"""Clip encoding: the common output format and frame-accurate re-encodes.

Used by ``cutpoint clips`` and ``cutpoint extend`` (and the audio fade by
``cutpoint assemble`` and the 360 renderer).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import state
from cutpoint.crop import Crop
from cutpoint.ffmpeg_utils import (
    READ_ERROR_RE,
    FFmpegError,
    MediaInfo,
    VideoEncoder,
    atomic_output,
    decode_raw,
    encode_with_fallback,
    ffmpeg_filters,
    probe,
    run,
)

logger = logging.getLogger(__name__)

CLIP_SHORT_S = 0.3  # an exported clip this much shorter than asked = truncated read
SAFE_END_MARGIN_S = 0.3  # trimmed re-encode stops this far before the failure

# Common audio of clips and assembled videos: 48 kHz stereo, AAC 192k.
A_FMT = "aresample=48000,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
AUDIO_OUT = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
SILENCE = "anullsrc=r=48000:cl=stereo"
# Audio this much shorter than its clip still covers it: below one AAC frame (21 ms at 48 kHz),
# the ms rounding of ``-t`` and stream ends, which leave the fade as it was. A shortfall of one
# whole AAC frame (21-23 ms) can cross it; harmless, as the fade then ends at the real audio end.
AUDIO_SHORT_S = 0.02


# Color (#289). HDR transfers (ffprobe ``color_transfer``) are tone-mapped to SDR Rec.709;
# a BT.2020 SDR source only changes primaries/matrix. Every other source keeps today's chain.
HDR_TRANSFERS = {"arib-std-b67": "HLG", "smpte2084": "PQ"}
BT2020_TRANSFERS = ("bt2020-10", "bt2020-12")
# Probed tags zscale takes as input values (same names as ffprobe's); others fall back to BT.2020.
_ZSCALE_PRIMARIES = ("bt709", "bt2020")
_ZSCALE_MATRICES = ("bt709", "bt2020nc", "bt2020c")
# Output tags of a converted clip, so players do not guess (other clips stay as they were).
BT709_TAGS = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
# Log/flat look of a 10-bit SDR source (one small frame, luma 0-1): mean HSV saturation
# below LOG_MAX_SATURATION, lifted blacks (1 % luma above LOG_MIN_BLACK), no near-white
# (99 % luma below LOG_MAX_WHITE) and still a picture (spread above LOG_MIN_SPREAD), so
# a black, white or uniform frame never counts.
LOG_MAX_SATURATION = 0.18
LOG_MIN_BLACK = 0.05
LOG_MAX_WHITE = 0.85
LOG_MIN_SPREAD = 0.15
_LOG_FRAME = (64, 36)
_HIGH_BIT_RE = re.compile(r"p0?1[0-6](?:le|be)?$")  # yuv420p10le, p010le, gbrp12le, ...

# Sources already reported this process (once per source, not once per clip).
_color_noted: set[str] = state.register_container(__name__, "_color_noted", set())


def color_kind(info: MediaInfo) -> str | None:
    """``"HLG"``, ``"PQ"`` or ``"BT.2020"`` when ``info`` needs an SDR Rec.709 conversion."""
    if info.color_transfer in HDR_TRANSFERS:
        return HDR_TRANSFERS[info.color_transfer]
    if (
        info.color_transfer in BT2020_TRANSFERS
        or info.color_primaries == "bt2020"
        or info.color_space in ("bt2020nc", "bt2020c")
    ):
        return "BT.2020"
    return None


def sdr_filter(info: MediaInfo, filters: frozenset[str] | None = None) -> str:
    """Filters converting an HDR or BT.2020 source to SDR Rec.709, "" for any other source.

    HLG and PQ are linearised, tone-mapped (``tonemap=mobius``) and re-encoded as BT.709
    with ``zscale``; BT.2020 SDR only changes primaries and matrix. They go after the
    scale and before ``format=yuv420p``. ``filters`` is this ffmpeg's filter list (default
    :func:`~cutpoint.ffmpeg_utils.ffmpeg_filters`); without ``zscale`` (or ``tonemap``)
    a WARNING says so once per source and the chain stays as it was.
    """
    kind = color_kind(info)
    if kind is None:
        return ""
    have = ffmpeg_filters() if filters is None else filters
    need = ("zscale", "tonemap") if kind in ("HLG", "PQ") else ("zscale",)
    missing = [f for f in need if f not in have]
    if missing:
        _note(
            info,
            logging.WARNING,
            "%s is %s video but this ffmpeg has no %s filter: exported without conversion to "
            "SDR (colors may look washed out). Use an ffmpeg build with zscale (libzimg)",
            info.path.name,
            kind,
            " or ".join(missing),
        )
        return ""
    pin = info.color_primaries if info.color_primaries in _ZSCALE_PRIMARIES else "bt2020"
    min_ = info.color_space if info.color_space in _ZSCALE_MATRICES else "bt2020nc"
    _note(info, logging.INFO, "%s is %s video: converting to SDR Rec.709", info.path.name, kind)
    if kind == "BT.2020":
        tin = info.color_transfer if info.color_transfer in BT2020_TRANSFERS else "bt709"
        return f"zscale=tin={tin}:pin={pin}:min={min_}:t=bt709:p=bt709:m=bt709:r=tv"
    # Every zscale names its input and output: a step reading untagged frames fails
    # ("no path between colorspaces"), and a zscale ``pin`` does not tag its output.
    # mobius keeps diffuse white (203 nits) near SDR white; hable left a synthetic HLG/PQ
    # chart about 25 % darker than its SDR original. CPU cost: about 30 fps at 1080p
    # (8 threads), so HDR sources export slower; SDR sources never run it.
    return (
        f"zscale=tin={info.color_transfer}:pin={pin}:min={min_}:t=linear:npl=100:p=bt709,"
        "format=gbrpf32le,tonemap=tonemap=mobius:desat=0,"
        "zscale=tin=linear:pin=bt709:t=bt709:p=bt709:m=bt709:r=tv"
    )


def _note(info: MediaInfo, level: int, msg: str, *args: object) -> None:
    """Log ``msg`` once per source and level."""
    key = f"{level}:{info.path}"
    if key not in _color_noted:
        _color_noted.add(key)
        logger.log(level, msg, *args)


def log_stats(frame: np.ndarray) -> tuple[float, float, float]:
    """Mean HSV saturation and the 1 % and 99 % luma (all 0-1) of an RGB uint8 frame."""
    rgb = frame.reshape(-1, 3).astype(np.float32) / 255.0
    hi, lo = rgb.max(axis=1), rgb.min(axis=1)
    sat = np.where(hi > 0, (hi - lo) / np.maximum(hi, 1e-6), 0.0)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    p1, p99 = np.percentile(luma, [1, 99])
    return float(sat.mean()), float(p1), float(p99)


def looks_log(saturation: float, black: float, white: float) -> bool:
    """True for flat stats: low saturation and compressed luma (see :func:`log_stats`)."""
    return (
        saturation < LOG_MAX_SATURATION
        and black > LOG_MIN_BLACK
        and white < LOG_MAX_WHITE
        and white - black > LOG_MIN_SPREAD
    )


def warn_if_log(src: Path, info: MediaInfo, at: float) -> None:
    """WARNING once per source when 10-bit SDR footage looks log/flat; never applies a LUT.

    Only 10-bit (or deeper) sources that get no HDR conversion are checked, with one
    64x36 frame of ``src`` at ``at`` s; a failed grab skips the check.
    """
    key = f"log:{src}"
    if key in _color_noted or color_kind(info) or not _HIGH_BIT_RE.search(info.pix_fmt):
        return
    _color_noted.add(key)
    w, h = _LOG_FRAME
    frames = decode_raw(
        src,
        size=(w, h),
        vf=f"scale={w}:{h}:flags=area,format=rgb24",
        start=at,
        duration=1.0,
        hwaccel=None,
        timeout=60,
    )
    try:
        got = next(frames, None)
    except FFmpegError as err:
        logger.debug("log check of %s skipped: %s", src.name, err)
        return
    finally:
        frames.close()
    if got is None:
        return
    sat, black, white = log_stats(got[1])
    logger.debug("%s: saturation %.3f, luma %.3f-%.3f", src.name, sat, black, white)
    if looks_log(sat, black, white):
        logger.warning(
            "%s looks like log/flat footage (10-bit, saturation %.2f, luma %.2f-%.2f): "
            "clips keep that look; apply your camera's LUT in an editor for normal colors",
            src.name,
            sat,
            black,
            white,
        )


def audio_fade_filter(dur: float, fade: float, audio: float | None = None) -> str:
    """``afade`` in/out for a clip of ``dur`` seconds ("" when ``fade`` is 0).

    ``audio`` is the audio the clip holds (s), None when unknown. Audio more than
    :data:`AUDIO_SHORT_S` shorter than the clip fades out at its own end (#250), not
    after it. The fade is at most a quarter of the clip (or of the shorter audio).
    Timestamps are reset first so ``st`` is relative to the clip start whatever the
    input seek produced.
    """
    if audio is not None and audio < dur - AUDIO_SHORT_S:
        dur = audio
    f = min(fade, dur / 4)
    if f <= 0.005:
        return ""
    return (
        f",asetpts=PTS-STARTPTS,afade=t=in:st=0:d={f:.3f}:curve=qsin,"
        f"afade=t=out:st={max(0.0, dur - f):.3f}:d={f:.3f}:curve=qsin"
    )


@dataclass(frozen=True)
class OutputFormat:
    """Common format every exported clip is normalised to."""

    width: int
    height: int
    fps: int

    def video_filter(self, vertical: bool, crop: Crop | None = None, color: str = "") -> str:
        """Filter chain producing exactly this format from any input.

        ``crop`` is applied first (trimmed to the output aspect when close, so a
        bordered recording fills the frame), then the usual scale/pad. ``color``
        (:func:`sdr_filter`) goes just before ``format=yuv420p``.
        """
        w, h = self.width, self.height
        pre = ""
        if crop is not None:
            box = crop if vertical else crop.fit_aspect(w / h)
            pre = f"crop={box.w}:{box.h}:{box.x}:{box.y},"
        if vertical:
            geo = f"crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',scale={w}:{h}:out_range=tv"
        else:
            geo = (
                f"scale={w}:{h}:force_original_aspect_ratio=decrease:out_range=tv,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
            )
        color = f"{color}," if color else ""
        return f"{pre}{geo},setsar=1,fps={self.fps},{color}format=yuv420p"


def choose_format(infos: list[MediaInfo], vertical: bool) -> OutputFormat:
    """Largest source resolution capped at 1080p (vertical: always 1080x1920).

    60 fps if any source is above 50 fps, else 30.
    """
    ref = max(
        infos, key=lambda i: (i.height, i.width)
    )  # ties: the wider source (a 9:16 capture must not make a 16:9 cut vertical)
    h = min(ref.height, 1080)
    fps = 60 if max(float(i.fps) for i in infos) > 50 else 30
    if vertical:
        return OutputFormat(width=1080, height=1920, fps=fps)
    w = round(h * ref.width / ref.height / 2) * 2
    return OutputFormat(width=w, height=h - h % 2, fps=fps)


def export_clip(
    src: Path,
    info: MediaInfo,
    start: float,
    end: float,
    out: Path,
    fmt: OutputFormat,
    vertical: bool,
    crop: Crop | None = None,
    audio_fade: float = 0.0,
) -> str:
    """Frame-accurate re-encode of [start, end] (absolute source time) into ``out``.

    ``audio_fade`` > 0 bakes a short audio fade-in/out into the clip (the
    video stays a hard cut), so a cut never slaps from sound to silence.
    ``out`` only appears once the clip is complete (:func:`atomic_output`); a failed
    auto-chosen GPU encode is redone on the CPU (:func:`encode_with_fallback`).

    Returns:
        ffmpeg's stderr (check it for read errors; ffmpeg exits 0 after them).
    """
    with atomic_output(out) as part:
        return _export(src, info, start, end, part, fmt, vertical, crop, audio_fade, out.name)


def _export(
    src: Path,
    info: MediaInfo,
    start: float,
    end: float,
    out: Path,
    fmt: OutputFormat,
    vertical: bool,
    crop: Crop | None,
    audio_fade: float,
    name: str,
) -> str:
    """:func:`export_clip` straight into ``out`` (no temporary name); ``name`` is for logs."""
    warn_if_log(src, info, start)
    return encode_with_fallback(
        lambda enc: _encode(enc, src, info, start, end, out, fmt, vertical, crop, audio_fade),
        name,
    )


def _encode(
    enc: VideoEncoder,
    src: Path,
    info: MediaInfo,
    start: float,
    end: float,
    out: Path,
    fmt: OutputFormat,
    vertical: bool,
    crop: Crop | None,
    audio_fade: float,
) -> str:
    """One ffmpeg run of :func:`export_clip` with ``enc``."""
    dur = end - start
    args = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{dur:.3f}",
        "-i",
        str(src),
    ]
    color = sdr_filter(info)
    vf = f"[0:v:0]{enc.chain(fmt.video_filter(vertical, crop, color))}[v]"
    audio = None if info.audio_duration is None else min(end, info.audio_duration) - start
    a_fmt = A_FMT + audio_fade_filter(dur, audio_fade, audio)
    if info.audio_streams == 0:
        args += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", SILENCE]
        af = f"[1:a]{a_fmt}[a]"
    elif info.audio_streams == 1:
        af = f"[0:a:0]{a_fmt}[a]"
    else:  # e.g. ShadowPlay game + separate mic track
        ins = "".join(f"[0:a:{i}]" for i in range(info.audio_streams))
        af = f"{ins}amix=inputs={info.audio_streams}:normalize=0,{a_fmt}[a]"
    args += [
        "-filter_complex",
        f"{vf};{af}",
        "-map",
        "[v]",
        "-map",
        "[a]",
        *enc.args(),
        "-color_range",
        "tv",
        *(BT709_TAGS if color else []),
        "-fps_mode",
        "cfr",
        *AUDIO_OUT,
        "-t",
        f"{dur:.3f}",
        "-video_track_timescale",
        "90000",
        "-movflags",
        "+faststart",
        str(out),
    ]
    return run(args)


def export_clip_checked(
    src: Path,
    info: MediaInfo,
    start: float,
    end: float,
    out: Path,
    fmt: OutputFormat,
    vertical: bool,
    crop: Crop | None = None,
    audio_fade: float = 0.0,
) -> dict[str, Any] | None:
    """:func:`export_clip`, guarded against silent truncation by read errors.

    If ffmpeg reports a read error or the clip comes out short, the read is
    retried once (marginal sectors sometimes read on a second try). If it is
    still short, the clip is re-encoded to end just before the failure so it
    ends cleanly instead of with mismatched audio/video. Every attempt writes to
    the temporary name; ``out`` only appears with the final result.

    Returns:
        None if the clip is complete, else a record for manifest.json with
        ``failure_time``, ``messages``, ``retried`` and the ``end`` used.
    """
    with atomic_output(out) as part:
        return _export_checked(src, info, start, end, part, fmt, vertical, crop, audio_fade, out)


def _export_checked(
    src: Path,
    info: MediaInfo,
    start: float,
    end: float,
    part: Path,
    fmt: OutputFormat,
    vertical: bool,
    crop: Crop | None,
    audio_fade: float,
    out: Path,
) -> dict[str, Any] | None:
    """:func:`export_clip_checked` writing to ``part``; ``out`` names the clip in logs."""
    expected = end - start
    messages: list[str] = []
    for attempt in (1, 2):
        text = _export(src, info, start, end, part, fmt, vertical, crop, audio_fade, out.name)
        got = probe(part).duration
        found = [m.strip() for m in READ_ERROR_RE.findall(text)]
        if not found and got >= expected - CLIP_SHORT_S:
            if attempt == 1:
                return None
            logger.warning("  %s: re-read succeeded on retry", out.name)
            return {"failure_time": None, "messages": messages, "retried": True, "end": end}
        messages = list(dict.fromkeys(messages + found))[:5]
        fail_t = start + got
        logger.warning("!" * 78)
        logger.warning(
            "READ ERROR exporting %s from %s at ~%.1f s (got %.2f of %.2f s)%s",
            out.name,
            src.name,
            fail_t,
            got,
            expected,
            "; retrying once" if attempt == 1 else "",
        )
        logger.warning("  ffmpeg: %s", "; ".join(messages) or "output shorter than requested")
        logger.warning("!" * 78)
    safe_end = max(start + 0.1, fail_t - SAFE_END_MARGIN_S)
    _export(src, info, start, safe_end, part, fmt, vertical, crop, audio_fade, out.name)
    logger.warning("  %s trimmed to end at %.2f (before the unreadable spot)", out.name, safe_end)
    return {
        "failure_time": round(fail_t, 3),
        "messages": messages,
        "retried": True,
        "end": round(safe_end, 3),
    }
