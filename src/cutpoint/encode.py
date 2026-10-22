"""Clip encoding: the common output format and frame-accurate re-encodes.

Used by ``cutpoint clips`` and ``cutpoint extend`` (and the audio fade by
``cutpoint assemble`` and the 360 renderer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cutpoint.crop import Crop
from cutpoint.ffmpeg_utils import READ_ERROR_RE, MediaInfo, probe, run, video_encoder

logger = logging.getLogger(__name__)

CLIP_SHORT_S = 0.3  # an exported clip this much shorter than asked = truncated read
SAFE_END_MARGIN_S = 0.3  # trimmed re-encode stops this far before the failure


def audio_fade_filter(dur: float, fade: float) -> str:
    """``afade`` in/out for a clip of ``dur`` seconds ("" when ``fade`` is 0).

    The fade is at most a quarter of the clip. Timestamps are reset first so
    ``st`` is relative to the clip start whatever the input seek produced.
    """
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

    def video_filter(self, vertical: bool, crop: Crop | None = None) -> str:
        """Filter chain producing exactly this format from any input.

        ``crop`` is applied first (trimmed to the output aspect when close, so a
        bordered recording fills the frame), then the usual scale/pad.
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
        return f"{pre}{geo},setsar=1,fps={self.fps},format=yuv420p"


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

    Returns:
        ffmpeg's stderr (check it for read errors; ffmpeg exits 0 after them).
    """
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
    enc = video_encoder()
    vf = f"[0:v:0]{enc.chain(fmt.video_filter(vertical, crop))}[v]"
    a_fmt = (
        "aresample=48000,aformat=sample_fmts=fltp:sample_rates=48000:"
        "channel_layouts=stereo" + audio_fade_filter(dur, audio_fade)
    )
    if info.audio_streams == 0:
        args += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
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
        "-fps_mode",
        "cfr",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        "2",
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
    ends cleanly instead of with mismatched audio/video.

    Returns:
        None if the clip is complete, else a record for manifest.json with
        ``failure_time``, ``messages``, ``retried`` and the ``end`` used.
    """
    expected = end - start
    messages: list[str] = []
    for attempt in (1, 2):
        text = export_clip(src, info, start, end, out, fmt, vertical, crop, audio_fade)
        got = probe(out).duration
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
    export_clip(src, info, start, safe_end, out, fmt, vertical, crop, audio_fade)
    logger.warning("  %s trimmed to end at %.2f (before the unreadable spot)", out.name, safe_end)
    return {
        "failure_time": round(fail_t, 3),
        "messages": messages,
        "retried": True,
        "end": round(safe_end, 3),
    }
