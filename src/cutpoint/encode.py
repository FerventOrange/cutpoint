"""Clip encoding: the common output format and frame-accurate re-encodes.

Used by ``cutpoint clips`` and ``cutpoint extend`` (and the audio fade by
``cutpoint assemble`` and the 360 renderer).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path, PurePath
from typing import Any, Protocol

import numpy as np

from cutpoint import jsonout, state
from cutpoint.audio_tracks import export_tracks
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
    has_soxr,
    mix_head,
    probe,
    run,
)

logger = logging.getLogger(__name__)

CLIP_SHORT_S = 0.3  # an exported clip this much shorter than asked = truncated read
SAFE_END_MARGIN_S = 0.3  # trimmed re-encode stops this far before the failure
# Keyframes forced this far (s) from both ends of every clip, so ``assemble --transition fade``
# re-encodes a fade of up to 0.5 s (default 0.25 s) plus little more and stream-copies the rest
# (#463). One per end costs about 5% in clip size; longer fades use the clips' own keyframes.
EDGE_KEYFRAMES_S = (0.5,)

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
# Nominal peak of HLG (BT.2100 reference display); its tone map's peak is this over npl (#449).
HLG_PEAK_NITS = 1000
# Probed tags zscale takes as input values (same names as ffprobe's); others fall back to BT.2020.
_ZSCALE_PRIMARIES = ("bt709", "bt2020")
_ZSCALE_MATRICES = ("bt709", "bt2020nc", "bt2020c")
# Output tags of a converted clip, so players do not guess (other clips stay as they were).
BT709_TAGS = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
# BT.601 SDR (#290 stage 0.5): NVIDIA DVR captures (tv, smpte170m) and SD video hold BT.601
# data. Exported as it was, a cut-joined montage stream-copies it next to BT.709 clips under
# the first clip's tags, and players reading only those showed the BT.601 part up to ~33 codes
# off on saturated colours. A matrix tag of that family (smpte240m has its own matrix) is
# converted to BT.709. Without a matrix tag, players pick the matrix by size, and so does this:
# SD (below 720 lines and 1280 columns, so cropped widescreen HD such as 1280x536 is HD) is
# BT.601 (a BT.601 primaries tag names which one), HD is BT.709. The transfer tag never
# decides: BT.601 and BT.709 share the curve. HD without a matrix tag (OBS, editors) was
# exported without one; on real footage tagging it bt709 changed no frame (framemd5).
BT601_TAGS = ("smpte170m", "bt470bg", "smpte240m")
BT601_MAX_HEIGHT = 719
BT601_MAX_WIDTH = 1279
# Matrix only (fast=1: primaries and transfer are not re-mapped, as players treat BT.601 SDR).
# ``colorspace`` is in every ffmpeg since 3.1 and works in YUV: on BT.601 bars (lossless),
# decoded with the BT.709 matrix, the mean RGB error inside the bars was 0.17 codes (6.0
# unconverted) against 1.15 for ``scale=in_color_matrix=...`` (0.28 with accurate_rnd+
# full_chroma_int) and 0.17 for ``zscale`` (not in every build). It costs about 7 % wall time
# on 1080p60. It is explicit: recent ffmpeg (9.0.2 measured) converts by itself to match the
# output tags, but older builds would only re-tag the BT.601 data. Its input is tv range: the
# scale before it always outputs that (full-range BT.601 is narrowed there, as full-range
# BT.709 always was).
_BT601_TO_709 = "colorspace=all=bt709:iall={}:irange=tv:range=tv:fast=1"
# Tags HD frames without a matrix tag bt709 without touching a pixel. The output tags alone would not do:
# recent ffmpeg (9.0.2 measured) then converts the untagged frames as BT.601 (on bars: 7 codes
# mean, 22 max).
TAG_BT709 = "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709"
# Full-range frames to TV range, matrix and tags kept (UI proxies of full-range BT.601, #441).
TV_RANGE = "scale=out_range=tv"
# To the BT.601 full range every decoder reads a JPEG as (UI stills, #441); the input range is
# the frame's. The mjpeg encoder only re-tags: a BT.709 still was 7.4 codes off (mean RGB on
# bars, 32 max) in the UI, while its export was right.
_TO_JPEG = "colorspace=all=bt470bg:iall={}:range=pc:fast=1"
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


def is_high_bit_sdr(info: MediaInfo) -> bool:
    """True for a 10-bit (or deeper) source that gets no HDR or BT.2020 conversion
    (:func:`color_kind`): the sources :func:`warn_if_log` checks and whose log look
    :mod:`cutpoint.color` skips."""
    return bool(_HIGH_BIT_RE.search(info.pix_fmt)) and not color_kind(info)


def sdr_filter(info: MediaInfo, filters: frozenset[str] | None = None) -> str:
    """Filters converting an HDR or BT.2020 source to SDR Rec.709, "" for any other source.

    HLG and PQ are linearised, tone-mapped (``tonemap``: mobius for HLG, hable for PQ) and
    re-encoded as BT.709 with ``zscale``; BT.2020 SDR only changes primaries and matrix. They
    go after the scale and before ``format=yuv420p``. ``filters`` is this ffmpeg's filter
    list (default :func:`~cutpoint.ffmpeg_utils.ffmpeg_filters`); without ``zscale`` (or
    ``tonemap``) a WARNING says so once per source and the chain stays as it was.
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
            "%s is %s video but this ffmpeg has no %s filter: shown and exported without "
            "conversion to SDR Rec.709 (colors may look washed out). "
            "Use an ffmpeg build with zscale (libzimg)",
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
    # Reference: libplacebo bt.2390 (not used: it needs Vulkan), mean luma of 3 frames each.
    # PQ: at npl=100 diffuse white (203 nits) is linear 2.0 and hable rolls it off close to
    # the reference (Eclipse +1.4 %, Swan -0.6 %); the peak is 1000 nits unless the frames
    # carry light-level metadata. HLG (#449): the same chain was 16 % darker on a real phone
    # clip (Xiaomi 14T Pro). mobius at npl=203 (diffuse white is 1.0, peak 1000/203) keeps
    # midtones linear and came within 3 % (-2.7 %, mean luma error 1.6 % against 9.5 %).
    # mobius had looked milky (lifted shadows) at npl=100, which puts diffuse white at 2.0.
    # On PQ (Eclipse, Swan) the HLG chain was 15 % too bright, so PQ keeps hable (Jellyfin
    # HDR10: -21 % with hable and with libplacebo peak_detect=0, a dynamic-peak gap).
    # desat>0 greys bright colours. CPU cost: about 30 fps at 1080p (8 threads), so HDR sources export slower;
    # SDR sources never run it.
    if kind == "HLG":
        npl, tone = 203, f"mobius:param=0.5:peak={HLG_PEAK_NITS / 203:.3f}:desat=0"
    else:
        npl, tone = 100, "hable:desat=0"
    return (
        f"zscale=tin={info.color_transfer}:pin={pin}:min={min_}:t=linear:npl={npl}:p=bt709,"
        f"format=gbrpf32le,tonemap=tonemap={tone},"
        "zscale=tin=linear:pin=bt709:t=bt709:p=bt709:m=bt709:r=tv"
    )


def bt601_matrix(info: MediaInfo) -> str | None:
    """The BT.601-family matrix (an ffprobe name) of an SDR source, None for any other.

    The matrix tag decides. Without one, SD (:func:`is_sd`) is BT.601: the primaries tag's
    matrix when it is one of :data:`BT601_TAGS`, else ``smpte170m``; HD is BT.709.
    """
    if color_kind(info):
        return None
    if info.color_space:
        return info.color_space if info.color_space in BT601_TAGS else None
    if not is_sd(info):
        return None
    return info.color_primaries if info.color_primaries in BT601_TAGS else "smpte170m"


def is_sd(info: MediaInfo) -> bool:
    """True below 720 lines and 1280 columns, the size players read as BT.601 when untagged."""
    return info.height <= BT601_MAX_HEIGHT and info.width <= BT601_MAX_WIDTH


def bt709_filter(info: MediaInfo, filters: frozenset[str] | None = None) -> str:
    """Color filters making an exported clip BT.709; they go where :func:`sdr_filter`'s go.

    HDR and BT.2020 sources get :func:`sdr_filter` (``filters`` as there); BT.601 ones
    (:func:`bt601_matrix`) a matrix conversion, with one INFO line per source; HD without a
    matrix tag only :data:`TAG_BT709`. Anything else (BT.709 among it) gets "", the chain of
    before. A non-empty result means the clip takes :data:`BT709_TAGS`.
    """
    if color_kind(info):
        return sdr_filter(info, filters)
    matrix = bt601_matrix(info)
    if matrix:
        why = (
            f"tagged {info.color_space}"
            if info.color_space
            else f"no matrix tag, {info.width}x{info.height}"
        )
        _note(
            info, logging.INFO, "%s is BT.601 video (%s): converting to BT.709", info.path.name, why
        )
        return _BT601_TO_709.format(matrix)
    return "" if info.color_space else TAG_BT709


def preview_filter(info: MediaInfo, filters: frozenset[str] | None = None) -> str:
    """Color filters of a UI preview proxy: :func:`sdr_filter`'s, for full-range BT.601
    :data:`TV_RANGE` (#441).

    A BT.601 proxy keeps its matrix and tags, which players read (and SD without tags is
    BT.601 to them too); only full range is narrowed to the TV range of exports. Every other
    source gets :func:`sdr_filter`'s result, so its proxy keeps its name.
    """
    if bt601_matrix(info) and info.color_range == "pc":
        return TV_RANGE
    return sdr_filter(info, filters)


def still_filter(info: MediaInfo, filters: frozenset[str] | None = None) -> str:
    """Color filters of a JPEG still, so it shows what the source's export shows (#441).

    A JPEG is BT.601 full range to every decoder, and the encoder only re-tags other matrices.
    BT.709 sources (tagged, or HD without a matrix tag, as :func:`bt709_filter` reads them)
    and HDR or BT.2020 ones after their :func:`sdr_filter` are converted to the BT.601 matrix
    and full range; smpte240m to BT.601 as well. Other BT.601 sources (they already are) and
    an HDR source this ffmpeg cannot tone-map get "", the chain of before.
    """
    if color_kind(info):
        sdr = sdr_filter(info, filters)
        return f"{sdr},{_TO_JPEG.format('bt709')}" if sdr else ""
    matrix = bt601_matrix(info)
    if matrix:
        return _TO_JPEG.format(matrix) if matrix == "smpte240m" else ""
    return _TO_JPEG.format("bt709") if info.color_space in (None, "bt709") else ""


def filter_path(path: PurePath) -> str:
    """``path`` as a filter option value inside a filtergraph (``-filter_complex``).

    Forward slashes on Windows (``as_posix``; on POSIX a backslash is a file name character),
    then the two escaping levels of ffmpeg: the option value (``\\``, ``'``, ``:``, so a drive
    colon is not an option separator) and the filtergraph (``\\``, ``'``, ``[``, ``]``, ``,``,
    ``;``). Spaces need no escaping inside a value.
    """
    text = path.as_posix()
    for special in ("\\':", "\\'[],;"):
        text = "".join("\\" + c if c in special else c for c in text)
    return text


def lut_filter(path: PurePath, strength: float = 1.0) -> str | None:
    """Filters grading a BT.709 tv-range clip with the 3D LUT ``path`` (#290 stage 1), for
    :meth:`OutputFormat.video_filter`'s ``color`` after :func:`bt709_filter`'s; None at
    strength 0.

    The frames go to full-range RGB with the BT.709 matrix, through ``lut3d`` (tetrahedral)
    and back to BT.709 tv range. The matrix is explicit: an older ffmpeg's automatic
    conversion would use BT.601. The LUT is assumed to be made for the footage's log or flat
    profile (a GoPro Flat or Insta360 Log to Rec.709 LUT, say); Cutpoint cannot check that.
    Below strength 1 the graded frames are blended over the original at that opacity, in YUV
    (a linear mix, the same as one in RGB, on fewer samples). Strength 1 has no blend.
    The result is tagged BT.709 tv range (:data:`TAG_BT709`): ffmpeg 6.1's ``scale`` back to
    YUV leaves the matrix unset (9.0.2 sets it), and a BT.709 source gets no output tags.
    """
    if strength <= 0:
        return None
    to_rgb = "scale=in_color_matrix=bt709:in_range=tv:out_range=pc"
    grade = f"lut3d=file={filter_path(path)}:interp=tetrahedral"
    to_yuv = "scale=out_color_matrix=bt709:out_range=tv"
    tag = f"{TAG_BT709}:range=tv"
    if strength >= 1:
        return f"{to_rgb},{grade},{to_yuv},{tag}"
    return (
        f"split=2[lut_o][lut_i];[lut_i]{to_rgb},{grade},{to_yuv}[lut_g];"
        f"[lut_g][lut_o]blend=all_mode=normal:all_opacity={strength:g},{tag}"
    )


LUT_SUFFIXES = (".cube", ".3dl")  # lut3d's Iridas/Resolve .cube and Autodesk .3dl
GRADE_KEYS = ("lut", "color")  # manifest.json keys of a graded clip (``color``: #290 stage 2)


class Grade(Protocol):
    """A clip's colour grade (#290), applied after the BT.709 conversion; :class:`Lut` is one.

    ``filter()`` is its video filter chain ("" for none); ``to_json()`` the manifest.json keys
    that record it, which :func:`grade_from_entry` reads back.
    """

    def filter(self) -> str: ...

    def to_json(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Lut:
    """A ``--lut`` grade: an absolute LUT file and its ``--lut-strength`` (0-1)."""

    path: Path
    strength: float = 1.0

    def filter(self) -> str:
        """:func:`lut_filter` ("" at strength 0)."""
        return lut_filter(self.path, self.strength) or ""

    def to_json(self) -> dict[str, Any]:
        """manifest.json keys of a clip graded with it: ``lut`` and ``lut_strength``."""
        return {"lut": str(self.path), "lut_strength": self.strength}

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> Lut | None:
        """The grade a manifest entry records (:meth:`to_json`), None for an ungraded clip."""
        if not entry.get("lut"):
            return None
        return cls(Path(entry["lut"]), float(entry.get("lut_strength", 1.0)))


def grade_from_entry(entry: dict[str, Any]) -> Grade | None:
    """The grade a manifest entry records (:meth:`Grade.to_json`), None for an ungraded clip:
    a ``--lut`` grade (:meth:`Lut.from_entry`) or a ``--color`` one
    (:meth:`cutpoint.color.AutoGrade.from_entry`, skipped ones included)."""
    from cutpoint.color import AutoGrade  # noqa: PLC0415 - it imports this module

    return Lut.from_entry(entry) or AutoGrade.from_entry(entry)


def check_lut(path: Path | str, strength: float = 1.0, what: str = "--lut") -> Lut:
    """Validate a LUT before any work: a readable ``.cube`` / ``.3dl`` file ffmpeg's
    ``lut3d`` loads, and a strength from 0 to 1.

    Returns:
        The :class:`Lut` with the absolute path.

    Raises:
        FileNotFoundError: If the file does not exist.
        jsonout.ArgumentError: If it is not a file, cannot be read, has another extension,
            ffmpeg cannot load it, or ``strength`` is outside 0-1.
    """
    path = Path(path).absolute()
    if not 0 <= strength <= 1:
        raise jsonout.ArgumentError(f"{what} strength {strength:g}: must be from 0 to 1")
    if path.suffix.lower() not in LUT_SUFFIXES:
        raise jsonout.ArgumentError(
            f"{what} {path}: not a LUT file (use {' or '.join(LUT_SUFFIXES)})"
        )
    if not path.exists():
        raise FileNotFoundError(f"{what} file not found: {path}")
    try:
        with path.open("rb") as f:
            f.read(1)
    except OSError as err:
        raise jsonout.ArgumentError(f"{what} {path}: cannot be read ({err})") from None
    # One tiny frame through lut3d: a malformed LUT fails now, not after hours of analysis.
    graph = f"color=s=16x16:d=0.04,format=gbrp,lut3d=file={filter_path(path)}"
    try:
        run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", graph, "-f", "null", "-"])
    except FFmpegError as err:
        raise jsonout.ArgumentError(f"{what} {path}: ffmpeg cannot load it ({err})") from None
    return Lut(path, strength)


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
    if key in _color_noted or not is_high_bit_sdr(info):
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
            "clips keep that look; give your camera's LUT with --lut FILE or apply it in an editor "
            "for normal colors",
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


def limiter_chain(limit_db: float, soxr: bool) -> str:
    """Oversampled (192 kHz) lookahead limiter at ``limit_db`` dBFS, back to 48 kHz afterwards
    (with soxr when ``soxr``, else ffmpeg's default resampler)."""
    lin = max(0.0625, 10 ** (limit_db / 20))
    resampler = ":resampler=soxr" if soxr else ""
    return (
        f"aresample=192000,alimiter=limit={lin:.4f}:attack=1:release=60:level=0:"
        f"latency=1,aresample=48000{resampler}"
    )


# Peak compression and clip leveling (#326, ``--compress-peaks``). Every clip of a set is
# measured (momentary loudness, BS.1770 blocks); the set's typical level is the median of
# its blocks that are not silence or room tone (above the pooled gated loudness minus
# PEAK_GATE_LU). An RMS compressor starts ``threshold`` dB above that level (set in LUFS;
# acompressor measures each channel's RMS, so on stereo it acts about 2-3 dB later), the same for
# every clip, so shouts, laughs into the mic and gunfire are pulled toward the speech of
# the whole set, not toward each clip's own (often near-silent) background. Each clip also
# gets a gain that moves its loudness after compression ``level`` of the way to the set's
# mean (at most ``max_boost`` up and ``max_cut`` down: a boost also lifts background noise).
# The gain goes before the compressor, so the bursts of a boosted clip are still caught.
# Measured on 12 clips of an R6 montage (gunfire, shouting), a voice-chat recording and
# GoPro wind, exported and assembled with --loudnorm (before #523, dynamic loudnorm; table
# in the #326 PR): the loudest
# moment above the speech level 12.2 dB -> 7.5 (light) / 5.2 (strong), 99th-percentile
# momentary loudness above speech 10.0 -> 5.9 / 2.8 dB, clip loudness spread 9.9 -> 4.3 /
# 4.7 LU, true peak unchanged (-2.2 / -2.0 dBTP). The cost: the quiet parts of boosted
# clips rise with them (commentary background +5.5 / +9.9 dB, +2.5 / +4.9 relative to
# speech); hence the boost caps. Attack 5 ms (1 ms caught 0.3 dB more, 20 ms let 0.7 dB
# more through); release 200 ms: a +19 dB burst recovers to within 1 dB 170-220 ms after
# it ends (500 ms ducks the next words for about half a second, 100 ms let 0.5 dB more by).
@dataclass(frozen=True)
class PeakPreset:
    """One ``--compress-peaks`` preset (see the comment above it)."""

    threshold: float  # dB above the set's typical loudness where compression starts
    ratio: float
    attack: float  # ms
    release: float  # ms
    level: float  # 0-1: share of the way each clip moves to the set's mean loudness
    max_boost: float  # dB
    max_cut: float  # dB


PEAK_PRESETS = {
    "light": PeakPreset(8.0, 3.0, 5.0, 200.0, 0.5, 6.0, 12.0),
    "strong": PeakPreset(4.0, 6.0, 5.0, 200.0, 1.0, 10.0, 20.0),
}
PEAK_GATE_LU = 20.0  # blocks this far below the pooled loudness are background, not "typical"
SILENT_LUFS = -70.0  # BS.1770 absolute gate
PEAK_THRESHOLD_MIN, PEAK_THRESHOLD_MAX = -60.0, -0.1  # dBFS acompressor accepts
# Container tag of a clip exported with compression (``assemble`` reads it, so a clip
# renamed during review is still not compressed twice).
PEAK_TAG = "cutpoint:compress_peaks"
# The limiter after a compressed clip's edge fade: AAC 192k adds up to +1.2 dB of true
# peak, so clips stay below about -1.8 dBTP.
CLIP_LIMIT_DB = -3.0
_GAIN_STEP = 0.25  # dB grid of the leveling search
_GAIN_SPAN = 40.0


def gated_loudness(blocks: np.ndarray) -> float | None:
    """BS.1770 integrated loudness of momentary ``blocks`` (LUFS), None when all are silent."""
    b = blocks[blocks > SILENT_LUFS]
    if b.size == 0:
        return None
    energy = 10 ** (b / 10)
    gated = energy[b > 10 * np.log10(energy.mean()) - 10]
    return float(10 * np.log10(gated.mean()))


def _compressed(blocks: np.ndarray, threshold: float, ratio: float) -> np.ndarray:
    """``blocks`` through the compressor's static curve (an estimate of its output)."""
    return np.where(blocks > threshold, threshold + (blocks - threshold) / ratio, blocks)


@dataclass(frozen=True)
class PeakPlan:
    """Compressor threshold and per-clip gains for one set of clips (:func:`plan_peaks`)."""

    preset: str
    threshold: float  # dB (RMS, dBFS) where the compressor starts
    gains: tuple[float | None, ...]  # dB per clip; None = the clip is left alone
    makeup: tuple[float | None, ...] = ()  # dB per clip after the compressor (:func:`plan_makeup`)

    def filter(self, i: int) -> str:
        """``,volume=...,acompressor=...[,volume=...]`` for clip ``i`` ("" when it is left
        alone); the last ``volume`` is its makeup gain."""
        gain = self.gains[i]
        if gain is None:
            return ""
        p = PEAK_PRESETS[self.preset]
        volume = f",volume={gain:.2f}dB" if gain else ""
        makeup = self.makeup[i] if self.makeup else None
        return (
            f"{volume},acompressor=threshold={10 ** (self.threshold / 20):.6f}:ratio={p.ratio:g}:"
            f"attack={p.attack:g}:release={p.release:g}:detection=rms:link=maximum"
            + (f",volume={makeup:.2f}dB" if makeup else "")
        )

    def measure_chain(self, i: int) -> str:
        """Clip ``i``'s gain and compressor, no makeup, after :data:`A_FMT` as in the export (a
        mono source reaches the compressor upmixed, 3 dB lower per channel), as a filter prefix
        ending in a comma (:func:`~cutpoint.ffmpeg_utils.clip_momentary` ``chain``; "" when it
        is left alone)."""
        return f"{A_FMT}{replace(self, makeup=()).filter(i)}," if self.gains[i] is not None else ""


def plan_peaks(
    preset: str,
    series: list[list[float]],
    durations: list[float],
    apply: list[bool] | None = None,
) -> PeakPlan | None:
    """Plan ``preset`` for clips with momentary loudness ``series``
    (:func:`~cutpoint.ffmpeg_utils.clip_momentary`) and ``durations`` (s).

    Every clip counts for the typical level and the mean; only clips with ``apply`` (default
    all) and audio above silence get a filter. Returns None when no clip gets one.
    """
    p = PEAK_PRESETS[preset]
    blocks = [np.asarray(s, dtype=float) for s in series]
    pooled = np.concatenate([np.zeros(0), *(b[b > SILENT_LUFS] for b in blocks)])
    whole = gated_loudness(pooled)
    if whole is None:
        return None
    threshold = float(np.median(pooled[pooled > whole - PEAK_GATE_LU])) + p.threshold
    # acompressor takes -60.2..0 dBFS only (near-silent or very loud footage would fail).
    threshold = min(PEAK_THRESHOLD_MAX, max(PEAK_THRESHOLD_MIN, threshold))
    loud = [gated_loudness(_compressed(b, threshold, p.ratio)) for b in blocks]
    heard = [(10 ** (v / 10), d) for v, d in zip(loud, durations, strict=True) if v is not None]
    target = 10 * np.log10(np.average([e for e, _ in heard], weights=[d for _, d in heard]))
    grid = np.arange(-_GAIN_SPAN, _GAIN_SPAN + _GAIN_STEP / 2, _GAIN_STEP)
    gains: list[float | None] = []
    for i, b in enumerate(blocks):
        if loud[i] is None or (apply is not None and not apply[i]):
            gains.append(None)
            continue
        # The gain whose compressed loudness is nearest the mean; of near-equal ones the
        # smallest (a heavily compressed clip hardly gets louder with more gain).
        cost = [
            abs((gated_loudness(_compressed(b + g, threshold, p.ratio)) or -120.0) - target)
            + 0.05 * abs(g)
            for g in grid
        ]
        full = float(grid[int(np.argmin(cost))])
        gains.append(round(max(-p.max_cut, min(p.max_boost, p.level * full)), 2))
    if all(g is None for g in gains):
        return None
    return PeakPlan(preset, round(threshold, 2), tuple(gains))


def plan_makeup(
    plan: PeakPlan, series: list[list[float]], compressed: list[list[float]]
) -> PeakPlan:
    """``plan`` with one makeup gain for the whole set, so the set keeps its loudness.

    ``series`` is each clip's momentary loudness as in :func:`plan_peaks`, ``compressed`` the
    same measured through :meth:`PeakPlan.measure_chain` (ignored for clips left alone). The
    makeup is the gain, after the compressor, that brings the set's integrated loudness (all
    clips pooled, so weighted by duration) back to that of ``series``. It is at most the
    preset's ``max_boost`` and at most ``max_boost`` minus the clip's own leveling gain, so the
    quiet parts of a clip never rise more than ``max_boost`` in all; at least ``-max_cut``.
    """
    p = PEAK_PRESETS[plan.preset]
    before = [np.asarray(s, dtype=float) for s in series]
    target = gated_loudness(np.concatenate([np.zeros(0), *before]))
    applied = [g is not None for g in plan.gains]
    if target is None or not any(applied):
        return replace(plan, makeup=tuple(None if g is None else 0.0 for g in plan.gains))
    after = [np.asarray(c if a else s, dtype=float)
             for s, c, a in zip(series, compressed, applied, strict=True)]  # fmt: skip

    def gains(m: float) -> list[float | None]:
        return [None if g is None else min(m, p.max_boost - g) for g in plan.gains]

    def loudness(m: float) -> float:
        shifted = [b + (g or 0.0) for b, g in zip(after, gains(m), strict=True)]
        return gated_loudness(np.concatenate([np.zeros(0), *shifted])) or -120.0

    grid = np.arange(-p.max_cut, p.max_boost + _GAIN_STEP / 2, _GAIN_STEP)
    best = float(grid[int(np.argmin([abs(loudness(m) - target) + 0.01 * abs(m) for m in grid]))])
    return replace(plan, makeup=tuple(None if g is None else round(g, 2) for g in gains(best)))


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
        (:func:`bt709_filter`, then any :func:`lut_filter`) goes just before
        ``format=yuv420p``.
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
    peaks: str = "",
    grade: Grade | None = None,
) -> str:
    """Frame-accurate re-encode of [start, end] (absolute source time) into ``out``.

    ``audio_fade`` > 0 bakes a short audio fade-in/out into the clip (the
    video stays a hard cut), so a cut never slaps from sound to silence.
    ``peaks`` (:meth:`PeakPlan.filter`) compresses and levels the audio before the fade;
    a true-peak limiter (:data:`CLIP_LIMIT_DB`) then ends the chain.
    ``grade`` (:class:`Grade`, e.g. a :class:`Lut`) grades the video after the BT.709
    conversion.
    ``out`` only appears once the clip is complete (:func:`atomic_output`); a failed
    auto-chosen GPU encode is redone on the CPU (:func:`encode_with_fallback`).

    Returns:
        ffmpeg's stderr (check it for read errors; ffmpeg exits 0 after them).
    """
    with atomic_output(out) as part:
        return _export(
            src, info, start, end, part, fmt, vertical, crop, audio_fade, out.name, peaks, grade
        )


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
    peaks: str = "",
    grade: Grade | None = None,
) -> str:
    """:func:`export_clip` straight into ``out`` (no temporary name); ``name`` is for logs."""
    warn_if_log(src, info, start)
    return encode_with_fallback(
        lambda enc: _encode(
            enc, src, info, start, end, out, fmt, vertical, crop, audio_fade, peaks, grade
        ),
        name,
    )


def edge_keyframes(enc: VideoEncoder, dur: float) -> list[str]:
    """``-force_key_frames`` at :data:`EDGE_KEYFRAMES_S` from the start and the end of a
    ``dur`` s clip (times inside it only).

    They must be IDR frames, as only those are keyframes in the MP4: NVENC needs
    ``-forced-idr 1``; libx264 (closed GOP) and VAAPI make every forced keyframe IDR. AMF
    makes them plain I frames (only the size cost), so it gets none: its clips keep only the
    encoder's own keyframes, which ``assemble`` reads instead (larger re-encoded windows, same
    result).
    """
    if enc.kind == "amf":
        return []
    times = sorted({t for d in EDGE_KEYFRAMES_S for t in (d, dur - d) if 0 < t < dur})
    if not times:
        return []
    idr = ["-forced-idr", "1"] if enc.kind == "nvenc" else []
    return ["-force_key_frames", ",".join(f"{t:.3f}" for t in times), *idr]


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
    peaks: str = "",
    grade: Grade | None = None,
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
    color = bt709_filter(info)
    chain = ",".join(f for f in (color, grade.filter() if grade else "") if f)
    vf = f"[0:v:0]{enc.chain(fmt.video_filter(vertical, crop, chain))}[v]"
    audio = None if info.audio_duration is None else min(end, info.audio_duration) - start
    a_fmt = A_FMT + peaks + audio_fade_filter(dur, audio_fade, audio)
    if peaks:
        a_fmt += "," + limiter_chain(CLIP_LIMIT_DB, has_soxr())
    if info.audio_streams == 0:
        args += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", SILENCE]
        af = f"[1:a]{a_fmt}[a]"
    else:  # several tracks are summed (e.g. ShadowPlay game + mic), duplicates left out
        length = info.audio_duration or info.duration
        tracks = export_tracks(src, info.audio_streams, length, (start, dur))
        af = f"{mix_head(tracks)}{a_fmt}[a]"
    args += [
        "-filter_complex",
        f"{vf};{af}",
        "-map",
        "[v]",
        "-map",
        "[a]",
        *enc.args(),
        *edge_keyframes(enc, dur),
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
        *(["-metadata", f"comment={PEAK_TAG}"] if peaks else []),
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
    peaks: str = "",
    grade: Grade | None = None,
) -> dict[str, Any] | None:
    """:func:`export_clip` (``peaks`` and ``grade`` as there), guarded against silent
    truncation by read errors.

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
        return _export_checked(
            src, info, start, end, part, fmt, vertical, crop, audio_fade, out, peaks, grade
        )


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
    peaks: str = "",
    grade: Grade | None = None,
) -> dict[str, Any] | None:
    """:func:`export_clip_checked` writing to ``part``; ``out`` names the clip in logs."""
    expected = end - start
    messages: list[str] = []
    for attempt in (1, 2):
        text = _export(
            src, info, start, end, part, fmt, vertical, crop, audio_fade, out.name, peaks, grade
        )
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
    _export(
        src, info, start, safe_end, part, fmt, vertical, crop, audio_fade, out.name, peaks, grade
    )
    logger.warning("  %s trimmed to end at %.2f (before the unreadable spot)", out.name, safe_end)
    return {
        "failure_time": round(fail_t, 3),
        "messages": messages,
        "retried": True,
        "end": round(safe_end, 3),
    }
