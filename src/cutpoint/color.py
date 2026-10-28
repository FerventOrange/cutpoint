"""Automatic colour correction of exported clips (#290 stage 2: ``auto`` and ``underwater``).

A clip is measured on up to :data:`SAMPLE_FRAMES` keyframes of its inner 80 %
(:func:`sample_stats`), seen through the same crop and BT.709 conversion as its export, and
:func:`estimate` turns those frames into an :class:`AutoGrade`: a levels stretch (from the
luma; per channel underwater), a white balance, a red-from-green channel mix (underwater only)
and a vibrance boost, every one clamped. The grade is ffmpeg filters only
(:meth:`AutoGrade.filter`: ``colorchannelmixer``, ``colorlevels`` and ``vibrance`` in 16-bit
RGB), so the export needs nothing new.

Before a grade is used, :func:`simulate` applies it to the sampled frames in numpy and
:func:`metrics` compares them before and after: a grade that clips more than its levels
intend, leaves a worse cast, or adds or takes away too much colour is vetoed. Clips that
already look good, log or flat footage from a 10-bit source (which needs the camera's LUT,
``--lut``) and, for ``underwater``, clips that do not look underwater get no grade.
:func:`auto_grade` caches each result in the source's cache folder (``color/<hash>.json``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.crop import Crop
from cutpoint.encode import (
    TAG_BT709,
    OutputFormat,
    bt709_filter,
    is_high_bit_sdr,
    log_stats,
    looks_log,
)
from cutpoint.ffmpeg_utils import FFmpegError, MediaInfo, decode_raw

logger = logging.getLogger(__name__)

# Bump when the measurement or estimate changes: it keys the cache and is recorded per clip.
# 2: per-channel levels for ``underwater``, the M3 drop veto and the relative cast veto.
# 3: ``underwater`` grades only clips that look underwater, vetoes a washed-out result by its
# saturation instead of its M3 and allows red-only highlight clipping (#494).
COLOR_VERSION = 3
MODES = ("auto", "underwater")
# Black and white points: from the luma, the same for every channel, or per channel. Each
# mode uses its own (MODE_LEVELS): ``auto`` the luma's, so a clip's overall balance is left to
# the white balance; ``underwater`` each channel's, which stretches the red the water took
# (``levels=`` of :func:`estimate` overrides it, for scripts/color_eval.py).
LEVELS = ("luma", "channel")
MODE_LEVELS = {"auto": "luma", "underwater": "channel"}

# Sampling: up to this many keyframes of the inner 80 % of a clip, this wide.
SAMPLE_FRAMES = 8
SAMPLE_WIDTH = 256
SAMPLE_INNER = 0.8
SAMPLE_TIMEOUT_S = 120
# Frames kept while decoding before thinning to every other one (bounded memory on long clips).
_SAMPLE_KEEP = 2 * SAMPLE_FRAMES

# Levels: luma percentiles, clamped (black at most, white at least, stretch at most).
LEVELS_PERCENTILES = (0.5, 99.5)
MAX_BLACK = 0.06
MIN_WHITE = 0.88
MAX_STRETCH = 1.25
# White balance: near-neutral mid-tones (luma range, HSV saturation below) of at least
# MIN_NEUTRAL of the pixels; WB_AMOUNT of the measured cast is corrected, each gain within
# 1 +- MAX_GAIN (underwater: red up to 1 + MAX_RED_GAIN, gray-world over the whole frame).
NEUTRAL_LUMA = (0.2, 0.8)
NEUTRAL_MAX_SAT = 0.25
MIN_NEUTRAL = 0.02
WB_AMOUNT = 0.6
MAX_GAIN = 0.12
MAX_RED_GAIN = 0.60
# Underwater: red borrowed from green (red = (1 - mix) red + mix green), mix at most MAX_MIX,
# aiming the frame's mean red/green at MIX_TARGET before the white balance.
MAX_MIX = 0.35
MIX_TARGET = 0.6
# Vibrance from the Hasler-Suesstrunk M3 colourfulness (0-255 scale): MAX_VIBRANCE at or
# below M3_FLAT ("slightly colourful"), none from M3_FULL ("moderately colourful") on.
MAX_VIBRANCE = 0.2
M3_FLAT = 15.0
M3_FULL = 33.0
# Already good: every gain within GOOD_GAIN of 1, levels within GOOD_LEVELS of full, M3 at
# least M3_FULL.
GOOD_GAIN = 0.03
GOOD_LEVELS = 0.03
# A clip that looks underwater (the gate of ``underwater``, and plain ``auto``'s hint): the
# frames' mean red below UNDERWATER_RG of their mean green and UNDERWATER_RB of their mean blue,
# and the green above UNDERWATER_GB of the blue. On the #489 windows the six scuba clips had
# R/G 0.60-0.73, the others 0.80 and up but for a blue-sky ski clip (R/G 0.66), which the G/B
# rule leaves out (G/B 0.72; scuba 0.86 and up). Water bluer than that (open ocean) does not
# pass either.
UNDERWATER_RG = 0.77
UNDERWATER_RB = 0.85
UNDERWATER_GB = 0.79
# Veto (:func:`_veto`): the clipped share rises by more than VETO_CLIPPED above the clipping
# the levels intend (:func:`_intended_clipping`: the source's own plus at most the
# LEVELS_PERCENTILES tails); the gray-world residual (a share of the luma range) gets worse;
# M3 rises by more than VETO_M3_RISE, or (``auto``) falls below VETO_M3_DROP of its value (a
# picture washed out to grey). ``underwater`` judges a washed-out result by the median HSV
# saturation instead (M3 counts the cast it removes as colour): below VETO_SATURATION_DROP of
# its value (flat cyan sand turned grey: 0.57-0.58 on the #489 scuba clips, the good grades
# 0.73-0.80). ``underwater`` also does not count pixels whose red alone clips high (the red
# gain on a bright highlight), up to UNDERWATER_RED_CLIP of the frame, as clipping; pixels
# clipped in the source and the red tail the levels intend are left out of that allowance, but
# green/blue tail pixels the gains hold below 1 while red clips stay in it (an overlap with the
# intended clipping of at most the green/blue tails).
# The residual is worse when it rises by more than VETO_RESIDUAL_EPS (0.5 % of the luma range,
# about a code), or by more than VETO_RESIDUAL_RATIO times and more than VETO_RESIDUAL_FLOOR.
# The relative rule catches a near-neutral clip given a cast (a GoPro clip went 0.0017 to
# 0.0048: 2.8 times, +0.0031, within the absolute rule); its floor, half a code of 8-bit
# (0.5 / 255), keeps rounding noise on an already neutral clip from vetoing it.
VETO_CLIPPED = 0.005
VETO_RESIDUAL_EPS = 0.005
VETO_RESIDUAL_RATIO = 1.5
VETO_RESIDUAL_FLOOR = 0.002
VETO_M3_RISE = 0.25
VETO_M3_DROP = 0.85
VETO_SATURATION_DROP = 0.63
UNDERWATER_RED_CLIP = 0.03
# Vibrance is kept this far below the M3 veto, so it never vetoes the grade by itself.
_M3_MARGIN = 0.98
_VIBRANCE_STEPS = 10

# BT.709 luma weights (the frames are BT.709 RGB).
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
# Explicit BT.709 conversions, as :func:`cutpoint.encode.lut_filter`'s (an older ffmpeg's
# automatic one would use BT.601).
_TO_RGB = "scale=in_color_matrix=bt709:in_range=tv:out_range=pc"
_TO_YUV = "scale=out_color_matrix=bt709:out_range=tv"
_RGB16 = "format=gbrp16le"
# ``vibrance`` with BT.709 luma weights (its defaults give red and blue each other's) in
# ``alternate`` mode: a channel moves away from the luma by 1 + v * (1 - saturation), so the
# least saturated colours gain most (ffmpeg 6.1's default mode uses 1 + v * (1 + saturation),
# boosting the saturated ones most, like a plain saturation that clips sooner).
_VIBRANCE_OPTS = "rlum=0.2126:glum=0.7152:blum=0.0722:alternate=1"

# Skip reasons (``skip`` of an :class:`AutoGrade`).
SKIP_GOOD = "good"
SKIP_LOG = "log"
SKIP_360 = "360"
SKIP_NO_FRAMES = "no_frames"
SKIP_NOT_UNDERWATER = "not_underwater"
VETO_CLIPPING = "veto_clipped"
VETO_CAST = "veto_cast"
VETO_COLOURFUL = "veto_m3"
VETO_M3_LOSS = "veto_m3_drop"  # auto
VETO_FADED = "veto_faded"  # underwater


@dataclass(frozen=True)
class Metrics:
    """What a grade is judged by, over a clip's sampled frames (see :func:`metrics`).

    ``m3``: Hasler-Suesstrunk colourfulness (0-255 scale); ``residual``: gray-world cast,
    the largest distance of a channel mean from their average over the luma range;
    ``clipped``: share of pixels with a channel at 0 or 255; ``black`` / ``white``: luma
    p0.5 / p99.5 (0-1); ``saturation``: mean HSV saturation (None in a record of version 2 or
    older).
    """

    m3: float
    residual: float
    clipped: float
    black: float
    white: float
    saturation: float | None = None

    def to_json(self) -> dict[str, float]:
        """The metrics as a manifest dict."""
        return {
            "m3": self.m3,
            "residual": self.residual,
            "clipped": self.clipped,
            "black": self.black,
            "white": self.white,
            "saturation": self.saturation,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> Metrics | None:
        """The metrics :meth:`to_json` wrote, None for none."""
        if not data:
            return None
        sat = data.get("saturation")
        return cls(
            *(float(data[k]) for k in ("m3", "residual", "clipped", "black", "white")),
            saturation=None if sat is None else float(sat),
        )


@dataclass(frozen=True)
class AutoGrade:
    """An automatic grade (``--color auto`` / ``underwater``) and the strength (0-1) it is
    applied at.

    The parameters are the full-strength estimate (rounded); :meth:`applied` folds the
    strength in, so a weaker grade is the same filters with milder values, not a blend.
    ``skip`` names why the clip gets no grade (:data:`SKIP_GOOD`, a veto, ...); a skipped
    grade keeps the parameters it would have had, for the record. ``before`` and
    ``predicted`` (the :func:`metrics` of the sampled frames and of their simulated grade)
    are recorded only: two grades with the same parameters are equal.
    """

    mode: str
    strength: float = 1.0
    black: float | tuple[float, float, float] = 0.0
    white: float | tuple[float, float, float] = 1.0
    gains: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mix: float = 0.0
    vibrance: float = 0.0
    skip: str | None = None
    before: Metrics | None = field(default=None, compare=False)
    predicted: Metrics | None = field(default=None, compare=False)
    # Plain ``auto`` only: the sampled frames look underwater (:func:`looks_underwater`), for
    # an INFO hint; it never changes the grade and is not recorded in manifest.json.
    underwater_hint: bool = field(default=False, compare=False)

    def applied(self) -> AutoGrade:
        """This grade with the strength folded into the parameters (and strength 1)."""
        s = min(max(self.strength, 0.0), 1.0)
        return replace(
            self,
            strength=1.0,
            black=_per(self.black, lambda b: b * s),
            white=_per(self.white, lambda w: 1.0 - (1.0 - w) * s),
            gains=tuple(1.0 + (g - 1.0) * s for g in self.gains),  # type: ignore[arg-type]
            mix=self.mix * s,
            vibrance=self.vibrance * s,
        )

    def filter(self) -> str:
        """Filters grading a BT.709 tv-range clip, for :meth:`OutputFormat.video_filter`'s
        ``color`` after :func:`~cutpoint.encode.bt709_filter`'s; "" for a skipped grade, at
        strength 0 or when nothing changes.

        Like :func:`~cutpoint.encode.lut_filter`: the frames go to full-range RGB with the
        BT.709 matrix (16-bit, ``gbrp16le``), through ``colorchannelmixer`` (underwater),
        ``colorlevels`` (levels and white balance in one) and ``vibrance``, back to BT.709 tv
        range, and are tagged BT.709 tv range.
        """
        if self.skip or self.strength <= 0:
            return ""
        g = self.applied()
        steps = []
        if g.mix > 0:
            steps.append(f"colorchannelmixer=rr={1.0 - g.mix:.5f}:rg={g.mix:.5f}")
        if max(_rgb(g.black)) > 0 or min(_rgb(g.white)) < 1 or any(k != 1 for k in g.gains):
            steps.append(_levels(g.black, g.white, g.gains))
        if g.vibrance > 0:
            steps.append(f"vibrance=intensity={g.vibrance:.5f}:{_VIBRANCE_OPTS}")
        if not steps:
            return ""
        return f"{_TO_RGB},{_RGB16},{','.join(steps)},{_TO_YUV},{TAG_BT709}:range=tv"

    def params(self) -> dict[str, Any]:
        """The full-strength parameters as a manifest dict."""
        return {
            "black": _json_point(self.black),
            "white": _json_point(self.white),
            "gains": list(self.gains),
            "mix": self.mix,
            "vibrance": self.vibrance,
        }

    def to_json(self) -> dict[str, Any]:
        """manifest.json key of a clip graded (or skipped) with it: ``color``."""
        return {
            "color": {
                "method": self.mode,
                "version": COLOR_VERSION,
                "strength": self.strength,
                "params": self.params(),
                "skip": self.skip,
                "before": self.before.to_json() if self.before else None,
                "predicted": self.predicted.to_json() if self.predicted else None,
            }
        }

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> AutoGrade | None:
        """The grade a manifest entry records (:meth:`to_json`), None for a clip without one.
        The recorded parameters are used as they are: nothing is measured again."""
        color = entry.get("color")
        if not isinstance(color, dict) or color.get("method") not in MODES:
            return None
        p = color.get("params") or {}
        gains = p.get("gains") or (1.0, 1.0, 1.0)
        return cls(
            mode=color["method"],
            strength=float(color.get("strength", 1.0)),
            black=_read_point(p.get("black", 0.0)),
            white=_read_point(p.get("white", 1.0)),
            gains=(float(gains[0]), float(gains[1]), float(gains[2])),
            mix=float(p.get("mix", 0.0)),
            vibrance=float(p.get("vibrance", 0.0)),
            skip=color.get("skip"),
            before=Metrics.from_json(color.get("before")),
            predicted=Metrics.from_json(color.get("predicted")),
        )

    @classmethod
    def skipped(cls, mode: str, reason: str, strength: float = 1.0) -> AutoGrade:
        """No grade for a clip, and why (for example :data:`SKIP_360`)."""
        return cls(mode=mode, strength=strength, skip=reason)


def _rgb(point: float | tuple[float, float, float]) -> tuple[float, float, float]:
    """A black or white point as one value per channel (r, g, b)."""
    if isinstance(point, tuple | list):
        return float(point[0]), float(point[1]), float(point[2])
    return float(point), float(point), float(point)


def _per(point: float | tuple[float, float, float], f: Any) -> Any:
    """``f`` applied to a point, per channel for a per-channel one."""
    if isinstance(point, tuple):
        return tuple(f(v) for v in point)
    return f(point)


def _json_point(point: float | tuple[float, float, float]) -> float | list[float]:
    """A point for manifest.json: a number, or ``[r, g, b]`` for a per-channel one."""
    return list(point) if isinstance(point, tuple) else point


def _read_point(value: Any) -> float | tuple[float, float, float]:
    """A point :func:`_json_point` wrote."""
    if isinstance(value, list | tuple):
        return float(value[0]), float(value[1]), float(value[2])
    return float(value)


def _levels(
    black: float | tuple[float, float, float],
    white: float | tuple[float, float, float],
    gains: tuple[float, float, float],
) -> str:
    """``colorlevels`` mapping each channel to ``(x - black) * gain / (white - black)`` (a
    per-channel point gives each channel its own).

    The input white of a channel is where that line reaches 1; a gain below 1 can put it
    above 1 (outside the filter's range), so the output white is lowered instead.
    """
    opts = []
    for c, gain, lo, hi in zip("rgb", gains, _rgb(black), _rgb(white), strict=True):
        k = gain / (hi - lo)
        imax, omax = lo + 1.0 / k, 1.0
        if imax > 1.0:
            imax, omax = 1.0, (1.0 - lo) * k
        opts.append(f"{c}imin={lo:.5f}:{c}imax={imax:.5f}:{c}omax={omax:.5f}")
    return "colorlevels=" + ":".join(opts)


# -- sampling -----------------------------------------------------------------------------------


@dataclass
class ColorStats:
    """Sampled frames of a clip: ``frames`` is ``(n, height, width, 3)`` BT.709 RGB ``uint8``.

    ``high_bit``: the source is 10-bit or deeper and gets no HDR conversion, the sources
    :func:`~cutpoint.encode.warn_if_log` checks; only those can be skipped as log (8-bit
    footage that looks flat is graded, the veto guarding it).
    """

    frames: np.ndarray
    high_bit: bool = False


def sample_filter(
    info: MediaInfo, fmt: OutputFormat, vertical: bool, crop: Crop | None = None
) -> tuple[str, tuple[int, int]]:
    """The sampling chain of a clip and the frame size it outputs.

    It mirrors :meth:`OutputFormat.video_filter` without the padding (bars would count as
    black): the same crop (trimmed to the output aspect, or the vertical crop), a scale to
    :data:`SAMPLE_WIDTH` px wide in tv range, the clip's :func:`bt709_filter` and an explicit
    BT.709 conversion to ``rgb24``.
    """
    pre, (w, h) = "", (info.width, info.height)
    if crop is not None:
        box = crop if vertical else crop.fit_aspect(fmt.width / fmt.height)
        pre, (w, h) = f"crop={box.w}:{box.h}:{box.x}:{box.y},", (box.w, box.h)
    if vertical:
        pre += "crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',"
        w, h = 9, 16
    height = max(2, round(SAMPLE_WIDTH * h / w / 2) * 2)
    color = bt709_filter(info)
    color = f"{color}," if color else ""
    chain = (
        f"{pre}scale={SAMPLE_WIDTH}:{height}:flags=area:out_range=tv,{color}{_TO_RGB},format=rgb24"
    )
    return chain, (SAMPLE_WIDTH, height)


def sample_stats(
    src: Path,
    info: MediaInfo,
    start: float,
    end: float,
    fmt: OutputFormat,
    vertical: bool = False,
    crop: Crop | None = None,
) -> ColorStats:
    """Up to :data:`SAMPLE_FRAMES` keyframes, evenly spread, of the inner 80 % of
    ``[start, end]`` (source time), through :func:`sample_filter`'s chain; one CPU pass.

    A clip with no keyframe in there gets its middle frame (a second, short pass).

    Raises:
        FFmpegError: If no frame can be decoded.
    """
    vf, size = sample_filter(info, fmt, vertical, crop)
    dur = max(end - start, 0.0)
    inner = dur * SAMPLE_INNER
    kept: list[np.ndarray] = []
    stride, k = 1, 0
    try:
        for _, frame in decode_raw(
            src,
            size=size,
            vf=vf,
            start=start + (dur - inner) / 2,
            duration=inner,
            keyframes_only=True,
            hwaccel=None,
            timeout=SAMPLE_TIMEOUT_S,
        ):
            if k % stride == 0:
                kept.append(frame)
                if len(kept) >= _SAMPLE_KEEP:
                    kept, stride = kept[::2], stride * 2
            k += 1
    except FFmpegError as err:
        logger.debug(
            "sampling of %s %.2f-%.2f s stopped after %d frames: %s", src.name, start, end, k, err
        )
    if not kept:
        frames = decode_raw(
            src,
            size=size,
            vf=vf,
            start=start + dur / 2,
            duration=1.0,
            hwaccel=None,
            timeout=SAMPLE_TIMEOUT_S,
        )
        try:
            kept = [next(frames)[1]]
        finally:
            frames.close()
    pick = np.unique(np.linspace(0, len(kept) - 1, min(SAMPLE_FRAMES, len(kept))).round())
    return ColorStats(np.stack([kept[int(i)] for i in pick]), is_high_bit_sdr(info))


# -- measurement, simulation and estimate --------------------------------------------------------


def _float(frames: np.ndarray) -> np.ndarray:
    """``(n, h, w, 3)`` frames as float32 0-1 (``uint8`` is divided by 255)."""
    if frames.dtype == np.uint8:
        return frames.astype(np.float32) / 255.0
    return frames.astype(np.float32, copy=False)


def _m3(rgb: np.ndarray) -> float:
    """Hasler-Suesstrunk M3 of one ``(h, w, 3)`` 0-1 frame, on their 0-255 scale."""
    r, g, b = (rgb[..., i].ravel() * 255.0 for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def _saturation(rgb: np.ndarray) -> float:
    """Mean HSV saturation, ``(max - min) / max``, of one ``(h, w, 3)`` 0-1 frame."""
    hi = rgb.max(axis=-1)
    return float(((hi - rgb.min(axis=-1)) / np.maximum(hi, 1e-6)).mean())


def _red_highlights(frames: np.ndarray, intended: np.ndarray | None = None) -> float:
    """Share of the pixels of ``frames`` (``uint8``, or float 0-1) whose red alone is
    clipped high: red at 255, green and blue below it, no channel at 0. Pixels set in
    ``intended`` (an ``(n, h, w)`` mask, :func:`_intended_high`) are left out: the intended
    clipping already counts them (it may also count some of the rest, see
    :func:`_intended_high`)."""
    rgb = _float(frames)
    high = rgb >= 254.5 / 255
    low = (rgb <= 0.5 / 255).any(axis=-1)
    red = high[..., 0] & ~high[..., 1] & ~high[..., 2] & ~low
    if intended is not None:
        red &= ~intended
    return round(float(red.mean()), 5)


def metrics(frames: np.ndarray) -> Metrics:
    """:class:`Metrics` of ``(n, h, w, 3)`` frames (``uint8``, or float 0-1): the median over
    the frames, the clipped share the mean."""
    rgb = _float(frames)
    m3, residual, black, white, clipped, sat = [], [], [], [], [], []
    for f in rgb:
        luma = f @ LUMA
        lo, hi = np.percentile(luma, LEVELS_PERCENTILES)
        means = f.reshape(-1, 3).mean(axis=0)
        m3.append(_m3(f))
        residual.append(float(np.abs(means - means.mean()).max()) / max(hi - lo, 0.05))
        black.append(lo)
        white.append(hi)
        ends = (f <= 0.5 / 255) | (f >= 254.5 / 255)
        clipped.append(float(ends.any(axis=-1).mean()))
        sat.append(_saturation(f))
    return Metrics(
        m3=round(float(np.median(m3)), 3),
        residual=round(float(np.median(residual)), 4),
        clipped=round(float(np.mean(clipped)), 5),
        black=round(float(np.median(black)), 4),
        white=round(float(np.median(white)), 4),
        saturation=round(float(np.median(sat)), 4),
    )


def _black_span(g: AutoGrade) -> tuple[Any, Any]:
    """The black point and ``white - black`` of ``g`` for numpy: floats, or float32 arrays
    (r, g, b) for per-channel points."""
    if not isinstance(g.black, tuple) and not isinstance(g.white, tuple):
        return g.black, g.white - g.black
    lo, hi = _rgb(g.black), _rgb(g.white)
    span = [w - b for b, w in zip(lo, hi, strict=True)]
    return np.asarray(lo, np.float32), np.asarray(span, np.float32)


def simulate(frames: np.ndarray, grade: AutoGrade) -> np.ndarray:
    """``frames`` (``uint8``, or float 0-1) as :meth:`AutoGrade.filter` grades them, in
    float 0-1: the strength folded in, each filter's result clipped as ffmpeg stores it."""
    g = grade.applied()
    rgb = _float(frames).copy()
    if g.mix > 0:
        rgb[..., 0] = np.clip((1.0 - g.mix) * rgb[..., 0] + g.mix * rgb[..., 1], 0.0, 1.0)
    lo, span = _black_span(g)
    k = np.asarray(g.gains, np.float32) / span
    rgb = np.clip((rgb - lo) * k, 0.0, 1.0)
    if g.vibrance > 0:
        # vf_vibrance: each channel moves away from the luma by 1 + v * (1 - saturation)
        sat = rgb.max(axis=-1, keepdims=True) - rgb.min(axis=-1, keepdims=True)
        luma = (rgb @ LUMA)[..., None]
        rgb = np.clip(luma + (rgb - luma) * (1.0 + g.vibrance * (1.0 - sat)), 0.0, 1.0)
    return rgb


def is_log(stats: ColorStats) -> bool:
    """True when a high-bit source's frames (``stats.high_bit``, the rule of
    :func:`~cutpoint.encode.warn_if_log`) look like log or flat footage
    (:func:`~cutpoint.encode.looks_log` on the median of their
    :func:`~cutpoint.encode.log_stats`)."""
    if not stats.high_bit:
        return False
    sat, black, white = np.median([log_stats(f) for f in stats.frames], axis=0)
    return looks_log(float(sat), float(black), float(white))


def looks_underwater(stats: ColorStats) -> bool:
    """True when the frames' mean red is below :data:`UNDERWATER_RG` of their mean green and
    :data:`UNDERWATER_RB` of their mean blue, and the green above :data:`UNDERWATER_GB` of the
    blue: the clips ``underwater`` grades (:data:`SKIP_NOT_UNDERWATER` for the others) and
    plain ``auto`` names in its hint (where it never changes the grade)."""
    r, g, b = _float(stats.frames).reshape(-1, 3).mean(axis=0)
    return bool(r < UNDERWATER_RG * g and r < UNDERWATER_RB * b and g > UNDERWATER_GB * b)


def _mix(rgb: np.ndarray) -> float:
    """Underwater red-from-green mix aiming the median red/green at :data:`MIX_TARGET`."""
    means = rgb.reshape(len(rgb), -1, 3).mean(axis=1)
    ratio = float(np.median(means[:, 0] / np.maximum(means[:, 1], 1e-4)))
    if ratio >= MIX_TARGET:
        return 0.0
    return min(MAX_MIX, (MIX_TARGET - ratio) / (1.0 - ratio))


def _clamp_points(lo: float, hi: float) -> tuple[float, float]:
    """Black and white points ``lo`` / ``hi`` clamped (see :data:`MAX_BLACK`,
    :data:`MIN_WHITE`, :data:`MAX_STRETCH`; a stretch above it moves both points back
    toward 0 and 1 in proportion)."""
    black, white = min(float(lo), MAX_BLACK), max(float(hi), MIN_WHITE)
    if white - black < 1.0 / MAX_STRETCH:
        t = (1.0 - 1.0 / MAX_STRETCH) / (black + 1.0 - white)
        black, white = black * t, 1.0 - (1.0 - white) * t
    return black, white


def _levels_points(rgb: np.ndarray) -> tuple[float, float]:
    """Black and white points: the median luma p0.5 / p99.5, clamped (:func:`_clamp_points`)."""
    lo, hi = np.median([np.percentile(f @ LUMA, LEVELS_PERCENTILES) for f in rgb], axis=0)
    return _clamp_points(lo, hi)


def _channel_points(
    rgb: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Per-channel black and white points (``levels="channel"``, ``underwater``'s): each
    channel's median p0.5 / p99.5, each pair clamped as :func:`_levels_points` clamps the
    luma's."""
    per = np.median(
        [np.percentile(f.reshape(-1, 3), LEVELS_PERCENTILES, axis=0) for f in rgb], axis=0
    )
    points = [_clamp_points(per[0, c], per[1, c]) for c in range(3)]
    black = (points[0][0], points[1][0], points[2][0])
    white = (points[0][1], points[1][1], points[2][1])
    return black, white


def _gains(rgb: np.ndarray, black: float, underwater: bool) -> tuple[float, float, float]:
    """White balance gains (r, g, b) on the black-subtracted frames.

    ``auto``: from the near-neutral mid-tones (:data:`NEUTRAL_LUMA`, :data:`NEUTRAL_MAX_SAT`)
    of each frame with at least :data:`MIN_NEUTRAL` of them; none if fewer than that share of
    all pixels are neutral. ``underwater``: gray-world over each whole frame. Each gain
    corrects :data:`WB_AMOUNT` of the cast (the median over the frames) and is clamped.
    """
    per_frame = []
    neutral = 0
    for f in rgb:
        px = f.reshape(-1, 3)
        if underwater:
            mask = np.ones(len(px), bool)
        else:
            luma = px @ LUMA
            hi = px.max(axis=1)
            sat = (hi - px.min(axis=1)) / np.maximum(hi, 1e-6)
            mask = (luma >= NEUTRAL_LUMA[0]) & (luma <= NEUTRAL_LUMA[1]) & (sat < NEUTRAL_MAX_SAT)
            neutral += int(mask.sum())
            if mask.mean() < MIN_NEUTRAL:
                continue
        means = np.maximum(px[mask].mean(axis=0) - black, 1e-4)
        per_frame.append(means.mean() / means)
    if not per_frame or (not underwater and neutral < MIN_NEUTRAL * rgb[..., 0].size):
        return 1.0, 1.0, 1.0
    full = np.median(per_frame, axis=0)
    gains = 1.0 + WB_AMOUNT * (full - 1.0)
    hi = np.array([MAX_RED_GAIN if underwater else MAX_GAIN, MAX_GAIN, MAX_GAIN])
    gains = np.clip(gains, 1.0 - MAX_GAIN, 1.0 + hi)
    return float(gains[0]), float(gains[1]), float(gains[2])


def _vibrance(frames: np.ndarray, grade: AutoGrade, before: Metrics) -> float:
    """Vibrance intensity of ``grade`` (whose own vibrance is 0) from the M3 of its result:
    :data:`MAX_VIBRANCE` at :data:`M3_FLAT`, none from :data:`M3_FULL`, linear between; then
    lowered until the predicted M3 stays below the veto (:data:`VETO_M3_RISE`)."""
    m3 = metrics(simulate(frames, grade)).m3
    want = MAX_VIBRANCE * min(max((M3_FULL - m3) / (M3_FULL - M3_FLAT), 0.0), 1.0)
    cap = before.m3 * (1.0 + VETO_M3_RISE) * _M3_MARGIN
    if want <= 0 or m3 >= cap:
        return 0.0
    if metrics(simulate(frames, replace(grade, vibrance=want))).m3 <= cap:
        return want
    lo, hi = 0.0, want
    for _ in range(_VIBRANCE_STEPS):
        mid = (lo + hi) / 2
        if metrics(simulate(frames, replace(grade, vibrance=mid))).m3 <= cap:
            lo = mid
        else:
            hi = mid
    return lo


def _round(grade: AutoGrade) -> AutoGrade:
    """``grade`` with its parameters rounded to 4 decimals (they go to the manifest)."""
    return replace(
        grade,
        black=_per(grade.black, lambda v: round(v, 4)),
        white=_per(grade.white, lambda v: round(v, 4)),
        gains=tuple(round(g, 4) for g in grade.gains),  # type: ignore[arg-type]
        mix=round(grade.mix, 4),
        vibrance=round(grade.vibrance, 4),
    )


def _intended_clipping(frames: np.ndarray, grade: AutoGrade) -> float:
    """The clipped share (as :func:`metrics` counts it) that ``grade``'s levels are meant to
    cause on ``frames``: the pixels already clipped, plus those whose luma (after the
    channel mix) the applied black and white points push to 0 or 1, at most the
    :data:`LEVELS_PERCENTILES` tail at each end (ties at a point and saturated colours a
    channel pushes out are not intended). A clamped point intends less than its tail."""
    g = grade.applied()
    rgb = _float(frames)
    mixed = simulate(rgb, AutoGrade(g.mode, mix=g.mix)) if g.mix > 0 else rgb
    tails = (LEVELS_PERCENTILES[0] / 100, 1.0 - LEVELS_PERCENTILES[1] / 100)
    shares = []
    per_channel = isinstance(g.black, tuple) or isinstance(g.white, tuple)
    lo, span = _black_span(g)
    for f, m in zip(rgb, mixed, strict=True):
        clipped = ((f <= 0.5 / 255) | (f >= 254.5 / 255)).any(axis=-1)
        if per_channel:  # each channel's points intend its own tails
            v = (m - lo) / span
            low_c = ~clipped[..., None] & (v <= 0.5 / 255)
            high_c = ~clipped[..., None] & (v >= 254.5 / 255)
            low = min(
                float(low_c.any(axis=-1).mean()),
                sum(min(float(low_c[..., c].mean()), tails[0]) for c in range(3)),
            )
            high = min(
                float(high_c.any(axis=-1).mean()),
                sum(min(float(high_c[..., c].mean()), tails[1]) for c in range(3)),
            )
            shares.append(float(clipped.mean()) + low + high)
            continue
        v = (m @ LUMA - g.black) / (g.white - g.black)
        low = float((~clipped & (v <= 0.5 / 255)).mean())
        high = float((~clipped & (v >= 254.5 / 255)).mean())
        shares.append(float(clipped.mean()) + min(low, tails[0]) + min(high, tails[1]))
    return round(float(np.mean(shares)), 5)


def _intended_high(frames: np.ndarray, grade: AutoGrade) -> np.ndarray:
    """Mask ``(n, h, w)`` of the pixels of ``frames`` that :func:`_intended_clipping` may
    count as clipped high in red: those already clipped, and those whose red (the luma, for
    one white point) after the channel mix the applied white point pushes to 1. A tail beyond
    :data:`LEVELS_PERCENTILES` is counted only up to the tail there, so leaving all of it out
    of :func:`_red_highlights` never allows more than the intended clipping does. With
    per-channel points, green or blue tail pixels the gains hold below 1 while red clips are
    not in the mask, though :func:`_intended_clipping` counts them."""
    g = grade.applied()
    rgb = _float(frames)
    mixed = simulate(rgb, AutoGrade(g.mode, mix=g.mix)) if g.mix > 0 else rgb
    clipped = ((rgb <= 0.5 / 255) | (rgb >= 254.5 / 255)).any(axis=-1)
    if isinstance(g.black, tuple) or isinstance(g.white, tuple):
        lo, span = _black_span(g)
        v = (mixed[..., 0] - lo[0]) / span[0]
    else:
        v = (mixed @ LUMA - g.black) / (g.white - g.black)
    return clipped | (v >= 254.5 / 255)


def _veto(
    before: Metrics,
    after: Metrics,
    intended: float,
    *,
    underwater: bool = False,
    red_clipped: float = 0.0,
) -> str | None:
    """The veto a predicted result earns, None for none (``underwater``: the rules of
    ``--color underwater``).

    Clipping: the p0.5 / p99.5 levels are meant to clip their tails, so only clipping more
    than :data:`VETO_CLIPPED` above ``intended`` (:func:`_intended_clipping`, never below
    ``before.clipped``) is vetoed: the white balance, mix or vibrance pushing more pixels
    out, or a mass of pixels at a levels point. Underwater, ``red_clipped`` (the predicted
    share of red-only highlight clipping, :func:`_red_highlights`, without the pixels clipped
    in the source and the red tail the levels intend) up to :data:`UNDERWATER_RED_CLIP` does
    not count; green/blue tail pixels the gains hold below 1 while red clips stay in it, so
    it can overlap ``intended`` by up to the green/blue tails.

    Cast: the residual rises by more than :data:`VETO_RESIDUAL_EPS`, or to more than
    :data:`VETO_RESIDUAL_RATIO` times its value and by more than :data:`VETO_RESIDUAL_FLOOR`.
    Colourfulness: M3 above ``1 +`` :data:`VETO_M3_RISE` times its value; ``auto``: or below
    :data:`VETO_M3_DROP` times; underwater: the saturation below :data:`VETO_SATURATION_DROP`
    times its value instead.
    """
    allowed = min(red_clipped, UNDERWATER_RED_CLIP) if underwater else 0.0
    if after.clipped - allowed - max(intended, before.clipped) > VETO_CLIPPED:
        return VETO_CLIPPING
    rise = after.residual - before.residual
    if rise > VETO_RESIDUAL_EPS or (
        after.residual > before.residual * VETO_RESIDUAL_RATIO and rise > VETO_RESIDUAL_FLOOR
    ):
        return VETO_CAST
    if after.m3 > before.m3 * (1.0 + VETO_M3_RISE):
        return VETO_COLOURFUL
    if underwater:
        if (
            before.saturation
            and after.saturation is not None
            and after.saturation < before.saturation * VETO_SATURATION_DROP
        ):
            return VETO_FADED
    elif after.m3 < before.m3 * VETO_M3_DROP:
        return VETO_M3_LOSS
    return None


def _is_good(grade: AutoGrade) -> bool:
    """True when ``grade`` would hardly change the clip: gains within :data:`GOOD_GAIN`, levels
    within :data:`GOOD_LEVELS` of full, no channel mix and no vibrance (M3 at least
    :data:`M3_FULL`)."""
    return (
        all(abs(g - 1.0) <= GOOD_GAIN for g in grade.gains)
        and max(_rgb(grade.black)) <= GOOD_LEVELS
        and min(_rgb(grade.white)) >= 1.0 - GOOD_LEVELS
        and grade.mix == 0
        and grade.vibrance == 0
    )


def estimate(
    stats: ColorStats, mode: str, strength: float = 1.0, levels: str | None = None
) -> AutoGrade:
    """The grade of a clip from its sampled frames (:func:`sample_stats`).

    Log or flat footage of a high-bit source is skipped (:data:`SKIP_LOG`, see
    :func:`is_log`). Otherwise, in order: the red-from-green
    mix (``underwater``), the levels on the mixed frames, the white balance on those levels,
    the vibrance on the result. ``underwater`` skips a clip that does not look underwater
    (:data:`SKIP_NOT_UNDERWATER`, :func:`looks_underwater`). A clip that would hardly change is
    skipped (:data:`SKIP_GOOD`); a grade the simulation at ``strength`` vetoes (:func:`_veto`) is
    skipped with the veto's name. Every result records the metrics before and predicted;
    plain ``auto`` also whether the frames look underwater (``underwater_hint``).

    ``levels`` (:data:`LEVELS`; None: the mode's, :data:`MODE_LEVELS`, what exports use):
    ``"luma"`` takes one black and white point from the luma; ``"channel"`` one per channel,
    with the white balance measured on the levelled frames. The veto's intended clipping
    follows the points (:func:`_intended_clipping`: per channel for per-channel points).

    Raises:
        ValueError: For a ``mode`` other than :data:`MODES` or ``levels`` other than
            :data:`LEVELS`.
    """
    if mode not in MODES:
        raise ValueError(f"unknown color mode {mode!r} (expected one of {list(MODES)})")
    levels = MODE_LEVELS[mode] if levels is None else levels
    if levels not in LEVELS:
        raise ValueError(f"unknown levels {levels!r} (expected one of {list(LEVELS)})")
    before = metrics(stats.frames)
    underwater = mode == "underwater"
    looks = looks_underwater(stats)
    hint = not underwater and looks
    if is_log(stats):
        skipped = AutoGrade.skipped(mode, SKIP_LOG, strength)
        return replace(skipped, before=before, underwater_hint=hint)
    rgb = _float(stats.frames)
    mix = _mix(rgb) if underwater else 0.0
    mixed = simulate(rgb, AutoGrade(mode, mix=mix)) if mix else rgb
    if levels == "channel":
        black, white = _channel_points(mixed)
        lo, span = _black_span(AutoGrade(mode, black=black, white=white))
        gains = _gains((mixed - lo) / span, 0.0, underwater)
    else:
        black, white = _levels_points(mixed)
        gains = _gains(mixed, black, underwater)
    grade = AutoGrade(mode, black=black, white=white, gains=gains, mix=mix, underwater_hint=hint)
    grade = _round(replace(grade, vibrance=_vibrance(rgb, grade, before)))
    if underwater and not looks:  # the grade is kept for the record, as a veto's
        grade = replace(grade, strength=strength)
        predicted = metrics(simulate(rgb, grade))
        return replace(grade, skip=SKIP_NOT_UNDERWATER, before=before, predicted=predicted)
    if _is_good(grade):
        return replace(grade, strength=strength, skip=SKIP_GOOD, before=before)
    grade = replace(grade, strength=strength)
    graded = simulate(rgb, grade)
    predicted = metrics(graded)
    veto = _veto(
        before,
        predicted,
        _intended_clipping(rgb, grade),
        underwater=underwater,
        red_clipped=_red_highlights(graded, _intended_high(rgb, grade)) if underwater else 0.0,
    )
    return replace(grade, skip=veto, before=before, predicted=predicted)


# -- cache --------------------------------------------------------------------------------------


def cache_path(src: Path, key: dict[str, Any]) -> Path:
    """Cache file of one measurement: ``<source cache>/color/<hash of key>.json``."""
    return cache.source_dir(src) / "color" / f"{cache.params_hash(key)}.json"


def auto_grade(
    src: Path,
    info: MediaInfo,
    start: float,
    end: float,
    mode: str,
    fmt: OutputFormat,
    *,
    strength: float = 1.0,
    vertical: bool = False,
    crop: Crop | None = None,
) -> AutoGrade:
    """:func:`estimate` of the clip ``[start, end]`` of ``src``, cached per source.

    The cache key is everything the result depends on: the version, ``mode``, ``strength``,
    the times and the sampling chain (:func:`sample_filter`: crop, size, BT.709 conversion).
    A clip whose frames cannot be decoded is skipped (:data:`SKIP_NO_FRAMES`, not cached).

    Raises:
        ValueError: For a ``mode`` other than :data:`MODES`.
    """
    if mode not in MODES:
        raise ValueError(f"unknown color mode {mode!r} (expected one of {list(MODES)})")
    vf, size = sample_filter(info, fmt, vertical, crop)
    key = {
        "v": COLOR_VERSION,
        "mode": mode,
        "strength": strength,
        "start": round(start, 3),
        "end": round(end, 3),
        "vf": vf,
        "size": list(size),
    }
    path = cache_path(src, key)
    entry = cache.load_json(path)
    got = AutoGrade.from_entry(entry) if isinstance(entry, dict) else None
    if got is not None:
        return replace(got, underwater_hint=entry.get("underwater_hint") is True)
    try:
        stats = sample_stats(src, info, start, end, fmt, vertical, crop)
    except FFmpegError as err:
        logger.debug("color of %s %.2f-%.2f s not measured: %s", src.name, start, end, err)
        return AutoGrade.skipped(mode, SKIP_NO_FRAMES, strength)
    grade = estimate(stats, mode, strength)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.save_json(path, grade.to_json() | {"underwater_hint": grade.underwater_hint})
    return grade
