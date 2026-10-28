"""Automatic colour correction of exported clips (#290 stage 2: ``auto`` and ``underwater``).

A clip is measured on up to :data:`SAMPLE_FRAMES` keyframes of its inner 80 %
(:func:`sample_stats`), seen through the same crop and BT.709 conversion as its export, and
:func:`estimate` turns those frames into an :class:`AutoGrade`: a levels stretch, a white
balance, a red-from-green channel mix (underwater only) and a vibrance boost, every one
clamped. The grade is ffmpeg filters only (:meth:`AutoGrade.filter`: ``colorchannelmixer``,
``colorlevels`` and ``vibrance`` in 16-bit RGB), so the export needs nothing new.

Before a grade is used, :func:`simulate` applies it to the sampled frames in numpy and
:func:`metrics` compares them before and after: a grade that clips more, leaves a worse cast
or adds too much colour is vetoed. Clips that already look good, and log or flat footage
(which needs the camera's LUT, ``--lut``), get no grade. :func:`auto_grade` caches each
result in the source's cache folder (``color/<hash>.json``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.crop import Crop
from cutpoint.encode import TAG_BT709, OutputFormat, bt709_filter, log_stats, looks_log
from cutpoint.ffmpeg_utils import FFmpegError, MediaInfo, decode_raw

logger = logging.getLogger(__name__)

# Bump when the measurement or estimate changes: it keys the cache and is recorded per clip.
COLOR_VERSION = 1
MODES = ("auto", "underwater")

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
# A clip that looks underwater (for plain ``auto``'s hint): mean red below this share of
# both green and blue.
UNDERWATER_RED = 0.6
# Veto: the clipped share rises by more than VETO_CLIPPED, the gray-world residual rises by
# more than VETO_RESIDUAL_EPS (0.5 % of the luma range, about a code: an invisible change),
# or M3 rises by more than VETO_M3_RISE.
VETO_CLIPPED = 0.005
VETO_RESIDUAL_EPS = 0.005
VETO_M3_RISE = 0.25
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
VETO_CLIPPING = "veto_clipped"
VETO_CAST = "veto_cast"
VETO_COLOURFUL = "veto_m3"


@dataclass(frozen=True)
class Metrics:
    """What a grade is judged by, over a clip's sampled frames (see :func:`metrics`).

    ``m3``: Hasler-Suesstrunk colourfulness (0-255 scale); ``residual``: gray-world cast,
    the largest distance of a channel mean from their average over the luma range;
    ``clipped``: share of pixels with a channel at 0 or 255; ``black`` / ``white``: luma
    p0.5 / p99.5 (0-1).
    """

    m3: float
    residual: float
    clipped: float
    black: float
    white: float

    def to_json(self) -> dict[str, float]:
        """The metrics as a manifest dict."""
        return {
            "m3": self.m3,
            "residual": self.residual,
            "clipped": self.clipped,
            "black": self.black,
            "white": self.white,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> Metrics | None:
        """The metrics :meth:`to_json` wrote, None for none."""
        if not data:
            return None
        return cls(*(float(data[k]) for k in ("m3", "residual", "clipped", "black", "white")))


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
    black: float = 0.0
    white: float = 1.0
    gains: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mix: float = 0.0
    vibrance: float = 0.0
    skip: str | None = None
    before: Metrics | None = field(default=None, compare=False)
    predicted: Metrics | None = field(default=None, compare=False)

    def applied(self) -> AutoGrade:
        """This grade with the strength folded into the parameters (and strength 1)."""
        s = min(max(self.strength, 0.0), 1.0)
        return replace(
            self,
            strength=1.0,
            black=self.black * s,
            white=1.0 - (1.0 - self.white) * s,
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
        if g.black > 0 or g.white < 1 or any(k != 1 for k in g.gains):
            steps.append(_levels(g.black, g.white, g.gains))
        if g.vibrance > 0:
            steps.append(f"vibrance=intensity={g.vibrance:.5f}:{_VIBRANCE_OPTS}")
        if not steps:
            return ""
        return f"{_TO_RGB},{_RGB16},{','.join(steps)},{_TO_YUV},{TAG_BT709}:range=tv"

    def params(self) -> dict[str, Any]:
        """The full-strength parameters as a manifest dict."""
        return {
            "black": self.black,
            "white": self.white,
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
            black=float(p.get("black", 0.0)),
            white=float(p.get("white", 1.0)),
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


def _levels(black: float, white: float, gains: tuple[float, float, float]) -> str:
    """``colorlevels`` mapping each channel to ``(x - black) * gain / (white - black)``.

    The input white of a channel is where that line reaches 1; a gain below 1 can put it
    above 1 (outside the filter's range), so the output white is lowered instead.
    """
    opts = []
    for c, gain in zip("rgb", gains, strict=True):
        k = gain / (white - black)
        imax, omax = black + 1.0 / k, 1.0
        if imax > 1.0:
            imax, omax = 1.0, (1.0 - black) * k
        opts.append(f"{c}imin={black:.5f}:{c}imax={imax:.5f}:{c}omax={omax:.5f}")
    return "colorlevels=" + ":".join(opts)


# -- sampling -----------------------------------------------------------------------------------


@dataclass
class ColorStats:
    """Sampled frames of a clip: ``frames`` is ``(n, height, width, 3)`` BT.709 RGB ``uint8``."""

    frames: np.ndarray


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
        logger.debug("no keyframe of %s in %.2f-%.2f s: %s", src.name, start, end, err)
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
    return ColorStats(np.stack([kept[int(i)] for i in pick]))


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


def metrics(frames: np.ndarray) -> Metrics:
    """:class:`Metrics` of ``(n, h, w, 3)`` frames (``uint8``, or float 0-1): the median over
    the frames, the clipped share the mean."""
    rgb = _float(frames)
    m3, residual, black, white, clipped = [], [], [], [], []
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
    return Metrics(
        m3=round(float(np.median(m3)), 3),
        residual=round(float(np.median(residual)), 4),
        clipped=round(float(np.mean(clipped)), 5),
        black=round(float(np.median(black)), 4),
        white=round(float(np.median(white)), 4),
    )


def simulate(frames: np.ndarray, grade: AutoGrade) -> np.ndarray:
    """``frames`` (``uint8``, or float 0-1) as :meth:`AutoGrade.filter` grades them, in
    float 0-1: the strength folded in, each filter's result clipped as ffmpeg stores it."""
    g = grade.applied()
    rgb = _float(frames).copy()
    if g.mix > 0:
        rgb[..., 0] = np.clip((1.0 - g.mix) * rgb[..., 0] + g.mix * rgb[..., 1], 0.0, 1.0)
    k = np.asarray(g.gains, np.float32) / (g.white - g.black)
    rgb = np.clip((rgb - g.black) * k, 0.0, 1.0)
    if g.vibrance > 0:
        # vf_vibrance: each channel moves away from the luma by 1 + v * (1 - saturation)
        sat = rgb.max(axis=-1, keepdims=True) - rgb.min(axis=-1, keepdims=True)
        luma = (rgb @ LUMA)[..., None]
        rgb = np.clip(luma + (rgb - luma) * (1.0 + g.vibrance * (1.0 - sat)), 0.0, 1.0)
    return rgb


def is_log(stats: ColorStats) -> bool:
    """True when the frames look like log or flat footage (:func:`~cutpoint.encode.looks_log`
    on the median of their :func:`~cutpoint.encode.log_stats`)."""
    sat, black, white = np.median([log_stats(f) for f in stats.frames], axis=0)
    return looks_log(float(sat), float(black), float(white))


def looks_underwater(stats: ColorStats) -> bool:
    """True when the frames' mean red is below :data:`UNDERWATER_RED` of both green and blue
    (a hint for plain ``auto``; it never changes the grade)."""
    r, g, b = _float(stats.frames).reshape(-1, 3).mean(axis=0)
    return bool(r < UNDERWATER_RED * g and r < UNDERWATER_RED * b)


def _mix(rgb: np.ndarray) -> float:
    """Underwater red-from-green mix aiming the median red/green at :data:`MIX_TARGET`."""
    means = rgb.reshape(len(rgb), -1, 3).mean(axis=1)
    ratio = float(np.median(means[:, 0] / np.maximum(means[:, 1], 1e-4)))
    if ratio >= MIX_TARGET:
        return 0.0
    return min(MAX_MIX, (MIX_TARGET - ratio) / (1.0 - ratio))


def _levels_points(rgb: np.ndarray) -> tuple[float, float]:
    """Black and white points: the median luma p0.5 / p99.5, clamped (see :data:`MAX_BLACK`,
    :data:`MIN_WHITE`, :data:`MAX_STRETCH`; a stretch above it moves both points back
    toward 0 and 1 in proportion)."""
    lo, hi = np.median([np.percentile(f @ LUMA, LEVELS_PERCENTILES) for f in rgb], axis=0)
    black, white = min(float(lo), MAX_BLACK), max(float(hi), MIN_WHITE)
    if white - black < 1.0 / MAX_STRETCH:
        t = (1.0 - 1.0 / MAX_STRETCH) / (black + 1.0 - white)
        black, white = black * t, 1.0 - (1.0 - white) * t
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
        black=round(grade.black, 4),
        white=round(grade.white, 4),
        gains=tuple(round(g, 4) for g in grade.gains),  # type: ignore[arg-type]
        mix=round(grade.mix, 4),
        vibrance=round(grade.vibrance, 4),
    )


def _veto(before: Metrics, after: Metrics) -> str | None:
    """The veto a predicted result earns (see :data:`VETO_CLIPPED`), None for none."""
    if after.clipped - before.clipped > VETO_CLIPPED:
        return VETO_CLIPPING
    if after.residual > before.residual + VETO_RESIDUAL_EPS:
        return VETO_CAST
    if after.m3 > before.m3 * (1.0 + VETO_M3_RISE):
        return VETO_COLOURFUL
    return None


def _is_good(grade: AutoGrade) -> bool:
    """True when ``grade`` would hardly change the clip: gains within :data:`GOOD_GAIN`, levels
    within :data:`GOOD_LEVELS` of full, no channel mix and no vibrance (M3 at least
    :data:`M3_FULL`)."""
    return (
        all(abs(g - 1.0) <= GOOD_GAIN for g in grade.gains)
        and grade.black <= GOOD_LEVELS
        and grade.white >= 1.0 - GOOD_LEVELS
        and grade.mix == 0
        and grade.vibrance == 0
    )


def estimate(stats: ColorStats, mode: str, strength: float = 1.0) -> AutoGrade:
    """The grade of a clip from its sampled frames (:func:`sample_stats`).

    Log or flat footage is skipped (:data:`SKIP_LOG`). Otherwise, in order: the red-from-green
    mix (``underwater``), the levels on the mixed frames, the white balance on those levels,
    the vibrance on the result. A clip that would hardly change is skipped
    (:data:`SKIP_GOOD`); a grade the simulation at ``strength`` vetoes (:func:`_veto`) is
    skipped with the veto's name. Every result records the metrics before and predicted.

    Raises:
        ValueError: For a ``mode`` other than :data:`MODES`.
    """
    if mode not in MODES:
        raise ValueError(f"unknown color mode {mode!r} (expected one of {list(MODES)})")
    before = metrics(stats.frames)
    if is_log(stats):
        return replace(AutoGrade.skipped(mode, SKIP_LOG, strength), before=before)
    underwater = mode == "underwater"
    rgb = _float(stats.frames)
    mix = _mix(rgb) if underwater else 0.0
    mixed = simulate(rgb, AutoGrade(mode, mix=mix)) if mix else rgb
    black, white = _levels_points(mixed)
    gains = _gains(mixed, black, underwater)
    grade = AutoGrade(mode, black=black, white=white, gains=gains, mix=mix)
    grade = _round(replace(grade, vibrance=_vibrance(rgb, grade, before)))
    if _is_good(grade):
        return replace(grade, strength=strength, skip=SKIP_GOOD, before=before)
    grade = replace(grade, strength=strength)
    predicted = metrics(simulate(rgb, grade))
    return replace(grade, skip=_veto(before, predicted), before=before, predicted=predicted)


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
        return got
    try:
        stats = sample_stats(src, info, start, end, fmt, vertical, crop)
    except FFmpegError as err:
        logger.debug("color of %s %.2f-%.2f s not measured: %s", src.name, start, end, err)
        return AutoGrade.skipped(mode, SKIP_NO_FRAMES, strength)
    grade = estimate(stats, mode, strength)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.save_json(path, grade.to_json())
    return grade
