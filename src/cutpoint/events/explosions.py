"""Confirmed explosions (#801): a heard boom that the loudness backs.

The audio layer reports PANNs' explosion classes as ``Explosion`` audio events
(:data:`cutpoint.events.audio_events.DEFAULT_GROUPS`, never a VLM seed by themselves). Gameplay
is full of booms that are not moments (engines, off-screen fire, sound coming back after a
loading screen), so an ``Explosion`` event counts as one only when :func:`confirm` finds it
**loud**: the momentary loudness (the cached ``loudness.json`` of ``cutpoint analyze``) jumps at
least :attr:`ExplosionParams.loud_lu` over the seconds before it, and those seconds were not
near-silent (:attr:`ExplosionParams.quiet_lufs`), and the picture was not black just before it
(below).

**Loading-screen guard**: game sound often comes back loud when a black loading or waiting screen
(mostly black, maybe a line of text) ends, which looks like a boom to the loudness. So each boom
that the loudness backs gets one short CPU decode of the picture from :attr:`ExplosionParams.black_s`
around it (5 frames/s, 64 px, gray); when at least :attr:`ExplosionParams.black_frames`
consecutive frames (0.4 s) have a mean luma (0..1 of full range) at most
:attr:`ExplosionParams.black_luma`, the boom is not confirmed and gets ``after_black: true``. A
single dark frame (a brief dark transition) never counts. A decode that fails (no video, ffmpeg error, out of range) keeps the loudness decision (fail open).

Two more backings are recorded for the reader and later tuning, not required (on labelled
footage neither raised precision: busy kill feeds put text near every boom, and scene cuts
rarely fall on one):

* **screen**: a scene cut (the cached ``scenes.json``) near the boom;
* **hud**: an on-screen text event of the OCR layer starting shortly before or after it.

The event keeps its fields and gets ``loud_jump`` (LU), ``backing`` (the names above that hold)
and ``confirmed`` (bool); a boom whose frames were checked also gets ``after_black`` (bool), the
others (not loud, or the decode failed) have no ``after_black``. Nothing is game-specific: no text,
no preset. Cheap: it reads two small cache files per source part and the OCR events already in
memory, plus a few short decodes per source (loud booms only), so nothing is cached here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.ffmpeg_utils import decode_raw

logger = logging.getLogger(__name__)

LABEL = "Explosion"  # the audio events' label (audio_events.DEFAULT_GROUPS)


@dataclass(frozen=True)
class ExplosionParams:
    """Thresholds of :func:`confirm` (part of events.json ``settings.explosions``)."""

    loud_lu: float = 6.0  # loudness jump at the boom over the median of the seconds before
    quiet_lufs: float = -50.0  # a "before" quieter than this is silence, not ongoing action
    pre_s: tuple[float, float] = (3.0, 0.6)  # the "before": from t - 3.0 to t - 0.6
    at_s: tuple[float, float] = (0.4, 0.6)  # the boom: t - 0.4 .. t + 0.6
    screen_s: tuple[float, float] = (1.0, 1.5)  # a cut from t - 1.0 to t + 1.5
    hud_s: tuple[float, float] = (1.0, 5.0)  # OCR text from t - 1.0 to t + 5.0
    # a frame this dark (mean luma, 0..1 of full range) is near-black: black measures about 0.007,
    # a black screen with a line of text about 0.01-0.03; dark transitions dip to 0.05 and more
    black_luma: float = 0.03
    black_frames: int = 2  # that many consecutive near-black frames (5 per s: 0.4 s) count
    black_s: tuple[float, float] = (3.0, 0.2)  # frames from t - 3.0 to t + 0.2 (a cut at t)

    def key(self) -> dict[str, Any]:
        """The thresholds as a JSON-able dict."""
        return asdict(self)


def levels(
    times: np.ndarray, lufs: np.ndarray, t: float, p: ExplosionParams
) -> tuple[float, float] | None:
    """(the loudest momentary LUFS at ``t``, the median LUFS before it); None when either stretch
    has no samples."""
    lufs = np.maximum(lufs, -70.0)  # the silence floor of analyze.loudness_signal
    at = lufs[(times >= t - p.at_s[0]) & (times <= t + p.at_s[1])]
    pre = lufs[(times >= t - p.pre_s[0]) & (times < t - p.pre_s[1])]
    if not len(at) or not len(pre):
        return None
    return float(at.max()), float(np.median(pre))


def loud_jump(
    times: np.ndarray, lufs: np.ndarray, t: float, p: ExplosionParams
) -> tuple[float, float] | None:
    """(how many LU the loudest momentary value at ``t`` is above the median before it, that
    median in LUFS); None when either stretch has no samples."""
    lv = levels(times, lufs, t, p)
    return (round(lv[0] - lv[1], 1), lv[1]) if lv else None


Parts = Iterable[tuple[float, Path]]  # (absolute start of the part, its cache folder)
# (t0, t1) in source time -> mean luma (0..1) of each sampled frame, None when not readable
LumaFn = Callable[[float, float], Sequence[float] | None]

_LUMA_FPS = 5.0
_LUMA_SIZE = (64, 36)  # the mean is all that is read: the aspect does not matter


def luma_reader(video: Path) -> LumaFn:
    """A :data:`LumaFn` reading ``video`` (one CPU decode per call, input seek)."""

    def read(t0: float, t1: float) -> list[float] | None:
        start = max(0.0, t0)
        if t1 <= start:
            return None
        w, h = _LUMA_SIZE
        try:
            frames = decode_raw(
                video,
                size=_LUMA_SIZE,
                vf=f"fps={_LUMA_FPS:g},scale={w}:{h}",
                pix_fmt="gray",
                start=start,
                duration=t1 - start,
                hwaccel=None,
                timeout=30.0,
            )
            return [float(f.mean()) / 255.0 for _, f in frames] or None
        except Exception as err:  # noqa: BLE001 - fail open: the loudness decides
            logger.debug("explosion frames %.1f-%.1fs of %s not read: %s", t0, t1, video, err)
            return None

    return read


def _loudness(parts: Parts) -> tuple[np.ndarray, np.ndarray] | None:
    ts, ms = [], []
    for offset, cdir in parts:
        data = cache.load_json(cdir / "loudness.json")
        if not isinstance(data, dict) or "t" not in data or "m" not in data:
            continue
        # ebur128 reports at the window end; centred on its 400 ms window as analyze does
        ts.append(np.asarray(data["t"], dtype=float) - 0.2 + offset)
        ms.append(np.asarray(data["m"], dtype=float))
    if not ts:
        return None
    t, m = np.concatenate(ts), np.concatenate(ms)
    order = np.argsort(t, kind="stable")
    return t[order], m[order]


def _cuts(parts: Parts) -> list[float]:
    out: list[float] = []
    for offset, cdir in parts:
        data = cache.load_json(cdir / "scenes.json")
        if isinstance(data, list):
            out += [float(x) + offset for x in data]
    return out


def has_booms(events: list[dict[str, Any]]) -> bool:
    """Whether a source's layer-1 ``events`` have an ``Explosion`` audio event."""
    return any(e.get("kind") == "audio" and e.get("label") == LABEL for e in events)


def confirm(
    events: list[dict[str, Any]],
    parts: Parts | None,
    params: ExplosionParams | None = None,
    *,
    loudness: tuple[np.ndarray, np.ndarray] | None = None,
    cuts: Iterable[float] | None = None,
    luma: LumaFn | None = None,
) -> int:
    """Mark the ``Explosion`` audio events of one source's layer-1 ``events`` (audio and OCR, in
    place) with ``loud_jump``, ``backing`` and ``confirmed``; returns how many are confirmed.

    ``parts``: ``(offset, cache folder)`` of each part of the source (``--recover`` makes more
    than one); their ``loudness.json`` and ``scenes.json`` are joined on the source's time.
    ``loudness`` (times, momentary LUFS) and ``cuts`` replace them (tests). Without loudness no event is
    confirmed. ``luma`` reads the mean luma of the frames around a loud boom (:func:`luma_reader`
    of the source video; a stub in tests); None skips the loading-screen guard.
    """
    p = params or ExplosionParams()
    booms = [e for e in events if e.get("kind") == "audio" and e.get("label") == LABEL]
    if not booms:
        return 0
    plist = list(parts) if parts is not None else []
    if loudness is None and plist:
        loudness = _loudness(plist)
    cut_list = list(cuts) if cuts is not None else _cuts(plist)
    texts = [float(e["t"]) for e in events if e.get("kind") == "ocr" and "t" in e]
    n = 0
    for e in booms:
        t = float(e.get("peak_t", e["t"]))
        lj = loud_jump(*loudness, t, p) if loudness is not None else None
        jump = lj[0] if lj else 0.0
        backing = []
        if lj and jump >= p.loud_lu and lj[1] >= p.quiet_lufs:
            backing.append("loud")
        if any(t - p.screen_s[0] <= c <= t + p.screen_s[1] for c in cut_list):
            backing.append("screen")
        if any(t - p.hud_s[0] <= x <= t + p.hud_s[1] for x in texts):
            backing.append("hud")
        e["loud_jump"] = jump
        e["backing"] = backing
        e["confirmed"] = "loud" in backing
        if e["confirmed"] and luma is not None:
            black = _after_black(luma, t, p)
            if black is not None:
                e["after_black"] = black
                e["confirmed"] = not black
        n += e["confirmed"]
    return n


def _after_black(luma: LumaFn, t: float, p: ExplosionParams) -> bool | None:
    """Whether ``black_frames`` consecutive frames from ``t - black_s[0]`` to ``t + black_s[1]``
    are near-black; None when the frames could not be read (never raises)."""
    try:
        values = luma(t - p.black_s[0], t + p.black_s[1])
    except Exception as err:  # noqa: BLE001 - fail open: the loudness decides
        logger.debug("explosion at %.1fs: frames not read: %s", t, err)
        return None
    if not values:
        return None
    run = 0
    for v in values:
        run = run + 1 if v <= p.black_luma else 0
        if run >= p.black_frames:
            return True
    return False
