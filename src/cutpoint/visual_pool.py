"""Visual candidates: beauty, cruising and group windows that become moments (#635).

The scorer ranks scenery and steady riding low on purpose, so on a day of action footage
they never reach ``highlights.json``. This module adds a few of them after the scored
candidates, for the styles ``action-cam``, ``action`` and ``sport`` (and learned presets
based on them) only, from cached layers alone (no video decode):

* ``action-cam``: the grids ``analyze`` saved in ``action_cam_features`` (beauty, people,
  group, airtime, crash, wipeout, carve and the lift/still mask; :func:`cutpoint.actioncam.
  load_part`);
* ``action`` / ``sport``: the cached visual layer through :func:`cutpoint.actioncam.
  layer_features` (beauty, people, group); the lift/still mask is the cached GPS speed
  below :data:`STILL_KMH`, when there is one;
* the cached GPS speed (:func:`cutpoint.telemetry.cached_speed`) for ``cruise``;
* ``crashes.json`` as a veto: no window within a listed crash.

Kinds (each outside the lift/still mask):

* ``view``: beauty peaks of at least :data:`VIEW_MIN`, 5-7 s;
* ``crew``: people + group peaks of at least :data:`CREW_MIN` with a detected person, 4-6 s;
* ``cruise``: at least :data:`CRUISE_RUN_S` of movement with a steady speed (spread below
  :data:`CRUISE_MAX_SPREAD`; with a carve grid, a steady carve too) and no airtime, crash
  or wipeout, cut to :data:`CRUISE_LEN_S`. No speed data: no ``cruise``.

Per source at most :data:`PER_KIND` of each kind and :data:`PER_SOURCE` in all, with no
overlap between them and :data:`SPACING_S` between two of a kind; a window that overlaps a
scored candidate of its source by :data:`SCORED_OVERLAP` of the shorter clip or more is
dropped. ``analyze`` writes them as ``pool: "visual"`` candidates (ids ``v001``...) with
``score`` = peak x 10 (display only); :func:`cutpoint.planning.select_moments` gives them a
few reserved slots (:func:`reserved_slots`, :data:`SLOT_MIN_PEAK`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import actioncam, telemetry
from cutpoint.config import path_key
from cutpoint.dsp import clean_mask, mask_runs
from cutpoint.styles import ACTION_CAM, Style, style_name

logger = logging.getLogger(__name__)

POOL = "visual"  # the ``pool`` value of these candidates
POOL_BASES = ("action", "sport")  # with the action-cam kind: the styles that get a pool
KINDS = ("view", "crew", "cruise")
LABELS = {
    "view": "scenic view / beauty shot",
    "crew": "group / friends shot",
    "cruise": "casual cruising run",
}
DT = actioncam.DT
# view: beauty is 0-1 (zero-shot contrast x aesthetic x steadiness); mix keeps scenery from
# 0.55, the pool lists a little more, so the assistant can find it.
VIEW_MIN = 0.5
VIEW_LEN = (5.0, 7.0)
# crew: 0.6 people + 0.4 group; people needs a person close to the camera, so a crowd-like
# scene with nobody close cannot pass on the group prompt alone (CREW_PEOPLE_MIN: distant
# other riders score about 0.45-0.5 there, real crew shots 0.8 and more).
CREW_W = {"people": 0.6, "group": 0.4}
CREW_MIN = 0.5
CREW_PEOPLE_MIN = 0.6
CREW_LEN = (4.0, 6.0)
# cruise: GPS km/h; casual riding is ~15-35 km/h, walking under 8.
CRUISE_MIN_KMH = 8.0
STILL_KMH = 3.0  # action / sport: slower than this counts as still (the mask)
CRUISE_RUN_S = 8.0
CRUISE_LEN_S = 8.0  # inside the 6-9 s the design allows; a whole run of 8 s fits
CRUISE_GAP_S = 1.0  # GPS dropouts this short do not end a run
CRUISE_MAX_SPREAD = 0.25  # speed std / mean over the window; peak = 1 - spread / 0.5
CRUISE_MAX_CARVE_STD = 0.25  # carve grid (0-1) std over the window, when there is one
CRUISE_VETO = 0.05  # airtime / crash / wipeout grid above this anywhere in the window
CRASH_PAD = (2.0, 3.0)  # crashes.json veto: (before t0, after t1) seconds
PER_KIND = 2
PER_SOURCE = 5
SPACING_S = {"view": 30.0, "crew": 30.0, "cruise": 60.0}  # between two of a kind
SCORED_OVERLAP = 0.5  # of the shorter clip
MAX_PICKS = 8  # peaks looked at per kind and part
# select_moments: the quality bar of a reserved slot (above the listing floors: only the
# clearer views, groups and steadier runs; cruise 0.6 = speed spread 0.2 or less).
SLOT_MIN_PEAK = 0.6
SLOTS_PER = 8  # about one reserved slot per this many automatic picks
SLOTS_FROM = 5  # no reserved slot below this many picks
SLOTS_MAX = 2
PROTECTED_KINDS = ("airtime", "crash", "wipeout")  # never give their slot to a pool moment
# ... nor a pick showing a crashes.json crash of at least this confidence (any style; the
# floor of a crash worth a clip in cutpoint.mix)
CRASH_PROTECT_MIN = 0.2


@dataclass
class Part:
    """One analysed source (part) of a run: ``index`` into the run's results."""

    source: Path  # as recorded in highlights.json
    index: int
    offset: float
    duration: float
    n: int  # analysis grid length (DT)


@dataclass
class Window:
    """A pool moment in source time."""

    source: Path
    index: int  # Part.index
    start: float
    end: float
    kind: str
    peak: float
    signals: dict[str, float] = field(default_factory=dict)


def is_pool_style(style: Style) -> bool:
    """Whether ``analyze`` with ``style`` builds the pool."""
    return style.kind == ACTION_CAM or style.base in POOL_BASES


def project_has_pool(hl: Mapping[str, Any]) -> bool:
    """Whether the analysis ``hl`` (a highlights.json object) is of a pool style: its
    ``settings.visual_pool`` record (``analyze`` writes it for those styles, learned presets
    too), else its style kind, recorded built-in base (``style_base``) or name (a file without ``style``: the fallback of
    :func:`cutpoint.styles.style_name`)."""
    settings = hl.get("settings")
    if isinstance(settings, dict) and isinstance(settings.get("visual_pool"), dict):
        return True
    return (
        hl.get("style_kind") == ACTION_CAM
        or hl.get("style_base") in POOL_BASES
        or style_name(hl) in POOL_BASES
    )


def is_pool(c: Mapping[str, Any]) -> bool:
    """Whether the candidate ``c`` is a pool moment."""
    return c.get("pool") == POOL


def peak_of(c: Mapping[str, Any]) -> float:
    """A pool candidate's 0-1 peak (``signals.peak``, else ``score`` / 10)."""
    sig = c.get("signals")
    if isinstance(sig, dict) and isinstance(sig.get("peak"), int | float):
        return float(sig["peak"])
    return float(c.get("score") or 0) / 10


def protected_spans(crashes: Mapping[str, Any] | None) -> dict[str, list[tuple[float, float]]]:
    """The impact spans (``t0``-``t1``, source time) of the ``crashes.json`` crashes of at
    least :data:`CRASH_PROTECT_MIN` confidence, by :func:`~cutpoint.config.path_key` of
    their source (for :func:`is_protected`)."""
    return _crash_spans(crashes, (0.0, 0.0), CRASH_PROTECT_MIN)


def is_protected(c: Mapping[str, Any], spans: Mapping[str, list[tuple[float, float]]]) -> bool:
    """Whether the scored candidate ``c`` keeps its slot: must-include, a ``kind`` of
    :data:`PROTECTED_KINDS` (action-cam), or overlapping one of the crash ``spans``
    (:func:`protected_spans`; action and sport candidates carry no crash evidence of their
    own)."""
    if c.get("must_include") or c.get("kind") in PROTECTED_KINDS:
        return True
    try:
        a, b = float(c["start"]), float(c["end"])
        mine = spans.get(path_key(str(Path(c["source"]))), [])
    except (KeyError, TypeError, ValueError):
        return False
    return any(a <= s1 and s0 <= b for s0, s1 in mine)


def reserved_slots(picks: int) -> int:
    """Reserved pool slots among ``picks`` automatic picks: about one per
    :data:`SLOTS_PER`, at least one from :data:`SLOTS_FROM`, at most :data:`SLOTS_MAX`."""
    if picks < SLOTS_FROM:
        return 0
    return min(SLOTS_MAX, max(1, picks // SLOTS_PER))


# --------------------------------------------------------------------------- inputs
def _resize(x: np.ndarray, n: int, fill: float = 0.0) -> np.ndarray:
    return actioncam._resize(np.asarray(x, dtype=float), n, fill)


def _speed(src: Path, offset: float, n: int) -> np.ndarray | None:
    """Cached GPS speed (km/h, NaN where invalid) of this part, or None without one."""
    try:
        sp = telemetry.cached_speed(src)
    except OSError:  # footage offline: its cache key needs the file
        return None
    if sp is None or not np.isfinite(sp).any():
        return None
    return _resize(sp[round(offset / DT) :], n, np.nan)


def _inputs(
    project: Path, part: Part, action_cam: bool
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray | None] | None:
    """``(grids, mask, speed)`` of a part, or None without visual data."""
    speed = None
    if action_cam:
        d = actioncam.load_part(project, part.source, part.offset)
        if d is None:
            return None
        mask = np.asarray(d["mask"], dtype=float)
        n = len(mask)
        grids = {k: _resize(v, n) for k, v in d["features"].items()}
        speed = _speed(part.source, part.offset, n)
    else:
        n = part.n
        speed = _speed(part.source, part.offset, n)
        # NaN (no fix, or past the track's end) is 0 km/h for the layer, as in
        # actioncam.build_features: a NaN there turns beauty, people and group NaN. The raw speed
        # stays for the still mask and cruise.
        grids = actioncam.layer_features(
            part.source, n, part.offset, None if speed is None else np.nan_to_num(speed, nan=0.0)
        )
        mask = np.zeros(n)
        if speed is not None:
            mask = (np.nan_to_num(speed, nan=np.inf) < STILL_KMH).astype(float)
    if not any(k in grids for k in ("beauty", "people", "group")) and speed is None:
        return None
    return grids, mask, speed


def _crash_spans(
    crashes: Mapping[str, Any] | None,
    pad: tuple[float, float] = CRASH_PAD,
    min_confidence: float = 0.0,
) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    found = crashes.get("crashes") if isinstance(crashes, Mapping) else None
    for c in found if isinstance(found, list) else []:
        try:
            if min_confidence and float(c.get("confidence") or 0) < min_confidence:
                continue
            span = (float(c["t0"]) - pad[0], float(c["t1"]) + pad[1])
            out.setdefault(path_key(str(Path(c["source"]))), []).append(span)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- windows
def _fit(
    a: float, b: float, t: float, lo: float, hi: float, duration: float
) -> tuple[float, float]:
    """``[a, b]`` clamped to ``lo``-``hi`` s around ``t`` inside ``[0, duration]``."""
    if b - a > hi:
        a = min(max(a, t - hi / 2), b - hi)
        b = a + hi
    if b - a < lo:
        grow = (lo - (b - a)) / 2
        a, b = a - grow, b + grow
    shift = max(0.0, -a) - max(0.0, b - duration)
    return round(max(0.0, a + shift), 2), round(min(duration, b + shift), 2)


def _peaks(
    grids: dict[str, np.ndarray],
    weights: dict[str, float],
    mask: np.ndarray,
    duration: float,
) -> list[actioncam.Pick]:
    """Peaks of the weighted ``grids`` outside ``mask`` (:func:`cutpoint.actioncam.pick`)."""
    if not all(k in grids for k in weights):
        return []
    n = len(mask)
    # a NaN anywhere would make actioncam.pick's maximum NaN, and then it picks nothing
    feats = {k: np.nan_to_num(grids[k][:n], nan=0.0) for k in weights}
    return actioncam.pick(feats, weights, mask, np.zeros(n), [], duration, MAX_PICKS)


def _views(grids: dict[str, np.ndarray], mask: np.ndarray, duration: float) -> list[Window]:
    out = []
    for p in _peaks(grids, {"beauty": 1.0}, mask, duration):
        if p.peak >= VIEW_MIN:
            a, b = _fit(p.start, p.end, p.peak_t, *VIEW_LEN, duration)
            out.append(_window("view", a, b, p.peak, {"beauty": p.signals.get("beauty", 0.0)}))
    return out


def _crews(grids: dict[str, np.ndarray], mask: np.ndarray, duration: float) -> list[Window]:
    out = []
    for p in _peaks(grids, CREW_W, mask, duration):
        if p.peak >= CREW_MIN and p.signals.get("people", 0.0) >= CREW_PEOPLE_MIN:
            a, b = _fit(p.start, p.end, p.peak_t, *CREW_LEN, duration)
            sig = {k: p.signals.get(k, 0.0) for k in ("people", "group")}
            out.append(_window("crew", a, b, p.peak, sig))
    return out


def _cruises(
    grids: dict[str, np.ndarray],
    mask: np.ndarray,
    speed: np.ndarray,
    duration: float,
    vetoed: np.ndarray,
) -> list[Window]:
    n = len(mask)
    sp = speed[:n]
    moving = np.nan_to_num(sp, nan=0.0) >= CRUISE_MIN_KMH
    moving = clean_mask(moving, round(CRUISE_GAP_S / DT), round(CRUISE_RUN_S / DT))
    bad = vetoed.copy()
    for k in PROTECTED_KINDS:
        if k in grids:
            bad |= grids[k][:n] > CRUISE_VETO
    ok = moving & (mask[:n] == 0) & ~bad
    carve = grids.get("carve")
    carve = carve[:n] if carve is not None and carve[:n].max(initial=0.0) > 0 else None
    length, step = round(CRUISE_LEN_S / DT), round(1.0 / DT)
    out = []
    for i, j in mask_runs(ok):
        if j - i < round(CRUISE_RUN_S / DT):
            continue
        best = None
        for s in range(i, j - length + 1, step):
            seg = sp[s : s + length]
            mean = float(np.nanmean(seg))  # >= 8 km/h bins in every window: never NaN or 0
            spread = float(np.nanstd(seg)) / mean
            if carve is not None and float(carve[s : s + length].std()) > CRUISE_MAX_CARVE_STD:
                continue
            if best is None or (spread, -mean) < (best[0], -best[1]):
                best = (spread, mean, s)
        if best is None or best[0] > CRUISE_MAX_SPREAD:
            continue
        spread, mean, s = best
        peak = float(np.clip(1 - spread / (2 * CRUISE_MAX_SPREAD), 0, 1))
        sig = {"speed_kmh": round(mean, 1), "speed_spread": round(spread, 3)}
        if carve is not None:
            sig["carve"] = round(float(carve[s : s + length].mean()), 3)
        a = round(s * DT, 2)
        out.append(_window("cruise", a, round(min(duration, a + CRUISE_LEN_S), 2), peak, sig))
    return out


def _window(kind: str, a: float, b: float, peak: float, signals: dict[str, float]) -> Window:
    """A part-relative window (``source`` / ``index`` set by :func:`part_windows`)."""
    sig = {"peak": round(float(peak), 3), **{k: round(float(v), 3) for k, v in signals.items()}}
    return Window(Path(), -1, a, b, kind, float(peak), sig)


def _vetoed(n: int, spans: Sequence[tuple[float, float]], offset: float) -> np.ndarray:
    v = np.zeros(n, dtype=bool)
    for a, b in spans:
        v[max(0, int((a - offset) / DT)) : max(0, int(np.ceil((b - offset) / DT)))] = True
    return v


def part_windows(
    part: Part,
    grids: dict[str, np.ndarray],
    mask: np.ndarray,
    speed: np.ndarray | None,
    crash_spans: Sequence[tuple[float, float]] = (),
) -> list[Window]:
    """Every ``view`` / ``crew`` / ``cruise`` window of one part, in source time, before
    the caps (:func:`select`). Windows touching a ``crash_spans`` span (source time) are
    left out."""
    n, dur = len(mask), part.duration
    vetoed = _vetoed(n, crash_spans, part.offset)
    out = _views(grids, mask, dur) + _crews(grids, mask, dur)
    if speed is not None:
        out += _cruises(grids, mask, _resize(speed, n, np.nan), dur, vetoed)
    keep = []
    for w in out:
        if vetoed[int(w.start / DT) : max(int(w.start / DT) + 1, int(w.end / DT))].any():
            continue
        w.source, w.index = part.source, part.index
        w.start, w.end = round(w.start + part.offset, 2), round(w.end + part.offset, 2)
        keep.append(w)
    return keep


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def select(
    windows: list[Window],
    scored: Sequence[Mapping[str, Any]],
    speech: Mapping[int, Sequence[tuple[float, float]]] | None = None,
) -> list[Window]:
    """The pool: ``windows`` without those overlapping a ``scored`` candidate of their
    source by :data:`SCORED_OVERLAP` of the shorter clip, then per source the best by peak,
    no two overlapping, :data:`SPACING_S` apart within a kind, :data:`PER_KIND` per kind
    and :data:`PER_SOURCE` in all. Best peak first. ``speech``: spoken ``(start, end)`` spans
    in source time by :attr:`Window.index`; of two windows with the same score as listed
    (``10 x peak`` to 0.1) the one with more speech over it goes first (a tie-break only, no
    speech data: the order by peak)."""
    speech = speech or {}

    def talk(w: Window) -> float:
        return sum(_overlap(w.start, w.end, a, b) for a, b in speech.get(w.index, ()))

    def order(w: Window) -> tuple[float, float, float, int, float]:
        return (-round(10 * w.peak, 1), -talk(w), -w.peak, w.index, w.start)

    by_src: dict[str, list[tuple[float, float]]] = {}
    for c in scored:
        by_src.setdefault(path_key(str(c["source"])), []).append(
            (float(c["start"]), float(c["end"]))
        )
    kept: list[Window] = []
    per_src: dict[str, list[Window]] = {}
    for w in sorted(windows, key=order):
        key = path_key(str(w.source))
        if any(
            _overlap(w.start, w.end, a, b) >= SCORED_OVERLAP * min(w.end - w.start, b - a)
            for a, b in by_src.get(key, [])
        ):
            continue
        mine = per_src.setdefault(key, [])
        same = [o for o in mine if o.kind == w.kind]
        if len(mine) >= PER_SOURCE or len(same) >= PER_KIND:
            continue
        if any(_overlap(w.start, w.end, o.start, o.end) > 0 for o in mine):
            continue
        gap = SPACING_S[w.kind]
        if any(w.start < o.end + gap and o.start < w.end + gap for o in same):
            continue
        mine.append(w)
        kept.append(w)
    return kept


def build(
    project: Path,
    parts: Sequence[Part],
    style: Style,
    scored: Sequence[Mapping[str, Any]],
    crashes: Mapping[str, Any] | None,
    speech: Mapping[int, Sequence[tuple[float, float]]] | None = None,
) -> list[Window]:
    """The visual pool of an analysis (:func:`select` over :func:`part_windows`).

    Args:
        project: Project folder (``action_cam_features`` of this run, ``activity.json``).
        parts: The run's analysed parts.
        style: Analysis style (the action-cam kind reads the saved grids).
        scored: The scored candidates (dicts with ``source``, ``start``, ``end``).
        crashes: ``crashes.json`` object, or None.
        speech: Spoken spans in source time by part index (tie-break, see :func:`select`).
    """
    from cutpoint import activity  # noqa: PLC0415

    action_cam = style.kind == ACTION_CAM
    spans = _crash_spans(crashes)
    found: list[Window] = []
    # the visual layer judged on the captions of the project's activity.json, as the
    # action-cam grids were (analyze)
    with activity.use_prompts(activity.project_prompts(project)):
        for part in parts:
            got = _inputs(project, part, action_cam)
            if got is None:
                continue
            grids, mask, speed = got
            found += part_windows(
                part, grids, mask, speed, spans.get(path_key(str(Path(part.source))), [])
            )
    return select(found, scored, speech)
