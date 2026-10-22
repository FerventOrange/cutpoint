"""Variety selection (``cutpoint mix``): a balanced set of moments from a ski day.

The ``cutpoint analyze --style ski`` features (``<project>/ski_features``) are
peak-picked once per *category* with :func:`cutpoint.ski.pick` and
category-specific weights, so a day of 360 footage does not collapse to the
one signal that happens to be strong (laughs, group shots):

    run      carve .55, airtime .25, people .2; window mean carve >= 0.4; never on
             lifts / stops; 6-9 s; kind ``follow`` with people around, else ``carve``
    jump     telemetry airtime events of medium / high confidence; kind ``airtime``
    crash    ``crashes.json``: confidence >= max(0.2, 0.6 x best) and corroborated
             (words / audio / telemetry / dense confirm); the best 3, each with its
             reaction span; kind ``wipeout``
    lift     telemetry lift mask, or (360) :func:`cutpoint.pano.views.lift_scores`
             above 0.5 for >= 20 s; the best beauty / people moment of each ride;
             5-7 s; kind ``beauty`` or ``social``
    scenery  beauty >= 0.55, off lifts; 3-5 s; kind ``beauty``
    social   laughter .5, people .3, voice .2 with someone audible; 5-9 s; kind ``laughter``

Without GPS (Insta360 X3) speed is 0, so ``run`` uses the IMU turn rhythm
(``carve``) only; with GPS ``carve`` already includes speed. Scores are
percentile-normalised per category, then :func:`select_mix` fills per-category
quotas of the budget (0.9 x target) greedily under variety rules (spacing,
per-source cap, CLIP near-duplicate veto, telemetry-run coverage). Categories
without candidates hand their quota to run / scenery.

Writes ``mix_segments.json`` (a :func:`cutpoint.clips.load_segments` list, in
chronological order; ``cutpoint clips --mix`` plays it in ``arc`` order),
``mix_report.json`` (with the arc and its :func:`content_qa`) and ``mix_sheet.jpg``.

Exclusions (``mix_exclude.json``, written by ``cutpoint 360 repair``) cut source
spans out of the candidates before selection, see :func:`apply_exclusions`.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache, ski
from cutpoint.clips import REACTION_LABEL, index_sources, order_moments
from cutpoint.config import remap_source, source_path
from cutpoint.crash import load_crashes
from cutpoint.events.sheets import contact_sheet
from cutpoint.events.visual import load_visual
from cutpoint.pano import views
from cutpoint.telemetry import DT

logger = logging.getLogger(__name__)

CATEGORIES = ("run", "jump", "crash", "lift", "scenery", "social")
DEFAULT_QUOTAS = {
    "run": 0.35,
    "social": 0.20,
    "crash": 0.12,
    "scenery": 0.10,
    "lift": 0.10,
    "jump": 0.06,
    "flex": 0.07,
}
FALLBACK = ("run", "scenery")  # categories that take over an unused quota
RUN_W = {"carve": 0.55, "airtime": 0.25, "people": 0.2}
SOCIAL_W = {"laughter": 0.5, "people": 0.3, "voice": 0.2}
LIFT_W = {"beauty": 0.6, "people": 0.4}
LENGTHS = {
    "run": (6.0, 9.0),
    "jump": (3.0, 6.0),
    "lift": (5.0, 7.0),
    "scenery": (3.0, 5.0),
    "social": (5.0, 9.0),
}
MIN_CLIP_S = 2.5  # shortest clip left after cutting out excluded spans
TRIM_FRAC = 0.75  # ... and at least this share of the category's LENGTHS minimum
EXCLUDE_FILE = "mix_exclude.json"
CRASH_MAX_S = 12.0  # crash clip (approach, impact, aftermath) cap
REACTION_MAX_S = 8.0
LIFT_MIN_S = 20.0
LIFT_ON = 0.5
QA_MIN_GROUPS = 4  # content acceptance: categories (lift counts as scenery)
QA_LENGTH_S = (180.0, 300.0)


@dataclass
class MixRules:
    """Variety rules of :func:`select_mix`."""

    spacing_s: float = 90.0  # same category (day time; same source without clock)
    per_source: int = 2  # per category
    social_apart_s: float = 300.0  # a second social clip of a source needs this gap
    dup_cos: float = 0.5  # near-duplicate veto (centred peak CLIP embeddings, same category)
    dup_window_s: float = 30.0  # ... and across categories this close in time
    run_cover: float = 0.7  # share of telemetry runs that contribute a clip
    budget_frac: float = 0.9
    max_share: float = 0.4  # no category past this share of the budget


@dataclass(eq=False)
class Cand:
    """One candidate moment (source times)."""

    category: str
    source: str
    start: float
    end: float
    peak_t: float
    score: float
    kind: str
    label: str
    norm: float = 0.0
    emb: np.ndarray | None = None
    run: int | None = None
    abs_t: float | None = None  # peak in day time (source start epoch + peak_t)
    extra: list[Cand] = field(default_factory=list)  # emitted with it (crash reaction)

    @property
    def seconds(self) -> float:
        return self.end - self.start + sum(e.end - e.start for e in self.extra)

    def spans(self) -> list[tuple[float, float]]:
        return [(self.start, self.end), *((e.start, e.end) for e in self.extra)]


# --------------------------------------------------------------------------- quotas
def parse_quotas(spec: str | None) -> dict[str, float]:
    """``run=.35,social=.2`` over :data:`DEFAULT_QUOTAS`, normalised to sum 1."""
    q = dict(DEFAULT_QUOTAS)
    for item in filter(None, (s.strip() for s in (spec or "").split(","))):
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or key not in q:
            raise ValueError(f"--quotas {item!r}: expected CATEGORY=SHARE, one of {', '.join(q)}")
        try:
            q[key] = float(value)
        except ValueError:
            raise ValueError(f"--quotas {item!r}: {value!r} is not a number") from None
        if q[key] < 0:
            raise ValueError(f"--quotas {item!r}: share must not be negative")
    total = sum(q.values())
    if total <= 0:
        raise ValueError("--quotas: shares sum to 0")
    return {k: v / total for k, v in q.items()}


def effective_quotas(quotas: dict[str, float], available: set[str]) -> dict[str, float]:
    """Move the quota of categories without candidates to run / scenery
    (proportionally; to every available category if neither has any)."""
    q = {k: v for k, v in quotas.items() if k == "flex" or k in available}
    spare = sum(v for k, v in quotas.items() if k not in q)
    heirs = [k for k in FALLBACK if k in available] or sorted(available)
    if spare and heirs:
        base = sum(q.get(k, 0.0) for k in heirs)
        for k in heirs:
            share = q.get(k, 0.0) / base if base else 1 / len(heirs)
            q[k] = q.get(k, 0.0) + spare * share
    elif spare:
        q["flex"] = q.get("flex", 0.0) + spare
    return q


def normalise(cands: list[Cand]) -> None:
    """``norm``: percentile rank of ``score`` within each category (0-1)."""
    for cat in {c.category for c in cands}:
        group = sorted((c for c in cands if c.category == cat), key=lambda c: c.score)
        for i, c in enumerate(group):
            c.norm = i / (len(group) - 1) if len(group) > 1 else 1.0


# --------------------------------------------------------------------------- selection
def _cos(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def allowed(c: Cand, chosen: list[Cand], rules: MixRules) -> bool:
    """No overlap with any chosen clip; spacing, per-source cap and near-duplicate
    veto within the category; near-duplicate veto across categories within
    ``dup_window_s``."""
    for o in chosen:
        if o.source == c.source and any(
            a < d and b > s for a, b in c.spans() for s, d in o.spans()
        ):
            return False
        if c.abs_t is not None and o.abs_t is not None:
            gap = abs(o.abs_t - c.abs_t)
        else:
            gap = abs(o.peak_t - c.peak_t) if o.source == c.source else math.inf
        if o.category != c.category:
            if gap <= rules.dup_window_s and _cos(o.emb, c.emb) > rules.dup_cos:
                return False
            continue
        if gap < rules.spacing_s or (
            c.category == "social" and o.source == c.source and gap < rules.social_apart_s
        ):
            return False
        if _cos(o.emb, c.emb) > rules.dup_cos:
            return False
    same = sum(o.source == c.source and o.category == c.category for o in chosen)
    return same < rules.per_source


def select_mix(
    cands: list[Cand],
    target_s: float = 240.0,
    quotas: dict[str, float] | None = None,
    rules: MixRules | None = None,
    n_runs: int = 0,
    fixed: list[Cand] | None = None,
) -> list[Cand]:
    """Greedy quota fill (best ``norm`` first), flex fill, then run-coverage swaps.

    Args:
        cands: Candidates with ``norm`` set (:func:`normalise`).
        target_s: Target edit length; the budget is ``budget_frac`` of it.
        quotas: Category shares (``flex`` included), default :data:`DEFAULT_QUOTAS`.
        rules: Variety rules.
        n_runs: Telemetry runs of the day (``Cand.run`` indexes them); 0 = no swaps.
        fixed: Already chosen candidates (kept, never swapped out); the rest fills up.
    """
    r = rules or MixRules()
    budget = r.budget_frac * target_s
    q = effective_quotas(quotas or DEFAULT_QUOTAS, {c.category for c in cands})
    ranked = sorted(cands, key=lambda c: (-c.norm, -c.score))
    chosen: list[Cand] = list(fixed or [])

    def secs(cat: str | None = None) -> float:
        return sum(c.seconds for c in chosen if cat is None or c.category == cat)

    count = {k: sum(c.category == k for c in cands) for k in q}
    for cat in sorted((k for k in q if k != "flex"), key=lambda k: count[k]):  # scarce first
        cap = min(q[cat], r.max_share) * budget
        for c in (c for c in ranked if c.category == cat):
            have = secs(cat)
            if have >= cap:
                break
            fits = have + c.seconds <= min(cap * 1.15, r.max_share * budget) or have == 0
            if fits and secs() + c.seconds <= budget and allowed(c, chosen, r):
                chosen.append(c)
    for c in ranked:  # flex: best of the rest, no category past max_share
        if c in chosen or secs() + c.seconds > budget:
            continue
        if secs(c.category) + c.seconds <= r.max_share * budget and allowed(c, chosen, r):
            chosen.append(c)
    if n_runs:
        _cover_runs(chosen, ranked, n_runs, budget, r, fixed or [])
    return chosen


def _cover_runs(
    chosen: list[Cand],
    ranked: list[Cand],
    n_runs: int,
    budget: float,
    r: MixRules,
    fixed: list[Cand],
) -> None:
    """Swap clips of over-represented runs for the best clip of an uncovered run
    until ``run_cover`` of the runs contribute (in place).

    A clip only gives way to one of its own category, or when its category keeps
    another clip; crashes never give way to other categories. No category goes
    past ``max_share``.
    """
    need = math.ceil(r.run_cover * n_runs)
    while len({c.run for c in chosen if c.run is not None}) < need:
        covered = {c.run for c in chosen}
        counts = {k: sum(c.run == k for c in chosen) for k in covered}
        per_cat = {k: sum(c.category == k for c in chosen) for k in {c.category for c in chosen}}
        swap = None
        for new in (c for c in ranked if c.run is not None and c.run not in covered):
            victims = sorted(
                (
                    o
                    for o in chosen
                    if o not in fixed
                    and (o.run is None or counts[o.run] > 1)
                    and (
                        o.category == new.category
                        or (o.category != "crash" and per_cat[o.category] > 1)
                    )
                ),
                key=lambda o: (o.category != new.category, o.norm),
            )
            for old in victims:
                rest = [o for o in chosen if o is not old]
                total = sum(o.seconds for o in rest) + new.seconds
                cat_s = sum(o.seconds for o in rest if o.category == new.category) + new.seconds
                if total <= budget and cat_s <= r.max_share * budget and allowed(new, rest, r):
                    swap = (old, new)
                    break
            if swap:
                break
        if swap is None:
            return
        chosen[chosen.index(swap[0])] = swap[1]


# --------------------------------------------------------------------------- candidates
def _fit(c: Cand, lo: float, hi: float, duration: float) -> Cand:
    """Clamp ``c`` to ``lo``-``hi`` s around its peak inside ``[0, duration]``."""
    a, b = c.start, c.end
    if b - a > hi:
        a = min(max(a, c.peak_t - hi / 2), b - hi)
        b = a + hi
    if b - a < lo:
        grow = (lo - (b - a)) / 2
        a, b = a - grow, b + grow
    shift = max(0.0, -a) - max(0.0, b - duration)
    return replace(c, start=round(max(0.0, a + shift), 2), end=round(min(duration, b + shift), 2))


def _mean(x: np.ndarray, a: float, b: float) -> float:
    s = x[int(a / DT) : max(int(a / DT) + 1, int(b / DT))]
    return float(s.mean()) if len(s) else 0.0


def _spans(mask: np.ndarray, min_s: float) -> list[tuple[float, float]]:
    """``(start, end)`` seconds of the runs of ``mask > 0`` lasting ``min_s`` or more."""
    d = np.diff(np.r_[0, (mask > 0).astype(int), 0])
    ij = zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True)
    return [(i * DT, j * DT) for i, j in ij if (j - i) * DT >= min_s]


def _paint(n: int, spans: list[tuple[float, float]]) -> np.ndarray:
    x = np.zeros(n)
    for a, b in spans:
        x[int(a / DT) : int(b / DT)] = 1.0
    return x


def lift_rides(src: Path, n: int, offset: float = 0.0) -> np.ndarray | None:
    """360: 1 on CLIP lift rides (score > LIFT_ON for LIFT_MIN_S or more) on the 0.1 s grid.

    Cached view layer only; None without one (GoPro, or ``360 analyze`` not run).
    """
    zs = views.lift_grid(Path(src), n, offset, DT)
    return None if zs is None else _paint(n, _spans(zs > LIFT_ON, LIFT_MIN_S))


def part_candidates(d: dict[str, Any], max_picks: int = 30) -> list[Cand]:
    """Run / jump / lift / scenery / social candidates of one saved feature part."""
    f, n = d["features"], len(d["mask"])
    src, off, dur = d["source"], float(d["offset"]), float(d["duration"])
    zero = np.zeros(n)
    tel_lift = (d["mask"] > 0) & (f["stopped"][:n] == 0)
    rides = lift_rides(Path(src), n, off)
    rides = zero if rides is None else rides
    lift = np.maximum(tel_lift.astype(float), rides)
    hard = np.maximum(d["mask"], lift)
    out: list[Cand] = []

    def picks(w: dict[str, float], mask: np.ndarray, events: list | None = None, k: int = 0):
        return ski.pick(f, w, mask, zero, events or [], dur, k or max_picks, off)

    def cand(cat: str, p: ski.Pick, kind: str, label: str, score: float | None = None) -> Cand:
        c = Cand(
            cat, src, p.start, p.end, p.peak_t, p.peak if score is None else score, kind, label
        )
        return _fit(c, *LENGTHS[cat], dur) if cat in LENGTHS else c

    for p in picks(RUN_W, hard, d["events"]):
        if _mean(f["carve"], p.start, p.end) >= 0.4:
            follow = _mean(f["people"], p.start, p.end) >= 0.4
            out.append(cand("run", p, "follow" if follow else "carve", "run"))
    for p in picks({"airtime": 1.0}, hard, d["events"]):
        if p.event and p.event.get("confidence") in ("medium", "high"):
            out.append(cand("jump", p, "airtime", p.label, p.event.get("strength", p.peak)))
    for p in picks({"beauty": 1.0}, lift):
        if p.peak >= 0.55:
            out.append(cand("scenery", p, "beauty", "view"))
    for p in picks(SOCIAL_W, zero):
        if p.signals.get("laughter", 0) >= 0.3 or _mean(f["voice"], p.start, p.end) >= 0.5:
            out.append(cand("social", p, "laughter", "laugh"))
    for a, b in _spans(lift, LIFT_MIN_S):
        outside = np.ones(n)
        outside[int(a / DT) : int(b / DT)] = 0.0
        for p in picks(LIFT_W, outside, k=1):
            beauty = LIFT_W["beauty"] * p.signals.get("beauty", 0)
            kind = "beauty" if beauty >= LIFT_W["people"] * p.signals.get("people", 0) else "social"
            c = cand("lift", p, kind, "lift")
            shift = max(0.0, a - c.start) - max(0.0, c.end - b)  # keep it on the ride
            c.start, c.end = round(c.start + shift, 2), round(c.end + shift, 2)
            out.append(c)
    for c in out:  # part-relative -> source time
        c.start, c.end, c.peak_t = c.start + off, c.end + off, c.peak_t + off
    return out


def crash_candidates(crashes: list[dict[str, Any]], top: int = 3) -> list[Cand]:
    """Confident, corroborated crashes (best ``top``), each with its reaction span."""
    best = max((c["confidence"] for c in crashes), default=0.0)
    floor = max(0.2, 0.6 * best)
    corroborate = {"words": 0.5, "audio": 0.5, "telemetry": 0.4, "dense": 0.3}
    ok = [
        c
        for c in crashes
        if c["confidence"] >= floor
        and any((c.get("scores") or {}).get(k, 0) >= v for k, v in corroborate.items())
    ]
    out = []
    for c in sorted(ok, key=lambda c: -c["confidence"])[:top]:
        clip = c["clip"]
        a = max(clip["start"], c["t0"] - 4.0)
        cd = Cand(
            "crash",
            c["source"],
            a,
            min(clip["end"], a + CRASH_MAX_S),
            c["t0"],
            c["confidence"],
            "wipeout",
            "crash",
        )
        if r := clip.get("reaction"):
            end = min(r["end"], r["start"] + REACTION_MAX_S)
            cd.extra.append(replace(cd, start=r["start"], end=end, label=REACTION_LABEL, extra=[]))
        out.append(cd)
    return out


def _embed(cands: list[Cand]) -> None:
    """``emb``: the source's mean CLIP crop embedding nearest each peak (if cached),
    minus the mean over all candidates.

    Raw CLIP embeddings of a snow day are all alike (cosine ~0.93 between
    unrelated moments), so the veto compares what sets a moment apart from the day.
    """
    for src in {c.source for c in cands}:
        vd = load_visual(remap_source(src))
        if vd is None or not len(vd.t):
            continue
        emb = vd.mean_emb
        for c in (c for c in cands if c.source == src):
            c.emb = emb[int(np.abs(vd.t - c.peak_t).argmin())]
    have = [c for c in cands if c.emb is not None]
    if have:
        mean = np.mean([c.emb for c in have], 0)
        for c in have:
            c.emb = c.emb - mean


def _runs_index(project: Path) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    """Telemetry runs and each source's start (epoch; None if unknown) from telemetry.json."""
    tj = cache.load_json(project / "telemetry.json") or {}
    starts = {ski._key(c["source"]): c.get("start_epoch") for c in tj.get("clips", [])}
    return tj.get("runs", []), starts


def _run_of(runs: list[dict[str, Any]], src: str, t: float) -> int | None:
    for i, r in enumerate(runs):
        for s in r.get("segments", []):
            if ski._key(s["source"]) == ski._key(src) and s["start"] <= t <= s["end"]:
                return i
    return None


# --------------------------------------------------------------------------- exclusions
def min_length(category: str) -> float:
    """Shortest clip of ``category`` left after a cut: :data:`TRIM_FRAC` of its
    :data:`LENGTHS` minimum, at least :data:`MIN_CLIP_S`."""
    return max(TRIM_FRAC * LENGTHS.get(category, (0.0,))[0], MIN_CLIP_S)


def load_exclusions(path: Path) -> list[dict[str, Any]]:
    """``[{source, start, end, category?, reason?}, ...]`` from ``path`` (none if missing).

    Raises:
        ValueError: If an entry lacks ``source`` / ``start`` / ``end``.
    """
    items = cache.load_json(path) or []
    if not isinstance(items, list) or not all(
        isinstance(e, dict) and {"source", "start", "end"} <= e.keys() for e in items
    ):
        raise ValueError(f"{path}: expected a list of {{source, start, end}} objects")
    return items


def _piece(
    a: float, b: float, holes: list[tuple[float, float]], need: float, peak: float | None
) -> tuple[float, float] | None:
    """Longest part of ``[a, b]`` outside ``holes`` (the one holding ``peak`` if given),
    or None if shorter than ``need``."""
    parts = [(a, b)]
    for s, e in holes:
        parts = [p for x, y in parts for p in ((x, min(y, s)), (max(x, e), y)) if p[1] > p[0]]
    if peak is not None:
        parts = [p for p in parts if p[0] <= peak <= p[1]]
    best = max(parts, key=lambda p: p[1] - p[0], default=None)
    if best is None or best[1] - best[0] < need - 1e-6:
        return None
    return round(best[0], 2), round(best[1], 2)


def apply_exclusions(cands: list[Cand], exclude: list[dict[str, Any]]) -> list[Cand]:
    """Cut excluded source spans (of every category, or only of the entry's
    ``category``) out of the candidates.

    A cut candidate keeps its longest remaining part (a crash the part holding its
    peak) when that is at least :func:`min_length`, else it is dropped; a crash
    reaction is cut the same way on its own (at least :data:`MIN_CLIP_S`).
    """
    out = []
    for c in cands:
        holes = [
            (float(e["start"]), float(e["end"]))
            for e in exclude
            if ski._key(e["source"]) == ski._key(c.source)
            and e.get("category") in (None, c.category)
        ]
        if not holes:
            out.append(c)
            continue
        peak = c.peak_t if c.category == "crash" else None
        main = _piece(c.start, c.end, holes, min_length(c.category), peak)
        if main is None:
            continue
        extra = [
            replace(e, start=p[0], end=p[1])
            for e in c.extra
            if (p := _piece(e.start, e.end, holes, MIN_CLIP_S, None))
        ]
        out.append(replace(c, start=main[0], end=main[1], extra=extra))
    return out


def _inside(start: float, end: float, seg: dict[str, Any]) -> bool:
    """``start``-``end`` lies within ``seg``: exclusions only shrink a clip, so a span
    reaching past it is a neighbour, not the same clip."""
    return start >= seg["start"] - 0.05 and end <= seg["end"] + 0.05


def _same(seg: dict[str, Any], s: Cand) -> bool:
    """``seg`` (a segments-file entry) and span ``s`` share source and category and ``s``
    lies within it (:func:`_inside`)."""
    return (
        ski._key(seg["source"]) == ski._key(s.source)
        and seg.get("category") == s.category
        and _inside(s.start, s.end, seg)
    )


def _carry(cands: list[Cand], previous: list[dict[str, Any]]) -> list[Cand]:
    """The candidates behind ``previous`` segments (best overlap, inside the segment) that
    survived the exclusions."""
    keep: list[Cand] = []
    for seg in previous:
        hits = [(c, s) for c in cands for s in (c, *c.extra) if _same(seg, s)]
        if not hits:
            logger.info("mix: %s (%s) is gone", seg.get("id"), seg.get("category"))
            continue
        c, _ = max(hits, key=lambda h: min(seg["end"], h[1].end) - max(seg["start"], h[1].start))
        if c not in keep:
            keep.append(c)
    return keep


def _seg_ids(
    flat: list[tuple[Cand, Cand]], previous: list[dict[str, Any]], fixed: list[Cand]
) -> list[str]:
    """The previous id of each span carried over (its candidate in ``fixed``); the rest
    get new ``mNNN`` ids above the highest previous one (``m001``... without ``previous``)."""
    nums = [int(p["id"][1:]) for p in previous if str(p["id"])[:1] == "m" and p["id"][1:].isdigit()]
    fresh = (f"m{k:03d}" for k in itertools.count(max(nums, default=0) + 1))
    return [
        (next((p["id"] for p in previous if _same(p, s)), None) if c in fixed else None)
        or next(fresh)
        for s, c in flat
    ]


# --------------------------------------------------------------------------- project
def run_mix(
    project: Path,
    target_s: float = 240.0,
    quotas: dict[str, float] | None = None,
    out: Path | None = None,
    rules: MixRules | None = None,
    sheet: bool = True,
    exclude: list[dict[str, Any]] | None = None,
    previous: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select, write ``mix_segments.json`` / ``mix_report.json`` / ``mix_sheet.jpg``.

    Args:
        exclude: Source spans to leave out (:func:`apply_exclusions`); default the
            project's :data:`EXCLUDE_FILE`.
        previous: An earlier selection (segments-file entries) to keep: its clips that
            survive the exclusions stay (cut if needed, same ids, never swapped out)
            and only the freed budget is filled anew.

    Returns:
        The report.
    """
    r = rules or MixRules()
    q = quotas or DEFAULT_QUOTAS
    parts = ski.load_project_features(project)
    cands = [c for d in parts for c in part_candidates(d)]
    cands += crash_candidates((load_crashes(project) or {}).get("crashes", []))
    if not cands:
        raise ValueError(f"No candidates in {project}; run `cutpoint analyze --style ski` first")
    runs, starts = _runs_index(project)
    for c in cands:
        c.run = _run_of(runs, c.source, c.peak_t)
        start = starts.get(ski._key(c.source))
        c.abs_t = None if start is None else start + c.peak_t
    normalise(cands)
    _embed(cands)
    if exclude is None:
        exclude = load_exclusions(project / EXCLUDE_FILE)
    cands = apply_exclusions(cands, exclude)  # after scoring: the others keep their norm
    fixed = _carry(cands, previous) if previous else None
    chosen = select_mix(cands, target_s, q, r, n_runs=len(runs), fixed=fixed)
    if not chosen:
        why = (
            f"none of {len(cands)} candidates fits the {target_s:.0f} s target"
            if cands
            else "every candidate is excluded"
        )
        raise ValueError(f"Nothing selected in {project}: {why}")
    flat = [(s, c) for c in chosen for s in (c, *c.extra)]  # chronological play order
    flat.sort(key=lambda p: (starts.get(ski._key(p[0].source)) or 0.0, p[0].source, p[0].start))
    ids = _seg_ids(flat, previous or [], fixed or [])
    segs = [
        {
            "id": ids[i],
            "source": s.source,
            "start": s.start,
            "end": s.end,
            "label": s.label,
            "kind": s.kind,
            "category": s.category,
            "score": round(c.norm, 3),
            "chapter": None if c.run is None else c.run + 1,
        }
        for i, (s, c) in enumerate(flat)
    ]
    out = out or project / "mix_segments.json"
    prior = previous if previous is not None else cache.load_json(out) or []
    for s in segs:  # keep the lead-in / tail bounds (360 repair) of the same clip
        p = next((p for p in prior if p.get("id") == s["id"]), None)
        if p is None or ski._key(p["source"]) != ski._key(s["source"]):
            continue
        if not _inside(s["start"], s["end"], p):  # a different clip
            continue
        if p.get("min_start") is not None:
            s["min_start"] = min(p["min_start"], s["start"])
        if p.get("max_end") is not None:
            s["max_end"] = max(p["max_end"], s["end"])
    cache.save_json(out, segs)
    report = mix_report(cands, chosen, q, r, target_s, len(runs), len(parts))
    arc = order_moments(index_sources([dict(s) for s in segs]), "arc")
    report["arc"] = [s["id"] for s in arc]
    report["content_qa"] = content_qa(
        arc,
        near_dups=report["near_dup_pairs"],
        runs_covered=report["runs_covered"],
        runs=len(runs),
        rules=r,
    )
    cache.save_json(out.with_name("mix_report.json"), report)
    logger.info(
        "mix: %d segments, %.0f s; %s; %d/%d runs, %d sources -> %s",
        len(segs),
        report["total_s"],
        ", ".join(f"{k} {v['seconds']:.0f}s" for k, v in report["categories"].items()),
        report["runs_covered"],
        len(runs),
        report["sources_covered"],
        out,
    )
    failed = [k for k, v in report["content_qa"]["checks"].items() if not v["ok"]]
    logger.info("mix content QA: %s", f"FAIL ({', '.join(failed)})" if failed else "pass")
    if sheet:
        _write_sheet(segs, out.with_name("mix_sheet.jpg"))
    return report


def content_qa(
    segs: list[dict[str, Any]],
    *,
    near_dups: int = 0,
    runs_covered: int = 0,
    runs: int = 0,
    rules: MixRules | None = None,
    durations: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Edit-level content acceptance of play-ordered segments.

    Checks: at least :data:`QA_MIN_GROUPS` categories (lift and scenery count as
    one), no category past ``max_share`` of the length, no near-duplicate pairs,
    ``run_cover`` of the telemetry runs covered (skipped without runs), a lift /
    scenery opener and a length within :data:`QA_LENGTH_S`. ``durations`` (by segment id,
    e.g. the rendered clips with lead-in and tails) replace ``end - start``.

    Returns:
        ``{"pass": bool, "checks": {name: {"value": ..., "ok": bool}}}``.
    """
    r = rules or MixRules()
    secs: dict[str, float] = {}
    for s in segs:
        cat = s.get("category") or "none"
        dur = (durations or {}).get(s.get("id"), s["end"] - s["start"])
        secs[cat] = secs.get(cat, 0.0) + dur
    total = sum(secs.values())
    groups = {"scenery" if c == "lift" else c for c in secs} - {"none"}
    share = max(secs.values(), default=0.0) / total if total else 0.0
    cover = runs_covered / runs if runs else None
    opener = segs[0].get("category") if segs else None
    lo, hi = QA_LENGTH_S
    checks = {
        "categories": {"value": len(groups), "ok": len(groups) >= QA_MIN_GROUPS},
        "max_share": {"value": round(share, 3), "ok": share <= r.max_share},
        "near_dups": {"value": near_dups, "ok": near_dups == 0},
        "runs_covered": {
            "value": None if cover is None else round(cover, 3),
            "ok": cover is None or cover >= r.run_cover,
        },
        "opener": {"value": opener, "ok": opener in ("lift", "scenery")},
        "length_s": {"value": round(total, 1), "ok": lo <= total <= hi},
    }
    return {"pass": all(c["ok"] for c in checks.values()), "checks": checks}


def mix_report(
    cands: list[Cand],
    chosen: list[Cand],
    quotas: dict[str, float],
    r: MixRules,
    target_s: float,
    n_runs: int,
    n_sources: int,
) -> dict[str, Any]:
    """Category breakdown, coverage and near-duplicate pairs of a selection."""
    total = sum(c.seconds for c in chosen)
    avail = {c.category for c in cands}
    cats = {}
    for cat in CATEGORIES:
        sel = [c for c in chosen if c.category == cat]
        secs = sum(c.seconds for c in sel)
        share = round(secs / total, 3) if total > 0 else 0.0
        cats[cat] = {"n": len(sel), "seconds": round(secs, 1), "share": share}
    dups = [
        (a.category, b.category)
        for i, a in enumerate(chosen)
        for b in chosen[i + 1 :]
        if _cos(a.emb, b.emb) > r.dup_cos
    ]
    return {
        "target_s": target_s,
        "total_s": round(total, 1),
        "categories": cats,
        "candidates": {k: sum(c.category == k for c in cands) for k in CATEGORIES},
        "quotas": {k: round(v, 3) for k, v in effective_quotas(quotas, avail).items()},
        "unused": [k for k in CATEGORIES if k not in avail],
        "sources_covered": len({c.source for c in chosen}),
        "sources": n_sources,
        "runs_covered": len({c.run for c in chosen if c.run is not None}),
        "runs": n_runs,
        "near_dup_pairs": sum(a == b for a, b in dups),
        "near_dup_pairs_any": len(dups),
        "crashes": [
            {"source": c.source, "t": c.peak_t, "confidence": c.score}
            for c in chosen
            if c.category == "crash"
        ],
    }


def _write_sheet(segs: list[dict[str, Any]], out: Path) -> Path | None:
    items = [
        {
            "path": str(remap_source(s["source"])),
            "t": (s["start"] + s["end"]) / 2,
            "caption": f"{s['id']} {s['category']} {source_path(s['source']).stem[-3:]}"
            f"@{s['start']:.0f} {s['end'] - s['start']:.0f}s",
        }
        for s in segs
    ]
    return contact_sheet(items, out, cols=6, width=320)
