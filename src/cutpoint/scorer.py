"""How highlight candidates are scored: weighted feature grids, then the events' bonus points.

Grid stage: the style's weights times the feature grids (:mod:`cutpoint.features`), summed in
the analysis signals' order (:func:`combined`; action-cam: :func:`action_cam_combined`, in its weights' order
and masked on lifts and stops). Candidates are picked on this raw sum (``analyze._pick_windows``,
``actioncam.pick``), so its cut-offs and the candidate set do not depend on which features are
available. A candidate's 0-100 score is its peak divided by the weights of the available
features (:func:`cutpoint.features.to_score`): on Cutpoint lite the ``events`` features are left
out of the divisor and the score is capped at 100; with every feature available it is the sum
of all weights. One exception (#950): an action-cam source without camera telemetry also leaves
the telemetry features out of its divisor, and its pick floor is scaled by the same ratio
(:func:`cutpoint.actioncam.source_available`, :func:`cutpoint.actioncam.source_floor`), so its
moments rank with those of the sources that have telemetry. Action-cam's ``social`` stays in the lite divisor: it is core, built from transcript
laughter (``social_said``, which runs on lite) and sound events (``social_heard``, ``events``
only), so a lite peak reaches at most 100 and the cap guards only grids cached by a full
install and learned presets.

Bonus stage: :func:`rescore_events` adds the ``events.json`` points (sound events, HUD event
types, reactions) to each candidate's analysis score and blends in the best overlapping VLM
window, boost-only (see :mod:`cutpoint.events.scoring` for the evidence).

The action-cam grids (:mod:`cutpoint.actioncam`) are registered features too (:data:`cutpoint.features.
ACTION_CAM_FEATURES`); action-cam keeps its own clip geometry and picking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cutpoint import default_fit, features

if TYPE_CHECKING:
    from cutpoint.analyze import GridPart
    from cutpoint.styles import Style

VLM_WEIGHT = 0.35  # share of the gap to the VLM window score that a candidate gains


def available_for(style: Style) -> frozenset[str] | None:
    """The features ``style``'s score is renormalised over (:func:`cutpoint.features.available`;
    the same for every style)."""
    return features.available()


def combined(sig: Mapping[str, np.ndarray], style: Style) -> np.ndarray:
    """The style's weighted sum of the analysis signals ``sig``, in ``sig``'s order."""
    return features.combine(sig, style.weights, tuple(sig))


def action_cam_combined(
    grids: Mapping[str, np.ndarray], weights: Mapping[str, float], keep: np.ndarray
) -> np.ndarray:
    """An action-cam weighted sum: ``weights`` times ``grids`` in ``weights`` order, over the non-zero
    weights of grids that exist, each grid cut to ``len(keep)``; times ``keep`` (0 where the
    mask forbids clips). As before the registry, zero weights and names without a grid add
    nothing, and the sum starts from float64 zeros (cached telemetry grids are float32), so
    the bytes are the same."""
    n = len(keep)
    order = [k for k, w in weights.items() if k in grids and w]
    grids = {k: grids[k][:n] for k in order}
    return features.combine(grids, weights, order, start=np.zeros(n)) * keep


def score(
    peak: float, style: Style, avail: frozenset[str] | None, must_include: bool = False
) -> float:
    """A candidate's 0-100 score from its combined-grid ``peak``.

    A must-include crash (action-cam) scores its confidence, so it also ranks first.
    """
    if must_include:
        return round(100 * peak, 1)
    return features.to_score(peak, style.weights, avail)


def rank_key(c: dict[str, Any]) -> tuple[float, float]:
    """Sort key of a candidate record, best first: its score, then (for the many nominees that
    tie on the score their window is floored to) the grid score it had before the floor. Only
    a nominee's own ``grid_score`` counts, so records that tie keep their order."""
    return -c["score"], -c.get("grid_score", c["score"])


def with_bonus(base: float, boost: float, vlm: float | None) -> float:
    """The bonus stage of one candidate: its analysis score ``base`` plus the event ``boost``
    points, raised toward the VLM window score ``vlm`` (0-100; None without a window) by
    :data:`VLM_WEIGHT` of the gap, capped at 100, one decimal."""
    score = base + boost
    if vlm is not None:
        # Boost-only: the window model often scores real action 0, so it may raise a
        # candidate toward its window score but never pull it down.
        score += VLM_WEIGHT * max(0.0, vlm - score)
    return round(min(100.0, score), 1)


def clamp_rated(rows: Sequence[tuple[float, float | None]]) -> list[float]:
    """The final scores of ``rows`` after the vision clamp (#909): ``(before, after)`` per
    moment in the ranking before vision, ``before`` its score without a vision rating and
    ``after`` its score with it (None: not rated). A rated moment's score stays between those
    of its nearest unrated neighbours, the one above it as the upper bound (100 without one)
    and the one below as the lower bound (0 without one), both inclusive; rated moments
    between the same neighbours may change places. An unrated moment keeps ``before``; a rated
    one is rounded to one decimal."""
    out = [before if after is None else after for before, after in rows]
    upper: list[float] = []
    hi = 100.0
    for before, after in rows:
        if after is None:
            hi = before
        upper.append(hi)
    lo = 0.0
    for i in range(len(rows) - 1, -1, -1):
        before, after = rows[i]
        if after is None:
            lo = before
        else:
            out[i] = round(min(max(after, lo), upper[i]), 1)
    return out


def apply_vision_clamp(cands: list[dict[str, Any]], before: Sequence[float | None]) -> int:
    """Clamp the scores of the candidate records ``cands`` with :func:`clamp_rated`; returns
    how many changed. ``before[i]`` is ``cands[i]``'s score without its vision rating, None
    when it has no rating that counts (never rated, the rating failed or went stale): its
    score is then its own. The ranking before vision orders them by that score
    (:func:`rank_key`; a tie keeps the order of ``cands``). ``must_include`` moments, visual
    pool moments and unrated moments the default edit skipped above its cut when vision ran
    (:func:`cutpoint.analyze.vision_skipped`; the edit cannot use them) are left out: they are
    no one's neighbours.

    A rated moment whose score equals an unrated neighbour's, which the final sort
    (:func:`rank_key`, then the order of ``cands``) would put past that neighbour, is moved
    0.1 inside its bounds: below the neighbour above, above the neighbour below, if that
    stays within the other bound and does not put it past the other neighbour instead. When
    the two bounds are equal, or 0.1 apart with no room on either side, it keeps the tie and
    the final sort places it by its ``grid_score`` and its place in ``cands``. A tie that
    would not reorder stays a tie. The result depends only on the inputs, so a rescore gives
    the same scores."""
    from cutpoint.analyze import vision_skipped  # noqa: PLC0415 - imports this module
    from cutpoint.visual_pool import is_pool  # noqa: PLC0415 - imports actioncam, which imports this

    def pre(i: int) -> float:
        b = before[i]
        return cands[i]["score"] if b is None else b

    idx = [
        i
        for i, c in enumerate(cands)
        if not c.get("must_include")
        and not is_pool(c)
        and not (before[i] is None and vision_skipped(c))
    ]
    idx.sort(key=lambda i: (-pre(i), -cands[i].get("grid_score", pre(i))))
    rows = [(pre(i), None if before[i] is None else cands[i]["score"]) for i in idx]
    scores = clamp_rated(rows)

    def key(k: int, score: float) -> tuple[float, float, int]:  # the final sort's, row k at score
        i = idx[k]
        return -score, -cands[i].get("grid_score", score), i

    above: list[int | None] = []  # the nearest unrated row above each row
    near: int | None = None
    for k, (_, after) in enumerate(rows):
        if after is None:
            near = k
        above.append(near)
    below: list[int | None] = [None] * len(rows)  # and below
    near = None
    for k in range(len(rows) - 1, -1, -1):
        if rows[k][1] is None:
            near = k
        below[k] = near

    def past(k: int, score: float, n: int | None) -> bool:  # row k at score sorts past row n
        if n is None or score != scores[n]:
            return False
        return key(k, score) < key(n, scores[n]) if n < k else key(k, score) > key(n, scores[n])

    for k, (_, after) in enumerate(rows):
        if after is None:
            continue
        up, down = above[k], below[k]
        hi = 100.0 if up is None else scores[up]
        lo = 0.0 if down is None else scores[down]
        if past(k, scores[k], up):
            nudged = round(scores[k] - 0.1, 1)
        elif past(k, scores[k], down):
            nudged = round(scores[k] + 0.1, 1)
        else:
            continue
        if lo <= nudged <= hi and not past(k, nudged, up) and not past(k, nudged, down):
            scores[k] = nudged
    changed = 0
    for i, score in zip(idx, scores, strict=True):
        if score != cands[i]["score"]:
            cands[i]["score"] = score
            changed += 1
    return changed


def rescore_events(
    hl: dict[str, Any], ev: dict[str, Any], project: Path, parts: list[GridPart] | None = None
) -> int:
    """The bonus stage of a whole analysis: rescore ``hl`` (a highlights.json object) of
    ``project`` from the events ``ev`` (an events.json object); returns the candidates changed.

    Each candidate keeps its analysis score as ``score_pre_events`` (so re-running is
    idempotent) and records the evidence in ``signals.events``. A vision rating blended into
    it counts only while the candidate keeps the span it was rated over
    (:func:`cutpoint.analyze.sync_vision`), and a rated candidate's score stays between those of
    its unrated neighbours in the ranking without the ratings, the moments the default edit
    skipped above its cut excepted (:func:`apply_vision_clamp`; the
    count goes to ``settings.vision.clamped`` when there is a ``settings.vision``). The
    candidates are re-sorted and ``suggested_top`` is recomputed. In styles with an open list (``gaming``), events that
    can score and that no candidate covers nominate candidates first
    (:func:`cutpoint.events.scoring.nominate`, scored from the grids ``parts`` or, without,
    from their caches); a nominee is rescored like the others. Visual pool moments
    (:mod:`cutpoint.visual_pool`) are left as they are, last, in their order, and outside
    ``suggested_top``.
    """
    from cutpoint import analyze  # noqa: PLC0415 - imports this module
    from cutpoint.events import scoring  # noqa: PLC0415 - the evidence; imports this module
    from cutpoint.events.eventtypes import DEFAULT_EVENT_TYPES, resolve  # noqa: PLC0415
    from cutpoint.visual_pool import is_pool  # noqa: PLC0415 - imports actioncam, which imports this

    scoring.nominate(hl, ev, project, parts)
    by_src: dict[str, list[dict[str, Any]]] = {}
    for e in ev.get("events", []):
        by_src.setdefault(e["source"], []).append(e)
    windows = ev.get("windows", [])
    types = resolve(project, ev)
    style = scoring.style_and_base(hl.get("style"))
    language = (hl.get("settings") or {}).get("language")
    segments: dict[str, list[dict[str, Any]] | None] = {}
    cands = [c for c in hl.get("candidates", []) if not is_pool(c)]
    pool = [c for c in hl.get("candidates", []) if is_pool(c)]
    old = [c["score"] for c in cands]
    unrated: list[float | None] = []  # the score without the vision rating, for the clamp
    for c in cands:
        base = c.get("score_pre_events", c["score"])
        rated = analyze.sync_vision(c)  # the vision blend only while the span is the one rated
        if rated is not None:
            base = rated
        nominated = c["signals"]["events"]["nominated"] if scoring.is_nominated(c) else None
        src = c["source"]
        src_types = types.get(src, DEFAULT_EVENT_TYPES)
        if src not in segments:
            reacts = any(t.reaction is not None and t.applies(*style) for t in src_types)
            segments[src] = scoring._segments(src, language) if reacts else None
        boost, evidence = scoring.event_boost(
            c, by_src.get(src, []), src_types, style=style, segments=segments[src]
        )
        win = scoring.best_window(c, windows)
        vlm = None
        if win:
            vlm = max(win.get("highlight_score", 0), win.get("funny_score", 0)) * 10
            evidence["vlm"] = {
                "highlight": win.get("highlight_score"),
                "funny": win.get("funny_score"),
                "label": win.get("label"),
                "window": [win["start"], win["end"]],
            }
        score = with_bonus(base, boost, vlm)
        active = rated is not None and "vision_stale" not in c  # a stale rating: unrated
        unrated.append(with_bonus(c["vision_rated"]["base"], boost, vlm) if active else None)
        c["score_pre_events"] = base
        c.setdefault("signals", {})["events"] = {
            "boost": round(boost, 1),
            **evidence,
            **({"nominated": nominated} if nominated else {}),
        }
        c["score"] = score
    clamped = apply_vision_clamp(cands, unrated)
    vision = (hl.get("settings") or {}).get("vision")
    if isinstance(vision, dict):
        vision["clamped"] = clamped
    changed = sum(c["score"] != o for c, o in zip(cands, old, strict=True))
    cands.sort(key=lambda c: (not c.get("must_include"), *rank_key(c)))  # forced crashes first
    hl["candidates"][:] = cands + pool  # in place, as the sort was
    target = (hl.get("settings") or {}).get("target_total")
    if target:
        n = len(default_fit.fit_target(cands, target))
        hl["suggested_top"] = n
    hl.setdefault("settings", {})["events_rescored"] = ev.get("created")
    return changed
