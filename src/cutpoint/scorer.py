"""How highlight candidates are scored: weighted feature grids, then the events' bonus points.

Grid stage: the style's weights times the feature grids (:mod:`cutpoint.features`), summed in
the analysis signals' order (:func:`combined`; ski: :func:`ski_combined`, in its weights' order
and masked on lifts and stops). Candidates are picked on this raw sum (``analyze._pick_windows``,
``ski.pick``), so its cut-offs and the candidate set do not depend on which features are
available. A candidate's 0-100 score is its peak divided by the weights of the available
features (:func:`cutpoint.features.to_score`): on Cutpoint lite the ``events`` features are left
out of the divisor and the score is capped at 100; with every feature available it is the sum
of all weights.

Bonus stage: :func:`rescore_events` adds the ``events.json`` points (sound events, HUD event
types, reactions) to each candidate's analysis score and blends in the best overlapping VLM
window, boost-only (see :mod:`cutpoint.events.scoring` for the evidence).

The ski grids (:mod:`cutpoint.ski`) are registered features too (:data:`cutpoint.features.
SKI_FEATURES`); ski keeps its own clip geometry and picking.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cutpoint import features

if TYPE_CHECKING:
    from cutpoint.styles import Style

VLM_WEIGHT = 0.35  # share of the gap to the VLM window score that a candidate gains


def available_for(style: Style) -> frozenset[str] | None:
    """The features ``style``'s score is renormalised over (:func:`cutpoint.features.available`;
    the same for every style)."""
    return features.available()


def combined(sig: Mapping[str, np.ndarray], style: Style) -> np.ndarray:
    """The style's weighted sum of the analysis signals ``sig``, in ``sig``'s order."""
    return features.combine(sig, style.weights, tuple(sig))


def ski_combined(
    grids: Mapping[str, np.ndarray], weights: Mapping[str, float], keep: np.ndarray
) -> np.ndarray:
    """A ski weighted sum: ``weights`` times ``grids`` in ``weights`` order, over the non-zero
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

    A must-include crash (ski) scores its confidence, so it also ranks first.
    """
    if must_include:
        return round(100 * peak, 1)
    return features.to_score(peak, style.weights, avail)


def with_bonus(base: float, boost: float, vlm: float | None) -> float:
    """The bonus stage of one candidate: its analysis score ``base`` plus the event ``boost``
    points, raised toward the VLM window score ``vlm`` (0-100; None without a window) by
    :data:`VLM_WEIGHT` of the gap, capped at 100, one decimal."""
    score = base + boost
    if vlm is not None:
        # Boost-only: Omni-7B often scores real action 0, so it may raise a
        # candidate toward its window score but never pull it down.
        score += VLM_WEIGHT * max(0.0, vlm - score)
    return round(min(100.0, score), 1)


def rescore_events(hl: dict[str, Any], ev: dict[str, Any], project: Path) -> int:
    """The bonus stage of a whole analysis: rescore ``hl`` (a highlights.json object) of
    ``project`` from the events ``ev`` (an events.json object); returns the candidates changed.

    Each candidate keeps its analysis score as ``score_pre_events`` (so re-running is
    idempotent) and records the evidence in ``signals.events``. The candidates are re-sorted
    and ``suggested_top`` is recomputed.
    """
    from cutpoint.events import scoring  # noqa: PLC0415 - the evidence; imports this module
    from cutpoint.events.eventtypes import DEFAULT_EVENT_TYPES, resolve  # noqa: PLC0415

    by_src: dict[str, list[dict[str, Any]]] = {}
    for e in ev.get("events", []):
        by_src.setdefault(e["source"], []).append(e)
    windows = ev.get("windows", [])
    types = resolve(project, ev)
    style = scoring.style_and_base(hl.get("style"))
    language = (hl.get("settings") or {}).get("language")
    segments: dict[str, list[dict[str, Any]] | None] = {}
    changed = 0
    for c in hl.get("candidates", []):
        base = c.get("score_pre_events", c["score"])
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
        c["score_pre_events"] = base
        c.setdefault("signals", {})["events"] = {"boost": round(boost, 1), **evidence}
        if score != c["score"]:
            changed += 1
        c["score"] = score
    hl["candidates"].sort(key=lambda c: -c["score"])
    target = (hl.get("settings") or {}).get("target_total")
    if target:
        total, n = 0.0, 0
        for c in hl["candidates"]:
            if total >= target:
                break
            total += c["end"] - c["start"]
            n += 1
        hl["suggested_top"] = n
    hl.setdefault("settings", {})["events_rescored"] = ev.get("created")
    return changed
