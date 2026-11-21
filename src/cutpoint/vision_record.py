"""The vision rating's record on a candidate: the blend into its score, and whether the
rating (and the default edit's skip mark of the vision step) still holds for its span.

Stdlib only, so :mod:`cutpoint.scorer` and the event layers import it without a cycle through
:mod:`cutpoint.analyze`, whose vision step rates the candidates
(``vision_step._rate_vision``).
"""

from __future__ import annotations

from typing import Any

VISION_SPAN_EPS = 1e-3  # seconds: a candidate span this close to its rated one still matches


def vision_blend(score: float, rating: float | None, weight: float) -> float:
    """``score`` with a 0-10 vision ``rating`` applied as a bounded multiplier
    (``1 + weight * (rating - 5) / 5``, ``weight`` being the style's
    ``vision_weight``), kept within 0-100; an unrated candidate (``rating``
    None) keeps its score."""
    if rating is None:
        return score
    return round(max(0.0, min(100.0, score * (1 + weight * (rating - 5) / 5))), 1)


def sync_vision(c: dict[str, Any]) -> float | None:
    """Keep the vision rating of the candidate ``c`` only while it describes the clip (#802).

    ``vision_rated`` (written by ``vision_step._rate_vision``) holds the span the frames were rated
    over and the analysis score before and after the blend. While ``c`` has that span (within
    :data:`VISION_SPAN_EPS`) the rating counts; once the span changes (widened, trimmed, merged
    back) it moves to ``vision_stale`` (``score``, ``label``, ``window``): ``signals.vision``
    becomes None and ``vision_label`` goes. The span coming back restores both. The clip's
    ``label`` is never touched. Returns the analysis score to rank by (the
    blended one or the one before the blend); None for a candidate without ``vision_rated``
    (a project rated before this, or never rated), which is left as it is."""
    rated = c.get("vision_rated")
    if not isinstance(rated, dict):
        return None
    sig = c.setdefault("signals", {})
    stale = c.get("vision_stale")
    if (
        abs(c["start"] - rated["start"]) <= VISION_SPAN_EPS
        and abs(c["end"] - rated["end"]) <= VISION_SPAN_EPS
    ):
        if isinstance(stale, dict):
            sig["vision"] = stale.get("score")
            if stale.get("label"):
                c["vision_label"] = stale["label"]
            del c["vision_stale"]
        return rated["blended"]
    if not isinstance(stale, dict):
        stale = c["vision_stale"] = {
            "score": sig.get("vision"),
            "label": c.pop("vision_label", None),
            "window": [rated["start"], rated["end"]],
        }
        sig["vision"] = None
    return rated["base"]


def vision_skipped(c: dict[str, Any]) -> bool:
    """Whether the candidate ``c`` is a moment the default edit skipped above its cut when the
    vision step ran (too long or overlapping a pick; ``vision_skipped``, the span it had then)
    and still has that span (within :data:`VISION_SPAN_EPS`). The vision clamp
    (:func:`cutpoint.scorer.apply_vision_clamp`) does not count such a moment as a neighbour.
    A re-rank (:func:`cutpoint.scorer.rescore_events`) also removes the mark once the default
    edit of the ranking without the ratings takes the moment (#949)."""
    span = c.get("vision_skipped")
    return (
        isinstance(span, list)
        and abs(c["start"] - span[0]) <= VISION_SPAN_EPS
        and abs(c["end"] - span[1]) <= VISION_SPAN_EPS
    )
