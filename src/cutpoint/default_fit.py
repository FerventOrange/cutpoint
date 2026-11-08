"""The default moment selection: the best moments that fit a style's ``target_total``.

One rule shared by ``analyze`` and the ``events`` re-rank (which count ``suggested_top``) and
by :func:`cutpoint.planning.select_moments` (which picks the moments of ``clips`` without
``--top``, ``--ids`` or ``--max-total``), so the count and the selection cannot drift apart.

Nothing here reads files.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from cutpoint.config import path_key

CLASH_SHARE = (
    0.5  # an automatic pick skips a candidate overlapping a pick by this share of the shorter
)


def span_overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        path_key(str(a["source"])) == path_key(str(b["source"]))
        and a["start"] < b["end"]
        and b["start"] < a["end"]
    )


def clashes(c: dict[str, Any], picked: Iterable[dict[str, Any]]) -> bool:
    """Whether ``c`` mostly overlaps a moment of ``picked`` from the same source: by at least
    :data:`CLASH_SHARE` of the shorter one (a candidate without a span, as in a hand-written
    highlights.json, overlaps nothing)."""
    keys = ("source", "start", "end")
    if not all(k in c for k in keys):
        return False
    return any(
        all(k in p for k in keys)
        and span_overlaps(c, p)
        and min(c["end"], p["end"]) - max(c["start"], p["start"])
        >= CLASH_SHARE * min(c["end"] - c["start"], p["end"] - p["start"])
        for p in picked
    )


def length(c: dict[str, Any]) -> float:
    """A candidate's length in seconds (``duration`` when it has no span)."""
    if "start" in c and "end" in c:
        return float(c["end"] - c["start"])
    return float(c["duration"])


def fit_target(candidates: Iterable[dict[str, Any]], target: float) -> list[dict[str, Any]]:
    """The default pick: the best candidates whose lengths fit in ``target`` seconds.

    ``candidates`` are the scored (not visual pool) moments. They are taken must-include first,
    then best score first (a tie keeps the given order). A candidate that does not fit in the
    room left, or that mostly overlaps a pick from the same source, is skipped and the next one
    is tried. A ``must_include`` candidate is always kept, even past ``target``. When nothing is
    picked, the best candidate is, so the edit is never empty. Returns the picks in that order.
    """
    ordered = sorted(candidates, key=lambda c: (not c.get("must_include"), -c["score"]))
    picked: list[dict[str, Any]] = []
    total = 0.0
    for c in ordered:
        if c.get("must_include") or (total + length(c) <= target and not clashes(c, picked)):
            picked.append(c)
            total += length(c)
    return picked or ordered[:1]
