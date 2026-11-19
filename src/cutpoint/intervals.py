"""Time intervals ``(start, end)`` in seconds: merge, overlap and the free stretch around a time.

Stdlib only. ``overlaps`` treats intervals as half-open: two that only touch do not overlap.
``merge`` joins touching intervals as well.
"""

from __future__ import annotations

from collections.abc import Iterable


def merge(spans: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """``spans`` sorted, the ones that overlap or touch merged."""
    out: list[tuple[float, float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length of the overlap of ``[a0, a1]`` and ``[b0, b1]`` (0.0 when they do not)."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    """Whether ``[a0, a1)`` and ``[b0, b1)`` share time (touching ones do not)."""
    return a0 < b1 and b0 < a1


def free_around(
    t: float, taken: Iterable[tuple[float, float]], lo: float, hi: float
) -> tuple[float, float]:
    """The stretch of ``[lo, hi]`` around ``t`` that no ``taken`` interval covers: ``(t, t)``
    when one covers ``t`` (strictly inside it); one that starts or ends at ``t`` bounds it."""
    taken = list(taken)
    if any(a < t < b for a, b in taken):
        return t, t
    return max([lo, *(b for a, b in taken if b <= t)]), min([hi, *(a for a, b in taken if a >= t)])
