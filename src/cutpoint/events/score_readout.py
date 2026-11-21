"""Score readout of the OCR layer (split from ``killfeed.py``, #1085 A347):
:class:`ScoreTracker` turns a "score" region's numbers into ``score`` events.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from cutpoint.events.discover import SCORE_SLOT_PX, score_value

# Score readout (:class:`ScoreTracker`): the number in a "score" region (a round score, a
# counter; :func:`~cutpoint.events.discover.score_value`) is read per slot (boxes whose centres
# lie within half of SCORE_SLOT_PX across). A value counts once read SCORE_CONFIRM times in a
# row at SCORE_MIN_CONF or more. A change is an event when the new value is one more or one
# less than the old, the old one held SCORE_HOLD_S, and the HUD showed no number for at most
# SCORE_MAX_GAP_S between them; each slot keeps the direction of its first change (a counter
# that counts down is fine, a step back afterwards is no event). Anything else is taken as the
# new value silently (a reset between matches). A value that comes back within SCORE_HOLD_S is
# a misread. A slot whose value changes more often than every SCORE_HOLD_S (median) or that
# shows more than SCORE_MAX_VALUES values is no score (a clock, ammo): no events from it.
# Nor is a slot that once moved by two or more in its direction to a value that held
# SCORE_HOLD_S, with at most SCORE_MAX_GAP_S between them (:func:`_score_jumps`): a
# score goes up or down one point at a time, a count of items (gadgets placed, charges used)
# drops by several at once; on a 60 min recording a gadget counter beside the team badge
# was taken for a round score that way (#872). A jump the other way is a reset and stays fine.
SCORE_CONFIRM = 2
SCORE_MIN_CONF = 0.7
SCORE_HOLD_S = 15.0
SCORE_MAX_GAP_S = 60.0
SCORE_MAX_VALUES = 12


@dataclass
class _ScoreRun:
    """Reads of one value in a row (:class:`ScoreTracker`)."""

    value: int
    first: float
    last: float
    n: int
    conf: float  # the sum of the reads' confidences

    def held(self) -> float:
        return self.last - self.first


def _score_runs(reads: list[tuple[float, int, float]]) -> list[_ScoreRun]:
    """The confirmed values of one slot in time order: runs of :data:`SCORE_CONFIRM` reads or
    more of one value (each within :data:`SCORE_MAX_GAP_S` of the last), runs of one value
    joined across the misreads between them, and a value that comes back within
    :data:`SCORE_HOLD_S` joined across the other one (a misread too)."""
    runs: list[_ScoreRun] = []
    for t, v, c in reads:
        last = runs[-1] if runs else None
        if last is not None and last.value == v and t - last.last <= SCORE_MAX_GAP_S:
            last.last, last.n, last.conf = t, last.n + 1, last.conf + c
        else:
            runs.append(_ScoreRun(v, t, t, 1, c))
    out: list[_ScoreRun] = []
    for r in (r for r in runs if r.n >= SCORE_CONFIRM):
        if out and out[-1].value == r.value and r.first - out[-1].last <= SCORE_MAX_GAP_S:
            out[-1] = _join(out[-1], r)
        else:
            out.append(r)
    i = 1
    while i < len(out) - 1:
        a, b, c = out[i - 1 : i + 2]
        if a.value == c.value and b.held() < SCORE_HOLD_S and c.first - a.last <= SCORE_MAX_GAP_S:
            out[i - 1 : i + 2] = [_join(a, c)]
            i = max(1, i - 1)
        else:
            i += 1
    return out


def _join(a: _ScoreRun, b: _ScoreRun) -> _ScoreRun:
    return _ScoreRun(a.value, a.first, b.last, a.n + b.n, a.conf + b.conf)


class ScoreTracker:
    """The score readout of "score" regions: their reads per slot during the pass
    (:meth:`see`), the changes of the value as ``score`` events at its end (:meth:`finish`,
    rules at :data:`SCORE_SLOT_PX`)."""

    def __init__(self) -> None:
        self.reads: dict[tuple[str, int], list[tuple[float, int, float]]] = {}
        self._slots: dict[str, list[float]] = {}  # per region the slots' centre x (reference px)

    def see(
        self,
        t: float,
        region: str,
        items: list[tuple[list, str, float]],
        ref: tuple[float, float, float, float],
    ) -> None:
        """Record the numbers of one read of ``region`` at ``t`` (``ref``: the crop's left and
        top in frame px and the frame-to-reference factors): per slot the surest one."""
        rx, _, sx, _ = ref
        best: dict[int, tuple[int, float]] = {}
        for box, text, score in items:
            v = score_value(text)
            if v is None or score < SCORE_MIN_CONF:
                continue
            cx = (rx + (min(p[0] for p in box) + max(p[0] for p in box)) / 2) * sx
            slot = self._slot(region, cx)
            if slot not in best or score > best[slot][1]:
                best[slot] = (v, float(score))
        for slot, (v, c) in best.items():
            self.reads.setdefault((region, slot), []).append((t, v, c))

    def _slot(self, region: str, cx: float) -> int:
        """The slot of a number centred at ``cx`` (reference px): the first one within half
        of :data:`SCORE_SLOT_PX`, or a new one there; named by its centre x."""
        xs = self._slots.setdefault(region, [])
        for x in xs:
            if abs(cx - x) <= SCORE_SLOT_PX / 2:
                return round(x)
        xs.append(cx)
        return round(cx)

    def finish(self, menu: set[float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """The ``score`` events of the pass, reads at the ``menu`` frame times left out,
        and per slot (``<region>@<x>``) its ``reads``, ``values``, ``changes`` and whether it
        was ``disabled`` (no score: it changes too often, shows too many values or jumped in
        its direction, :func:`_score_jumps`)."""
        events: list[dict[str, Any]] = []
        stats: dict[str, Any] = {}
        for (region, slot), reads in self.reads.items():
            runs = _score_runs([r for r in reads if r[0] not in menu])
            spans = [b.first - a.first for a, b in pairwise(runs)]
            disabled = (
                len({r.value for r in runs}) > SCORE_MAX_VALUES
                or (len(spans) >= 2 and float(np.median(spans)) < SCORE_HOLD_S)
                or _score_jumps(runs) > 0
            )
            found = [] if disabled else _score_changes(runs, region, slot)
            events += found
            stats[f"{region}@{slot}"] = {
                "reads": len(reads),
                "values": len({r.value for r in runs}),
                "changes": len(found),
                "disabled": disabled,
            }
        return events, stats


def _in_play(a: _ScoreRun, b: _ScoreRun) -> bool:
    """Whether the change from ``a`` to ``b`` happened in play: ``a`` held
    :data:`SCORE_HOLD_S` and at most :data:`SCORE_MAX_GAP_S` passed until ``b``."""
    return a.held() >= SCORE_HOLD_S and b.first - a.last <= SCORE_MAX_GAP_S


def _score_jumps(runs: list[_ScoreRun]) -> int:
    """How often one slot's value moved by two or more in its direction (the sign of its
    first in-play change by one, :func:`_in_play`) to a value that then held
    :data:`SCORE_HOLD_S`, at most :data:`SCORE_MAX_GAP_S` after the old one: a count of
    items, no score (:data:`SCORE_MAX_VALUES`). The old value need not have held: menus
    and the HUD hidden between rounds leave it few reads. None without a direction."""
    pairs = list(pairwise(runs))
    step = next(
        (b.value - a.value for a, b in pairs if abs(b.value - a.value) == 1 and _in_play(a, b)), 0
    )
    return sum(
        (b.value - a.value) * step >= 2
        and b.held() >= SCORE_HOLD_S
        and b.first - a.last <= SCORE_MAX_GAP_S
        for a, b in pairs
    )


def _score_changes(runs: list[_ScoreRun], region: str, slot: int) -> list[dict[str, Any]]:
    """The ``score`` events of one slot's confirmed values (:data:`SCORE_SLOT_PX`): at the last
    read of the old value, ending at the first read of the new one."""
    out = []
    step = 0  # the slot's direction, once set
    for a, b in pairwise(runs):
        d = b.value - a.value
        gap = b.first - a.last
        if abs(d) != 1 or not _in_play(a, b):
            continue  # a reset or a misread: the new value counts from here
        if step and d != step:
            continue  # a step back
        step = d
        out.append(
            {
                "t": round(a.last, 2),
                "end": round(b.first, 2),
                "type": "score",
                "text": f"{a.value} > {b.value}",
                "confidence": round(min(a.conf / a.n, b.conf / b.n), 3),
                "region": region,
                "before": a.value,
                "after": b.value,
                "seen_at": round(b.first, 2),
                "gap_s": round(gap, 2),
                "slot": slot,
            }
        )
    return out
