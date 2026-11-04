"""Fill a proposal that is still short of the length the user asked for, in code.

A local model gives up after a few send-backs well short of "about 4 minutes". When the final
proposal is still more than the tolerance under the target, :func:`top_up` appends real ops to it,
marked ``"auto": true``, so the user reviews and applies them like any other:

1. lengthen the ends of the sequence's clips that end in the middle of speech or are very short,
   each within the per-edge cap of a proposal
   (``ops.MAX_GROW_S``) and never past its source, snapped to the end of the phrase there by the
   trim op;
2. then take turns: add one of this turn's search results (``find_moments`` / ``look_for``, one
   from each search in turn) that is not in the sequence and does not overlap a clip of the same
   source, and lengthen the next clip.

It stops once the proposal is close enough to the target and never goes over the tolerance.
It measures every try with ``evaluate`` (the session's own check of ops: the same apply path as
the model's proposal), so nothing here knows how a sequence is applied.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cutpoint.assistant.ops import MAX_GROW_S

STOP_SHARE = 0.95  # of the target: close enough, stop filling
MIN_STEP_S = 0.5  # a lengthening of less than this is not worth an op
SHORT_CLIP_S = 5.0  # a clip shorter than this is lengthened before any moment is added

Evaluate = Callable[[list[dict[str, Any]]], dict[str, Any] | None]


@dataclass
class TopUp:
    """What :func:`top_up` did: the ``checked`` proposal with the extra ops, the ids of the
    clips ``lengthened`` and ``added``, and whether the length is within the tolerance."""

    checked: dict[str, Any]
    lengthened: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    reached: bool = False

    @property
    def count(self) -> int:
        return len(self.lengthened) + len(self.added)


def _overlaps(row: Mapping[str, Any], others: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        o["source"] == row["source"] and o["start"] < row["end"] and row["start"] < o["end"]
        for o in others
    )


def _round_robin(groups: Sequence[Sequence[str]]) -> Iterator[str]:
    """The ids of ``groups`` one from each in turn (the first of each, then the second of each,
    ...), every id once."""
    seen: set[str] = set()
    for rank in range(max((len(g) for g in groups), default=0)):
        for group in groups:
            if rank < len(group) and group[rank] not in seen:
                seen.add(group[rank])
                yield group[rank]


def kind_groups(
    rows: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    kind_of: Callable[[Mapping[str, Any]], str],
) -> list[list[str]]:
    """Search-less groups for :func:`top_up`: per kind already in ``rows`` (in the order they
    first appear), the unused moments of that kind, best score first. When no unused moment has
    such a kind, one group of all the unused ones."""
    used = {r["id"] for r in rows}
    unused = sorted(
        (c for c in candidates.values() if c["id"] not in used),
        key=lambda c: -(c["score"] if isinstance(c.get("score"), (int, float)) else 0),
    )
    present = list(dict.fromkeys(kind_of(candidates.get(r["id"]) or r) for r in rows))
    groups = [[c["id"] for c in unused if kind_of(c) == k] for k in present]
    groups = [g for g in groups if g]
    return groups or ([[c["id"] for c in unused]] if unused else [])


def top_up(
    checked: dict[str, Any],
    *,
    target: float,
    tolerance: float,
    evaluate: Evaluate,
    before: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Mapping[str, Any]],
    searched: Sequence[Sequence[str]],
    durations: Mapping[str, float | None],
    phrase_end: Callable[[str, float], float] | None = None,
) -> TopUp | None:
    """Fill ``checked`` (a proposal that :func:`evaluate` made: ``ops``, ``after``, ``length``)
    towards ``target`` seconds; None when it is not short of it or nothing could be added.

    Args:
        checked: The proposal to fill.
        target: The length the user asked for, in seconds.
        tolerance: The share of ``target`` that is still near enough (0.15).
        evaluate: ``ops -> checked`` for the whole op list on the turn's sequence, or None when
            they do not apply.
        before: The sequence the proposal changes.
        candidates: The moments ``add`` can take, by id.
        searched: The ids each of this turn's searches found, one list per search, in
            order; moments are taken one from each search in turn.
        durations: Source path -> seconds, the end a clip cannot pass.
        phrase_end: Where the speech at a time of a source ends (to find clips cut mid-speech).
    """
    if checked["length"] >= target * (1 - tolerance):
        return None
    upper = target * (1 + tolerance)
    base = list(checked["ops"])
    extra: list[dict[str, Any]] = []
    cur = checked
    result = TopUp(checked)
    origin = {s["id"]: s for s in before}

    def attempt(op: dict[str, Any]) -> bool:
        nonlocal cur
        got = evaluate([*base, *extra, op])
        if got is None or got["length"] > upper or got["length"] <= cur["length"]:
            return False
        extra.append(op)
        cur = got
        return True

    def done() -> bool:
        return cur["length"] >= target * STOP_SHARE

    def abrupt(row: Mapping[str, Any]) -> bool:
        return phrase_end is not None and phrase_end(row["source"], row["end"]) > row["end"] + 1e-6

    rows = sorted(cur["after"] or [], key=lambda r: (not abrupt(r), r["end"] - r["start"]))
    urgent = [r for r in rows if abrupt(r) or r["end"] - r["start"] < SHORT_CLIP_S]
    rest = iter([r for r in rows if r not in urgent])

    def lengthen(row: Mapping[str, Any]) -> bool:
        first = origin.get(row["id"]) or candidates.get(row["id"]) or row
        room = float(first["end"]) + MAX_GROW_S - float(row["end"])
        source_end = durations.get(row["source"])
        if source_end is not None:
            room = min(room, float(source_end) - float(row["end"]))
        want = min(room, target - cur["length"])
        if want < MIN_STEP_S:
            return False
        end = round(float(row["end"]) + want, 3)
        if attempt({"op": "trim", "id": row["id"], "end": end, "auto": True}):
            result.lengthened.append(row["id"])
            return True
        return False

    def lengthen_next() -> bool:
        return any(lengthen(row) for row in rest)  # the next row that takes it

    hits = _round_robin(searched)

    def add_next() -> bool:
        for cid in hits:
            cand = candidates.get(cid)
            if cand is None or any(r["id"] == cid for r in cur["after"] or []):
                continue
            if _overlaps(cand, cur["after"] or []):
                continue
            if attempt({"op": "add", "id": cid, "auto": True}):
                result.added.append(cid)
                return True
        return False

    for row in urgent:  # the abrupt and the very short ends first
        if done():
            break
        lengthen(row)
    adding = True  # then take turns, so neither the clips nor the searches are used up first
    while not done():
        if not (add_next() if adding else lengthen_next()) and not (
            lengthen_next() if adding else add_next()
        ):
            break
        adding = not adding
    if not result.count:
        return None
    result.checked = cur
    result.reached = cur["length"] >= target * (1 - tolerance)
    return result
