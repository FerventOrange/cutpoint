"""Sequence edits the assistant proposes, checked and applied without side effects.

A sequence is the review's segment list (``{id, source, start, end, label}``, as
``cutpoint.ui.validate_segments`` returns it). :func:`apply_ops` applies the model's ops to a
copy and checks the result with the same rules, so a proposal that applies here also saves.
Its errors are plain sentences the model reads and retries on.

Ops (positions count from 1):

* ``{"op": "add", "id", "at"?, "start"?, "end"?}``: a moment into the sequence (default: last,
  the moment's own range).
* ``{"op": "remove", "id"}``.
* ``{"op": "move", "id", "to"}``.
* ``{"op": "trim", "id", "start"?, "end"?}`` in source seconds, or ``start_delta``/``end_delta``
  in seconds added to the current edge (negative: earlier). An end moved later is snapped to
  the end of the utterance it lands in (:data:`Snap`), never past the source's end.
* ``{"op": "order", "ids"}``: every sequence id once, in the new order.
* ``{"op": "sort", "by": "time" | "score" | "length", "desc"?}``.
* ``{"op": "color", "mode": "off" | "auto" | "underwater", "strength"?}``: the project's Color
  setting (``strength`` 0-1; left as it is when omitted), not a row of the sequence. At most one
  per proposal; :attr:`Outcome.color` has it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from cutpoint.clips import MIN_EXPORT_S
from cutpoint.config import COLOR_MODES

MAX_SEGMENTS = 500  # as cutpoint.ui.MAX_SEGMENTS (ui needs the optional fastapi)
MAX_OPS = 100  # ops in one proposal
LABEL_MAX = 200
OPS = ("add", "remove", "move", "trim", "order", "sort", "color")
SORT_KEYS = ("time", "score", "length")
LENGTH_EPS = 0.05  # a clip's length must change by more than this to be lengthened/shortened
# Where an end moved later to ``t`` (source path, seconds) ends instead: the end of the phrase it
# lands in (cutpoint.assistant.edges.EdgePlanner.phrase_end); ``t`` itself when it lands in silence.
Snap = Callable[[str, float], float]
# Words of a summary that claim a change, and the changes of the diff that make the claim true.
# The words are edit verbs, not adjectives or nouns ("longer", "colorful", "top-grade", "a steep
# drop-in" and "smooth moves" claim nothing); a hyphen before or after a word is no verb either.
CLAIMS = (
    ("extended", r"extend(?:s|ed|ing)?|lengthen(?:s|ed|ing)?", ("lengthened",)),
    ("shortened", r"shorten(?:s|ed|ing)?", ("shortened",)),
    (
        "trimmed",
        r"trim(?:s|med|ming)?|tighten(?:s|ed|ing)?",
        ("trimmed", "lengthened", "shortened"),
    ),
    ("added", r"add(?:s|ed|ing)?|insert(?:s|ed|ing)?", ("added",)),
    (
        "removed",
        r"remov(?:e|es|ed|ing)|delet(?:e|es|ed|ing)|drop(?:s|ped|ping)?|cut(?:s|ting)? (?:out|down)",
        ("removed",),
    ),
    (
        "moved",
        r"move|moved|moving|reorder(?:s|ed|ing)?|rearrang(?:e|es|ed|ing)|shuffl(?:e|es|ed|ing)",
        ("moved",),
    ),
    # "sorted by time" over clips just added is true: they are in that order
    ("sorted", r"sort(?:s|ed|ing)?|order(?:ed|ing)", ("moved", "added")),
)
_CLAIM_WORDS = [(n, re.compile(rf"(?<![\w-])(?:{w})(?![\w-])", re.I), c) for n, w, c in CLAIMS]
_COLOR_WORDS = re.compile(
    r"(?<![\w-])(?:colou?r(?:s|ed|ing)?|regrad(?:e|es|ed|ing)|colou?r[ -]grad\w+)(?![\w-])", re.I
)
MAX_GROW_S = 10.0  # a trim moves an edge outward by at most this much per proposal
ADD_SLACK_S = 10.0  # an add's start and end stay this close to the moment's own window


class OpsError(ValueError):
    """An op or the sequence it gives breaks a rule; the message is for the model."""


@dataclass
class Outcome:
    """What :func:`apply_ops` gives: ``after`` and its ``diff``, or the ``error``; ``color`` is
    the ``color`` op's ``{mode, strength}`` (strength None: leave it), None without one."""

    after: list[dict[str, Any]] | None = None
    diff: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    color: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)  # edges of an add that were ignored


class Durations(Mapping[str, float | None]):
    """Source path -> duration in seconds (None: cannot be read). The keys are the project's
    sources; a duration not known yet is asked of ``probe`` once, when an op needs it."""

    def __init__(self, known: Mapping[str, float | None], probe: Callable[[str], float | None]):
        self.known = dict(known)
        self.probe = probe
        self.probed: set[str] = set()

    def __getitem__(self, path: str) -> float | None:
        value = self.known[path]
        if value is None and path not in self.probed:
            self.probed.add(path)
            value = self.known[path] = self.probe(path)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self.known)

    def __len__(self) -> int:
        return len(self.known)


def base_key(segments: list[dict[str, Any]], highlights: dict[str, Any] | None) -> str:
    """Hash of a sequence and the analysis it was made from: a proposal made on one key applies
    as is only while both are unchanged."""
    text = json.dumps([segments, highlights], sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def apply_ops(
    before: list[dict[str, Any]],
    ops: Any,
    candidates: list[dict[str, Any]],
    durations: Mapping[str, float | None],
    *,
    name: Callable[[str], str] = str,
    snap: Snap | None = None,
) -> Outcome:
    """Apply ``ops`` to a copy of the sequence ``before``.

    Args:
        before: The current sequence (never changed).
        ops: The model's ops (any JSON; checked here).
        candidates: The analysis moments (``moments_view`` ids), the only ones ``add`` takes.
        durations: The project's sources and their durations (:class:`Durations`).
        name: How an id is written in error messages (the short id the model knows).
        snap: Where a trim that moves an end later puts it (None: exactly where asked).

    Returns:
        ``after`` (checked and cleaned as ``validate_segments`` cleans it) and a per-row
        :func:`diff`, or ``error`` naming the op or row and what is wrong.
    """
    color: dict[str, Any] = {}
    notes: list[str] = []
    try:
        rows = _apply(before, ops, candidates, durations, name, color, snap, notes)
        after = _check(rows, durations, name)
    except OpsError as err:
        return Outcome(error=str(err))
    return Outcome(after, diff(before, after), color=color or None, notes=notes)


def _number(op: dict[str, Any], key: str, where: str) -> float | None:
    v = op.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise OpsError(f"{where}: {key} must be a number of seconds")
    return float(v)


def _whole(v: Any) -> Any:
    """``v`` as an int when it is a whole number given as an int, a float (``3.0``) or a string
    of digits (``"3"``); anything else (bool, fraction, other text) as it is."""
    if isinstance(v, float) and math.isfinite(v) and v == int(v):
        return int(v)
    if isinstance(v, str) and v.isascii() and v.isdigit():
        return int(v)
    return v


def _position(op: dict[str, Any], key: str, top: int, where: str) -> int | None:
    v = _whole(op.get(key))
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= top:
        raise OpsError(f"{where}: {key} must be a position from 1 to {top}")
    return v


def _apply(
    before: list[dict[str, Any]],
    ops: Any,
    candidates: list[dict[str, Any]],
    durations: Mapping[str, float | None],
    name: Callable[[str], str],
    color: dict[str, Any],
    snap: Snap | None = None,
    notes: list[str] | None = None,
) -> list[dict[str, Any]]:
    notes = [] if notes is None else notes
    if not isinstance(ops, list) or not ops:
        raise OpsError("ops must be a non-empty list of edits")
    if len(ops) > MAX_OPS:
        raise OpsError(f"at most {MAX_OPS} ops in one proposal")
    rows = [dict(s) for s in before]
    origin = {s["id"]: s for s in before}
    cands = {c["id"]: c for c in candidates}
    order = {path: i for i, path in enumerate(durations)}

    def index(sid: Any, where: str) -> int:
        if not isinstance(sid, str):
            raise OpsError(f"{where}: id must be a moment id")
        for i, row in enumerate(rows):
            if row["id"] == sid:
                return i
        raise OpsError(f"{where}: {name(sid)} is not in the sequence")

    for n, op in enumerate(ops, 1):
        if not isinstance(op, dict) or op.get("op") not in OPS:
            raise OpsError(f"op {n}: expected an object with op one of {', '.join(OPS)}")
        kind = op["op"]
        sid = op.get("id")
        where = f"op {n} ({kind} {name(sid)})" if isinstance(sid, str) else f"op {n} ({kind})"
        if kind == "color":
            if color:
                raise OpsError(f"{where}: only one color op in a proposal")
            color.update(_color_op(op, where))
        elif kind == "add":
            cand = cands.get(sid) if isinstance(sid, str) else None
            if cand is None:
                raise OpsError(f"{where}: not a moment of this project")
            if any(row["id"] == sid for row in rows):
                raise OpsError(f"{where}: already in the sequence; use move or trim")
            at = _whole(op.get("at"))
            if at is not None and (isinstance(at, bool) or not isinstance(at, int) or at < 0):
                raise OpsError(f"{where}: at must be a position from 1 to {len(rows) + 1}")
            at = at if at and at <= len(rows) + 1 else len(rows) + 1  # 0 or too far: append
            start, end = _number(op, "start", where), _number(op, "end", where)
            own = f"{cand['start']:g}-{cand['end']:g}"
            said = f"add {name(sid)}"
            for edge, value in (("start", start), ("end", end)):
                if value is not None and abs(value - float(cand[edge])) > ADD_SLACK_S:
                    notes.append(
                        f"{said}: {edge} {value:g} ignored (not near the moment's {own}); "
                        "added as it is"
                    )
                    start, end = (None, end) if edge == "start" else (start, None)
            first = float(cand["start"]) if start is None else start
            last = float(cand["end"]) if end is None else end
            if (start, end) != (None, None) and (first < 0 or last - first < MIN_EXPORT_S):
                given = " and ".join(
                    f"{e} {v:g}" for e, v in (("start", start), ("end", end)) if v is not None
                )
                notes.append(
                    f"{said}: {given} ignored (the clip would start before 0 or be under "
                    f"{MIN_EXPORT_S:g} s); added as it is"
                )
                start = end = None
            label = cand.get("label") if isinstance(cand.get("label"), str) else ""
            rows.insert(
                at - 1,
                {
                    "id": sid,
                    "source": cand["source"],
                    "start": float(cand["start"]) if start is None else start,
                    "end": float(cand["end"]) if end is None else end,
                    "label": label[:LABEL_MAX],
                },
            )
        elif kind == "remove":
            rows.pop(index(sid, where))
        elif kind == "move":
            i = index(sid, where)
            to = _position(op, "to", len(rows) + 1, where)  # one past the end: last
            to = None if to is None else min(to, len(rows))
            if to is None:
                raise OpsError(f"{where}: needs to, a position from 1 to {len(rows)}")
            rows.insert(to - 1, rows.pop(i))
        elif kind == "trim":
            row = rows[index(sid, where)]
            edges = 0
            for edge in ("start", "end"):
                absolute = _number(op, edge, where)
                delta = _number(op, f"{edge}_delta", where)
                if absolute is not None and delta is not None:
                    raise OpsError(f"{where}: give {edge} or {edge}_delta, not both")
                if absolute is not None or delta is not None:
                    edges += 1
                    was = row[edge]
                    row[edge] = absolute if absolute is not None else was + delta
                    first = origin.get(row["id"]) or cands.get(row["id"])  # or the moment's own
                    limit = durations.get(row["source"]) or math.inf
                    if first is not None and row[edge] <= limit + 0.01:  # past the end: _check
                        grew = row[edge] - first[edge] if edge == "end" else first[edge] - row[edge]
                        if grew > MAX_GROW_S + 1e-6:
                            raise OpsError(
                                f"{where}: {edge} may move outward by at most +{MAX_GROW_S:g} s "
                                f"per edge in a proposal (asked {grew:+g} s)"
                            )
                    if edge == "end" and snap is not None and row[edge] > was:
                        asked = row[edge]
                        row[edge] = _snapped(row, snap, durations)
                        if first is not None:  # the snap stays within the cap too
                            row[edge] = max(asked, min(row[edge], float(first["end"]) + MAX_GROW_S))
            if not edges:
                raise OpsError(f"{where}: needs start, end, start_delta or end_delta")
        elif kind == "order":
            ids = op.get("ids")
            if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
                raise OpsError(f"{where}: ids must be a list of moment ids")
            have = [row["id"] for row in rows]
            missing = [x for x in have if x not in ids]
            unknown = [x for x in ids if x not in have]
            if missing or unknown or len(ids) != len(set(ids)):
                parts = [f"missing {', '.join(map(name, missing))}" if missing else ""]
                parts += [f"not in the sequence {', '.join(map(name, unknown))}" if unknown else ""]
                said = "; ".join(p for p in parts if p) or "an id is listed twice"
                raise OpsError(f"{where}: ids must list every moment in the sequence once ({said})")
            by_id = {row["id"]: row for row in rows}
            rows = [by_id[x] for x in ids]
        else:  # sort
            by, desc = op.get("by"), op.get("desc")
            if by not in SORT_KEYS:
                raise OpsError(f"{where}: by must be one of {', '.join(SORT_KEYS)}")
            if desc is not None and not isinstance(desc, bool):
                raise OpsError(f"{where}: desc must be true or false")
            keys: dict[str, Callable[[dict[str, Any]], Any]] = {
                "time": lambda r: (order.get(r["source"], len(order)), r["start"]),
                "score": lambda r: _score(cands.get(r["id"])),
                "length": lambda r: r["end"] - r["start"],
            }
            # sort(reverse=...) keeps tied rows in their order: no false "moved" rows
            rows.sort(key=keys[by], reverse=desc if desc is not None else by == "score")
    return rows


def _snapped(row: dict[str, Any], snap: Snap, durations: Mapping[str, float | None]) -> float:
    """The end of ``row`` (just moved later) at the end of the phrase it lands in, never past
    the source's end."""
    end = row["end"]
    try:
        end = max(end, float(snap(row["source"], end)))
    except (TypeError, ValueError, KeyError):  # an end that cannot be snapped stays as it is
        return end
    duration = durations.get(row["source"])
    return min(end, duration) if duration is not None else end


def _color_op(op: dict[str, Any], where: str) -> dict[str, Any]:
    """``{mode, strength}`` of a color op; ``strength`` is None for off (it has none) and when
    the op gives none."""
    mode = op.get("mode")
    if mode not in COLOR_MODES:
        raise OpsError(f"{where}: mode must be one of {', '.join(COLOR_MODES)}")
    strength = op.get("strength")
    if strength is not None and (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not 0 <= strength <= 1
    ):
        raise OpsError(f"{where}: strength must be a number from 0 to 1")
    if strength is not None:
        strength = round(strength * 20) / 20  # the slider's 5% steps
    return {"mode": mode, "strength": None if mode == "off" else strength}


def extra_moments(raw: Any) -> list[dict[str, Any]]:
    """The ``moments`` a saved proposal carries (look_for finds that are not in
    highlights.json) that ``add`` can take again: dicts with an id, a source and times."""

    def usable(m: Any) -> bool:
        return (
            isinstance(m, dict)
            and isinstance(m.get("id"), str)
            and isinstance(m.get("source"), str)
            and all(
                isinstance(m.get(k), (int, float)) and math.isfinite(m[k]) for k in ("start", "end")
            )
        )

    return [m for m in raw if usable(m)] if isinstance(raw, list) else []


def _score(cand: dict[str, Any] | None) -> float:
    v = (cand or {}).get("score")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def _check(
    rows: list[dict[str, Any]],
    durations: Mapping[str, float | None],
    name: Callable[[str], str],
) -> list[dict[str, Any]]:
    """``rows`` cleaned, with the rules of ``validate_segments``; else :class:`OpsError`."""
    if len(rows) > MAX_SEGMENTS:
        raise OpsError(f"the sequence can hold at most {MAX_SEGMENTS} moments")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        sid, label = row.get("id"), row.get("label", "segment")
        if not isinstance(sid, str) or not 0 < len(sid) <= 64 or sid in seen:
            raise OpsError(f"{name(str(sid))}: ids must be unique strings of 1-64 characters")
        seen.add(sid)
        who = name(sid)
        source = row.get("source")
        if not isinstance(source, str) or source not in durations:
            raise OpsError(f"{who}: its source is not part of this project")
        start, end = row.get("start"), row.get("end")
        if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (start, end)):
            raise OpsError(f"{who}: start and end must be numbers")
        if start < 0 or end - start < MIN_EXPORT_S:
            raise OpsError(
                f"{who}: needs 0 <= start and end at least {MIN_EXPORT_S}s after start "
                f"(got {start:g}-{end:g})"
            )
        duration = durations[source]
        if duration is None:
            raise OpsError(f"{who}: its source cannot be read")
        if end > duration + 0.01:
            raise OpsError(f"{who}: end {end:g}s is past the source's {duration:g}s")
        if not isinstance(label, str) or len(label) > LABEL_MAX:
            raise OpsError(f"{who}: label must be text of at most {LABEL_MAX} characters")
        out.append(
            {
                "id": sid,
                "source": source,
                "start": round(float(start), 3),
                "end": round(float(end), 3),
                "label": "".join(c for c in label if c.isprintable()).strip() or "segment",
            }
        )
    return out


def _in_order(ids: list[str], old: dict[str, tuple[int, Any]]) -> set[str]:
    """The most ``ids`` (kept rows, in their new order) that are still in their old order: the
    longest increasing run of old positions. The others count as moved."""
    tails: list[int] = []  # tails[k]: index in ids ending the best run of length k + 1
    prev: list[int] = [-1] * len(ids)
    for i, sid in enumerate(ids):
        pos = old[sid][0]
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if old[ids[tails[mid]]][0] < pos:
                lo = mid + 1
            else:
                hi = mid
        prev[i] = tails[lo - 1] if lo else -1
        if lo == len(tails):
            tails.append(i)
        else:
            tails[lo] = i
    out: set[str] = set()
    i = tails[-1] if tails else -1
    while i >= 0:
        out.add(ids[i])
        i = prev[i]
    return out


CHANGE_ORDER = ("added", "removed", "moved", "lengthened", "shortened", "trimmed", "relabelled")


def change_counts(rows: list[dict[str, Any]], color: dict[str, Any] | None = None) -> str:
    """``"2 added, 1 lengthened, color auto"`` for a :func:`diff` and its ``color`` op: what a
    proposal really changes (``"no change"`` when nothing)."""
    n: dict[str, int] = {}
    for row in rows:
        for c in row.get("changes", []):
            n[c] = n.get(c, 0) + 1
    parts = [f"{n[c]} {c}" for c in CHANGE_ORDER if n.get(c)]
    if color:
        parts.append(f"color {color.get('mode')}")
    return ", ".join(parts) or "no change"


def summary_gap(
    summary: str, rows: list[dict[str, Any]], color: dict[str, Any] | None = None
) -> str | None:
    """A sentence for the model when ``summary`` claims a change (extended, shortened, trimmed,
    added, removed, moved, color) that the :func:`diff` ``rows`` and ``color`` op do not make;
    None when every claim holds."""
    made = {c for row in rows for c in row.get("changes", [])}
    false = [
        name
        for name, words, needs in _CLAIM_WORDS
        if words.search(summary) and not made & set(needs)
    ]
    if _COLOR_WORDS.search(summary) and not color:
        false.append("color")
    if not false:
        return None
    return (
        f"The summary says it {', '.join(false)}, but the ops really make: "
        f"{change_counts(rows, color)}. Make the ops do what the summary says, or fix the "
        "summary, and propose again."
    )


def diff(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per row, in ``after``'s order then the removed rows: ``{id, label, changes, before,
    after}``. ``changes`` is ``["added"]``, ``["removed"]`` or any of ``moved`` (out of its old
    order among the kept rows, counting as few rows as possible), ``lengthened`` and
    ``shortened`` (the clip got longer or shorter), ``trimmed`` (its edges moved, its length did
    not) and ``relabelled``; ``before``/``after`` are ``{pos, start, end, label}`` or None."""
    old = {s["id"]: (i, s) for i, s in enumerate(before, 1)}
    new_ids = {s["id"] for s in after}
    stayed = _in_order([s["id"] for s in after if s["id"] in old], old)

    def place(pos: int, s: dict[str, Any]) -> dict[str, Any]:
        return {"pos": pos, "start": s["start"], "end": s["end"], "label": s.get("label", "")}

    rows: list[dict[str, Any]] = []
    for i, s in enumerate(after, 1):
        was = old.get(s["id"])
        if was is None:
            changes = ["added"]
        else:
            changes = []
            if s["id"] not in stayed:
                changes.append("moved")
            if (was[1]["start"], was[1]["end"]) != (s["start"], s["end"]):
                grew = (s["end"] - s["start"]) - (was[1]["end"] - was[1]["start"])
                changes.append(
                    "lengthened"
                    if grew > LENGTH_EPS
                    else "shortened"
                    if grew < -LENGTH_EPS
                    else "trimmed"
                )
            if was[1].get("label", "") != s.get("label", ""):
                changes.append("relabelled")
        rows.append(
            {
                "id": s["id"],
                "label": s.get("label", ""),
                "changes": changes,
                "before": place(*was) if was else None,
                "after": place(i, s),
            }
        )
    for i, s in enumerate(before, 1):
        if s["id"] not in new_ids:
            rows.append(
                {
                    "id": s["id"],
                    "label": s.get("label", ""),
                    "changes": ["removed"],
                    "before": place(i, s),
                    "after": None,
                }
            )
    return rows
