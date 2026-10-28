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
  in seconds added to the current edge (negative: earlier).
* ``{"op": "order", "ids"}``: every sequence id once, in the new order.
* ``{"op": "sort", "by": "time" | "score" | "length", "desc"?}``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from cutpoint.clips import MIN_EXPORT_S

MAX_SEGMENTS = 500  # as cutpoint.ui.MAX_SEGMENTS (ui needs the optional fastapi)
MAX_OPS = 100  # ops in one proposal
LABEL_MAX = 200
OPS = ("add", "remove", "move", "trim", "order", "sort")
SORT_KEYS = ("time", "score", "length")


class OpsError(ValueError):
    """An op or the sequence it gives breaks a rule; the message is for the model."""


@dataclass
class Outcome:
    """What :func:`apply_ops` gives: ``after`` and its ``diff``, or the ``error``."""

    after: list[dict[str, Any]] | None = None
    diff: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


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
) -> Outcome:
    """Apply ``ops`` to a copy of the sequence ``before``.

    Args:
        before: The current sequence (never changed).
        ops: The model's ops (any JSON; checked here).
        candidates: The analysis moments (``moments_view`` ids), the only ones ``add`` takes.
        durations: The project's sources and their durations (:class:`Durations`).
        name: How an id is written in error messages (the short id the model knows).

    Returns:
        ``after`` (checked and cleaned as ``validate_segments`` cleans it) and a per-row
        :func:`diff`, or ``error`` naming the op or row and what is wrong.
    """
    try:
        after = _check(_apply(before, ops, candidates, durations, name), durations, name)
    except OpsError as err:
        return Outcome(error=str(err))
    return Outcome(after, diff(before, after))


def _number(op: dict[str, Any], key: str, where: str) -> float | None:
    v = op.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise OpsError(f"{where}: {key} must be a number of seconds")
    return float(v)


def _position(op: dict[str, Any], key: str, top: int, where: str) -> int | None:
    v = op.get(key)
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
) -> list[dict[str, Any]]:
    if not isinstance(ops, list) or not ops:
        raise OpsError("ops must be a non-empty list of edits")
    if len(ops) > MAX_OPS:
        raise OpsError(f"at most {MAX_OPS} ops in one proposal")
    rows = [dict(s) for s in before]
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
        if kind == "add":
            cand = cands.get(sid) if isinstance(sid, str) else None
            if cand is None:
                raise OpsError(f"{where}: not a moment of this project")
            if any(row["id"] == sid for row in rows):
                raise OpsError(f"{where}: already in the sequence; use move or trim")
            at = _position(op, "at", len(rows) + 1, where) or len(rows) + 1
            start, end = _number(op, "start", where), _number(op, "end", where)
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
            to = _position(op, "to", len(rows), where)
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
                    row[edge] = absolute if absolute is not None else row[edge] + delta
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


def diff(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per row, in ``after``'s order then the removed rows: ``{id, label, changes, before,
    after}``. ``changes`` is ``["added"]``, ``["removed"]`` or any of ``moved`` (out of its old
    order among the kept rows, counting as few rows as possible), ``trimmed`` and
    ``relabelled``; ``before``/``after`` are ``{pos, start, end, label}`` or None."""
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
                changes.append("trimmed")
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
