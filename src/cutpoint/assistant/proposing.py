"""Proposals of a chat turn: the ops checked with :func:`cutpoint.assistant.ops.apply_ops`,
sent back for their length or summary, topped up (:mod:`cutpoint.assistant.topup`) and paced
(:mod:`cutpoint.assistant.pacing`) by code, by :class:`Proposing`, the proposal part of
:class:`cutpoint.assistant.session.Turn`.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from cutpoint.assistant import context, intent, pacing
from cutpoint.assistant.edges import length
from cutpoint.assistant.ops import apply_ops, change_counts, rows_after, summary_gap
from cutpoint.assistant.topup import kind_groups, top_up

logger = logging.getLogger(__name__)

MAX_SUMMARY = 500
MAX_SUGGESTIONS = 4
WORD_EDGE_S = 0.1  # an add or trim edge this close to a find_words hit edge is moved onto it, exact
TOLERANCE = 0.15  # a proposal this far over or under the asked length is sent back
MAX_LENGTH_BACKS = 3  # ... up to this many times a turn, then the closest one is offered


class Proposing:
    """The proposals of :class:`cutpoint.assistant.session.Turn` (it works on the turn's
    state): ``propose_edit`` and the proposals code offers."""

    def _checked(self, ops: Any) -> tuple[dict[str, Any] | None, str]:
        """``ops`` with real ids, applied to the turn's sequence: ``({ops, after, diff}, "")``
        or ``(None, why)``."""
        mapped, problem = _real_ops(ops, self.table)
        if mapped is None:
            return None, problem
        return self._checked_real(mapped)

    def _checked_real(self, mapped: Any) -> tuple[dict[str, Any] | None, str]:
        """:meth:`_checked` for ops that already have real ids."""
        # a guess of look_for goes into the sequence only once view_moment showed it; the user
        # can add it themself from the chat
        blocked = self._unsure - self._viewed
        for n, op in enumerate(mapped if isinstance(mapped, list) else [], 1):
            if isinstance(op, dict) and op.get("op") == "add" and op.get("id") in blocked:
                return None, (
                    f"op {n}: {self.table.name(op['id'])} is an unconfirmed look_for window; "
                    "confirm it with view_moment first, or tell the user it is uncertain "
                    "(they can add it from the chat)"
                )
        p = self.project
        known = [c for c in self.found if c["id"] not in blocked]
        mapped = self._at_words(mapped, [*p.candidates, *known])
        out = apply_ops(
            p.segments,
            mapped,
            [*p.candidates, *known],
            p.durations,
            name=self.table.name,
            snap=p.edges.phrase_end if p.edges is not None else None,
        )
        if out.error:
            return None, out.error
        if out.color is None and not any(row["changes"] for row in out.diff):
            return None, "these ops leave the sequence as it is"
        if p.segments and not out.after and not intent.CLEARS.search(self.text):
            return None, (
                "these ops remove every clip, but the user did not ask to clear or start over; "
                "keep the clips that stay and change only what was asked"
            )
        removed = sum("removed" in row["changes"] for row in out.diff)
        planned = p.edges.plan(out.after or []) if p.edges is not None else None
        have = length(p.segments, p.edges.plan(p.segments) if p.edges is not None else None)
        wants_less = bool(self.target) and self.target < have * (1 - TOLERANCE)  # asks for removal
        if (
            p.segments
            and 2 * removed >= len(p.segments)
            and not (intent.wants_removal(self.text) or wants_less)
        ):
            return None, (
                f"you removed {removed} of {len(p.segments)} clips; the user didn't ask to "
                "remove any. Keep the clips that are there and add or change only what was asked"
            )
        checked: dict[str, Any] = {
            "ops": mapped,
            "after": out.after,
            "diff": out.diff,
            "length": round(length(out.after or [], planned), 1),  # seconds as exported
        }
        if out.color is not None:
            checked["color"] = out.color
        if out.notes:
            checked["notes"] = out.notes
        used = {row["id"] for row in out.after or []}
        if moments := [c for c in known if c["id"] in used]:
            checked["moments"] = moments  # look_for finds: the page's analysis does not have them
        return checked, ""

    def _at_words(self, ops: Any, moments: list[dict[str, Any]]) -> Any:
        """``ops`` with each add or trim edge (in seconds or as a delta) within
        :data:`WORD_EDGE_S` of a find_words hit edge (this message's or an earlier one's) moved
        onto the nearest one, and ``"exact": true``: the cut the user asked for at a word stays
        there at export, as an add_range row's. An op with ``"exact": false`` is left as it is.
        The proposal keeps the flag, so a rebase on save marks the row again.

        A delta counts from the row's edges as :func:`rows_after` leaves them after the ops
        before it; when those ops are wrong, the delta is left as given."""
        if not self._word_edges or not isinstance(ops, list):
            return ops
        p = self.project
        snap = p.edges.phrase_end if p.edges is not None else None
        own = {m["id"]: m for m in moments}
        out: list[Any] = []
        for op in ops:
            kind = op.get("op") if isinstance(op, dict) else None
            sid = op.get("id") if isinstance(op, dict) else None
            if not isinstance(sid, str) or kind not in ("add", "trim"):
                out.append(op)
                continue
            row: dict[str, Any] = dict(own.get(sid) or {})  # an add: the moment's own range
            if kind == "trim":
                rows = rows_after(p.segments, out, moments, p.durations, snap=snap)
                row = next((r for r in rows or [] if r.get("id") == sid), {})
            source = row.get("source") or (own.get(sid) or {}).get("source")
            hits = [w for src, w in self._word_edges if src == source]
            for edge in ("start", "end"):
                t = _num(op.get(edge))
                delta = _num(op.get(f"{edge}_delta")) if kind == "trim" else None
                if t is None and delta is not None and _num(row.get(edge)) is not None:
                    t = row[edge] + delta
                if t is None:
                    continue
                near = min(hits, key=lambda w: abs(w - t), default=None)
                close = near is not None and abs(near - t) <= WORD_EDGE_S + 1e-6
                if close and op.get("exact") is not False:
                    # on the word (a small model rounds 3627.04 to 3627): the hit's own time
                    op = {k: v for k, v in op.items() if k != f"{edge}_delta"}
                    op |= {edge: near, "exact": True}
            out.append(op)
        return out

    def _tool_propose_edit(self, args: dict[str, Any]) -> tuple[str, bool]:
        summary = args.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return "error: summary must be one sentence for the user", False
        main, problem = self._checked(args.get("ops"))
        if main is None:
            return self._rejected(
                problem, f"error: proposal rejected: {problem}. Fix the ops and propose again."
            )
        gap = summary_gap(summary, main["diff"], main.get("color"))
        if wrong := self._length_gap(summary, main, lambda: self._other_lengths(main)):
            gap = f"{gap} {wrong}" if gap else wrong
        if note := self._send_back(summary, main, gap):
            if main.get("notes"):  # the model learns that its edges were not used
                note = f"{note} {' '.join(main['notes'])}"
            return self._rejected(note, note)
        if gap:  # a summary that claims what the ops do not do is replaced by what they do
            summary = self._true_summary(main)
        raw = args.get("suggestions")
        if self.target and self._candidates:  # a sent-back one may be closer than this retry
            was = self._closest()
            if was is not None and self._rank(was[1]) < self._rank(main):
                summary, main = was
                raw = []  # its suggestions were not kept
        if self.target and self._rank(main)[0]:  # never shown over the length by more than 15%
            over = main["length"] - self.target
            note = (
                f"This proposal makes {main['length']:.0f} s of the {self.target:.0f} s asked; "
                f"cut about {over:.0f} s (shorten or remove clips) and propose again."
            )
            return self._rejected(note, note)
        summary, edges = self._reserve(summary, main)
        summary, main = self._paced(*self._filled(summary, main))
        summary = self._noted(summary, main, edges)
        suggestions = []
        for sug in (raw if isinstance(raw, list) else [])[:MAX_SUGGESTIONS]:
            label = sug.get("label") if isinstance(sug, dict) else None
            checked = self._checked(sug.get("ops"))[0] if isinstance(label, str) else None
            if checked is None or not label.strip():
                logger.debug("assistant: dropped a suggestion that does not apply: %r", sug)
                continue
            checked.pop("notes", None)
            suggestions.append(self._numbered("s", label.strip()[:80], checked))
        event = {"kind": "proposal", **self._numbered("p", summary.strip(), main)}
        self._emit({**event, "suggestions": suggestions})
        return "", True

    @staticmethod
    def _reserve(summary: str, main: dict[str, Any]) -> tuple[str, str]:
        """``(summary, note)``: the note on the edges of an add that were ignored (at most 150
        characters), and the summary cut to leave room for it and for the fill's own notes."""
        note = " ".join(main.get("notes") or [])
        if not note:
            return summary, ""
        if len(note) > 150:
            note = note[:149] + "…"
        return summary.strip()[: max(MAX_SUMMARY - 200 - len(note) - 1, 100)], note

    @staticmethod
    def _noted(summary: str, main: dict[str, Any], note: str) -> str:
        """``summary`` with the edge ``note`` after it; the proposal ``main`` loses its notes."""
        main.pop("notes", None)
        return f"{summary.strip()} {note}" if note else summary

    def _rejected(self, reason: str, result: str) -> tuple[str, bool]:
        """Tell the model ``result`` (the turn goes on) and the page ``reason`` (a muted line of
        the chat); logged, so a run that went wrong can be read back."""
        logger.info("assistant: proposal not accepted: %s", reason)
        self._emit({"kind": "rejected", "text": reason[:MAX_SUMMARY]})
        return result, False

    @staticmethod
    def _length_gap(
        summary: str,
        main: dict[str, Any],
        others: Callable[[], Iterable[float]] | None = None,
    ) -> str | None:
        """A sentence for the model when ``summary`` gives the sequence a length ("now about 3
        minutes", read by :func:`intent.claimed_seconds`) more than the tolerance off the
        proposal's exported length; None when it gives none or the right one. ``others`` gives
        the other lengths a summary may name (the change in length, each clip's): a claim that
        fits one of them is taken for it, not flagged. A summary that names the exported length
        anywhere ("now 34 s; reaching 60 s total needs more footage") is not flagged either."""
        claim = intent.claimed_seconds(summary)
        have = main["length"]

        def fits(seconds: float, said: float | None = claim) -> bool:
            return abs(said - seconds) <= max(said * TOLERANCE, 0.5)  # type: ignore[operator]

        if claim is None or fits(have) or (others is not None and any(map(fits, others()))):
            return None
        if any(fits(have, said) for said in intent.named_seconds(summary)):
            return None
        return (
            f"The summary says the sequence is {claim:.0f} s, but this proposal makes {have:.0f} s "
            "as exported. Fix the summary or the ops and propose again."
        )

    def _other_lengths(self, main: dict[str, Any]) -> list[float]:
        """The change in length that ``main`` makes and each of its clips' exported length."""
        p = self.project
        after = main["after"] or []
        was = length(p.segments, p.edges.plan(p.segments) if p.edges is not None else None)
        planned = p.edges.plan(after) if p.edges is not None else None
        return [abs(main["length"] - was), *(length([r], planned) for r in after)]

    @staticmethod
    def _true_summary(main: dict[str, Any]) -> str:
        """A summary made from the diff, for a proposal whose own summary says too much."""
        counts = change_counts(main["diff"], main.get("color"))
        return f"Changes: {counts}; {main['length']:.0f} s in all."

    def _kind(self, item: Mapping[str, Any]) -> str:
        """What a moment is, for a fill that has no search to go by: its ``kind``, else its
        strongest cue, else its label."""
        real = item["id"]
        cand = self.table.candidates.get(real) or item
        if isinstance(cand.get("kind"), str) and cand["kind"]:
            return cand["kind"]
        look = self.table.looks.get(real)
        if look is not None and look.cues:
            return next(iter(look.cues))
        return str(cand.get("label") or "")

    def _offer_own(self) -> bool:
        """The turn ends with no proposal of the model's while the sequence is short of the
        length asked for: offer one that code builds from the current sequence (the fill), for
        the user to review. False when the message asks no edit, nothing is short or there is
        nothing to fill it with. An advice question (:func:`intent.asks_advice`) gets none."""
        p = self.project
        if not self.target or intent.asks_advice(self.text):
            return False
        planned = p.edges.plan(p.segments) if p.edges is not None else None
        have = length(p.segments, planned)
        if have >= self.target * (1 - TOLERANCE):
            return False
        base = {"ops": [], "after": [dict(s) for s in p.segments], "diff": [], "length": have}
        why = " (the assistant didn't propose an edit)"
        summary, main = self._filled("", base, why)
        if "auto" not in main:
            return False
        summary, main = self._paced(summary, main)
        event = {"kind": "proposal", **self._numbered("p", summary.strip(), main)}
        return self._emit({**event, "suggestions": []})

    def _filled(
        self, summary: str, main: dict[str, Any], why: str = ""
    ) -> tuple[str, dict[str, Any]]:
        """``(summary, proposal)`` topped up by code when it is still short of the asked length
        (:func:`topup.top_up`): real ops marked ``auto``, their rows and the summary say so."""
        p = self.project
        if not self.target or main["length"] >= self.target * (1 - TOLERANCE):
            return summary, main
        goal = f"~{int(self.target) // 60}:{int(self.target) % 60:02d}"
        found = top_up(
            main,
            target=self.target,
            tolerance=TOLERANCE,
            evaluate=lambda ops: self._checked_real(ops)[0],
            before=p.segments,
            candidates={c["id"]: c for c in [*p.candidates, *self.found]},
            searched=self._searches
            or kind_groups(
                main["after"] or [], {c["id"]: c for c in [*p.candidates, *self.found]}, self._kind
            ),
            durations=p.durations,
            phrase_end=p.edges.phrase_end if p.edges is not None else None,
        )
        base = summary.strip()[:300]
        if found is None:
            short = self.target - main["length"]
            return f"{base} (Still {short:.0f} s short of {goal}: nothing to fill it with.)", main
        done = found.checked
        auto = set(found.lengthened) | set(found.added)
        for row in done["diff"]:
            if row["id"] in auto:
                row["auto"] = True
        parts = []
        if found.lengthened:
            parts.append(f"lengthened {len(set(found.lengthened))} clips")
        if found.added:
            parts.append(f"added {len(found.added)} from your searches")
        head = (f"to reach {goal}" if found.reached else f"towards {goal}") + why
        done["auto"] = {
            "target": self.target,
            "lengthened": found.lengthened,
            "added": found.added,
            "reached": found.reached,
        }
        note = f" Auto-filled {head}: {', '.join(parts)}; now {done['length']:.0f} s."
        return base + note, done

    def _pace_kind(self, item: Mapping[str, Any]) -> str:
        """The pacing kind (:func:`pacing.moment_kind`) of a moment or sequence row."""
        real = item["id"]
        if real not in self._kinds:
            cand = self.table.candidates.get(real)
            look = self.table.looks.get(real)

            def spoken() -> bool:
                said = (cand or {}).get("transcript")
                if not isinstance(said, str) or len(said.split()) < pacing.SPOKEN_WORDS:
                    row = cand or item
                    lines = context.transcript_lines(
                        self.project.transcript,
                        row["source"],
                        row["start"],
                        row["end"],
                        shown=self.table.file(row["source"]),
                    )
                    said = " ".join(line.partition("] ")[2] for line in lines)
                return len(said.split()) >= pacing.SPOKEN_WORDS

            label = str(item.get("label") or "")
            self._kinds[real] = pacing.moment_kind(
                label, cand, look.cues if look is not None else {}, spoken
            )
        return self._kinds[real]

    def _paced(self, summary: str, main: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """``(summary, proposal)`` with the kinds spread between the action clips
        (:func:`pacing.pace`) when the message asks to break up or mix kinds: real ops marked
        ``auto``, their rows and the summary say so. Unchanged for any other message."""
        if not self.mix:
            return summary, main
        p = self.project
        pool = {c["id"]: c for c in [*p.candidates, *self.found]}
        found = pacing.pace(
            main,
            kind_of=self._pace_kind,
            evaluate=lambda ops: self._checked_real(ops)[0],
            candidates=pool,
            searched=self._searches,
            dialogue=self.dialogue,
            upper=self.target * (1 + TOLERANCE) if self.target else None,
            removable=(main.get("auto") or {}).get("added", []),
        )
        if found is None:
            return summary, main
        done = found.checked
        if "auto" in main:  # the fill's record stays with the proposal
            done["auto"] = main["auto"]
        marked = {r["id"]: r["auto"] for r in main["diff"] if r.get("auto")}
        for row in done["diff"]:
            if row["id"] in marked:
                row["auto"] = marked[row["id"]]
            elif row["id"] in (*found.added, *found.replaced) or (
                row["id"] in found.moved and "moved" in row["changes"]
            ):
                row["auto"] = "spread"  # a row the model left in place and code moved or swapped
        done["pace"] = {
            "moved": found.moved,
            "added": found.added,
            "replaced": found.replaced,
            "spread": found.spread,
        }
        parts = []
        if found.spread:
            names = [f"{n} {k}" for k, n in found.spread.items()]
            listed = ", ".join(names[:-1]) + (" and " if len(names) > 1 else "") + names[-1]
            parts.append(f"Spread {listed} clips between the action clips.")
        if found.added:
            n, r = len(found.added), len(found.replaced)
            more = f", replacing {r} action clip{'s' * (r != 1)}" if r else ""
            parts.append(f"Added {n} dialogue moment{'s' * (n != 1)} you asked for{more}.")
        text = summary.strip()
        if "auto" not in main:  # the fill capped it already; else leave room for the note
            text = text[:300]
        if found.added:  # the length changed: say the new one
            now = f"now {done['length']:.0f} s."
            text, n = re.subn(r"now \d+ s\.$", now, text)
            if not n:
                parts.append(now.capitalize())
        note = " ".join(parts)
        return f"{text[: MAX_SUMMARY - len(note) - 1]} {note}".strip(), done

    def _send_back(self, summary: str, main: dict[str, Any], gap: str | None = None) -> str | None:
        """Why a valid proposal goes back to the model, once a turn for each problem: it makes
        less than the length the user asked for, or its summary says a change the ops do not
        make. None: accept it."""
        notes: list[str] = []
        have = main["length"]
        by_length = False
        if self.target and self._length_backs < MAX_LENGTH_BACKS:
            need = self.target - have
            if abs(need) > self.target * TOLERANCE:
                self._length_backs += 1
                by_length = True
                fix = (
                    f"add about {need:.0f} s more (extend clips within their room or add moments)"
                    if need > 0
                    else f"cut about {-need:.0f} s (shorten or remove clips)"
                )
                notes.append(
                    f"This proposal makes {have:.0f} s of the {self.target:.0f} s asked; {fix} "
                    "and propose again."
                )
        # a summary alone is no reason to send a proposal back: it gets a true one instead
        if by_length and gap and "summary" not in self._sent_back:
            self._sent_back.add("summary")
            notes.append(gap)
        if by_length:  # a candidate to offer later, with a summary that says what its ops do
            self._candidates.append((self._true_summary(main) if gap else summary, main))
        return " ".join(notes) or None

    def _rank(self, main: dict[str, Any]) -> tuple[bool, float]:
        """Sort key of a proposal against the asked length: one over it by more than the
        tolerance loses to any other, then the closest."""
        target = self.target or 0.0
        return main["length"] > target * (1 + TOLERANCE), abs(main["length"] - target)

    def _closest(self) -> tuple[str, dict[str, Any]] | None:
        """The sent-back proposal closest to the asked length, never one over it by more than
        the tolerance."""
        best = min(self._candidates, key=lambda c: self._rank(c[1]), default=None)
        return best if best is not None and not self._rank(best[1])[0] else None

    def _offer_fallback(self) -> bool:
        """Show the closest proposal that was sent back, when the turn would end without one.
        False when there is none."""
        found = self._closest()
        if found is None:
            return False
        summary, edges = self._reserve(*found)
        summary, main = self._paced(*self._filled(summary, found[1]))
        summary = self._noted(summary, main, edges)
        self._candidates = []
        event = {"kind": "proposal", **self._numbered("p", summary.strip(), main)}
        return self._emit({**event, "suggestions": []})

    def _numbered(self, kind: str, summary: str, checked: dict[str, Any]) -> dict[str, Any]:
        self._proposals += 1
        return {
            "id": f"{self.id}-{kind}{self._proposals}",
            "summary": summary[:MAX_SUMMARY],
            **checked,
            "base_key": self.project.base_key,
        }


def _real_ops(ops: Any, table: context.MomentTable) -> tuple[list[Any] | None, str]:
    """``ops`` with their short ids replaced by the real ones; ``(None, why)`` for an unknown
    id. Anything else is left for :func:`apply_ops` to judge."""
    if isinstance(ops, str):  # some models send the list as a JSON string
        text = ops
        try:
            ops = json.loads(ops)
        except json.JSONDecodeError as err:
            if err.msg != "Extra data":
                return (
                    None,
                    f"ops is a string that is not valid JSON: {err.msg} at position {err.pos}",
                )
            # a valid list followed by punctuation or a fence is the list
            ops, end = json.JSONDecoder().raw_decode(text.lstrip())
            if text.lstrip()[end:].strip(" \t\r\n.,;`"):
                return None, f"ops has text after the JSON list, at position {end}"
        except ValueError:
            return ops, ""
    if not isinstance(ops, list):
        return ops, ""
    out: list[Any] = []
    for n, op in enumerate(ops, 1):
        if not isinstance(op, dict):
            out.append(op)
            continue
        op = dict(op)
        if op.get("op") == "add_range":  # a file name, not a moment: no id to map
            op.pop("id", None)
            source, problem = _source_of(op.pop("file", op.get("source")), table, n)
            if source is None:
                return None, problem
            op["source"] = source
        if "id" in op:
            real = table.real(op["id"])
            if real is None:
                return None, f"op {n}: unknown moment id {op['id']!r}"
            op["id"] = real
        if isinstance(op.get("ids"), list):
            reals = [table.real(x) for x in op["ids"]]
            unknown = [str(x) for x, r in zip(op["ids"], reals, strict=True) if r is None]
            if unknown:
                return None, f"op {n}: unknown moment ids {', '.join(unknown)}"
            op["ids"] = reals
        out.append(op)
    return out, ""


def _source_of(name: Any, table: context.MomentTable, n: int) -> tuple[str | None, str]:
    """``(source path, "")`` of the file ``name`` an ``add_range`` op gives (a name as
    :meth:`context.MomentTable.file` shows it, a bare file name when it is unique, or the path
    itself), else ``(None, why)``."""
    paths = list(table.durations)
    if isinstance(name, str) and name in paths:
        return name, ""
    shown = {p: table.file(p) for p in paths}
    slashed = name.replace("\\", "/") if isinstance(name, str) else None
    given = slashed.strip("/") if slashed is not None else None
    # as shown first: a source at a root without a drive is shown as "/clip.mp4"
    hits = [p for p in paths if slashed is not None and slashed == shown[p]]
    hits = hits or [p for p in paths if given is not None and given == shown[p]]
    hits = hits or [p for p in paths if given is not None and given == context.file_name(p)]
    if len(hits) == 1:
        return hits[0], ""
    listed = ", ".join(shown.values())
    if hits:
        return None, f"op {n}: several sources are named {name!r}: use one of {listed}"
    return None, f"op {n}: add_range needs file, one of the project's files ({listed})"


def _num(v: Any) -> float | None:
    """``v`` as a finite number of seconds, else None."""
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    return float(v) if ok else None
