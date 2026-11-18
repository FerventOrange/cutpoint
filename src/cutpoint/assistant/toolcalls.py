"""The tools of a chat turn: their definitions (:data:`TOOLS`, :data:`JSON_SCHEMA`) and the
read tools a model calls, run by :class:`ToolCalls`, the tool part of
:class:`cutpoint.assistant.session.Turn` (``propose_edit`` is in
:mod:`cutpoint.assistant.proposing`).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from cutpoint.assistant import context, errtext, lookfor, ranking, watch, words
from cutpoint.assistant.proposing import MAX_SUMMARY, _num
from cutpoint.assistant.providers import ToolCall

logger = logging.getLogger(__name__)

MAX_PAD_S = 60.0  # get_transcript's pad
EARLIER_HITS = 3  # hits of each earlier find_words search the model is reminded of


def _tool(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


_ID = {"type": "string", "description": "short moment id such as m3"}
_OPS = {
    "type": "array",
    "items": {"type": "object"},
    "description": "edits in order: add, add_range, remove, move, trim, order, sort or color ops",
}
TOOLS = [
    _tool(
        "find_moments",
        "Moments whose label, transcript, seen text, heard sounds or cue names (5+ of 10, "
        "such as beauty or crash) have the query's words, best match first.",
        {"query": {"type": "string"}, "limit": {"type": "integer"}},
        ["query"],
    ),
    lookfor.TOOL,
    words.TOOL,
    _tool("get_moment", "One moment with its whole transcript snippet.", {"id": _ID}, ["id"]),
    _tool(
        "get_transcript",
        "Transcript lines around a moment, pad seconds before and after (default 5).",
        {"id": _ID, "pad": {"type": "number"}},
        ["id"],
    ),
    _tool(
        "show_moments",
        "Highlight moments in the user's list, for example the answer to a search.",
        {"ids": {"type": "array", "items": _ID}, "note": {"type": "string"}},
        ["ids"],
    ),
    _tool(
        "propose_edit",
        "Propose a change to the sequence; the user previews it and applies it or not. "
        "suggestions: up to 4 follow-up proposals offered as buttons, each {label, ops}.",
        {
            "summary": {"type": "string", "description": "one sentence for the user"},
            "ops": _OPS,
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}, "ops": _OPS},
                    "required": ["label", "ops"],
                },
            },
        },
        ["summary", "ops"],
    ),
]
TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
FINAL_TOOL = "propose_edit"  # the one tool the closing round offers
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "enum": [*TOOL_NAMES, "reply"]},
        "arguments": {"type": "object"},
        "text": {"type": "string"},
    },
    "required": ["tool"],
}
_JSON_TOOLS = [
    "find_moments {query, limit?}",
    lookfor.JSON_TOOL,
    words.JSON_TOOL,
    "get_moment {id}",
    "get_transcript {id, pad?}",
    "show_moments {ids, note?}",
    "propose_edit {summary, ops, suggestions?: [{label, ops}]}",
]


class ToolCalls:
    """The tool calls of :class:`cutpoint.assistant.session.Turn` (it works on the turn's
    state): :meth:`_call` runs the ``_tool_<name>`` method of a call."""

    # -- tools -------------------------------------------------------------------------
    def _call(self, call: ToolCall) -> tuple[str, bool]:
        """``(result for the model, whether the turn is over)``."""
        if call.error:
            return f"error: {call.name}: {call.error}", False
        names = [*self._names(), *self._extra()]
        handler = getattr(self, f"_tool_{call.name}", None) if call.name in names else None
        if handler is None:
            return f"error: unknown tool {call.name!r}; the tools are {', '.join(names)}", False
        return handler(call.arguments)

    def _moment(self, short: Any) -> tuple[str | None, str]:
        real = self.table.real(short)
        if real is None:
            return None, f"error: unknown moment id {short!r}"
        return real, ""

    def _tool_find_moments(self, args: dict[str, Any]) -> tuple[str, bool]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return "error: query must be text", False
        limit = args.get("limit")
        if isinstance(limit, bool) or not isinstance(limit, int):
            limit = 10
        found = self.table.find(query, max(1, min(limit, context.MAX_FOUND)))
        self._remember(found)
        self._emit(
            {"kind": "tool", "name": "find_moments", "query": query[:200], "found": len(found)}
        )
        if not found:
            return "no moment's label, transcript, seen text or cues have those words", False
        rows = "\n".join(json.dumps(self.table.row(r), ensure_ascii=False) for r in found)
        return context.data_block("moments", rows), False

    def _tool_look_for(self, args: dict[str, Any]) -> tuple[str, bool]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return "error: query must be text", False
        query = query.strip()[: lookfor.MAX_QUERY]
        limit = args.get("limit")
        if isinstance(limit, bool) or not isinstance(limit, int):
            limit = lookfor.DEFAULT_LIMIT
        limit = max(1, min(limit, lookfor.MAX_LIMIT))
        p = self.project
        # a guess is not a moment yet: a second look flags it again
        covering = [c for c in self.table.candidates.values() if c["id"] not in self._unsure]
        covering += p.segments
        try:
            got = self._finder.search(query, limit, covering, p.durations)
        except lookfor.LookForError as err:
            # the lite edition's install hint can name a local source folder (#1017)
            logger.info("assistant: look_for unavailable: %s", err)
            why = errtext.no_paths(str(err))
            self._emit(
                {
                    "kind": "tool",
                    "name": lookfor.NAME,
                    "query": query[:200],
                    "error": why,
                    **({"reason": err.reason} if err.reason else {}),
                }
            )
            return f"error: look_for is not available: {why}", False
        rows = []
        possible = []
        group: list[str] = []
        for hit, covered in got.hits:
            cover = covered
            if cover is None:
                cand = lookfor.new_moment(hit, query)
                if cand["id"] not in self.table.candidates:
                    self.found.append(cand)
                cover = cand["id"]
                # a clear one is a new moment; the rest are guesses, kept out of the ranked
                # moments (find_moments, the table) and out of proposals until viewed
                self.table.add(cand, rank=hit.clear)
                if hit.clear:
                    self._unsure.discard(cover)
                else:
                    self._unsure.add(cover)
            group.append(cover)
            row = {**self.table.row(cover), "match": hit.match, "confidence": hit.confidence}
            if covered is None:
                row["new" if hit.clear else "unconfirmed"] = True
                if not hit.clear:
                    possible.append(
                        {
                            "id": cover,
                            "source": self.table.candidates[cover]["source"],
                            "start": self.table.candidates[cover]["start"],
                            "end": self.table.candidates[cover]["end"],
                            "label": self.table.candidates[cover]["label"],
                            "confidence": hit.confidence,
                        }
                    )
            rows.append(row)
        self._remember(group)
        self._emit(
            {
                "kind": "tool",
                "name": lookfor.NAME,
                "query": query[:200],
                "found": len(rows),
                "new": sum(1 for r in rows if r.get("new")),
                "weak": sum(1 for r in rows if r["confidence"] == "weak"),
                "unconfirmed": len(possible),
                "possible": possible,  # the page offers each for the user to add
            }
        )
        bare = got.total - got.searched - got.other
        parts = [f"{bare} of {got.total} sources have no cached frames"] if bare else []
        if got.other:
            parts.append(
                f"{got.other} of {got.total} have a frame index made with another CLIP model"
            )
        searched = "neither was searched" if len(parts) == 2 else "they were not searched"
        note = f" ({' and '.join(parts)}; {searched})" if parts else ""
        if not rows:
            return (
                f"no window matches closely (the best frame is {got.best}/10){note}; "
                "the footage probably does not show it, or find_moments may know it by words"
            ), False
        if any(r["confidence"] == "weak" or r.get("unconfirmed") for r in rows):
            note += (
                " (weak and unconfirmed windows are guesses: confirm them or call them uncertain)"
            )
        return context.data_block(
            "look_for", chr(10).join(json.dumps(r, ensure_ascii=False) for r in rows)
        ) + note, False

    def _tool_find_words(self, args: dict[str, Any]) -> tuple[str, bool]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return "error: query must be the word or phrase to find", False
        query = " ".join(query.split())[:200]
        limit = args.get("limit")
        if isinstance(limit, bool) or not isinstance(limit, int):
            limit = words.DEFAULT_LIMIT
        limit = max(1, min(limit, words.MAX_LIMIT))
        p = self.project
        got = words.find(list(p.durations), p.words or _no_words, query, limit)
        apart, splice = ("", []) if got.hits or not got.with_words else self._apart(query)
        # the places given to the model go into chat.json too: a later message remembers them
        # (_hits_of)
        hits = [{k: h[k] for k in ("source", "start", "end")} for h in got.hits or splice]
        self._remember_hits(hits)
        self._emit(
            {"kind": "tool", "name": words.NAME, "query": query, "found": got.total, "hits": hits}
        )
        if not got.with_words:
            return (
                "no source of this project has transcript words (analyze transcribes them); "
                "what is said cannot be searched"
            ), False
        note = ""
        if got.with_words < got.sources:
            note = f" ({got.sources - got.with_words} of {got.sources} sources have no words)"
        self._found_words = self._found_words or bool(hits)
        if not got.hits:
            where = "in one place" if apart else "in the transcripts"
            return f"nobody says {query!r} {where}{note}{apart}", False
        rows = [
            {
                "file": self.table.file(h["source"]),
                "start": h["start"],
                "end": h["end"],
                "said": h["said"],
            }
            for h in got.hits
        ]
        if got.total > len(rows):
            note = f" ({len(rows)} of {got.total} shown; limit up to {words.MAX_LIMIT}){note}"
        body = chr(10).join(json.dumps(r, ensure_ascii=False) for r in rows)
        # a small model takes hits that are no moment for something it cannot add, and refuses
        # to join words said apart as "not said together"
        use = (
            "\nAdd a hit with an add_range op (its file, start and end); no moment id needed. "
            "Hits of different words, added in order, make a person say what they never said "
            "in one go: that is a splice, so do it when asked. To cut a moment right after a "
            "word, use the hit's end; right before it, the hit's start."
        )
        return context.data_block("words", body) + note + use, False

    def _remember_hits(self, hits: list[dict[str, Any]]) -> None:
        """Note both edges of each find_words hit (``source``, ``start``, ``end``)."""
        for h in hits:
            self._word_edges += [(h["source"], h["start"]), (h["source"], h["end"])]

    def _hits_of(self, record: Mapping[str, Any]) -> str:
        """The find_words hits of an earlier turn's ``record`` (its events, as chat.json keeps
        them), remembered for :meth:`_at_words` and as a line the model reads after that
        turn's reply, so a later message can cut at them without searching again."""
        events = record.get("events")
        said = []
        for e in events if isinstance(events, list) else []:
            if not isinstance(e, Mapping) or e.get("name") != words.NAME:
                continue
            hits = [
                h
                for h in e.get("hits") or []
                if isinstance(h, Mapping)
                and h.get("source") in self.project.durations
                and _num(h.get("start")) is not None
                and _num(h.get("end")) is not None
            ]
            self._remember_hits(hits)
            places = [
                f"{self.table.file(h['source'])} {h['start']:g}-{h['end']:g}"
                for h in hits[:EARLIER_HITS]
            ]
            if places:
                said.append(f"{str(e.get('query', ''))[:40]!r}: {', '.join(places)}")
        return f"\n(find_words hits: {'; '.join(said)})" if said else ""

    def _apart(self, query: str) -> tuple[str, list[dict[str, Any]]]:
        """For a phrase nobody says in one place: how often each of its words is said, and when
        each is said somewhere, the add_range ops of a splice (the first hit of each word, in
        order) so the model proposes them instead of giving up. Returns the text and those hits."""
        tokens = [w for w in query.split() if words.norm(w)]
        if len(set(tokens)) < 2:
            return "", []
        p = self.project
        found = {
            t: words.find(list(p.durations), p.words or _no_words, t, 1)
            for t in dict.fromkeys(tokens)
        }
        said = ", ".join(f"{t!r} {f.total} times" for t, f in found.items())
        if any(not f.hits for f in found.values()):
            return f". Its words alone: {said}.", []
        hits = [found[t].hits[0] for t in tokens]  # a word said twice: its hit twice
        ops = [
            {"op": "add_range", "file": self.table.file(h["source"])}
            | {k: h[k] for k in ("start", "end")}
            for h in hits
        ]
        # a refusal gets these ops in SPLICE_NUDGE too
        self._splice = json.dumps(ops, ensure_ascii=False)
        return (
            f". Its words are said apart: {said}. To make someone say it, splice them: "
            "propose_edit with these ops, in order:\n" + context.data_block("splice", self._splice)
        ), hits

    def _tool_get_moment(self, args: dict[str, Any]) -> tuple[str, bool]:
        real, problem = self._moment(args.get("id"))
        if real is None:
            return problem, False
        self._emit({"kind": "tool", "name": "get_moment", "id": real})
        row = json.dumps(self.table.row(real, full=True), ensure_ascii=False)
        return context.data_block("moment", row), False

    def _tool_get_transcript(self, args: dict[str, Any]) -> tuple[str, bool]:
        real, problem = self._moment(args.get("id"))
        if real is None:
            return problem, False
        pad = args.get("pad")
        if isinstance(pad, bool) or not isinstance(pad, (int, float)):
            pad = 5.0
        pad = max(0.0, min(float(pad), MAX_PAD_S))
        item = next((s for s in self.project.segments if s["id"] == real), None)
        item = item or self.table.candidates[real]
        self._emit({"kind": "tool", "name": "get_transcript", "id": real})
        start, end = float(item["start"]) - pad, float(item["end"]) + pad
        shown = self.table.file(item["source"])
        lines = context.transcript_lines(
            self.project.transcript, item["source"], start, end, shown=shown
        )
        if not lines:
            return "no transcript lines in that range", False
        return context.data_block("transcript", "\n".join(lines)), False

    def _tool_show_moments(self, args: dict[str, Any]) -> tuple[str, bool]:
        ids = args.get("ids")
        if not isinstance(ids, list) or not ids:
            return "error: ids must be a list of moment ids", False
        real = [self.table.real(x) for x in ids]
        unknown = [str(x) for x, r in zip(ids, real, strict=True) if r is None]
        if unknown:
            return f"error: unknown moment ids {', '.join(unknown)}", False
        note = args.get("note") if isinstance(args.get("note"), str) else ""
        self._emit({"kind": "show", "ids": list(dict.fromkeys(real)), "note": note[:MAX_SUMMARY]})
        return f"{len(set(real))} moments shown to the user", False

    def _tool_view_moment(self, args: dict[str, Any]) -> tuple[str, bool]:
        real, problem = self._moment(args.get("id"))
        if real is None or self.viewer is None:
            return problem or "error: view_moment is not available", False
        short = self.table.name(real)
        if real in self._viewed:
            return f"{short} was already viewed in this turn: its sheet is above", False
        cap = watch.view_cap(self.provider)
        if len(self._viewed) >= cap:
            self._emit({"kind": "tool", "name": watch.NAME, "id": real, "error": "limit"})
            self._spent = True  # the next round is the closing one
            return (
                f"error: at most {watch.views(cap)} can be viewed per message and the limit is "
                "reached; answer the user now with what you have seen",
                False,
            )
        item = next((s for s in self.project.segments if s["id"] == real), None)
        item = item or self.table.candidates[real]
        view = self.viewer.view(short, real, item, self.table.row(real), self.project.transcript)
        self._emit(view.event)
        if view.sheet is not None:
            self._viewed.add(real)
            text = f"{watch.NAME} {short}: timeline sheet"
            self._sheets.append(
                watch.image_message(text, view.sheet, watch.sheet_tokens(self.provider))
            )
        return view.text, False

    def _remember(self, ids: list[str]) -> None:
        """Note the moments one search found, for :func:`topup.top_up`."""
        if ids:
            self._searches.append(list(ids))

    def _tool_preview_ranking(self, args: dict[str, Any]) -> tuple[str, bool]:
        """Show the page's preview card of the moments ranked in ``style`` (the user applies it
        or not) and end the turn, as a proposal does. A style not offered, the current one, a
        busy project or a refused preview go back to the model."""
        hook = self.project.ranking
        style = args.get("style")
        others = [n for n, _ in self._styles if n != self._current]
        if hook is None or not isinstance(style, str) or style not in others:
            now = f" (the moments are ranked as {self._current} now)" if self._current else ""
            return f"error: style must be one of {', '.join(others)}{now}", False
        reason = args.get("reason")
        reason = " ".join(reason.split())[: ranking.MAX_REASON] if isinstance(reason, str) else ""
        try:
            payload = hook.preview(style)
        except ranking.RankingError as err:
            logger.info("assistant: ranking preview refused: %s", err)
            why = errtext.no_paths(str(err))  # the model and chat.json get no file path
            self._emit(
                {
                    "kind": "tool",
                    "name": ranking.NAME,
                    "style": style,
                    "error": "the project is busy" if err.busy else why,
                }
            )
            if err.busy:
                return (
                    "error: the project is busy (an analysis or another ranking runs): tell the "
                    "user to ask again when it is done",
                    False,
                )
            return f"error: the ranking could not be previewed: {why}. Tell the user", False
        except Exception:  # a text with a file path must not reach the model or the reply
            logger.exception("assistant: ranking preview failed")
            self._emit(
                {
                    "kind": "tool",
                    "name": ranking.NAME,
                    "style": style,
                    "error": "an error on this computer",
                }
            )
            return (
                "error: the ranking could not be previewed (an error on this computer). "
                "Tell the user",
                False,
            )
        self._emit({"kind": "ranking", **payload, "reason": reason})
        return "", True


def _no_words(_source: str) -> None:
    return None
