"""One chat turn, run in a thread: the model looks moments up and ends with a proposal or a reply.

The loop offers the read tools (``find_moments``, ``get_moment``, ``get_transcript``,
``show_moments``) and the final ``propose_edit``, for at most :data:`MAX_ROUNDS` model replies.
A proposal is checked with :func:`cutpoint.assistant.ops.apply_ops`; a rejected one goes back to
the model with the reason. A model without tool calls (or one that refuses them) answers in JSON
mode: one ``{"tool", "arguments"}`` object per reply, constrained by :data:`JSON_SCHEMA`.

Nothing here writes project files or runs commands: a turn only adds events (tool use, moments
to show, a reply, proposals) that the page polls, the way it polls jobs.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cutpoint.assistant import context
from cutpoint.assistant.ops import apply_ops
from cutpoint.assistant.providers import (
    Provider,
    ProviderError,
    ToolCall,
    ToolsUnsupportedError,
    describe,
)

logger = logging.getLogger(__name__)

MAX_ROUNDS = 8  # model replies in one turn
MAX_RESULT_CHARS = 6000  # of one tool result
MAX_TEXT = 2000  # characters of a user message
MAX_SUMMARY = 500
MAX_SUGGESTIONS = 4
MAX_KEPT = 50  # finished turns kept in memory (chat.json has them all)
MAX_PAD_S = 60.0  # get_transcript's pad


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
    "description": "edits in order: add, remove, move, trim, order or sort ops",
}
TOOLS = [
    _tool(
        "find_moments",
        "Moments whose label or transcript has the query's words, best match first.",
        {"query": {"type": "string"}, "limit": {"type": "integer"}},
        ["query"],
    ),
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
    "get_moment {id}",
    "get_transcript {id, pad?}",
    "show_moments {ids, note?}",
    "propose_edit {summary, ops, suggestions?: [{label, ops}]}",
]


class BusyError(RuntimeError):
    """A turn of this project is still running."""


@dataclass
class Project:
    """What a turn works on, read when it starts."""

    candidates: list[dict[str, Any]]  # the analysis moments, with moments_view ids
    segments: list[dict[str, Any]]  # the sequence the proposals change (checked)
    durations: Mapping[str, float | None]  # the project's sources (ops.Durations)
    transcript: str  # transcript.txt ("" when there is none)
    base_key: str  # ops.base_key of segments and highlights.json


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Turn:
    """One user message and the model's work on it; :meth:`view` is what the page polls."""

    def __init__(
        self,
        provider: Provider,
        project: Project,
        text: str,
        earlier: list[dict[str, Any]] | None = None,
    ) -> None:
        self.id = secrets.token_hex(6)
        self.provider = provider
        self.project = project
        self.text = text
        self.earlier = [
            {"user": str(t.get("user", "")), "reply": str(t.get("reply", ""))}
            for t in earlier or []
        ]
        self.table = context.MomentTable(project.candidates, project.segments)
        self.status = "running"  # running | done | failed | cancelled
        self.error: str | None = None
        self.events: list[dict[str, Any]] = []
        self.started = _now()
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.on_finish: Callable[[Turn], None] | None = None
        self._proposals = 0
        self._saved = False

    # -- what the page sees ------------------------------------------------------------
    def view(self, after: int = 0) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "status": self.status,
                "text": self.text,
                "started": self.started,
                "provider": describe(self.provider),
                "events": self.events[max(0, after) :],
                "next": len(self.events),
                "error": self.error,
            }

    def record(self) -> dict[str, Any]:
        """The finished turn as ``chat.json`` keeps it; ``reply`` is what later turns see."""
        with self.lock:
            reply = self.error or ""
            for e in self.events:
                if e["kind"] == "reply":
                    reply = e["text"]
                elif e["kind"] == "proposal":
                    reply = f"Proposed: {e['summary']}"
            return {
                "id": self.id,
                "user": self.text,
                "started": self.started,
                "status": self.status,
                "error": self.error,
                "events": list(self.events),
                "reply": reply,
            }

    def proposal(self, pid: str) -> dict[str, Any] | None:
        with self.lock:
            for e in self.events:
                if e["kind"] == "proposal":
                    for p in (e, *e["suggestions"]):
                        if p["id"] == pid:
                            return p
        return None

    # -- running -----------------------------------------------------------------------
    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name=f"assistant-{self.id}").start()

    def cancel(self) -> None:
        """Stop the turn; a reply still on its way is dropped (its request is closed at the
        next chunk). The turn is saved now, so it never lands after a newer turn."""
        with self.lock:
            if self.status != "running":
                return
            self.status = "cancelled"
        self._save()

    def _save(self) -> None:
        """Call ``on_finish`` once: when the turn is cancelled or its thread ends."""
        with self.lock:
            if self._saved:
                return
            self._saved = True
        try:
            if self.on_finish is not None:
                self.on_finish(self)
        except Exception:  # logged; the turn itself is finished
            logger.exception("could not save assistant turn %s", self.id)

    def _emit(self, event: dict[str, Any]) -> bool:
        """Add ``event``; False (nothing added) once the turn was cancelled."""
        with self.lock:
            if self.status != "running":
                return False
            self.events.append({"n": len(self.events), **event})
            return True

    def _finish(self, status: str, error: str | None = None) -> None:
        with self.lock:
            if self.status != "running":
                return
            self.status, self.error = status, error
            if error:
                self.events.append({"n": len(self.events), "kind": "error", "text": error})

    def _run(self) -> None:
        try:
            self._loop()  # returns with the turn still running only when the rounds ran out
            self._finish("failed", f"no answer after {MAX_ROUNDS} rounds; try a simpler request")
        except ProviderError as err:
            self._finish("failed", str(err))
        except Exception as err:  # a bug must not leave the turn running forever
            logger.exception("assistant turn %s failed", self.id)
            self._finish("failed", f"the assistant stopped: {err}")
        finally:
            self._save()
            self.done.set()

    def _messages(self, json_mode: bool) -> list[dict[str, Any]]:
        plan = context.budget(self.provider.context_tokens)
        tools = _JSON_TOOLS if json_mode else None
        return context.build_messages(self.table, self.earlier, self.text, plan, json_tools=tools)

    def _loop(self) -> None:
        caps = self.provider.capabilities()
        if not (caps.tool_calls or caps.json_schema):
            raise ProviderError(
                f"{self.provider.model} supports neither tool calls nor JSON output; "
                "choose another model"
            )
        json_mode = not caps.tool_calls
        messages = self._messages(json_mode)
        for _ in range(MAX_ROUNDS):
            if self.status != "running":
                return
            try:
                reply = self.provider.chat(
                    messages,
                    tools=None if json_mode else TOOLS,
                    schema=JSON_SCHEMA if json_mode else None,
                    cancelled=lambda: self.status != "running",
                )
            except ToolsUnsupportedError:
                json_mode = True
                messages = self._messages(True)
                continue
            if self.status != "running":
                return
            if json_mode:
                messages.append({"role": "assistant", "content": reply.text})
                calls, text, problem = _from_json(reply.text)
                if problem:
                    messages.append({"role": "user", "content": f"error: {problem}"})
                    continue
            else:
                messages.append(reply.message)
                calls, text = reply.tool_calls, reply.text.strip()
            if not calls:
                if not text:
                    raise ProviderError(f"{self.provider.model} gave an empty answer")
                self._emit({"kind": "reply", "text": text})
                self._finish("done")
                return
            for call in calls:
                result, final = self._call(call)
                if final:
                    self._finish("done")
                    return
                result = result[:MAX_RESULT_CHARS]
                if json_mode:
                    messages.append({"role": "user", "content": f"{call.name} result:\n{result}"})
                else:
                    messages.append(self.provider.tool_result(call, result))

    # -- tools -------------------------------------------------------------------------
    def _call(self, call: ToolCall) -> tuple[str, bool]:
        """``(result for the model, whether the turn is over)``."""
        if call.error:
            return f"error: {call.name}: {call.error}", False
        handler = getattr(self, f"_tool_{call.name}", None) if call.name in TOOL_NAMES else None
        if handler is None:
            return (
                f"error: unknown tool {call.name!r}; the tools are {', '.join(TOOL_NAMES)}",
                False,
            )
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
        self._emit(
            {"kind": "tool", "name": "find_moments", "query": query[:200], "found": len(found)}
        )
        if not found:
            return "no moment's label or transcript has those words", False
        rows = "\n".join(json.dumps(self.table.row(r), ensure_ascii=False) for r in found)
        return context.data_block("moments", rows), False

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
        lines = context.transcript_lines(self.project.transcript, item["source"], start, end)
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

    def _checked(self, ops: Any) -> tuple[dict[str, Any] | None, str]:
        """``ops`` with real ids, applied to the turn's sequence: ``({ops, after, diff}, "")``
        or ``(None, why)``."""
        mapped, problem = _real_ops(ops, self.table)
        if mapped is None:
            return None, problem
        p = self.project
        out = apply_ops(p.segments, mapped, p.candidates, p.durations, name=self.table.name)
        if out.error:
            return None, out.error
        if not any(row["changes"] for row in out.diff):
            return None, "these ops leave the sequence as it is"
        return {"ops": mapped, "after": out.after, "diff": out.diff}, ""

    def _tool_propose_edit(self, args: dict[str, Any]) -> tuple[str, bool]:
        summary = args.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return "error: summary must be one sentence for the user", False
        main, problem = self._checked(args.get("ops"))
        if main is None:
            return f"error: proposal rejected: {problem}. Fix the ops and propose again.", False
        suggestions = []
        raw = args.get("suggestions")
        for sug in (raw if isinstance(raw, list) else [])[:MAX_SUGGESTIONS]:
            label = sug.get("label") if isinstance(sug, dict) else None
            checked = self._checked(sug.get("ops"))[0] if isinstance(label, str) else None
            if checked is None or not label.strip():
                logger.debug("assistant: dropped a suggestion that does not apply: %r", sug)
                continue
            suggestions.append(self._numbered("s", label.strip()[:80], checked))
        event = {"kind": "proposal", **self._numbered("p", summary.strip(), main)}
        self._emit({**event, "suggestions": suggestions})
        return "proposal shown to the user", True

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
    if not isinstance(ops, list):
        return ops, ""
    out: list[Any] = []
    for n, op in enumerate(ops, 1):
        if not isinstance(op, dict):
            out.append(op)
            continue
        op = dict(op)
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


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def _from_json(text: str) -> tuple[list[ToolCall], str, str | None]:
    """A JSON-mode reply as ``(tool calls, reply text, problem)``."""
    raw = _FENCE.sub("", text.strip())
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("tool"), str):
        return [], "", 'reply with one JSON object such as {"tool": "reply", "text": "..."}'
    if data["tool"] == "reply":
        reply = data.get("text")
        if not isinstance(reply, str) or not reply.strip():
            return [], "", 'a reply needs "text"'
        return [], reply.strip(), None
    args = data.get("arguments")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return [ToolCall(data["tool"], {}, "arguments must be a JSON object")], "", None
    return [ToolCall(data["tool"], args)], "", None


class Turns:
    """The turns of every project, at most one running per project.

    ``on_finish`` of a turn (it saves the turn to ``chat.json``) runs under the lock and only
    if the chat was not cleared since the turn started, so a Clear chat is never followed by an
    older turn's write.
    """

    def __init__(self) -> None:
        # reentrant: clear() cancels turns under it, and a cancelled turn saves itself at once
        self.lock = threading.RLock()
        self.turns: dict[str, tuple[str, Turn]] = {}  # turn id -> (project key, turn)
        self.cleared: dict[str, int] = {}  # project key -> times its chat was cleared

    def start(self, key: str, turn: Turn, on_finish: Callable[[Turn], None]) -> Turn:
        with self.lock:
            if any(k == key and t.status == "running" for k, t in self.turns.values()):
                raise BusyError("the assistant is still answering; wait or cancel it")
            generation = self.cleared.get(key, 0)
            done = [i for i, (_, t) in self.turns.items() if t.done.is_set()]
            for i in done[: max(0, len(done) + 1 - MAX_KEPT)]:
                del self.turns[i]
            self.turns[turn.id] = (key, turn)

        def finished(t: Turn) -> None:
            with self.lock:
                if self.cleared.get(key, 0) == generation:
                    on_finish(t)

        turn.on_finish = finished
        turn.start()
        return turn

    def get(self, key: str, tid: str) -> Turn | None:
        with self.lock:
            hit = self.turns.get(tid)
        return hit[1] if hit and hit[0] == key else None

    def of(self, key: str) -> list[Turn]:
        with self.lock:
            return [t for k, t in self.turns.values() if k == key]

    def clear(self, key: str) -> None:
        """Cancel the project's running turn and forget its turns (their results are not
        saved)."""
        with self.lock:
            self.cleared[key] = self.cleared.get(key, 0) + 1
            for tid, (k, t) in list(self.turns.items()):
                if k == key:
                    t.cancel()
                    del self.turns[tid]

    def shutdown(self) -> None:
        with self.lock:
            for _, t in self.turns.values():
                t.cancel()
