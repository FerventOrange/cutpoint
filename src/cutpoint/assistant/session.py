"""One chat turn, run in a thread: the model looks moments up and ends with a proposal or a reply.

The loop offers the read tools (``find_moments``, ``look_for`` (CLIP search of the whole
timeline, :mod:`cutpoint.assistant.lookfor`), ``get_moment``, ``get_transcript``,
``show_moments``, and ``view_moment`` for a model that takes images, :mod:`cutpoint.assistant.watch`)
and the final ``propose_edit``, for at most :data:`MAX_ROUNDS` model replies.
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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cutpoint.assistant import context, lookfor, pacing, watch
from cutpoint.assistant.edges import EdgePlanner, length
from cutpoint.assistant.ops import apply_ops, change_counts, summary_gap
from cutpoint.assistant.providers import (
    OllamaProvider,
    OutputCapError,
    Provider,
    ProviderError,
    Reply,
    ToolCall,
    ToolsUnsupportedError,
    TruncatedError,
    describe,
)
from cutpoint.assistant.topup import kind_groups, top_up

logger = logging.getLogger(__name__)

MAX_ROUNDS = 8  # model replies in one turn
FIT_CHARS_PER_TOKEN = context.FIT_CHARS_PER_TOKEN
MAX_RESULT_CHARS = context.MAX_RESULT_CHARS
MAX_TEXT = 2000  # characters of a user message
MAX_SUMMARY = 500
MAX_SUGGESTIONS = 4
MAX_KEPT = 50  # finished turns kept in memory (chat.json has them all)
MAX_PAD_S = 60.0  # get_transcript's pad
NUDGE = (
    "If the user asked for a change to the sequence, call propose_edit now with the ops; "
    "otherwise give your answer again."
)
_MOMENT_ID = re.compile(r"\bm\d+\b")
TOLERANCE = 0.15  # a proposal this far over or under the asked length is sent back
MAX_LENGTH_BACKS = 3  # ... up to this many times a turn, then the closest one is offered
SHORT_REPLY = (
    "Your reply was too long. Reply shorter: call propose_edit now with only the ops "
    "(a one-sentence summary), or answer in a few lines."
)
# A message that asks to empty the sequence: only then may a proposal remove every clip.
# A message that asks for an edit: a turn that ends in text gets the one nudge toward a proposal.
# A question asks none unless it is a request ("can you add something funny?").
_REQUEST = re.compile(r"\s*(?:please|can you|could you|would you|will you)\b", re.I)
_EDIT_ASK = re.compile(
    r"\b(?:make|build|add|put|place|change|extend|lengthen|trim|shorten|remove|delete|move|reorder"
    r"|sort|create|assemble|cut|fix|longer|shorter|arrange|include|insert|swap|replace|color|grade)\b",
    re.I,
)
_REMOVES = re.compile(
    r"\b(?:remove|delete|drop|clear|get rid|discard|take out|cut out|replace|swap|start over"
    r"|from scratch)\b",
    re.I,
)


def _wants_removal(text: str) -> bool:
    """Whether a message asks to take clips out (so a proposal may drop half of them)."""
    return bool(_REMOVES.search(text) or _CLEARS.search(text))


_CLEARS = re.compile(
    r"\b(?:start (?:over|again|fresh|from scratch)|from scratch|begin again"
    r"|(?:clear|empty)(?: out)? (?:the |my )?(?:whole |entire )?(?:sequence|timeline)"
    r"|(?:remove|delete|drop|clear)\s+(?:everything|all (?:of )?(?:the |my )?clips?"
    r"(?=\s*(?:$|[.,;!]|and\b|from\b|in\b))))",
    re.I,
)


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
    "description": "edits in order: add, remove, move, trim, order, sort or color ops",
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
    lookfor.JSON_TOOL,
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
    windows: list[dict[str, Any]] = field(default_factory=list)  # context.vlm_windows(events.json)
    # exported edges of rows and phrase ends; None: the raw ranges, and trims are not snapped
    edges: EdgePlanner | None = None
    # some source has cached frame embeddings (lookfor.indexed): without any, look_for is not offered
    frame_index: bool = True


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def cut_again_error(provider: Any, err: TruncatedError, again: bool = True) -> ProviderError:
    """The failure for a reply cut off after the conversation was already made smaller: the
    provider's own message, and the context advice only where ``assistant.context_tokens``
    is what the model is given (Ollama's ``num_ctx``). ``again``: false when the reply was cut
    off only once (in the final round)."""
    text = f"{err} (cut off again)" if again else str(err)
    if isinstance(provider, OllamaProvider):
        text += (
            f"; the context is {provider.context_tokens} tokens: ask for less, or raise "
            "assistant.context_tokens in config.json"
        )
    return ProviderError(text)


class Turn:
    """One user message and the model's work on it; :meth:`view` is what the page polls."""

    def __init__(
        self,
        provider: Provider,
        project: Project,
        text: str,
        earlier: list[dict[str, Any]] | None = None,
        viewer: watch.Viewer | None = None,
    ) -> None:
        self.id = secrets.token_hex(6)
        self.provider = provider
        self.project = project
        self.text = text
        self.earlier = [
            {"user": str(t.get("user", "")), "reply": str(t.get("reply", ""))}
            for t in earlier or []
        ]
        self.table = context.MomentTable(project.candidates, project.segments, project.windows)
        self.table.durations = project.durations
        self.target = context.target_seconds(text)  # seconds the user asked for, if any
        self.mix = pacing.asks_for_mix(text)  # asks to break up / mix in kinds: code spreads them
        self.dialogue = pacing.names_dialogue(text)
        self._kinds: dict[str, str] = {}  # id -> pacing kind (memo of :meth:`_pace_kind`)
        self.status = "running"  # running | done | failed | cancelled
        self.error: str | None = None
        self.events: list[dict[str, Any]] = []
        self.started = _now()
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.on_finish: Callable[[Turn], None] | None = None
        self._proposals = 0
        self._sent_back: set[str] = set()  # summary problem sent back already (once a turn)
        self._searches: list[list[str]] = []  # real ids each search found, in order (top-up)
        self._length_backs = 0  # proposals sent back for their length (up to MAX_LENGTH_BACKS)
        # the valid proposals sent back only for their length ((summary, checked)): the closest
        # to the asked length is offered when the retries do not get there
        self._candidates: list[tuple[str, dict[str, Any]]] = []
        self._nudged: str | None = None  # the text-only reply that got the nudge (this turn's one)
        self._results: list[dict[str, Any]] = []  # tool result messages of the conversation
        self._saved = False
        # view_moment: offered (decided in _loop), moments viewed, sheets waiting to be sent and
        # the image messages sent (taken out again if the model refuses images)
        self.viewer = viewer
        self._watch = False
        self._viewed: set[str] = set()
        self._sheets: list[dict[str, Any]] = []
        self._images: list[dict[str, Any]] = []
        # look_for: the finder (reads each source's cached embeddings once) and the new moments
        # it made, which add ops can take and proposals carry
        self._finder = lookfor.Finder(list(project.durations))
        self.found: list[dict[str, Any]] = []
        self._unsure: set[str] = set()  # ids of found windows that are only guesses

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
                    # what the ops really change, so a later turn does not take the summary's word
                    counts = change_counts(e["diff"], e.get("color"))
                    auto = e.get("auto")
                    if auto:
                        counts += f"; {len(auto['lengthened']) + len(auto['added'])} auto-filled"
                    if pace := e.get("pace"):
                        n = len({*pace["moved"], *pace["added"], *pace["replaced"]})
                        counts += f"; {n} auto-spread"
                    reply = f"Proposed: {e['summary']} (changes: {counts})"
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

    def _messages(self, json_mode: bool, *, tight: bool = False) -> list[dict[str, Any]]:
        # the tool definitions of a tool-call request count; JSON mode lists them in the rules
        sent = None if json_mode else self._tools()
        sheets = watch.view_cap(self.provider) if self._watch else 0
        plan = context.budget(
            self.provider.context_tokens,
            sent,
            tight=tight,
            images=sheets,
            image_tokens=watch.sheet_tokens(self.provider),
        )
        tools = None
        if json_mode:
            listed = [t for t in _JSON_TOOLS if self.project.frame_index or t != lookfor.JSON_TOOL]
            tools = [*listed, *([watch.JSON_TOOL] if self._watch else [])]
        # look_for's rules, then how and when to watch, after the others and measured with them
        late = lookfor.rules(bool(self._watch)) if self.project.frame_index else ""
        if self._watch:
            late += watch.RULES.replace("{cap}", watch.views(sheets))
        out = context.build_messages(
            self.table,
            self.earlier,
            self.text,
            plan,
            json_tools=tools,
            images=bool(self._watch),
            target=self.target,
            late_rules=late,
        )
        return out

    def _names(self) -> list[str]:
        """The names of the tools offered: ``look_for`` only with cached frame embeddings."""
        return [n for n in TOOL_NAMES if self.project.frame_index or n != lookfor.NAME]

    def _tools(self) -> list[dict[str, Any]]:
        base = TOOLS if self.project.frame_index else [t for t in TOOLS if t is not lookfor.TOOL]
        return [*base, watch.TOOL] if self._watch else base

    def _schema(self) -> dict[str, Any]:
        if not self._watch and self.project.frame_index:
            return JSON_SCHEMA
        names = [*self._names(), *([watch.NAME] if self._watch else []), "reply"]
        props = {**JSON_SCHEMA["properties"], "tool": {"type": "string", "enum": names}}
        return {**JSON_SCHEMA, "properties": props}

    def _ask(self, messages: list[dict[str, Any]], json_mode: bool) -> Reply:
        """One reply. A model that refuses the sheets sent (:func:`watch.rejected_image`) is
        asked once more with them taken out, and is offered no more this session."""

        def once() -> Reply:
            return self.provider.chat(
                messages,
                tools=None if json_mode else self._tools(),
                schema=self._schema() if json_mode else None,
                cancelled=lambda: self.status != "running",
            )

        try:
            return once()
        except (ToolsUnsupportedError, TruncatedError):
            raise
        except ProviderError as err:
            if not self._images or not watch.rejected_image(err):
                raise
            logger.info("assistant: %s takes no images (%s)", self.provider.model, err)
        watch.mark_text_only(self.provider)
        self._watch = False
        # nothing was seen after all: no guess counts as viewed, and the rules stop saying it can
        # be confirmed with view_moment
        self._viewed.clear()
        messages[0] = {
            **messages[0],
            "content": messages[0]["content"].replace(lookfor.WEAK_IMAGES, lookfor.WEAK_TEXT),
        }
        for m in self._images:  # the same dicts as in messages
            m["content"] = "(the timeline sheet could not be sent: this model takes no images)"
        self._images = []
        return once()

    def _loop(self) -> None:
        caps = self.provider.capabilities()
        if not (caps.tool_calls or caps.json_schema):
            raise ProviderError(
                f"{self.provider.model} supports neither tool calls nor JSON output; "
                "choose another model"
            )
        json_mode = not caps.tool_calls
        if (planner := self.project.edges) is not None:  # the lengths the model sees: as exported
            self.table.planned = planner.plan(self.project.segments)
        self._watch = self.viewer is not None and watch.takes_images(self.provider, caps)
        messages = self._messages(json_mode)
        base = len(messages)  # the context built up front; what follows is this turn's exchange
        cut = False  # a reply was cut off once: the conversation was shrunk and asked again
        capped = False  # a reply hit the output cap once: asked again for a shorter one
        last_cut: TruncatedError | None = None  # the latest round's cut-off, if it was one
        for rnd in range(MAX_ROUNDS):
            if self.status != "running":
                return
            self._fit(messages, json_mode)
            try:
                reply = self._ask(messages, json_mode)
                last_cut = None
            except ToolsUnsupportedError:
                json_mode = True
                self._results = []
                messages = self._messages(True)
                base = len(messages)
                continue
            except OutputCapError as err:  # the context is not the problem: ask for less output
                if capped:
                    raise ProviderError(
                        f"{self.provider.model} ran past its reply limit ({err.tokens} tokens) "
                        "twice; ask for one change at a time"
                    ) from None
                capped = True
                logger.info("assistant: reply hit the output cap (%s tokens)", err.tokens)
                messages.append({"role": "user", "content": SHORT_REPLY})
                continue
            except TruncatedError as err:
                if cut:
                    raise cut_again_error(self.provider, err) from None
                last_cut = err
                cut = True
                logger.info("assistant: reply cut off; asking again with less")
                # old results go; with none to drop, the context is built smaller and the
                # exchange of this turn (the user's message, the latest result) is kept as is
                if not self._fit(messages, json_mode, tight=True):
                    smaller = self._messages(json_mode, tight=True)
                    messages = smaller + messages[base:]
                    base = len(smaller)
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
                if self._nudged is not None and not text:
                    text = self._nudged  # the answer given again is empty: the first one stands
                if not text:
                    raise ProviderError(f"{self.provider.model} gave an empty answer")
                if (
                    self._nudged is None
                    and rnd < MAX_ROUNDS - 1  # a nudge in the last round would leave no answer
                    and (self._names_moments(text) or self._asks_edit())
                ):
                    # moments chosen in text, or an edit asked for and answered in text: ask
                    # once more (the nudge stays in this conversation; the page and chat.json
                    # never get it)
                    self._nudged = text
                    messages.append({"role": "user", "content": NUDGE})
                    continue
                if self._offer_fallback():  # the earlier proposal beats an answer in text
                    self._finish("done")
                    return
                self._emit({"kind": "reply", "text": text})
                self._offer_own()  # a short sequence gets the fill to review, next to the text
                self._finish("done")
                return
            for call in calls:
                result, final = self._call(call)
                if final:
                    self._finish("done")
                    return
                result = result[:MAX_RESULT_CHARS]
                if json_mode:
                    sent = {"role": "user", "content": f"{call.name} result:\n{result}"}
                else:
                    sent = self.provider.tool_result(call, result)
                messages.append(sent)
                self._results.append(sent)
            # sheets after every tool result of the reply (OpenAI wants the results together)
            messages += self._sheets
            self._images += self._sheets
            self._sheets = []
        if self.status != "running":
            return
        if self._offer_fallback() or self._offer_own():  # the rounds ran out: better than none
            self._finish("done")
            return
        if last_cut is not None:
            raise cut_again_error(self.provider, last_cut, again=False) from None
        if self._nudged is not None:  # the rounds ran out after the nudge: the first answer stands
            self._emit({"kind": "reply", "text": self._nudged})
            self._finish("done")

    def _asks_edit(self) -> bool:
        """Whether the user's message asks for a change to the sequence: a length or an edit
        word ("add", "trim", "make", ...), and no question ("anything to fix?") unless it is a
        request ("can you add something funny?")."""
        if self.text.rstrip().endswith("?") and not _REQUEST.match(self.text):
            return False
        return bool(self.target or _EDIT_ASK.search(self.text))

    def _names_moments(self, text: str) -> bool:
        """Whether ``text`` names a moment of the table (``m3``)."""
        return any(self.table.real(m) is not None for m in _MOMENT_ID.findall(text))

    def _fit(self, messages: list[dict[str, Any]], json_mode: bool, *, tight: bool = False) -> bool:
        """Leave out the oldest tool results until the conversation, the tool definitions and
        room for a reply fit the context (``tight``: all but the latest). The system messages,
        the user's messages and the latest result stay. True if a result was left out."""
        limit = (self.provider.context_tokens - context.REPLY_TOKENS) * FIT_CHARS_PER_TOKEN
        total = context.size(messages) + (0 if json_mode else len(json.dumps(self._tools())))
        left = False
        for msg in self._results[:-1]:
            if total <= limit and not tight:
                break
            before = context.size(msg)
            _leave_out(msg)
            total -= before - context.size(msg)
            left = left or before != context.size(msg)
        return left

    # -- tools -------------------------------------------------------------------------
    def _call(self, call: ToolCall) -> tuple[str, bool]:
        """``(result for the model, whether the turn is over)``."""
        if call.error:
            return f"error: {call.name}: {call.error}", False
        names = [*self._names(), *([watch.NAME] if self._watch else [])]
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
            self._emit(
                {
                    "kind": "tool",
                    "name": lookfor.NAME,
                    "query": query[:200],
                    "error": str(err),
                    **({"reason": err.reason} if err.reason else {}),
                }
            )
            return f"error: look_for is not available: {err}", False
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
        skipped = got.total - got.searched
        note = (
            f" ({skipped} of {got.total} sources have no cached frames and were not searched)"
            if skipped
            else ""
        )
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
            return (
                f"error: at most {watch.views(cap)} can be viewed per message; answer with what you "
                "have seen, or ask the user to send another message",
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

    def _checked(self, ops: Any) -> tuple[dict[str, Any] | None, str]:
        """``ops`` with real ids, applied to the turn's sequence: ``({ops, after, diff}, "")``
        or ``(None, why)``."""
        mapped, problem = _real_ops(ops, self.table)
        if mapped is None:
            return None, problem
        return self._checked_real(mapped)

    def _remember(self, ids: list[str]) -> None:
        """Note the moments one search found, for :func:`topup.top_up`."""
        if ids:
            self._searches.append(list(ids))

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
        if p.segments and not out.after and not _CLEARS.search(self.text):
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
            and not (_wants_removal(self.text) or wants_less)
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
        if wrong := self._length_gap(summary, main):
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
        return "proposal shown to the user", True

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
    def _length_gap(summary: str, main: dict[str, Any]) -> str | None:
        """A sentence for the model when ``summary`` gives the sequence a length ("now about 3
        minutes", read by :func:`context.claimed_seconds`) more than the tolerance off the
        proposal's exported length; None when it gives none or the right one."""
        claim = context.claimed_seconds(summary)
        have = main["length"]
        if claim is None or abs(claim - have) <= claim * TOLERANCE:
            return None
        return (
            f"The summary says the sequence is {claim:.0f} s, but this proposal makes {have:.0f} s "
            "as exported. Fix the summary or the ops and propose again."
        )

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
        nothing to fill it with."""
        p = self.project
        if not self.target or (
            not _EDIT_ASK.search(self.text) and self.text.rstrip().endswith("?")
        ):
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
                        self.project.transcript, row["source"], row["start"], row["end"]
                    )
                    said = " ".join(line.partition("] ")[2] for line in lines)
                return len(said.split()) >= pacing.SPOKEN_WORDS

            label = str(item.get("label") or "")
            self._kinds[real] = pacing.moment_kind(
                label, cand, look.cues if look is not None else {}, spoken
            )
        return self._kinds[real]

    def _paced(self, summary: str, main: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """``(summary, proposal)`` with the kinds spread between the skiing runs
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
            parts.append(f"Spread {listed} clips between the skiing runs.")
        if found.added:
            n, r = len(found.added), len(found.replaced)
            more = f", replacing {r} skiing clip{'s' * (r != 1)}" if r else ""
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


def _leave_out(msg: dict[str, Any]) -> None:
    """Replace the result in a tool result message (any provider's shape) with a note."""
    if "output" in msg:  # OpenAI Responses
        msg["output"] = context.LEFT_OUT
    elif isinstance(msg.get("content"), list):  # Anthropic: tool_result blocks
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                block["content"] = context.LEFT_OUT
    else:
        msg["content"] = context.LEFT_OUT


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
