"""One chat turn, run in a thread: the model looks moments up and ends with a proposal or a reply.

The loop offers the read tools (``find_moments``, ``look_for`` (CLIP search of the whole
timeline, :mod:`cutpoint.assistant.lookfor`), ``find_words`` (where words are said, to the
word, :mod:`cutpoint.assistant.words`), ``get_moment``, ``get_transcript``,
``show_moments``, and ``view_moment`` for a model that takes images, :mod:`cutpoint.assistant.watch`)
and the final ``propose_edit`` (or ``preview_ranking``, the whole list ranked in another style,
:mod:`cutpoint.assistant.ranking`, for a project the page gives a ranking hook), for at most :data:`MAX_ROUNDS` model replies. The last of them,
and the one after ``view_moment`` hit its limit, is the closing round: it offers only
``propose_edit`` and asks for the answer (:data:`CLOSING`); a model that still looks something up
gets that round once more, then the turn ends with the best it has (an earlier proposal, its
first answer, or a short answer of the code's), never without one.
A proposal is checked with :func:`cutpoint.assistant.ops.apply_ops`; a rejected one goes back to
the model with the reason. A model without tool calls (or one that refuses them) answers in JSON
mode: one ``{"tool", "arguments"}`` object per reply, constrained by
:data:`cutpoint.assistant.toolcalls.JSON_SCHEMA`.

:class:`Turn` is the state machine and the loop; its parts are in
:mod:`cutpoint.assistant.prompting` (the messages and tools sent),
:mod:`cutpoint.assistant.toolcalls` (the tools and their calls) and
:mod:`cutpoint.assistant.proposing` (proposals).

Nothing here writes project files or runs commands: a turn only adds events (tool use, moments
to show, a reply, proposals) that the page polls, the way it polls jobs.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cutpoint.assistant import context, errtext, intent, lookfor, ranking, watch
from cutpoint.assistant.edges import EdgePlanner
from cutpoint.assistant.ops import change_counts
from cutpoint.assistant.prompting import Prompting, _earlier_reply, _without_folder, _without_names
from cutpoint.assistant.proposing import Proposing
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
from cutpoint.assistant.toolcalls import FINAL_TOOL, ToolCalls

logger = logging.getLogger(__name__)

MAX_ROUNDS = 8  # model replies in one turn (plus one if the closing round is asked again)
MAX_TEXT = 2000  # characters of a user message
MAX_KEPT = 50  # finished turns kept in memory (chat.json has them all)
MAX_LOOKED = 4  # searches the code's own answer names (_no_answer)
NUDGE = (
    "If the user asked for a change to the sequence, call propose_edit now with the ops; "
    "otherwise give your answer again."
)
# The closing round: the lookups are spent, so the model answers with what it has. A small model
# that found nothing keeps searching or viewing until the rounds run out; this asks it for the
# plain answer instead ("I couldn't find ...; here is what they show", with something it can do)
CLOSING = (
    "No more lookups in this turn: answer the user now with what you have found. If the user "
    "asked for a change and the moments you found fit it, call propose_edit with the ops; "
    "otherwise answer in plain text. If the moments do not show what the user asked for, say "
    'so plainly ("I couldn\'t find ... in these moments"), say in a sentence or two what they '
    "do show, and offer what you can do instead, such as a montage of the best moments."
)
CLOSING_JSON = (
    f'{CLOSING} Reply with one JSON object: {{"tool": "reply", "text": "..."}}, or propose_edit.'
)
# A lookup asked for in the closing round: not run (the result each call needs)
NOT_RUN = (
    "error: not run: no more lookups in this turn. Answer the user now in plain text with what "
    "you have found, or call propose_edit"
)
# The nudge after find_words found words: a small model answers "make him say X" with "the words
# are never said together" (they never are: that is what a splice is for)
SPLICE_NUDGE = (
    "If the user asked for a change to the sequence, call propose_edit now with the ops; to make "
    "someone say words, add_range one find_words hit of each word, in order, even though they "
    "were never said together. Otherwise give your answer again."
)
# The nudge after a reply that promises a lookup and makes no call ("I'll search each one now."):
# a small model ends its turn there (#1013)
PROMISE_NUDGE = (
    "You said you would look something up, but you made no tool call. Make that call now; if "
    "you already have what you need, give your answer instead."
)
_MOMENT_ID = re.compile(r"\bm\d+\b")
SHORT_REPLY = (
    "Your reply was too long. Reply shorter: call propose_edit now with only the ops "
    "(a one-sentence summary), or answer in a few lines."
)


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
    # without them, why and what adds them (lookfor.index_state): the rules tell the model
    index_state: Mapping[str, Any] | None = None
    # a source's cached transcript words (edges.EdgePlanner.words); None: find_words finds none
    words: Callable[[str], list[dict[str, Any]] | None] | None = None
    # the library styles to preview a ranking in (ranking.Hook); None: preview_ranking is not
    # offered (no analysis, a 360 project)
    ranking: ranking.Hook | None = None
    # ui.ranking_key of highlights.json: kept in the turn record, so a later turn knows the
    # moment ids of an earlier one changed (another ranking or analysis since)
    ranking_key: str | None = None
    # the project's files the model may name (context.project_files): names only
    files: list[dict[str, str]] = field(default_factory=list)
    # the absolute project folder: given only to a model on this machine (Turn.folder)
    folder: str = ""


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


def _once_more(rnd: int, again: bool) -> tuple[int, bool]:
    """``(rounds, again)`` when the closing round gave no answer: that round is asked once
    more, then the turn ends with its best."""
    return rnd + (1 if again else 2), True


class Turn(Prompting, ToolCalls, Proposing):
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
        self.table = context.MomentTable(project.candidates, project.segments, project.windows)
        self.table.durations = project.durations
        self.table.local = self._on_this_machine()  # before the earlier turns name files
        # the edges of the find_words hits of this turn and the earlier ones: an add or trim to
        # one makes its row exact
        self._word_edges: list[tuple[str, float]] = []  # (source, s): both edges of each hit
        # only the turns the model is shown (context.budget's window) give their hits
        window = context.budget(provider.context_tokens).turns
        earlier = list(earlier or [])
        seen = len(earlier) - window
        key = project.ranking_key
        # a model not given the folder gets no earlier reply naming it (chat.json keeps it),
        # and the names of sources of one file name a model of the other kind was shown become
        # this one's
        hidden = project.folder if self.folder() is None else ""
        renamed = self.table.other_names()

        def reply(t: Mapping[str, Any], i: int) -> str:
            text = _without_names(_earlier_reply(t), renamed)
            text += self._hits_of(t) if i >= seen else ""
            return _without_folder(text, hidden) if hidden else text

        self.earlier = [
            {
                "user": str(t.get("user", "")),
                "reply": reply(t, i),
                # its short ids (m1...) were of another ranking (context.IDS_CHANGED)
                "ids_changed": isinstance(key, str)
                and isinstance(t.get("ranking_key"), str)
                and t["ranking_key"] != key,
            }
            for i, t in enumerate(earlier)
        ]
        self.target = intent.target_seconds(text)  # seconds the user asked for, if any
        self.mix = intent.asks_for_mix(text)  # asks to break up / mix in kinds: code spreads them
        self.dialogue = intent.names_dialogue(text)
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
        self._splice: str | None = None  # the add_range ops of a phrase said apart (_apart)
        # find_words found the words, this turn or an earlier one the model is shown
        # (SPLICE_NUDGE)
        self._found_words = bool(self._word_edges)
        self._results: list[dict[str, Any]] = []  # tool result messages of the conversation
        self._saved = False
        # view_moment: offered (decided in _loop), moments viewed, sheets waiting to be sent and
        # the image messages sent (taken out again if the model refuses images)
        self.viewer = viewer
        self._watch = False
        self._viewed: set[str] = set()
        self._spent = False  # a view was refused for the limit: the next round is the closing one
        self._sheets: list[dict[str, Any]] = []
        self._images: list[dict[str, Any]] = []
        # look_for: the finder (reads each source's cached embeddings once) and the new moments
        # it made, which add ops can take and proposals carry
        self._finder = lookfor.Finder(list(project.durations))
        self.found: list[dict[str, Any]] = []
        self._unsure: set[str] = set()  # ids of found windows that are only guesses
        # preview_ranking: the styles offered ((name, note), ranking.listed) and the one the
        # moments are ranked in now; no other style, no tool
        self._styles: list[tuple[str, str]] = []
        self._current = ""
        if project.ranking is not None:
            try:
                self._current = str(project.ranking.current() or "")
                self._styles = ranking.listed(project.ranking.styles())
            except Exception:  # a hook that fails offers no rankings; the turn goes on
                logger.exception("assistant: the ranking styles could not be read")
                self._styles = []
            if not [n for n, _ in self._styles if n != self._current]:
                self._styles = []

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
                "ranking_key": self.project.ranking_key,
                # the files the model was told of: the page links a reply naming one
                "files": [{"name": f["name"], "target": f["target"]} for f in self.project.files],
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
                elif e["kind"] == "ranking":
                    reply = ranking.reply(e)
            out = {
                "id": self.id,
                "user": self.text,
                "started": self.started,
                "status": self.status,
                "error": self.error,
                "events": list(self.events),
                "reply": reply,
            }
            if self.project.ranking_key is not None:  # the ranking its short ids are of
                out["ranking_key"] = self.project.ranking_key
            return out

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
        # the error is the turn's reply in chat.json, sent to the model on later turns: no
        # folders of this computer in it (#1017); the log has the full text
        cleaned = errtext.no_paths(error) if error else error
        if cleaned != error:
            logger.info("assistant turn %s failed: %s", self.id, error)
        error = cleaned
        with self.lock:
            if self.status != "running":
                return
            self.status, self.error = status, error
            if error:
                self.events.append({"n": len(self.events), "kind": "error", "text": error})

    def _run(self) -> None:
        try:
            self._loop()
        except ProviderError as err:
            self._finish("failed", str(err))
        except Exception as err:  # a bug must not leave the turn running forever
            logger.exception("assistant turn %s failed", self.id)
            self._finish("failed", f"the assistant stopped: {errtext.error_text(err)}")
        finally:
            self._save()
            self.done.set()

    def _ask(self, messages: list[dict[str, Any]], json_mode: bool, closing: bool = False) -> Reply:
        """One reply. A model that refuses the sheets sent (:func:`watch.rejected_image`) is
        asked once more with them taken out, and is offered no more this session."""

        def once() -> Reply:
            return self.provider.chat(
                messages,
                tools=None if json_mode else self._tools(closing),
                schema=self._schema(closing) if json_mode else None,
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
        rounds = MAX_ROUNDS  # one more when the closing round looks something up
        closing = False  # the closing round: only propose_edit, and the answer is asked for
        noted = False  # CLOSING is in the conversation
        again = False  # the closing round was asked once more
        for rnd in itertools.count():
            if rnd >= rounds or self.status != "running":
                break
            closing = closing or self._spent or rnd >= MAX_ROUNDS - 1
            if closing and not noted:
                messages.append({"role": "user", "content": CLOSING_JSON if json_mode else CLOSING})
                noted = True
            self._fit(messages, json_mode)
            try:
                reply = self._ask(messages, json_mode, closing)
                last_cut = None
            except ToolsUnsupportedError:
                json_mode = True
                self._results = []
                messages = self._messages(True)
                base = len(messages)
                noted = False
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
                    if closing:  # no answer in the closing round: once more, then the best
                        rounds, again = _once_more(rnd, again)
                    continue
            else:
                messages.append(reply.message)
                calls, text = reply.tool_calls, reply.text.strip()
            if not calls:
                if self._nudged is not None and not text:
                    text = self._nudged  # the answer given again is empty: the first one stands
                if not text and closing:  # once more, then the turn's best
                    rounds, again = _once_more(rnd, again)
                    continue
                if not text:
                    raise ProviderError(f"{self.provider.model} gave an empty answer")
                # a nudge in the closing round would leave no answer
                nudge = None if self._nudged is not None or closing else self._nudge(text)
                if nudge is not None:
                    # ask once more (the nudge stays in this conversation; the page and chat.json
                    # never get it, and the reply after it is the answer)
                    self._nudged = text
                    messages.append({"role": "user", "content": nudge})
                    continue
                if self._offer_fallback():  # the earlier proposal beats an answer in text
                    self._finish("done")
                    return
                self._emit({"kind": "reply", "text": text})
                self._offer_own()  # a short sequence gets the fill to review, next to the text
                self._finish("done")
                return
            for call in calls:
                if closing and call.name != FINAL_TOOL:
                    logger.info("assistant: %s in the closing round: not run", call.name)
                    result, final = NOT_RUN, False
                else:
                    result, final = self._call(call)
                if final:
                    self._finish("done")
                    return
                result = context.cut_result(
                    result, context.MAX_RESULT_CHARS
                )  # a data block keeps its end
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
            if closing:  # looked something up instead of answering: once more, then the best
                rounds, again = _once_more(rnd, again)
        if self.status != "running":
            return
        if self._offer_fallback() or self._offer_own():  # the rounds ran out: better than none
            self._finish("done")
            return
        if last_cut is not None:
            raise cut_again_error(self.provider, last_cut, again=False) from None
        # the rounds ran out after the nudge: the first answer stands; else the code's own
        self._emit({"kind": "reply", "text": self._nudged or self._no_answer()})
        self._finish("done")

    def _nudge(self, text: str) -> str | None:
        """The one nudge of a turn for its text-only reply ``text``, or None. Moments chosen in
        text (not for an advice question: its examples name moments), more work announced to an
        edit request, or an edit asked for and answered in text: :data:`NUDGE` (with a phrase
        found, :data:`SPLICE_NUDGE`). Otherwise a lookup promised and not made (#1013):
        :data:`PROMISE_NUDGE`."""
        if (
            (self._names_moments(text) and not intent.asks_advice(self.text))
            or self._announces_work(text)
            or self._asks_edit()
        ):
            if self._found_words:  # with the ops of a phrase said apart, if any
                ops = context.data_block("splice", self._splice) if self._splice else ""
                return SPLICE_NUDGE + (f" The ops:\n{ops}" if ops else "")
            return NUDGE
        return PROMISE_NUDGE if intent.promises(text) else None

    def _no_answer(self) -> str:
        """The reply of a turn whose model gave none: the last rejection of its proposals, or
        what was looked for; and what to do."""
        with self.lock:
            rejected = [e["text"] for e in self.events if e["kind"] == "rejected"]
            looked = [e["query"] for e in self.events if e["kind"] == "tool" and e.get("query")]
        if rejected:
            return (
                "I couldn't make an edit that passes the checks (the last problem: "
                f"{rejected[-1]}). Try asking for one change at a time."
            )
        looked = list(dict.fromkeys(looked))[:MAX_LOOKED]
        viewed = sorted(self.table.name(r) for r in self._viewed)
        done = []
        if looked:
            done.append("searched for " + ", ".join(repr(q) for q in looked))
        if viewed:
            done.append("viewed " + ", ".join(viewed))
        how = f" (I {' and '.join(done)})" if done else ""
        return (
            f"I couldn't find an answer to this in the moments{how}. Try describing it in other "
            "words, or ask for something they have, such as a montage of the best moments."
        )

    def _announces_work(self, text: str) -> bool:
        """Whether ``text`` says more work follows ("Now let me search for ...") to a message
        that asks for an edit, instead of doing it."""
        return bool(intent.ANNOUNCES.search(text) and (self.target or intent.edits(self.text)))

    def _asks_edit(self) -> bool:
        """Whether the user's message asks for a change to the sequence: a length or an edit
        word ("add", "trim", "make", ...). A question asks one only with an edit word and not
        for advice: "could we add the laugh?" and "add the laugh, ok?" do; "what would you
        trim?", "anything to fix?" and "can it be 40 s long?" do not."""
        if self.text.rstrip().endswith("?"):
            return not intent.asks_advice(self.text)
        return bool(self.target or intent.edits(self.text))

    def _names_moments(self, text: str) -> bool:
        """Whether ``text`` names a moment of the table (``m3``). The nudge skips this for an
        advice question, whose answer cites moments as examples."""
        return any(self.table.real(m) is not None for m in _MOMENT_ID.findall(text))


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
