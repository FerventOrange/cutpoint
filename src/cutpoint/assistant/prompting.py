"""What a chat turn sends the model: the messages (:meth:`Prompting._messages`), the tools
offered and their JSON schema, the conversation fitted to the context, and the earlier turns'
replies without this computer's folders or other names, by :class:`Prompting`, the prompt part
of :class:`cutpoint.assistant.session.Turn`.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cutpoint.assistant import context, errtext, lookfor, ranking, watch, words
from cutpoint.assistant.providers import is_consent_exempt
from cutpoint.assistant.toolcalls import _JSON_TOOLS, FINAL_TOOL, JSON_SCHEMA, TOOL_NAMES, TOOLS


def _earlier_reply(turn: Mapping[str, Any]) -> str:
    """The reply of a saved turn for the model; a failed turn's error saved before #1017 can
    name folders of this computer, so an error reply loses its paths (a model's answer is
    sent as it is)."""
    reply = str(turn.get("reply", ""))
    error = turn.get("error")
    return errtext.no_paths(reply) if isinstance(error, str) and error and reply == error else reply


def _without_folder(text: str, folder: str) -> str:
    """``text`` of an earlier turn for a model not given the project ``folder`` (#1010): a
    model on this machine may have answered with a full path, and that reply would go to a
    hosted model the user switched to. Every spelling of the folder goes (so the name inside
    it stays), then any other absolute path becomes its bare name (:func:`errtext.no_paths`)."""
    path = Path(folder)
    spellings = {folder, path.as_posix(), folder.replace("/", "\\")}
    with contextlib.suppress(OSError, RuntimeError):
        spellings.add(str(path.resolve()))
    bases = sorted((s.rstrip("/\\") for s in spellings if s.rstrip("/\\")), key=len, reverse=True)
    for base in bases:  # with a separator after it first: "<folder>/x.txt" leaves "x.txt"
        text = text.replace(base + "/", "").replace(base + "\\", "")
    for base in bases:
        text = text.replace(base, "")
    return errtext.no_paths(text)


def _without_names(text: str, renamed: Mapping[str, str]) -> str:
    """``text`` with each name of ``renamed`` (:meth:`context.MomentTable.other_names`) that
    does not go on another name (``xday1/a.mp4``) replaced by the name shown instead, in one
    pass, the longest first."""
    if not renamed:
        return text
    keys = sorted(renamed, key=len, reverse=True)
    names = "|".join(f"(?P<n{i}>{re.escape(n)})" for i, n in enumerate(keys))

    def shown(m: re.Match[str]) -> str:  # the spelling as written first, else the name matched
        return renamed.get(m.group()) or renamed[keys[int(str(m.lastgroup)[1:])]]

    # Windows paths differ in case, so the match ignores it
    return re.sub(rf"(?<![\w.-])(?:{names})", shown, text, flags=re.IGNORECASE)


class Prompting:
    """The prompt of :class:`cutpoint.assistant.session.Turn` (it works on the turn's state)."""

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
            listed = [t for t, n in zip(_JSON_TOOLS, TOOL_NAMES, strict=True) if self._offered(n)]
            tools = [
                *listed,
                *([ranking.JSON_TOOL] if self._styles else []),
                *([watch.JSON_TOOL] if self._watch else []),
            ]
        # find_words', look_for's and preview_ranking's rules, then how and when to watch, after
        # the others and measured with them
        late = words.RULES
        if self.project.frame_index:
            late += lookfor.rules(bool(self._watch))
        else:
            late += lookfor.unindexed_rules(self.project.index_state or {})
        after = watch.RULES.replace("{cap}", watch.views(sheets)) if self._watch else ""

        def build(rank_rules: str) -> list[dict[str, Any]]:
            return context.build_messages(
                self.table,
                self.earlier,
                self.text,
                plan,
                json_tools=tools,
                images=bool(self._watch),
                target=self.target,
                late_rules=late + rank_rules + after,
                files=self.project.files,
                folder=self.folder(),
            )

        if not self._styles:
            return build("")
        out = build(ranking.rules(self._styles, self._current))
        floor = int(plan.tokens // 4 * context.CHARS_PER_TOKEN) // (2 if plan.tight else 1)
        if plan.chars > floor:
            return out
        # At the budget's floor (a small context) the brief rules, without the built-in styles'
        # notes, when the full ones would cost the moment table rows: it only grows with more
        brief = build(ranking.brief(self._styles, self._current, names=json_mode))
        return brief if len(brief[1]["content"]) > len(out[1]["content"]) else out

    def _on_this_machine(self) -> bool:
        """Whether nothing sent to the model leaves this machine
        (:func:`~cutpoint.assistant.providers.is_consent_exempt` of its URL: not a hosted API,
        not the subscription proxy, not Ollama on another machine)."""
        if self.provider.local is not True:
            return False
        url = getattr(self.provider, "url", None)
        return url is None or is_consent_exempt(url)

    def folder(self) -> str | None:
        """The project folder's path for the model, only when it runs on this machine
        (:meth:`_on_this_machine`); else None."""
        return self.project.folder if self.project.folder and self._on_this_machine() else None

    def _offered(self, name: str) -> bool:
        """Whether the tool ``name`` is offered: ``look_for`` only with cached frame embeddings."""
        return self.project.frame_index or name != lookfor.NAME

    def _names(self) -> list[str]:
        """The names of the tools of :data:`TOOLS` offered (:meth:`_offered`)."""
        return [n for n in TOOL_NAMES if self._offered(n)]

    def _extra(self) -> list[str]:
        """The per-turn tools offered besides :data:`TOOLS`: ``preview_ranking`` with a style to
        choose, ``view_moment`` for a model that takes images."""
        return [*([ranking.NAME] if self._styles else []), *([watch.NAME] if self._watch else [])]

    def _tools(self, closing: bool = False) -> list[dict[str, Any]]:
        """The tools of a request; the closing round offers only ``propose_edit`` (a list that
        is not empty: a provider wants tools defined next to the earlier calls)."""
        if closing:
            return [t for t in TOOLS if t["function"]["name"] == FINAL_TOOL]
        base = [t for t in TOOLS if self._offered(t["function"]["name"])]
        if self._styles:  # this turn's styles are the enum of its style
            base.append(ranking.tool([n for n, _ in self._styles]))
        return [*base, watch.TOOL] if self._watch else base

    def _schema(self, closing: bool = False) -> dict[str, Any]:
        if not closing and not self._extra() and self._names() == TOOL_NAMES:
            return JSON_SCHEMA
        names = [*self._names(), *self._extra(), "reply"]
        if closing:
            names = [FINAL_TOOL, "reply"]
        props = {**JSON_SCHEMA["properties"], "tool": {"type": "string", "enum": names}}
        return {**JSON_SCHEMA, "properties": props}

    def _fit(self, messages: list[dict[str, Any]], json_mode: bool, *, tight: bool = False) -> bool:
        """Leave out the oldest tool results until the conversation, the tool definitions and
        room for a reply fit the context (``tight``: all but the latest). The system messages,
        the user's messages and the latest result stay. True if a result was left out."""
        limit = (self.provider.context_tokens - context.REPLY_TOKENS) * context.FIT_CHARS_PER_TOKEN
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
