"""Shared helpers of the ``cutpoint ui`` assistant tests: a scripted :class:`FakeProvider`, the
sample moments, ``requests`` doubles (:class:`Wire`, :class:`_Resp`) and the reply builders of the
OpenAI, Responses and Anthropic wire formats.

A plain module (not a test module): the assistant test files import these from here.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from cutpoint.assistant import ops, providers, session
from cutpoint.assistant.providers import Capabilities, Reply, ToolCall

A, B = "/v/a.mp4", "/v/b.mp4"

CANDS = [
    {"id": "c1", "source": A, "start": 10.0, "end": 20.0, "score": 90, "label": "big laugh"},
    {"id": "c2", "source": A, "start": 40.0, "end": 50.0, "score": 70, "label": "jump"},
    {"id": "c3", "source": B, "start": 5.0, "end": 9.0, "score": 80, "label": "crash"},
]

SEQ = [
    {"id": "c2", "source": A, "start": 40.0, "end": 50.0, "label": "jump"},
    {"id": "c3", "source": B, "start": 5.0, "end": 9.0, "label": "crash"},
    {"id": "x9", "source": A, "start": 60.0, "end": 70.0, "label": "kept by hand"},
]

TOOLS_AND_JSON = Capabilities(True, True)


def durations() -> ops.Durations:
    return ops.Durations({A: 100.0, B: 30.0}, lambda _p: None)


class FakeProvider:
    """A model that answers with ``script`` in order (a :class:`Reply` or a callable taking the
    messages) and records every request."""

    name = "fake"
    model = "fake-model"
    local = True
    host = "localhost"
    context_tokens = 8192

    def __init__(self, script: list[Any], caps: Capabilities = TOOLS_AND_JSON) -> None:
        self.script = list(script)
        self.caps = caps
        self.requests: list[dict[str, Any]] = []

    def capabilities(self) -> Capabilities:
        return self.caps

    def chat(self, messages, *, tools=None, schema=None, cancelled=None) -> Reply:
        self.requests.append({"messages": list(messages), "tools": tools, "schema": schema})
        step = self.script.pop(0) if self.script else call("find_moments", query="laugh")
        return step(messages) if callable(step) else step

    def tool_result(self, call: ToolCall, content: str) -> dict[str, Any]:
        return {"role": "tool", "tool_name": call.name, "content": content}


def call(name: str, **args: Any) -> Reply:
    """A reply with one tool call."""
    raw = {"function": {"name": name, "arguments": args}}
    return Reply(
        "", [ToolCall(name, args)], {"role": "assistant", "content": "", "tool_calls": [raw]}
    )


def project(segments=SEQ, transcript: str = "") -> session.Project:
    return session.Project(list(CANDS), list(segments), durations(), transcript, "key0")


def run(provider: FakeProvider, text: str = "tidy up", **kw: Any) -> session.Turn:
    turn = session.Turn(provider, kw.pop("proj", None) or project(), text)
    turn.start()
    assert turn.done.wait(10)
    return turn


class _Resp:
    """A ``requests`` response; ``lines``: the streamed body, read by ``iter_lines``."""

    def __init__(self, body: Any = None, status: int = 200, lines: list | None = None) -> None:
        self.body, self.status_code, self.lines = body, status, lines or []
        self.text, self.reason = json.dumps(body), "x"
        self.closed = False

    def json(self) -> Any:
        return self.body

    def iter_lines(self):
        for line in self.lines:
            yield json.dumps(line).encode()

    def close(self) -> None:
        self.closed = True


def _chunks(content: str = "", calls: list | None = None) -> list[dict]:
    """A streamed /api/chat reply: the text in two pieces, then the tool calls and done."""
    half = len(content) // 2
    return [
        {"message": {"role": "assistant", "content": content[:half]}, "done": False},
        {"message": {"role": "assistant", "content": content[half:]}, "done": False},
        {"message": {"role": "assistant", "content": "", "tool_calls": calls or []}, "done": True},
    ]


KEY = "sk-test-0123456789abcdefSECRET"  # shaped like a real key, so the pattern redacts it too

OTHER = "plain-secret-without-prefix-42"

XKEY = "xai-0123456789abcdefghijSECRET"  # shaped like an xAI key


class NoRing:
    """A keyring without a backend, as keyring's fail backend: every call raises."""

    def _fail(self, *args: str) -> None:
        raise RuntimeError("No recommended backend was available")

    get_password = set_password = delete_password = _fail


class FakeRing:
    """The ``keyring`` module's three calls, in memory."""

    def __init__(self) -> None:
        self.saved: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, user: str) -> str | None:
        return self.saved.get((service, user))

    def set_password(self, service: str, user: str, value: str) -> None:
        self.saved[(service, user)] = value

    def delete_password(self, service: str, user: str) -> None:
        del self.saved[(service, user)]


class Wire:
    """``requests.get``/``post`` of the providers: the answers in order (a response or an
    exception to raise); every request is recorded."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *answers: Any) -> None:
        self.answers = list(answers)
        self.sent: list[dict[str, Any]] = []
        monkeypatch.setattr(
            providers.requests, "post", lambda url, **kw: self._next("POST", url, kw)
        )
        monkeypatch.setattr(providers.requests, "get", lambda url, **kw: self._next("GET", url, kw))

    def _next(self, method: str, url: str, kw: dict[str, Any]) -> Any:
        self.sent.append(
            {
                "method": method,
                "url": url,
                "json": copy.deepcopy(kw.get("json")),
                "headers": dict(kw.get("headers") or {}),
                "timeout": kw.get("timeout"),
                "allow_redirects": kw.get("allow_redirects", True),  # requests' default
            }
        )
        if not self.answers:
            raise AssertionError(f"unexpected request {method} {url}")
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def oa_reply(content: str | None = "", calls: list | None = None) -> _Resp:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if calls is not None:
        msg["tool_calls"] = calls
    return _Resp({"choices": [{"message": msg, "finish_reason": "stop"}]})


def oa_call(cid: str | None, name: str, args: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {"type": "function", "function": {"name": name, "arguments": args}}
    if cid is not None:
        raw["id"] = cid
    return raw


def rs_reply(*items: dict[str, Any], status: str = "completed") -> _Resp:
    """A ``/v1/responses`` answer with these output items."""
    return _Resp({"object": "response", "status": status, "output": list(items)})


def rs_text(text: str, phase: str | None = None) -> dict[str, Any]:
    content = [{"type": "output_text", "text": text, "annotations": []}]
    item = {"type": "message", "role": "assistant", "status": "completed", "content": content}
    return item if phase is None else {**item, "phase": phase}


def rs_call(cid: str, name: str, args: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{cid}",
        "call_id": cid,
        "name": name,
        "arguments": args,
    }


def an_reply(*blocks: dict[str, Any]) -> _Resp:
    return _Resp({"content": list(blocks), "stop_reason": "end_turn"})


def tool_use(cid: str, name: str, **args: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": cid, "name": name, "input": args}
