"""Bring-your-own AI for the ``cutpoint ui`` assistant (#346): the OpenAI-compatible and Anthropic
adapters, local detection, API keys (lookup, keyring, never leaked) and what a hosted model sees.

HTTP is stubbed (:class:`Wire`); the keyring is a fake; no request leaves the machine.
"""

from __future__ import annotations

import copy
import io
import json
import logging
import os
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest
import requests

from conftest import set_config
from cutpoint import config
from cutpoint.assistant import keys, ops, providers, session
from cutpoint.assistant.providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    ToolsUnsupportedError,
)
from test_assistant import _Resp

KEY = "sk-test-0123456789abcdefSECRET"  # shaped like a real key, so the pattern redacts it too
OTHER = "plain-secret-without-prefix-42"


@pytest.fixture(autouse=True)
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real key variable or keyring reaches a test."""
    for name in keys.ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(keys, "_keyring", NoRing)
    monkeypatch.setattr(keys, "_SEEN", set())
    monkeypatch.setattr(keys, "_CAPTURED", {})


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


@pytest.fixture
def ring(monkeypatch: pytest.MonkeyPatch) -> FakeRing:
    fake = FakeRing()
    monkeypatch.setattr(keys, "_keyring", lambda: fake)
    return fake


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


# -- local or hosted -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://127.0.0.1:11434", True),
        ("localhost:1234", True),
        ("http://LOCALHOST:8080/v1", True),
        ("http://box.localhost:8080", True),
        ("http://127.5.6.7", True),
        ("http://[::1]:11434", True),
        ("http://10.10.10.235:11434", False),
        ("http://192.168.1.4:1234/v1", False),
        ("http://0.0.0.0:11434", False),
        ("https://api.anthropic.com", False),
        ("http://localhost.evil.example", False),
        (r"http://evil.example\@127.0.0.1", False),  # requests goes to evil.example
        (r"http://evil.example\@localhost:1234/v1", False),
        ("", False),
    ],
)
def test_local_means_this_machine_only(url: str, local: bool) -> None:
    assert providers.is_local(url) is local


def test_a_remote_ollama_is_hosted_and_needs_consent() -> None:
    """PR 1 labelled every Ollama "Local", even one on another machine."""
    here = OllamaProvider("http://127.0.0.1:11434", "m", 8192)
    there = OllamaProvider("10.10.10.235:11434", "m", 8192)
    assert here.local and providers.consent_key(here) is None
    assert providers.describe(here)["label"] == "Ollama · m · this machine"
    assert not there.local and there.host == "10.10.10.235"
    assert providers.describe(there)["label"] == "Ollama · m · 10.10.10.235"
    assert providers.consent_key(there) == "ollama@10.10.10.235"


# -- OpenAI-compatible -----------------------------------------------------------------------


def test_a_compatible_server_keeps_chat_completions_with_tools(monkeypatch) -> None:
    """#536: only api.openai.com moved to /v1/responses; OpenAI-compatible servers (OpenRouter
    here, LM Studio, vLLM, Ollama's /v1) still get /v1/chat/completions."""
    calls = [
        oa_call("c1", "find_moments", '{"query": "laugh"}'),
        oa_call(None, "get_moment", {"id": "m1"}),  # no id, arguments as an object
        oa_call("c3", "get_moment", "{bad"),
    ]
    wire = Wire(monkeypatch, _Resp({"data": [{"id": "gpt-x"}]}), oa_reply(None, calls))
    p = OpenAIProvider("https://openrouter.ai/api/v1", "gpt-x", 0, KEY)
    assert (p.local, p.host, p.context_tokens) == (False, "openrouter.ai", 32768)
    assert p.capabilities() == providers.Capabilities(True, True)
    reply = p.chat([{"role": "user", "content": "hi"}], tools=session.TOOLS)

    check, ask = wire.sent
    assert check["method"] == "GET" and check["url"] == "https://openrouter.ai/api/v1/models"
    assert ask["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert ask["headers"] == {"Authorization": f"Bearer {KEY}"}
    assert ask["json"]["tools"] == session.TOOLS and ask["json"]["model"] == "gpt-x"
    assert "temperature" not in ask["json"]  # the server's default
    assert ask["timeout"] == (providers.CONNECT_TIMEOUT_S, providers.HOSTED_TIMEOUT_S)
    assert check["allow_redirects"] is False and ask["allow_redirects"] is False
    got = [(c.name, c.arguments, c.error, c.id) for c in reply.tool_calls]
    assert got == [
        ("find_moments", {"query": "laugh"}, None, "c1"),
        ("get_moment", {"id": "m1"}, None, "call_2"),
        ("get_moment", {}, "arguments are not valid JSON", "c3"),
    ]
    # the assistant message goes back as OpenAI wants it: ids, arguments as JSON text
    assert [c["id"] for c in reply.message["tool_calls"]] == ["c1", "call_2", "c3"]
    assert reply.message["tool_calls"][1]["function"]["arguments"] == '{"id": "m1"}'
    assert p.tool_result(reply.tool_calls[1], "r") == {
        "role": "tool",
        "tool_call_id": "call_2",
        "content": "r",
    }
    assert KEY not in repr(p) and KEY not in json.dumps(providers.describe(p))


def test_openai_makes_tool_calls_over_the_responses_api(monkeypatch) -> None:
    """#536: Sol, Terra and Luna make tool calls only over /v1/responses, so api.openai.com is
    asked there: system text as instructions, flat function tools, and the reply's items
    (reasoning included) sent back before the function_call_output that answers each call."""
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc"}
    wire = Wire(
        monkeypatch,
        _Resp({"data": [{"id": "gpt-6.1-sol"}]}),
        rs_reply(reasoning, rs_text("Looking."), rs_call("call_a", "find_moments", '{"q": 1}')),
        rs_reply(
            rs_text("Checking.", "commentary"),
            rs_text("Done.", "final_answer"),
            rs_text("Bye.", "final_answer"),
        ),
    )
    p = OpenAIProvider("", "gpt-6.1-sol", 0, KEY)
    assert p.capabilities() == providers.Capabilities(True, True)
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "system", "content": "context"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "assistant", "content": ""},  # an earlier turn without a reply: left out
        {"role": "user", "content": "find x"},
    ]
    reply = p.chat(messages, tools=session.TOOLS)
    ask = wire.sent[1]
    assert ask["url"] == "https://api.openai.com/v1/responses"
    assert ask["headers"] == {"Authorization": f"Bearer {KEY}"} and not ask["allow_redirects"]
    body = ask["json"]
    assert body["instructions"] == "rules\n\ncontext"
    assert body["store"] is False  # OpenAI keeps no copy; reasoning replays as encrypted items
    assert body["input"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "phase": "final_answer"},  # an earlier reply
        {"role": "user", "content": "find x"},
    ]
    fn = session.TOOLS[0]["function"]
    assert body["tools"][0] == {
        "type": "function",
        "name": fn["name"],
        "description": fn["description"],
        "parameters": fn["parameters"],
        "strict": False,  # as Chat Completions: Responses would otherwise make it strict
    }
    assert "messages" not in body and "temperature" not in body
    assert reply.text == "Looking."
    assert [(c.name, c.arguments, c.id) for c in reply.tool_calls] == [
        ("find_moments", {"q": 1}, "call_a")
    ]

    convo = [*messages, reply.message, p.tool_result(reply.tool_calls[0], "found")]
    # the final answer's message items, one paragraph each, without the commentary
    assert p.chat(convo, tools=session.TOOLS).text == "Done.\n\nBye."
    assert wire.sent[2]["json"]["input"][3:] == [
        reasoning,
        rs_text("Looking."),
        rs_call("call_a", "find_moments", '{"q": 1}'),
        {"type": "function_call_output", "call_id": "call_a", "output": "found"},
    ]


@pytest.mark.parametrize(
    ("answer", "want"),
    [
        (
            _Resp(
                {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [rs_call("call_a", "propose_edit", '{"summary": "Op')],
                }
            ),
            r"cut off \(max_output_tokens\)",
        ),
        (
            rs_reply(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "I can't help with that."}],
                }
            ),
            "declined",
        ),
    ],
)
def test_a_responses_reply_cut_off_or_refused_runs_no_call(monkeypatch, answer, want) -> None:
    """#536: a /v1/responses answer cut off midway (its call's arguments are partial) or a
    refusal alone is an error, never a tool call to run or an empty reply."""
    Wire(monkeypatch, answer)
    with pytest.raises(ProviderError, match=want):
        OpenAIProvider("", "m", 0, KEY).chat(HI, tools=session.TOOLS)


@pytest.mark.parametrize(
    ("url", "want"),
    [
        ("", "https://api.openai.com/v1"),
        ("http://localhost:1234", "http://localhost:1234/v1"),
        ("localhost:8080/", "http://localhost:8080/v1"),
        ("https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1"),
    ],
)
def test_openai_base_url_gets_v1_only_when_it_has_no_path(url: str, want: str) -> None:
    assert OpenAIProvider(url, "m", 0, KEY).url == want


def test_a_local_openai_server_needs_no_key_and_serves_its_only_model(monkeypatch) -> None:
    wire = Wire(monkeypatch, _Resp({"data": [{"id": "qwen.gguf"}]}), oa_reply("Hello"))
    p = OpenAIProvider("http://127.0.0.1:8080", "", 0, None)
    assert p.local and p.context_tokens == providers.LOCAL_CONTEXT_TOKENS
    assert p.model == providers.AUTO_SERVER_MODEL
    p.capabilities()
    reply = p.chat([{"role": "user", "content": "hi"}])
    assert reply.text == "Hello" and not reply.tool_calls
    label = providers.describe(p)["label"]
    assert p.model == "qwen.gguf" and label == "OpenAI-compatible · qwen.gguf · this machine"
    assert all("Authorization" not in s["headers"] for s in wire.sent)
    assert wire.sent[1]["timeout"][1] == providers.CHAT_TIMEOUT_S


def test_an_openai_server_with_several_models_asks_for_one(monkeypatch) -> None:
    Wire(monkeypatch, _Resp({"data": [{"id": "a"}, {"id": "b"}]}))
    with pytest.raises(ProviderError, match=r"Choose a model .* in Settings; it lists a, b"):
        OpenAIProvider("http://localhost:1234", "", 0, None).capabilities()


def test_openai_json_mode_constrains_the_reply_to_the_schema(monkeypatch) -> None:
    wire = Wire(
        monkeypatch,
        oa_reply('{"tool": "reply", "text": "ok"}'),
        rs_reply(rs_text('{"tool": "reply", "text": "ok"}')),
    )
    hi = [{"role": "user", "content": "hi"}]
    reply = OpenAIProvider("http://127.0.0.1:1234", "m", 0, None).chat(
        hi, schema=session.JSON_SCHEMA
    )
    assert wire.sent[0]["json"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "cutpoint_reply", "schema": session.JSON_SCHEMA},
    }
    assert "tools" not in wire.sent[0]["json"] and reply.text.startswith("{")
    # over the Responses API (api.openai.com) the schema is text.format
    reply = OpenAIProvider("", "gpt-x", 0, KEY).chat(hi, schema=session.JSON_SCHEMA)
    assert wire.sent[1]["json"]["text"] == {
        "format": {"type": "json_schema", "name": "cutpoint_reply", "schema": session.JSON_SCHEMA}
    }
    assert "tools" not in wire.sent[1]["json"] and reply.text.startswith("{")


@pytest.mark.parametrize(
    ("url", "why"),
    [
        (
            "http://localhost:8000",
            '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set',
        ),
        ("http://localhost:8080", "tools param requires --jinja flag"),
        ("http://localhost:11434", "registry.ollama.ai/library/gemma:2b does not support tools"),
        ("", "Function tools are not supported with this model."),  # /v1/responses (#536)
    ],
)
def test_a_server_refusing_tools_is_asked_again_in_json_mode(monkeypatch, url, why) -> None:
    wire = Wire(monkeypatch, _Resp({"error": {"message": why}}, 400))
    with pytest.raises(ToolsUnsupportedError):
        OpenAIProvider(url, "m", 0, KEY).chat([], tools=session.TOOLS)
    endpoint = "/v1/chat/completions" if url else "/v1/responses"
    assert wire.sent[0]["url"].endswith(endpoint)


# -- Anthropic ---------------------------------------------------------------------------------


def test_anthropic_messages_api_with_tool_use(monkeypatch) -> None:
    wire = Wire(
        monkeypatch,
        _Resp({"id": providers.DEFAULT_ANTHROPIC_MODEL, "type": "model"}),
        an_reply({"type": "text", "text": "Looking."}, tool_use("tu1", "find_moments", query="x")),
        an_reply({"type": "text", "text": "Done."}),
    )
    p = AnthropicProvider("", "", 0, KEY)
    assert p.model == "claude-sonnet-5-5" and p.url == "https://api.anthropic.com"
    assert p.capabilities() == providers.Capabilities(True, False)
    check = wire.sent[0]
    assert check["url"] == "https://api.anthropic.com/v1/models/claude-sonnet-5-5"
    assert check["headers"] == {"anthropic-version": "2023-06-01", "x-api-key": KEY}

    messages = [
        {"role": "system", "content": "rules"},
        {"role": "system", "content": "context"},
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": ""},  # an earlier turn without a reply: left out
        {"role": "user", "content": "find x"},
    ]
    reply = p.chat(messages, tools=session.TOOLS)
    ask = wire.sent[1]
    assert check["allow_redirects"] is False and ask["allow_redirects"] is False
    assert ask["url"] == "https://api.anthropic.com/v1/messages"
    assert ask["headers"]["x-api-key"] == KEY
    body = ask["json"]
    assert body["system"] == "rules\n\ncontext" and body["max_tokens"] == providers.MAX_TOKENS
    assert "temperature" not in body  # newer models answer 400: `temperature` is deprecated
    assert body["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "earlier"}, {"type": "text", "text": "find x"}],
        }
    ]
    first = body["tools"][0]
    assert first["name"] == "find_moments"
    assert first["input_schema"] == session.TOOLS[0]["function"]["parameters"]
    assert reply.text == "Looking."
    assert [(c.name, c.arguments, c.id) for c in reply.tool_calls] == [
        ("find_moments", {"query": "x"}, "tu1")
    ]

    # The reply and both tool results go back: the results in one user message after it.
    second = providers.ToolCall("get_moment", {"id": "m1"}, None, "tu2")
    convo = [
        *messages,
        reply.message,
        p.tool_result(reply.tool_calls[0], "found"),
        p.tool_result(second, "moment"),
    ]
    assert p.chat(convo, tools=session.TOOLS).text == "Done."
    wire_msgs = wire.sent[2]["json"]["messages"]
    assert wire_msgs[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Looking."},
            tool_use("tu1", "find_moments", query="x"),
        ],
    }
    assert wire_msgs[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu1", "content": "found"},
            {"type": "tool_result", "tool_use_id": "tu2", "content": "moment"},
        ],
    }


def test_opus_is_selectable_and_the_key_comes_from_the_environment(home, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    set_config(home, assistant={"provider": "anthropic", "model": "claude-opus-5-5"})
    p = providers.make()
    assert isinstance(p, AnthropicProvider) and p.model == "claude-opus-5-5"
    assert "claude-opus-5-5" in providers.ANTHROPIC_MODELS
    assert p._key == KEY and KEY not in repr(p)
    assert p.context_tokens == providers.HOSTED_CONTEXT_TOKENS
    set_config(home, assistant={"provider": "anthropic", "context_tokens": 100000})
    assert providers.make().context_tokens == 100000


# -- errors, timeouts and the missing key ------------------------------------------------------


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
@pytest.mark.parametrize(
    ("answer", "want"),
    [
        (
            _Resp({"error": {"message": f"Incorrect API key provided: {KEY}"}}, 401),
            "rejected the API key",
        ),
        (
            _Resp({"error": {"type": "not_found_error", "message": "model: nope"}}, 404),
            "has no model",
        ),
        (_Resp({"error": {"message": "slow down"}}, 429), "rate limiting or out of credit"),
        (
            _Resp({"type": "error", "error": {"message": "Overloaded"}}, 529),
            "answered 529: Overloaded",
        ),
        (requests.Timeout("read timed out"), "did not answer within 180 s"),
        (requests.ConnectionError(f"refused, header x-api-key={KEY}"), "is not reachable"),
    ],
)
def test_hosted_errors_say_what_failed_never_the_key(monkeypatch, kind, answer, want) -> None:
    Wire(monkeypatch, answer)
    p = (OpenAIProvider if kind == "openai" else AnthropicProvider)("", "m", 0, KEY)
    with pytest.raises(ProviderError) as err:
        p.chat([{"role": "user", "content": "hi"}], tools=session.TOOLS)
    assert want in str(err.value) and KEY not in str(err.value)


def test_an_unreadable_answer_is_an_error(monkeypatch) -> None:
    Wire(monkeypatch, _Resp(["not", "an", "object"]), _Resp({"content": "text"}))
    with pytest.raises(ProviderError, match="unreadable"):
        OpenAIProvider("", "m", 0, KEY).chat([])
    with pytest.raises(ProviderError, match="no content"):
        AnthropicProvider("", "m", 0, KEY).chat([])


@pytest.mark.parametrize(
    ("cls", "env"), [(OpenAIProvider, "OPENAI_API_KEY"), (AnthropicProvider, "ANTHROPIC_API_KEY")]
)
def test_a_hosted_provider_without_a_key_says_how_to_add_one(monkeypatch, cls, env) -> None:
    wire = Wire(monkeypatch)  # nothing is sent
    p = cls("", "m", 0, None)
    for ask in (p.capabilities, lambda: p.chat([{"role": "user", "content": "hi"}])):
        with pytest.raises(ProviderError, match="No API key") as err:
            ask()
        assert env in str(err.value) and "CUTPOINT_ASSISTANT_API_KEY" not in str(err.value)
        assert "keyring" in str(err.value)
    assert wire.sent == []
    assert "No API key" in (providers.check(p) or "")


def test_a_cancelled_hosted_turn_stops_after_the_reply(monkeypatch) -> None:
    Wire(monkeypatch, oa_reply("late"))
    stop = {"now": False}

    def cancelled() -> bool:
        was, stop["now"] = stop["now"], True  # false before the request, true after
        return was

    with pytest.raises(providers.CancelledError):
        OpenAIProvider("", "m", 0, KEY).chat([], cancelled=cancelled)


# -- whole turns over each adapter, and what a hosted model is sent ----------------------------

DIR_A = "/home/someone/private footage"
DIR_B = "D:\\Clips\\secret trip"
SRC_A, SRC_B = f"{DIR_A}/a.mp4", f"{DIR_B}\\b.mp4"
CANDS = [
    {
        "id": "c1",
        "source": SRC_A,
        "start": 10.0,
        "end": 20.0,
        "score": 90,
        "label": "big laugh",
        "transcript": "that was a big laugh",
        "signals": {"loudness": 0.8},
    },
    {"id": "c2", "source": SRC_B, "start": 4.0, "end": 9.0, "score": 70, "label": "crash"},
]
SEQ = [{"id": "c2", "source": SRC_B, "start": 4.0, "end": 9.0, "label": "crash"}]
TRANSCRIPT = "[a.mp4 00:00:10.0-00:00:12.0] that was a big laugh\n[b.mp4 00:00:05.0-00:00:06.0] ow"


def hosted_project() -> session.Project:
    durations = ops.Durations({SRC_A: 100.0, SRC_B: 30.0}, lambda _p: None)
    return session.Project(list(CANDS), list(SEQ), durations, TRANSCRIPT, "key0")


def run_turn(provider: providers.Provider, text: str = "open on the laugh") -> session.Turn:
    turn = session.Turn(provider, hosted_project(), text, [{"user": "hi", "reply": "hello"}])
    turn.start()
    assert turn.done.wait(10)
    return turn


TOOL_STEPS = [
    ("find_moments", {"query": "laugh"}),
    ("get_moment", {"id": "m1"}),
    ("get_transcript", {"id": "m2", "pad": 5}),
    ("show_moments", {"ids": ["m1"]}),
    ("propose_edit", {"summary": "Open on the laugh", "ops": [{"op": "add", "id": "m1", "at": 1}]}),
]


def scripted(kind: str) -> list[_Resp]:
    """A capability check, then one reply per tool step, over ``kind``'s wire format."""
    if kind == "openai":
        check = _Resp({"data": [{"id": "gpt-x"}]})
        steps = [
            rs_reply(rs_call(f"c{n}", name, json.dumps(a)))
            for n, (name, a) in enumerate(TOOL_STEPS)
        ]
    else:
        check = _Resp({"id": "claude-sonnet-5-5"})
        steps = [an_reply(tool_use(f"tu{n}", name, **a)) for n, (name, a) in enumerate(TOOL_STEPS)]
    return [check, *steps]


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
def test_a_turn_runs_over_each_hosted_adapter_and_sends_file_names_never_paths(
    monkeypatch, kind: str
) -> None:
    wire = Wire(monkeypatch, *scripted(kind))
    p = (
        OpenAIProvider("", "gpt-x", 0, KEY)
        if kind == "openai"
        else AnthropicProvider("", "", 0, KEY)
    )
    turn = run_turn(p)
    assert turn.status == "done", turn.error
    prop = next(e for e in turn.events if e["kind"] == "proposal")
    assert [s["id"] for s in prop["after"]] == ["c1", "c2"]

    sent = json.dumps([s["json"] for s in wire.sent], ensure_ascii=False)
    for secret in (DIR_A, "someone", "private footage", "secret trip", "Clips", "D:"):
        assert secret not in sent, secret
    for shown in ("a.mp4", "b.mp4", "big laugh", "that was a big laugh", "ow"):
        assert shown in sent, shown
    # each tool result answers its call by id
    if kind == "openai":  # over /v1/responses (#536)
        last = wire.sent[-1]["json"]["input"]
        outputs = [m["call_id"] for m in last if m.get("type") == "function_call_output"]
        assert outputs == ["c0", "c1", "c2", "c3"]
    else:
        last = wire.sent[-1]["json"]["messages"]
        results = [
            b["tool_use_id"] for m in last for b in m["content"] if b["type"] == "tool_result"
        ]
        assert results == ["tu0", "tu1", "tu2", "tu3"]
    assert KEY not in sent  # the key only ever goes in a header


# -- keys ----------------------------------------------------------------------------------


def test_a_key_comes_from_the_provider_variable_then_the_keyring_then_the_cutpoint_one(
    monkeypatch, ring: FakeRing
) -> None:
    an, oa = "api.anthropic.com", "api.openai.com"
    assert keys.lookup("anthropic", an) == (None, None) and keys.status("openai", oa) is None
    ring.saved[("cutpoint", "anthropic")] = "ring-key-123456"
    assert keys.lookup("anthropic", an) == ("ring-key-123456", "keyring")
    assert keys.lookup("anthropic", "proxy.example.com") == (None, None)  # another host's entry
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-1234567")
    assert keys.lookup("anthropic", an) == ("env-key-1234567", "env")
    assert keys.lookup("anthropic", "API.Anthropic.com") == ("env-key-1234567", "env")
    assert keys.lookup("anthropic", "proxy.example.com") == (None, None)  # official host only
    assert keys.lookup("openai", oa) == (None, None)  # another provider's variable is not used
    monkeypatch.setenv("CUTPOINT_ASSISTANT_API_KEY", "cp-key-12345678")
    assert keys.lookup("anthropic", an) == ("env-key-1234567", "env")  # the own one wins (#504)
    assert keys.lookup("openai", oa) == (None, None)  # never for an official host (#504)
    assert keys.lookup("openai", "openrouter.ai") == ("cp-key-12345678", "env")
    ring.saved[("cutpoint", "openai@openrouter.ai")] = "ring-key-654321"
    assert keys.lookup("openai", "openrouter.ai") == ("ring-key-654321", "keyring")


def test_the_own_key_beats_the_shared_key_on_the_wire(home, monkeypatch) -> None:
    """#504: with both CUTPOINT_ASSISTANT_API_KEY (an OpenRouter key) and ANTHROPIC_API_KEY set,
    api.anthropic.com gets only the Anthropic key."""
    monkeypatch.setenv("CUTPOINT_ASSISTANT_API_KEY", OTHER)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    set_config(home, assistant={"provider": "anthropic", "model": "m"})
    wire = Wire(monkeypatch, an_reply({"type": "text", "text": "ok"}))
    providers.make().chat(HI)
    sent = wire.sent[0]
    assert sent["url"].startswith("https://api.anthropic.com/")
    assert sent["headers"]["x-api-key"] == KEY and OTHER not in json.dumps(sent["headers"])


@pytest.mark.parametrize(
    ("name", "url"),
    [
        ("anthropic", ""),
        ("xai", ""),
        ("openai", "https://api.x.ai/v1"),  # another provider's official host: never either
        ("openai", "https://api.openai.com./v1"),  # the same host with a trailing dot
    ],
)
def test_the_shared_key_never_reaches_an_official_host(home, monkeypatch, name, url) -> None:
    """#504: CUTPOINT_ASSISTANT_API_KEY is for custom and self-hosted URLs only."""
    monkeypatch.setenv("CUTPOINT_ASSISTANT_API_KEY", KEY)
    set_config(home, assistant={"provider": name, "model": "m", "url": url})
    wire = Wire(monkeypatch)  # any request fails the test
    p = providers.make()
    assert not p.has_key
    with pytest.raises(ProviderError, match="No API key"):
        p.chat(HI)
    assert "No API key" in (providers.check(p) or "")
    assert wire.sent == []


def test_keys_are_saved_and_cleared_in_the_keyring_only(ring: FakeRing) -> None:
    oa = "api.openai.com"
    keys.save("openai", f"  {OTHER} ", oa)
    assert ring.saved == {("cutpoint", "openai"): OTHER}
    assert keys.status("openai", oa) == "keyring"
    assert keys.status("openai", "openrouter.ai") is None
    keys.clear("openai", oa)
    keys.clear("openai", oa)  # nothing left: fine
    assert ring.saved == {} and keys.status("openai", oa) is None
    for bad in ("", "has space", "x" * (keys.MAX_KEY + 1), 42):
        with pytest.raises(ValueError, match="printable"):
            keys.save("openai", bad, oa)
    with pytest.raises(ValueError, match="provider"):
        keys.save("ollama", OTHER, "localhost")


def test_a_key_for_another_host_is_saved_under_provider_at_host(ring: FakeRing) -> None:
    keys.save("openai", OTHER, "OpenRouter.ai")
    assert ring.saved == {("cutpoint", "openai@openrouter.ai"): OTHER}
    assert keys.lookup("openai", "openrouter.ai") == (OTHER, "keyring")
    assert keys.lookup("openai", "api.openai.com") == (None, None)
    keys.clear("openai", "openrouter.ai")
    assert ring.saved == {}


def test_saving_a_key_without_a_keyring_backend_says_how_to_set_one() -> None:
    assert keys.lookup("openai", "api.openai.com") == (None, None)  # a failed read: no key
    with pytest.raises(keys.KeyringError, match=r"did not save the key.*OPENAI_API_KEY") as err:
        keys.save("openai", OTHER, "api.openai.com")
    assert OTHER not in str(err.value)
    with pytest.raises(keys.KeyringError, match="did not remove the key"):
        keys.clear("anthropic", "api.anthropic.com")


def test_child_processes_get_no_key_variable(monkeypatch) -> None:
    env = {"PATH": "/bin", "ANTHROPIC_API_KEY": "a", "openai_api_key": "b", "CUTPOINT_HOME": "h"}
    env["CUTPOINT_ASSISTANT_API_KEY"] = "c"
    assert keys.child_env(env) == {"PATH": "/bin", "CUTPOINT_HOME": "h"}
    monkeypatch.setenv("OPENAI_API_KEY", OTHER)
    assert "OPENAI_API_KEY" not in keys.child_env()


def test_capture_takes_the_variables_out_of_the_environment_and_still_finds_them(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", OTHER)
    keys.capture()
    try:
        assert "OPENAI_API_KEY" not in os.environ
        assert keys.lookup("openai", "api.openai.com") == (OTHER, "env")
        assert keys.redact(f"x {OTHER} y") == "x [redacted] y"
    finally:
        keys._CAPTURED.clear()


def test_redact_removes_read_keys_and_key_shaped_text(monkeypatch) -> None:
    monkeypatch.setenv("CUTPOINT_ASSISTANT_API_KEY", OTHER)
    assert keys.redact(f"bad key {OTHER}!") == "bad key [redacted]!"
    assert keys.redact("got sk-ant-api03-abcdefghijklmnopqrstuv") == "got [redacted]"
    assert keys.redact("short sk-1 stays") == "short sk-1 stays"


def test_log_records_never_carry_a_key(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", OTHER)
    keys.lookup("anthropic", "api.anthropic.com")
    out = io.StringIO()
    handler = logging.StreamHandler(out)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        keys.redact_logs()
        keys.redact_logs()  # once per handler
        assert handler.filters.count(keys._FILTER) == 1
        log = logging.getLogger("cutpoint.test")
        log.error("asked with %s and %s", OTHER, KEY)
        try:
            raise RuntimeError(f"header x-api-key: {OTHER}")
        except RuntimeError:
            log.exception("failed")
    finally:
        root.removeHandler(handler)
    text = out.getvalue()
    assert "asked with [redacted] and [redacted]" in text and "failed" in text
    assert "RuntimeError: header x-api-key: [redacted]" in text
    assert OTHER not in text and KEY not in text


def test_config_never_holds_a_key(home, monkeypatch) -> None:
    """There is no key setting: config.json's assistant section has none to read."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", OTHER)
    set_config(home, assistant={"provider": "anthropic"})
    settings = config.assistant_settings()
    assert OTHER not in repr(settings) and OTHER not in repr(config.get_config())


HI = [{"role": "user", "content": "hi"}]


def sent_key(wire: Wire) -> bool:
    return any(KEY in json.dumps(s["headers"]) for s in wire.sent)


@pytest.mark.parametrize(
    "url", ["http://192.168.1.4:1234/v1", "https://openrouter.ai/api/v1", "http://127.0.0.1:8080"]
)
def test_the_openai_variable_goes_only_to_api_openai_com(home, monkeypatch, url) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    set_config(home, assistant={"provider": "openai", "model": "m", "url": url})
    wire = Wire(monkeypatch, oa_reply("ok"))
    p = providers.make()
    if p.local:
        p.chat(HI)  # a server on this machine needs no key: asked without one
        assert not sent_key(wire)
    else:
        with pytest.raises(ProviderError, match="No API key"):
            p.chat(HI)
        assert wire.sent == []
        assert "OPENAI_API_KEY" not in (providers.check(p) or "")  # nor offered as the fix


def test_the_openai_variable_goes_to_the_official_host(home, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    set_config(home, assistant={"provider": "openai", "model": "m"})
    wire = Wire(monkeypatch, rs_reply(rs_text("ok")))
    providers.make().chat(HI)
    assert wire.sent[0]["url"].startswith("https://api.openai.com/") and sent_key(wire)


@pytest.mark.parametrize(
    ("url", "sent"),
    [("https://openrouter.ai/api/v1", True), ("http://127.0.0.1:8080", True)],
)
def test_the_cutpoint_variable_goes_to_any_host(home, monkeypatch, url, sent) -> None:
    monkeypatch.setenv("CUTPOINT_ASSISTANT_API_KEY", KEY)
    set_config(home, assistant={"provider": "openai", "model": "m", "url": url})
    wire = Wire(monkeypatch, oa_reply("ok"))
    providers.make().chat(HI)
    assert sent_key(wire) is sent


def test_a_key_is_never_sent_over_http_to_another_machine(home, monkeypatch) -> None:
    monkeypatch.setenv("CUTPOINT_ASSISTANT_API_KEY", KEY)
    for name in ("openai", "anthropic"):
        set_config(
            home, assistant={"provider": name, "model": "m", "url": "http://192.168.1.4:1234/v1"}
        )
        wire = Wire(monkeypatch)  # nothing is sent
        p = providers.make()
        with pytest.raises(ProviderError, match=r"use an https:// base URL") as err:
            p.chat(HI)
        assert "192.168.1.4" in str(err.value) and KEY not in str(err.value)
        assert wire.sent == []


def test_a_keyring_key_for_another_host_is_used_only_there(home, monkeypatch, ring) -> None:
    ring.saved[("cutpoint", "openai@openrouter.ai")] = KEY
    set_config(
        home, assistant={"provider": "openai", "model": "m", "url": "https://openrouter.ai/api/v1"}
    )
    wire = Wire(monkeypatch, oa_reply("ok"))
    providers.make().chat(HI)
    assert sent_key(wire)
    set_config(home, assistant={"provider": "openai", "model": "m"})
    with pytest.raises(ProviderError, match="No API key"):
        providers.make().chat(HI)


@pytest.mark.parametrize(
    ("stop", "why"),
    [
        ("max_tokens", r"cut off at 4096 tokens; try a simpler request"),
        ("refusal", r"declined this request"),
    ],
)
def test_an_anthropic_reply_cut_off_or_refused_is_an_error(monkeypatch, stop, why) -> None:
    reply = _Resp({"content": [tool_use("t1", "get_moment", id="m1")], "stop_reason": stop})
    Wire(monkeypatch, reply)
    with pytest.raises(ProviderError, match=why):
        AnthropicProvider("", "m", 0, KEY).chat(HI)


def test_messages_never_show_the_password_of_a_url(monkeypatch) -> None:
    assert providers.public("at http://user:pw@box:11434/v1 and https://a@b.example/x") == (
        "at http://box:11434/v1 and https://b.example/x"
    )
    assert providers.public("mail me@example.com") == "mail me@example.com"  # not a URL
    Wire(monkeypatch, requests.ConnectionError("refused"), requests.ConnectionError("refused"))
    p = OllamaProvider("http://user:pw@box:11434", "m", 0)
    for ask in (p.capabilities, lambda: p.chat(HI)):
        with pytest.raises(ProviderError, match="not reachable") as err:
            ask()
        assert "user" not in str(err.value) and "pw" not in str(err.value)
        assert "box:11434" in str(err.value)


# -- spawning: jobs and dialogs ------------------------------------------------------------


def test_jobs_and_dialogs_start_without_the_key_variables(monkeypatch) -> None:
    from cutpoint import ui  # noqa: PLC0415

    for name in keys.ENV_VARS:
        monkeypatch.setenv(name, OTHER)
    seen: list[dict] = []

    def popen(args, **kwargs):
        seen.append({"args": list(args), "env": kwargs.get("env")})
        return SimpleNamespace(pid=1, communicate=lambda: ("", ""), returncode=0, wait=lambda: 0)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(ui, "_join_tree_job", lambda _proc: None)
    ui.spawn(["x", "y"])
    job = seen[-1]
    assert job["env"]["CUTPOINT_PROGRESS"] == "1"
    assert not set(job["env"]) & set(keys.ENV_VARS)
    assert OTHER not in json.dumps(job)  # nor in the arguments
    monkeypatch.setattr(ui, "path_picker", lambda: lambda kind, start: (["dialog"], None))
    ui.Picker().ask("file", None)
    assert not set(seen[-1]["env"]) & set(keys.ENV_VARS)


# -- URLs, redirects and what the page is told ------------------------------------------------

EVIL_URLS = [r"https://evil.example\@api.openai.com/v1", r"http://evil.example\@127.0.0.1:1234/v1"]


@pytest.mark.parametrize("url", EVIL_URLS)
@pytest.mark.parametrize("name", ["openai", "anthropic"])
def test_a_hand_edited_url_with_an_unclear_host_sends_nothing(home, monkeypatch, url, name) -> None:
    """urlsplit reads api.openai.com / 127.0.0.1; requests would go to evil.example."""
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    assert providers.host_of(url) == "evil.example" and not providers.unambiguous(url)
    set_config(home, assistant={"provider": name, "model": "m", "url": url})
    wire = Wire(monkeypatch)  # nothing is sent
    with pytest.raises(providers.AmbiguousURLError, match="backslash") as err:
        providers.make().chat(HI)
    assert isinstance(err.value, ValueError) and KEY not in str(err.value)
    assert wire.sent == []


def test_hosts_are_read_as_requests_reads_them() -> None:
    assert providers.host_of("http://[::1]:8080/v1") == "::1"
    assert providers.host_of("HTTPS://API.OpenAI.com/v1") == "api.openai.com"
    assert providers.host_of("http://a b") == ""
    assert providers.unambiguous("https://openrouter.ai/api/v1")
    assert not providers.unambiguous("https://user:pw@openrouter.ai/api/v1")
    assert not providers.unambiguous(r"https://openrouter.ai\api")


def test_a_redirect_is_never_followed(monkeypatch) -> None:
    """requests would resend the key (and on 307/308 the body) to the new host."""
    moved = _Resp({}, status=307)
    moved.headers = {"Location": "https://elsewhere.example/v1/chat/completions"}
    wire = Wire(monkeypatch, moved)
    with pytest.raises(ProviderError, match=r"answered 307, redirecting to elsewhere\.example"):
        OpenAIProvider("https://openrouter.ai/api/v1", "m", 0, KEY).chat(HI)
    assert len(wire.sent) == 1 and wire.sent[0]["allow_redirects"] is False


@pytest.mark.parametrize("model", ["", "m"])  # /api/tags (auto model), else /api/show
def test_ollama_never_follows_a_redirect(monkeypatch, model) -> None:
    moved = _Resp({}, status=301)
    moved.headers = {"Location": "http://elsewhere.example:11434/api/show"}
    wire = Wire(monkeypatch, moved)
    with pytest.raises(
        ProviderError,
        match=r"Ollama answered 301, redirecting to elsewhere\.example; set that as the URL",
    ):
        OllamaProvider("http://10.0.0.2:11434", model, 0).capabilities()
    assert len(wire.sent) == 1 and wire.sent[0]["allow_redirects"] is False


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example@api.openai.com/v1",
        "https://api.openai.com@evil.example/v1",
        "https://user:pw@openrouter.ai/api/v1",
    ],
)
def test_a_url_with_a_user_name_or_password_sends_nothing(home, monkeypatch, url) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    assert providers.has_userinfo(url) and not providers.unambiguous(url)
    set_config(home, assistant={"provider": "openai", "model": "m", "url": url})
    wire = Wire(monkeypatch)
    with pytest.raises(providers.AmbiguousURLError, match="user name or password") as err:
        providers.make().chat(HI)
    assert "pw" not in str(err.value) and wire.sent == []
    assert not providers.has_userinfo("https://openrouter.ai/api/v1")


def test_a_redirect_without_a_location_names_no_host(monkeypatch) -> None:
    Wire(monkeypatch, _Resp({}, status=302))
    with pytest.raises(ProviderError, match="redirecting to another address"):
        AnthropicProvider("", "m", 0, KEY).chat(HI)


def test_only_a_key_for_another_host_is_noted_as_checked_later() -> None:
    later = providers.KEY_CHECKED_LATER
    assert (
        providers.check_note(OpenAIProvider("https://openrouter.ai/api/v1", "m", 0, KEY)) == later
    )
    assert providers.check_note(OpenAIProvider("", "m", 0, KEY)) is None
    assert providers.check_note(AnthropicProvider("", "m", 0, KEY)) is None
    assert providers.check_note(OpenAIProvider("http://127.0.0.1:8080", "m", 0, None)) is None
    assert providers.check_note(OllamaProvider("http://127.0.0.1:11434", "m", 0)) is None


def test_a_missing_key_is_named_with_where_to_set_it() -> None:
    why = providers.missing_key(OpenAIProvider("https://openrouter.ai/api/v1", "m", 0, None))
    assert why and "CUTPOINT_ASSISTANT_API_KEY" in why and "OPENAI_API_KEY" not in why
    assert providers.missing_key(OpenAIProvider("http://127.0.0.1:8080", "m", 0, None)) is None
    assert providers.missing_key(AnthropicProvider("", "m", 0, KEY)) is None
    assert providers.missing_key(OllamaProvider("http://10.0.0.2:11434", "m", 0)) is None


def test_codex_list_models_queries_cliproxy(monkeypatch) -> None:
    monkeypatch.setattr(providers.cliproxy, "status", lambda p: {"logged_in": True})
    monkeypatch.setattr(
        providers.cliproxy,
        "list_models",
        lambda p: [
            {"id": "gpt-6.1-sol", "name": "GPT-6.1 Sol"},
            {"id": "gpt-5.6-sol", "name": "gpt-5.6-sol"},
        ],
    )
    p = providers.SubscriptionProvider("codex", "", "", 0)
    models = p.list_models()
    assert models == [
        {"id": "gpt-6.1-sol", "name": "GPT-6.1 Sol"},
        {"id": "gpt-5.6-sol", "name": "gpt-5.6-sol"},
    ]


def test_codex_list_models_honest_errors_no_fake_fallback(monkeypatch) -> None:
    def fake_err(p):
        raise providers.cliproxy.ProxyError("proxy refused login (401)")

    monkeypatch.setattr(providers.cliproxy, "list_models", fake_err)
    p = providers.SubscriptionProvider("codex", "", "", 0)
    with pytest.raises(ProviderError, match="proxy refused login"):
        p.list_models()

    def not_logged_in(p):
        raise providers.cliproxy.ProxyError("not logged in; run cutpoint login codex")

    monkeypatch.setattr(providers.cliproxy, "list_models", not_logged_in)
    with pytest.raises(ProviderError, match="not logged in"):
        p.list_models()


def test_claude_code_list_models_queries_cliproxy(monkeypatch) -> None:
    monkeypatch.setattr(providers.cliproxy, "status", lambda p: {"logged_in": True})
    monkeypatch.setattr(
        providers.cliproxy,
        "list_models",
        lambda p: [{"id": "claude-haiku-5-5", "name": "Claude Haiku 5.5"}],
    )
    p = providers.SubscriptionProvider("claude-code", "", "", 0)
    models = p.list_models()
    assert models == [{"id": "claude-haiku-5-5", "name": "Claude Haiku 5.5"}]


def test_model_list_subscription_login_requires_project_consent(
    monkeypatch,
) -> None:
    providers.clear_model_cache()
    monkeypatch.setattr(providers.cliproxy, "status", lambda p: {"logged_in": True})
    monkeypatch.setattr(
        providers.cliproxy,
        "list_models",
        lambda p: [{"id": "gpt-6.1-sol", "name": "GPT-6.1 Sol"}],
    )
    codex = providers.SubscriptionProvider("codex", "", "", 0)
    # Subscription provider without consent requires consent
    assert providers.model_list(codex, consent=())["state"] == "consent_needed"

    # With consent, model list loads
    res = providers.model_list(codex, consent=(providers.consent_key(codex),))
    assert res["state"] == "ok"
    assert res["models"] == [{"id": "gpt-6.1-sol", "name": "GPT-6.1 Sol"}]

    # Keyed/custom hosted provider still enforces project data consent boundary
    openai = providers.OpenAIProvider("", "m", 0, KEY)
    assert providers.model_list(openai, consent=())["state"] == "consent_needed"

    # Missing login returns key_needed, not bypassed
    monkeypatch.setattr(
        providers.cliproxy,
        "status",
        lambda p: {"logged_in": False, "error": "not logged in to Codex subscription"},
    )
    assert (
        providers.model_list(codex, consent=(providers.consent_key(codex),), refresh=True)["state"]
        == "key_needed"
    )


def test_clear_model_cache_invalidates_transient_cache(monkeypatch) -> None:
    providers.clear_model_cache()
    call_count = 0

    def fake_list(p):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [{"id": "gpt-6.1-sol", "name": "GPT-6.1 Sol"}]
        return [{"id": "gpt-6.1-terra", "name": "GPT-6.1 Terra"}]

    monkeypatch.setattr(providers.cliproxy, "status", lambda p: {"logged_in": True})
    monkeypatch.setattr(providers.cliproxy, "list_models", fake_list)
    codex = providers.SubscriptionProvider("codex", "", "", 0)
    res1 = providers.model_list(codex, consent=(providers.consent_key(codex),))
    assert res1["models"][0]["id"] == "gpt-6.1-sol"
    assert call_count == 1

    # Cached: no second request
    res2 = providers.model_list(codex, consent=(providers.consent_key(codex),))
    assert res2["models"][0]["id"] == "gpt-6.1-sol"
    assert call_count == 1

    # Invalidate cache for codex
    providers.clear_model_cache("codex")
    res3 = providers.model_list(codex, consent=(providers.consent_key(codex),))
    assert res3["models"][0]["id"] == "gpt-6.1-terra"
    assert call_count == 2
    providers.clear_model_cache()
