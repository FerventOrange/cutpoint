"""The Antigravity subscription provider: OpenAI format chat through cliproxy,
tool call and reply parsing, hosted consent, and model listing.
"""

from __future__ import annotations

from typing import Any

import pytest

from cutpoint.assistant import providers
from cutpoint.assistant.providers import ProviderError


@pytest.fixture(autouse=True)
def logged_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        providers.cliproxy,
        "status",
        lambda p: {"logged_in": True, "email": "dev@example.com", "error": None},
    )


def test_it_is_registered_hosted_and_needs_consent() -> None:
    assert "antigravity" in providers.PROVIDERS and "antigravity" in providers.TITLES
    p = providers.SubscriptionProvider("antigravity", "", "", 0)
    assert (p.name, p.local, p.host) == ("antigravity", False, "daily-cloudcode-pa.googleapis.com")
    assert providers.consent_key(p) == "antigravity@daily-cloudcode-pa.googleapis.com"


def test_custom_url_rejected_for_subscription() -> None:
    with pytest.raises(ProviderError, match="does not accept a custom base URL"):
        providers.SubscriptionProvider("antigravity", "http://custom-proxy.internal:8080", "", 0)


def test_a_turn_is_sent_to_cliproxy_and_the_reply_read(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, Any]] = []

    def fake_chat(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
        sent.append({"provider": provider, "payload": payload})
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello from Gemini",
                        "reasoning_content": "thinking trace",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    monkeypatch.setattr(providers.cliproxy, "chat", fake_chat)
    p = providers.SubscriptionProvider("antigravity", "", "gemini-3.1-pro", 0)
    reply = p.chat([{"role": "user", "content": "hi"}])
    assert reply.text == "Hello from Gemini"
    assert reply.message["reasoning_content"] == "thinking trace"
    assert len(sent) == 1
    assert sent[0]["provider"] == "antigravity"
    assert sent[0]["payload"]["model"] == "gemini-3.1-pro"
    assert sent[0]["payload"]["messages"] == [{"role": "user", "content": "hi"}]


def test_tool_calls_and_their_results_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, Any]] = []

    def fake_chat(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
        sent.append({"provider": provider, "payload": payload})
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"query": "moment"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    monkeypatch.setattr(providers.cliproxy, "chat", fake_chat)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Find moment",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]
    p = providers.SubscriptionProvider("antigravity", "", "gemini-3.1-pro", 0)
    reply = p.chat([{"role": "user", "content": "search"}], tools=tools)
    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert call.name == "lookup" and call.arguments == {"query": "moment"} and call.id == "call_1"

    # Tool result formatting
    result_msg = p.tool_result(call, "result text")
    assert result_msg == {"role": "tool", "tool_call_id": "call_1", "content": "result text"}


def test_not_logged_in_raises_actionable_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        providers.cliproxy,
        "status",
        lambda p: {"logged_in": False, "error": "run cutpoint login antigravity first"},
    )
    p = providers.SubscriptionProvider("antigravity", "", "", 0)
    with pytest.raises(ProviderError, match="run cutpoint login antigravity first"):
        p.chat([{"role": "user", "content": "hi"}])


def test_a_reply_cut_off_raises_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_chat(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "partial text"},
                    "finish_reason": "length",
                }
            ]
        }

    monkeypatch.setattr(providers.cliproxy, "chat", fake_chat)
    p = providers.SubscriptionProvider("antigravity", "", "gemini-3.1-pro", 0)
    with pytest.raises(ProviderError, match="cut off"):
        p.chat([{"role": "user", "content": "hi"}])


def test_model_list_queries_cliproxy(monkeypatch: pytest.MonkeyPatch) -> None:
    models = [
        {"id": "gemini-3.1-pro", "name": "Gemini 3.1 Pro"},
        {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
    ]
    monkeypatch.setattr(providers.cliproxy, "list_models", lambda p: models)
    p = providers.SubscriptionProvider("antigravity", "", "", 0)
    assert p.list_models() == models
    assert p.resolve_model() == "gemini-3.1-pro"


def test_capabilities_honest_reporting() -> None:
    p = providers.SubscriptionProvider("antigravity", "", "", 0)
    caps = p.capabilities()
    assert caps.tool_calls is True
    assert caps.json_schema is False
