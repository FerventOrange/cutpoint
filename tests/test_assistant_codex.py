"""The Codex subscription provider: OpenAI format chat through cliproxy,
structured outputs (json_schema), tool calls, image URLs, and hosted consent.
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


def test_codex_registration_and_hosted_consent() -> None:
    assert "codex" in providers.PROVIDERS and "codex" in providers.TITLES
    p = providers.SubscriptionProvider("codex", "", "", 0)
    assert (p.name, p.local, p.host) == ("codex", False, "chatgpt.com")
    assert providers.consent_key(p) == "codex@chatgpt.com"


def test_codex_capabilities() -> None:
    p = providers.SubscriptionProvider("codex", "", "", 0)
    caps = p.capabilities()
    assert caps.tool_calls is True
    assert caps.json_schema is False


def test_custom_url_rejected_for_codex() -> None:
    with pytest.raises(ProviderError, match="does not accept a custom base URL"):
        providers.SubscriptionProvider("codex", "http://codex.internal:8080", "", 0)


def test_content_filter_and_incomplete_tool_calls_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    p = providers.SubscriptionProvider("codex", "", "gpt-5.6-sol", 0)

    monkeypatch.setattr(
        providers.cliproxy,
        "chat",
        lambda prov, req: {
            "choices": [
                {"message": {"role": "assistant", "content": ""}, "finish_reason": "content_filter"}
            ]
        },
    )
    with pytest.raises(ProviderError, match="content filter"):
        p.chat([{"role": "user", "content": "filter me"}])

    monkeypatch.setattr(
        providers.cliproxy,
        "chat",
        lambda prov, req: {
            "choices": [
                {"message": {"role": "assistant", "content": ""}, "finish_reason": "tool_calls"}
            ]
        },
    )
    with pytest.raises(ProviderError, match="without completed tool calls"):
        p.chat([{"role": "user", "content": "call tool"}])


def test_codex_chat_sends_structured_schema_and_preserves_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, Any]] = []

    def fake_chat(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
        sent.append({"provider": provider, "payload": payload})
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"analysis": "done"}',
                        "reasoning": "step-by-step reasoning",
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    monkeypatch.setattr(providers.cliproxy, "chat", fake_chat)
    p = providers.SubscriptionProvider("codex", "", "gpt-5.6-sol", 0)
    schema = {"type": "object", "properties": {"analysis": {"type": "string"}}}
    reply = p.chat([{"role": "user", "content": "analyze"}], schema=schema)
    assert reply.text == '{"analysis": "done"}'
    assert reply.message["reasoning"] == "step-by-step reasoning"
    assert len(sent) == 1
    assert sent[0]["provider"] == "codex"
    assert sent[0]["payload"]["model"] == "gpt-5.6-sol"
    assert sent[0]["payload"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "cutpoint_reply", "schema": schema},
    }


def test_codex_chat_converts_neutral_images_to_image_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, Any]] = []

    def fake_chat(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
        sent.append({"provider": provider, "payload": payload})
        return {
            "choices": [{"message": {"role": "assistant", "content": "I see a cat"}}],
        }

    monkeypatch.setattr(providers.cliproxy, "chat", fake_chat)
    p = providers.SubscriptionProvider("codex", "", "gpt-5.6-sol", 0)
    b64_data = "aW1hZ2VkYXRh"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image", "media_type": "image/png", "data": b64_data},
            ],
        }
    ]
    reply = p.chat(messages)
    assert reply.text == "I see a cat"
    assert len(sent) == 1
    content_parts = sent[0]["payload"]["messages"][0]["content"]
    assert content_parts[0] == {"type": "text", "text": "what is this?"}
    assert content_parts[1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64_data}"},
    }


def test_codex_not_logged_in_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        providers.cliproxy,
        "status",
        lambda p: {"logged_in": False, "error": "run cutpoint login codex first"},
    )
    p = providers.SubscriptionProvider("codex", "", "", 0)
    with pytest.raises(ProviderError, match="run cutpoint login codex first"):
        p.chat([{"role": "user", "content": "hi"}])
