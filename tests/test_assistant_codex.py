"""The subscription providers (Codex, Antigravity): OpenAI format chat through cliproxy,
structured outputs (json_schema), the replies they refuse, and the model they pick.

Registration, hosted consent and the custom-URL refusal are in ``test_assistant_presets`` and
``test_assistant_routes``; image parts in ``test_vision_assistant``; the OpenAI tool-call parser
in ``test_assistant_byo``.
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


def _answer(message: dict[str, Any], finish: str | None = "stop") -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", **message}, "finish_reason": finish}]}


@pytest.mark.parametrize(
    ("name", "model", "schema", "reasoning_key"),
    [
        ("codex", "gpt-5.6-sol", {"type": "object", "properties": {"a": {"type": "string"}}},
         "reasoning"),
        ("antigravity", "gemini-3.1-pro", None, "reasoning_content"),
    ],
)  # fmt: skip
def test_a_turn_is_sent_to_cliproxy_and_the_reply_read(
    monkeypatch: pytest.MonkeyPatch, name, model, schema, reasoning_key
) -> None:
    """The turn goes to cliproxy under the provider's name with the chosen model and the
    messages as sent; a schema asks for json_schema output; the reply text and the model's
    reasoning field come back."""
    sent: list[tuple[str, dict[str, Any]]] = []

    def fake_chat(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
        sent.append((provider, payload))
        return _answer({"content": '{"a": "done"}', reasoning_key: "step by step"})

    monkeypatch.setattr(providers.cliproxy, "chat", fake_chat)
    p = providers.SubscriptionProvider(name, "", model, 0)
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "go"}]
    reply = p.chat(messages, schema=schema)
    assert reply.text == '{"a": "done"}'
    assert reply.message[reasoning_key] == "step by step"
    ((provider, payload),) = sent
    assert provider == name and payload["model"] == model
    assert payload["messages"] == messages
    if schema:
        assert payload["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "cutpoint_reply", "schema": schema},
        }
    else:
        assert "response_format" not in payload


@pytest.mark.parametrize(
    ("finish", "error"),
    [
        ("length", "cut off"),
        ("content_filter", "content filter"),
        ("tool_calls", "without completed tool calls"),
    ],
)
def test_an_unfinished_reply_is_a_provider_error(
    monkeypatch: pytest.MonkeyPatch, finish, error
) -> None:
    monkeypatch.setattr(
        providers.cliproxy, "chat", lambda prov, req: _answer({"content": "partial"}, finish)
    )
    p = providers.SubscriptionProvider("codex", "", "gpt-5.6-sol", 0)
    with pytest.raises(ProviderError, match=error):
        p.chat([{"role": "user", "content": "hi"}])


def test_not_logged_in_raises_actionable_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        providers.cliproxy,
        "chat",
        lambda prov, req: (_ for _ in ()).throw(
            providers.cliproxy.ProxyError("run cutpoint login codex first")
        ),
    )
    p = providers.SubscriptionProvider("codex", "", "gpt-5.6-sol", 0)
    with pytest.raises(ProviderError, match="run cutpoint login codex first"):
        p.chat([{"role": "user", "content": "hi"}])


@pytest.mark.parametrize(
    ("name", "listed", "chosen"),
    [
        ("antigravity", ["gemini-2.5-flash", "gemini-3.1-pro"], "gemini-3.1-pro"),  # the default
        ("antigravity", ["claude-x", "gemini-2.5-flash"], "gemini-2.5-flash"),  # first Gemini
        ("codex", ["gpt-a", "gpt-b"], "gpt-a"),  # default not listed: the first
    ],
)
def test_without_a_model_the_default_or_a_listed_one_is_asked(
    monkeypatch: pytest.MonkeyPatch, name, listed, chosen
) -> None:
    monkeypatch.setattr(
        providers.cliproxy, "list_models", lambda p: [{"id": m, "name": m} for m in listed]
    )
    assert providers.SubscriptionProvider(name, "", "", 0).resolve_model() == chosen


def test_codex_chat_does_not_check_management_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        providers.cliproxy,
        "status",
        lambda p: (_ for _ in ()).throw(AssertionError("unexpected management status request")),
    )
    monkeypatch.setattr(providers.cliproxy, "chat", lambda prov, req: _answer({"content": "ok"}))
    p = providers.SubscriptionProvider("codex", "", "gpt-5.6-sol", 0)
    assert p.chat([{"role": "user", "content": "hi"}]).text == "ok"
