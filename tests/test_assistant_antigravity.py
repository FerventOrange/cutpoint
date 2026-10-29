"""The Antigravity (Cloud Code Gemini) provider: request envelope, message and tool conversion,
reply parsing and a refused login. HTTP is stubbed (``Wire``); no request leaves the machine."""

from __future__ import annotations

import json

import pytest
import requests

from cutpoint.assistant import grants, oauth, providers, session
from cutpoint.assistant.providers import AntigravityProvider, ProviderError
from test_assistant import _Resp
from test_assistant_byo import Wire

TOKEN = "ya29.SECRET-ACCESS-TOKEN"


def login(monkeypatch, endpoint: str = "", project: str = "proj-1") -> None:
    grant = grants.Grant("antigravity", TOKEN, "R", 0, project_id=project, api_endpoint=endpoint)
    monkeypatch.setattr(providers.grants, "load", lambda _name: grant)
    monkeypatch.setattr(providers.grants, "access_token", lambda _name: TOKEN)


def gem_reply(*parts: dict, finish: str = "STOP") -> requests.Response:
    content = {"role": "model", "parts": list(parts)}
    events = [
        {"response": {"candidates": [{"content": content}]}},
        {"response": {"candidates": [{"finishReason": finish}]}},
    ]
    response = requests.Response()
    response.status_code = 200
    response._content = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
    response._content_consumed = True
    return response


def test_a_turn_is_sent_as_a_cloud_code_envelope_and_the_reply_read(monkeypatch) -> None:
    login(monkeypatch)
    wire = Wire(
        monkeypatch,
        _Resp({"models": {providers.DEFAULT_ANTIGRAVITY_MODEL: {"displayName": "Gemini"}}}),
        gem_reply({"text": "thinking", "thought": True}, {"text": "Hello"}),
    )
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {
            "role": "user",
            "content": [{"type": "image", "media_type": "image/jpeg", "data": "QQ=="}],
        },
    ]
    reply = AntigravityProvider("", "", 0).chat(messages, tools=session.TOOLS)
    sent = wire.sent[1]
    assert sent["url"] == f"{oauth.CLOUDCODE}/v1internal:streamGenerateContent?alt=sse"
    assert sent["headers"] == {"Authorization": f"Bearer {TOKEN}", **oauth.ANTIGRAVITY_HEADERS}
    assert sent["allow_redirects"] is False
    body = sent["json"]
    assert body["project"] == "proj-1" and body["model"] == providers.DEFAULT_ANTIGRAVITY_MODEL
    req = body["request"]
    assert req["systemInstruction"] == {"role": "user", "parts": [{"text": "rules"}]}
    assert req["contents"] == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "hello"}]},
        {"role": "user", "parts": [{"inlineData": {"mimeType": "image/jpeg", "data": "QQ=="}}]},
    ]
    decls = req["tools"][0]["functionDeclarations"]
    assert [d["name"] for d in decls] == [t["function"]["name"] for t in session.TOOLS]
    assert reply.text == "Hello"  # the thought part is not shown


def test_tool_calls_and_their_results_round_trip(monkeypatch) -> None:
    login(monkeypatch, endpoint="https://staging-cloudcode-pa.googleapis.com")
    call = {"functionCall": {"name": "get_moment", "args": {"id": "m1"}}, "thoughtSignature": "sig"}
    wire = Wire(monkeypatch, gem_reply(call), gem_reply({"text": "done"}))
    provider = AntigravityProvider("", "gemini-x", 0)
    reply = provider.chat([{"role": "user", "content": "go"}], tools=session.TOOLS)
    (tc,) = reply.tool_calls
    assert (tc.name, tc.arguments, tc.id) == ("get_moment", {"id": "m1"}, "call_1")
    result = provider.tool_result(tc, "ok")
    provider.chat([{"role": "user", "content": "go"}, reply.message, result, result])
    second = wire.sent[1]
    assert (
        second["url"]
        == "https://staging-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse"
    )
    assert second["json"]["model"] == "gemini-x"
    model_turn, user_turn = second["json"]["request"]["contents"][1:]
    assert model_turn["parts"][0]["thoughtSignature"] == "sig"  # sent back as it came
    assert model_turn["parts"][0]["functionCall"]["id"] == "call_1"
    assert len(user_turn["parts"]) == 2  # both results in one user turn
    assert user_turn["parts"][0] == {
        "functionResponse": {"name": "get_moment", "id": "call_1", "response": {"result": "ok"}}
    }


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_login_says_to_log_in_again_never_the_token(monkeypatch, status) -> None:
    login(monkeypatch)
    Wire(monkeypatch, _Resp({"error": {"message": f"bad {TOKEN}"}}, status))
    with pytest.raises(ProviderError, match="run cutpoint login antigravity again") as err:
        AntigravityProvider("", "", 0).chat([{"role": "user", "content": "hi"}])
    assert TOKEN not in str(err.value)


def test_no_project_means_login_first_and_nothing_is_sent(monkeypatch) -> None:
    login(monkeypatch, project="")
    wire = Wire(monkeypatch)
    with pytest.raises(ProviderError, match="run cutpoint login antigravity first"):
        AntigravityProvider("", "", 0).chat([{"role": "user", "content": "hi"}])
    assert wire.sent == []


def test_a_reply_cut_off_runs_no_call(monkeypatch) -> None:
    login(monkeypatch)
    Wire(monkeypatch, gem_reply({"functionCall": {"name": "x", "args": {}}}, finish="MAX_TOKENS"))
    with pytest.raises(ProviderError, match="cut off"):
        AntigravityProvider("", providers.DEFAULT_ANTIGRAVITY_MODEL, 0).chat(
            [{"role": "user", "content": "hi"}]
        )


def test_it_is_registered_hosted_and_needs_consent(monkeypatch) -> None:
    login(monkeypatch)
    assert "antigravity" in providers.PROVIDERS and "antigravity" in providers.TITLES
    p = AntigravityProvider("", "", 0)
    assert (
        not p.local and providers.consent_key(p) == "antigravity@daily-cloudcode-pa.googleapis.com"
    )


def test_account_roster_excludes_disabled_internal_and_image_models(monkeypatch) -> None:
    login(monkeypatch)
    Wire(
        monkeypatch,
        _Resp(
            {
                "models": {
                    "gemini-served": {"displayName": "Served"},
                    "gemini-unavailable": {"disabled": True},
                    "internal": {"isInternal": True},
                    "image-only": {},
                },
                "imageGenerationModelIds": ["image-only"],
            }
        ),
    )
    p = AntigravityProvider("", "", 0)
    assert p.resolve_model() == "gemini-served"


def test_truncated_stream_does_not_return_a_partial_tool_call(monkeypatch) -> None:
    login(monkeypatch)
    response = gem_reply({"functionCall": {"name": "x", "args": {}}}, finish="")
    Wire(monkeypatch, response)
    with pytest.raises(ProviderError, match="ended without a completed response"):
        AntigravityProvider("", "gemini-served", 0).chat([{"role": "user", "content": "hi"}])
