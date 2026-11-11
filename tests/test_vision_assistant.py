"""The vision rater and 360 VL check use the configured vision_assistant provider."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import pytest

from conftest import set_config
from cutpoint import vision
from cutpoint.assistant import keys, providers
from cutpoint.assistant.providers import AnthropicProvider, OpenAIProvider, Reply, XAIProvider
from cutpoint.pano import vlcheck
from test_assistant import _Resp
from test_assistant_byo import NoRing, Wire, an_reply, oa_reply, rs_reply, rs_text

JPEG = b"\xff\xd8jpeg"
B64 = base64.b64encode(JPEG).decode()
CONSENT = "vision-v2:codex@chatgpt.com"


class FakeSubscription:
    name, model, local, host = "codex", "gpt-x", False, "chatgpt.com"

    def __init__(self, text: str) -> None:
        self.text, self.sent = text, []

    def capabilities(self) -> None:
        pass

    def chat(self, messages, **_kw) -> Reply:
        self.sent.append(messages)
        return Reply(self.text)


@pytest.fixture
def fake(home, monkeypatch) -> FakeSubscription:
    """Ollama is unreachable (any request fails the test); the provider is a fake."""
    fake = FakeSubscription('{"score": 8, "label": "big jump"}')
    monkeypatch.setattr(providers, "make", lambda *_a, **_k: fake)

    def no_network(*_a, **_k):
        raise AssertionError("Ollama was asked")

    monkeypatch.setattr(vision.requests, "post", no_network)
    monkeypatch.setattr(vision.requests, "get", no_network)
    monkeypatch.setattr(vision, "grab_frame_jpeg", lambda _src, _t, **_k: JPEG)
    set_config(home, vision_assistant={"provider": "codex", "model": "gpt-x", "consent": [CONSENT]})
    return fake


def test_the_rater_sends_its_frames_to_the_provider(fake, home) -> None:
    """vision_assistant is its own setting: the chat assistant's provider and model are not
    used."""
    set_config(
        home,
        assistant={"provider": "anthropic", "model": "claude-3-7-sonnet", "consent": []},
        vision_assistant={"provider": "codex", "model": "gpt-x", "consent": [CONSENT]},
    )
    rater = vision.VisionRater()
    assert rater.available() and rater.model == "gpt-x"
    res = rater.rate(Path("a.mp4"), 0, 4, "action-cam")
    assert res == vision.VisionResult(8.0, "big jump")
    (msg,) = fake.sent[0]
    assert msg["role"] == "user" and msg["content"][0]["type"] == "text"
    assert msg["content"][1:] == [{"type": "image", "media_type": "image/jpeg", "data": B64}] * 3


def test_nothing_is_sent_before_the_user_agreed(fake, home) -> None:
    """Consent given to the chat assistant does not cover vision: no frame is sent."""
    set_config(
        home,
        assistant={"provider": "codex", "consent": ["assistant-v3:codex@chatgpt.com"]},
        vision_assistant={"provider": "codex", "model": "gpt-x", "consent": []},
    )
    rater = vision.VisionRater()
    assert not rater.available()
    assert rater.rate(Path("a.mp4"), 0, 4, "action-cam") is None and fake.sent == []


def test_consent_for_another_provider_does_not_send_frames(fake) -> None:
    """Consent is per provider and host: codex's does not cover openai."""
    fake.name = "openai"
    assert not vision.VisionRater().available()


def test_the_vl_check_asks_the_provider_and_stops_when_it_fails(fake) -> None:
    fake.text = '{"blocked": true, "where": "left", "what": "glove", "low_info": false}'
    checker = vlcheck.VLChecker()
    assert checker.available()
    assert checker.ask(JPEG) == {
        "blocked": True,
        "where": "left",
        "what": "glove",
        "low_info": False,
    }
    assert fake.sent[0][0]["content"][1]["data"] == B64
    fake.chat = lambda *_a, **_k: (_ for _ in ()).throw(providers.ProviderError("down"))
    assert checker.ask(JPEG) is None and checker.dead


def test_image_parts_reach_each_wire_format() -> None:
    part = {"type": "image", "media_type": "image/jpeg", "data": B64}
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}, part]}]
    _, wire = providers._anthropic_messages(msgs)
    assert wire[0]["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": B64},
    }
    formatted = providers._format_openai_message(msgs[0])
    assert formatted["content"][1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{B64}"},
    }


def test_the_assistant_s_versioned_consent_does_not_send_frames(fake, home) -> None:
    set_config(
        home,
        assistant={"provider": "codex", "consent": ["assistant-v3:codex@chatgpt.com"]},
        vision_assistant={"provider": "codex", "consent": ["assistant-v3:codex@chatgpt.com"]},
    )
    rater = vision.VisionRater()
    assert not rater.available()
    assert rater.rate(Path("a.mp4"), 0, 4, "action-cam") is None
    assert fake.sent == []


def test_consent_under_the_old_vision_wording_no_longer_counts(fake, home) -> None:
    # "codex@chatgpt.com" was the Vision key before frames, on-screen text and telemetry were
    # named in the prompt: the user must agree again (vision-v2:)
    set_config(
        home,
        vision_assistant={"provider": "codex", "model": "gpt-x", "consent": ["codex@chatgpt.com"]},
    )
    rater = vision.VisionRater()
    assert not rater.available()
    assert rater.rate(Path("a.mp4"), 0, 4, "action-cam") is None
    assert fake.sent == []


def test_rating_cache_key_distinguishes_provider() -> None:
    k_codex = vision.rating_key("model-a", "gaming", [1.0, 2.0], provider="codex")
    k_antigravity = vision.rating_key("model-a", "gaming", [1.0, 2.0], provider="antigravity")
    k_ollama = vision.rating_key("model-a", "gaming", [1.0, 2.0], provider="ollama")
    assert k_codex != k_antigravity
    assert k_codex != k_ollama


def test_the_provider_gets_the_speech_with_the_frames(fake) -> None:
    rater = vision.VisionRater()
    assert rater.rate(Path("a.mp4"), 0, 4, "gaming", transcript="I'm laughing so hard")
    (msg,) = fake.sent[0]
    assert "<transcript>\nI'm laughing so hard\n</transcript>" in msg["content"][0]["text"]
    assert len(msg["content"]) == 4  # the prompt and the 3 frames; the speech is in the prompt


def test_offline_refresh_skips_vision_instead_of_raising(home, monkeypatch) -> None:
    set_config(
        home,
        vision_assistant={
            "provider": "codex",
            "model": "gpt-x",
            "consent": [CONSENT],
        },
    )

    codex = providers.SubscriptionProvider("codex", "", "gpt-x", 0)
    monkeypatch.setattr(
        providers.cliproxy,
        "status",
        lambda _name: {"logged_in": False, "error": "network unreachable"},
    )
    monkeypatch.setattr(providers, "make", lambda _cfg: codex)

    ast_vis = vision.AssistantVision()
    assert ast_vis.available() is False
    assert ast_vis.ask("describe", [JPEG]) is None


# -- hosted Vision through an API key ---------------------------------------------------------

KEY = "plain-secret-key-never-shown-42"
SCHEMA = {"type": "object", "properties": {"score": {"type": "number"}}}
REPLY = '{"score": 7, "label": "wave"}'
API = {  # provider -> (official host, key variable, the answers to the check and to the request)
    "anthropic": (
        "api.anthropic.com",
        "ANTHROPIC_API_KEY",
        lambda: [_Resp({"id": "m"}), an_reply({"type": "text", "text": REPLY})],
    ),
    "openai": (
        "api.openai.com",
        "OPENAI_API_KEY",
        lambda: [_Resp({"data": [{"id": "m"}]}), rs_reply(rs_text(REPLY))],
    ),
    "xai": (
        "api.x.ai",
        "XAI_API_KEY",
        lambda: [_Resp({"data": [{"id": "m"}]}), oa_reply(REPLY)],
    ),
}


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real key variable or keyring reaches a test."""
    for name in keys.ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(keys, "_keyring", NoRing)
    monkeypatch.setattr(keys, "_SEEN", set())
    monkeypatch.setattr(keys, "_CAPTURED", {})


def api_vision(home: Path, name: str, consent: list[str], **extra) -> vision.AssistantVision:
    set_config(home, vision_assistant={"provider": name, "model": "m", "consent": consent, **extra})
    return vision.AssistantVision()


@pytest.mark.parametrize("name", list(API))
def test_an_api_provider_with_vision_consent_gets_the_frames(
    name, home, no_keys, monkeypatch
) -> None:
    """No subscription is needed: an API key and the vision-v2 consent send the frames."""
    host, env, answers = API[name]
    monkeypatch.setenv(env, KEY)
    wire = Wire(monkeypatch, *answers())
    ast = api_vision(home, name, [f"vision-v2:{name}@{host}"])
    assert ast.available()
    assert ast.ask("describe", [JPEG, JPEG]) == REPLY
    sent = wire.sent[-1]
    assert sent["method"] == "POST" and host in sent["url"]
    assert sent["json"]["model"] == "m" and B64 in str(sent["json"])
    assert KEY in str(sent["headers"]) and KEY not in str(sent["json"])


@pytest.mark.parametrize("name", list(API))
def test_the_assistant_consent_does_not_unlock_vision_for_an_api_provider(
    name, home, no_keys, monkeypatch, caplog
) -> None:
    host, env, _ = API[name]
    monkeypatch.setenv(env, KEY)
    wire = Wire(monkeypatch)  # any request fails the test
    ast = api_vision(home, name, [f"assistant-v3:{name}@{host}", f"{name}@{host}"])
    with caplog.at_level(logging.INFO, logger="cutpoint.vision"):
        assert not ast.available()
        assert ast.ask("describe", [JPEG]) is None
    assert "needs your consent before frames are sent to it" in caplog.text
    assert f'add "vision-v2:{name}@{host}" to vision_assistant.consent' in caplog.text
    assert wire.sent == []


@pytest.mark.parametrize("name", list(API))
def test_a_missing_api_key_skips_vision_with_a_log_line_and_no_request(
    name, home, no_keys, monkeypatch, caplog
) -> None:
    host, env, _ = API[name]
    wire = Wire(monkeypatch)
    ast = api_vision(home, name, [f"vision-v2:{name}@{host}"])
    with caplog.at_level(logging.INFO, logger="cutpoint.vision"):
        assert not ast.available()
        assert ast.ask("describe", [JPEG]) is None
    assert "No API key" in caplog.text and env in caplog.text and "skipping vision" in caplog.text
    assert wire.sent == []


def test_a_local_openai_server_needs_no_vision_consent_and_no_key(
    home, no_keys, monkeypatch
) -> None:
    """A base URL on this machine: nothing leaves it, so no consent (consent key None)."""
    wire = Wire(monkeypatch, _Resp({"data": [{"id": "m"}]}), oa_reply(REPLY))
    ast = api_vision(home, "openai", [], url="http://localhost:1234/v1")
    assert providers.consent_key(ast.provider, "vision") is None
    assert ast.available()
    assert ast.ask("describe", [JPEG]) == REPLY
    assert wire.sent[-1]["url"] == "http://localhost:1234/v1/chat/completions"
    assert B64 in str(wire.sent[-1]["json"])


@pytest.mark.parametrize(
    ("url", "host"),
    [
        ("http://192.168.1.5:1234/v1", "192.168.1.5:1234"),
        ("https://openrouter.ai/api/v1", "openrouter.ai"),
    ],
)
def test_a_custom_non_loopback_url_needs_its_own_vision_consent(
    url, host, home, no_keys, monkeypatch
) -> None:
    """Consent for the official OpenAI host does not cover another server: nothing is sent."""
    monkeypatch.setenv(API["openai"][1], KEY)
    wire = Wire(monkeypatch)  # any request fails the test
    ast = api_vision(home, "openai", ["vision-v2:openai@api.openai.com"], url=url)
    key = providers.consent_key(ast.provider, "vision")
    assert key is not None and key.startswith("vision-v2:openai@")
    assert key == f"vision-v2:openai@{host}"
    assert not ast.available()
    assert ast.ask("describe", [JPEG]) is None
    assert wire.sent == []


# -- what goes on the wire ---------------------------------------------------------------------

FRAME = {"type": "image", "media_type": "image/jpeg", "data": B64}
MSGS = [{"role": "user", "content": [{"type": "text", "text": "rate"}, FRAME]}]
DATA_URL = f"data:image/jpeg;base64,{B64}"


@pytest.mark.parametrize(
    "make_provider",
    [
        lambda: OpenAIProvider("https://openrouter.ai/api/v1", "m", 0, KEY),
        lambda: XAIProvider("https://llm.example.com/v1", "m", 0, KEY),
        lambda: XAIProvider("", "m", 0, KEY),  # api.x.ai speaks Chat Completions too
    ],
    ids=["openai-compatible", "xai-compatible", "xai-official"],
)
def test_chat_completions_carry_frames_as_image_urls_and_the_schema_as_response_format(
    monkeypatch, make_provider
) -> None:
    wire = Wire(monkeypatch, oa_reply(REPLY), oa_reply(REPLY))
    p = make_provider()
    assert not p.responses
    p.chat(MSGS, schema=SCHEMA)
    p.chat(MSGS)
    with_schema, without = (s["json"] for s in wire.sent)
    assert wire.sent[0]["url"].endswith("/chat/completions")
    assert with_schema["messages"][0]["content"] == [
        {"type": "text", "text": "rate"},
        {"type": "image_url", "image_url": {"url": DATA_URL}},
    ]
    assert with_schema["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "cutpoint_reply", "schema": SCHEMA},
    }
    assert "response_format" not in without
    assert MSGS[0]["content"][1] == FRAME  # the caller's message is not rewritten


def test_the_official_openai_host_takes_frames_and_schema_over_the_responses_api(
    monkeypatch,
) -> None:
    wire = Wire(monkeypatch, rs_reply(rs_text(REPLY)))
    p = OpenAIProvider("", "m", 0, KEY)
    assert p.responses and p.chat(MSGS, schema=SCHEMA).text == REPLY
    sent = wire.sent[0]
    assert sent["url"] == "https://api.openai.com/v1/responses"
    assert sent["json"]["input"][0]["content"] == [
        {"type": "input_text", "text": "rate"},
        {"type": "input_image", "image_url": DATA_URL},
    ]
    assert sent["json"]["text"]["format"]["type"] == "json_schema"
    assert sent["json"]["text"]["format"]["schema"] == SCHEMA
    assert "response_format" not in sent["json"]


def test_anthropic_sends_frames_as_base64_sources_and_ignores_the_schema(monkeypatch) -> None:
    wire = Wire(monkeypatch, an_reply({"type": "text", "text": REPLY}))
    p = AnthropicProvider("", "m", 0, KEY)
    assert p.chat(MSGS, schema=SCHEMA).text == REPLY  # no error for a schema
    sent = wire.sent[0]["json"]
    assert wire.sent[0]["url"] == "https://api.anthropic.com/v1/messages"
    assert sent["messages"][0]["content"] == [
        {"type": "text", "text": "rate"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": B64}},
    ]
    assert not {"response_format", "schema", "text"} & set(sent)


# -- ask_json holds hosted replies to the answer schema (#920) ---------------------------------

ANSWER = {
    "type": "object",
    "properties": {"footage": {"type": "string", "enum": ["camera", "screen"]}},
    "required": ["footage"],
}


@pytest.mark.parametrize(
    ("name", "reply", "sent_schema"),
    [
        ("openai", lambda t: rs_reply(rs_text(t)), lambda j: j["text"]["format"]["schema"]),
        ("xai", oa_reply, lambda j: j["response_format"]["json_schema"]["schema"]),
        ("anthropic", lambda t: an_reply({"type": "text", "text": t}), None),
    ],
)
def test_ask_json_sends_the_schema_to_a_hosted_provider_that_takes_one(
    name, reply, sent_schema, home, no_keys, monkeypatch
) -> None:
    """Anthropic's request has no schema field: the prompt asks for JSON and the reply is
    checked against the schema like every other."""
    host, env, _ = API[name]
    monkeypatch.setenv(env, KEY)
    check = _Resp({"id": "m"} if name == "anthropic" else {"data": [{"id": "m"}]})
    wire = Wire(monkeypatch, check, reply('{"footage": "camera"}'))
    set_config(
        home,
        vision_assistant={"provider": name, "model": "m", "consent": [f"vision-v2:{name}@{host}"]},
    )
    assert vision.VisionRater().ask_json("q", [JPEG], ANSWER) == {"footage": "camera"}
    sent = wire.sent[-1]["json"]
    if sent_schema is None:
        assert not {"response_format", "text", "output_format", "output_config"} & set(sent)
    else:
        assert sent_schema(sent) == ANSWER


@pytest.mark.parametrize(
    "text",
    [
        '{"answer": "screen"}',
        "{}",
        '["screen"]',
        "not json",
    ],  # wrong key, empty, no object, no JSON
)
def test_ask_json_rejects_a_hosted_reply_without_the_required_keys(fake, text) -> None:
    fake.text = text
    assert vision.VisionRater().ask_json("q", [JPEG], ANSWER) is None


@pytest.mark.parametrize(
    "text", ['{"footage": "screen"}', '{"footage": "sport"}', '{"footage": 3}']
)
def test_ask_json_leaves_enums_and_types_to_the_caller(fake, text) -> None:
    """#920: only the top level is checked (an object with the required keys)."""
    fake.text = text
    assert vision.VisionRater().ask_json("q", [JPEG], ANSWER) == json.loads(text)
