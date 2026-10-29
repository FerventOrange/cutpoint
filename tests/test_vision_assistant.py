"""``vision_backend: assistant``: the frame rater and the 360 VL check ask the subscription
provider (consent first) instead of Ollama. No network: the provider and ``requests`` are fakes."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from conftest import set_config
from cutpoint import vision
from cutpoint.assistant import oauth, providers
from cutpoint.assistant.providers import Reply
from cutpoint.pano import vlcheck

JPEG = b"\xff\xd8jpeg"
B64 = base64.b64encode(JPEG).decode()
CONSENT = "codex@chatgpt.com"


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
    set_config(home, vision_backend="assistant", assistant={"consent": [CONSENT]})
    return fake


def test_the_rater_sends_its_frames_to_the_provider(fake) -> None:
    rater = vision.VisionRater()
    assert rater.available() and rater.model == "gpt-x"
    res = rater.rate(Path("a.mp4"), 0, 4, "ski")
    assert res == vision.VisionResult(8.0, "big jump")
    (msg,) = fake.sent[0]
    assert msg["role"] == "user" and msg["content"][0]["type"] == "text"
    assert msg["content"][1:] == [{"type": "image", "media_type": "image/jpeg", "data": B64}] * 3


def test_nothing_is_sent_before_the_user_agreed(fake, home) -> None:
    set_config(home, vision_backend="assistant", assistant={"consent": []})
    rater = vision.VisionRater()
    assert not rater.available()
    assert rater.rate(Path("a.mp4"), 0, 4, "ski") is None and fake.sent == []


def test_only_a_subscription_provider_is_asked(fake) -> None:
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
    codex = providers.CodexProvider("", "m", 8192)
    items = codex._responses_payload(msgs, None, None)["input"]
    assert items[0]["content"] == [
        {"type": "input_text", "text": "hi"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{B64}"},
    ]
    # Antigravity wire parts
    gemini_parts = providers._gemini_parts([part])
    assert gemini_parts == [{"inlineData": {"mimeType": "image/jpeg", "data": B64}}]


def test_a_bad_backend_is_a_config_error(home) -> None:
    set_config(home, vision_backend="gemini")
    with pytest.raises(ValueError, match="vision_backend"):
        vision.VisionRater()


def test_independent_vision_assistant_settings(fake, home) -> None:
    set_config(
        home,
        assistant={"provider": "anthropic", "model": "claude-3-7-sonnet", "consent": []},
        vision_assistant={
            "provider": "codex",
            "model": "gpt-x",
            "consent": ["codex@chatgpt.com"],
        },
    )
    rater = vision.VisionRater()
    assert rater.available()
    assert rater.model == "gpt-x"
    res = rater.rate(Path("a.mp4"), 0, 4, "ski")
    assert res == vision.VisionResult(8.0, "big jump")


def test_independent_vision_consent_privacy_gate(fake, home) -> None:
    # Assistant has consent, but vision_assistant does not have consent: frames MUST NOT be sent
    set_config(
        home,
        assistant={"provider": "codex", "consent": ["codex@chatgpt.com"]},
        vision_assistant={"provider": "codex", "consent": []},
    )
    rater = vision.VisionRater()
    assert not rater.available()
    assert rater.rate(Path("a.mp4"), 0, 4, "ski") is None
    assert fake.sent == []


def test_antigravity_is_allowed_vision_subscription(fake, home) -> None:
    fake.name = "antigravity"
    fake.host = "daily-cloudcode-pa.googleapis.com"
    set_config(
        home,
        vision_assistant={
            "provider": "antigravity",
            "model": "gemini-3.8-flash",
            "consent": ["antigravity@daily-cloudcode-pa.googleapis.com"],
        },
    )
    rater = vision.VisionRater()
    assert rater.available()


def test_rating_cache_key_distinguishes_provider() -> None:
    k_codex = vision.rating_key("model-a", "gaming", [1.0, 2.0], provider="codex")
    k_antigravity = vision.rating_key("model-a", "gaming", [1.0, 2.0], provider="antigravity")
    k_ollama = vision.rating_key("model-a", "gaming", [1.0, 2.0], provider="ollama")
    assert k_codex != k_antigravity
    assert k_codex != k_ollama


def test_offline_refresh_skips_vision_instead_of_raising(home, monkeypatch) -> None:
    set_config(
        home,
        vision_backend="assistant",
        vision_assistant={
            "provider": "codex",
            "model": "gpt-x",
            "consent": ["codex@chatgpt.com"],
        },
    )

    codex = providers.CodexProvider("", "gpt-x", 0)

    def _broken_token():
        raise oauth.OAuthError("network unreachable")

    monkeypatch.setattr(codex, "_token", _broken_token)
    monkeypatch.setattr(providers, "make", lambda _cfg: codex)

    ast_vis = vision.AssistantVision()
    assert ast_vis.available() is False
    assert ast_vis.ask("describe", [JPEG]) is None
