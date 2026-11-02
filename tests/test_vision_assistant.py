"""The vision rater and 360 VL check use the configured vision_assistant provider."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from conftest import set_config
from cutpoint import vision
from cutpoint.assistant import providers
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
    res = rater.rate(Path("a.mp4"), 0, 4, "ski")
    assert res == vision.VisionResult(8.0, "big jump")
    (msg,) = fake.sent[0]
    assert msg["role"] == "user" and msg["content"][0]["type"] == "text"
    assert msg["content"][1:] == [{"type": "image", "media_type": "image/jpeg", "data": B64}] * 3


def test_nothing_is_sent_before_the_user_agreed(fake, home) -> None:
    """Consent given to the chat assistant does not cover vision: no frame is sent."""
    set_config(
        home,
        assistant={"provider": "codex", "consent": [CONSENT]},
        vision_assistant={"provider": "codex", "model": "gpt-x", "consent": []},
    )
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
    formatted = providers._format_openai_message(msgs[0])
    assert formatted["content"][1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{B64}"},
    }


def test_rating_cache_key_distinguishes_provider() -> None:
    k_codex = vision.rating_key("model-a", "gaming", [1.0, 2.0], provider="codex")
    k_antigravity = vision.rating_key("model-a", "gaming", [1.0, 2.0], provider="antigravity")
    k_ollama = vision.rating_key("model-a", "gaming", [1.0, 2.0], provider="ollama")
    assert k_codex != k_antigravity
    assert k_codex != k_ollama


def test_offline_refresh_skips_vision_instead_of_raising(home, monkeypatch) -> None:
    set_config(
        home,
        vision_assistant={
            "provider": "codex",
            "model": "gpt-x",
            "consent": ["codex@chatgpt.com"],
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
