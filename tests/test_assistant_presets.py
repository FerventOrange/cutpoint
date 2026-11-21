"""Settings' assistant presets (None/Ollama, Claude, Codex, Grok), the xAI key and model lists.

No request reaches a real provider: ``requests`` is replaced by :class:`Wire` in every test
that could send one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from assistant_helpers import KEY, OTHER, XKEY, Wire, _Resp, oa_reply
from conftest import set_config
from cutpoint.assistant import keys, providers
from cutpoint.config import AssistantSettings

pytestmark = pytest.mark.usefixtures("no_keys")  # no real key variable or keyring

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


# -- presets: provider, base URL, host -------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "cls", "url", "host"),
    [
        (
            "anthropic",
            providers.AnthropicProvider,
            "https://api.anthropic.com",
            "api.anthropic.com",
        ),
        ("openai", providers.OpenAIProvider, "https://api.openai.com/v1", "api.openai.com"),
        ("xai", providers.XAIProvider, "https://api.x.ai/v1", "api.x.ai"),
    ],
)
def test_each_hosted_preset_is_its_provider_at_its_official_url(home, name, cls, url, host) -> None:
    p = providers.make(AssistantSettings(name, "m"))
    assert type(p) is cls and p.name == name
    assert (p.url, p.host, p.local) == (url, host, False)
    assert providers.consent_key(p) == f"assistant-v3:{name}@{host}"
    assert providers.key_host(name, "") == host and keys.official(name, host)
    assert providers.is_preset(name, "") and providers.is_preset(name, url)


@pytest.mark.parametrize(
    ("provider_id", "expected_host"),
    [
        ("claude-code", "api.anthropic.com"),
        ("codex", "chatgpt.com"),
        ("antigravity", "daily-cloudcode-pa.googleapis.com"),
    ],
)
def test_constructing_subscription_providers_from_settings(
    home, provider_id, expected_host
) -> None:
    p = providers.make(AssistantSettings(provider=provider_id))
    assert isinstance(p, providers.SubscriptionProvider)
    assert p.name == provider_id
    assert p.host == expected_host
    assert p.local is False
    assert providers.consent_key(p) == f"assistant-v3:{provider_id}@{expected_host}"


def test_the_none_preset_is_the_configured_ollama(home, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11999")
    p = providers.make(AssistantSettings("ollama", "llama3"))
    assert type(p) is providers.OllamaProvider and p.url == "http://127.0.0.1:11999"
    assert p.model == "llama3" and p.local
    assert providers.make(AssistantSettings("ollama")).model == providers.DEFAULT_OLLAMA_MODEL
    assert providers.is_preset("ollama", "") and providers.is_preset("ollama", "127.0.0.1:11999")
    assert not providers.is_preset("ollama", "http://nas:11434")


@pytest.mark.parametrize(
    ("name", "url", "preset"),
    [
        ("openai", "https://api.openai.com", True),  # a bare host gets /v1
        ("openai", "https://API.openai.com/v1/", True),
        ("openai", "https://openrouter.ai/api/v1", False),
        ("openai", "https://api.x.ai/v1", False),  # Grok through the OpenAI entry: custom
        ("anthropic", "https://api.anthropic.com/", True),
        ("anthropic", "https://proxy.example.com", False),
        ("xai", "https://api.x.ai", True),
        ("xai", "http://127.0.0.1:8080/v1", False),
    ],
)
def test_a_url_other_than_the_presets_is_custom(name, url, preset) -> None:
    assert providers.is_preset(name, url) is preset


# -- the xAI key -------------------------------------------------------------------------------


def test_grok_sends_the_xai_key_to_api_x_ai_only(home, monkeypatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", XKEY)
    set_config(home, assistant={"provider": "xai", "model": "grok-x"})
    wire = Wire(monkeypatch, oa_reply("ok"))
    p = providers.make()
    assert providers.describe(p)["label"] == "xAI · grok-x · api.x.ai"
    p.chat([{"role": "user", "content": "hi"}])
    sent = wire.sent[0]
    assert sent["url"] == "https://api.x.ai/v1/chat/completions"
    assert sent["headers"]["Authorization"] == f"Bearer {XKEY}" and not sent["allow_redirects"]
    # the OpenAI entry pointed at api.x.ai (a custom URL) never gets the xAI key
    set_config(home, assistant={"provider": "openai", "model": "m", "url": "https://api.x.ai/v1"})
    with pytest.raises(providers.ProviderError, match="No API key") as err:
        providers.make().chat([{"role": "user", "content": "hi"}])
    assert "XAI_API_KEY" not in str(err.value) and len(wire.sent) == 1


def test_a_missing_grok_key_names_the_xai_variable(home) -> None:
    p = providers.make(AssistantSettings("xai", "grok-x"))
    why = providers.missing_key(p) or ""
    assert "No API key for xAI" in why and "set XAI_API_KEY before" in why
    assert "CUTPOINT_ASSISTANT_API_KEY" not in why


# -- Settings routes: presets round-trip, custom kept, model lists -----------------------------


def _ui():
    pytest.importorskip("httpx")
    from cutpoint import ui  # noqa: PLC0415

    return ui


@pytest.fixture
def checked(monkeypatch: pytest.MonkeyPatch) -> list[providers.Provider]:
    """The provider Save checks, recorded instead of asked (no request)."""
    seen: list[providers.Provider] = []
    monkeypatch.setattr(providers, "check", lambda p: seen.append(p))
    return seen


def saved(home: Path) -> dict:
    return json.loads((home / "config.json").read_text("utf-8"))["assistant"]


@pytest.mark.parametrize(
    ("name", "model", "cls", "url"),
    [
        ("ollama", "llama3.2:3b", providers.OllamaProvider, "http://127.0.0.1:11434"),
        ("ollama", "", providers.OllamaProvider, "http://127.0.0.1:11434"),
        ("anthropic", "claude-opus-5-5", providers.AnthropicProvider, providers.ANTHROPIC_URL),
        ("openai", "my-openai-model", providers.OpenAIProvider, providers.OPENAI_URL),
        ("xai", "my-grok-model", providers.XAIProvider, providers.XAI_URL),
    ],
)
def test_each_preset_round_trips_through_settings(
    ui_client: TestClient, home: Path, checked, name, model, cls, url
) -> None:
    # Save checks only then (#504)
    agreed = [f"assistant-v3:{n}@{h}" for n, h in keys.OFFICIAL_HOST.items()]
    set_config(
        home,
        assistant={"provider": "openai", "url": "https://openrouter.ai/api/v1", "consent": agreed},
    )
    assert ui_client.get("/api/settings").json()["assistant"]["custom"] is True
    body = {"assistant": {"provider": name, "model": model, "url": ""}}
    res = ui_client.put("/api/settings", json=body)
    assert res.status_code == 200, res.text
    want = {"provider": name, "model": model, "url": "", "consent": agreed}
    assert saved(home) == want  # the URL is the preset's
    got = ui_client.get("/api/settings").json()["assistant"]
    assert (got["provider"], got["model"], got["url"], got["custom"]) == (name, model, "", False)
    p = checked[-1]
    assert type(p) is cls and p.url == url


def test_settings_offer_the_presets_with_their_model_fields(ui_client: TestClient) -> None:
    got = ui_client.get("/api/settings").json()["assistant"]
    offers = {p["name"]: p for p in got["providers"]}
    assert [p["name"] for p in got["providers"]] == [
        "ollama",
        "anthropic",
        "openai",
        "xai",
        "claude-code",
        "codex",
        "antigravity",
    ]
    assert offers["ollama"]["placeholder"] == "qwen3.5:9b" and not offers["ollama"]["key_env"]
    assert offers["anthropic"]["placeholder"] == "claude-sonnet-5-5"
    assert offers["anthropic"]["models"] == ["claude-sonnet-5-5", "claude-opus-5-5"]
    for name, env, own in (
        ("openai", "OPENAI_API_KEY", providers.OPENAI_MODELS),
        ("xai", "XAI_API_KEY", providers.XAI_MODELS),
    ):  # offered when no list came; still no default, so "Choose a model" comes first
        assert offers[name]["model_required"] is True and offers[name]["models"] == list(own)
        assert offers[name]["key_env"] == [env]  # the shared key is not for it (#504)
    assert not offers["ollama"]["model_required"] and not offers["anthropic"]["model_required"]
    assert got["key_hosts"]["xai"] == "api.x.ai"


def test_every_settings_provider_has_an_offer_in_the_provider_module() -> None:
    """A419: the provider table lives beside PRESETS, one Offer per preset and subscription."""
    got = providers.offers("http://box:11434", "some-vl:7b")
    assert set(got) == set(providers.PRESETS) | set(providers.SUBSCRIPTION_PROVIDERS)
    assert got["ollama"] == ("http://box:11434", "some-vl:7b", ["some-vl:7b"], "some-vl:7b", False)
    assert got["codex"].model == providers.DEFAULT_CODEX_MODEL and got["codex"].url == ""


@pytest.mark.parametrize(
    "section",
    [
        {"provider": "openai", "model": "m", "url": "https://openrouter.ai/api/v1"},
        {"provider": "ollama", "model": "q", "url": "http://nas:11434"},
        {"provider": "anthropic", "url": "https://proxy.example.com"},
    ],
)
def test_a_legacy_custom_url_is_shown_as_custom_and_kept(
    ui_client: TestClient, home: Path, monkeypatch, section
) -> None:
    set_config(home, assistant=section)
    got = ui_client.get("/api/settings").json()["assistant"]
    assert got["custom"] is True and got["url"] == section["url"]
    assert got["provider"] == section["provider"]
    p = providers.make()
    assert p.host == providers.host_of(section["url"])  # still the URL in use
    res = ui_client.put("/api/settings", json={"vision": True})  # another setting: kept as is
    assert res.status_code == 200 and saved(home) == section
    monkeypatch.setattr(providers, "check", lambda p: None)
    res = ui_client.put("/api/settings", json={"assistant": {"model": "other-model"}})  # Custom
    assert res.status_code == 200 and saved(home) == {**section, "model": "other-model"}
    assert res.json()["assistant"]["custom"] is True


def test_custom_vision_url_is_shown_as_custom_and_kept(ui_client: TestClient, home: Path) -> None:
    section = {"provider": "ollama", "url": "http://gpu-box:11434", "model": "llava:7b"}
    set_config(home, vision_assistant=section)
    got = ui_client.get("/api/settings").json()["vision_assistant"]
    assert got["custom"] is True
    assert got["url"] == "http://gpu-box:11434"
    assert got["provider"] == "ollama"

    res = ui_client.put("/api/settings", json={"vision_assistant": {"model": "llava:13b"}})
    assert res.status_code == 200
    out = res.json()["vision_assistant"]
    assert out["custom"] is True
    assert out["url"] == "http://gpu-box:11434"
    assert out["model"] == "llava:13b"
    assert out["provider"] == "ollama"
    cfg = json.loads((home / "config.json").read_text("utf-8"))["vision_assistant"]
    assert cfg["url"] == "http://gpu-box:11434"
    assert cfg["model"] == "llava:13b"


def test_chat_save_does_not_create_vision_assistant_on_legacy_config(
    ui_client: TestClient, home: Path, monkeypatch
) -> None:
    set_config(home, assistant={"provider": "ollama", "model": "llama3.2"})
    monkeypatch.setattr(providers, "check", lambda p: None)
    res = ui_client.put(
        "/api/settings",
        json={"assistant": {"provider": "ollama", "model": "llama3.3", "url": ""}},
    )
    assert res.status_code == 200
    cfg = json.loads((home / "config.json").read_text("utf-8"))
    assert "vision_assistant" not in cfg


def test_the_shared_key_can_be_left_out_of_a_lookup(monkeypatch) -> None:
    monkeypatch.setenv("CUTPOINT_ASSISTANT_API_KEY", KEY)
    assert keys.lookup("xai", "proxy.example") == (KEY, "env")
    assert keys.lookup("xai", "proxy.example", shared=False) == (None, None)
    assert keys.lookup("xai", "api.x.ai", shared=False) == (None, None)
    monkeypatch.setenv("XAI_API_KEY", XKEY)
    assert keys.lookup("xai", "api.x.ai", shared=False) == (XKEY, "env")
    assert keys.lookup("xai", "proxy.example", shared=False) == (None, None)


def test_a_custom_url_keeps_its_key_host(ui_client: TestClient, home: Path, monkeypatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", XKEY)
    set_config(home, assistant={"provider": "xai", "model": "g", "url": "https://proxy.example"})
    got = ui_client.get("/api/settings").json()["assistant"]
    xai = next(p for p in got["providers"] if p["name"] == "xai")
    assert got["custom"] and got["key_hosts"]["xai"] == "proxy.example"
    assert xai["key_env"] == ["CUTPOINT_ASSISTANT_API_KEY"] and got["keys"]["xai"] is None


# -- model lists ------------------------------------------------------------------------------


def models(ui_client: TestClient, provider: str, **params) -> dict:
    res = ui_client.get("/api/assistant/models", params={"provider": provider, **params})
    assert res.status_code == 200, res.text
    return res.json()


@pytest.mark.parametrize("name", ["anthropic", "openai", "xai"])
def test_no_hosted_model_list_is_asked_without_a_key_or_consent(
    ui_client: TestClient, home: Path, monkeypatch, name
) -> None:
    wire = Wire(monkeypatch)  # any request fails the test
    got = models(ui_client, name)
    assert got["state"] == "key_needed" and got["models"] == [] and "No API key" in got["message"]
    monkeypatch.setenv(keys.PROVIDER_ENV[name], KEY)
    got = models(ui_client, name)
    assert got["state"] == "consent_needed" and got["models"] == []
    # another host's consent
    set_config(home, assistant={"consent": [f"assistant-v3:{name}@other.example"]})
    assert models(ui_client, name)["state"] == "consent_needed"
    assert wire.sent == []


def test_a_preset_not_saved_never_gets_the_shared_key(
    ui_client: TestClient, home: Path, monkeypatch
) -> None:
    """CUTPOINT_ASSISTANT_API_KEY may be the saved provider's (an OpenRouter key): choosing
    another preset in Settings never sends it to that preset's host."""
    monkeypatch.setenv("CUTPOINT_ASSISTANT_API_KEY", KEY)
    set_config(
        home,
        assistant={
            "provider": "openai",
            "url": "https://openrouter.ai/api/v1",
            "consent": [
                "assistant-v3:anthropic@api.anthropic.com",
                "assistant-v3:openai@api.openai.com",
                "assistant-v3:xai@api.x.ai",
            ],
        },
    )
    wire = Wire(monkeypatch)  # any request fails the test
    shared_only = "CUTPOINT_ASSISTANT_API_KEY is only for custom base URLs"
    for name in ("anthropic", "openai", "xai"):  # openai too: the saved URL is custom
        got = models(ui_client, name)
        assert got["state"] == "key_needed" and got["models"] == [], name
        assert keys.PROVIDER_ENV[name] in got["message"] and shared_only in got["message"]
    set_config(
        home,
        assistant={"provider": "ollama", "consent": ["assistant-v3:anthropic@api.anthropic.com"]},
    )
    assert shared_only in models(ui_client, "anthropic")["message"]  # Ollama saved, Claude picked
    monkeypatch.delenv("CUTPOINT_ASSISTANT_API_KEY")
    got = models(ui_client, "anthropic")  # no key at all: how to add one
    assert got["state"] == "key_needed" and "ANTHROPIC_API_KEY" in got["message"]
    assert wire.sent == []
    monkeypatch.setenv("CUTPOINT_ASSISTANT_API_KEY", KEY)
    assert not providers.preset("anthropic").has_key


def test_the_saved_preset_never_gets_the_shared_key_but_its_own(
    ui_client: TestClient, home: Path, monkeypatch
) -> None:
    """#504: the saved, agreed-to official host never gets CUTPOINT_ASSISTANT_API_KEY, even
    as a fallback; its own key wins."""
    monkeypatch.setenv("CUTPOINT_ASSISTANT_API_KEY", KEY)
    set_config(
        home,
        assistant={
            "provider": "anthropic",
            "consent": ["assistant-v3:anthropic@api.anthropic.com"],
        },
    )
    wire = Wire(monkeypatch, _Resp({"data": [{"id": "claude-z"}], "has_more": False}))
    assert models(ui_client, "anthropic")["state"] == "key_needed" and wire.sent == []
    monkeypatch.setenv("ANTHROPIC_API_KEY", OTHER)
    assert models(ui_client, "anthropic")["models"] == [{"id": "claude-z", "name": "claude-z"}]
    assert wire.sent[0]["headers"]["x-api-key"] == OTHER
    assert KEY not in json.dumps(wire.sent)


def test_the_anthropic_list_reads_every_page(
    ui_client: TestClient, home: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    set_config(home, assistant={"consent": ["assistant-v3:anthropic@api.anthropic.com"]})
    page1 = {
        "data": [{"id": "claude-a", "display_name": "Claude A"}, {"id": "claude-b"}],
        "has_more": True,
        "last_id": "claude-b",
    }
    page2 = {"data": [{"id": "claude-c", "display_name": "Claude C"}], "has_more": False}
    wire = Wire(monkeypatch, _Resp(page1), _Resp(page2))
    res = ui_client.get("/api/assistant/models", params={"provider": "anthropic"})
    got = res.json()
    assert got["state"] == "ok"
    assert got["models"] == [
        {"id": "claude-a", "name": "Claude A"},
        {"id": "claude-b", "name": "claude-b"},
        {"id": "claude-c", "name": "Claude C"},
    ]
    first, second = wire.sent
    assert first["url"] == "https://api.anthropic.com/v1/models?limit=1000"
    assert second["url"] == "https://api.anthropic.com/v1/models?limit=1000&after_id=claude-b"
    for sent in wire.sent:
        assert sent["headers"] == {"anthropic-version": "2023-06-01", "x-api-key": KEY}
        assert sent["allow_redirects"] is False and sent["timeout"][1] <= providers.CHECK_TIMEOUT_S
    assert KEY not in res.text


def test_the_openai_list_leaves_out_models_that_do_not_chat(
    ui_client: TestClient, home: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    set_config(home, assistant={"consent": ["assistant-v3:openai@api.openai.com"]})
    ids = [
        "text-embedding-3-small",
        "tts-1-hd",
        "whisper-1",
        "dall-e-3",
        "omni-moderation-latest",
        "text-moderation-stable",
        "gpt-image-1",
        "sora-2",
        "davinci-002",
        "babbage-002",
        "chat-x-tts",
        "chat-x-transcribe",
        "chat-x-realtime-preview",
        "chat-x-audio-preview",
        "chat-x-codex",
        "codex-mini-latest",
        "model-b",
        "model-a",
    ]
    wire = Wire(monkeypatch, _Resp({"data": [{"id": i} for i in ids]}))
    got = models(ui_client, "openai")  # *-codex chats over the Responses API (#536)
    assert [m["id"] for m in got["models"]] == [
        "chat-x-codex",
        "codex-mini-latest",
        "model-a",
        "model-b",
    ]
    assert wire.sent[0]["url"] == "https://api.openai.com/v1/models"
    assert wire.sent[0]["headers"] == {"Authorization": f"Bearer {KEY}"}


def test_the_xai_list_is_asked_at_api_x_ai_with_the_xai_key(
    ui_client: TestClient, home: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XAI_API_KEY", XKEY)
    set_config(home, assistant={"consent": ["assistant-v3:xai@api.x.ai"]})
    rows = [{"id": i} for i in ("grok-one", "grok-x-image", "grok-imagine-video", "grok-codex")]
    wire = Wire(monkeypatch, _Resp({"data": rows}))  # xAI's image and codex models are left out too
    res = ui_client.get("/api/assistant/models", params={"provider": "xai"})
    assert res.json()["models"] == [{"id": "grok-one", "name": "grok-one"}]
    assert wire.sent[0]["url"] == "https://api.x.ai/v1/models"
    assert wire.sent[0]["headers"] == {"Authorization": f"Bearer {XKEY}"}
    assert XKEY not in res.text


def test_the_ollama_list_is_its_pulled_models(ui_client: TestClient, monkeypatch) -> None:
    tags = {"models": [{"name": "qwen3.5:9b"}, {"name": "llama3.2:3b"}, {"size": 1}]}
    wire = Wire(monkeypatch, _Resp(tags))
    got = models(ui_client, "ollama")  # this machine: no key, no consent
    assert got == {
        "state": "ok",
        "models": [
            {"id": "qwen3.5:9b", "name": "qwen3.5:9b"},
            {"id": "llama3.2:3b", "name": "llama3.2:3b"},
        ],
        "message": None,
    }
    assert wire.sent[0]["url"] == "http://127.0.0.1:11434/api/tags"
    assert wire.sent[0]["allow_redirects"] is False


def test_a_list_is_kept_ten_minutes_unless_refreshed(ui_client: TestClient, monkeypatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(providers, "_clock", lambda: now[0])
    wire = Wire(
        monkeypatch,
        _Resp({"models": [{"name": "a"}]}),
        _Resp({"models": [{"name": "b"}]}),
        _Resp({"models": [{"name": "c"}]}),
    )
    assert models(ui_client, "ollama")["models"][0]["id"] == "a"
    now[0] += providers.MODELS_TTL_S - 1
    assert models(ui_client, "ollama")["models"][0]["id"] == "a" and len(wire.sent) == 1
    assert models(ui_client, "ollama", refresh=1)["models"][0]["id"] == "b"
    now[0] += providers.MODELS_TTL_S + 1
    assert models(ui_client, "ollama")["models"][0]["id"] == "c" and len(wire.sent) == 3


def test_a_failed_list_says_why_never_the_key(
    ui_client: TestClient, home: Path, monkeypatch, caplog
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", OTHER)
    set_config(home, assistant={"consent": ["assistant-v3:anthropic@api.anthropic.com"]})
    answer = _Resp({"error": {"message": f"invalid x-api-key {OTHER}"}}, status=401)
    Wire(monkeypatch, answer, _Resp({"data": [{"id": "claude-z"}]}))
    caplog.set_level(logging.DEBUG)
    res = ui_client.get("/api/assistant/models", params={"provider": "anthropic"})
    got = res.json()
    assert got["state"] == "error" and got["models"] == []
    assert "rejected the API key" in got["message"] and "[redacted]" in got["message"]
    assert OTHER not in res.text and OTHER not in caplog.text
    assert models(ui_client, "anthropic")["models"] == [{"id": "claude-z", "name": "claude-z"}]


def test_an_unreachable_ollama_is_an_error_state(ui_client: TestClient, monkeypatch) -> None:
    import requests  # noqa: PLC0415

    Wire(monkeypatch, requests.ConnectionError("refused"))
    got = models(ui_client, "ollama")
    assert got["state"] == "error" and "not reachable" in got["message"]


def test_the_models_route_takes_presets_or_custom_and_the_token(
    ui_client: TestClient, home: Path, monkeypatch
) -> None:
    assert ui_client.get("/api/assistant/models", params={"provider": "gemini"}).status_code == 400
    bare = ui_client.get("/api/assistant/models", headers={"X-Cutpoint-Token": "wrong"})
    assert bare.status_code == 403
    set_config(home, assistant={"provider": "openai", "url": "http://127.0.0.1:1234/v1"})
    wire = Wire(monkeypatch, _Resp({"data": [{"id": "local-model"}]}))
    got = models(ui_client, "custom")  # the configured server on this machine: no key needed
    assert got["models"] == [{"id": "local-model", "name": "local-model"}]
    assert wire.sent[0]["url"] == "http://127.0.0.1:1234/v1/models"
