"""Settings' assistant presets (None/Ollama, Claude, Codex, Grok), the xAI key and model lists.

No request reaches a real provider: ``requests`` is replaced by :class:`Wire` in every test
that could send one.
"""

from __future__ import annotations

import io
import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from conftest import set_config
from cutpoint import config
from cutpoint.assistant import keys, providers
from cutpoint.config import AssistantSettings
from test_assistant import _Resp
from test_assistant_byo import KEY, OTHER, NoRing, Wire, oa_reply

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

XKEY = "xai-0123456789abcdefghijSECRET"  # shaped like an xAI key


@pytest.fixture(autouse=True)
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real key variable, keyring or cached model list reaches a test."""
    for name in keys.ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(keys, "_keyring", NoRing)
    monkeypatch.setattr(keys, "_SEEN", set())
    monkeypatch.setattr(keys, "_CAPTURED", {})
    monkeypatch.setattr(providers, "_MODELS", {})


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
    assert providers.consent_key(p) == f"{name}@{host}"
    assert providers.key_host(name, "") == host and keys.official(name, host)
    assert providers.is_preset(name, "") and providers.is_preset(name, url)


def test_the_none_preset_is_the_configured_ollama(home, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11999")
    p = providers.make(AssistantSettings("ollama", "llama3"))
    assert type(p) is providers.OllamaProvider and p.url == "http://127.0.0.1:11999"
    assert p.model == "llama3" and p.local
    assert providers.make(AssistantSettings("ollama")).model == providers.AUTO_MODEL
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


def test_xai_is_a_provider_in_config(home) -> None:
    set_config(home, assistant={"provider": "xai", "model": "grok-x"})
    assert config.assistant_settings() == AssistantSettings("xai", "grok-x")
    assert "xai" in config.ASSISTANT_PROVIDERS and list(providers.PRESETS) == [
        "ollama",
        "anthropic",
        "openai",
        "xai",
    ]


# -- the xAI key -------------------------------------------------------------------------------


def test_the_xai_variable_is_only_for_api_x_ai(monkeypatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", XKEY)
    assert keys.lookup("xai", "api.x.ai") == (XKEY, "env")
    assert keys.lookup("xai", "API.X.AI") == (XKEY, "env")
    assert keys.lookup("xai", "proxy.example.com") == (None, None)
    assert keys.lookup("openai", "api.x.ai") == (None, None)  # another provider's entry
    assert keys.lookup("openai", "api.openai.com") == (None, None)
    assert keys.lookup("anthropic", "api.anthropic.com") == (None, None)
    monkeypatch.setenv("OPENAI_API_KEY", OTHER)
    assert keys.lookup("xai", "api.x.ai") == (XKEY, "env")  # never the OpenAI key
    monkeypatch.setenv("CUTPOINT_ASSISTANT_API_KEY", KEY)
    assert keys.lookup("xai", "api.x.ai") == (KEY, "env")  # the Cutpoint one, for any host


def test_an_xai_key_is_saved_in_the_keyring_under_xai(monkeypatch) -> None:
    from test_assistant_byo import FakeRing  # noqa: PLC0415

    ring = FakeRing()
    monkeypatch.setattr(keys, "_keyring", lambda: ring)
    keys.save("xai", OTHER, "api.x.ai")
    keys.save("xai", KEY, "proxy.example.com")
    assert ring.saved == {("cutpoint", "xai"): OTHER, ("cutpoint", "xai@proxy.example.com"): KEY}
    assert keys.lookup("xai", "api.x.ai") == (OTHER, "keyring")


def test_saving_an_xai_key_without_the_extra_names_its_variable() -> None:
    with pytest.raises(keys.KeyringError, match=r"XAI_API_KEY"):
        keys.save("xai", OTHER, "api.x.ai")


def test_the_xai_variable_never_reaches_a_child_process(monkeypatch) -> None:
    assert "XAI_API_KEY" in keys.ENV_VARS
    env = {"PATH": "/bin", "XAI_API_KEY": "a", "xai_api_key": "b", "XAI_OTHER": "c"}
    assert keys.child_env(env) == {"PATH": "/bin", "XAI_OTHER": "c"}
    monkeypatch.setenv("XAI_API_KEY", XKEY)
    keys.capture()
    try:
        import os  # noqa: PLC0415

        assert "XAI_API_KEY" not in os.environ and "XAI_API_KEY" not in keys.child_env()
        assert keys.lookup("xai", "api.x.ai") == (XKEY, "env")  # still found after capture
    finally:
        keys._CAPTURED.clear()


def test_an_xai_key_is_redacted_from_messages_and_logs(monkeypatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", OTHER)
    assert keys.redact(f"bad key {OTHER}") == "bad key [redacted]"  # read from the variable
    assert keys.redact(f"echo {XKEY} back") == "echo [redacted] back"  # shaped like one
    assert keys.redact("xai-short stays") == "xai-short stays"
    out = io.StringIO()
    handler = logging.StreamHandler(out)
    handler.addFilter(keys._FILTER)
    log = logging.getLogger("cutpoint.test.xai")
    log.addHandler(handler)
    try:
        log.error("sent %s and %s", OTHER, XKEY)
    finally:
        log.removeHandler(handler)
    assert out.getvalue().strip() == "sent [redacted] and [redacted]"


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
    assert "No API key for xAI" in why and "XAI_API_KEY (or CUTPOINT_ASSISTANT_API_KEY)" in why


# -- Settings routes: presets round-trip, custom kept, model lists -----------------------------

PORT = 8765


def _ui():
    pytest.importorskip("httpx")
    from cutpoint import ui  # noqa: PLC0415

    return ui


@pytest.fixture
def client() -> Iterator[TestClient]:
    from fastapi.testclient import TestClient  # noqa: PLC0415

    app = _ui().create_app(PORT)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        c.headers["X-Cutpoint-Token"] = app.state.token
        yield c


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
    client: TestClient, home: Path, checked, name, model, cls, url
) -> None:
    set_config(home, assistant={"provider": "openai", "url": "https://openrouter.ai/api/v1"})
    assert client.get("/api/settings").json()["assistant"]["custom"] is True
    body = {"assistant": {"provider": name, "model": model, "url": ""}}
    res = client.put("/api/settings", json=body)
    assert res.status_code == 200, res.text
    assert saved(home) == {"provider": name, "model": model, "url": ""}  # the URL is the preset's
    got = client.get("/api/settings").json()["assistant"]
    assert (got["provider"], got["model"], got["url"], got["custom"]) == (name, model, "", False)
    p = checked[-1]
    assert type(p) is cls and p.url == url


def test_settings_offer_the_presets_with_their_model_fields(client: TestClient) -> None:
    got = client.get("/api/settings").json()["assistant"]
    offers = {p["name"]: p for p in got["providers"]}
    assert [p["title"] for p in got["providers"]] == [
        "None (local Ollama)",
        "Claude (Anthropic)",
        "Codex (OpenAI)",
        "Grok (xAI)",
    ]
    assert offers["ollama"]["placeholder"] == "qwen3.5:9b" and not offers["ollama"]["key_env"]
    assert offers["anthropic"]["placeholder"] == "claude-sonnet-5-5"
    assert offers["anthropic"]["models"] == ["claude-sonnet-5-5", "claude-opus-5-5"]
    for name, env in (("openai", "OPENAI_API_KEY"), ("xai", "XAI_API_KEY")):
        assert offers[name]["model_required"] is True and offers[name]["models"] == []
        assert offers[name]["key_env"] == ["CUTPOINT_ASSISTANT_API_KEY", env]
    assert not offers["ollama"]["model_required"] and not offers["anthropic"]["model_required"]
    assert got["key_hosts"]["xai"] == "api.x.ai"


@pytest.mark.parametrize(
    "section",
    [
        {"provider": "openai", "model": "m", "url": "https://openrouter.ai/api/v1"},
        {"provider": "ollama", "model": "q", "url": "http://nas:11434"},
        {"provider": "anthropic", "url": "https://proxy.example.com"},
    ],
)
def test_a_legacy_custom_url_is_shown_as_custom_and_kept(
    client: TestClient, home: Path, monkeypatch, section
) -> None:
    set_config(home, assistant=section)
    got = client.get("/api/settings").json()["assistant"]
    assert got["custom"] is True and got["url"] == section["url"]
    assert got["provider"] == section["provider"]
    p = providers.make()
    assert p.host == providers.host_of(section["url"])  # still the URL in use
    res = client.put("/api/settings", json={"vision": True})  # another setting: kept as is
    assert res.status_code == 200 and saved(home) == section
    monkeypatch.setattr(providers, "check", lambda p: None)
    res = client.put("/api/settings", json={"assistant": {"model": "other-model"}})  # Custom
    assert res.status_code == 200 and saved(home) == {**section, "model": "other-model"}
    assert res.json()["assistant"]["custom"] is True


def test_the_shared_key_can_be_left_out_of_a_lookup(monkeypatch) -> None:
    monkeypatch.setenv("CUTPOINT_ASSISTANT_API_KEY", KEY)
    assert keys.lookup("xai", "api.x.ai") == (KEY, "env")
    assert keys.lookup("xai", "api.x.ai", shared=False) == (None, None)
    monkeypatch.setenv("XAI_API_KEY", XKEY)
    assert keys.lookup("xai", "api.x.ai", shared=False) == (XKEY, "env")
    assert keys.lookup("xai", "proxy.example", shared=False) == (None, None)


def test_a_custom_url_keeps_its_key_host(client: TestClient, home: Path, monkeypatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", XKEY)
    set_config(home, assistant={"provider": "xai", "model": "g", "url": "https://proxy.example"})
    got = client.get("/api/settings").json()["assistant"]
    xai = next(p for p in got["providers"] if p["name"] == "xai")
    assert got["custom"] and got["key_hosts"]["xai"] == "proxy.example"
    assert xai["key_env"] == ["CUTPOINT_ASSISTANT_API_KEY"] and got["keys"]["xai"] is None


# -- model lists ------------------------------------------------------------------------------


def models(client: TestClient, provider: str, **params) -> dict:
    res = client.get("/api/assistant/models", params={"provider": provider, **params})
    assert res.status_code == 200, res.text
    return res.json()


@pytest.mark.parametrize("name", ["anthropic", "openai", "xai"])
def test_no_hosted_model_list_is_asked_without_a_key_or_consent(
    client: TestClient, home: Path, monkeypatch, name
) -> None:
    wire = Wire(monkeypatch)  # any request fails the test
    got = models(client, name)
    assert got["state"] == "key_needed" and got["models"] == [] and "No API key" in got["message"]
    monkeypatch.setenv(keys.PROVIDER_ENV[name], KEY)
    got = models(client, name)
    assert got["state"] == "consent_needed" and got["models"] == []
    set_config(home, assistant={"consent": [f"{name}@other.example"]})  # another host's consent
    assert models(client, name)["state"] == "consent_needed"
    assert wire.sent == []


def test_a_preset_not_saved_never_gets_the_shared_key(
    client: TestClient, home: Path, monkeypatch
) -> None:
    """CUTPOINT_ASSISTANT_API_KEY may be the saved provider's (an OpenRouter key): choosing
    another preset in Settings never sends it to that preset's host."""
    monkeypatch.setenv("CUTPOINT_ASSISTANT_API_KEY", KEY)
    set_config(
        home,
        assistant={
            "provider": "openai",
            "url": "https://openrouter.ai/api/v1",
            "consent": ["anthropic@api.anthropic.com", "openai@api.openai.com", "xai@api.x.ai"],
        },
    )
    wire = Wire(monkeypatch)  # any request fails the test
    # Settings shows the shared key as "set (env)": the list says why it is not used yet
    shared_only = (
        "The model list loads once this provider is saved: "
        "CUTPOINT_ASSISTANT_API_KEY is used only for the saved provider."
    )
    for name in ("anthropic", "openai", "xai"):  # openai too: the saved URL is custom
        got = models(client, name)
        assert got == {"state": "key_needed", "models": [], "message": shared_only}, name
    set_config(home, assistant={"provider": "ollama", "consent": ["anthropic@api.anthropic.com"]})
    assert models(client, "anthropic")["message"] == shared_only  # Ollama saved, Claude picked
    monkeypatch.delenv("CUTPOINT_ASSISTANT_API_KEY")
    got = models(client, "anthropic")  # no key at all: how to add one
    assert got["state"] == "key_needed" and "ANTHROPIC_API_KEY" in got["message"]
    assert wire.sent == []
    monkeypatch.setenv("CUTPOINT_ASSISTANT_API_KEY", KEY)
    assert not providers.preset("anthropic").has_key


def test_the_saved_preset_still_uses_the_shared_key(
    client: TestClient, home: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CUTPOINT_ASSISTANT_API_KEY", KEY)
    set_config(
        home, assistant={"provider": "anthropic", "consent": ["anthropic@api.anthropic.com"]}
    )
    wire = Wire(monkeypatch, _Resp({"data": [{"id": "claude-z"}], "has_more": False}))
    assert models(client, "anthropic")["models"] == [{"id": "claude-z", "name": "claude-z"}]
    assert wire.sent[0]["headers"]["x-api-key"] == KEY


def test_the_anthropic_list_reads_every_page(client: TestClient, home: Path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    set_config(home, assistant={"consent": ["anthropic@api.anthropic.com"]})
    page1 = {
        "data": [{"id": "claude-a", "display_name": "Claude A"}, {"id": "claude-b"}],
        "has_more": True,
        "last_id": "claude-b",
    }
    page2 = {"data": [{"id": "claude-c", "display_name": "Claude C"}], "has_more": False}
    wire = Wire(monkeypatch, _Resp(page1), _Resp(page2))
    res = client.get("/api/assistant/models", params={"provider": "anthropic"})
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
    client: TestClient, home: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    set_config(home, assistant={"consent": ["openai@api.openai.com"]})
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
    got = models(client, "openai")
    assert [m["id"] for m in got["models"]] == ["model-a", "model-b"]
    assert wire.sent[0]["url"] == "https://api.openai.com/v1/models"
    assert wire.sent[0]["headers"] == {"Authorization": f"Bearer {KEY}"}


def test_the_xai_list_is_asked_at_api_x_ai_with_the_xai_key(
    client: TestClient, home: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XAI_API_KEY", XKEY)
    set_config(home, assistant={"consent": ["xai@api.x.ai"]})
    rows = [{"id": i} for i in ("grok-one", "grok-x-image", "grok-imagine-video")]
    wire = Wire(monkeypatch, _Resp({"data": rows}))  # xAI's image models are left out too
    res = client.get("/api/assistant/models", params={"provider": "xai"})
    assert res.json()["models"] == [{"id": "grok-one", "name": "grok-one"}]
    assert wire.sent[0]["url"] == "https://api.x.ai/v1/models"
    assert wire.sent[0]["headers"] == {"Authorization": f"Bearer {XKEY}"}
    assert XKEY not in res.text


def test_the_ollama_list_is_its_pulled_models(client: TestClient, monkeypatch) -> None:
    tags = {"models": [{"name": "qwen3.5:9b"}, {"name": "llama3.2:3b"}, {"size": 1}]}
    wire = Wire(monkeypatch, _Resp(tags))
    got = models(client, "ollama")  # this machine: no key, no consent
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


def test_a_list_is_kept_ten_minutes_unless_refreshed(client: TestClient, monkeypatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(providers, "_clock", lambda: now[0])
    wire = Wire(
        monkeypatch,
        _Resp({"models": [{"name": "a"}]}),
        _Resp({"models": [{"name": "b"}]}),
        _Resp({"models": [{"name": "c"}]}),
    )
    assert models(client, "ollama")["models"][0]["id"] == "a"
    now[0] += providers.MODELS_TTL_S - 1
    assert models(client, "ollama")["models"][0]["id"] == "a" and len(wire.sent) == 1
    assert models(client, "ollama", refresh=1)["models"][0]["id"] == "b"
    now[0] += providers.MODELS_TTL_S + 1
    assert models(client, "ollama")["models"][0]["id"] == "c" and len(wire.sent) == 3


def test_a_failed_list_says_why_never_the_key(
    client: TestClient, home: Path, monkeypatch, caplog
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", OTHER)
    set_config(home, assistant={"consent": ["anthropic@api.anthropic.com"]})
    answer = _Resp({"error": {"message": f"invalid x-api-key {OTHER}"}}, status=401)
    Wire(monkeypatch, answer, _Resp({"data": [{"id": "claude-z"}]}))
    caplog.set_level(logging.DEBUG)
    res = client.get("/api/assistant/models", params={"provider": "anthropic"})
    got = res.json()
    assert got["state"] == "error" and got["models"] == []
    assert "rejected the API key" in got["message"] and "[redacted]" in got["message"]
    assert OTHER not in res.text and OTHER not in caplog.text
    assert models(client, "anthropic")["models"] == [{"id": "claude-z", "name": "claude-z"}]


def test_an_unreachable_ollama_is_an_error_state(client: TestClient, monkeypatch) -> None:
    import requests  # noqa: PLC0415

    Wire(monkeypatch, requests.ConnectionError("refused"))
    got = models(client, "ollama")
    assert got["state"] == "error" and "not reachable" in got["message"]


def test_the_models_route_takes_presets_or_custom_and_the_token(
    client: TestClient, home: Path, monkeypatch
) -> None:
    assert client.get("/api/assistant/models", params={"provider": "gemini"}).status_code == 400
    bare = client.get("/api/assistant/models", headers={"X-Cutpoint-Token": "wrong"})
    assert bare.status_code == 403
    set_config(home, assistant={"provider": "openai", "url": "http://127.0.0.1:1234/v1"})
    wire = Wire(monkeypatch, _Resp({"data": [{"id": "local-model"}]}))
    got = models(client, "custom")  # the configured server on this machine: no key needed
    assert got["models"] == [{"id": "local-model", "name": "local-model"}]
    assert wire.sent[0]["url"] == "http://127.0.0.1:1234/v1/models"


# -- the Settings dialog (contract pins) ------------------------------------------------------


def _web(name: str) -> str:
    return (_ui().WEB_DIR / name).read_text(encoding="utf-8")


def test_the_settings_dialog_has_a_provider_choice_and_no_base_url_for_presets() -> None:
    html = _web("index.html")
    section = html[
        html.index('id="setAssistant"') : html.index("</fieldset>", html.index('id="setAssistant"'))
    ]
    assert '<select id="setAiProvider"' in section
    assert '<input id="setAiModel" list="setAiModels"' in section  # any name can be typed
    assert '<datalist id="setAiModels">' in section
    assert 'id="setAiModelsRefresh"' in section and 'id="setAiModelsStatus"' in section
    url_field = re.search(
        r'<label class="field" id="setAiUrlField" hidden>.*?</label>', section, re.S
    )
    assert url_field and "readonly" in url_field[0]  # Custom only, never edited here


def test_the_settings_script_shows_fields_by_the_provider_chosen() -> None:
    js = _web("app.js")
    fill = js[js.index("function fillAiSettings(") : js.index("const aiCustom = ")]
    assert (
        "if (a.custom) options.push(h('option', { value: AI_CUSTOM, text: 'Custom (from config.json)' }));"
        in fill
    )
    assert "sel.value = sel.dataset.saved = a.custom ? AI_CUSTOM : a.provider;" in fill
    render = js[js.index("function renderAiFields(") : js.index("async function loadAiModels(")]
    assert "$('#setAiUrlField').hidden = $('#setAiUrlHint').hidden = !custom;" in render
    assert "readOnly" not in render  # a custom provider's model can still be changed
    assert "model.placeholder = custom ? '' : p.placeholder;" in render
    assert "renderAiModelList();" in render
    listing = js[js.index("function renderAiModelList(") : js.index("async function loadAiModels(")]
    assert "const defaults = aiCustom() ? [] : p.models;" in listing  # fallback list
    changes = js[js.index("function aiChanges(") : js.index("/** Save the typed key")]
    # Custom sends only a changed model: its provider and base URL are never rewritten
    assert (
        "if (aiCustom()) return model.value.trim() !== model.dataset.saved"
        " ? { model: model.value.trim() } : null;"
    ) in changes
    assert "url: ''" in changes  # a preset's URL is its own
    load = js[js.index("async function loadAiModels(") : js.index("function aiChanges(")]
    assert "/api/assistant/models?provider=${encodeURIComponent(choice)}" in load
    assert "AI_CUSTOM" not in load  # Custom lists its models too (?provider=custom)
    assert "renderAiModelList(); // only the list" in load and "renderAiFields()" not in load
    assert ".replace(/\\.?$/, '.')} You can type a name." in load
    save = js[js.index("async function saveSettings(") : js.index("// The AI assistant's provider")]
    assert "if (ai && !ai.model && !aiCustom() && aiOffer()?.model_required) {" in save
    assert "*-codex models are not supported (chat models only)." in js
    assert "$('#setAiModelsRefresh').addEventListener('click', () => loadAiModels(true));" in js
