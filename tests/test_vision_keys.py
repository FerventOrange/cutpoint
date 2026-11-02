"""The Vision role's API key and model: CUTPOINT_ASSISTANT_API_KEY goes only to the assistant's
key host, and the caches are keyed on the model an OpenAI-compatible server resolved.

HTTP is stubbed (:class:`Wire`); the keyring is a fake; no request leaves the machine.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import test_vlm_vision as tvv
from conftest import set_config
from cutpoint import ui, vision
from cutpoint.assistant import keys, providers
from cutpoint.config import AssistantSettings
from cutpoint.events import pipeline
from cutpoint.pano import vlcheck
from test_assistant import _Resp
from test_assistant_byo import KEY, FakeRing, NoRing, Wire, oa_reply

OPENROUTER = "https://openrouter.ai/api/v1"
OTHER_HOST = "https://llm.example.com/v1"
LOCAL = "http://localhost:1234/v1"


@pytest.fixture(autouse=True)
def no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real key variable or keyring reaches a test."""
    for name in keys.ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(keys, "_keyring", NoRing)
    monkeypatch.setattr(keys, "_SEEN", set())
    monkeypatch.setattr(keys, "_CAPTURED", {})


def _vision(provider: str, url: str = "", model: str = "m") -> AssistantSettings:
    return AssistantSettings(provider=provider, model=model, url=url)


# -- F3: the shared key only for the assistant's key host ---------------------------------------


@pytest.mark.parametrize(
    ("chat", "vision_url", "shared"),
    [
        (("openai", OPENROUTER), OPENROUTER, True),  # the host the variable is set for
        (("openai", OPENROUTER), OTHER_HOST, False),  # another custom host
        (("anthropic", OPENROUTER), OPENROUTER, True),  # any keyed provider, same host
        (("ollama", ""), OPENROUTER, False),  # the assistant takes no key
        (("codex", ""), OPENROUTER, False),  # a subscription takes no key
    ],
)
def test_the_shared_key_reaches_the_vision_provider_only_on_the_assistant_s_key_host(
    home, monkeypatch, chat, vision_url, shared
) -> None:
    monkeypatch.setenv(keys.ENV_KEY, KEY)
    set_config(home, assistant={"provider": chat[0], "url": chat[1]})
    assert providers.vision_key_shared("openai", vision_url, *chat) is shared
    p = providers.make_vision(_vision("openai", vision_url))
    assert p.has_key is shared
    assert (providers.missing_key(p) is None) is shared


def test_the_shared_key_never_goes_to_an_official_host_even_for_the_same_host(
    home, monkeypatch
) -> None:
    """#504 still holds: the assistant on api.openai.com does not open it to Vision there."""
    monkeypatch.setenv(keys.ENV_KEY, KEY)
    set_config(home, assistant={"provider": "openai", "url": ""})
    assert providers.vision_key_shared("openai", "", "openai", "")
    assert not providers.make_vision(_vision("openai")).has_key


def test_the_vision_host_s_own_keyring_key_works_on_any_host(home, monkeypatch) -> None:
    fake = FakeRing()
    monkeypatch.setattr(keys, "_keyring", lambda: fake)
    monkeypatch.setenv(keys.ENV_KEY, "cp-shared-key-123456")
    keys.save("openai", KEY, "llm.example.com")
    set_config(home, assistant={"provider": "openai", "url": OPENROUTER})
    p = providers.make_vision(_vision("openai", OTHER_HOST))
    assert p._key == KEY


def test_an_explicit_chat_section_wins_over_config_json(home, monkeypatch) -> None:
    monkeypatch.setenv(keys.ENV_KEY, KEY)
    set_config(home, assistant={"provider": "ollama"})
    chat = AssistantSettings(provider="openai", url=OPENROUTER)
    assert providers.make_vision(_vision("openai", OPENROUTER), chat).has_key


def test_subscriptions_and_ollama_build_as_before(home) -> None:
    set_config(home)
    assert providers.make_vision(_vision("ollama", model="")).name == "ollama"
    assert providers.make_vision(_vision("codex", model="")).name == "codex"


def test_the_assistant_vision_provider_follows_the_rule(home, monkeypatch) -> None:
    monkeypatch.setenv(keys.ENV_KEY, KEY)
    set_config(
        home,
        assistant={"provider": "openai", "url": OPENROUTER},
        vision_assistant={"provider": "openai", "model": "m", "url": OTHER_HOST},
    )
    assert not vision.AssistantVision().provider.has_key


@pytest.mark.parametrize(("vision_url", "where"), [(OPENROUTER, "env"), (OTHER_HOST, None)])
def test_the_models_tab_vision_key_status_says_whether_the_shared_key_is_used(
    monkeypatch, vision_url, where
) -> None:
    monkeypatch.setenv(keys.ENV_KEY, KEY)
    data = {
        "assistant": {"provider": "openai", "url": OPENROUTER},
        "vision_assistant": {"provider": "openai", "url": vision_url},
    }
    assert ui.key_status(data, "vision_assistant")["keys"]["openai"] == where
    assert ui.key_status(data)["keys"]["openai"] == "env"  # the assistant's own: unchanged
    vp = ui._vision_provider_from_data(data)
    assert vp is not None and vp.has_key is (where == "env")


# -- F4: the caches are keyed on the resolved model ---------------------------------------------


@pytest.fixture
def local_server(home, monkeypatch) -> Wire:
    """An OpenAI-compatible server on this machine with one model; no model configured."""
    set_config(home, vision_assistant={"provider": "openai", "model": "", "url": LOCAL})
    return Wire(monkeypatch, _Resp({"data": [{"id": "served-7b"}]}), oa_reply("{}"))


def test_the_assistant_vision_model_is_the_server_s_once_checked(local_server) -> None:
    ast = vision.AssistantVision()
    assert ast.model == providers.AUTO_SERVER_MODEL
    assert ast.available()
    assert ast.model == "served-7b"


def test_the_rater_keys_its_ratings_on_the_resolved_model(local_server) -> None:
    rater = vision.VisionRater()
    assert rater.available()
    assert rater.model == "served-7b"
    assert vision.rating_key(rater.model, "action", [1.0], provider="openai") != vision.rating_key(
        providers.AUTO_SERVER_MODEL, "action", [1.0], provider="openai"
    )


def test_the_vl_check_keys_its_frames_on_the_resolved_model(local_server) -> None:
    checker = vlcheck.VLChecker()
    assert checker.available()
    assert checker.model == "served-7b"


def test_an_unavailable_provider_keeps_the_configured_model(home, monkeypatch) -> None:
    set_config(home, vision_assistant={"provider": "openai", "model": "", "url": OPENROUTER})
    Wire(monkeypatch)  # no consent: nothing is asked
    rater = vision.VisionRater()
    assert not rater.available()
    assert rater.model == providers.AUTO_SERVER_MODEL


def test_the_events_cache_is_looked_up_under_the_resolved_model(
    local_server, tmp_path, monkeypatch
) -> None:
    """F6: the VLM layer resolves the server's model before its first cache lookup, so a rerun
    finds the descriptions the first run saved under it and asks the server nothing."""
    monkeypatch.setattr(pipeline.importlib.util, "find_spec", lambda name: object())
    local_server.answers[1] = oa_reply(json.dumps(tvv.REPLY))  # /models, then the window
    first = tvv._run_vlm(tmp_path, monkeypatch)
    assert first["settings"]["vlm"]["model"] == "openai:served-7b"
    assert [e["label"] for e in first["events"] if e["kind"] == "vlm"] == ["player gets a kill"]

    raters: list[Any] = []
    real = pipeline.describe_windows

    def spy(rater: Any, *a: Any, **k: Any) -> Any:
        raters.append(rater)
        return real(rater, *a, **k)

    monkeypatch.setattr(pipeline, "describe_windows", spy)
    wire = Wire(monkeypatch, _Resp({"data": [{"id": "served-7b"}]}))  # /models only: no chat
    again = tvv._run_vlm(tmp_path, monkeypatch)
    assert [r["method"] for r in wire.sent] == ["GET"]
    assert raters == [None]  # the cached lookup found every window
    assert [e["label"] for e in again["events"] if e["kind"] == "vlm"] == ["player gets a kill"]
    assert again["stats"]["vlm"]["cached"] == 1


# -- F7: the key hints of a Vision provider on another host than the assistant's ----------------


@pytest.mark.parametrize(("vision_url", "shared"), [(OPENROUTER, True), (OTHER_HOST, False)])
def test_the_vision_key_hints_name_the_shared_variable_only_where_it_is_read(
    home, vision_url, shared
) -> None:
    set_config(home, assistant={"provider": "openai", "url": OPENROUTER})
    missing = providers.missing_key(providers.make_vision(_vision("openai", vision_url)))
    assert missing is not None
    assert (f"set {keys.ENV_KEY}" in missing) is shared
    assert (f"{keys.ENV_KEY} is only for the assistant's host" in missing) is not shared
    data = {
        "assistant": {"provider": "openai", "url": OPENROUTER},
        "vision_assistant": {"provider": "openai", "url": vision_url},
    }
    view = ui.assistant_view(data, "vision_assistant")
    env = {p["name"]: p["key_env"] for p in view["providers"]}
    assert env["openai"] == ([keys.ENV_KEY] if shared else [])
    chat = {p["name"]: p["key_env"] for p in ui.assistant_view(data)["providers"]}
    assert chat["openai"] == [keys.ENV_KEY]  # the assistant's own: unchanged
