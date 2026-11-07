"""An Ollama on another machine is hosted for the Vision role: nothing is sent to it until the
user agreed in the Models tab of ``cutpoint ui`` (a ``vision-v2:ollama@<host:port>`` entry).

HTTP is stubbed to fail the test on any request; no network.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests
from fastapi.testclient import TestClient

from conftest import set_config
from cutpoint import activity, analyze, jsonout, ui, vision
from cutpoint.assistant import cliproxy, providers
from cutpoint.events import pipeline
from cutpoint.pano import vlcheck
from test_activity import fake_clip, footage  # noqa: F401
from test_analyze_steps import last_states, progress_lines, run  # noqa: F401
from test_assistant_byo import no_keys as no_keys  # the autouse fixture: no real keyring
from test_vlm_vision import REPLY, Ollama, _run_vlm, extra_ok  # noqa: F401

REMOTE = "http://10.0.0.5:11434"
KEY = "vision-v2:ollama@10.0.0.5:11434"


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every request the vision module makes: any one fails the test."""
    sent: list[str] = []

    def record(url: str, *_a: Any, **_k: Any) -> Any:
        sent.append(url)
        raise AssertionError(f"a request went out to {url}")

    monkeypatch.setattr(vision.requests, "get", record)
    monkeypatch.setattr(vision.requests, "post", record)
    return sent


# -- providers: host_port, consent_key, is_local ---------------------------------------------


@pytest.mark.parametrize(
    ("url", "want"),
    [
        ("https://api.openai.com/v1", "api.openai.com"),
        ("http://box:80", "box"),
        ("https://box:443/x", "box"),
        ("192.168.1.5:8080", "192.168.1.5:8080"),
        ("http://10.0.0.5:11434", "10.0.0.5:11434"),
        ("https://box:8443", "box:8443"),
        ("http://[fd00::5]:8080", "[fd00::5]:8080"),
        ("http://[fd00::5]", "fd00::5"),
        ("HTTP://Box:81", "box:81"),
    ],
)
def test_host_port_omits_only_the_default_port(url: str, want: str) -> None:
    assert providers.host_port(url) == want


def test_the_consent_key_has_the_port_of_a_non_default_endpoint() -> None:
    there = providers.OllamaProvider("10.0.0.5:11434", "m", 0)
    other = providers.OllamaProvider("10.0.0.5:11435", "m", 0)
    assert providers.consent_key(there, "vision") == KEY
    assert providers.consent_key(there) == "assistant-v3:ollama@10.0.0.5:11434"
    assert providers.consent_key(other, "vision") != providers.consent_key(there, "vision")
    official = providers.OpenAIProvider("https://api.openai.com/v1", "m", 0, "k")
    assert providers.consent_key(official, "vision") == "vision-v2:openai@api.openai.com"
    lan = providers.OpenAIProvider("http://192.168.1.5:1234/v1", "m", 0, "k")
    assert providers.consent_key(lan, "vision") == "vision-v2:openai@192.168.1.5:1234"


@pytest.mark.parametrize("name", ["claude-code", "codex", "antigravity"])
def test_a_subscription_keeps_its_host_only_key(name: str) -> None:
    sub = providers.SubscriptionProvider(name)
    host = providers.SUBSCRIPTION_HOSTS[name]
    assert providers.consent_key(sub, "vision") == f"vision-v2:{name}@{host}"
    assert providers.consent_key(sub) == f"assistant-v3:{name}@{host}"


@pytest.mark.parametrize(
    ("url", "local"),
    [
        ("http://localhost:11434", True),
        ("http://127.0.0.1:11434", True),
        ("http://127.5.6.7", True),
        ("http://[::1]:11434", True),
        ("http://foo.localhost:11434", False),
        ("http://10.0.0.5:11434", False),
    ],
)
def test_is_local_takes_loopback_and_localhost_only(url: str, local: bool) -> None:
    assert providers.is_local(url) is local


# -- the Cutpoint proxy's port is hosted (#650) ----------------------------------------------

PROXY = f"https://127.0.0.1:{cliproxy.DEFAULT_PORT}/v1"
PROXY_VISION_KEY = f"vision-v2:openai@127.0.0.1:{cliproxy.DEFAULT_PORT}"


@pytest.mark.parametrize(
    ("url", "exempt"),
    [
        (PROXY, False),
        (f"http://localhost:{cliproxy.DEFAULT_PORT}", False),
        (f"http://[::1]:{cliproxy.DEFAULT_PORT}/v1", False),
        (f"127.0.0.1:{cliproxy.DEFAULT_PORT}", False),
        ("http://127.0.0.1:11434", True),  # Ollama
        ("http://localhost:11434", True),
        ("http://localhost:1234/v1", True),  # LM Studio, a relay on another port
        ("http://127.0.0.1", True),
        (f"http://10.0.0.5:{cliproxy.DEFAULT_PORT}", False),  # not on this machine anyway
    ],
)
def test_only_the_proxy_port_of_this_machine_needs_consent(url: str, exempt: bool) -> None:
    assert providers.is_consent_exempt(url) is exempt
    assert providers.is_proxy(url) is (providers.is_local(url) and not exempt)


def test_the_proxy_port_is_still_on_this_machine() -> None:
    """Timeouts, context and key rules keep reading is_local: the proxy runs here."""
    assert providers.is_local(PROXY)
    p = providers.OpenAIProvider(PROXY, "m", 0, None)
    assert p.on_machine and not p.local
    assert p.timeout == providers.CHAT_TIMEOUT_S
    assert p.context_tokens == providers.ON_MACHINE_CONTEXT_TOKENS
    assert providers.missing_key(p) is None  # a server on this machine may take no key
    assert providers.consent_key(p, "vision") == PROXY_VISION_KEY
    assert providers.consent_key(p) == f"assistant-v3:openai@127.0.0.1:{cliproxy.DEFAULT_PORT}"


@pytest.mark.parametrize("where", ["env", "keys"])
def test_a_configured_proxy_port_replaces_the_default(home, monkeypatch, where: str) -> None:
    if where == "env":
        monkeypatch.setenv("CUTPOINT_CLIPROXY_PORT", "19000")
    else:
        cliproxy.keys_path().parent.mkdir(parents=True, exist_ok=True)
        cliproxy.keys_path().write_text(json.dumps({"port": 19000}), "utf-8")
    assert providers.proxy_port() == 19000
    assert not providers.is_consent_exempt("https://127.0.0.1:19000/v1")
    assert providers.is_consent_exempt(PROXY)  # the default port is just another local port


def test_an_unreadable_proxy_port_falls_back_to_the_default(home, monkeypatch) -> None:
    monkeypatch.setenv("CUTPOINT_CLIPROXY_PORT", "not-a-port")
    with pytest.raises(cliproxy.ProxyError):
        cliproxy.port()
    assert providers.proxy_port() == cliproxy.DEFAULT_PORT
    assert not providers.is_consent_exempt(PROXY)

    def broken() -> int:
        raise OSError("keys file unreadable")

    monkeypatch.setattr(cliproxy, "port", broken)
    assert providers.proxy_port() == cliproxy.DEFAULT_PORT
    assert providers.is_consent_exempt("http://127.0.0.1:11434")


def test_a_vision_provider_on_the_proxy_port_needs_consent(home, hosted) -> None:
    set_config(home, vision_assistant={"provider": "openai", "url": PROXY})
    rater = vision.VisionRater()
    assert rater.via is not None and rater.consent_needed == PROXY_VISION_KEY
    assert not rater.available() and rater.ask_json("q") is None
    assert hosted == []
    set_config(
        home, vision_assistant={"provider": "openai", "url": PROXY, "consent": [PROXY_VISION_KEY]}
    )
    assert vision.VisionRater().consent_needed is None


def test_an_ollama_on_the_proxy_port_needs_consent(home, wire) -> None:
    url = f"http://127.0.0.1:{cliproxy.DEFAULT_PORT}"
    assert vision.VisionRater(url=url, model="m").consent_needed == (
        f"vision-v2:ollama@127.0.0.1:{cliproxy.DEFAULT_PORT}"
    )
    assert wire == []


def test_the_settings_view_lists_the_proxy_port_as_hosted() -> None:
    view = ui.settings_view(
        {
            "assistant": {"provider": "openai", "url": PROXY},
            "vision_assistant": {"provider": "openai", "url": PROXY},
        }
    )
    assert view["assistant"]["local"] is False and view["vision_assistant"]["local"] is False
    assert view["vision_consent"]["needed"] is True
    assert view["vision_consent"]["key"] == PROXY_VISION_KEY
    other = ui.settings_view(
        {"assistant": {"provider": "openai", "url": "http://localhost:1234/v1"}}
    )
    assert other["assistant"]["local"] is True


# -- rating_key -------------------------------------------------------------------------------


def _rating_key(provider: str, host: str) -> str:
    return vision.rating_key("m", "gaming", [1.0, 2.0], provider=provider, host=host)


def test_the_rating_key_tells_hosts_apart() -> None:
    assert _rating_key("openai", "a.example:1") != _rating_key("openai", "b.example:1")
    assert _rating_key("openai", "a.example:1") != _rating_key("openai", "")
    # Ollama too (#673); without a host it keeps the old key
    assert _rating_key("ollama", "127.0.0.1:11434") != _rating_key("ollama", "10.0.0.5:11434")
    assert _rating_key("ollama", "10.0.0.5:11434") != _rating_key("ollama", "10.0.0.5:11435")
    assert _rating_key("ollama", "") == _rating_key("", "")


def test_hosted_rating_keys_are_unchanged() -> None:
    # Pinned: a change here drops every user's cached hosted ratings (#621)
    assert _rating_key("openai", "a.example:1") == "89335c0665@1.000,2.000"
    assert _rating_key("openai", "") == "4b4446f1ba@1.000,2.000"


# -- VisionRater ------------------------------------------------------------------------------


def test_a_local_ollama_needs_no_consent(home, wire) -> None:
    for url in ("http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"):
        assert vision.VisionRater(url=url, model="m").consent_needed is None


def test_a_remote_ollama_needs_consent_and_is_not_contacted(home, wire, caplog) -> None:
    rater = vision.VisionRater(url=REMOTE, model="m")
    assert rater.consent_needed == KEY
    with caplog.at_level(logging.WARNING), jsonout.run("x") as res:
        assert not rater.available()
        assert rater.ask_json("q") is None
        assert rater.describe("q", [b"\xff\xd8"], {}) is None
        assert rater._rate_frames(Path("a.mp4"), [1.0], "gaming") is None
    assert wire == []
    assert caplog.text.count("not on this machine") == 1  # warned once
    assert [w["code"] for w in res.warnings] == ["vision_consent_needed"]
    want = (
        "Ollama at 10.0.0.5:11434 is not on this machine: allow vision requests to it in the "
        "Models tab of `cutpoint ui` (Vision role, Ollama with this URL), or add "
        f'"{KEY}" to vision_assistant.consent in config.json; skipping vision'
    )
    assert want in caplog.text
    assert res.warnings[0]["message"] == want


@pytest.mark.parametrize(
    "consent",
    [
        ["vision-v2:ollama@10.0.0.5:11435"],
        ["assistant-v3:ollama@10.0.0.5:11434"],
        ["vision-v2:ollama@10.0.0.5"],
        ["ollama@10.0.0.5"],
    ],
    ids=["other-port", "assistant", "no-port", "unprefixed"],
)
def test_no_other_entry_unlocks_a_remote_ollama(home, wire, consent) -> None:
    set_config(home, vision_assistant={"provider": "ollama", "consent": consent})
    assert vision.VisionRater(url=REMOTE, model="m").consent_needed == KEY


def test_with_consent_a_remote_ollama_is_used(home, monkeypatch) -> None:
    set_config(home, vision_assistant={"provider": "ollama", "consent": [KEY]})
    fake = Ollama()
    monkeypatch.setattr(vision.requests, "get", fake.get)
    monkeypatch.setattr(vision.requests, "post", fake.post)
    rater = vision.VisionRater(url=REMOTE, model="qwen3.5:9b")
    assert rater.consent_needed is None
    assert rater.available()


def test_the_rating_cache_of_ollama_keys_on_the_host(home, monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    here = vision.VisionRater(url="http://127.0.0.1:11434", model="m")
    there = vision.VisionRater(url=REMOTE, model="m")
    port = vision.VisionRater(url="http://10.0.0.5:11435", model="m")
    assert (here.host, there.host, port.host) == (
        "127.0.0.1:11434",
        "10.0.0.5:11434",
        "10.0.0.5:11435",
    )
    # the default URL: the same host on every run, so the same key
    default = vision.VisionRater(model="m")
    assert default.host == vision.VisionRater(model="m").host == "127.0.0.1:11434"


# -- the steps that use the rater -------------------------------------------------------------


def test_analyze_vision_skips_a_remote_ollama_without_consent(
    run,  # noqa: F811
    home,
    wire,
    capsys,
    caplog,
    monkeypatch,
) -> None:
    monkeypatch.setattr(analyze, "VisionRater", vision.VisionRater)  # the run fixture fakes it
    with caplog.at_level(logging.WARNING), jsonout.run("analyze") as res:
        run.go(vision=True, ollama_url=REMOTE)
    state = last_states(progress_lines(capsys))["vision"]
    assert state["state"] == "skipped" and f'add "{KEY}" to' in state["detail"]
    assert f'add "{KEY}" to vision_assistant.consent' in caplog.text
    assert [w["code"] for w in res.warnings] == ["vision_consent_needed"]
    assert KEY in res.warnings[0]["message"]
    assert wire == []
    assert (run.proj / "highlights.json").is_file()  # the run succeeds


def test_events_vlm_layer_skips_a_remote_ollama_without_consent(
    extra_ok,  # noqa: F811
    home,
    wire,
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    with caplog.at_level(logging.WARNING), jsonout.run("events") as res:
        data = _run_vlm(tmp_path, monkeypatch, ollama_url=REMOTE, ollama_model="qwen3.5:9b")
    assert not [e for e in data["events"] if e["kind"] == "vlm"]
    assert f'add "{KEY}" to vision_assistant.consent' in caplog.text
    msgs = [w["message"] for w in res.warnings if w["code"] == "vision_consent_needed"]
    assert len(msgs) == 1 and KEY in msgs[0]
    assert wire == []


def test_events_vlm_layer_uses_a_remote_ollama_with_consent(
    extra_ok,  # noqa: F811
    home,
    tmp_path,
    monkeypatch,
) -> None:
    set_config(home, vision_assistant={"provider": "ollama", "consent": [KEY]})
    fake = Ollama(REPLY)
    monkeypatch.setattr(vision.requests, "get", fake.get)
    monkeypatch.setattr(vision.requests, "post", fake.post)
    data = _run_vlm(tmp_path, monkeypatch, ollama_url=REMOTE)
    assert len(fake.chats) == 1
    assert [e["label"] for e in data["events"] if e["kind"] == "vlm"] == ["player gets a kill"]


def test_the_pano_vl_check_sends_no_frame_to_a_remote_ollama_without_consent(
    home, wire, caplog
) -> None:
    chk = vlcheck.VLChecker(url=REMOTE, model="m")
    with caplog.at_level(logging.WARNING), jsonout.run("360") as res:
        assert chk.ask(b"jpeg") is None
    assert chk.dead
    assert wire == []
    assert "not on this machine" in caplog.text
    assert [w["code"] for w in res.warnings] == ["vision_consent_needed"]
    assert KEY in res.warnings[0]["message"]


# -- activity detection (profile --context, crashes, the crash step of analyze) ---------------


def test_activity_detection_sends_nothing_to_a_remote_ollama_without_consent(
    home, wire, monkeypatch, caplog
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", REMOTE)
    with caplog.at_level(logging.WARNING), jsonout.run("profile") as res:
        assert activity.default_ask() == (None, None)
    assert wire == []
    assert caplog.text.count("not on this machine") == 1
    tail = f'"{KEY}" to vision_assistant.consent in config.json; activity detection goes on'
    assert f"{tail} without the vision model" in caplog.text
    assert [w["code"] for w in res.warnings] == ["vision_consent_needed"]
    assert KEY in res.warnings[0]["message"]


def test_activity_detection_is_not_unlocked_by_the_assistant_consent(home, wire) -> None:
    set_config(
        home,
        ollama_url=REMOTE,
        assistant={
            "provider": "ollama",
            "url": REMOTE,
            "consent": [KEY.replace("vision", "assistant")],
        },
    )
    assert vision.VisionRater(local_only=True).consent_needed == KEY


def test_activity_detection_uses_a_remote_ollama_with_consent(home, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", REMOTE)
    set_config(home, vision_assistant={"provider": "ollama", "consent": [KEY]})
    fake = Ollama({"activity": "packrafting"})
    monkeypatch.setattr(vision.requests, "get", fake.get)
    monkeypatch.setattr(vision.requests, "post", fake.post)
    ask, model = activity.default_ask()
    assert ask is not None and model == "qwen3.5:9b"
    assert ask("q", [b"jpeg"], None) == {"activity": "packrafting"}
    assert len(fake.chats) == 1


def test_activity_detection_with_a_local_ollama_needs_no_consent(home, wire) -> None:
    assert vision.VisionRater(local_only=True).consent_needed is None


def test_resolve_falls_back_without_the_model_and_warns_once(
    home,
    wire,
    tmp_path,
    footage,  # noqa: F811
    fake_clip,  # noqa: F811
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", REMOTE)
    footage["v"] = fake_clip("ski", "mtb")  # ambiguous: the LLM would be asked
    with caplog.at_level(logging.WARNING), jsonout.run("profile") as res:
        rec = activity.resolve(tmp_path, refresh=True, context="packrafting trip")
    assert rec["source"] == "fallback"
    assert wire == []
    assert caplog.text.count("not on this machine") == 1
    codes = [w["code"] for w in res.warnings]
    assert codes == ["vision_consent_needed", "context_not_applied"]
    assert KEY in res.warnings[0]["message"]


# -- a hosted Vision provider without consent --------------------------------------------------

HOSTED_KEY = "vision-v2:openai@api.openai.com"
HOSTED_WANT = (
    "Vision provider openai (api.openai.com) needs your consent before frames are sent to it: "
    "allow it in the Models tab of `cutpoint ui` (Vision role), or add "
    f'"{HOSTED_KEY}" to vision_assistant.consent in config.json; skipping vision'
)


@pytest.fixture
def hosted(home, wire, monkeypatch) -> list[str]:
    """The Vision role on OpenAI's API without its consent (the assistant's does not count);
    any request, also through a session, fails the test."""
    set_config(
        home,
        vision_assistant={"provider": "openai", "consent": ["assistant-v3:openai@api.openai.com"]},
    )

    def record(_self: Any, _method: str, url: str, *_a: Any, **_k: Any) -> Any:
        wire.append(url)
        raise AssertionError(f"a request went out to {url}")

    monkeypatch.setattr(vision.requests.Session, "request", record)
    return wire


def test_a_hosted_provider_without_consent_warns_once_with_its_key(hosted, caplog) -> None:
    rater = vision.VisionRater()
    assert rater.via is not None and rater.consent_needed == HOSTED_KEY
    with caplog.at_level(logging.INFO), jsonout.run("x") as res:
        assert not rater.via._check()
        assert not rater.via._check()
        assert not rater.available()
        assert rater.ask_json("q") is None
        assert rater.describe("q", [b"\xff\xd8"], {}) is None
        assert rater._rate_frames(Path("a.mp4"), [1.0], "gaming") is None
        assert rater.warn_consent() == HOSTED_WANT
    assert hosted == []
    assert caplog.text.count("needs your consent") == 1  # warned once
    assert HOSTED_WANT in caplog.text
    assert [w["code"] for w in res.warnings] == ["vision_consent_needed"]
    assert res.warnings[0]["message"] == HOSTED_WANT


@pytest.mark.parametrize("remote", [False, True], ids=["hosted", "remote-ollama"])
def test_two_raters_in_one_run_warn_once(home, hosted, remote, caplog) -> None:
    """edit --vision --events builds one rater for analyze and one for the VLM layer."""
    make = (lambda: vision.VisionRater(url=REMOTE, model="m")) if remote else vision.VisionRater
    if remote:
        set_config(home, vision_assistant={"provider": "ollama"})
    with caplog.at_level(logging.WARNING), jsonout.run("edit") as res:
        first, second = make(), make()
        assert first is not second and first.consent_needed and second.consent_needed
        assert not first.available()
        assert not second.available()
        assert first.warn_consent() == second.warn_consent()
    assert hosted == []
    assert len(caplog.records) == 1
    assert [w["code"] for w in res.warnings] == ["vision_consent_needed"]


def test_with_consent_a_hosted_provider_needs_none(home) -> None:
    set_config(home, vision_assistant={"provider": "openai", "consent": [HOSTED_KEY]})
    assert vision.VisionRater().consent_needed is None


def test_analyze_vision_skips_a_hosted_provider_without_consent(
    run,  # noqa: F811
    hosted,
    capsys,
    caplog,
    monkeypatch,
) -> None:
    monkeypatch.setattr(analyze, "VisionRater", vision.VisionRater)  # the run fixture fakes it
    with caplog.at_level(logging.WARNING), jsonout.run("analyze") as res:
        run.go(vision=True)
    state = last_states(progress_lines(capsys))["vision"]
    assert state["state"] == "skipped" and state["detail"] == HOSTED_WANT
    assert caplog.text.count("needs your consent") == 1
    assert [w["message"] for w in res.warnings if w["code"] == "vision_consent_needed"] == [
        HOSTED_WANT
    ]
    assert hosted == []
    assert (run.proj / "highlights.json").is_file()  # the run succeeds


def test_events_vlm_layer_skips_a_hosted_provider_without_consent(
    extra_ok,  # noqa: F811
    hosted,
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    with caplog.at_level(logging.WARNING), jsonout.run("events") as res:
        data = _run_vlm(tmp_path, monkeypatch)
    assert not [e for e in data["events"] if e["kind"] == "vlm"]
    assert caplog.text.count("needs your consent") == 1
    msgs = [w["message"] for w in res.warnings if w["code"] == "vision_consent_needed"]
    assert msgs == [HOSTED_WANT]
    assert hosted == []


@pytest.mark.parametrize("agreed", [False, True], ids=["no-consent", "consent"])
def test_events_uses_no_cached_descriptions_without_consent(
    extra_ok,  # noqa: F811
    home,
    agreed,
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    """An Ollama on another machine whose consent was withdrawn gets none of the descriptions
    it cached (as analyze's ratings) and is asked nothing; one with consent is asked, and the
    descriptions the local Ollama cached for the same model are not used: they are keyed on
    the host (#670)."""
    fake = Ollama()
    fake.replies = [REPLY]
    monkeypatch.setattr(vision.requests, "get", fake.get)
    monkeypatch.setattr(vision.requests, "post", fake.post)
    if agreed:  # cached by the local Ollama
        first = _run_vlm(tmp_path, monkeypatch)
    else:  # cached by the remote one, then the consent is withdrawn
        set_config(home, vision_assistant={"provider": "ollama", "consent": [KEY]})
        first = _run_vlm(tmp_path, monkeypatch, ollama_url=REMOTE)
    assert [e for e in first["events"] if e["kind"] == "vlm"] and first["windows"]

    sent: list[str] = []

    def record(url: str, *_a: Any, **_k: Any) -> Any:
        sent.append(url)
        raise requests.ConnectionError("refused")  # the remote Ollama is down

    monkeypatch.setattr(vision.requests, "get", record)
    monkeypatch.setattr(vision.requests, "post", record)
    set_config(home, vision_assistant={"provider": "ollama", "consent": [KEY] if agreed else []})
    with caplog.at_level(logging.WARNING), jsonout.run("events") as res:
        data = _run_vlm(tmp_path, monkeypatch, ollama_url=REMOTE)
    vlm_events = [e for e in data["events"] if e["kind"] == "vlm"]
    msgs = [w["message"] for w in res.warnings if w["code"] == "vision_consent_needed"]
    if agreed:
        assert sent and all(url.startswith(REMOTE) for url in sent)
        assert vlm_events == [] and data["stats"]["vlm"]["cached"] == 0  # not the local ones
        assert msgs == [] and [w["code"] for w in res.warnings] == ["vlm_unavailable"]
    else:
        assert sent == []
        assert vlm_events == [] and data["windows"] == []
        assert len(msgs) == 1 and KEY in msgs[0]
        assert caplog.text.count("not on this machine") == 1
        assert not [w for w in res.warnings if w["code"] == "vlm_unavailable"]


def test_action_cam_reused_windows_warn_without_consent_and_no_vlm_layer(
    extra_ok,  # noqa: F811
    home,
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    """An action-cam rerun reuses earlier described windows; once the consent is withdrawn
    they go, and a run without the VLM layer says so once, also when its new windows have no
    cache. The windows are described by the remote Ollama: the cache is keyed on its host."""
    monkeypatch.setattr(pipeline, "project_kind", lambda hl: pipeline.ACTION_CAM)
    monkeypatch.setattr(pipeline, "_telemetry_layer", lambda *a: {})
    fake = Ollama()
    fake.replies = [REPLY]
    monkeypatch.setattr(vision.requests, "get", fake.get)
    monkeypatch.setattr(vision.requests, "post", fake.post)
    set_config(home, vision_assistant={"provider": "ollama", "consent": [KEY]})
    first = _run_vlm(tmp_path, monkeypatch, ollama_url=REMOTE)
    assert first["windows"]
    src = str(tmp_path / "game.mp4")
    later = {"id": "h002", "source": src, "start": 45.0, "end": 49.0, "score": 70}
    monkeypatch.setattr(
        pipeline,
        "load_highlights",
        lambda p: {"sources": [{"path": src}], "candidates": [later]},
    )

    sent: list[str] = []

    def record(url: str, *_a: Any, **_k: Any) -> Any:
        sent.append(url)
        raise AssertionError(f"a request went out to {url}")

    monkeypatch.setattr(vision.requests, "get", record)
    monkeypatch.setattr(vision.requests, "post", record)
    set_config(home, vision_assistant={"provider": "ollama"})
    with caplog.at_level(logging.WARNING), jsonout.run("events") as res:
        jpath, _ = pipeline.run_events(tmp_path / "proj", layers=(), ollama_url=REMOTE)
    data = json.loads(Path(jpath).read_text(encoding="utf-8"))
    assert sent == []
    assert not [e for e in data["events"] if e["kind"] == "vlm"]
    msgs = [w["message"] for w in res.warnings if w["code"] == "vision_consent_needed"]
    assert len(msgs) == 1 and KEY in msgs[0]


def test_the_pano_vl_check_sends_no_frame_to_a_hosted_provider_without_consent(
    hosted, caplog
) -> None:
    chk = vlcheck.VLChecker()
    with caplog.at_level(logging.WARNING), jsonout.run("360") as res:
        assert chk.ask(b"jpeg") is None
        assert chk.ask(b"jpeg") is None
    assert chk.dead
    assert hosted == []
    assert caplog.text.count("needs your consent") == 1
    assert [w["code"] for w in res.warnings] == ["vision_consent_needed"]
    assert HOSTED_KEY in res.warnings[0]["message"]


# -- the description and frame caches of hosted providers (#621) ------------------------------


def test_the_events_descriptions_of_a_hosted_provider_are_keyed_on_its_host(
    home, tmp_path, monkeypatch
) -> None:
    class Stop(Exception):
        pass

    def stop(*_a: Any, **_k: Any) -> Any:
        raise Stop  # the key is set by now: the rest of the run is not needed

    hl = {"sources": [{"path": str(tmp_path / "a.mp4")}]}
    monkeypatch.setattr(pipeline, "load_highlights", lambda project: hl)
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "load_source", stop)
    vlms = []
    for host in ("a.example:8080", "b.example:8080"):
        rater = SimpleNamespace(provider_name="openai", host=host, model="m")
        monkeypatch.setattr(pipeline, "VisionRater", lambda *_a, r=rater, **_k: r)
        vlm = pipeline.VlmParams()
        with pytest.raises(Stop):
            pipeline.run_events(tmp_path, vlm=vlm, layers=("audio",))
        vlms.append(vlm)
    assert [v.model for v in vlms] == ["openai@a.example:8080:m", "openai@b.example:8080:m"]
    assert vlms[0].key() != vlms[1].key()


def test_the_events_descriptions_of_an_ollama_are_keyed_on_its_host(
    home, tmp_path, monkeypatch
) -> None:
    """As a hosted provider's (#621, providers.host_port): a local and a remote Ollama with
    one model id describe apart, and the local default URL keeps one key (#670)."""

    class Stop(Exception):
        pass

    def stop(*_a: Any, **_k: Any) -> Any:
        raise Stop  # the key is set by now: the rest of the run is not needed

    hl = {"sources": [{"path": str(tmp_path / "a.mp4")}]}
    monkeypatch.setattr(pipeline, "load_highlights", lambda project: hl)
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "load_source", stop)

    def model(url: str | None = None) -> str:
        vlm = pipeline.VlmParams()
        with pytest.raises(Stop):
            pipeline.run_events(tmp_path, vlm=vlm, layers=("audio",), ollama_url=url)
        return vlm.model

    local, remote = model(), model(REMOTE)
    assert local == "ollama@127.0.0.1:11434:qwen3.5:9b"  # the default URL
    assert remote == "ollama@10.0.0.5:11434:qwen3.5:9b"
    assert local != remote
    assert model() == local  # stable across runs
    assert model("http://127.0.0.1:11434/") == local  # a trailing slash is the same server
    assert model("http://10.0.0.5:11435") != remote  # another port is another server


def test_the_pano_frame_key_tells_hosts_apart() -> None:
    path = {"keys": [{"t": 0.0, "yaw": 0, "pitch": 0, "fov": 90}]}

    def key(provider: str, host: str) -> str:
        return vlcheck.frame_key(path, 1.0, "m", provider=provider, host=host)

    assert key("openai", "a.example:1") != key("openai", "b.example:1")
    assert key("openai", "a.example:1") != key("openai", "")
    assert key("openai", "") == vlcheck.frame_key(path, 1.0, "m", provider="openai")
    # hosted keys are unchanged (#621)
    assert key("openai", "a.example:1") == "2|m|openai@a.example:1|h|1.00|0|0|90"
    # Ollama keys on its host too (#673), in the same format
    assert key("ollama", "127.0.0.1:11434") == "2|m|ollama@127.0.0.1:11434|h|1.00|0|0|90"
    assert key("ollama", "127.0.0.1:11434") != key("ollama", "10.0.0.5:11434")
    assert key("ollama", "10.0.0.5:11434") != key("ollama", "10.0.0.5:11435")
    assert key("ollama", "") == key("", "") == vlcheck.frame_key(path, 1.0, "m")
    with_wording = vlcheck.frame_key(path, 1.0, "m", provider="ollama", wording="ab", host="h:1")
    assert with_wording == "2|m|ollama@h:1|wab|h|1.00|0|0|90"


def test_the_vl_checker_keys_on_the_rater_host(home, monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert vlcheck.VLChecker(url=REMOTE, model="m").host == "10.0.0.5:11434"
    assert vlcheck.VLChecker(url="http://127.0.0.1:11434", model="m").host == "127.0.0.1:11434"
    assert vlcheck.VLChecker(model="m").host == vlcheck.VLChecker(model="m").host


# -- the settings view ------------------------------------------------------------------------


def test_settings_view_asks_vision_consent_for_a_remote_ollama_only() -> None:
    def view(url: str, consent: list[str] | None = None) -> dict[str, Any]:
        section: dict[str, Any] = {"provider": "ollama", "url": url}
        if consent is not None:
            section["consent"] = consent
        return ui.settings_view({"vision_assistant": section})["vision_consent"]

    local = view("http://127.0.0.1:11434")
    assert local["needed"] is False and local["key"] is None
    remote = view(REMOTE)
    assert remote["needed"] is True and remote["key"] == KEY
    assert remote["host"] == "10.0.0.5"
    assert view(REMOTE, [KEY])["needed"] is False


def test_the_ui_consent_route_unlocks_a_remote_ollama_for_the_rater(home) -> None:
    set_config(home, vision_assistant={"provider": "ollama", "url": REMOTE})
    assert vision.VisionRater().consent_needed == KEY
    app = ui.create_app(8765)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        client.headers["X-Cutpoint-Token"] = app.state.token
        res = client.post("/api/assistant/consent", json={"role": "vision", "key": KEY})
    assert res.status_code == 200, res.text
    saved = json.loads((home / "config.json").read_text("utf-8"))
    assert saved["vision_assistant"]["consent"] == [KEY]
    assert vision.VisionRater().consent_needed is None


# -- unload_ollama (the events run frees the GPU first) ---------------------------------------


def test_unload_sends_nothing_to_a_remote_ollama_without_consent(home, wire, monkeypatch) -> None:
    from cutpoint.events import common  # noqa: PLC0415

    monkeypatch.setattr(common.requests, "get", vision.requests.get)
    monkeypatch.setattr(common.requests, "post", vision.requests.post)
    assert common.unload_ollama(REMOTE) == []
    assert wire == []
