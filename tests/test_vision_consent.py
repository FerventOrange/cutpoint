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
from fastapi.testclient import TestClient

from conftest import set_config
from cutpoint import activity, analyze, jsonout, ui, vision
from cutpoint.assistant import providers
from cutpoint.events import pipeline
from cutpoint.pano import vlcheck
from test_activity import fake_clip, footage  # noqa: F401
from test_analyze_steps import last_states, progress_lines, run  # noqa: F401
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
    assert providers.consent_key(there) == "assistant-v2:ollama@10.0.0.5:11434"
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
    assert providers.consent_key(sub) == f"assistant-v2:{name}@{host}"


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


# -- rating_key -------------------------------------------------------------------------------


def test_the_rating_key_tells_hosts_apart_for_a_hosted_provider_only() -> None:
    def key(provider: str, host: str) -> str:
        return vision.rating_key("m", "gaming", [1.0, 2.0], provider=provider, host=host)

    assert key("openai", "a.example:1") != key("openai", "b.example:1")
    assert key("openai", "a.example:1") != key("openai", "")
    assert key("ollama", "a.example:1") == key("ollama", "") == key("", "")


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
    assert "10.0.0.5:11434" in caplog.text
    assert [w["code"] for w in res.warnings] == ["vision_consent_needed"]


@pytest.mark.parametrize(
    "consent",
    [
        ["vision-v2:ollama@10.0.0.5:11435"],
        ["assistant-v2:ollama@10.0.0.5:11434"],
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


def test_the_rating_cache_of_ollama_ignores_the_host(home) -> None:
    here = vision.VisionRater(url="http://127.0.0.1:11434", model="m")
    there = vision.VisionRater(url=REMOTE, model="m")
    assert here.host == there.host == ""


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
    assert state["state"] == "skipped" and "not on this machine" in state["detail"]
    assert "not on this machine" in caplog.text
    assert [w["code"] for w in res.warnings] == ["vision_consent_needed"]
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
    assert "not on this machine" in caplog.text
    assert "vision_consent_needed" in [w["code"] for w in res.warnings]
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


# -- activity detection (profile --context, crashes, the crash step of analyze) ---------------


def test_activity_detection_sends_nothing_to_a_remote_ollama_without_consent(
    home, wire, monkeypatch, caplog
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", REMOTE)
    with caplog.at_level(logging.WARNING), jsonout.run("profile") as res:
        assert activity.default_ask() == (None, None)
    assert wire == []
    assert caplog.text.count("not on this machine") == 1
    assert "activity detection goes on without the vision model" in caplog.text
    assert [w["code"] for w in res.warnings] == ["vision_consent_needed"]


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
    ollama = SimpleNamespace(provider_name="ollama", host="", model="m")
    monkeypatch.setattr(pipeline, "VisionRater", lambda *_a, **_k: ollama)
    vlm = pipeline.VlmParams()
    with pytest.raises(Stop):
        pipeline.run_events(tmp_path, vlm=vlm, layers=("audio",))
    assert vlm.model == "m"  # Ollama keys unchanged


def test_the_pano_frame_key_tells_hosts_apart_for_a_hosted_provider_only() -> None:
    path = {"keys": [{"t": 0.0, "yaw": 0, "pitch": 0, "fov": 90}]}

    def key(provider: str, host: str) -> str:
        return vlcheck.frame_key(path, 1.0, "m", provider=provider, host=host)

    assert key("openai", "a.example:1") != key("openai", "b.example:1")
    assert key("openai", "a.example:1") != key("openai", "")
    assert key("openai", "") == vlcheck.frame_key(path, 1.0, "m", provider="openai")
    assert key("ollama", "a.example:1") == key("ollama", "") == key("", "")
    assert key("", "") == vlcheck.frame_key(path, 1.0, "m")  # Ollama keys unchanged


def test_the_vl_checker_keys_on_the_rater_host(home) -> None:
    assert vlcheck.VLChecker(url=REMOTE, model="m").host == ""  # Ollama: no host


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
