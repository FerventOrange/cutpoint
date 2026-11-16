"""What goes to hosted models and into project files (#1085): an Ollama login is never saved
(A639), and source file names go only to a vision model on this machine (A653)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import requests

from conftest import set_config
from cutpoint import backends, cache, cli, runstate, vision
from cutpoint.assistant import providers
from cutpoint.assistant.providers import Reply
from cutpoint.events import vlm_windows as vw
from cutpoint.events.common import Source

REMOTE = "10.0.0.5"
REMOTE_KEY = f"vision-v2:ollama@{REMOTE}:11434"

# (as given, as recorded): the password holds an "@" or a "/" in some
LOGINS = [
    ("http://alice:s3cret@myhost:11434", "http://myhost:11434"),
    ("http://alice:s3c@ret@myhost:11434", "http://myhost:11434"),
    ("http://alice:s3c/ret@myhost:11434", "http://myhost:11434"),
    ("https://alice:s3cret@myhost/ollama", "https://myhost/ollama"),
    ("alice:s3cret@myhost", "myhost"),
    ("alice:s3c@ret@myhost:8080", "myhost:8080"),
    ("http://alice:pa#ss@myhost:11434", "http://myhost:11434"),
    ("http://myhost:11434", "http://myhost:11434"),
    ("myhost", "myhost"),
    # a hex-only login is no IPv6 address (F5)
    ("http://dad:fade/x@h:11434", "http://h:11434"),
    ("dad:fade/x@h:11434", "h:11434"),
    ("fe80::1", "fe80::1"),
    ("[::1]:8080", "[::1]:8080"),
    # an "@" in the path is no login (F2)
    ("http://h/p@x", "http://h/p@x"),
    ("http://h:11434/p@x", "http://h:11434/p@x"),
    ("http://alice:s3c@ret@h/p@x", "http://h/p@x"),
    ("http://[::1]:11434", "http://[::1]:11434"),
]


# --------------------------------------------------------------------------- A639: not saved


@pytest.mark.parametrize(("raw", "want"), LOGINS)
def test_without_login_leaves_out_user_and_password(raw, want):
    assert backends.without_login(raw) == want


@pytest.mark.parametrize("raw", [r for r, _ in LOGINS])
@pytest.mark.parametrize("form", ["separate", "equals", "abbreviated"])
def test_recorded_argv_has_no_login(raw, form):
    head = ["crashes", "p", "--json"]
    opt = {
        "separate": ["--ollama-url", raw],
        "equals": [f"--ollama-url={raw}"],
        "abbreviated": ["--ollama-u", raw],
    }[form]
    got = runstate.recorded_argv([*head, *opt, "--ollama-model", "m"])
    assert "s3c" not in json.dumps(got) and "alice" not in json.dumps(got)
    assert got[:3] == head and got[-2:] == ["--ollama-model", "m"]
    assert len(got) == len(head) + len(opt) + 2


def test_recorded_argv_keeps_what_follows_a_double_dash():
    argv = ["analyze", "--", "--ollama-url=http://a:b@h"]
    assert runstate.recorded_argv(argv) == argv


def test_a_bare_url_with_a_login_parses():
    parts, port = backends.split_ollama_url("alice:s3cret@myhost:11434", "--ollama-url")
    assert (parts.hostname, port, parts.username) == ("myhost", 11434, "alice")
    assert (
        backends.normalize_ollama_url("alice:s3cret@::1", "x") == "http://alice:s3cret@[::1]:11434"
    )


@pytest.mark.parametrize(
    "url",
    ["http://alice:s3c#ret@myhost:11434", "alice:s3c?ret@myhost", "alice:s3cret@myhost:99999"],
)
def test_a_bad_ollama_url_error_shows_no_login(home, capsys, url):
    with pytest.raises(ValueError, match="invalid port") as err:
        backends.split_ollama_url(url, "--ollama-url")
    assert "s3c" not in str(err.value) and "alice" not in str(err.value)
    with pytest.raises((SystemExit, cli.UsageError)) as stop:  # exit 2 (bad_args)
        cli.main(["crashes", "p", "--json", "--ollama-url", url])
    assert getattr(stop.value, "code", None) == 2 or "2" in str(stop.value)
    out, errs = capsys.readouterr()
    assert "s3c" not in out + errs and "alice" not in out + errs
    assert json.loads(out)["error"]["code"] == "bad_args"


@pytest.fixture
def project(home: Path) -> Path:
    p = home / "projects" / "p"
    p.mkdir(parents=True)
    cache.save_json(p / "highlights.json", {"sources": [{"path": "/videos/a.mp4"}]})
    return p


@pytest.mark.parametrize("url", ["http://alice:s3c@ret@myhost:11434", "alice:s3c@ret@myhost"])
def test_progress_json_records_the_ollama_url_without_its_login(project, monkeypatch, url):
    monkeypatch.setattr(cli, "_cmd_crashes", lambda a: None)
    assert cli.main(["crashes", "p", "--ollama-url", url]) == 0
    text = (project / runstate.FILE).read_text(encoding="utf-8")
    assert "s3c" not in text and "alice" not in text
    rec = json.loads(text)
    assert rec["argv"] == ["crashes", "p", "--ollama-url", backends.without_login(url)]
    # the options key is that of the URL without the login
    assert cli.main(["crashes", "p", f"--ollama-url={backends.without_login(url)}"]) == 0
    again = json.loads((project / runstate.FILE).read_text("utf-8"))
    assert again["options_key"] == rec["options_key"]


def test_an_old_highlights_json_loses_its_login_on_the_next_write(home, tmp_path, monkeypatch):
    """Strip on next write: a rewrite (here `crashes --context`) saves the URL without it."""
    from cutpoint import activity  # noqa: PLC0415

    hl = {
        "sources": [],
        "candidates": [],
        "settings": {"ollama_url": "http://alice:s3c@ret@myhost:11434", "vision_used": False},
    }
    proj = tmp_path / "proj"
    proj.mkdir()
    cache.save_json(proj / "highlights.json", hl)
    assert activity.record_context(proj, "downhill mtb")
    text = (proj / "highlights.json").read_text(encoding="utf-8")
    assert "s3c" not in text and "alice" not in text
    assert json.loads(text)["settings"]["ollama_url"] == "http://myhost:11434"
    assert backends.saved_settings({"settings": None}) == {"settings": None}


# --------------------------------------------------------------------------- A639: re-supplied


def test_with_login_takes_the_login_of_ollama_host_for_the_same_server(home, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://alice:s3c%40ret@myhost:11434")
    assert backends.with_login("http://myhost:11434") == "http://alice:s3c%40ret@myhost:11434"
    # another port, host or scheme gets none
    assert backends.with_login("http://myhost:8080") == "http://myhost:8080"
    assert backends.with_login("http://other:11434") == "http://other:11434"
    assert backends.with_login("https://myhost:11434") == "https://myhost:11434"
    # a URL with its own login keeps it
    assert backends.with_login("http://bob:pw@myhost:11434") == "http://bob:pw@myhost:11434"


@pytest.mark.parametrize(
    "cfg",
    [
        {"ollama_url": "alice:s3cret@myhost"},  # bare: http and Ollama's port
        {"vision_assistant": {"provider": "ollama", "url": "http://alice:s3cret@myhost:11434"}},
        {"assistant": {"provider": "ollama", "url": "http://alice:s3cret@MyHost:11434/"}},
    ],
    ids=["ollama_url", "vision", "chat"],
)
def test_with_login_takes_the_login_of_config_json_for_the_same_server(home, cfg):
    set_config(home, **cfg)
    got = backends.with_login("http://myhost:11434")
    assert requests.utils.get_auth_from_url(got) == ("alice", "s3cret")


class Resp:
    def __init__(self, body: Any, status: int = 200) -> None:
        self.body, self.status_code, self.is_redirect = body, status, False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self) -> Any:
        return self.body


def _consented_remote(home: Path, **cfg: Any) -> None:
    set_config(home, vision_assistant={"provider": "ollama", "consent": [REMOTE_KEY]}, **cfg)


def test_a_replayed_url_without_login_is_sent_with_the_configured_one(home, monkeypatch):
    _consented_remote(home, ollama_url=f"http://alice:s3cret@{REMOTE}:11434")
    seen: list[dict[str, Any]] = []

    def get(url: str, **kw: Any) -> Resp:
        seen.append({"url": url, **kw})
        return Resp({"models": [{"name": "qwen3.5:9b", "digest": "d"}]})

    monkeypatch.setattr(backends.requests, "get", get)
    rater = vision.VisionRater(f"http://{REMOTE}:11434")  # as recorded in highlights.json
    assert rater.available()
    (call,) = seen
    assert call["url"] == f"http://{REMOTE}:11434/api/tags"  # no login in the URL itself
    assert (call["auth"].username, call["auth"].password) == ("alice", "s3cret")


@pytest.mark.parametrize("status", [401, 403])
def test_a_server_that_wants_a_login_skips_vision_with_a_warning(home, monkeypatch, caplog, status):
    _consented_remote(home)
    monkeypatch.setattr(backends.requests, "get", lambda url, **kw: Resp({}, status))
    rater = vision.VisionRater(f"http://{REMOTE}:11434")
    with caplog.at_level(logging.INFO):
        assert not rater.available()
    (warn,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    text = warn.getMessage()
    assert f"HTTP {status}" in text and "login" in text and "skipping vision" in text


# --------------------------------------------------------------------------- A653: file names

REPLY = {
    "events": [{"t": 1.0, "what": "a kill", "funny": 1, "hype": 6}],
    "summary": "A kill.",
    "funny_score": 1,
    "highlight_score": 6,
    "label": "kill",
}


def _sources(tmp_path: Path) -> dict[str, Source]:
    (tmp_path / "cache").mkdir(exist_ok=True)
    out = {}
    for name in ("first_session.mp4", "secret_meeting.mp4"):
        src = Source(tmp_path / name, 60.0, tmp_path / "cache" / name, [])
        src.cdir.mkdir(exist_ok=True)
        out[str(src.path)] = src
    return out


def _prompts(tmp_path, monkeypatch, rater, *, action_cam: bool = False) -> list[str]:
    """The text each window of the second source sent, through describe_windows."""
    sources = _sources(tmp_path)
    second = list(sources)[1]
    monkeypatch.setattr(vw, "grab_frames", lambda *a, **k: [b"\xff\xd8x"])
    sent: list[str] = []
    real = rater.describe

    def spy(prompt: str, images: list[bytes], schema: dict, **kw: Any):
        sent.append(prompt)
        return real(prompt, images, schema, **kw)

    monkeypatch.setattr(rater, "describe", spy)
    params = vw.VlmParams(model="m", action_cam=vw.default_wording() if action_cam else None)
    lines = [{"name": "secret_meeting.mp4", "start": 6.0, "end": 8.0, "text": "nice shot"}]
    win = vw.Window(second, 5.0, 20.0, 1.0, ["seed"])
    vw.describe_windows(rater, sources, [win], params, {}, lines, {}, force=True)
    return sent


def _ollama(monkeypatch) -> None:
    def post(url: str, **_kw: Any) -> Resp:
        return Resp({"message": {"content": json.dumps(REPLY)}})

    monkeypatch.setattr(backends.requests, "post", post)
    monkeypatch.setattr(vision.memguard, "check_vision", lambda rater: None)
    monkeypatch.setattr(vision.memguard, "ollama_context", lambda *a: None)


@pytest.mark.parametrize("action_cam", [False, True], ids=["game", "action-cam"])
def test_a_local_ollama_gets_the_file_name(home, tmp_path, monkeypatch, action_cam):
    _ollama(monkeypatch)
    rater = vision.VisionRater()
    assert rater.local
    (prompt,) = _prompts(tmp_path, monkeypatch, rater, action_cam=action_cam)
    assert "in secret_meeting.mp4." in prompt and "nice shot" in prompt


@pytest.mark.parametrize("action_cam", [False, True], ids=["game", "action-cam"])
def test_a_remote_ollama_gets_source_n(home, tmp_path, monkeypatch, action_cam):
    _consented_remote(home)
    _ollama(monkeypatch)
    rater = vision.VisionRater(f"http://{REMOTE}:11434")
    assert not rater.local
    (prompt,) = _prompts(tmp_path, monkeypatch, rater, action_cam=action_cam)
    assert "in source 2." in prompt and "nice shot" in prompt  # the transcript still goes
    assert "secret_meeting" not in prompt and ".mp4" not in prompt


class FakeSubscription:
    name, model, local, host = "codex", "gpt-x", False, "chatgpt.com"

    def __init__(self) -> None:
        self.sent: list[list[dict[str, Any]]] = []

    def capabilities(self) -> None:
        pass

    def chat(self, messages, **kw) -> Reply:
        self.sent.append(messages)
        return Reply(json.dumps(REPLY))


def test_a_hosted_vision_role_gets_source_n(home, tmp_path, monkeypatch):
    fake = FakeSubscription()
    monkeypatch.setattr(providers, "make", lambda *_a, **_k: fake)
    monkeypatch.setattr(vision.memguard, "check_vision", lambda rater: None)
    set_config(
        home,
        vision_assistant={
            "provider": "codex",
            "model": "gpt-x",
            "consent": ["vision-v2:codex@chatgpt.com"],
        },
    )
    rater = vision.VisionRater()
    assert rater.via is not None and not rater.local
    (prompt,) = _prompts(tmp_path, monkeypatch, rater)
    assert "in source 2." in prompt
    assert "secret_meeting" not in json.dumps(fake.sent) and ".mp4" not in json.dumps(fake.sent)
