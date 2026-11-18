"""The one Ollama client (:mod:`cutpoint.backends`, #1085 A328): a URL password never shows in
errors or logs (A641), and an answer of another JSON shape is an ordinary failure (A116).

``requests`` is answered by a stand-in transport (``HTTPAdapter.send``) or a stubbed
``requests.post``/``get``; no Ollama and no network."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pytest
import requests

from cutpoint import backends, cli, vision
from cutpoint.assistant.providers import OllamaProvider, ProviderError, public
from cutpoint.events import common

SECRET = "s3cret"
URL = f"http://user:{SECRET}@127.0.0.1:38797"


class _Server:
    """``HTTPAdapter.send`` stand-in: ``answers`` maps a path to ``(status, body)``; any
    other path gets a 404. Records what was sent."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.sent: list[tuple[requests.PreparedRequest, dict[str, Any]]] = []
        monkeypatch.setattr(
            requests.adapters.HTTPAdapter, "send", lambda _a, req, **kw: self._send(req, **kw)
        )

    def _send(self, request: requests.PreparedRequest, **kw: Any) -> requests.Response:
        self.sent.append((request, kw.get("proxies") or {}))
        path = request.path_url
        status, body = self.answers.get(path, (404, {"error": "not found"}))
        resp = requests.Response()
        resp.status_code, resp.reason = status, "Not Found" if status == 404 else "OK"
        resp._content = json.dumps(body).encode()
        resp.request, resp.url = request, request.url  # the URL as sent: with user:pw@
        return resp


@pytest.fixture
def quiet_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vision.memguard, "check_vision", lambda _r: None)
    monkeypatch.setattr(vision, "grab_frame_jpeg", lambda _src, _t, **_k: b"jpeg")


# --------------------------------------------------------------------------- A641


def test_a_404_error_has_no_password(monkeypatch: pytest.MonkeyPatch) -> None:
    _Server(monkeypatch, {})
    with pytest.raises(requests.HTTPError) as got:
        backends.ollama_get(URL, "/api/tags", timeout=1)
    assert got.value.response is not None and got.value.response.status_code == 404
    assert SECRET not in str(got.value) and "user@" not in str(got.value)
    assert "127.0.0.1:38797/api/tags" in str(got.value)


def test_a_request_error_has_no_password(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(url: str, **_k: Any) -> Any:
        raise requests.ConnectionError(f"cannot reach {url}")

    monkeypatch.setattr(requests, "post", refused)
    with pytest.raises(requests.ConnectionError) as got:
        backends.ollama_post(URL, "/api/chat", {}, timeout=1)
    assert SECRET not in str(got.value)


def test_an_invalid_port_error_has_no_password() -> None:
    with pytest.raises(ValueError, match="invalid port") as got:
        backends.normalize_ollama_url(f"http://user:{SECRET}@host:99999", "--ollama-url")
    assert SECRET not in str(got.value)


def test_a_no_host_cli_error_has_no_password() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="no host") as got:
        cli._ollama_url_arg(f"http://user:{SECRET}@:11434")
    assert SECRET not in str(got.value)


@pytest.mark.parametrize("password", ["pa/ss", "p@ss"])
def test_a_password_with_a_slash_or_at_never_shows(password: str) -> None:
    """An unencoded "/" makes urlsplit read the netloc as ``user:pa``: the port error."""
    assert backends.strip_userinfo(f"http://user:{password}@host:1") == "http://host:1"
    with pytest.raises(ValueError, match="invalid port") as got:
        backends.normalize_ollama_url("http://user:pa/ss@host", "--ollama-url")
    assert "user" not in str(got.value) and "ss" not in str(got.value)
    with pytest.raises(argparse.ArgumentTypeError, match=r"no host|invalid port") as got:
        cli._ollama_url_arg(f"http://user:{password}@:11434")
    assert "user" not in str(got.value) and "ss" not in str(got.value)


def test_strip_userinfo_keeps_text_without_a_login() -> None:
    assert backends.strip_userinfo("http://host:1/api/tags") == "http://host:1/api/tags"
    assert backends.strip_userinfo("http://u:p@s/s@h:1") == "http://h:1"
    assert backends.strip_userinfo("a@b http://u:p/x@h/api") == "a@b http://h/api"


def test_the_vision_paths_log_no_password(monkeypatch, quiet_guard, caplog) -> None:
    _Server(monkeypatch, {})
    rater = vision.VisionRater(url=URL, model="m")
    with caplog.at_level(logging.DEBUG):
        assert rater.available() is False
        assert rater.ask_json("p", schema=None) is None
        assert rater.describe("p", [b"jpeg"], {"type": "object"}) is None
        assert rater.rate(Path("a.mp4"), 0, 4, "action") is None
        assert rater.load(5) is False
        rater.unload()
    assert "404" in caplog.text
    assert SECRET not in caplog.text


def test_the_events_unload_logs_no_password(monkeypatch, caplog) -> None:
    _Server(monkeypatch, {"/api/ps": (200, {"models": [{"name": "m"}]})})  # unload: a 404
    with caplog.at_level(logging.DEBUG):
        assert common.unload_ollama(URL) == []
    assert "Could not unload Ollama model m: 404" in caplog.text
    assert SECRET not in caplog.text


def test_the_assistant_errors_have_no_password(monkeypatch, caplog) -> None:
    def refused(url: str, **_k: Any) -> Any:
        raise requests.ConnectionError(f"cannot reach {url}")

    monkeypatch.setattr(requests, "get", refused)
    monkeypatch.setattr(requests, "post", refused)
    provider = OllamaProvider(URL, "m", 0)
    with caplog.at_level(logging.DEBUG), pytest.raises(ProviderError) as got:
        provider.list_models()
    with pytest.raises(ProviderError) as posted:
        provider.capabilities()
    for text in (str(got.value), str(posted.value), caplog.text):
        assert SECRET not in text


def test_the_url_login_is_sent_but_never_a_netrc_one(monkeypatch, tmp_path: Path) -> None:
    """The URL's own user:password goes as basic auth (what requests did before); without one
    no ~/.netrc login is sent, and a server on this machine skips the environment's proxy."""
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 127.0.0.1 login someone password from-netrc\n", encoding="utf-8")
    netrc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc))
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    server = _Server(monkeypatch, {"/api/tags": (200, {"models": []})})
    backends.ollama_get(URL, "/api/tags", timeout=1)
    backends.ollama_get("http://127.0.0.1:38797", "/api/tags", timeout=1)
    (with_login, proxies), (without, _) = server.sent
    assert with_login.headers["Authorization"].startswith("Basic ")
    assert "Authorization" not in without.headers
    assert not any(proxies.values())


# --------------------------------------------------------------------------- A116


class _Body:
    is_redirect = False
    status_code = 200

    def __init__(self, body: Any) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return self.body


BAD_REPLIES = [[1, 2], {"message": None}, {"message": "text"}, {"message": {"content": 5}}]
BAD_IDS = ["list", "message-null", "message-text", "content-not-text"]


@pytest.mark.parametrize("body", BAD_REPLIES, ids=BAD_IDS)
def test_a_reply_of_another_shape_is_a_failed_request(monkeypatch, quiet_guard, body) -> None:
    monkeypatch.setattr(requests, "post", lambda *_a, **_k: _Body(body))
    rater = vision.VisionRater(url="http://127.0.0.1:1", model="m")
    assert rater.ask_json("p", schema=None, images=[b"jpeg"]) is None  # /api/chat
    assert rater.describe("p", [b"jpeg"], {"type": "object"}) is None
    assert rater.rate(Path("a.mp4"), 0, 4, "action") is None  # _rate_frames
    qwen = vision.VisionRater(url="http://127.0.0.1:1", model="qwen3.5:9b")
    assert qwen.ask_json("p", schema=None) is None  # /api/generate: no "response"


@pytest.mark.parametrize(
    "body",
    [[1, 2], {"models": [1, 2]}, {"models": "m"}, {"models": [{"name": 5}, {"name": "m"}]}],
    ids=["list", "models-not-objects", "models-text", "name-not-text"],
)
def test_a_model_list_of_another_shape_is_no_crash(monkeypatch, body) -> None:
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: _Body(body))
    rater = vision.VisionRater(url="http://127.0.0.1:1", model="m")
    has_m = isinstance(body, dict) and body["models"] == [{"name": 5}, {"name": "m"}]
    assert rater.available() is has_m
    assert rater.on_gpu() is None


def test_events_unload_skips_entries_of_another_shape(monkeypatch) -> None:
    monkeypatch.setattr(common.time, "sleep", lambda _s: None)
    answers = iter([{"models": [1, None, {"name": "m"}]}, {"models": []}])
    monkeypatch.setattr(requests, "get", lambda *_a, **_k: _Body(next(answers)))
    posted: list[Any] = []
    monkeypatch.setattr(requests, "post", lambda _u, json, **_k: posted.append(json) or _Body({}))
    assert common.unload_ollama("http://127.0.0.1:1") == ["m"]
    assert posted == [{"model": "m", "keep_alive": 0}]


def test_ollama_reply_reads_chat_and_generate_answers() -> None:
    assert backends.ollama_reply({"message": {"content": "hi"}}) == "hi"
    assert backends.ollama_reply({"response": "hi"}, generate=True) == "hi"
    with pytest.raises(ValueError, match="no reply text"):
        backends.ollama_reply({"response": "hi"})


def test_public_strips_a_password_with_a_slash_or_at() -> None:
    for url in ("http://user:pa/ss@host:1", "http://user:p@ss@host:1"):
        shown = public(f"{url} is not reachable")
        assert "pa" not in shown.replace("path", "") and "ss" not in shown
        assert shown == "http://host:1 is not reachable"
    assert public("http://host:1/v1") == "http://host:1/v1"


# --------------------------------------------------------------------------- A298


@pytest.mark.parametrize(
    ("given", "sent"),
    [
        (300, (backends.CONNECT_TIMEOUT_S, 300)),
        (30.0, (backends.CONNECT_TIMEOUT_S, 30.0)),
        (5, (5, 5)),  # the connection never gets longer than the number given
        ((1, 3), (1, 3)),  # a pair as given
    ],
)
def test_a_read_timeout_gets_a_short_connect_timeout(monkeypatch, given, sent) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(requests, "post", lambda _u, **kw: seen.append(kw["timeout"]) or _Body({}))
    backends.ollama_post("http://127.0.0.1:1", "/api/chat", {}, timeout=given)
    assert seen == [sent]


def test_every_ollama_request_connects_within_the_connect_timeout(monkeypatch, quiet_guard) -> None:
    """A black-holed Ollama fails within CONNECT_TIMEOUT_S on every path: the vision rating,
    the events labelling and unload, the crash and profile steps (ask_json) and the AI Chat."""
    seen: list[tuple[str, Any]] = []

    def black_hole(url: str, **kw: Any) -> Any:
        seen.append((url.rsplit("/", 1)[-1], kw["timeout"]))
        raise requests.ConnectTimeout("timed out")

    monkeypatch.setattr(requests, "post", black_hole)
    rater = vision.VisionRater(url="http://127.0.0.1:1", model="m")
    rater.used = True
    assert rater.ask_json("p", schema=None) is None
    assert rater.describe("p", [b"jpeg"], {"type": "object"}) is None
    assert rater.describe("p", [b"jpeg"], {"type": "object"}, timeout=3) is None
    assert rater.timed_out is True  # a stalled server: no retry of the window, as before
    assert rater.rate(Path("a.mp4"), 0, 4, "action") is None
    assert rater.load(60) is False
    rater.unload()
    monkeypatch.setattr(
        requests, "get", lambda *_a, **_k: _Body({"models": [{"name": "m"}]})
    )  # events: what is loaded
    common.unload_ollama("http://127.0.0.1:1")
    with pytest.raises(ProviderError, match="not reachable"):
        OllamaProvider("http://127.0.0.1:1", "m", 0).capabilities()
    connect = backends.CONNECT_TIMEOUT_S
    assert seen[:6] == [
        ("chat", (connect, 600)),
        ("chat", (connect, 600)),
        ("chat", (3, 3)),
        ("chat", (connect, 300)),
        ("generate", (connect, 60)),
        ("generate", (connect, 10)),
    ]
    assert seen[6] == ("generate", (connect, 30))
    assert seen[-1] == ("show", (5.0, 5.0))
