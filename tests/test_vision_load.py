"""``VisionRater.load()`` and ``VisionRater.on_gpu()`` (#845) at HTTP level: ``requests`` is
replaced, so no Ollama and no network."""

from __future__ import annotations

from typing import Any

import pytest
import requests

from cutpoint import vision

URL = "http://127.0.0.1:1"


class _Resp:
    is_redirect = False
    status_code = 200

    def __init__(self, body: Any = None, *, status: int = 200, bad_json: bool = False) -> None:
        self.body, self.status_code, self.bad_json = body, status, bad_json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        if self.bad_json:
            raise ValueError("not json")
        return self.body


def _redirect() -> requests.Response:
    resp = requests.Response()
    resp.status_code = 307
    resp.headers["Location"] = "http://elsewhere.example.com/api/generate"
    return resp


@pytest.fixture
def rater(monkeypatch) -> vision.VisionRater:
    monkeypatch.setattr(vision.memguard, "check_vision", lambda _r: None)
    return vision.VisionRater(url=URL, model="qwen-test")


def _post(monkeypatch, reply) -> list[dict]:
    sent: list[dict] = []

    def post(url, json=None, timeout=None, allow_redirects=True):
        sent.append({"url": url, "json": json, "timeout": timeout, "redir": allow_redirects})
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(vision.requests, "post", post)
    return sent


def _get(monkeypatch, reply) -> None:
    def get(url, timeout=None, allow_redirects=True):
        assert url == f"{URL}/api/ps" and allow_redirects is False
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(vision.requests, "get", get)


def test_load_success_sends_a_promptless_generate(monkeypatch, rater):
    sent = _post(monkeypatch, _Resp({"done": True}))
    assert rater.load(30) is True
    assert rater.timed_out is False and rater.used is True
    assert sent == [
        {
            "url": f"{URL}/api/generate",
            "json": {"model": "qwen-test"},
            "timeout": 30,
            "redir": False,
        }
    ]


@pytest.mark.parametrize(
    ("reply", "timed_out"),
    [
        (requests.ReadTimeout("slow"), True),
        (requests.ConnectTimeout("slow"), True),
        (_Resp(status=400), False),
        (_Resp(status=500), False),
        (requests.ConnectionError("refused"), False),
        (_redirect(), False),
    ],
    ids=["read-timeout", "connect-timeout", "400", "500", "refused", "redirect"],
)
def test_load_failure(monkeypatch, rater, reply, timed_out):
    _post(monkeypatch, reply)
    assert rater.load(5) is False
    assert rater.timed_out is timed_out


def test_load_resets_timed_out(monkeypatch, rater):
    _post(monkeypatch, requests.ReadTimeout("slow"))
    assert rater.load(5) is False and rater.timed_out is True
    _post(monkeypatch, _Resp({"done": True}))
    assert rater.load(5) is True and rater.timed_out is False


def test_load_hosted_sends_nothing(monkeypatch, rater):
    sent = _post(monkeypatch, _Resp({}))
    rater.via = object()  # type: ignore[assignment]
    assert rater.load(5) is True
    assert sent == [] and rater.used is False


@pytest.mark.parametrize(
    ("models", "expected"),
    [
        ([{"name": "qwen-test", "size_vram": 5000}], True),
        ([{"name": "qwen-test:latest", "size_vram": 5000}], True),
        ([{"model": "qwen-test", "size_vram": 5000}], True),
        ([{"name": "other", "size_vram": 9}, {"model": "qwen-test", "size_vram": 1}], True),
        ([{"name": "qwen-test", "size_vram": 0}], False),
        ([{"name": "qwen-test"}], None),
        ([{"name": "qwen-test", "size_vram": "lots"}], None),
        ([{"name": "other", "size_vram": 5000}], None),
        ([], None),
        (None, None),
        ("nonsense", None),
    ],
    ids=[
        "name",
        "latest",
        "model-field",
        "second-entry",
        "cpu-only",
        "no-vram-field",
        "vram-not-int",
        "not-listed",
        "empty",
        "null",
        "not-a-list",
    ],
)
def test_on_gpu_from_api_ps(monkeypatch, rater, models, expected):
    _get(monkeypatch, _Resp({"models": models}))
    assert rater.on_gpu() is expected


@pytest.mark.parametrize(
    "reply",
    [
        _Resp(bad_json=True),
        _Resp([1, 2]),  # a JSON list: no .get
        _Resp(status=500),
        requests.ConnectionError("refused"),
        requests.ReadTimeout("slow"),
        _redirect(),
    ],
    ids=["bad-json", "json-not-object", "500", "refused", "timeout", "redirect"],
)
def test_on_gpu_unknown_when_no_usable_answer(monkeypatch, rater, reply):
    _get(monkeypatch, reply)
    assert rater.on_gpu() is None


def test_on_gpu_hosted_asks_nothing(monkeypatch, rater):
    def get(*_a, **_k):
        raise AssertionError("no request expected")

    monkeypatch.setattr(vision.requests, "get", get)
    rater.via = object()  # type: ignore[assignment]
    assert rater.on_gpu() is None


def _chat_ok() -> _Resp:
    return _Resp({"message": {"content": '{"score": 7}'}})


def _describe(rater: vision.VisionRater, timeout: float = 5) -> dict | None:
    return rater.describe("rate", [b"jpg"], {"type": "object"}, timeout=timeout)


@pytest.mark.parametrize(
    ("reply", "timed_out"),
    [
        (requests.ReadTimeout("slow"), True),
        (requests.ConnectTimeout("slow"), True),
        (requests.ConnectionError("refused"), False),
        (_Resp(status=500), False),
    ],
    ids=["read-timeout", "connect-timeout", "refused", "500"],
)
def test_describe_failure_sets_timed_out_only_for_timeouts(monkeypatch, rater, reply, timed_out):
    sent = _post(monkeypatch, reply)
    assert _describe(rater, timeout=12) is None
    assert rater.timed_out is timed_out
    assert sent[0]["url"] == f"{URL}/api/chat" and sent[0]["timeout"] == 12


def test_describe_resets_timed_out_after_a_success(monkeypatch, rater):
    _post(monkeypatch, requests.ReadTimeout("slow"))
    assert _describe(rater) is None and rater.timed_out is True
    _post(monkeypatch, _chat_ok())
    assert _describe(rater) == {"score": 7} and rater.timed_out is False
