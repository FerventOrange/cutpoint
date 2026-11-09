"""The vision model reuses the context Ollama has it loaded with (#732): the AI Chat loads
qwen3.5:9b with num_ctx 50000 and Ollama reloads a model asked for another context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import requests

from cutpoint import memguard, vision
from test_resume import _Resp

URL = "http://127.0.0.1:11434"
MODEL = "qwen3.5:9b"
# the probe itself: conftest's no_host_probes turns it off for every other test
_ollama_context = memguard.ollama_context


@pytest.fixture(autouse=True)
def probe(monkeypatch):
    monkeypatch.setattr(memguard, "ollama_context", _ollama_context)


class _Ollama:
    """Ollama on this machine: one loaded model (``ctx`` None: none), reloaded by a request for
    another context (counted in ``loads``); ``ps``: False makes /api/ps fail."""

    def __init__(self, ctx: int | None = None, *, reply: str = '{"score": 7, "label": "x"}'):
        self.ctx, self.reply, self.ps = ctx, reply, True
        self.loads = 0
        self.sent: list[int] = []  # num_ctx of each chat/generate request
        self.ps_asked = 0

    def get(self, url: str, **_kw: Any) -> _Resp:
        assert url == f"{URL}/api/ps"
        self.ps_asked += 1
        if not self.ps:
            raise requests.ConnectionError("down")
        models = [] if self.ctx is None else [{"name": MODEL, "context_length": self.ctx}]
        return _Resp({"models": models})

    def post(self, url: str, json: dict, **_kw: Any) -> _Resp:
        ctx = json["options"]["num_ctx"]
        self.sent.append(ctx)
        load = 0.0
        if ctx != self.ctx:
            self.ctx, self.loads, load = ctx, self.loads + 1, 7.0
        body: dict[str, Any] = {"load_duration": load, "done_reason": "stop"}
        if url.endswith("/api/generate"):
            return _Resp(body | {"response": self.reply})
        return _Resp(body | {"message": {"content": self.reply}})


@pytest.fixture
def ollama(home, monkeypatch):
    fake = _Ollama()
    monkeypatch.setattr(requests, "get", fake.get)
    monkeypatch.setattr(requests, "post", fake.post)
    return fake


@pytest.fixture
def guarded(monkeypatch):
    """The num_ctx each memory guard check was for."""
    checks: list[int | None] = []
    monkeypatch.setattr(memguard, "check_vision", lambda r: checks.append(r.num_ctx))
    return checks


def _chat(ollama: _Ollama) -> None:
    """An AI Chat turn: num_ctx 50000 (assistant.providers.OLLAMA_CONTEXT_TOKENS)."""
    requests.post(f"{URL}/api/chat", json={"model": MODEL, "options": {"num_ctx": 50000}})


def test_describe_reuses_the_larger_context_the_chat_loaded(ollama, guarded):
    ollama.ctx = 50000
    rater = vision.VisionRater(URL, MODEL)
    assert rater.describe("q", [b"x"], {}) == {"score": 7, "label": "x"}
    assert ollama.sent == [50000] and ollama.loads == 0
    assert guarded == [50000]


def test_a_request_never_asks_for_less_than_it_needs(ollama, guarded):
    rater = vision.VisionRater(URL, MODEL)
    rater.describe("q", [b"x"], {})  # nothing loaded: what it needs
    ollama.ctx = 8192  # loaded smaller: still what it needs
    rater.describe("q", [b"x"], {})
    assert ollama.sent == [16384, 16384]


def test_ask_json_reuses_the_larger_context(ollama, guarded):
    ollama.ctx = 50000
    rater = vision.VisionRater(URL, MODEL)
    assert rater.ask_json("q", num_ctx=16384) == {"score": 7, "label": "x"}
    assert ollama.sent == [50000]


def test_interleaved_chat_and_vision_load_the_model_once(ollama, guarded):
    rater = vision.VisionRater(URL, MODEL)
    _chat(ollama)
    for _ in range(3):
        rater.describe("q", [b"x"], {})
        _chat(ollama)
    assert ollama.loads == 1  # the chat's first load; no reload per switch


def test_the_retry_ladder_still_climbs_when_nothing_larger_is_loaded(
    ollama, guarded, monkeypatch, tmp_path
):
    monkeypatch.setattr(vision, "grab_frame_jpeg", lambda *a, **k: b"jpeg")
    monkeypatch.setattr(vision, "rating_profile", lambda _m: None)  # the frames as they are
    ollama.reply = "not json"
    rater = vision.VisionRater(URL, MODEL)
    assert rater._rate_frames(Path(tmp_path / "a.mp4"), [1.0, 2.0], "gaming") is None
    assert ollama.sent == [a[0] for a in vision._ATTEMPTS]  # 8192, then 12288
    assert guarded == [8192, 12288]  # each context change is checked
    ollama.sent.clear()
    ollama.ctx = 50000
    rater._rate_frames(Path(tmp_path / "a.mp4"), [1.0, 2.0], "gaming")
    assert ollama.sent == [50000, 50000]


def test_ps_failing_keeps_what_the_request_needs(ollama, guarded):
    ollama.ctx, ollama.ps = 50000, False
    rater = vision.VisionRater(URL, MODEL)
    rater.describe("q", [b"x"], {})
    assert ollama.sent == [16384]


def test_an_ollama_on_another_machine_is_not_asked(ollama):
    rater = vision.VisionRater(URL, MODEL)
    rater.url = "http://192.168.1.5:11434"
    assert rater._context(8192) == 8192
    assert ollama.ps_asked == 0


def test_the_guard_checks_again_only_for_another_context(monkeypatch):
    checks: list[int | None] = []
    monkeypatch.setattr(memguard, "check_vision", lambda r: checks.append(r.num_ctx))
    rater = vision.VisionRater(URL, MODEL)
    rater.check_memory(8192)
    rater.used = True
    rater.check_memory(8192)  # loaded with this context: no check
    rater.check_memory(50000)  # Ollama reloads it: checked
    rater.check_memory()  # a hosted Vision role names none: no check once used
    assert checks == [8192, 50000]


# -- memguard ---------------------------------------------------------------------------------


def _ps(monkeypatch, models: list[dict]) -> None:
    monkeypatch.setattr(requests, "get", lambda url, **kw: _Resp({"models": models}))


def test_ollama_context_reads_context_length(monkeypatch):
    _ps(
        monkeypatch,
        [{"name": "llava", "context_length": 4096}, {"name": MODEL, "context_length": 50000}],
    )
    assert memguard.ollama_context(URL, MODEL) == 50000
    assert memguard.ollama_context(URL, "other") is None
    _ps(monkeypatch, [{"name": MODEL}])  # an Ollama that does not report it
    assert memguard.ollama_context(URL, MODEL) is None
    _ps(monkeypatch, [{"name": MODEL, "context_length": True}])
    assert memguard.ollama_context(URL, MODEL) is None


def test_a_pending_context_change_is_not_loaded(monkeypatch):
    _ps(monkeypatch, [{"model": MODEL, "context_length": 8192}])
    assert memguard._ollama_loaded(URL, MODEL, 8192)
    assert not memguard._ollama_loaded(URL, MODEL, 50000)  # Ollama would reload it
    assert memguard._ollama_loaded(URL, MODEL)  # no context named: the name alone
    _ps(monkeypatch, [{"name": f"{MODEL}:latest"}])  # no context_length: the name alone
    assert memguard._ollama_loaded(URL, MODEL, 50000)
