"""Ollama vision rating: per-style prompts on the retry and unloading the model after a run.

HTTP and frame grabs are stubbed; no Ollama, no ffmpeg (except the analyze test).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar

import pytest
import requests

import cutpoint.analyze
from conftest import needs_ffmpeg
from cutpoint import cli, vision


class _Resp:
    def __init__(self, body: dict) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self.body


@pytest.fixture
def posts(monkeypatch) -> list[tuple[str, dict]]:
    """Every POST as (url, payload); chat replies are empty (done_reason=length) until the
    retry, which answers with a score."""
    seen: list[tuple[str, dict]] = []

    def post(url, json, timeout):
        seen.append((url, json))
        if url.endswith("/api/chat"):
            chats = sum(u.endswith("/api/chat") for u, _ in seen)
            content = "" if chats % 2 else '{"score": 7, "label": "big jump"}'
            return _Resp({"message": {"content": content}, "done_reason": "length"})
        return _Resp({"done": True})

    monkeypatch.setattr(vision.requests, "post", post)
    monkeypatch.setattr(vision, "grab_frame_jpeg", lambda _src, _t: b"jpeg")
    return seen


def _chat_prompts(posts: list[tuple[str, dict]]) -> list[str]:
    return [p["messages"][0]["content"] for u, p in posts if u.endswith("/api/chat")]


def test_ski_retry_uses_the_ski_prompt(posts):
    res = vision.VisionRater(url="http://x:1", model="m").rate(Path("a.mp4"), 0, 4, "ski")
    assert res == vision.VisionResult(7.0, "big jump")
    first, retry = _chat_prompts(posts)
    assert first == vision._PROMPT_SKI.format(n=3, style="ski")
    assert retry == vision._RETRY_PROMPT_SKI.format(n=3, style="ski")
    assert "scoreboard" not in retry and "ski" in retry


def test_other_styles_keep_the_default_prompts(posts):
    vision.VisionRater(url="http://x:1", model="m").rate(Path("a.mp4"), 0, 4, "gaming")
    first, retry = _chat_prompts(posts)
    assert first == vision._PROMPT.format(n=3, style="gaming")
    assert retry == vision._RETRY_PROMPT.format(n=3, style="gaming")


def test_unload_sends_keep_alive_zero_for_the_model_used(posts):
    rater = vision.VisionRater(url="http://x:1", model="qwen3-vl:8b")
    rater.rate(Path("a.mp4"), 0, 4, "ski")
    posts.clear()
    rater.unload()
    assert posts == [("http://x:1/api/generate", {"model": "qwen3-vl:8b", "keep_alive": 0})]
    rater.unload()  # already unloaded: nothing more to send
    assert len(posts) == 1


def test_unload_does_nothing_when_the_model_was_not_used(posts):
    vision.VisionRater(url="http://x:1", model="m").unload()
    assert posts == []


def test_unload_failure_is_not_fatal(monkeypatch, caplog):
    def post(*_a, **_k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(vision.requests, "post", post)
    rater = vision.VisionRater(url="http://x:1", model="m")
    rater.used = True
    with caplog.at_level(logging.DEBUG, logger="cutpoint.vision"):
        rater.unload()  # no exception
    assert [r.levelno for r in caplog.records] == [logging.DEBUG]
    assert "Could not unload Ollama model m" in caplog.text


class _FakeRater:
    """A reachable Ollama with the model pulled; records every call."""

    calls: ClassVar[list[str]] = []
    fail = False

    def __init__(self, **_kw) -> None:
        _FakeRater.calls.append("init")

    def available(self) -> bool:
        _FakeRater.calls.append("available")
        return True

    def rate(self, *_a, **_k):
        _FakeRater.calls.append("rate")
        if _FakeRater.fail:
            raise RuntimeError("boom")
        return vision.VisionResult(5.0, "moment")

    def unload(self) -> None:
        _FakeRater.calls.append("unload")


@pytest.fixture
def rater(monkeypatch) -> type[_FakeRater]:
    monkeypatch.setattr(cutpoint.analyze, "VisionRater", _FakeRater)
    monkeypatch.setattr(_FakeRater, "calls", [])
    return _FakeRater


def _argv(video: Path, *flags: str, style: str = "commentary") -> list[str]:
    """``--skip`` leaves out the HUD/OCR discovery a gaming style would run (minutes with the
    events extras); none of these tests is about those steps."""
    skip = ["--skip", "reactions", "keyframes", "crashes"]
    return ["analyze", str(video), "--style", style, "--out", "v", "--no-scenes", *flags, *skip]


@needs_ffmpeg
def test_analyze_does_not_use_vision_by_default_even_when_ollama_is_reachable(video, home, rater):
    assert cli.main(_argv(video, style="gaming")) == 0
    assert rater.calls == []  # not even asked whether Ollama is there
    st = json.loads((home / "projects" / "v" / "highlights.json").read_text("utf-8"))["settings"]
    assert (st["vision_requested"], st["vision_used"]) == (False, False)


@needs_ffmpeg
@pytest.mark.parametrize("fail", [False, True])
def test_analyze_vision_rates_then_unloads_the_model(video, home, rater, monkeypatch, fail):
    monkeypatch.setattr(_FakeRater, "fail", fail)
    assert (cli.main(_argv(video, "--vision")) == 0) is not fail
    assert rater.calls[:3] == ["init", "available", "rate"]
    assert rater.calls[-1] == "unload" and rater.calls.count("unload") == 1
