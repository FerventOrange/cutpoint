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
from cutpoint.styles import get_style


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


@pytest.mark.parametrize(
    ("content", "want"),
    [
        ('Sure: {"score": 7, "label": "x"} done', {"score": 7, "label": "x"}),
        ("no json here", None),
        ("{not json}", None),
    ],
)
def test_ask_json_and_parse_share_the_reply_json(monkeypatch, content, want):
    monkeypatch.setattr(
        vision.requests, "post", lambda *_a, **_k: _Resp({"message": {"content": content}})
    )
    assert vision.VisionRater(url="http://x:1", model="m").ask_json("q") == want
    assert vision._parse(content) == (vision.VisionResult(7.0, "x") if want else None)


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


@pytest.mark.parametrize(("weight", "high", "low"), [(0.25, 50.0, 30.0), (0.45, 58.0, 22.0)])
def test_vision_blend_is_a_bounded_multiplier(weight, high, low):
    """#284: 5/10 is neutral, 10/10 lifts by ``weight`` (the style's vision_weight), 0/10
    lowers by as much."""
    blend = cutpoint.analyze.vision_blend
    assert blend(40.0, 5.0, weight) == 40.0
    assert blend(40.0, 10.0, weight) == high
    assert blend(40.0, 0.0, weight) == low


def test_vision_blend_keeps_an_unrated_score():
    assert cutpoint.analyze.vision_blend(23.4, None, 0.3) == 23.4


@pytest.mark.parametrize("score", [0.0, 0.1, 50.0, 80.0, 99.9, 100.0])
@pytest.mark.parametrize("rating", [0.0, 2.0, 5.0, 7.5, 10.0])
@pytest.mark.parametrize("weight", [0.2, 0.45])
def test_vision_blend_stays_within_0_100(score, rating, weight):
    assert 0.0 <= cutpoint.analyze.vision_blend(score, rating, weight) <= 100.0


@pytest.mark.parametrize(
    ("rating", "want"),
    [(0.0, None), (2.0, None), (6.9, None), (7.0, "big-jump"), (10.0, "big-jump")],
)
def test_vision_label_only_from_a_good_rating(rating, want):
    assert cutpoint.analyze.vision_label(rating, "Big jump") == want
    assert cutpoint.analyze.vision_label(10.0, "") is None


class _ByStartRater:
    """Rates each candidate by its start time; records nothing."""

    def __init__(self, ratings: dict[float, tuple[float, str]]) -> None:
        self.ratings = ratings

    def rate(self, _src, start, _end, _style):
        return vision.VisionResult(*self.ratings[start])

    def unload(self) -> None:
        pass


def test_rate_vision_low_rating_no_longer_outranks_an_unrated_candidate():
    """#284: the rating (0-10) is a multiplier on the score, not a 0-100 term mixed into a
    ~15-27 style score: 2/10 lowers it and keeps the clip's label, unrated scores stay put."""
    style = get_style("ski")
    cands = [
        {"source": "a.mp4", "start": 0.0, "end": 5.0, "score": 22.0, "label": "ski-moment",
         "signals": {}},
        {"source": "a.mp4", "start": 10.0, "end": 15.0, "score": 20.0, "label": "ski-moment",
         "signals": {}},
        {"source": "a.mp4", "start": 20.0, "end": 25.0, "score": 26.0, "label": "fast-run",
         "signals": {}},
    ]  # fmt: skip
    rater = _ByStartRater({0.0: (2.0, "boring run no action"), 10.0: (10.0, "big jump")})
    used, failed = cutpoint.analyze._rate_vision(cands, rater, 2, style)  # type: ignore[arg-type]
    assert (used, failed) == (True, 0)
    low, high, unrated = cands
    assert low["score"] == 19.4 and low["label"] == "ski-moment"  # ski: vision_weight 0.2
    assert low["signals"]["vision"] == 2.0
    assert high["score"] == 24.0 and high["label"] == "big-jump"
    assert unrated["score"] == 26.0 and unrated["label"] == "fast-run"
    assert unrated["score"] > low["score"]


def test_rate_vision_mid_rating_does_not_lift_a_score_past_an_unrated_one():
    """#284: a middling 5/10 leaves the score as it was (the old 0-100 blend gave 22.0 -> 27.6,
    above an unrated 26.0)."""
    cands = [
        {"source": "a.mp4", "start": 30.0, "end": 35.0, "score": 22.0, "label": "ski-moment",
         "signals": {}},
        {"source": "a.mp4", "start": 40.0, "end": 45.0, "score": 26.0, "label": "fast-run",
         "signals": {}},
    ]  # fmt: skip
    rater = _ByStartRater({30.0: (5.0, "ok run")})
    cutpoint.analyze._rate_vision(cands, rater, 1, get_style("ski"))  # type: ignore[arg-type]
    mid, unrated = cands
    assert mid["score"] == 22.0 < unrated["score"] == 26.0


def test_rate_vision_keeps_the_model_label_apart_from_the_clip_label():
    """The model's label is recorded as ``vision_label`` even for a low rating; ``label``
    only takes it from a good rating."""
    cands = [
        {"source": "a.mp4", "start": 0.0, "end": 5.0, "score": 22.0, "label": "ski-moment",
         "signals": {}},
        {"source": "a.mp4", "start": 10.0, "end": 15.0, "score": 20.0, "label": "ski-moment",
         "signals": {}},
    ]  # fmt: skip
    rater = _ByStartRater({0.0: (2.0, "boring run"), 10.0: (9.0, "big jump")})
    cutpoint.analyze._rate_vision(cands, rater, 2, get_style("ski"))  # type: ignore[arg-type]
    low, high = cands
    assert low["vision_label"] == "boring run" and low["label"] == "ski-moment"
    assert high["vision_label"] == "big jump" and high["label"] == "big-jump"


@pytest.mark.parametrize(
    ("content", "want"),
    [
        ('{"score": NaN, "label": "x"}', None),
        ('{"score": Infinity, "label": "x"}', 10.0),
        ('{"score": -Infinity, "label": "x"}', 0.0),
    ],
)
def test_parse_treats_a_non_finite_score_as_no_rating(content, want):
    res = vision._parse(content)
    assert (res.score if res else None) == want


@pytest.mark.parametrize(("style_name", "want"), [("commentary", 24.0), ("action", 29.0)])
def test_rate_vision_uses_the_style_vision_weight(style_name, want):
    """The style's vision_weight sets the multiplier strength: 10/10 on 20.0 gives
    20 * (1 + vision_weight)."""
    style = get_style(style_name)
    cand = {"source": "a.mp4", "start": 0.0, "end": 5.0, "score": 20.0, "label": "x",
            "signals": {}}  # fmt: skip
    cutpoint.analyze._rate_vision([cand], _ByStartRater({0.0: (10.0, "")}), 1, style)  # type: ignore[arg-type]
    assert cand["score"] == want == round(20.0 * (1 + style.vision_weight), 1)


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
