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
from cutpoint import cache, cli, vision
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
    monkeypatch.setattr(vision, "grab_frame_jpeg", lambda _src, _t, **_k: b"jpeg")
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

    hits = asked = 0

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
    hits = asked = missed = 0

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


# ---- #377: ratings cached per source and rating input


@pytest.fixture
def src(home, tmp_path) -> Path:
    """A source file to key the cache by (frames are stubbed); the cache under ``home``."""
    path = tmp_path / "videos" / "a.mp4"
    path.parent.mkdir()
    path.write_bytes(b"not really a video")
    return path


def _chats(posts: list[tuple[str, dict]]) -> int:
    return sum(u.endswith("/api/chat") for u, _ in posts)


def test_rerun_asks_the_model_nothing_and_gives_the_same_ratings(posts, src, caplog):
    """A second run (a new rater, as a new `analyze` would make) takes every rating from the
    cache: no request at all, not even the unload, and the same scores and labels."""

    def cands() -> list[dict]:
        return [
            {"source": str(src), "start": float(s), "end": s + 5.0, "score": 20.0 + s,
             "label": "x", "signals": {}}
            for s in (0, 10, 20)
        ]  # fmt: skip

    style = get_style("gaming")
    first = cands()
    rater = vision.VisionRater(url="http://x:1", model="m")
    assert cutpoint.analyze._rate_vision(first, rater, 3, style) == (True, 0)
    assert (rater.hits, rater.asked) == (0, 3) and _chats(posts) == 6
    posts.clear()
    again = cands()
    rater = vision.VisionRater(url="http://x:1", model="m")
    with caplog.at_level(logging.INFO, logger="cutpoint.analyze"):
        assert cutpoint.analyze._rate_vision(again, rater, 3, style) == (True, 0)
    assert posts == []
    assert again == first
    assert (rater.hits, rater.asked) == (3, 0)
    assert "Vision: 3 ratings from the cache, 0 asked the model" in caplog.text


def test_the_ollama_url_is_not_part_of_the_key(posts, src):
    vision.VisionRater(url="http://x:1", model="m").rate(src, 0, 4, "ski")
    posts.clear()
    res = vision.VisionRater(url="http://other:2", model="m").rate(src, 0, 4, "ski")
    assert res == vision.VisionResult(7.0, "big jump") and posts == []


@pytest.mark.parametrize(
    "change",
    ["model", "version", "prompt", "temperature", "times", "frames", "style"],
)
def test_a_changed_rating_input_misses(posts, src, monkeypatch, change):
    vision.VisionRater(url="http://x:1", model="m").rate(src, 0, 4, "gaming")
    posts.clear()
    model, start, n_frames, style = "m", 0.0, 3, "gaming"
    if change == "model":
        model = "m2"
    elif change == "version":
        monkeypatch.setattr(vision, "RATING_VERSION", vision.RATING_VERSION + 1)
    elif change == "prompt":
        monkeypatch.setattr(
            vision, "_DEFAULT_PROMPTS", (vision._PROMPT + " ", vision._RETRY_PROMPT)
        )
    elif change == "temperature":
        monkeypatch.setattr(vision, "_TEMPERATURES", (0.2, 0.5))
    elif change == "times":
        start = 0.5
    elif change == "frames":
        n_frames = 4
    else:
        style = "action"
    rater = vision.VisionRater(url="http://x:1", model=model)
    assert rater.rate(src, start, 4, style, n_frames=n_frames) is not None
    assert _chats(posts) == 2 and (rater.hits, rater.asked) == (0, 1)


def test_a_failed_cache_write_keeps_the_rating(posts, src, monkeypatch, caplog):
    def locked(*_a, **_k):
        raise PermissionError("locked")

    monkeypatch.setattr(vision.cache, "save_json", locked)
    res = vision.VisionRater(url="http://x:1", model="m").rate(src, 0, 4, "ski")
    assert res == vision.VisionResult(7.0, "big jump")
    assert "Could not save the vision rating cache" in caplog.text


def test_a_failed_request_is_not_cached(posts, src, monkeypatch):
    """Ollama down mid-run: no rating, nothing cached; the next run asks again."""
    working = vision.requests.post

    def down(*_a, **_k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(vision.requests, "post", down)
    rater = vision.VisionRater(url="http://x:1", model="m")
    assert rater.rate(src, 0, 4, "ski") is None
    assert (rater.hits, rater.asked) == (0, 1)
    assert not (cache.source_dir(src) / vision.RATINGS_FILE).exists()
    monkeypatch.setattr(vision.requests, "post", working)
    rater = vision.VisionRater(url="http://x:1", model="m")
    assert rater.rate(src, 0, 4, "ski") == vision.VisionResult(7.0, "big jump")
    assert (rater.hits, rater.asked) == (0, 1) and _chats(posts) == 2


def test_an_unparseable_reply_is_not_cached(monkeypatch, src):
    monkeypatch.setattr(vision, "grab_frame_jpeg", lambda _src, _t, **_k: b"jpeg")
    monkeypatch.setattr(
        vision.requests, "post", lambda *_a, **_k: _Resp({"message": {"content": "no json"}})
    )
    assert vision.VisionRater(url="http://x:1", model="m").rate(src, 0, 4, "ski") is None
    assert not (cache.source_dir(src) / vision.RATINGS_FILE).exists()


@pytest.mark.parametrize(
    "text",
    [
        "{not json",
        "[1, 2]",
        "\xff\xfe",
        None,  # a valid file whose entry is malformed
    ],
)
def test_a_corrupt_cache_file_is_a_miss(posts, src, text):
    path = cache.source_dir(src) / vision.RATINGS_FILE
    if text is None:
        key = vision.rating_key("m", "ski", [1.0, 2.0, 3.0])
        path.write_text(json.dumps({key: {"score": "high", "label": "x"}}), "utf-8")
    else:
        path.write_bytes(text.encode("latin-1"))
    rater = vision.VisionRater(url="http://x:1", model="m")
    assert rater.rate(src, 0, 4, "ski") == vision.VisionResult(7.0, "big jump")
    assert (rater.hits, rater.asked) == (0, 1)
    posts.clear()
    rater = vision.VisionRater(url="http://x:1", model="m")
    assert rater.rate(src, 0, 4, "ski") == vision.VisionResult(7.0, "big jump")  # rewritten
    assert posts == [] and rater.hits == 1


def test_a_missing_source_still_reports_the_grab_failure(
    posts, home, tmp_path, monkeypatch, caplog
):
    """No cache key for a source that is not there: the frame grab reports it, as before."""

    def grab(_src, _t, **_k):
        raise vision.FFmpegError("no such file")

    monkeypatch.setattr(vision, "grab_frame_jpeg", grab)
    rater = vision.VisionRater(url="http://x:1", model="m")
    assert rater.rate(tmp_path / "gone.mp4", 0, 4, "ski") is None
    assert "Vision frame grab failed" in caplog.text and posts == []
    assert (rater.hits, rater.asked) == (0, 0)


def test_the_attempts_keep_their_temperatures(posts):
    """Moving the temperatures into ``_TEMPERATURES`` keeps the request payloads: 0.2 for the
    first attempt, 0.4 for the retry."""
    vision.VisionRater(url="http://x:1", model="m").rate(Path("a.mp4"), 0, 4, "gaming")
    temps = [p["options"]["temperature"] for u, p in posts if u.endswith("/api/chat")]
    assert temps == [0.2, 0.4]


# ---- #413: the model digest in the cache; cached ratings when Ollama is down


@pytest.fixture
def ollama(posts, monkeypatch) -> dict:
    """A fake ``/api/tags``: model ``m`` with ``state["digest"]``; ``state["up"] = False``
    refuses the connection (and any POST then fails the test)."""
    state: dict = {"up": True, "digest": "sha256:aaa"}
    working = vision.requests.post

    def get(url, timeout):
        if not state["up"]:
            raise requests.ConnectionError("refused")
        assert url.endswith("/api/tags")
        return _Resp({"models": [{"name": "m:latest", "digest": state["digest"]}]})

    def post(*a, **k):
        assert state["up"], "a request went to Ollama while it was down"
        return working(*a, **k)

    monkeypatch.setattr(vision.requests, "get", get)
    monkeypatch.setattr(vision.requests, "post", post)
    return state


def _rater(ollama: dict) -> vision.VisionRater:
    """A new rater (as a new `analyze` makes) after ``available()``; cache-only when down,
    as analyze sets it."""
    rater = vision.VisionRater(url="http://x:1", model="m")
    rater.cache_only = not rater.available()
    assert rater.cache_only is not ollama["up"]
    return rater


def _entries(src: Path) -> list[dict]:
    return list(cache.load_json(cache.source_dir(src) / vision.RATINGS_FILE).values())


def test_an_unchanged_digest_reuses_the_rating(posts, src, ollama):
    assert _rater(ollama).rate(src, 0, 4, "ski") == vision.VisionResult(7.0, "big jump")
    assert _entries(src) == [{"score": 7.0, "label": "big jump", "digest": "sha256:aaa"}]
    posts.clear()
    rater = _rater(ollama)
    assert rater.rate(src, 0, 4, "ski") == vision.VisionResult(7.0, "big jump")
    assert posts == [] and (rater.hits, rater.asked) == (1, 0)


def test_a_changed_digest_rates_again_and_replaces_the_entry(posts, src, ollama):
    _rater(ollama).rate(src, 0, 4, "ski")
    posts.clear()
    ollama["digest"] = "sha256:bbb"  # the model pulled again under the same tag
    rater = _rater(ollama)
    assert rater.rate(src, 0, 4, "ski") == vision.VisionResult(7.0, "big jump")
    assert _chats(posts) == 2 and (rater.hits, rater.asked) == (0, 1)
    assert [e["digest"] for e in _entries(src)] == ["sha256:bbb"]


def test_a_cached_rating_is_used_when_ollama_is_down(posts, src, ollama):
    _rater(ollama).rate(src, 0, 4, "ski")
    posts.clear()
    ollama["up"] = False
    rater = _rater(ollama)
    assert rater.cached(src, 0, 4, "ski") == vision.VisionResult(7.0, "big jump")
    assert rater.rate(src, 0, 4, "ski") == vision.VisionResult(7.0, "big jump")
    assert (rater.hits, rater.asked, rater.missed) == (1, 0, 0)
    rater.unload()  # nothing was asked: nothing to unload
    assert posts == []


def test_a_miss_when_ollama_is_down_is_no_rating(posts, src, ollama, monkeypatch):
    _rater(ollama).rate(src, 0, 4, "ski")
    posts.clear()
    ollama["up"] = False

    def grab(*_a, **_k):
        raise AssertionError("no frame grab for a miss when Ollama is down")

    monkeypatch.setattr(vision, "grab_frame_jpeg", grab)
    rater = _rater(ollama)
    assert rater.cached(src, 10, 14, "ski") is None
    assert rater.rate(src, 10, 14, "ski") is None
    assert rater.rate(src, 0, 4, "ski") is not None
    assert posts == [] and (rater.hits, rater.asked, rater.missed) == (1, 0, 1)


def test_an_entry_without_a_digest_is_used_only_when_ollama_is_down(posts, src, ollama):
    """A rating cached before #413 has no digest, so no known model build: it is rated
    again when Ollama answers, and still used when Ollama is down (no digest to compare)."""
    key = vision.rating_key("m", "ski", vision._frame_times(0, 4, 3))
    path = cache.source_dir(src) / vision.RATINGS_FILE
    path.write_text(json.dumps({key: {"score": 3.0, "label": "old"}}), "utf-8")
    ollama["up"] = False
    assert _rater(ollama).rate(src, 0, 4, "ski") == vision.VisionResult(3.0, "old")
    ollama["up"] = True
    rater = _rater(ollama)
    assert rater.rate(src, 0, 4, "ski") == vision.VisionResult(7.0, "big jump")
    assert (rater.hits, rater.asked) == (0, 1)


def test_no_digest_is_stored_when_ollama_lists_none(posts, src, monkeypatch):
    monkeypatch.setattr(
        vision.requests, "get", lambda *_a, **_k: _Resp({"models": [{"name": "m"}]})
    )
    rater = vision.VisionRater(url="http://x:1", model="m")
    assert rater.available() and rater.digest is None
    rater.rate(src, 0, 4, "ski")
    assert _entries(src) == [{"score": 7.0, "label": "big jump"}]
