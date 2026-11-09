"""Region discovery with the vision model's labels: the feed score floor, splitting a group
of text of two styles, label -> region mapping, and the model-less fallback (synthetic OCR
frames and a fake model; no network)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from cutpoint.events import discover, feed_labels
from cutpoint.events.discover import (
    DiscoverParams,
    apply_labels,
    cluster_regions,
    proposals_to_regions,
)
from cutpoint.events.regions import Region

NAMES = ["Alder", "Boxer", "Kip", "Nova", "Quill", "Orrin", "Sable", "Tarn", "Vex", "Wren"]


def _box(text: str, x: int, y: int, w: int, h: int) -> dict[str, Any]:
    return {"x0": x, "y0": y, "x1": x + w, "y1": y + h, "text": text, "score": 0.9}


def _frame(b: int, k: int, boxes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": f"rec.mp4@{b * 60 + k * 1.5:.1f}",
        "source": "rec.mp4",
        "t": b * 60 + k * 1.5,
        "burst": b,
        "k": k,
        "frame": f"b{b:03d}_{k}.jpg",
        "boxes": boxes,
    }


def _two_style_frame(b: int, k: int, chat_h: int = 36) -> dict[str, Any]:
    """Six chat rows (36 px text) over a four-row log (24 px text) in one column at the
    right edge; every row's text changes in every frame."""
    n = b * 4 + k
    boxes = [_box("HEALTH", 20, 1040, 100, 24)]
    for r in range(6):
        m = n + r
        text = f"{NAMES[m % 10]}: {NAMES[m // 3 % 10]} {NAMES[m // 7 % 10]}"
        boxes.append(_box(text, 1500, 100 + r * 44, 380, chat_h))
    for r in range(4):
        m = n * 3 + r
        text = f"{NAMES[m % 10]} downed {NAMES[m // 10 % 10]}"
        boxes.append(_box(text, 1500, 380 + r * 30, 380, 24))
    return _frame(b, k, boxes)


FRAMES = [_two_style_frame(b, k) for b in range(40) for k in range(4)]


def _prop(name: str, kind: str, score: float, x: int = 1000, y: int = 100) -> dict[str, Any]:
    return {"name": name, "kind": kind, "score": score, "x": x, "y": y, "w": 300, "h": 100}


def test_a_feed_scoring_next_to_nothing_is_no_region() -> None:
    """The only feed proposal was a loadout box scoring 0.004, kept because it was the best
    feed; the real feed was a notice."""
    proposals = [_prop("feed", "feed", 0.004), _prop("notice", "notice", 77.0, y=400)]
    assert [r.name for r in proposals_to_regions(proposals)] == ["notice"]
    proposals = [_prop("feed", "feed", 0.5), _prop("feed2", "feed", 0.01, y=300)]
    assert [r.name for r in proposals_to_regions(proposals)] == ["feed"]


def test_text_of_two_styles_in_one_column_makes_two_feeds() -> None:
    """Chat above a battle log merged into one region, which the model then called chat."""
    proposals, _ = cluster_regions(FRAMES, DiscoverParams())
    feeds = sorted((p for p in proposals if p["kind"] == "feed"), key=lambda p: p["y"])
    assert len(feeds) == 2
    top, bottom = feeds
    assert top["y"] + top["h"] <= bottom["y"] + 24  # side by side, not stacked on each other
    assert top["y"] < 200 and bottom["y"] >= 360


def test_one_style_stays_one_feed() -> None:
    frames = [_two_style_frame(b, k, chat_h=24) for b in range(40) for k in range(4)]
    proposals, _ = cluster_regions(frames, DiscoverParams())
    assert len([p for p in proposals if p["kind"] == "feed"]) == 1


def test_labels_make_the_regions() -> None:
    proposals = [
        _prop("feed", "feed", 0.5),
        _prop("feed2", "feed", 0.4, y=300),
        _prop("notice", "notice", 1.0, y=700),
        _prop("extra_notice", "notice", 77.0, y=900),
    ]
    regions = [
        Region(p["name"], p["x"], p["y"], p["w"], p["h"], p["kind"], gate="det")
        for p in proposals[:3]
    ]
    extra = [{"name": "extra_notice", "kind": "notice", "origin": "outside"}]
    labels = {
        "feed": {"label": "menu", "model": "m"},
        "feed2": {"label": "chat", "model": "m"},
        "extra_notice": {"label": "kill_feed", "model": "m"},
    }
    out, extra_out = apply_labels(proposals, labels, regions, extra)
    by = {r.name: r for r in out}
    assert out[0].name == "extra_notice" and out[0].kind == "feed" and out[0].every == 1
    assert by["feed"].kind == "notice"  # a feed the model calls a menu is read as text
    assert by["feed2"].kind == "feed"  # chat stays: reading it tells chat from a feed
    assert by["notice"].kind == "notice"
    assert extra_out == []
    # a second kill_feed proposal on the same rectangle is not read twice
    labels["feed3"] = {"label": "kill_feed", "model": "m"}
    proposals.append(_prop("feed3", "feed", 0.2, y=910))
    out, _ = apply_labels(proposals, labels, regions, extra)
    assert [r.name for r in out if r.kind == "feed" and r.name != "feed2"] == ["extra_notice"]


def test_a_large_separate_feed_is_kept_beside_two_small_ones() -> None:
    """The overlap test compares the pair, not the smallest region of all."""
    big = {"name": "big", "kind": "feed", "score": 0.5, "x": 0, "y": 0, "w": 600, "h": 400}
    tiny = {"name": "tiny", "kind": "feed", "score": 0.5, "x": 700, "y": 700, "w": 40, "h": 20}
    tiny2 = {"name": "tiny2", "kind": "feed", "score": 0.5, "x": 700, "y": 700, "w": 40, "h": 20}
    other = {"name": "other", "kind": "feed", "score": 0.5, "x": 560, "y": 0, "w": 300, "h": 100}
    proposals = [tiny, tiny2, other, big]
    labels = {p["name"]: {"label": "kill_feed", "model": "m"} for p in proposals}
    out, _ = apply_labels(proposals, labels, [], [])
    # tiny2 repeats tiny; big overlaps other slightly (far under 0.6 of the smaller): kept
    assert [r.name for r in out] == ["tiny", "other", "big"]


def test_an_outside_notice_labelled_kill_feed_keeps_its_rectangle() -> None:
    p = {
        "name": "extra_notice",
        "kind": "notice",
        "score": 77.0,
        "x": 100,
        "y": 200,
        "w": 300,
        "h": 80,
        "evidence": {"outside": True},
    }
    labels = {"extra_notice": {"label": "kill_feed", "model": "m"}}
    extra = [{"name": "extra_notice", "kind": "notice", "origin": "outside"}]
    out, extra_out = apply_labels([p], labels, [], extra)
    assert [(r.x, r.y, r.w, r.h, r.kind, r.every) for r in out] == [(100, 200, 300, 80, "feed", 1)]
    assert extra_out == []


def _write_frames(folder: Path) -> None:
    folder.mkdir(parents=True)
    img = np.full((720, 1280, 3), 40, np.uint8)
    for b in range(40):
        for k in range(4):
            cv2.imwrite(str(folder / f"b{b:03d}_{k}.jpg"), img)


def test_labels_are_asked_once_and_cached_per_model(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames")
    asked: list[tuple[str, int]] = []

    def ask(prompt: str, jpegs: list[bytes], schema: dict[str, Any]) -> dict[str, Any] | None:
        asked.append((prompt, len(jpegs)))
        return {"text_in_box": "x", "label": "kill_feed"}

    proposals = [
        _prop("feed", "feed", 0.5, x=1500, y=100),
        _prop("banner", "banner", 0.5),  # sure enough: not asked
        _prop("readout", "readout", 0.5, y=500),
    ]

    def run(model: str) -> dict[str, dict[str, str]]:
        return feed_labels.label_proposals(proposals, FRAMES, tmp_path / "frames", ask, model)

    first = run("ollama@h:m")
    assert list(first) == ["feed"] and first["feed"]["label"] == "kill_feed"
    assert len(asked) == 1 and asked[0][1] == feed_labels.CROPS
    assert "top-right of the screen, 300x100 px of 1920x1080" in asked[0][0]
    assert run("ollama@h:m") == first and len(asked) == 1  # cached
    run("ollama@h:other")
    assert len(asked) == 2  # another model: asked again


def test_a_label_outside_the_list_is_no_label(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames")
    calls = []

    def ask(*_a: Any) -> dict[str, Any] | None:
        calls.append(1)
        return {"text_in_box": "x", "label": "scoreboard"}

    proposals = [
        _prop("feed", "feed", 0.5, x=1500, y=100),
        _prop("feed2", "feed", 0.5, x=1500, y=400),
    ]
    out = feed_labels.label_proposals(proposals, FRAMES, tmp_path / "frames", ask, "m")
    assert out == {} and len(calls) == feed_labels.STOP_AFTER


def test_failed_calls_in_a_row_stop_the_labelling(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames")
    asked: list[int] = []

    def ask(*_a: Any) -> dict[str, Any] | None:
        asked.append(1)
        return None

    proposals = [_prop(f"feed{i}", "feed", 0.5, x=1500, y=100 + 100 * i) for i in range(5)]
    out = feed_labels.label_proposals(proposals, FRAMES, tmp_path / "frames", ask, "m")
    assert out == {} and len(asked) == feed_labels.STOP_AFTER


def _rater(**kw: Any) -> SimpleNamespace:
    warned: list[bool] = []
    base = {
        "consent_needed": None,
        "available": lambda: True,
        "warn_consent": lambda: warned.append(True),
        "provider_name": "ollama",
        "host": "h",
        "url": "http://localhost:11434",
        "model": "m",
        "unload": lambda: None,
        "describe": lambda *a, **k: {"text_in_box": "", "label": "chat"},
        "load": lambda _timeout: True,
        "on_gpu": lambda: True,
        "used": False,
        "timed_out": False,
        "warned": warned,
        "loads": [],
    }
    rater = SimpleNamespace(**(base | kw))
    inner, inner_load = rater.describe, rater.load

    def describe(*a: Any, **k: Any) -> Any:
        rater.used = True  # as VisionRater: set when a request goes out, before it can fail
        return inner(*a, **k)

    def load(timeout: float) -> bool:
        rater.used = True
        rater.loads.append(timeout)
        return inner_load(timeout)

    rater.describe, rater.load = describe, load
    return rater


def test_no_consent_asks_nothing_and_warns(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames")
    sent = []
    rater = _rater(consent_needed="vision-v2:host", describe=lambda *a, **k: sent.append(1))
    proposals = [_prop("feed", "feed", 0.5, x=1500, y=100)]
    assert feed_labels.make_labeler(rater)(proposals, FRAMES, tmp_path / "frames") == {}
    assert rater.warned == [True] and not sent


def test_no_model_gives_no_labels(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames")
    rater = _rater(available=lambda: False)
    proposals = [_prop("feed", "feed", 0.5, x=1500, y=100)]
    assert feed_labels.make_labeler(rater)(proposals, FRAMES, tmp_path / "frames") == {}
    assert not rater.warned


def test_low_memory_gives_no_labels_and_frees_the_model(tmp_path: Path) -> None:
    from cutpoint.errors import LowMemoryError  # noqa: PLC0415

    _write_frames(tmp_path / "frames")
    unloaded: list[bool] = []

    def describe(*_a: Any, **_k: Any) -> None:
        raise LowMemoryError("Not enough memory for the model", "1 GB free")

    rater = _rater(describe=describe, unload=lambda: unloaded.append(True))
    proposals = [_prop("feed", "feed", 0.5, x=1500, y=100)]
    assert feed_labels.make_labeler(rater)(proposals, FRAMES, tmp_path / "frames") == {}
    assert unloaded == [True]


def _discover(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kw: Any) -> dict[str, Any]:
    monkeypatch.setattr(discover, "_ocr_frames", lambda *_a, **_k: FRAMES)
    monkeypatch.setattr(discover, "probe", lambda _p: None)
    return discover.discover_regions([], tmp_path / "disc", "zonewar", **kw)


def test_without_labels_the_rules_decide_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = _discover(tmp_path / "a", monkeypatch)
    none = _discover(tmp_path / "b", monkeypatch, labeler=lambda *_a: {})
    assert plain["regions"] == none["regions"]
    assert "labels" not in none["discovery"]


def test_labelled_discovery_saves_the_labels_in_the_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def labeler(proposals: list[dict[str, Any]], frames: list[Any], folder: Path) -> dict[str, Any]:
        top = min((p for p in proposals if p["kind"] == "feed"), key=lambda p: p["y"])
        return {top["name"]: {"label": "chat", "model": "ollama@h:m"}}

    preset = _discover(tmp_path, monkeypatch, labeler=labeler)
    assert preset["source"] == "discovered"
    assert len(preset["discovery"]["labels"]) == 1
    assert [r["label"] for r in preset["regions"] if "label" in r] == ["chat"]


def test_a_named_feed_reads_icon_lines_as_kills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def labeler(proposals: list[dict[str, Any]], frames: list[Any], folder: Path) -> dict[str, Any]:
        top = min((p for p in proposals if p["kind"] == "feed"), key=lambda p: p["y"])
        return {top["name"]: {"label": "kill_feed", "model": "ollama@h:m"}}

    preset = _discover(tmp_path, monkeypatch, labeler=labeler)
    assert preset["feed"] == {"row_notices": True}


def test_the_cache_key_follows_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = FRAMES[:4]
    p = _prop("feed", "feed", 0.5)
    before = feed_labels._key("m", p, frames)
    assert feed_labels._key("m", p, frames) == before
    monkeypatch.setattr(feed_labels, "_PROMPT", feed_labels._PROMPT + " Be brief.")
    assert feed_labels._key("m", p, frames) != before


def test_the_cache_key_follows_the_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = FRAMES[:4]
    p = _prop("feed", "feed", 0.5)
    before = feed_labels._key("m", p, frames)
    monkeypatch.setattr(feed_labels, "SCHEMA", feed_labels.SCHEMA | {"title": "other"})
    assert feed_labels._key("m", p, frames) != before


def test_an_outside_notice_the_model_labelled_carries_its_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = {
        "name": "outside1",
        "kind": "notice",
        "score": 1.0,
        "x": 20,
        "y": 600,
        "w": 400,
        "h": 120,
        "evidence": {"median_text_h": 20, "outside": True, "bursts": 5, "frames": 9},
    }
    monkeypatch.setattr(discover, "notice_proposals", lambda *_a, **_k: [outside])

    def labeler(proposals: list[dict[str, Any]], frames: list[Any], folder: Path) -> dict[str, Any]:
        return {"outside1": {"label": "chat", "model": "ollama@h:m"}}

    preset = _discover(tmp_path, monkeypatch, labeler=labeler)
    got = [r for r in preset["regions"] if r["name"] == "outside1"]
    assert [r["label"] for r in got] == ["chat"] and got[0]["origin"] == "outside"
    # the loader ignores the extra key
    from cutpoint.events import regions as regions_mod  # noqa: PLC0415

    loaded = {r.name: r for r in regions_mod._regions(preset)}
    assert loaded["outside1"].kind == "notice"
    unlabelled = _discover(tmp_path / "plain", monkeypatch, labeler=lambda *_a: {})
    assert all("label" not in r for r in unlabelled["regions"])


def test_a_call_that_times_out_stops_the_labelling(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames")
    seen: list[Any] = []

    def describe(*_a: Any, **kw: Any) -> None:
        seen.append(kw.get("timeout"))
        rater.timed_out = True  # as VisionRater.describe when the request times out
        return None

    rater = _rater(describe=describe)
    proposals = [_prop(f"feed{i}", "feed", 0.5, x=1500, y=100 + 100 * i) for i in range(4)]
    assert feed_labels.make_labeler(rater)(proposals, FRAMES, tmp_path / "frames") == {}
    # the load gets the long timeout, the call the short one, and the next would time out too
    assert rater.loads == [feed_labels.FIRST_ASK_TIMEOUT]
    assert seen == [feed_labels.ASK_TIMEOUT]
    assert feed_labels.FIRST_ASK_TIMEOUT > feed_labels.ASK_TIMEOUT


def test_a_model_on_the_cpu_only_is_not_asked(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames")
    sent: list[int] = []
    unloaded: list[bool] = []
    rater = _rater(
        on_gpu=lambda: False,
        describe=lambda *a, **k: sent.append(1),
        unload=lambda: unloaded.append(True),
    )
    proposals = [_prop("feed", "feed", 0.5, x=1500, y=100)]
    assert feed_labels.make_labeler(rater)(proposals, FRAMES, tmp_path / "frames") == {}
    assert not sent and unloaded == [True]  # loaded to find out, then freed


def test_a_refused_load_falls_back_to_a_long_first_call(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames")
    seen: list[Any] = []
    checked: list[bool] = []

    def describe(*_a: Any, **kw: Any) -> dict[str, str]:
        seen.append(kw.get("timeout"))
        return {"text_in_box": "", "label": "chat"}

    # the load request failed without a timeout (a proxy or an old Ollama refused it)
    rater = _rater(
        load=lambda _t: False, describe=describe, on_gpu=lambda: checked.append(True) or False
    )
    proposals = [_prop(f"feed{i}", "feed", 0.5, x=1500, y=100 + 100 * i) for i in range(2)]
    got = feed_labels.make_labeler(rater)(proposals, FRAMES, tmp_path / "frames")
    # GPU status unknown: asked as before, the first call loads the model
    assert len(got) == 2 and not checked
    assert seen == [feed_labels.FIRST_ASK_TIMEOUT, feed_labels.ASK_TIMEOUT]


def test_a_load_that_times_out_stops_the_labelling(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames")
    sent: list[int] = []

    def load(_t: float) -> bool:
        rater.timed_out = True
        return False

    rater = _rater(load=load, describe=lambda *a, **k: sent.append(1))
    proposals = [_prop("feed", "feed", 0.5, x=1500, y=100)]
    assert feed_labels.make_labeler(rater)(proposals, FRAMES, tmp_path / "frames") == {}
    assert not sent


def test_an_unknown_gpu_status_still_asks(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames")
    rater = _rater(on_gpu=lambda: None)
    proposals = [_prop("feed", "feed", 0.5, x=1500, y=100)]
    got = feed_labels.make_labeler(rater)(proposals, FRAMES, tmp_path / "frames")
    assert got["feed"]["label"] == "chat"


def test_the_budget_ends_the_labelling_and_keeps_the_labels_so_far(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_frames(tmp_path / "frames")
    clock = [0.0]
    monkeypatch.setattr(feed_labels.time, "monotonic", lambda: clock[0])
    seen: list[Any] = []

    def describe(*_a: Any, **kw: Any) -> dict[str, str]:
        seen.append(kw.get("timeout"))
        clock[0] += 50  # each call takes 50 s
        return {"text_in_box": "", "label": "chat"}

    rater = _rater(describe=describe)
    proposals = [_prop(f"feed{i}", "feed", 0.5, x=1500, y=100 + 100 * i) for i in range(4)]
    got = feed_labels.make_labeler(rater)(proposals, FRAMES, tmp_path / "frames")
    # 120 s: two calls of 50 s, the third gets the 20 s left, then fewer than MIN_ASK are left
    assert seen == [60, 60, 20] and len(got) == 3


def test_a_hosted_rater_is_never_given_the_model_load_timeout(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames")
    seen: list[Any] = []

    def describe(*_a: Any, **kw: Any) -> dict[str, str]:
        seen.append(kw.get("timeout"))
        return {"text_in_box": "", "label": "chat"}

    rater = _rater(describe=describe, provider_name="anthropic")
    proposals = [
        _prop("feed", "feed", 0.5, x=1500, y=100),
        _prop("feed2", "feed", 0.5, x=1500, y=400),
    ]
    assert len(feed_labels.make_labeler(rater)(proposals, FRAMES, tmp_path / "frames")) == 2
    assert seen == [feed_labels.ASK_TIMEOUT] * 2


def test_only_the_model_load_gets_the_load_timeout(tmp_path: Path) -> None:
    _write_frames(tmp_path / "frames")
    seen: list[Any] = []

    def describe(*_a: Any, **kw: Any) -> dict[str, str]:
        seen.append(kw.get("timeout"))
        return {"text_in_box": "", "label": "chat"}

    proposals = [
        _prop("feed", "feed", 0.5, x=1500, y=100),
        _prop("feed2", "feed", 0.5, x=1500, y=400),
    ]
    rater = _rater(describe=describe)
    feed_labels.make_labeler(rater)(proposals, FRAMES, tmp_path / "frames")
    # the load has its own request: every call gets the short timeout
    assert rater.loads == [feed_labels.FIRST_ASK_TIMEOUT]
    assert seen == [feed_labels.ASK_TIMEOUT, feed_labels.ASK_TIMEOUT]


def test_each_game_has_its_own_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for game in ("a", "b"):
        _write_frames(tmp_path / game / "frames")
    clock = [0.0]
    monkeypatch.setattr(feed_labels.time, "monotonic", lambda: clock[0])
    seen: list[Any] = []

    def describe(*_a: Any, **kw: Any) -> dict[str, str]:
        seen.append(kw.get("timeout"))
        clock[0] += feed_labels.LABEL_BUDGET  # the first game spends the whole budget
        return {"text_in_box": "", "label": "chat"}

    rater = _rater(describe=describe)
    labeler = feed_labels.make_labeler(rater)
    proposals = [_prop("feed", "feed", 0.5, x=1500, y=100)]
    assert labeler(proposals, FRAMES, tmp_path / "a" / "frames")
    rater.used = False  # freed after the first game, loaded again for the next
    assert labeler(proposals, FRAMES, tmp_path / "b" / "frames")
    assert seen == [feed_labels.ASK_TIMEOUT] * 2


def test_a_model_on_the_cpu_only_is_checked_once_for_all_games(tmp_path: Path) -> None:
    for game in ("a", "b"):
        _write_frames(tmp_path / game / "frames")
    sent: list[int] = []
    checks: list[int] = []

    def on_gpu() -> bool:
        checks.append(1)
        return False

    rater = _rater(on_gpu=on_gpu, describe=lambda *a, **k: sent.append(1))
    labeler = feed_labels.make_labeler(rater)
    proposals = [_prop("feed", "feed", 0.5, x=1500, y=100)]
    assert labeler(proposals, FRAMES, tmp_path / "a" / "frames") == {}
    rater.used = False  # freed after the first game
    assert labeler(proposals, FRAMES, tmp_path / "b" / "frames") == {}
    assert rater.loads == [feed_labels.FIRST_ASK_TIMEOUT] and len(checks) == 1 and not sent
