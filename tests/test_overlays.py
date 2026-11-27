"""The overlay layer of non-gameplay footage (#1069 (c)): the vision model points at a
scoreboard, gauges or a lower-third, OCR confirms and reads them, the events boost moments.

Frames, OCR text and model replies are invented; the vision model, the OCR engine and ffmpeg
are fakes (no GPU, no Ollama, no network)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from cutpoint import jsonout, scorer
from cutpoint.errors import LowMemoryError
from cutpoint.events import eventtypes, hud_map, layers, overlays, pipeline, scoring
from cutpoint.events.common import Source
from cutpoint.events.discover import score_value
from cutpoint.events.feed_labels import VisionCalls
from cutpoint.events.layers import OVERLAY_STYLES, overlay_kinds
from cutpoint.events.score_readout import ScoreTracker
from ocr_helpers import StopAtVlm, quad, stub_ocr_layer

# --------------------------------------------------------------------------- the faked world

SCORES = [("3", "5"), ("4", "5"), ("7", "5"), ("13", "5")]  # a scoreboard's states
CHANGES_AT = (60, 140, 220)  # +1, +3 and +6 on the left number
SCOREBOARD_ANSWER = {
    "elements": [{"kind": "scoreboard", "box": [450, 25, 560, 110], "text_seen": ""}]
}


def _sample_items(words: bool = False) -> list[tuple[list, str, float]]:
    """What the OCR reads on a 1280 x 720 sample frame: two score numbers (or words)."""
    if words:
        return [(quad(600, 27, 120, 47), "WELCOME BACK", 0.95)]
    return [(quad(600, 27, 27, 47), "3", 0.95), (quad(667, 27, 27, 47), "5", 0.95)]


def _rater(**kw: Any) -> SimpleNamespace:
    warned: list[bool] = []
    asked: list[dict[str, Any]] = []

    def describe(prompt: str, jpegs: list[bytes], schema: dict, **k: Any) -> dict | None:
        asked.append({"prompt": prompt, "jpegs": jpegs, "schema": schema, **k})
        return SCOREBOARD_ANSWER

    base: dict[str, Any] = {
        "consent_needed": None,
        "available": lambda: True,
        "warn_consent": lambda: warned.append(True),
        "provider_name": "hosted",
        "host": "h",
        "model": "m",
        "unload": lambda: None,
        "describe": describe,
        "timed_out": False,
        "warned": warned,
        "asked": asked,
    }
    return SimpleNamespace(**(base | kw))


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """One 300 s source ``match.mp4`` whose overlay is a scoreboard changing at
    :data:`CHANGES_AT`; ffmpeg, OCR and the vision model are fakes that count their calls."""
    src = Source(
        path=tmp_path / "videos" / "match.mp4",
        duration=300.0,
        cdir=tmp_path / "cache",
        audio_parts=[],
    )
    w = SimpleNamespace(
        src=src,
        rater=_rater(),
        sample_words=False,
        grabs=0,
        decodes=0,
        engines=0,
        timeline=lambda t: sum(t >= x for x in CHANGES_AT),
    )
    gray = np.full((720, 1280, 3), 90, np.uint8)
    jpeg = cv2.imencode(".jpg", gray)[1].tobytes()

    def grab_images(*a: Any, **k: Any) -> list[bytes]:
        w.grabs += 1
        return [jpeg]

    def decode_raw(path: Any, *, size: tuple[int, int], **k: Any) -> Any:
        w.decodes += 1
        cw, ch = size
        for t in range(300):
            yield float(t), np.full((ch, cw, 3), (w.timeline(t) + 1) * 40, np.uint8)

    def engine(img: np.ndarray, min_score: float) -> list:
        w.engines += 1
        if img.shape[1] == 1280:
            return _sample_items(w.sample_words)
        a, b = SCORES[round(float(img.mean()) / 40) - 1]
        return [(quad(14, 10, 28, 40), a, 0.95), (quad(84, 10, 28, 40), b, 0.95)]

    w.make_calls = 0

    def make_engine() -> Any:
        w.make_calls += 1
        return engine

    w.make_engine = make_engine
    w.calls = lambda: VisionCalls(w.rater)
    monkeypatch.setattr("cutpoint.ffmpeg_utils.grab_images", grab_images)
    monkeypatch.setattr("cutpoint.ffmpeg_utils.decode_raw", decode_raw)
    monkeypatch.setattr("cutpoint.ffmpeg_utils.find_tool", lambda name: name)
    monkeypatch.setattr(
        "cutpoint.ffmpeg_utils.probe", lambda path: SimpleNamespace(width=1920, height=1080)
    )

    def run(kinds: tuple[str, ...] = ("scoreboard",), calls: Any = "default", **kw: Any) -> Any:
        return overlays.run_overlays(
            [src],
            kinds,
            w.calls() if calls == "default" else calls,
            make_engine=w.make_engine,
            **kw,
        )

    w.run = run
    return w


# --------------------------------------------------------------------------- vision points, OCR confirms


def test_a_scoreboard_vision_points_and_ocr_confirms_gives_the_score_changes(world) -> None:
    out, stats = world.run()
    events = out[str(world.src.path)]
    assert [(e["type"], e["before"], e["after"]) for e in events] == [
        ("score", 3, 4),  # +1
        ("score", 4, 7),  # +3
        ("score", 7, 13),  # +6
    ]
    assert [e["t"] for e in events] == [59.0, 139.0, 219.0]
    assert all(e["kind"] == "ocr" and e["overlay"] is True for e in events)
    assert all(e["overlay_kind"] == "scoreboard" for e in events)
    assert stats["events"] == {"score": 3}
    assert [r["kind"] for r in stats["regions"]] == ["scoreboard"]


def test_vision_pointing_without_ocr_confirmation_gives_no_events(world) -> None:
    """The model says there is a scoreboard, OCR reads only words there: it never scores alone."""
    world.sample_words = True
    out, stats = world.run()
    assert out == {} and world.decodes == 0  # nothing is read through the footage
    assert world.rater.asked  # the model was asked
    assert [e["confirmed"] for e in stats["elements"]] == [False]
    assert stats["regions"] == []
    assert overlays.summary(stats)["confirmed"] == []


def test_no_text_on_the_samples_asks_the_model_nothing(world, monkeypatch) -> None:
    monkeypatch.setattr(world, "sample_words", False)
    world.make_engine = lambda: lambda img, s: []
    out, stats = world.run()
    assert out == {} and world.rater.asked == []
    assert stats["samples"] > 0


def test_the_model_answering_nothing_usable_reads_nothing(world) -> None:
    world.rater.describe = lambda *a, **k: {"not": "elements"}
    out, stats = world.run()
    assert out == {} and stats["skipped"] == "no answer" and world.decodes == 0


def test_an_overlay_of_a_kind_that_is_not_wanted_is_dropped(world) -> None:
    """The style wants lower-thirds: the model's scoreboard is no element at all."""
    out, stats = world.run(kinds=("lower_third",))
    assert out == {} and stats["regions"] == [] and stats["elements"] == []
    assert world.decodes == 0


# --------------------------------------------------------------------------- no model, no consent


def test_without_a_vision_client_nothing_is_asked_and_nothing_breaks(world) -> None:
    out, stats = world.run(calls=None)
    assert out == {} and stats["skipped"] == "no vision client"
    assert world.grabs == 0 and world.make_calls == 0


def test_a_hosted_model_without_consent_is_not_asked(world, caplog) -> None:
    world.rater.consent_needed = "vision-v2:h"
    with caplog.at_level(logging.INFO, logger="cutpoint.events.overlays"):
        out, stats = world.run()
    assert out == {} and stats["skipped"] == "no vision consent"
    assert world.rater.asked == [] and world.grabs == 0 and world.decodes == 0
    assert "Vision consent" in caplog.text


def test_a_model_that_does_not_answer_is_not_asked(world) -> None:
    world.rater.available = lambda: False
    out, stats = world.run()
    assert out == {} and stats["skipped"] == "no vision model answers"
    assert world.rater.asked == [] and world.grabs == 0


def test_no_overlay_kinds_do_nothing(world) -> None:
    assert world.run(kinds=()) == ({}, {})
    assert world.grabs == 0 and world.rater.asked == []


def test_the_model_sees_frames_only_never_a_path_or_a_name(world) -> None:
    world.run(context="a table-tennis match")
    assert world.rater.asked
    for call in world.rater.asked:
        flat = json.dumps({k: v for k, v in call.items() if k != "jpegs"}, default=str)
        assert "match.mp4" not in flat and str(world.src.path.parent) not in flat
        assert "videos" not in flat
        assert all(isinstance(j, bytes) for j in call["jpegs"])
    assert "a table-tennis match" in world.rater.asked[0]["prompt"]  # the user's hint, delimited


def test_the_prompt_names_no_game_sport_or_activity() -> None:
    text = overlays.prompt(("scoreboard", "gauge", "lower_third")).lower()
    for word in ("soccer", "football", "basketball", "tennis", "gameplay", "skiing", "cycling"):
        assert word not in text


# --------------------------------------------------------------------------- caches


def test_a_second_run_is_served_from_the_caches(world) -> None:
    first, _ = world.run()
    work = (world.grabs, world.decodes, world.engines, len(world.rater.asked), world.make_calls)
    assert work[0] > 0 and work[1] == 1 and work[3] == overlays.MAP_FRAMES
    second, _ = world.run()
    assert second == first
    assert (
        world.grabs,
        world.decodes,
        world.engines,
        len(world.rater.asked),
        world.make_calls,
    ) == work


def test_force_ignores_the_caches(world) -> None:
    world.run()
    decodes, asked = world.decodes, len(world.rater.asked)
    world.run(force=True)
    assert world.decodes == decodes + 1 and len(world.rater.asked) == asked + overlays.MAP_FRAMES


@pytest.mark.parametrize("change", ["model", "context"])
def test_a_changed_model_or_context_asks_again_but_reads_the_samples_from_the_cache(
    world, change
) -> None:
    world.run(context="match")
    grabs, asked = world.grabs, len(world.rater.asked)
    if change == "model":
        world.rater.model = "m2"
        world.run(context="match")
    else:
        world.run(context="another description")
    assert len(world.rater.asked) == asked + overlays.MAP_FRAMES  # the answers: new key
    assert world.grabs == grabs  # the sample frames: same key


def test_a_new_overlay_version_misses_every_cache(world, monkeypatch) -> None:
    world.run()
    grabs, decodes, asked = world.grabs, world.decodes, len(world.rater.asked)
    monkeypatch.setattr(overlays, "OVERLAY_VERSION", overlays.OVERLAY_VERSION + 1)
    world.run()
    assert world.grabs > grabs and world.decodes == decodes + 1
    assert len(world.rater.asked) == asked + overlays.MAP_FRAMES


@pytest.mark.parametrize("content", ["{not json", "[]", '{"v": 1, "events": 5, "frames": 3}', ""])
def test_an_old_or_damaged_cache_file_does_not_crash(world, content) -> None:
    first, _ = world.run()
    files = [
        *(world.src.cdir / "overlays").glob("*.json"),
        *world.src.cdir.glob("events-overlays*"),
    ]
    assert len(files) >= 4  # samples, vision, result, the reads
    for f in files:
        f.write_text(content, encoding="utf-8")
    again, _ = world.run()
    assert again == first
    assert world.run(calls=None)[0] == first  # the repaired result file is read back


def test_a_sample_whose_jpeg_is_gone_is_grabbed_again(world) -> None:
    world.run()
    grabs = world.grabs
    for jpg in (world.src.cdir / "overlays").glob("*.jpg"):
        jpg.unlink()
    world.run()
    assert world.grabs > grabs


def test_stored_elements_with_junk_are_ignored() -> None:
    kinds = ("scoreboard", *overlays.IGNORE_KINDS)
    assert overlays._stored("nope", kinds) is None
    good = {"kind": "scoreboard", "box": [10, 10, 100, 40]}
    junk = [{"kind": "x", "box": [1, 2, 3, 4]}, {"kind": "scoreboard", "box": "no"}, 5, good]
    assert overlays._stored(junk, kinds) == [good]


# --------------------------------------------------------------------------- earlier events are kept


def test_without_a_run_the_events_of_an_earlier_run_with_the_same_kinds_are_kept(world) -> None:
    first, _ = world.run()
    kept, stats = world.run(run=False)
    assert kept == first and "skipped" in stats
    assert world.run(calls=None)[0] == first  # no vision client: the same
    assert world.run(kinds=("gauge",), calls=None)[0] == {}  # other kinds: not those events


def test_kept_events_of_another_overlay_version_are_not_used(world, monkeypatch) -> None:
    world.run()
    monkeypatch.setattr(overlays, "OVERLAY_VERSION", overlays.OVERLAY_VERSION + 1)
    assert world.run(calls=None)[0] == {}


# --------------------------------------------------------------------------- scoreboard readout


def _score_reads(*parts: tuple[int, int, str], cx: float = 100.0) -> list[list[Any]]:
    """Reads at one a second: ``(from, to, text)``, both ends included."""
    return [[float(t), [[text, 0.95, cx]]] for a, b, text in parts for t in range(a, b + 1)]


def _changes(*parts: tuple[int, int, str]) -> list[tuple[int, int]]:
    events, _ = overlays.score_changes("sb", _score_reads(*parts))
    return [(e["before"], e["after"]) for e in events]


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        ([(0, 40, "2"), (41, 80, "3")], [(2, 3)]),  # +1
        ([(0, 40, "2"), (41, 80, "5")], [(2, 5)]),  # +3
        ([(0, 40, "2"), (41, 80, "8")], [(2, 8)]),  # +6
        ([(0, 40, "2"), (41, 80, "9")], []),  # +7: a count or a reset, no score
        ([(0, 40, "98"), (41, 80, "101")], [(98, 101)]),  # 3 digits
        ([(0, 40, "2"), (41, 80, "3"), (81, 120, "2")], [(2, 3)]),  # a step back is none
        ([(0, 60, "2"), (61, 66, "5"), (67, 120, "2")], []),  # flicker
        ([(0, 30, "4"), (31, 31, "9"), (32, 60, "4"), (61, 90, "7")], [(4, 7)]),  # a misread
        ([(0, 10, "2"), (11, 60, "3")], []),  # the old value did not hold
    ],
)
def test_score_steps_of_one_to_six_count(parts, expected) -> None:
    assert _changes(*parts) == expected


def test_a_jump_of_seven_switches_the_slot_off_even_after_a_real_step() -> None:
    events, stats = overlays.score_changes(
        "sb", _score_reads((0, 40, "2"), (41, 80, "3"), (81, 120, "10"))
    )
    assert events == [] and next(iter(stats.values()))["disabled"] is True


def test_an_overlay_scoreboard_has_no_cap_on_its_values() -> None:
    values = [str(v) for v in range(40, 60)]  # a basketball score: many values
    parts = [(i * 20, i * 20 + 18, v) for i, v in enumerate(values)]
    assert len(_changes(*parts)) == 19


def test_two_numbers_of_a_scoreboard_are_two_slots() -> None:
    reads = [
        [float(t), [["2" if t < 50 else "3", 0.95, 100.0], ["7", 0.95, 400.0]]] for t in range(100)
    ]
    events, stats = overlays.score_changes("sb", reads)
    assert [(e["before"], e["after"]) for e in events] == [(2, 3)] and len(stats) == 2


def test_the_gameplay_score_tracker_is_unchanged_without_rules() -> None:
    """A move of 2 still marks a count of items, no score: the slot is off with the defaults."""
    reads = _score_reads((0, 40, "0"), (41, 80, "1"), (81, 120, "3"))

    def track(tracker: ScoreTracker) -> tuple[list, dict]:
        for t, items in reads:
            tracker.see(t, "r", [(quad(100, 0, 30, 40), x[0], x[1]) for x in items], (0, 0, 1, 1))
        return tracker.finish(set())

    events, stats = track(ScoreTracker())
    assert events == [] and next(iter(stats.values()))["disabled"] is True
    events, _ = track(ScoreTracker(overlays.OVERLAY_SCORE_RULES))
    assert [(e["before"], e["after"]) for e in events] == [(0, 1), (1, 3)]


def test_a_three_digit_number_is_a_score_only_under_the_overlay_rules() -> None:
    assert score_value("123") is None and score_value("123", 3) == 123
    assert score_value("1234", 3) is None
    assert score_value("+12", 3) is None and score_value("2:49", 3) is None


def test_an_overlay_scoreboard_counts_increases_only() -> None:
    """F1: the first change a drop (5 to 2) is a reset, no event and no direction; the rise
    after it counts."""
    assert _changes((0, 40, "5"), (41, 80, "2"), (81, 120, "3")) == [(2, 3)]


def test_a_big_drop_is_a_reset_and_does_not_switch_the_slot_off() -> None:
    """F1: 21 to 0 (a new match or period) is no count of items."""
    events, stats = overlays.score_changes(
        "sb", _score_reads((0, 40, "20"), (41, 80, "21"), (81, 120, "0"), (121, 160, "1"))
    )
    assert [(e["before"], e["after"]) for e in events] == [(20, 21), (0, 1)]
    assert next(iter(stats.values()))["disabled"] is False


def test_the_gameplay_rules_keep_their_count_down_direction() -> None:
    """F1: without ``increase_only`` (the gameplay default) a count down still counts."""
    from cutpoint.events.score_readout import GAMEPLAY_SCORE_RULES  # noqa: PLC0415

    assert GAMEPLAY_SCORE_RULES.increase_only is False
    reads = _score_reads((0, 40, "5"), (41, 80, "4"), (81, 120, "3"))
    tracker = ScoreTracker()
    for t, items in reads:
        tracker.see(t, "r", [(quad(100, 0, 30, 40), x[0], x[1]) for x in items], (0, 0, 1, 1))
    events, _ = tracker.finish(set())
    assert [(e["before"], e["after"]) for e in events] == [(5, 4), (4, 3)]


def test_a_minute_mark_is_no_overlay_score() -> None:
    """F2: a match clock's minute ("65'") is no score number."""
    for text in ("65'", "65\u2032", "65\u2019"):
        assert score_value(text, 3) is None
    assert score_value("65", 3) == 65


def _minute_reads(clock_cx: float = 300.0, clock_cy: float = 60.0) -> list[list[Any]]:
    """Ten minutes of a scoreboard: the score 1 (at x 100) and a minute counter in the clock
    box (at ``clock_cx``) climbing by one every 60 s, read once a second."""
    return [
        [
            float(t),
            [["1", 0.95, 100.0, 20.0, 60.0], [str(60 + t // 60), 0.95, clock_cx, 30.0, clock_cy]],
        ]
        for t in range(600)
    ]


def test_a_minute_counter_inside_the_scoreboard_gives_no_score_events() -> None:
    """F2: the clock box the model marked keeps the minute out of the scoreboard's reads."""
    events, _ = overlays.score_changes("sb", _minute_reads())
    assert events  # without the clock box the minute reads as a score (the bug)
    clock = [270, 40, 60, 40]  # x, y, w, h (reference px) around (300, 60)
    events, stats = overlays.score_changes("sb", _minute_reads(), [clock])
    assert events == [] and len(stats) == 1  # the score's slot only


def test_the_clock_box_reaches_the_reader_through_judge_regions_and_to_events() -> None:
    """F2: judge drops the clock's digits from the OCR test and hands the clock on."""

    def box(text: str, x: int, y: int) -> dict[str, Any]:
        return {"text": text, "x0": x - 10, "y0": y - 15, "x1": x + 10, "y1": y + 15}

    # the minute sits between the two scores, inside the scoreboard: "1  65  0"
    frames = [
        {"boxes": [box("1", 520, 60), box(f"{60 + i}", 600, 60), box("0", 680, 60)]}
        for i in range(5)
    ]
    group = {"kind": "scoreboard", "box": (500, 40, 220, 40), "frames": {0, 1}}
    clock = (580, 40, 40, 40)
    rec = overlays.judge(group, frames, [clock])
    assert rec["confirmed"] and rec["evidence"]["score_values"] == [0, 1]
    assert rec["clocks"] == [list(clock)]
    regs = overlays.regions([rec])
    assert regs[0]["clocks"] == [list(clock)]
    reads = {regs[0]["name"]: _minute_reads(600.0, 60.0)}
    events, _ = overlays.to_events(regs, reads)
    assert events == []


def test_a_one_box_score_is_two_slots_and_a_goal_registers() -> None:
    """F3: "0-0" in one OCR box is split into its two numbers."""
    reads = [
        [float(t), [["0-0" if t < 60 else "1-0", 0.95, 500.0, 60.0, 50.0]]] for t in range(120)
    ]
    events, stats = overlays.score_changes("sb", reads)
    assert len(stats) == 2
    assert [(e["before"], e["after"]) for e in events] == [(0, 1)]


def test_a_named_one_box_score_is_split_too() -> None:
    reads = [
        [float(t), [["LIV 2 - 1 MCI" if t < 60 else "LIV 2 - 2 MCI", 0.95, 500.0, 160.0, 50.0]]]
        for t in range(120)
    ]
    events, stats = overlays.score_changes("sb", reads)
    assert len(stats) == 2 and [(e["before"], e["after"]) for e in events] == [(1, 2)]


def test_a_one_box_score_confirms_a_scoreboard() -> None:
    """F3: the OCR test counts both numbers of a "2-1" box."""
    assert overlays.score_numbers("2-1") == [2, 1]
    assert overlays.score_numbers("LIV 2 \u2013 1 MCI") == [2, 1]
    assert overlays.score_numbers("12:30") == [] and overlays.score_numbers("7") == [7]
    frames = [{"boxes": [{"text": "2-1", "x0": 500, "y0": 45, "x1": 560, "y1": 75}]}] * 4
    st = overlays.ocr_stats(frames, (480, 30, 100, 60))
    assert st["score_frames"] == 4 and st["score_values"] == [1, 2]


# --------------------------------------------------------------------------- gauges


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("42", 42.0),
        ("1,234 m", 1234.0),
        ("1 234", 1234.0),
        ("1.234", 1234.0),
        ("12,345.6", 12345.6),
        ("1043m", 1043.0),
        ("12.5", 12.5),
        ("12,5", 12.5),
        ("12.5 km/h", 12.5),
        (" 87 ", 87.0),
    ],
)
def test_gauge_value_reads_numbers(text: str, value: float) -> None:
    assert overlays.gauge_value(text) == value


@pytest.mark.parametrize("text", ["", "abc", "2:49", "50%", "0.125", "--", "12 34 56 78"])
def test_gauge_value_rejects_junk(text: str) -> None:
    assert overlays.gauge_value(text) is None


def _gauge_reads(values: dict[int, float], n: int = 120) -> list[list[Any]]:
    return [[float(t), [[f"{values.get(t, 30 + t % 3):g}", 0.9, 50.0]]] for t in range(n)]


def test_gauge_peaks_are_the_highest_reads_that_stand_out() -> None:
    values = {58: 88, 59: 90, 60: 92, 61: 90, 62: 88, 98: 70, 99: 72, 100: 74, 101: 72, 102: 70}
    values[20] = 8000  # one misread digit: no other read agrees, no peak
    events, stats = overlays.gauge_peaks("g1", _gauge_reads(values))
    assert [(e["t"], e["value"], e["type"]) for e in events] == [
        (60.0, 92.0, "gauge_peak"),
        (100.0, 74.0, "gauge_peak"),
    ]
    assert stats["peaks"] == 2 and "disabled" not in stats


def test_peaks_of_one_climb_are_one_event() -> None:
    values = {t: 60 + (t - 50) * 3 for t in range(50, 60)} | {
        t: 90 - (t - 59) for t in range(60, 70)
    }
    events, _ = overlays.gauge_peaks("g1", _gauge_reads(values))
    assert len(events) == 1 and events[0]["value"] == 89.0


@pytest.mark.parametrize("n", [10, 119])
def test_a_gauge_that_does_not_move_or_is_barely_read_has_no_peaks(n: int) -> None:
    flat = [[float(t), [["50", 0.9, 50.0]]] for t in range(n)]
    events, stats = overlays.gauge_peaks("g1", flat)
    assert events == [] and stats["disabled"] is True


def test_a_gauge_with_too_few_reads_has_no_peaks() -> None:
    events, stats = overlays.gauge_peaks("g1", _gauge_reads({5: 90, 6: 91}, n=10))
    assert events == [] and stats["disabled"] is True


# --------------------------------------------------------------------------- lower-thirds


def _lt_reads(shown: dict[int, str], n: int = 100) -> list[list[Any]]:
    return [[float(t), [[shown[t], 0.9, 100.0]] if t in shown else []] for t in range(n)]


def test_a_lower_third_is_an_event_when_its_words_come_on_screen() -> None:
    shown = dict.fromkeys(range(20, 25), "Jane Doe Host") | dict.fromkeys(
        range(60, 64), "John Roe Guest"
    )
    shown[80] = "Blip Words"  # one read only: not confirmed
    events, stats = overlays.lower_thirds("lt", _lt_reads(shown))
    assert [(e["t"], e["end"], e["text"], e["type"]) for e in events] == [
        (20.0, 24.0, "Jane Doe Host", "lower_third"),
        (60.0, 63.0, "John Roe Guest", "lower_third"),
    ]
    assert stats["shown"] == 2


def test_a_title_that_stays_on_screen_is_no_lower_third() -> None:
    shown = dict.fromkeys(range(5, 95), "Channel Name Live")
    events, stats = overlays.lower_thirds("lt", _lt_reads(shown))
    assert events == [] and stats["disabled"] is True


def test_numbers_are_no_lower_third() -> None:
    shown = dict.fromkeys(range(20, 30), "12 34 56")
    assert overlays.lower_thirds("lt", _lt_reads(shown))[0] == []


# --------------------------------------------------------------------------- styles and the boost


def test_the_kinds_follow_the_style() -> None:
    assert overlay_kinds("sport") == ("scoreboard", "gauge")
    assert overlay_kinds("action") == ("scoreboard", "gauge")
    assert overlay_kinds("action-cam") == ("gauge",)
    assert overlay_kinds("commentary") == ("lower_third",)
    assert overlay_kinds("custom", "commentary") == ("lower_third",)  # a learned style's base
    for gameplay in ("gaming", "funny", "general"):
        assert overlay_kinds(gameplay) == ()


def test_removing_lower_third_from_the_styles_switches_it_off(monkeypatch) -> None:
    monkeypatch.delitem(OVERLAY_STYLES, "lower_third")
    assert overlay_kinds("commentary") == ()
    assert overlay_kinds("sport") == ("scoreboard", "gauge")


def test_the_overlay_layer_is_in_the_layer_lists() -> None:
    assert "overlay" in layers.LAYERS
    assert "overlay" in layers.DEFAULT_LAYERS and "overlay" in layers.ACTION_CAM_LAYERS


def _ov(kind: str, type_: str, t: float = 10.0) -> dict[str, Any]:
    """An events.json entry of an overlay event."""
    return {
        "source": "rec.mp4",
        "start": t,
        "end": t + 1,
        "kind": kind,
        "type": type_,
        "label": "x",
        "confidence": 0.9,
        "details": {"overlay": True},
    }


CAND = {"start": 0.0, "end": 30.0}
SCORE, GAUGE, LT = (
    _ov("ocr", "score"),
    _ov("ocr", "gauge_peak", 12.0),
    _ov("ocr", "lower_third", 14.0),
)


@pytest.mark.parametrize(
    ("style", "events", "points"),
    [
        ("sport", [SCORE], 3.0),
        ("action", [GAUGE], 2.0),
        ("action-cam", [GAUGE], 2.0),
        ("commentary", [LT], 1.0),
        ("sport", [LT], 0.0),  # lower-thirds boost commentary only
        ("action", [LT], 0.0),
        ("commentary", [SCORE, GAUGE], 0.0),
        ("gaming", [SCORE, GAUGE, LT], 0.0),  # gameplay: none
        ("funny", [SCORE, GAUGE, LT], 0.0),
    ],
)
def test_overlay_events_boost_by_their_kind_and_style(style, events, points) -> None:
    boost, evidence = scoring.event_boost(CAND, events, style=style)
    assert boost == pytest.approx(points)
    assert all(v >= 0 for v in evidence.values() if isinstance(v, int | float))


def test_overlapping_overlay_events_add_at_most_three_points() -> None:
    events = [SCORE, GAUGE, _ov("ocr", "lower_third", 16.0)]
    for style in ("sport", "action"):
        boost, evidence = scoring.event_boost(CAND, events, style=style)
        assert boost == eventtypes.OVERLAY_MAX_POINTS == 3.0
        assert evidence["overlay_scores"] == 1 and evidence["overlay_gauges"] == 1
    assert scoring.event_boost(CAND, [GAUGE], style="sport")[0] == 2.0  # under the cap: whole


def test_an_overlay_moment_has_no_negative_points_and_the_layer_nominates_nothing() -> None:
    for t in eventtypes.OVERLAY_TYPES:
        assert t.score > 0 and not t.nominate and not t.payoff
    boost, _ = scoring.event_boost(CAND, [], style="sport")
    assert boost == 0.0


def test_a_gameplay_score_event_is_no_overlay_event() -> None:
    plain = {**SCORE, "details": {}}
    _, evidence = scoring.event_boost(CAND, [plain], style="sport")
    assert "overlay_scores" not in evidence


def test_overlay_evidence_keys_only_appear_with_a_count() -> None:
    _, evidence = scoring.event_boost(CAND, [], style="sport")
    assert not any(k.startswith("overlay_") for k in evidence)
    _, evidence = scoring.event_boost(CAND, [SCORE], style="sport")
    assert evidence["overlay_scores"] == 1


# --------------------------------------------------------------------------- the pipeline step


def test_hud_map_parse_keeps_only_the_wanted_kinds() -> None:
    reply = {"elements": [{"kind": "scoreboard", "box": [100, 100, 300, 200], "text_seen": ""}]}
    assert hud_map.parse(reply, (1280, 720)) == []  # not a gameplay HUD kind
    got = hud_map.parse(reply, (1280, 720), ("scoreboard",))
    assert got is not None and [e["kind"] for e in got] == ["scoreboard"]


class _Ocr:
    released = 0

    def __init__(self, **kw: Any) -> None:
        pass

    def release(self) -> None:
        type(self).released += 1


@pytest.fixture
def layer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """``pipeline._overlay_layer`` over a fake run with ``overlays.run_overlays`` faked."""
    monkeypatch.setattr("cutpoint.events.ocr_engine.SharedOcr", _Ocr)
    state = SimpleNamespace(calls=[], result=({}, {"kinds": ["x"]}), error=None)

    def fake(sources: Any, kinds: Any, calls: Any, **kw: Any) -> Any:
        state.calls.append((tuple(kinds), calls, kw))
        if state.error:
            raise state.error
        return state.result

    monkeypatch.setattr(overlays, "run_overlays", fake)
    state.run = lambda style="commentary", layers=("overlay",), **hl: SimpleNamespace(
        hl={"style": style, **hl},
        sources=[SimpleNamespace(path=tmp_path / "a.mp4")],
        layers=layers,
        force=(),
        bad={},
        layer1={str(tmp_path / "a.mp4"): []},
        stats={},
        timings={},
        vision_off=False,
    )
    state.path = str(tmp_path / "a.mp4")
    return state


def test_the_step_does_nothing_for_gameplay(layer) -> None:
    run = layer.run("gaming")
    pipeline._overlay_layer(run, None, None, hwaccel=True)
    assert layer.calls == [] and run.stats == {}


def test_the_step_hands_the_events_to_layer_one_and_records_the_stats(layer) -> None:
    ev = {"t": 3.0, "end": 4.0, "type": "lower_third", "kind": "ocr", "overlay": True}
    layer.result = ({layer.path: [ev]}, {"kinds": ["lower_third"], "v": 1})
    rater = _rater()
    run = layer.run("commentary", settings={"context": "  two\nhosts "})
    pipeline._overlay_layer(run, rater, None, hwaccel=False)
    kinds, calls, kw = layer.calls[0]
    assert kinds == ("lower_third",) and calls.rater is rater
    assert kw["run"] is True and kw["hwaccel"] is False and kw["force"] is False
    assert kw["context"] == "two hosts"  # the analysis's --context words
    assert run.layer1[layer.path] == [ev]
    assert run.stats["overlays"]["kinds"] == ["lower_third"] and "overlay_s" in run.timings
    assert _Ocr.released >= 1


def test_the_step_without_the_layer_keeps_the_earlier_events_and_does_not_run(layer) -> None:
    run = layer.run("sport", layers=("ocr",))
    pipeline._overlay_layer(run, None, None, hwaccel=True)
    kinds, calls, kw = layer.calls[0]
    assert kinds == ("scoreboard", "gauge") and calls is None and kw["run"] is False
    assert "overlay_s" not in run.timings


def test_a_learned_style_uses_the_kinds_of_its_base(layer) -> None:
    pipeline._overlay_layer(layer.run("mine", style_base="commentary"), None, None, hwaccel=True)
    assert layer.calls[0][0] == ("lower_third",)


def test_removing_the_kind_from_the_styles_stops_the_step(layer, monkeypatch) -> None:
    monkeypatch.delitem(OVERLAY_STYLES, "lower_third")
    pipeline._overlay_layer(layer.run("commentary"), None, None, hwaccel=True)
    assert layer.calls == []


@pytest.mark.parametrize("error", [RuntimeError("model blew up"), LowMemoryError("too little")])
def test_a_failing_step_warns_and_the_run_goes_on(layer, error, caplog) -> None:
    layer.error = error
    run = layer.run("sport")
    with jsonout.run("events") as res, caplog.at_level(logging.WARNING):
        pipeline._overlay_layer(run, None, None, hwaccel=True)
    assert [w["code"] for w in res.warnings] == ["step_failed"]
    assert str(error) in res.warnings[0]["message"] and "Overlays skipped" in caplog.text
    assert run.layer1[layer.path] == [] and "overlays" not in run.stats
    assert _Ocr.released >= 1  # the engine is released all the same


def test_a_failing_overlay_layer_does_not_stop_the_other_layers(
    layer, tmp_path, monkeypatch, caplog
) -> None:
    """Through ``run_events``: the VLM layer after it is still reached."""
    srcs = [SimpleNamespace(path=tmp_path / "a.mp4", name="a.mp4", duration=60.0)]
    stub_ocr_layer(monkeypatch, srcs, {"style": "commentary"})
    layer.error = RuntimeError("model blew up")
    with jsonout.run("events") as res, pytest.raises(StopAtVlm):
        pipeline.run_events(tmp_path / "proj", layers=("overlay", "vlm"))
    assert len(layer.calls) == 1
    assert [w["code"] for w in res.warnings] == ["step_failed"]


# --------------------------------------------------------------------------- settings.overlays


def test_summary_of_a_run_names_what_was_confirmed(world) -> None:
    _, stats = world.run()
    got = overlays.summary(stats)
    assert got == {
        "v": overlays.OVERLAY_VERSION,
        "kinds": ["scoreboard"],
        "model": "hosted@h:m",
        "calls": overlays.MAP_FRAMES,
        "events": {"score": 3},
        "confirmed": ["scoreboard"],
    }
    json.dumps(got)  # goes into highlights.json


@pytest.mark.parametrize("stats", [None, {}, {"kinds": []}, "x"])
def test_no_summary_when_the_layer_did_not_look(stats) -> None:
    assert overlays.summary(stats) is None


def test_the_summary_says_why_the_layer_was_skipped() -> None:
    stats = {"v": 1, "kinds": ["gauge"], "skipped": "no vision consent"}
    assert overlays.summary(stats) == {"v": 1, "kinds": ["gauge"], "skipped": "no vision consent"}


def test_highlights_record_settings_overlays_and_drop_a_stale_one(tmp_path) -> None:
    stats = {"v": 1, "kinds": ["scoreboard"], "model": "m", "calls": 3}
    stats["regions"] = [{"kind": "scoreboard"}]
    summary = overlays.summary(stats)
    assert summary is not None and summary["confirmed"] == ["scoreboard"]
    hl: dict[str, Any] = {"style": "sport", "settings": {}, "candidates": []}
    scorer.rescore_events(hl, {"events": [], "windows": [], "stats": {"overlays": stats}}, tmp_path)
    assert hl["settings"]["overlays"] == summary
    scorer.rescore_events(hl, {"events": [], "windows": [], "stats": {}}, tmp_path)
    assert "overlays" not in hl["settings"]  # a gameplay re-run: absent


# --------------------------------------------------------------------------- --no-vision (V1)


def test_no_vision_asks_the_model_nothing_for_the_overlays(world, caplog) -> None:
    with caplog.at_level(logging.INFO):
        out, stats = world.run(vision=False)
    assert world.rater.asked == [] and world.grabs == 0 and world.decodes == 0
    assert out == {} and stats["skipped"] == "vision off"
    lines = [m for m in caplog.messages if m.startswith("Vision is off")]
    assert lines == ['Vision is off (--no-vision or "vision": false): overlays not read']


def test_no_vision_keeps_the_overlay_events_of_an_earlier_run(world) -> None:
    first, _ = world.run()
    assert first
    world.rater.asked.clear()
    again, _ = world.run(vision=False)
    assert again == first and world.rater.asked == []


@pytest.mark.parametrize(
    ("vision", "settings", "off"),
    [
        (False, {}, True),
        (True, {"vision_off": True}, False),  # given on the command line: it wins
        (None, {"vision_off": True}, True),  # recorded by the analysis: events follow it
        (None, {}, False),
        (None, {"vision_off": "yes"}, False),
    ],
)
def test_vision_off_follows_the_option_else_the_analysis(vision, settings, off) -> None:
    assert pipeline.vision_off(vision, {"settings": settings}) is off


@pytest.mark.parametrize("off", [True, False])
def test_no_vision_gives_the_hud_discovery_no_vision_map(
    tmp_path, monkeypatch, caplog, off
) -> None:
    """V1: with vision off the HUD discovery gets no mapper (no HUD map, no top-up) and no
    feed labeler, one INFO line each; without it both are made as before."""
    from cutpoint.events import feed_labels  # noqa: PLC0415
    from cutpoint.events.killfeed import OcrParams  # noqa: PLC0415

    got: list[Any] = []
    made: list[Any] = []
    labelers: list[Any] = []

    def discover(*a: Any, **k: Any) -> dict:
        got.append((a[7], a[8]))  # the feed labeler and the hud_mapper
        return {}

    monkeypatch.setattr(pipeline, "_discover", discover)
    monkeypatch.setattr(hud_map, "make_mapper", lambda *a, **k: made.append(a) or "mapper")
    monkeypatch.setattr(feed_labels, "make_labeler", lambda *a, **k: labelers.append(a) or "lab")
    run = SimpleNamespace(
        project=tmp_path, sources=[], hl={"settings": {}}, stats={}, timings={}, vision_off=off
    )
    with caplog.at_level(logging.INFO):
        pipeline._ocr_discovery(
            run, OcrParams(), _rater(), 0.0, discover=None, bursts=40, scope="user", hint="",
            context=None, prefetch=None,
        )  # fmt: skip
    lines = [m for m in caplog.messages if m.startswith("Vision is off")]
    if off:
        assert got == [(None, None)] and made == [] and labelers == [] and len(lines) == 2
        assert "no HUD map or top-up" in lines[0] and "no feed labels" in lines[1]
    else:
        assert got == [("lab", "mapper")] and len(made) == len(labelers) == 1 and lines == []
