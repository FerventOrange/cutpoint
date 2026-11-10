"""Score readout (#800 phase 2): small on-screen numbers that step by one, found by discovery
and read as ``score`` events (invented boxes, values and names only)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cutpoint import paths
from cutpoint.events import (
    discover,
    eventtypes,
    killfeed,
    ocr_raw,
    pipeline,
    report,
    scoring,
    tails,
    vlm_windows,
)
from cutpoint.events.common import Source
from cutpoint.events.discover import (
    DiscoverParams,
    cluster_regions,
    proposals_to_regions,
    score_region,
    score_value,
)
from cutpoint.events.killfeed import (
    FrameRec,
    OcrInterpretation,
    OcrParams,
    OcrReader,
    ScoreTracker,
)
from cutpoint.events.regions import Region, preset_from_dict
from hud_fixtures import verbfeed, write_preset

REF = (100.0, 0.0, 1.0, 1.0)  # a crop at x 100 of a 1920x1080 frame


def _box(x: float, w: float = 30.0) -> list[list[float]]:
    return [[x, 10.0], [x + w, 10.0], [x + w, 50.0], [x, 50.0]]


def _reads(*parts: tuple[float, float, str], conf: float = 0.95, x: float = 20.0) -> list:
    """Reads at 1 a second: ``(from, to, text)`` per part, both ends included."""
    out = []
    for a, b, text in parts:
        out += [(float(t), [(_box(x), text, conf)]) for t in range(int(a), int(b) + 1)]
    return out


def _track(reads: list, menu: set[float] | None = None) -> tuple[list[dict[str, Any]], dict]:
    tracker = ScoreTracker()
    for t, items in sorted(reads, key=lambda r: r[0]):
        tracker.see(t, "score", items, REF)
    return tracker.finish(menu or set())


def _changes(reads: list, menu: set[float] | None = None) -> list[str]:
    return [e["text"] for e in _track(reads, menu)[0]]


# --------------------------------------------------------------------------- tokens


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("3", 3),
        ("12", 12),
        ("3X", 3),
        ("x7", 7),
        (" 4 ", 4),
        ("2:49", None),
        ("50%", None),
        ("+10", None),
        ("+1", None),
        ("123", None),
        ("A12B3", None),
        ("ROUND", None),
        ("", None),
    ],
)
def test_score_value(text: str, value: int | None) -> None:
    assert score_value(text) == value


# --------------------------------------------------------------------------- tracker


def test_a_count_down_step_is_an_event_at_the_last_read_of_the_old_value() -> None:
    events, stats = _track(_reads((0, 40, "5"), (41, 80, "4")))
    assert len(events) == 1
    e = events[0]
    assert (e["t"], e["end"], e["type"], e["text"]) == (40.0, 41.0, "score", "5 > 4")
    assert (e["before"], e["after"], e["seen_at"], e["gap_s"]) == (5, 4, 41.0, 1.0)
    assert e["region"] == "score" and e["slot"] == 135  # (100 + 20 + 15) reference px
    assert stats == {"score@135": {"reads": 81, "values": 2, "changes": 1, "disabled": False}}


def test_a_hidden_hud_up_to_a_minute_still_compares() -> None:
    events, _ = _track(_reads((300, 390, "0"), (406, 450, "1")))
    assert [(e["t"], e["seen_at"], e["gap_s"]) for e in events] == [(390.0, 406.0, 16.0)]
    assert _changes(_reads((200, 300, "0"), (391, 450, "1"))) == []  # 91 s: a new match


def test_the_old_value_must_hold() -> None:
    assert _changes(_reads((0, 10, "2"), (11, 60, "3"))) == []  # 10 s only
    assert _changes(_reads((0, 15, "2"), (16, 60, "3"))) == ["2 > 3"]


def test_jitter_and_resets_give_no_event() -> None:
    assert _changes(_reads((0, 60, "2"), (61, 66, "1"), (67, 120, "2"))) == []
    assert _changes(_reads((0, 60, "1"), (61, 120, "5"))) == []  # a reset between matches
    assert _changes(_reads((0, 60, "0"), (61, 120, "32307"))) == []  # a count-up is no score


def test_a_lone_misread_is_no_value() -> None:
    reads = _reads((0, 30, "4"), (31, 31, "7"), (32, 60, "4"), (61, 90, "3"))
    assert _changes(reads) == ["4 > 3"]


def test_each_slot_keeps_its_first_direction() -> None:
    up = _reads((0, 40, "0"), (41, 80, "1"), (81, 120, "2"), (121, 160, "1"), (161, 200, "2"))
    assert _changes(up) == ["0 > 1", "1 > 2", "1 > 2"]
    down = _reads((0, 40, "3"), (41, 80, "2"), (81, 120, "3"), (121, 160, "2"))
    assert _changes(down) == ["3 > 2", "3 > 2"]


def test_a_counter_that_drops_by_several_is_no_score() -> None:
    """Items used two at a time, then one: a count, not a score (#872)."""
    reads = _reads((0, 30, "9"), (40, 100, "7"), (101, 160, "6"))
    events, stats = _track(reads)
    assert events == [] and stats["score@135"]["disabled"] is True


def test_a_jump_counts_without_a_long_old_value() -> None:
    """The old value had few reads (the HUD hidden before it), the new one held: a jump."""
    events, stats = _track(_reads((0, 5, "8"), (12, 70, "6"), (71, 130, "5")))
    assert events == [] and stats["score@135"]["disabled"] is True


def test_a_jump_to_a_value_that_does_not_hold_is_no_count() -> None:
    reads = _reads((0, 40, "5"), (41, 80, "4"), (81, 85, "2"), (86, 130, "3"), (131, 170, "2"))
    assert _changes(reads) == ["5 > 4", "3 > 2"]


def test_a_reset_the_other_way_keeps_the_score() -> None:
    up = _reads((0, 40, "0"), (41, 80, "1"), (81, 120, "2"), (125, 160, "0"), (161, 200, "1"))
    assert _changes(up) == ["0 > 1", "1 > 2", "0 > 1"]
    down = _reads((0, 40, "3"), (41, 80, "2"), (81, 120, "1"), (125, 160, "4"), (161, 200, "3"))
    assert _changes(down) == ["3 > 2", "2 > 1", "4 > 3"]


def test_a_jump_before_the_direction_is_known_counts() -> None:
    reads = _reads((0, 40, "2"), (41, 80, "4"), (81, 120, "5"))  # +2, then +1
    events, stats = _track(reads)
    assert events == [] and stats["score@135"]["disabled"] is True


def test_a_jump_over_a_long_gap_is_a_new_match() -> None:
    reads = _reads((0, 40, "1"), (41, 80, "2"), (141, 200, "5"), (201, 240, "6"))
    assert _changes(reads) == ["1 > 2", "5 > 6"]


def test_reads_need_confidence_and_two_in_a_row() -> None:
    assert _changes(_reads((0, 40, "5")) + _reads((41, 80, "4"), conf=0.65)) == []
    assert _changes(_reads((0, 40, "5"), (41, 41, "4"))) == []


def test_a_slot_that_changes_too_often_is_no_score() -> None:
    fast = [part for k in range(20) for part in [(4 * k, 4 * k + 3, str(k))]]
    events, stats = _track(_reads(*fast))
    assert events == [] and stats["score@135"]["disabled"] is True
    many = [(40 * k, 40 * k + 39, str(k)) for k in range(14)]  # 14 values, each held 40 s
    events, stats = _track(_reads(*many))
    assert events == [] and stats["score@135"] == {
        "reads": 560,
        "values": 14,
        "changes": 0,
        "disabled": True,
    }


def test_menu_frame_reads_are_left_out() -> None:
    reads = _reads((0, 50, "5"), (51, 80, "4"))
    assert _changes(reads) == ["5 > 4"]
    assert _changes(reads, menu={float(t) for t in range(51, 81)}) == []


def test_two_slots_of_one_region_are_apart() -> None:
    left = _reads((0, 40, "1"), (41, 80, "2"), x=0.0)
    right = _reads((0, 60, "3"), (61, 80, "4"), x=200.0)
    events, stats = _track(left + right)
    assert sorted((e["slot"], e["text"]) for e in events) == [(115, "1 > 2"), (315, "3 > 4")]
    assert sorted(stats) == ["score@115", "score@315"]


def test_a_wider_number_stays_in_its_slot() -> None:
    reads = [(float(t), [(_box(20.0), "9", 0.95)]) for t in range(40)]
    reads += [(float(t), [(_box(8.0, 50.0), "10", 0.95)]) for t in range(40, 80)]
    events, _ = _track(reads)
    assert [e["text"] for e in events] == ["9 > 10"]


# --------------------------------------------------------------------------- interpretation

SCORE = Region("score", 100, 20, 80, 60, "score", gate="read", every=1, upscale=2.5)
NOTICE = Region("notice", 600, 700, 700, 100, "notice", gate="det", every=1)


def _interpret(
    reads: list, regions: tuple[Region, ...] = (SCORE,), preset: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {}
    interp = OcrInterpretation(
        "rec.mp4",
        OcrParams(game="tilefall", fps=1.0),
        preset_from_dict({"game": "tilefall", "regions": []}, None) if preset else None,
        regions,
        (1920, 1080),
        stats,
    )
    by_t = dict(reads)
    for n in range(int(max(by_t)) + 1):
        recs: dict[str, dict[str, Any]] = {}
        for r in regions:
            if r.kind == "score":
                recs[r.name] = {"g": "read", "items": by_t.get(float(n), [])}
            else:
                recs[r.name] = {"g": "det", "boxes": []}
        interp.add(FrameRec(n=n, t=float(n), regions=recs))
    return interp.finish(), stats


def test_score_regions_make_score_events() -> None:
    events, stats = _interpret(_reads((0, 40, "5"), (41, 80, "4")))
    assert [(e["kind"], e["type"], e["text"], e["t"]) for e in events] == [
        ("ocr", "score", "5 > 4", 40.0)
    ]
    assert stats["score_slots"]["score@135"]["changes"] == 1
    assert stats["ocr_calls"] == {"score": 81}
    normal = report.normalize("rec.mp4", events[0])
    assert normal["label"] == "5 > 4" and normal["details"]["before"] == 5
    assert report._event_line("rec.mp4", normal).endswith("OCR score: 5 > 4")


def test_a_disabled_slot_is_logged_once_per_source(caplog: pytest.LogCaptureFixture) -> None:
    fast = [part for k in range(20) for part in [(4 * k, 4 * k + 3, str(k))]]
    with caplog.at_level("INFO", logger=killfeed.logger.name):
        events, stats = _interpret(_reads(*fast))
        assert not caplog.records  # logged by detect_ocr_events, fresh or cached, not here
        killfeed._log_score_slots("rec.mp4", stats)
    assert events == [] and stats["score_slots"]["score@135"]["disabled"] is True
    lines = [r.getMessage() for r in caplog.records if "score readout off" in r.getMessage()]
    assert lines == ["rec.mp4: score readout off for score@135 (a counter, not a score)"]
    caplog.clear()
    with caplog.at_level("INFO", logger=killfeed.logger.name):
        killfeed._log_score_slots("rec.mp4", _interpret(_reads((0, 40, "5"), (41, 80, "4")))[1])
    assert not [r for r in caplog.records if "score readout off" in r.getMessage()]


def test_a_cached_result_logs_its_disabled_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    region = {"name": "score", "x": 100, "y": 20, "w": 80, "h": 60, "kind": "score"}
    preset = preset_from_dict({"game": "tilefall", "regions": [region | {"gate": "read"}]}, None)
    monkeypatch.setattr(killfeed, "load_preset", lambda *_a, **_k: preset)
    slots = {"score@135": {"disabled": True}, "score@300": {"disabled": False}}
    hit = {"events": [], "stats": {"score_slots": slots}}
    monkeypatch.setattr(killfeed.cache, "load_json", lambda _p: hit)
    src = Source(tmp_path / "rec.mp4", 60.0, tmp_path, [])
    with caplog.at_level("INFO", logger=killfeed.logger.name):
        assert killfeed.detect_ocr_events(src, OcrParams(game="tilefall", fps=1.0), None) == (
            [],
            hit["stats"],
        )
    lines = [r.getMessage() for r in caplog.records if "score readout off" in r.getMessage()]
    assert lines == ["rec.mp4: score readout off for score@135 (a counter, not a score)"]


def test_score_region_reads_make_no_text_event_and_are_not_learned() -> None:
    """A counter box misread as a CJK character and a number is no big text: the tracker
    sees it, the announcement learner never does."""
    reads = [
        (float(t), [(_box(20.0), f"\u6708{t % 5 + 1}", 0.95)])
        for t in range(0, 3000, 240)
        for t in (t, t + 1)
    ]
    events, _ = _interpret(reads)
    assert not any(e["type"] in ("bigtext", "notice", "banner") for e in events)
    assert not any(e.get("announce") for e in events)


def test_the_interpreter_ignores_a_score_region() -> None:
    interp = killfeed.Interpreter(OcrParams(game="tilefall", fps=1.0), 1.0)
    interp.handle(5.0, SCORE, None, [(_box(20.0), "\u670810", 0.95)])
    assert interp.tracker.events == [] and interp.digits.events == []


def test_the_score_tracker_still_sees_the_misread() -> None:
    events, _ = _track(_reads((0, 40, "\u67085"), (41, 80, "\u67086")))
    assert [e["type"] for e in events] == ["score"]


BIGTEXT = Region("big", 0, 0, 1920, 1080, "bigtext", gate="read", every=1, min_text_h=36)


def _bigtext_lines(
    regions: tuple[Region, ...],
    *lines: tuple[float, str],
    size: tuple[int, int] = (1920, 1080),
) -> list[str]:
    """The texts the big-text region's reads leave in the tracker; ``lines``: ``(x, text)`` of
    a 60 px tall line, the first at y 20 and the others each 100 px lower, read for 5 frames."""
    interp = OcrInterpretation(
        "rec.mp4",
        OcrParams(game="tilefall", fps=1.0),
        preset_from_dict({"game": "tilefall", "regions": []}, None),
        regions,
        size,
        {},
    )
    for n in range(5):
        items = [
            ([[x, y], [x + 40, y], [x + 40, y + 60], [x, y + 60]], text, 0.95)
            for (x, text), y in zip(lines, range(20, 1000, 100), strict=False)
        ]
        recs: dict[str, dict[str, Any]] = {r.name: {"g": "read", "items": items} for r in regions}
        interp.add(FrameRec(n=n, t=float(n), regions=recs))
    return [e["text"] for e in interp.interp.tracker.events]


def test_a_text_line_centred_in_a_score_box_is_no_text() -> None:
    got = _bigtext_lines((SCORE, BIGTEXT), (110.0, "ZONE 7"), (110.0, "OTHER WORDS"))
    assert got == ["OTHER WORDS"]  # the rest of the big region's lines stay


def test_score_box_test_applies_the_text_region_offset_and_the_frame_scale() -> None:
    """On a 3840x2160 frame the score box (500, 400, 80x60 reference px) spans x 1000-1160,
    y 800-920 and the text region (400, 300) starts at 800, 600: the line (x 200, third row, y 220)
    is centred at (1020, 850) and is a counter; the line (x 1000, ninth row, y 820) is centred
    in the box only when the crop offset is ignored, so it stays."""
    score = Region("score2", 500, 400, 80, 60, "score", gate="read", every=1)
    text = Region("big2", 400, 300, 1000, 500, "bigtext", gate="read", every=1, min_text_h=18)
    lines = [(1500.0, f"FILLER {i}") for i in range(9)]
    lines[2] = (200.0, "COUNTER")
    lines[8] = (1000.0, "REAL TEXT")
    got = _bigtext_lines((score, text), *lines, size=(3840, 2160))
    assert "COUNTER" not in got and "REAL TEXT" in got


def test_without_score_regions_text_lines_stay() -> None:
    got = _bigtext_lines((BIGTEXT,), (110.0, "ZONE 7"), (110.0, "OTHER WORDS"))
    assert sorted(got) == ["OTHER WORDS", "ZONE 7"]


def test_menu_frames_drop_score_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    def rules(frames: list[dict[str, Any]], **_k: Any) -> tuple[dict[Any, str], dict]:
        return {f["id"]: "cover" if f["t"] > 50 else "none" for f in frames}, {}

    monkeypatch.setattr(killfeed, "menu_rules", rules)
    events, stats = _interpret(_reads((0, 50, "5"), (51, 80, "4")), (SCORE, NOTICE))
    assert events == [] and stats["menu_frames"] == 30


def test_no_score_region_no_score_stats() -> None:
    _, stats = _interpret([(0.0, []), (5.0, [])], (NOTICE,))
    assert "score_slots" not in stats


# --------------------------------------------------------------------------- reading


class _Engine:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, crop: np.ndarray, min_score: float) -> list[tuple[list, str, float]]:
        self.shapes.append(crop.shape)
        return [([[10.0, 5.0], [40.0, 5.0], [40.0, 45.0], [10.0, 45.0]], "3", 0.9)]


def _reader(region: Region, height: int) -> tuple[OcrReader, _Engine]:
    engine = _Engine()
    width = height * 16 // 9
    reader = OcrReader(
        (region,),
        {region.name: (0, 0, 80, 60)},
        (0, 0, 80, 60),
        engine,  # type: ignore[arg-type]
        OcrParams(),
        height / 1080,
        "bottom",
        (width, height),
    )
    return reader, engine


@pytest.mark.parametrize(("height", "shape"), [(1080, (150, 200, 3)), (2160, (75, 100, 3))])
def test_read_gate_enlarges_to_the_reference_size(height: int, shape: tuple[int, ...]) -> None:
    reader, engine = _reader(SCORE, height)
    crop = np.zeros((60, 80, 3), np.uint8)
    g, rows, items = reader._look(SCORE, crop)
    assert g == {} and rows is None and engine.shapes == [shape]
    f = shape[0] / 60
    assert items == [
        ([[10 / f, 5 / f], [40 / f, 5 / f], [40 / f, 45 / f], [10 / f, 45 / f]], "3", 0.9)
    ]


class _DetEngine(_Engine):
    def detect(self, crop: np.ndarray, scale: float = 1.0) -> list[list]:
        return []


def test_the_static_test_area_leaves_score_regions_out() -> None:
    """A score box above the text regions grows the decoded area, not the thumbnails the
    static-screen test compares nor its area share (#872)."""
    notice = Region("notice", 0, 124, 400, 200, "notice", gate="det", every=1)
    score = Region("score", 40, 0, 60, 40, "score", gate="read", every=3, upscale=2.5)
    rects = {"notice": (0, 124, 400, 200), "score": (40, 0, 60, 40)}
    assert killfeed._still_rect((notice, score), rects) == (0, 124, 400, 200)
    assert killfeed._still_rect((score,), {"score": rects["score"]}) == (40, 0, 60, 40)
    engine = _DetEngine()
    union = killfeed._union(list(rects.values()))
    params, size = OcrParams(), (1920, 1080)
    reader = OcrReader((notice, score), rects, union, engine, params, 1.0, "bottom", size)  # type: ignore[arg-type]
    frame = np.random.default_rng(0).integers(0, 255, (union[3], union[2], 3), np.uint8)
    rec, _ = reader.read(1, 0.0, frame)
    assert rec.thumb is not None
    np.testing.assert_array_equal(rec.thumb, killfeed._thumb_bgr(frame[124:324, 0:400]))
    stats: dict[str, Any] = {}
    preset = preset_from_dict({"game": "tilefall", "regions": []}, None)
    OcrInterpretation("rec.mp4", params, preset, (notice, score), size, stats)
    assert stats["static_area"] == round(400 * 200 / (1920 * 1080), 2)


def test_no_upscale_reads_the_crop_as_decoded() -> None:
    plain = Region("score", 100, 20, 80, 60, "score", gate="read")
    reader, engine = _reader(plain, 720)  # a small source: still as decoded
    reader._look(plain, np.zeros((60, 80, 3), np.uint8))
    assert engine.shapes == [(60, 80, 3)]


# --------------------------------------------------------------------------- presets


def test_region_key_and_preset_round_trip() -> None:
    old = Region("feed", 1500, 0, 400, 400, "feed", gate="det")
    assert "upscale" not in old.key()  # keys of existing presets stay as they were
    assert SCORE.key()["upscale"] == 2.5 and SCORE.key()["kind"] == "score"
    loaded = preset_from_dict({"game": "g", "regions": [SCORE.key(), old.key()]}, None)
    assert loaded.regions == (SCORE, old)
    legacy = preset_from_dict({"game": "g", "regions": [{**old.key(), "label": "kill_feed"}]})
    assert legacy.regions[0].upscale == 1.0


# --------------------------------------------------------------------------- discovery


def _b(text: str, x: int, y: int, w: int = 30, h: int = 36) -> dict[str, Any]:
    return {"x0": x, "y0": y, "x1": x + w, "y1": y + h, "text": text, "score": 0.9}


def _frame(b: int, k: int, slots: int = 1) -> dict[str, Any]:
    """A steady counter at the top left (4 values, each for 10 bursts), a clock, ammo that
    changes within a burst, "+N" pop-ups, a fixed digit and a large number."""
    boxes = [
        _b(str(4 - b // 10), 60, 30),  # the counter
        _b(f"{b % 3}:{50 - k:02d}", 900, 20, w=60),  # a clock
        _b(str(30 - (b + k) % 12), 1700, 1000),  # ammo: changes inside a burst
        _b(f"+{10 * (k + 1)}", 1400, 500),  # point pop-ups
        _b("2", 40, 1000),  # a fixed key hint: one value
        _b(str(40 + b // 10), 1800, 40),  # steady, but above 30
        _b("HEALTH", 120, 1040, w=100),
    ]
    if slots > 1:
        boxes += [_b(str(b // 13), 1000 + 200 * s, 30) for s in range(slots - 1)]
    return {"id": f"rec.mp4@{b * 60 + k * 1.5:.1f}", "source": "rec.mp4", "burst": b, "k": k} | {
        "frame": f"b{b:03d}_{k}.jpg",
        "boxes": boxes,
    }


def _frames(slots: int = 1) -> list[dict[str, Any]]:
    return [_frame(b, k, slots) for b in range(40) for k in range(4)]


def test_discovery_finds_a_steady_counter_only() -> None:
    proposals, _ = cluster_regions(_frames(), DiscoverParams())
    scores = [p for p in proposals if p["kind"] == "score"]
    assert [(p["name"], p["x"], p["y"], p["w"], p["h"]) for p in scores] == [
        ("score", 60, 30, 30, 36)
    ]
    assert scores[0]["evidence"]["values"] == [1, 2, 3, 4]
    regions = [r for r in proposals_to_regions(proposals) if r.kind == "score"]
    assert regions == [Region("score", 38, 18, 74, 60, "score", gate="read", every=3, upscale=2.5)]


def test_a_two_digit_value_fits_a_region_found_on_one_digit_values() -> None:
    one = {
        "name": "score",
        "x": 60,
        "y": 30,
        "w": 30,
        "h": 36,
        "evidence": {"values": [1, 4], "median_text_h": 36},
    }
    two = {**one, "evidence": {"values": [8, 12]}}
    r = score_region(one)
    assert (r.x, r.w, r.y, r.h) == (38, 74, 18, 60)
    assert (score_region(two).x, score_region(two).w) == (48, 54)


def test_discovery_keeps_two_score_slots_at_most() -> None:
    proposals, _ = cluster_regions(_frames(slots=4), DiscoverParams())
    assert [p["name"] for p in proposals if p["kind"] == "score"] == ["score", "score2"]


def test_discovery_needs_six_bursts() -> None:
    frames = [f for f in _frames() if f["burst"] in (5, 9, 10, 11, 20)]
    proposals, _ = cluster_regions(frames, DiscoverParams())
    assert not [p for p in proposals if p["kind"] == "score"]


def test_a_new_preset_gets_the_score_region(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(discover, "_ocr_frames", lambda *_a, **_k: _frames())
    monkeypatch.setattr(discover, "probe", lambda _p: None)
    preset = discover.discover_regions([], tmp_path / "disc", "tilefall")
    got = [r for r in preset["regions"] if r["kind"] == "score"]
    assert got == [SCORE.key() | {"name": "score", "x": 38, "y": 18, "w": 74, "h": 60, "every": 3}]
    assert all("upscale" not in r for r in preset["regions"] if r["kind"] != "score")


@pytest.fixture
def home(isolated_env: Path) -> Path:
    return isolated_env


def _discover_existing(tmp_path: Path, monkeypatch, data: dict[str, Any]) -> dict[str, Any]:
    write_preset(paths.presets_dir(), data)
    monkeypatch.setattr(discover, "_ocr_frames", lambda *_a, **_k: _frames())
    monkeypatch.setattr(discover, "probe", lambda _p: None)
    path = tmp_path / "rec.mp4"
    src = SimpleNamespace(path=path, io_path=path, name="rec.mp4", duration=2400.0)
    out = pipeline._discover(tmp_path / "proj", [src], OcrParams(game="verbfeed"), True, 40, "user")
    return out["verbfeed"]


def test_an_existing_preset_is_suggested_the_score_region(
    home: Path, tmp_path: Path, monkeypatch
) -> None:
    rec = _discover_existing(tmp_path, monkeypatch, verbfeed())
    assert rec["installed"] is False
    assert [r["name"] for r in rec["suggested_regions"] if r["kind"] == "score"] == ["score"]
    preset = json.loads((paths.presets_dir() / "verbfeed.json").read_text(encoding="utf-8"))
    assert not [r for r in preset["regions"] if r["kind"] == "score"]  # never written into it


def test_a_preset_with_a_score_region_gets_no_suggestion(
    home: Path, tmp_path: Path, monkeypatch
) -> None:
    data = verbfeed()
    data["regions"] = [*data["regions"], SCORE.key()]
    rec = _discover_existing(tmp_path, monkeypatch, data)
    assert not [r for r in rec["suggested_regions"] if r["kind"] == "score"]


# --------------------------------------------------------------------------- event type


def _json_event(t: float, text: str = "4 > 3") -> dict[str, Any]:
    return {
        "source": "rec.mp4",
        "start": t,
        "end": t + 2,
        "kind": "ocr",
        "type": "score",
        "label": text,
        "confidence": 0.9,
        "details": {"before": 4, "after": 3, "region": "score", "slot": 135},
    }


def test_score_changes_type() -> None:
    types = eventtypes.DEFAULT_EVENT_TYPES
    t = eventtypes.first_match(types, _json_event(10.0))
    assert t is not None and t.name == "score_changes"
    assert (t.score, t.max, t.payoff, t.seed, t.seed_gap_s) == (6.0, 3, True, 1.3, 60.0)
    names = [x.name for x in types]
    assert names.index("score_changes") < names.index("announcements")
    flat = {"kind": "ocr", "type": "score", "t": 10.0, "text": "4 > 3"}
    assert vlm_windows._seed_priority(flat, types) == 1.3


def test_event_boost_counts_a_kill_each_up_to_three() -> None:
    events = [_json_event(t) for t in (2.0, 8.0, 14.0, 20.0)]
    boost, evidence = scoring.event_boost({"start": 0.0, "end": 30.0}, events, style="gaming")
    assert boost == 18.0 and evidence["score_changes"] == 4
    _, evidence = scoring.event_boost({"start": 0.0, "end": 30.0}, [], style="gaming")
    assert evidence["score_changes"] == 0


def test_seeds_are_a_minute_apart() -> None:
    events = [_json_event(t) for t in (100.0, 130.0, 170.0, 240.0)]
    got = scoring.seeds(events, eventtypes.DEFAULT_EVENT_TYPES, ("gaming",))
    assert [s.start for s in got if s.name == "score_changes"] == [100.0, 170.0, 240.0]


def test_a_score_change_is_a_tail_payoff(tmp_path: Path) -> None:
    (tmp_path / "events.json").write_text(json.dumps({"events": [_json_event(5.0)]}))
    loaded = tails.load_events(tmp_path)
    assert loaded is not None
    e = next(iter(loaded.values()))[0]
    assert e["_payoff"] is True and tails.is_payoff(e, tails.EventTailSettings())


def test_vlm_hint_names_the_score() -> None:
    ev = {"kind": "ocr", "type": "score", "t": 12.0, "text": "4 > 3"}
    assert vlm_windows._hints([ev], 10.0, 20.0) == "2.0s on-screen score: 4 > 3"


def test_raw_key_marks_only_score_regions_off_the_stills() -> None:
    plain = [Region("a", 0, 0, 100, 50, "feed"), Region("b", 0, 60, 100, 50, "notice")]
    args = (2.0, "bottom", 0.5, [])
    base = ocr_raw.raw_key(args[0], [r.key() for r in plain], *args[1:])
    assert ocr_raw.raw_key(args[0], killfeed._raw_region_keys(plain), *args[1:]) == base
    with_score = [*plain, Region("c", 0, 120, 40, 30, "score")]
    keys = killfeed._raw_region_keys(with_score)
    assert [k.get("still") for k in keys] == [None, None, False]
    assert ocr_raw.raw_key(args[0], keys, *args[1:]) != ocr_raw.raw_key(
        args[0], [r.key() for r in with_score], *args[1:]
    )
