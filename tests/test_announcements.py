"""Learned announcements: big-text lines that come back as short pop-ups all through a source
(invented lines and names only)."""

from __future__ import annotations

from typing import Any

import pytest

from cutpoint.events import eventtypes, scoring, vlm_windows
from cutpoint.events.killfeed import (
    GENERIC_VERSION,
    FrameRec,
    Interpreter,
    OcrInterpretation,
    OcrParams,
    learn_announcements,
)
from cutpoint.events.regions import Region, preset_from_dict

FEED = Region("feed", 1500, 0, 400, 400, "feed")
NOTICE = Region("notice", 600, 700, 700, 100, "notice")
BIG = Region("big", 300, 110, 1320, 700, "bigtext")
REGIONS = (FEED, NOTICE, BIG)
EVERY_4_MIN = [60.0 + 240 * k for k in range(12)]  # 12 times over 48 min, in 5 stretches


def _line(
    t: float,
    text: str = "SIGNAL RELAYED",
    length: float = 3.0,
    etype: str = "bigtext",
    region: str = "big",
    **extra: Any,
) -> dict[str, Any]:
    """A finished OCR event as the interpretation leaves it (layer 1)."""
    return {
        "t": t,
        "end": t + length,
        "type": etype,
        "text": text,
        "confidence": 0.9,
        "sightings": 3,
        "region": region,
        "unknown": True,
    } | extra


def _flagged(events: list[dict[str, Any]]) -> list[float]:
    return sorted(e["t"] for e in events if e.get("announce"))


# --------------------------------------------------------------------------- the gate


def test_a_recurring_pop_up_is_learned() -> None:
    events = learn_announcements([_line(t) for t in EVERY_4_MIN])
    assert _flagged(events) == EVERY_4_MIN
    assert events[0]["announce_words"] == ["SIGNAL", "RELAYED"]
    assert events[0]["type"] == "bigtext"  # the OCR type stays


def test_a_one_word_line_is_learned() -> None:
    out = learn_announcements([_line(t, "BREACH") for t in EVERY_4_MIN])
    assert _flagged(out) == EVERY_4_MIN
    assert out[0]["announce_words"] == ["BREACH"]


def test_a_recurring_notice_sized_prompt_is_not_learned() -> None:
    """Only big text is an announcement: an item or action prompt in a notice-sized line passes
    every other gate (count, short, spread, gaps) but is never learned nor counted."""
    prompts = [_line(t, "GRAB LANTERN", etype="notice", region="notice") for t in EVERY_4_MIN]
    assert _flagged(learn_announcements(prompts)) == []
    # the same reads as big text are learned
    assert _flagged(learn_announcements([_line(t, "GRAB LANTERN") for t in EVERY_4_MIN])) == (
        EVERY_4_MIN
    )
    # notice reads add no appearances to a big-text key: 9 big-text ones stay under the gate
    mixed = [_line(t, "GRAB LANTERN") for t in EVERY_4_MIN[:9]]
    mixed += [_line(t, "GRAB LANTERN", etype="notice", region="notice") for t in EVERY_4_MIN[9:]]
    assert _flagged(learn_announcements(mixed)) == []


@pytest.mark.parametrize(
    "times",
    [
        [60.0 + 100 * k for k in range(12)],  # 20 minutes: 2 stretches only
        EVERY_4_MIN[:9],  # 9 times
        [60.0 + 20 * k for k in range(12)] + [700.0, 1300.0, 1900.0],  # median gap 20 s
    ],
)
def test_too_few_too_close_or_too_short_a_source_is_not_learned(times: list[float]) -> None:
    assert _flagged(learn_announcements([_line(t) for t in times])) == []


def test_a_line_that_stays_is_a_label_not_a_pop_up() -> None:
    events = [_line(t, "EAST HALL", length=45.0) for t in EVERY_4_MIN]
    assert _flagged(learn_announcements(events)) == []


def test_feed_rows_banners_and_known_lines_are_never_learned() -> None:
    feed_rows = [_line(t, etype="notice", region="feed") for t in EVERY_4_MIN]
    banners = [_line(t, etype="banner") | {"unknown": False} for t in EVERY_4_MIN]
    known = [_line(t, "TOWER CLAIMED") | {"unknown": False} for t in EVERY_4_MIN]
    assert _flagged(learn_announcements([*feed_rows, *banners, *known])) == []


def test_one_pop_up_read_in_two_regions_is_one_appearance() -> None:
    events = [_line(t) for t in EVERY_4_MIN]
    events += [_line(t + 2, region="banner") for t in EVERY_4_MIN]
    out = learn_announcements(events)
    assert _flagged(out) == EVERY_4_MIN  # the earlier read of each, not the second region's


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ("SIGNAL RELAYED ORCHIDFOX", [1100.0]),  # a name stays on the screen: the first only
        ("SIGNAL RELAYED", [1100.0, 1120.0, 1140.0]),  # the same line alone: no name, all kept
    ],
)
def test_text_held_on_a_screen_keeps_only_its_first(text: str, kept: list[float]) -> None:
    base = [_line(t) for t in EVERY_4_MIN]
    held = [_line(t, text) for t in (1100.0, 1120.0, 1140.0)]
    assert _flagged(learn_announcements(base + held)) == sorted([*EVERY_4_MIN, *kept])


@pytest.mark.parametrize("text", ["SIGNAL RELAYED", "YOU SCORED", "RED TEAM SCORED"])
def test_a_nameless_multi_word_pop_up_within_a_minute_stays_flagged(text: str) -> None:
    """The announcement's other words, short ones too, are no name beside it."""
    times = sorted([*EVERY_4_MIN, 1100.0, 1120.0, 1140.0])
    flagged = _flagged(learn_announcements([_line(t, text) for t in times]))
    assert flagged == times


def test_a_pop_up_with_another_name_each_time_is_not_held() -> None:
    base = [_line(t) for t in EVERY_4_MIN]
    names = ("ORCHIDFOX", "BLUEWREN", "MAPLETON")
    named = [
        _line(t, f"SIGNAL RELAYED {n}")
        for t, n in zip((1100.0, 1120.0, 1140.0), names, strict=True)
    ]
    flagged = _flagged(learn_announcements(base + named))
    assert flagged == sorted([*EVERY_4_MIN, 1100.0, 1120.0, 1140.0])


def test_digit_lines_are_learned_by_their_shape_and_others_dropped() -> None:
    extra = [_line(t, f"{k % 5 + 1} VS 5") for k, t in enumerate(EVERY_4_MIN)]
    extra.append(_line(500.0, "12 34"))  # once: not learned, not kept
    out = learn_announcements([_line(30.0, "PLAIN WORDS")], extra)
    assert [e["text"] for e in out][:2] == ["PLAIN WORDS", "1 VS 5"]
    assert len(out) == 13 and all(e["announce_words"] == ["# VS #"] for e in out[1:])


@pytest.mark.parametrize("text", ["\u670810", "X 7", "x10"])
def test_one_letter_beside_digits_is_not_learned(text: str) -> None:
    assert learn_announcements([], [_line(t, text) for t in EVERY_4_MIN]) == []


def test_a_two_letter_cjk_word_with_digits_is_learned() -> None:
    out = learn_announcements([], [_line(t, "\u52dd\u5229 3") for t in EVERY_4_MIN])
    assert len(out) == 12 and all(e["announce_words"] == ["\u52dd\u5229 #"] for e in out)


def test_bare_numbers_are_never_learned() -> None:
    points = [_line(t, f"+{100 * (k % 3 + 1)}") for k, t in enumerate(EVERY_4_MIN)]
    clocks = [_line(t + 100, f"{k % 3 + 1} - {k % 4}") for k, t in enumerate(EVERY_4_MIN)]
    sides = [_line(t + 30, f"{k % 4 + 1} VS 5") for k, t in enumerate(EVERY_4_MIN)]
    out = learn_announcements([], [*points, *clocks, *sides])
    assert [e["text"] for e in out] == [e["text"] for e in sides]
    assert all(e["announce_words"] == ["# VS #"] for e in out)


# --------------------------------------------------------------------------- interpretation


def _interpret(frames: dict[float, dict[str, list[str]]]) -> list[dict[str, Any]]:
    """Events of invented OCR reads at 1 frame a second: ``frames`` maps a time to the lines
    each region reads then."""
    data = {"game": "tilefall", "regions": []}
    interp = OcrInterpretation(
        "rec.mp4",
        OcrParams(game="tilefall", fps=1.0),
        preset_from_dict(data, None),
        REGIONS,
        (1920, 1080),
        {},
    )
    for n in range(int(max(frames)) + 1):
        recs = {}
        for r in REGIONS:
            lines = frames.get(float(n), {}).get(r.name, [])
            box = [[0, 0], [300, 0], [300, 60], [0, 60]]
            recs[r.name] = {"g": "read", "items": [(box, line, 0.95) for line in lines]}
        interp.add(FrameRec(n=n, t=float(n), regions=recs))
    return interp.finish()


def test_digit_lines_reach_only_the_announcement_learner() -> None:
    interp = Interpreter(OcrParams(game="tilefall", fps=1.0), 1.0)
    box = [[0, 0], [300, 0], [300, 60], [0, 60]]
    interp.handle(5.0, BIG, None, [(box, "3 VS 4", 0.9)])
    interp.handle(6.0, BIG, None, [(box, "-- //", 0.9)])  # no digit: dropped as before
    interp.handle(7.0, NOTICE, None, [(box, "2 VS 4", 0.9)])  # a notice: dropped as before
    assert interp.tracker.events == []  # the kill, feed and menu rules never see them
    assert [e["text"] for e in interp.digits.events] == ["3 VS 4"]


def test_digit_pop_ups_through_the_interpretation() -> None:
    frames: dict[float, dict[str, list[str]]] = {}
    for k, t in enumerate(EVERY_4_MIN):
        for dt in (0, 1):
            frames[t + dt] = {"big": [f"{k % 4 + 1} VS 5"]}
    frames |= {900.0: {"big": ["77"]}, 901.0: {"big": ["77"]}}  # once: no event
    events = _interpret(frames)
    assert [e["t"] for e in events] == EVERY_4_MIN
    assert all(e["announce"] and e["type"] == "bigtext" and e["kind"] == "ocr" for e in events)


def test_notice_digit_pop_ups_leave_no_event() -> None:
    frames: dict[float, dict[str, list[str]]] = {}
    for k, t in enumerate(EVERY_4_MIN):
        for dt in (0, 1):
            frames[t + dt] = {"notice": [f"{k % 4 + 1} VS 5"]}
    assert _interpret(frames) == []


def test_without_learning_the_digit_lines_leave_no_event() -> None:
    frames = {t: {"big": ["2 VS 5"]} for t in (10.0, 11.0, 300.0, 301.0)}
    assert _interpret(frames) == []


def test_generic_version_is_in_the_cache_key() -> None:
    preset = preset_from_dict({"game": "tilefall", "regions": []}, None)
    assert GENERIC_VERSION == 24
    assert OcrParams().key("tilefall", preset)["preset"]["interp"] == 24


# --------------------------------------------------------------------------- scoring


def _json_event(start: float, words: int = 1, conf: float = 0.9, **details: Any) -> dict:
    """An events.json entry of a flagged big-text line."""
    return {
        "source": "rec.mp4",
        "start": start,
        "end": start + 2,
        "kind": "ocr",
        "type": "bigtext",
        "label": "SIGNAL RELAYED",
        "confidence": conf,
        "details": {"announce": True, "announce_words": ["SIGNAL"] * words} | details,
    }


def test_first_match_and_a_preset_type_of_the_same_name() -> None:
    types = eventtypes.DEFAULT_EVENT_TYPES
    assert eventtypes.first_match(types, _json_event(10.0)).name == "announcements"
    flat = {"kind": "ocr", "type": "bigtext", "text": "SIGNAL RELAYED", "announce": True}
    assert eventtypes.first_match(types, flat).name == "announcements"
    assert eventtypes.first_match(types, flat | {"announce": False}) is None
    assert eventtypes.first_match(types, flat | {"type": "notice"}) is None
    preset = preset_from_dict(
        {
            "game": "tilefall",
            "regions": [],
            "events": [{"name": "announcements", "when": [{"type": "notice"}], "score": 1}],
        },
        None,
    )
    mine = [t for t in eventtypes.types_for(preset) if t.name == "announcements"]
    assert (
        len(mine) == 1 and mine[0].score == 1.0 and mine[0].seed_gap_s is None and mine[0].nominate
    )


def test_event_boost_counts_half_a_kill_up_to_two() -> None:
    cand = {"start": 0.0, "end": 30.0}
    events = [_json_event(t) for t in (5.0, 10.0, 15.0)]
    boost, evidence = scoring.event_boost(cand, events, style="gaming")
    assert boost == 6.0 and evidence["announcements"] == 3


def test_announcements_key_is_always_in_the_evidence() -> None:
    _, evidence = scoring.event_boost({"start": 0.0, "end": 30.0}, [], style="gaming")
    assert evidence["announcements"] == 0


def test_announcements_are_boost_only_and_seed_no_moment() -> None:
    types = eventtypes.DEFAULT_EVENT_TYPES
    (mine,) = [t for t in types if t.name == "announcements"]
    assert not mine.nominate and mine.score
    events = [_json_event(t, words=2) for t in (100.0, 130.0, 280.0)]
    assert scoring.seeds(events, types, ("gaming",)) == []
    kill = {
        "source": "rec.mp4",
        "start": 101.0,
        "end": 105.0,
        "kind": "ocr",
        "type": "kill",
        "label": "Alderwick > Boxer77",
        "confidence": 0.9,
        "details": {"player_kill": True},
    }
    got = scoring.seeds([*events, kill], types, ("gaming",))
    assert [s.name for s in got] == ["player_kills"]


def test_announcement_still_adds_points_to_an_overlapping_candidate() -> None:
    cand = {"start": 90.0, "end": 120.0}
    boost, evidence = scoring.event_boost(cand, [_json_event(100.0)], style="gaming")
    assert boost == 3.0 and evidence["announcements"] == 1
    far, _ = scoring.event_boost(cand, [_json_event(300.0)], style="gaming")
    assert far == 0.0


# --------------------------------------------------------------------------- VLM


def test_no_vlm_seed_for_a_flagged_big_text_line() -> None:
    ev = {
        "kind": "ocr",
        "type": "bigtext",
        "t": 12.0,
        "text": "SIGNAL RELAYED",
        "announce": True,
    }
    types = eventtypes.DEFAULT_EVENT_TYPES
    assert vlm_windows._seed_priority(ev, types) is None  # boost-only: no window of its own
    assert vlm_windows._seed_priority(ev | {"type": "notice"}, types) is None


# a custom type with a seed gap (the spacing is generic, announcements no longer use it)
SPACED = (
    eventtypes.EventType(
        "relay", (eventtypes.Matcher("bigtext"),), score=3.0, seed=0.6, seed_gap_s=60.0
    ),
)


def test_seeds_are_a_minute_apart_the_best_first() -> None:
    events = [
        _json_event(100.0, conf=0.9),
        _json_event(130.0, words=2, conf=0.8),  # the best: two learned words
        _json_event(150.0, conf=0.95),
        _json_event(220.0, conf=0.7),  # 90 s from 130, 60 s from 280
        _json_event(280.0, conf=0.9),
    ]
    got = scoring.seeds(events, SPACED, ("gaming",))
    assert [s.start for s in got] == [130.0, 220.0, 280.0]


def test_seeds_a_candidate_covers_take_no_room() -> None:
    events = [_json_event(100.0, words=2), _json_event(150.0)]
    got = scoring.seeds(events, SPACED, ("gaming",), covered=lambda s: s.start < 120)
    assert [s.start for s in got] == [100.0, 150.0]
    assert [s.start for s in scoring.seeds(events, SPACED, ("gaming",))] == [100.0]
