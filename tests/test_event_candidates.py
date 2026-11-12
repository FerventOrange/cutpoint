"""Candidates nominated by events, and the two-tier floor of the picked ones.

Synthetic signals and events only: grids are built in memory, the caches (for the events
path) are written by hand for a stand-in source file, and the HUD preset is made up.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import threading
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cutpoint import analyze, cache, default_fit, paths, planning, scorer
from cutpoint.assistant import context
from cutpoint.events import scoring
from cutpoint.events.eventtypes import DEFAULT_EVENT_TYPES, EventType, Matcher
from cutpoint.sources import TRANSCRIPT_MODELS
from cutpoint.styles import STYLES, get_style
from cutpoint.telemetry import DT
from cutpoint.transcribe import transcript_name
from cutpoint.ui import moments_view

SIGNALS = ("loudness", "speech", "laughter", "scene", "continuity")


# --------------------------------------------------------------------------- helpers
def bump(t: float, n: int, height: float = 1.0, width: float = 1.0) -> np.ndarray:
    x = np.arange(n) * DT
    return height * np.exp(-(((x - t) / width) ** 2))


def part_of(sig: dict[str, np.ndarray], n: int, src: str = "a.mp4") -> analyze.SourceResult:
    ps = analyze.PartSignals(n, sig, [], [], [], [], [])
    return analyze.SourceResult(Path(src), None, 0.0, n / 10, [], [], None, ps)


def loud_picks(
    style_name: str, heights: list[float], max_candidates: int | None = None
) -> list[analyze.Candidate]:
    """Candidates from loudness bumps of ``heights``, 60 s apart (wider than any clip)."""
    n = int(60 * (len(heights) + 1) / DT)
    sig = {k: np.zeros(n) for k in SIGNALS}
    for i, h in enumerate(heights):
        sig["loudness"] += bump(30.0 + 60.0 * i, n, h)
    return analyze.pick_candidates(part_of(sig, n), get_style(style_name), max_candidates, {})


def spans(cands: list[analyze.Candidate]) -> list[tuple[float, float]]:
    return [(c.start, c.end) for c in cands]


def picks_at(
    style_name: str, peaks: list[tuple[float, float]], max_candidates: int | None = None
) -> list[analyze.Candidate]:
    """Candidates from loudness bumps ``(second, height)``."""
    n = int((max(t for t, _ in peaks) + 30) / DT) + 1
    sig = {k: np.zeros(n) for k in SIGNALS}
    for t, h in peaks:
        sig["loudness"] += bump(t, n, h)
    return analyze.pick_candidates(part_of(sig, n), get_style(style_name), max_candidates, {})


def within(seconds: float, fn, *args):
    """``fn(*args)``, failing the test (instead of hanging it) if it takes longer."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["result"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001 - handed to the test thread
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        pytest.fail(f"still running after {seconds} s")
    if "error" in box:
        raise box["error"]
    return box["result"]


def overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] - 1e-6 and b[0] < a[1] - 1e-6


# --------------------------------------------------------------------------- usual_count, cap
def test_usual_count_grows_with_a_long_target():
    assert analyze.usual_count(get_style("gaming")) > 30  # 90 s target, 8 s clips


def test_only_the_gaming_style_has_an_open_list():
    assert {n for n, s in STYLES.items() if s.open_list} == {"gaming"}


def test_candidate_cap_is_open_only_for_an_open_list_style():
    gaming, action = get_style("gaming"), get_style("action")
    assert analyze.candidate_cap(gaming) == sys.maxsize
    assert analyze.candidate_cap(action) == analyze.candidate_cap(action, None)
    assert analyze.candidate_cap(action) == analyze.usual_count(action)
    assert analyze.candidate_cap(gaming, 7) == analyze.candidate_cap(action, 7) == 7


# --------------------------------------------------------------------------- the two-tier floor
def heights_with_tail(usual: int) -> list[float]:
    """``usual`` strong peaks, then 3 at 0.6 (above the long-list floor), 3 at 0.4 (between)."""
    return [1.0 - 0.1 * i / usual for i in range(usual)] + [0.6] * 3 + [0.4] * 3


def test_default_keeps_the_weak_peaks_of_a_short_source():
    # fewer peaks than the usual count: the 0.25 floor alone, as before the long list
    got = loud_picks("gaming", [1.0, 0.6, 0.4, 0.3, 0.2])
    assert len(got) == 4  # 0.2 is under the quarter floor, 0.3 and 0.4 are kept


def test_an_open_list_applies_the_long_list_floor_only_past_the_usual_count():
    usual = analyze.usual_count(get_style("gaming"))
    got = loud_picks("gaming", heights_with_tail(usual))
    assert len(got) == usual + 3  # the 0.6 peaks follow, the 0.4 ones do not


@pytest.mark.parametrize(
    "name", sorted(n for n, s in STYLES.items() if s.kind == "generic" and not s.open_list)
)
def test_every_other_generic_style_gives_the_usual_count_exactly(name):
    usual = analyze.usual_count(get_style(name))
    heights = heights_with_tail(usual)
    default = loud_picks(name, heights)
    assert len(default) == usual  # capped, the tail is not listed
    assert spans(default) == spans(loud_picks(name, heights, usual))
    # and a short source keeps the quarter floor, as it always did
    assert len(loud_picks(name, [1.0, 0.6, 0.4, 0.3, 0.2])) == 4


@pytest.mark.parametrize("gap", [5.5, 9.0, 20.0])
def test_a_longer_list_never_changes_the_top_of_a_shorter_one(gap):
    # The tail peak sits within merge_gap of (or inside the roll of) the first top pick: the
    # head is finished as the usual-count run finishes it, the tail only fills what is free.
    style = get_style("gaming")
    usual = analyze.usual_count(style)
    peaks = [(30.0 + 60.0 * i, 1.0 - 0.1 * i / usual) for i in range(usual)]
    peaks.append((30.0 + gap, 0.6))
    default = picks_at("gaming", peaks)
    explicit = picks_at("gaming", peaks, usual)
    assert len(explicit) == usual
    assert spans(default[:usual]) == spans(explicit)
    for tail in default[usual:]:
        assert tail.end - tail.start >= style.min_len - 1e-6
        assert not any(overlap((tail.start, tail.end), h) for h in spans(default[:usual]))
    assert len(default) == usual + 1


def test_an_explicit_count_keeps_the_quarter_floor_throughout():
    usual = analyze.usual_count(get_style("gaming"))
    heights = heights_with_tail(usual)
    got = loud_picks("gaming", heights, usual + 6)
    assert len(got) == usual + 6  # the 0.4 peaks too
    assert len(loud_picks("gaming", heights, usual + 2)) == usual + 2  # a cap is a cap


def test_a_default_cap_source_whose_last_sample_is_a_peak_ends():
    # the pick loop zeroes a window; one that ends at the last sample once left the peak itself
    n = int(120 / DT) + 1
    sig = {k: np.zeros(n) for k in SIGNALS}
    sig["loudness"] += bump(30.0, n, 0.8)
    sig["loudness"][-1] = 1.0
    got = within(20, analyze.pick_candidates, part_of(sig, n), get_style("gaming"), None, {})
    assert got
    assert max(c.end for c in got) == pytest.approx((n - 1) * DT, abs=1.0)
    assert len(got) <= analyze.usual_count(get_style("gaming"))


# --------------------------------------------------------------------------- describe_candidates
def test_describe_candidates_scores_and_moves_to_source_time():
    words = [
        {"word": w, "start": 11.0 + 0.4 * i, "end": 11.3 + 0.4 * i, "probability": 0.95}
        for i, w in enumerate(["what", "a", "shot"])
    ]
    n = 300
    ps = analyze.PartSignals(n, {k: np.zeros(n) for k in SIGNALS}, [], [], words, [], [])
    c = analyze.Candidate(Path("a.mp4"), 10.0, 18.0, 12.0, 0.5, dict.fromkeys(SIGNALS, 0.0))
    (out,) = analyze.describe_candidates([c], get_style("gaming"), ps, 100.0)
    assert (out.start, out.end, out.peak_t) == (110.0, 118.0, 112.0)
    assert out.transcript == "what a shot"
    assert out.score == scorer.score(
        0.5, get_style("gaming"), scorer.available_for(get_style("gaming")), False
    )


# --------------------------------------------------------------------------- events
SRC = "game.mp4"


def feed(t: float, **flags: bool) -> dict[str, Any]:
    return {
        "source": SRC, "start": t, "end": t + 3, "kind": "ocr", "type": "kill",
        "label": "Ash > Bob", "player": True, "player_kill": True, "player_death": False,
        "teamkill": False, **flags,
    }  # fmt: skip


def banner(t: float, text: str = "VICTORY", kind: str = "banner") -> dict[str, Any]:
    return {"source": SRC, "start": t, "end": t + 2, "kind": "ocr", "type": kind, "label": text}


def sound(t: float, label: str = "Laughter", confidence: float = 0.4) -> dict[str, Any]:
    return {
        "source": SRC, "start": t, "end": t + 1.5, "kind": "audio", "label": label,
        "confidence": confidence,
    }  # fmt: skip


STYLE = ("gaming",)
PLACER = EventType(
    "placer", (Matcher("banner", prefix=("BOX PLACED",)),), score=6.0, max=2, tail_s=3.0
)
LAUGH_ONLY = EventType("hype", (Matcher("banner", prefix=("HYPE",)),), score=4.0, styles=("funny",))
NO_POINTS = EventType("flavour", (Matcher("banner", prefix=("FLAVOUR",)),))


@pytest.mark.parametrize(
    ("event", "types", "style", "expected"),
    [
        (feed(10), DEFAULT_EVENT_TYPES, STYLE, ["player_kills"]),
        (feed(10, teamkill=True), DEFAULT_EVENT_TYPES, STYLE, ["teamkills"]),
        (feed(10, player_death=True), DEFAULT_EVENT_TYPES, STYLE, ["self_kills"]),
        (banner(10), DEFAULT_EVENT_TYPES, STYLE, ["banners"]),
        (banner(10, "BOX PLACED 2"), (PLACER, *DEFAULT_EVENT_TYPES), STYLE, ["placer"]),
        (banner(10, "BOX PLACED 2"), DEFAULT_EVENT_TYPES, STYLE, []),
        (sound(10), DEFAULT_EVENT_TYPES, STYLE, ["laughter"]),
        (sound(10, "Scream"), DEFAULT_EVENT_TYPES, STYLE, ["scream"]),
        (sound(10, "Cheering"), DEFAULT_EVENT_TYPES, STYLE, ["cheering"]),
        # nothing that scores: a death (reaction points only), bigtext, notice, other sounds,
        # a sound without confidence, a type without points, one that is not for this style
        (feed(10, player_kill=False, player_death=True), DEFAULT_EVENT_TYPES, STYLE, []),
        (banner(10, "ROUND 3", "bigtext"), DEFAULT_EVENT_TYPES, STYLE, []),
        (banner(10, "VICTORY", "notice"), DEFAULT_EVENT_TYPES, STYLE, []),
        (sound(10, "Speech"), DEFAULT_EVENT_TYPES, STYLE, []),
        (sound(10, confidence=0.0), DEFAULT_EVENT_TYPES, STYLE, []),
        (banner(10, "FLAVOUR"), (NO_POINTS, *DEFAULT_EVENT_TYPES), STYLE, []),
        (banner(10, "HYPE"), (LAUGH_ONLY, *DEFAULT_EVENT_TYPES), STYLE, []),
        (banner(10, "HYPE"), (LAUGH_ONLY, *DEFAULT_EVENT_TYPES), ("funny",), ["hype"]),
    ],
)
def test_seeds_are_the_events_that_can_score(event, types, style, expected):
    assert [s.name for s in scoring.seeds([event], types, style)] == expected


def test_seeds_hold_and_tail():
    long_banner = {**banner(10, "BOX PLACED"), "end": 30.0}
    by_name = {
        s.name: s
        for s in scoring.seeds([long_banner, sound(50)], (PLACER, *DEFAULT_EVENT_TYPES), STYLE)
    }
    assert by_name["placer"].hold == 1.0 and by_name["placer"].tail_s == 3.0  # OCR_HOLD_S cap
    assert by_name["laughter"].hold == 1.5 and by_name["laughter"].tail_s is None
    assert (by_name["laughter"].start, by_name["laughter"].end) == (50, 51.5)


def test_seeds_skip_events_that_start_in_menu_time():
    events = [feed(10), feed(60), feed(100)]
    got = scoring.seeds(events, DEFAULT_EVENT_TYPES, STYLE, [[50.0, 70.0]])
    assert [s.start for s in got] == [10, 100]


@pytest.mark.parametrize(
    ("c", "expected"),
    [
        ({"signals": {"events": {"boost": 6.0, "nominated": ["player_kills"]}}}, True),
        ({"signals": {"events": {"boost": 6.0}}}, False),
        ({"signals": {"events": {"nominated": []}}}, False),
        ({"signals": {"events": None}}, False),
        ({"signals": {}}, False),
        ({}, False),
    ],
)
def test_is_nominated(c, expected):
    assert scoring.is_nominated(c) is expected


# --------------------------------------------------------------------------- nominate
N = 6000  # a 600 s part


def grid(src: str = SRC, n: int = N, offset: float = 0.0, index: int = 0) -> analyze.GridPart:
    ps = analyze.PartSignals(n, {k: np.zeros(n) for k in SIGNALS}, [], [], [], [], [])
    return analyze.GridPart(Path(src), index, offset, n * DT, ps)


def own(cid: str, start: float, score: float = 50.0, end: float | None = None) -> dict[str, Any]:
    return {
        "id": cid, "source": SRC, "source_index": 0, "start": start,
        "end": start + 8 if end is None else end, "duration": 8, "score": score,
        "signals": {"keywords": [], "vision": None}, "label": "gaming-moment", "transcript": "",
    }  # fmt: skip


def highlights(cands: list[dict[str, Any]], style: str = "gaming", **settings) -> dict[str, Any]:
    return {"style": style, "settings": settings, "candidates": cands, "suggested_top": 1}


def events_of(events: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"events": events, "windows": [], "sources": [SRC], **extra}


@pytest.fixture(autouse=True)
def no_transcript(monkeypatch):
    monkeypatch.setattr(scoring, "_segments", lambda src, language=None: [])


def nominees(hl: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in hl["candidates"] if scoring.is_nominated(c)]


def test_nominate_adds_a_candidate_for_an_event_nothing_covers(tmp_path):
    hl = highlights([own("h001", 20.0, 60.0), own("h002", 300.0, 40.0)])
    n = scoring.nominate(hl, events_of([feed(100.0)]), tmp_path, [grid()])
    assert n == 1
    (nom,) = nominees(hl)
    assert "id" not in nom  # numbered by number_candidates, after the rescore
    assert nom["signals"]["events"]["nominated"] == ["player_kills"]
    assert nom["start"] < 100.0 < nom["end"]
    assert nom["label"] == "player-kills"
    # the grids say nothing here: its analysis score is the floor, 0.6 of the best other (60)
    assert scoring.NOMINEE_FLOOR == 0.6
    assert nom["score"] == pytest.approx(36.0)
    assert nom["grid_score"] == 0.0  # what the grids gave, kept for ties


def test_an_event_inside_a_candidate_nominates_nothing(tmp_path):
    hl = highlights([own("h001", 90.0, end=100.0)])
    # inside the window, and starting where it ends
    events = events_of([feed(95.0), feed(100.0, player_kill=True)])
    assert scoring.nominate(hl, events, tmp_path, [grid()]) == 0
    assert len(hl["candidates"]) == 1


def test_an_event_after_a_clip_gets_a_window_of_its_own_after_it(tmp_path):
    # LOOKAHEAD_S makes the clip's boost count it, but the clip does not cover it
    taken = own("h001", 90.0, end=100.0)
    hl = highlights([taken])
    t = 100.0 + scoring.LOOKAHEAD_S - 0.5
    assert scoring.nominate(hl, events_of([feed(t)]), tmp_path, [grid()]) == 1
    (nom,) = nominees(hl)
    assert nom["start"] >= taken["end"] - 1e-6
    assert nom["start"] <= t <= nom["end"]
    assert hl["candidates"][0] is taken


def test_a_burst_of_events_is_one_candidate(tmp_path):
    hl = highlights([own("h001", 20.0)])
    burst = [feed(100.0), feed(102.0), feed(104.0)]
    scoring.nominate(hl, events_of(burst), tmp_path, [grid()])
    (nom,) = nominees(hl)
    assert nom["start"] < 100.0 and nom["end"] > 105.0


def test_nominees_keep_the_style_geometry_and_stay_out_of_the_others(tmp_path):
    style = get_style("gaming")
    taken = [own("h001", 96.0, end=104.0), own("h002", 190.0, end=200.0)]
    hl = highlights(list(taken))  # nominate extends the list in place
    # 110.5 is within LOOKAHEAD_S of h001's end (covered); 194.0 is inside h002
    events = [feed(t) for t in (30.0, 110.5, 150.0, 151.0, 152.0, 153.0, 194.0, 400.0, 401.5)]
    scoring.nominate(hl, events_of(events), tmp_path, [grid()])
    noms = sorted(nominees(hl), key=lambda c: c["start"])
    assert len(noms) >= 3
    for c in noms:
        assert style.min_len - 1e-6 <= c["end"] - c["start"] <= style.max_len + 1e-6
        assert c["duration"] == pytest.approx(c["end"] - c["start"], abs=1e-3)
        for o in taken:
            assert c["end"] <= o["start"] + 1e-6 or c["start"] >= o["end"] - 1e-6
    for a, b in pairwise(noms):
        assert a["end"] <= b["start"] + 1e-6


def test_a_type_that_cannot_score_nominates_nothing(tmp_path):
    hl = highlights([own("h001", 20.0)])
    events = [
        banner(100.0, "ROUND 3", "bigtext"),
        banner(200.0, "OBJECTIVE", "notice"),
        feed(300.0, player_kill=False, player_death=True),
    ]
    assert scoring.nominate(hl, events_of(events), tmp_path, [grid()]) == 0
    assert [c["id"] for c in hl["candidates"]] == ["h001"]


def test_every_scoring_type_nominates_by_type_not_by_kills(tmp_path):
    hl = highlights([own("h001", 20.0)])
    events = [
        banner(100.0),
        sound(200.0, "Laughter"),
        sound(300.0, "Scream"),
        sound(400.0, "Cheering"),
    ]
    assert scoring.nominate(hl, events_of(events), tmp_path, [grid()]) == 4
    got = sorted((c["start"], c["signals"]["events"]["nominated"]) for c in nominees(hl))
    assert [g[1] for g in got] == [["banners"], ["laughter"], ["scream"], ["cheering"]]


def test_a_preset_defined_type_nominates_in_a_game_with_no_kills(tmp_path):
    (tmp_path / "hud_presets").mkdir()
    preset = {
        "game": "foo", "regions": [], "banner_phrases": [],
        "events": [{"name": "placer", "when": [{"type": "banner", "prefix": ["BOX PLACED"]}],
                    "score": 6, "max": 2, "tail_s": 3}],
    }  # fmt: skip
    (tmp_path / "hud_presets" / "foo.json").write_text(json.dumps(preset), encoding="utf-8")
    ev = events_of([banner(100.0, "BOX PLACED 2")], settings={"ocr": {"game": "foo"}})
    hl = highlights([own("h001", 20.0)])
    assert scoring.nominate(hl, ev, tmp_path, [grid()]) == 1
    assert nominees(hl)[0]["signals"]["events"]["nominated"] == ["placer"]
    # the same events with no such preset: nothing scores
    plain = highlights([own("h001", 20.0)])
    plain_ev = events_of(ev["events"])
    assert scoring.nominate(plain, plain_ev, tmp_path, [grid()]) == 0


def test_menu_time_nominates_nothing(tmp_path):
    ev = events_of(
        [feed(100.0), feed(300.0)],
        stats={"ocr": {SRC: {"menu_spans": [[90.0, 130.0]]}}},
    )
    hl = highlights([own("h001", 20.0)])
    scoring.nominate(hl, ev, tmp_path, [grid()])
    # the event in the menu span is no seed; the one outside is
    assert [c["signals"]["events"]["nominated"] for c in nominees(hl)] == [["player_kills"]]
    assert all(c["start"] > 200 for c in nominees(hl))


def test_a_window_mostly_in_menu_time_is_dropped(tmp_path):
    # the event starts just after the menu span, so it seeds; most of its window is menu time
    ev = events_of(
        [{**feed(100.0), "end": 100.2}],
        stats={"ocr": {SRC: {"menu_spans": [[60.0, 99.9]]}}},
    )
    hl = highlights([own("h001", 20.0)])
    scoring.nominate(hl, ev, tmp_path, [grid()])
    assert nominees(hl) == []


def test_an_explicit_max_candidates_caps_the_part_with_the_nominees_in_it(tmp_path):
    events = [feed(t) for t in (100.0, 200.0, 300.0, 400.0)]
    hl = highlights([own("h001", 20.0)], max_candidates=2)
    assert scoring.nominate(hl, events_of(events), tmp_path, [grid()]) == 1  # 2 in all
    assert len(hl["candidates"]) == 2
    full = highlights([own("h001", 20.0)], max_candidates=1)
    assert scoring.nominate(full, events_of(events), tmp_path, [grid()]) == 0
    open_ = highlights([own("h001", 20.0)])
    assert scoring.nominate(open_, events_of(events), tmp_path, [grid()]) == 4


def test_a_capped_part_keeps_the_nominee_with_the_best_boost(tmp_path):
    burst = [feed(300.0), feed(301.5), feed(303.0)]
    hl = highlights([own("h001", 20.0)], max_candidates=2)
    scoring.nominate(hl, events_of([feed(100.0), *burst]), tmp_path, [grid()])
    (nom,) = nominees(hl)
    assert nom["start"] < 300.0 < nom["end"]


def test_max_candidates_is_the_total_of_the_part(tmp_path):
    # --max-candidates 33: 30 picked candidates and ten events nothing covers -> 33 in all
    own_ = [own(f"h{k + 1:03d}", 15.0 * k, end=15.0 * k + 8.0) for k in range(30)]
    hl = highlights(own_, max_candidates=33)
    events = [feed(15.0 * k + 10.0) for k in range(10)]
    scoring.nominate(hl, events_of(events), tmp_path, [grid()])
    assert len(hl["candidates"]) == 33
    assert len(nominees(hl)) == 3


def test_action_cam_and_pool_candidates_are_left_alone(tmp_path):
    cam = highlights([own("h001", 20.0)], style="action-cam")
    before = copy.deepcopy(cam)
    assert scoring.nominate(cam, events_of([feed(100.0)]), tmp_path, [grid()]) == 0
    assert cam == before


def learned(name: str, base: str) -> None:
    folder = paths.styles_dir()
    folder.mkdir(parents=True, exist_ok=True)
    preset = {"name": name, "base": base, "weights": {"loudness": 0.5}}
    (folder / f"{name}.json").write_text(json.dumps(preset), encoding="utf-8")


def test_a_learned_style_with_a_generic_base_nominates_and_follows_its_open_list(tmp_path):
    learned("mygame", "gaming")
    learned("myaction", "action")
    assert get_style("mygame").open_list is True
    assert get_style("myaction").open_list is False
    assert analyze.candidate_cap(get_style("mygame")) == sys.maxsize
    assert analyze.candidate_cap(get_style("myaction")) == analyze.usual_count(get_style("action"))
    hl = highlights([own("h001", 20.0)], style="mygame")
    assert scoring.nominate(hl, events_of([feed(100.0)]), tmp_path, [grid()]) == 1
    assert nominees(hl)[0]["signals"]["events"]["nominated"] == ["player_kills"]


def test_an_unknown_style_without_a_recorded_base_nominates_nothing(tmp_path):
    """An older file (no style_base) of a style the library lacks: nothing, as before."""
    for base in (None, "no-such-base", 3):
        hl = highlights([own("h001", 20.0)], style="no-such-style")
        if base is not None:
            hl["style_base"] = base
        assert scoring.nominate(hl, events_of([feed(100.0)]), tmp_path, [grid()]) == 0


@pytest.mark.parametrize(("base", "expected"), [("gaming", 1), ("action", 0)])
def test_a_deleted_user_style_nominates_as_its_recorded_base(tmp_path, base, expected):
    """The file wins: a deleted user style nominates with its recorded built-in base (gaming:
    open list), and one over a base without an open list (action) still nominates nothing."""
    learned("mine", base)
    hl = highlights([own("h001", 20.0)], style="mine")
    hl["style_base"] = base
    (paths.styles_dir() / "mine.json").unlink()
    with pytest.raises(ValueError, match="Unknown style"):
        get_style("mine")
    assert scoring.nominate(hl, events_of([feed(100.0)]), tmp_path, [grid()]) == expected
    if expected:
        assert nominees(hl)[0]["signals"]["events"]["nominated"] == ["player_kills"]


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("{not json", "Cannot read learned style"),
        (json.dumps({"schema": 1, "base": "gaming", "weights": {"nope": 1}}),
         "not a valid style file"),
        (json.dumps({"base": "no-such-base", "weights": {}}), "unknown base style"),
    ],
)  # fmt: skip
def test_recorded_style_of_a_broken_style_file_raises_its_own_error(text, error):
    """A style file that is there but unusable is no deleted style: its error, no fallback."""
    folder = paths.styles_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "mine.json").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        scoring.recorded_style({"style": "mine", "style_base": "gaming"})
    hl = highlights([own("h001", 20.0)], style="mine")
    hl["style_base"] = "gaming"
    assert scoring.nominate(hl, events_of([feed(100.0)]), Path(), [grid()]) == 0


def test_recorded_style_degrades_a_missing_style_to_its_recorded_base(caplog):
    with caplog.at_level(logging.DEBUG, logger=scoring.__name__):
        got = scoring.recorded_style({"style": "gone", "style_base": "sport"}, logging.DEBUG)
    assert got is get_style("sport")
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (
            logging.DEBUG,
            "Style 'gone' is not in the style library (deleted?); using its base 'sport'",
        )
    ]
    caplog.clear()
    assert scoring.recorded_style({"style": "gaming", "style_base": "sport"}) is get_style("gaming")
    assert not caplog.records  # a style in the library: no fallback, no line
    for hl in ({"style": "gone"}, {"style": "gone", "style_base": "gone"}, {"style_base": None}):
        with pytest.raises(ValueError, match="not in the style library"):
            scoring.recorded_style(hl)


# --------------------------------------------------------------------------- rescore
def run_rescore(hl, ev, project, parts):
    """What ``rescore_highlights`` does to the analysis, without the files."""
    scorer.rescore_events(hl, ev, project, parts)
    scoring.number_candidates(hl)
    return hl


def test_rescoring_again_gives_the_same_candidates(tmp_path):
    hl = highlights([own("h001", 20.0, 60.0), own("h002", 300.0, 40.0)], target_total=60)
    ev = events_of([feed(100.0), feed(450.0), sound(520.0)])
    run_rescore(hl, ev, tmp_path, [grid()])
    first = copy.deepcopy(hl)
    for _ in range(2):
        run_rescore(hl, ev, tmp_path, [grid()])
        assert hl == first
    assert len(nominees(hl)) == 3


def test_ids_of_existing_candidates_are_stable_and_all_ids_unique(tmp_path):
    hl = highlights([own("h001", 20.0, 60.0), own("h005", 300.0, 40.0)])
    run_rescore(hl, events_of([feed(100.0), feed(450.0)]), tmp_path, [grid()])
    ids = [c["id"] for c in hl["candidates"]]
    assert len(ids) == len(set(ids)) == 4
    by_id = {c["id"]: c for c in hl["candidates"]}
    assert (by_id["h001"]["start"], by_id["h005"]["start"]) == (20.0, 300.0)
    assert sorted(i for i in ids if i not in ("h001", "h005")) == ["h006", "h007"]
    # a nominee that stays keeps its id when events are added later
    kept = {c["id"]: (c["start"], c["end"]) for c in nominees(hl)}
    run_rescore(hl, events_of([feed(100.0), feed(450.0), feed(550.0)]), tmp_path, [grid()])
    after = {c["id"]: (c["start"], c["end"]) for c in hl["candidates"]}
    assert all(after[i] == w for i, w in kept.items())
    ids = [c["id"] for c in hl["candidates"]]
    assert len(ids) == len(set(ids)) == 5
    assert "h008" in ids  # the new one: after the highest there ever was


def test_a_nominee_is_rescored_with_its_events_and_ranks_by_score(tmp_path):
    hl = highlights([own("h001", 20.0, 50.0), own("h002", 300.0, 20.0)])
    run_rescore(hl, events_of([feed(100.0), feed(101.0), feed(102.0)]), tmp_path, [grid()])
    nom = nominees(hl)[0]
    assert nom["signals"]["events"]["boost"] > 0
    assert nom["signals"]["events"]["nominated"] == ["player_kills"]
    assert nom["score_pre_events"] == 30.0  # 0.6 of the best other (50), then the boost
    assert nom["score"] == scorer.with_bonus(30.0, nom["signals"]["events"]["boost"], None)
    assert nom["score"] > 30.0
    scores = [c["score"] for c in hl["candidates"]]
    assert scores == sorted(scores, reverse=True)
    assert [c["id"] for c in hl["candidates"]][::2] == ["h001", "h002"]  # the nominee between


def test_the_rank_ties_on_the_floored_score_break_by_the_grid_score():
    def c(cid: str, **extra: Any) -> dict[str, Any]:
        return {**own(cid, 0.0, 50.0), **extra}

    plain, low, high = c("plain"), c("low", grid_score=5.0), c("high", grid_score=12.0)
    assert [x["id"] for x in sorted([low, plain, high], key=scorer.rank_key)] == [
        "plain", "high", "low"
    ]  # fmt: skip
    # the sort of the rescore
    for order in ([low, high, plain], [plain, low, high], [high, plain, low]):
        hl = highlights([copy.deepcopy(x) for x in order])
        scorer.rescore_events(hl, events_of([]), Path(), [grid()])
        assert [x["id"] for x in hl["candidates"]] == ["plain", "high", "low"]


def test_a_rated_nominee_keeps_its_rating_through_the_post_vision_rescore(tmp_path):
    hl = highlights([own("h001", 20.0, 50.0)])
    ev = events_of([feed(100.0), feed(101.0)])
    scorer.rescore_events(hl, ev, tmp_path, [grid()])
    (nom,) = nominees(hl)
    # what _rate_vision does: the rating on the record, blended into the analysis score
    nom["signals"]["vision"] = 8.0
    nom["score_pre_events"] = analyze.vision_blend(nom["score_pre_events"], 8.0, 0.35)
    window, grid_score = (nom["start"], nom["end"]), nom["grid_score"]
    scorer.rescore_events(hl, ev, tmp_path, [grid()])
    (again,) = nominees(hl)
    assert again is nom  # the same record: nothing nominated twice, nothing re-rated
    assert again["signals"]["vision"] == 8.0
    assert (again["start"], again["end"], again["grid_score"]) == (*window, grid_score)
    assert again["score_pre_events"] == analyze.vision_blend(30.0, 8.0, 0.35)
    boost = again["signals"]["events"]["boost"]
    assert again["score"] == scorer.with_bonus(again["score_pre_events"], boost, None)


def test_suggested_top_follows_the_target_with_the_nominees(tmp_path):
    hl = highlights([own("h001", 20.0, 30.0)], target_total=20)
    run_rescore(hl, events_of([feed(100.0), feed(300.0), feed(500.0)]), tmp_path, [grid()])
    n = len(default_fit.fit_target(hl["candidates"], 20))
    assert hl["suggested_top"] == n >= 2


def test_pool_moments_stay_last_and_outside_suggested_top(tmp_path):
    pool = {**own("v001", 400.0), "pool": "visual", "score": 99.0}
    hl = highlights([own("h001", 20.0), pool], target_total=500)
    run_rescore(hl, events_of([feed(100.0)]), tmp_path, [grid()])
    assert hl["candidates"][-1]["id"] == "v001"
    assert hl["suggested_top"] == 2


# --------------------------------------------------------------------------- the two paths
def write_caches(src: Path, span: float, cuts: list[float]) -> None:
    cdir = cache.source_dir(src)
    times = np.arange(1, int(span) * 10 + 1) * 0.1
    loud = -30.0 + 12.0 * np.exp(-(((times - 40.0) / 1.0) ** 2))
    cache.save_json(cdir / "loudness.json", {"t": times.tolist(), "m": loud.tolist()})
    words = [
        {"word": w, "start": 40.0 + 0.4 * i, "end": 40.3 + 0.4 * i, "probability": 0.95}
        for i, w in enumerate(["no", "way", "that", "worked"])
    ]
    seg = {"start": 40.0, "end": 41.6, "text": "no way that worked", "words": words}
    cache.save_json(cdir / transcript_name(TRANSCRIPT_MODELS[0]), [seg])
    cache.save_json(cdir / "scenes.json", cuts)


def analysis_of(src: Path, span: float, **entry: Any) -> dict[str, Any]:
    return {
        "style": "gaming",
        "settings": {"scenes": True},
        "sources": [{"path": str(src), "duration": span, **entry}],
        "candidates": [],
    }


def test_cached_grids_equal_the_grids_of_the_analysis(tmp_path):
    src = tmp_path / "game.mp4"
    src.write_bytes(b"x")
    write_caches(src, 120.0, [30.0, 70.0])
    style = get_style("gaming")
    (got,) = analyze.cached_grid_parts(analysis_of(src, 120.0), style)
    assert (got.source, got.index, got.offset, got.duration) == (src, 0, 0.0, 120.0)

    # what analyze_source builds from the same caches
    cdir = cache.source_dir(src)
    data = cache.load_json(cdir / "loudness.json")
    lufs = (np.asarray(data["t"]) - 0.2, np.asarray(data["m"]))
    segs = cache.load_json(cdir / transcript_name(TRANSCRIPT_MODELS[0]))
    want = analyze.part_signals(120.0, lufs, segs, [30.0, 70.0], style, src.name)
    assert got.signals.n == want.n
    for k in SIGNALS:
        np.testing.assert_allclose(got.signals.sig[k], want.sig[k], err_msg=k)
    assert got.signals.sig["loudness"].max() > 0.5 and got.signals.sig["scene"].max() > 0


def test_nominating_from_the_caches_equals_nominating_from_the_run(tmp_path):
    src = tmp_path / "game.mp4"
    src.write_bytes(b"x")
    # a cut within 1 s of the window of the kill at 41 s (37.5-44.5 unsnapped): it snaps there
    cuts = [37.0, 70.0]
    write_caches(src, 120.0, cuts)
    ev = {
        "events": [{**feed(41.0), "source": str(src)}, {**feed(100.0), "source": str(src)}],
        "windows": [],
        "sources": [str(src)],
    }
    style = get_style("gaming")

    def hl_for() -> dict[str, Any]:
        base = analysis_of(src, 120.0)
        base["candidates"] = [{**own("h001", 5.0), "source": str(src)}]
        return base

    # the analysis hands over its own grids; the events command reads the caches
    cdir = cache.source_dir(src)
    data = cache.load_json(cdir / "loudness.json")
    lufs = (np.asarray(data["t"]) - 0.2, np.asarray(data["m"]))
    segs = cache.load_json(cdir / transcript_name(TRANSCRIPT_MODELS[0]))
    ps = analyze.part_signals(120.0, lufs, segs, cuts, style, src.name)
    run = hl_for()
    scoring.nominate(run, ev, tmp_path, [analyze.GridPart(src, 0, 0.0, 120.0, ps)])
    events_cmd = hl_for()
    scoring.nominate(events_cmd, ev, tmp_path)
    assert len(nominees(run)) == 2
    assert min(c["start"] for c in nominees(run)) == 37.0  # snapped to the cut
    assert events_cmd == run


def test_part_signals_keeps_the_cuts_for_snapping():
    style = get_style("gaming")
    assert analyze.part_signals(60.0, None, [], [12.0, 30.0], style, "a.mp4").cuts == [12.0, 30.0]
    off = analyze.part_signals(60.0, None, [], None, style, "a.mp4")  # scene detection off
    assert off.cuts == [] and not off.sig["scene"].any()


def test_cached_grids_skip_recovered_sources_and_missing_caches(tmp_path):
    a, b, c = (tmp_path / f"{n}.mp4" for n in "abc")
    for p in (a, b, c):
        p.write_bytes(b"x")
    write_caches(a, 60.0, [])
    write_caches(c, 60.0, [])
    hl = analysis_of(a, 60.0)
    hl["sources"] += [
        {"path": str(b), "duration": 60.0},  # no caches at all
        {"path": str(c), "duration": 60.0, "read_errors": [{"at": 30.0}]},  # a recovered part
    ]
    parts = analyze.cached_grid_parts(hl, get_style("gaming"))
    assert [p.source for p in parts] == [a]


def test_cached_grids_cover_the_analysed_seconds_only(tmp_path):
    src = tmp_path / "game.mp4"
    src.write_bytes(b"x")
    write_caches(src, 120.0, [])
    hl = analysis_of(src, 120.0, analyzed_s=50.0)
    (got,) = analyze.cached_grid_parts(hl, get_style("gaming"))
    assert got.duration == 50.0 and got.signals.n == int(50.0 / DT) + 1


# --------------------------------------------------------------------------- readers
def test_a_nominee_without_transcript_or_keywords_does_not_break_the_readers(tmp_path):
    hl = highlights([own("h001", 20.0, 60.0)])
    run_rescore(hl, events_of([feed(100.0)]), tmp_path, [grid()])
    nom = nominees(hl)[0]
    assert nom["transcript"] == "" and nom["signals"]["keywords"] == []

    view = moments_view(hl)
    got = {c["cli_id"]: c for c in view["candidates"]}
    assert got[nom["id"]]["display_label"] == "player-kills"
    assert len({c["id"] for c in view["candidates"]}) == 2

    ids = [c["id"] for c in hl["candidates"]]
    table = context.MomentTable(hl["candidates"], [])
    for cid in ids:
        row = table.row(cid, full=True)
        assert row["len"] > 0
        json.dumps(row)
    assert table.label(nom["id"], nom["label"]) == "player-kills"


# --------------------------------------------------------------------------- the events command
def project_of(tmp_path: Path, cands: list[dict[str, Any]], **settings: Any) -> tuple[Path, Path]:
    """A project folder with a highlights.json of ``cands`` for a stand-in source whose caches
    are written (a loudness peak at 40 s, scene cuts at 30 and 70 s)."""
    src = tmp_path / "game.mp4"
    src.write_bytes(b"x")
    write_caches(src, 120.0, [30.0, 70.0])
    proj = tmp_path / "proj"
    proj.mkdir()
    hl = analysis_of(src, 120.0)
    hl["settings"].update(settings)
    hl["candidates"] = [{**c, "source": str(src)} for c in cands]
    hl["suggested_top"] = 1
    cache.save_json(proj / "highlights.json", hl)
    return proj, src


def write_events(proj: Path, src: Path, times: list[float]) -> None:
    events = [{**feed(t), "source": str(src)} for t in times]
    cache.save_json(proj / "events.json", {"events": events, "windows": [], "sources": [str(src)]})


def saved(proj: Path) -> dict[str, Any]:
    return json.loads((proj / "highlights.json").read_text(encoding="utf-8"))


def test_the_events_command_numbers_the_nominees_when_no_candidate_is_left(tmp_path):
    proj, src = project_of(tmp_path, [])  # e.g. every candidate fell in menu time
    write_events(proj, src, [41.0, 100.0])
    scoring.rescore_highlights(proj)
    hl = saved(proj)
    ids = [c["id"] for c in hl["candidates"]]
    assert sorted(ids) == ["h001", "h002"] and all(map(scoring.is_nominated, hl["candidates"]))
    assert hl["settings"]["last_candidate_id"] == 2
    moments, style = planning.select_moments(proj, ids=["h002", "h001"])
    assert [m["id"] for m in moments] == ["h002", "h001"] and style == "gaming"
    assert len(planning.select_moments(proj, top=1)[0]) == 1


def test_a_vanished_nominees_id_is_not_given_to_another_window(tmp_path):
    proj, src = project_of(tmp_path, [])
    seen: dict[str, tuple[float, float]] = {}

    def run(times: list[float]) -> dict[str, tuple[float, float]]:
        write_events(proj, src, times)
        scoring.rescore_highlights(proj)
        got = {c["id"]: (c["start"], c["end"]) for c in saved(proj)["candidates"]}
        for cid, window in got.items():  # an id names one window, ever
            assert seen.setdefault(cid, window) == window
        return got

    first = run([41.0, 100.0])  # 41 s sits on the loudness peak: it ranks first
    assert sorted(first) == ["h001", "h002"]
    second = run([41.0])  # the window of h002 is gone
    assert list(second) == ["h001"]
    third = run([41.0, 110.0])  # a new window must not take h002
    assert sorted(third) == ["h001", "h003"]
    fourth = run([41.0, 110.0, 90.0])
    assert sorted(fourth) == ["h001", "h003", "h004"]
    assert saved(proj)["settings"]["last_candidate_id"] == 4


def test_number_candidates_goes_past_the_highest_id_there_ever_was():
    hl = highlights([own("h001", 20.0), {k: v for k, v in own("x", 90.0).items() if k != "id"}])
    hl["settings"]["last_candidate_id"] = 5
    scoring.number_candidates(hl)
    assert [c["id"] for c in hl["candidates"]] == ["h001", "h006"]
    assert hl["settings"]["last_candidate_id"] == 6


# --------------------------------------------------------------------------- no audio
def test_a_source_with_scenes_and_no_audio_nominates_on_both_paths(tmp_path):
    src = tmp_path / "silent.mp4"
    src.write_bytes(b"x")
    cache.save_json(cache.source_dir(src) / "scenes.json", [30.0])  # no loudness, no transcript
    style = get_style("gaming")
    hl = analysis_of(src, 120.0)
    ev = {"events": [{**feed(100.0), "source": str(src)}], "windows": [], "sources": [str(src)]}

    (cached,) = analyze.cached_grid_parts(hl, style)
    assert not cached.signals.sig["loudness"].any() and cached.signals.sig["scene"].max() > 0
    from_caches = copy.deepcopy(hl)
    assert scoring.nominate(from_caches, ev, tmp_path) == 1

    ps = analyze.part_signals(120.0, None, [], [30.0], style, src.name)  # what a run builds
    from_run = copy.deepcopy(hl)
    assert scoring.nominate(from_run, ev, tmp_path, [analyze.GridPart(src, 0, 0.0, 120.0, ps)]) == 1
    assert from_run == from_caches


# --------------------------------------------------------------------------- _free_around, _place_free
@pytest.mark.parametrize(
    ("t", "taken", "expected"),
    [
        (30.0, [], (0.0, 100.0)),
        (30.0, [(10.0, 20.0), (40.0, 60.0)], (20.0, 40.0)),  # between two
        (5.0, [(10.0, 20.0)], (0.0, 10.0)),  # before the first
        (90.0, [(10.0, 20.0)], (20.0, 100.0)),  # after the last
        (15.0, [(10.0, 20.0)], (15.0, 15.0)),  # inside one: nothing is free
        (40.0, [(40.0, 50.0), (10.0, 20.0)], (20.0, 40.0)),  # at a start, the list unsorted
        (50.0, [(40.0, 50.0)], (50.0, 100.0)),  # at an end
        (90.0, [(10.0, 20.0), (130.0, 140.0)], (20.0, 100.0)),  # a window past the part
    ],
)
def test_free_around_is_the_stretch_no_taken_window_covers(t, taken, expected):
    assert analyze._free_around(t, taken, 100.0) == expected


def free_cand(start: float, end: float, peak_t: float) -> analyze.Candidate:
    return analyze.Candidate(Path("a.mp4"), start, end, peak_t, 0.5, dict.fromkeys(SIGNALS, 0.0))


def place(
    cands, taken, span: float = 100.0, cuts: list[float] | None = None, segments=()
) -> list[analyze.Candidate]:
    n = int(span / DT) + 1
    ps = analyze.PartSignals(
        n, {k: np.zeros(n) for k in SIGNALS}, [], [], [], list(segments), list(cuts or [])
    )
    return analyze._place_free(cands, taken, ps, span, get_style("gaming"))


def assert_placed(out, taken, span: float = 100.0) -> None:
    style = get_style("gaming")
    spans_ = spans(out)
    for a, b in spans_:
        assert a >= 0.0 and b <= span + 1e-6
        assert style.min_len - 1e-6 <= b - a <= style.max_len + 1e-6
        assert not any(overlap((a, b), t) for t in taken)
    assert not any(overlap(x, y) for x, y in pairwise(sorted(spans_)))


def test_place_free_keeps_a_candidate_in_the_stretch_beside_a_taken_window():
    taken = [(0.0, 50.0), (70.0, 100.0)]
    (got,) = place([free_cand(48.5, 56.0, 52.0)], taken)
    assert_placed([got], taken)
    assert got.start >= 50.0 and got.peak_t == 52.0


def test_place_free_keeps_a_cut_snap_out_of_the_taken_window():
    taken = [(0.0, 50.0)]
    # the start (50) snaps to the cut at 49.5, inside the taken window; it is fitted again
    (got,) = place([free_cand(50.0, 56.0, 53.0)], taken, cuts=[49.5])
    assert_placed([got], taken)
    assert got.start == 50.0 and got.peak_t == 53.0


def test_place_free_keeps_two_candidates_a_snap_brought_together_apart():
    taken = [(0.0, 44.0), (59.6, 100.0)]
    # one word, 53.5-55.8: the first window's end (54.5) snaps out to its end, the second's
    # start (55) back to its start, so they overlap. Resolved, the second is left 3.65 s long;
    # fitting it to min_len moves it back over the first, and what remains is too short to keep.
    word = {"start": 53.5, "end": 55.8, "word": "gg"}
    segments = [{"start": 53.5, "end": 55.8, "text": "gg", "words": [word]}]
    cands = [free_cand(44.0, 54.5, 50.0), free_cand(55.0, 59.6, 58.0)]
    out = place(cands, taken, segments=segments)
    assert_placed(out, taken)
    assert spans(out) == [(44.0, pytest.approx(55.95))]


def test_place_free_drops_what_the_free_stretch_is_too_short_for():
    taken = [(0.0, 50.0), (52.0, 100.0)]  # 2 s free, min_len is 4
    assert place([free_cand(48.5, 55.0, 51.0)], taken) == []
    assert place([free_cand(48.5, 55.0, 49.0)], [(0.0, 50.0)]) == []  # peak inside the taken one


def test_place_free_stops_at_the_end_of_the_part():
    taken = [(0.0, 50.0)]
    (got,) = place([free_cand(55.5, 62.0, 59.0)], taken, span=60.0)
    assert got.end <= 60.0 + 1e-6
    assert_placed([got], taken, span=60.0)


def test_place_free_keeps_several_candidates_of_one_stretch_apart():
    taken = [(0.0, 50.0), (80.0, 100.0)]
    cands = [free_cand(50.5, 56.5, 52.0), free_cand(54.0, 60.0, 56.0), free_cand(70.0, 76.0, 73.0)]
    out = place(cands, taken)
    assert 1 <= len(out) <= 3
    assert_placed(out, taken)
    # a crowd for a stretch with room for one window: the rest are dropped, none overlaps
    tight = [(0.0, 50.0), (58.0, 100.0)]
    crowd = [free_cand(49.0 + i, 55.0 + i, 52.0 + i) for i in range(0, 5, 2)]
    out = place(crowd, tight)
    assert 1 <= len(out) <= 2
    assert_placed(out, tight)


def test_place_free_merges_per_free_stretch_not_across_a_taken_window():
    # gaming merge_gap is 3: the taken 50-52 is narrower than that, and the two windows (gap
    # 2 s, 12 s together) would merge into one over it if they were merged as one group
    taken = [(50.0, 52.0)]
    out = place([free_cand(45.0, 50.0, 48.0), free_cand(52.0, 57.0, 54.0)], taken)
    assert_placed(out, taken)
    assert [c.peak_t for c in sorted(out, key=lambda c: c.start)] == [48.0, 54.0]
    first, second = sorted(out, key=lambda c: c.start)
    assert first.end <= 50.0 + 1e-6 and second.start >= 52.0 - 1e-6


# --------------------------------------------------------------------------- cycle 2 fixes
def quiet_grid(span: float, offset: float = 0.0, index: int = 0, bumps=()):
    """A part of ``span`` s with loudness bumps ``(second, height)``."""
    n = int(span / DT) + 1
    ps = analyze.PartSignals(n, {k: np.zeros(n) for k in SIGNALS}, [], [], [], [], [])
    for t, h in bumps:
        ps.sig["loudness"] += bump(t, n, h)
    return analyze.GridPart(Path(SRC), index, offset, span, ps)


def short(t: float, **flags: bool) -> dict[str, Any]:
    return {**feed(t, **flags), "end": t + 0.5}


def test_a_recovered_part_clamps_the_free_stretch_to_the_part(tmp_path):
    # parts at 0 and 520 s; the kill at 520.5 sits right after the second part starts, with a
    # candidate at 523-531 (3 s of room, under min_len) and another in the first part
    parts = [quiet_grid(500.0), quiet_grid(1000.0, offset=520.0, index=1)]
    cands = [own("h001", 100.0, end=110.0), own("h002", 523.0, end=531.0)]
    hl = highlights(cands, target_total=60)
    events = [short(520.5), short(700.0)]
    scoring.nominate(hl, events_of(events), tmp_path, parts)  # must not raise
    for c in nominees(hl):
        assert c["start"] >= 520.0 and c["end"] <= 1520.0 + 1e-6
        assert not overlap((c["start"], c["end"]), (523.0, 531.0))
    assert any(c["start"] < 700.0 < c["end"] for c in nominees(hl))


def test_the_cap_cut_keeps_the_nominees_with_the_best_grid_score(tmp_path):
    # three kills with the same boost, all floored to 0.6 of the best other (36): only the grids
    # tell them apart, and two fit under max_candidates=3 (one candidate is picked already)
    part = quiet_grid(600.0, bumps=[(100.0, 0.05), (200.0, 0.2), (300.0, 0.1)])
    hl = highlights([own("h001", 20.0, 60.0)], max_candidates=3)
    events = [short(100.0), short(200.0), short(300.0)]
    assert scoring.nominate(hl, events_of(events), tmp_path, [part]) == 2
    noms = sorted(nominees(hl), key=lambda c: c["start"])
    assert [c["score"] for c in noms] == [36.0, 36.0]  # floored: the score cannot decide
    assert noms[0]["start"] < 200.0 < noms[0]["end"]
    assert noms[1]["start"] < 300.0 < noms[1]["end"]
    assert all(0.0 < c["grid_score"] < 36.0 for c in noms)


def test_a_window_with_an_event_on_each_side_nominates_both_events(tmp_path):
    # gaming (max_len 12, merge_gap 3): the window 100-104 is taken, kills at 99 (before) and
    # 104.5 (after), neither inside it. Their windows, 95.5-102.5 and 101-107.5, would merge
    # into one of 12 s over the taken one; fitted first they are 4 s apart and stay two.
    hl = highlights([own("h001", 100.0, end=104.0)])
    ev = events_of([short(99.0), short(104.5)])
    assert scoring.nominate(hl, ev, tmp_path, [grid()]) == 2
    before, after = sorted(nominees(hl), key=lambda c: c["start"])
    assert before["start"] < 99.0 < before["end"] <= 100.0 + 1e-6
    assert 104.0 - 1e-6 <= after["start"] < 104.5 < after["end"]


def end_loud(n: int = 1001) -> np.ndarray:
    """A bump at 50 s and a taller final sample (the source ends loud)."""
    combined = bump(50.0, n, 0.8)
    combined[-1] = 1.0
    return combined


def test_an_open_list_ends_and_finds_the_peaks_before_a_loud_last_sample():
    combined = end_loud()
    sig = {"loudness": combined}
    out = within(
        10, analyze._pick_windows, Path("a.mp4"), combined, sig, get_style("gaming"),
        100.0, sys.maxsize, 5,
    )  # fmt: skip
    assert [c.peak_t for c in out] == [pytest.approx(100.0), pytest.approx(50.0)]


def test_a_list_with_a_cap_does_not_repeat_a_loud_last_sample():
    # #771: the last sample is picked once, then the peak before it (not the last sample again)
    combined = end_loud()
    out = analyze._pick_windows(
        Path("a.mp4"), combined, {"loudness": combined}, get_style("action"), 100.0, 3
    )
    assert [c.peak_t for c in out] == [pytest.approx(100.0), pytest.approx(50.0)]


def test_a_capped_style_keeps_every_peak_of_a_600_s_source_that_ends_loud():
    # ten bumps 50 s apart and a taller final sample: eleven distinct picks, not the last one again
    n = 6001
    combined = sum(bump(50.0 * k, n, 0.6) for k in range(1, 11))
    combined[-1] = 1.0
    out = analyze._pick_windows(
        Path("a.mp4"), combined, {"loudness": combined}, get_style("action"), 600.0, 11
    )
    peaks = sorted(round(c.peak_t) for c in out)
    assert peaks == [50 * k for k in range(1, 11)] + [600]


def ranked_ids(style: str, events: list[dict[str, Any]], tmp_path) -> tuple[list[str], int]:
    hl = highlights(
        [own("h001", 20.0, 50.0), own("h002", 300.0, 20.0)], style=style, target_total=20
    )
    scorer.rescore_events(hl, events_of(events), tmp_path, [grid()])
    ids = ["nominee" if scoring.is_nominated(c) else c["id"] for c in hl["candidates"]]
    return ids, hl["suggested_top"]


def test_an_open_list_ranks_the_nominee_by_score(tmp_path):
    burst = [short(100.0), short(101.5), short(103.0)]
    # the nominee (floor 30 plus the boost of three kills) beats the weak pick (20) on score
    ids, top = ranked_ids("gaming", burst, tmp_path)
    assert ids == ["h001", "nominee", "h002"]
    assert top == 2  # the third does not fit the 20 s target


@pytest.mark.parametrize("style", ["action", "funny", "sport"])
def test_a_style_without_an_open_list_gets_no_nominee_and_keeps_its_suggested_top(style, tmp_path):
    burst = [short(100.0), short(101.5), short(103.0)]
    ids, top = ranked_ids(style, burst, tmp_path)
    assert ids == ["h001", "h002"]
    assert top == ranked_ids(style, [], tmp_path)[1]


@pytest.mark.parametrize("order", [("c", "a", "b"), ("a", "b", "c"), ("b", "c", "a")])
def test_records_that_tie_are_ordered_by_start_on_a_rescore(order):
    # not nominated, equal score: their start orders them, not the id or the list (#949)
    starts = {"a": 200.0, "b": 0.0, "c": 100.0}
    hl = highlights([own(k, starts[k], 50.0) for k in order])
    scorer.rescore_events(hl, events_of([]), Path(), [grid()])
    assert [c["id"] for c in hl["candidates"]] == ["b", "c", "a"]


# --------------------------------------------------------------------------- weak heard laughs
WEAK = scoring.LAUGH_STRONG / 2  # a laugh the detector found only by its relative rule


def talk(*pieces: tuple[float, str]) -> list[dict[str, Any]]:
    """One transcript segment per ``(start, text)``, each word a second long."""
    return [
        {
            "start": t, "end": t + len(text.split()),
            "words": [
                {"word": f" {w}", "start": t + i, "end": t + i + 1.0}
                for i, w in enumerate(text.split())
            ],
        }
        for t, text in pieces
    ]  # fmt: skip


def test_a_strong_laugh_nominates_a_moment_as_before(tmp_path):
    hl = highlights([own("h001", 20.0)])
    scoring.nominate(
        hl, events_of([sound(100.0, confidence=scoring.LAUGH_STRONG)]), tmp_path, [grid()]
    )
    (nom,) = nominees(hl)
    assert nom["signals"]["events"]["nominated"] == ["laughter"] and nom["label"] == "laughter"


def test_a_weak_laugh_alone_makes_no_moment_and_no_label(tmp_path):
    hl = highlights([own("h001", 20.0)])
    assert scoring.nominate(hl, events_of([sound(100.0, confidence=WEAK)]), tmp_path, [grid()]) == 0
    assert nominees(hl) == []


def test_a_weak_laugh_far_from_everything_stays_out_even_beside_other_events(tmp_path):
    far = scoring.BACKUP_S + 2
    events = [sound(100.0, confidence=WEAK), sound(100.0 + far + 1.5, confidence=WEAK)]
    assert scoring.seeds(events, DEFAULT_EVENT_TYPES, STYLE) == []
    assert (
        scoring.seeds([*events, feed(104.0)], DEFAULT_EVENT_TYPES, STYLE)[0].name == "player_kills"
    )


@pytest.mark.parametrize("other", ["Laughter", "Scream", "Cheering"])
def test_a_weak_laugh_with_another_sound_nearby_is_a_seed(other):
    events = [sound(100.0, confidence=WEAK), sound(103.0, other, confidence=0.3)]
    names = {s.name for s in scoring.seeds(events, DEFAULT_EVENT_TYPES, STYLE)}
    assert "laughter" in names


def test_a_weak_laugh_with_a_laughter_word_in_the_transcript_nominates(tmp_path, monkeypatch):
    words = talk((98.0, "that is so funny"), (101.0, "[laughter] okay"))
    monkeypatch.setattr(scoring, "_segments", lambda src, language=None: words)
    hl = highlights([own("h001", 20.0)])
    scoring.nominate(hl, events_of([sound(100.0, confidence=WEAK)]), tmp_path, [grid()])
    assert [c["label"] for c in nominees(hl)] == ["laughter"]


def test_the_spoken_word_laughter_does_not_back_a_weak_laugh(tmp_path, monkeypatch):
    words = talk(
        (98.0, "he said laughter twice"),
    )
    monkeypatch.setattr(scoring, "_segments", lambda src, language=None: words)
    hl = highlights([own("h001", 20.0)])
    scoring.nominate(hl, events_of([sound(100.0, confidence=WEAK)]), tmp_path, [grid()])
    assert nominees(hl) == []


def _loud_grid(at: float) -> analyze.GridPart:
    g = grid()
    g.signals.sig["loudness"] = bump(at, N, 1.0, 1.0)
    return g


def test_loud_speech_backs_a_weak_laugh_but_quiet_speech_or_loud_noise_do_not(
    tmp_path, monkeypatch
):
    event = events_of([sound(100.0, confidence=WEAK)])
    speech = talk((99.0, "oh come on now"))
    monkeypatch.setattr(scoring, "_segments", lambda src, language=None: speech)
    loud, quiet = highlights([own("h001", 20.0)]), highlights([own("h001", 20.0)])
    scoring.nominate(loud, event, tmp_path, [_loud_grid(100.5)])
    scoring.nominate(quiet, event, tmp_path, [grid()])
    assert len(nominees(loud)) == 1 and nominees(quiet) == []
    monkeypatch.setattr(scoring, "_segments", lambda src, language=None: [])  # loud, nobody talking
    noise = highlights([own("h001", 20.0)])
    scoring.nominate(noise, event, tmp_path, [_loud_grid(100.5)])
    assert nominees(noise) == []


def test_loud_speech_backs_a_weak_laugh_at_the_stream_start(tmp_path, monkeypatch):
    speech = talk((0.0, "oh come on now"))
    monkeypatch.setattr(scoring, "_segments", lambda src, language=None: speech)
    hl = highlights([own("h001", 20.0)])
    scoring.nominate(hl, events_of([sound(0.5, confidence=WEAK)]), tmp_path, [_loud_grid(1.0)])
    assert len(nominees(hl)) == 1


def test_a_weak_laugh_after_a_part_boundary_reads_its_own_part_not_the_previous(
    tmp_path, monkeypatch
):
    speech = talk((599.0, "oh come on now"))
    monkeypatch.setattr(scoring, "_segments", lambda src, language=None: speech)
    loud_before = _loud_grid(599.5)  # loud at the end of the first part only
    quiet_after = grid(offset=N * DT, index=1)
    hl = highlights([own("h001", 20.0)])
    event = events_of([sound(N * DT + 0.5, confidence=WEAK)])
    scoring.nominate(hl, event, tmp_path, [loud_before, quiet_after])
    assert nominees(hl) == []


def test_a_weak_laugh_inside_a_candidate_still_counts_for_its_boost():
    # the backup rule is about making a moment, not about the evidence the boost reads
    boost, evidence = scoring.event_boost(
        own("h001", 90.0, end=110.0), [sound(100.0, confidence=WEAK)]
    )
    assert boost > 0 and evidence["laughter"] == round(WEAK, 3)


@pytest.mark.parametrize(
    ("laughter", "heard"), [(0.12, True), (0.5, True), (0.119, False), (0.0, False), (None, False)]
)
def test_laughter_is_heard_in_a_candidate_from_its_audio_evidence(laughter, heard):
    events = {} if laughter is None else {"laughter": laughter}
    assert scoring.heard_laughter({"signals": {"events": events, "laughter": 0.9}}) is heard
    assert scoring.heard_laughter({"signals": {}}) is False
