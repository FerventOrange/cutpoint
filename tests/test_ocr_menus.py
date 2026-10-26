"""Menu frames in the OCR events layer (fake decoder and OCR engine, invented HUD texts)."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from cutpoint.events import killfeed
from cutpoint.events.common import Source
from cutpoint.events.killfeed import (
    GENERIC_VERSION,
    EventTracker,
    FeedParser,
    OcrParams,
    detect_ocr_events,
    menu_rules,
    menu_times,
)
from cutpoint.events.regions import preset_from_dict

NOTICE = (300, 200, 1320, 700)  # a centred notice region (reference px)
FEED = (1400, 40, 500, 400)  # an edge feed region
FPS = 3.0


def _preset() -> dict[str, Any]:
    x, y, w, h = NOTICE
    return {
        "game": "zonewar",
        "regions": [
            {"name": "notice", "x": x, "y": y, "w": w, "h": h, "kind": "notice", "gate": "det"}
        ],
        "banner_phrases": ["Squad wiped", "Hull breached"],
        "bigtext": False,
    }


def _menu() -> list[tuple[str, float, float]]:
    """A centred options grid: 31 lines over the screen centre."""
    out = [("Hull breached", 380, 330)]  # a menu line that is also a banner phrase
    out += [(f"Option {chr(65 + i)}", 260 + (i % 6) * 240, 260 + (i // 6) * 150) for i in range(30)]
    return out


def _screen(t: float) -> list[tuple[str, float, float]]:
    """(text, x, y) of what the screen shows at ``t``: gameplay, a menu at 20-30 s."""
    if 20 <= t < 30:
        return _menu()
    out = [("Hold", 900, 820)]
    if 5 <= t < 8 or 40 <= t < 43:
        out.append(("Squad wiped", 860, 400))
    return out


class _Engine:
    """Detection and recognition of the screen at the decoder's current time.

    ``screen(t)`` gives ``(text, x, y)`` lines; a crop is told apart by its size
    (each region shows the lines whose box starts inside it).
    """

    def __init__(self, screen=None, regions: tuple[tuple[int, int, int, int], ...] = (NOTICE,)):
        self.t = 0.0
        self.screen = screen or _screen
        self.regions = regions

    def _boxes(self, crop: np.ndarray) -> list[tuple[str, list[list[float]]]]:
        h, w = crop.shape[:2]
        x0, y0, _, _ = next(r for r in self.regions if (r[2], r[3]) == (w, h))
        return [
            (text, [[x - x0, y - y0], [x - x0 + bw, y - y0], [x - x0 + bw, y - y0 + 24]])
            for text, x, y in self.screen(self.t)
            if x0 <= x < x0 + w and y0 <= y < y0 + h
            for bw in [min(200, 12 * len(text))]
        ]

    def detect(self, crop: np.ndarray, scale: float = 1.0) -> list[list[list[float]]]:
        return [b for _, b in self._boxes(crop)]

    def read_rows(self, crop, rows, min_score, pad=6.0) -> list[tuple[list, str, float]]:
        out = []
        for y0, x0, y1, x1 in rows:
            for text, b in self._boxes(crop):
                if y0 <= (b[0][1] + b[2][1]) / 2 <= y1 and x0 <= b[0][0] <= x1:
                    out.append((b, text, 0.95))
        return out


def _run(
    tmp_path: Path,
    monkeypatch,
    engine: _Engine | None,
    preset: dict[str, Any] | None = None,
    still=lambda t: False,
    on_progress=None,
    paint=None,
) -> tuple[list[dict], dict]:
    """Detect the OCR events of a 60 s source. Its frames are moving gameplay (a new
    grey level every frame) except where ``still(t)`` (one grey level); ``paint(t,
    frame)`` may draw on each frame."""
    data = preset or _preset()
    monkeypatch.setattr(killfeed, "load_preset", lambda *_a, **_k: preset_from_dict(data))
    monkeypatch.setattr(
        killfeed, "probe", lambda _p: SimpleNamespace(width=1920, height=1080, duration=60.0)
    )

    def frames(path, fps, rect, hw, errors, start=0.0, dur=None):
        _, _, w, h = rect
        for n in range(int(60 * fps)):
            t = n / fps
            if engine:
                engine.t = t
            frame = np.full((h, w, 3), 90 if still(t) else n * 53 % 256, np.uint8)
            if paint:
                paint(t, frame)
            yield t, frame

    monkeypatch.setattr(killfeed, "iter_frames", frames)
    src = Source(tmp_path / "rec.mp4", 60.0, tmp_path, [])
    return detect_ocr_events(
        src, OcrParams(game="zonewar", fps=FPS), engine, on_progress=on_progress
    )


def test_menu_frames_give_no_events(tmp_path: Path, monkeypatch) -> None:
    events, stats = _run(tmp_path, monkeypatch, _Engine())
    banners = [(e["text"], e["t"]) for e in events if e["type"] == "banner"]
    assert banners == [("Squad wiped", 5.0), ("Squad wiped", 40.0)]
    assert not [e for e in events if 20 <= e["t"] < 30]  # menu lines are no notices
    assert stats["menu_frames"] == 30  # 10 s at 3 fps
    # cached: read back without an engine (no OCR)
    assert _run(tmp_path, monkeypatch, None) == (events, stats)


def test_menu_and_event_frames_are_logged_at_debug(tmp_path: Path, monkeypatch, caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="cutpoint.events.killfeed")
    _run(tmp_path, monkeypatch, _Engine())
    lines = [r.getMessage() for r in caplog.records if " menu test " in r.getMessage()]
    menus = [x for x in lines if not x.endswith("-> none")]
    assert len(menus) == 30  # the menu frames, 20-30 s at 3 fps
    assert re.fullmatch(
        r"rec\.mp4 menu test 20\.0s: \d+ boxes \(min 24\), centre cover 0\.\d\d "
        r"\(cover >= 0\.50, list >= 0\.25\), median 1\.0, jump off, "
        r"still 0\.0s \(static >= 5\.0s with menu text, >= 10\.0s without\), menu text (yes|no), "
        r"\d+ OCR events -> cover",
        menus[0],
    )
    events = [x for x in lines if x.endswith("-> none")]
    assert events and all(" 0 OCR events" not in x for x in events)  # frames with sightings
    assert any(x.startswith("rec.mp4 menu test 5.0s: 2 boxes (min 24)") for x in events)
    texts = ("Squad", "Hull", "Option", "Hold")
    assert not [x for x in lines if any(t in x for t in texts)]  # no OCR text


def test_menu_rules_name_the_rule() -> None:
    def frame(n: int, boxes: int, spread: bool) -> dict[str, Any]:
        pos = [
            (300 + 240 * (i % 6), 200 + 130 * (i // 6)) if spread else (20, 30 * i)
            for i in range(boxes)
        ]
        edge = [{"x0": x, "y0": y, "x1": x + 60, "y1": y + 20} for x, y in pos]
        return {"id": n, "t": n / 3, "source": "rec.mp4", "boxes": edge}

    frames = [frame(n, 30, False) for n in range(10)] + [frame(10, 30, True), frame(11, 70, False)]
    rules, info = menu_rules(frames)
    assert [rules[n] for n in range(12)] == ["none"] * 10 + ["cover", "jump"]
    assert info == {
        "median": 30.0,
        "min_boxes": 24,
        "cover": 0.5,
        "list": float("inf"),
        "jump": 60.0,
        "static_s": float("inf"),  # no feed times given: the static rule is off
        "static_bare_s": float("inf"),
        "static_diff": killfeed.STATIC_MAX_DIFF,
    }


def test_menu_rule_changes_the_cache_key() -> None:
    preset = preset_from_dict(_preset())
    assert GENERIC_VERSION >= 8
    assert OcrParams().key("zonewar", preset)["preset"]["interp"] == GENERIC_VERSION


def test_menu_times_use_the_median_of_the_frames() -> None:
    def frame(n: int, boxes: int, spread: bool) -> dict[str, Any]:
        return {
            "id": n,
            "t": n / 3,
            "source": "rec.mp4",
            "boxes": [
                {"x0": x, "y0": y, "x1": x + 60, "y1": y + 20}
                for i in range(boxes)
                for x, y in [
                    (300 + 240 * (i % 6), 200 + 130 * (i // 6)) if spread else (20, 30 * i)
                ]
            ],
        }

    frames = [frame(n, 30, False) for n in range(10)]  # a busy edge HUD: no menu
    frames += [frame(10, 30, True)]  # the same box count over the centre: a menu
    frames += [frame(11, 70, False)]  # over twice the median: a menu
    assert menu_times(frames) == {10 / 3, 11 / 3}


def test_list_menu_with_a_quiet_median() -> None:
    # the 2-column, 32-line options list (centre cover ~0.33) among frames of 1-2 boxes
    def frame(n: int, lines: list[tuple[str, float, float]]) -> dict[str, Any]:
        boxes = [{"x0": x, "y0": y, "x1": x + 96, "y1": y + 24} for _, x, y in lines]
        return {"id": n, "t": n / 3, "source": "rec.mp4", "boxes": boxes}

    listed = [("Hull breached", 500, 260)]
    listed += [
        (f"Option {chr(65 + i)}", 500 + (i % 2) * 700, 290 + (i // 2) * 38) for i in range(31)
    ]
    frames = [frame(n, _screen(n / 3 + 40)) for n in range(20)] + [frame(20, listed)]
    assert menu_times(frames) == {20 / 3}
    rules, info = menu_rules(frames)
    assert rules[20] == "list" and info["jump"] == float("inf")


def test_box_jump_needs_a_busy_median() -> None:
    def frame(n: int, boxes: int) -> dict[str, Any]:
        edge = [{"x0": 20, "y0": 30 * i, "x1": 80, "y1": 30 * i + 20} for i in range(boxes)]
        return {"id": n, "t": n / 3, "source": "rec.mp4", "boxes": edge}

    # region boxes are few: 30 boxes along an edge are no jump over a median of 4
    frames = [frame(n, 2 + n % 5) for n in range(20)] + [frame(20, 30)]
    assert menu_times(frames) == set()
    # a quiet HUD (median 0): the jump is off, not inf x 0
    assert menu_rules([frame(n, 0) for n in range(5)])[1]["jump"] == float("inf")


def test_full_feed_is_no_menu(tmp_path: Path, monkeypatch) -> None:
    names = ["Orrin", "Pell", "Quade", "Sorrel", "Tamsin", "Ulric", "Varo", "Wynn", "Yarrow"]
    names += ["Zeb"]

    def screen(t: float) -> list[tuple[str, float, float]]:
        rows = 10 if abs(t - 30) < 1e-6 else 1 + int(t) % 2  # one frame with a full feed
        out = [("Hold", 900, 820)]
        for r in range(rows):
            y = 60 + r * 30
            out += [(names[r], 1410, y), ("destroyed", 1500, y), (names[9 - r], 1640, y)]
        return out

    fx, fy, fw, fh = FEED
    preset = _preset() | {"banner_phrases": []}
    preset["regions"] = [
        *preset["regions"],
        {"name": "feed", "x": fx, "y": fy, "w": fw, "h": fh, "kind": "feed", "gate": "det"},
    ]
    events, stats = _run(tmp_path, monkeypatch, _Engine(screen, (NOTICE, FEED)), preset)
    assert stats["menu_frames"] == 0
    full = {(e["killer"], e["victim"]) for e in events if e["type"] == "kill" and e["t"] == 30.0}
    assert len(full) >= 8  # the rows new in that frame (the first ones were already on screen)


def test_drop_sightings() -> None:
    tr = EventTracker(persist_s=8.0, fuzzy=0.8)
    for t in (1.0, 2.0, 3.0):
        tr.see(t, "banner", "SQUAD WIPED", text="SQUAD WIPED")
    tr.see(2.0, "banner", "HULL BREACHED", text="HULL BREACHED")
    tr.drop_sightings({1.0, 2.0})
    assert [(e["text"], e["t"], e["sightings"]) for e in tr.finish()] == [("SQUAD WIPED", 3.0, 1)]
    # kept events (feed sightings, kills) keep every sighting
    tr.see(4.0, "kill", "ORRIN > PELL", text="Orrin > Pell", region="feed")
    tr.drop_sightings({3.0, 4.0}, keep=lambda e: e["type"] == "kill")
    assert [(e["text"], e["t"]) for e in tr.finish()] == [("Orrin > Pell", 4.0)]


def test_kill_words_match_whole_words() -> None:
    parser = FeedParser({}, (), 1.0)  # default kill words, "assist" among them
    assert parser._word_split("Nym requests assistance now") is None
    assert parser._word_split("Nym assist Vel") == ("Nym", "Vel", "assist")
    assert parser._word_split("NymAssistVel") == ("Nym", "Vel", "assist")  # camelCase joined
    assert parser._word_split("NymassistedVel") is None  # inside a longer word
    assert parser._word_split("Qorv2assist Vel") == ("Qorv2", "Vel", "assist")  # after digits
    assert parser._word_split("Nym SHOT  DOWN Vel") == ("Nym", "Vel", "kill")
    # long kill words may be joined to a name on one side; short ones never
    assert parser._word_split("Ashdestroyed Kip") == ("Ash", "Kip", "kill")
    assert parser._word_split("ASHDESTROYED KIP") == ("ASH", "KIP", "kill")
    assert parser._word_split("Kiphas crashed") == ("Kip", "", "crash")
    assert parser._word_split("Nym skilled Vel") is None
    crash = parser.parse(
        np.zeros((40, 400, 3), np.uint8),
        [([[0, 0], [300, 0], [300, 24], [0, 24]], "Kiphas crashed", 0.9)],
    )
    assert [(e["type"], e["victim"]) for e in crash] == [("crash", "Kip")]


# --------------------------------------------------------------------------- static screens


def _quiet_menu(t: float) -> list[tuple[str, float, float]]:
    """Gameplay, and at 20-30 s a menu of 8 lines (far below the 24 boxes every box
    rule needs) with a banner phrase on it."""
    if not 20 <= t < 30:
        return [("Hold", 900, 820)]
    out = [("Hull breached", 320, 220)]
    out += [(f"Tab {chr(65 + i)}", 320 + 180 * i, 860) for i in range(7)]
    return out


def _dirs(tmp_path: Path, *names: str) -> list[Path]:
    """One folder (and so one OCR cache) per run."""
    out = [tmp_path / n for n in names]
    for d in out:
        d.mkdir()
    return out


def test_still_menu_with_few_boxes_gives_no_events(tmp_path: Path, monkeypatch) -> None:
    moving, still = _dirs(tmp_path, "moving", "still")
    # moving frames: the box rules let the quiet menu through
    events, stats = _run(moving, monkeypatch, _Engine(_quiet_menu))
    assert [e["t"] for e in events if e["type"] == "banner"] == [20.0]
    assert stats["menu_frames"] == stats["static_frames"] == 0
    # the menu holds still: a static screen
    events, stats = _run(still, monkeypatch, _Engine(_quiet_menu), still=lambda t: 20 <= t < 30)
    assert not [e for e in events if 20 <= e["t"] < 30]
    assert stats["static_area"] == 0.45  # the notice region: 1320 x 700 of 1920 x 1080
    assert stats["menu_frames"] == stats["static_frames"] == 30  # 10 s at 3 fps


def test_still_menu_is_logged_with_the_static_rule(tmp_path: Path, monkeypatch, caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="cutpoint.events.killfeed")
    _run(tmp_path, monkeypatch, _Engine(_quiet_menu), still=lambda t: 20 <= t < 30)
    lines = [r.getMessage() for r in caplog.records if " menu test " in r.getMessage()]
    static = [x for x in lines if x.endswith("-> static")]
    assert len(static) == 30
    assert re.fullmatch(
        r"rec\.mp4 menu test 20\.0s: 8 boxes \(min 24\), centre cover 0\.\d\d "
        r"\(cover >= 0\.50, list >= 0\.25\), median 1\.0, jump off, "
        r"still 9\.7s \(static >= 5\.0s with menu text, >= 10\.0s without\), menu text yes, "
        r"\d+ OCR events -> static",
        static[0],
    )
    assert not [x for x in lines if any(t in x for t in ("Hull", "Tab", "Hold"))]  # no OCR text


def _feed_screen(t: float) -> list[tuple[str, float, float]]:
    """Still gameplay at 20-40 s: a kill in the feed from 24 s, a banner beside it at
    24-26 s and one far from it at 32-34 s."""
    out = [("Hold", 900, 820)]
    if t >= 24:
        out += [("Orrin", 1410, 60), ("destroyed", 1500, 60), ("Pell", 1640, 60)]
    if 24 <= t < 26:
        out.append(("Squad wiped", 860, 400))
    if 32 <= t < 34:
        out.append(("Hull breached", 860, 400))
    return out


def _feed_preset() -> dict[str, Any]:
    fx, fy, fw, fh = FEED
    preset = _preset()
    preset["regions"] = [
        *preset["regions"],
        {"name": "feed", "x": fx, "y": fy, "w": fw, "h": fh, "kind": "feed", "gate": "det"},
    ]
    return preset


def test_still_gameplay_keeps_feed_kills(tmp_path: Path, monkeypatch) -> None:
    moving, still = _dirs(tmp_path, "moving", "still")
    engine = _Engine(_feed_screen, (NOTICE, FEED))
    ref, _ = _run(moving, monkeypatch, engine, _feed_preset())
    events, stats = _run(still, monkeypatch, engine, _feed_preset(), still=lambda t: 20 <= t < 40)
    kills = [(e["t"], e["killer"], e["victim"]) for e in events if e["type"] == "kill"]
    assert kills == [(e["t"], e["killer"], e["victim"]) for e in ref if e["type"] == "kill"]
    assert kills == [(24.0, "Orrin", "Pell")]
    banners = [(e["text"], e["t"]) for e in events if e["type"] == "banner"]
    # beside the kill the banner stays; the still stretch far from it is a static screen
    assert banners == [("Squad wiped", 24.0)]
    assert {e["text"] for e in ref if e["type"] == "banner"} == {"Squad wiped", "Hull breached"}
    # 20-21.7 s is too short; 26.3-39.7 s (> 2 s from the kill, no menu text, over
    # STATIC_BARE_MIN_S) is static
    assert stats["static_frames"] == 41


def test_box_rule_menu_keeps_feed_kills(tmp_path: Path, monkeypatch) -> None:
    def screen(t: float) -> list[tuple[str, float, float]]:
        out = _screen(t)  # the options grid at 20-30 s (rule cover)
        if t >= 25:  # a kill in the feed while the grid is open
            out += [("Orrin", 1410, 60), ("destroyed", 1500, 60), ("Pell", 1640, 60)]
        return out

    events, stats = _run(tmp_path, monkeypatch, _Engine(screen, (NOTICE, FEED)), _feed_preset())
    assert stats["menu_frames"] == 30 and stats["static_frames"] == 0
    kills = [(e["t"], e["killer"], e["victim"]) for e in events if e["type"] == "kill"]
    assert kills == [(25.0, "Orrin", "Pell")]
    assert not [e for e in events if 20 <= e["t"] < 30 and e["type"] != "kill"]  # menu text


def _scoped(t: float) -> list[tuple[str, float, float]]:
    """Still gameplay at 20-26.3 s (no feed): a banner phrase at 22-24 s."""
    out = [("Hold", 900, 820)]
    if 22 <= t < 24:
        out.append(("Hull breached", 860, 400))
    return out


def _minimap(t: float, frame: np.ndarray) -> None:
    """A small HUD corner (40 x 40 px) that changes every frame."""
    frame[:40, :40] = int(t * 3) * 97 % 256


def test_still_gameplay_with_a_moving_small_hud_is_no_static_screen(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    caplog.set_level(logging.DEBUG, logger="cutpoint.events.killfeed")
    events, stats = _run(
        tmp_path, monkeypatch, _Engine(_scoped), still=lambda t: 20 <= t < 26.5, paint=_minimap
    )
    # the HUD area holds still 6.3 s (the corner moves its thumbnail by far less than
    # STATIC_MAX_DIFF), over STATIC_MIN_S, but shows no menu text: below STATIC_BARE_MIN_S
    assert stats["static_frames"] == stats["menu_frames"] == 0
    assert [(e["text"], e["t"]) for e in events if e["type"] == "banner"] == [
        ("Hull breached", 22.0)
    ]
    lines = [r.getMessage() for r in caplog.records if " menu test 22.0s" in r.getMessage()]
    assert len(lines) == 1 and "still 6.3s" in lines[0] and "menu text no" in lines[0]


def test_short_still_screen_with_menu_text_is_static(tmp_path: Path, monkeypatch) -> None:
    def screen(t: float) -> list[tuple[str, float, float]]:
        out = _scoped(t)
        if 20 <= t < 26.5:  # a result screen: a phrase and words of its own
            out += [("Tally board", 700, 300), ("Proceed", 900, 600)]
        return out

    events, stats = _run(
        tmp_path, monkeypatch, _Engine(screen), still=lambda t: 20 <= t < 26.5, paint=_minimap
    )
    assert stats["static_frames"] == stats["menu_frames"] == 20  # 20-26.3 s at 3 fps
    assert not [e for e in events if e["type"] == "banner"]


def test_long_still_menu_without_menu_text_is_static(tmp_path: Path, monkeypatch) -> None:
    def screen(t: float) -> list[tuple[str, float, float]]:
        # a 16 s still menu showing only a banner phrase (no new menu text)
        return [("Hull breached", 320, 220)] if 18 <= t < 34 else [("Hold", 900, 820)]

    events, stats = _run(tmp_path, monkeypatch, _Engine(screen), still=lambda t: 18 <= t < 34)
    assert stats["static_frames"] == stats["menu_frames"] == 48  # 16 s at 3 fps
    assert not [e for e in events if e["type"] == "banner"]


def test_moving_gameplay_is_no_static_screen(tmp_path: Path, monkeypatch) -> None:
    events, stats = _run(
        tmp_path, monkeypatch, _Engine(_feed_screen, (NOTICE, FEED)), _feed_preset()
    )
    assert {e["text"] for e in events if e["type"] == "banner"} == {"Squad wiped", "Hull breached"}
    assert stats["static_frames"] == 0


def _still_records(times: list[float], grey: list[float]) -> list[dict[str, Any]]:
    """Frame records at ``times`` whose thumbnails are one grey level each, with
    their still run ids (:class:`StillRuns` at 3 fps)."""
    runs = killfeed.StillRuns(FPS)
    shape = killfeed.STATIC_THUMB[::-1]
    return [
        {"id": n, "t": t, "source": "rec.mp4", "boxes": [], "still": runs.add(t, np.full(shape, g))}
        for n, (t, g) in enumerate(zip(times, grey, strict=True))
    ]


def test_static_rule_needs_still_runs_and_feed_times() -> None:
    times = [n / 3 for n in range(40)]  # 13 s still
    still = _still_records(times, [90.0] * 40)
    assert set(menu_rules(still, feed_times=set())[0].values()) == {"static"}
    assert set(menu_rules(still)[0].values()) == {"none"}  # no feed times: off
    # records without a still run (made before still runs existed): never static
    old = [{k: v for k, v in f.items() if k != "still"} for f in still]
    assert set(menu_rules(old, feed_times=set())[0].values()) == {"none"}
    # a small change (a cursor) stays still; a larger one starts a new run
    diff = killfeed.STATIC_MAX_DIFF
    grey = [90.0] * 10 + [90 + 2 * diff] * 10 + [90 + 2.5 * diff] * 5 + [90 + 2 * diff] * 15
    # menu text read as the cursor's screen opens (3.5 s): 0-3 s is too short; 3.3-13 s
    # holds still with the cursor
    rules = menu_rules(_still_records(times, grey), feed_times=set(), text_times={3.5})[0]
    assert [rules[n] for n in (0, 9, 10, 22, 39)] == ["none", "none", "static", "static", "static"]
    # without menu text 9.7 s is below STATIC_BARE_MIN_S
    rules = menu_rules(_still_records(times, grey), feed_times=set())[0]
    assert set(rules.values()) == {"none"}


def test_static_runs_end_at_gaps() -> None:
    # equal thumbnails across a 30 s gap (a skipped range): no still run over it
    times = [0.0, 1 / 3, 2 / 3, 1.0, 9.0, 9 + 1 / 3, 9 + 2 / 3, 40.0, 40 + 1 / 3]
    records = _still_records(times, [90.0] * len(times))
    assert len({f["still"] for f in records}) == 3
    assert killfeed.static_frames(records, set()) == set()
    # long enough but too few frames (a very low OCR rate): no static screen
    sparse = [dict(f, still=0) for f in _still_records([0.0, 6.0], [90.0, 90.0])]
    assert killfeed.static_frames(sparse, set()) == set()


def test_small_hud_area_has_no_static_rule(tmp_path: Path, monkeypatch) -> None:
    corner = {"name": "notice", "x": 20, "y": 20, "w": 400, "h": 200, "kind": "notice"}
    preset = _preset() | {"regions": [corner | {"gate": "det"}]}

    def screen(t: float) -> list[tuple[str, float, float]]:
        return [("Hull breached", 40, 40)] if 20 <= t < 30 else []

    events, stats = _run(
        tmp_path, monkeypatch, _Engine(screen, ((20, 20, 400, 200),)), preset, lambda t: True
    )
    assert stats["static_area"] == 0.04  # 400 x 200 of 1920 x 1080
    assert stats["static_frames"] == 0
    assert [e["t"] for e in events if e["type"] == "banner"] == [20.0]


def test_old_cache_is_read_again(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(killfeed, "GENERIC_VERSION", 7)  # a result cached before static screens
    _run(tmp_path, monkeypatch, _Engine())
    assert _run(tmp_path, monkeypatch, None)[1].get("skipped") is None  # cached
    monkeypatch.setattr(killfeed, "GENERIC_VERSION", 8)
    # interpreted again from the raw OCR cache, without an engine (no OCR)
    events, stats = _run(tmp_path, monkeypatch, None)
    assert stats.get("skipped") is None and stats["static_frames"] == 0
    assert [(e["text"], e["t"]) for e in events if e["type"] == "banner"] == [
        ("Squad wiped", 5.0),
        ("Squad wiped", 40.0),
    ]
    # without the raw read the footage is read again
    for folder in tmp_path.glob("ocr-raw-*"):
        shutil.rmtree(folder)
    monkeypatch.setattr(killfeed, "GENERIC_VERSION", 9)
    assert _run(tmp_path, monkeypatch, None) == ([], {"skipped": "not cached"})


def test_progress_within_a_source(tmp_path: Path, monkeypatch) -> None:
    shares: list[float] = []
    _run(tmp_path, monkeypatch, _Engine(), on_progress=shares.append)
    assert shares == sorted(set(shares))  # rising, once per percent
    assert len(shares) == 100 and shares[0] == 0.0 and shares[-1] == 0.99
