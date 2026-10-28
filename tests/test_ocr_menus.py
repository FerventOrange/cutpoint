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

    def frames(path, fps, rect, hw, errors, start=0.0, dur=None, cancel=None, exit_codes=None):
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
        r"\d+ staying lines, \d+ together \(text >= 5, together >= 3\), \d+ OCR events -> cover",
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
        "text_lines": float("inf"),  # off too
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
    """Gameplay, and at 20-30 s a menu of 8 boxes (far below the 24 boxes every box
    rule needs) with a banner phrase on it and 2 rows of tabs."""
    if not 20 <= t < 30:
        return [("Hold", 900, 820)]
    out = [("Hull breached", 320, 220)]
    out += [(f"Tab {chr(65 + i)}", 320 + 180 * (i % 4), 760 + 100 * (i // 4)) for i in range(7)]
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
        r"\d+ staying lines, \d+ together \(text >= 5, together >= 3\), \d+ OCR events -> static",
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


def test_one_notice_in_still_gameplay_is_no_menu_text(tmp_path: Path, monkeypatch) -> None:
    def screen(t: float) -> list[tuple[str, float, float]]:
        out = [("Hold", 900, 820)]
        if 22 <= t < 24:  # one gameplay notice, sliding in (read twice)
            out.append(("Ridge sighted", 860, 400 if t < 23 else 420))
        return out

    events, stats = _run(
        tmp_path, monkeypatch, _Engine(screen), still=lambda t: 20 <= t < 26.5, paint=_minimap
    )
    # 6.3 s still with one line of new text: no menu text, below STATIC_BARE_MIN_S
    assert stats["static_frames"] == stats["menu_frames"] == 0
    notices = [(e["text"], e["t"], e["sightings"]) for e in events if e["text"] != "Hold"]
    assert notices == [("Ridge sighted", 22.0, 2)]  # its sighting is kept


def test_short_still_menu_in_a_sparsely_read_region_is_static(tmp_path: Path, monkeypatch) -> None:
    def screen(t: float) -> list[tuple[str, float, float]]:
        if 20.3 <= t < 25.5:  # a result screen with 2 lines of its own
            return [("Tally board", 700, 300), ("Proceed", 900, 600)]
        return [("Hold", 900, 820)]

    preset = _preset()
    preset["regions"] = [preset["regions"][0] | {"every": 6}]  # read every 2 s at 3 fps
    _, stats = _run(
        tmp_path, monkeypatch, _Engine(screen), preset, still=lambda t: 20.3 <= t < 25.4
    )
    # 20.33-25.33 s still; its lines are read at 22 s and kept alive at 24 s only, so
    # their ends are padded by the 2 s read interval to cover half of the part
    assert stats["static_frames"] == stats["menu_frames"] == 16


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
    # 2 menu lines read as the cursor's screen opens (3.5 s), on screen to the end: 0-3 s
    # is too short; 3.3-13 s holds still with the cursor
    lines = [(3.5, 13.0), (3.5, 13.0)]
    rules = menu_rules(_still_records(times, grey), feed_times=set(), text_spans=lines)[0]
    assert [rules[n] for n in (0, 9, 10, 22, 39)] == ["none", "none", "static", "static", "static"]
    # without menu text 9.7 s is below STATIC_BARE_MIN_S
    rules = menu_rules(_still_records(times, grey), feed_times=set())[0]
    assert set(rules.values()) == {"none"}


def test_static_menu_text_needs_lines_that_stay() -> None:
    times = [n / 3 for n in range(19)]  # 0-6.0 s still (a spawn map)
    records = _still_records(times, [90.0] * 19)

    def rules(spans: list[tuple[float, float]]) -> set[str]:
        return set(menu_rules(records, feed_times=set(), text_spans=spans)[0].values())

    assert rules([]) == {"none"}  # no menu text: below STATIC_BARE_MIN_S
    assert rules([(0.0, 6.0), (0.3, 6.0)]) == {"static"}  # 2 menu lines that stay
    assert rules([(-1.5, 6.0), (-1.5, 6.0)]) == {"static"}  # read as the menu opened
    assert rules([(0.0, 6.0)]) == {"none"}  # one notice line, however long
    assert rules([(2.0, 4.0), (2.0, 4.0)]) == {"none"}  # a 2-line notice box flashing 2 s
    assert rules([(2.0, 2.0), (3.0, 3.0)]) == {"none"}  # one-frame reads
    assert rules([(-3.0, 6.0), (-3.0, 6.0)]) == {"none"}  # read before the guard window


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


# --------------------------------------------------------------------------- text screens


def _moving_records(times: list[float]) -> list[dict[str, Any]]:
    """Frame records at ``times`` of a screen that never holds still (each frame its
    own still run) and has few boxes: no box or static rule fires."""
    return [
        {"id": n, "t": t, "source": "rec.mp4", "boxes": [], "still": n} for n, t in enumerate(times)
    ]


def test_text_screen_needs_many_lines_that_stay() -> None:
    times = [n / 3 for n in range(75)]  # 25 s of moving frames
    records = _moving_records(times)

    def text(spans: list[tuple[float, float]]) -> list[float]:
        rules = menu_rules(records, feed_times=set(), text_spans=spans)[0]
        return [f["t"] for f in records if rules[f["id"]] == "text"]

    lobby = [(4.0, 14.0)] * 5  # 5 lines on screen 4-14 s
    assert text(lobby) == [t for t in times if 4.0 <= t <= 14.0]
    assert text(lobby[:4]) == []  # 4 lines: an objective, a location and an overlay or two
    assert text([(4.0, 9.5)] * 5) == []  # lines that stay 5.5 s (< TEXT_STAY_S)
    # a HUD player list on screen all match (> TEXT_STAY_MAX_S) and a 3-line notice box
    assert text([(-80.0, 20.0)] * 7 + [(4.0, 14.0)] * 3) == []
    assert text([(-80.0, 20.0)] * 7 + lobby) == [t for t in times if 4.0 <= t <= 14.0]
    # flashing notices and kill lines next to two lasting labels, however many at once
    assert text([(0.0, 20.0)] * 2 + [(4 + i / 3, 5 + i / 3) for i in range(30)]) == []
    # held 2.7 s (< TEXT_MIN_S): the stretch of frames with 5 such lines at once
    assert text([(4.0, 12.0)] * 4 + [(9.3, 15.3)]) == []
    assert text([(4.0, 12.0)] * 4 + [(9.0, 15.0)]) == [t for t in times if 9.0 <= t <= 12.0]
    # off without feed times or text spans (a HUD area too small for the static rule)
    assert set(menu_rules(records, text_spans=lobby)[0].values()) == {"none"}
    assert set(menu_rules(records, feed_times=set())[0].values()) == {"none"}
    assert menu_rules(records, feed_times=set(), text_spans=lobby)[1]["text_lines"] == 5


def test_text_screen_ends_at_gaps() -> None:
    # 5 lasting lines, but the frames 4-6 s and 8-10 s are apart by 2 s (a skipped range)
    times = [4.0, 4 + 1 / 3, 4 + 2 / 3, 5.0, 6.0, 8.0, 9.0, 10.0, 20.0]
    records = _moving_records(times)
    rules = menu_rules(records, feed_times=set(), text_spans=[(0.0, 12.0)] * 5)[0]
    assert set(rules.values()) == {"none"}  # each part holds 2 s
    # (staying lines, first read together): the sixth line came 5 s after the others
    shown = killfeed.staying_lines(records, [(0.0, 12.0)] * 5 + [(5.0, 12.0)])
    assert shown == [(5, 5)] * 3 + [(6, 5)] * 5 + [(0, 0)]


def test_text_screen_at_low_ocr_frame_rates() -> None:
    # --ocr-fps 0.5: frames 2 s apart, as far apart as StillRuns allows (2 frame intervals)
    records = _moving_records([2.0 * n for n in range(13)])  # 0-24 s
    lobby = [(4.0, 14.0)] * 5
    gap = killfeed.StillRuns(0.5).max_gap
    assert gap == 4.0
    rules = menu_rules(records, feed_times=set(), text_spans=lobby, max_gap=gap)[0]
    assert [f["t"] for f in records if rules[f["id"]] == "text"] == [
        4.0,
        6.0,
        8.0,
        10.0,
        12.0,
        14.0,
    ]
    rules = menu_rules(records, feed_times=set(), text_spans=lobby)[0]  # 1 s: never a stretch
    assert set(rules.values()) == {"none"}


def test_text_screen_skips_hud_lines_of_short_sources() -> None:
    times = [n / 3 for n in range(135)]  # a 45 s source, under TEXT_STAY_MAX_S
    records = _moving_records(times)

    def text(spans: list[tuple[float, float]], keys: list | None = None) -> list[float]:
        rules = menu_rules(records, feed_times=set(), text_spans=spans, text_keys=keys)[0]
        return [f["t"] for f in records if rules[f["id"]] == "text"]

    hud, lobby = [(0.0, 45.0)] * 7, [(4.0, 14.0)] * 5  # a player list on screen all along
    assert text(hud) == []
    # first read late: a missed read, a region read every few frames, a fade-in
    assert text([(1.5, 45.0)] * 7) == []
    assert text([(4.5, 45.0)] * 7) == []
    assert text(hud + lobby) == [t for t in times if 4.0 <= t <= 14.0]
    # the list hidden by a scoreboard at 20-25 s comes back as new spans of the same lines
    pieces = [(0.0, 20.0)] * 7 + [(25.0, 45.0)] * 7
    names = [f"player {i}" for i in range(7)] * 2
    assert text(pieces, names) == []
    assert text(pieces) != []  # without keys each piece is a screen of its own
    # the same lobby at the start and the end: on screen for under TEXT_HUD_SHARE of the read
    twice = [(0.0, 8.0)] * 5 + [(37.0, 45.0)] * 5
    shown = text(twice, [f"map {i}" for i in range(5)] * 2)
    assert shown == [t for t in times if t <= 8.0 or t >= 37.0]


def test_text_screen_needs_lines_read_together() -> None:
    times = [n / 3 for n in range(75)]
    records = _moving_records(times)

    def text(spans: list[tuple[float, float]]) -> list[float]:
        rules = menu_rules(records, feed_times=set(), text_spans=spans)[0]
        return [f["t"] for f in records if rules[f["id"]] == "text"]

    # gameplay labels that arrive one by one, 2.5 s apart: 5 on screen at 11-17 s
    labels = [(2.0 + 2.5 * i, 17.0) for i in range(4)] + [(11.0, 17.0)]
    assert text(labels) == []
    # a screen that opens with 3 of them at once
    assert text(labels[:2] + [(10.0, 17.0)] * 2 + [(11.0, 17.0)]) == [
        t for t in times if 11.0 <= t <= 17.0
    ]


def _lobby(t: float) -> list[tuple[str, float, float]]:
    """Gameplay, and at 20-32 s a lobby of 6 lines of its own, one per row (no box
    rule sees 6 boxes); a notice flashes at 40-41 s."""
    if 20 <= t < 32:
        words = ("Armory", "Barracks", "Courtyard", "Dockyard", "Embassy", "Foundry")
        return [(w, 600, 230 + 75 * i) for i, w in enumerate(words)]
    out = [("Hold", 900, 820)]
    if 40 <= t < 41:
        out.append(("Squad wiped", 860, 400))
    return out


def test_moving_lobby_gives_no_events(tmp_path: Path, monkeypatch) -> None:
    events, stats = _run(tmp_path, monkeypatch, _Engine(_lobby))  # never still
    assert not [e for e in events if 20 <= e["t"] < 32]
    assert [(e["text"], e["t"]) for e in events if e["type"] == "banner"] == [("Squad wiped", 40.0)]
    assert stats["static_frames"] == 0
    assert stats["menu_frames"] == stats["text_frames"] > 0


def test_row_notices_that_stay_are_no_feed_lines(tmp_path: Path, monkeypatch) -> None:
    """A row_notices feed: a row that scrolls out is a notice, a row that stays all
    clip (lobby text in the feed's area) is none."""
    x, y, w, h = FEED
    preset = _preset() | {"feed": {"row_notices": True}}
    preset["regions"] = [
        *preset["regions"],
        {"name": "feed", "x": x, "y": y, "w": w, "h": h, "kind": "feed", "gate": "det"},
    ]

    def screen(t: float) -> list[tuple[str, float, float]]:
        out = [("Hold", 900, 820), ("Waiting for other players", 1650, 50)]
        if 10 <= t < 14:
            out.append(("Kip has left the game", 1650, 100))
        if 12 <= t < 16:  # a new row: Kip's row is read again (Ana's once: OCR noise)
            out.append(("Ana has left the game", 1650, 130))
        return out

    events, _ = _run(tmp_path, monkeypatch, _Engine(screen, (NOTICE, FEED)), preset)
    notices = [(e["text"], e["t"]) for e in events if e.get("region") == "feed"]
    assert notices == [("Kip has left the game", 10.0)]  # not the 60 s "Waiting" row


def _menu_over_feed(t: float) -> list[tuple[str, float, float]]:
    """The options menu at 20-30 s, with settings rows ("name | value") in the feed's
    area; an icon kill line (two names an icon apart) at 40-43 s."""
    if 20 <= t < 30:
        rows = [("Brightness", "Medium"), ("Subtitles", "Enabled")]
        feed = [(a, 1410, 100 + 40 * r) for r, (a, _) in enumerate(rows)]
        feed += [(b, 1640, 100 + 40 * r) for r, (_, b) in enumerate(rows)]
        return _menu() + feed
    out = [("Hold", 900, 820)]
    if 40 <= t < 43:
        out += [("Alder", 1410, 100), ("Boxer7", 1640, 100)]
    return out


def test_menu_rows_over_an_icon_feed_are_no_kills(tmp_path: Path, monkeypatch) -> None:
    """Settings rows of a menu over a discovered icon feed (``row_notices``) read as icon
    lines on the menu frames; those kills are dropped with the menu, the kill after it
    stays. Without the flag no icon line is read."""
    engine = _Engine(_menu_over_feed, (NOTICE, FEED))
    preset = _feed_preset() | {"feed": {"row_notices": True}}
    events, stats = _run(tmp_path, monkeypatch, engine, preset)
    assert stats["menu_frames"] == 30
    kills = [(e["killer"], e["victim"], e["t"]) for e in events if e["type"] == "kill"]
    assert kills == [("Alder", "Boxer7", 40.0)]
    (plain := tmp_path / "plain").mkdir()
    events, stats = _run(plain, monkeypatch, engine, _feed_preset())
    assert stats["menu_frames"] == 30
    assert [e for e in events if e["type"] == "kill"] == []
