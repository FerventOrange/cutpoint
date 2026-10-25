"""Menu frames in the OCR events layer (fake decoder and OCR engine, invented HUD texts)."""

from __future__ import annotations

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
    tmp_path: Path, monkeypatch, engine: _Engine | None, preset: dict[str, Any] | None = None
) -> tuple[list[dict], dict]:
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
            yield t, np.zeros((h, w, 3), np.uint8)

    monkeypatch.setattr(killfeed, "iter_frames", frames)
    src = Source(tmp_path / "rec.mp4", 60.0, tmp_path, [])
    return detect_ocr_events(src, OcrParams(game="zonewar", fps=FPS), engine)


def test_menu_frames_give_no_events(tmp_path: Path, monkeypatch) -> None:
    events, stats = _run(tmp_path, monkeypatch, _Engine())
    banners = [(e["text"], e["t"]) for e in events if e["type"] == "banner"]
    assert banners == [("Squad wiped", 5.0), ("Squad wiped", 40.0)]
    assert not [e for e in events if 20 <= e["t"] < 30]  # menu lines are no notices
    assert stats["menu_frames"] == 30  # 10 s at 3 fps
    # cached: read back without an engine (no OCR)
    assert _run(tmp_path, monkeypatch, None) == (events, stats)


def test_menu_rule_changes_the_cache_key() -> None:
    preset = preset_from_dict(_preset())
    assert GENERIC_VERSION >= 7
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


def test_box_jump_needs_a_busy_median() -> None:
    def frame(n: int, boxes: int) -> dict[str, Any]:
        edge = [{"x0": 20, "y0": 30 * i, "x1": 80, "y1": 30 * i + 20} for i in range(boxes)]
        return {"id": n, "t": n / 3, "source": "rec.mp4", "boxes": edge}

    # region boxes are few: 30 boxes along an edge are no jump over a median of 4
    frames = [frame(n, 2 + n % 5) for n in range(20)] + [frame(20, 30)]
    assert menu_times(frames) == set()


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
