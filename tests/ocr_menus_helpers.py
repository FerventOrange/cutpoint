"""Helpers shared by test modules, moved out of test_ocr_menus.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from cutpoint.events import killfeed
from cutpoint.events.common import Source
from cutpoint.events.killfeed import OcrParams, detect_ocr_events
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


def _feed_preset() -> dict[str, Any]:
    fx, fy, fw, fh = FEED
    preset = _preset()
    preset["regions"] = [
        *preset["regions"],
        {"name": "feed", "x": fx, "y": fy, "w": fw, "h": fh, "kind": "feed", "gate": "det"},
    ]
    return preset


def _moving_records(times: list[float]) -> list[dict[str, Any]]:
    """Frame records at ``times`` of a screen that never holds still (each frame its
    own still run) and has few boxes: no box or static rule fires."""
    return [
        {"id": n, "t": t, "source": "rec.mp4", "boxes": [], "still": n} for n, t in enumerate(times)
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
