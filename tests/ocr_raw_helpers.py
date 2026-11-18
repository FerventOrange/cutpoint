"""Helpers shared by test modules, moved out of test_ocr_raw.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from cutpoint import ffmpeg_utils
from cutpoint.events import killfeed
from cutpoint.events.common import Source
from cutpoint.events.killfeed import OcrParams, detect_ocr_events
from cutpoint.events.regions import preset_from_dict

NOTICE = (300, 200, 1320, 700)  # a centred notice region ("det" gate at half size)


FEED = (1400, 40, 500, 400)  # a feed with coloured names ("det" gate)


BANNER = (600, 940, 700, 120)  # a banner region ("mask" gate, every 2nd frame)


ORIGIN = (300, 40)  # top left of the decoded HUD area


FPS = 3.0


RED, BLUE, WHITE = (0, 0, 255), (255, 0, 0), (255, 255, 255)


def _preset(**notice: Any) -> dict[str, Any]:
    def region(name: str, rect: tuple[int, ...], kind: str, **extra: Any) -> dict[str, Any]:
        x, y, w, h = rect
        return {"name": name, "x": x, "y": y, "w": w, "h": h, "kind": kind, **extra}

    return {
        "game": "zonewar",
        "regions": [
            region("notice", NOTICE, "notice", gate="det", det_scale=0.5, **notice),
            region("feed", FEED, "feed", gate="det"),
            region("banner", BANNER, "banner", every=2),
        ],
        "banner_phrases": ["Squad wiped", "Hull breached", "Zone held"],
        "feed": {"entry": "one_line", "colors": {"red": "enemy", "blue": "ally"}},
        "bigtext": False,
    }


def _screen(t: float) -> list[tuple[str, float, float, tuple[int, int, int] | None]]:
    """(text, x, y, colour) of what the screen shows at ``t``: a still menu at 8-20 s,
    coloured feed names from 24 s and 40 s, banners."""
    out: list[tuple[str, float, float, tuple[int, int, int] | None]] = []
    if 8 <= t < 20:
        out += [("Hull breached", 320, 220, None)]
        out += [(f"Tab {chr(65 + i)}", 320 + 180 * i, 860, None) for i in range(7)]
    else:
        out.append(("Hold", 900, 820, None))  # still on screen: the notice is kept alive
    if 4 <= t < 7 or 44 <= t < 47:
        out.append(("Squad wiped", 860, 400, None))
    if t >= 24:
        out += [("Orrin", 1410, 60, RED), ("Pell", 1640, 60, BLUE)]
    if t >= 40:
        out += [("Ash", 1410, 90, BLUE), ("Kip", 1640, 90, RED)]
    if 50 <= t < 53:
        out.append(("Zone held", 700, 960, WHITE))
    return out


def _width(text: str) -> int:
    return min(200, 12 * len(text))


def _paint(t: float, frame: np.ndarray) -> None:
    """Draw the coloured lines of the screen at ``t`` (decoded HUD area pixels)."""
    for text, x, y, colour in _screen(t):
        if colour is not None:
            x0, y0 = x - ORIGIN[0], y - ORIGIN[1]
            frame[y0 : y0 + 24, x0 : x0 + _width(text)] = colour


class _Engine:
    """Detection and recognition of the screen at the decoder's current time; a crop
    is told apart by its size. ``calls`` counts every OCR call."""

    def __init__(self) -> None:
        self.t = 0.0
        self.calls = 0

    def _boxes(self, crop: np.ndarray) -> list[tuple[str, list[list[float]]]]:
        self.calls += 1
        h, w = crop.shape[:2]
        x0, y0, _, _ = next(r for r in (NOTICE, FEED, BANNER) if (r[2], r[3]) == (w, h))
        return [
            (text, [[x - x0, y - y0], [x - x0 + bw, y - y0], [x - x0 + bw, y - y0 + 24]])
            for text, x, y, _ in _screen(self.t)
            if x0 <= x < x0 + w and y0 <= y < y0 + h
            for bw in [_width(text)]
        ]

    def detect(self, crop: np.ndarray, scale: float = 1.0) -> list[list[list[float]]]:
        return [b for _, b in self._boxes(crop)]

    def read_rows(self, crop, rows, min_score, pad=6.0) -> list[tuple[list, str, float]]:
        boxes = self._boxes(crop)
        out = []
        for y0, x0, y1, x1 in rows:
            for text, b in boxes:
                if y0 <= (b[0][1] + b[2][1]) / 2 <= y1 and x0 <= b[0][0] <= x1:
                    out.append((b, text, 0.95))
        return out

    def __call__(self, crop: np.ndarray, min_score: float) -> list[tuple[list, str, float]]:
        return [(b, text, 0.9) for text, b in self._boxes(crop)]


class _Stop(Exception):
    """The read is interrupted."""


def _run(
    cdir: Path,
    monkeypatch,
    engine: _Engine | None,
    preset: dict[str, Any] | None = None,
    fps: float = FPS,
    avoid: list[tuple[float, float]] | None = None,
    stop_after: int | None = None,
    starts: list[float] | None = None,
    fail_at: float | None = None,
    cut_after: int | None = None,
    on_frame: Any = None,
    isolated: list[bool] | None = None,
    exit_after: int | None = None,
    fail_hw_at: float | None = None,
    on_cut: Any = None,
    gpu: bool = True,
    det_skip: bool = False,
    **kwargs: Any,
) -> tuple[list[dict], dict]:
    """Detect the OCR events of a 60 s source whose frames are moving gameplay (a new
    grey level every frame) except for the still menu at 8-20 s. The decoder starts
    at ``start`` (logged in ``starts``) and raises :class:`_Stop` after ``stop_after``
    frames; a decode starting at ``fail_at`` gives no frames. After ``cut_after``
    frames a pass ends quietly (ffmpeg stopped by the console); ``on_frame`` is called
    with the number of frames made, and ``isolated`` gets each pass's ``isolate``. After
    ``exit_after`` frames ffmpeg exits with code 255 that the decode did not cause; a CUDA
    pass starting at ``fail_hw_at`` gives no frames (the CPU pass does); ``on_cut`` is
    called when a pass ends after ``cut_after`` frames. ``gpu`` False: no GPU decoder fits the
    source, so a ``hw`` pass is a CPU pass. ``det_skip``: ``OcrParams.det_skip``."""
    data = preset or _preset()
    monkeypatch.setattr(killfeed, "load_preset", lambda *_a, **_k: preset_from_dict(data))
    monkeypatch.setattr(
        killfeed, "probe", lambda _p: SimpleNamespace(width=1920, height=1080, duration=60.0)
    )
    monkeypatch.setattr(
        killfeed, "decoder_for", lambda _p: ffmpeg_utils.HwDecoder("cuda") if gpu else None
    )
    made = [0]

    def frames(path, fps, rect, hw, errors, start=0.0, dur=None, cancel=None, exit_codes=None):
        _, _, w, h = rect
        if starts is not None:
            starts.append(start)
        if isolated is not None:
            isolated.append(cancel is not None)
        if start == fail_at or (hw and start == fail_hw_at):
            raise RuntimeError("no frames")
        end = 60.0 if dur is None else start + dur
        k = 0
        while start + k / fps < end - 1e-6:
            if stop_after is not None and made[0] >= stop_after:
                raise _Stop
            if cut_after is not None and made[0] >= cut_after:
                if on_cut is not None:
                    on_cut()
                return
            if exit_after is not None and made[0] >= exit_after:
                if exit_codes is not None:  # only a background read asks for them
                    exit_codes.append(255)  # Ctrl+Break: one frame short at most
                return
            made[0] += 1
            if on_frame is not None:
                on_frame(made[0])
            t = start + k / fps  # as ffmpeg_utils.decode_raw stamps it
            tc = round(t * fps) / fps  # the screen's time, however t was summed up
            made_by = kwargs.get("make_engine")
            current = engine or (made_by.engines[-1] if made_by and made_by.engines else None)
            if current:
                current.t = tc
            frame = np.full((h, w, 3), 90 if 8 <= tc < 20 else round(t * fps) * 53 % 256, np.uint8)
            _paint(tc, frame)
            yield t, frame
            k += 1

    monkeypatch.setattr(killfeed, "iter_frames", frames)
    src = Source(cdir / "rec.mp4", 60.0, cdir, [])
    params = OcrParams(game="zonewar", fps=fps, det_skip=det_skip)
    return detect_ocr_events(src, params, engine, avoid=avoid, **kwargs)


def _dirs(tmp_path: Path, *names: str) -> list[Path]:
    out = [tmp_path / n for n in names]
    for d in out:
        d.mkdir()
    return out
