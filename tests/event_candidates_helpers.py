"""Helpers shared by test modules, moved out of test_event_candidates.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cutpoint import analyze, cache
from cutpoint.events import scoring
from cutpoint.sources import TRANSCRIPT_MODELS
from cutpoint.telemetry import DT
from cutpoint.transcribe import transcript_name

SIGNALS = ("loudness", "speech", "laughter", "scene", "continuity")


# --------------------------------------------------------------------------- helpers
def bump(t: float, n: int, height: float = 1.0, width: float = 1.0) -> np.ndarray:
    x = np.arange(n) * DT
    return height * np.exp(-(((x - t) / width) ** 2))


# --------------------------------------------------------------------------- events
SRC = "game.mp4"


def feed(t: float, **flags: bool) -> dict[str, Any]:
    return {
        "source": SRC, "start": t, "end": t + 3, "kind": "ocr", "type": "kill",
        "label": "Ash > Bob", "player": True, "player_kill": True, "player_death": False,
        "teamkill": False, **flags,
    }  # fmt: skip


def sound(t: float, label: str = "Laughter", confidence: float = 0.4) -> dict[str, Any]:
    return {
        "source": SRC, "start": t, "end": t + 1.5, "kind": "audio", "label": label,
        "confidence": confidence,
    }  # fmt: skip


STYLE = ("gaming",)


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


@pytest.fixture
def no_transcript(monkeypatch):
    monkeypatch.setattr(scoring, "_segments", lambda src, language=None: [])


def nominees(hl: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in hl["candidates"] if scoring.is_nominated(c)]


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
    cache.save_json(cdir / analyze.SCENES_FILE, cuts)


def analysis_of(src: Path, span: float, **entry: Any) -> dict[str, Any]:
    return {
        "style": "gaming",
        "settings": {"scenes": True},
        "sources": [{"path": str(src), "duration": span, **entry}],
        "candidates": [],
    }


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


# --------------------------------------------------------------------------- cycle 2 fixes
def quiet_grid(span: float, offset: float = 0.0, index: int = 0, bumps=()):
    """A part of ``span`` s with loudness bumps ``(second, height)``."""
    n = int(span / DT) + 1
    ps = analyze.PartSignals(n, {k: np.zeros(n) for k in SIGNALS}, [], [], [], [], [])
    for t, h in bumps:
        ps.sig["loudness"] += bump(t, n, h)
    return analyze.GridPart(Path(SRC), index, offset, span, ps)
