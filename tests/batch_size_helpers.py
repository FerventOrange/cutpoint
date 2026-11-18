"""Helpers shared by test modules, moved out of test_batch_size.py."""

from __future__ import annotations

import shutil
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import cutpoint.analyze
from conftest import write
from cutpoint import cache
from cutpoint.ffmpeg_utils import MediaInfo


def _without(data: dict, *keys: str) -> dict:
    return {k: v for k, v in data.items() if k not in keys}


class _Whisper:
    """Fake Whisper: loaded on the first transcript after a close (counted)."""

    model_name, language, device = "fake", None, "cpu"
    loads = closes = 0
    calls: list[str] = []  # noqa: RUF012 - reset by the fixture
    stop: int | None = None  # KeyboardInterrupt on this transcribe call (1-based)

    def __init__(self, **_kw: Any) -> None:
        self.loaded = False

    def transcribe(self, wav: Path, on_progress=None) -> list[dict]:
        type(self).calls.append(wav.parent.name)
        if len(self.calls) == self.stop:
            raise KeyboardInterrupt
        if not self.loaded:
            type(self).loads += 1
            self.loaded = True
        n = len(wav.parent.name)
        return [{"start": 1.0, "end": 2.0 + n % 3, "text": f"hi {wav.parent.name}", "words": []}]

    def close(self) -> None:
        if self.loaded:
            type(self).closes += 1
        self.loaded = False


@pytest.fixture
def analyze_run(home, tmp_path, monkeypatch):
    """``analyze videos --out p`` over five sources with Whisper and the per-source analysis
    faked (scores differ per source); ``state.fresh()`` drops the sources' caches."""
    monkeypatch.chdir(tmp_path)
    videos = [write(tmp_path / "videos" / f"{n}.mp4", n * 3) for n in "abcde"]

    def analyze_source(src, style, tr, use_scenes, max_cand, timings, **kw):
        cdir = cache.source_dir(src)
        segments = cutpoint.analyze._transcript(cdir / "audio16k.wav", cdir, tr)
        kw["on_transcript"]()
        info = MediaInfo(
            src, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 1, None, None, None
        )
        score = 40.0 + 7 * videos.index(src)
        cand = cutpoint.analyze.Candidate(src, 10.0, 20.0, 15.0, 1.0, {"loudness": 1.0}, score)
        return cutpoint.analyze.SourceResult(src, info, 0.0, 60.0, [cand], segments, None)

    _Whisper.loads = _Whisper.closes = 0
    _Whisper.calls = []
    _Whisper.stop = None
    monkeypatch.setattr(cutpoint.analyze, "Transcriber", _Whisper)
    monkeypatch.setattr(cutpoint.analyze, "analyze_source", analyze_source)
    monkeypatch.setattr(cutpoint.analyze, "plan_layers", lambda *a: cutpoint.analyze.LayerPlan())

    def fresh() -> None:
        for v in videos:
            shutil.rmtree(cache.source_dir(v), ignore_errors=True)
        _Whisper.loads = _Whisper.closes = 0
        _Whisper.calls.clear()

    def outputs() -> tuple[dict, bytes]:
        hl = cache.load_json(project / "highlights.json")
        hl = _without(hl, "created", "timings")
        # the vision step's own seconds (#909) and the time budget made of them differ per run
        if isinstance(hl["settings"].get("vision"), dict):
            hl["settings"] = hl["settings"] | {
                "vision": _without(hl["settings"]["vision"], "vision_s", "pre_vision_s", "budget_s")
            }
        return hl, (project / "transcript.txt").read_bytes()

    project = home / "projects" / "p"
    argv = ["analyze", "videos", "--style", "funny", "--no-vision", "--out", "p"]
    return SimpleNamespace(videos=videos, project=project, argv=argv, fresh=fresh, outputs=outputs)
