"""Helpers shared by test modules, moved out of test_planning.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import media_info
from cutpoint import boundaries, clips, planning
from cutpoint.events.tails import EventTailSettings
from cutpoint.tail import TailSettings


@pytest.fixture
def readable(monkeypatch):
    """Readable part of every source: ``ranges[0]`` (default the whole source)."""
    ranges: list[tuple[float, float] | None] = [None]
    monkeypatch.setattr(planning.sources, "readable_range", lambda src, t, d: ranges[0] or (0.0, d))
    return ranges


@pytest.fixture
def speech_env(monkeypatch, readable, tmp_path):
    """One utterance 10-14 s in every source; records the speech-map and words calls."""
    calls: dict[str, list[Any]] = {"maps": [], "words": []}

    def fake_map(src, words, events, settings):
        calls["maps"].append((src, words, events))
        span = boundaries.Span(10.0, 14.0, frozenset({"word"}), "a long sentence here")
        return boundaries.SpeechMap([span], settings)

    def fake_words(src, language=None):
        calls["words"].append(src)
        return [{"start": 10.0, "end": 14.0, "word": "sentence"}]

    monkeypatch.setattr(planning.boundaries, "build_speech_map", fake_map)
    monkeypatch.setattr(planning.sources, "source_words", fake_words)
    return tmp_path, calls


# -- clips.plan_clips ----------------------------------------------------------------
SRC_A, SRC_B = r"D:\a.mp4", r"D:\b.mp4"


def _moment(cid: str, source: str, start: float, end: float, **extra: Any) -> dict[str, Any]:
    return {"id": cid, "source": source, "start": start, "end": end, "score": 1.0, **extra}


@pytest.fixture
def plan_env(monkeypatch, speech_env):
    """plan_clips inputs for two sources; every source is fully readable, no ffmpeg."""
    monkeypatch.setattr(planning.sources, "readable_end", lambda src, t, d: d)
    monkeypatch.setattr(planning.sources, "audio_window", lambda *a: pytest.fail("audio read"))
    project, calls = speech_env

    def run(ordered, **kw: Any) -> list[clips.ClipPlan]:
        kw.setdefault("tail", TailSettings(enabled=False))
        kw.setdefault("speech", boundaries.SpeechSettings(enabled=False))
        return clips.plan_clips(
            ordered,
            project,
            io_paths={SRC_A: Path("a.mp4"), SRC_B: Path("b.mp4")},
            infos={SRC_A: media_info(duration=100.0), SRC_B: media_info(duration=50.0)},
            events=None,
            event_settings=EventTailSettings(enabled=False),
            **kw,
        )

    return run, calls
