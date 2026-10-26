"""The audio layer loads PANNs only for a source it must detect (#370)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cutpoint import cache
from cutpoint.events import audio_events, common, pipeline, sheets
from cutpoint.events.audio_events import AudioParams, cached_audio_events
from cutpoint.events.common import AudioPart, Source, layer_cache_path


class _Stop(Exception):
    pass


def _source(tmp_path: Path, name: str, audio: bool = True) -> Source:
    parts = [AudioPart(0.0, tmp_path / f"{name}.wav", 60.0)] if audio else []
    return Source(
        path=tmp_path / f"{name}.mp4", duration=60.0, cdir=tmp_path / name, audio_parts=parts
    )


def _cache(src: Source, events: list) -> None:
    cpath = layer_cache_path(src.cdir, "audio", AudioParams().key())
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cache.save_json(cpath, {"source": str(src.path), "params": {}, "events": events})


@pytest.fixture
def run(tmp_path: Path, monkeypatch):
    """``run_events`` with the audio layer only over ``srcs``; returns the detector count
    and the ``(source, det, force)`` calls to ``detect_audio_events``. Stops after the audio layer."""

    def go(srcs: list[Source], force: tuple[str, ...] = ()):
        by_path = {str(s.path): s for s in srcs}
        monkeypatch.setattr(
            pipeline, "load_highlights", lambda p: {"sources": [{"path": k} for k in by_path]}
        )
        monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
        monkeypatch.setattr(pipeline, "load_source", lambda path, dur: by_path[str(path)])
        monkeypatch.setattr(pipeline, "bad_ranges", lambda *a: {})
        monkeypatch.setattr(pipeline, "load_transcript_lines", lambda project: [])
        monkeypatch.setattr(pipeline, "unload_ollama", list)
        monkeypatch.setattr(common, "gpu_memory_used_mib", lambda: None)
        made: list[object] = []

        class Det:
            device = "cpu"

            def __init__(self):
                made.append(self)

            def close(self):
                pass

        calls: list[tuple[str, object, bool]] = []

        def detect(s, det, params, force=False):
            calls.append((s.name, det, force))
            return [{"t": 1.0, "end": 2.0, "type": "audio", "label": "laugh"}]

        def stop(project, sources):
            raise _Stop  # the audio layer is done: the rest of the run is not needed

        monkeypatch.setattr(pipeline, "PannsDetector", Det)
        monkeypatch.setattr(pipeline, "detect_audio_events", detect)
        monkeypatch.setattr(sheets, "write_visual_sheets", stop)
        with pytest.raises(_Stop):
            pipeline.run_events(
                tmp_path, layers=("audio",), force=force, discover=False, sheets=True
            )
        return made, calls

    return go


def test_cached_audio_events_tells_a_miss_from_an_empty_hit(tmp_path: Path):
    src = _source(tmp_path, "a")
    assert cached_audio_events(src, AudioParams()) is None
    _cache(src, [])
    assert cached_audio_events(src, AudioParams()) == []
    assert audio_events.detect_audio_events(src, None, AudioParams()) == []


def test_a_cached_empty_result_loads_no_detector(tmp_path: Path, run):
    srcs = [_source(tmp_path, "a"), _source(tmp_path, "b")]
    for s in srcs:
        _cache(s, [])
    made, calls = run(srcs)
    assert made == [] and calls == []


def test_a_cache_miss_loads_the_detector_once(tmp_path: Path, run):
    srcs = [_source(tmp_path, n) for n in "abc"]
    _cache(srcs[1], [])  # b is cached (empty); a and c are misses
    made, calls = run(srcs)
    assert len(made) == 1
    assert calls == [("a.mp4", made[0], True), ("c.mp4", made[0], True)]


def test_force_detects_a_cached_source(tmp_path: Path, run):
    src = _source(tmp_path, "a")
    _cache(src, [])
    made, calls = run([src], force=("audio",))
    assert len(made) == 1 and calls == [("a.mp4", made[0], True)]


def test_a_source_without_audio_loads_no_detector(tmp_path: Path, run, caplog):
    caplog.set_level(logging.WARNING, logger="cutpoint.events.pipeline")
    made, calls = run([_source(tmp_path, "a", audio=False)])
    assert made == [] and calls == []
    assert "a.mp4: no cached 16 kHz audio; run `cutpoint analyze` first" in caplog.text
