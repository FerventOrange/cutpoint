"""The voice layer reuses the PANNs detector the audio layer loaded (#372)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cutpoint.events import common, pipeline, sheets, voice
from cutpoint.events.common import AudioPart, Source


class _Stop(Exception):
    pass


def _source(tmp_path: Path, name: str) -> Source:
    return Source(
        path=tmp_path / f"{name}.mp4",
        duration=60.0,
        cdir=tmp_path / name,
        audio_parts=[AudioPart(0.0, tmp_path / f"{name}.wav", 60.0)],
    )


def _cache_voice(src: Source) -> None:
    path = voice.cache_path(src, voice.VoiceParams())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")  # needs_run checks only that the file exists


@pytest.fixture
def run(tmp_path: Path, monkeypatch):
    """``run_events`` over ``srcs`` (audio always a cache miss); returns the detectors made
    and a log of ``made``/``detect``/``close``/``voice``/``after`` steps. Stops before
    the visual sheets ("after")."""

    def go(srcs: list[Source], layers: tuple[str, ...], fail: bool = False):
        by_path = {str(s.path): s for s in srcs}
        monkeypatch.setattr(
            pipeline, "load_highlights", lambda p: {"sources": [{"path": k} for k in by_path]}
        )
        monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
        monkeypatch.setattr(pipeline, "load_source", lambda path, dur: by_path[str(path)])
        monkeypatch.setattr(pipeline, "bad_ranges", lambda *a: {})
        monkeypatch.setattr(pipeline, "load_transcript_lines", lambda project: [])
        monkeypatch.setattr(pipeline, "unload_ollama", list)
        monkeypatch.setattr(pipeline, "cached_audio_events", lambda s, p: None)
        monkeypatch.setattr(common, "gpu_memory_used_mib", lambda: None)
        made: list[object] = []
        log: list[tuple[str, object]] = []

        class Det:
            device = "cpu"

            def __init__(self):
                made.append(self)
                log.append(("made", self))

            def close(self):
                log.append(("close", self))

        def detect(s, det, params, force=False):
            log.append(("detect", det))
            if fail:
                raise RuntimeError("boom")
            return []

        def run_layer(sources, params, force=False, det=None):
            log.append(("voice", det))
            return {}

        def stop(project, sources):
            log.append(("after", None))
            raise _Stop

        monkeypatch.setattr(pipeline, "PannsDetector", Det)
        monkeypatch.setattr(pipeline, "detect_audio_events", detect)
        monkeypatch.setattr(voice, "run_layer", run_layer)
        monkeypatch.setattr(sheets, "write_visual_sheets", stop)
        with pytest.raises(RuntimeError if fail else _Stop):
            pipeline.run_events(tmp_path, layers=layers, discover=False, sheets=True)
        return made, log

    return go


def test_voice_gets_the_audio_detector_and_it_is_closed_once(tmp_path: Path, run):
    made, log = run([_source(tmp_path, "a"), _source(tmp_path, "b")], ("audio", "voice"))
    assert len(made) == 1
    det = made[0]
    assert log == [
        ("made", det),
        ("detect", det),
        ("detect", det),
        ("voice", det),
        ("close", det),
        ("after", None),
    ]


def test_without_voice_the_detector_is_closed_after_audio(tmp_path: Path, run):
    made, log = run([_source(tmp_path, "a")], ("audio",))
    det = made[0]
    assert log == [("made", det), ("detect", det), ("close", det), ("after", None)]


def test_cached_voice_grids_close_the_detector_after_audio(tmp_path: Path, run):
    src = _source(tmp_path, "a")
    _cache_voice(src)
    made, log = run([src], ("audio", "voice"))
    det = made[0]
    assert log == [
        ("made", det),
        ("detect", det),
        ("close", det),
        ("voice", None),
        ("after", None),
    ]


def test_a_failed_audio_layer_closes_the_detector(tmp_path: Path, run):
    made, log = run([_source(tmp_path, "a")], ("audio", "voice"), fail=True)
    det = made[0]
    assert log == [("made", det), ("detect", det), ("close", det)]


def test_needs_run_checks_the_cache_files(tmp_path: Path):
    a, b = _source(tmp_path, "a"), _source(tmp_path, "b")
    params = voice.VoiceParams()
    _cache_voice(a)
    assert not voice.needs_run([a], params)
    assert voice.needs_run([a], params, force=True)
    assert voice.needs_run([a, b], params)


def test_run_layer_leaves_a_given_detector_open(tmp_path: Path, monkeypatch):
    closed: list[bool] = []

    class Det:
        device = "cuda"

        def close(self):
            closed.append(True)

    monkeypatch.setattr(voice, "SileroVad", lambda: None)
    monkeypatch.setattr(voice, "compute_voice", lambda s, det, vad, params, force=False: {})
    stats = voice.run_layer([_source(tmp_path, "a")], voice.VoiceParams(), det=Det())
    assert stats == {"cached": 0, "computed": 1, "device": "cuda"} and closed == []
