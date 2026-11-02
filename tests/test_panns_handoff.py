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


@pytest.mark.parametrize(
    ("layers", "cached", "fail", "voice_gets"),
    [
        (("audio", "voice"), False, False, "det"),  # voice reuses the audio detector
        (("audio",), False, False, "absent"),
        (("audio", "voice"), True, False, "none"),  # cached grids: freed before voice
        (("audio", "voice"), False, True, "absent"),  # a failed audio layer still closes it
    ],
    ids=["voice", "no-voice", "cached-voice", "audio-fails"],
)
def test_one_audio_detector_is_shared_and_closed_once_before_it_is_not_needed(
    tmp_path: Path, run, layers, cached, fail, voice_gets
):
    srcs = [_source(tmp_path, "a"), _source(tmp_path, "b")]
    if cached:
        for src in srcs:
            _cache_voice(src)
    made, log = run(srcs[:1] if fail else srcs, layers, fail=fail)
    (det,) = made
    assert [d for k, d in log if k == "detect"] == [det] * (1 if fail else 2)
    assert log.count(("close", det)) == 1
    voiced = [d for k, d in log if k == "voice"]
    assert voiced == {"det": [det], "none": [None], "absent": []}[voice_gets]
    if voiced:
        before = log.index(("close", det)) < log.index(("voice", voiced[0]))
        assert before is (voice_gets == "none")
    if fail:
        assert ("after", None) not in log
    else:
        assert log.index(("close", det)) < log.index(("after", None))


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
