"""The voice layer reuses the PANNs detector the audio layer loaded (#372)."""

from __future__ import annotations

import logging
import os
import types
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
    path.write_bytes(b"")  # needs_run checks only that the file exists at the keyed path


@pytest.fixture
def run(tmp_path: Path, monkeypatch):
    """``run_events`` over ``srcs`` (audio always a cache miss); returns the detectors made
    and a log of ``made``/``detect``/``close``/``voice``/``after`` steps. Stops before
    the visual sheets ("after")."""

    def go(
        srcs: list[Source],
        layers: tuple[str, ...],
        fail: bool = False,
        load_fail: bool = False,
        stage=None,
    ):
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
                if load_fail:
                    log.append(("load_failed", None))
                    raise RuntimeError("panns.pth cannot be loaded safely")
                made.append(self)
                log.append(("made", self))

            def close(self):
                log.append(("close", self))

        def detect(s, det, params, force=False):
            log.append(("detect", det))
            if fail:
                raise RuntimeError("boom")
            return []

        def run_layer(sources, params, force=False, det=None, errors=None):
            log.append(("voice", det))
            if det is None and voice.needs_run(sources, params, force):
                Det()  # loads its own PANNs, as the real one does
            return {}

        def stop(project, sources):
            log.append(("after", None))
            raise _Stop

        monkeypatch.setattr(pipeline, "PannsDetector", Det)
        monkeypatch.setattr(pipeline, "detect_audio_events", detect)
        monkeypatch.setattr(voice, "run_layer", run_layer)
        monkeypatch.setattr(sheets, "write_visual_sheets", stop)
        stops = RuntimeError if (fail or load_fail) and stage is None else _Stop
        with pytest.raises(stops):
            pipeline.run_events(tmp_path, layers=layers, discover=False, sheets=True, stage=stage)
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


@pytest.mark.parametrize("cached", [False, True], ids=["voice-needs-panns", "cached-voice"])
def test_a_failed_panns_load_in_analyze_is_tried_once_and_fails_the_step_once(
    tmp_path: Path, run, monkeypatch, caplog, cached
):
    """#629: the voice layer does not load PANNs again after the audio layer failed to."""
    from cutpoint import analyze, progress  # noqa: PLC0415

    warned: list[str] = []
    monkeypatch.setattr(
        analyze.jsonout, "warn", lambda code, message, files=(): warned.append(code)
    )
    srcs = [_source(tmp_path, "a"), _source(tmp_path, "b")]
    if cached:
        for src in srcs:
            _cache_voice(src)
    caplog.set_level(logging.INFO)
    with progress.run(("reactions",)) as steps:
        made, log = run(srcs, ("audio", "voice"), load_fail=True, stage=analyze._StepHook(steps))
    assert made == [] and log.count(("load_failed", None)) == 1
    assert warned == ["step_failed"]
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "cannot be loaded safely" in errors[0].getMessage()
    skipped = [r for r in caplog.records if "Voice layer skipped" in r.getMessage()]
    if cached:  # nothing to compute: the voice layer runs as before and loads nothing
        assert [d for k, d in log if k == "voice"] == [None] and skipped == []
    else:
        assert [k for k, _ in log if k == "voice"] == []
        assert [r.levelno for r in skipped] == [logging.INFO]


def test_a_failed_panns_load_in_events_stops_before_the_voice_layer(tmp_path: Path, run):
    made, log = run([_source(tmp_path, "a")], ("audio", "voice"), load_fail=True)
    assert made == [] and log == [("load_failed", None)]


def test_needs_run_checks_the_cache_files(tmp_path: Path):
    a, b = _source(tmp_path, "a"), _source(tmp_path, "b")
    params = voice.VoiceParams()
    _cache_voice(a)
    assert not voice.needs_run([a], params)
    assert voice.needs_run([a], params, force=True)
    assert voice.needs_run([a, b], params)


def test_needs_run_reruns_a_changed_source(tmp_path: Path, monkeypatch):
    """The cache path carries the source key (path, size, mtime): a changed source misses
    the old grids (#661)."""
    monkeypatch.setenv("CUTPOINT_CACHE_DIR", str(tmp_path / "cache"))
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x" * 10)
    os.utime(video, ns=(1, 10**18))
    params = voice.VoiceParams()
    _cache_voice(common.load_source(video))
    assert not voice.needs_run([common.load_source(video)], params)  # unchanged
    os.utime(video, ns=(1, 2 * 10**18))
    assert voice.needs_run([common.load_source(video)], params)  # mtime changed
    _cache_voice(common.load_source(video))
    video.write_bytes(b"y" * 20)
    os.utime(video, ns=(1, 2 * 10**18))  # same mtime, new size
    assert voice.needs_run([common.load_source(video)], params)


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


# --- the checkpoint is loaded with weights_only (#598) ---

RAN: list[str] = []


def _mark(text: str) -> str:
    RAN.append(text)
    return text


class _Evil:
    def __reduce__(self):
        return (_mark, ("executed",))


@pytest.fixture
def panns_ckpt(tmp_path: Path, monkeypatch):
    torch = pytest.importorskip("torch")
    from cutpoint.events import audio_events, paths  # noqa: PLC0415

    class Net(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)

    ckpt = tmp_path / "panns.pth"
    monkeypatch.setattr(paths, "panns_ckpt", lambda: ckpt)
    monkeypatch.setattr(
        audio_events,
        "_import_panns_models",
        lambda: types.SimpleNamespace(Cnn14_DecisionLevelMax=Net),
    )
    RAN.clear()
    return torch, audio_events, ckpt, Net


def test_panns_checkpoint_with_a_plain_state_dict_loads(panns_ckpt):
    torch, audio_events, ckpt, Net = panns_ckpt
    torch.save({"model": Net().state_dict()}, ckpt)
    det = audio_events.PannsDetector(device="cpu")
    assert isinstance(det.model, Net)


def test_panns_checkpoint_with_code_in_it_is_refused_unexecuted(panns_ckpt):
    torch, audio_events, ckpt, Net = panns_ckpt
    torch.save({"model": Net().state_dict(), "extra": _Evil()}, ckpt)
    with pytest.raises(RuntimeError, match="cannot be loaded safely"):
        audio_events.PannsDetector(device="cpu")
    assert RAN == []


def test_panns_stub_package_keeps_the_events_extra_check_working(tmp_path: Path, monkeypatch):
    """#669: the ``panns_inference`` stub has a spec, so a later ``find_spec`` works."""
    import importlib.util  # noqa: PLC0415
    import sys  # noqa: PLC0415

    from cutpoint.events import audio_events  # noqa: PLC0415

    pkg = tmp_path / "panns_inference"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "raise RuntimeError('package __init__ ran')\n", encoding="utf-8"
    )
    (pkg / "models.py").write_text("class Cnn14_DecisionLevelMax: ...\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("panns_inference", "panns_inference.models"):
        monkeypatch.setitem(sys.modules, name, None)  # undo restores or removes the key
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(pipeline, "EVENTS_MODULES", ("panns_inference",))

    models = audio_events._import_panns_models()

    assert models.Cnn14_DecisionLevelMax
    assert importlib.util.find_spec("panns_inference") is not None
    pipeline._require_events_extra(("audio",))
