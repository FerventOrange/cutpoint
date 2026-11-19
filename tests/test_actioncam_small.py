"""Action-cam fixes of audit #1085 (A185, A191, A192, A686/A315, A721, A722).

A183 (``decode_window`` without a frame) is covered by ``test_small_robustness.py``; A191 and
A192 (why an ask failed, the model behind it) by ``test_activity.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint import cache, crash, insta360, telemetry
from cutpoint.errors import FFmpegError
from cutpoint.telemetry import TelemetryParams

# ----------------------------------------------------------------- A185: read errors are not cached


def _parser(monkeypatch, err: Exception) -> None:
    class Parser:
        def __init__(self, _path):
            raise err

    monkeypatch.setitem(sys.modules, "telemetry_parser", SimpleNamespace(Parser=Parser))


def test_parse_imu_says_none_only_for_a_file_without_telemetry(monkeypatch):
    _parser(monkeypatch, OSError("Unsupported file format"))
    assert telemetry.parse_imu(Path("plain.mp4")) is None
    for err in (OSError("Input/output error"), FileNotFoundError("gone"), RuntimeError("x")):
        _parser(monkeypatch, err)
        with pytest.raises(type(err)):
            telemetry.parse_imu(Path("ride.mp4"))


def _extract_env(home, tmp_path, monkeypatch) -> tuple[Path, Path]:
    src = tmp_path / "ride.mp4"
    src.write_bytes(b"x")
    monkeypatch.setattr(telemetry, "imu_read_path", lambda s: (src, None, src, False))
    stem = cache.source_dir(src) / f"telemetry-{TelemetryParams().key()}"
    return src, stem.with_suffix(".json")


def test_extract_does_not_cache_a_failed_read_as_no_telemetry(home, tmp_path, monkeypatch):
    src, meta = _extract_env(home, tmp_path, monkeypatch)
    _parser(monkeypatch, OSError("Input/output error"))
    with pytest.raises(OSError, match="Input/output"):
        telemetry.extract(src, 10.0)
    assert not meta.exists()  # asked again next time
    _parser(monkeypatch, OSError("Unsupported file format"))
    assert telemetry.extract(src, 10.0) is None
    assert cache.load_json(meta) == {"none": "no_track"}  # a file without telemetry is remembered


def test_extract_reads_again_an_old_none_entry(home, tmp_path, monkeypatch):
    """A bare ``{"none": true}`` may follow a read error of an older run: a cache miss."""
    src, meta = _extract_env(home, tmp_path, monkeypatch)
    cache.save_json(meta, {"none": True})
    cache.save_npz(meta.with_suffix(".npz"))
    _parser(monkeypatch, OSError("Unsupported file format"))
    assert telemetry.extract(src, 10.0) is None
    assert cache.load_json(meta) == {"none": "no_track"}  # read again, now remembered as such
    _parser(monkeypatch, OSError("Input/output error"))
    assert telemetry.extract(src, 10.0) is None  # the new entry is trusted: no read


def test_imu_grids_lets_a_read_error_through_without_caching_it(home, tmp_path, monkeypatch):
    src, _ = _extract_env(home, tmp_path, monkeypatch)
    _parser(monkeypatch, OSError("Input/output error"))
    with pytest.raises(OSError, match="Input/output"):
        crash.imu_grids(src, 10.0)
    assert not crash._imu_cache(src).exists()
    _parser(monkeypatch, OSError("Unsupported file format"))
    assert crash.imu_grids(src, 10.0) is None
    assert crash._imu_cache(src).exists()


def test_imu_grids_reads_again_an_old_none_cache(home, tmp_path, monkeypatch):
    src, _ = _extract_env(home, tmp_path, monkeypatch)
    cache.save_npz(crash._imu_cache(src), none=np.zeros(1))  # older runs: also after an error
    _parser(monkeypatch, OSError("Input/output error"))
    with pytest.raises(OSError, match="Input/output"):
        crash.imu_grids(src, 10.0)
    _parser(monkeypatch, OSError("Unsupported file format"))
    assert crash.imu_grids(src, 10.0) is None
    _parser(monkeypatch, OSError("Input/output error"))
    assert crash.imu_grids(src, 10.0) is None  # the new "none" is trusted


# ----------------------------------------------------------------- A686 / A315: one probe per file


@pytest.fixture
def probes(monkeypatch):
    calls = SimpleNamespace(imu=0, ffprobe=0, streams=["avc1", "gpmd"], error=None)
    telemetry._HAS_TELEMETRY.clear()

    def imu_source(path):
        calls.imu += 1

    def ffprobe(path, **kw):
        calls.ffprobe += 1
        if calls.error:
            raise calls.error
        return {
            "streams": [{"index": i, "codec_tag_string": t} for i, t in enumerate(calls.streams)]
        }

    monkeypatch.setattr(insta360, "imu_source", imu_source)
    monkeypatch.setattr(telemetry, "ffprobe_json", ffprobe)
    yield calls
    telemetry._HAS_TELEMETRY.clear()


def test_has_telemetry_probes_a_file_once(tmp_path, probes):
    video = tmp_path / "ride.mp4"
    video.write_bytes(b"x")
    assert telemetry.has_telemetry(video) and telemetry.has_telemetry(video)
    assert (probes.imu, probes.ffprobe) == (1, 1)
    video.write_bytes(b"xy")  # another size: another file
    probes.streams = ["avc1"]
    assert not telemetry.has_telemetry(video)
    assert not telemetry.has_telemetry(video)
    assert (probes.imu, probes.ffprobe) == (2, 2)


def test_has_telemetry_does_not_remember_a_failed_probe(tmp_path, probes):
    video = tmp_path / "ride.mp4"
    video.write_bytes(b"x")
    probes.error = FFmpegError("busy")
    assert not telemetry.has_telemetry(video)
    probes.error = None
    assert telemetry.has_telemetry(video)  # a later probe works; the failure was not kept


# ----------------------------------------------------------------- A721 / A722: per-source constants


def _source(n: int = 600) -> crash.SourceInputs:
    rng = np.random.default_rng(3)
    imu = {k: rng.random(n) for k in ("acc_g", "gyro_dps", "offaxis_deg", "sat")}
    voice = {k: rng.random(n) for k in ("raw_cheer", "hp_cheer", "hp_laugh")}
    return crash.SourceInputs(
        path=Path("a.mp4"),
        duration=n / 10,
        imu=imu,
        voice=voice,
        visual=SimpleNamespace(t=np.arange(60, dtype=np.float32)),
        kf_crash=np.linspace(0, 1, 60),
    )


def test_source_constants_are_computed_once_and_equal_the_per_candidate_values(monkeypatch):
    inp = _source()
    off = np.convolve(inp.imu["offaxis_deg"], np.ones(10), mode="same")
    assert np.array_equal(inp.offaxis_1s, off) and inp.offaxis_1s is inp.offaxis_1s
    assert np.array_equal(inp.visual_t, inp.visual.t.astype(float))
    assert inp.visual_t.dtype == np.float64 and inp.visual_t is inp.visual_t
    yell, ref_y, ref_l = inp.voice_refs
    v = inp.voice
    assert np.array_equal(yell, np.maximum(v["raw_cheer"], v["hp_cheer"]))
    assert ref_y == max(float(np.percentile(yell, 99)), crash.AUDIO_REF_YELL)
    assert ref_l == max(float(np.percentile(v["hp_laugh"], 99)), crash.AUDIO_REF_LAUGH)
    calls = []
    real = np.percentile
    monkeypatch.setattr(np, "percentile", lambda *a, **k: calls.append(1) or real(*a, **k))
    first = [crash.audio_feature(inp, t, t + 2.0) for t in (5.0, 20.0, 40.0)]
    assert len(calls) == 0  # the references were taken above, once
    assert first[0]["score"] != first[1]["score"] or first[0]["yell"] != first[1]["yell"]


def _voice_source(level: float, loud: float) -> crash.SourceInputs:
    """60 s of PANNs cheer and laughter at ``level``, with a 2 s stretch at ``loud`` (20-22 s,
    3% of the source: the 99th percentile) and the rest of the source as it was."""
    n = 600
    cheer, laugh = np.full(n, level), np.full(n, level)
    cheer[200:220] = laugh[200:220] = loud
    voice = {"raw_cheer": cheer, "hp_cheer": cheer, "hp_laugh": laugh}
    return crash.SourceInputs(path=Path("a.mp4"), duration=n / 10, voice=voice)


def test_the_loudest_moment_of_a_quiet_source_is_no_yell():
    # #1085 A189: a source whose loudest 1% is under the level the voice layer reads as
    # laughter at all; scored against its own 99th percentile it reached 0.5
    inp = _voice_source(5e-5, 1.2e-4)
    assert crash.audio_feature(inp, 20.0, 21.0)["score"] < 0.1  # was 0.5


def test_a_yell_of_normal_footage_scores_as_before():
    # laughter around 1e-3 (the voice layer's calibration footage): above the references' floor
    inp = _voice_source(1e-4, 2e-3)
    _, ref_y, ref_l = inp.voice_refs
    assert (ref_y, ref_l) == (2e-3, 2e-3)  # the 99th percentile, not the floor
    assert crash.audio_feature(inp, 20.0, 21.0)["score"] == 0.5
    assert crash.audio_feature(inp, 40.0, 41.0)["score"] == 0.0  # the quiet rest


def test_the_audio_references_floor_is_where_the_voice_layer_starts_to_read_laughter():
    from cutpoint.events.voice import VoiceCalibration  # noqa: PLC0415

    cal = VoiceCalibration()
    assert pytest.approx(10**cal.laugh_log_thr) == crash.AUDIO_REF_LAUGH
    assert pytest.approx(10**cal.laugh_log_thr / cal.cheer_weight) == crash.AUDIO_REF_YELL


def test_source_constants_are_none_without_their_layer():
    bare = crash.SourceInputs(path=Path("a.mp4"), duration=10.0)
    assert bare.offaxis_1s is None and bare.visual_t is None and bare.voice_refs is None
    assert crash.audio_feature(bare, 1.0, 2.0) == {"score": 0.0}


def test_telemetry_features_of_a_candidate_are_asked_once_per_cue(monkeypatch):
    inp = _source()
    inp.imu["acc_g"][:] = 0.0
    inp.imu["acc_g"][300:303] = 9.0  # one telemetry candidate near 30 s
    inp.hits = [(31.0, 1.0, "a"), (80.0, 1.0, "b"), (130.0, 1.0, "c")]
    real = crash.telemetry_features
    calls: list[float] = []

    def counting(i, t0, t1, p):
        calls.append(t0)
        return real(i, t0, t1, p)

    monkeypatch.setattr(crash, "telemetry_features", counting)
    crash.candidate_events(inp, crash.CrashParams())
    assert len(calls) == len(set(calls))  # never twice for one candidate


# ----------------------------------------------------------------- read errors do not stop the steps


def test_crash_inputs_without_telemetry_when_it_cannot_be_read(home, tmp_path, monkeypatch, caplog):
    """A185: a read error scores the source without telemetry (WARNING, nothing cached)."""
    import logging  # noqa: PLC0415

    from cutpoint import jsonout  # noqa: PLC0415

    src, meta = _extract_env(home, tmp_path, monkeypatch)
    _parser(monkeypatch, OSError("Input/output error"))
    with caplog.at_level(logging.WARNING), jsonout.run("crashes") as result:
        inp = crash.load_inputs(src, 10.0, crash.CrashParams())
    assert inp.tel is None and inp.imu is None
    lines = [r.getMessage() for r in caplog.records if "telemetry not read" in r.getMessage()]
    assert len(lines) == 1 and "ride.mp4: telemetry not read (Input/output error)" in lines[0]
    assert len([w for w in result.warnings if w["code"] == "step_failed"]) == 1
    assert not meta.exists() and not crash._imu_cache(src).exists()


def test_a_forced_imu_read_error_does_not_leave_the_old_grids_in_use(
    home, tmp_path, monkeypatch, caplog
):
    """F3: ``crashes --force`` whose IMU read fails drops the old cache, so the source is
    read again and warned for instead of scored from the old grids silently."""
    import json  # noqa: PLC0415
    from dataclasses import replace  # noqa: PLC0415

    from cutpoint import mix, sources  # noqa: PLC0415
    from cutpoint.events import visual, voice  # noqa: PLC0415

    src, _ = _extract_env(home, tmp_path, monkeypatch)
    project = tmp_path / "proj"
    project.mkdir()
    hl = {"sources": [{"path": str(src), "duration": 60.0}]}
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    cache.save_npz(crash._imu_cache(src), acc_g=np.ones(5, np.float32))  # an earlier run's
    _parser(monkeypatch, OSError("Input/output error"))
    monkeypatch.setattr(telemetry, "extract", lambda s, d: None)
    monkeypatch.setattr(voice, "load_voice", lambda s: None)
    monkeypatch.setattr(visual, "load_visual", lambda s: None)
    monkeypatch.setattr(sources, "source_words", lambda s, lang: None)
    monkeypatch.setattr(mix, "lift_rides", lambda s, n: None)
    monkeypatch.setattr(crash, "candidate_events", lambda inp, p: [])
    own_words = replace(crash.CrashParams(), pos=("a",), neg=("b",), words=())
    with caplog.at_level("WARNING", logger="cutpoint.crash"):
        crash.detect_project(project, params=own_words, dense=False, sheets=0, force=True)
    lines = [r.getMessage() for r in caplog.records if "telemetry not read" in r.getMessage()]
    assert len(lines) == 1 and "Input/output error" in lines[0]
    assert not crash._imu_cache(src).exists()


def test_imu_grids_falls_back_to_the_full_file_when_the_gpmf_copy_fails(
    home, tmp_path, monkeypatch
):
    src = tmp_path / "ride.mp4"
    src.write_bytes(b"x")
    copy = tmp_path / "copy.gpmf.mp4"
    parsed: list[Path] = []
    grids = {"imu_hz": np.array([200.0])}

    def compute(path, duration, force=False):
        parsed.append(path)
        if path == copy:
            raise OSError("Input/output error")
        return grids

    monkeypatch.setattr(telemetry, "imu_read_path", lambda s: (src, None, src, True))
    monkeypatch.setattr(crash, "_gpmf_copy", lambda io, tmp: copy)
    monkeypatch.setattr(crash, "_compute_imu", compute)
    assert crash.imu_grids(src, 10.0) is not None
    assert parsed == [copy, src]  # the full file was parsed after the copy failed
    # the full file failing too raises, and nothing is cached
    crash._imu_cache(src).unlink()

    def both_fail(path, duration, force=False):
        raise OSError("Input/output error")

    monkeypatch.setattr(crash, "_compute_imu", both_fail)
    with pytest.raises(OSError, match="Input/output"):
        crash.imu_grids(src, 10.0)
    assert not crash._imu_cache(src).exists()
