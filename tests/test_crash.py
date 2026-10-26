"""Crash detection decisions on small synthetic IMU / GPS / keyframe inputs."""

from __future__ import annotations

import functools
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint import config, crash, stage, telemetry
from cutpoint.pano import x3

N = 600  # 60 s on the 0.1 s grid


def _inputs(*, spike: bool) -> crash.SourceInputs:
    acc = np.zeros(N)
    speed = np.full(N, 30.0)  # skiing at 30 km/h ...
    kf = np.zeros(60)  # ... one keyframe per second, none looks like a crash
    if spike:
        acc[300:303] = 8.0  # 8 g impact at t = 30 s
        speed[303:] = 0.0  # then standing still
        kf[30:34] = 0.95  # and the camera looks like a crash
    imu = {k: np.zeros(N) for k in ("gyro_dps", "offaxis_deg", "sat")}
    imu["acc_g"] = acc
    tel = SimpleNamespace(n=N, grid={"speed_kmh": speed, "lift_mask": np.zeros(N)})
    visual = SimpleNamespace(t=np.arange(60, dtype=float))
    return crash.SourceInputs(
        path=Path("a.mp4"), duration=60.0, tel=tel, imu=imu, visual=visual, kf_crash=kf
    )


def test_impact_with_stop_and_visual_change_is_a_candidate_above_the_threshold():
    p = crash.CrashParams()
    inp = _inputs(spike=True)
    (cand,) = crash.candidate_events(inp, p)
    assert cand["origin"] == "telemetry" and 29.5 <= cand["t0"] <= cand["t1"] <= 31.0
    scored = crash.score_candidate(inp, cand, p)
    assert scored["pre_score"] >= p.prefilter
    clip = crash.plan_clip(inp, cand, p)
    assert clip["start"] == cand["t0"] - p.lead_s  # the approach is kept
    assert clip["end"] >= cand["t1"] + p.min_after_s  # and the aftermath


def test_quiet_window_has_no_candidates():
    assert crash.candidate_events(_inputs(spike=False), crash.CrashParams()) == []


@pytest.mark.parametrize(("ride_start_s", "damped"), [(32, True), (45, False)])
def test_impact_just_before_a_lift_ride_is_damped_without_gps(ride_start_s, damped):
    """Boarding knocks (360, no GPS lift mask): CLIP lift rides damp events up to 5 s before."""
    p = crash.CrashParams()
    inp = _inputs(spike=True)
    (cand,) = crash.candidate_events(inp, p)
    base = crash.telemetry_features(inp, cand["t0"], cand["t1"], p)
    assert not base["on_lift"]
    inp.lift = np.zeros(N)
    inp.lift[ride_start_s * 10 :] = 1.0
    tel = crash.telemetry_features(inp, cand["t0"], cand["t1"], p)
    assert tel["on_lift"] is damped
    assert tel["score"] == pytest.approx(base["score"] * (0.2 if damped else 1.0), abs=2e-3)


def _project(tmp_path: Path, *sources: Path) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    hl = {"sources": [{"path": str(s), "duration": 60.0} for s in sources]}
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    return project


def test_detect_project_all_footage_offline_fails_before_writing(tmp_path):
    project = _project(tmp_path, tmp_path / "gone.mp4")
    with pytest.raises(ValueError, match="No footage of proj is reachable"):
        crash.detect_project(project)
    assert not (project / "crashes.json").exists()
    assert not (project / "activity.json").exists()


def test_detect_project_skips_offline_source_and_processes_the_rest(tmp_path, monkeypatch, caplog):
    here = tmp_path / "here.mp4"
    here.write_bytes(b"x")
    project = _project(tmp_path, tmp_path / "gone.mp4", here)
    loaded = []
    monkeypatch.setattr(
        crash, "load_inputs", lambda src, dur, p, tmp: loaded.append(src) or _inputs(spike=False)
    )
    monkeypatch.setattr(crash, "candidate_events", lambda inp, p: [])
    own_words = replace(crash.CrashParams(), words=())  # keeps the activity profile out of it
    with caplog.at_level("WARNING"):
        out = crash.detect_project(project, params=own_words, dense=False)
    assert loaded == [here]
    assert "gone.mp4: footage not reachable" in caplog.text
    assert json.loads(out.read_text(encoding="utf-8"))["n_candidates"] == 0


class _FakeClip:
    def __init__(self):
        self.at = "cuda"
        self.moves = []

    def to(self, device):
        self.at = device
        self.moves.append(device)
        return self


class _FakeModels:
    def __init__(self):
        self.torch, self.device, self.dtype = object(), "cpu", "f16"
        self.clip, self.mean, self.std = _FakeClip(), "mean", "std"
        self.closed = 0

    def close(self):
        self.closed += 1


def test_clip_encoder_from_visual_shares_the_model_parks_it_and_double_close_is_safe(monkeypatch):
    def boom(self, *a, **k):
        raise AssertionError("CLIP must not be loaded again")

    monkeypatch.setattr(crash.ClipEncoder, "__init__", boom)
    monkeypatch.setattr("cutpoint.events.common.free_torch", lambda: None)
    models = _FakeModels()
    models.device = "cuda"
    enc = crash.ClipEncoder.from_visual(models)
    assert enc.model is models.clip and enc.device == "cuda" and enc.mean == "mean"
    assert models.clip.moves == ["cpu"] and not enc.on_device  # parked until the first embed
    empty = np.zeros((0, 8, 8, 3), np.uint8)  # no torch needed to trigger the move back
    assert enc.embed(empty).shape == (0, 3, 768)
    enc.embed(empty)
    assert models.clip.moves == ["cpu", "cuda"] and enc.on_device  # moved back once
    enc.close()
    enc.close()
    assert enc.model is None


@pytest.mark.parametrize(
    ("tel", "expected"),
    [
        ({"speed_before_kmh": 71, "speed_min_after_kmh": 0.4, "on_lift": False}, True),
        ({"speed_before_kmh": 30, "speed_min_after_kmh": 0.4, "on_lift": False}, False),
        ({"speed_before_kmh": 71, "speed_min_after_kmh": 8, "on_lift": False}, False),
        ({"speed_before_kmh": 71, "speed_min_after_kmh": 0.4, "on_lift": True}, False),
        ({"speed_before_kmh": None, "speed_min_after_kmh": None, "on_lift": False}, False),
    ],
)
def test_speed_drop_only_for_a_fast_rider_stopped_dead(tel, expected):
    cand = {"telemetry": tel}
    assert crash._speed_drop(cand, crash.CrashParams()) is expected


def _cand(name, pre, before, after, on_lift=False):
    tel = {"speed_before_kmh": before, "speed_min_after_kmh": after, "on_lift": on_lift}
    return {
        "name": name,
        "source": "s.mp4",
        "t0": 10.0,
        "t1": 12.0,
        "origin": "telemetry",
        "pre_score": pre,
        "telemetry": tel,
        "scores": {"telemetry": 0, "visual": 0, "audio": 0, "words": 0},
    }


def test_detect_project_dense_checks_speed_drops_below_prefilter(tmp_path, monkeypatch):
    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    project = _project(tmp_path, src)
    cands = [
        _cand("high", 0.5, 10, 10),
        _cand("drop1", 0.161, 71, 0.4),
        _cand("slow", 0.16, 30, 0.4),
        _cand("notstopped", 0.15, 71, 8),
        _cand("lift", 0.14, 71, 0.4, on_lift=True),
        *[_cand(f"drop{k}", 0.1, 80, 0.0) for k in range(2, 9)],
    ]
    for c in cands:
        c["source"] = str(src)
    checked, ends = [], []
    enc = SimpleNamespace(close=lambda: ends.append(1))
    monkeypatch.setattr(crash, "load_inputs", lambda *a: _inputs(spike=False))
    monkeypatch.setattr(crash, "candidate_events", lambda inp, p: cands)
    monkeypatch.setattr(crash, "score_candidate", lambda inp, c, p: c)
    monkeypatch.setattr(crash, "plan_clip", lambda i, c, p: {"start": c["t0"], "end": c["t1"]})
    monkeypatch.setattr(crash, "ClipEncoder", lambda: pytest.fail("loaded a new encoder"))

    def dense(e, inp, c, p, store):
        assert e is enc
        checked.append(c["name"])
        return {"score": 0.5}

    monkeypatch.setattr(crash, "dense_feature", dense)
    own_words = replace(crash.CrashParams(), words=())
    crash.detect_project(project, params=own_words, encoder=enc, sheets=0)
    p = crash.CrashParams()
    assert checked == ["high", "drop1", *[f"drop{k}" for k in range(2, 1 + p.drop_top)]]
    assert ends == [1]


def test_detect_project_dense_zero_never_lowers_a_speed_drop_bypass(tmp_path, monkeypatch):
    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    project = _project(tmp_path, src)
    normal, drop = _cand("normal", 0.5, 10, 10), _cand("drop", 0.161, 71, 0.4)
    drop["t0"], drop["t1"] = 40.0, 42.0  # outside the other clip, so it is not absorbed
    for c in (normal, drop):
        c["source"] = str(src)
        c["scores"].update(telemetry=0.8, visual=0.6)
    enc = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(crash, "load_inputs", lambda *a: _inputs(spike=False))
    monkeypatch.setattr(crash, "candidate_events", lambda inp, p: [normal, drop])
    monkeypatch.setattr(crash, "score_candidate", lambda inp, c, p: c)
    monkeypatch.setattr(crash, "plan_clip", lambda i, c, p: {"start": c["t0"], "end": c["t1"]})
    pre = {
        c["name"]: crash._confidence(c, dict(crash.CrashParams().weights)) for c in (normal, drop)
    }
    monkeypatch.setattr(crash, "dense_feature", lambda e, inp, c, p, store: {"score": 0.0})
    own_words = replace(crash.CrashParams(), words=())
    crash.detect_project(project, params=own_words, encoder=enc, sheets=0)
    got = {
        c["name"]: c
        for c in json.loads((project / "crashes.json").read_text(encoding="utf-8"))["crashes"]
    }
    assert got["drop"]["confirmed"] and got["normal"]["confirmed"]
    assert got["drop"]["confidence"] >= round(pre["drop"], 3)  # the bypass is never penalised
    assert got["normal"]["confidence"] < round(pre["normal"], 3)  # a normal candidate still drops


# -- IMU parse shared with telemetry ----------------------------------------------------------


def test_gravity_is_the_unit_low_passed_accelerometer() -> None:
    acc = np.tile([0.0, 9.8, 0.0], (50, 1))
    acc[25] = [0.0, 0.0, 9.8]  # one off-axis sample barely moves a 10-sample mean
    g = telemetry.gravity(acc, 10)
    assert np.allclose(np.linalg.norm(g, axis=1), 1.0)
    assert g[25, 1] > 0.99
    assert np.allclose(telemetry.gravity(acc, 0)[25], [0.0, 0.0, 1.0])  # k < 1: no smoothing


def test_parse_imu_drops_samples_without_both_sensors(monkeypatch) -> None:
    samples = [
        {"timestamp_ms": 5.0 * i, "accl": [0.0, 9.8, 0.0], "gyro": [1.0, 2.0, 3.0]}
        for i in range(12)
    ]
    samples[3]["gyro"] = None

    class Parser:
        def __init__(self, _path):
            pass

        def telemetry(self):
            return [{}]

        def normalized_imu(self):
            return samples

    monkeypatch.setitem(sys.modules, "telemetry_parser", SimpleNamespace(Parser=Parser))
    _, groups, t, acc, gyr = telemetry.parse_imu(Path("a.mp4"))
    assert groups == [{}] and acc.shape == gyr.shape == (11, 3)
    assert t[:4].tolist() == [0.0, 0.005, 0.01, 0.02]
    samples[4:] = []  # 3 usable samples: too few for the crash grids
    assert len(telemetry.parse_imu(Path("a.mp4"))[2]) == 3
    assert crash._compute_imu(Path("a.mp4"), 10.0) is None


@pytest.mark.parametrize(
    ("trailer", "staged", "remote", "read", "unstaged"),
    [
        ("/nas/VID_00_.insv", None, True, "/nas/VID_00_.insv", False),  # X3: the IMU trailer
        (None, "/local/GX01.mp4", True, "/local/GX01.mp4", False),  # the staged copy
        (None, None, True, "/nas/GX01.mp4", True),  # remote, read in place
        (None, None, False, "/nas/GX01.mp4", False),
    ],
)
def test_imu_read_path_prefers_the_x3_trailer_then_the_staged_copy(
    monkeypatch, trailer, staged, remote, read, unstaged
) -> None:
    src = Path("/nas/GX01.mp4")
    monkeypatch.setattr(config, "remap_source", Path)
    monkeypatch.setattr(x3, "imu_source", lambda _p: trailer and Path(trailer))
    monkeypatch.setattr(stage, "staged_path", functools.cache(lambda p: staged or p))
    monkeypatch.setattr(stage, "is_remote", lambda _p: remote)
    io, x3_imu, read_path, got = telemetry.imu_read_path(src)
    assert (io, x3_imu, read_path, got) == (src, trailer and Path(trailer), Path(read), unstaged)
