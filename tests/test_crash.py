"""Crash detection decisions on small synthetic IMU / GPS / keyframe inputs."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint import crash

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
