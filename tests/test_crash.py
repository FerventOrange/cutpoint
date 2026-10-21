"""Crash detection decisions on small synthetic IMU / GPS / keyframe inputs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

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
