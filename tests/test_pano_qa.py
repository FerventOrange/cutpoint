"""Rendered-frame direction QA (``cutpoint 360 qa``): flags, pan metrics, aggregation.

No models, ffmpeg or footage: detections, prompt scores and director paths are
synthetic; :func:`run_qa` runs with its decoder and models monkeypatched.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cutpoint.events.visual as V
import cutpoint.ffmpeg_utils as F
from cutpoint.events.visual import VisualData
from cutpoint.pano import geometry as G
from cutpoint.pano import project, qa

TH = qa.QAThresholds()


def flags_for(
    boxes: list[tuple[float, float, float, float, float, bool]],
    *,
    target: str = "friend",
    gray_std: float = 0.2,
    occl: float = 0.0,
    low: float = 0.0,
    roll: float | None = None,
    th: qa.QAThresholds = TH,
) -> dict:
    """One frame's record for person boxes ``(score, x1, y1, x2, y2, is_self)``."""
    det = np.array([[0, 1, *b[:5]] for b in boxes], float).reshape(-1, 7)
    is_self = np.array([b[5] for b in boxes], bool)
    return qa.frame_flags(
        det,
        is_self,
        1,
        gray_std=np.array([gray_std]),
        occl_zs=np.array([occl]),
        lowinfo_zs=np.array([low]),
        targets=[target],
        roll=None if roll is None else np.array([roll]),
        th=th,
    )[0]


GOOD = (0.99, 0.40, 0.30, 0.60, 0.95, False)  # well framed friend, h 0.65


# --------------------------------------------------------------------------- per-frame flags
def test_clean_frame_has_no_flags_and_a_subject() -> None:
    r = flags_for([GOOD])
    assert r["flags"] == []
    assert r["subj"] == [0.4, 0.3, 0.6, 0.95]
    assert r["size_ok"] is True
    assert r["safe"] is True
    assert r["boxes"] == [[0.99, 0.4, 0.3, 0.6, 0.95, False]]


def test_subject_is_largest_kept_person_not_self_or_tiny_or_weak() -> None:
    r = flags_for(
        [
            (0.99, 0.10, 0.40, 0.20, 0.60, False),  # small friend, area 0.02
            (0.99, 0.30, 0.20, 0.70, 0.95, True),  # wearer's own body (self)
            (0.40, 0.50, 0.10, 0.90, 0.90, False),  # low score
            (0.99, 0.80, 0.50, 0.81, 0.53, False),  # h 0.03 < subject_min_h
        ]
    )
    assert r["subj"] == [0.1, 0.4, 0.2, 0.6]
    assert len(r["self"]) == 1


def test_occluded_by_large_self_box() -> None:
    assert "occluded" in flags_for([GOOD, (0.99, 0.0, 0.3, 0.6, 1.0, True)])["flags"]
    # a small self box (ski tips at the bottom) is fine
    assert "occluded" not in flags_for([GOOD, (0.99, 0.4, 0.8, 0.7, 1.0, True)])["flags"]


def test_wearer_coming_in_from_the_bottom_is_allowed() -> None:
    arms = (0.99, 0.0, 0.62, 1.0, 1.0, True)  # area 0.38, bottom only
    middle = (0.99, 0.2, 0.3, 0.8, 0.9, True)  # area 0.36 mid-frame
    assert qa.wearer_allowed(np.array([arms[1:5], middle[1:5]]).T, TH).tolist() == [True, False]
    assert flags_for([GOOD, arms])["flags"] == []
    assert "occluded" in flags_for([GOOD, middle])["flags"]


def test_occluded_by_prompt_score() -> None:
    assert "occluded" in flags_for([GOOD], occl=TH.occl_zs + 0.05)["flags"]
    assert "occluded" not in flags_for([GOOD], occl=TH.occl_zs - 0.05)["flags"]


def test_too_close_by_area_or_full_height_and_wide() -> None:
    assert "too_close" in flags_for([(0.99, 0.1, 0.1, 0.95, 0.95, False)])["flags"]  # area .72
    assert "too_close" in flags_for([(0.99, 0.2, 0.0, 0.88, 0.97, False)])["flags"]  # h .97 w .68
    # a seated friend filling half the frame (area 0.5) is a medium shot, not too close
    assert "too_close" not in flags_for([(0.99, 0.25, 0.08, 0.80, 0.99, False)])["flags"]
    # self boxes count too
    assert "too_close" in flags_for([(0.99, 0.0, 0.0, 0.9, 1.0, True)])["flags"]


def test_edge_cut_left_right_and_top() -> None:
    assert "edge_cut" in flags_for([(0.99, 0.0, 0.2, 0.4, 0.98, False)])["flags"]
    assert "edge_cut" in flags_for([(0.99, 0.6, 0.2, 1.0, 0.98, False)])["flags"]
    # head cut at the top while the body ends above the bottom
    assert "edge_cut" in flags_for([(0.99, 0.4, 0.0, 0.6, 0.6, False)])["flags"]


def test_edge_cut_ignores_feet_tight_full_height_and_small_background() -> None:
    assert "edge_cut" not in flags_for([(0.99, 0.4, 0.3, 0.6, 1.0, False)])["flags"]  # feet
    assert "edge_cut" not in flags_for([(0.99, 0.4, 0.0, 0.6, 0.97, False)])["flags"]  # tight
    small_bg = (0.99, 0.0, 0.0, 0.15, 0.48, False)  # area 0.07 at the left edge
    r = flags_for([(0.99, 0.35, 0.2, 0.65, 0.98, False), small_bg])
    assert "edge_cut" not in r["flags"]
    big_bg = (0.99, 0.0, 0.0, 0.25, 0.6, False)  # area 0.15
    assert "edge_cut" in flags_for([(0.99, 0.35, 0.2, 0.65, 0.98, False), big_bg])["flags"]
    # a tiny far subject touching the edge is background, not a framing error
    assert "edge_cut" not in flags_for([(0.99, 0.95, 0.5, 1.0, 0.65, False)])["flags"]


def test_edge_cut_ignores_the_wearer_detected_twice() -> None:
    """040_crash t=0: an untagged box around the wearer's self-tagged box at the left edge."""
    twin = (0.99, 0.001, 0.034, 0.206, 0.984, False)
    assert "edge_cut" not in flags_for([twin, (0.99, 0.0015, 0.519, 0.205, 0.994, True)])["flags"]
    assert "edge_cut" in flags_for([twin])["flags"]


def test_low_info_needs_flat_picture_or_confident_prompt() -> None:
    assert "low_info" in flags_for([], gray_std=0.03, low=0.65)["flags"]
    assert "low_info" not in flags_for([], gray_std=0.2, low=0.65)["flags"]
    assert "low_info" in flags_for([], gray_std=0.2, low=0.85)["flags"]


def test_too_far_only_on_friend_target() -> None:
    far = (0.99, 0.45, 0.4, 0.5, 0.5, False)  # h 0.10
    r = flags_for([far], target="friend")
    assert "too_far" in r["flags"]
    assert r["size_ok"] is False
    assert "too_far" not in flags_for([far], target="view")["flags"]


def test_tilt_and_safe_area() -> None:
    assert "tilt" in flags_for([GOOD], roll=-12.5)["flags"]
    assert "tilt" not in flags_for([GOOD], roll=11.0)["flags"]
    assert flags_for([(0.99, 0.02, 0.3, 0.3, 0.95, False)])["safe"] is False


# --------------------------------------------------------------------------- path metrics
def make_path(
    speed: float = 10.0, t0: float = 0.0, t1: float = 20.0, cut_at: float | None = None
) -> dict:
    keys = []
    for t in np.arange(t0, t1 + 1e-6, 0.5):
        k = {"t": float(t), "yaw": speed * (t - t0), "pitch": 0.0, "fov": 86.0, "target": "friend"}
        if cut_at is not None and abs(t - cut_at) < 1e-6:
            k["cut"] = True
        if cut_at is not None and t >= cut_at:
            k["yaw"] += 120.0
            k["target"] = "view"
        keys.append(k)
    return {"keys": keys, "cuts": [cut_at] if cut_at is not None else [], "segment": {}}


def test_pan_metrics_constant_pan() -> None:
    m = qa.pan_metrics(make_path(speed=10.0), 2.0, 10.0)
    assert m["speed_p95"] == pytest.approx(10.0, abs=0.2)
    assert m["speed_max"] == pytest.approx(10.0, abs=0.2)
    assert m["accel_p95"] < 1.0
    assert m["fov_rate_p95"] == 0.0
    assert m["cuts"] == 0
    assert m["min_shot_s"] == pytest.approx(10.0)


def test_pan_metrics_excludes_cut_crossing_and_measures_shots() -> None:
    m = qa.pan_metrics(make_path(speed=5.0, cut_at=6.0), 2.0, 10.0)
    assert m["cuts"] == 1
    assert m["cut_times"] == [4.0]
    assert m["speed_max"] < 10.0  # the 120 deg jump at the cut is not a pan
    assert m["min_shot_s"] == pytest.approx(4.0)
    assert m["cuts_per_min"] == pytest.approx(6.0)


def test_targets_at_nearest_key() -> None:
    path = make_path(cut_at=6.0)
    assert qa.targets_at(path, np.array([1.0, 7.2])) == ["friend", "view"]
    assert qa.targets_at(None, np.array([1.0])) == ["unknown"]


def _rolling(roll_deg: np.ndarray) -> G.Orientation:
    """Sky lens up (camera -> world Rp(-90)), rolled about the travel direction (world z)."""
    h = np.radians(roll_deg) / 2
    m = np.sqrt(0.5)
    # (cos h, 0, 0, sin h) * (m, -m, 0, 0)
    q = np.stack([np.cos(h) * m, -np.cos(h) * m, -np.sin(h) * m, np.sin(h) * m], 1)
    n = len(roll_deg)
    z = np.zeros(n)
    return G.Orientation(np.arange(n) / 30.0, z, z, z, {"fps": 30.0}, q=q)


def test_horizon_roll_ignores_shake_and_shows_followed_lean() -> None:
    path = make_path(speed=0.0, t1=30.0)
    times = np.arange(2.0, 28.0, 0.1)
    jolt = np.zeros(900)
    jolt[150] = 8.0  # one-frame roll jolt: stabilised away
    assert np.abs(qa.horizon_roll(_rolling(jolt), path, times)).max() < 0.1
    lean = _rolling(10 * np.sin(2 * np.pi * np.arange(900) / 30.0 / 4))  # slow +-10 deg
    roll = np.abs(qa.horizon_roll(lean, path, times)).max()
    assert 1.0 < roll < 3.2  # 30% of the (smoothed) lean is followed by default


def test_covered_lens_is_dark_and_fails_the_clip() -> None:
    gray = np.full((3, 36, 64), 0.5)
    gray[0, 4:] = 0.03  # a hand over the lens, light at the top edge
    gray[1] = 0.18  # dim, but no near-black picture (dusk, fog)
    assert qa.dark_frames(gray, TH).tolist() == [True, False, False]
    dim = np.full((4, 10, 10), 0.4)
    dim[:, :6] = 0.05  # a dim clip throughout: mean 0.19, 60 % near black, but its normal
    assert not qa.dark_frames(dim, TH).any()
    recs = qa.frame_flags(
        np.zeros((0, 7)),
        np.zeros(0, bool),
        3,
        gray_std=np.full(3, 0.2),
        occl_zs=np.zeros(3),
        lowinfo_zs=np.zeros(3),
        targets=["view"] * 3,
        roll=None,
        th=TH,
        dark=qa.dark_frames(gray, TH),
    )
    assert [r["flags"] for r in recs] == [["dark"], [], []]
    frames = [rec(i / 2, []) for i in range(40)]
    frames[30]["flags"] = ["dark"]
    assert qa.aggregate_clip(frames, 2.0, PAN_OK, None, TH)["fails"] == ["dark 0.5s"]


# --------------------------------------------------------------------------- aggregation
def rec(t: float, flags: list[str], subj: list[float] | None = None, target: str = "friend"):
    if subj is None and "nosubj" not in flags:
        subj = [0.4, 0.3, 0.6, 0.95]
    flags = [f for f in flags if f != "nosubj"]
    return {"t": t, "target": target, "flags": flags, "subj": subj, "size_ok": True, "safe": True}


PAN_OK = {"speed_p95": 20.0, "speed_max": 30.0, "min_shot_s": 5.0}


def test_aggregate_clean_clip_passes() -> None:
    frames = [rec(i / 2, []) for i in range(40)]
    a = qa.aggregate_clip(frames, 2.0, PAN_OK, {"p95": 0.5, "max": 1.0}, TH)
    assert a["pass"] is True
    assert a["score"] == 100.0
    assert a["share"]["subject_present"] == 1.0
    assert a["flags"] == []


def test_aggregate_shares_runs_spans_and_score() -> None:
    frames = [rec(i / 2, []) for i in range(40)]
    for i in (4, 5, 6):
        frames[i]["flags"] = ["occluded"]
    frames[20]["flags"] = ["edge_cut"]
    a = qa.aggregate_clip(frames, 2.0, PAN_OK, None, TH)
    assert a["share"]["occluded"] == pytest.approx(0.075)
    assert a["runs"]["occluded"] == pytest.approx(1.5)
    assert a["runs"]["occl_close"] == pytest.approx(1.5)
    assert a["flags"] == [
        {"what": "occluded", "t0": 2.0, "t1": 3.5},
        {"what": "edge_cut", "t0": 10.0, "t1": 10.5},
    ]
    assert a["score"] == pytest.approx(100 * (1 - 2 * 0.075 - 1.5 * 0.025), abs=0.05)
    assert a["pass"] is False
    assert any("occluded+too_close" in f for f in a["fails"])
    assert any("run 1.5s" in f for f in a["fails"])


@pytest.mark.parametrize(
    ("mutate", "fail"),
    [
        (lambda f, p, h: [f[i]["flags"].append("too_close") for i in (3, 4)], "close run"),
        (lambda f, p, h: [f[i]["flags"].append("edge_cut") for i in (1, 2, 3)], "edge_cut run"),
        (lambda f, p, h: [f[i].update(subj=None) for i in range(9)], "subject present"),
        (lambda f, p, h: [f[i]["flags"].append("low_info") for i in (1, 9, 17)], "low_info"),
        (lambda f, p, h: p.update(speed_p95=40.0), "pan"),
        (lambda f, p, h: p.update(speed_max=60.0), "pan"),
        (lambda f, p, h: p.update(min_shot_s=1.5), "shot"),
        (lambda f, p, h: h.update(p95=12.5), "horizon"),
    ],
)
def test_pass_rule(mutate, fail: str) -> None:
    frames = [rec(i / 2, []) for i in range(40)]
    pan, hor = dict(PAN_OK), {"p95": 0.5, "max": 1.0}
    mutate(frames, pan, hor)
    a = qa.aggregate_clip(frames, 2.0, pan, hor, TH)
    assert a["pass"] is False
    assert any(fail in f for f in a["fails"]), a["fails"]


def test_single_short_flags_within_limits_pass() -> None:
    frames = [rec(i / 2, []) for i in range(40)]
    frames[10]["flags"] = ["occluded"]  # 2.5%, run 0.5 s
    frames[20]["flags"] = ["edge_cut"]
    frames[21]["flags"] = ["edge_cut"]  # run 1.0 s (fails only above 1.0)
    a = qa.aggregate_clip(frames, 2.0, PAN_OK, None, TH)
    assert a["pass"] is True, a["fails"]


@pytest.mark.parametrize("flag", ["occluded"])
@pytest.mark.parametrize(("flagged", "ok"), [((9,), True), ((3, 12), False)])
def test_single_flagged_frame_is_not_a_share_failure(flag: str, flagged, ok: bool) -> None:
    # 19 frames: one flagged frame is 5.3 %, above the 3 % / 5 % shares
    frames = [rec(i / 2, []) for i in range(19)]
    for i in flagged:
        frames[i]["flags"] = [flag]
    a = qa.aggregate_clip(frames, 2.0, PAN_OK, None, TH)
    assert a["pass"] is ok, a["fails"]


def test_aggregate_edit_is_duration_weighted() -> None:
    clips = [
        {
            "duration": 30.0,
            "score": 100.0,
            "pass": True,
            "share": {"occluded": 0.0, "subject_present": 1.0},
            "runs": {"occl_close": 0.0},
            "pan": {"speed_p95": 10.0, "speed_max": 20.0, "jerk_p95": 5.0, "min_shot_s": 30.0},
            "horizon": {"p95": 0.5, "max": 1.0},
        },
        {
            "duration": 10.0,
            "score": 60.0,
            "pass": False,
            "share": {"occluded": 0.2, "subject_present": None},
            "runs": {"occl_close": 1.5},
            "pan": None,
            "horizon": None,
        },
    ]
    e = qa.aggregate_edit(clips)
    assert e["score"] == pytest.approx(90.0)
    assert e["passed"] == 1
    assert e["pass_share"] == 0.5
    assert e["share"]["occluded"] == pytest.approx(0.05)
    assert e["share"]["subject_present"] == 1.0
    assert e["longest_occl_close_run_s"] == 1.5
    assert e["pan_max"] == 20.0
    assert e["horizon_p95_max"] == 0.5


# --------------------------------------------------------------------------- inputs + JSON
def test_find_inputs_preview_folder(tmp_path: Path) -> None:
    d = tmp_path / "director"
    d.mkdir()
    (d / "001_a.mp4").write_bytes(b"")
    (d / "001_a.json").write_text(json.dumps({**make_path(), "segment": {"start": 12.5}}))
    (d / "002_b.mp4").write_bytes(b"")  # no path: skipped
    items = qa.find_inputs(d)
    assert [i["file"] for i in items] == ["001_a.mp4"]
    assert items[0]["offset"] == 12.5
    assert items[0]["path"]["keys"]


def test_find_inputs_clips_folder(tmp_path: Path) -> None:
    d = tmp_path / "clips"
    d.mkdir()
    pf = tmp_path / "p.json"
    pf.write_text(json.dumps(make_path()))
    (d / "001_x.mp4").write_bytes(b"")
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "file": "001_x.mp4",
                        "start": 3.0,
                        "kind": "laughter",
                        "pano": {"path_file": str(pf), "mode": "dfisheye", "cmd_every": 2},
                    }
                ]
            }
        )
    )
    items = qa.find_inputs(d)
    assert items[0]["offset"] == 3.0
    assert items[0]["kind"] == "laughter"
    assert items[0]["path"]["keys"]


def test_run_qa_json_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = tmp_path / "preview"
    d.mkdir()
    (d / "001_a.mp4").write_bytes(b"")
    (d / "001_a.json").write_text(json.dumps({**make_path(t1=30.0), "segment": {"start": 2.0}}))
    n = 8
    ts = np.arange(n) / 2.0
    frames = np.full((n, 36, 64, 3), 128, np.uint8)
    frames[:, :, ::2] = 0  # contrast, so nothing is low-info
    det = np.array([[i, 1, 0.99, 0.4, 0.3, 0.6, 0.95] for i in range(n)], np.float32)
    det[1] = [1, 1, 0.99, 0.0, 0.1, 0.8, 1.0]  # frame 1: close + edge cut

    class Models:
        def close(self) -> None:
            pass

    monkeypatch.setattr(V, "VisualModels", Models)
    monkeypatch.setattr(qa, "decode", lambda video, fps, size: (ts, frames))
    monkeypatch.setattr(
        V,
        "visual_data_from_frames",
        lambda m, t, f: VisualData(t=t, emb=np.zeros((n, 3, 768), np.float16), det=det),
    )
    monkeypatch.setattr(V, "self_boxes", lambda data, rules=None: np.zeros(len(data.det), bool))
    monkeypatch.setattr(V, "prompt_score", lambda emb, pos, neg, models=None: np.zeros((n, 3)))
    monkeypatch.setattr(V, "frame_scores", lambda data, models=None: {})
    monkeypatch.setattr(V, "beauty_score", lambda sc: np.full(n, 0.4))
    monkeypatch.setattr(F, "probe", lambda p: SimpleNamespace(duration=4.0))
    rep = qa.run_qa(d, json_only=True)
    out = tmp_path / "preview_qa" / "qa.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) >= {"version", "source_dir", "mode", "thresholds", "clips", "edit"}
    assert data["mode"] == "rendered"
    assert data["thresholds"]["occl_zs"] == TH.occl_zs
    c = data["clips"][0]
    assert set(c) >= {
        "file",
        "kind",
        "duration",
        "share",
        "runs",
        "pan",
        "horizon",
        "beauty_mean",
        "score",
        "pass",
        "flags",
        "frames",
    }
    assert c["file"] == "001_a.mp4"
    assert c["beauty_mean"] == pytest.approx(0.4)
    assert c["horizon"] is None  # no cached IMU orientation
    assert c["frames"][1]["flags"] == ["too_close", "edge_cut"]
    assert set(c["frames"][0]) >= {"t", "target", "flags", "subj", "occl_zs", "lowinfo_zs"}
    assert {"what": "too_close", "t0": 0.5, "t1": 1.0} in c["flags"]
    assert c["pan"]["speed_p95"] == pytest.approx(10.0, abs=0.2)
    assert data["edit"]["clips"] == 1
    assert rep["sheets"] == []


def write_manifest(d: Path, pano: dict, start: float = 3.0) -> None:
    d.mkdir(exist_ok=True)
    (d / "001_x.mp4").write_bytes(b"")
    (d / "manifest.json").write_text(
        json.dumps({"clips": [{"file": "001_x.mp4", "start": start, "pano": pano}]})
    )


@pytest.mark.parametrize(
    ("sidecar", "expect"), [(["-x"], True), (["-enable_flowstate"], False), (None, False)]
)
def test_sdk_horizon_only_when_imu_levelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sidecar, expect: bool
) -> None:
    eq = tmp_path / "033.equirect.mp4"
    eq.write_bytes(b"")
    if sidecar is not None:
        project.equirect_sidecar(eq).write_text(json.dumps({"stitch_args": sidecar}))
    pf = tmp_path / "p.json"
    pf.write_text(json.dumps({**make_path(t1=30.0), "source": str(tmp_path / "src.lrv")}))
    d = tmp_path / "clips"
    write_manifest(d, {"path_file": str(pf), "mode": "sdk", "inputs": [str(eq)], "cmd_every": 2})
    orient = _rolling(np.zeros(900))
    seen = []
    monkeypatch.setattr(G, "load_orientation", lambda src: seen.append(src) or orient)
    item = qa.find_inputs(d)[0]
    assert item["imu_levelled"] is expect
    roll = qa.horizon(item, np.array([5.0, 6.0]))
    assert (roll is not None) is expect
    if expect:
        assert seen == [tmp_path / "src.lrv"]


def test_missing_path_file_is_reported(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    d = tmp_path / "clips"
    write_manifest(d, {"path_file": str(tmp_path / "gone.json"), "mode": "dfisheye"})
    with caplog.at_level("WARNING"):
        item = qa.find_inputs(d)[0]
    assert item["path_missing"] is True
    assert item["path"] is None
    assert "not found" in caplog.text


def test_run_qa_skips_failing_clip_resolves_dot_and_clears_sheets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d = tmp_path / "preview"
    d.mkdir()
    (d / "001_a.json").write_text(json.dumps({**make_path(), "segment": {"start": 0.0}}))
    (d / "001_a.mp4").write_bytes(b"")
    out = tmp_path / "preview_qa"
    out.mkdir()
    (out / "flagged_01.jpg").write_bytes(b"old")

    class Models:
        def close(self) -> None:
            pass

    def boom(video, fps, size):
        raise F.FFmpegError("decode failed")

    monkeypatch.setattr(V, "VisualModels", Models)
    monkeypatch.setattr(qa, "decode", boom)
    monkeypatch.chdir(d)
    rep = qa.run_qa(Path(), json_only=True)
    assert rep["out_dir"] == str(out)
    assert rep["clips"] == []
    assert rep["skipped"] == [{"file": "001_a.mp4", "error": "decode failed"}]
    assert not (out / "flagged_01.jpg").exists()
    assert json.loads((out / "qa.json").read_text(encoding="utf-8"))["skipped"]
    sub = d / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)
    rep = qa.run_qa(Path(".."), json_only=True)
    assert Path(rep["out_dir"]) == tmp_path / "preview_qa"
    assert rep["source_dir"] == str(d)


def test_visual_data_from_frames_assembles_batches() -> None:
    calls = []

    class FakeModels:
        def run(self, frames: np.ndarray):
            calls.append(len(frames))
            b = len(frames)
            emb = np.ones((b, 3, 768), np.float16)
            dets = [np.array([[1, 0.9, 0.1, 0.2, 0.3, 0.4]]) for _ in range(b)]
            be = []
            for j in range(b):
                v = np.full((1, 768), np.nan if (sum(calls[:-1]) + j) % 2 else 0.5, np.float32)
                be.append(v)
            return emb, dets, be

    ts = [0.0, 0.5, 1.0, 1.5, 2.0]
    frames = [np.zeros((4, 4, 3), np.uint8)] * 5
    data = V.visual_data_from_frames(FakeModels(), ts, frames, batch=2)
    assert calls == [2, 2, 1]
    assert data.det[:, 0].tolist() == [0, 1, 2, 3, 4]
    assert data.t.tolist() == ts
    assert data.emb.shape == (5, 3, 768)
    assert data.box_idx.tolist() == [0, 2, 4]  # frames whose crop was embedded
    assert data.box_emb.shape == (3, 768)
    assert data.meta["frames"] == 5
    with pytest.raises(ValueError, match="at least one frame"):
        V.visual_data_from_frames(FakeModels(), [], [])
