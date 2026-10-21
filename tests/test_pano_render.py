"""360 render sources (FlowState sidecar -> ``levelled``) and ``v360`` command composition.

No ffmpeg or footage: ``render.run`` / ``render.probe`` are monkeypatched and the
IMU orientation is synthetic.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint import paths
from cutpoint.pano import geometry as G
from cutpoint.pano import project, render
from cutpoint.pano.x3 import X3Clip

FPS = 30.0


@pytest.fixture
def clip() -> X3Clip:
    return X3Clip(
        clip_id="001",
        start_local="2026-01-01T10:00:00",
        mode="5.7k",
        lrv="LRV_20260101_100000_11_001.lrv",
        insv=["VID_20260101_100000_00_001.insv", "VID_20260101_100000_10_001.insv"],
        fps=FPS,
        frames=300,
        duration=10.0,
    )


@pytest.fixture(autouse=True)
def fresh_warnings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(project, "_flowstate_warned", set())


def _stitched(tmp_path: Path) -> Path:
    d = tmp_path / "stitched"
    d.mkdir(exist_ok=True)
    cfg = paths.config_file()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"pano": {"stitched_dir": str(d)}}), encoding="utf-8")
    return d


# --------------------------------------------------------------------------- resolve_source
class TestResolveSource:
    def _sdk_file(self, tmp_path: Path, clip: X3Clip, args: list[str] | None) -> Path:
        _stitched(tmp_path)
        out = project.sdk_output(clip)
        out.write_bytes(b"x")
        if args is not None:
            meta = {"version": 1, "stitch_args": args, "sdk_version": "3.1.5"}
            project.equirect_sidecar(out).write_text(json.dumps(meta), encoding="utf-8")
        return out

    def test_sidecar_without_flowstate_is_not_levelled(self, tmp_path, clip, caplog):
        self._sdk_file(tmp_path, clip, ["-stitch_type", "optflow", "-enable_stitchfusion"])
        with caplog.at_level(logging.WARNING):
            src = project.resolve_source(clip, "sdk")
        assert (src.mode, src.projection, src.levelled) == ("sdk", "equirect", False)
        assert caplog.text == ""

    def test_sidecar_with_flowstate_is_levelled_with_warning(self, tmp_path, clip, caplog):
        self._sdk_file(tmp_path, clip, ["-enable_flowstate", "-enable_stitchfusion"])
        with caplog.at_level(logging.WARNING):
            src = project.resolve_source(clip, "auto")
        assert (src.mode, src.levelled) == ("sdk", True)
        assert "stitched with FlowState" in caplog.text
        assert "cutpoint 360 stitch <project> --clips 001 --force" in caplog.text

    def test_missing_sidecar_is_treated_as_flowstate(self, tmp_path, clip, caplog):
        out = self._sdk_file(tmp_path, clip, None)
        with caplog.at_level(logging.WARNING):
            src = project.resolve_source(clip, "sdk")
            project.resolve_source(clip, "sdk")
        assert src.levelled is True
        assert f"no sidecar {project.equirect_sidecar(out).name}" in caplog.text
        assert "--force" in caplog.text
        assert caplog.text.count("no sidecar") == 1  # once per file

    def test_studio_is_levelled_with_warning(self, tmp_path, clip, caplog, monkeypatch):
        export = tmp_path / "export_001.mp4"
        monkeypatch.setattr(project, "find_studio_export", lambda c: (export, "Studio export"))
        with caplog.at_level(logging.WARNING):
            src = project.resolve_source(clip, "studio")
        assert (src.mode, src.projection, src.levelled) == ("studio", "equirect", True)
        assert "Studio export, assumed FlowState" in caplog.text

    def test_dfisheye_is_not_levelled(self, clip):
        src = project.resolve_source(clip, "dfisheye")
        assert (src.projection, src.levelled) == ("dfisheye", False)


# --------------------------------------------------------------------------- command_lines
def _q(axis: tuple[float, float, float], deg: float) -> np.ndarray:
    """Quaternion ``(w, x, y, z)`` of a right-handed turn by ``deg`` about ``axis``."""
    h = math.radians(deg) / 2
    return np.r_[math.cos(h), math.sin(h) * np.asarray(axis, float)]


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


# Sky lens up, skis (FORWARD_V) along world +z: camera -> world = Rp(-90).
MOUNT = _q((1, 0, 0), -90)


def _orient(n: int, heading_deg: np.ndarray | None = None) -> G.Orientation:
    """Camera on the usual mount, turning about the vertical by ``heading_deg``."""
    hd = np.zeros(n) if heading_deg is None else heading_deg
    q = np.array([_qmul(_q((0, 1, 0), h), MOUNT) for h in hd])
    return G.Orientation(
        t=np.arange(n) / FPS,
        up=np.tile([0.0, 0.0, -1.0], (n, 1)),
        heading=np.zeros(n),
        heading_smooth=np.zeros(n),
        meta={"fps": FPS},
        q=q,
    )


def _path(*keys: tuple[float, float, float]) -> dict:
    return {"keys": [{"t": t, "yaw": y, "pitch": p, "fov": 90.0} for t, y, p in keys]}


def _src(projection: str, levelled: bool) -> project.RenderSource:
    return project.RenderSource("sdk", ("in.mp4",), projection, levelled, False, "test")


def _init_matrix(init: dict) -> np.ndarray:
    return G.rot_r(init["roll"]) @ G.rot_y(init["yaw"]) @ G.rot_p(init["pitch"])


def _replay(init: dict, lines: list[str]) -> np.ndarray:
    """The rotation v360 ends up with after the incremental sendcmd lines."""
    cur = _init_matrix(init)
    axes = {"roll": G.rot_r, "yaw": G.rot_y, "pitch": G.rot_p}
    for line in lines:
        for part in line.split(" ", 1)[1].rstrip(";").split(", "):
            _, name, val = part.split()
            if name in axes:
                cur = cur @ axes[name](float(val))
    return cur


class TestCommandLines:
    FMT = render.Format360(640, 360)

    def test_known_case_forward_ray(self, clip):
        """Sky lens straight up: R_VW = Rp(90), so the view centre samples C @ FORWARD_V."""
        orient = _orient(300)
        assert np.allclose(render.stabilised_r_vw(orient, np.arange(30, 45), 0.7, 0.5), G.rot_p(90))
        path = _path((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
        _, init = render.command_lines(
            clip, path, _src("equirect", False), 1.0, 0.5, self.FMT, 2, orient
        )
        centre = _init_matrix(init) @ np.array([0.0, 0.0, 1.0])
        assert np.allclose(centre, G.SDK_EQUIRECT_FROM_CAMERA @ G.FORWARD_V, atol=1e-4)

    def test_equirect_is_constant_rotation_of_dfisheye(self, clip):
        orient = _orient(300, np.linspace(0, 40, 300) ** 1.5 / 40**0.5)  # turning, speeding up
        path = _path((0.0, -30.0, -5.0), (10.0, 60.0, 10.0))
        c = G.SDK_EQUIRECT_FROM_CAMERA
        fish_lines, fish_init = render.command_lines(
            clip, path, _src("dfisheye", False), 2.0, 3.0, self.FMT, 2, orient
        )
        eq_lines, eq_init = render.command_lines(
            clip, path, _src("equirect", False), 2.0, 3.0, self.FMT, 2, orient
        )
        assert np.allclose(_init_matrix(eq_init), c @ _init_matrix(fish_init), atol=1e-4)
        # after all incremental commands both reach the same target, up to C
        assert eq_lines and fish_lines
        got = _replay(eq_init, eq_lines)
        assert np.allclose(got, c @ _replay(fish_init, fish_lines), atol=3e-3)
        # ... which is C @ R_VW @ view of the frame the last command is for
        j = round((2.0 + float(eq_lines[-1].split()[0]) + render.CMD_LEAD_S) * FPS)
        yaw, pitch, _ = render.D.evaluate(path, np.array([j / FPS]))
        r_vw = render.stabilised_r_vw(orient, np.arange(60, j + 1), 0.7, 0.5)[-1]
        target = c @ r_vw @ G.view_matrix(float(yaw[0]), float(pitch[0]))
        assert np.allclose(got, target, atol=3e-3)

    def test_commands_lead_their_frames_after_a_seek(self, clip):
        """After ``-ss`` the first frame is source frame ceil(start * fps): every
        command lands just before its own frame's timestamp, one per frame."""
        clip = dataclasses.replace(clip, fps=30000 / 1001)
        path = _path((0.0, 0.0, 0.0), (10.0, 90.0, 0.0))
        start = 1.01
        lines, _ = render.command_lines(
            clip, path, _src("equirect", True), start, 1.0, self.FMT, 1, None
        )
        times = np.array([float(ln.split()[0]) for ln in lines])
        pts = (math.ceil(start * clip.fps) + 1 + np.arange(len(times))) / clip.fps - start
        assert pts[0] - 1 / clip.fps == pytest.approx(0.0244, abs=1e-4)  # frame 31, not 0
        assert np.allclose(pts - times, render.CMD_LEAD_S, atol=1e-4)
        assert len(times) >= 30  # covers the whole second

    def test_level_strength_locks_or_follows_tilt(self):
        """Camera rolling +-10 deg about the travel direction: a full lock keeps the
        display up vertical, 0 follows the camera's (smoothed) roll."""
        n = 240
        roll = 10 * np.sin(2 * np.pi * np.arange(n) / FPS / 4)
        q = np.array([_qmul(_q((0, 0, 1), r), MOUNT) for r in roll])
        orient = dataclasses.replace(_orient(n), q=q)
        idx = np.arange(n)

        def display_tilt(strength: float) -> np.ndarray:
            r_vw = render.stabilised_r_vw(orient, idx, strength, 0.5)
            up = np.einsum("kij,kjl,l->ki", orient.cam_to_world(idx), r_vw, [0.0, -1.0, 0.0])
            return np.degrees(np.arccos(np.clip(-up[:, 1], -1, 1)))

        assert display_tilt(1.0).max() < 0.01
        assert 5 < display_tilt(0.0).max() < 10.5
        assert display_tilt(0.7).max() == pytest.approx(0.3 * display_tilt(0.0).max(), rel=0.05)

    def test_levelled_equirect_uses_path_angles_only(self, clip):
        path = _path((0.0, 25.0, -5.0), (10.0, 25.0, -5.0))
        lines, init = render.command_lines(
            clip, path, _src("equirect", True), 1.0, 1.0, self.FMT, 2, None
        )
        assert lines == []  # still path: nothing to send
        assert np.allclose(_init_matrix(init), G.view_matrix(25.0, -5.0), atol=1e-4)


# --------------------------------------------------------------------------- render_segment
class TestRenderSegmentSendcmd:
    @pytest.fixture
    def ffmpeg(self, monkeypatch):
        calls: list[list[str]] = []

        def run(args, cwd=None):
            calls.append(args)
            return ""

        monkeypatch.setattr(render, "run", run)
        monkeypatch.setattr(render, "probe", lambda p: SimpleNamespace(duration=1.0))
        return calls

    def _filter(self, args: list[str]) -> str:
        return args[args.index("-filter_complex") + 1]

    def test_still_path_skips_sendcmd(self, clip, tmp_path, ffmpeg):
        path = _path((0.0, 10.0, -5.0), (10.0, 10.0, -5.0))
        out = tmp_path / "still.mp4"
        rec = render.render_segment(
            clip, path, 3.0, 4.0, out, src=_src("equirect", True), work_dir=tmp_path
        )
        vf = self._filter(ffmpeg[0])
        assert "sendcmd" not in vf
        assert "v360@cam=input=equirect" in vf and ":yaw=10.000:" in vf
        assert rec["commands"] == 0 and rec["cmd_file"] is None
        assert not list(tmp_path.glob("*.cmd"))

    def test_moving_path_uses_sendcmd(self, clip, tmp_path, ffmpeg):
        path = _path((0.0, 0.0, -5.0), (10.0, 90.0, -5.0))
        out = tmp_path / "pan.mp4"
        rec = render.render_segment(
            clip, path, 3.0, 4.0, out, src=_src("equirect", True), work_dir=tmp_path
        )
        assert "sendcmd=f=pan.cmd," in self._filter(ffmpeg[0])
        assert rec["commands"] > 0 and rec["cmd_file"] == "pan.cmd"
        assert (tmp_path / "pan.cmd").read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------- orientation filter
class TestQuatTrack:
    T = np.arange(0, 30.0, 0.001)  # 1 kHz IMU

    def test_gyro_only_recovers_constant_rate(self):
        """90 deg/s about the camera's up axis for 1 s, no accelerometer pull."""
        t = self.T[:1001]
        a_v = np.tile([0.0, -G.G, 0.0], (len(t), 1))
        w_v = np.tile([0.0, math.radians(90), 0.0], (len(t), 1))
        q = G.quat_track(t, a_v, w_v, gain=0.0)
        assert np.allclose(G.quat_matrix(q[-1:])[0], G.rot_y(90), atol=1e-6)

    def test_accelerometer_keeps_it_level(self):
        """A 2 deg/s gyro bias on a still, tilted camera: the pull holds the up
        estimate on gravity (steady error ~ bias / gain), it would drift 60 deg."""
        tilt = np.array([0.3, -1.0, 0.0]) / np.linalg.norm([0.3, -1.0, 0.0])
        a_v = np.tile(G.G * tilt, (len(self.T), 1))
        w_v = np.tile([math.radians(2.0), 0.0, 0.0], (len(self.T), 1))
        q = G.quat_track(self.T, a_v, w_v, gain=0.6)
        up = G.quat_matrix(q[-1:])[0].T @ [0.0, -1.0, 0.0]
        assert math.degrees(math.acos(up @ tilt)) < 5

    def test_full_lock_matches_the_directors_frame(self):
        """Fused turning + 20 deg lean IMU: at level_strength 1 the stabilised display
        frame is the director's ``r_vw`` frame (so planned yaws land the same)."""
        fps, dt = 30.0, 0.001
        t = np.arange(0, 12.0, dt)
        psi = 40 * np.sin(2 * np.pi * t / 6)  # turning
        roll = 20 + 1.5 * np.sin(2 * np.pi * 7 * t)  # lean + shake
        q = np.array(
            [
                _qmul(_qmul(_q((0, 1, 0), p), _q((0, 0, 1), r)), MOUNT)
                for p, r in zip(psi, roll, strict=True)
            ]
        )
        rm = G.quat_matrix(q)
        rel = np.einsum("kji,kjl->kil", rm[:-1], rm[1:])  # small body-frame steps
        w = np.stack(
            [rel[:, 2, 1] - rel[:, 1, 2], rel[:, 0, 2] - rel[:, 2, 0], rel[:, 1, 0] - rel[:, 0, 1]],
            1,
        ) / (2 * dt)
        w = np.vstack([np.zeros(3), w])
        a_v = np.einsum("kji,j->ki", rm, [0.0, -G.G, 0.0])
        o = G.fuse(t, a_v @ G.M_IMU_TO_V, np.degrees(w) @ G.M_IMU_TO_V, fps, int(12 * fps) - 1)
        idx = np.arange(60, len(o.t) - 60)
        new = render.stabilised_r_vw(o, idx, 1.0, 0.5)
        old = np.stack([o.r_vw(i / fps) for i in idx])
        cos = (np.trace(np.einsum("kji,kjl->kil", old, new), axis1=1, axis2=2) - 1) / 2
        assert np.degrees(np.arccos(np.clip(cos, -1, 1))).max() < 2.0
