"""The 360 horizon QA measures the roll from the emitted v360 commands (GitHub issue #12).

A correct render passes; a render whose commands roll the picture (a perturbed command
file, a broken renderer convention) is flagged, which the renderer's own model cannot see.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import media_info
from cutpoint.ffmpeg_utils import VideoEncoder
from cutpoint.pano import geometry as G
from cutpoint.pano import project, qa, render
from cutpoint.pano import qa_horizon as H
from pano_helpers import LRV, orientation, x3clip

FPS = 30.0
N = 900  # 30 s of IMU


def _q(axis: tuple[float, float, float], deg: np.ndarray) -> np.ndarray:
    """Quaternions ``(w, x, y, z)`` of right-handed turns by ``deg`` about ``axis``."""
    h = np.radians(deg) / 2
    return np.concatenate([np.cos(h)[:, None], np.sin(h)[:, None] * np.asarray(axis)], 1)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        1,
    )


def _riding() -> G.Orientation:
    """Sky lens up (camera -> world Rp(-90)), turning, leaning +-20 deg and shaking."""
    t = np.arange(N) / FPS
    mount = _q((1, 0, 0), np.full(N, -90.0))
    lean = 20 * np.sin(2 * np.pi * t / 6) + 3 * np.sin(2 * np.pi * t * 7)
    heading = 15 * t + 4 * np.sin(2 * np.pi * t * 3)
    pitch = 8 * np.sin(2 * np.pi * t / 5)
    q = _qmul(
        _q((0, 1, 0), heading), _qmul(_q((1, 0, 0), pitch), _qmul(_q((0, 0, 1), lean), mount))
    )
    return orientation(N, FPS, q)


PATH = {
    "keys": [
        {"t": float(t), "yaw": 6.0 * t - 40, "pitch": -5.0 + t / 4, "fov": 86.0, "target": "view"}
        for t in np.arange(0.0, 30.5, 0.5)
    ],
    "cuts": [],
}


@pytest.fixture
def rendered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``render(name)``: renders 10-14 s of the riding clip into a clips folder through
    :func:`render.render_segment` (ffmpeg faked, the ``.cmd`` file real) and returns
    the folder's :func:`qa.find_inputs` item."""
    orient = _riding()
    clip = x3clip(lrv=LRV, fps=FPS, frames=N, duration=N / FPS)
    calls: list[list[str]] = []

    def run(args: list[str], cwd: Path | None = None) -> str:
        calls.append(args)
        Path(args[-1]).write_bytes(b"x")  # the output, which render_segment then renames
        return ""

    monkeypatch.setattr(render, "run", run)
    monkeypatch.setattr(render, "probe", lambda p: media_info(duration=4.0))
    monkeypatch.setattr(render, "encode_with_fallback", lambda f, what: f(VideoEncoder("cpu")))
    monkeypatch.setattr(render, "imu_source", lambda p: p)
    monkeypatch.setattr(G, "orientation", lambda *a, **k: orient)
    monkeypatch.setattr(G, "load_orientation", lambda src: orient)
    paths_dir = tmp_path / "pano_paths" / "clips"
    paths_dir.mkdir(parents=True)
    clips = tmp_path / "clips"
    clips.mkdir()
    src = project.RenderSource("dfisheye", tuple(clip.insv), "dfisheye", False, True, "t")

    def render_clip(name: str = "001_run") -> dict:
        pf = paths_dir / f"{name}.json"
        pf.write_text(json.dumps({**PATH, "source": LRV}), encoding="utf-8")
        out = clips / f"{name}.mp4"
        rec = render.render_segment(
            clip, PATH, 10.0, 14.0, out, fmt=render.Format360(640, 360), src=src,
            work_dir=paths_dir,
        )  # fmt: skip
        entry = {"file": out.name, "start": 10.0, "pano": {**rec, "path_file": str(pf)}}
        (clips / "manifest.json").write_text(json.dumps({"clips": [entry]}), encoding="utf-8")
        (item,) = qa.find_inputs(clips)
        return item

    render_clip.calls = calls
    render_clip.cmd_dir = paths_dir
    render_clip.reread = lambda: qa.find_inputs(clips)[0]  # after editing the .cmd file
    return render_clip


def _summary(item: dict) -> dict:
    t = item["offset"] + np.arange(0.0, 4.0, 1 / 30.0)
    return qa.horizon_summary(qa.horizon(item, t), qa.horizon_source(item))


def _fails(hor: dict) -> list[str]:
    agg = qa.aggregate_clip([], 2.0, None, hor, qa.QAThresholds())
    return [f for f in agg["fails"] if f.startswith("horizon")]


def test_correct_render_passes_and_matches_the_intended_tilt(rendered) -> None:
    item = rendered()
    assert item["commands"] is not None
    hor = _summary(item)
    assert hor["source"] == "commands"
    assert _fails(hor) == []
    # the measurement agrees with what the renderer meant to show (30 % of the lean)
    t = item["offset"] + np.arange(0.0, 4.0, 1 / 30.0)
    measured = qa.horizon(item, t)
    intended = qa.horizon_roll(G.load_orientation(LRV), item["path"], t)
    assert np.abs(intended).max() > 2.0  # the lean shows: the comparison is not trivial
    assert np.abs(measured - intended).max() < 0.3


def test_cmd_header_holds_the_initial_v360_options(rendered) -> None:
    rendered()
    args = rendered.calls[0]
    vf = args[args.index("-filter_complex") + 1]
    head = (rendered.cmd_dir / "001_run.cmd").read_text(encoding="utf-8").splitlines()[0]
    assert head.startswith(render.CMD_HEADER + " input=dfisheye rorder=ryp yaw=")
    rot = " ".join(head.split()[4:7])  # yaw= pitch= roll=
    assert f":rorder=ryp:{rot.replace(' ', ':')}:" in vf
    assert f"-ss {head.split('start=')[1].split()[0]}" in " ".join(args)


def test_qa_does_not_use_the_stabilisation_model(rendered, monkeypatch) -> None:
    item = rendered()

    def boom(*_a, **_k):
        raise AssertionError("the horizon QA must not call the renderer's model")

    monkeypatch.setattr(render, "stabilised_r_vw", boom)
    monkeypatch.setattr(qa, "horizon_roll", boom)
    assert _summary(item)["source"] == "commands"


def test_perturbed_roll_commands_are_flagged(rendered) -> None:
    """A sign error in the emitted roll deltas: the model QA sees nothing, the command QA
    flags the clip."""
    rendered()
    cmd = rendered.cmd_dir / "001_run.cmd"
    text = cmd.read_text(encoding="utf-8")
    assert "v360@cam roll " in text
    cmd.write_text(text.replace("v360@cam roll ", "v360@cam roll -").replace("--", ""), "utf-8")
    item = rendered.reread()
    hor = _summary(item)
    assert _fails(hor), hor
    t = item["offset"] + np.arange(0.0, 4.0, 1 / 30.0)
    old = qa.horizon_summary(qa.horizon_roll(G.load_orientation(LRV), item["path"], t))
    assert _fails(old) == []  # the renderer's own model could not catch it


def test_renderer_roll_sign_error_is_flagged(rendered, monkeypatch) -> None:
    """The renderer's command maths gets the roll sign wrong (its levelling is right): the
    replay flags the clip, the model check does not."""
    real = G.decompose_ryp

    def wrong_roll(r: np.ndarray) -> tuple[float, float, float]:
        y, p, roll = real(r)
        return y, p, -roll

    monkeypatch.setattr(G, "decompose_ryp", wrong_roll)
    item = rendered()
    assert _fails(_summary(item))
    t = item["offset"] + np.arange(0.0, 4.0, 1 / 30.0)
    old = qa.horizon_summary(qa.horizon_roll(G.load_orientation(LRV), item["path"], t))
    assert _fails(old) == []


def test_commands_timed_a_frame_late_show_the_shake(rendered) -> None:
    """Every command shifted one frame late: the levelling misses the 7 Hz shake."""
    item = rendered()
    t = item["offset"] + np.arange(0.0, 4.0, 1 / 30.0)
    good = qa.horizon(item, t)
    cmd = rendered.cmd_dir / "001_run.cmd"
    head, *lines = cmd.read_text(encoding="utf-8").splitlines()
    late = []
    for ln in lines:
        at, rest = ln.split(" ", 1)
        late.append(f"{float(at) + 1 / FPS:.4f} {rest}")
    cmd.write_text("\n".join([head, *late]) + "\n", encoding="utf-8")
    bad = qa.horizon(rendered.reread(), t)
    assert np.abs(np.diff(bad)).max() > 2 * np.abs(np.diff(good)).max()


def test_clip_without_a_header_falls_back_to_the_model(rendered) -> None:
    rendered()
    cmd = rendered.cmd_dir / "001_run.cmd"
    cmd.write_text("".join(cmd.read_text(encoding="utf-8").splitlines(True)[1:]), "utf-8")
    item = rendered.reread()
    assert item["commands"] is None
    assert _summary(item)["source"] == "model"


def test_rerender_without_commands_drops_the_stale_cmd(rendered) -> None:
    rendered()
    cmd = rendered.cmd_dir / "001_run.cmd"
    assert cmd.is_file()
    still = {"keys": [{"t": t, "yaw": 10.0, "pitch": -5.0, "fov": 86.0} for t in (0.0, 30.0)]}
    src = project.RenderSource("studio", ("in.mp4",), "equirect", True, False, "t")
    out = cmd.parents[2] / "clips" / "001_run.mp4"
    rec = render.render_segment(
        x3clip(lrv=LRV, fps=FPS, frames=N), still, 10.0, 14.0, out, src=src,
        work_dir=rendered.cmd_dir,
    )  # fmt: skip
    assert rec["cmd_file"] is None
    assert not cmd.exists()
    assert "sendcmd" not in " ".join(rendered.calls[-1])


def test_first_frame_is_the_seek_frame_at_ntsc_rates(tmp_path) -> None:
    """A 6-decimal header start at 29.97 fps maps back to its own frame, not the next."""
    fps = 30000 / 1001
    t = np.arange(1200) / fps
    q = _qmul(_q((0, 0, 1), t * 30.0), _q((1, 0, 0), np.full(1200, -90.0)))  # rolling fast
    orient = orientation(1200, fps, q)
    for j0 in range(900, 1000):
        cmd = tmp_path / "a.cmd"
        cmd.write_text(
            f"{render.CMD_HEADER} input=dfisheye rorder=ryp yaw=0.000 pitch=90.000 roll=0.000 "
            f"start={j0 / fps:.6f} fps={fps:.6f}\n",
            "utf-8",
        )
        c = H.read_commands(cmd)
        early, first = H.emitted_roll(c, orient, np.array([(j0 - 1) / fps, j0 / fps]))
        assert early == pytest.approx(first)
        assert first == pytest.approx(H.emitted_roll(c, orient, np.array([j0 / fps]))[0])
        nxt = H.emitted_roll(c, orient, np.array([(j0 + 1) / fps]))[0]
        assert abs(nxt - first) > 0.5  # the rolling camera tells the frames apart


def test_replay_follows_the_v360_conventions(tmp_path) -> None:
    """The QA's own v360 turns agree with the renderer's measured conventions."""
    for name, rot in (("yaw", G.rot_y), ("pitch", G.rot_p), ("roll", G.rot_r)):
        assert np.allclose(H.turn(name, 23.0), rot(23.0))
    cmd = tmp_path / "a.cmd"
    cmd.write_text(
        f"{render.CMD_HEADER} input=equirect rorder=ryp yaw=10.000 pitch=-5.000 roll=2.000 "
        "start=3.000000 fps=30.000000\n"
        "0.0283 v360@cam roll 1.000, v360@cam yaw -2.000, v360@cam h_fov 80.000;\n"
        "0.0617 v360@cam pitch 0.500;\n",
        "utf-8",
    )
    c = H.read_commands(cmd)
    init = G.rot_r(2.0) @ G.rot_y(10.0) @ G.rot_p(-5.0)
    one = init @ G.rot_r(1.0) @ G.rot_y(-2.0)
    got = c.rotation(np.array([0.0, 0.03, 0.07]))
    assert np.allclose(got, [init, one, one @ G.rot_p(0.5)])
    assert (c.projection, c.start, c.fps) == ("equirect", 3.0, 30.0)
    assert H.read_commands(tmp_path / "missing.cmd") is None


def test_level_camera_through_a_still_view_shows_no_roll(tmp_path) -> None:
    """Sky lens up and level, view straight ahead: R = Rp(90), the horizon is level."""
    orient = orientation(60, FPS, _q((1, 0, 0), np.full(60, -90.0)))
    cmd = tmp_path / "a.cmd"
    y, p, r = G.decompose_ryp(G.rot_p(90.0) @ G.rot_y(30.0))
    cmd.write_text(
        f"{render.CMD_HEADER} input=dfisheye rorder=ryp yaw={y:.3f} pitch={p:.3f} "
        f"roll={r:.3f} start=0.000000 fps=30.000000\n",
        "utf-8",
    )
    c = H.read_commands(cmd)
    assert np.abs(H.emitted_roll(c, orient, np.arange(0.0, 1.0, 0.1))).max() < 0.01
    # an output ray turned by roll +9 (+x toward +y) shows the horizon rolled by -9
    tilted = dataclasses.replace(c, init=c.init @ H.turn("roll", 9.0))
    assert H.emitted_roll(tilted, orient, np.array([0.5])) == pytest.approx([-9.0], abs=0.01)


def test_render_rotations_from_the_commands_match_the_renderer_model(rendered) -> None:
    """The wearer cone of the rendered QA (#11) sees the picture through the emitted
    commands; they agree with the renderer's own R_VW @ view."""
    item = rendered()
    assert item["commands"] is not None
    t = item["offset"] + np.arange(0.2, 3.8, 0.5)
    orient = G.load_orientation(LRV)
    measured = qa.render_rotations(item, orient, t)
    model = qa.render_rotations({**item, "commands": None}, orient, t)
    ang = [
        np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))
        for a, b in zip(measured, model, strict=True)
    ]
    assert max(ang) < 0.5
