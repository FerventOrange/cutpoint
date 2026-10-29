"""360 moment preview rendering (#517): ``render_segment(progress=)``, the upright horizon of a
``level_forward`` preview and an end-to-end preview render (the ``preview_path`` kinds are tested
in test_pano_candidates.py).

No models or footage except in the ``ffmpeg`` tests, which render a synthetic dual fisheye.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

import pano_helpers as helpers
from conftest import media_info, needs_ffmpeg
from cutpoint import cache
from cutpoint.ffmpeg_utils import probe
from cutpoint.pano import candidates as C
from cutpoint.pano import geometry as G
from cutpoint.pano import project as P
from cutpoint.pano import render as R


def _dump(x) -> str:
    return json.dumps(x, sort_keys=True)


@pytest.fixture(autouse=True)
def cpu_render(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(R, "decoder_for", lambda _path: None)


# --------------------------------------------------------------------------- upright
def test_upright_orientation_is_the_nominal_mount() -> None:
    o = R.upright_orientation(30.0, 5)
    assert len(o.q) == 5 and np.allclose(o.up, [0.0, 0.0, -1.0])
    rw = o.cam_to_world(np.array([0, 4]))
    assert np.allclose(rw @ np.array([0.0, 0.0, -1.0]), [0.0, -1.0, 0.0], atol=1e-6)  # sky up
    assert np.allclose(rw @ G.FORWARD_V, [0.0, 0.0, 1.0], atol=1e-6)  # heading 0
    assert np.allclose(o.r_vw(0.1), G.level_matrix(np.array([0.0, 0.0, -1.0])), atol=1e-6)


# --------------------------------------------------------------------------- orientation cache
@pytest.fixture
def imu_clip(home, tmp_path, monkeypatch):
    """A source file and a stubbed IMU read (counted): a still, upright camera."""
    src = tmp_path / "clip.lrv"
    src.write_bytes(b"x")
    reads: list[Path] = []
    t = np.arange(0.0, 2.0, 1e-3)
    acc = np.tile(G.M_IMU_TO_V @ np.array([0.0, 0.0, -G.G]), (len(t), 1))

    def load_imu(insv00):
        reads.append(insv00)
        return t, acc, np.zeros((len(t), 3))

    monkeypatch.setattr(G, "load_imu", load_imu)
    return src, reads


def test_orientation_cache_is_written_atomically(imu_clip, monkeypatch) -> None:
    src, reads = imu_clip
    o = G.orientation(src, Path("i00.insv"), 30.0, 60)
    npz = G._orient_path(src)
    assert npz.is_file() and not list(npz.parent.glob("*.tmp"))
    assert np.allclose(G.load_orientation(src).q, o.q, atol=1e-6)
    assert G.orientation(src, Path("i00.insv"), 30.0, 60) is not None and len(reads) == 1
    npz.unlink()

    def replace_fails(a, b):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cache.os, "replace", replace_fails)
    with pytest.raises(OSError, match="No space"):
        G.orientation(src, Path("i00.insv"), 30.0, 60)
    assert not npz.exists() and not list(npz.parent.glob("*.tmp"))  # nothing half-written


def test_a_corrupt_orientation_cache_is_computed_anew(imu_clip) -> None:
    src, reads = imu_clip
    G.orientation(src, Path("i00.insv"), 30.0, 60)
    npz = G._orient_path(src)
    good = npz.read_bytes()
    npz.write_bytes(good[: len(good) // 2])  # a half-written file
    assert G.load_orientation(src) is None
    o = G.orientation(src, Path("i00.insv"), 30.0, 60)
    assert len(reads) == 2 and len(o.q) == 60
    assert G.load_orientation(src) is not None  # rewritten, readable again


# --------------------------------------------------------------------------- progress
@pytest.fixture
def fake_ffmpeg(monkeypatch):
    """``render.run`` writing ``-progress`` lines (``ticks`` updates over ``spacing`` s each)."""
    calls: list[list[str]] = []
    opts = {"ticks": 8, "spacing": 0.12}

    def run(args, cwd=None):
        calls.append(args)
        if "-progress" in args:
            dur = float(args[args.index("-t", args.index("-map")) + 1])
            with Path(args[args.index("-progress") + 1]).open("a", encoding="utf-8") as fh:
                for i in range(1, opts["ticks"] + 1):
                    fh.write(f"out_time_us={int(dur * 1e6 * i / opts['ticks'])}\n")
                    fh.write("progress=continue\n")
                    fh.flush()
                    time.sleep(opts["spacing"])
                fh.write("progress=end\n")
        Path(args[-1]).write_bytes(b"x")
        return ""

    monkeypatch.setattr(R, "run", run)
    monkeypatch.setattr(R, "probe", lambda p: media_info(duration=1.0))
    return calls, opts


def _render(clip, tmp_path, progress):
    src = P.RenderSource("sdk", ("in.mp4",), "equirect", True, False, "test")
    path = {"keys": [{"t": 0.0, "yaw": 0.0, "pitch": 0.0, "fov": 90.0}]}
    return R.render_segment(
        clip, path, 3.0, 4.0, tmp_path / "p.mp4", src=src, work_dir=tmp_path, progress=progress
    )


def test_progress_rises_to_one_and_stops(fake_ffmpeg, tmp_path) -> None:
    calls, _ = fake_ffmpeg
    clip = helpers.x3clip(lrv="x.lrv", fps=30.0, frames=300, duration=10.0)
    seen: list[tuple[float, float]] = []
    _render(clip, tmp_path, lambda f: seen.append((f, time.monotonic())))
    n = len(seen)
    vals = [v for v, _ in seen]
    assert n >= 3 and vals == sorted(vals) and vals[-1] == 1.0 and vals.count(1.0) == 1
    assert all(0.0 < v < 1.0 for v in vals[:-1])
    gaps = np.diff([t for _, t in seen[:-1]])
    assert (gaps >= R.PROGRESS_MIN_S - 0.01).all()
    time.sleep(0.3)
    assert len(seen) == n  # nothing after the return
    assert not list(Path(calls[0][calls[0].index("-progress") + 1]).parent.glob("cutpoint-pro*"))


def test_progress_is_rate_limited(fake_ffmpeg, tmp_path) -> None:
    _, opts = fake_ffmpeg
    opts.update(ticks=200, spacing=0.002)  # 200 updates; the sleeps' real length varies by OS
    clip = helpers.x3clip(lrv="x.lrv", fps=30.0, frames=300, duration=10.0)
    seen: list[float] = []
    stamps: list[float] = []

    def on_progress(frac: float) -> None:
        seen.append(frac)
        stamps.append(time.monotonic())

    t0 = time.monotonic()
    _render(clip, tmp_path, on_progress)
    elapsed = time.monotonic() - t0
    # One call per PROGRESS_MIN_S of the measured run, plus the first and the final 1.0
    # (and one more for a timer that fires late), however slow the machine is.
    assert len(seen) <= elapsed / R.PROGRESS_MIN_S + 3
    assert seen[-1] == 1.0 and seen == sorted(seen) and len(set(seen)) == len(seen)
    # Not a call per tick: the calls before the final 1.0 are spaced by the limit (half of it
    # here, since a loaded machine can delay one callback after the limiter's own timestamp).
    gaps = np.diff(stamps[:-1])
    assert (gaps >= R.PROGRESS_MIN_S * 0.5).all()


def test_a_raising_callback_does_not_break_the_render(fake_ffmpeg, tmp_path, caplog) -> None:
    clip = helpers.x3clip(lrv="x.lrv", fps=30.0, frames=300, duration=10.0)
    seen: list[float] = []

    def bad(f):
        seen.append(f)
        raise RuntimeError("UI gone")

    with caplog.at_level(logging.DEBUG, logger="cutpoint.pano.render"):
        rec = _render(clip, tmp_path, bad)
    assert (tmp_path / "p.mp4").is_file() and rec["read_error"] is None
    assert len(seen) == 1
    assert [r.levelname for r in caplog.records if "progress callback" in r.message] == ["DEBUG"]


def test_a_locked_progress_file_does_not_fail_the_render(fake_ffmpeg, tmp_path, monkeypatch):
    real = Path.unlink
    locked: list[Path] = []

    def unlink(self, missing_ok=False):
        if self.name.startswith("cutpoint-progress-"):
            locked.append(self)
            raise PermissionError(13, "The process cannot access the file", str(self))
        return real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    clip = helpers.x3clip(lrv="x.lrv", fps=30.0, frames=300, duration=10.0)
    seen: list[float] = []
    rec = _render(clip, tmp_path, seen.append)
    assert locked and rec["read_error"] is None and seen[-1] == 1.0
    assert (tmp_path / "p.mp4").is_file()
    for f in locked:
        real(f, missing_ok=True)


def test_no_progress_keeps_the_command(fake_ffmpeg, tmp_path) -> None:
    calls, _ = fake_ffmpeg
    clip = helpers.x3clip(lrv="x.lrv", fps=30.0, frames=300, duration=10.0)
    _render(clip, tmp_path, None)
    _render(clip, tmp_path, lambda f: None)
    assert "-progress" not in calls[0]
    with_p = calls[1]
    i = with_p.index("-progress")
    assert with_p[:i] + with_p[i + 3 :] == calls[0]  # only -progress FILE -nostats added


# --------------------------------------------------------------------------- real ffmpeg
@needs_ffmpeg
def test_level_forward_preview_renders_a_synthetic_dfisheye(home, tmp_path) -> None:
    """End to end without IMU or layers: preview_path falls back to an upright forward view,
    and render_segment renders it from a dual fisheye with real ffmpeg progress."""
    cam = tmp_path / "cam"
    cam.mkdir()
    src = cam / "fisheye.mp4"  # not an X3 name: no _00_.insv, no IMU
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
         "testsrc2=size=640x320:rate=30:duration=3", "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=48000:duration=3", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)],
        check=True,
    )  # fmt: skip
    clip = helpers.x3clip(
        lrv=str(src), insv=[str(src)], mode="4k", fps=30.0, frames=90, duration=3.0
    )
    proj = tmp_path / "proj"
    proj.mkdir()
    cache.save_json(proj / "pano.json", {"clips": [clip.to_json()]})
    got = C.preview_path(proj, src, 0.5, 2.5)
    assert got["kind"] == "level_forward" and got["path"]["level"] == R.UPRIGHT
    seen: list[float] = []
    out = tmp_path / "preview.mp4"
    rec = R.render_segment(
        clip,
        got["path"],
        0.5,
        2.5,
        out,
        fmt=R.Format360(320, 180),
        src=P.RenderSource("dfisheye", (str(src),), "dfisheye", False, True, "test"),
        work_dir=tmp_path / "work",
        progress=seen.append,
    )
    info = probe(out)
    assert (info.width, info.height) == (320, 180) and info.duration == pytest.approx(2.0, abs=0.2)
    assert rec["read_error"] is None and rec["commands"] == 0  # a still view
    assert seen and seen == sorted(seen) and seen[-1] == 1.0 and seen.count(1.0) == 1
