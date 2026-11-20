"""Builders shared by the 360 tests: synthetic views, IMU orientation and X3 clips."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cutpoint import activity, cache
from cutpoint.pano import geometry as G
from cutpoint.pano import views as VW
from cutpoint.pano.x3 import X3Clip

INSV = ["VID_20260101_100000_00_001.insv", "VID_20260101_100000_10_001.insv"]
LRV = "LRV_20260101_100000_11_001.lrv"
SKY_UP = [0.0, 0.0, -1.0]  # camera up vector with the sky lens up


def make_views(
    boxes: dict[int, list[tuple]], source: str = "fake.lrv", n: int = 10
) -> VW.PanoViews:
    """View layer with ``boxes[frame] = [(score, yaw, pitch, w, h, ray), ...]`` (label person),
    one view frame every 1.2 s."""
    rows = [
        [f, 1, sc, y, p, w, h, 0.0, *ray] for f, bs in boxes.items() for sc, y, p, w, h, ray in bs
    ]
    return VW.PanoViews(
        t=(np.arange(n) * 1.2).astype(np.float32),
        emb=np.zeros((n, 9, 4), np.float16),
        det=np.asarray(rows, np.float32).reshape(-1, len(VW.DET_COLS)),
        box_emb=np.zeros((0, 768), np.float16),
        box_idx=np.zeros(0, np.int64),
        meta={"source": source},
    )


def orientation(
    n: int = 600,
    fps: float = 30.0,
    q: np.ndarray | None = None,
    up: list[float] = SKY_UP,
) -> G.Orientation:
    """Still, with the camera's ``up`` vector (sky lens up by default); ``q`` (camera -> world
    quaternions) when the test needs rotation."""
    return G.Orientation(
        t=np.arange(n) / fps,
        up=np.tile(np.asarray(up, float), (n, 1)),
        heading=np.zeros(n),
        heading_smooth=np.zeros(n),
        meta={"fps": fps},
        q=q,
    )


def x3clip(**over) -> X3Clip:
    """A 5.7K clip ``001`` recorded as :data:`INSV`; ``over`` replaces any field."""
    fields = {
        "clip_id": "001",
        "start_local": "2026-01-01T10:00:00",
        "mode": "5.7k",
        "lrv": None,
        "insv": list(INSV),
    }
    return X3Clip(**(fields | over))


def quat(axis: tuple[float, float, float], deg) -> np.ndarray:
    """Quaternion(s) ``(w, x, y, z)`` of right-handed turns by ``deg`` (a number or an array)
    about ``axis``."""
    h = np.radians(np.asarray(deg, float)) / 2
    axis_ = np.asarray(axis, float)
    return np.concatenate([np.cos(h)[..., None], np.sin(h)[..., None] * axis_], -1)


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions (or stacks of them, broadcast) ``a * b``."""
    w1, x1, y1, z1 = np.moveaxis(np.asarray(a), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(b), -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        -1,
    )


def fuse_rotations(t: np.ndarray, rm: np.ndarray, fps: float, frames: int) -> G.Orientation:
    """``geometry.fuse`` of the synthetic IMU of a camera whose camera -> world rotation at the
    sample times ``t`` (uniform steps) is ``rm`` ``(K, 3, 3)``: the gyro is the body-frame rate
    between samples, the accelerometer reads gravity only."""
    dt = float(t[1] - t[0])
    rel = np.einsum("kji,kjl->kil", rm[:-1], rm[1:])  # small body-frame steps
    w = np.stack(
        [rel[:, 2, 1] - rel[:, 1, 2], rel[:, 0, 2] - rel[:, 2, 0], rel[:, 1, 0] - rel[:, 0, 1]], 1
    ) / (2 * dt)
    w = np.vstack([np.zeros(3), w])
    a_v = np.einsum("kji,j->ki", rm, [0.0, -G.G, 0.0])
    return G.fuse(t, a_v @ G.M_IMU_TO_V, np.degrees(w) @ G.M_IMU_TO_V, fps, frames)


def rot(axis: int, deg) -> np.ndarray:
    """``(K, 3, 3)`` rotations by ``deg`` (a list or an array) about coordinate ``axis``."""
    a = np.radians(np.asarray(deg, float))
    c, s = np.cos(a), np.sin(a)
    i, j = [k for k in range(3) if k != axis]
    r = np.zeros((len(a), 3, 3))
    r[:, axis, axis] = 1.0
    r[:, i, i], r[:, j, j], r[:, i, j], r[:, j, i] = c, c, -s, s
    return r


def swaying(
    mount: np.ndarray,
    turn_dps: float = 0.0,
    sway_deg: float = 0.0,
    *,
    fps: float = 30.0,
    dt: float = 0.001,
    dur: float = 12.0,
) -> G.Orientation:
    """Fused track of a camera on ``mount`` (camera -> world at rest) turning steadily about
    the vertical and swaying (roll and pitch about world axes, a few Hz), ``dur`` seconds of
    IMU samples ``dt`` apart."""
    t = np.arange(0, dur, dt)
    psi = turn_dps * t
    roll = sway_deg * np.sin(2 * np.pi * 1.3 * t)
    pitch = sway_deg * np.sin(2 * np.pi * 0.7 * t + 1.0)
    rm = rot(1, psi) @ rot(2, roll) @ rot(0, pitch) @ mount
    return fuse_rotations(t, rm, fps, int(dur * fps) - 1)


# camera -> world of the usual mount: sky lens up, FORWARD_V along world +z (Rp(-90)).
SKY_MOUNT = quat((1, 0, 0), -90)


def turning(n: int, fps: float, heading_deg: np.ndarray | None = None) -> G.Orientation:
    """Camera on the usual mount (sky lens up, FORWARD_V along world +z), turning about the
    vertical by ``heading_deg`` (still without it)."""
    hd = np.zeros(n) if heading_deg is None else heading_deg
    return orientation(n, fps, np.array([qmul(quat((0, 1, 0), h), SKY_MOUNT) for h in hd]))


def write_orient(src: Path, up: list[float]) -> None:
    """Cache a ten-sample IMU orientation of ``src`` with the given ``up`` vector."""
    path = G._orient_path(src)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, up=np.tile(np.asarray(up, np.float32), (10, 1)))


def write_views(src: Path, meta: dict, det: np.ndarray | None = None) -> None:
    """Cache an empty view layer of ``src`` with sidecar ``meta`` (and ``det`` rows if given)."""
    path = VW.cache_path(src)
    path.parent.mkdir(parents=True, exist_ok=True)
    z = np.zeros(0)
    d = np.zeros((0, len(VW.DET_COLS)), np.float32) if det is None else det
    np.savez(path, t=z, emb=z, det=d, box_emb=z, box_idx=z)
    cache.save_json(path.with_suffix(".json"), meta)


def profile_prompts(raw: dict) -> activity.Prompts:
    """The prompts of the library profile ``raw`` (an invented profile dict)."""
    return activity.prompts(activity.full(activity.validate(raw)))


def write_activity(proj: Path, name: str, raw: dict) -> None:
    """Write ``proj``'s ``activity.json`` for the library profile ``name`` (profile dict ``raw``)."""
    rec = {
        "version": activity.PROFILE_VERSION,
        "name": name,
        "source": "library",
        "profile": activity.full(activity.validate(raw)),
    }
    proj.mkdir(parents=True, exist_ok=True)
    (proj / activity.ACTIVITY_FILE).write_text(json.dumps(rec), encoding="utf-8")
