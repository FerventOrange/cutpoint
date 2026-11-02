"""Builders shared by the 360 tests: synthetic views, IMU orientation and X3 clips."""

from __future__ import annotations

import numpy as np

from cutpoint.pano import geometry as G
from cutpoint.pano import views as VW
from cutpoint.pano.x3 import X3Clip

INSV = ["VID_20260101_100000_00_001.insv", "VID_20260101_100000_10_001.insv"]
LRV = "LRV_20260101_100000_11_001.lrv"


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


def orientation(n: int = 600, fps: float = 30.0, q: np.ndarray | None = None) -> G.Orientation:
    """Sky lens up, still; ``q`` (camera -> world quaternions) when the test needs rotation."""
    return G.Orientation(
        t=np.arange(n) / fps,
        up=np.tile([0.0, 0.0, -1.0], (n, 1)),
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
