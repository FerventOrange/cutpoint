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
