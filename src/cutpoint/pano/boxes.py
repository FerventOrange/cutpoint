"""Overlap of frame boxes ``(x1, y1, x2, y2)``: IoU, share of the smaller box and
per-frame non-maximum suppression (the 360 QA and the director's detector model)."""

from __future__ import annotations

import numpy as np


def _inter_areas(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersections ``(N, M)`` of each box in ``a`` ``(N, 4)`` with each in ``b``
    ``(M, 4)``, and the areas of ``a`` and ``b``."""
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(hi - lo, 0.0, None).prod(2)
    aa, ab = ((x[:, 2:] - x[:, :2]).prod(1) for x in (a, b))
    return inter, aa, ab


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over union of each box in ``a`` ``(N, 4)`` with each in ``b``
    ``(M, 4)``, ``(N, M)`` (0 where the union is empty)."""
    inter, aa, ab = _inter_areas(a, b)
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection of each box in ``a`` ``(N, 4)`` with each in ``b`` ``(M, 4)`` as a
    share of the smaller one's area, ``(N, M)``.

    The smaller, not ``a``'s own: in the 360 QA the wearer's duplicate box is the larger
    one (a head and shoulders box around the self-tagged lower half, e.g. 0.21 x 0.95
    around 0.20 x 0.48), which a share of ``a`` misses. The cost: a small self box inside
    a friend's box at the edge hides that friend's edge cut too.
    """
    inter, aa, ab = _inter_areas(a, b)
    return inter / np.maximum(np.minimum(aa[:, None], ab[None, :]), 1e-9)


def nms(frames: np.ndarray, scores: np.ndarray, boxes: np.ndarray, thr: float) -> np.ndarray:
    """Which rows survive greedy non-maximum suppression within each frame: a box whose
    IoU with a kept, higher-scored box of the same frame is above ``thr`` is dropped."""
    keep = np.ones(len(frames), bool)
    order = np.lexsort((-scores, frames)) if len(frames) else np.zeros(0, int)
    for f in np.unique(frames):
        idx = order[frames[order] == f]
        m = iou(boxes[idx], boxes[idx])
        for i in range(len(idx)):
            if keep[idx[i]]:
                keep[idx[i + 1 :]] &= m[i, i + 1 :] <= thr
    return keep
