"""Box overlap helpers of the 360 QA (``cutpoint.pano.boxes``, A369)."""

from __future__ import annotations

import numpy as np

from cutpoint.pano import boxes as B


def test_iou_and_share_of_the_smaller_box() -> None:
    a = np.array([[0.0, 0.0, 0.4, 0.4]])
    b = np.array([[0.2, 0.2, 0.6, 0.6], [0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.5, 0.9]])
    assert np.allclose(B.iou(a, b), [[0.04 / 0.28, 0.01 / 0.16, 0.0]])
    assert np.allclose(B.overlap(a, b), [[0.04 / 0.16, 1.0, 0.0]])  # empty box: 0


def _greedy(frames: np.ndarray, scores: np.ndarray, boxes: np.ndarray, thr: float) -> list:
    """Pairwise greedy NMS, one box pair at a time (the reference)."""
    keep = [True] * len(frames)
    order = sorted(range(len(frames)), key=lambda i: (frames[i], -scores[i]))
    for x, i in enumerate(order):
        for j in order[x + 1 :]:
            if keep[i] and keep[j] and frames[i] == frames[j]:
                keep[j] = B.iou(boxes[i : i + 1], boxes[j : j + 1])[0, 0] <= thr
    return keep


def test_nms_suppresses_within_each_frame_only() -> None:
    rng = np.random.default_rng(3)
    for _ in range(30):
        n = int(rng.integers(0, 14))
        frames = rng.integers(0, 3, n).astype(float)
        scores = rng.uniform(0, 1, n)
        lo = rng.uniform(0, 0.6, (n, 2))
        boxes = np.c_[lo, lo + rng.uniform(0.05, 0.4, (n, 2))]
        assert B.nms(frames, scores, boxes, 0.45).tolist() == _greedy(frames, scores, boxes, 0.45)
