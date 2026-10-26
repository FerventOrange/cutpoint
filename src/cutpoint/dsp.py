"""Small signal-processing helpers shared by the analysis modules."""

from __future__ import annotations

from itertools import pairwise

import numpy as np


def conv_same(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Centred ``np.convolve(x, k)`` that always returns ``len(x)`` samples.

    numpy's "same" mode returns ``max(len(x), len(k))`` samples, which breaks
    signals shorter than the kernel (a 2 s clip vs a 10 s window).
    """
    full = np.convolve(x, k, mode="full")
    start = (len(k) - 1) // 2
    return full[start : start + len(x)]


def mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` index pairs of True runs."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(int)))
    return list(zip(d[::2], d[1::2], strict=True))  # padded mask: rises and falls pair up


def clean_mask(mask: np.ndarray, close_bins: int, min_bins: int) -> np.ndarray:
    """Close gaps of up to ``close_bins``, then drop runs shorter than ``min_bins``."""
    out = mask.astype(bool).copy()
    runs = mask_runs(out)
    for (_, e0), (s1, _) in pairwise(runs):
        if s1 - e0 <= close_bins:
            out[e0:s1] = True
    for s, e in mask_runs(out):
        if e - s < min_bins:
            out[s:e] = False
    return out
