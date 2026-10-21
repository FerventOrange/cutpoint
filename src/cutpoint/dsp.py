"""Small signal-processing helpers shared by the analysis modules."""

from __future__ import annotations

import numpy as np


def conv_same(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Centred ``np.convolve(x, k)`` that always returns ``len(x)`` samples.

    numpy's "same" mode returns ``max(len(x), len(k))`` samples, which breaks
    signals shorter than the kernel (a 2 s clip vs a 10 s window).
    """
    full = np.convolve(x, k, mode="full")
    start = (len(k) - 1) // 2
    return full[start : start + len(x)]
