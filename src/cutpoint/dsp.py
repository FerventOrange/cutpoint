"""Small signal-processing helpers shared by the analysis modules."""

from __future__ import annotations

from itertools import pairwise
from typing import Literal

import numpy as np


def conv_same(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Centred ``np.convolve(x, k)`` that always returns ``len(x)`` samples.

    numpy's "same" mode returns ``max(len(x), len(k))`` samples, which breaks
    signals shorter than the kernel (a 2 s clip vs a 10 s window).
    """
    full = np.convolve(x, k, mode="full")
    start = (len(k) - 1) // 2
    return full[start : start + len(x)]


def gauss_kernel(sigma: float, dt: float = 1.0, normalise: bool = True) -> tuple[int, np.ndarray]:
    """Gaussian over +-3 sigma: ``(radius, k)`` with ``len(k) == 2 * radius + 1``.

    ``sigma`` is in the unit of the step ``dt`` (samples by default; seconds with
    ``dt`` the grid step). ``normalise``: weights sum to 1, else the centre is 1.
    """
    r = int(3 * sigma / dt)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) * dt / sigma) ** 2)
    if normalise:
        k /= k.sum()
    return r, k


def gauss_smooth(
    x: np.ndarray,
    sigma: float,
    min_sigma: float | None = None,
    min_len: int = 0,
    *,
    dt: float = 1.0,
    pad: Literal["edge", "zero"] = "edge",
) -> np.ndarray:
    """Gaussian low-pass along axis 0 (``sigma`` in units of ``dt``, samples by default).

    The kernel is :func:`gauss_kernel`; ``pad`` is how the ends are extended: ``"edge"``
    repeats the end samples, ``"zero"`` pulls toward 0 (:func:`conv_same`). A copy of ``x``
    comes back unsmoothed when ``sigma <= min_sigma`` (None: no such floor) or ``x`` has
    fewer than ``min_len`` samples.
    """
    if (min_sigma is not None and sigma <= min_sigma) or len(x) < min_len:
        return x.copy()
    r, k = gauss_kernel(sigma, dt)
    if pad == "zero":
        if x.ndim == 1:
            return conv_same(x, k)
        return np.apply_along_axis(lambda c: conv_same(c, k), 0, x)
    if x.ndim == 1:
        return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")
    xp = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, xp)


BIN_EPS = 1e-6  # in bins: ``0.3 / 0.1`` is 2.9999999999999996, still bin 3


def bin_of(t: float, dt: float) -> int:
    """Index of the ``dt`` grid bin that holds time ``t`` (s): ``t / dt`` rounded down, with
    :data:`BIN_EPS` of slack so a time on a bin edge (a time rounded to 0.1 s on a 0.1 s grid)
    is not one bin early through float error. A negative ``t`` truncates toward 0, as
    ``int(t / dt)`` does (callers clamp it)."""
    return int(t / dt + BIN_EPS)


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
