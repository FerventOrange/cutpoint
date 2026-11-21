"""Shared signal helpers: boolean runs on the analysis grid, Gaussian smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from cutpoint.analyze import DT, _impulse_signal, keyword_signal, loudness_signal
from cutpoint.dsp import clean_mask, conv_same, gauss_kernel, gauss_smooth, mask_runs


def test_mask_runs_are_half_open_and_include_runs_at_both_edges() -> None:
    mask = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
    assert mask_runs(mask) == [(0, 2), (4, 5), (6, 9)]
    assert mask_runs(np.zeros(4, bool)) == []
    assert mask_runs(np.array([0.0, 2.0, np.nan, 0.0])) == [(1, 3)]  # nonzero (and NaN) is True


def test_clean_mask_closes_short_gaps_then_drops_short_runs() -> None:
    mask = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1], bool)
    # the 1-bin gap closes; 3-bin gaps stay; the lone bin at 6 is shorter than 3
    assert mask_runs(clean_mask(mask, 1, 3)) == [(0, 3), (10, 13)]
    assert mask_runs(clean_mask(mask, 3, 1)) == [(0, 13)]
    assert not mask[1]  # the input is not modified


def test_gauss_smooth_edges_and_axis() -> None:
    x = np.r_[np.zeros(20), np.ones(20)]
    y = gauss_smooth(x, 2.0)
    assert len(y) == len(x)
    assert y[0] == 0.0 and y[-1] == 1.0  # edge-padded: no pull toward 0 at the ends
    assert y[19] == pytest.approx(1 - y[20])  # symmetric about the step
    two = gauss_smooth(np.stack([x, 3 * x], 1), 2.0)  # columns smoothed along axis 0
    assert np.array_equal(two[:, 0], y) and np.allclose(two[:, 1], 3 * y)


def test_gauss_smooth_min_sigma_and_min_len_return_an_unsmoothed_copy() -> None:
    x = np.r_[np.zeros(10), np.ones(10)]
    for sigma, kw in ((0.5, {"min_sigma": 0.5}), (0.0, {"min_sigma": 0.0}), (2.0, {"min_len": 21})):
        y = gauss_smooth(x, sigma, **kw)
        assert np.array_equal(y, x) and y is not x
    # just above the floor, or long enough: smoothed
    assert not np.array_equal(gauss_smooth(x, 0.6, min_sigma=0.5), x)
    assert not np.array_equal(gauss_smooth(x, 2.0, min_sigma=0.0, min_len=20), x)
    short = np.array([0.0, 1.0])  # two samples: smoothed without a length floor
    assert not np.array_equal(gauss_smooth(short, 1.0), short)
    assert np.array_equal(gauss_smooth(short, 1.0, min_len=3), short)


# Reference copies of the Gaussian formulas before analyze moved onto dsp.gauss_kernel
# (#1085 A370): the new path must give the same floats, bit for bit.
def _old_dsp_gauss_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    if x.ndim == 1:
        return np.convolve(np.pad(x, r, mode="edge"), k, mode="valid")
    xp = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, xp)


def _old_analyze_kernel(sigma_s: float) -> tuple[int, np.ndarray]:
    radius = int(3 * sigma_s / DT)
    return radius, np.exp(-0.5 * (np.arange(-radius, radius + 1) * DT / sigma_s) ** 2)


def _old_analyze_smooth(x: np.ndarray, sigma_s: float) -> np.ndarray:
    _, k = _old_analyze_kernel(sigma_s)
    return conv_same(x, k / k.sum())


def _old_impulse(n: int, events: list[tuple[float, float]], sigma_s: float) -> np.ndarray:
    x = np.zeros(n)
    for t, w in events:
        i = int(t / DT)
        if 0 <= i < n:
            x[i] += w
    _, k = _old_analyze_kernel(sigma_s)
    return 1.0 - np.exp(-conv_same(x, k))


def _old_capped(n: int, hits: list[tuple[float, float, str]], cap: float, sigma_s: float):
    radius, k = _old_analyze_kernel(sigma_s)
    total, strongest = np.zeros(n), np.zeros(n)
    for t, w, _ in hits:
        i = int(t / DT)
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        if lo >= hi:
            continue
        contrib = w * k[lo - (i - radius) : hi - (i - radius)]
        total[lo:hi] += contrib
        np.maximum(strongest[lo:hi], contrib, out=strongest[lo:hi])
    return 1.0 - np.exp(-np.minimum(cap, strongest + 0.3 * (total - strongest)))


# 0.4, 1.0 and 1.5 s are analyze's sigmas; 1.7, 3.4 and 3.9 s are where 3 * (s / DT) and
# 3 * s / DT round to different radii, so sigma must stay in seconds with dt=DT.
_SIGMAS_S = (0.1, 0.25, 0.4, 0.7, 1.0, 1.5, 1.7, 2.0, 3.4, 3.9)


@pytest.mark.parametrize("sigma_s", _SIGMAS_S)
def test_gauss_kernel_matches_the_old_analyze_kernel(sigma_s: float) -> None:
    r0, k0 = _old_analyze_kernel(sigma_s)
    r1, k1 = gauss_kernel(sigma_s, DT, normalise=False)
    assert r1 == r0 and np.array_equal(k1, k0) and k1[r1] == 1.0  # peak height 1


@pytest.mark.parametrize("sigma_s", _SIGMAS_S)
@pytest.mark.parametrize("n", (1, 3, 8, 50, 400))  # shorter and longer than the kernel
def test_zero_padded_smooth_and_impulses_match_the_old_analyze_formulas(
    sigma_s: float, n: int
) -> None:
    rng = np.random.default_rng(int(sigma_s * 100) + n)
    x = rng.uniform(0, 6, n)
    x[0], x[-1] = 9.0, 7.0  # big values at both ends: the zero padding pulls them down
    assert np.array_equal(
        gauss_smooth(x, sigma_s, dt=DT, pad="zero"), _old_analyze_smooth(x, sigma_s)
    )
    events = [(0.0, 1.0), (n * DT / 2, 0.7), (n * DT / 2 + 0.05, 0.4), ((n - 1) * DT, 2.0)]
    events += [(-1.0, 1.0), (n * DT + 5, 1.0)]  # off the grid: ignored
    assert np.array_equal(_impulse_signal(n, events, sigma_s), _old_impulse(n, events, sigma_s))
    hits = [(t, w, "w") for t, w in events]
    for cap in (0.5, 1.2, 9.0):
        assert np.array_equal(
            keyword_signal(n, hits, cap, sigma_s), _old_capped(n, hits, cap, sigma_s)
        )


def test_loudness_signal_matches_the_old_formula_and_has_no_sigma_floor() -> None:
    rng = np.random.default_rng(7)
    for n in (2, 5, 30, 600):
        lufs = rng.uniform(-80, -10, n)
        x = np.maximum(lufs, -70.0)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) * 1.4826 + 1e-6
        z = np.clip((x - med) / max(mad, 1.0), 0, None)
        old = np.clip(_old_analyze_smooth(z, 0.4) / 4.0, 0, 1)
        assert np.array_equal(loudness_signal(lufs), old)
    # no min_sigma or min_len: a tiny sigma and a one-sample signal are still smoothed
    one = np.array([4.0])
    assert gauss_smooth(one, 0.4, dt=DT, pad="zero")[0] < 4.0


@pytest.mark.parametrize("sigma", (0.6, 1.0, 2.0, 2.5, 7.3, 30.0))
@pytest.mark.parametrize("n", (2, 5, 40, 300))
def test_edge_padded_gauss_smooth_is_unchanged(sigma: float, n: int) -> None:
    rng = np.random.default_rng(n)
    x = rng.normal(size=n)
    assert np.array_equal(gauss_smooth(x, sigma), _old_dsp_gauss_smooth(x, sigma))
    two = np.stack([x, 3 * x], 1)
    assert np.array_equal(gauss_smooth(two, sigma), _old_dsp_gauss_smooth(two, sigma))
    r0 = int(3 * sigma)
    k0 = np.exp(-0.5 * (np.arange(-r0, r0 + 1) / sigma) ** 2)
    r1, k1 = gauss_kernel(sigma)
    assert r1 == r0 and np.array_equal(k1, k0 / k0.sum())


def test_zero_padding_smooths_each_column_like_a_single_signal() -> None:
    x = np.r_[np.ones(10), np.zeros(10)]
    y = gauss_smooth(x, 2.0, pad="zero")
    assert y[0] < 1.0  # pulled toward 0 at the start, unlike edge padding
    two = gauss_smooth(np.stack([x, 3 * x], 1), 2.0, pad="zero")
    assert np.array_equal(two[:, 0], y) and np.allclose(two[:, 1], 3 * y)
