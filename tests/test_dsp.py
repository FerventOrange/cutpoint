"""Shared signal helpers: boolean runs on the analysis grid, Gaussian smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from cutpoint.dsp import clean_mask, gauss_smooth, mask_runs


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
