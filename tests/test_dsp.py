"""Shared signal helpers: boolean runs on the analysis grid."""

from __future__ import annotations

import numpy as np

from cutpoint.dsp import clean_mask, mask_runs


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
