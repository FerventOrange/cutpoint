"""The interval helpers: half-open overlap, merging touching spans, the free stretch."""

from __future__ import annotations

import pytest

from cutpoint import intervals


def test_overlapping_or_touching_spans_merge_in_order():
    got = intervals.merge([(30.0, 35.0), (10.0, 15.0), (14.0, 19.0), (19.0, 20.0), (12.0, 13.0)])
    assert got == [(10.0, 20.0), (30.0, 35.0)]
    assert intervals.merge([]) == []
    assert intervals.merge([(5.0, 5.0), (5.0, 6.0)]) == [(5.0, 6.0)]  # an empty span touches


@pytest.mark.parametrize(
    ("a", "b", "length", "shared"),
    [
        ((0.0, 10.0), (5.0, 20.0), 5.0, True),
        ((0.0, 10.0), (10.0, 20.0), 0.0, False),  # touching: half-open, no overlap
        ((0.0, 10.0), (12.0, 20.0), 0.0, False),  # apart: clamped at 0, never negative
        ((5.0, 5.0), (0.0, 10.0), 0.0, True),  # an empty span inside one: shares no length
    ],
)
def test_overlap_length_and_whether_spans_share_time(a, b, length, shared):
    assert intervals.overlap(*a, *b) == intervals.overlap(*b, *a) == length
    assert intervals.overlaps(*a, *b) is intervals.overlaps(*b, *a) is shared


@pytest.mark.parametrize(
    ("t", "taken", "expected"),
    [
        (30.0, [], (0.0, 100.0)),
        (30.0, [(10.0, 20.0), (40.0, 60.0)], (20.0, 40.0)),  # between two
        (5.0, [(10.0, 20.0)], (0.0, 10.0)),  # before the first
        (15.0, [(10.0, 20.0)], (15.0, 15.0)),  # inside one: nothing is free
        (40.0, [(40.0, 50.0), (10.0, 20.0)], (20.0, 40.0)),  # at a start, the list unsorted
        (50.0, [(40.0, 50.0)], (50.0, 100.0)),  # at an end
        (90.0, [(10.0, 20.0), (130.0, 140.0)], (20.0, 100.0)),  # after the last; one past the end
    ],
)
def test_free_around_is_the_stretch_no_taken_window_covers(t, taken, expected):
    assert intervals.free_around(t, taken, 0.0, 100.0) == expected
