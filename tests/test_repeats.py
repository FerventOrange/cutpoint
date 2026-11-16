"""Repeat damping (#1071): a key said often in a source counts less each time."""

from __future__ import annotations

import math

import pytest

from cutpoint import repeats


@pytest.mark.parametrize(
    ("count", "span_s", "expected"),
    [
        (3, 600, 1.0),  # a short source still allows three
        (4, 600, math.sqrt(3 / 4)),
        (36, 3 * 3600, math.sqrt(9 / 36)),  # 3 h: allowance 9
        (9, 3 * 3600, 1.0),  # up to the allowance keeps the full weight
        (12, 4 * 3600, 1.0),  # allowance 12
        (100, 0, math.sqrt(3 / 100)),
    ],
)
def test_repeat_factor_is_the_square_root_of_allowance_over_count(count, span_s, expected):
    assert repeats.repeat_factor(count, span_s) == pytest.approx(expected)


def test_a_damped_key_total_grows_only_with_the_square_root_of_its_count():
    span = 3 * 3600
    for count in (36, 144):
        total = count * repeats.repeat_factor(count, span)
        assert total == pytest.approx(math.sqrt(9 * count))


def test_repeat_factors_count_each_distinct_key_once_per_occurrence():
    got = repeats.repeat_factors(["a"] * 12 + ["b"] * 2, 600)
    assert got["b"] == 1.0
    assert got["a"] == pytest.approx(math.sqrt(3 / 12))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Round 2", "ROUND3"),
        ("Attacker safeguards active", "Attackersafeguardsactive"),
        ("4VS4", "12 vs 12"),
    ],
)
def test_text_key_ignores_digits_spacing_and_case(a, b):
    assert repeats.text_key(a) == repeats.text_key(b)


def test_text_key_keeps_different_words_apart():
    assert repeats.text_key("ACTION PHASE") != repeats.text_key("PREP PHASE")
    assert repeats.text_key("Round 2") == "ROUND#"
