"""The vision band, the per-kind centring and the time cap, as pure functions (#909 part 3):
which moments the vision model rates, how its ratings are put on one scale and when it stops.
"""

from __future__ import annotations

import pytest

from cutpoint import analyze, vision

# ---- band_k


@pytest.mark.parametrize(
    ("picks", "k"),
    [
        (1, 2),  # at least 2
        (5, 2),  # 1.5 rounds up to 2
        (6, 2),  # 1.8
        (7, 3),  # 2.1 rounds up
        (10, 3),  # 3.0 exactly: no float error adds one
        (11, 4),
        (80, 24),  # 24.0 exactly
        (200, 24),  # at most 24
    ],
)
def test_band_k_is_a_share_of_the_edits_moments_within_2_and_24(picks, k):
    assert analyze.band_k(picks) == k


# ---- vision_band


def _ranked(n: int, **flags) -> list[dict]:
    return [{"id": i, **flags} for i in range(n)]


def _ids(order: list[dict]) -> list[int]:
    return [c["id"] for c in order]


def test_the_band_is_the_edit_and_k_below_the_cut_closest_first():
    ranked = _ranked(12)
    picks = ranked[:6]  # the edit takes the 6 best: k = 2, the cut is after rank 5
    order, cut, k = analyze.vision_band(ranked, picks, None)
    assert (cut, k) == (6, 2)
    # the last pick, the first below, the pick before, the second below, then the other picks
    assert _ids(order) == [5, 6, 4, 7, 3, 2, 1, 0]


def test_the_sides_interleave_and_the_longer_one_goes_on():
    ranked = _ranked(30)
    picks = ranked[:10]  # k = band_k(10) = 3
    order, cut, k = analyze.vision_band(ranked, picks, None)
    assert (cut, k) == (10, 3)
    assert _ids(order) == [9, 10, 8, 11, 7, 12, 6, 5, 4, 3, 2, 1, 0]


def test_k_comes_from_the_number_of_picks():
    ranked = _ranked(40)
    for n in (1, 7, 11):
        _, cut, k = analyze.vision_band(ranked, ranked[:n], None)
        assert (cut, k) == (n, analyze.band_k(n))


def test_the_band_below_is_clipped_at_the_end():
    ranked = _ranked(3)
    order, cut, k = analyze.vision_band(ranked, ranked[:3], None)  # the edit takes everything
    assert (cut, k) == (3, 2) and _ids(order) == [2, 1, 0]

    ranked = _ranked(8)
    order, cut, k = analyze.vision_band(ranked, ranked[:7], None)  # one moment below the cut
    assert (cut, k) == (7, 3) and _ids(order) == [6, 7, 5, 4, 3, 2, 1, 0]


def test_short_lists():
    assert analyze.vision_band([], [], None)[0] == []
    one = _ranked(1)
    order, cut, _ = analyze.vision_band(one, one, None)
    assert cut == 1 and _ids(order) == [0]  # a single moment is its own band
    ranked = _ranked(5)
    order, cut, _ = analyze.vision_band(ranked, ranked[:1], None)  # one pick: cut 1, k 2
    assert cut == 1 and _ids(order) == [0, 1, 2]


def test_picks_that_are_not_a_prefix_skip_the_moments_left_out_for_their_length():
    ranked = _ranked(10)
    picks = [ranked[i] for i in (0, 1, 3, 6)]  # 2, 4 and 5 were too long: k = band_k(4) = 2
    order, cut, k = analyze.vision_band(ranked, picks, None)
    assert (cut, k) == (7, 2)
    assert _ids(order) == [6, 7, 3, 8, 1, 0]  # not 2, 4 or 5: they are not in the edit


def test_a_must_include_pick_is_not_rated_and_not_the_cut():
    ranked = _ranked(10)  # the candidates a forced crash is not among
    forced = {"id": "crash", "must_include": True}
    picks = [forced, *ranked[:5]]  # 6 picks: k = 2; the cut follows the last ranked pick
    order, cut, k = analyze.vision_band(ranked, picks, None)
    assert (cut, k) == (5, 2) and _ids(order) == [4, 5, 3, 6, 2, 1, 0]

    order, cut, _ = analyze.vision_band(ranked, [forced], None)  # only forced picks: cut 0
    assert cut == 0 and _ids(order) == [0, 1]


def test_max_n_caps_the_whole_band_keeping_the_moments_closest_to_the_cut():
    ranked = _ranked(12)
    order, cut, _ = analyze.vision_band(ranked, ranked[:6], 3)
    assert cut == 6 and _ids(order) == [5, 6, 4]
    assert _ids(analyze.vision_band(ranked, ranked[:6], 5)[0]) == [5, 6, 4, 7, 3]
    assert _ids(analyze.vision_band(ranked, ranked[:6], 99)[0]) == [5, 6, 4, 7, 3, 2, 1, 0]


# ---- rating_centres and centred_rating


def test_a_kind_with_five_ratings_is_centred_on_their_mean(pinned_prior):
    got = analyze.rating_centres([("frames", r) for r in (2.0, 4.0, 6.0, 8.0, 10.0)])
    assert got == {"frames": 6.0}  # no shrinking from PRIOR_N on


def test_a_kind_with_few_ratings_is_shrunk_toward_its_prior(pinned_prior):
    # (sum + (5 - n) * prior) / 5, prior 3.2 for frames, 5.1 for speech
    got = analyze.rating_centres([("frames", 2.0), ("frames", 4.0), ("speech", 7.0)])
    assert got["frames"] == pytest.approx((2 + 4 + 3 * 3.2) / 5)  # 3.12
    assert got["speech"] == pytest.approx((7 + 4 * 5.1) / 5)  # 5.48


def test_only_the_kinds_rated_have_a_centre_and_each_is_apart(pinned_prior):
    assert analyze.rating_centres([]) == {}
    got = analyze.rating_centres([("speech", 9.0)] * 6 + [("frames", 1.0)] * 5)
    assert got == {"speech": 9.0, "frames": 1.0}


def test_the_prior_is_one_constant_per_kind_for_every_model():
    assert set(vision.RATING_PRIOR) == {"frames", "speech"}
    assert vision.PRIOR_N == 5
    assert not hasattr(vision, "SPEECH_PROMPT_LIFT")  # replaced by the centring


@pytest.mark.parametrize(
    ("rating", "centre", "want"),
    [
        (4.0, 4.0, 5.0),  # the centre is the neutral 5
        (6.0, 4.0, 7.0),
        (2.0, 4.0, 3.0),
        (10.0, 4.0, 10.0),  # 11 clips to 10
        (0.0, 4.0, 1.0),
        (1.0, 7.0, 0.0),  # -1 clips to 0
    ],
)
def test_centred_rating_moves_the_centre_to_5_within_0_and_10(rating, centre, want):
    assert analyze.centred_rating(rating, centre) == pytest.approx(want)


# ---- the time cap


@pytest.mark.parametrize(
    ("now", "deadline", "calls", "want"),
    [
        (0.0, 10.0, 0, False),  # before the deadline
        (10.0, 10.0, 1, True),  # at it
        (11.0, 10.0, 3, True),
        (9.9, 10.0, 5, False),
        (50.0, 10.0, 0, False),  # past it but no call yet: one call is always made
        (0.0, 0.0, 0, False),  # share 0: the first call still goes
        (0.0, 0.0, 1, True),  # ... and then it stops
    ],
)
def test_the_cap_stops_calls_only_after_one_was_made_and_the_deadline_passed(
    now, deadline, calls, want
):
    assert analyze.cap_reached(now, deadline, calls) is want


def test_the_budget_is_a_share_of_the_time_before_vision():
    assert analyze.vision_budget(0.3, 100.0, None) == (pytest.approx(30.0), "pre_vision_s")


def test_a_rerank_measures_against_the_first_analysis():
    budget, base = analyze.vision_budget(0.3, 5.0, 200.0)  # this run's own 5 s is mostly cached
    assert (budget, base) == (pytest.approx(60.0), "earlier_total_s")


def test_a_share_of_0_is_a_budget_of_0():
    assert analyze.vision_budget(0.0, 100.0, None)[0] == 0.0
    assert analyze.vision_budget(0.0, 100.0, 300.0)[0] == 0.0
