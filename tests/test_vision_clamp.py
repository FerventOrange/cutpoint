"""The vision clamp (#909 part 3): a moment the vision model rated never passes a moment it did
not rate. Pure functions first, then the rescore that applies it after the events' bonus.

Every model call is faked (no network).
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import pytest

from cutpoint import analyze, scorer
from cutpoint.events import scoring
from test_event_candidates import (  # noqa: F401  (no_transcript is autouse there)
    events_of,
    feed,
    grid,
    highlights,
    no_transcript,
    own,
)
from test_vision_stale import STYLE, _Rater, talk

pytestmark = pytest.mark.usefixtures("pinned_prior")  # the rating centres on the prior

# ---- clamp_rated: rows are (score before vision, score with it or None), best first


@pytest.mark.parametrize(
    ("rows", "want"),
    [
        # a rating cannot lift a moment past the unrated one above it ...
        ([(90.0, None), (70.0, 95.0), (50.0, None)], [90.0, 90.0, 50.0]),
        # ... or lower it below the unrated one under it
        ([(90.0, None), (70.0, 20.0), (50.0, None)], [90.0, 50.0, 50.0]),
        # within the bounds the rating stands
        ([(90.0, None), (70.0, 77.0), (50.0, None)], [90.0, 77.0, 50.0]),
        # a tie with the neighbour is allowed (inclusive bounds)
        ([(90.0, None), (70.0, 90.0), (50.0, None)], [90.0, 90.0, 50.0]),
        ([(90.0, None), (70.0, 50.0), (50.0, None)], [90.0, 50.0, 50.0]),
        # neighbours that tied before vision give no room
        ([(50.0, None), (50.0, 60.0), (50.0, None)], [50.0, 50.0, 50.0]),
        # no unrated neighbour above: the bound is 100; none below: 0
        ([(70.0, 95.0), (60.0, None)], [95.0, 60.0]),
        ([(60.0, None), (50.0, 10.0)], [60.0, 10.0]),
        ([(80.0, 10.0), (70.0, 99.0)], [10.0, 99.0]),  # all rated: free
        # a rated stretch is bounded by the unrated ones around it
        ([(90.0, None), (80.0, 99.0), (70.0, 10.0), (60.0, None)], [90.0, 90.0, 60.0, 60.0]),
        # one decimal
        ([(90.0, None), (70.0, 77.77), (50.0, None)], [90.0, 77.8, 50.0]),
        ([], []),
    ],
)
def test_clamp_rated_keeps_a_rated_moment_between_its_unrated_neighbours(rows, want):
    assert scorer.clamp_rated(rows) == want


def test_rated_moments_between_the_same_neighbours_may_change_places():
    rows = [(90.0, None), (80.0, 70.0), (60.0, 85.0), (50.0, None)]
    assert scorer.clamp_rated(rows) == [90.0, 70.0, 85.0, 50.0]  # the lower-ranked one is now first


def test_no_rated_and_unrated_pair_is_ever_reordered():
    """Property: whatever the ratings, a moment that ranked above (below) an unrated one
    without a rating, or below (above) it with one, still ranks above (below) it, and an
    unrated moment keeps its score."""
    rng = random.Random(909)
    for _ in range(500):
        n = rng.randint(1, 14)
        before = sorted((round(rng.uniform(0, 100), 1) for _ in range(n)), reverse=True)
        rows = [(b, round(rng.uniform(0, 100), 1) if rng.random() < 0.5 else None) for b in before]
        out = scorer.clamp_rated(rows)
        assert len(out) == n
        for i, (b, after) in enumerate(rows):
            if after is None:
                assert out[i] == b
        for i in range(n):
            for j in range(i + 1, n):  # i ranked above j before vision
                if (rows[i][1] is None) != (rows[j][1] is None):
                    assert out[i] >= out[j], (rows, out, i, j)


# ---- apply_vision_clamp


def _cand(cid: str, score: float, **extra) -> dict:
    return {**own(cid, 0.0, score), **extra}


def test_apply_vision_clamp_changes_the_records_in_any_order_and_counts():
    a, b, c = _cand("a", 90.0), _cand("b", 95.0), _cand("c", 50.0)  # b was 70 before its rating
    cands = [c, b, a]
    assert scorer.apply_vision_clamp(cands, [None, 70.0, None]) == 1
    assert (a["score"], b["score"], c["score"]) == (90.0, 89.9, 50.0)  # 0.1 inside: a leads
    assert scorer.apply_vision_clamp(cands, [None, 70.0, None]) == 0  # nothing more to do


def _order(cands: list[dict]) -> list[str]:
    return [c["id"] for c in sorted(cands, key=scorer.rank_key)]


def test_a_rated_moment_on_the_score_of_an_unrated_one_above_is_nudged_under_it():
    """The clamp lands R on 60, U's score; R has no grid score, so the final sort ranks it by
    60 and U (grid 40) by 40 behind it: R is moved to 59.9 and U stays first."""
    u = _cand("u", 60.0, grid_score=40.0)
    r = _cand("r", 70.0)  # 55 before its rating
    assert scorer.apply_vision_clamp([u, r], [None, 55.0]) == 1
    assert (u["score"], r["score"]) == (60.0, 59.9)
    assert _order([u, r]) == ["u", "r"]

    assert scorer.apply_vision_clamp([u, r], [None, 55.0]) == 0  # a second apply
    assert (u["score"], r["score"]) == (60.0, 59.9)


def test_a_rated_moment_on_the_score_of_an_unrated_one_below_is_nudged_over_it():
    """Mirror: R is clamped down onto L (50); L's grid score 70 would rank it first."""
    r = _cand("r", 30.0)  # 60 before its rating
    low = _cand("l", 50.0, grid_score=70.0)
    assert scorer.apply_vision_clamp([r, low], [60.0, None]) == 1
    assert (r["score"], low["score"]) == (50.1, 50.0)
    assert _order([r, low]) == ["r", "l"]

    assert scorer.apply_vision_clamp([r, low], [60.0, None]) == 0
    assert r["score"] == 50.1


def test_equal_bounds_leave_the_tie():
    above, rated, below = _cand("a", 60.0), _cand("r", 90.0), _cand("b", 60.0)
    assert (
        scorer.apply_vision_clamp([above, rated, below], [None, 60.0, None]) == 1
    )  # 90 held at 60
    assert (above["score"], rated["score"], below["score"]) == (60.0, 60.0, 60.0)
    assert _order([above, rated, below]) == ["a", "r", "b"]  # by their places in the list


def test_bounds_0_1_apart_keep_the_tie_when_the_nudge_would_pass_the_neighbour_below():
    """R is clamped to 60 (U above, grid 40, would rank behind it). 59.9 would put it behind
    L, whose grid score is higher: no room, so R keeps 60."""
    u = _cand("u", 60.0, grid_score=40.0)
    r = _cand("r", 99.0)  # 59.95 before its rating
    low = _cand("l", 59.9, grid_score=70.0)
    assert scorer.apply_vision_clamp([u, r, low], [None, 59.95, None]) == 1
    assert (u["score"], r["score"], low["score"]) == (60.0, 60.0, 59.9)


def test_bounds_0_1_apart_with_room_nudge_to_the_lower_bound():
    u = _cand("u", 60.0, grid_score=40.0)
    r = _cand("r", 99.0)  # 59.9 before its rating
    low = _cand("l", 59.9)
    assert scorer.apply_vision_clamp([u, r, low], [None, 59.9, None]) == 1
    assert (u["score"], r["score"], low["score"]) == (60.0, 59.9, 59.9)
    assert _order([u, r, low]) == ["u", "r", "l"]


def test_a_tie_that_would_not_reorder_stays_a_tie():
    u = _cand("u", 60.0, grid_score=70.0)  # ranks ahead of R at 60 anyway
    r = _cand("r", 90.0)  # 55 before its rating
    assert scorer.apply_vision_clamp([u, r], [None, 55.0]) == 1
    assert (u["score"], r["score"]) == (60.0, 60.0)
    assert _order([u, r]) == ["u", "r"]


def test_the_final_order_never_puts_a_rated_moment_past_a_neighbour_it_was_held_by():
    """Property over the records: random scores on a coarse grid (many ties) and random grid
    scores; after ``rank_key`` no (rated, unrated) pair is reordered, unless the rated moment's
    bounds were equal (no room), and a second apply changes nothing."""
    rng = random.Random(909)
    for _ in range(500):
        n = rng.randint(2, 12)
        cands, before = [], []
        for k in range(n):
            pre = float(rng.randint(0, 10) * 10)
            grid = float(rng.randint(0, 10) * 10) if rng.random() < 0.5 else None
            rated = rng.random() < 0.5
            after = float(rng.randint(0, 10) * 10)
            extra = {} if grid is None else {"grid_score": grid}
            cands.append(_cand(f"c{k}", after if rated else pre, **extra))
            before.append(pre if rated else None)
        pre = [b if b is not None else c["score"] for b, c in zip(before, cands, strict=True)]
        ranked = sorted(
            range(n), key=lambda i: (-pre[i], -cands[i].get("grid_score", pre[i]), i)
        )  # the ranking before vision
        copy_before = copy.deepcopy(cands)
        scorer.apply_vision_clamp(cands, before)
        final = {c["id"]: p for p, c in enumerate(sorted(cands, key=scorer.rank_key))}
        for pos, i in enumerate(ranked):
            if before[i] is None:
                assert cands[i]["score"] == copy_before[i]["score"]
                continue
            up = [pre[j] for j in ranked[:pos] if before[j] is None]
            down = [pre[j] for j in ranked[pos + 1 :] if before[j] is None]
            hi, lo = (up[-1] if up else 100.0), (down[0] if down else 0.0)
            if hi - lo < 0.2:  # no room to tell it from the neighbours
                continue
            for pos2, j in enumerate(ranked):
                if before[j] is not None:
                    continue
                ctx = (copy_before, before, cands, i, j)
                if pos2 < pos:
                    assert final[cands[j]["id"]] < final[cands[i]["id"]], ctx
                else:
                    assert final[cands[j]["id"]] > final[cands[i]["id"]], ctx
        again = [c["score"] for c in cands]
        assert scorer.apply_vision_clamp(cands, before) == 0
        assert [c["score"] for c in cands] == again


def test_must_include_and_pool_moments_are_no_ones_neighbours():
    forced = _cand("m", 100.0, must_include=True)
    pool = _cand("v001", 99.0, pool="visual")
    rated = _cand("r", 95.0)  # 70 before its rating
    low = _cand("l", 40.0)
    assert scorer.apply_vision_clamp([forced, pool, rated, low], [None, None, 70.0, None]) == 0
    assert rated["score"] == 95.0  # no unrated moment above it: the bound is 100


def test_a_moment_without_a_rating_that_counts_is_an_unrated_neighbour():
    """A failed, capped or stale rating is ``None``: the moment keeps its own score and bounds
    the rated ones around it."""
    stuck = _cand("s", 60.0)  # its rating failed or went stale
    rated = _cand("r", 99.0)  # 40 before
    assert scorer.apply_vision_clamp([stuck, rated], [None, 40.0]) == 1
    assert (stuck["score"], rated["score"]) == (60.0, 60.0)


def test_a_tie_before_vision_is_ranked_by_grid_score():
    """Two moments on the same score: the one with the higher ``grid_score`` ranks first, so
    the unrated one above the rated bounds it only when its grid score is higher."""
    above = _cand("u", 70.0, grid_score=80.0)
    rated = _cand("r", 95.0, grid_score=70.0)  # 70 before
    assert scorer.apply_vision_clamp([above, rated], [None, 70.0]) == 1
    assert rated["score"] == 70.0

    below = _cand("u", 70.0, grid_score=10.0)  # now the unrated one ranks under it
    rated = _cand("r", 95.0, grid_score=70.0)
    assert scorer.apply_vision_clamp([below, rated], [None, 70.0]) == 0
    assert rated["score"] == 95.0


# ---- rescore_events: the bonus and the clamp


def _rate(c: dict, rating: float = 10.0) -> None:
    analyze._rate_vision([c], _Rater(rating), STYLE, None, 0, deadline=float("inf"))  # type: ignore[arg-type]


def test_a_rated_moment_with_a_bonus_stays_under_the_unrated_one_above_it(tmp_path):
    """Rating 10 lifts h002 from 50 to 67.5; the kill on it adds 6 points. Without a rating
    it would be 56, under the unrated h001 (70): the clamp keeps it there, and the same on a
    second rescore."""
    cands = [own("h001", 0.0, 70.0), own("h002", 100.0, 50.0), own("h003", 300.0, 40.0)]
    _rate(cands[1])
    assert cands[1]["score"] == 67.5
    hl = highlights(cands, vision={})
    ev = events_of([feed(100.0)])
    scorer.rescore_events(hl, ev, tmp_path, [grid()])
    first, second, third = hl["candidates"]
    assert [c["id"] for c in hl["candidates"]] == ["h001", "h002", "h003"]
    assert second["signals"]["events"]["boost"] == 6.0
    assert second["score_pre_events"] == 67.5  # the blend itself is not clamped
    assert (first["score"], second["score"], third["score"]) == (70.0, 70.0, 40.0)
    assert hl["settings"]["vision"]["clamped"] == 1

    again = copy.deepcopy(hl)
    for _ in range(2):
        scorer.rescore_events(hl, ev, tmp_path, [grid()])
        assert hl == again  # idempotent


def test_a_stale_rating_counts_as_unrated_and_bounds_the_rated_below_it(tmp_path):
    """A dialogue stretch widens h002 over the span it was rated: its rating is stale, so it
    ranks as unrated (50 + 3 points) and h004's rating cannot lift that one past it."""
    cands = [own("h001", 0.0, 70.0), own("h002", 100.0, 50.0), own("h004", 400.0, 45.0)]
    _rate(cands[1])
    _rate(cands[2])
    hl = highlights(cands, vision={})
    ev = events_of([talk(60.0, 140.0)])
    scorer.rescore_events(hl, ev, tmp_path, [grid()])
    by_id = {c["id"]: c for c in hl["candidates"]}
    stale, rated = by_id["h002"], by_id["h004"]
    assert "vision_stale" in stale and "vision_stale" not in rated
    assert stale["score"] == pytest.approx(50.0 + scoring.DIALOGUE_POINTS)  # no vision lift
    assert rated["score_pre_events"] == pytest.approx(45.0 * 1.35, abs=0.1)  # 60.75, unclamped
    assert rated["score"] == stale["score"]  # held at the unrated one above it
    assert by_id["h001"]["score"] == 70.0
    assert hl["settings"]["vision"]["clamped"] == 1

    again = copy.deepcopy(hl)
    scorer.rescore_events(hl, ev, Path(tmp_path), [grid()])
    assert hl == again


# ---- moments the default edit skipped above its cut (vision_skipped)


def _skipped(cid: str, start: float, score: float) -> dict:
    c = own(cid, start, score)
    c["vision_skipped"] = [c["start"], c["end"]]
    return c


def test_a_moment_the_edit_skipped_is_no_neighbour():
    """S, skipped by the edit above its cut, does not bound the rated R: R passes it. The
    unrated U under them still bounds R."""
    s, r, u = _skipped("s", 0.0, 80.0), _cand("r", 99.0), _cand("u", 30.0)  # r was 70
    assert scorer.apply_vision_clamp([s, r, u], [None, 70.0, None]) == 0
    assert (s["score"], r["score"], u["score"]) == (80.0, 99.0, 30.0)
    r["score"] = 10.0  # rated low instead: held by U, not by S
    assert scorer.apply_vision_clamp([s, r, u], [None, 70.0, None]) == 1
    assert r["score"] == 30.0


def test_a_skipped_moment_with_another_span_is_a_neighbour_again():
    """Once its span changes (a menu trim, a dialogue stretch) the moment is no longer the one
    the edit skipped, so it bounds the rated moments again."""
    s, r = _skipped("s", 0.0, 80.0), _cand("r", 99.0)  # r was 70
    s["end"] += 5.0
    assert not analyze.vision_skipped(s)
    assert scorer.apply_vision_clamp([s, r], [None, 70.0]) == 1
    assert r["score"] == 80.0  # held at S (a tie that keeps S first)


def test_rescore_lets_a_rating_pass_a_skipped_moment_and_is_idempotent(tmp_path):
    """The rescore after the events' bonus reads the same mark: h002 rated 10 (67.5) passes
    the skipped h001 (60) and stays under nothing; a second rescore changes nothing. Without
    events the result is the clamp's own."""
    cands = [_skipped("h001", 0.0, 60.0), own("h002", 100.0, 50.0), own("h003", 300.0, 40.0)]
    _rate(cands[1])
    direct = copy.deepcopy(cands)
    scorer.apply_vision_clamp(direct, [None, direct[1]["vision_rated"]["base"], None])
    hl = highlights(cands, vision={})
    scorer.rescore_events(hl, events_of([]), tmp_path, [grid()])
    assert [c["id"] for c in hl["candidates"]] == ["h002", "h001", "h003"]
    assert {c["id"]: c["score"] for c in hl["candidates"]} == {c["id"]: c["score"] for c in direct}
    assert hl["candidates"][0]["score"] == 67.5
    assert hl["settings"]["vision"]["clamped"] == 0

    again = copy.deepcopy(hl)
    for _ in range(2):
        scorer.rescore_events(hl, events_of([]), tmp_path, [grid()])
        assert hl == again


def test_a_stale_rating_still_bounds_with_a_skipped_moment_around(tmp_path):
    """A dialogue stretch widens h002 over the span it was rated: its stale rating counts as
    unrated, so it bounds the rated h004 below it; the skipped h001 above them does not."""
    cands = [
        _skipped("h001", 0.0, 70.0),
        own("h002", 100.0, 50.0),
        own("h004", 400.0, 45.0),
    ]
    _rate(cands[1])
    _rate(cands[2])
    hl = highlights(cands, vision={})
    ev = events_of([talk(60.0, 140.0)])
    scorer.rescore_events(hl, ev, tmp_path, [grid()])
    by_id = {c["id"]: c for c in hl["candidates"]}
    stale, rated = by_id["h002"], by_id["h004"]
    assert "vision_stale" in stale and "vision_stale" not in rated
    assert rated["score"] == stale["score"]  # held at the stale one above it
    assert by_id["h001"]["score"] == 70.0


# ---- #949: the mark goes once the default edit before vision takes the moment


def _skip_set(pick_score: float) -> list[dict]:
    """h000 (``pick_score``) overlaps the skipped h001 (60) by 6 of their 8 s; h002 (50) is
    rated 10 (67.5); h003 (40) does not fit in the 16 s edit."""
    cands = [
        own("h000", 2.0, pick_score),
        _skipped("h001", 0.0, 60.0),
        own("h002", 100.0, 50.0),
        own("h003", 300.0, 40.0),
    ]
    _rate(cands[2])
    return cands


def test_a_moment_still_skipped_keeps_its_mark(tmp_path):
    """The pick h000 (70) still overlaps h001 in the edit before vision: h001 keeps its mark
    and the rated h002 passes it, held only by h000."""
    hl = highlights(_skip_set(70.0), vision={}, target_total=16.0)
    scorer.rescore_events(hl, events_of([]), tmp_path, [grid()])
    by_id = {c["id"]: c for c in hl["candidates"]}
    assert by_id["h001"]["vision_skipped"] == [0.0, 8.0]
    assert by_id["h002"]["score"] == 67.5
    assert [c["id"] for c in hl["candidates"]] == ["h000", "h002", "h001", "h003"]


def test_a_skipped_moment_the_edit_now_takes_loses_its_mark(tmp_path):
    """A re-rank drops the pick h000 to 45, under h001 and h002: the edit before vision now
    takes h001 and h002, so the mark goes and h001 bounds the rated h002 again (0.1 under it,
    as h002 is listed first after the first rescore); a second rescore does not mark it
    again and h002 stays under or level with it."""
    hl = highlights(_skip_set(70.0), vision={}, target_total=16.0)
    scorer.rescore_events(hl, events_of([]), tmp_path, [grid()])
    pick = next(c for c in hl["candidates"] if c["id"] == "h000")
    pick["score_pre_events"] = 45.0  # the re-rank's analysis score
    scorer.rescore_events(hl, events_of([]), tmp_path, [grid()])
    by_id = {c["id"]: c for c in hl["candidates"]}
    assert "vision_skipped" not in by_id["h001"]
    assert by_id["h002"]["score"] == 59.9  # held under h001
    assert hl["settings"]["vision"]["clamped"] == 1
    assert [c["id"] for c in hl["candidates"]] == ["h001", "h002", "h000", "h003"]
    scorer.rescore_events(hl, events_of([]), tmp_path, [grid()])
    by_id = {c["id"]: c for c in hl["candidates"]}
    assert "vision_skipped" not in by_id["h001"]
    assert [c["id"] for c in hl["candidates"]][:2] == ["h001", "h002"]
    assert by_id["h002"]["score"] <= by_id["h001"]["score"] == 60.0


def test_no_vision_project_is_left_as_it_is(tmp_path):
    """Without vision there is no mark to drop and none is added."""
    cands = [own("h000", 2.0, 70.0), own("h001", 0.0, 60.0), own("h002", 100.0, 50.0)]
    hl = highlights(cands, target_total=16.0)
    scorer.rescore_events(hl, events_of([]), tmp_path, [grid()])
    assert not any("vision_skipped" in c for c in hl["candidates"])
    assert [c["score"] for c in hl["candidates"]] == [70.0, 60.0, 50.0]
