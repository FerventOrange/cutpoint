"""A vision rating holds only for the span it rated (#802): a candidate widened by a dialogue
stretch, merged back or trimmed out of menu time loses the rating's blend and its label until
the span is the rated one again. Projects rated before ``vision_rated`` existed stay as they were.

Every model call is faked (no network).
"""

from __future__ import annotations

from typing import Any

import pytest

from cutpoint import analyze, scorer, vision
from cutpoint.events import scoring
from cutpoint.events.menu_spans import drop_menu_time
from cutpoint.styles import get_style
from test_event_candidates import (  # noqa: F401  (no_transcript is autouse there)
    SRC,
    events_of,
    grid,
    highlights,
    no_transcript,
    own,
)

STYLE = get_style("gaming")


class _Rater:
    """Rates every span ``rating`` with ``label``; records nothing."""

    hits = asked = 0
    cache_only = False

    def __init__(self, rating: float = 10.0, label: str = "big jump") -> None:
        self.rating, self.label = rating, label

    def cached(self, *_a, **_k) -> None:
        return None

    def rate(self, _src, _start, _end, _style, activity=None, transcript=None, laughter=False):
        return vision.VisionResult(self.rating, self.label)

    def unload(self) -> None:
        pass


def talk(start: float, end: float) -> dict[str, Any]:
    return {
        "source": SRC, "start": start, "end": end, "kind": "text", "type": "dialogue",
        "confidence": 1.0, "funny": None, "hype": None, "label": "funny", "description": "a bit",
    }  # fmt: skip


def rate(c: dict[str, Any]) -> None:
    analyze._rate_vision([c], _Rater(), 1, STYLE)  # type: ignore[arg-type]


BLENDED = analyze.vision_blend(40.0, 10.0, STYLE.vision_weight)


def test_rating_records_the_span_and_both_scores():
    c = own("h001", 150.0, 40.0, end=158.0)
    rate(c)
    assert c["vision_rated"] == {
        "start": 150.0,
        "end": 158.0,
        "base": 40.0,
        "blended": BLENDED,
    }
    assert c["score"] == BLENDED != 40.0 and c["label"] == "gaming-moment"
    assert c["vision_label"] == "big jump"


def test_a_widened_rated_candidate_ranks_without_the_rating_and_gets_it_back(tmp_path):
    hl = highlights([own("h001", 150.0, 40.0, end=158.0)])
    rate(hl["candidates"][0])
    scorer.rescore_events(hl, events_of([talk(100.0, 180.0)]), tmp_path, [grid()])
    (c,) = hl["candidates"]
    assert (c["start"], c["end"]) == (100.0, 180.0)  # widened: the frames are not this clip
    assert c["vision_stale"] == {
        "score": 10.0,
        "label": "big jump",
        "window": [150.0, 158.0],
    }
    assert c["signals"]["vision"] is None and "vision_label" not in c
    assert c["label"] == "gaming-moment"  # the rating never touched it
    assert c["score_pre_events"] == 40.0
    assert c["score"] == pytest.approx(40.0 + scoring.DIALOGUE_POINTS)

    scorer.rescore_events(hl, events_of([]), tmp_path, [grid()])  # the stretch is gone
    (c,) = hl["candidates"]
    assert (c["start"], c["end"]) == (150.0, 158.0)
    assert "vision_stale" not in c
    assert c["signals"]["vision"] == 10.0 and c["vision_label"] == "big jump"
    assert c["label"] == "gaming-moment"
    assert c["score_pre_events"] == c["score"] == BLENDED


def test_a_candidate_rated_over_its_widened_span_keeps_the_rating_on_a_rerank(tmp_path):
    """The analysis rates after the first re-rank widened it; the re-rank after the rating
    widens it back to the same span, so the rating holds."""
    hl = highlights([own("h001", 150.0, 40.0, end=158.0)])
    events = events_of([talk(100.0, 180.0)])
    scorer.rescore_events(hl, events, tmp_path, [grid()])
    rate(hl["candidates"][0])
    scorer.rescore_events(hl, events, tmp_path, [grid()])
    (c,) = hl["candidates"]
    rated = c["vision_rated"]
    assert (c["start"], c["end"]) == (rated["start"], rated["end"]) == (100.0, 180.0)
    assert "vision_stale" not in c and c["signals"]["vision"] == 10.0
    assert c["score_pre_events"] == BLENDED
    assert c["score"] == pytest.approx(BLENDED + scoring.DIALOGUE_POINTS)


def test_a_rating_under_a_dialogue_title_gives_back_the_own_label(tmp_path):
    """Rated while it carries a stretch's title: once the stretch goes the candidate gets its
    own label (not the title's slug), and the title again when the stretch is back."""
    hl = highlights([own("h001", 150.0, 40.0, end=158.0)])
    titled = events_of([{**talk(100.0, 180.0), "title": "Bob and the door"}])
    scorer.rescore_events(hl, titled, tmp_path, [grid()])
    assert hl["candidates"][0]["label"] == "bob-and-the-door"
    rate(hl["candidates"][0])
    assert hl["candidates"][0]["label"] == "bob-and-the-door"
    scorer.rescore_events(hl, titled, tmp_path, [grid()])
    scoring.restore_widened(hl["candidates"])
    scorer.rescore_events(hl, events_of([]), tmp_path, [grid()])
    (c,) = hl["candidates"]
    assert (c["start"], c["end"]) == (150.0, 158.0)
    assert c["label"] == "gaming-moment" and "dialogue_title" not in c
    assert "vision_stale" in c
    scorer.rescore_events(hl, titled, tmp_path, [grid()])
    (c,) = hl["candidates"]
    assert c["label"] == "bob-and-the-door" and "vision_stale" not in c
    assert c["vision_label"] == "big jump"


def test_a_menu_trim_makes_the_rating_stale():
    c = {**own("h001", 10.0, 40.0, end=30.0), "duration": 20.0}
    rate(c)
    hl = highlights([c])
    ev = {"sources": [SRC], "events": [], "stats": {"ocr": {SRC: {"menu_spans": [[5, 12]]}}}}
    drop_menu_time(hl, ev)
    (c,) = hl["candidates"]
    assert (c["start"], c["end"]) == (12, 30)
    assert c["vision_stale"]["window"] == [10.0, 30.0]
    assert c["signals"]["vision"] is None and "vision_label" not in c
    assert c["label"] == "gaming-moment"
    assert analyze.sync_vision(c) == 40.0  # what a rescore ranks it by


def test_a_label_changed_since_the_rating_is_not_touched():
    c = own("h001", 150.0, 40.0, end=158.0)
    rate(c)
    c["label"] = "renamed-by-hand"
    c["start"] = 100.0
    assert analyze.sync_vision(c) == 40.0
    assert c["label"] == "renamed-by-hand" and "renamed" not in c["vision_stale"]
    c["start"] = 150.0
    assert analyze.sync_vision(c) == BLENDED
    assert c["label"] == "renamed-by-hand" and c["vision_label"] == "big jump"


def test_a_span_within_the_epsilon_still_matches():
    c = own("h001", 150.0, 40.0, end=158.0)
    rate(c)
    c["end"] = 158.0 + analyze.VISION_SPAN_EPS / 2
    assert analyze.sync_vision(c) == BLENDED and "vision_stale" not in c


def test_a_candidate_rated_before_vision_rated_existed_is_unchanged(tmp_path):
    legacy = own("h001", 150.0, BLENDED, end=158.0)
    legacy["signals"]["vision"] = 10.0
    legacy["vision_label"] = "big jump"
    legacy["label"] = "big-jump"
    hl = highlights([legacy])
    scorer.rescore_events(hl, events_of([talk(100.0, 180.0)]), tmp_path, [grid()])
    (c,) = hl["candidates"]
    assert (c["start"], c["end"]) == (100.0, 180.0)
    assert "vision_stale" not in c and "vision_rated" not in c
    assert c["signals"]["vision"] == 10.0 and c["vision_label"] == "big jump"
    assert c["label"] == "big-jump" and c["score_pre_events"] == BLENDED
    assert analyze.sync_vision(c) is None
