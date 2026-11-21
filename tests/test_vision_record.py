"""The vision rating's record on a candidate (:mod:`cutpoint.vision_record`), on synthetic
candidates, and that the scorer reaches it without importing the analysis."""

from __future__ import annotations

import subprocess
import sys

import pytest

from cutpoint import vision_record
from cutpoint.vision_record import VISION_SPAN_EPS


def _cand(start: float = 10.0, end: float = 20.0) -> dict:
    return {"source": "a.mp4", "start": start, "end": end, "score": 50.0, "signals": {}}


@pytest.mark.parametrize(
    ("end", "skipped"),
    [(20.0, True), (20.0 + VISION_SPAN_EPS / 2, True), (20.0 + 2 * VISION_SPAN_EPS, False)],
)
def test_vision_skipped_holds_only_while_the_span_is_the_one_marked(end, skipped):
    c = _cand(end=end) | {"vision_skipped": [10.0, 20.0]}
    assert vision_record.vision_skipped(c) is skipped


def test_vision_skipped_is_false_without_a_mark_or_with_a_malformed_one():
    assert not vision_record.vision_skipped(_cand())
    assert not vision_record.vision_skipped(_cand() | {"vision_skipped": {"start": 10.0}})


def test_sync_vision_ranks_by_the_blend_on_the_rated_span_and_the_base_off_it():
    c = _cand() | {"vision_rated": {"start": 10.0, "end": 20.0, "base": 40.0, "blended": 46.0}}
    c["signals"]["vision"] = 8.0
    assert vision_record.sync_vision(c) == 46.0
    c["end"] = 25.0
    assert vision_record.sync_vision(c) == 40.0
    assert c["signals"]["vision"] is None and c["vision_stale"]["score"] == 8.0


def test_the_scorer_and_menu_spans_use_the_vision_record_without_importing_analyze():
    """#1085 A770: the vision clamp and the menu trim no longer import cutpoint.analyze."""
    code = (
        "import sys\n"
        "from cutpoint import scorer\n"
        "from cutpoint.events import menu_spans\n"
        "c = [{'source': 'a.mp4', 'start': 0.0, 'end': 5.0, 'score': 50.0,\n"
        "      'vision_skipped': [0.0, 5.0]}]\n"
        "scorer.apply_vision_clamp(c, [None])\n"
        "print('cutpoint.analyze' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert out.stdout.strip() == "False"
