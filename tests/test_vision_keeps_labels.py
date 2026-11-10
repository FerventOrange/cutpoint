"""--vision never renames a clip (#913): the model's description goes to ``vision_label`` and
``label`` (and so the clip file name) is what it was without vision, whatever the rating.

Every model call is faked (no network, no Ollama).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import cutpoint.analyze
from conftest import needs_ffmpeg
from cutpoint import analyze, cli, scorer, vision
from cutpoint.styles import get_style
from cutpoint.utils import slugify
from test_event_candidates import (  # noqa: F401  (no_transcript is autouse there)
    events_of,
    grid,
    highlights,
    no_transcript,
    own,
)
from test_vision_stale import BLENDED, rate, talk


class _HighRater:
    """A reachable Ollama that rates every span 9/10 and describes it."""

    hits = asked = missed = 0
    stats = dict.fromkeys(
        ("calls", "grab_s", "load_s", "prompt_s", "reply_s", "prompt_tokens", "reply_tokens"), 0
    )
    model = "m"
    provider_name = "ollama"
    consent_needed = None
    cache_only = False

    def __init__(self, **_kw) -> None:
        pass

    def cached(self, *_a, **_k) -> None:
        return None

    def available(self) -> bool:
        return True

    def rate(self, *_a, **_k):
        return vision.VisionResult(9.0, "Spectacular Big Jump")

    def unload(self) -> None:
        pass


def _clip_name(idx: int, c: dict[str, Any]) -> str:
    """The file name ``clips.py`` gives the ``idx``-th moment (plan_clips_preview)."""
    return f"{idx:03d}_{slugify(c.get('label', ''), 40)}.mp4"


def _load(home: Path, name: str) -> dict[str, Any]:
    return json.loads((home / "projects" / name / "highlights.json").read_text("utf-8"))


def _argv(video: Path, out: str, *flags: str) -> list[str]:
    skip = ["--skip", "reactions", "keyframes", "crashes"]
    return ["analyze", str(video), "--style", "commentary", "--out", out, "--no-scenes",
            *flags, *skip]  # fmt: skip


@needs_ffmpeg
def test_analyze_with_vision_gives_the_labels_and_clip_names_of_a_run_without(
    video, home, monkeypatch
):
    monkeypatch.setattr(cutpoint.analyze, "VisionRater", _HighRater)
    assert cli.main(_argv(video, "plain")) == 0
    assert cli.main(_argv(video, "seen", "--vision")) == 0
    plain, seen = _load(home, "plain"), _load(home, "seen")
    assert seen["settings"]["vision_used"] is True
    by_span = lambda hl: sorted(hl["candidates"], key=lambda c: (c["start"], c["end"]))  # noqa: E731
    a, b = by_span(plain), by_span(seen)
    assert a and [(c["start"], c["end"]) for c in a] == [(c["start"], c["end"]) for c in b]
    assert [c["label"] for c in a] == [c["label"] for c in b]
    assert [_clip_name(i, c) for i, c in enumerate(a, 1)] == [
        _clip_name(i, c) for i, c in enumerate(b, 1)
    ]
    rated = [c for c in b if "vision_rated" in c]
    assert rated
    for c in rated:
        assert c["vision_label"] == "Spectacular Big Jump"
        assert c["label"] != "spectacular-big-jump"
        assert "label_before" not in c["vision_rated"] and "renamed" not in c["vision_rated"]
        assert "label_before" not in c and "renamed" not in c
    assert not any("vision_label" in c or "vision_rated" in c for c in a)


@pytest.mark.parametrize("rating", [0.0, 6.9, 7.0, 10.0])
def test_a_candidate_keeps_its_label_whatever_the_rating(rating):
    c = own("h001", 150.0, 40.0, end=158.0)
    c["transcript"] = "hello there"
    analyze._rate_vision(
        [c],
        _RatingRater(rating),
        1,
        get_style("action-cam"),  # type: ignore[arg-type]
    )
    assert c["label"] == "gaming-moment"
    assert c["vision_label"] == "big jump"
    assert not {"label_before", "renamed"} & (set(c) | set(c["vision_rated"]))


class _RatingRater(_HighRater):
    def __init__(self, rating: float) -> None:
        self.rating = rating

    def rate(self, *_a, **_k):
        return vision.VisionResult(self.rating, "big jump")


def test_a_high_rating_keeps_the_transcript_slug_event_label_and_dialogue_title(tmp_path):
    """Each kind of label a candidate can carry survives a rating of 7 or more."""
    for label in ("hello-there-friend", "kill-feed", "gaming-moment"):
        c = {**own("h001", 150.0, 40.0, end=158.0), "label": label}
        rate(c)
        assert c["label"] == label and c["vision_label"] == "big jump"

    hl = highlights([own("h001", 150.0, 40.0, end=158.0)])
    titled = events_of([{**talk(100.0, 180.0), "title": "Bob and the door"}])
    scorer.rescore_events(hl, titled, tmp_path, [grid()])
    assert hl["candidates"][0]["label"] == "bob-and-the-door"
    rate(hl["candidates"][0])
    assert hl["candidates"][0]["label"] == "bob-and-the-door"
    assert hl["candidates"][0]["dialogue_title"]
    scorer.rescore_events(hl, titled, tmp_path, [grid()])
    assert hl["candidates"][0]["label"] == "bob-and-the-door"


def test_an_old_project_with_label_before_and_renamed_loads_and_rescores(tmp_path):
    """Written by an older Cutpoint: a rating that renamed the clip. It must not crash, and
    the keys are ignored (the label stays as saved)."""
    old = own("h001", 150.0, BLENDED, end=158.0)
    old.update(
        label="big-jump",
        vision_label="big jump",
        vision_rated={
            "start": 150.0, "end": 158.0, "base": 40.0, "blended": BLENDED,
            "label_before": "gaming-moment", "label": "big-jump",
        },
    )  # fmt: skip
    old["signals"]["vision"] = 10.0
    hl = highlights([old])
    json.loads(json.dumps(hl))  # round-trips as saved JSON

    assert analyze.sync_vision(old) == BLENDED
    assert old["label"] == "big-jump"

    scorer.rescore_events(hl, events_of([talk(100.0, 180.0)]), tmp_path, [grid()])
    (c,) = hl["candidates"]  # span changed: the rating goes stale, the label is not restored
    assert c["vision_stale"]["window"] == [150.0, 158.0]
    assert "renamed" not in c["vision_stale"]
    assert c["label"] == "big-jump"
    assert c["signals"]["vision"] is None

    stale = {**c, "vision_stale": {**c["vision_stale"], "renamed": True}}  # an old stale block
    stale["start"], stale["end"] = 150.0, 158.0
    assert analyze.sync_vision(stale) == BLENDED
    assert stale["label"] == "big-jump" and "vision_stale" not in stale
