"""Funny style: the player's own death plus a reaction (laughter, scream, speech after a
pause) adds points once. Synthetic data only."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from cutpoint import cache, sources
from cutpoint.events import eventtypes, scoring
from cutpoint.events.eventtypes import DEFAULT_EVENT_TYPES
from cutpoint.events.regions import load_preset

SRC = "a.mp4"
CAND = {"start": 0.0, "end": 100.0}


def death(t: float = 10.0) -> dict[str, Any]:
    return {
        "source": SRC,
        "start": t,
        "end": t + 3,
        "kind": "ocr",
        "type": "kill",
        "label": "Bob > Ash",
        "player": True,
        "player_kill": False,
        "player_death": True,
        "teamkill": False,
    }


def audio(label: str, start: float, end: float, conf: float = 0.5) -> dict[str, Any]:
    return {
        "source": SRC,
        "start": start,
        "end": end,
        "kind": "audio",
        "label": label,
        "confidence": conf,
    }


def seg(start: float, end: float) -> dict[str, Any]:
    return {"start": start, "end": end, "text": "hey"}


@pytest.mark.parametrize(
    ("events", "segments", "expected"),
    [
        ([], [seg(2, 4), seg(12, 13)], True),  # silence, then speech 2 s after
        ([], [seg(8, 11), seg(12, 13)], False),  # speech already running at the event
        ([], [seg(8.8, 9.0), seg(12, 13)], False),  # ended within the 1.5 s pause window
        ([], [seg(2, 3), seg(12, 13)], True),  # ended well before: a pause
        ([audio("Laughter", 12, 14)], None, True),
        ([audio("Scream", 11, 12)], None, True),
        ([audio("Laughter", 9, 12)], None, True),  # overlapping the event's start
        ([audio("Cheering", 12, 14)], [], False),  # not a reaction sound
        ([], [seg(17, 18)], False),  # speech after 6 s
        ([audio("Laughter", 17, 18)], None, False),
        ([], [], False),
        ([], None, False),
    ],
)
def test_has_reaction(events, segments, expected):
    assert scoring.has_reaction(10.0, events, segments) is expected


def test_continuous_banter_is_no_reaction():
    segments = [seg(s, s + 1.8) for s in range(0, 30, 2)]
    assert scoring.has_reaction(10.0, [], segments) is False
    boost, ev = scoring.event_boost(CAND, [death()], style="funny", segments=segments)
    assert boost == 0.0 and ev["player_deaths_reaction"] == 0


def test_funny_death_then_speech_after_pause():
    boost, ev = scoring.event_boost(
        CAND, [death()], style="funny", segments=[seg(1, 3), seg(12, 14)]
    )
    assert boost == 8.0
    assert ev["player_deaths_reaction"] == 1


@pytest.mark.parametrize("style", ["highlights", "gaming", None, ("highlights",)])
def test_other_styles_get_nothing(style):
    got = scoring.event_boost(CAND, [death()], style=style, segments=[seg(1, 3), seg(12, 14)])
    assert got == (
        0.0,
        {"laughter": 0.0, "scream": 0.0, "player_kills": 0, "teamkills": 0, "banners": 0},
    )


def test_reaction_counted_once_for_two_deaths():
    events = [death(10), death(40)]
    segments = [seg(12, 13), seg(42, 43)]
    boost, ev = scoring.event_boost(CAND, events, style="funny", segments=segments)
    assert boost == 8.0 and ev["player_deaths_reaction"] == 1


def test_max_boost_cap_holds():
    events = [death(), audio("Laughter", 12, 20)]  # laughter alone is worth 12
    kills = [
        {
            "source": SRC, "start": 50.0, "end": 53.0, "kind": "ocr", "type": "banner",
            "label": "ACE",
        },
        {
            "source": SRC, "start": 60.0, "end": 63.0, "kind": "ocr", "type": "banner",
            "label": "MATCH POINT",
        },
        {
            "source": SRC, "start": 70.0, "end": 73.0, "kind": "ocr", "type": "kill",
            "label": "x", "player": True, "player_kill": True, "teamkill": False,
        },
    ]  # fmt: skip
    boost, _ = scoring.event_boost(CAND, [*events, *kills], style="funny", segments=[])
    assert boost == scoring.MAX_BOOST


def _learned(home: Path, name: str, base: str) -> None:
    folder = home / "styles"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(
        json.dumps({"name": name, "base": base, "weights": {}}), encoding="utf-8"
    )


def test_style_and_base(home):
    _learned(home, "mine", "funny")
    _learned(home, "other", "gaming")
    assert scoring.style_and_base("mine") == ("mine", "funny")
    assert scoring.style_and_base("funny") == ("funny",)
    assert scoring.style_and_base("other") == ("other", "gaming")
    assert scoring.style_and_base("nope") == ("nope",)  # unknown falls back to the name
    assert scoring.style_and_base(None) == ()


def test_learned_funny_style_applies(home):
    _learned(home, "mine", "funny")
    _learned(home, "other", "gaming")
    segs = [seg(12, 13)]
    got = scoring.event_boost(CAND, [death()], style=scoring.style_and_base("mine"), segments=segs)
    assert got[0] == 8.0
    got = scoring.event_boost(CAND, [death()], style=scoring.style_and_base("other"), segments=segs)
    assert got[0] == 0.0
    got = scoring.event_boost(CAND, [death()], style=scoring.style_and_base("nope"), segments=segs)
    assert got[0] == 0.0


@pytest.mark.parametrize("name", ["r6", "wt"])
@pytest.mark.parametrize("style", [None, "funny", "gaming", "highlights"])
def test_own_death_is_first_match_in_every_style(name, style):
    e = death()
    if name == "wt":
        e["verb"] = "kill"
    t = eventtypes.first_match(DEFAULT_EVENT_TYPES, e)
    assert t is not None and t.name == "player_deaths"
    assert (t.reaction, t.styles) == (8.0, ("funny",))
    assert scoring.event_boost(CAND, [e], style=style)[1]["player_kills"] == 0


def _custom(tmp_path: Path, events: list[Any]) -> Path:
    project = tmp_path / "proj"
    (project / "hud_presets").mkdir(parents=True)
    (project / "hud_presets" / "foo.json").write_text(
        json.dumps({"game": "foo", "regions": [], "events": events}), encoding="utf-8"
    )
    return project


BANNER = {"name": "mine", "when": [{"type": "banner", "prefix": ["X"]}], "score": 2}


def test_from_dict_reaction_and_styles(tmp_path, home):
    events = [
        {**BANNER, "name": "big", "reaction": 99, "styles": "funny"},
        {**BANNER, "name": "many", "reaction": 3, "styles": ["funny", "gaming"]},
        {**BANNER, "name": "all", "reaction": 3},
    ]
    types = {t.name: t for t in eventtypes.types_for(load_preset("foo", _custom(tmp_path, events)))}
    assert (types["big"].reaction, types["big"].styles) == (12.0, ("funny",))
    assert types["many"].styles == ("funny", "gaming")
    assert (types["all"].reaction, types["all"].styles) == (3.0, None)


def test_unknown_style_skipped_with_one_warning(tmp_path, home, caplog, monkeypatch):
    monkeypatch.setattr(eventtypes, "_warned_types", set())
    bad = {**BANNER, "name": "bad", "styles": ["funy"]}
    preset = load_preset("foo", _custom(tmp_path, [bad, {**BANNER, "name": "ok"}]))
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            by_name = {t.name: t for t in eventtypes.types_for(preset)}
    assert "bad" not in by_name and "ok" in by_name and "player_kills" in by_name
    assert sum("Ignoring invalid event type" in r.getMessage() for r in caplog.records) == 1


def test_reserved_reaction_name_skipped_with_one_warning(tmp_path, home, caplog, monkeypatch):
    monkeypatch.setattr(eventtypes, "_warned_types", set())
    bad = {**BANNER, "name": "foo_reaction"}
    with pytest.raises(ValueError, match="reserved"):
        eventtypes.from_dict(bad)
    preset = load_preset("foo", _custom(tmp_path, [bad, {**BANNER, "name": "ok"}]))
    with caplog.at_level(logging.WARNING, logger=eventtypes.logger.name):
        for _ in range(2):
            names = {t.name for t in eventtypes.types_for(preset)}
    assert "foo_reaction" not in names and "ok" in names
    assert sum("Ignoring invalid event type" in r.getMessage() for r in caplog.records) == 1


def test_learned_style_name_accepted_in_preset(tmp_path, home):
    _learned(home, "mine", "funny")
    ev = {**BANNER, "name": "ok", "styles": ["mine"]}
    types = eventtypes.types_for(load_preset("foo", _custom(tmp_path, [ev])))
    assert next(t for t in types if t.name == "ok").styles == ("mine",)


def _transcribe(tmp_path: Path, segments: list[dict[str, Any]]) -> Path:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    cache.save_json(cache.source_dir(video) / "transcript-small.json", segments)
    return video


def test_source_segments_and_words_round_trip(tmp_path, home):
    segments = [
        {
            "start": 1.0, "end": 3.0, "text": "hello there", "avg_logprob": -0.3,
            "words": [
                {"word": "hello", "start": 1.0, "end": 1.5},
                {"word": "there", "start": 1.6, "end": 3.0},
            ],
        },
        {
            "start": 5.0, "end": 6.0, "text": "bye",
            "words": [{"word": "bye", "start": 5.0, "end": 6.0}],
        },
    ]  # fmt: skip
    video = _transcribe(tmp_path, segments)
    got = sources.source_segments(video)
    assert got is not None
    assert [(s["start"], s["end"], s["text"]) for s in got] == [
        (1.0, 3.0, "hello there"),
        (5.0, 6.0, "bye"),
    ]
    assert sources.source_words(video) == [
        {"word": "hello", "start": 1.0, "end": 1.5, "avg_logprob": -0.3},
        {"word": "there", "start": 1.6, "end": 3.0, "avg_logprob": -0.3},
        {"word": "bye", "start": 5.0, "end": 6.0},
    ]
    other = tmp_path / "untranscribed.mp4"
    other.write_bytes(b"y")
    assert sources.source_segments(other) is None
    assert sources.source_words(other) is None


def _project(tmp_path: Path, style: str, video: Path, laughter: bool = True) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    src = str(video)
    cand = {"id": "h001", "source": src, "start": 0.0, "end": 30.0, "score": 40.0}
    cache.save_json(project / "highlights.json", {"style": style, "candidates": [cand]})
    ev = [{**death(10), "source": src}]
    if laughter:
        ev.append({**audio("Laughter", 12, 13), "source": src, "confidence": 0.01})
    cache.save_json(project / "events.json", {"settings": {}, "events": ev, "windows": []})
    return project


@pytest.mark.parametrize(("style", "boost"), [("funny", 8.0), ("gaming", 0.0)])
def test_rescore_highlights_death_reaction(tmp_path, home, style, boost):
    video = tmp_path / "gone.mp4"  # no transcript: only the laughter can count
    project = _project(tmp_path, style, video)
    scoring.rescore_highlights(project)
    c = cache.load_json(project / "highlights.json")["candidates"][0]
    ev = c["signals"]["events"]
    laugh = ev["laughter"]
    assert c["score"] == pytest.approx(40.0 + round(min(12.0, 60 * laugh), 3) + boost, abs=0.1)
    assert ("player_deaths_reaction" in ev) == (style == "funny")
    assert ev.get("player_deaths_reaction", 0) == (1 if style == "funny" else 0)


def test_rescore_highlights_death_reaction_from_cached_transcript(tmp_path, home):
    video = _transcribe(tmp_path, [seg(1, 3), seg(12, 13)])  # a pause, then speech
    project = _project(tmp_path, "funny", video, laughter=False)
    scoring.rescore_highlights(project)
    c = cache.load_json(project / "highlights.json")["candidates"][0]
    assert c["signals"]["events"]["player_deaths_reaction"] == 1
    assert c["score"] == pytest.approx(48.0, abs=0.1)


def test_rescore_highlights_gaming_does_not_load_transcript(tmp_path, home, monkeypatch):
    monkeypatch.setattr(sources, "source_segments", lambda _p: pytest.fail("loaded"))
    video = _transcribe(tmp_path, [seg(1, 3), seg(12, 13)])
    project = _project(tmp_path, "gaming", video, laughter=False)
    scoring.rescore_highlights(project)
    c = cache.load_json(project / "highlights.json")["candidates"][0]
    assert "player_deaths_reaction" not in c["signals"]["events"]
    assert c["score"] == pytest.approx(40.0, abs=0.1)
