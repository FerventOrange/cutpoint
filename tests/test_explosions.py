"""Confirmed explosions (#801): a heard boom the loudness backs (screen and HUD recorded).

All data is invented: a flat loudness track with jumps, scene cuts and OCR rows.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from cutpoint.assistant import watch
from cutpoint.events import common, eventtypes, explosions, pipeline, scoring, sheets, vlm_windows
from cutpoint.events.audio_events import DEFAULT_GROUPS
from cutpoint.events.common import Source
from cutpoint.events.report import _event_line, normalize
from cutpoint.sources import Part

QUIET = -30.0


def _loudness(jumps: dict[float, float], length: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """10 Hz momentary loudness at QUIET LUFS, ``+lu`` for 0.3 s from each jump time."""
    t = np.arange(0.1, length, 0.1)
    m = np.full_like(t, QUIET)
    for at, lu in jumps.items():
        m[(t >= at) & (t < at + 0.3)] += lu
    return t, m


def _boom(t: float, conf: float = 0.3) -> dict[str, Any]:
    return {
        "kind": "audio",
        "t": t - 0.2,
        "end": t + 0.6,
        "peak_t": t,
        "label": "Explosion",
        "type": "audio",
        "confidence": conf,
        "top_class": "Explosion",
        "mean": conf / 2,
        "baseline": 0.0,
        "relative": False,
        "seed": False,
    }


def _text(t: float) -> dict[str, Any]:
    return {"kind": "ocr", "type": "notice", "text": "SOMETHING HAPPENED", "t": t, "end": t + 2}


def test_explosion_group_is_not_a_seed_and_has_no_gunshot():
    group = next(g for g in DEFAULT_GROUPS if g.name == "Explosion")
    assert not group.seed
    assert "Gunshot, gunfire" not in group.labels


@pytest.mark.parametrize(
    ("jump", "cuts", "texts", "backing", "confirmed"),
    [
        (12.0, [20.5], [], ["loud", "screen"], True),  # boom + the picture changes
        (12.0, [], [22.0], ["loud", "hud"], True),  # boom + text right after it
        (12.0, [20.5], [21.0], ["loud", "screen", "hud"], True),
        (12.0, [], [], ["loud"], True),  # loudness alone confirms
        (12.0, [25.0], [30.0], ["loud"], True),  # screen and text too late: only recorded
        (3.0, [20.5], [21.0], ["screen", "hud"], False),  # not louder than before
    ],
)
def test_confirm(jump, cuts, texts, backing, confirmed):
    events = [_boom(20.0), *(_text(t) for t in texts)]
    n = explosions.confirm(events, None, loudness=_loudness({20.0: jump}), cuts=cuts)
    assert events[0]["backing"] == backing
    assert events[0]["confirmed"] is confirmed
    assert n == int(confirmed)
    assert events[0]["loud_jump"] == pytest.approx(jump, abs=0.1)


def test_a_jump_out_of_silence_is_not_loud():
    # sound coming back after a near-silent stretch (a loading screen, a menu): a big jump, but
    # not a blast within the action
    t, m = _loudness({20.0: 12.0})
    m[(t >= 15.0) & (t < 20.0)] = -65.0
    events = [_boom(20.0), _text(21.0)]
    assert explosions.confirm(events, None, loudness=(t, m), cuts=[20.3]) == 0
    assert events[0]["backing"] == ["screen", "hud"]
    assert events[0]["loud_jump"] > 30


def test_confirm_without_loudness_confirms_nothing():
    events = [_boom(20.0), _text(21.0)]
    assert explosions.confirm(events, None, cuts=[20.2]) == 0
    assert events[0]["confirmed"] is False
    assert events[0]["backing"] == ["screen", "hud"]


def test_confirm_reads_the_source_cache(tmp_path):
    t, m = _loudness({20.0: 10.0})
    (tmp_path / "loudness.json").write_text(json.dumps({"t": t.tolist(), "m": m.tolist()}))
    (tmp_path / "scenes.json").write_text("[20.4]")
    events = [_boom(20.0), _boom(40.0)]
    assert explosions.confirm(events, [(0.0, tmp_path)]) == 1
    assert [e["confirmed"] for e in events] == [True, False]


def test_other_audio_events_are_left_alone():
    laugh = {**_boom(20.0), "label": "Laughter"}
    assert explosions.confirm([laugh], None, loudness=_loudness({20.0: 12.0}), cuts=[20.0]) == 0
    assert "confirmed" not in laugh


def _confirmed(t: float, source: str = "a.mp4") -> dict[str, Any]:
    e = _boom(t)
    explosions.confirm([e], None, loudness=_loudness({t: 12.0}), cuts=[t + 0.3])
    return normalize(source, e)


def test_normalize_keeps_the_backing():
    e = _confirmed(20.0)
    assert e["details"]["confirmed"] is True
    assert e["details"]["backing"] == ["loud", "screen"]


def test_audio_matcher():
    m = eventtypes.Matcher("Explosion", flag="confirmed", kind="audio")
    assert m.matches(_confirmed(20.0))
    unconfirmed = _boom(20.0)
    explosions.confirm([unconfirmed], None, loudness=_loudness({}), cuts=[])
    assert not m.matches(normalize("a.mp4", unconfirmed))
    # an OCR matcher never matches a sound, whatever its label
    assert not eventtypes.Matcher("Explosion").matches(_confirmed(20.0))


def test_explosions_score_in_gaming_not_in_funny():
    cand = {"source": "a.mp4", "start": 15.0, "end": 30.0}
    evs = [_confirmed(18.0), _confirmed(22.0), _confirmed(26.0)]
    boost, ev = scoring.event_boost(cand, evs, style="gaming")
    assert ev["explosions"] == 3
    assert boost == 8.0  # 4 points each, at most 2 counted
    boost, ev = scoring.event_boost(cand, evs, style="funny")
    assert (boost, ev["explosions"]) == (0.0, 0)


def test_unconfirmed_boom_scores_nothing():
    cand = {"source": "a.mp4", "start": 15.0, "end": 30.0}
    e = _boom(20.0)
    explosions.confirm([e], None, loudness=_loudness({20.0: 3.0}), cuts=[20.3])  # no jump
    boost, ev = scoring.event_boost(cand, [normalize("a.mp4", e)], style="gaming")
    assert (boost, ev["explosions"]) == (0.0, 0)


def test_confirmed_explosion_is_a_seed():
    evs = [_confirmed(20.0)]
    seeds = scoring.seeds(evs, eventtypes.DEFAULT_EVENT_TYPES, ("gaming",))
    assert [(s.name, s.start) for s in seeds] == [("explosions", evs[0]["start"])]
    assert scoring.seeds(evs, eventtypes.DEFAULT_EVENT_TYPES, ("funny",)) == []


def test_preset_audio_matcher_and_bad_kind():
    t = eventtypes.from_dict(
        {"name": "blasts", "when": [{"type": "Explosion", "kind": "audio"}], "score": 2}
    )
    assert t.when[0].kind == "audio"
    with pytest.raises(ValueError, match="kind"):
        eventtypes.from_dict({"name": "x", "when": [{"type": "kill", "kind": "video"}]})


def test_two_parts_are_read_with_their_offsets(tmp_path):
    first, second = tmp_path / "p0", tmp_path / "p1"
    for d in (first, second):
        d.mkdir()
    t, m = _loudness({20.0: 10.0}, length=40.0)
    (first / "loudness.json").write_text(json.dumps({"t": t.tolist(), "m": m.tolist()}))
    # the second part starts at 100 s of the source and has its own boom at 20 s of its own time
    (second / "loudness.json").write_text(json.dumps({"t": t.tolist(), "m": m.tolist()}))
    (second / "scenes.json").write_text("[20.4]")
    events = [_boom(20.0), _boom(120.0), _boom(60.0)]
    parts = [(0.0, first), (100.0, second)]
    assert explosions.confirm(events, parts) == 2
    assert [e["confirmed"] for e in events] == [True, True, False]
    assert events[1]["backing"] == ["loud", "screen"]  # the part's cut at 100 + 20.4
    assert events[0]["backing"] == ["loud"]


def test_loud_jump_is_the_confirm_calculation():
    t, m = _loudness({20.0: 12.0})
    p = explosions.ExplosionParams()
    assert explosions.loud_jump(t, m, 20.0, p) == pytest.approx((12.0, QUIET), abs=0.1)
    assert explosions.loud_jump(t, m, 0.2, p) is None


@pytest.mark.parametrize(
    ("style", "confirmed", "seed"),
    [
        (("gaming",), True, 1.0),
        (("funny",), True, None),  # the type is limited to other styles
        (("gaming",), False, None),  # an unconfirmed boom
    ],
)
def test_a_boom_seeds_a_vlm_window_only_where_the_type_applies(style, confirmed, seed):

    ev = _boom(20.0)
    ev["confirmed"] = confirmed
    ev = normalize("a.mp4", ev)
    types = eventtypes.DEFAULT_EVENT_TYPES
    assert vlm_windows._seed_priority(ev, types, style) == seed


def test_report_line_and_heard_tell_confirmed_from_not():

    yes = _confirmed(20.0)
    b = _boom(40.0)
    explosions.confirm([b], None, loudness=_loudness({}), cuts=[])
    no = normalize("a.mp4", b)
    assert _event_line("a", yes).endswith(" confirmed")
    assert _event_line("a", no).endswith(" unconfirmed")
    laugh = normalize("a.mp4", {**_boom(30.0), "label": "Laughter"})
    assert _event_line("a", laugh).endswith("s)")
    got = watch.heard([yes, no, laugh], "a.mp4", 0.0, 60.0)
    assert len(got) == 2
    assert any("Explosion" in g for g in got)
    assert any("Laughter" in g for g in got)
    assert sum("Explosion" in g for g in got) == 1


def test_run_events_confirms_booms_and_writes_the_settings(tmp_path, monkeypatch):

    cdir = tmp_path / "cache"
    cdir.mkdir()
    t, m = _loudness({20.0: 12.0})
    (cdir / "loudness.json").write_text(json.dumps({"t": t.tolist(), "m": m.tolist()}))
    src = Source(path=tmp_path / "a.mp4", duration=60.0, cdir=cdir, audio_parts=[])
    project = tmp_path / "proj"
    project.mkdir()
    cached = [
        {k: v for k, v in _boom(20.0).items() if k != "kind"},
        {k: v for k, v in _boom(40.0).items() if k != "kind"},
    ]
    monkeypatch.setattr(
        pipeline,
        "load_highlights",
        lambda p: {"sources": [{"path": str(src.path)}], "style": "gaming", "candidates": []},
    )
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "load_source", lambda path, dur: src)
    monkeypatch.setattr(pipeline, "bad_ranges", lambda *a: {})
    monkeypatch.setattr(pipeline, "load_transcript_lines", lambda project: [])
    monkeypatch.setattr(pipeline, "unload_ollama", list)
    monkeypatch.setattr(pipeline, "cached_audio_events", lambda s, p: cached)
    monkeypatch.setattr(pipeline, "source_parts", lambda path: [Part(0.0, cdir, None)])
    monkeypatch.setattr(common, "gpu_memory_used_mib", lambda: None)
    monkeypatch.setattr(sheets, "write_visual_sheets", lambda *a, **k: None)
    jpath, _ = pipeline.run_events(project, layers=("audio",), discover=False, sheets=False)
    data = json.loads(jpath.read_text())
    booms = [e for e in data["events"] if e["label"] == "Explosion"]
    assert [e["details"]["confirmed"] for e in booms] == [True, False]
    assert data["settings"]["explosions"] == json.loads(
        json.dumps(explosions.ExplosionParams().key())
    )
