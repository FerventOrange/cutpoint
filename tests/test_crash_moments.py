"""Crashes are moments (#486, decision B): crashes.json crashes become candidates in every
style, forced ones at or above ``must_include_crash`` and weaker ones ranked like any moment."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import StubTranscriber, media_info, write
from cutpoint import analyze, cache, crash, mix, scorer, ui
from cutpoint.events import scoring
from cutpoint.events.menu_spans import drop_menu_time
from cutpoint.styles import ACTION_CAM, STYLES, get_style, load_learned
from cutpoint.telemetry import DT

N = 2400  # 240 s
SRC = Path("a.mp4")


def result(sig: dict[str, np.ndarray] | None = None) -> analyze.SourceResult:
    sig = sig or {k: np.zeros(N) for k in ("loudness", "speech", "laughter", "scene")}
    ps = analyze.PartSignals(N, sig, [], [], [], [], [])
    return analyze.SourceResult(SRC, None, 0.0, N * DT, [], [], None, ps)


def record(cid: str, t: float, conf: float, source: Path = SRC) -> dict[str, Any]:
    return {
        "source": str(source), "id": cid, "t0": t, "t1": t + 1.5, "confidence": conf,
        "scores": {"telemetry": 0.8},
        "clip": {"start": t - 5, "end": t + 8, "reaction": {"start": t + 8, "end": t + 13}},
    }  # fmt: skip


def cand(start: float, end: float, score: float = 50.0) -> analyze.Candidate:
    return analyze.Candidate(SRC, start, end, (start + end) / 2, 0.5, {}, score=score)


def test_every_built_in_style_forces_crashes_at_0_6():
    assert {n: s.must_include_crash for n, s in STYLES.items()} == dict.fromkeys(STYLES, 0.6)


def test_a_learned_preset_still_overrides_must_include_crash(tmp_path):
    path = tmp_path / "mine.json"
    path.write_text('{"name": "mine", "base": "action", "must_include_crash": 0.8}', "utf-8")
    assert load_learned(path).must_include_crash == 0.8


@pytest.mark.parametrize("style", ["action", "sport", "gaming"])
def test_a_confident_crash_and_its_reaction_replace_the_moments_they_overlap(style):
    r = result()
    r.candidates = [cand(96.0, 104.0), cand(200.0, 208.0)]
    crashes = {"crashes": [record("c1", 100.0, 0.9)]}
    analyze.add_crash_candidates([r], get_style(style), crashes)
    forced = [c for c in r.candidates if c.must_include]
    assert [c.label_hint for c in forced] == ["crash", "crash reaction"]
    assert all(c.kind == "crash" and c.event["id"] == "c1" for c in forced)
    assert all(c.score == 90.0 for c in forced)  # its confidence
    assert [(c.start, c.end) for c in r.candidates if not c.must_include] == [(200.0, 208.0)]


def test_a_weak_crash_without_a_moment_is_a_candidate_scored_from_the_grids():
    sig = {k: np.zeros(N) for k in ("loudness", "speech", "laughter", "scene")}
    sig["loudness"][int(150 / DT) : int(152 / DT)] = 0.8
    r = result(sig)
    crashes = {"crashes": [record("c1", 50.0, 0.9), record("c2", 150.0, 0.55)]}
    analyze.add_crash_candidates([r], get_style("action"), crashes)
    [weak] = [c for c in r.candidates if c.event and c.event["id"] == "c2"]
    assert not weak.must_include and weak.kind == "crash" and weak.label_hint == "crash"
    assert (weak.start, weak.end) == (145.0, 158.0)  # its planned clip, no reaction
    assert 0 < weak.score < 90  # the style's score of its span, not the confidence
    assert weak.signals["loudness"] == pytest.approx(0.8)


def test_a_weak_crash_marks_the_moment_it_overlaps_most():
    r = result()
    host, other = cand(140.0, 150.0, 40.0), cand(155.0, 170.0, 30.0)
    r.candidates = [host, other]
    analyze.add_crash_candidates([r], get_style("sport"), {"crashes": [record("c2", 147.0, 0.5)]})
    assert len(r.candidates) == 2  # no new candidate
    assert host.event["id"] == "c2" and other.event is None
    assert (host.start, host.end, host.score, host.must_include) == (140.0, 150.0, 40.0, False)
    rec = analyze._candidate_record(host, 0, get_style("sport"))
    assert (rec["crash_confidence"], rec["crash_forced"]) == (0.5, False)
    assert "must_include" not in rec


def test_crashes_below_the_project_floor_and_of_other_sources_are_left_out():
    r = result()
    crashes = {
        "crashes": [
            record("c1", 50.0, 0.9),
            record("c2", 150.0, 0.5),  # below 0.6 x 0.9
            record("c3", 200.0, 0.95, Path("b.mp4")),
        ]
    }
    analyze.add_crash_candidates([r], get_style("action"), crashes)
    assert {c.event["id"] for c in r.candidates} == {"c1"}


def test_must_include_crash_0_forces_none_but_keeps_the_crash_moment():
    r = result()
    style = replace(get_style("action"), must_include_crash=0.0)
    analyze.add_crash_candidates([r], style, {"crashes": [record("c1", 50.0, 0.9)]})
    assert [(c.label_hint, c.must_include) for c in r.candidates] == [("crash", False)]


def test_a_forced_crash_record_has_the_additive_keys():
    r = result()
    analyze.add_crash_candidates([r], get_style("action"), {"crashes": [record("c1", 50.0, 0.9)]})
    rec = analyze._candidate_record(r.candidates[0], 0, get_style("action"))
    assert rec["must_include"] is True and rec["kind"] == "crash"
    assert rec["crash"]["id"] == "c1"
    assert (rec["crash_confidence"], rec["crash_forced"]) == (0.9, True)


def test_no_crashes_json_adds_nothing():
    r = result()
    r.candidates = [cand(10.0, 20.0)]
    for doc in (None, {}, {"crashes": [{"source": "a.mp4"}]}):
        analyze.add_crash_candidates([r], get_style("action"), doc)
    assert len(r.candidates) == 1 and r.candidates[0].event is None


def test_action_cam_forced_crashes_keep_their_ski_record_plus_the_new_keys():
    style = get_style(ACTION_CAM)
    c = analyze.Candidate(SRC, 95.0, 108.0, 100.0, 0.72, {}, kind="wipeout", must_include=True)
    c.event = {"type": "crash", "id": "c1", "t": 100.0, "end": 101.5, "confidence": 0.72}
    rec = analyze._candidate_record(c, 0, style)
    assert rec["must_include"] is True and rec["crash"] is c.event and rec["kind"] == "wipeout"
    assert (rec["crash_confidence"], rec["crash_forced"]) == (0.72, True)


def test_the_shared_crash_floor():
    assert crash.crash_floor([]) == 0.2
    assert crash.crash_floor([0.1, 0.3]) == 0.2
    assert crash.crash_floor([0.5, 0.9]) == pytest.approx(0.54)


def test_mix_uses_the_shared_floor(monkeypatch):
    seen = []
    real = crash.crash_floor
    monkeypatch.setattr(mix, "crash_floor", lambda xs: seen.append(list(xs)) or real([]))
    mix.crash_candidates([record("c1", 50.0, 0.9)])
    assert seen == [[0.9]]


def test_the_rail_leaves_out_crashes_below_the_shared_floor(tmp_path):
    crashes = [record("c1", 50.0, 0.9), record("c2", 150.0, 0.5), record("c3", 200.0, 0.6)]
    cache.save_json(tmp_path / "crashes.json", {"crashes": crashes})
    out = ui._crash_events(tmp_path, [])
    assert [e["start"] for e in out] == [50.0, 200.0]  # 0.5 < max(0.2, 0.6 x 0.9)


def test_a_crash_without_a_usable_reaction_is_skipped():
    bad = record("c1", 50.0, 0.9)
    bad["clip"]["reaction"] = {"text": "no times"}
    no_reaction = record("c2", 150.0, 0.9)
    del no_reaction["clip"]["reaction"]
    assert [c["id"] for c in analyze._crash_entries({"crashes": [bad, no_reaction]})] == ["c2"]


def _analyze_with_crash(tmp_path, monkeypatch, **kw: Any) -> tuple[Path, dict[str, Any]]:
    write(tmp_path / "a.mp4")
    monkeypatch.setattr(
        analyze,
        "probe",
        lambda path: media_info(
            path, 60.0, width=640, height=360, audio_streams=0,
            acodec=None, sample_rate=None, channels=None,
        ),
    )  # fmt: skip
    monkeypatch.setattr(analyze, "Transcriber", StubTranscriber)
    proj = tmp_path / "proj"
    src = tmp_path / "a.mp4"
    proj.mkdir()
    cache.save_json(proj / "crashes.json", {"crashes": [record("c1", 20.0, 0.9, src)]})
    analyze.analyze(src, get_style("action"), proj, vision=False, use_scenes=False, **kw)
    return proj, json.loads((proj / "highlights.json").read_text(encoding="utf-8"))


def test_analyze_puts_a_forced_crash_first_and_records_the_threshold(home, tmp_path, monkeypatch):
    _, hl = _analyze_with_crash(tmp_path, monkeypatch)
    first = hl["candidates"][0]
    assert first["id"] == "h001" and first["must_include"] is True
    assert (first["crash_forced"], first["crash_confidence"], first["kind"]) == (True, 0.9, "crash")
    assert first["crash"]["id"] == "c1"
    assert hl["settings"]["must_include_crash"] == 0.6
    assert "crash_moments" not in hl["settings"]


def test_a_360_analysis_makes_no_crash_moments_and_reanalyze_keeps_that(
    home, tmp_path, monkeypatch
):
    proj, hl = _analyze_with_crash(tmp_path, monkeypatch, crash_moments=False)
    assert not any("crash" in c for c in hl["candidates"])
    assert hl["settings"]["crash_moments"] is False
    analyze.reanalyze(proj)
    again = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
    assert not any("crash" in c for c in again["candidates"])
    # a 360 project analysed before crash_moments was recorded: pano.json says it is one
    del again["settings"]["crash_moments"]
    cache.save_json(proj / "highlights.json", again)
    cache.save_json(proj / "pano.json", {"clips": []})
    analyze.reanalyze(proj)
    older = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
    assert not any("crash" in c for c in older["candidates"])
    assert older["settings"]["crash_moments"] is False


def test_a_dialogue_stretch_does_not_widen_a_forced_crash_moment():
    forced = {"source": "a.mp4", "start": 10.0, "end": 20.0, "score": 90.0, "must_include": True}
    other = {"source": "a.mp4", "start": 22.0, "end": 26.0, "score": 40.0}
    cands = [forced, other]
    stretch = {"kind": "text", "type": scoring.DIALOGUE_TYPE, "start": 15.0, "end": 30.0}
    scoring.widen_for_dialogue(cands, {"a.mp4": [stretch]})
    assert cands[0] is forced and (forced["start"], forced["end"]) == (10.0, 20.0)
    assert "widened_from" not in forced


def test_a_dialogue_stretch_does_not_widen_a_moment_across_a_forced_crash():
    forced = {"source": "a.mp4", "start": 10.0, "end": 20.0, "score": 90.0, "must_include": True}
    other = {"source": "a.mp4", "start": 22.0, "end": 26.0, "score": 40.0}
    clear = {"source": "a.mp4", "start": 100.0, "end": 104.0, "score": 40.0}
    cands = [forced, other, clear]
    stretches = [
        {"kind": "text", "type": scoring.DIALOGUE_TYPE, "start": 15.0, "end": 30.0},
        {"kind": "text", "type": scoring.DIALOGUE_TYPE, "start": 98.0, "end": 110.0},
    ]
    handled: set[tuple[str, float, float]] = set()
    scoring.widen_for_dialogue(cands, {"a.mp4": stretches}, handled=handled)
    assert (other["start"], other["end"]) == (22.0, 26.0) and "widened_from" not in other
    assert handled == {("a.mp4", 98.0, 110.0)}  # the crash's stretch is left to nominate
    assert (clear["start"], clear["end"]) == (98.0, 110.0)


def test_rescore_events_keeps_a_forced_crash_first(tmp_path):
    def c(cid: str, score: float, forced: bool = False) -> dict[str, Any]:
        return {
            "id": cid, "source": "a.mp4", "start": 0.0, "end": 5.0, "score": score,
            "signals": {}, "must_include": forced,
        }  # fmt: skip

    hl = {"style": "action", "settings": {}, "candidates": [c("h001", 80), c("h002", 20, True)]}
    scorer.rescore_events(hl, {"events": [], "windows": []}, tmp_path)
    assert [x["id"] for x in hl["candidates"]] == ["h002", "h001"]


def test_menu_time_does_not_drop_a_forced_crash_moment():
    forced = {"source": "/v/a.mp4", "start": 10.0, "end": 20.0, "score": 90.0, "must_include": True}
    plain = {**forced, "start": 30.0, "end": 40.0, "must_include": False}
    hl = {"style": "gaming", "settings": {}, "candidates": [forced, plain]}
    ev = {"sources": ["/v/a.mp4"], "stats": {"ocr": {"a.mp4": {"menu_spans": [[0, 50]]}}}}
    drop_menu_time(hl, ev)
    assert hl["candidates"] == [forced]


def test_the_ui_offers_crashes_become_moments_for_every_style(tmp_path):
    names = ("gaming", "action", "commentary", ACTION_CAM)
    assert [ui._style_crash(s) for s in names] == [0.6] * 4
    assert ui._style_crash("auto") == 0.6  # auto picks a built-in style
    assert ui._style_crash("nope") is None
    hl = {"style": "gaming", "settings": {"must_include_crash": 0.4}}
    assert ui.analyze_options(tmp_path, {"style": "gaming"}, hl)["must_include_crash"] == 0.4
    hl["settings"]["must_include_crash"] = 0.6  # the style's own: no flag
    assert ui.analyze_options(tmp_path, {"style": "gaming"}, hl)["must_include_crash"] is None


def test_the_analyze_field_is_renamed_and_the_crashes_kind_has_a_chip():
    html = (ui.WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert ">Crashes become moments at</label>" in html and "Must-include crash" not in html
    assert 'data-kind="crash" aria-pressed="false">Crashes</button>' in html
