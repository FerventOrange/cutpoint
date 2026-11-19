"""Clip planning steps: select_moments, Padding / pad, speech_pass, plan_clips, import layout."""

from __future__ import annotations

import ast
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import cutpoint
from conftest import media_info, write
from cutpoint import boundaries, clips, config, extend, folders, jsonout, planning
from cutpoint.encode import OutputFormat
from cutpoint.events.tails import EventTailSettings
from cutpoint.ffmpeg_utils import MediaInfo
from cutpoint.planning import Padding, manifest_parts, pad, select_moments, speech_pass
from cutpoint.styles import get_style
from cutpoint.tail import TailSettings
from export_helpers import install_fake_export
from planning_helpers import SRC_A, SRC_B, _moment


# -- select_moments ------------------------------------------------------------------
def _project(tmp_path: Path, cands: list[dict[str, Any]], **extra: Any) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    data = {"style": "ski", "candidates": cands, **extra}
    (project / "highlights.json").write_text(json.dumps(data), encoding="utf-8")
    return project


def _cand(cid: str, score: float, duration: float = 10.0, **extra: Any) -> dict[str, Any]:
    return {"id": cid, "score": score, "duration": duration, **extra}


def test_select_moments_top_by_score(tmp_path):
    project = _project(tmp_path, [_cand("a", 1), _cand("b", 3), _cand("c", 2)])
    chosen, style = select_moments(project, top=2)
    assert [c["id"] for c in chosen] == ["b", "c"]
    assert style == "ski"


def test_select_moments_suggested_top_then_default_5(tmp_path):
    cands = [_cand(f"c{i}", i) for i in range(8)]
    project = _project(tmp_path, cands, suggested_top=3)
    assert len(select_moments(project)[0]) == 3
    (project / "highlights.json").write_text(json.dumps({"candidates": cands}), encoding="utf-8")
    chosen, style = select_moments(project)
    assert len(chosen) == 5 and style == "action"  # no style recorded: auto's, not gaming


@pytest.mark.parametrize(
    ("recorded", "want"),
    [
        ({}, "action"),  # auto's style for unclear footage (#1002)
        ({"style_base": "sport"}, "sport"),
        ({"detection": {"style": "commentary"}}, "commentary"),
        ({"style_base": "zz-unknown", "detection": {"style": "funny"}}, "funny"),
        ({"style_base": "sport", "detection": {"style": "commentary"}}, "sport"),
        ({"detection": {"style": "zz-unknown"}}, "action"),
    ],
)
def test_select_moments_of_a_file_without_style_uses_what_it_records(tmp_path, recorded, want):
    project = tmp_path / "proj"
    project.mkdir()
    data = {"candidates": [_cand("a", 1)], **recorded}
    (project / "highlights.json").write_text(json.dumps(data), encoding="utf-8")
    assert select_moments(project)[1] == want


def test_select_moments_ids_keep_their_order(tmp_path):
    project = _project(tmp_path, [_cand("a", 1), _cand("b", 3), _cand("c", 2)])
    chosen, _ = select_moments(project, ids=["a", "c"])
    assert [c["id"] for c in chosen] == ["a", "c"]
    with pytest.raises(ValueError, match="Unknown ids: zz"):
        select_moments(project, ids=["a", "zz"])


def _span(cid, score, start, end, source="a.mp4"):
    return _cand(cid, score, end - start, source=source, start=start, end=end)


def test_select_moments_skips_a_candidate_inside_a_picked_one(tmp_path):
    project = _project(
        tmp_path,
        [_span("long", 9, 100, 400), _span("short", 8, 150, 160), _span("far", 7, 500, 510)],
    )
    chosen, _ = select_moments(project, top=2)
    assert [c["id"] for c in chosen] == ["long", "far"]  # the skipped one uses no slot
    chosen, _ = select_moments(project, max_total=1000)
    assert [c["id"] for c in chosen] == ["long", "far"]


def test_select_moments_keeps_overlapping_candidates_of_other_sources_and_explicit_ids(tmp_path):
    project = _project(
        tmp_path, [_span("long", 9, 100, 400), _span("short", 8, 150, 160, source="b.mp4")]
    )
    assert [c["id"] for c in select_moments(project, top=2)[0]] == ["long", "short"]
    (tmp_path / "x").mkdir()
    project = _project(tmp_path / "x", [_span("long", 9, 100, 400), _span("short", 8, 150, 160)])
    assert [c["id"] for c in select_moments(project, ids=["long", "short"])[0]] == ["long", "short"]


def test_select_moments_never_skips_an_overlapping_must_include_crash(tmp_path):
    cands = [
        dict(_span("c1", 9, 100, 140), must_include=True),
        dict(_span("c2", 8, 110, 130), must_include=True),
    ]
    project = _project(tmp_path, cands)
    assert [c["id"] for c in select_moments(project, max_total=180)[0]] == ["c1", "c2"]
    assert [c["id"] for c in select_moments(project, top=3)[0]] == ["c1", "c2"]


def test_select_moments_keeps_candidates_that_overlap_only_at_the_edge(tmp_path):
    project = _project(tmp_path, [_span("a", 9, 100, 110.3), _span("b", 8, 109.8, 120)])
    assert [c["id"] for c in select_moments(project, top=2)[0]] == ["a", "b"]


def test_select_moments_max_total_takes_the_short_one_when_the_long_does_not_fit(tmp_path):
    project = _project(tmp_path, [_span("long", 9, 100, 400), _span("short", 8, 150, 160)])
    chosen, _ = select_moments(project, max_total=60)
    assert [c["id"] for c in chosen] == ["short"]


def test_select_moments_must_include_added_and_kept_past_max_total(tmp_path):
    cands = [
        _cand("a", 5, 30),
        _cand("b", 4, 30),
        _cand("crash", 0.1, 20, must_include=True),
    ]
    project = _project(tmp_path, cands)
    chosen, _ = select_moments(project, top=1)
    assert [c["id"] for c in chosen] == ["a", "crash"]
    # max_total: the crash goes first and stays; the rest fill what is left
    chosen, _ = select_moments(project, max_total=50)
    assert [c["id"] for c in chosen] == ["a", "crash"]
    chosen, _ = select_moments(project, max_total=10)
    assert [c["id"] for c in chosen] == ["crash"]


def test_select_moments_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"highlights\.json"):
        select_moments(tmp_path)
    project = _project(tmp_path, [_cand("a", 1, 30)])
    with pytest.raises(ValueError, match="No moments selected"):
        select_moments(project, max_total=5)


# -- Padding / pad -------------------------------------------------------------------
def test_padding_from_args_and_override():
    assert Padding.from_args(2.0, None, 5.0) == Padding(2.0, 5.0)
    assert Padding.from_args(None, None, None) == Padding(0.0, 0.0)
    assert not Padding().active and Padding(0, 1).active
    base = Padding(2.0, 2.0)
    assert base.override({}) is base
    assert base.override({"extend": 4}) == Padding(4.0, 4.0)
    assert base.override({"extend": 4, "extend_end": 1}) == Padding(4.0, 1.0)
    assert base.override({"extend_start": 0}) == Padding(0.0, 2.0)
    with pytest.raises(ValueError, match="negative"):
        Padding(-1, 0)


def test_pad_applies_and_records(readable):
    start, end, rec = pad(Path("v.mp4"), media_info(), 10.0, 20.0, Padding(2.0, 3.0))
    assert (start, end) == (8.0, 23.0)
    assert rec == {
        "requested": {"start": 2.0, "end": 3.0},
        "applied": {"start": 2.0, "end": 3.0},
    }


def test_pad_clamps_to_source_and_readable_part(readable):
    start, end, rec = pad(Path("v.mp4"), media_info(duration=25.0), 1.0, 24.0, Padding(4.0, 4.0))
    assert (start, end) == (0.0, 25.0)
    assert rec["clamped"] == ["start", "end"]
    readable[0] = (5.0, 21.0)  # recovered part starting at 5 s, read error at 21 s
    start, end, rec = pad(Path("v.mp4"), media_info(), 6.0, 20.0, Padding(4.0, 4.0))
    assert (start, end) == (5.0, 21.0)
    assert rec["applied"] == {"start": 1.0, "end": 1.0}
    # never shorter than the clip, even past the readable end
    start, end, _ = pad(Path("v.mp4"), media_info(), 6.0, 22.0, Padding(0.0, 4.0))
    assert end == 22.0


def test_pad_past_last_readable_point_only_extends_forward(readable):
    readable[0] = (0.0, 30.0)
    start, end, _ = pad(Path("v.mp4"), media_info(), 40.0, 50.0, Padding(4.0, 4.0))
    assert (start, end) == (40.0, 54.0)


# -- speech_pass ---------------------------------------------------------------------
@dataclass
class _Plan:
    name: str
    entry: dict[str, Any]
    src: Path
    info: MediaInfo
    start: float
    end: float
    keep: tuple[float, float]


def _plan(name: str, start: float, end: float, **entry: Any) -> _Plan:
    e = {"source": "D:\\v.mp4", **entry}
    return _Plan(name, e, Path("v.mp4"), media_info(), start, end, (start, end))


def test_speech_pass_moves_end_out_of_an_utterance(speech_env):
    project, calls = speech_env
    plans = [_plan("a.mp4", 5.0, 12.0)]
    recs = speech_pass(plans, boundaries.SpeechSettings(), project, fit_end=lambda e: True)
    assert plans[0].start == 5.0
    assert plans[0].end == pytest.approx(14.4)  # utterance end + breath
    assert recs["a.mp4"]["end"]["moved"] > 0
    assert len(calls["maps"]) == 1 and calls["words"] == [Path("v.mp4")]


def test_speech_pass_respects_opt_outs_and_fixed_ends(speech_env):
    project, _ = speech_env
    plans = [
        _plan("off.mp4", 5.0, 12.0, speech_aware=False),
        _plan("fixed.mp4", 5.0, 12.5, tail=1.0),
    ]
    recs = speech_pass(
        plans, boundaries.SpeechSettings(), project, fit_end=lambda e: "tail" not in e
    )
    assert (plans[0].start, plans[0].end) == (5.0, 12.0)
    assert "off.mp4" not in recs
    assert plans[1].end == 12.5  # fit_end False: the out point stays


def test_speech_pass_per_clip_cap_and_hi_clamp(speech_env):
    project, _ = speech_env
    plans = [_plan("a.mp4", 5.0, 12.0, speech_max_extend=1.0)]
    speech_pass(plans, boundaries.SpeechSettings(), project, fit_end=lambda e: True)
    assert plans[0].end <= 13.0 + 1e-6  # utterance would need 2.4 s > 1 s cap

    plans = [_plan("b.mp4", 5.0, 12.0)]
    seen = []

    def clamp(plan, hi):
        seen.append((plan.name, hi))
        return 13.0

    speech_pass(plans, boundaries.SpeechSettings(), project, fit_end=lambda e: True, hi_clamp=clamp)
    assert seen == [("b.mp4", 100.0)]
    assert plans[0].end <= 13.0


def test_speech_pass_disabled_and_shared_words_cache(speech_env):
    project, calls = speech_env
    plans = [_plan("a.mp4", 5.0, 12.0)]
    off = boundaries.SpeechSettings(enabled=False)
    assert speech_pass(plans, off, project, fit_end=lambda e: True) == {}
    assert (plans[0].start, plans[0].end) == (5.0, 12.0) and not calls["maps"]

    cache = {Path("v.mp4"): [{"start": 1.0, "end": 2.0, "word": "hi"}]}
    plans = [_plan("a.mp4", 5.0, 12.0), _plan("b.mp4", 20.0, 30.0)]
    speech_pass(
        plans, boundaries.SpeechSettings(), project, fit_end=lambda e: True, words_cache=cache
    )
    assert calls["words"] == []  # cached words used, one map per source
    assert len(calls["maps"]) == 1 and calls["maps"][0][1] == cache[Path("v.mp4")]


def test_speech_pass_rejects_duplicate_names(speech_env):
    project, _ = speech_env
    plans = [_plan("a.mp4", 5.0, 12.0), _plan("a.mp4", 20.0, 30.0)]
    with pytest.raises(ValueError, match=r"Duplicate clip names: a\.mp4"):
        speech_pass(plans, boundaries.SpeechSettings(), project, fit_end=lambda e: True)


# -- extend's speech-pass policy -------------------------------------------------------
def test_extend_hi_clamp_stays_before_read_error_but_never_cuts():
    early = _plan("a.mp4", 5.0, 18.0, read_error={"end": 20.0})
    assert extend._hi_clamp(early, 30.0) == 20.0
    late = _plan("b.mp4", 5.0, 22.0, read_error={"end": 20.0})
    assert extend._hi_clamp(late, 30.0) == 22.0
    assert extend._hi_clamp(_plan("c.mp4", 5.0, 18.0), 30.0) == 30.0


def test_speech_pass_with_extend_clamp_respects_read_error(speech_env):
    project, _ = speech_env
    plans = [_plan("a.mp4", 5.0, 12.0, read_error={"end": 13.0})]  # utterance runs to 14 s
    speech_pass(
        plans,
        boundaries.SpeechSettings(),
        project,
        fit_end=lambda e: not extend._fixed_tail(e),
        hi_clamp=extend._hi_clamp,
    )
    assert 12.0 <= plans[0].end <= 13.0


def test_plan_clips_names_tails_padding_and_crop(plan_env):
    run, calls = plan_env
    ordered = [
        _moment("c2", SRC_A, 30.0, 40.0, label="Big Jump!"),
        _moment("c1", SRC_A, 41.0, 45.0, label="landing", tail=3.0),
        _moment("c3", SRC_B, 20.0, 60.0, label="end", crop="none"),
    ]
    plans = run(ordered, padding=Padding(1.0, 1.0), crop="640:480:0:0")
    assert [p.name for p in plans] == ["001_big-jump.mp4", "002_landing.mp4", "003_end.mp4"]
    assert [p.entry["id"] for p in plans] == ["c2", "c1", "c3"]
    assert [p.src for p in plans] == [Path("a.mp4"), Path("a.mp4"), Path("b.mp4")]
    first, fixed, last = plans
    assert first.tail.reason == "smart tail off" and first.keep == (30.0, 40.0)
    assert (first.start, first.end) == (29.0, 41.0)  # padded after the tail
    assert fixed.tail.reason == "fixed tail 3s" and fixed.keep == (41.0, 48.0)
    assert last.keep == (20.0, 50.0) and last.end == 50.0  # clamped to the source
    assert str(first.crop) == "640:480:0:0" and first.crop_source == "cli"
    assert last.crop is None and last.crop_source == "segment"
    assert all(p.speech is None and not p.pano for p in plans)
    assert calls["words"] == [] and calls["maps"] == []  # tails off, speech off: no reads


def test_plan_clips_tail_stops_at_next_moment_of_same_source(plan_env):
    run, _ = plan_env
    ordered = [_moment("a", SRC_A, 10.0, 20.0, tail=8.0), _moment("b", SRC_A, 24.0, 30.0)]
    plans = run(ordered)
    assert plans[0].end == 24.0 and plans[0].tail.extension == 4.0
    assert plans[1].end == 30.0


def test_plan_clips_speech_pass_and_opt_out(plan_env):
    run, calls = plan_env
    ordered = [
        _moment("a", SRC_A, 5.0, 12.0),
        _moment("b", SRC_B, 5.0, 12.0, speech_aware=False),
    ]
    plans = run(ordered, speech=boundaries.SpeechSettings())
    assert plans[0].keep == (5.0, 12.0)
    assert plans[0].end == pytest.approx(14.4)  # out of the utterance
    assert plans[0].speech["end"]["moved"] > 0
    assert (plans[1].end, plans[1].speech) == (12.0, None)
    assert calls["words"] == [Path("a.mp4")]


def test_an_exact_segment_is_cut_at_its_own_edges(plan_env, tmp_path):
    """A word splice ("exact": true) gets no tail, padding or speech-aware move, also with all
    of them on: a 0.3 s word inside an utterance stays 0.3 s."""
    run, calls = plan_env
    (tmp_path / "a.mp4").write_bytes(b"x")
    seg_file = tmp_path / "segments.json"
    seg_file.write_text(
        json.dumps(
            [
                {"source": "a.mp4", "start": 10.5, "end": 10.8, "exact": True, "tail": 4.0},
                {"source": "a.mp4", "start": 5.0, "end": 12.0, "extend": 1.0},
            ]
        ),
        encoding="utf-8",
    )
    word, plain = clips.load_segments(seg_file)
    word["source"] = SRC_A  # the planning inputs of plan_env
    assert word["exact"] is True and "tail" not in word and "extend" not in word
    assert {k: word[k] for k in clips.EXACT_KEYS} == clips.EXACT_KEYS
    assert "exact" not in plain
    plans = run(
        [word],
        tail=TailSettings(enabled=True),
        speech=boundaries.SpeechSettings(),
        padding=Padding(1.0, 1.0),
    )
    assert (plans[0].start, plans[0].end) == (10.5, 10.8) and plans[0].speech is None
    assert calls["words"] == []  # no speech pass read for it
    assert planning.exact_fade(word, 0.25) == planning.EXACT_AUDIO_FADE_S
    assert planning.exact_fade(plain, 0.25) == 0.25
    assert planning.exact_fade(word, 0.0) == 0.0  # never more than the export's own fade


def test_load_segments_rejects_an_exact_that_is_not_true_or_false(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    seg_file = tmp_path / "segments.json"
    seg_file.write_text(json.dumps([{"source": "a.mp4", "start": 1, "end": 2, "exact": "yes"}]))
    with pytest.raises(ValueError, match="Segment 1: exact must be true or false"):
        clips.load_segments(seg_file)


def test_plan_clips_honour_min_start_and_max_end(plan_env):
    run, _ = plan_env
    seg = _moment("a", SRC_A, 10.0, 20.0, tail=8.0, min_start=9.0, max_end=23.0)
    plans = run([seg], padding=Padding(2.0, 2.0))
    assert (plans[0].start, plans[0].end) == (9.0, 23.0)  # tail and padding capped
    plans = run([_moment("b", SRC_A, 5.0, 12.0, max_end=13.0)], speech=boundaries.SpeechSettings())
    assert 12.0 <= plans[0].end <= 13.0  # not out of the utterance to 14.4


def test_plan_clips_errors(plan_env):
    run, _ = plan_env
    with pytest.raises(ValueError, match=r"x: 60\.00-70\.00 is outside b\.mp4"):
        run([_moment("x", SRC_B, 60.0, 70.0)])
    with pytest.raises(ValueError, match=r"does not fit a\.mp4"):
        run([_moment("y", SRC_A, 1.0, 5.0)], crop="1920:1080:8:0")


def test_render_and_manifest_after_read_error(plan_env, monkeypatch, tmp_path, caplog):
    run, _ = plan_env
    plans = run(
        [_moment("a", SRC_A, 10.0, 20.0, tail=3.0), _moment("b", SRC_B, 5.0, 9.0, tail=1.0)]
    )
    assert [p.end for p in plans] == [23.0, 10.0]
    err = {"failure_time": 21.5, "messages": [], "retried": True, "end": 21.0}
    calls = []

    def fake_checked(src, info, start, end, out, *a):
        calls.append(out)
        return err if out.name == plans[0].name else None

    monkeypatch.setattr(clips, "export_clip_checked", fake_checked)
    fmt = OutputFormat(1920, 1080, 30)
    results = clips.render_clips(plans, tmp_path, fmt=fmt, vertical=False, audio_fade=0.25)
    assert [r.out for r in results] == calls == [tmp_path / p.name for p in plans]
    assert (results[0].end, results[0].read_error) == (21.0, err)
    assert (results[1].end, results[1].read_error) == (10.0, None)

    caplog.set_level(logging.INFO, logger="cutpoint.clips")
    clips.write_manifest(
        tmp_path,
        tmp_path,
        plans,
        results,
        order="segments",
        fmt=fmt,
        vertical=False,
        style=get_style("gaming"),
        tail=TailSettings(enabled=False),
        event_settings=EventTailSettings(enabled=False),
        padding=None,
        speech=boundaries.SpeechSettings(enabled=False),
        audio_fade=0.25,
        crop=None,
    )
    entries = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["clips"]
    first = entries[0]
    assert (first["start"], first["end"], first["duration"]) == (10.0, 21.0, 11.0)
    assert first["read_error"] == err
    assert first["pick"] == {"start": 10.0, "end": 23.0}
    assert "read_error" not in entries[1] and entries[1]["end"] == 10.0
    # 3 s + 1 s of tail, minus the 2 s the read error cut off
    assert "Smart tail added 2.0s over 2 of 2 clips" in caplog.messages


# -- import layout (B5) --------------------------------------------------------------
PKG = Path(cutpoint.__file__).resolve().parent


def _module_name(path: Path) -> str:
    rel = path.relative_to(PKG.parent).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


def _import_graph() -> dict[str, set[str]]:
    """``{module: cutpoint modules it imports}``, counting lazy (function-level) imports."""
    graph: dict[str, set[str]] = {}
    for path in PKG.rglob("*.py"):
        name = _module_name(path)
        deps: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                if node.module.startswith("cutpoint"):
                    deps.add(node.module)
                    deps.update(f"{node.module}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                deps.update(a.name for a in node.names if a.name.startswith("cutpoint"))
        graph[name] = deps
    return {m: {d for d in deps if d in graph and d != m} for m, deps in graph.items()}


def test_planning_and_encode_do_not_import_commands():
    graph = _import_graph()
    for low in ("cutpoint.planning", "cutpoint.encode"):
        assert not graph[low] & {"cutpoint.clips", "cutpoint.extend"}, low
    assert "cutpoint.clips" not in graph["cutpoint.extend"]


@pytest.mark.slow
def test_clip_modules_import_on_their_own():
    """Each module imports first, with no cutpoint module loaded (an import cycle fails)."""
    script = (
        "import importlib, sys\n"
        "for m in ('cutpoint.extend', 'cutpoint.clips', 'cutpoint.planning', 'cutpoint.encode'):\n"
        "    for k in [k for k in sys.modules if k == 'cutpoint' or k.startswith('cutpoint.')]:\n"
        "        del sys.modules[k]\n"
        "    importlib.import_module(m)\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=120)


def test_load_segments_keeps_kind_category_and_drops_an_empty_one(tmp_path: Path) -> None:
    video = tmp_path / "a.mp4"
    video.write_bytes(b"")
    seg = tmp_path / "segments.json"
    items = [
        {"source": "a.mp4", "start": 1, "end": 4, "kind": "laughter", "category": "social"},
        {"source": "a.mp4", "start": 5, "end": 8, "kind": ""},
    ]
    seg.write_text(json.dumps(items), encoding="utf-8")
    first, second = clips.load_segments(seg)
    assert (first["kind"], first["category"]) == ("laughter", "social")
    assert "kind" not in second


@pytest.mark.parametrize("preset", ["off", "light"])
def test_manifest_records_compress_peaks_only_where_applied(plan_env, tmp_path, preset):
    run, _ = plan_env
    plans = run([_moment("a", SRC_A, 10.0, 20.0), _moment("b", SRC_B, 5.0, 9.0)])
    results = [clips.RenderResult(tmp_path / p.name, p.end, None, None, None) for p in plans]
    peaks = [",acompressor=x", ""] if preset != "off" else ["", ""]
    clips.write_manifest(
        tmp_path, tmp_path, plans, results, order="segments", fmt=OutputFormat(1920, 1080, 30),
        vertical=False, style=get_style("gaming"), tail=TailSettings(enabled=False),
        event_settings=EventTailSettings(enabled=False), padding=None,
        speech=boundaries.SpeechSettings(enabled=False), audio_fade=0.25, crop=None,
        compress_peaks=preset, peaks=peaks,
    )  # fmt: skip
    doc = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    entries = doc["clips"]
    if preset == "off":
        assert "compress_peaks" not in doc
        assert all("compress_peaks" not in e for e in entries)
    else:
        assert doc["compress_peaks"] == "light"
        assert entries[0]["compress_peaks"] == "light" and "compress_peaks" not in entries[1]


@pytest.mark.parametrize(
    ("extra_a", "extra_b", "on", "tails", "speech"),
    [
        ({}, {}, True, True, True),
        ({}, {}, False, False, False),  # the run turned them off
        ({"smart_tail": False, "speech_aware": False}, {"tail": 2.0, "speech_aware": False},
         True, False, False),
        ({"smart_tail": False}, {"speech_aware": False}, True, True, True),
    ],
)  # fmt: skip
def test_manifest_header_says_whether_smart_tail_and_speech_were_applied(
    plan_env, tmp_path, extra_a, extra_b, on, tails, speech
):
    """The header's ``enabled`` is off when every segment turned the step off ("smart_tail":
    false or a fixed "tail"; "speech_aware": false), not only the run's setting; the clips'
    records are unchanged."""
    run, _ = plan_env
    plans = run(
        [_moment("a", SRC_A, 10.0, 20.0, **extra_a), _moment("b", SRC_B, 5.0, 9.0, **extra_b)]
    )
    results = [clips.RenderResult(tmp_path / p.name, p.end, None, None, None) for p in plans]
    clips.write_manifest(
        tmp_path, tmp_path, plans, results, order="segments", fmt=OutputFormat(1920, 1080, 30),
        vertical=False, style=get_style("gaming"), tail=TailSettings(enabled=on),
        event_settings=EventTailSettings(enabled=False), padding=None,
        speech=boundaries.SpeechSettings(enabled=on), audio_fade=0.25, crop=None,
    )  # fmt: skip
    doc = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert doc["smart_tail"]["enabled"] is tails
    assert doc["speech_aware"] == boundaries.SpeechSettings().to_json() | {"enabled": speech}
    a, b = doc["clips"]
    assert ("speech_aware" in a, "speech_aware" in b) == (
        "speech_aware" in extra_a,
        "speech_aware" in extra_b,
    )


@pytest.mark.parametrize(
    ("extra", "on", "clause"),
    [
        ({}, True, "smart tail on (min 1.5s, max 5.0s)"),
        ({}, False, "smart tail off"),  # the run turned it off
        ({"smart_tail": False}, True, "smart tail off"),
        ({"tail": 2.0}, True, "smart tail off"),
    ],
)
def test_export_line_says_smart_tail_off_when_no_clip_gets_one(
    home, monkeypatch, caplog, extra, on, clause
):
    """The ``Exporting ...`` line matches the manifest header: ``smart tail off`` when every
    segment turned it off, the bounds only when a clip gets one."""
    caplog.set_level(logging.INFO)
    project = home / "projects" / "p"
    src = write(project / "src.mp4")
    seg = write(
        project / "seg.json",
        json.dumps([{"source": str(src), "start": 0.0, "end": 2.0, "label": "a", **extra}]),
    )
    monkeypatch.setattr(clips, "probe", media_info)
    install_fake_export(monkeypatch, clips, content=b"n")
    monkeypatch.setattr(clips.sources, "source_words", lambda _src, _lang=None: None)
    monkeypatch.setattr(clips.sources, "audio_window", lambda *a: None)
    clips.export_clips(
        project, segments=seg, smart_tail_on=on, tail_min=1.5, tail_max=5.0, speech_aware=False
    )
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Exporting 1")]
    assert line.endswith(f"(segments order); {clause}")


# -- a project whose user style was deleted ---------------------------------------------
def _deleted_style_project(home: Path, **extra: Any) -> Path:
    project = home / "projects" / "p"
    src = write(project / "src.mp4")
    hl = {
        "style": "gone",
        "candidates": [
            {
                "id": "h001",
                "source": str(src),
                "start": 0.0,
                "end": 2.0,
                "score": 9.0,
                "duration": 2.0,
                "label": "a",
            }
        ],
        **extra,
    }
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    return project


def _fake_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``clips`` run without ffmpeg: probe and the encoder faked, no transcript or audio."""
    monkeypatch.setattr(clips, "probe", media_info)
    install_fake_export(monkeypatch, clips, content=b"n")
    monkeypatch.setattr(clips.sources, "source_words", lambda _src, _lang=None: None)
    monkeypatch.setattr(clips.sources, "audio_window", lambda *a: None)


def test_clips_of_a_deleted_user_style_use_its_recorded_base(home, monkeypatch, caplog):
    """The style is gone from the library: clips reads the base the file records (its playback
    order and tails), with one INFO line, and does not rewrite highlights.json."""
    caplog.set_level(logging.INFO)
    project = _deleted_style_project(home, style_base="commentary")
    before = (project / "highlights.json").read_bytes()
    _fake_export(monkeypatch)
    clips.export_clips(project, top=1, speech_aware=False)
    doc = json.loads((project / "clips" / "manifest.json").read_text(encoding="utf-8"))
    base = get_style("commentary")
    assert doc["order"] == base.default_order == "chronological"
    assert doc["smart_tail"]["style"] == "commentary"
    assert doc["smart_tail"]["tail_min"] == base.tail_min
    assert "Style 'gone' is not in the style library (deleted?); using its base 'commentary'" in [
        r.getMessage() for r in caplog.records
    ]
    assert (project / "highlights.json").read_bytes() == before


def test_clips_of_a_deleted_user_style_without_a_recorded_base_is_a_clear_error(
    home, monkeypatch, capsys, caplog
):
    """An older file (no style_base): one ERROR line, exit 1, nothing exported."""
    from cutpoint import cli  # noqa: PLC0415

    project = _deleted_style_project(home)
    _fake_export(monkeypatch)
    code = cli.main(["clips", str(project), "--top", "1"])
    assert code == 1
    [error] = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert "'gone' of this analysis is not in the style library" in error.getMessage()
    assert "run 'cutpoint analyze' again with another --style" in error.getMessage()
    assert error.exc_info is None
    assert not (project / "clips").exists()


def test_select_moments_max_total_takes_a_candidate_without_duration(tmp_path):
    """A101: the span gives the length, as without --max-total."""
    cands = [
        {"id": "a", "score": 2, "start": 0.0, "end": 10.0},
        {"id": "b", "score": 1, "start": 20.0, "end": 45.0},
    ]
    chosen, _ = select_moments(_project(tmp_path, cands), top=2, max_total=30)
    assert [c["id"] for c in chosen] == ["a"]


def test_manifest_parts_rejects_other_shapes():
    fmt = {"width": 1, "height": 1, "fps": 30}
    assert manifest_parts({"format": fmt, "clips": [{"file": "a.mp4"}]}, need_format=True) == (
        fmt,
        [{"file": "a.mp4"}],
    )
    assert manifest_parts({"clips": []}) == ({}, [])
    for bad in ([], {"clips": {}}, {"clips": [{"x": 1}]}, {"format": 5, "clips": []}):
        with pytest.raises(ValueError, match="not a clips manifest"):
            manifest_parts(bad)
    with pytest.raises(ValueError, match="not a clips manifest"):
        manifest_parts({"clips": []}, need_format=True)
    ranged = {"clips": [{"file": "a.mp4", "source": "s", "start": 1, "end": 2.5}]}
    assert manifest_parts(ranged, need_range=True)[1] == ranged["clips"]
    for entry in ({"file": "a.mp4"}, {"file": "a.mp4", "source": "s", "start": 1}):
        with pytest.raises(ValueError, match=r"m\.json is not a clips manifest"):
            manifest_parts({"clips": [entry]}, need_range=True, path=Path("m.json"))


def test_speech_pass_drops_a_clip_inside_a_longer_one(speech_env, caplog):
    """A102: nested picks of one source keep the outer clip whole and warn."""
    project, _ = speech_env
    plans = [_plan("long.mp4", 10.0, 40.0), _plan("inner.mp4", 12.0, 16.0)]
    plans[0].keep, plans[1].keep = (12.0, 38.0), (13.0, 15.0)
    with caplog.at_level("WARNING"):
        speech_pass(plans, boundaries.SpeechSettings(), project, fit_end=lambda e: True)
    assert [p.name for p in plans] == ["long.mp4"]
    assert plans[0].start <= 10.0 and plans[0].end >= 40.0
    assert "inner.mp4" in caplog.text and "long.mp4" in caplog.text


def test_partly_overlapping_clips_are_still_cut_apart():
    a = boundaries.ClipBounds("a", "k", 10.0, 20.0, 12.0, 18.0, 0.0, 100.0, None)
    b = boundaries.ClipBounds("b", "k", 16.0, 30.0, 18.0, 28.0, 0.0, 100.0, None)
    assert boundaries.separate([a, b]) == []
    assert a.end == b.start and 10.0 < a.end < 30.0


def test_export_clips_takes_a_list_shaped_highlights_file(tmp_path):
    """A229: no AttributeError on the .get of a list."""
    project = tmp_path / "proj"
    write(project / "highlights.json", "[1]")
    with pytest.raises(Exception) as err:  # whatever it says, not a traceback of .get
        clips.export_clips(project)
    assert not isinstance(err.value, AttributeError)


# -- replace_plan / warn_kept: what clips --force and extend may replace -----------------
# Each edge case runs once as clips (no clip_sources) and once as extend (clip_sources: every
# source of the clips folder's manifest); both must give the same plan.
MODES = {"clips": lambda _keys: None, "extend": lambda keys: keys}


def _clips_folder(tmp_path: Path, listed: list[dict[str, Any]] | None) -> Path:
    out = tmp_path / "clips"
    out.mkdir()
    if listed is not None:
        manifest = {"generator": planning.MANIFEST_GENERATOR, "clips": listed}
        write(out / "manifest.json", json.dumps(manifest))
    return out


def _keys(*paths: Path) -> set[Any]:
    return folders.source_keys((str(p), p, []) for p in paths)


@pytest.mark.parametrize("mode", MODES)
def test_replace_plan_replaces_only_what_an_earlier_run_made(tmp_path, mode):
    out = _clips_folder(
        tmp_path,
        [{"file": "001_a.mp4", "srt": "001_a.srt"}, {"file": "002_b.mp4", "srt": "002_b.srt"}],
    )
    for n in ("001_a.mp4", "001_a.srt", "002_b.mp4", "050_mine.mp4", "050_mine.srt", "notes.txt"):
        write(out / n, n)
    write(out / ".003_c.part.mp4", "leftover of a hard-killed run")
    src = write(tmp_path / "footage.mp4", "source")
    plan = planning.replace_plan(
        out, ["001_a.mp4", "003_c.mp4"], _keys(src), clip_sources=MODES[mode](_keys(src))
    )
    # 002_b is not written again: the new manifest.json no longer lists it, so it goes too
    assert sorted(plan.replace) == [".003_c.part.mp4", "001_a.mp4", "001_a.srt", "002_b.mp4"]
    assert plan.kept == ["050_mine.mp4"] and plan.kept_sources == []
    assert plan.had_manifest is True and plan.renamed == {}
    assert plan.protected_srts == {config.path_key("050_mine.srt", out)}


@pytest.mark.parametrize("mode", MODES)
def test_replace_plan_never_replaces_a_source_in_the_folder(tmp_path, mode):
    out = _clips_folder(tmp_path, [{"file": "001_a.mp4"}, {"file": "002_b.mp4"}])
    listed_src = write(out / "002_b.mp4", "a listed clip used as a source")
    src = write(out / "footage.mp4", "source")
    write(out / "001_a.mp4", "old")
    keys = _keys(src, listed_src)
    plan = planning.replace_plan(out, ["001_a.mp4"], keys, clip_sources=MODES[mode](keys))
    assert plan.replace == ["001_a.mp4"]
    assert plan.kept_sources == ["002_b.mp4", "footage.mp4"] and plan.kept == []
    with pytest.raises(ValueError, match="reads it as a source"):
        planning.replace_plan(out, ["footage.mp4"], keys, clip_sources=MODES[mode](keys))


@pytest.mark.parametrize("mode", MODES)
def test_replace_plan_keeps_a_clip_renamed_in_case_only(tmp_path, mode, case_insensitive):
    """The manifest lists 001_Joke; review renamed it 001_joke. On a case-insensitive file
    system the listed name finds it, but it stays (kept, with its .srt) unless the run writes
    its clip again; then it goes under the spelling on disk, .srt included."""
    out = _clips_folder(tmp_path, [{"file": "001_Joke.mp4", "srt": "001_Joke.srt"}])
    write(out / "001_joke.mp4", "renamed")
    write(out / "001_joke.srt", "its subtitles")
    case_insensitive(out)
    clip_sources = MODES[mode](set())
    plan = planning.replace_plan(out, ["001_b.mp4"], set(), clip_sources=clip_sources)
    assert plan.replace == [] and plan.kept == ["001_joke.mp4"]
    plan = planning.replace_plan(out, ["001_Joke.mp4"], set(), clip_sources=clip_sources)
    assert sorted(plan.replace) == ["001_joke.mp4", "001_joke.srt"] and plan.kept == []


@pytest.mark.parametrize("mode", MODES)
def test_replace_plan_names_a_clip_renamed_during_review(tmp_path, mode):
    out = _clips_folder(tmp_path, [{"file": "001_a.mp4", "size": 7}])
    write(out / "099_mine.mp4", "renamed")  # 7 bytes, and 001_a.mp4 is gone
    plan = planning.replace_plan(out, ["001_b.mp4"], set(), clip_sources=MODES[mode](set()))
    assert plan.kept == ["099_mine.mp4"] and plan.renamed == {"099_mine.mp4": "001_a.mp4"}


@pytest.mark.parametrize("mode", MODES)
def test_replace_plan_stops_before_replacing_what_cutpoint_did_not_make(tmp_path, mode):
    out = _clips_folder(tmp_path, [{"file": "001_a.mp4"}])
    write(out / "002_mine.mp4", "added by hand")
    clip_sources = MODES[mode](set())
    with pytest.raises(FileExistsError, match=r"Cannot replace 002_mine.mp4"):
        planning.replace_plan(out, ["002_mine.mp4"], set(), clip_sources=clip_sources)
    write(out / "manifest.json", json.dumps({"generator": "another tool", "clips": []}))
    with pytest.raises(FileExistsError, match="may not be one cutpoint wrote"):
        planning.replace_plan(out, ["001_a.mp4"], set(), clip_sources=clip_sources)


@pytest.mark.parametrize("mode", MODES)
def test_replace_plan_without_a_manifest_keeps_all_but_leftovers(tmp_path, mode, caplog):
    out = _clips_folder(tmp_path, None)
    for n in ("001_a.mp4", "001_a.srt"):
        write(out / n, n)
    write(out / ".001_a.part.mp4", "leftover")
    clip_sources = MODES[mode](set())
    plan = planning.replace_plan(out, ["002_b.mp4"], set(), clip_sources=clip_sources)
    assert plan.replace == [".001_a.part.mp4"] and plan.kept == ["001_a.mp4"]
    assert plan.had_manifest is False
    assert plan.protected_srts == {config.path_key("001_a.srt", out)}
    assert "has no readable manifest.json" in caplog.text
    with pytest.raises(FileExistsError, match=r"no readable manifest.json"):
        planning.replace_plan(out, ["001_a.mp4"], set(), clip_sources=clip_sources)


@pytest.mark.parametrize("mode", MODES)
def test_replace_plan_of_a_missing_folder_is_empty(tmp_path, mode):
    plan = planning.replace_plan(
        tmp_path / "new", ["001_a.mp4"], set(), clip_sources=MODES[mode](set())
    )
    assert plan == planning.ReplacePlan([], [], [], True, set(), {})


def test_replace_plan_names_a_source_of_the_clips_as_one_only_in_extend(tmp_path):
    """extend: a source of a clip it does not re-export is named as a kept source; not read
    this run, it is protected no further than clips --force protects it."""
    out = _clips_folder(tmp_path, [{"file": "001_a.mp4"}])
    other = write(out / "other.mp4", "source of a clip not re-exported")
    as_clips = planning.replace_plan(out, ["001_a.mp4"], set())
    assert as_clips.kept == ["other.mp4"] and as_clips.kept_sources == []
    as_extend = planning.replace_plan(out, ["001_a.mp4"], set(), clip_sources=_keys(other))
    assert as_extend.kept == [] and as_extend.kept_sources == ["other.mp4"]


def test_warn_kept_warns_kept_sources_and_kept_clips(tmp_path, caplog, capsys):
    out = tmp_path / "clips"
    plan = planning.ReplacePlan(
        [], ["footage.mp4"], ["099_mine.mp4"], True, set(), {"099_mine.mp4": "001_a.mp4"}
    )
    with caplog.at_level(logging.WARNING), jsonout.run("clips") as result:
        planning.warn_kept(out, plan, "a source of these clips is never replaced;")
    hint = "assemble includes them, so move them out if they are not part of the edit"
    sources_msg = (
        f"Kept footage.mp4 in {out.absolute()}: a source of these clips is never replaced; {hint}"
    )
    clips_msg = (
        f"Kept 099_mine.mp4 in {out.absolute()}: cutpoint exported 099_mine.mp4 as 001_a.mp4 "
        f"(renamed during review); {hint}"
    )
    assert caplog.messages == [sources_msg, clips_msg]
    assert result.envelope()["warnings"] == [
        {"code": "kept_sources", "message": sources_msg, "files": [str(out / "footage.mp4")]},
        {"code": "kept_clips", "message": clips_msg, "files": [str(out / "099_mine.mp4")]},
    ]
    caplog.clear()
    planning.warn_kept(out, planning.ReplacePlan([], [], [], True, set(), {}), "unused")
    assert caplog.messages == []
