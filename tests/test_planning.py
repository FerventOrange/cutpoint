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
from cutpoint import boundaries, clips, extend, planning
from cutpoint.encode import OutputFormat
from cutpoint.events.tails import EventTailSettings
from cutpoint.ffmpeg_utils import MediaInfo
from cutpoint.planning import Padding, pad, select_moments, speech_pass
from cutpoint.styles import get_style
from cutpoint.tail import TailSettings


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
    assert len(chosen) == 5 and style == "gaming"


def test_select_moments_ids_keep_their_order(tmp_path):
    project = _project(tmp_path, [_cand("a", 1), _cand("b", 3), _cand("c", 2)])
    chosen, _ = select_moments(project, ids=["a", "c"])
    assert [c["id"] for c in chosen] == ["a", "c"]
    with pytest.raises(ValueError, match="Unknown ids: zz"):
        select_moments(project, ids=["a", "zz"])


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


@pytest.fixture
def readable(monkeypatch):
    """Readable part of every source: ``ranges[0]`` (default the whole source)."""
    ranges: list[tuple[float, float] | None] = [None]
    monkeypatch.setattr(planning.sources, "readable_range", lambda src, t, d: ranges[0] or (0.0, d))
    return ranges


def test_pad_applies_and_records(readable):
    start, end, rec = pad(Path("v.mp4"), media_info(), 10.0, 20.0, Padding(2.0, 3.0))
    assert (start, end) == (8.0, 23.0)
    assert rec == {
        "requested": {"start": 2.0, "end": 3.0},
        "applied": {"start": 2.0, "end": 3.0},
        "pre_pad": {"start": 10.0, "end": 20.0},
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


@pytest.fixture
def speech_env(monkeypatch, readable, tmp_path):
    """One utterance 10-14 s in every source; records the speech-map and words calls."""
    calls: dict[str, list[Any]] = {"maps": [], "words": [], "languages": []}

    def fake_map(src, words, events, settings):
        calls["maps"].append((src, words, events))
        span = boundaries.Span(10.0, 14.0, frozenset({"word"}), "a long sentence here")
        return boundaries.SpeechMap([span], settings)

    def fake_words(src, language=None):
        calls["words"].append(src)
        calls["languages"].append(language)
        return [{"start": 10.0, "end": 14.0, "word": "sentence"}]

    monkeypatch.setattr(planning.boundaries, "build_speech_map", fake_map)
    monkeypatch.setattr(planning.sources, "source_words", fake_words)
    return tmp_path, calls


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


def test_speech_pass_reads_the_words_of_the_given_language(speech_env):
    """#371: the transcript language of the analysis reaches the words read."""
    project, calls = speech_env
    plans = [_plan("a.mp4", 5.0, 12.0)]
    speech_pass(plans, boundaries.SpeechSettings(), project, fit_end=lambda e: True)
    speech_pass(plans, boundaries.SpeechSettings(), project, fit_end=lambda e: True, language="de")
    assert calls["languages"] == [None, "de"]


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


# -- clips.plan_clips ----------------------------------------------------------------
SRC_A, SRC_B = r"D:\a.mp4", r"D:\b.mp4"


def _moment(cid: str, source: str, start: float, end: float, **extra: Any) -> dict[str, Any]:
    return {"id": cid, "source": source, "start": start, "end": end, "score": 1.0, **extra}


@pytest.fixture
def plan_env(monkeypatch, speech_env):
    """plan_clips inputs for two sources; every source is fully readable, no ffmpeg."""
    monkeypatch.setattr(planning.sources, "readable_end", lambda src, t, d: d)
    monkeypatch.setattr(planning.sources, "audio_window", lambda *a: pytest.fail("audio read"))
    project, calls = speech_env

    def run(ordered, **kw: Any) -> list[clips.ClipPlan]:
        kw.setdefault("tail", TailSettings(enabled=False))
        kw.setdefault("speech", boundaries.SpeechSettings(enabled=False))
        return clips.plan_clips(
            ordered,
            project,
            io_paths={SRC_A: Path("a.mp4"), SRC_B: Path("b.mp4")},
            infos={SRC_A: media_info(duration=100.0), SRC_B: media_info(duration=50.0)},
            events=None,
            event_settings=EventTailSettings(enabled=False),
            **kw,
        )

    return run, calls


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
    assert first.pad_rec["pre_pad"] == {"start": 30.0, "end": 40.0}
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
@pytest.mark.parametrize(
    "module", ["cutpoint.extend", "cutpoint.clips", "cutpoint.planning", "cutpoint.encode"]
)
def test_clip_modules_import_in_a_fresh_interpreter(module):
    subprocess.run([sys.executable, "-c", f"import {module}"], check=True, timeout=120)


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

    def export(src, info, start, end, out, *_a, **_kw):
        out.write_bytes(b"n")

    monkeypatch.setattr(clips, "export_clip_checked", export)
    monkeypatch.setattr(clips.sources, "source_words", lambda _src, _lang=None: None)
    monkeypatch.setattr(clips.sources, "audio_window", lambda *a: None)
    clips.export_clips(
        project, segments=seg, smart_tail_on=on, tail_min=1.5, tail_max=5.0, speech_aware=False
    )
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Exporting 1")]
    assert line.endswith(f"(segments order); {clause}")
