"""``analyze`` steps: reactions, keyframes and crashes in a fixed order, the progress lines,
``--skip``, the layer plan and the single atomic write of highlights.json."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import cutpoint.actioncam as actioncam
from conftest import HOP, StubTranscriber, add_profile, run_json, set_config, write
from cutpoint import (
    activity,
    analyze,
    cache,
    category,
    cli,
    config,
    crash,
    default_fit,
    insta360,
    jsonout,
    progress,
    transcribe,
    vision,
)
from cutpoint.errors import FFmpegError
from cutpoint.events import cli as events_cli
from cutpoint.events import common, pipeline, report, scoring
from cutpoint.events import paths as event_paths
from cutpoint.ffmpeg_utils import MediaInfo
from cutpoint.styles import get_style

ORDER = ("audio", "voice", "visual", "ocr", "crash")
REAL_CUDA = analyze._cuda


def progress_lines(capsys) -> list[dict]:
    err = capsys.readouterr().err
    return [
        json.loads(x.removeprefix("PROGRESS "))
        for x in err.splitlines()
        if x.startswith("PROGRESS ")
    ]


def last_states(lines: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for g in lines:
        out[g["step"]] = g
    return out


@pytest.fixture
def run(home, tmp_path, monkeypatch):
    """``analyze.analyze`` with Whisper, probing, the signal analysis and the event layers
    replaced by fakes. ``state.plan`` is what plan_layers returns; ``state.fail`` names a
    layer whose fake raises (an Exception, or KeyboardInterrupt with ``state.cancel``)."""
    monkeypatch.setenv(progress.ENV, "1")
    video = write(tmp_path / "videos" / "a.mp4")
    proj = tmp_path / "proj"
    state = SimpleNamespace(
        video=video, proj=proj, plan=analyze.LayerPlan(), fail=None, cancel=False,
        events_calls=[], layers_seen=[], saves=[], vision_ok=True, vision_cached=False,
        rated=0, segments=[], transcripts=[], events=[], cats=[], indexes=[],
    )  # fmt: skip

    def fake_analyze_source(src, style, tr, use_scenes, max_cand, timings, **kw):
        info = MediaInfo(
            src, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None
        )
        cand = analyze.Candidate(src, 10.0, 20.0, 15.0, 1.0, {"loudness": 1.0}, score=50.0)
        return analyze.SourceResult(src, info, 0.0, 60.0, [cand], list(state.segments), None)

    def fake_run_events(project, *, layers, stage, **kw):
        state.events_calls.append((project, tuple(layers), kw))
        state.transcripts.append(common.load_transcript_lines(project))
        # the analysis in progress is visible, highlights.json is not written yet
        assert common.load_highlights(project)["candidates"]
        assert not (project / "highlights.json").exists() or state.old is not None
        for layer in [x for x in ORDER if x in layers]:
            with stage(layer):
                state.layers_seen.append(layer)
                if layer == "crash":  # its files, as detect_project writes them
                    fake_crash_outputs(project)
                if state.fail == layer:
                    raise KeyboardInterrupt if state.cancel else RuntimeError("model blew up")
        with stage("write"):
            lines = common.load_transcript_lines(project)
            out = report.write_outputs(project, [str(video)], state.events, [], lines, {}, {}, {})
            if state.fail == "write" and state.cancel:
                raise KeyboardInterrupt
        return out

    def fake_crash_outputs(project):
        """activity.json, the crash_check sheets and crashes.json with the real writers."""
        add_profile(HOP)  # a library profile, apart from the default (generic)
        activity.resolve(project, activity="puddle-hop", llm=False)
        for f in (project / crash.CHECK_DIR).glob("*.jpg"):
            cache.remove(f)
        c = {"source": str(video), "t0": 10.0, "t1": 11.0, "confidence": 0.9}
        crash.write_sheet(c, project / crash.CHECK_DIR / "c001_a_10.jpg")
        cache.save_json(project / crash.CRASHES_FILE, {"crashes": [c]})

    class FakeRater:
        hits = asked = missed = 0
        stats = dict.fromkeys(
            ("calls", "grab_s", "load_s", "prompt_s", "reply_s", "prompt_tokens", "reply_tokens"), 0
        )
        model = "m"
        provider_name = "ollama"
        consent_needed = None
        cache_only = False

        def __init__(self, **_):
            pass

        def available(self):
            return state.vision_ok

        def cached(self, *a, **k):  # when Ollama is down: the cached rating, if any
            return SimpleNamespace(score=8.0, label="nice") if state.vision_cached else None

        def rate(self, *a, **k):
            state.rated += 1
            if state.cancel and state.fail == "vision":
                raise KeyboardInterrupt
            return SimpleNamespace(score=8.0, label="nice")

        def unload(self):
            pass

    real_save = analyze.cache.save_json

    def save(path, data, **kw):
        state.saves.append(Path(path).name)
        real_save(path, data, **kw)

    state.old = None
    frames = (np.array([10.0]), np.zeros((1, 24, 32, 3), np.uint8))
    monkeypatch.setattr(crash, "decode_window", lambda *a, **k: frames)
    monkeypatch.setattr(analyze, "Transcriber", StubTranscriber)
    monkeypatch.setattr(analyze, "analyze_source", fake_analyze_source)
    monkeypatch.setattr(analyze, "VisionRater", FakeRater)
    monkeypatch.setattr(
        analyze,
        "plan_layers",
        lambda style, sources, skip, cat, visual_index=False: (
            state.cats.append(cat) or state.indexes.append(visual_index) or state.plan
        ),
    )
    monkeypatch.setattr(pipeline, "run_events", fake_run_events)
    monkeypatch.setattr(analyze.cache, "save_json", save)

    def go(style="funny", **kw):
        # vision_time_share: a cap no fake run reaches (the real clock decides the deadline);
        # the tests of the cap pass their own share or clock
        kw = {
            "vision": False,
            "use_scenes": False,
            "layers": True,
            "vision_time_share": 1000.0,
        } | kw
        return analyze.analyze(
            video, get_style(style) if isinstance(style, str) else style, proj, **kw
        )

    state.go = go
    return state


def plan(**layers) -> analyze.LayerPlan:
    return analyze.LayerPlan(layers=layers)


# ----------------------------------------------------------------- the steps and their lines
def test_a_full_run_emits_every_step_in_order_and_ends_done(run, capsys):
    run.plan = plan(reactions=("audio", "voice"), keyframes=("visual",), crashes=("crash",))
    run.go("action", vision=True)
    got = progress_lines(capsys)
    assert {g["total"] for g in got} == {7}
    started = [g["step"] for g in got if g["state"] == "running"]
    # the dialogue reader is for open-list styles (gaming) only: the action style skips it
    ran = [s for s in progress.ANALYZE_STEPS if s != "dialogue"]
    assert list(dict.fromkeys(started)) == ran
    states = {s: g["state"] for s, g in last_states(got).items()}
    assert states == dict.fromkeys(progress.ANALYZE_STEPS, "done") | {"dialogue": "skipped"}
    assert "open-list" in last_states(got)["dialogue"]["detail"]
    assert run.layers_seen == ["audio", "voice", "visual", "crash"]
    assert run.rated == 1


def test_events_run_once_with_all_planned_layers_before_highlights_exist(run):
    run.plan = plan(reactions=("audio",), keyframes=("ocr",))
    run.go()
    ((project, layers, kw),) = run.events_calls
    assert project == run.proj and layers == ("audio", "ocr")
    assert kw["cached_ocr"] is False
    assert kw["llm"] == activity.AUTO  # profile discovery: any local Ollama, quietly
    assert kw["discover_hint"] == analyze.DISCOVER_HINT  # no --no-discover on analyze
    assert (run.proj / "events.json").is_file()
    hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
    assert hl["candidates"][0]["id"] == "h001"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ({"ollama_url": "http://10.0.0.5:11434", "ollama_model": "qwen3.5:9b"},
         ("http://10.0.0.5:11434", "qwen3.5:9b")),
        ({}, (None, None)),  # none given: run_events reads config.json and the environment
    ],
)  # fmt: skip
def test_the_ollama_options_reach_the_event_layers(run, given, expected):
    """#660: analyze --ollama-url/--ollama-model reach run_events (the VLM layer)."""
    run.plan = plan(reactions=("audio",))
    run.go(**given)
    ((_project, _layers, kw),) = run.events_calls
    assert (kw["ollama_url"], kw["ollama_model"]) == expected


def test_skipped_steps_carry_the_plan_reasons(run, capsys):
    run.plan = analyze.LayerPlan(
        skipped={
            "reactions": "events extra not installed",
            "keyframes": "left out by --skip or analysis.skip",
            "crashes": "not action-cam footage",
        }
    )
    run.go()
    final = last_states(progress_lines(capsys))
    assert run.events_calls == []
    assert [(s, final[s]["state"], final[s]["detail"]) for s in progress.ANALYZE_STEPS[1:4]] == [
        ("reactions", "skipped", "events extra not installed"),
        ("keyframes", "skipped", "left out by --skip or analysis.skip"),
        ("crashes", "skipped", "not action-cam footage"),
    ]
    assert final["vision"]["state"] == "skipped"
    assert "vision is off" in final["vision"]["detail"]
    assert final["transcribe"]["state"] == final["ranking"]["state"] == "done"


def test_vision_is_skipped_when_ollama_is_unreachable(run, capsys):
    run.vision_ok = False
    run.go(vision=True)
    with_warning = last_states(progress_lines(capsys))["vision"]
    assert with_warning["state"] == "skipped"
    assert "not reachable" in with_warning["detail"]
    assert (run.proj / "highlights.json").is_file()
    assert run.rated == 0


def test_vision_is_skipped_on_a_bad_ollama_port(run, capsys, monkeypatch, caplog):
    """#825: a bad port in OLLAMA_HOST skips the vision step with a warning, not a failure."""
    monkeypatch.setattr(analyze, "VisionRater", vision.VisionRater)  # the real one parses the URL
    monkeypatch.setenv("OLLAMA_HOST", "myhost:99999")
    with jsonout.run("analyze") as result:
        run.go(vision=True)
    got = last_states(progress_lines(capsys))["vision"]
    assert got["state"] == "skipped" and "invalid port" in got["detail"]
    codes = [w["code"] for w in result.warnings]
    assert "vision_unavailable" in codes and "step_failed" not in codes
    assert "vision model not usable" in caplog.text
    assert (run.proj / "highlights.json").is_file()


def test_vision_uses_the_cached_ratings_when_ollama_is_unreachable(run, capsys, caplog):
    """#413: Ollama down, but the top candidate has a cached rating: the vision step runs on
    the cache and the warning says how many candidates had none."""
    run.vision_ok, run.vision_cached = False, True
    run.go(vision=True)
    assert last_states(progress_lines(capsys))["vision"]["state"] == "done"
    assert run.rated == 1
    hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
    (cand,) = hl["candidates"]
    assert cand["signals"]["vision"] == 8.0
    assert hl["settings"]["vision_used"] is True
    assert hl["settings"]["vision"].items() >= {"requested": 1, "failed": 0, "uncached": 0}.items()
    assert "using 1 cached rating, 0 candidates had no cached rating" in caplog.text


def test_vision_counts_the_candidates_without_a_cached_rating(run, caplog, monkeypatch):
    """#413: two candidates, only the first has a cached rating. The settings and the warning
    count the miss; an online run has no ``uncached`` key."""

    def two(src, style, tr, use_scenes, max_cand, timings, **kw):
        info = MediaInfo(
            src, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None
        )
        cands = [
            analyze.Candidate(src, 10.0, 20.0, 15.0, 1.0, {"loudness": 1.0}, score=60.0),
            analyze.Candidate(src, 30.0, 40.0, 35.0, 1.0, {"loudness": 1.0}, score=50.0),
        ]
        return analyze.SourceResult(src, info, 0.0, 60.0, cands, [], None)

    class Rater:
        cache_only = False
        hits = asked = missed = 0
        stats = dict.fromkeys(
            ("calls", "grab_s", "load_s", "prompt_s", "reply_s", "prompt_tokens", "reply_tokens"), 0
        )
        model = "m"
        provider_name = "ollama"
        consent_needed = None

        def __init__(self, **_):
            pass

        def available(self):
            return run.vision_ok

        def cached(self, src, start, end, base, activity=None, transcript=None, laughter=False):
            hit = not self.cache_only or start == 10.0
            return SimpleNamespace(score=8.0, label="nice") if hit else None

        def rate(self, src, start, end, base, activity=None, transcript=None, laughter=False):
            res = self.cached(src, start, end, base)
            if res is None:
                self.missed += 1
            return res

        def unload(self):
            pass

    monkeypatch.setattr(analyze, "analyze_source", two)
    monkeypatch.setattr(analyze, "VisionRater", Rater)
    run.vision_ok = False
    run.go(vision=True, vision_top=2)
    hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
    assert hl["settings"]["vision_used"] is True
    assert hl["settings"]["vision"].items() >= {"requested": 2, "failed": 0, "uncached": 1}.items()
    warning = next(r.getMessage() for r in caplog.records if "using " in r.getMessage())
    assert "1 cached" in warning and "1 candidate" in warning

    run.vision_ok = True  # online: nothing is "uncached"
    run.go(vision=True, vision_top=2)
    hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
    assert hl["settings"]["vision"].items() >= {"requested": 2, "failed": 0}.items()


def _two_candidates(src, style, tr, use_scenes, max_cand, timings, **kw):
    info = MediaInfo(
        src, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None
    )
    cands = [
        analyze.Candidate(src, 10.0, 20.0, 15.0, 1.0, {"loudness": 1.0}, score=60.0),
        analyze.Candidate(src, 30.0, 40.0, 35.0, 1.0, {"loudness": 1.0}, score=50.0),
    ]
    return analyze.SourceResult(src, info, 0.0, 60.0, cands, [], None)


def _timed_rater(timed: bool):
    """A rater whose first candidate comes from the cache and whose second is asked (one call
    of 3 s load, 1 s prompt, 2 s reply, 100 and 20 tokens); ``timed`` False leaves the stats 0."""

    class Rater:
        cache_only = False
        asked = missed = 0
        model = "qwen-test"
        provider_name = "ollama"
        consent_needed = None

        def __init__(self, **_):
            self.hits = 0
            self.stats = dict.fromkeys(("calls", "grab_s", "load_s", "prompt_s", "reply_s"), 0) | {
                "prompt_tokens": 0,
                "reply_tokens": 0,
            }

        def available(self):
            return True

        def cached(self, *a, **k):
            return None

        def rate(self, src, start, end, base, **k):
            if start == 10.0:
                self.hits += 1
                return SimpleNamespace(score=9.0, label="cached one")
            if timed:
                self.stats |= {
                    "calls": 1, "grab_s": 0.25, "load_s": 3.0, "prompt_s": 1.0, "reply_s": 2.0,
                    "prompt_tokens": 100, "reply_tokens": 20,
                }  # fmt: skip
            return SimpleNamespace(score=4.0, label="asked one")

        def unload(self):
            pass

    return Rater


@pytest.mark.usefixtures("pinned_prior")
def test_the_vision_settings_carry_the_run_stats(run, caplog, monkeypatch):
    """#909: settings.vision has every stat key, the cached ratings are counted apart from the
    calls, and one INFO line says what the step cost."""
    monkeypatch.setattr(analyze, "analyze_source", _two_candidates)
    monkeypatch.setattr(analyze, "VisionRater", _timed_rater(True))
    with caplog.at_level(logging.INFO, logger="cutpoint.analyze"):
        run.go(vision=True, vision_top=2)
    hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
    v = hl["settings"]["vision"]
    assert set(v) == {
        "requested", "failed", "rated", "cached", "calls", "vision_s", "pre_vision_s", "grab_s",
        "load_s", "prompt_s", "reply_s", "prompt_tokens", "reply_tokens", "model", "provider",
        "cut", "band_k", "rated_by_kind", "centre", "share", "budget_s", "budget_base", "capped",
        "skipped_by_cap", "clamped",
    }  # fmt: skip
    assert (v["requested"], v["failed"], v["rated"]) == (2, 0, 2)
    # both moments fit the edit's target: the cut is after the 2nd, k = band_k(2) = 2
    assert (v["cut"], v["band_k"]) == (2, 2)
    assert v["rated_by_kind"] == {"frames": 2}
    # (9 + 4 + 3 * 3.2) / 5: two frames ratings, shrunk toward the prior
    assert v["centre"] == {"speech": None, "frames": 4.52}
    assert (v["share"], v["budget_base"]) == (1000.0, "pre_vision_s")
    # pre_vision_s is stored to 0.01 s
    assert v["budget_s"] == pytest.approx(1000.0 * v["pre_vision_s"], abs=1000.0 * 0.006)
    assert (v["capped"], v["skipped_by_cap"], v["clamped"]) == (False, 0, 0)
    assert (v["cached"], v["calls"]) == (1, 1)  # the hit is not a call
    assert (v["grab_s"], v["load_s"], v["prompt_s"], v["reply_s"]) == (0.25, 3.0, 1.0, 2.0)
    assert (v["prompt_tokens"], v["reply_tokens"]) == (100, 20)
    assert (v["model"], v["provider"]) == ("qwen-test", "ollama")
    assert v["vision_s"] >= 0 and v["pre_vision_s"] >= 0
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Vision: 2 "))
    assert re.fullmatch(
        r"Vision: 2 rated \(1 cached\) in \d+ s \(\d+% of analyze before vision\)", line
    )


def test_a_rerank_is_marked_and_has_no_share_of_the_analysis(run, caplog, monkeypatch):
    """#909: on a re-rank (earlier_total_s given) settings.vision says so and the INFO line leaves
    out the share of analyze, which the re-rank did not spend; a first run is not marked."""
    monkeypatch.setattr(analyze, "analyze_source", _two_candidates)
    monkeypatch.setattr(analyze, "VisionRater", _timed_rater(True))

    def vision_run(**kw):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="cutpoint.analyze"):
            run.go(vision=True, vision_top=2, **kw)
        hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
        line = next(
            r.getMessage() for r in caplog.records if r.getMessage().startswith("Vision: 2 ")
        )
        return hl, line

    hl, line = vision_run()
    assert "rerank" not in hl["settings"]["vision"] and "%" in line

    hl, line = vision_run(earlier_total_s=12.0)
    assert hl["settings"]["vision"]["rerank"] is True
    assert re.fullmatch(r"Vision: 2 rated \(1 cached\) in \d+ s", line)
    assert "%" not in line
    assert hl["timings"]["total_s"] == 12.0 and "rerank_s" in hl["timings"]


def test_the_vision_stats_do_not_change_the_ranking(run, monkeypatch):
    """#909 is timing only: the scores, labels and order are the same whatever the stats say."""
    monkeypatch.setattr(analyze, "analyze_source", _two_candidates)
    out = {}
    for timed in (False, True):
        monkeypatch.setattr(analyze, "VisionRater", _timed_rater(timed))
        run.go(vision=True, vision_top=2)
        hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
        out[timed] = [(c["id"], c["start"], c["score"], c["label"]) for c in hl["candidates"]]
    assert out[False] == out[True]
    assert [c[1] for c in out[True]] == [10.0, 30.0]  # the 9.0 rated one stays first


def test_without_layers_no_event_step_runs(run, capsys):
    run.plan = plan(reactions=("audio",))
    run.go(layers=False)  # reanalyze / 360 analyze: events.json reused as it is
    assert run.events_calls == []
    assert progress_lines(capsys) == []  # not a progress run at all
    assert (run.proj / "highlights.json").is_file()


def test_the_events_json_rescores_the_ranking(run, monkeypatch):
    run.plan = plan(reactions=("audio",))
    run.go()
    ev = {
        "created": "t",
        "events": [
            {"source": str(run.video), "kind": "audio", "label": "Laughter",
             "confidence": 1.0, "start": 12.0, "end": 14.0},
        ],
    }  # fmt: skip
    write(run.proj / "events.json", json.dumps(ev))
    monkeypatch.setattr(pipeline, "run_events", lambda *a, **k: None)  # keep the file written
    run.go()
    hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
    cand = hl["candidates"][0]
    assert cand["score"] > cand["score_pre_events"] == 50.0


# ----------------------------------------------------------------- transcribe pct
def test_transcribe_pct_is_the_extract_then_whisper(run, capsys, monkeypatch):
    """analyze_source reports its source's share done; the loop shows it in the step's pct
    and never lowers it (a recovered part's extract starts again at its offset)."""
    monkeypatch.setattr(progress, "MIN_INTERVAL_S", 0.0)  # every update is a line
    fake = analyze.analyze_source

    def source(src, *a, report=None, **kw):
        assert report is not None  # progress is on
        for done in (0.05, 0.1, 0.55, 1.0, 0.1):  # extract, Whisper, then a lower value
            report(done)
        return fake(src, *a, **kw)

    monkeypatch.setattr(analyze, "analyze_source", source)
    run.go()
    lines = [g for g in progress_lines(capsys) if g["step"] == "transcribe"]
    assert [(g["state"], g["pct"]) for g in lines] == [
        ("running", None),
        ("running", 0),
        ("running", 5),
        ("running", 10),
        ("running", 55),
        ("running", 100),
        ("done", 100),
    ]
    assert {g["detail"] for g in lines[1:-1]} == {run.video.name}


class _Stop(Exception):
    pass


@pytest.mark.parametrize(
    "offset, cached_wav, want",
    [
        (0.0, False, [0.05, 0.1, 0.55, 1.0]),
        (50.0, False, [0.075, 0.1, 0.775, 1.0]),  # a recovered part: the source's second half
        (0.0, True, [0.55, 1.0]),  # the WAV is cached: no extract to watch
    ],
)
def test_analyze_source_reports_extract_then_whisper(
    tmp_path, monkeypatch, offset, cached_wav, want
):
    src = write(tmp_path / "a.mp4")
    info = MediaInfo(src, 100.0, 100.0, 640, 360, Fraction(30), "yuv420p", "h264", 1, "aac",
                     16000, 1)  # fmt: skip
    monkeypatch.setattr(analyze, "probe", lambda path: info)
    monkeypatch.setattr(analyze.cache, "source_dir", lambda s, off=0.0: tmp_path / "cache")
    if cached_wav:
        write(tmp_path / "cache" / "audio16k.wav")

    @contextlib.contextmanager
    def watch(wav, seconds, report):
        assert seconds == 100.0 - offset
        report(0.5)
        report(1.0)
        yield

    monkeypatch.setattr(analyze, "_watch_extract", watch)
    monkeypatch.setattr(analyze, "extract_audio", lambda *a, **k: write(a[1]))
    monkeypatch.setattr(analyze, "check_read", lambda *a: None)
    monkeypatch.setattr(analyze, "_loudness", lambda wav, cdir: (np.zeros(1), np.zeros(1)))

    class Tr:
        model_name = "m"
        language = None

        def transcribe(self, wav, on_progress=None):
            on_progress(0.5)
            on_progress(1.0)
            raise _Stop  # the rest of the analysis is not needed here

    got: list[float] = []
    with pytest.raises(_Stop):
        analyze.analyze_source(
            src, get_style("funny"), Tr(), False, None, {}, offset=offset, report=got.append
        )
    assert got == pytest.approx(want)


def test_whisper_reports_each_segment_end_as_it_is_decoded(tmp_path, monkeypatch):
    seen: list[float] = []

    def segments():  # faster-whisper decodes lazily: progress comes as segments are yielded
        for end in (10.0, 40.0, 100.0):
            seen.append(end)
            yield SimpleNamespace(start=end - 5, end=end, text=" hi", avg_logprob=-0.1,
                                  no_speech_prob=0.0, compression_ratio=1.0, words=[])  # fmt: skip

    model = SimpleNamespace(
        transcribe=lambda audio, **kw: (segments(), SimpleNamespace(duration=100.0))
    )
    monkeypatch.setattr(transcribe, "load_wav", lambda wav: None)
    tr = transcribe.Transcriber(model="tiny")
    tr._model = model
    got: list[tuple[float, list[float]]] = []
    out = tr.transcribe(tmp_path / "a.wav", lambda frac: got.append((frac, list(seen))))
    assert [g[0] for g in got] == [0.1, 0.4, 1.0]
    assert [len(g[1]) for g in got] == [1, 2, 3]  # each report before the next decode
    assert len(out) == 3
    assert len(tr.transcribe(tmp_path / "a.wav")) == 3  # the callback is optional


def test_whisper_stores_each_word_probability(tmp_path, monkeypatch):
    words = [
        SimpleNamespace(start=0.0, end=0.4, word=" hello", probability=0.98765),
        SimpleNamespace(start=0.5, end=0.9, word=" world", probability=0.5),
    ]
    seg = SimpleNamespace(start=0.0, end=1.0, text=" hello world", avg_logprob=-0.1,
                          no_speech_prob=0.0, compression_ratio=1.0, words=words)  # fmt: skip
    model = SimpleNamespace(
        transcribe=lambda audio, **kw: (iter([seg]), SimpleNamespace(duration=1.0))
    )
    monkeypatch.setattr(transcribe, "load_wav", lambda wav: None)
    tr = transcribe.Transcriber(model="tiny")
    tr._model = model
    got = tr.transcribe(tmp_path / "a.wav")[0]["words"]
    assert [(w["word"], w["probability"]) for w in got] == [("hello", 0.988), ("world", 0.5)]


def test_an_update_that_changes_nothing_writes_no_line(monkeypatch, capsys):
    monkeypatch.setenv(progress.ENV, "1")
    monkeypatch.setattr(progress, "MIN_INTERVAL_S", 0.0)
    with progress.run(("transcribe",)) as steps:
        steps.start("transcribe", "a.mp4")
        for pct in (5, 5, 5.4, 6, 6):
            steps.update(pct)
    got = [(g["state"], g["pct"]) for g in progress_lines(capsys)]
    assert got == [("running", None), ("running", 5), ("running", 6), ("done", 100)]


def test_no_extract_watch_without_progress(run, monkeypatch):
    monkeypatch.delenv(progress.ENV)
    seen = []
    fake = analyze.analyze_source

    def source(src, *a, report=None, **kw):
        seen.append(report)
        return fake(src, *a, **kw)

    monkeypatch.setattr(analyze, "analyze_source", source)
    run.go()
    assert seen == [None]  # no poller thread when nobody reads the lines


def test_watch_extract_reports_the_growing_temp_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "EXTRACT_POLL_S", 0.01)
    wav = tmp_path / "audio16k.wav"
    tmp = wav.with_suffix(".tmp.wav")
    got: list[float] = []
    seconds = 10.0
    tmp.write_bytes(b"\0" * analyze.WAV_HEADER)
    with analyze._watch_extract(wav, seconds, got.append):
        for k in range(1, 5):  # 2.5 s, 5 s, 7.5 s, 10 s of audio written
            # append like ffmpeg does; rewriting the file lets the poller see it truncated
            with tmp.open("ab") as f:
                f.write(b"\0" * int(2.5 * analyze.WAV_BYTES_PER_S))
            deadline = time.monotonic() + 2.0
            while (not got or got[-1] < k / 4) and time.monotonic() < deadline:
                time.sleep(0.005)
    assert got == sorted(got) and got[-1] == 1.0
    assert {0.25, 0.5, 0.75, 1.0} <= set(got)


def test_whisper_progress_is_passed_only_on_a_transcript_cache_miss(tmp_path):
    calls = []

    def whisper(wav, on_progress=None):
        on_progress(1.0)
        return [{"start": 0, "end": 1}]

    tr = SimpleNamespace(model_name="m", language=None, transcribe=whisper)
    assert analyze._transcript(tmp_path / "a.wav", tmp_path, tr, calls.append)
    assert calls == [1.0]
    analyze._transcript(tmp_path / "a.wav", tmp_path, tr, calls.append)
    assert calls == [1.0]  # cached: Whisper does not run


@pytest.mark.parametrize("cached", [0, 2])
def test_transcribe_says_how_many_sources_are_cached_on_a_rerun(run, capsys, caplog, cached):
    """#109: "transcribe: N/M done, continuing" (INFO and the step's PROGRESS detail) only
    when transcripts of this model are cached; a fresh run looks as before."""
    videos = [run.video, write(run.video.with_name("b.mp4")), write(run.video.with_name("c.mp4"))]
    for v in videos[:cached]:
        write(cache.source_dir(v) / transcribe.transcript_name("stub"), "[]")
    write(cache.source_dir(videos[2]) / transcribe.transcript_name("other"), "[]")  # other model
    with caplog.at_level("INFO"):
        analyze.analyze(
            run.video.parent,
            get_style("funny"),
            run.proj,
            vision=False,
            use_scenes=False,
            layers=True,
        )
    text = "transcribe: 2/3 done, continuing"
    details = [g["detail"] for g in progress_lines(capsys) if g["step"] == "transcribe"]
    assert (text in caplog.messages) is bool(cached)
    assert (text in details) is bool(cached)
    assert not any("continuing" in m for m in caplog.messages if m != text)


def test_cached_transcripts_without_a_transcriber_counts_none(tmp_path):
    """#109: no transcriber (stubbed out by callers that fail analyze early) counts 0
    cached transcripts, so no "continuing" line, instead of failing on ``model_name``."""
    assert analyze._cached_transcripts([write(tmp_path / "a.mp4")], None) == 0


# ----------------------------------------------------------------- failures
def test_a_layer_failure_fails_its_step_warns_and_the_later_steps_run(run, capsys, caplog):
    run.plan = plan(reactions=("audio", "voice"), keyframes=("visual",), crashes=("crash",))
    run.fail = "visual"
    with jsonout.run("analyze") as res:
        run.go("action")
    got = progress_lines(capsys)
    final = last_states(got)
    assert final["reactions"]["state"] == "done"
    assert final["keyframes"]["state"] == "failed"
    assert "model blew up" in final["keyframes"]["detail"]
    assert final["crashes"]["state"] == "done"  # still ran
    assert final["ranking"]["state"] == "done"
    assert run.layers_seen == ["audio", "voice", "visual", "crash"]
    assert [w["code"] for w in res.warnings] == ["step_failed"]
    assert "Keyframes failed: model blew up" in res.warnings[0]["message"]
    assert "Keyframes failed" in caplog.text
    assert (run.proj / "highlights.json").is_file()


def test_a_failed_visual_index_on_gameplay_leaves_the_hud_text_step_done(run, capsys, caplog):
    run.plan = analyze.LayerPlan(layers={"keyframes": ("visual", "ocr")}, game=True)
    run.fail = "visual"
    with jsonout.run("analyze") as res:
        run.go("gaming", visual_index=True)
    final = last_states(progress_lines(capsys))
    assert final["keyframes"]["state"] == "done"
    assert run.layers_seen == ["visual", "ocr"]
    assert [w["code"] for w in res.warnings] == ["step_failed"]
    assert "Visual index failed: model blew up; the HUD text still runs" in caplog.text


def test_analyze_json_reports_a_failed_step_as_a_warning_not_an_error(run, capsys):
    """Through ``cli.main``: a failed layer is a ``step_failed`` warning; the run succeeds."""
    run.plan = plan(reactions=("audio", "voice"), keyframes=("visual",))
    run.fail = "visual"

    code, doc = run_json(capsys, ["analyze", str(run.video), "--style", "action", "--out", "s"])

    assert code == 0 and doc["ok"] is True and doc["error"] is None
    [warning] = [w for w in doc["warnings"] if w["code"] == "step_failed"]
    assert "Keyframes failed: model blew up" in warning["message"]
    assert Path(doc["outputs"]["highlights"]).is_file()


def test_a_failure_of_the_events_run_itself_skips_its_steps_and_goes_on(run, capsys, monkeypatch):
    run.plan = plan(reactions=("audio",), keyframes=("ocr",))

    def boom(project, *, layers, stage, **kw):
        with stage("audio"):
            pass
        raise OSError("disk full")

    monkeypatch.setattr(pipeline, "run_events", boom)
    with jsonout.run("analyze") as res:
        run.go()
    final = last_states(progress_lines(capsys))
    assert final["reactions"]["state"] == "failed"  # the step running when it stopped
    assert "disk full" in final["reactions"]["detail"]
    assert final["keyframes"]["state"] == "skipped"
    assert final["keyframes"]["detail"] == "not run: an earlier step failed"
    assert final["ranking"]["state"] == "done"
    assert [w["code"] for w in res.warnings] == ["step_failed"]


def test_an_events_error_before_its_first_layer_keeps_transcribe_done(run, capsys, monkeypatch):
    """F3: run_events fails before any stage: the transcription stays done and the first
    planned step carries the failure."""
    run.plan = plan(reactions=("audio",), keyframes=("ocr",))

    def boom(project, **kw):
        raise ValueError("No footage of proj is reachable")

    monkeypatch.setattr(pipeline, "run_events", boom)
    with jsonout.run("analyze") as res:
        run.go()
    final = last_states(progress_lines(capsys))
    assert (final["transcribe"]["state"], final["transcribe"]["pct"]) == ("done", 100)
    assert final["reactions"]["state"] == "failed"
    assert "No footage" in final["reactions"]["detail"]
    assert final["keyframes"]["state"] == "skipped"
    assert final["ranking"]["state"] == "done"
    assert [w["code"] for w in res.warnings] == ["step_failed"]


def test_an_early_events_error_blames_the_first_step_in_step_order(run, capsys, monkeypatch):
    """F10: plan_layers adds crashes first; the failure still goes to reactions."""
    run.plan = plan(crashes=("crash",), reactions=("audio", "voice"))

    def boom(project, **kw):
        raise ValueError("No footage of proj is reachable")

    monkeypatch.setattr(pipeline, "run_events", boom)
    run.go("action")
    final = last_states(progress_lines(capsys))
    assert final["reactions"]["state"] == "failed"
    assert final["crashes"]["state"] == "skipped"
    assert final["crashes"]["detail"] == "not run: an earlier step failed"


@pytest.mark.parametrize("doc", ["[]", '{"settings": null}', '{"settings": {"ocr": null}}'])
def test_ocr_options_ignore_a_malformed_events_json(run, doc):
    """F11: a malformed events.json falls back to the defaults instead of stopping."""
    write(run.proj / "events.json", doc)
    run.plan = plan(keyframes=("ocr",))
    run.go()
    ((_, _, kw),) = run.events_calls
    assert kw["ocr"].game == "auto" and kw["ocr"].player_tags == config.player_tags()


def test_a_run_stopping_error_fails_the_step_with_its_text(run, capsys, monkeypatch):
    def unreadable(*a, **k):
        raise RuntimeError("a.mp4 is unreadable")

    monkeypatch.setattr(analyze, "analyze_source", unreadable)
    with pytest.raises(RuntimeError):
        run.go()
    last = progress_lines(capsys)[-1]
    assert (last["step"], last["state"], last["detail"]) == (
        "transcribe", "failed", "a.mp4 is unreadable",
    )  # fmt: skip
    assert not (run.proj / "highlights.json").exists()


def test_no_input_fails_the_first_step(run, capsys, tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze.analyze(tmp_path / "empty", get_style("funny"), run.proj, layers=True)
    last = progress_lines(capsys)[-1]
    assert (last["step"], last["state"]) == ("transcribe", "failed")


# ----------------------------------------------------------------- highlights.json
def test_stale_pre_events_copy_is_removed_only_after_the_final_write(run, monkeypatch):
    """F4: ranked with events, the old highlights.pre-events.json goes, but only once
    highlights.json is written."""
    pre = write(run.proj / "highlights.pre-events.json", "{}")
    write(run.proj / "events.json", json.dumps({"events": [], "created": "t"}))
    monkeypatch.setattr(pipeline, "run_events", lambda *a, **k: None)  # keep events.json
    seen = []
    save = analyze.cache.save_json

    def spy(path, data, **kw):
        if Path(path).name == "highlights.json":
            seen.append(pre.exists())
        save(path, data, **kw)

    monkeypatch.setattr(analyze.cache, "save_json", spy)
    run.plan = plan(reactions=("audio",))
    run.go(vision=True)
    assert seen == [True] and not pre.exists()


def test_a_cancel_in_vision_keeps_the_pre_events_copy(run):
    pre = write(run.proj / "highlights.pre-events.json", "{}")
    run.plan = plan(reactions=("audio",))
    run.cancel, run.fail = True, "vision"
    with pytest.raises(KeyboardInterrupt):
        run.go(vision=True)
    assert pre.exists()


def test_without_events_the_pre_events_copy_stays(run):
    pre = write(run.proj / "highlights.pre-events.json", "{}")
    run.go()  # no layers planned, no events.json
    assert pre.exists()


def _laugh_events(run) -> str:
    return json.dumps({
        "created": "t",
        "events": [
            {"source": str(run.video), "kind": "audio", "label": "Laughter",
             "confidence": 1.0, "start": 12.0, "end": 14.0},
        ],
    })  # fmt: skip


@pytest.mark.usefixtures("pinned_prior")
def test_vision_blends_into_the_analysis_score_then_the_events_apply_again(run, monkeypatch):
    """F5a: with events.json, the vision rating is blended into score_pre_events and the
    event boost is added on top of the blended score."""
    run.plan = plan(reactions=("audio",))
    write(run.proj / "events.json", _laugh_events(run))
    monkeypatch.setattr(pipeline, "run_events", lambda *a, **k: None)  # keep the file written
    run.go()
    (plain,) = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))["candidates"]
    boost = plain["score"] - plain["score_pre_events"]
    assert boost > 0 and plain["score_pre_events"] == 50.0
    run.go(vision=True)
    (cand,) = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))["candidates"]
    w = get_style("funny").vision_weight
    assert (
        cand["signals"]["vision"] == 8.0
        and cand["label"] == plain["label"] == "loud-moment"
        and cand["vision_label"] == "nice"
    )
    # #909: the lone frames rating is centred: (8 + 4 * 3.2) / 5 = 4.16, so 8 becomes
    # 5 + 8 - 4.16 = 8.84 and the blend is 50 * (1 + 0.3 * (8.84 - 5) / 5) = 61.52
    centre = (8.0 + 4 * vision.RATING_PRIOR["frames"]) / 5
    assert centre == pytest.approx(4.16)
    assert cand["vision_rated"]["centre"] == 4.16 and cand["vision_rated"]["kind"] == "frames"
    assert cand["score_pre_events"] == pytest.approx(
        50.0 * (1 + w * (5 + 8.0 - centre - 5) / 5), abs=0.05
    )
    assert cand["score_pre_events"] == 61.5  # funny: vision_weight 0.3
    assert cand["score"] == pytest.approx(cand["score_pre_events"] + boost, abs=0.15)


def test_the_grids_of_the_run_reach_both_rescores_and_the_nominees_keep_their_ratings(
    run, monkeypatch
):
    """The signal grids of this run feed the events' nominees at the ranking and again after
    the vision step; a nominee is rated, keeps the rating through the second rescore, gets a
    fresh id, and of two nominees that tie on the floored score the one the grids like more
    ranks first."""
    sig = {k: np.zeros(6001) for k in ("loudness", "speech", "laughter", "scene", "continuity")}
    sig["loudness"] += 0.3 * np.exp(-(((np.arange(6001) * 0.1 - 100.0) / 1.0) ** 2))
    signals = analyze.PartSignals(6001, sig, [], [], [], [], [])

    def source(src, style, tr, use_scenes, max_cand, timings, **kw):
        info = MediaInfo(
            src, 600.0, 600.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None
        )
        cand = analyze.Candidate(src, 10.0, 20.0, 15.0, 1.0, {"loudness": 1.0}, score=50.0)
        return analyze.SourceResult(src, info, 0.0, 600.0, [cand], [], None, signals)

    monkeypatch.setattr(analyze, "analyze_source", source)
    monkeypatch.setattr(pipeline, "run_events", lambda *a, **k: None)
    seen = []
    real = analyze.rescore

    def rescore(hl, ev, project, parts=None):
        seen.append(parts)
        return real(hl, ev, project, parts)

    monkeypatch.setattr(analyze, "rescore", rescore)
    run.go("gaming")  # no events.json yet: the rescore does not run
    kills = [
        {"source": str(run.video), "start": t, "end": t + 3, "kind": "ocr", "type": "kill",
         "label": "Ash > Bob", "player": True, "player_kill": True, "player_death": False,
         "teamkill": False}
        for t in (300.0, 100.0)
    ]  # fmt: skip
    write(run.proj / "events.json", json.dumps({"created": "t", "events": kills}))
    seen.clear()
    run.go("gaming", vision=True)
    assert len(seen) == 2  # the ranking and after the vision step
    for parts in seen:
        assert [(g.source, g.index, g.offset, g.duration) for g in parts] == [
            (run.video, 0, 0.0, 600.0)
        ]
        assert parts[0].signals is signals
    hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
    cands = hl["candidates"]
    assert [c["id"] for c in cands] == ["h001", "h002", "h003"]
    assert hl["settings"]["last_candidate_id"] == 3
    noms = [c for c in cands if scoring.is_nominated(c)]
    assert len(noms) == 2
    assert all(c["signals"]["vision"] == 8.0 for c in noms)  # rated, and kept by the rescore
    # the same boost and floor; the kill on the loudness bump ranks above the one that is not
    assert noms[0]["score"] == noms[1]["score"]
    assert noms[0]["grid_score"] > noms[1]["grid_score"]
    assert noms[0]["start"] < 100.0 < noms[0]["end"]


def test_a_style_without_an_open_list_gets_no_nominee_and_the_same_suggested_top_on_both_paths(
    run, monkeypatch
):
    """Kills nothing covers nominate no window under a style that does not go past its usual
    list, and the suggestion is the one of a run without events, on the analyze path and when
    ``cutpoint events`` rescores the file afterwards."""

    sig = {k: np.zeros(6001) for k in ("loudness", "speech", "laughter", "scene", "continuity")}
    signals = analyze.PartSignals(6001, sig, [], [], [], [], [])  # grids: a nominee could be scored

    def source(src, style, tr, use_scenes, max_cand, timings, **kw):
        info = MediaInfo(
            src, 600.0, 600.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None
        )
        cands = [
            analyze.Candidate(src, t, t + 30.0, t + 15.0, 1.0, {"loudness": 1.0}, score=s)
            for t, s in ((10.0, 60.0), (200.0, 50.0), (400.0, 40.0))
        ]
        return analyze.SourceResult(src, info, 0.0, 600.0, cands, [], None, signals)

    def saved() -> dict:
        return json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))

    monkeypatch.setattr(analyze, "analyze_source", source)
    monkeypatch.setattr(pipeline, "run_events", lambda *a, **k: None)
    run.go("action")  # no events.json yet
    plain = saved()
    assert len(plain["candidates"]) == 3 and plain["suggested_top"] == 2
    kills = [
        {"source": str(run.video), "start": t, "end": t + 3, "kind": "ocr", "type": "kill",
         "label": "Ash > Bob", "player": True, "player_kill": True, "player_death": False,
         "teamkill": False}
        for t in (120.0, 330.0)
    ]  # fmt: skip
    write(run.proj / "events.json", json.dumps({"created": "t", "events": kills}))
    run.go("action")
    for hl in (saved(), (scoring.rescore_highlights(run.proj), saved())[1]):
        assert [c["id"] for c in hl["candidates"]] == ["h001", "h002", "h003"]
        assert not any(scoring.is_nominated(c) for c in hl["candidates"])
        assert hl["suggested_top"] == plain["suggested_top"]


@pytest.mark.parametrize(("layers", "prepared"), [(True, 2), (False, 1)])
def test_ski_features_are_prepared_again_after_the_layers(run, monkeypatch, layers, prepared):
    """F5b: ski candidates come from the feature grid; it is built again once this run's
    layers wrote events.json, and only once without layers."""
    calls = []
    monkeypatch.setattr(actioncam, "prepare", lambda *a, **k: calls.append(a) or SimpleNamespace())
    monkeypatch.setattr(analyze, "probe", lambda path: SimpleNamespace(duration=60.0))
    picked = []

    def pick(r, style, max_candidates, timings, ctx):
        picked.append(ctx)
        return r.candidates

    monkeypatch.setattr(analyze, "pick_candidates", pick)
    run.plan = plan(reactions=("audio", "voice"), crashes=("crash",))
    run.go("action-cam", layers=layers)
    assert len(calls) == prepared
    assert len(run.events_calls) == int(layers)
    assert len(picked) == 1  # the candidates are picked once, from the last features


@pytest.mark.parametrize(
    ("configured", "tags", "info"),
    [
        ([], ("PlayerOne", "PlayerTwo"), "2 player tags and game somegame"),
        (["ConfigName"], ("ConfigName",), "game somegame"),  # config.json wins for the tags
    ],
)
def test_ocr_options_of_the_last_events_run_are_reused(home, run, caplog, configured, tags, info):
    """Follow-up A: `cutpoint events --player ... --game ...` stays in effect on a
    re-analyze; player_tags in config.json, when set, wins over the saved tags."""
    set_config(home, player_tags=configured)
    settings = {"ocr": {"tags": ["PlayerOne", "PlayerTwo"], "game": "somegame"}}
    write(run.proj / "events.json", json.dumps({"events": [], "settings": settings}))
    run.plan = plan(keyframes=("ocr",))
    caplog.set_level("INFO", logger=analyze.logger.name)
    run.go()
    ((_, _, kw),) = run.events_calls
    assert kw["ocr"].player_tags == tags
    assert kw["ocr"].game == "somegame"
    infos = [r.getMessage() for r in caplog.records if "Reusing" in r.getMessage()]
    assert infos == [f"Reusing {info} from the last `cutpoint events` run"]
    assert "PlayerOne" not in caplog.text  # the names themselves are not logged


def test_ocr_options_default_without_an_earlier_events_run(run, caplog):
    run.plan = plan(keyframes=("ocr",))
    caplog.set_level("INFO", logger=analyze.logger.name)
    run.go()
    ((_, _, kw),) = run.events_calls
    assert kw["ocr"].game == "auto"
    assert kw["ocr"].player_tags == config.player_tags()
    assert "Reusing" not in caplog.text


@pytest.mark.parametrize(
    ("reason", "note"),
    [
        (analyze.SKIP_NO_CUDA, "no CUDA GPU for the keyframes step"),
        (analyze.SKIP_OPTION, "keyframes left out by --skip or analysis.skip"),
        (None, None),  # keyframes ran (or the style has none): the usual advice
    ],
)
def test_the_crash_layer_learns_why_there_is_no_visual_layer(run, monkeypatch, reason, note):
    """Follow-up B: inside analyze, crashes.json's no_visual_layer warning says why."""
    seen = []

    def events(project, **kw):
        seen.append(crash._NO_VISUAL_REASON.get())

    monkeypatch.setattr(pipeline, "run_events", events)
    run.plan = analyze.LayerPlan(
        layers={"crashes": ("crash",)}, skipped={"keyframes": reason} if reason else {}
    )
    run.go("action")
    assert seen == [note]
    assert crash._NO_VISUAL_REASON.get() is None  # reset after the run


def test_highlights_json_is_written_once_atomically(run):
    run.plan = plan(reactions=("audio",), keyframes=("ocr",))
    run.go(vision=True)
    assert run.saves.count("highlights.json") == 1
    assert [p.name for p in run.proj.iterdir() if p.name.startswith(".")] == []  # no temp left


@pytest.mark.parametrize("where", ["reactions", "ranking-vision"])
def test_a_cancel_leaves_the_previous_highlights_json_byte_identical(run, capsys, where):
    old = b'{"style": "old", "candidates": []}\n'
    write(run.proj / "highlights.json", old)
    run.old = old
    run.plan = plan(reactions=("audio",))
    run.cancel = True
    if where == "reactions":
        run.fail = "audio"
    else:
        run.fail = "vision"
    with pytest.raises(KeyboardInterrupt):
        run.go(vision=where != "reactions")
    assert (run.proj / "highlights.json").read_bytes() == old
    assert "highlights.json" not in run.saves
    last = progress_lines(capsys)[-1]
    assert last["state"] == "failed" and last["detail"] == "cancelled"
    assert last["step"] == ("reactions" if where == "reactions" else "vision")


# ----------------------------------------------------------------- transcript.txt
SPEECH = [
    {"start": 1.0, "end": 2.5, "text": "Here we go."},
    {"start": 4.0, "end": 5.0, "text": "again"},
]


def test_transcript_txt_is_written_at_the_end_and_the_layers_read_it_in_memory(run):
    run.segments = SPEECH
    run.plan = plan(reactions=("audio",))
    run.go()
    want = f"# {run.video}\n[a.mp4 00:00:01.0-00:00:02.5] Here we go.\n"
    want += "[a.mp4 00:00:04.0-00:00:05.0] again\n"
    assert (run.proj / "transcript.txt").read_text(encoding="utf-8") == want
    (seen,) = run.transcripts  # before transcript.txt exists
    assert [t["text"] for t in seen] == ["Here we go.", "again"]
    assert seen == common.load_transcript_lines(run.proj)  # same as from the file after
    assert [p.name for p in run.proj.iterdir() if p.name.startswith("transcript.txt.")] == []


def test_without_layers_the_transcript_is_still_written(run):
    run.go()
    assert (run.proj / "transcript.txt").read_text(encoding="utf-8") == (
        f"# {run.video}\n(no speech)\n"
    )


def test_pending_transcript_is_parsed_only_inside_the_block(tmp_path):
    write(tmp_path / "transcript.txt", "[a.mp4 00:00:01.0-00:00:02.0] file\n")
    text = "# a.mp4\n[a.mp4 00:00:03.0-00:00:04.0] memory\n"
    with common.pending_highlights(tmp_path, {"sources": []}, text):
        assert [t["text"] for t in common.load_transcript_lines(tmp_path)] == ["memory"]
    assert [t["text"] for t in common.load_transcript_lines(tmp_path)] == ["file"]
    with common.pending_highlights(tmp_path, {"sources": []}):  # no text: the file
        assert [t["text"] for t in common.load_transcript_lines(tmp_path)] == ["file"]


# ----------------------------------------------------------------- held result files (#408)
ALL_LAYERS = {"reactions": ("audio",), "keyframes": ("visual",), "crashes": ("crash",)}
LAUGH = {"source": "", "kind": "audio", "type": "audio", "label": "Laughter",
         "confidence": 1.0, "start": 12.0, "end": 14.0}  # fmt: skip
OLD_FILES = {
    "events.json": b'{"events": [], "old": true}\r\n',
    "events.txt": b"# old events\r\n",
    "crashes.json": b'{"crashes": [], "old": true}\n',
    "crash_check/c001_old_5.jpg": b"old sheet",
    "activity.json": b'{"name": "old"}',
    "telemetry.json": b'{"old": true}',
    "action_cam_features/old-0123abcd.npz": b"old features",
}


def tree(folder: Path) -> dict[str, bytes]:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def test_hold_outputs_keeps_writes_and_deletions_in_memory_until_commit(tmp_path):
    keep = write(tmp_path / "a.json", '{"v": "old"}')
    gone = write(tmp_path / "sheets" / "old.jpg", b"jpg")
    other = tmp_path / "other.json"
    with cache.hold_outputs(tmp_path, ["a.json", "new.txt", "sheets", "sub"]) as held:
        cache.save_json(keep, {"v": "new"})
        cache.save_bytes(tmp_path / "new.txt", b"text\r\n")
        cache.save_bytes(tmp_path / "sub" / "x.bin", b"x", parents=True)
        cache.remove(gone)
        cache.save_json(other, {"v": 1})  # not held: written at once
        assert keep.read_text(encoding="utf-8") == '{"v": "old"}' and gone.is_file()
        assert not (tmp_path / "new.txt").exists() and not (tmp_path / "sub").exists()
        assert cache.load_json(keep) == {"v": "new"} == cache.read_json(keep)
        assert cache.load_json(gone) is None
        with pytest.raises(FileNotFoundError):
            cache.read_json(gone)
        assert cache.load_json(other) == {"v": 1}
        held.commit()
        direct = tmp_path / "direct.json"
        cache.save_json(direct, {"v": "new"})
        assert keep.read_bytes() == direct.read_bytes()  # what save_json writes itself
        cache.save_json(keep, {"v": "after"})  # committed: no longer held
    assert json.loads(keep.read_text(encoding="utf-8")) == {"v": "after"}
    assert (tmp_path / "new.txt").read_bytes() == b"text\r\n"
    assert (tmp_path / "sub" / "x.bin").read_bytes() == b"x" and not gone.exists()
    with cache.hold_outputs(tmp_path, ["a.json"]):  # left without a commit: dropped
        cache.save_json(keep, {"v": "dropped"})
    assert json.loads(keep.read_text(encoding="utf-8")) == {"v": "after"}
    assert cache.load_json(keep) == {"v": "after"}


@pytest.mark.parametrize("fail", ["stage", "rename", "cancel+rename"])
def test_a_failed_commit_leaves_the_project_untouched(tmp_path, monkeypatch, fail):
    """F1: every file is staged before anything is deleted or renamed; a failure while
    staging (the second file) leaves the old tree byte-identical, with no temp file or new
    folder. A failed rename removes the temp files that were not renamed; with a Ctrl+C
    before it, the KeyboardInterrupt follows the error. The signal handlers are put back."""
    write(tmp_path / "a.json", b'{"v": "old"}')
    write(tmp_path / "sheets" / "old.jpg", b"jpg")
    old = tree(tmp_path)
    calls = []
    real_mode, real_replace = cache.match_mode, cache.os.replace

    def mode(tmp, target):
        calls.append(target.name)
        if fail == "stage" and len(calls) == 2:
            raise OSError("disk full")
        real_mode(tmp, target)

    def replace(src, dst):
        if Path(dst).name == "x.bin":
            if fail == "cancel+rename":
                signal.raise_signal(signal.SIGINT)
            raise OSError("locked")
        real_replace(src, dst)

    monkeypatch.setattr(cache, "match_mode", mode)
    if fail != "stage":
        monkeypatch.setattr(cache.os, "replace", replace)
    before = {s: signal.getsignal(s) for s in cache._INTERRUPTS}
    with cache.hold_outputs(tmp_path, ["a.json", "sheets", "sub"]) as held:
        cache.save_json(tmp_path / "a.json", {"v": "new"})
        cache.save_bytes(tmp_path / "sub" / "deep" / "x.bin", b"x")
        cache.remove(tmp_path / "sheets" / "old.jpg")
        if fail == "cancel+rename":
            with pytest.raises(KeyboardInterrupt) as got:
                held.commit()
            assert isinstance(got.value.__context__, OSError)
            assert str(got.value.__context__) == "locked"
        else:
            with pytest.raises(OSError, match="disk full" if fail == "stage" else "locked"):
                held.commit()
    assert {s: signal.getsignal(s) for s in cache._INTERRUPTS} == before
    leftovers = [p.name for p in tmp_path.rglob("*.tmp")]
    assert leftovers == []
    if fail == "stage":
        assert calls == ["a.json", "x.bin"]
        assert tree(tmp_path) == old and not (tmp_path / "sub").exists()
    else:  # deletions and the renames before the failed one are made
        assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8")) == {"v": "new"}


@pytest.mark.parametrize("sig", ["SIGINT", "SIGBREAK", "SIGTERM"])
def test_a_cancel_during_the_commit_renames_every_file_then_interrupts(tmp_path, monkeypatch, sig):
    """Ctrl+C (or the CLI's Ctrl+Break, or its SIGTERM on POSIX) arriving between two renames
    waits for the last one: the project gets the whole new set, then KeyboardInterrupt, and
    the handlers are put back."""
    if sig not in {"SIGINT", "SIGBREAK" if os.name == "nt" else "SIGTERM"}:
        pytest.skip(f"the CLI does not stop on {sig} here")
    write(tmp_path / "a.json", b'{"v": "old"}')
    write(tmp_path / "b.json", b'{"v": "old"}')
    write(tmp_path / "sheets" / "old.jpg", b"jpg")
    signum, real_replace = getattr(signal, sig), cache.os.replace
    renamed = []

    def replace(src, dst):
        if not renamed:  # a cancel after the deletion, before a rename
            signal.raise_signal(signum)
        real_replace(src, dst)
        renamed.append(Path(dst).name)

    monkeypatch.setattr(cache.os, "replace", replace)
    with contextlib.ExitStack() as stack:
        if sig != "SIGINT":
            stack.enter_context(cli._stop_signal_as_interrupt())
        before = {s: signal.getsignal(s) for s in cache._INTERRUPTS}
        with cache.hold_outputs(tmp_path, ["a.json", "b.json", "sheets"]) as held:
            cache.save_json(tmp_path / "a.json", {"v": "new"})
            cache.save_json(tmp_path / "b.json", {"v": "new"})
            cache.remove(tmp_path / "sheets" / "old.jpg")
            with pytest.raises(KeyboardInterrupt):
                held.commit()
        assert {s: signal.getsignal(s) for s in cache._INTERRUPTS} == before
    assert renamed == ["a.json", "b.json"]
    assert sorted(tree(tmp_path)) == ["a.json", "b.json"]
    for name in renamed:
        assert json.loads((tmp_path / name).read_text(encoding="utf-8")) == {"v": "new"}


def test_a_commit_off_the_main_thread_defers_nothing(tmp_path, monkeypatch):
    """signal.signal works only on the main thread: elsewhere the commit runs as it is."""
    before, seen, errors = signal.getsignal(signal.SIGINT), [], []
    real_replace = cache.os.replace

    def replace(src, dst):
        seen.append(signal.getsignal(signal.SIGINT))
        real_replace(src, dst)

    def commit():
        try:
            held.commit()
        except BaseException as err:  # noqa: BLE001 - reported to the test below
            errors.append(err)

    monkeypatch.setattr(cache.os, "replace", replace)
    with cache.hold_outputs(tmp_path, ["a.json"]) as held:
        cache.save_json(tmp_path / "a.json", {"v": "new"})
        thread = threading.Thread(target=commit)
        thread.start()
        thread.join()
    assert errors == [] and seen == [before]
    assert json.loads((tmp_path / "a.json").read_text(encoding="utf-8")) == {"v": "new"}


def test_result_files_are_written_at_the_end_and_read_from_memory_meanwhile(run, monkeypatch):
    write(run.proj / "crash_check" / "c001_old_5.jpg", b"old sheet")
    run.segments = SPEECH
    run.events = [{**LAUGH, "source": str(run.video)}]
    run.plan = plan(**ALL_LAYERS)
    seen = []
    real = analyze.rescore

    def rescore(hl, ev, project, parts=None):
        seen.append((ev["events"], (project / "events.json").exists()))
        return real(hl, ev, project, parts)

    monkeypatch.setattr(analyze, "rescore", rescore)
    run.go("action")
    assert seen == [(run.events, False)]  # this run's events, before events.json exists
    (cand,) = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))["candidates"]
    assert cand["score"] > cand["score_pre_events"] == 50.0
    got = tree(run.proj)
    assert sorted(got) == [
        "activity.json", "crash_check/c001_a_10.jpg", "crashes.json", "events.json",
        "events.txt", "highlights.json", "transcript.txt",
    ]  # fmt: skip
    assert json.loads(got["events.json"])["events"] == run.events
    assert json.loads(got["crashes.json"])["crashes"][0]["t0"] == 10.0
    assert json.loads(got["activity.json"])["name"] == "puddle-hop"
    assert got["crash_check/c001_a_10.jpg"][:2] == b"\xff\xd8"  # a JPEG
    # events.txt as write_text gave it: the transcript and the event, text mode's endings
    want = [
        "# events.txt: transcript lines interleaved with detected events.",
        "# AUDIO = PANNs sound events (confidence, duration); OCR = on-screen HUD text;",
        "# VLM WINDOW = Vision model summary of a window (hl = highlight 0-10);",
        "# VLM = moment inside a window. Times are source time.",
        "# DIALOGUE = talk a text model marked (kind, strength, duration).",
        "",
        f"# {run.video}",
        "[a.mp4 00:00:01.0-00:00:02.5] Here we go.",
        "[a.mp4 00:00:04.0-00:00:05.0] again",
        "[a.mp4 00:00:12.0] AUDIO: Laughter (1.00, 2.0s)",
        "",
    ]
    assert got["events.txt"] == os.linesep.join(want).encode("utf-8")


def test_the_standalone_writers_still_write_at_once(tmp_path, monkeypatch):
    """`cutpoint events` and `crashes` run no hold: their files are on disk at once, the
    sheet with the bytes cv2.imwrite gave."""
    jpath, tpath = report.write_outputs(tmp_path, ["a.mp4"], [], [], [], {}, {}, {})
    assert json.loads(jpath.read_text(encoding="utf-8"))["events"] == []
    assert tpath.read_bytes().startswith(b"# events.txt")
    images = []
    frames = (np.array([1.5]), np.full((1, 24, 32, 3), 90, np.uint8))
    monkeypatch.setattr(crash, "decode_window", lambda *a, **k: frames)
    c = {"source": "a.mp4", "t0": 1.0, "t1": 2.0, "caption": "c001"}
    sheet = crash.write_sheet(c, tmp_path / "crash_check" / "c.jpg")
    assert sheet is not None and sheet.is_file()
    real_encode = cv2.imencode

    def encode(ext, img, opts):
        images.append(img)
        return real_encode(ext, img, opts)

    monkeypatch.setattr(cv2, "imencode", encode)
    crash.write_sheet(c, tmp_path / "again.jpg")
    plain = tmp_path / "plain.jpg"
    cv2.imwrite(str(plain), images[0], [cv2.IMWRITE_JPEG_QUALITY, 82])
    assert sheet.read_bytes() == plain.read_bytes() == (tmp_path / "again.jpg").read_bytes()


@pytest.mark.parametrize(
    ("where", "earlier"),
    [
        ("audio", True),
        ("visual", False),
        ("crash", True),
        ("write", False),
        ("ranking", True),
        ("vision", False),
    ],
)
def test_a_cancel_at_any_step_leaves_the_result_files_as_they_were(
    run, monkeypatch, where, earlier
):
    """A cancel in reactions, keyframes, crashes, the events write, ranking or vision: the
    earlier result files stay byte-identical and a first run leaves no project folder."""
    old = {**OLD_FILES, "highlights.json": b'{"style": "old", "candidates": []}'}
    if earlier:
        for name, data in old.items():
            write(run.proj / name, data)
        run.old = old["highlights.json"]
    run.events = [{**LAUGH, "source": str(run.video)}]
    run.plan = plan(**ALL_LAYERS)
    run.fail, run.cancel = where, True
    if where == "ranking":

        def rescore(*a):
            raise KeyboardInterrupt

        monkeypatch.setattr(analyze, "rescore", rescore)
    with pytest.raises(KeyboardInterrupt):
        run.go("action", vision=where == "vision")
    if earlier:
        assert tree(run.proj) == old
    else:
        assert not run.proj.exists()


def test_a_cancel_during_the_final_commit_leaves_every_result_file_new(run, monkeypatch):
    """Ctrl+C between the renames of the final commit: transcript.txt and highlights.json
    are new too, with events.json and the rest, then the run is cancelled."""
    old = {
        **OLD_FILES,
        "highlights.json": b'{"style": "old", "candidates": []}',
        "highlights.pre-events.json": b"{}",
        "transcript.txt": b"[a.mp4 00:00:00.0-00:00:01.0] old\r\n",
    }
    for name, data in old.items():
        write(run.proj / name, data)
    run.old = old["highlights.json"]
    run.segments = SPEECH
    run.events = [{**LAUGH, "source": str(run.video)}]
    run.plan = plan(**ALL_LAYERS)
    real_replace, renamed = cache.os.replace, []

    def replace(src, dst):
        if not renamed:
            signal.raise_signal(signal.SIGINT)
        real_replace(src, dst)
        renamed.append(Path(dst).name)

    monkeypatch.setattr(cache.os, "replace", replace)
    with pytest.raises(KeyboardInterrupt):
        run.go("action")
    assert renamed[-1] == "highlights.json"
    got = tree(run.proj)
    assert sorted(got) == [
        "action_cam_features/old-0123abcd.npz", "activity.json", "crash_check/c001_a_10.jpg",
        "crashes.json", "events.json", "events.txt", "highlights.json", "telemetry.json",
        "transcript.txt",
    ]  # fmt: skip
    for name in ["activity.json", "crashes.json", "events.json", "events.txt"]:
        assert got[name] != old[name]
    assert json.loads(got["highlights.json"])["candidates"][0]["id"] == "h001"
    assert b"Here we go." in got["transcript.txt"]


@pytest.mark.parametrize(("crashes", "want"), [(True, "puddle-hop"), (False, "generic")])
def test_action_cam_candidates_are_picked_with_the_projects_activity_prompts(
    run, monkeypatch, crashes, want
):
    """F3: the ranking step scores the beauty / people features (``actioncam.layer_features``,
    ``frame_scores``) with the captions of the activity.json the crash step wrote."""
    monkeypatch.setattr(actioncam, "prepare", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(analyze, "probe", lambda path: SimpleNamespace(duration=60.0))
    seen = []

    def pick(r, style, max_candidates, timings, ctx):
        seen.append(activity.active_prompts().name)
        return r.candidates

    monkeypatch.setattr(analyze, "pick_candidates", pick)
    run.plan = plan(crashes=("crash",)) if crashes else plan()
    run.go("action-cam", layers=crashes)
    assert seen == [want]
    assert activity.active_prompts() == activity.default_prompts()


@pytest.mark.parametrize("cancel", [False, True])
def test_ski_features_and_telemetry_json_are_written_at_the_end(run, monkeypatch, cancel):
    """The stale features go and the new ones and telemetry.json come only when the run
    completes; the second actioncam.prepare reads this run's crashes.json from memory."""
    for name in ("telemetry.json", "action_cam_features/old-0123abcd.npz"):
        write(run.proj / name, OLD_FILES[name])
    monkeypatch.setattr(analyze, "probe", lambda path: SimpleNamespace(duration=60.0))
    monkeypatch.setattr(actioncam, "extract", lambda src, duration: None)  # no telemetry
    seen = []

    def pick(r, style, max_candidates, timings, ctx):
        seen.append((len(ctx.crashes), (run.proj / "crashes.json").exists()))
        grid = np.zeros(5, np.float32)
        actioncam.save_features(ctx.project, r.source, 0.0, 60.0, {"speed": grid}, grid, grid, [])
        assert not actioncam.features_path(run.proj, r.source).exists()
        return r.candidates

    monkeypatch.setattr(analyze, "pick_candidates", pick)
    run.plan = plan(crashes=("crash",))
    run.fail, run.cancel = ("vision" if cancel else None), cancel
    with pytest.raises(KeyboardInterrupt) if cancel else contextlib.nullcontext():
        run.go("action-cam", vision=cancel)
    assert seen == [(1, False)]
    got = tree(run.proj)
    if cancel:
        assert got == {
            k: OLD_FILES[k] for k in ("telemetry.json", "action_cam_features/old-0123abcd.npz")
        }
        return
    new = actioncam.features_path(run.proj, run.video)
    assert "action_cam_features/old-0123abcd.npz" not in got
    assert actioncam.load_features(new)["features"]["speed"].tolist() == [0.0] * 5
    assert json.loads(got["telemetry.json"])["sources"] == {str(run.video): None}


# ----------------------------------------------------------------- background OCR (#375)
class FakePrefetch:
    """``_ocr_prefetch``'s reader: records offers, the reader ``run_events`` gets, and
    whether the project folder still existed when it was closed."""

    def __init__(self, proj: Path) -> None:
        self.proj = proj
        self.offers: list[tuple[str, float, bool]] = []
        self.closed_with_folder: bool | None = None
        self.closed_early: bool | None = None
        self.probe = lambda: None
        self.mode = None  # CUTPOINT_ANALYZE_SHARED_DECODE off
        self.scenes: list[object] = []
        self.waits: dict[str, float] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.closed_with_folder is None:  # the first close
            self.closed_with_folder = self.proj.is_dir()
            self.closed_early = self.probe()

    def offer(self, src, duration, ok, scene=None):
        self.offers.append((Path(src).name, duration, ok))
        self.scenes.append(scene)


@pytest.fixture
def prefetched(run, monkeypatch):
    """``run`` with a fake OCR prefetch; ``run.read_error`` / ``run.device`` shape the
    transcription the fake ``analyze_source`` reports."""
    pre = FakePrefetch(run.proj)
    # closed after the layers ran, before ranking ends with highlights.json
    pre.probe = lambda: bool(run.events_calls) and "highlights.json" not in run.saves
    run.pre, run.read_error, run.device = pre, None, "cuda"
    monkeypatch.setattr(analyze, "_ocr_prefetch", lambda *a: pre)
    inner = analyze.analyze_source

    def analyze_source(src, style, tr, *a, read_checked=None, **kw):
        tr.device = run.device
        res = inner(src, style, tr, *a, **kw)
        res.read_error = run.read_error
        if read_checked is not None:
            read_checked(res.info, run.read_error)
        return res

    monkeypatch.setattr(analyze, "analyze_source", analyze_source)
    return run


@pytest.mark.parametrize(
    ("device", "read_error", "ok"),
    [("cuda", None, True), ("cpu", None, False), ("cuda", {"failure_time": 30.0}, False)],
)
def test_sources_are_offered_to_the_ocr_prefetch(prefetched, device, read_error, ok):
    prefetched.plan = plan(reactions=("audio",), keyframes=("ocr",))
    prefetched.device, prefetched.read_error = device, read_error
    prefetched.go("gaming")
    assert prefetched.pre.offers == [("a.mp4", 60.0, ok)]
    (_, _, kw) = prefetched.events_calls[0]
    assert kw["prefetch"] is prefetched.pre
    assert prefetched.pre.closed_early  # stopped once the layers are done


@pytest.mark.parametrize(("device", "mode", "planned"), [("cuda", "split", True),
                         ("cpu", "split", False), ("cuda", None, False)])  # fmt: skip
def test_a_shared_decode_mode_offers_the_scene_plan(prefetched, monkeypatch, device, mode, planned):
    prefetched.plan = plan(reactions=("audio",), keyframes=("ocr",))
    prefetched.device, prefetched.pre.mode = device, mode
    monkeypatch.setattr(analyze, "_scene_plan", lambda src, duration: ("plan", src.name, duration))
    prefetched.go("gaming", use_scenes=True)
    assert prefetched.pre.scenes == [("plan", "a.mp4", 60.0) if planned else None]


def test_the_ocr_prefetch_stops_before_a_cancelled_run_cleans_up(prefetched):
    prefetched.plan = plan(reactions=("audio",), keyframes=("ocr",))
    prefetched.fail, prefetched.cancel = "audio", True
    with pytest.raises(KeyboardInterrupt):
        prefetched.go("gaming")
    assert prefetched.pre.closed_with_folder is True  # output_dir's cleanup comes after


# ----------------------------------------------------------------- --skip and analysis.skip
@pytest.mark.parametrize(
    ("cfg", "argv", "skip"),
    [
        (None, [], ()),
        (["crashes"], [], ("crashes",)),
        (["crashes", "crashes", "reactions"], [], ("crashes", "reactions")),  # de-duplicated
        (["crashes"], ["--skip", "keyframes"], ("keyframes",)),  # the option wins
    ],
)
def test_config_analysis_skip_is_the_default(home, tmp_path, monkeypatch, cfg, argv, skip):
    if cfg is not None:
        set_config(home, analysis={"skip": cfg})
    video = write(tmp_path / "a.mp4")
    seen: list = []
    monkeypatch.setattr(analyze, "analyze", lambda *a, **kw: seen.append(kw["skip"]))
    assert cli.main(["analyze", str(video), "--out", "p", *argv]) == 0
    assert seen == [skip]
    assert config.analysis_skip() == tuple(dict.fromkeys(cfg or ()))


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"analysis": []}, "analysis must be an object"),
        ({"analysis": {"skip": "crashes"}}, "analysis.skip must be a list"),
        ({"analysis": {"skip": ["ranking"]}}, "analysis.skip has 'ranking'"),
    ],
)
def test_config_analysis_skip_is_validated(home, data, message):
    set_config(home, **data)
    with pytest.raises(ValueError, match=message):
        config.get_config()


# ----------------------------------------------------------------- plan_layers
@pytest.fixture
def env(monkeypatch):
    """A machine with the events extra, every model file, CUDA and no telemetry."""
    state = SimpleNamespace(cuda=True, telemetry=False, missing=[], modules=("json",))
    monkeypatch.setattr(analyze, "EVENTS_MODULES", state.modules)
    monkeypatch.setattr(analyze, "_cuda", lambda: state.cuda)
    monkeypatch.setattr(analyze, "has_telemetry", lambda src: state.telemetry)
    monkeypatch.setattr(
        event_paths,
        "missing_files",
        lambda layers: [(c, Path(c)) for c, ls in state.missing if set(ls) & set(layers)],
    )
    return state


def planned(style: str, skip=()):
    p = analyze.plan_layers(get_style(style), [Path("a.mp4")], skip)
    return p.layers, p.skipped


@pytest.mark.parametrize(
    ("style", "layers", "skipped"),
    [
        ("funny", {"reactions": ("audio",), "keyframes": ("ocr",)}, {"crashes"}),
        ("gaming", {"reactions": ("audio",), "keyframes": ("ocr",)}, {"crashes"}),
        (
            "action",
            {"reactions": ("audio", "voice"), "keyframes": ("visual",), "crashes": ("crash",)},
            set(),
        ),
        ("commentary", {"reactions": ("audio",)}, {"keyframes", "crashes"}),
    ],
)
def test_plan_by_style(env, style, layers, skipped):
    got, why = planned(style)
    assert got == layers and set(why) == skipped
    if "crashes" in why:
        assert why["crashes"] == "not action-cam footage"


@pytest.mark.parametrize(
    ("style", "cat", "keyframes", "crashes"),
    [
        ("funny", category.GENERAL, ("visual",), True),  # funny on camera footage: no HUD OCR
        ("action", category.GAMEPLAY, ("ocr",), False),
        ("commentary", category.TALK, None, False),
    ],
)
def test_plan_follows_the_detected_category_not_the_style(env, style, cat, keyframes, crashes):
    p = analyze.plan_layers(get_style(style), [Path("a.mp4")], (), cat)
    assert p.layers.get("keyframes") == keyframes and ("crashes" in p.layers) == crashes
    assert p.game == (cat == category.GAMEPLAY)


@pytest.mark.parametrize(
    ("style", "off", "on"),
    [
        ("gaming", ("ocr",), ("ocr", "visual")),  # the index first, the HUD text after it
        ("funny", ("ocr",), ("ocr", "visual")),
        ("commentary", None, ("visual",)),
        ("action", ("visual",), ("visual",)),  # always has it: the option changes nothing
        ("sport", ("visual",), ("visual",)),
        ("action-cam", ("visual",), ("visual",)),
    ],
)
def test_plan_visual_index_is_opt_in_for_the_styles_without_it(env, style, off, on):
    for want, flag in ((off, False), (on, True)):
        p = analyze.plan_layers(get_style(style), [Path("a.mp4")], (), None, flag)
        assert p.layers.get("keyframes") == want
    assert (
        analyze.plan_layers(get_style(style), [Path("a.mp4")], ()).layers.get("keyframes") == off
    )  # the default is off


def test_plan_visual_index_follows_the_skip_option(env):
    p = analyze.plan_layers(get_style("gaming"), [Path("a.mp4")], ("keyframes",), None, True)
    assert "keyframes" not in p.layers and p.skipped["keyframes"] == analyze.SKIP_OPTION


def test_plan_visual_index_without_cuda_keeps_the_hud_text(env, caplog):
    env.cuda = False
    caplog.set_level("INFO")
    p = analyze.plan_layers(get_style("gaming"), [Path("a.mp4")], (), None, True)
    assert p.layers["keyframes"] == ("ocr",) and "keyframes" not in p.skipped
    assert "the visual index skipped" in caplog.text
    talk = analyze.plan_layers(get_style("commentary"), [Path("a.mp4")], (), None, True)
    assert "keyframes" not in talk.layers and talk.skipped["keyframes"].startswith("no CUDA GPU")


def test_plan_visual_index_without_clip_files_keeps_the_hud_text(env, caplog):
    env.missing = [("visual", ["visual"])]
    caplog.set_level("INFO")
    p = analyze.plan_layers(get_style("gaming"), [Path("a.mp4")], (), None, True)
    assert p.layers["keyframes"] == ("ocr",) and "keyframes" not in p.skipped
    assert "cutpoint setup --only visual" in caplog.text


def test_plan_visual_index_without_the_events_extra_is_skipped_like_the_rest(env, monkeypatch):
    monkeypatch.setattr(analyze, "EVENTS_MODULES", ("no_such_module_xyz",))
    p = analyze.plan_layers(get_style("commentary"), [Path("a.mp4")], (), None, True)
    assert not p.layers and p.skipped["keyframes"] == analyze.SKIP_NO_EXTRA


def test_analyze_hands_the_option_to_the_plan(run):
    run.go("gaming")
    run.go("gaming", visual_index=True)
    assert run.indexes == [False, True]


def test_plan_ski_is_action_footage(env):
    got, _ = planned("action-cam")
    assert got["crashes"] == ("crash",) and got["keyframes"] == ("visual",)


def test_plan_telemetry_turns_crashes_on_for_any_style(env):
    env.telemetry = True
    got, why = planned("funny")
    assert got["crashes"] == ("crash",) and got["reactions"] == ("audio", "voice")
    assert got["keyframes"] == ("ocr",) and why == {}


def test_plan_without_cuda_leaves_out_the_clip_keyframes(env, caplog):
    env.cuda = False
    caplog.set_level("INFO")
    got, why = planned("action")
    assert "keyframes" not in got and why["keyframes"].startswith("no CUDA GPU")
    assert got["crashes"] == ("crash",)
    assert "No CUDA GPU" in caplog.text
    assert planned("funny")[0]["keyframes"] == ("ocr",)  # HUD text needs no CUDA


def test_plan_cuda_check_uses_torch(env, monkeypatch):
    monkeypatch.undo()  # the real _cuda
    for on in (True, False):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda on=on: on))
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert analyze._cuda() is on


@pytest.mark.parametrize("broken", ["import", "call"])
def test_plan_a_broken_torch_counts_as_no_cuda(env, monkeypatch, caplog, broken):
    """F2: torch that fails to load (a missing DLL) or to answer leaves keyframes out
    instead of failing the analysis."""
    monkeypatch.setattr(analyze, "_cuda", REAL_CUDA)
    if broken == "import":
        monkeypatch.setitem(sys.modules, "torch", None)  # import raises ImportError
    else:

        def dll_error():
            raise OSError("[WinError 126] c10.dll could not be loaded")

        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
            is_available=dll_error)))  # fmt: skip
    caplog.set_level("DEBUG", logger=analyze.logger.name)
    got, why = planned("action")
    assert "keyframes" not in got and why["keyframes"] == analyze.SKIP_NO_CUDA
    assert got["crashes"] == ("crash",)
    assert "torch CUDA check failed" in caplog.text


def test_plan_skip_option_leaves_steps_out(env):
    got, why = planned("action", skip=("reactions", "crashes"))
    assert got == {"keyframes": ("visual",)}
    assert why == dict.fromkeys(("reactions", "crashes"), "left out by --skip or analysis.skip")
    # skipping crashes drops the voice layer
    assert planned("action", skip=("crashes",))[0]["reactions"] == ("audio",)


def test_plan_everything_skipped_is_an_empty_plan(env):
    got, why = planned("action", skip=analyze.SKIPPABLE)
    assert got == {} and set(why) == set(analyze.SKIPPABLE)


def test_plan_without_the_events_extra_skips_every_step(env, monkeypatch, caplog):
    monkeypatch.setattr(analyze, "EVENTS_MODULES", ("no_such_module_for_cutpoint_tests",))
    caplog.set_level("INFO")
    got, why = planned("action")
    assert got == {}
    assert why == dict.fromkeys(("reactions", "keyframes", "crashes"), "events extra not installed")
    assert "not available in Cutpoint lite (the events extra is not installed)" in caplog.text


def test_plan_without_the_extra_still_reports_the_steps_not_wanted(env, monkeypatch):
    monkeypatch.setattr(analyze, "EVENTS_MODULES", ("no_such_module_for_cutpoint_tests",))
    _got, why = planned("commentary")
    assert why["keyframes"].startswith("no keyframe layer")
    assert why["reactions"] == "events extra not installed"


def test_plan_missing_voice_files_keep_the_other_reaction_layers(env):
    env.missing = [("voice", ("voice",))]
    got, why = planned("action")
    assert got["reactions"] == ("audio",) and "reactions" not in why


def test_plan_missing_model_files_skip_that_step_only(env, caplog):
    env.missing = [("panns", ("ocr",))]
    caplog.set_level("INFO")
    got, why = planned("funny")
    assert got == {"reactions": ("audio",)}
    assert why["keyframes"].startswith("model files missing (cutpoint setup --only panns")


# ----------------------------------------------------------------- has_telemetry
@pytest.fixture
def probe_env(monkeypatch):
    state = SimpleNamespace(imu=None, streams=[], error=None)
    monkeypatch.setattr(insta360, "imu_source", lambda p: state.imu)

    def fake_probe(path, **kw):
        if state.error:
            raise state.error
        return {"streams": [{"codec_tag_string": t} for t in state.streams]}

    monkeypatch.setattr(analyze, "ffprobe_json", fake_probe)
    return state


@pytest.mark.parametrize(
    ("imu", "streams", "error", "want"),
    [
        (Path("x_00_.insv"), [], None, True),  # Insta360 trailer: no probe needed
        (None, ["avc1", "mp4a", "gpmd"], None, True),  # GoPro GPMF track
        (None, ["avc1", "mp4a"], None, False),
        (None, [], FFmpegError("bad file"), False),
        (None, [], OSError("gone"), False),
    ],
)
def test_has_telemetry(probe_env, imu, streams, error, want):
    probe_env.imu, probe_env.streams, probe_env.error = imu, streams, error
    assert analyze.has_telemetry(Path("a.mp4")) is want


# ----------------------------------------------------------------- common / scoring helpers
def test_pending_highlights_is_visible_only_inside_the_block(tmp_path):
    write(tmp_path / "highlights.json", json.dumps({"sources": [], "from": "file"}))
    data = {"sources": [], "from": "memory"}
    with common.pending_highlights(tmp_path, data):
        assert common.load_highlights(tmp_path) is data
        assert common.load_highlights(tmp_path / "." / "") is data  # same folder, any spelling
    assert common.load_highlights(tmp_path)["from"] == "file"


def test_pending_highlights_is_removed_when_the_block_raises(tmp_path):
    write(tmp_path / "highlights.json", json.dumps({"sources": []}))
    with pytest.raises(RuntimeError), common.pending_highlights(tmp_path, {"sources": ["x"]}):
        raise RuntimeError
    assert common.load_highlights(tmp_path) == {"sources": []}


def test_rescore_in_memory_equals_rescore_highlights(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "_segments", lambda src, language=None: [])
    src = str(tmp_path / "a.mp4")
    cands = [
        {"id": "h001", "source": src, "start": 10.0, "end": 20.0, "score": 40.0, "signals": {}},
        {"id": "h002", "source": src, "start": 100.0, "end": 110.0, "score": 60.0, "signals": {}},
    ]
    hl = {"style": "funny", "settings": {"target_total": 15}, "candidates": cands,
          "suggested_top": 1}  # fmt: skip
    ev = {
        "created": "2026-01-01T00:00:00",
        "events": [
            {
                "source": src,
                "kind": "audio",
                "label": "Laughter",
                "confidence": 0.9,
                "start": 12.0,
                "end": 13.0,
            },
        ],
        "windows": [],
    }
    proj = tmp_path / "proj"
    write(proj / "highlights.json", json.dumps(hl))
    write(proj / "events.json", json.dumps(ev))
    changed_disk = scoring.rescore_highlights(proj)
    memory = json.loads(json.dumps(hl))
    changed_mem = scoring.rescore(memory, ev, proj)
    on_disk = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
    assert changed_mem == changed_disk == 1
    assert memory == on_disk
    boosted = next(c for c in memory["candidates"] if c["id"] == "h001")
    assert boosted["score"] > 40.0 and boosted["score_pre_events"] == 40.0


# ----------------------------------------------------------------- cached OCR and pct
@pytest.mark.parametrize(("layers", "ocr_updates"), [((), 0), (("ocr",), 2)])
def test_only_a_running_ocr_layer_reports_pct(tmp_path, monkeypatch, layers, ocr_updates):
    """Follow-up C: reading cached OCR events (``--skip keyframes`` on a game style) sends
    no pct to whatever step is running."""
    srcs = [SimpleNamespace(path=tmp_path / f"{n}.mp4", name=f"{n}.mp4") for n in "ab"]
    hl = {"sources": [{"path": str(s.path)} for s in srcs]}
    monkeypatch.setattr(pipeline, "load_highlights", lambda project: hl)
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "load_source", lambda path, dur: srcs["ab".index(path.stem)])
    monkeypatch.setattr(pipeline, "bad_ranges", lambda *a: {})
    monkeypatch.setattr(pipeline, "load_transcript_lines", lambda project: [])
    monkeypatch.setattr(pipeline, "detect_audio_events", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "OcrEngine", lambda: None)
    monkeypatch.setattr(pipeline, "load_preset", lambda *a: None)
    updates: list = []
    monkeypatch.setattr(progress, "update", lambda *a, **k: updates.append(a))
    calls = []

    def ocr_events(s, *a, **k):
        calls.append(s.name)
        if len(calls) == 2:
            raise _Stop  # both sources read: the rest of the run is not needed
        return [], {}

    monkeypatch.setattr(pipeline, "detect_ocr_events", ocr_events)
    with pytest.raises(_Stop):
        pipeline.run_events(tmp_path, layers=layers, cached_ocr=True, discover=False)
    assert calls == ["a.mp4", "b.mp4"]
    assert len(updates) == ocr_updates


# ----------------------------------------------------------------- --style auto (#513)
GAME_PROBE = {"handler": "VideoHandle", "frames": 16, "hud_static": 0.01, "static_share": 0.05,
              "noise_floor": 0.2, "shake": 0.0}  # fmt: skip
CAMERA_PROBE = {"make": "Apple", "frames": 16, "hud_static": 0.0, "static_share": 0.0,
                "noise_floor": 3.0, "shake": 4.0}  # fmt: skip


@pytest.fixture
def probed(run, monkeypatch):
    """``run`` with the footage probe faked: ``run.probe`` is the SourceProbe fields."""
    run.probe, run.probes = dict(GAME_PROBE), []

    def fake(src, vision=None):
        run.probes.append(src)
        return category.SourceProbe(str(src), 60.0, False, **run.probe)

    monkeypatch.setattr(category, "probe_source", fake)
    return run


def _hl(run) -> dict:
    return json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))


def test_style_auto_records_the_style_it_chose_and_why(probed, capsys):
    code, doc = run_json(capsys, ["analyze", str(probed.video), "--style", "auto", "--out", "s"])
    assert code == 0 and doc["warnings"] == []
    hl = json.loads(Path(doc["outputs"]["highlights"]).read_text(encoding="utf-8"))
    assert hl["style"] == doc["outputs"]["style"] == "gaming"
    assert hl["settings"]["style_requested"] == "auto"
    det = hl["detection"]
    assert det == doc["outputs"]["detection"]
    assert (det["category"], det["emphasis"], det["style"], det["fallback"]) == (
        "gameplay", "hype", "gaming", None,
    )  # fmt: skip
    assert [s["path"] for s in det["sources"]] == [str(probed.video)]
    assert probed.cats == ["gameplay"]


@pytest.mark.parametrize(
    ("score", "style", "named"),
    [
        # leans to gameplay (some HUD evidence): the best guess's style
        ((category.GAMEPLAY_MIN + category.GAMEPLAY_SURE) / 2, "gaming", "Gameplay"),
        # leans to nothing: the General style (#896)
        ((category.GENERAL_SURE + category.GAMEPLAY_MIN) / 2, "action", "General"),
    ],
)
def test_an_ambiguous_detection_follows_its_best_guess_and_warns(
    probed, capsys, monkeypatch, score, style, named
):
    monkeypatch.setattr(category, "gameplay_score", lambda p: (score, ["screen recorder tags"]))
    code, doc = run_json(capsys, ["analyze", str(probed.video), "--style", "auto", "--out", "s"])
    assert code == 0
    det = doc["outputs"]["detection"]
    assert det["ambiguous"] is True and det["fallback"] == style == doc["outputs"]["style"]
    [warning] = doc["warnings"]
    assert warning["code"] == "style_ambiguous" and warning["files"] == [str(probed.video)]
    assert f"using the {named} style ({style})" in warning["message"]
    assert probed.cats == [None]  # the fallback style plans its own layers


def test_talk_is_decided_after_transcription(probed, monkeypatch):
    """General footage long enough for talk waits for the transcript: mostly speech makes it
    commentary, and the transcript signals wait for that style."""
    monkeypatch.setattr(category, "TALK_MIN_S", 30.0)
    probed.probe = dict(CAMERA_PROBE)
    words = [{"start": float(t), "end": t + 0.9, "word": "so"} for t in range(50)]
    probed.segments = [{"start": 0.0, "end": 50.0, "text": "so", "words": words}]
    seen = []
    real = analyze.analyze_source
    monkeypatch.setattr(
        analyze, "analyze_source", lambda *a, **kw: seen.append(kw) or real(*a, **kw)
    )
    probed.go(analyze.AutoRequest("hype"))
    hl = _hl(probed)
    assert hl["style"] == "commentary" and hl["detection"]["category"] == "talk"
    assert [(kw["pick"], kw["text"]) for kw in seen] == [(False, False)]
    assert probed.cats == ["talk"]


def test_an_explicit_style_never_probes(run, capsys, monkeypatch):
    """No frame probe and no HUD text check (OCR) with an explicit style."""
    monkeypatch.setattr(category, "probe_source", lambda src, vision=None: pytest.fail("probed"))
    monkeypatch.setattr(category, "text_probe", lambda *a: pytest.fail("OCR check"))
    run.go("gaming")
    code, doc = run_json(capsys, ["analyze", str(run.video), "--style", "gaming", "--out", "s"])
    assert code == 0 and "detection" not in doc["outputs"]


def test_sources_of_two_categories_warn_in_the_json_envelope(probed, capsys, monkeypatch):
    other = write(probed.video.parent / "b.mp4")
    fields = {str(probed.video): GAME_PROBE, str(other): CAMERA_PROBE}
    monkeypatch.setattr(
        category,
        "probe_source",
        lambda src, vision=None: category.SourceProbe(str(src), 60.0, False, **fields[str(src)]),
    )
    code, doc = run_json(
        capsys, ["analyze", str(probed.video.parent), "--style", "auto", "--out", "s"]
    )
    assert code == 0
    [warning] = [w for w in doc["warnings"] if w["code"] == "style_sources_disagree"]
    assert len(warning["files"]) == 1 and warning["files"][0] in fields
    assert set(doc["outputs"]["detection"]["scores"].values()) >= {0.5}


def test_reanalyze_keeps_the_detection_and_never_probes_again(probed, monkeypatch):
    probed.go(analyze.AutoRequest("hype"))
    before = _hl(probed)
    monkeypatch.setattr(
        category, "probe_source", lambda src, vision=None: pytest.fail("probed again")
    )
    analyze.reanalyze(probed.proj)
    after = _hl(probed)
    assert after["style"] == "gaming" and after["detection"] == before["detection"]
    assert after["settings"]["style_requested"] == "auto"


# ----------------------------------------------------------------- per-source timings (#725)
@pytest.mark.parametrize("cached", [True, False])
def test_analyze_source_times_each_stage_and_flags_a_cached_source(tmp_path, monkeypatch, cached):
    src = write(tmp_path / "a.mp4")
    info = MediaInfo(src, 100.0, 100.0, 640, 360, Fraction(30), "yuv420p", "h264", 1, "aac",
                     16000, 1)  # fmt: skip
    cdir = tmp_path / "cache"
    if cached:
        write(cdir / "audio16k.wav")
        write(cdir / "transcript-m.json", "[]")
    monkeypatch.setattr(analyze, "probe", lambda path: info)
    monkeypatch.setattr(analyze.cache, "source_dir", lambda s, off=0.0: cdir)
    monkeypatch.setattr(analyze, "extract_audio", lambda *a, **k: write(a[1]))
    monkeypatch.setattr(analyze, "check_read", lambda *a: None)
    monkeypatch.setattr(analyze, "_loudness", lambda wav, cdir: (np.zeros(1), np.zeros(1)))
    monkeypatch.setattr(analyze, "scene_cuts", lambda *a, **k: [])

    class Tr:
        model_name = "m"
        language = None

        def transcribe(self, wav, on_progress=None):
            return []

    timings: dict = {}
    res = analyze.analyze_source(src, get_style("gaming"), Tr(), True, None, timings)
    assert set(res.stages) == {*analyze.SOURCE_STAGES, "cached"}
    assert res.stages["cached"] is cached
    assert all(res.stages[k] >= 0 for k in analyze.SOURCE_STAGES)
    assert {"audio_extract_s", "loudness_s", "transcribe_s", "scenes_s"} <= set(timings)


def test_source_timings_lists_each_part_with_the_ocr_wait_on_the_first(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    info = MediaInfo(a, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None)

    def result(path, offset=0.0, cached=False):
        stages = {"probe_s": 0.1, "scenes_s": 2.0, "cached": cached}
        return analyze.SourceResult(path, info, offset, 60.0, [], [], None, stages=stages)

    pre = SimpleNamespace(waits={str(a): 1.234})
    got = analyze._source_timings([result(a), result(a, 30.0), result(b, cached=True)], pre)
    assert [(g["source"], g.get("offset"), g["ocr_wait_s"], g["cached"]) for g in got] == [
        (str(a), None, 1.23, False),
        (str(a), 30.0, 0.0, False),
        (str(b), None, 0.0, True),
    ]
    assert all(g[k] >= 0 for g in got for k in (*analyze.SOURCE_STAGES, "ocr_wait_s"))
    assert got[0]["scenes_s"] == 2.0 and got[0]["loudness_s"] == 0.0
    assert analyze._source_timings([result(a)], None)[0]["ocr_wait_s"] == 0.0


def test_highlights_json_has_the_per_source_timings(prefetched):
    run = prefetched
    run.plan = plan(keyframes=("ocr",))
    run.pre.waits = {str(run.video): 0.5}  # keyed as the real OcrPrefetch.take() keys it
    run.go()
    hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
    (rec,) = hl["timings"]["sources"]
    assert rec["source"] == hl["sources"][0]["path"]
    assert rec["cached"] is False and rec["ocr_wait_s"] == 0.5
    assert {"total_s", "events_s"} <= set(hl["timings"])  # the sums stay


# ----------------------------------------------------------------- vision with speech
def _speech_run(run, monkeypatch, transcript, rating=8.0):
    """``run`` with a recording rater: every ``cached``/``rate`` call keeps its ``transcript``
    keyword; the one candidate (score 50) has ``transcript`` and the rater rates ``rating``."""
    calls: list[tuple[str, object]] = []

    def one(src, style, tr, use_scenes, max_cand, timings, **kw):
        info = MediaInfo(
            src, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None
        )
        cand = analyze.Candidate(
            src, 10.0, 20.0, 15.0, 1.0, {"loudness": 1.0}, score=50.0, transcript=transcript
        )
        return analyze.SourceResult(src, info, 0.0, 60.0, [cand], [], None)

    class Rater:
        cache_only = False
        hits = asked = missed = 0
        stats = dict.fromkeys(
            ("calls", "grab_s", "load_s", "prompt_s", "reply_s", "prompt_tokens", "reply_tokens"), 0
        )
        model = "m"
        provider_name = "ollama"
        consent_needed = None

        def __init__(self, **_):
            pass

        def available(self):
            return run.vision_ok

        def cached(self, src, start, end, base, activity=None, transcript=None, laughter=False):
            calls.append(("cached", transcript))
            return SimpleNamespace(score=rating, label="nice")

        def rate(self, src, start, end, base, activity=None, transcript=None, laughter=False):
            calls.append(("rate", transcript))
            return SimpleNamespace(score=rating, label="nice")

        def unload(self):
            pass

    monkeypatch.setattr(analyze, "analyze_source", one)
    monkeypatch.setattr(analyze, "VisionRater", Rater)
    return calls


def _only_candidate(run):
    hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
    (cand,) = hl["candidates"]
    return cand


def test_the_candidates_speech_goes_to_rate_and_to_both_cached_lookups(run, monkeypatch):
    said = "No way, that was close"
    calls = _speech_run(run, monkeypatch, said)
    run.go(vision=True, vision_top=1)  # online: the resume lookup, then the rating
    assert calls == [("cached", said), ("rate", said)]
    assert _only_candidate(run)["transcript"] == said

    calls.clear()
    run.vision_ok = False  # offline: the lookup before the step, and the one of the step
    run.go(vision=True, vision_top=1)
    assert [n for n, _ in calls].count("cached") >= 1  # the offline lookup has speech too
    assert [t for _, t in calls] == [said] * len(calls)


@pytest.mark.usefixtures("pinned_prior")
def test_a_rating_with_speech_is_centred_on_its_own_kind_not_lifted(run, monkeypatch):
    """#909: the speech prompt rates higher than the frames-only one, so each kind is centred
    on its own mean (here shrunk toward its prior, one rating of 5), not lowered by 1.9."""
    weight = get_style("funny").vision_weight
    assert weight > 0
    _speech_run(run, monkeypatch, "Wow, look at that", rating=5.9)
    run.go(vision=True, vision_top=1)
    cand = _only_candidate(run)
    centre = (5.9 + 4 * vision.RATING_PRIOR["speech"]) / 5  # 5.26
    assert centre == pytest.approx(5.26)
    assert cand["vision_rated"]["kind"] == "speech" and cand["vision_rated"]["centre"] == 5.26
    assert cand["score"] == analyze.vision_blend(50.0, 5 + 5.9 - centre, weight)  # 5.64 -> 51.92
    assert cand["score"] == 51.9
    assert cand["signals"]["vision"] == 5.9  # the raw rating

    # a frames-only rating the same distance above its prior ends at the same score
    _speech_run(run, monkeypatch, "", rating=4.0)
    run.go(vision=True, vision_top=1)
    other = _only_candidate(run)
    assert other["vision_rated"]["kind"] == "frames" and other["vision_rated"]["centre"] == 3.36
    assert other["score"] == cand["score"]


@pytest.mark.usefixtures("pinned_prior")
@pytest.mark.parametrize(
    ("style", "said", "rating", "kind", "centre", "centred"),
    [
        ("funny", "Wow", 1.0, "speech", 4.28, 1.72),  # a low rating stays low, never negative
        ("funny", "Wow", 8.0, "speech", 5.68, 7.32),
        ("funny", "   \n ", 8.0, "frames", 4.16, 8.84),  # no words: the frames-only prompt
        ("funny", "<<>>", 8.0, "frames", 4.16, 8.84),  # only brackets: no words
        ("funny", None, 8.0, "frames", 4.16, 8.84),
        ("action-cam", "Wow", 8.0, "frames", 4.16, 8.84),  # action-cam never sends speech
    ],
)
def test_the_prompt_kind_is_speech_only_when_there_are_words(
    style, said, rating, kind, centre, centred
):
    """One rating is centred on (r + 4 * prior) / 5 of its kind; the centre becomes 5."""
    weight = get_style(style).vision_weight
    cand = {
        "source": "a.mp4", "start": 1.0, "end": 5.0, "score": 50.0, "label": "x",
        "signals": {}, "transcript": said,
    }  # fmt: skip

    class Rater:
        cache_only = True
        hits = asked = missed = 0
        stats = dict.fromkeys(
            ("calls", "grab_s", "load_s", "prompt_s", "reply_s", "prompt_tokens", "reply_tokens"), 0
        )
        model = "m"
        provider_name = "ollama"

        def rate(self, *a, **k):
            return SimpleNamespace(score=rating, label="nice")

        def unload(self):
            pass

    out = analyze._rate_vision([cand], Rater(), get_style(style), None, 0, deadline=float("inf"))
    assert out.used is True and out.by_kind == {kind: 1}
    assert out.centres == {kind: pytest.approx(centre)}
    assert cand["vision_rated"]["kind"] == kind and cand["vision_rated"]["centre"] == centre
    assert cand["score"] == analyze.vision_blend(50.0, centred, weight)
    assert cand["signals"]["vision"] == rating


# ----------------------------------------------------------------- HUD OCR det skip (#807)
class _Ran(Exception):
    """run_events was reached."""


def _events_det_skip(project: Path, monkeypatch, *argv: str) -> bool:
    """The ``det_skip`` a ``cutpoint events <project> ...`` run gives its OCR layer."""
    import argparse  # noqa: PLC0415

    seen: list = []

    def run_events(project, **kw):
        seen.append(kw["ocr"].det_skip)
        raise _Ran

    monkeypatch.setattr(events_cli.pipeline, "run_events", run_events)
    parser = argparse.ArgumentParser()
    events_cli.add_arguments(parser)
    with pytest.raises(_Ran):
        events_cli.run_args(parser.parse_args([str(project), *argv]))
    return seen[0]


def test_events_reads_ocr_as_the_analysis_did(home, run, monkeypatch) -> None:
    run.plan = plan(reactions=("audio",), keyframes=("ocr",))
    run.go("gaming")
    hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
    assert "ocr_det_skip" not in hl["settings"]  # off: highlights.json as before
    assert _events_det_skip(run.proj, monkeypatch, "--layers", "vlm") is False
    # config on does not change it: the analysis read without the skip
    set_config(home, ocr_det_skip=True)
    assert _events_det_skip(run.proj, monkeypatch, "--layers", "vlm") is False
    set_config(home, ocr_det_skip=False)

    run.go("gaming", ocr_det_skip=True)
    hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
    assert hl["settings"]["ocr_det_skip"] is True
    # no flag, config off: the analysis's setting, so the cached OCR read is found
    assert _events_det_skip(run.proj, monkeypatch, "--layers", "vlm") is True
    assert _events_det_skip(run.proj, monkeypatch) is True
    assert _events_det_skip(run.proj, monkeypatch, "--no-ocr-det-skip") is False
    # a re-rank keeps it
    analyze.reanalyze(run.proj)
    hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
    assert hl["settings"]["ocr_det_skip"] is True


def test_prefetch_and_layers_get_the_same_ocr_params(run, monkeypatch) -> None:
    from cutpoint.events import ocr_prefetch  # noqa: PLC0415

    made: list = []

    def prefetch(project, sources, params, batch_size=0):
        made.append(params)
        return FakePrefetch(project)

    monkeypatch.setattr(analyze, "EVENTS_MODULES", ())
    monkeypatch.setattr(event_paths, "missing_files", lambda *a: [])
    monkeypatch.setattr(ocr_prefetch, "enabled", lambda: True)
    monkeypatch.setattr(ocr_prefetch, "OcrPrefetch", prefetch)
    run.plan = plan(reactions=("audio",), keyframes=("ocr",))
    run.go("gaming", ocr_det_skip=True)
    ((_project, _layers, kw),) = run.events_calls
    (pre,) = made
    assert pre.det_skip is True and kw["ocr"].det_skip is True
    assert pre == kw["ocr"]  # else the prefetch's read is thrown away


# ----------------------------------------------------------------- the vision band and its cap (#909)
class _Clock:
    """A fake clock: it moves only when the fake model is asked (a call takes 10 s)."""

    now = 0.0


def _twelve(src, style, tr, use_scenes, max_cand, timings, **kw):
    """Twelve 9 s moments 40 s apart, scored 90, 89, ... 79: the funny style (60 s target)
    takes the first six, so the cut is after the sixth (index 5) and k = 2."""
    info = MediaInfo(
        src, 600.0, 600.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None
    )
    cands = [
        analyze.Candidate(
            src, 40.0 * (i + 1), 40.0 * (i + 1) + 9.0, 40.0 * (i + 1) + 4.0, 1.0,
            {"loudness": 1.0}, score=90.0 - i,
        )
        for i in range(12)
    ]  # fmt: skip
    return analyze.SourceResult(src, info, 0.0, 600.0, cands, [], None)


def _start(rank: int) -> float:
    """Start second of the moment ranked ``rank`` (0 is the best) in :func:`_twelve`."""
    return 40.0 * (rank + 1)


@pytest.fixture
def band_run(run, monkeypatch):
    """``run`` over :func:`_twelve` with a fake model that rates every moment ``rating`` (10),
    or ``ratings[start]`` when given, takes 10 s of the fake clock per call and answers the
    start times in ``cached`` for free.
    ``deadline`` replaces the real deadline with a point on the fake clock (None: the real
    clock and budget). ``asked`` lists, in order, the starts the model was asked about."""
    state = SimpleNamespace(asked=[], cached=set(), rating=10.0, ratings={}, deadline=None)
    _Clock.now = 0.0

    class Rater:
        cache_only = False
        asked = missed = 0
        model = "m"
        provider_name = "ollama"
        consent_needed = None

        def __init__(self, **_):
            self.hits = 0
            self.stats = dict.fromkeys(
                ("calls", "grab_s", "load_s", "prompt_s", "reply_s", "prompt_tokens",
                 "reply_tokens"), 0
            )  # fmt: skip

        def available(self):
            return True

        def cached(self, src, start, *a, **k):
            if start in state.cached:
                return SimpleNamespace(score=state.ratings.get(start, state.rating), label="nice")
            return None

        def rate(self, src, start, *a, **k):
            if start in state.cached:
                self.hits += 1
            else:
                state.asked.append(start)
                self.stats["calls"] += 1
                _Clock.now += 10.0
            return SimpleNamespace(score=state.ratings.get(start, state.rating), label="nice")

        def unload(self):
            pass

    real = analyze._rate_vision

    def rate_vision(order, rater, style, activity, batch_size, *, deadline, clock=None):
        if state.deadline is None:  # the real clock and the real deadline
            return real(order, rater, style, activity, batch_size, deadline=deadline)
        return real(
            order, rater, style, activity, batch_size,
            deadline=state.deadline, clock=lambda: _Clock.now,
        )  # fmt: skip

    monkeypatch.setattr(analyze, "analyze_source", _twelve)
    monkeypatch.setattr(analyze, "VisionRater", Rater)
    monkeypatch.setattr(analyze, "_rate_vision", rate_vision)

    def load() -> dict:
        return json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))

    def go(**kw):
        run.go(vision=True, **kw)
        hl = load()
        return hl, hl["settings"]["vision"]

    state.run, state.go, state.load = run, go, load
    return state


def _rated_starts(hl) -> list[float]:
    return sorted(c["start"] for c in hl["candidates"] if c["signals"].get("vision") is not None)


def test_vision_rates_the_edit_and_the_band_below_its_cut_closest_first(band_run, caplog):
    with caplog.at_level(logging.INFO, logger="cutpoint.analyze"):
        hl, v = band_run.go(vision_time_share=1000.0)
    # the edit takes ranks 0-5 (6 x 9 s <= 60 s): cut 6, k = band_k(6) = 2, so ranks 0-7
    assert (v["cut"], v["band_k"], v["requested"], v["rated"]) == (6, 2, 8, 8)
    # the last pick, the first below, the pick before, the second below, then the other picks
    assert band_run.asked == [_start(r) for r in (5, 6, 4, 7, 3, 2, 1, 0)]
    assert _rated_starts(hl) == [_start(r) for r in range(8)]
    assert (
        "Vision: rating 8 moments (6 of the default edit, 2 below the cut at 6) of 12 candidates"
        in caplog.text
    )
    assert len(hl["candidates"]) == 12  # the rest stay as they were


def test_vision_top_caps_the_band_keeping_the_moments_closest_to_the_cut(band_run, caplog):
    with caplog.at_level(logging.INFO, logger="cutpoint.analyze"):
        _, v = band_run.go(vision_time_share=1000.0, vision_top=3)
    assert band_run.asked == [_start(5), _start(6), _start(4)]
    assert (v["requested"], v["rated"]) == (3, 3)
    assert (  # the picks actually rated
        "Vision: rating 3 moments (2 of the default edit, 1 below the cut at 6) of 12 candidates"
        in caplog.text
    )


def test_the_time_cap_leaves_the_far_moments_unrated(band_run, caplog):
    band_run.deadline = 15.0  # the fake clock: a call at 0 s, one at 10 s, then 20 s is past it
    with caplog.at_level(logging.INFO, logger="cutpoint.analyze"):
        hl, v = band_run.go(vision_time_share=0.3)
    assert band_run.asked == [_start(5), _start(6)]  # the two closest to the cut
    assert _rated_starts(hl) == [_start(5), _start(6)]
    assert (v["capped"], v["skipped_by_cap"], v["rated"], v["requested"]) == (True, 6, 2, 8)
    assert v["calls"] == 2 and v["share"] == 0.3
    lines = [m for m in caplog.messages if m.startswith("Vision: time cap reached")]
    assert len(lines) == 1  # once
    assert re.fullmatch(
        r"Vision: time cap reached after 2 calls \(\d+ s, budget \d+ s = 30% of \d+ s\); "
        r"6 moments near the cut not rated",
        lines[0],
    )


def test_a_cached_rating_is_still_taken_after_the_cap(band_run):
    band_run.deadline = 15.0
    band_run.cached = {_start(4)}  # third in line: after the cap, but free
    hl, v = band_run.go(vision_time_share=0.3)
    assert band_run.asked == [_start(5), _start(6)]  # the model is not asked for the rest
    assert _rated_starts(hl) == [_start(4), _start(5), _start(6)]
    assert (v["capped"], v["skipped_by_cap"], v["rated"]) == (True, 5, 3)
    assert v["cached"] == 1 and v["calls"] == 2


def test_a_share_of_0_still_makes_one_call(band_run):
    _, v = band_run.go(vision_time_share=0.0)  # the real clock: the deadline is the step's start
    assert band_run.asked == [_start(5)]
    assert (v["share"], v["budget_s"]) == (0.0, 0.0)
    assert (v["capped"], v["skipped_by_cap"], v["rated"]) == (True, 7, 1)


def test_a_rerank_takes_its_budget_from_the_first_analysis(band_run):
    _, v = band_run.go(vision_time_share=0.5, earlier_total_s=200.0)
    assert (v["budget_base"], v["budget_s"], v["share"]) == ("earlier_total_s", 100.0, 0.5)
    assert v["rerank"] is True and v["capped"] is False

    _, v = band_run.go(vision_time_share=0.5)
    assert v["budget_base"] == "pre_vision_s" and "rerank" not in v


def test_the_share_comes_from_config_json_when_not_given(band_run, home):
    set_config(home, vision_time_share=0.0)
    band_run.run.go(vision=True, vision_time_share=None)
    v = band_run.load()["settings"]["vision"]
    assert v["share"] == 0.0 and band_run.asked == [_start(5)]


def _twelve_long3(*args, **kw):
    """:func:`_twelve` with rank 3 39 s long: the edit (60 s) skips it for its length and takes
    ranks 0-2 and 4-6, so the cut is after rank 6 and rank 3 is not rated."""
    res = _twelve(*args, **kw)
    res.candidates[3].end = res.candidates[3].start + 39.0
    return res


def test_a_moment_the_edit_skipped_above_the_cut_is_no_neighbour(band_run, monkeypatch):
    """#909: rank 3, skipped by the edit for its length, is not rated and is marked
    ``vision_skipped``. The clamp does not count it: a top rating on the band below it (ranks
    0-2 rated 0) lifts those moments past its 87 and the low-rated ones above it fall under
    it. The unrated moment under the band (rank 9) still bounds every rated one."""
    monkeypatch.setattr(analyze, "analyze_source", _twelve_long3)
    band_run.ratings = {_start(r): 0.0 for r in (0, 1, 2)}  # so 10 is above the centre
    hl, v = band_run.go(vision_time_share=1000.0)
    assert (v["cut"], v["band_k"], v["requested"]) == (7, 2, 8)
    by_start = {c["start"]: c for c in hl["candidates"]}
    skipped = by_start[_start(3)]
    assert skipped["signals"]["vision"] is None and skipped["score"] == 87.0
    assert skipped["vision_skipped"] == [skipped["start"], skipped["end"]]
    assert [c["start"] for c in hl["candidates"] if "vision_skipped" in c] == [_start(3)]
    for rank in (4, 5, 6, 7, 8):
        c = by_start[_start(rank)]
        assert c["signals"]["vision"] == 10.0
        assert c["score"] == c["vision_rated"]["blended"] > skipped["score"]  # passed it
    for rank in (0, 1, 2):
        assert by_start[_start(rank)]["score"] < skipped["score"]  # fell under it
    below = by_start[_start(9)]  # unrated, under the band: still a neighbour
    assert below["signals"]["vision"] is None
    assert all(by_start[_start(r)]["score"] >= below["score"] for r in range(9) if r != 3)


def test_a_weak_pick_above_a_skipped_moment_can_leave_the_edit(band_run, monkeypatch):
    """#909: rank 0, a pick above rank 3 (skipped by the edit for its length), rated 0 while
    the rest get 10: the skipped moment no longer holds it, so it falls below the cut and out
    of the edit, held only by the first unrated moment under the band (rank 9)."""
    monkeypatch.setattr(analyze, "analyze_source", _twelve_long3)
    band_run.ratings = {_start(0): 0.0}
    hl, _ = band_run.go(vision_time_share=1000.0)
    by_start = {c["start"]: c for c in hl["candidates"]}
    weak, skipped, floor = by_start[_start(0)], by_start[_start(3)], by_start[_start(9)]
    assert weak["score"] < skipped["score"]
    assert weak["score"] >= floor["score"]
    picks = default_fit.fit_target(hl["candidates"], 60.0)
    assert _start(0) not in [c["start"] for c in picks]


def test_an_unrated_pick_is_still_a_neighbour(band_run, monkeypatch):
    """With ``--vision-top 3`` only ranks 6, 7 and 5 are rated: the picks left unrated (ranks
    0-2 and 4) bound them as before; only rank 3, which the edit skipped, does not count."""
    monkeypatch.setattr(analyze, "analyze_source", _twelve_long3)
    hl, _ = band_run.go(vision_time_share=1000.0, vision_top=3)
    assert band_run.asked == [_start(6), _start(7), _start(5)]
    by_start = {c["start"]: c for c in hl["candidates"]}
    order = [c["start"] for c in hl["candidates"]]
    pick = by_start[_start(4)]
    assert pick["signals"]["vision"] is None and "vision_skipped" not in pick
    for rank in (5, 6, 7):
        c = by_start[_start(rank)]
        assert c["vision_rated"]["blended"] > pick["score"]  # the blend alone would pass it
        assert c["score"] <= pick["score"]  # the clamp holds it under
        assert order.index(_start(rank)) > order.index(_start(4))
    assert "vision_skipped" in by_start[_start(3)]


def test_a_weak_pick_falls_below_the_cut(band_run):
    """#909: the whole edit is rated, so a pick the model rates low can drop out of it. Rank 5,
    the last pick but one, rated 0 while the rest get 10, falls under ranks 6 and 7 below the
    cut, held only by the first unrated moment under the band (rank 8)."""
    band_run.ratings = {_start(5): 0.0}
    hl, v = band_run.go(vision_time_share=1000.0)
    assert v["requested"] == 8
    order = [c["start"] for c in hl["candidates"]]
    by_start = {c["start"]: c for c in hl["candidates"]}
    weak, floor = by_start[_start(5)], by_start[_start(8)]
    assert floor["signals"]["vision"] is None
    assert order.index(_start(5)) > order.index(_start(7))  # past the two below the cut
    assert weak["score"] >= floor["score"]  # the unrated moment under the band bounds it
    picks = default_fit.fit_target(hl["candidates"], 60.0)
    assert _start(5) not in [c["start"] for c in picks]  # out of the edit
    assert _start(6) in [c["start"] for c in picks]
