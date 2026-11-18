"""Helpers shared by test modules, moved out of test_analyze_steps.py."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import HOP, StubTranscriber, add_profile, write
from cutpoint import activity, analyze, cache, crash, progress
from cutpoint.events import common, pipeline, report
from cutpoint.ffmpeg_utils import MediaInfo
from cutpoint.styles import get_style

ORDER = ("audio", "voice", "visual", "ocr", "crash")


def progress_lines(capsys) -> list[dict]:
    err = capsys.readouterr().err
    return [
        json.loads(x.removeprefix("PROGRESS "))
        for x in err.splitlines()
        if x.startswith("PROGRESS ")
    ]


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
        rated=0, segments=[], transcripts=[], events=[], cats=[], indexes=[], telemetry=True,
    )  # fmt: skip
    # camera telemetry in the source: action-cam runs stay action-cam (#939 falls back without)
    monkeypatch.setattr(analyze, "has_telemetry", lambda src, timeout=60: state.telemetry)

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


def last_states(lines: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for g in lines:
        out[g["step"]] = g
    return out
