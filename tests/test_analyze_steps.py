"""``analyze`` steps: reactions, keyframes and crashes in a fixed order, the progress lines,
``--skip``, the layer plan and the single atomic write of highlights.json."""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

import cutpoint.pano.x3 as x3
from conftest import set_config, write
from cutpoint import analyze, cli, config, jsonout, progress
from cutpoint.events import common, pipeline, scoring
from cutpoint.events import paths as event_paths
from cutpoint.ffmpeg_utils import FFmpegError, MediaInfo
from cutpoint.styles import get_style

ORDER = ("audio", "voice", "visual", "ocr", "crash")


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
        events_calls=[], layers_seen=[], saves=[], vision_ok=True, rated=0,
    )  # fmt: skip

    class StubTranscriber:
        model_name, device = "stub", "cpu"

        def __init__(self, **_):
            pass

        def close(self):
            pass

    def fake_analyze_source(src, style, tr, use_scenes, max_cand, timings, **kw):
        info = MediaInfo(
            src, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None
        )
        cand = analyze.Candidate(src, 10.0, 20.0, 15.0, 1.0, {"loudness": 1.0}, score=50.0)
        return analyze.SourceResult(src, info, 0.0, 60.0, [cand], [], None)

    def fake_run_events(project, *, layers, stage, **kw):
        state.events_calls.append((project, tuple(layers), kw))
        # the analysis in progress is visible, highlights.json is not written yet
        assert common.load_highlights(project)["candidates"]
        assert not (project / "highlights.json").exists() or state.old is not None
        for layer in [x for x in ORDER if x in layers]:
            with stage(layer):
                state.layers_seen.append(layer)
                if state.fail == layer:
                    raise KeyboardInterrupt if state.cancel else RuntimeError("model blew up")
        with stage("write"):
            write(project / "events.json", json.dumps({"events": [], "created": "t"}))
        return project / "events.json", project / "events.txt"

    class FakeRater:
        def __init__(self, **_):
            pass

        def available(self):
            return state.vision_ok

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
    monkeypatch.setattr(analyze, "Transcriber", StubTranscriber)
    monkeypatch.setattr(analyze, "analyze_source", fake_analyze_source)
    monkeypatch.setattr(analyze, "VisionRater", FakeRater)
    monkeypatch.setattr(analyze, "plan_layers", lambda style, sources, skip: state.plan)
    monkeypatch.setattr(pipeline, "run_events", fake_run_events)
    monkeypatch.setattr(analyze.cache, "save_json", save)

    def go(style="funny", **kw):
        kw = {"vision": False, "use_scenes": False, "layers": True} | kw
        return analyze.analyze(video, get_style(style), proj, **kw)

    state.go = go
    return state


def plan(**layers) -> analyze.LayerPlan:
    return analyze.LayerPlan(layers=layers)


# ----------------------------------------------------------------- the steps and their lines
def test_a_full_run_emits_every_step_in_order_and_ends_done(run, capsys):
    run.plan = plan(reactions=("audio", "voice"), keyframes=("visual",), crashes=("crash",))
    run.go("action", vision=True)
    got = progress_lines(capsys)
    assert {g["total"] for g in got} == {6}
    started = [g["step"] for g in got if g["state"] == "running"]
    assert list(dict.fromkeys(started)) == list(progress.ANALYZE_STEPS)
    assert {s: g["state"] for s, g in last_states(got).items()} == dict.fromkeys(
        progress.ANALYZE_STEPS, "done"
    )
    assert run.layers_seen == ["audio", "voice", "visual", "crash"]
    assert run.rated == 1


def test_events_run_once_with_all_planned_layers_before_highlights_exist(run):
    run.plan = plan(reactions=("audio",), keyframes=("ocr",))
    run.go()
    ((project, layers, kw),) = run.events_calls
    assert project == run.proj and layers == ("audio", "ocr")
    assert kw["cached_ocr"] is False and kw["llm"] is False
    assert (run.proj / "events.json").is_file()
    hl = json.loads((run.proj / "highlights.json").read_text(encoding="utf-8"))
    assert hl["candidates"][0]["id"] == "h001"


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


# ----------------------------------------------------------------- --skip and analysis.skip
@pytest.mark.parametrize(
    ("argv", "skip"),
    [
        ([], ()),
        (["--skip", "reactions"], ("reactions",)),
        (["--skip", "keyframes", "crashes"], ("keyframes", "crashes")),
    ],
)
def test_skip_option_reaches_analyze(home, tmp_path, monkeypatch, argv, skip):
    video = write(tmp_path / "a.mp4")
    seen: list = []
    monkeypatch.setattr(analyze, "analyze", lambda *a, **kw: seen.append(kw))
    assert cli.main(["analyze", str(video), "--out", "p", *argv]) == 0
    assert seen[0]["skip"] == skip and seen[0]["layers"] is True


@pytest.mark.parametrize("bad", ["transcribe", "ranking", "vision", "nope"])
def test_skip_rejects_steps_that_cannot_be_skipped(home, tmp_path, capsys, bad):
    video = write(tmp_path / "a.mp4")
    with pytest.raises(SystemExit) as err:
        cli.main(["analyze", str(video), "--skip", bad])
    assert err.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_skip_needs_at_least_one_step(home, tmp_path, capsys):
    video = write(tmp_path / "a.mp4")
    with pytest.raises(SystemExit) as err:
        cli.main(["analyze", str(video), "--skip"])
    assert err.value.code == 2


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
        ({"analysis": {"skip": ["reactions", 3]}}, "analysis.skip must be a list"),
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


def test_plan_ski_is_action_footage(env):
    got, _ = planned("ski")
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


def test_plan_cuda_is_not_needed_for_hud_text(env):
    env.cuda = False
    assert planned("funny")[0]["keyframes"] == ("ocr",)


def test_plan_cuda_check_uses_torch(env, monkeypatch):
    monkeypatch.undo()  # the real _cuda
    for on in (True, False):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda on=on: on))
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert analyze._cuda() is on


def test_plan_skip_option_leaves_steps_out(env):
    got, why = planned("action", skip=("reactions", "crashes"))
    assert got == {"keyframes": ("visual",)}
    assert why == dict.fromkeys(("reactions", "crashes"), "left out by --skip or analysis.skip")


def test_plan_skipping_crashes_drops_the_voice_layer(env):
    got, _ = planned("action", skip=("crashes",))
    assert got["reactions"] == ("audio",)


def test_plan_everything_skipped_is_an_empty_plan(env):
    got, why = planned("action", skip=analyze.SKIPPABLE)
    assert got == {} and set(why) == set(analyze.SKIPPABLE)


def test_plan_without_the_events_extra_skips_every_step(env, monkeypatch, caplog):
    monkeypatch.setattr(analyze, "EVENTS_MODULES", ("no_such_module_for_cutpoint_tests",))
    caplog.set_level("INFO")
    got, why = planned("action")
    assert got == {}
    assert why == dict.fromkeys(("reactions", "keyframes", "crashes"), "events extra not installed")
    assert "events extra is not installed" in caplog.text


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
    monkeypatch.setattr(x3, "imu_source", lambda p: state.imu)

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
    monkeypatch.setattr(scoring, "_segments", lambda src: [])
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
