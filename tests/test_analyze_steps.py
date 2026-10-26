"""``analyze`` steps: reactions, keyframes and crashes in a fixed order, the progress lines,
``--skip``, the layer plan and the single atomic write of highlights.json."""

from __future__ import annotations

import contextlib
import json
import sys
import time
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cutpoint.pano.x3 as x3
import cutpoint.ski as ski
from conftest import set_config, write
from cutpoint import analyze, cli, config, crash, jsonout, progress, transcribe
from cutpoint.events import common, pipeline, scoring
from cutpoint.events import paths as event_paths
from cutpoint.ffmpeg_utils import FFmpegError, MediaInfo
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
    with analyze._watch_extract(wav, seconds, got.append):
        for k in range(1, 5):  # 2.5 s, 5 s, 7.5 s, 10 s of audio written
            tmp.write_bytes(b"\0" * (analyze.WAV_HEADER + int(k * 2.5 * analyze.WAV_BYTES_PER_S)))
            deadline = time.monotonic() + 2.0
            while (not got or got[-1] < k / 4) and time.monotonic() < deadline:
                time.sleep(0.005)
    assert got == sorted(got) and got[-1] == 1.0
    assert {0.25, 0.5, 0.75, 1.0} <= set(got)


def test_watch_extract_is_a_no_op_without_a_report(tmp_path):
    with analyze._watch_extract(tmp_path / "a.wav", 10.0, None):
        pass  # no thread, nothing to stop


def test_whisper_progress_is_passed_only_on_a_transcript_cache_miss(tmp_path):
    calls = []

    def whisper(wav, on_progress=None):
        on_progress(1.0)
        return [{"start": 0, "end": 1}]

    tr = SimpleNamespace(model_name="m", transcribe=whisper)
    assert analyze._transcript(tmp_path / "a.wav", tmp_path, tr, calls.append)
    assert calls == [1.0]
    analyze._transcript(tmp_path / "a.wav", tmp_path, tr, calls.append)
    assert calls == [1.0]  # cached: Whisper does not run


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
    assert cand["signals"]["vision"] == 8.0 and cand["label"] == "nice"
    assert cand["score_pre_events"] == pytest.approx(analyze.vision_blend(50.0, 8.0, w))
    assert cand["score_pre_events"] == 59.0  # funny: vision_weight 0.3, 8/10 -> x1.18
    assert cand["score"] == pytest.approx(cand["score_pre_events"] + boost, abs=0.15)


@pytest.mark.parametrize(("layers", "prepared"), [(True, 2), (False, 1)])
def test_ski_features_are_prepared_again_after_the_layers(run, monkeypatch, layers, prepared):
    """F5b: ski candidates come from the feature grid; it is built again once this run's
    layers wrote events.json, and only once without layers."""
    calls = []
    monkeypatch.setattr(ski, "prepare", lambda *a, **k: calls.append(a) or SimpleNamespace())
    monkeypatch.setattr(analyze, "probe", lambda path: SimpleNamespace(duration=60.0))
    picked = []

    def pick(r, style, max_candidates, timings, ctx):
        picked.append(ctx)
        return r.candidates

    monkeypatch.setattr(analyze, "pick_candidates", pick)
    run.plan = plan(reactions=("audio", "voice"), crashes=("crash",))
    run.go("ski", layers=layers)
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
