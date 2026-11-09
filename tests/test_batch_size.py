"""``--batch-size N`` on analyze, events and 360 analyze (#109 phase 4): each model is closed
after N sources (the vision model after N VLM windows) and loaded again; the outputs are the
same as without it."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import requests

import cutpoint.analyze
from conftest import envelope, write
from cutpoint import cache, cli, config, memguard, progress, runstate, vision
from cutpoint.assistant import providers
from cutpoint.events import common, pipeline
from cutpoint.events.common import AudioPart, Source
from cutpoint.ffmpeg_utils import MediaInfo
from cutpoint.vision import VisionResult


def _read(project: Path) -> dict:
    return json.loads((project / runstate.FILE).read_text(encoding="utf-8"))


def _without(data: dict, *keys: str) -> dict:
    return {k: v for k, v in data.items() if k not in keys}


# -- helpers and the option -------------------------------------------------------------------


def test_batches_split_in_order_and_zero_is_one_batch():
    items = list("abcde")
    assert progress.batches(items, 0) == [items]
    assert progress.batches(items, 5) == [items]
    assert progress.batches(items, 2) == [["a", "b"], ["c", "d"], ["e"]]
    assert progress.batches([], 2) == [[]]


def test_merge_stats_gives_the_stats_of_one_run():
    a = {"cached": 2, "computed": 0, "clips": {}}
    b = {"cached": 0, "computed": 2, "clips": {"c1": 1}, "device": "cuda", "seconds": 1.5}
    assert pipeline.merge_stats(a, b) == {
        "cached": 2,
        "computed": 2,
        "clips": {"c1": 1},
        "device": "cuda",
        "seconds": 1.5,
    }
    assert list(pipeline.merge_stats(a, b)) == ["cached", "computed", "clips", "device", "seconds"]
    got = pipeline.merge_stats({"ok": 1, "seconds": [0.5]}, {"ok": 2, "seconds": [0.7]})
    assert got == {"ok": 3, "seconds": [0.5, 0.7]}


@pytest.mark.parametrize(
    "argv",
    [["analyze", "a.mp4"], ["events", "p"], ["360", "analyze", "p"], ["crashes", "p"]],
    ids=["analyze", "events", "360-analyze", "crashes"],
)
def test_batch_size_is_a_whole_number_from_zero(argv, capsys):
    parser = cli.build_parser()
    assert cli._parse(parser, argv).batch_size == 0
    assert cli._parse(parser, [*argv, "--batch-size", "0"]).batch_size == 0
    assert cli._parse(parser, [*argv, "--batch-size", "3"]).batch_size == 3
    for bad in ("-1", "x"):
        with pytest.raises(SystemExit) as exc:
            cli.main([*argv, "--batch-size", bad, "--json"])
        assert exc.value.code == 2
        doc = envelope(capsys.readouterr().out)
        assert doc["error"]["code"] == "bad_args" and "--batch-size" in doc["error"]["message"]


@pytest.mark.parametrize("command", ["clips", "extend", "edit"])
def test_no_batch_size_on_the_commands_without_models(command, capsys):
    argv = {"clips": ["clips", "p"], "extend": ["extend", "p/clips"], "edit": ["edit", "a.mp4"]}
    with pytest.raises(SystemExit) as exc:
        cli._parse(cli.build_parser(), [*argv[command], "--batch-size", "2"])
    assert exc.value.code == 2


# -- analyze: Whisper every N sources ---------------------------------------------------------


class _Whisper:
    """Fake Whisper: loaded on the first transcript after a close (counted)."""

    model_name, language, device = "fake", None, "cpu"
    loads = closes = 0
    calls: list[str] = []  # noqa: RUF012 - reset by the fixture
    stop: int | None = None  # KeyboardInterrupt on this transcribe call (1-based)

    def __init__(self, **_kw: Any) -> None:
        self.loaded = False

    def transcribe(self, wav: Path, on_progress=None) -> list[dict]:
        type(self).calls.append(wav.parent.name)
        if len(self.calls) == self.stop:
            raise KeyboardInterrupt
        if not self.loaded:
            type(self).loads += 1
            self.loaded = True
        n = len(wav.parent.name)
        return [{"start": 1.0, "end": 2.0 + n % 3, "text": f"hi {wav.parent.name}", "words": []}]

    def close(self) -> None:
        if self.loaded:
            type(self).closes += 1
        self.loaded = False


@pytest.fixture
def analyze_run(home, tmp_path, monkeypatch):
    """``analyze videos --out p`` over five sources with Whisper and the per-source analysis
    faked (scores differ per source); ``state.fresh()`` drops the sources' caches."""
    monkeypatch.chdir(tmp_path)
    videos = [write(tmp_path / "videos" / f"{n}.mp4", n * 3) for n in "abcde"]

    def analyze_source(src, style, tr, use_scenes, max_cand, timings, **kw):
        cdir = cache.source_dir(src)
        segments = cutpoint.analyze._transcript(cdir / "audio16k.wav", cdir, tr)
        kw["on_transcript"]()
        info = MediaInfo(
            src, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 1, None, None, None
        )
        score = 40.0 + 7 * videos.index(src)
        cand = cutpoint.analyze.Candidate(src, 10.0, 20.0, 15.0, 1.0, {"loudness": 1.0}, score)
        return cutpoint.analyze.SourceResult(src, info, 0.0, 60.0, [cand], segments, None)

    _Whisper.loads = _Whisper.closes = 0
    _Whisper.calls = []
    _Whisper.stop = None
    monkeypatch.setattr(cutpoint.analyze, "Transcriber", _Whisper)
    monkeypatch.setattr(cutpoint.analyze, "analyze_source", analyze_source)
    monkeypatch.setattr(cutpoint.analyze, "plan_layers", lambda *a: cutpoint.analyze.LayerPlan())

    def fresh() -> None:
        for v in videos:
            shutil.rmtree(cache.source_dir(v), ignore_errors=True)
        _Whisper.loads = _Whisper.closes = 0
        _Whisper.calls.clear()

    def outputs() -> tuple[dict, bytes]:
        hl = cache.load_json(project / "highlights.json")
        return _without(hl, "created", "timings"), (project / "transcript.txt").read_bytes()

    project = home / "projects" / "p"
    argv = ["analyze", "videos", "--style", "funny", "--no-vision", "--out", "p"]
    return SimpleNamespace(videos=videos, project=project, argv=argv, fresh=fresh, outputs=outputs)


def test_analyze_closes_whisper_every_n_sources_with_the_same_outputs(analyze_run, caplog):
    r = analyze_run
    assert cli.main(r.argv) == 0
    reference = r.outputs()
    assert _Whisper.loads == 1 and _Whisper.closes == 1
    for size, loads in (("0", 1), ("2", 3), ("5", 1)):
        r.fresh()
        caplog.clear()
        with caplog.at_level("INFO"):
            assert cli.main([*r.argv, "--batch-size", size]) == 0
        assert r.outputs() == reference
        assert _Whisper.loads == _Whisper.closes == loads == math.ceil(5 / (int(size) or 5))
        boundaries = [m for m in caplog.messages if m.startswith("transcribe: ")]
        assert boundaries == (
            ["transcribe: 2/5 done, continuing", "transcribe: 4/5 done, continuing"]
            if size == "2"
            else []
        )
    rec = _read(r.project)  # resume replays the option
    assert rec["argv"][-2:] == ["--batch-size", "5"] and rec["status"] == "done"


@pytest.mark.parametrize(("stop", "redone"), [(3, 3), (2, 4)], ids=["at-boundary", "inside"])
def test_analyze_interrupted_in_a_batch_resumes_to_the_same_outputs(analyze_run, stop, redone):
    """Stopped on the first source of the second batch, or inside the first one: resume
    transcribes only the rest and writes what an uninterrupted run does."""
    r = analyze_run
    assert cli.main(r.argv) == 0
    reference = r.outputs()
    for name in ("highlights.json", "transcript.txt"):
        (r.project / name).unlink()
    r.fresh()
    _Whisper.stop = stop
    argv = [*r.argv, "--batch-size", "2"]
    assert cli.main(argv) == 130
    assert _read(r.project)["status"] == "interrupted"
    assert _read(r.project)["steps"]["transcribe"]["state"] == "failed"
    _Whisper.stop = None
    _Whisper.calls.clear()
    assert cli.main(["resume", "p"]) == 0
    assert len(_Whisper.calls) == redone  # the transcripts made before are cached
    assert r.outputs() == reference
    assert _read(r.project)["status"] == "done" and _read(r.project)["argv"] == argv


def test_a_rerun_with_batch_size_continues_the_interrupted_record(analyze_run, capsys):
    """Review F5: re-running an interrupted analyze with --batch-size (to use less memory)
    is the same run: no resume_args_changed, the recorded sources kept; argv records it."""
    r = analyze_run
    _Whisper.stop = 2
    assert cli.main(r.argv) == 130
    before = _read(r.project)
    assert before["status"] == "interrupted"
    _Whisper.stop = None
    capsys.readouterr()
    argv = [*r.argv, "--batch-size", "2", "--json"]
    assert cli.main(argv) == 0
    doc = envelope(capsys.readouterr().out)
    assert "resume_args_changed" not in [w["code"] for w in doc["warnings"]]
    rec = _read(r.project)
    assert rec["options_key"] == before["options_key"] and rec["argv"] == argv
    assert rec["sources"] == before["sources"] == [str(v) for v in r.videos]
    assert rec["status"] == "done"


# -- W-F1: progress.json once the first transcript is there -----------------------------------


def test_a_first_analyze_stopped_in_scene_detection_can_be_resumed(home, tmp_path, monkeypatch):
    """Windows W-F1: a single-source first analyze stopped after Whisper, during the scene
    cuts, leaves progress.json (interrupted), and resume finishes it."""
    from cutpoint import analyze  # noqa: PLC0415

    monkeypatch.chdir(tmp_path)
    video = write(tmp_path / "videos" / "a.mp4")
    info = MediaInfo(video, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 1, "aac",
                     16000, 1)  # fmt: skip
    stop = {"scenes": True}

    def scene_cuts(*a, **k):
        if stop["scenes"]:
            raise KeyboardInterrupt
        return [30.0]

    monkeypatch.setattr(analyze, "probe", lambda path: info)
    monkeypatch.setattr(analyze, "_analysis_tracks", lambda *a: (0,))
    monkeypatch.setattr(analyze, "extract_audio", lambda *a, **k: write(a[1]))
    monkeypatch.setattr(analyze, "check_read", lambda *a: None)
    monkeypatch.setattr(analyze, "_loudness", lambda wav, cdir: (np.zeros(1), np.zeros(1)))
    monkeypatch.setattr(analyze, "scene_cuts", scene_cuts)
    monkeypatch.setattr(analyze, "Transcriber", _Whisper)
    monkeypatch.setattr(analyze, "plan_layers", lambda *a: analyze.LayerPlan())
    _Whisper.calls, _Whisper.stop = [], None
    argv = ["analyze", "videos/a.mp4", "--style", "action", "--no-vision", "--out", "p"]
    assert cli.main(argv) == 130
    project = home / "projects" / "p"
    rec = _read(project)
    assert rec["status"] == "interrupted" and rec["sources"] == [str(video)]
    assert not (project / "highlights.json").exists()
    stop["scenes"] = False
    assert cli.main(["resume", "p"]) == 0
    assert _Whisper.calls == [cache.source_dir(video).name]  # the transcript was kept
    assert (project / "highlights.json").is_file() and _read(project)["status"] == "done"


def test_a_first_analyze_stopped_before_its_transcript_leaves_no_folder(
    home, tmp_path, monkeypatch
):
    """The first-run contract stays: a cancel before the first transcript leaves nothing."""
    from cutpoint import analyze  # noqa: PLC0415

    monkeypatch.chdir(tmp_path)
    write(tmp_path / "videos" / "a.mp4")
    info = MediaInfo(tmp_path / "videos" / "a.mp4", 60.0, 60.0, 640, 360, Fraction(30),
                     "yuv420p", "h264", 1, "aac", 16000, 1)  # fmt: skip
    monkeypatch.setattr(analyze, "probe", lambda path: info)
    monkeypatch.setattr(analyze, "_analysis_tracks", lambda *a: (0,))
    monkeypatch.setattr(analyze, "extract_audio", lambda *a, **k: write(a[1]))
    monkeypatch.setattr(analyze, "check_read", lambda *a: None)
    monkeypatch.setattr(analyze, "_loudness", lambda wav, cdir: (np.zeros(1), np.zeros(1)))
    monkeypatch.setattr(analyze, "Transcriber", _Whisper)
    _Whisper.calls, _Whisper.stop = [], 1
    argv = ["analyze", "videos/a.mp4", "--style", "funny", "--no-vision", "--out", "p"]
    assert cli.main(argv) == 130
    assert not (home / "projects" / "p").exists()


# -- events: every layer's model every N sources ----------------------------------------------


@pytest.fixture
def layers_run(home, tmp_path, monkeypatch):
    """``cutpoint events p --layers audio voice visual ocr`` over five sources with each
    layer's model faked; ``state.models`` counts loads and closes per layer and
    ``state.fresh()`` drops what the layers "cached"."""
    from cutpoint.events import visual, voice  # noqa: PLC0415

    project = home / "projects" / "p"
    srcs = [
        Source(
            tmp_path / f"{n}.mp4",
            60.0,
            tmp_path / "cache" / n,
            [AudioPart(0.0, tmp_path / n, 60.0)],
        )
        for n in "abcde"
    ]
    by_path = {str(s.path): s for s in srcs}
    project.mkdir(parents=True)
    cache.save_json(project / "highlights.json", {"sources": [{"path": p} for p in by_path]})
    st = SimpleNamespace(project=project, models={}, done={}, handoff=[], stop=None)

    def model(layer: str) -> Any:
        counts = st.models.setdefault(layer, {"loads": 0, "closes": 0})
        counts["loads"] += 1
        return SimpleNamespace(
            device="cpu", close=lambda: counts.__setitem__("closes", counts["closes"] + 1)
        )

    def detect_audio(s, det, params, force=False):
        """The PANNs events of ``s`` ("cached" once made; without a detector: those)."""
        audio = st.done.setdefault("audio", {})
        if det is not None:
            audio[s.name] = [{"t": 5.0 + len(s.name) + "abcde".index(s.name[0]), "end": 9.0,
                              "label": "laughter", "top_class": "Laughter", "confidence": 0.8,
                              "peak_t": 6.0}]  # fmt: skip
        return audio.get(s.name, [])

    def layer(name: str):
        def run_layer(sources, params, force=False, det=None, bad=None, handoff=None):
            need = [s for s in sources if s.name not in st.done.setdefault(name, set())]
            out: dict[str, Any] = {"cached": len(sources) - len(need), "computed": 0}
            st.handoff.append(handoff)
            if not need:
                return out
            m = det or model(name)
            out["device"] = m.device
            for s in need:
                st.done[name].add(s.name)
                out["computed"] += 1
            if handoff is not None:
                handoff(SimpleNamespace(clip=None))
            if det is None:
                m.close()
            return out

        return run_layer

    class Engine:
        def __init__(self):
            model("ocr")

    def detect_ocr(s, params, engine, *, make_engine=None, **kw):
        if st.stop == s.name:
            raise KeyboardInterrupt
        if make_engine is not None and s.name not in st.done.setdefault("ocr", set()):
            make_engine()
            st.done["ocr"].add(s.name)
        return [], {"frames": 1}

    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "load_source", lambda path, dur: by_path[str(path)])
    monkeypatch.setattr(pipeline, "bad_ranges", lambda *a: {})
    monkeypatch.setattr(pipeline, "load_transcript_lines", lambda project: [])
    monkeypatch.setattr(pipeline, "unload_ollama", list)
    monkeypatch.setattr(common, "gpu_memory_used_mib", lambda: None)
    monkeypatch.setattr(pipeline, "PannsDetector", lambda: model("audio"))
    monkeypatch.setattr(
        pipeline, "cached_audio_events", lambda s, p: st.done.get("audio", {}).get(s.name)
    )
    monkeypatch.setattr(pipeline, "detect_audio_events", detect_audio)
    monkeypatch.setattr(voice, "run_layer", layer("voice"))
    monkeypatch.setattr(visual, "run_layer", layer("visual"))
    monkeypatch.setattr(pipeline, "OcrEngine", Engine)
    monkeypatch.setattr(pipeline, "detect_ocr_events", detect_ocr)
    monkeypatch.setattr(pipeline, "load_preset", lambda game, project: None)
    monkeypatch.setattr(pipeline, "game_key_for_path", lambda *a: "g")

    def fresh() -> None:
        st.models.clear()
        st.done.clear()
        st.handoff.clear()

    def events() -> dict:
        data = cache.load_json(project / "events.json")
        return _without(data, "created", "timings")

    st.fresh, st.events = fresh, events
    st.argv = ["events", "p", "--layers", "audio", "voice", "visual", "ocr", "--no-rescore"]
    st.argv += ["--no-discover"]
    return st


def test_events_closes_each_layer_model_every_n_sources_with_the_same_events(layers_run, caplog):
    s = layers_run
    assert cli.main(s.argv) == 0
    reference = s.events()
    # PANNs: the audio layer's goes on to the voice layer (no load of its own there)
    assert s.models == {
        "audio": {"loads": 1, "closes": 1},
        "visual": {"loads": 1, "closes": 1},
        "ocr": {"loads": 1, "closes": 0},  # the engine is dropped, not closed
    }
    for size in ("0", "2"):
        s.fresh()
        caplog.clear()
        with caplog.at_level("INFO"):
            assert cli.main([*s.argv, "--batch-size", size]) == 0
        assert s.events() == reference, size
        if size == "0":
            continue
        # audio: 3 PANNs (the last handed to voice's first batch), voice: 2 of its own
        assert s.models["audio"] == {"loads": 3, "closes": 3}
        assert s.models["voice"] == {"loads": 2, "closes": 2}
        assert s.models["visual"] == {"loads": 3, "closes": 3}
        assert s.models["ocr"]["loads"] == 3
        for layer in ("audio", "voice", "visual", "ocr"):
            assert f"{layer}: 2/5 done, continuing" in caplog.messages
            assert f"{layer}: 4/5 done, continuing" in caplog.messages


def test_events_records_its_layers_as_steps_and_dry_run_lists_them(layers_run, capsys, monkeypatch):
    """Windows follow-up: events' progress.json has steps (no PROGRESS lines), so resume
    --dry-run shows them; an interrupt in the OCR layer resumes to the same events."""
    s = layers_run
    monkeypatch.setenv(progress.ENV, "1")  # even so: events writes no PROGRESS lines
    assert cli.main(s.argv) == 0
    reference = s.events()
    assert "PROGRESS" not in capsys.readouterr().err
    steps = _read(s.project)["steps"]
    assert list(steps) == list(progress.EVENTS_STEPS)
    assert {k: v["state"] for k, v in steps.items()} == {
        "audio": "done",
        "voice": "done",
        "visual": "done",
        "ocr": "done",
        "crash": "skipped",
        "vlm": "skipped",
        "write": "done",
    }
    s.fresh()
    s.stop = "d.mp4"
    argv = [*s.argv, "--batch-size", "2"]
    assert cli.main(argv) == 130
    rec = _read(s.project)
    assert rec["status"] == "interrupted" and rec["steps"]["ocr"]["state"] == "failed"
    assert rec["steps"]["ocr"]["done"] == 2 and rec["steps"]["ocr"]["total"] == 5
    capsys.readouterr()
    assert cli.main(["resume", "p", "--dry-run"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "cutpoint " + " ".join(argv)
    assert "  audio: done, 5/5 done" in out and "  ocr: failed, 2/5 done" in out
    s.stop = None
    s.models.clear()
    assert cli.main(["resume", "p"]) == 0
    # the same events; the stats count what was cached before the interrupt
    assert _without(s.events(), "stats") == _without(reference, "stats")
    assert s.models["ocr"]["loads"] == 2  # d.mp4 (batch c, d) and e.mp4 only
    assert _read(s.project)["status"] == "done"


def test_the_write_step_ends_before_the_rerank(layers_run, monkeypatch):
    """Review F1: the action-cam re-rank after the layers (its transcribe counts, its
    progress) is not recorded in the write step."""
    s = layers_run
    hl = cache.load_json(s.project / "highlights.json")
    cache.save_json(s.project / "highlights.json", {**hl, "style_kind": "action-cam"})
    monkeypatch.setattr(pipeline, "_telemetry_layer", lambda *a: {})

    def rerank(project):
        progress.continuing("transcribe", 5, 5)
        progress.update(50)
        return project / "highlights.json"

    monkeypatch.setattr(cutpoint.analyze, "reanalyze_if_reachable", rerank)
    assert cli.main(["events", "p", "--layers", "audio"]) == 0
    write = _read(s.project)["steps"]["write"]
    assert write["state"] == "done" and write["done"] is None and write["total"] is None


def test_the_visual_layer_hands_clip_over_only_in_its_last_batch(layers_run, monkeypatch):
    """With the crash layer and no OCR, CLIP goes to the dense confirm from the last batch."""
    from cutpoint import crash  # noqa: PLC0415

    s = layers_run
    got: dict[str, Any] = {}
    clip = SimpleNamespace(close=lambda: None)

    def detect_project(project, **kw):
        got.update(kw)

    monkeypatch.setattr(crash, "detect_project", detect_project)
    monkeypatch.setattr(crash.ClipEncoder, "from_visual", classmethod(lambda cls, m: clip))
    argv = ["events", "p", "--layers", "visual", "crash", "--no-rescore", "--batch-size", "2"]
    assert cli.main(argv) == 0
    assert [h is not None for h in s.handoff] == [False, False, True]
    assert got["encoder"] is clip and got["batch_size"] == 2


# -- crashes: the dense confirm's CLIP every N dense windows (#713) -------------------------


@pytest.mark.parametrize("size", [None, "2"])
def test_crashes_passes_the_batch_size_to_the_dense_confirm(monkeypatch, tmp_path, size):
    from cutpoint import crash  # noqa: PLC0415

    project = tmp_path / "p"
    project.mkdir()
    got: dict[str, Any] = {}

    def detect_project(project_, params=None, **kw):
        got.update(kw)
        return project_ / crash.CRASHES_FILE

    monkeypatch.setattr(crash, "detect_project", detect_project)
    argv = ["crashes", str(project), "--no-reanalyze"]
    assert cli.main([*argv, "--batch-size", size] if size else argv) == 0
    assert got["batch_size"] == (int(size) if size else 0)


def test_crashes_records_its_steps_and_dry_run_lists_them(home, monkeypatch, capsys):
    """#721: crashes' progress.json has steps (no PROGRESS lines); a dense-window batch
    boundary updates the counts, so resume --dry-run shows them."""
    from cutpoint import crash  # noqa: PLC0415

    project = write(home / "p" / "highlights.json", json.dumps({"sources": []})).parent
    stop = {"at": True}

    def detect_project(project_, params=None, **kw):
        progress.start("inputs")
        progress.start("dense")
        progress.continuing("dense check", 0, 4)
        progress.continuing("dense check", 2, 4)  # a --batch-size boundary
        if stop["at"]:
            raise KeyboardInterrupt
        progress.start("result")
        return project_ / crash.CRASHES_FILE

    monkeypatch.setattr(crash, "detect_project", detect_project)
    monkeypatch.setenv(progress.ENV, "1")  # even so: crashes writes no PROGRESS lines
    argv = ["crashes", str(project), "--no-reanalyze", "--batch-size", "2"]
    assert cli.main(argv) == 130
    rec = _read(project)
    assert rec["status"] == "interrupted" and list(rec["steps"]) == ["profile", "inputs", "dense"]
    assert rec["steps"]["dense"]["state"] == "failed"
    assert (rec["steps"]["dense"]["done"], rec["steps"]["dense"]["total"]) == (2, 4)
    assert "PROGRESS" not in capsys.readouterr().err
    assert cli.main(["resume", str(project), "--dry-run"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert "  profile: skipped" in out and "  inputs: done" in out
    assert "  dense: failed, 2/4 done" in out
    stop["at"] = False
    assert cli.main(["resume", str(project)]) == 0
    rec = _read(project)
    assert rec["status"] == "done" and list(rec["steps"]) == list(progress.CRASHES_STEPS)
    assert rec["steps"]["dense"]["state"] == "done" and rec["steps"]["dense"]["done"] == 4


def test_the_crashes_steps_are_no_steps_of_the_other_runs():
    """Inside analyze or events, the crash detection starts none of their steps."""
    others = {*progress.EDIT_STEPS, *progress.PANO_STEPS, *progress.EVENTS_STEPS}
    assert not others & set(progress.CRASHES_STEPS)


# -- 360 analyze: the view layer per batch, the option passed on ------------------------------


def test_360_analyze_runs_the_views_per_batch_and_passes_the_size_on(monkeypatch, tmp_path):
    from cutpoint import analyze as analyze_mod  # noqa: PLC0415
    from cutpoint.pano import cli as pano_cli  # noqa: PLC0415
    from cutpoint.pano import views  # noqa: PLC0415

    clips = [argparse.Namespace(lrv=f"{n}.lrv", clip_id=n) for n in "abcde"]
    calls: list[tuple[list[str], bool]] = []
    passed: dict[str, int] = {}

    def run_layer(part, force=False, handoff=None):
        calls.append(([c.clip_id for c in part], handoff is not None))
        return {"cached": 0, "computed": len(part), "clips": {c.clip_id: 1 for c in part}}

    monkeypatch.setattr(pano_cli, "_project", lambda arg: tmp_path)
    monkeypatch.setattr(pano_cli, "_clips", lambda project: clips)
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(
        pipeline, "run_events", lambda *a, **k: passed.__setitem__("events", k["batch_size"])
    )
    monkeypatch.setattr(views, "run_layer", run_layer)
    monkeypatch.setattr(
        analyze_mod, "analyze", lambda *a, **k: passed.__setitem__("analyze", k["batch_size"])
    )
    monkeypatch.setattr(analyze_mod, "reanalyze", lambda project: None)
    monkeypatch.setattr("cutpoint.crash.dense_needs_run", lambda paths: True)
    monkeypatch.setattr("cutpoint.pano.project.default_style", lambda: "action-cam")
    monkeypatch.setattr("cutpoint.styles.get_style", lambda name: name)
    args = cli._parse(cli.build_parser(), ["360", "analyze", "p", "--batch-size", "2"])
    pano_cli.cmd_analyze(args)
    assert calls == [(["a", "b"], False), (["c", "d"], False), (["e"], True)]
    assert passed == {"analyze": 2, "events": 2}
    timings = json.loads((tmp_path / pano_cli.TIMINGS_FILE).read_text(encoding="utf-8"))
    assert timings["views"]["computed"] == 5 and list(timings["views"]["clips"]) == list("abcde")


@pytest.mark.parametrize("size", [0, 2])
def test_the_360_views_bar_never_goes_back_across_batches(monkeypatch, size):
    """Review F2: each batch reports in its share of the views step (run_layer reports 0-100
    per call); one batch reports as before."""
    from cutpoint.pano import cli as pano_cli  # noqa: PLC0415
    from cutpoint.pano import views  # noqa: PLC0415

    clips = [argparse.Namespace(lrv=f"{n}.lrv", clip_id=n) for n in "abcde"]
    seen: list[int] = []

    def run_layer(part, force=False, handoff=None):
        for i in range(len(part)):
            progress.update(progress.fraction(i, len(part)))
        return {"cached": 0, "computed": len(part)}

    def observe(step, state, pct, done, total):
        if state == "running" and pct is not None:
            seen.append(pct)

    monkeypatch.setattr(views, "run_layer", run_layer)
    monkeypatch.setattr(progress.Run, "observers", [observe])
    args = argparse.Namespace(batch_size=size, force_views=False)
    with progress.run(progress.PANO_STEPS) as steps, steps.step("views"):
        pano_cli._views(clips, args, None)
    assert seen == sorted(seen)  # without the shares: 0, 20, 0, 20, 0
    assert sorted(set(seen)) == [0, 20, 40, 60, 80]


# -- W-F1: the VLM layer's batches ------------------------------------------------------------


def _windows(*sources: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(source=s, key=f"{s}{i}") for i, s in enumerate(sources)]


def test_vlm_batches_end_only_at_a_source_after_n_windows():
    ws = _windows("a", "a", "a", "b", "b", "c")
    keys = [[w.key for w in part] for part in pipeline.vlm_batches(ws, 2)]
    assert keys == [["a0", "a1", "a2"], ["b3", "b4"], ["c5"]]
    assert pipeline.vlm_batches(ws, 4) == [ws[:5], ws[5:]]
    assert pipeline.vlm_batches(ws, 0) == [ws] == pipeline.vlm_batches(ws, 9)
    one = _windows(*"a" * 41)  # one recording: never reloaded within it (Windows design note)
    assert pipeline.vlm_batches(one, 2) == [one]
    assert pipeline.vlm_batches([], 2) == [[]]


@pytest.mark.parametrize(("size", "unload", "unloads"), [(2, True, 2), (2, False, 0), (0, True, 0)])
def test_describe_batches_gives_the_results_of_one_call(monkeypatch, size, unload, unloads):
    """Windows W-F1: 5 windows of three sources in batches of 2 per source give the results and
    stats of one call; the model is unloaded at each boundary (2x), never without ``unload``."""
    calls: list[list[str]] = []

    def describe(rater, by_path, part, vlm, prompts, transcript, layer1, force=False):
        calls.append([w.key for w in part])
        res = [{"key": w.key, "source": w.source} for w in part]
        return res, {"requested": len(part), "ok": len(part), "seconds": [1.0] * len(part)}

    rater = SimpleNamespace(unloads=0)
    rater.unload = lambda: setattr(rater, "unloads", rater.unloads + 1)
    monkeypatch.setattr(pipeline, "describe_windows", describe)
    ws = _windows("a", "a", "b", "b", "c")

    def run(n: int, unload: bool) -> tuple[list, dict]:
        return pipeline._describe_batches(
            rater, {}, ws, None, {}, [], {}, force=False, batch_size=n, unload=unload
        )

    reference = run(0, True)
    calls.clear()
    rater.unloads = 0
    assert run(size, unload) == reference
    assert rater.unloads == unloads
    assert calls == ([["a0", "a1"], ["b2", "b3"], ["c4"]] if size else [[w.key for w in ws]])


# -- the memory guard (#109 phase 5) at a batch; the boundary after the cached part --------


def _resume_target(message: str) -> Path:
    found = re.search(r"`cutpoint resume (.+?)` to continue", message)
    assert found, message
    return Path(found.group(1)).resolve()


class _GuardedWhisper(_Whisper):
    """The fake Whisper, checking the memory guard as it loads (as the real one does)."""

    def transcribe(self, wav: Path, on_progress=None) -> list[dict]:
        if not self.loaded:
            memguard.check("whisper-small", label="Whisper small")
        return super().transcribe(wav, on_progress)


def test_analyze_stops_with_low_memory_at_a_batch_and_resume_finishes(
    analyze_run, mem, monkeypatch, capsys, caplog
):
    """Whisper loads again after a --batch-size boundary with too little memory: stopped
    resumable after the first batch; resume with memory free writes the outputs of a plain
    run and transcribes only the rest."""
    r = analyze_run
    monkeypatch.setattr(cutpoint.analyze, "Transcriber", _GuardedWhisper)
    assert cli.main(r.argv) == 0
    reference = r.outputs()
    for name in ("highlights.json", "transcript.txt"):
        (r.project / name).unlink()
    r.fresh()
    loads: list[int] = []

    def check(model, **kw):
        loads.append(1)
        if len(loads) == 2:  # the second batch's load
            mem.ram = 1.0
        return real(model, **kw)

    real = memguard.check
    monkeypatch.setattr(memguard, "check", check)
    argv = [*r.argv, "--batch-size", "2"]
    capsys.readouterr()
    with caplog.at_level(logging.ERROR):
        assert cli.main([*argv, "--json"]) == 1
    doc = envelope(capsys.readouterr().out)
    assert doc["error"]["code"] == "low_memory"
    assert mem.unloaded == 1 and mem.freed >= 1  # analyze frees at its boundaries too
    rec = json.loads((r.project / runstate.FILE).read_text(encoding="utf-8"))
    assert rec["status"] == "low_memory"
    assert rec["steps"]["transcribe"]["state"] == "failed"  # the PROGRESS states
    message = next(r.getMessage() for r in caplog.records if r.levelno == logging.ERROR)
    assert _resume_target(message) == r.project.resolve()
    assert not (r.project / "highlights.json").exists()
    mem.ram = 64.0
    _Whisper.calls.clear()
    assert cli.main(["resume", "p"]) == 0
    assert len(_Whisper.calls) == 3  # the first batch's transcripts are cached
    assert r.outputs() == reference
    assert json.loads((r.project / runstate.FILE).read_text(encoding="utf-8"))["status"] == "done"


def test_a_resumed_batch_boundary_after_the_cached_part_says_continuing_once(analyze_run, caplog):
    """Windows cosmetic: resume with --batch-size 1 after one cached source logs
    ``transcribe: 1/N done, continuing`` once, not again at the boundary right after it."""
    r = analyze_run
    _Whisper.stop = 2
    assert cli.main([*r.argv, "--batch-size", "1"]) == 130
    _Whisper.stop = None
    with caplog.at_level(logging.INFO):
        assert cli.main(["resume", "p"]) == 0
    done = [m for m in caplog.messages if m.startswith("transcribe: ")]
    assert done == [f"transcribe: {k}/5 done, continuing" for k in range(1, 5)]


def test_a_layer_short_of_memory_stops_analyze_with_low_memory(analyze_run, monkeypatch, capsys):
    """Review F7: PANNs short of memory in the reactions step is no failed step analyze goes
    on without: exit 1, low_memory in the envelope and progress.json, no step_failed."""
    r = analyze_run
    plan = cutpoint.analyze.LayerPlan(layers={"reactions": ("audio",)})
    monkeypatch.setattr(cutpoint.analyze, "plan_layers", lambda *a: plan)

    def run_events(project, *, stage, **kw):
        with stage("audio"):
            memguard.check("panns", label="PANNs")  # as PannsDetector does
        pytest.fail("went on after the layer")

    monkeypatch.setattr(pipeline, "run_events", run_events)
    monkeypatch.setattr(memguard, "_free", lambda url=None: None)
    monkeypatch.setattr(memguard, "ram_free", lambda: memguard.GIB)  # 1 GB: short
    capsys.readouterr()
    assert cli.main([*r.argv, "--json"]) == 1
    doc = envelope(capsys.readouterr().out)
    assert doc["error"]["code"] == "low_memory"
    assert "step_failed" not in [w["code"] for w in doc["warnings"]]
    rec = _read(r.project)
    assert rec["status"] == "low_memory" and rec["steps"]["reactions"]["state"] == "failed"
    assert not (r.project / "highlights.json").exists()


# -- #713: analyze --vision in batches, the providers' unload hook ----------------------------


class _Rater:
    """A reachable vision model whose ratings are cached across runs (as the real cache is,
    per source); records rate and unload calls, KeyboardInterrupt on rate call ``stop``."""

    calls: list[str] = []  # noqa: RUF012 - reset by the fixture
    store: dict[tuple[str, float], float] = {}  # noqa: RUF012
    stop: int | None = None
    consent_needed = None
    cache_only = False
    hits = asked = missed = 0

    def __init__(self, **_kw: Any) -> None:
        pass

    def available(self) -> bool:
        return True

    def cached(self, src, start, _end, _style, activity=None, transcript=None, laughter=False):
        hit = self.store.get((Path(src).name, start))
        return None if hit is None else VisionResult(hit, "moment")

    def rate(self, src, start, end, style, activity=None, transcript=None, laughter=False):
        if (hit := self.cached(src, start, end, style)) is not None:
            return hit
        type(self).calls.append(f"rate {Path(src).stem}")
        if sum(c.startswith("rate") for c in self.calls) == self.stop:
            raise KeyboardInterrupt
        score = 2.0 + 1.5 * len(Path(src).stem) + start / 10
        type(self).store[(Path(src).name, start)] = score
        return VisionResult(score, "moment")

    def unload(self) -> None:
        type(self).calls.append("unload")


@pytest.fixture
def vision_run(analyze_run, monkeypatch):
    """``analyze_run`` with ``--vision`` and the cached fake rater: one candidate per source,
    all five rated."""
    _Rater.calls, _Rater.store, _Rater.stop = [], {}, None
    monkeypatch.setattr(cutpoint.analyze, "VisionRater", _Rater)
    analyze_run.argv = [a if a != "--no-vision" else "--vision" for a in analyze_run.argv]
    return analyze_run


def test_vision_batches_group_n_sources_and_keep_rank_order_in_one_batch():
    cands = [{"source": s, "start": float(i)} for i, s in enumerate("abacbd")]
    starts = [[c["start"] for c in part] for part in cutpoint.analyze._vision_batches(cands, 2)]
    assert starts == [[0.0, 2.0, 1.0, 4.0], [3.0, 5.0]]  # a and b, then c and d
    for size in (0, 4, 9):  # off, or every source in one batch: rank order as before
        assert cutpoint.analyze._vision_batches(cands, size) == [cands]
    assert cutpoint.analyze._vision_batches([], 2) == [[]]


def test_analyze_vision_unloads_every_n_sources_with_the_same_outputs(vision_run, caplog):
    r = vision_run
    assert cli.main(r.argv) == 0
    reference = r.outputs()
    assert _Rater.calls.count("unload") == 1 and reference[0]["settings"]["vision_used"]
    for size, unloads in (("0", 1), ("2", 3), ("5", 1)):
        r.fresh()
        _Rater.calls, _Rater.store = [], {}
        caplog.clear()
        with caplog.at_level("INFO"):
            assert cli.main([*r.argv, "--batch-size", size]) == 0
        assert r.outputs() == reference
        assert _Rater.calls.count("unload") == unloads and _Rater.calls[-1] == "unload"
        boundaries = [m for m in caplog.messages if m.startswith("vision: ")]
        assert boundaries == (
            ["vision: 2/5 done, continuing", "vision: 4/5 done, continuing"] if size == "2" else []
        )
        step = _read(r.project)["steps"]["vision"]
        assert (step["state"], step["done"], step["total"]) == ("done", 5, 5)


def test_analyze_vision_interrupted_in_a_batch_resumes_to_the_same_outputs(vision_run, caplog):
    """Stopped on the third rating (the second batch's first): resume rates only the rest,
    says how many were cached and writes what an uninterrupted run does."""
    r = vision_run
    assert cli.main(r.argv) == 0
    reference = r.outputs()
    (r.project / "highlights.json").unlink()
    _Rater.calls, _Rater.store, _Rater.stop = [], {}, 3
    assert cli.main([*r.argv, "--batch-size", "2"]) == 130
    rec = _read(r.project)
    assert rec["status"] == "interrupted" and rec["steps"]["vision"]["state"] == "failed"
    assert rec["steps"]["vision"]["done"] == 2
    _Rater.calls, _Rater.stop = [], None
    with caplog.at_level("INFO"):
        assert cli.main(["resume", "p"]) == 0
    assert sum(c.startswith("rate") for c in _Rater.calls) == 3  # two ratings were cached
    assert "vision: 2/5 done, continuing" in caplog.messages
    assert r.outputs() == reference
    assert _read(r.project)["status"] == "done"


class _Provider:
    """A Vision role provider on this machine: replies with a rating, counts unload calls."""

    name, model, host, url, local = "openai", "m", "127.0.0.1", "http://127.0.0.1:8080/v1", True

    def __init__(self, freed: bool) -> None:
        self.freed, self.unloads, self.chats = freed, 0, 0

    def chat(self, messages, *, schema=None, **_kw):
        self.chats += 1
        return SimpleNamespace(text='{"score": 7, "label": "x"}')

    def unload(self) -> bool:
        self.unloads += 1
        return self.freed


@pytest.mark.parametrize("freed", [True, False])
def test_a_vision_role_unloads_through_its_provider_only_after_a_request(monkeypatch, freed):
    """VisionRater.unload of a Vision role other than Ollama asks the provider's unload hook,
    only once a request went to it since the last one, and never Ollama's /api/generate."""
    p = _Provider(freed)
    monkeypatch.setattr(providers, "make_vision", lambda _s: p)
    settings = config.AssistantSettings("openai", url=p.url, model="m")
    cfg = SimpleNamespace(vision_assistant=settings)
    monkeypatch.setattr(vision, "get_config", lambda: cfg)
    monkeypatch.setattr(vision.requests, "post", lambda *a, **k: pytest.fail("Ollama asked"))
    monkeypatch.setattr(memguard, "check_vision", lambda _r: None)
    rater = vision.VisionRater()
    assert rater.via is not None
    rater.unload()
    assert p.unloads == 0  # nothing was sent: nothing to ask
    rater.via._ready = True
    assert rater.describe("p", [b"jpg"], {"type": "object"}) is not None
    rater.unload()
    rater.unload()
    assert (p.chats, p.unloads) == (1, 1)


def test_http_providers_send_nothing_to_unload(monkeypatch):
    """Hosted APIs (nothing on this machine) and OpenAI-compatible servers (llama-server, LM
    Studio: no unload request in that API) get no extra request."""

    def sent(*_a, **_k):
        pytest.fail("a request was sent")

    monkeypatch.setattr(requests, "request", sent)
    monkeypatch.setattr(requests, "post", sent)
    monkeypatch.setattr(requests.Session, "request", sent)
    for p in (
        providers.OpenAIProvider("http://127.0.0.1:8080", "m", 0, None),
        providers.OpenAIProvider("", "gpt-6-luna", 0, "sk-test"),
        providers.AnthropicProvider("", "claude-sonnet-5-5", 0, "sk-test"),
        providers.XAIProvider("", "grok-4.7", 0, "xai-test"),
    ):
        assert p.unload() is False


@pytest.mark.parametrize(("status", "want"), [(200, True), (404, False), (302, False)])
def test_the_ollama_provider_unloads_with_keep_alive_0(monkeypatch, status, want):
    sent: list[tuple[str, dict, bool]] = []

    def post(url, json=None, timeout=None, allow_redirects=True, **_k):
        sent.append((url, json, allow_redirects))
        return SimpleNamespace(status_code=status, headers={}, close=lambda: None)

    monkeypatch.setattr(providers.requests, "post", post)
    assert providers.OllamaProvider("127.0.0.1:11434", "", 0).unload() is False  # auto: unset
    assert sent == []
    assert providers.OllamaProvider("127.0.0.1:11434", "qwen3.5:9b", 0).unload() is want
    assert sent == [
        ("http://127.0.0.1:11434/api/generate", {"model": "qwen3.5:9b", "keep_alive": 0}, False)
    ]

    def down(*_a, **_k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(providers.requests, "post", down)
    assert providers.OllamaProvider("127.0.0.1:11434", "m", 0).unload() is False


def test_events_hand_the_ollama_flags_to_the_crash_layer(layers_run, monkeypatch):
    """#816: the crash layer's profile detection gets --ollama-url/--ollama-model."""
    from cutpoint import crash  # noqa: PLC0415

    got: dict[str, Any] = {}
    monkeypatch.setattr(crash, "detect_project", lambda project, **kw: got.update(kw))
    monkeypatch.setattr(
        crash.ClipEncoder,
        "from_visual",
        classmethod(lambda cls, m: SimpleNamespace(close=lambda: None)),
    )
    argv = ["events", "p", "--layers", "visual", "crash", "--no-rescore"]
    assert cli.main(argv) == 0
    assert (got["ollama_url"], got["ollama_model"]) == (None, None)
