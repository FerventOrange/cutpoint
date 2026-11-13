"""One bad source skips only itself (#985, #1033): action-cam telemetry, the contact sheets,
the crash dense check and the per-source layers of the events pipeline go on with the other
sources, with a WARNING and a ``step_failed`` JSON warning naming the one that failed."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint import actioncam, crash, jsonout, telemetry
from cutpoint.errors import LowMemoryError
from cutpoint.events import audio_events, pipeline, sheets, visual, voice
from cutpoint.events.common import AudioPart, Source

N = 50  # 5 s on the 0.1 s grid
BAD = "'list' object cannot be interpreted as an integer"


def _tel() -> SimpleNamespace:
    grid = {
        "gps_ok": np.ones(N),
        "lift_mask": np.zeros(N),
        "still_mask": np.zeros(N),
        "speed_kmh": np.full(N, 20.0),
    }
    return SimpleNamespace(n=N, grid=grid, events=[])


def _source(tmp_path: Path, name: str) -> Source:
    return Source(
        path=tmp_path / f"{name}.mp4",
        duration=N / 10,
        cdir=tmp_path / name,
        audio_parts=[AudioPart(0.0, tmp_path / f"{name}.wav", N / 10)],
    )


def _extract(bad: set[str], err: Exception | None = None):
    def extract(src, duration):
        if Path(src).stem in bad:
            raise err or TypeError(BAD)
        return _tel()

    return extract


def _warned(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def _step_failed(result) -> list[tuple[str, list[str]]]:
    return [(w["message"], w["files"]) for w in result.warnings if w["code"] == "step_failed"]


# --------------------------------------------------------------------------- action-cam prepare
def test_prepare_scores_a_source_whose_telemetry_fails_without_it(tmp_path, monkeypatch, caplog):
    srcs = [tmp_path / f"{n}.mp4" for n in ("ridge", "gully", "bowl")]
    monkeypatch.setattr(actioncam, "extract", _extract({"gully"}))
    monkeypatch.setattr(actioncam, "join_runs", lambda tels: {"runs": len(tels)})
    monkeypatch.setattr(actioncam, "write_telemetry_json", lambda project, ctx: None)
    with caplog.at_level(logging.WARNING), jsonout.run("analyze") as result:
        ctx = actioncam.prepare(srcs, {str(s): N / 10 for s in srcs}, tmp_path)
    assert ctx.telemetry[str(srcs[0])] is not None and ctx.telemetry[str(srcs[2])] is not None
    assert ctx.telemetry[str(srcs[1])] is None
    assert ctx.runs == {"runs": 2}  # the good ones still make the day
    assert ctx.speed_ref.size == 2 * N
    message = f"gully.mp4: telemetry not read ({BAD}); scored without it"
    assert _warned(caplog) == [message]
    assert _step_failed(result) == [(message, [str(srcs[1])])]


def test_prepare_raises_when_every_source_fails(tmp_path, monkeypatch):
    srcs = [tmp_path / f"{n}.mp4" for n in ("ridge", "gully")]
    monkeypatch.setattr(actioncam, "extract", _extract({"ridge", "gully"}))
    with pytest.raises(TypeError, match="cannot be interpreted"):
        actioncam.prepare(srcs, {str(s): N / 10 for s in srcs}, tmp_path)


def test_prepare_lets_a_memory_shortage_stop_the_run(tmp_path, monkeypatch):
    srcs = [tmp_path / f"{n}.mp4" for n in ("ridge", "gully")]
    monkeypatch.setattr(actioncam, "extract", _extract({"ridge"}, LowMemoryError("short")))
    with pytest.raises(LowMemoryError):
        actioncam.prepare(srcs, {str(s): N / 10 for s in srcs}, tmp_path)


# --------------------------------------------------------------------------- contact sheets
@pytest.fixture
def sheet_fakes(monkeypatch):
    """Fake visual data (2 frames per source, a person in frame 0) and contact sheets that
    record the paths they show."""
    shown: dict[str, set[str]] = {}

    def load_visual(s):
        det = np.array([[0, visual.COCO_PERSON, 0.9, 0.1, 0.1, 0.4, 0.9]])
        return SimpleNamespace(t=np.array([1.0, 2.0]), det=det)

    def frame_scores(d):
        k = len(d.t)
        return {
            "beauty_zs": np.ones(k),
            "aesthetic": np.ones(k),
            "count": np.ones(k),
        }

    def contact_sheet(items, out):
        if not items:
            return None
        shown[out.stem] = {Path(i["path"]).name for i in items}
        return out

    monkeypatch.setattr(visual, "load_visual", load_visual)
    monkeypatch.setattr(visual, "frame_scores", frame_scores)
    monkeypatch.setattr(visual, "friends_score", lambda sc: np.ones(len(sc["count"])))
    monkeypatch.setattr(visual, "beauty_score", lambda sc: np.ones(len(sc["count"])))
    monkeypatch.setattr(visual, "self_boxes", lambda d, rules: [False] * len(d.det))
    monkeypatch.setattr(sheets, "contact_sheet", contact_sheet)
    monkeypatch.setattr(sheets, "_io", lambda path: path)
    return shown


def test_sheets_leave_out_a_source_whose_telemetry_fails(
    tmp_path, monkeypatch, caplog, sheet_fakes
):
    srcs = [_source(tmp_path, n) for n in ("reef", "wreck", "drift")]
    monkeypatch.setattr(telemetry, "extract", _extract({"wreck"}))
    with caplog.at_level(logging.WARNING), jsonout.run("events") as result:
        written = sheets.write_visual_sheets(tmp_path, srcs, n=6)
    assert [p.stem for p in written] == ["beauty_top", "beauty_bottom", "friends_top"]
    assert sheet_fakes["beauty_top"] == {"reef.mp4", "drift.mp4"}
    message = f"wreck.mp4: contact sheets skipped this source ({BAD})"
    assert _warned(caplog) == [message]
    assert _step_failed(result) == [(message, [str(srcs[1].path)])]


def test_sheets_raise_when_every_source_fails(tmp_path, monkeypatch, sheet_fakes):
    srcs = [_source(tmp_path, n) for n in ("reef", "wreck")]
    monkeypatch.setattr(telemetry, "extract", _extract({"reef", "wreck"}))
    with pytest.raises(TypeError, match="cannot be interpreted"):
        sheets.write_visual_sheets(tmp_path, srcs)
    assert sheet_fakes == {}


# --------------------------------------------------------------------------- events: telemetry
def test_telemetry_layer_keeps_a_failing_source_events_without_coverage(
    tmp_path, monkeypatch, caplog
):
    srcs = [_source(tmp_path, n) for n in ("ridge", "gully")]
    crash = {"type": "crash", "t": 2.0, "peak_g": 6.0, "strength": 0.7}
    data = {"sources": {str(s.path): {"events": [crash]} for s in srcs}}
    (tmp_path / "telemetry.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(telemetry, "extract", _extract({"gully"}))
    with caplog.at_level(logging.WARNING), jsonout.run("events") as result:
        out = pipeline._telemetry_layer(tmp_path, srcs, pipeline.VlmParams(window_s=1.0))
    good, bad = out[str(srcs[0].path)], out[str(srcs[1].path)]
    assert [e["type"] for e in bad] == ["crash"]
    assert good[0]["type"] == "crash" and {e["type"] for e in good[1:]} == {"coverage"}
    message = f"gully.mp4: Telemetry coverage skipped this source ({BAD})"
    assert _warned(caplog) == [message]
    assert _step_failed(result) == [(message, [str(srcs[1].path)])]


def test_telemetry_layer_raises_when_every_source_fails(tmp_path, monkeypatch):
    srcs = [_source(tmp_path, n) for n in ("ridge", "gully")]
    data = {"sources": {str(s.path): {"events": []} for s in srcs}}
    (tmp_path / "telemetry.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(telemetry, "extract", _extract({"ridge", "gully"}))
    with pytest.raises(TypeError, match="cannot be interpreted"):
        pipeline._telemetry_layer(tmp_path, srcs, pipeline.VlmParams())


def test_telemetry_layer_counts_a_source_without_telemetry_as_fine(tmp_path, monkeypatch):
    """Only every source failing fails the layer, as in the action-cam read."""
    srcs = [_source(tmp_path, n) for n in ("ridge", "gully")]
    data = {"sources": {str(srcs[1].path): {"events": []}}}  # ridge has no telemetry record
    (tmp_path / "telemetry.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(telemetry, "extract", _extract({"gully"}))
    out = pipeline._telemetry_layer(tmp_path, srcs, pipeline.VlmParams())
    assert out == {str(srcs[1].path): []}


# --------------------------------------------------------------------------- events: audio
@pytest.fixture
def run_audio(tmp_path, monkeypatch):
    """``run_events`` over ``srcs`` with ``layers`` (only audio by default; its detection fails
    for the names in ``bad`` with ``err``); returns the events written and the detectors
    closed."""

    def go(
        srcs: list[Source],
        bad: set[str],
        err: Exception | None = None,
        layers: tuple[str, ...] = ("audio",),
        batch_size: int = 0,
    ):
        by_path = {str(s.path): s for s in srcs}
        closed: list[object] = []

        class Det:
            device = "cpu"

            def close(self):
                closed.append(self)

        def detect(s, det, params, force=False):
            if s.path.stem in bad:
                raise err or ValueError("truncated wav")
            return [{"t": 1.0, "end": 2.0, "label": "Laughter", "score": 0.9}]

        def vision_rater(url, model):
            raise ValueError("no vision model in tests")

        written: dict[str, object] = {}

        def write_outputs(project, paths, events, *a):
            written["events"] = events
            return project / "events.json", project / "events.txt"

        monkeypatch.setattr(
            pipeline, "load_highlights", lambda p: {"sources": [{"path": k} for k in by_path]}
        )
        monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
        monkeypatch.setattr(pipeline, "VisionRater", vision_rater)
        monkeypatch.setattr(pipeline, "load_source", lambda path, dur: by_path[str(path)])
        monkeypatch.setattr(pipeline, "bad_ranges", lambda *a: {})
        monkeypatch.setattr(pipeline, "load_transcript_lines", lambda project: [])
        monkeypatch.setattr(pipeline, "unload_ollama", list)
        monkeypatch.setattr(pipeline, "cached_audio_events", lambda s, p: None)
        monkeypatch.setattr(pipeline, "PannsDetector", Det)
        monkeypatch.setattr(audio_events, "PannsDetector", Det)
        monkeypatch.setattr(pipeline, "detect_audio_events", detect)
        monkeypatch.setattr(pipeline, "game_key_for_path", lambda *a: "none")
        monkeypatch.setattr(pipeline, "load_preset", lambda *a: None)
        monkeypatch.setattr(pipeline, "select_windows", lambda *a, **k: [])
        monkeypatch.setattr(pipeline, "describe_windows", lambda *a, **k: ([], {}))
        monkeypatch.setattr(pipeline, "kept_dialogue", lambda *a: [])
        monkeypatch.setattr(pipeline, "normalize", lambda src, e: {**e, "source": src})
        monkeypatch.setattr(pipeline, "write_outputs", write_outputs)
        pipeline.run_events(
            tmp_path, layers=layers, discover=False, cached_ocr=False, batch_size=batch_size
        )
        return written["events"], closed

    return go


def test_audio_layer_skips_a_source_whose_detection_fails(tmp_path, run_audio, caplog):
    srcs = [_source(tmp_path, n) for n in ("cave", "arch", "kelp")]
    with caplog.at_level(logging.WARNING), jsonout.run("events") as result:
        events, closed = run_audio(srcs, {"arch"})
    assert sorted(Path(e["source"]).name for e in events) == ["cave.mp4", "kelp.mp4"]
    assert len(closed) == 1  # one detector for all of them, closed once
    message = "arch.mp4: Audio events skipped this source (truncated wav)"
    assert _warned(caplog) == [message]
    assert _step_failed(result) == [(message, [str(srcs[1].path)])]


def test_audio_layer_fails_when_every_source_fails(tmp_path, run_audio):
    srcs = [_source(tmp_path, n) for n in ("cave", "arch")]
    with pytest.raises(ValueError, match="truncated wav"):
        run_audio(srcs, {"cave", "arch"})


def test_audio_layer_lets_a_memory_shortage_stop_the_run(tmp_path, run_audio):
    srcs = [_source(tmp_path, n) for n in ("cave", "arch")]
    with pytest.raises(LowMemoryError):
        run_audio(srcs, {"cave"}, LowMemoryError("short"))


# --------------------------------------------------------------------------- events: voice
class _Det:
    device = "cpu"

    def close(self):
        pass


def _voice(monkeypatch, bad: set[str], err: Exception | None = None) -> list[str]:
    """Voice grids that fail for the names in ``bad``; returns the names computed."""
    done: list[str] = []

    def compute(s, det, vad, params, force=False):
        if s.path.stem in bad:
            raise err or ValueError("truncated wav")
        done.append(s.name)
        return {}

    monkeypatch.setattr(voice, "load_voice", lambda s, params: None)
    monkeypatch.setattr(voice, "SileroVad", object)
    monkeypatch.setattr(voice, "compute_voice", compute)
    return done


def test_voice_layer_skips_a_source_whose_grids_fail(tmp_path, monkeypatch, caplog):
    srcs = [_source(tmp_path, n) for n in ("cave", "arch", "kelp")]
    done = _voice(monkeypatch, {"arch"})
    with caplog.at_level(logging.WARNING), jsonout.run("events") as result:
        stats = voice.run_layer(srcs, voice.VoiceParams(), det=_Det())
    assert done == ["cave.mp4", "kelp.mp4"] and stats["computed"] == 2
    message = "arch.mp4: Voice layer skipped this source (truncated wav)"
    assert _warned(caplog) == [message]
    assert _step_failed(result) == [(message, [str(srcs[1].path)])]


def test_voice_layer_fails_when_every_source_fails(tmp_path, monkeypatch):
    srcs = [_source(tmp_path, n) for n in ("cave", "arch")]
    _voice(monkeypatch, {"cave", "arch"})
    with pytest.raises(ValueError, match="truncated wav"):
        voice.run_layer(srcs, voice.VoiceParams(), det=_Det())


def test_voice_layer_hands_the_errors_to_the_caller_of_a_batch(tmp_path, monkeypatch):
    """A batch of only bad files does not fail the layer: the caller counts every batch's."""
    srcs = [_source(tmp_path, n) for n in ("cave", "arch")]
    _voice(monkeypatch, {"cave", "arch"})
    errors: list[Exception] = []
    voice.run_layer(srcs, voice.VoiceParams(), det=_Det(), errors=errors)
    assert [str(e) for e in errors] == ["truncated wav", "truncated wav"]


def test_voice_layer_lets_a_memory_shortage_stop_the_run(tmp_path, monkeypatch):
    srcs = [_source(tmp_path, n) for n in ("cave", "arch")]
    _voice(monkeypatch, {"cave"}, LowMemoryError("short"))
    with pytest.raises(LowMemoryError):
        voice.run_layer(srcs, voice.VoiceParams(), det=_Det(), errors=[])


# --------------------------------------------------------------------------- events: visual
def _visual(monkeypatch, bad: set[str], err: Exception | None = None) -> SimpleNamespace:
    """A visual layer whose frames fail for the names in ``bad``; records the names computed,
    the models closed and handed over."""
    st = SimpleNamespace(done=[], closed=0, handed=[])

    class Models:
        device = "cpu"

        def __init__(self, **kw):
            pass

        def close(self):
            st.closed += 1

    def compute(s, models, params, force=False, avoid=None):
        if s.path.stem in bad:
            raise err or OSError("moov atom not found")
        st.done.append(s.name)
        return object()

    monkeypatch.setattr(visual, "load_visual", lambda s, params: None)
    monkeypatch.setattr(visual, "VisualModels", Models)
    words = dict.fromkeys(
        ("highlight_pos", "highlight_neg", "group_pos", "group_neg", "person_pos", "person_neg"),
        (),
    )
    monkeypatch.setattr(visual, "_prompts", lambda prompts: SimpleNamespace(**words))
    monkeypatch.setattr(visual, "text_embeddings", lambda texts, models: None)
    monkeypatch.setattr(visual, "compute_visual", compute)
    return st


def test_visual_layer_skips_a_source_whose_frames_fail(tmp_path, monkeypatch, caplog):
    srcs = [_source(tmp_path, n) for n in ("reef", "wreck", "drift")]
    st = _visual(monkeypatch, {"wreck"})
    with caplog.at_level(logging.WARNING), jsonout.run("events") as result:
        stats = visual.run_layer(srcs, visual.VisualParams(), handoff=st.handed.append)
    assert st.done == ["reef.mp4", "drift.mp4"] and stats["computed"] == 2
    assert len(st.handed) == 1 and st.closed == 1  # CLIP still handed over, models freed
    message = "wreck.mp4: Visual layer skipped this source (moov atom not found)"
    assert _warned(caplog) == [message]
    assert _step_failed(result) == [(message, [str(srcs[1].path)])]


def test_visual_layer_fails_when_every_source_fails(tmp_path, monkeypatch):
    srcs = [_source(tmp_path, n) for n in ("reef", "wreck")]
    st = _visual(monkeypatch, {"reef", "wreck"})
    with pytest.raises(OSError, match="moov atom"):
        visual.run_layer(srcs, visual.VisualParams(), handoff=st.handed.append)
    assert st.handed == [] and st.closed == 1


def test_visual_layer_hands_the_errors_to_the_caller_of_a_batch(tmp_path, monkeypatch):
    srcs = [_source(tmp_path, n) for n in ("reef", "wreck")]
    _visual(monkeypatch, {"reef", "wreck"})
    errors: list[Exception] = []
    visual.run_layer(srcs, visual.VisualParams(), errors=errors)
    assert [str(e) for e in errors] == ["moov atom not found"] * 2


def test_visual_layer_lets_a_memory_shortage_stop_the_run(tmp_path, monkeypatch):
    srcs = [_source(tmp_path, n) for n in ("reef", "wreck")]
    st = _visual(monkeypatch, {"reef"}, LowMemoryError("short"))
    with pytest.raises(LowMemoryError):
        visual.run_layer(srcs, visual.VisualParams(), errors=[])
    assert st.closed == 1


# --------------------------------------------------------------------------- crash dense check
@pytest.fixture
def dense_project(tmp_path, monkeypatch):
    """``crash.detect_project`` over one candidate per source with the real
    :func:`crash.dense_feature` and :func:`crash.write_sheet`; ``bad`` names the sources whose
    decode raises ``err`` (``vanish``: after deleting the file), ``bad_inputs`` those whose
    inputs raise it and ``gone`` those deleted once their inputs loaded."""

    def go(
        names: list[str],
        bad: set[str],
        err: Exception | None = None,
        sheets: int = 0,
        vanish: bool = False,
        bad_inputs: frozenset[str] = frozenset(),
        gone: frozenset[str] = frozenset(),
    ) -> Path:
        srcs = [tmp_path / f"{n}.mp4" for n in names]
        for s in srcs:
            s.write_bytes(b"x")
        project = tmp_path / "proj"
        project.mkdir()
        hl = {"sources": [{"path": str(s), "duration": 60.0} for s in srcs]}
        (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")

        def inputs(src, dur, p, tmp, lang):
            if src.stem in bad_inputs:
                raise err or ValueError("bad IMU block")
            if src.stem in gone:
                src.unlink()
            return crash.SourceInputs(
                path=src, duration=dur, tel=None, imu=None, visual=None, kf_crash=None
            )

        def cands(inp, p):
            tel = {"speed_before_kmh": 10, "speed_min_after_kmh": 10, "on_lift": False}
            scores = {"telemetry": 0.8, "visual": 0.6, "audio": 0, "words": 0}
            return [
                {"name": inp.path.name, "source": str(inp.path), "t0": 10.0, "t1": 12.0,
                 "origin": "telemetry", "pre_score": 0.9, "telemetry": tel, "scores": scores}
            ]  # fmt: skip

        def decode(path, a, b, fps, width=448):
            if Path(path).stem in bad:
                if vanish:
                    Path(path).unlink(missing_ok=True)
                raise err or FileNotFoundError(f"{Path(path).name}: no such file")
            ts = np.arange(a, b, 1.0)
            return ts, np.zeros((len(ts), 32, 32, 3), np.uint8)

        enc = SimpleNamespace(
            embed=lambda frames: np.tile(np.linspace(0.1, 0.9, len(frames))[:, None], (1, 4)),
            close=lambda: None,
        )
        monkeypatch.setattr(crash, "load_inputs", inputs)
        monkeypatch.setattr(crash, "candidate_events", cands)
        monkeypatch.setattr(crash, "score_candidate", lambda inp, c, p: c)
        monkeypatch.setattr(crash, "plan_clip", lambda i, c, p: {"start": c["t0"], "end": c["t1"]})
        monkeypatch.setattr(crash, "decode_window", decode)
        monkeypatch.setattr(crash, "crash_prompt_scores", lambda emb, p: emb[:, 0])
        monkeypatch.setattr(crash, "view_change", lambda emb: np.zeros(len(emb)))
        params = replace(crash.CrashParams(), pos=("a",), neg=("b",), words=())  # no profile
        return crash.detect_project(project, params=params, sheets=sheets, encoder=enc)

    return go


def test_crash_dense_check_skips_a_source_whose_decode_fails(dense_project, caplog):
    with caplog.at_level(logging.WARNING), jsonout.run("crashes") as result:
        out = dense_project(["ridge", "gully", "bowl"], {"gully"})
    got = {Path(c["source"]).name: c for c in json.loads(out.read_text("utf-8"))["crashes"]}
    assert set(got) == {"ridge.mp4", "gully.mp4", "bowl.mp4"}  # every source keeps its crash
    assert got["ridge.mp4"]["confirmed"] and got["bowl.mp4"]["confirmed"]
    assert not got["gully.mp4"]["confirmed"]  # unconfirmed, not lost
    message = "gully.mp4: crash dense check skipped this source (gully.mp4: no such file)"
    assert _warned(caplog) == [message]
    gully = str(Path(got["gully.mp4"]["source"]))
    assert _step_failed(result) == [(message, [gully])]


@pytest.mark.parametrize("vanish", [False, True])
def test_a_source_whose_dense_check_fails_gets_no_sheet_and_crashes_json_is_written(
    dense_project, caplog, vanish
):
    """The default 15 contact sheets skip the source (a file gone since is not saved to)."""
    with caplog.at_level(logging.WARNING), jsonout.run("crashes") as result:
        out = dense_project(["ridge", "gully", "bowl"], {"gully"}, sheets=15, vanish=vanish)
    got = {Path(c["source"]).name: c for c in json.loads(out.read_text("utf-8"))["crashes"]}
    assert set(got) == {"ridge.mp4", "gully.mp4", "bowl.mp4"}
    assert "sheet" not in got["gully.mp4"]
    assert Path(got["ridge.mp4"]["sheet"]).is_file() and Path(got["bowl.mp4"]["sheet"]).is_file()
    assert [f for _, f in _step_failed(result)] == [[got["gully.mp4"]["source"]]]  # warned once
    assert len(_warned(caplog)) == 1


def test_a_source_whose_sheet_fails_keeps_its_crash(dense_project, monkeypatch, caplog):
    """Without the dense check the sheet's decode is the first to fail: warned once, and
    crashes.json is still written."""
    real = crash.write_sheet

    def write_sheet(c, out, *a, **k):
        if Path(c["source"]).stem == "gully":
            raise OSError("moov atom not found")
        return real(c, out, *a, **k)

    monkeypatch.setattr(crash, "write_sheet", write_sheet)
    with caplog.at_level(logging.WARNING), jsonout.run("crashes") as result:
        out = dense_project(["ridge", "gully"], set(), sheets=15)
    got = {Path(c["source"]).name: c for c in json.loads(out.read_text("utf-8"))["crashes"]}
    assert "sheet" not in got["gully.mp4"] and "sheet" in got["ridge.mp4"]
    message = "gully.mp4: crash contact sheets skipped this source (moov atom not found)"
    assert _warned(caplog) == [message]
    assert _step_failed(result) == [(message, [got["gully.mp4"]["source"]])]


def test_crash_dense_check_fails_when_every_source_fails(dense_project, tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file"):
        dense_project(["ridge", "gully"], {"ridge", "gully"})
    assert not (tmp_path / "proj" / crash.CRASHES_FILE).exists()


def test_crash_dense_check_lets_a_memory_shortage_stop_the_run(dense_project):
    with pytest.raises(LowMemoryError):
        dense_project(["ridge", "gully"], {"ridge"}, LowMemoryError("short"))


# --------------------------------------------------------------------------- run_events batches
@pytest.mark.parametrize("layer", ["voice", "visual"])
def test_a_batch_of_one_bad_source_does_not_fail_the_layer(tmp_path, run_audio, monkeypatch, layer):
    """``--batch-size 1``: the bad file is a batch on its own, the layer still ends."""
    srcs = [_source(tmp_path, n) for n in ("cave", "arch", "kelp")]
    done = (
        _voice(monkeypatch, {"arch"}) if layer == "voice" else _visual(monkeypatch, {"arch"}).done
    )
    with jsonout.run("events") as result:
        run_audio(srcs, set(), layers=(layer,), batch_size=1)
    assert done == ["cave.mp4", "kelp.mp4"]
    assert [f for _, f in _step_failed(result)] == [[str(srcs[1].path)]]


@pytest.mark.parametrize("layer", ["voice", "visual"])
def test_every_batch_failing_fails_the_layer(tmp_path, run_audio, monkeypatch, layer):
    srcs = [_source(tmp_path, n) for n in ("cave", "arch")]
    if layer == "voice":
        _voice(monkeypatch, {"cave", "arch"})
    else:
        _visual(monkeypatch, {"cave", "arch"})
    with pytest.raises((ValueError, OSError), match=r"truncated wav|moov atom"):
        run_audio(srcs, set(), layers=(layer,), batch_size=1)


def test_a_failed_visual_layer_frees_the_clip_it_handed_to_the_crash_layer(
    tmp_path, run_audio, monkeypatch
):
    from cutpoint import crash as crash_mod  # noqa: PLC0415

    srcs = [_source(tmp_path, n) for n in ("reef", "wreck")]
    _visual(monkeypatch, {"reef", "wreck"})
    closed: list[int] = []
    enc = SimpleNamespace(close=lambda: closed.append(1))
    monkeypatch.setattr(crash_mod.ClipEncoder, "from_visual", classmethod(lambda cls, m: enc))
    monkeypatch.setattr(crash_mod, "detect_project", lambda *a, **k: pytest.fail("crash layer ran"))
    with pytest.raises(OSError, match="moov atom"):
        run_audio(srcs, set(), layers=("visual", "crash"))
    assert closed == [1]


# --------------------------------------------------------------------------- out of memory
def _oom() -> RuntimeError:
    return RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 7.79 GiB total)")


def _stopped(info, what: str, name: str) -> None:
    """The run stopped with a resumable LowMemoryError chained to the GPU's error."""
    assert str(info.value).startswith(f"Out of memory in {what} ({name}); close other programs")
    assert isinstance(info.value.__cause__, RuntimeError)


def test_out_of_memory_in_the_telemetry_read_stops_the_run(tmp_path, monkeypatch):
    srcs = [tmp_path / f"{n}.mp4" for n in ("ridge", "gully")]
    monkeypatch.setattr(actioncam, "extract", _extract({"ridge"}, _oom()))
    with pytest.raises(LowMemoryError) as info:
        actioncam.prepare(srcs, {str(s): N / 10 for s in srcs}, tmp_path)
    _stopped(info, "the telemetry read", "ridge.mp4")


def test_out_of_memory_in_the_contact_sheets_stops_the_run(tmp_path, monkeypatch, sheet_fakes):
    srcs = [_source(tmp_path, n) for n in ("reef", "wreck")]
    monkeypatch.setattr(telemetry, "extract", _extract({"reef"}, _oom()))
    with pytest.raises(LowMemoryError) as info:
        sheets.write_visual_sheets(tmp_path, srcs)
    _stopped(info, "the contact sheets", "reef.mp4")
    assert sheet_fakes == {}


def test_out_of_memory_in_the_telemetry_coverage_stops_the_run(tmp_path, monkeypatch):
    srcs = [_source(tmp_path, n) for n in ("ridge", "gully")]
    data = {"sources": {str(s.path): {"events": []} for s in srcs}}
    (tmp_path / "telemetry.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(telemetry, "extract", _extract({"ridge"}, _oom()))
    with pytest.raises(LowMemoryError) as info:
        pipeline._telemetry_layer(tmp_path, srcs, pipeline.VlmParams())
    _stopped(info, "Telemetry coverage", "ridge.mp4")


def test_out_of_memory_in_the_audio_layer_stops_the_run(tmp_path, run_audio):
    srcs = [_source(tmp_path, n) for n in ("cave", "arch")]
    with pytest.raises(LowMemoryError) as info:
        run_audio(srcs, {"cave"}, _oom())
    _stopped(info, "Audio events", "cave.mp4")


def test_out_of_memory_in_the_voice_layer_stops_the_run(tmp_path, monkeypatch):
    srcs = [_source(tmp_path, n) for n in ("cave", "arch")]
    done = _voice(monkeypatch, {"cave"}, _oom())
    with pytest.raises(LowMemoryError) as info:
        voice.run_layer(srcs, voice.VoiceParams(), det=_Det(), errors=[])
    _stopped(info, "Voice layer", "cave.mp4")
    assert done == []  # arch is not computed after it


def test_out_of_memory_in_the_visual_layer_stops_the_run(tmp_path, monkeypatch):
    srcs = [_source(tmp_path, n) for n in ("reef", "wreck")]
    st = _visual(monkeypatch, {"reef"}, _oom())
    with pytest.raises(LowMemoryError) as info:
        visual.run_layer(srcs, visual.VisualParams(), errors=[], handoff=st.handed.append)
    _stopped(info, "Visual layer", "reef.mp4")
    assert st.done == [] and st.handed == [] and st.closed == 1


def test_out_of_memory_in_the_crash_inputs_stops_the_run(dense_project, tmp_path):
    with pytest.raises(LowMemoryError) as info:
        dense_project(["ridge", "gully"], set(), _oom(), bad_inputs=frozenset({"ridge"}))
    _stopped(info, "crash detection", "ridge.mp4")
    assert not (tmp_path / "proj" / crash.CRASHES_FILE).exists()


def test_out_of_memory_in_the_crash_dense_check_stops_the_run(dense_project, tmp_path):
    with pytest.raises(LowMemoryError) as info:
        dense_project(["ridge", "gully"], {"ridge"}, _oom())
    _stopped(info, "the crash dense check", "ridge.mp4")
    assert not (tmp_path / "proj" / crash.CRASHES_FILE).exists()


def test_out_of_memory_in_the_crash_sheets_stops_the_run(dense_project, monkeypatch, tmp_path):
    def write_sheet(c, out, *a, **k):
        raise _oom()

    monkeypatch.setattr(crash, "write_sheet", write_sheet)
    with pytest.raises(LowMemoryError) as info:
        dense_project(["ridge", "gully"], set(), sheets=15)
    assert "Out of memory in the crash contact sheets" in str(info.value)
    assert not (tmp_path / "proj" / crash.CRASHES_FILE).exists()


# --------------------------------------------------------------------------- crash dense cache
def test_a_source_gone_before_its_dense_cache_is_read_is_skipped(dense_project, caplog):
    with caplog.at_level(logging.WARNING), jsonout.run("crashes") as result:
        out = dense_project(["ridge", "gully"], set(), sheets=15, gone=frozenset({"gully"}))
    got = {Path(c["source"]).name: c for c in json.loads(out.read_text("utf-8"))["crashes"]}
    assert got["ridge.mp4"]["confirmed"] and not got["gully.mp4"]["confirmed"]
    assert "sheet" not in got["gully.mp4"]
    (message,) = _warned(caplog)
    assert message.startswith("gully.mp4: crash dense check skipped this source (")
    assert [f for _, f in _step_failed(result)] == [[got["gully.mp4"]["source"]]]
