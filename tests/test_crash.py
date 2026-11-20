"""Crash detection decisions on small synthetic IMU / GPS / keyframe inputs."""

from __future__ import annotations

import functools
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint import activity, config, crash, insta360, stage, telemetry

N = 600  # 60 s on the 0.1 s grid


def _inputs(*, spike: bool) -> crash.SourceInputs:
    acc = np.zeros(N)
    speed = np.full(N, 30.0)  # skiing at 30 km/h ...
    kf = np.zeros(60)  # ... one keyframe per second, none looks like a crash
    if spike:
        acc[300:303] = 8.0  # 8 g impact at t = 30 s
        speed[303:] = 0.0  # then standing still
        kf[30:34] = 0.95  # and the camera looks like a crash
    imu = {k: np.zeros(N) for k in ("gyro_dps", "offaxis_deg", "sat")}
    imu["acc_g"] = acc
    tel = SimpleNamespace(n=N, grid={"speed_kmh": speed, "lift_mask": np.zeros(N)})
    visual = SimpleNamespace(t=np.arange(60, dtype=float))
    return crash.SourceInputs(
        path=Path("a.mp4"), duration=60.0, tel=tel, imu=imu, visual=visual, kf_crash=kf
    )


def test_impact_with_stop_and_visual_change_is_a_candidate_above_the_threshold():
    p = crash.CrashParams()
    inp = _inputs(spike=True)
    (cand,) = crash.candidate_events(inp, p)
    assert cand["origin"] == "telemetry" and 29.5 <= cand["t0"] <= cand["t1"] <= 31.0
    scored = crash.score_candidate(inp, cand, p)
    assert scored["pre_score"] >= p.prefilter
    clip = crash.plan_clip(inp, cand, p)
    assert clip["start"] == cand["t0"] - p.lead_s  # the approach is kept
    assert clip["end"] >= cand["t1"] + p.min_after_s  # and the aftermath


def test_quiet_window_has_no_candidates():
    assert crash.candidate_events(_inputs(spike=False), crash.CrashParams()) == []


@pytest.mark.parametrize(("ride_start_s", "damped"), [(32, True), (45, False)])
def test_impact_just_before_a_lift_ride_is_damped_without_gps(ride_start_s, damped):
    """Boarding knocks (360, no GPS lift mask): CLIP lift rides damp events up to 5 s before."""
    p = crash.CrashParams()
    inp = _inputs(spike=True)
    (cand,) = crash.candidate_events(inp, p)
    base = crash.telemetry_features(inp, cand["t0"], cand["t1"], p)
    assert not base["on_lift"]
    inp.lift = np.zeros(N)
    inp.lift[ride_start_s * 10 :] = 1.0
    tel = crash.telemetry_features(inp, cand["t0"], cand["t1"], p)
    assert tel["on_lift"] is damped
    assert tel["score"] == pytest.approx(base["score"] * (0.2 if damped else 1.0), abs=2e-3)


def _project(tmp_path: Path, *sources: Path) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    hl = {"sources": [{"path": str(s), "duration": 60.0} for s in sources]}
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    return project


def test_detect_project_all_footage_offline_fails_before_writing(tmp_path):
    project = _project(tmp_path, tmp_path / "gone.mp4")
    with pytest.raises(ValueError, match="No footage of proj is reachable"):
        crash.detect_project(project)
    assert not (project / "crashes.json").exists()
    assert not (project / "activity.json").exists()


def test_detect_project_skips_offline_source_and_processes_the_rest(tmp_path, monkeypatch, caplog):
    here = tmp_path / "here.mp4"
    here.write_bytes(b"x")
    project = _project(tmp_path, tmp_path / "gone.mp4", here)
    loaded = []
    monkeypatch.setattr(
        crash,
        "load_inputs",
        lambda src, dur, p, tmp, lang: loaded.append(src) or _inputs(spike=False),
    )
    monkeypatch.setattr(crash, "candidate_events", lambda inp, p: [])
    own_words = replace(crash.CrashParams(), pos=("a",), neg=("b",), words=())  # no profile
    with caplog.at_level("WARNING"):
        out = crash.detect_project(project, params=own_words, dense=False)
    assert loaded == [here]
    assert "gone.mp4: footage not reachable" in caplog.text
    assert json.loads(out.read_text(encoding="utf-8"))["crashes"] == []


def test_one_source_whose_telemetry_fails_is_skipped_and_the_others_get_crashes(
    tmp_path, monkeypatch, caplog
):
    """#985: telemetry, IMU or input loading that raises on one source skips only that source,
    with one WARNING naming it and a ``step_failed`` JSON warning; the rest get their crashes."""
    from cutpoint import jsonout, mix, sources  # noqa: PLC0415
    from cutpoint.events import visual, voice  # noqa: PLC0415

    srcs = [tmp_path / f"{name}.mp4" for name in ("reef", "wreck", "cave", "drift")]
    for s in srcs:
        s.write_bytes(b"x")
    project = _project(tmp_path, *srcs)
    good = _inputs(spike=True)

    def extract(src, duration):
        if src.name == "wreck.mp4":
            raise TypeError("'list' object cannot be interpreted as an integer")
        return good.tel

    def imu_grids(src, duration, force=False, tmp_dir=None):
        if src.name == "cave.mp4":
            raise ValueError("bad IMU block")
        return good.imu

    monkeypatch.setattr(telemetry, "extract", extract)
    monkeypatch.setattr(crash, "imu_grids", imu_grids)
    monkeypatch.setattr(voice, "load_voice", lambda src: None)
    monkeypatch.setattr(visual, "load_visual", lambda src: good.visual)
    monkeypatch.setattr(sources, "source_words", lambda src, lang: None)
    monkeypatch.setattr(mix, "lift_rides", lambda src, n: None)
    monkeypatch.setattr(crash, "crash_prompt_scores", lambda emb, p: good.kf_crash)
    monkeypatch.setattr(crash, "_flip_deg", lambda inp: np.zeros(N))
    good.visual.emb = None
    own_words = replace(crash.CrashParams(), pos=("a",), neg=("b",), words=())  # no profile
    with caplog.at_level("DEBUG", logger="cutpoint.crash"), jsonout.run("crashes") as result:
        out = crash.detect_project(project, params=own_words, dense=False, sheets=0)

    data = json.loads(out.read_text(encoding="utf-8"))
    # wreck (telemetry) and cave (IMU) are scored without what could not be read (A185):
    # wreck keeps its IMU and its crash, cave has only the transcript left
    assert sorted(Path(c["source"]).name for c in data["crashes"]) == [
        "drift.mp4",
        "reef.mp4",
        "wreck.mp4",
    ]
    warned = [
        r.getMessage()
        for r in caplog.records
        if r.levelname == "WARNING" and "telemetry not read" in r.getMessage()
    ]
    assert warned == [
        "wreck.mp4: telemetry not read ('list' object cannot be interpreted as an integer); "
        "scored without it",
        "cave.mp4: telemetry not read (bad IMU block); scored without it",
    ]
    tracebacks = [r for r in caplog.records if r.levelname == "DEBUG" and r.exc_info]
    assert [r.exc_info[0] for r in tracebacks] == [TypeError, ValueError]
    failed = [w for w in result.warnings if w["code"] == "step_failed"]
    assert [(w["message"], w["files"]) for w in failed] == [
        (warned[0], [str(srcs[1])]),
        (warned[1], [str(srcs[2])]),
    ]


def test_when_every_source_fails_the_error_propagates_and_crashes_json_stays(
    tmp_path, monkeypatch, caplog
):
    """A project-wide failure (every source raises) fails the step: the last error is raised
    and the crashes.json of an earlier run is not replaced by an empty one."""
    srcs = [tmp_path / f"{name}.mp4" for name in ("kelp", "arch")]
    for s in srcs:
        s.write_bytes(b"x")
    project = _project(tmp_path, *srcs)
    earlier = '{"crashes": [{"id": "c001"}]}'
    (project / "crashes.json").write_text(earlier, encoding="utf-8")

    def load_inputs(src, *a):
        raise RuntimeError("no CLIP weights")

    monkeypatch.setattr(crash, "load_inputs", load_inputs)
    own_words = replace(crash.CrashParams(), pos=("a",), neg=("b",), words=())
    with caplog.at_level("WARNING"), pytest.raises(RuntimeError) as exc:
        crash.detect_project(project, params=own_words, dense=False, sheets=0)
    assert str(exc.value) == "no CLIP weights"
    assert not [r for r in caplog.records if r.levelname == "ERROR"]  # the CLI logs the error
    final = [r for r in caplog.records if "for every source" in r.getMessage()]
    assert [(r.levelname, r.getMessage()) for r in final] == [
        ("WARNING", "crash detection failed for every source (last: arch.mp4: no CLIP weights)")
    ]
    assert (project / "crashes.json").read_text(encoding="utf-8") == earlier
    assert sum("crash detection skipped this source" in r.getMessage() for r in caplog.records) == 2


def test_when_every_source_fails_the_original_exception_keeps_its_attributes(tmp_path, monkeypatch):
    """cli.py maps a ModuleNotFoundError by its ``name`` (missing_extra): it is raised as is."""
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    project = _project(tmp_path, src)

    def load_inputs(*a):
        raise ModuleNotFoundError("No module named 'open_clip'", name="open_clip")

    monkeypatch.setattr(crash, "load_inputs", load_inputs)
    own_words = replace(crash.CrashParams(), pos=("a",), neg=("b",), words=())
    with pytest.raises(ModuleNotFoundError) as exc:
        crash.detect_project(project, params=own_words, dense=False, sheets=0)
    assert exc.value.name == "open_clip"


def test_a_low_memory_error_on_one_source_still_stops_the_run(tmp_path, monkeypatch):
    from cutpoint.errors import LowMemoryError  # noqa: PLC0415

    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    project = _project(tmp_path, src)

    def load_inputs(*a):
        raise LowMemoryError("short of memory")

    monkeypatch.setattr(crash, "load_inputs", load_inputs)
    own_words = replace(crash.CrashParams(), pos=("a",), neg=("b",), words=())
    with pytest.raises(LowMemoryError):
        crash.detect_project(project, params=own_words, dense=False, sheets=0)
    assert not (project / "crashes.json").exists()


class _FakeClip:
    def __init__(self):
        self.at = "cuda"
        self.moves = []

    def to(self, device):
        self.at = device
        self.moves.append(device)
        return self


class _FakeModels:
    def __init__(self):
        self.torch, self.device, self.dtype = object(), "cpu", "f16"
        self.clip, self.mean, self.std = _FakeClip(), "mean", "std"
        self.closed = 0

    def close(self):
        self.closed += 1


def test_clip_encoder_from_visual_shares_the_model_parks_it_and_double_close_is_safe(monkeypatch):
    def boom(self, *a, **k):
        raise AssertionError("CLIP must not be loaded again")

    monkeypatch.setattr(crash.ClipEncoder, "__init__", boom)
    monkeypatch.setattr("cutpoint.events.common.free_torch", lambda: None)
    models = _FakeModels()
    models.device = "cuda"
    enc = crash.ClipEncoder.from_visual(models)
    assert enc.model is models.clip and enc.device == "cuda" and enc.mean == "mean"
    assert models.clip.moves == ["cpu"] and not enc.on_device  # parked until the first embed
    empty = np.zeros((0, 8, 8, 3), np.uint8)  # no torch needed to trigger the move back
    assert enc.embed(empty).shape == (0, 3, 768)
    enc.embed(empty)
    assert models.clip.moves == ["cpu", "cuda"] and enc.on_device  # moved back once
    enc.close()
    enc.close()
    assert enc.model is None


class _FakeTensor:
    def __init__(self, at: str) -> None:
        self.at = at

    def to(self, device):
        return _FakeTensor(device)


@pytest.mark.parametrize(("vram", "device"), [(8.0, "cuda"), (0.2, "cpu")])
def test_the_parked_clip_goes_back_to_the_gpu_only_when_its_vram_is_free(
    mem, monkeypatch, caplog, vram, device
):
    """#721: the move of the handed-over CLIP to the GPU is checked as a fresh load
    (memguard.device_for): short of VRAM by then, it runs on the CPU in float32."""
    monkeypatch.setattr("cutpoint.events.common.free_torch", lambda: None)
    mem.vram = vram
    models = _FakeModels()
    models.device, models.dtype = "cuda", "f16"
    models.torch = SimpleNamespace(float32="f32")
    models.mean, models.std = _FakeTensor("cuda"), _FakeTensor("cuda")
    moves = []
    models.clip.to = lambda *a: moves.append(a) or models.clip
    enc = crash.ClipEncoder.from_visual(models)
    with caplog.at_level("WARNING"):
        enc.embed(np.zeros((0, 8, 8, 3), np.uint8))
    assert enc.device == device and enc.mean.at == enc.std.at == device
    if device == "cuda":
        assert moves == [("cpu",), ("cuda",)] and enc.dtype == "f16"
        assert "running on the CPU" not in caplog.text
    else:
        assert moves == [("cpu",), ("cpu", "f32")] and enc.dtype == "f32"
        assert "CLIP: 0.2 GB VRAM free, 1.0 GB needed; running on the CPU" in caplog.text


def test_a_parked_clip_on_the_cpu_is_not_checked_again(monkeypatch):
    monkeypatch.setattr(crash.memguard, "device_for", lambda *a, **k: pytest.fail("checked"))
    models = _FakeModels()  # device cpu
    enc = crash.ClipEncoder.from_visual(models)
    enc.embed(np.zeros((0, 8, 8, 3), np.uint8))
    assert enc.device == "cpu" and enc.on_device


@pytest.mark.parametrize(
    ("tel", "expected"),
    [
        ({"speed_before_kmh": 71, "speed_min_after_kmh": 0.4, "on_lift": False}, True),
        ({"speed_before_kmh": 30, "speed_min_after_kmh": 0.4, "on_lift": False}, False),
        ({"speed_before_kmh": 71, "speed_min_after_kmh": 8, "on_lift": False}, False),
        ({"speed_before_kmh": 71, "speed_min_after_kmh": 0.4, "on_lift": True}, False),
        ({"speed_before_kmh": None, "speed_min_after_kmh": None, "on_lift": False}, False),
    ],
)
def test_speed_drop_only_for_a_fast_rider_stopped_dead(tel, expected):
    cand = {"telemetry": tel}
    assert crash._speed_drop(cand, crash.CrashParams()) is expected


def _cand(name, pre, before, after, on_lift=False):
    tel = {"speed_before_kmh": before, "speed_min_after_kmh": after, "on_lift": on_lift}
    return {
        "name": name,
        "source": "s.mp4",
        "t0": 10.0,
        "t1": 12.0,
        "origin": "telemetry",
        "pre_score": pre,
        "telemetry": tel,
        "scores": {"telemetry": 0, "visual": 0, "audio": 0, "words": 0},
    }


def test_detect_project_dense_checks_speed_drops_below_prefilter(tmp_path, monkeypatch):
    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    project = _project(tmp_path, src)
    cands = [
        _cand("high", 0.5, 10, 10),
        _cand("drop1", 0.161, 71, 0.4),
        _cand("slow", 0.16, 30, 0.4),
        _cand("notstopped", 0.15, 71, 8),
        _cand("lift", 0.14, 71, 0.4, on_lift=True),
        *[_cand(f"drop{k}", 0.1, 80, 0.0) for k in range(2, 9)],
    ]
    for c in cands:
        c["source"] = str(src)
    checked, ends = [], []
    enc = SimpleNamespace(close=lambda: ends.append(1))
    monkeypatch.setattr(crash, "load_inputs", lambda *a: _inputs(spike=False))
    monkeypatch.setattr(crash, "candidate_events", lambda inp, p: cands)
    monkeypatch.setattr(crash, "score_candidate", lambda inp, c, p: c)
    monkeypatch.setattr(crash, "plan_clip", lambda i, c, p: {"start": c["t0"], "end": c["t1"]})
    monkeypatch.setattr(crash, "ClipEncoder", lambda: pytest.fail("loaded a new encoder"))

    def dense(e, inp, c, p, store):
        assert e is enc
        checked.append(c["name"])
        return {"score": 0.5}

    monkeypatch.setattr(crash, "dense_feature", dense)
    own_words = replace(crash.CrashParams(), pos=("a",), neg=("b",), words=())
    crash.detect_project(project, params=own_words, encoder=enc, sheets=0)
    p = crash.CrashParams()
    assert checked == ["high", "drop1", *[f"drop{k}" for k in range(2, 1 + p.drop_top)]]
    assert ends == [1]


def test_detect_project_dense_zero_never_lowers_a_speed_drop_bypass(tmp_path, monkeypatch):
    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    project = _project(tmp_path, src)
    normal, drop = _cand("normal", 0.5, 10, 10), _cand("drop", 0.161, 71, 0.4)
    drop["t0"], drop["t1"] = 40.0, 42.0  # outside the other clip, so it is not absorbed
    for c in (normal, drop):
        c["source"] = str(src)
        c["scores"].update(telemetry=0.8, visual=0.6)
    enc = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(crash, "load_inputs", lambda *a: _inputs(spike=False))
    monkeypatch.setattr(crash, "candidate_events", lambda inp, p: [normal, drop])
    monkeypatch.setattr(crash, "score_candidate", lambda inp, c, p: c)
    monkeypatch.setattr(crash, "plan_clip", lambda i, c, p: {"start": c["t0"], "end": c["t1"]})
    pre = {
        c["name"]: crash._confidence(c, dict(crash.CrashParams().weights)) for c in (normal, drop)
    }
    monkeypatch.setattr(crash, "dense_feature", lambda e, inp, c, p, store: {"score": 0.0})
    own_words = replace(crash.CrashParams(), pos=("a",), neg=("b",), words=())
    crash.detect_project(project, params=own_words, encoder=enc, sheets=0)
    got = {
        c["name"]: c
        for c in json.loads((project / "crashes.json").read_text(encoding="utf-8"))["crashes"]
    }
    assert got["drop"]["confirmed"] and got["normal"]["confirmed"]
    assert got["drop"]["confidence"] >= round(pre["drop"], 3)  # the bypass is never penalised
    assert got["normal"]["confidence"] < round(pre["normal"], 3)  # a normal candidate still drops


@pytest.fixture
def dense_run(tmp_path, monkeypatch):
    """``detect_project`` over four candidates with distinct dense windows, the real dense
    store and :func:`crash.dense_feature`; the decode and the CLIP embedding are fakes.
    ``state.stop_at`` interrupts the decode of that window (KeyboardInterrupt)."""
    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    project = _project(tmp_path, src)
    state = SimpleNamespace(src=src, project=project, decoded=[], stop_at=None, loads=0, closes=0)
    starts = (10.0, 40.0, 70.0, 100.0)  # 4 s before, 6 s after: no two windows overlap

    def cands(inp, p):
        out = []
        for k, t0 in enumerate(starts):
            c = _cand(f"c{k}", 0.9 - 0.1 * k, 10, 10)  # strongest first: c0 .. c3
            c.update(source=str(src), t0=t0, t1=t0 + 2.0)
            out.append(c)
        return out

    def decode(path, a, b, fps):
        if a == state.stop_at:
            raise KeyboardInterrupt
        state.decoded.append(a)
        ts = np.arange(a, b, 1.0)
        return ts, np.zeros((len(ts), 2, 2, 3), np.uint8)

    class Enc:
        def __init__(self):
            state.loads += 1

        def embed(self, frames):
            n = len(frames)
            return np.tile(np.linspace(0.1, 0.9, n)[:, None, None], (1, 3, 4))

        def close(self):
            state.closes += 1

    monkeypatch.setattr(crash, "load_inputs", lambda *a: _inputs(spike=False))
    monkeypatch.setattr(crash, "candidate_events", cands)
    monkeypatch.setattr(crash, "score_candidate", lambda inp, c, p: c)
    monkeypatch.setattr(crash, "plan_clip", lambda i, c, p: {"start": c["t0"], "end": c["t1"]})
    monkeypatch.setattr(crash, "decode_window", decode)
    monkeypatch.setattr(crash, "crash_prompt_scores", lambda emb, p: emb[:, 1, 0])
    monkeypatch.setattr(crash, "ClipEncoder", Enc)
    params = replace(crash.CrashParams(), pos=("a",), neg=("b",), words=())
    state.store = crash._dense_cache(src, params)
    state.params = params
    state.go = lambda **kw: crash.detect_project(project, params=params, sheets=0, **kw)
    state.windows = [round(max(0.0, t - params.dense_pre_s), 1) for t in starts]
    return state


def _crashes(project: Path) -> dict:
    data = json.loads((project / crash.CRASHES_FILE).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k != "timings"}


def test_dense_windows_are_saved_as_they_finish_and_a_rerun_computes_only_the_rest(
    dense_run, caplog
):
    """#109: Ctrl+C after window k keeps windows 1..k; the re-run decodes only k+1.. and
    ends with the same dense store and crashes.json as an uninterrupted run."""
    w = dense_run.windows
    dense_run.go()  # uninterrupted
    full, crashes = dense_run.store.read_bytes(), _crashes(dense_run.project)
    assert dense_run.decoded == w
    dense_run.store.unlink()
    dense_run.decoded.clear()

    dense_run.stop_at = w[2]  # interrupted after the second window
    with caplog.at_level("INFO"), pytest.raises(KeyboardInterrupt):
        dense_run.go()
    assert dense_run.decoded == w[:2]
    assert list(json.loads(dense_run.store.read_text(encoding="utf-8"))) == [
        f"{a:.1f}-{a + 12.0:.1f}" for a in w[:2]
    ]
    assert "continuing" not in caplog.text  # a fresh run looks as before

    dense_run.stop_at, dense_run.loads = None, 0
    dense_run.decoded.clear()
    caplog.clear()
    with caplog.at_level("INFO"):
        dense_run.go()
    assert dense_run.decoded == w[2:]
    assert "dense check: 2/4 done, continuing" in caplog.messages
    assert dense_run.store.read_bytes() == full
    assert _crashes(dense_run.project) == crashes

    dense_run.decoded.clear()
    dense_run.loads = 0
    caplog.clear()
    with caplog.at_level("INFO"):
        dense_run.go()  # everything cached: no model, no decode, nothing to continue
    assert dense_run.decoded == [] and dense_run.loads == 0
    assert not any("continuing" in m for m in caplog.messages)


@pytest.mark.parametrize("size", [0, 1, 3, 4])
def test_batch_size_closes_clip_every_n_dense_windows_with_the_same_crashes(
    dense_run, caplog, size
):
    """#109 --batch-size: CLIP is loaded ceil(4/N) times and closed as often; same output."""
    dense_run.go()
    full, crashes = dense_run.store.read_bytes(), _crashes(dense_run.project)
    dense_run.store.unlink()
    dense_run.loads = dense_run.closes = 0
    with caplog.at_level("INFO"):
        dense_run.go(batch_size=size)
    want = -(-4 // size) if size else 1
    assert dense_run.loads == dense_run.closes == want
    assert dense_run.store.read_bytes() == full and _crashes(dense_run.project) == crashes
    done = [m for m in caplog.messages if m.startswith("dense check: ")]
    want_done = range(size, 4, size) if size else []
    assert done == [f"dense check: {k}/4 done, continuing" for k in want_done]


@pytest.mark.parametrize("stop", [2, 1], ids=["at-boundary", "inside"])
def test_batch_size_interrupted_dense_check_resumes_to_the_same_crashes(dense_run, stop):
    w = dense_run.windows
    dense_run.go()
    full, crashes = dense_run.store.read_bytes(), _crashes(dense_run.project)
    dense_run.store.unlink()
    dense_run.decoded.clear()
    dense_run.stop_at = w[stop]
    with pytest.raises(KeyboardInterrupt):
        dense_run.go(batch_size=2)
    dense_run.stop_at = None
    dense_run.decoded.clear()
    dense_run.go(batch_size=2)
    assert dense_run.decoded == w[stop:]
    assert dense_run.store.read_bytes() == full and _crashes(dense_run.project) == crashes


def test_dense_batch_boundaries_update_the_crashes_step_counts(dense_run, monkeypatch):
    """#721: inside cutpoint crashes' quiet run, the steps run in order and each dense batch
    boundary updates the dense counts; crashes.json is the same as without the run."""
    from cutpoint import progress  # noqa: PLC0415

    dense_run.go(batch_size=2)
    crashes = _crashes(dense_run.project)
    dense_run.store.unlink()
    seen = []
    obs = lambda step, state, pct, done, total: seen.append((step, state, done, total))  # noqa: E731
    monkeypatch.setattr(progress.Run, "observers", [obs])
    with progress.run(progress.CRASHES_STEPS, quiet=True):
        dense_run.go(batch_size=2)
    assert _crashes(dense_run.project) == crashes
    assert [s for s in seen if s[1] != "running"] == [
        ("profile", "skipped", None, None),  # the params bring their own words
        ("inputs", "done", None, None),
        ("dense", "done", None, None),
        ("result", "done", None, None),
    ]
    counts = [(d, t) for step, _, d, t in seen if step == "dense" and d is not None]
    assert counts == [(0, 4), (2, 4)]


def test_no_dense_check_records_no_counts(dense_run, monkeypatch):
    """#721 F1: with --no-dense no dense counts land on another step."""
    from cutpoint import progress  # noqa: PLC0415

    seen = []
    obs = lambda step, state, pct, done, total: seen.append((step, state, done, total))  # noqa: E731
    monkeypatch.setattr(progress.Run, "observers", [obs])
    with progress.run(progress.CRASHES_STEPS, quiet=True):
        crash.detect_project(dense_run.project, params=dense_run.params, sheets=0, dense=False)
    assert all(d is None and t is None for _, _, d, t in seen)
    assert ("dense", "skipped", None, None) in seen and dense_run.loads == 0


# -- IMU parse shared with telemetry ----------------------------------------------------------


# ---------------------------------------------------------------- the generic floor (#932)
@pytest.fixture
def captions(monkeypatch):
    """Fake CLIP text tower: one embedding axis per caption. ``frame(*captions)`` is a
    ``(1, 3, 768)`` keyframe embedding (three equal crops) between those captions."""
    axes: dict[str, int] = {}

    def text_embeddings(prompts, models=None, *, save=True):
        e = np.zeros((len(prompts), 768), np.float32)
        for k, s in enumerate(prompts):
            e[k, axes.setdefault(s, len(axes))] = 1.0
        return e

    def frame(*names: str, weights: tuple[float, ...] | None = None) -> np.ndarray:
        v = np.zeros(768, np.float32)
        for n, w in zip(names, weights or (1.0,) * len(names), strict=True):
            v[axes.setdefault(n, len(axes))] = w
        v /= np.linalg.norm(v)
        return np.tile(v, (1, 3, 1))

    monkeypatch.setattr("cutpoint.events.visual.text_embeddings", text_embeddings)
    return frame


def _profile_params(raw: dict) -> crash.CrashParams:
    return crash.CrashParams().with_profile(activity.full(activity.validate(raw)))


def _generic_params() -> crash.CrashParams:
    return crash.CrashParams().with_profile(activity.full(activity.library()["generic"]))


def test_a_profile_without_negatives_no_longer_saturates(captions):
    """The no-model fallback (the floor's captions, no negatives) scored every frame 1.0."""
    floor = activity.base()["incident"]
    assert floor["pos"] and not floor["neg"]
    calm = captions(activity.builtin("generic")["incident"]["neg"][0])
    old = replace(crash.CrashParams(), pos=tuple(floor["pos"]), neg=(), words=())
    assert crash.crash_prompt_scores(calm, old)[0] == pytest.approx(1.0)  # the saturation
    bare = _profile_params({"name": "bare", "incident": {"pos": floor["pos"]}})
    assert bare.neg == bare.floor_neg and bare.neg
    assert crash.crash_prompt_scores(calm, bare)[0] < 0.1
    # an activity.json written before the fix: its stored profile has no negatives
    stored = activity.full(activity.validate({"name": "bare"}))
    stored["incident"] = {**stored["incident"], "neg": []}
    rec = {"version": 1, "name": "bare", "source": "synthesized", "profile": stored}
    old_rec = crash.CrashParams().with_profile(activity.record_profile(rec))
    assert crash.crash_prompt_scores(calm, old_rec)[0] < 0.1


def test_a_profile_with_on_domain_negatives_never_scores_below_generic(captions, monkeypatch):
    """Synthesized negatives (normal footage of the same activity) sit close to the crash
    frames: alone they score a fall low; with the floor it is at least generic's share."""
    tumble = activity.base()["incident"]["pos"][1]  # "a blurry tumbling view"
    near = "a blurry fast ride down a rough trail"  # an on-domain negative
    raw = {"name": "rough-ride", "incident": {"pos": ["a rider falling off"], "neg": [near]}}
    prof, generic = _profile_params(raw), _generic_params()
    assert prof.floor_pos == generic.pos and prof.floor_neg == generic.neg
    frames = np.concatenate(
        [
            captions(tumble, near, weights=(1.0, 1.0)),  # a fall: like the negative too
            captions(near),  # normal riding
            captions("a rider falling off", tumble),
        ]
    )
    alone = replace(prof, floor_pos=None, floor_neg=None)
    got = crash.crash_prompt_scores(frames, prof)
    ref = crash.crash_prompt_scores(frames, generic)
    assert crash.crash_prompt_scores(frames, alone)[0] < ref[0]  # the problem of #932
    assert (got >= ref - 1e-6).all() and (got >= crash.crash_prompt_scores(frames, alone)).all()
    # the dense confirm reads the same share per frame
    monkeypatch.setattr(crash, "decode_window", lambda *a: (np.arange(len(frames)) + 9.0, frames))
    enc = SimpleNamespace(embed=lambda f: f)
    c = {"t0": 10.0, "t1": 11.0}
    inp = _inputs(spike=False)
    stores: dict[str, dict] = {"prof": {}, "generic": {}, "alone": {}}
    for k, p in (("prof", prof), ("generic", generic), ("alone", alone)):
        crash.dense_feature(enc, inp, c, p, stores[k])
    cs = {k: np.asarray(next(iter(s.values()))["crash"]) for k, s in stores.items()}
    assert (cs["prof"] >= cs["generic"] - 1e-3).all()
    assert cs["prof"][0] > cs["alone"][0] + 1e-3  # the floor lifts the stored fall share


def test_generic_itself_is_scored_once_and_keys_change_with_the_floor(captions):
    generic = _generic_params()
    assert (generic.pos, generic.neg) == (generic.floor_pos, generic.floor_neg)
    other = _profile_params({"name": "x", "incident": {"pos": ["a"], "neg": ["b"]}})
    assert other.dense_key() != replace(other, floor_pos=None, floor_neg=None).dense_key()
    own = replace(crash.CrashParams(), pos=("a",), neg=("b",), words=())
    assert own.resolved and own.floor_pos is None  # prompts given: no profile, no floor


def test_the_no_model_fallback_scores_exactly_like_generic(tmp_path, monkeypatch):
    """#916: without a model the profile is generic (CLIP's guess only in its description),
    so crash detection keys and scores it as ``--activity generic``."""
    v = np.zeros((1, 768), np.float32)
    v[0, 0] = 1.0
    monkeypatch.setattr(activity, "keyframes", lambda project, **kw: (v, []))
    monkeypatch.setattr(activity, "_text_emb", lambda p: np.eye(len(p), 768, dtype=np.float32))
    rec = activity.resolve(tmp_path, llm=False)
    got = crash.CrashParams().with_profile(activity.record_profile(rec))
    generic = _generic_params()
    assert rec["name"] == "generic" and rec["described_by"] == "clip"
    assert (got.key(), got.dense_key()) == (generic.key(), generic.dense_key())


def test_an_llm_crash_phrase_alone_makes_no_crash_candidate():
    """#916: LLM phrases are capped below the 0.8 a spoken cue needs to be a candidate of
    its own; the floor's phrases still are one."""
    words = {"flipped the raft": 1.0, "took a swim": 0.95}
    raw = {"name": "raft", "incident": {"pos": ["a flipped raft"], "words": words}}

    def ask(prompt, images, schema):
        return raw

    prof = activity.synthesize("rafting", None, ask, [])
    assert max(prof["incident"]["words"].values()) == activity.GEN_PHRASE_CAP < 0.8
    inp = _inputs(spike=False)
    inp.hits = [(30.0, w, k) for k, w in prof["incident"]["words"].items()]
    assert crash.candidate_events(inp, crash.CrashParams()) == []
    inp.hits = [(30.0, activity.base()["incident"]["words"]["wiped out"], "wiped out")]
    assert [c["origin"] for c in crash.candidate_events(inp, crash.CrashParams())] == ["words"]


def test_stale_dense_caches_are_not_read_and_do_not_break_the_check(tmp_path):
    """CRASH_VERSION 3: the dense store is a new file; an old one stays harmless."""
    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    p = _generic_params()
    stale = crash.cache.source_dir(src) / "crashdense-01769b39a2.json"  # a version 2 key
    stale.write_text(json.dumps({"1.0-11.0": {"t": [1.0], "crash": [1.0], "change": [0.0]}}))
    assert crash._dense_cache(src, p) != stale
    assert not crash._dense_cache(src, p).exists()
    assert crash.dense_needs_run([src]) is False  # any store counts: only a hint


def test_gravity_is_the_unit_low_passed_accelerometer() -> None:
    acc = np.tile([0.0, 9.8, 0.0], (50, 1))
    acc[25] = [0.0, 0.0, 9.8]  # one off-axis sample barely moves a 10-sample mean
    g = telemetry.gravity(acc, 10)
    assert np.allclose(np.linalg.norm(g, axis=1), 1.0)
    assert g[25, 1] > 0.99
    assert np.allclose(telemetry.gravity(acc, 0)[25], [0.0, 0.0, 1.0])  # k < 1: no smoothing


def test_parse_imu_drops_samples_without_both_sensors(monkeypatch) -> None:
    samples = [
        {"timestamp_ms": 5.0 * i, "accl": [0.0, 9.8, 0.0], "gyro": [1.0, 2.0, 3.0]}
        for i in range(12)
    ]
    samples[3]["gyro"] = None

    class Parser:
        def __init__(self, _path):
            pass

        def telemetry(self):
            return [{}]

        def normalized_imu(self):
            return samples

    monkeypatch.setitem(sys.modules, "telemetry_parser", SimpleNamespace(Parser=Parser))
    _, groups, t, acc, gyr = telemetry.parse_imu(Path("a.mp4"))
    assert groups == [{}] and acc.shape == gyr.shape == (11, 3)
    assert t[:4].tolist() == [0.0, 0.005, 0.01, 0.02]
    samples[4:] = []  # 3 usable samples: too few for the crash grids
    assert len(telemetry.parse_imu(Path("a.mp4"))[2]) == 3
    assert crash._compute_imu(Path("a.mp4"), 10.0) is None


@pytest.mark.parametrize(
    ("trailer", "staged", "remote", "read", "unstaged"),
    [
        ("/nas/VID_00_.insv", None, True, "/nas/VID_00_.insv", False),  # X3: the IMU trailer
        (None, "/local/GX01.mp4", True, "/local/GX01.mp4", False),  # the staged copy
        (None, None, True, "/nas/GX01.mp4", True),  # remote, read in place
        (None, None, False, "/nas/GX01.mp4", False),
    ],
)
def test_imu_read_path_prefers_the_x3_trailer_then_the_staged_copy(
    monkeypatch, trailer, staged, remote, read, unstaged
) -> None:
    src = Path("/nas/GX01.mp4")
    monkeypatch.setattr(config, "remap_source", Path)
    monkeypatch.setattr(insta360, "imu_source", lambda _p: trailer and Path(trailer))
    monkeypatch.setattr(stage, "staged_path", functools.cache(lambda p, repaired=True: staged or p))
    monkeypatch.setattr(stage, "is_remote", lambda _p: remote)
    io, x3_imu, read_path, got = telemetry.imu_read_path(src)
    assert (io, x3_imu, read_path, got) == (src, trailer and Path(trailer), Path(read), unstaged)


# ---------------------------------------------------------------- the stop after a GPS gap (A181)
def test_a_gps_gap_after_the_impact_does_not_move_the_stop_earlier():
    """#1085 A181: NaN speed bins between the impact and the stop shifted the stop (and the
    words and movement measured from it) one bin earlier each."""
    inp = _inputs(spike=True)
    speed = np.full(N, 30.0)
    speed[303:331] = np.nan  # no fix for 2.8 s after the impact
    speed[340:] = 0.0  # stopped at 34 s
    speed[500:] = 20.0  # moving again at 50 s
    inp.tel.grid["speed_kmh"] = speed
    inp.words = [
        {"start": 32.0, "end": 32.4, "word": "whoa"},  # still sliding: before the stop
        {"start": 45.0, "end": 45.5, "word": "ouch"},  # the first words after it
    ]
    c = {"t0": 30.0, "t1": 30.2}
    clip = crash.plan_clip(inp, c, crash.CrashParams())
    assert (clip["end"], clip["until"]) == (46.0, "first words after")
    inp.words = []
    clip = crash.plan_clip(inp, c, crash.CrashParams())
    assert (clip["end"], clip["until"]) == (50.0, "moving again")


# ---------------------------------------------------------------- portrait crops (A182)
def _pooling_encoder() -> crash.ClipEncoder:
    """A ClipEncoder whose fake model embeds a 224x224 crop as its 16x16 average pool."""
    torch = pytest.importorskip("torch")

    class Model:
        def encode_image(self, x):
            return torch.nn.functional.adaptive_avg_pool2d(x, 16).flatten(1) + 1.0

    enc = crash.ClipEncoder.__new__(crash.ClipEncoder)
    enc.torch, enc.device, enc.dtype, enc.model = torch, "cpu", torch.float32, Model()
    enc.mean, enc.std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    enc.on_device = True
    return enc


def _embed_before_a182(enc, frames: np.ndarray, batch: int, centre: bool) -> np.ndarray:
    """ClipEncoder.embed as it was before #1085 A182 (crops along the width only)."""
    torch = enc.torch
    crops = 1 if centre else 3
    out = []
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(frames[i : i + batch]).permute(0, 3, 1, 2).float() / 255
        b, _, h, w = x.shape
        s = min(h, w)
        if centre:
            y0, x0 = (h - s) // 2, (w - s) // 2
            c = x[:, :, y0 : y0 + s, x0 : x0 + s]
        else:
            c = torch.cat([x[:, :, :, o : o + s] for o in (0, (w - s) // 2, w - s)], 0)
        c = torch.nn.functional.interpolate(
            c, size=(224, 224), mode="bicubic", antialias=True, align_corners=False
        )
        c = ((c.clamp(0, 1) - enc.mean) / enc.std).to(enc.dtype)
        e = enc.model.encode_image(c).float()
        e = e / e.norm(dim=-1, keepdim=True)
        out.append(e.view(crops, b, -1).permute(1, 0, 2).numpy())
    return np.concatenate(out)


@pytest.mark.parametrize("centre", [False, True], ids=["three-crops", "centre"])
def test_landscape_embeddings_are_byte_identical_to_before(centre):
    enc = _pooling_encoder()
    frames = np.random.default_rng(7).integers(0, 256, (5, 36, 64, 3), dtype=np.uint8)
    got = enc.embed(frames, batch=2, centre=centre)
    assert got.tobytes() == _embed_before_a182(enc, frames, 2, centre).tobytes()


def test_portrait_frames_are_cropped_top_centre_bottom_like_the_visual_layer():
    """#1085 A182: a tall frame was squashed whole into all three crops."""
    from cutpoint.events.visual import square_crops  # noqa: PLC0415

    enc = _pooling_encoder()
    frames = np.zeros((2, 80, 40, 3), np.uint8)
    frames[:, :20, :, 0] = 255  # red top quarter
    frames[:, 60:, :, 2] = 255  # blue bottom quarter
    emb = enc.embed(frames, batch=1)
    top, mid, bottom = emb[0]
    assert not np.allclose(top, mid) and not np.allclose(mid, bottom)
    assert not np.allclose(top, bottom)
    assert np.allclose(enc.embed(frames, centre=True)[:, 0], emb[:, 1], atol=1e-6)
    # the crops VisualModels.run takes
    crops = square_crops(frames.transpose(0, 3, 1, 2))
    assert [c.shape for c in crops] == [(2, 3, 40, 40)] * 3
    assert crops[0][:, 0].max() == 255 and crops[0][:, 2].max() == 0  # top: red only
    assert crops[2][:, 2].max() == 255 and crops[2][:, 0].max() == 0  # bottom: blue only
    wide = square_crops(frames.transpose(0, 3, 2, 1))  # the same frames on their side
    assert all(np.array_equal(a, b.transpose(0, 1, 3, 2)) for a, b in zip(crops, wide, strict=True))


@pytest.mark.parametrize("portrait", [True, False], ids=["portrait", "landscape"])
def test_old_dense_windows_of_a_portrait_source_are_computed_again(tmp_path, monkeypatch, portrait):
    """#1085 A182: windows a dense store kept from before the portrait crops are dropped for a
    portrait source only; a landscape source keeps them, marked so it is not probed again."""
    from conftest import media_info  # noqa: PLC0415

    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    p = _generic_params()
    old = {"t": [1.0], "crash": [0.5], "change": [0.0]}
    new = {**old, "crops": crash.DENSE_CROPS}
    crash.cache.save_json(crash._dense_cache(src, p), {"1.0-11.0": old, "20.0-30.0": new})
    size = {"width": 1080, "height": 1920} if portrait else {}
    probes = []
    monkeypatch.setattr(
        "cutpoint.ffmpeg_utils.probe", lambda path: probes.append(path) or media_info(path, **size)
    )
    store = crash._load_dense(src, p)
    want = {"20.0-30.0": new} if portrait else {"1.0-11.0": new, "20.0-30.0": new}
    assert store == want and len(probes) == 1
    crash.cache.save_json(crash._dense_cache(src, p), store)
    assert crash._load_dense(src, p) == want and len(probes) == 1  # all marked: no probe


def test_a_dense_store_of_a_source_that_cannot_be_probed_is_kept(tmp_path, monkeypatch):
    from cutpoint.errors import FFmpegError  # noqa: PLC0415

    def probe(path):
        raise FFmpegError("no video stream")

    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    p = _generic_params()
    crash.cache.save_json(crash._dense_cache(src, p), {"1.0-11.0": {"t": [1.0]}})
    monkeypatch.setattr("cutpoint.ffmpeg_utils.probe", probe)
    store = crash._load_dense(src, p)
    assert list(store) == ["1.0-11.0"]
    assert "crops" not in store["1.0-11.0"]
    crash.cache.save_json(crash._dense_cache(src, p), [1, 2])  # not a store
    assert crash._load_dense(src, p) == {}


# ---------------------------------------------------------------- the IMU half (A490)
FS = 200.0
BINS = int(60.0 / crash.DT) + 1  # the grids of a 60 s source
GOPRO = [{"Accelerometer": {"Scale": [32767.0 / (16 * crash.G)]}}]  # +-16 g per axis
INSTA360 = [{"Default": {"Metadata": {}}}]


def _imu_stream() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """60 s of a camera ridden upright (gravity on +y): a 5 g bump at 10 s, a crash at 30 s
    (one axis at 32.3 g, saturated on every camera, then a 1 s roll at 300 deg/s) and lying on
    its side (gravity on +x) from 31 s on. Events start mid-bin."""
    t = np.arange(int(60 * FS)) / FS
    acc = np.tile([0.0, crash.G, 0.0], (len(t), 1))
    acc += np.random.default_rng(3).normal(0, 0.3, acc.shape)  # riding vibration
    acc[(t >= 10.05) & (t < 10.07), 0] = 5 * crash.G
    acc[(t >= 30.05) & (t < 30.07), 0] = 32.3 * crash.G
    acc[t >= 31.05] = [crash.G, 0.0, 0.0]
    gyr = np.zeros_like(acc)
    gyr[(t >= 30.05) & (t < 31.05), 2] = 300.0  # roll: perpendicular to gravity
    return t, acc, gyr


def _fake_parse(monkeypatch, groups, t, acc, gyr) -> list[Path]:
    calls = []

    def parse_imu(path, level=20):
        calls.append(path)
        return object(), groups, t, acc, gyr

    monkeypatch.setattr(telemetry, "parse_imu", parse_imu)
    return calls


@pytest.mark.parametrize("groups", [GOPRO, INSTA360, [{}]], ids=["gopro", "insta360", "no-scale"])
def test_imu_grids_bin_peaks_saturation_rotation_and_gravity(monkeypatch, groups):
    _fake_parse(monkeypatch, groups, *_imu_stream())
    g = crash._compute_imu(Path("a.mp4"), 60.0)
    assert all(len(v) == BINS for k, v in g.items() if k != "imu_hz")
    assert g["imu_hz"][0] == pytest.approx(FS, rel=1e-3)
    assert g["acc_g"][100] == pytest.approx(5.1, abs=0.05) and g["acc_g"][300] > 32
    assert np.flatnonzero(g["sat"]).tolist() == [300]  # the crash, not the bump
    assert g["gyro_dps"][300:311].min() == pytest.approx(300.0)
    assert g["rot_deg"][295:315].sum() == pytest.approx(300.0, rel=0.01)  # 1 s at 300 deg/s
    assert g["offaxis_deg"][295:315].sum() == pytest.approx(300.0, rel=0.01)
    assert g["offaxis_deg"][:295].max() == 0.0
    assert g["grav_y"][50:250].mean() == pytest.approx(1.0, abs=0.01)  # upright
    assert g["grav_x"][330:600].mean() == pytest.approx(1.0, abs=0.01)  # on its side


def test_a_crash_turns_the_camera_away_from_its_riding_orientation(monkeypatch):
    """The synthetic stream end to end: the IMU grids, the flip angle against the riding
    orientation (moving, from GPS) and the telemetry decision."""
    _fake_parse(monkeypatch, GOPRO, *_imu_stream())
    inp = _inputs(spike=False)
    inp.imu = crash._compute_imu(Path("a.mp4"), 60.0)
    inp.tel.grid["speed_kmh"][300:] = 0.0  # stopped at the crash
    inp.flip = crash._flip_deg(inp)
    assert inp.flip[200:290].max() < 2  # riding: the riding orientation
    assert inp.flip[330:600].min() == pytest.approx(90.0, abs=2)  # on its side
    p = crash.CrashParams()
    bump, hit = cands = crash.candidate_events(inp, p)
    assert [c["origin"] for c in cands] == ["telemetry", "telemetry"]
    assert bump["t0"] == 10.0 and 29.5 <= hit["t0"] <= 30.0
    bump, hit = (crash.telemetry_features(inp, c["t0"], c["t1"], p) for c in cands)
    assert hit["saturated"] == 1 and hit["flip_after_deg"] == pytest.approx(90, abs=2)
    assert hit["parts"]["flip"] == 1.0 and hit["parts"]["stop"] == 1.0
    assert hit["score"] >= 0.5 > bump["score"]
    assert bump["saturated"] == 0 and bump["parts"]["flip"] == 0.0


def test_flip_without_gps_is_measured_against_the_whole_clip():
    inp = _inputs(spike=False)
    inp.tel = None
    grav = np.zeros((N, 3))
    grav[:, 1] = 1.0
    grav[500:] = [0.0, -1.0, 0.0]  # upside down for the last 10 s
    grav[100:110] = np.nan  # no gravity samples
    inp.imu = {f"grav_{a}": grav[:, k] for k, a in enumerate("xyz")}
    flip = crash._flip_deg(inp)
    assert flip[:500].max() == pytest.approx(0.0, abs=0.01)  # NaN bins count as unturned
    assert flip[500:].min() == pytest.approx(180.0, abs=0.01)
    inp.imu = {k: np.full(N, np.nan) for k in inp.imu}
    assert crash._flip_deg(inp).tolist() == [0.0] * N  # no gravity at all


def _grav_group(t_us: int, *vecs: tuple[float, float, float]) -> dict:
    return {
        "GravityVector": {
            "TimestampUs": t_us,
            "Data": [{"x": x, "y": y, "z": z} for x, y, z in vecs],
        }
    }


def test_the_camera_gravity_stream_is_spread_normalised_and_preferred(monkeypatch):
    groups = [_grav_group(k * 1_000_000, *[(0.0, -2.0, 0.0)] * 30) for k in range(60)]
    groups[0]["GravityVector"]["Data"][1] = "not a sample"
    gt, gv = crash._grav_stream(groups)
    assert len(gt) == 60 * 30 - 1
    assert gt[:2].tolist() == pytest.approx([0.0, 2 / 30])  # spread over its payload
    assert np.allclose(gv, [0.0, -1.0, 0.0])  # unit vectors
    assert crash._grav_stream([{"GravityVector": {"Data": [1]}}])[0].size == 0
    # the camera's fused gravity wins over the accelerometer (upright there)
    _fake_parse(monkeypatch, [*GOPRO, *groups], *_imu_stream())
    g = crash._compute_imu(Path("a.mp4"), 60.0)
    assert g["grav_y"][:600].tolist() == pytest.approx([-1.0] * 600)
    assert np.isnan(g["grav_y"][600])  # past the stream: no data


def test_imu_grids_are_cached_per_source_and_a_corrupt_cache_is_recomputed(tmp_path, monkeypatch):
    src = tmp_path / "GX01.mp4"
    src.write_bytes(b"x")
    monkeypatch.setattr(telemetry, "imu_read_path", lambda s: (s, None, s, False))
    calls = _fake_parse(monkeypatch, GOPRO, *_imu_stream())
    first = crash.imu_grids(src, 60.0)
    path = crash.cache.source_dir(src) / f"crashimu-v{crash.IMU_VERSION}.npz"
    assert path.exists() and len(calls) == 1
    again = crash.imu_grids(src, 60.0)
    assert len(calls) == 1  # a hit
    assert sorted(again) == sorted(first) and all(v.dtype == np.float64 for v in again.values())
    assert all(np.allclose(again[k], first[k], equal_nan=True, atol=1e-3) for k in first)
    path.write_bytes(b"not an npz")
    assert crash.imu_grids(src, 60.0) is not None and len(calls) == 2  # recomputed
    crash.imu_grids(src, 60.0, force=True)
    assert len(calls) == 3


def test_a_source_without_imu_is_cached_as_none(tmp_path, monkeypatch):
    src = tmp_path / "GX02.mp4"
    src.write_bytes(b"x")
    monkeypatch.setattr(telemetry, "imu_read_path", lambda s: (s, None, s, False))
    calls = []
    monkeypatch.setattr(telemetry, "parse_imu", lambda path, level=20: calls.append(path))
    assert crash.imu_grids(src, 60.0) is None and len(calls) == 1
    assert crash.imu_grids(src, 60.0) is None and len(calls) == 1  # the "none" marker


def _parse_gopro(monkeypatch) -> list[Path]:
    """A fake GoPro parse (camera named, for telemetry.extract); returns the parsed paths."""
    calls = []
    parsed = SimpleNamespace(camera="GoPro", model="HERO11 Black"), GOPRO, *_imu_stream()

    def parse_imu(path, level=20):
        calls.append(path)
        return parsed

    monkeypatch.setattr(telemetry, "parse_imu", parse_imu)
    return calls


def test_extract_parses_a_source_once_for_its_telemetry_and_the_crash_imu_grids(
    tmp_path, monkeypatch
):
    src = tmp_path / "GX03.mp4"
    src.write_bytes(b"x")
    monkeypatch.setattr(telemetry, "imu_read_path", lambda s: (s, None, s, False))
    calls = _parse_gopro(monkeypatch)
    tel = telemetry.extract(src, 60.0)
    grids = crash.imu_grids(src, 60.0)
    assert tel is not None and grids is not None
    assert calls == [src]  # one parse fills both caches
    expected = crash._compute_imu(src, 60.0)  # as imu_grids computed them on its own
    assert sorted(grids) == sorted(expected)
    assert all(np.allclose(grids[k], expected[k], equal_nan=True, atol=1e-3) for k in expected)


def test_extract_keeps_valid_crash_imu_grids_and_caches_a_file_without_telemetry_as_none(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(telemetry, "imu_read_path", lambda s: (s, None, s, False))
    src = tmp_path / "GX04.mp4"
    src.write_bytes(b"x")
    path = crash._imu_cache(src)
    crash.cache.save_npz(path, acc_g=np.ones(3))
    _parse_gopro(monkeypatch)
    telemetry.extract(src, 60.0)
    assert crash.cache.load_npz(path)["acc_g"].tolist() == [1.0, 1.0, 1.0]  # kept
    other = tmp_path / "GX05.mp4"
    other.write_bytes(b"x")
    calls = []
    monkeypatch.setattr(telemetry, "parse_imu", lambda path, level=20: calls.append(path))
    assert telemetry.extract(other, 60.0) is None
    # None is "no telemetry" (a read error raises): cached for imu_grids, which does not parse
    assert crash.imu_grids(other, 60.0) is None and calls == [other]


@pytest.mark.parametrize(
    ("unstaged", "remux_ok", "parsed"),
    [(True, True, ["small"]), (True, False, ["full"]), (False, True, ["full"])],
    ids=["remote-remuxed", "remux-failed-full-file", "local-not-remuxed"],
)
def test_imu_grids_read_a_remote_unstaged_file_through_the_gpmf_remux(
    tmp_path, monkeypatch, unstaged, remux_ok, parsed
):
    src = tmp_path / "GX06.mp4"
    src.write_bytes(b"x")
    full, small = tmp_path / "full.mp4", tmp_path / "small.mp4"
    monkeypatch.setattr(telemetry, "imu_read_path", lambda s: (s, None, full, unstaged))
    remuxed = []

    def gpmf_copy(io, tmp):
        remuxed.append(io)
        return small if remux_ok else None

    monkeypatch.setattr(crash, "_gpmf_copy", gpmf_copy)
    calls = _parse_gopro(monkeypatch)
    assert crash.imu_grids(src, 60.0) is not None
    assert remuxed == ([src] if unstaged else [])
    assert calls == [{"small": small, "full": full}[p] for p in parsed]


# ---------------------------------------------------------------- the reaction pick (A491)
def _talking_inputs(words, hits, *, kf=None):
    """A rider at 30 km/h all along (no stop), so the clip ends at the minimum aftermath."""
    inp = _inputs(spike=False)
    inp.words = [{"start": s, "end": e, "word": w} for s, e, w in words]
    inp.hits = hits
    inp.kf_crash = kf
    return inp


_C = {"t0": 10.0, "t1": 10.2}  # clip: 5.0 to 18.2


def test_the_reaction_is_the_utterance_of_the_strongest_crash_word_after_the_clip():
    inp = _talking_inputs(
        [(24.0, 24.3, "ok"), (40.0, 40.3, " oh"), (40.4, 40.8, " no "), (50.0, 50.4, "meh")],
        [(24.0, 0.92, "ok"), (40.0, 0.95, "oh"), (50.0, 0.5, "meh")],
    )
    clip = crash.plan_clip(inp, _C, crash.CrashParams())
    assert clip["end"] == 18.2
    # the later 0.95 hit beats the earlier 0.92 one; its whole utterance chain is read, and a
    # hit under 0.9 is no reaction
    assert clip["reaction"] == {"start": 39.6, "end": 42.3, "text": "oh no"}


def test_no_reaction_without_a_strong_hit_in_range_or_when_it_is_too_close():
    p = crash.CrashParams()
    weak = _talking_inputs([(24.0, 24.3, "x")], [(24.0, 0.85, "x")])
    assert "reaction" not in crash.plan_clip(weak, _C, p)
    close = _talking_inputs([(19.0, 19.3, "x")], [(19.0, 1.0, "x")])  # within 2 s of the end
    assert "reaction" not in crash.plan_clip(close, _C, p)
    far = _talking_inputs([(300.0, 300.3, "x")], [(300.0, 1.0, "x")])
    far.duration = 400.0  # more than 240 s after the end
    assert "reaction" not in crash.plan_clip(far, _C, p)


def test_the_reaction_prefers_words_said_once_the_view_is_back_to_normal():
    kf = np.zeros(60)
    kf[24:27] = 0.95  # still lying on the ground at 24 s
    inp = _talking_inputs(
        [(24.0, 24.3, "ouch"), (40.0, 40.4, "fine")],
        [(24.0, 1.0, "ouch"), (40.0, 0.92, "fine")],
        kf=kf,
    )
    inp.visual = SimpleNamespace(t=np.arange(60, dtype=float))
    clip = crash.plan_clip(inp, _C, crash.CrashParams())
    assert clip["reaction"]["text"] == "fine"
    assert clip["reaction"]["start"] == pytest.approx(39.6)


# --------------------------------------------------------------------------- sheets and helpers
def test_write_sheet_tiles_the_window_in_order_and_marks_the_candidate_red(
    tmp_path: Path, monkeypatch
):
    import cv2  # noqa: PLC0415

    h, w = 48, 64
    ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # the candidate is 1.0-2.0 s
    frames = np.stack([np.full((h, w, 3), v, np.uint8) for v in (10, 60, 110, 160, 210)])
    monkeypatch.setattr(crash, "decode_window", lambda *a, **k: (ts, frames))
    c = {"source": "a.mp4", "t0": 1.0, "t1": 2.0, "caption": "c001 title"}
    out = crash.write_sheet(c, tmp_path / "s.jpg", cols=3, width=w)
    sheet = cv2.imread(str(out))
    # a title strip (26 px) above two rows of three tiles, the last one blank
    assert sheet.shape == (26 + 2 * h, 3 * w, 3)
    assert (sheet[:26].max(axis=2) > 200).any()

    def tile(i: int) -> np.ndarray:
        y, x = 26 + h * (i // 3), w * (i % 3)
        return sheet[y : y + h, x : x + w]

    centres = [int(tile(i)[5:15, 20:40].mean()) for i in range(6)]
    assert centres == pytest.approx([10, 60, 110, 160, 210, 0], abs=6)

    def band(i: int) -> np.ndarray:  # the caption band's left edge, clear of the text
        return tile(i)[-3:-1, :2].reshape(-1, 3).mean(axis=0)

    assert band(1)[2] > 120 and band(1)[0] < 40 and band(2)[2] > 120  # red: 1.0 and 2.0 s
    assert band(0).max() < 40 and band(3).max() < 40  # black


def test_write_sheet_without_frames_writes_nothing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(crash, "decode_window", lambda *a, **k: (np.array([]), np.array([])))
    assert crash.write_sheet({"source": "a.mp4", "t0": 1.0, "t1": 2.0}, tmp_path / "s.jpg") is None
    assert not (tmp_path / "s.jpg").exists()


def test_source_failed_names_the_file_and_warns_unless_told_not_to(caplog, monkeypatch):
    from cutpoint import jsonout  # noqa: PLC0415
    from cutpoint.events.pipeline import SourceGuard  # noqa: PLC0415

    guard = SourceGuard("crash dense check", log=crash.logger)
    warned: list[str] = []
    monkeypatch.setattr(jsonout, "warn", lambda code, message, files=(): warned.append(code))
    with caplog.at_level("WARNING"):
        crash._source_failed(guard, "D:/v/a.mp4", OSError("gone"))
        crash._source_failed(guard, Path("b.mp4"), OSError("gone"), warn=False)
    assert "a.mp4: crash dense check skipped this source (gone)" in caplog.text
    assert "b.mp4" not in caplog.text
    assert list(guard.failed) == ["D:/v/a.mp4", "b.mp4"]
    assert warned == ["step_failed"]


def _candidate(source: str, k: int) -> dict:
    scores = {"telemetry": 0.5, "visual": 0.5, "audio": 0.1, "words": 0.2}
    return {
        "id": f"c{k:03d}", "name": Path(source).name, "source": source, "t0": 10.0 * k,
        "t1": 10.0 * k + 2, "confidence": 0.9, "scores": scores,
    }  # fmt: skip


def test_write_sheets_captions_the_top_candidates_and_skips_a_failed_source(
    tmp_path: Path, monkeypatch
):
    cands = [
        _candidate("a.mp4", 1),
        _candidate("bad.mp4", 2),
        _candidate("bad.mp4", 3),
        _candidate("a.mp4", 4),
    ]
    out_dir = tmp_path / "crash_check"
    out_dir.mkdir()
    (out_dir / "old.jpg").write_bytes(b"x")
    bad_calls: list[str] = []

    def write(c, path):
        if c["source"] == "bad.mp4":
            bad_calls.append(c["id"])
            raise OSError("unreadable")
        path.write_bytes(b"jpg")
        return path

    monkeypatch.setattr(crash, "write_sheet", write)
    written = crash._write_sheets(cands, out_dir, 3, {})
    assert [p.name for p in written] == ["c001_a_10.jpg"]
    assert not (out_dir / "old.jpg").exists()
    assert cands[0]["sheet"] == str(written[0]) and "sheet" not in cands[1]
    assert cands[0]["caption"].startswith("c001 a.mp4 10.0-12.0s conf 0.90 tel 0.50")
    assert "caption" not in cands[3]  # beyond the first three
    # a failed source is not tried again: bad.mp4 was written once, its second candidate none
    assert bad_calls == ["c002"] and "sheet" not in cands[2]
    # sheets=0 keeps the old files and writes none
    assert crash._write_sheets(cands, out_dir, 0, {}) == []
    assert (out_dir / "c001_a_10.jpg").exists()


# ------------------------------------------------------- words_feature, _chain, _normal_view
def _hits_inputs(hits, kf=None, kf_t=None) -> crash.SourceInputs:
    visual = None if kf_t is None else SimpleNamespace(t=np.asarray(kf_t, float))
    return crash.SourceInputs(
        path=Path("a.mp4"),
        duration=600.0,
        hits=hits,
        visual=visual,
        kf_crash=None if kf is None else np.asarray(kf, float),
    )


def test_words_feature_decays_after_the_event_and_saturates():
    # a mutated decay constant, weight or score formula changes these numbers
    at_end = crash.words_feature(_hits_inputs([(20.0, 1.0, "ouch")]), 10.0, 20.0)
    assert at_end["sum"] == 1.0 and at_end["score"] == pytest.approx(1 - np.exp(-1), abs=1e-3)
    later = crash.words_feature(_hits_inputs([(110.0, 1.0, "ouch")]), 10.0, 20.0)
    assert later["sum"] == pytest.approx(np.exp(-1), abs=0.01)  # 90 s after: 1/e
    during = crash.words_feature(_hits_inputs([(15.0, 2.0, "ouch")]), 10.0, 20.0)
    assert during["sum"] == 2.0  # inside the event: no decay, weight counts
    two = crash.words_feature(_hits_inputs([(15.0, 1.0, "a"), (16.0, 1.0, "b")]), 10.0, 20.0)
    assert two["sum"] == 2.0 and two["score"] > at_end["score"]


@pytest.mark.parametrize(
    ("t", "counted"),
    [
        (8.99, False),  # more than 1 s before the event
        (9.0, True),  # exactly 1 s before: in
        (260.0, True),  # exactly horizon after the end: in
        (260.01, False),
    ],
)
def test_words_feature_window_edges(t, counted):
    got = crash.words_feature(_hits_inputs([(t, 1.0, "x")]), 10.0, 20.0)
    assert bool(got["phrases"]) is counted and (got["sum"] > 0) is counted


def test_words_feature_empty_horizon_and_phrase_cap():
    empty = crash.words_feature(_hits_inputs([]), 10.0, 20.0)
    assert empty == {"score": 0.0, "sum": 0.0, "phrases": []}
    hits = [(20.0, 0.5, "late")]
    assert crash.words_feature(_hits_inputs(hits), 0.0, 5.0, horizon_s=10.0)["phrases"] == []
    many = [(20.0 + i * 0.01, 0.1, f"p{i}") for i in range(15)]
    got = crash.words_feature(_hits_inputs(many), 10.0, 20.0)
    assert [p for _, p in got["phrases"]] == [f"p{i}" for i in range(12)]  # first 12, in order
    assert got["phrases"][0][0] == 20.0


def _w(*spans):
    return [{"start": a, "end": b} for a, b in spans]


@pytest.mark.parametrize(
    ("words", "i", "want"),
    [
        (_w((0, 1), (1.5, 2), (2.5, 3)), 1, (0, 2)),  # gaps of 0.5: one utterance
        (_w((0, 1), (1.8, 2), (2.5, 3)), 2, (1, 2)),  # a gap of exactly 0.8 breaks the chain
        (_w((0, 1), (1.79, 2), (2.5, 3)), 0, (0, 2)),  # just under 0.8 holds
        (_w((0, 1), (5, 6), (10, 11)), 1, (1, 1)),  # a lone word
        (_w((0, 1)), 0, (0, 0)),  # a single-word list
        (_w((0, 1), (1.2, 2), (4, 5)), 0, (0, 1)),  # the start of the list is a bound
    ],
)
def test_chain_gives_the_first_and_last_word_of_an_utterance(words, i, want):
    assert crash._chain(words, i) == want


def test_chain_gap_is_a_parameter():
    words = _w((0, 1), (2.0, 3))
    assert crash._chain(words, 0) == (0, 0)
    assert crash._chain(words, 0, gap=1.5) == (0, 1)


def test_normal_view_reads_the_median_of_the_keyframes_in_the_window():
    t = [0, 1, 2, 3, 4]
    kf = [0.9, 0.9, 0.1, 0.1, 0.1]
    # no keyframe layer: nothing says it is not normal
    assert crash._normal_view(_hits_inputs([]), 0.0, 4.0) is True
    inp = _hits_inputs([], kf=kf, kf_t=t)
    assert crash._normal_view(inp, 0.0, 1.0) is False  # lying on the ground
    assert crash._normal_view(inp, 2.0, 4.0) is True  # back up
    assert crash._normal_view(inp, 10.0, 12.0) is True  # no frame in the window
    # the median, not the mean: median 0.5 is normal, the mean (about 0.67) would not be
    mixed = _hits_inputs([], kf=[0.5, 0.5, 1.0], kf_t=[0, 1, 2])
    assert crash._normal_view(mixed, 0.0, 2.0) is True
    # both window edges are inclusive: dropping either 0.9 edge frame gives median 0.5
    edges = _hits_inputs([], kf=[0.9, 0.1, 0.9], kf_t=[0, 1, 2])
    assert crash._normal_view(edges, 0.0, 2.0) is False
    # the 0.6 limit: a median of exactly 0.6 is not normal
    edge = _hits_inputs([], kf=[0.6, 0.6], kf_t=[0, 1])
    assert crash._normal_view(edge, 0.0, 1.0) is False
    below = _hits_inputs([], kf=[0.59, 0.59], kf_t=[0, 1])
    assert crash._normal_view(below, 0.0, 1.0) is True
