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
    assert json.loads(out.read_text(encoding="utf-8"))["n_candidates"] == 0


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
    assert sorted(Path(c["source"]).name for c in data["crashes"]) == ["drift.mp4", "reef.mp4"]
    warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warned == [
        "wreck.mp4: crash detection skipped this source "
        "('list' object cannot be interpreted as an integer)",
        "cave.mp4: crash detection skipped this source (bad IMU block)",
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
        raise RuntimeError(f"no CLIP weights ({src.name})")

    monkeypatch.setattr(crash, "load_inputs", load_inputs)
    own_words = replace(crash.CrashParams(), pos=("a",), neg=("b",), words=())
    with caplog.at_level("WARNING"), pytest.raises(RuntimeError, match=r"arch\.mp4"):
        crash.detect_project(project, params=own_words, dense=False, sheets=0)
    assert (project / "crashes.json").read_text(encoding="utf-8") == earlier
    assert sum("crash detection skipped this source" in r.getMessage() for r in caplog.records) == 2


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
    return {k: v for k, v in data.items() if k not in ("created", "timings")}


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
    dense = {
        k: crash.dense_feature(enc, inp, c, p, {})["crash_topq"]
        for k, p in (("prof", prof), ("generic", generic))
    }
    assert dense["prof"] >= dense["generic"]


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
    monkeypatch.setattr(stage, "staged_path", functools.cache(lambda p: staged or p))
    monkeypatch.setattr(stage, "is_remote", lambda _p: remote)
    io, x3_imu, read_path, got = telemetry.imu_read_path(src)
    assert (io, x3_imu, read_path, got) == (src, trailer and Path(trailer), Path(read), unstaged)
