"""Action-cam and 360 fixes of audit #1085 (A184, A187, A194, A357, A728)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint import actioncam, crash, dsp, mix, telemetry
from cutpoint.events import visual as V
from cutpoint.events import voice as VO
from cutpoint.events.visual import VisualModels, square_crops
from cutpoint.pano import views
from cutpoint.telemetry import DT, TelemetryParams

# ------------------------------------------------------------------ A187: one time-to-bin rule


@pytest.mark.parametrize(
    ("t", "expected"),
    [(0.0, 0), (0.3, 3), (0.7, 7), (2.3, 23), (0.36, 3), (0.39, 3), (10800.0, 108000), (-0.05, 0)],
)
def test_bin_of_rounds_down_without_float_error_on_bin_edges(t, expected):
    assert dsp.bin_of(t, 0.1) == expected


def test_bin_edges_rounded_to_a_tenth_paint_their_own_bin():
    """``int(0.3 / 0.1)`` is 2: a span from 0.3 s started one bin early."""
    assert np.flatnonzero(telemetry.paint_spans(10, [(0.3, 0.7, 1.0)])).tolist() == [3, 4, 5, 6, 7]
    assert np.flatnonzero(mix._paint(10, [(0.3, 0.7)])).tolist() == [3, 4, 5, 6]
    x = np.arange(10, dtype=float)
    assert mix._mean(x, 0.3, 0.4) == 3.0


def test_build_and_layer_features_slice_a_recovered_part_at_the_same_bin(monkeypatch):
    """A187: telemetry was sliced at ``int(offset / DT)``, the voice grid at ``round()``."""
    ramp = np.arange(100, dtype=float)
    seen: dict[str, float] = {}

    def voice_features(grids, moving):
        seen["voice"] = grids["laughter"][0]
        return {"laughter": grids["laughter"], "voice": grids["laughter"]}

    monkeypatch.setattr(VO, "load_voice", lambda src: {"laughter": ramp})
    monkeypatch.setattr(VO, "voice_features", voice_features)
    monkeypatch.setattr(V, "load_visual", lambda src: None)
    monkeypatch.setattr(views, "layer_grids", lambda *a: None)
    out = actioncam.layer_features(Path("a.mp4"), 10, 0.36)
    assert seen["voice"] == 3.0 == dsp.bin_of(0.36, DT)  # round() gave 4
    assert out["voice"][0] == 3.0


# ------------------------------------------------------------------ A184: no reading is not a zero


def test_no_vibration_before_an_impact_at_the_start_is_not_moving():
    fs = 100.0
    t = np.arange(int(10 * fs)) / fs
    am = np.ones_like(t)
    am[(t >= 0.2) & (t < 0.23)] = 6.0  # no bins before it: no reading, not "0 g"
    grav = np.tile([0.0, 0.0, 1.0], (len(t), 1))
    grav[t > 0.7] = [1.0, 0.0, 0.0]
    vib = np.full(100, 0.5)  # moving later in the clip: a negative window end must not read it
    vib[15:45] = 0.0
    speed = np.full(100, np.nan)
    assert telemetry._crash_events(t, am, grav, speed, vib, TelemetryParams()) == []


# ------------------------------------------------------------------ A357: one CLIP embedding


def test_clip_encoder_embeds_as_the_visual_layer_does():
    torch = pytest.importorskip("torch")

    class Model:
        def encode_image(self, x):
            return torch.nn.functional.adaptive_avg_pool2d(x, 16).flatten(1) + 1.0

    enc = crash.ClipEncoder.__new__(crash.ClipEncoder)
    enc.torch, enc.device, enc.dtype, enc.model = torch, "cpu", torch.float32, Model()
    enc.mean, enc.std = torch.full((1, 3, 1, 1), 0.4), torch.full((1, 3, 1, 1), 0.3)
    enc.on_device = True
    vm = VisualModels.__new__(VisualModels)
    vm.torch, vm.dtype, vm.clip, vm.mean, vm.std = torch, enc.dtype, enc.model, enc.mean, enc.std
    rng = np.random.default_rng(1)
    for h, w in ((48, 80), (80, 48)):
        frames = rng.integers(0, 256, (3, h, w, 3), dtype=np.uint8)
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255
        want = vm._embed(torch.cat(square_crops(x), 0)).view(3, 3, -1).permute(1, 0, 2).numpy()
        np.testing.assert_allclose(enc.embed(frames, batch=2), want, rtol=1e-5, atol=1e-6)
        centre = enc.embed(frames, centre=True)
        np.testing.assert_allclose(centre[:, 0], want[:, 1], rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------ A194: detect_project helpers


def test_no_reachable_footage_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(crash, "_project_sources", lambda project: [(tmp_path / "gone.mp4", 9.0)])
    with pytest.raises(ValueError, match="reachable"):
        crash._reachable_sources(tmp_path, warn_unreachable=False)


def test_rank_keeps_a_bypass_candidate_confidence_and_numbers_the_best_first(monkeypatch):
    monkeypatch.setattr(crash, "plan_clip", lambda inp, c, p: {"start": c["t0"]})
    monkeypatch.setattr(crash, "_absorb", lambda cands: cands)
    conf = {"a": (0.4, 0.2), "b": (0.5, 0.9)}  # (before, after) the dense score

    def confidence(c, w):
        return conf[c["name"]][1 if "dense" in c["scores"] else 0]

    monkeypatch.setattr(crash, "_confidence", confidence)
    cands = [
        {"name": n, "source": "s", "t0": 1.0, "scores": {}, "dense": {"score": 1.0}}
        for n in ("a", "b")
    ]
    out = crash._rank(cands, {"s": None}, crash.CrashParams(), {id(cands[0])})
    assert [(c["id"], c["name"], c["confidence"]) for c in out] == [
        ("c001", "b", 0.9),
        ("c002", "a", 0.4),  # bypass: never lowered by its dense score
    ]
    assert all(c["confirmed"] for c in out)


# ------------------------------------------------------------------ A728: cheap presence check


def test_run_layer_tests_cached_views_without_loading_them(monkeypatch):
    monkeypatch.setattr(views, "load_views", lambda *a, **k: pytest.fail("loaded"))
    monkeypatch.setattr(views, "views_current", lambda src, params=None: True)
    clips = [SimpleNamespace(lrv="a.lrv"), SimpleNamespace(lrv=None)]
    assert views.run_layer(clips)["cached"] == 1
