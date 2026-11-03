"""A wearer axis off the default mount recomputes only the selfie view, byte-equal to a full
recompute (#376 item 2). Torch, CLIP, the detector and the fisheye sampler are numpy fakes."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint import insta360
from cutpoint.events import visual as V
from cutpoint.pano import geometry as G
from cutpoint.pano import views
from cutpoint.pano.x3 import X3Clip
from pano_fakes import FAKE_TORCH, Models, Sampler, frames
from pano_helpers import orientation

TIMING = ("elapsed_s", "gpu_s")


@pytest.fixture
def pano(tmp_path: Path, monkeypatch):
    lrv = tmp_path / "LRV_001.lrv"
    lrv.write_bytes(b"x")
    clip = X3Clip("001", "2026-01-01", "5.7k", str(lrv), [], frames=600, duration=20.0)
    rec = SimpleNamespace(built=[], closed=0, det_calls=0, embeds=[], clip=clip, lrv=lrv)
    monkeypatch.setitem(sys.modules, "torch", FAKE_TORCH)
    monkeypatch.setattr(G, "FisheyeSampler", Sampler)
    monkeypatch.setattr(G, "orientation", lambda *a: orientation())
    monkeypatch.setattr(V, "iter_frames", frames)
    monkeypatch.setattr(V, "VisualModels", lambda **kw: Models(rec, **kw))
    monkeypatch.setattr(V, "require_detector", lambda: None)  # the fake models need no file
    monkeypatch.setattr(V, "text_embeddings", lambda prompts, models=None: None)
    monkeypatch.setattr(insta360, "imu_source", lambda src: src)
    rec.p = views.ViewParams(view_px=4, eq_width=32)
    return rec


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(a @ b, -1, 1))))


def _same(a: views.PanoViews, b: views.PanoViews) -> None:
    for k in ("t", "emb", "det", "box_emb", "box_idx", "rays"):
        x, y = getattr(a, k), getattr(b, k)
        assert x.dtype == y.dtype and x.shape == y.shape, k
        assert x.tobytes() == y.tobytes(), k
    drop = lambda m: {k: v for k, v in m.items() if k not in TIMING}  # noqa: E731
    assert drop(a.meta) == drop(b.meta)


def test_selfie_only_recompute_equals_a_full_recompute(pano):
    models = Models(pano)
    orient = orientation()
    first = views.compute_views(pano.clip, models, pano.p, orient)
    axis = views.wearer_axis(first)
    assert _angle(axis, G.WEARER_V) > 25  # the run_layer trigger
    full = views.compute_views(pano.clip, models, pano.p, orient, wearer_v=axis)
    det_calls, embeds = pano.det_calls, len(pano.embeds)
    sel = views.compute_views(pano.clip, models, pano.p, orient, wearer_v=axis, reuse=first)
    _same(sel, full)
    assert pano.det_calls == det_calls  # no detector, no person-crop embeds
    assert pano.embeds[embeds:] == [8 * 9, 8 * 9, 4 * 9]  # the 9-view batches of a full run
    # The axis matters where it should, and only there.
    assert first.emb[:, views.SELFIE].tobytes() != full.emb[:, views.SELFIE].tobytes()
    assert first.emb[:, : views.SELFIE].tobytes() == full.emb[:, : views.SELFIE].tobytes()
    assert not np.array_equal(first.det[:, 7], full.det[:, 7])
    assert np.array_equal(np.delete(first.det, 7, 1), np.delete(full.det, 7, 1))
    assert len(full.box_idx) and len(full.det)
    # wearer_deg needs the float64 rays: float32-rounded ones give other bytes.
    r32 = first.rays.astype(np.float32).astype(np.float64)
    wear = axis / np.linalg.norm(axis)
    col7 = np.asarray([views._wearer_deg(dv, wear) for dv in r32], np.float32)
    assert col7.tobytes() != full.det[:, 7].tobytes()


@pytest.mark.parametrize(
    "stale",
    [
        lambda first: {"t": first.t + 1},  # other frames
        lambda first: {"rays": None},  # as load_views gives it
    ],
    ids=["other-frames", "no-rays"],
)
def test_views_that_cannot_be_reused_fall_back_to_a_full_recompute(pano, stale):
    models = Models(pano)
    orient = orientation()
    first = views.compute_views(pano.clip, models, pano.p, orient)
    axis = views.wearer_axis(first)
    full = views.compute_views(pano.clip, models, pano.p, orient, wearer_v=axis)
    reuse = views.PanoViews(**{**first.__dict__, **stale(first)})
    det_calls = pano.det_calls
    _same(views.compute_views(pano.clip, models, pano.p, orient, wearer_v=axis, reuse=reuse), full)
    assert pano.det_calls > det_calls


def test_saved_views_drop_the_in_process_scores(tmp_path: Path, monkeypatch):
    src = tmp_path / "LRV_002.lrv"
    src.write_bytes(b"x")
    monkeypatch.setattr(
        V,
        "text_embeddings",
        lambda prompts, models=None: np.random.default_rng(len(prompts)).normal(
            size=(len(prompts), 768)
        ),
    )
    monkeypatch.setattr(V, "aesthetic", lambda e: 5.0 + e[..., 0])
    rng = np.random.default_rng(0)
    old = views.PanoViews(
        t=np.arange(4, dtype=np.float32),
        emb=rng.normal(size=(4, 9, 768)).astype(np.float16),
        det=np.zeros((0, len(views.DET_COLS)), np.float32),
        box_emb=np.zeros((0, 768), np.float16),
        box_idx=np.zeros(0, np.int64),
        meta={"source": str(src)},
    )
    new = views.PanoViews(
        **{**old.__dict__, "emb": rng.normal(size=(4, 9, 768)).astype(np.float16)}
    )
    before = views.view_scores(old)["group_zs"].copy()
    occl = views.occlusion_scores(old)["selfie_ok"].copy()
    views.save_views(src, new, views.ViewParams())
    after = views.view_scores(new)["group_zs"]
    assert not np.array_equal(after, before)
    views._SCORES.clear()
    assert np.array_equal(after, views.view_scores(new)["group_zs"])  # a fresh compute
    assert not np.array_equal(views.occlusion_scores(new)["selfie_ok"], occl)


def _written(lrv: Path, p: views.ViewParams) -> dict:
    """The cache files run_layer wrote: npz arrays, JSON without the timings."""
    out = {}
    pv_path = views.cache_path(lrv, p)
    vis_path = V.cache_path(lrv, V.VisualParams())
    for path in (pv_path, vis_path):
        with np.load(path) as z:
            out[path.name] = {k: (z[k].dtype, z[k].shape, z[k].tobytes()) for k in z.files}
    meta = views.cache.load_json(pv_path.with_suffix(".json"))
    out["meta"] = {k: v for k, v in meta.items() if k not in TIMING}
    out["visual.json"] = vis_path.with_suffix(".json").read_bytes()
    return out


def test_run_layer_writes_the_caches_of_a_full_recompute(pano):
    st = views.run_layer([pano.clip], pano.p)
    assert st["computed"] == 1 and len(pano.built) == 1 and pano.closed == 1
    assert pano.det_calls == 3  # the selfie pass skipped the detector
    assert pano.embeds.count(8 * 9) == 4  # ... but embedded its 9-view batches again
    selfie_only = _written(pano.lrv, pano.p)
    axis = views.wearer_axis(views.load_views(pano.lrv, pano.p))
    assert _angle(axis, G.WEARER_V) > 25

    # A forced re-run starts at the cached axis: one full pass, no selfie pass.
    pano.det_calls, pano.embeds = 0, []
    views.run_layer([pano.clip], pano.p, force=True)
    assert pano.det_calls == 3 and pano.embeds.count(8 * 9) == 2
    assert _written(pano.lrv, pano.p) == selfie_only
