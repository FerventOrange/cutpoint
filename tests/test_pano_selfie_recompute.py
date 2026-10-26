"""A wearer axis off the default mount recomputes only the selfie view, byte-equal to a full
recompute (#376 item 2). Torch, CLIP, the detector and the fisheye sampler are numpy fakes."""

from __future__ import annotations

import contextlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint.events import visual as V
from cutpoint.pano import geometry as G
from cutpoint.pano import views, x3
from cutpoint.pano.x3 import X3Clip
from pano_helpers import orientation

N_FRAMES = 20  # 3 batches of 8, 8, 4
TIMING = ("elapsed_s", "gpu_s")


class _T(np.ndarray):
    """The few ``torch.Tensor`` methods ``compute_views`` uses, on numpy."""

    def to(self, *a, **k):
        return self

    def permute(self, *dims):
        return self.transpose(dims)

    def float(self):
        return self.astype(np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def view(self, *args, **kw):
        if args and all(isinstance(a, int) for a in args):
            return self.reshape(args)
        return super().view(*args, **kw)


def _interpolate(x, size, **kw):
    """Nearest resize of ``(1, 3, h, w)`` to ``size``."""
    h, w = x.shape[2:]
    yi = np.arange(size[0]) * h // size[0]
    xi = np.arange(size[1]) * w // size[1]
    return np.ascontiguousarray(x[:, :, yi][:, :, :, xi]).view(_T)


FAKE_TORCH = SimpleNamespace(
    tensor=lambda x, device=None: np.asarray(x),
    from_numpy=lambda a: a.view(_T),
    inference_mode=contextlib.nullcontext,
    cat=lambda xs, dim: np.concatenate(xs, dim).view(_T),
    isin=np.isin,
    nn=SimpleNamespace(functional=SimpleNamespace(interpolate=_interpolate)),
)


class _Sampler:
    """Stand-in for ``FisheyeSampler``: each pixel is the frame's mean colour plus its ray in
    camera coordinates, so the output depends on the frame and on the view rotation."""

    def __init__(self, device, fov=G.DFISHEYE_FOV):
        pass

    def rays(self, d):
        return np.asarray(d.reshape(-1, 3), np.float32), d.shape[:-1]

    def sample(self, frames, r_vw, rays, view_rot=None):
        d, shape = rays
        full = r_vw[:, None] if view_rot is None else np.einsum("bij,kjl->bkil", r_vw, view_rot)
        cam = np.einsum("bkij,nj->bkjn", full, d)  # (B, K, 3, N)
        base = np.asarray(frames).mean((2, 3))[:, None, :, None]
        out = (base + 0.1 * cam).astype(np.float32).reshape(*full.shape[:2], 3, *shape)
        out = out.view(_T)
        return out if view_rot is not None else out[:, 0]


class _Detector:
    """Per frame: the wearer (a ~90-deg box near straight ahead, off the default mount and
    off the pixel grid, so its ray is not axis-aligned), another
    person moving with the frame, and a COCO label that is not kept."""

    def __init__(self, rec):
        self.rec = rec
        self.transform = SimpleNamespace(min_size=None, max_size=None)
        self.roi_heads = SimpleNamespace(score_thresh=None)

    def __call__(self, images):
        self.rec.det_calls += 1
        outs = []
        for img in images:
            m = float(np.asarray(img).mean())
            x0 = 2.0 + (m * 997.0) % 10.0
            outs.append(
                {
                    "boxes": np.asarray(
                        [[15.3, 3.7, 24.1, 12.9], [x0, 6, x0 + 9, 15], [5, 5, 7, 7]], np.float32
                    ).view(_T),
                    "labels": np.asarray([1, 1, 3], np.int64).view(_T),
                    "scores": np.asarray([0.95, 0.4 + (m * 31.0) % 0.5, 0.9], np.float32).view(_T),
                }
            )
        return outs


class _Models:
    def __init__(self, rec, **kw):
        self.rec = rec
        self.device = "cpu"
        self.det = _Detector(rec)
        rec.built.append(self)

    def _embed(self, c):
        self.rec.embeds.append(len(c))
        flat = np.asarray(c, np.float32).reshape(len(c), -1)
        out = np.zeros((len(c), 768), np.float32)
        out[:, : min(768, flat.shape[1])] = flat[:, :768]
        return out.view(_T)

    def close(self):
        self.rec.closed += 1


def _frames(path, width, height, keyframes, hwaccel):
    for i in range(N_FRAMES):
        frame = np.full((8, 16, 3), (i * 37) % 251, np.uint8)
        frame[:, :, 1] = (i * 11) % 7
        yield i * 1.0, frame


@pytest.fixture
def pano(tmp_path: Path, monkeypatch):
    lrv = tmp_path / "LRV_001.lrv"
    lrv.write_bytes(b"x")
    clip = X3Clip("001", "2026-01-01", "5.7k", str(lrv), [], frames=600, duration=20.0)
    rec = SimpleNamespace(built=[], closed=0, det_calls=0, embeds=[], clip=clip, lrv=lrv)
    monkeypatch.setitem(sys.modules, "torch", FAKE_TORCH)
    monkeypatch.setattr(G, "FisheyeSampler", _Sampler)
    monkeypatch.setattr(G, "orientation", lambda *a: orientation())
    monkeypatch.setattr(V, "iter_frames", _frames)
    monkeypatch.setattr(V, "VisualModels", lambda **kw: _Models(rec, **kw))
    monkeypatch.setattr(V, "text_embeddings", lambda prompts, models=None: None)
    monkeypatch.setattr(x3, "imu_source", lambda src: src)
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
    models = _Models(pano)
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


def test_other_frames_fall_back_to_a_full_recompute(pano):
    models = _Models(pano)
    orient = orientation()
    first = views.compute_views(pano.clip, models, pano.p, orient)
    axis = views.wearer_axis(first)
    full = views.compute_views(pano.clip, models, pano.p, orient, wearer_v=axis)
    stale = views.PanoViews(**{**first.__dict__, "t": first.t + 1})
    det_calls = pano.det_calls
    _same(views.compute_views(pano.clip, models, pano.p, orient, wearer_v=axis, reuse=stale), full)
    assert pano.det_calls > det_calls


def test_views_without_rays_fall_back_to_a_full_recompute(pano):
    models = _Models(pano)
    orient = orientation()
    first = views.compute_views(pano.clip, models, pano.p, orient)
    axis = views.wearer_axis(first)
    full = views.compute_views(pano.clip, models, pano.p, orient, wearer_v=axis)
    cached = views.PanoViews(**{**first.__dict__, "rays": None})  # as load_views gives it
    det_calls = pano.det_calls
    _same(views.compute_views(pano.clip, models, pano.p, orient, wearer_v=axis, reuse=cached), full)
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
