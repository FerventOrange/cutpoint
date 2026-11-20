"""The 360 forward axis (yaw 0) of an upright clip: :func:`geometry.forward_axis`, the
view layer's ``r_vw`` and the renderer's ``stabilised_r_vw`` in its frame, and the caches
that depend on it (views, director paths). Synthetic IMU fused by ``geometry.fuse``."""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint import cache
from cutpoint.events import visual as V
from cutpoint.pano import cli as PC
from cutpoint.pano import geometry as G
from cutpoint.pano import render
from cutpoint.pano import views as VW
from cutpoint.pano.x3 import X3Clip
from pano_fakes import FAKE_TORCH, Models, Sampler, frames
from pano_helpers import rot, swaying, write_orient, write_views

FPS, DUR = 30.0, 12.0  # the defaults of swaying()


# camera -> world at rest. Helmet: sky lens (-z) up, FORWARD_V along world +z.
HELMET = rot(0, [-90.0])[0]
# Upright, lenses horizontal: camera -y up, tipped 6 deg forward (|horizontal| of FORWARD_V
# ~0.1, as on the cruise footage).
UPRIGHT = rot(0, [6.0])[0]


def _display_yaw(o: G.Orientation, r_vw: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """World yaw (deg, unwrapped) of the display's forward ray at frames ``idx``."""
    f = np.einsum("kij,kj->ki", o.cam_to_world(idx) @ r_vw, np.tile([0.0, 0.0, 1.0], (len(idx), 1)))
    return np.degrees(np.unwrap(np.arctan2(f[:, 0], f[:, 2])))


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angles (deg) between rotation stacks ``a`` and ``b``."""
    cos = (np.trace(np.einsum("kji,kjl->kil", a, b), axis1=1, axis2=2) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


IDX = np.arange(60, int(DUR * FPS) - 61)


# --------------------------------------------------------------------------- axis choice
def test_axis_choice() -> None:
    assert G.forward_axis(np.tile([0.0, 0.0, -1.0], (5, 1))) is G.FORWARD_V
    assert G.forward_axis(np.tile([0.0, -1.0, 0.0], (5, 1))) is G.LENS_V
    assert G.forward_axis(np.tile([0.0, 1.0, 0.0], (5, 1))) is G.LENS_V  # upside down
    assert G.forward_axis(np.tile([-1.0, 0.0, 0.0], (5, 1))) is G.FORWARD_V  # on its side
    assert G.forward_axis(np.zeros((0, 3))) is G.FORWARD_V
    # FORWARD_V 60 deg from vertical (|horizontal| 0.87): still the helmet axis
    tilt = np.array([0.0, math.sin(math.radians(60)), -math.cos(math.radians(60))])
    assert G.forward_axis(np.tile(tilt, (5, 1))) is G.FORWARD_V
    assert G.axis_name(G.FORWARD_V) == "-y" and G.axis_name(G.LENS_V) == "+z"


@pytest.mark.parametrize(("mount", "axis"), [(HELMET, "-y"), (UPRIGHT, "+z")])
def test_axis_choice_is_stable_under_sway(mount, axis) -> None:
    """One axis per clip from the median: a 25 deg sway does not flip it."""
    o = swaying(mount, turn_dps=20.0, sway_deg=25.0)
    assert o.forward_axis == axis
    for part in (slice(0, 120), slice(100, 220), slice(200, None)):
        assert G.axis_name(G.forward_axis(o.up[part])) == axis


# --------------------------------------------------------------------------- upright frames
def test_upright_turn_has_a_smooth_display_heading_in_both_frames() -> None:
    """Turning 30 deg/s while swaying: the display yaw follows the turn without jumps in the
    view layer's ``r_vw`` and the renderer's frame, and the two frames agree."""
    o = swaying(UPRIGHT, turn_dps=30.0, sway_deg=5.0)
    assert o.forward_axis == "+z" and o.meta["forward_axis"] == "+z"
    view = np.stack([o.r_vw(i / FPS) for i in IDX])
    rend = render.stabilised_r_vw(o, IDX, 1.0, 0.5)
    assert _angle(view, rend).max() < 2.0
    for r in (view, rend):
        yaw = _display_yaw(o, r, IDX)
        rate = np.polyfit(IDX / FPS, yaw, 1)[0]
        assert abs(abs(rate) - 30.0) < 1.5
        assert np.abs(np.diff(yaw) - np.diff(yaw).mean()).max() < 0.5  # deg per frame


def test_upright_still_camera_holds_its_view() -> None:
    """A swaying, not turning, upright camera: a fixed-yaw view stays put in the world
    (the forward axis near vertical spun it before)."""
    o = swaying(UPRIGHT, sway_deg=8.0)
    view = np.stack([o.r_vw(i / FPS) for i in IDX])
    # the view layer's gyro heading wobbles a few degrees with sway (3 deg on the helmet mount)
    assert np.ptp(_display_yaw(o, view, IDX)) < 6.0
    assert np.ptp(_display_yaw(o, render.stabilised_r_vw(o, IDX, 0.7, 0.5), IDX)) < 1.0


# --------------------------------------------------------------------------- helmet unchanged
def test_helmet_mount_matrices_are_unchanged() -> None:
    o = swaying(HELMET, turn_dps=30.0, sway_deg=15.0)
    assert o.fwd is G.FORWARD_V and "forward_axis" not in o.meta
    for i in IDX[::17]:
        before = G.level_matrix(o.up[i]) @ G.rot_y(o.heading[i] - o.heading_smooth[i])
        assert np.array_equal(o.r_vw(i / FPS), before)


def test_upright_axis_logged_once(home, tmp_path, monkeypatch, caplog) -> None:
    src = tmp_path / "up.lrv"
    src.write_bytes(b"x")
    o = swaying(UPRIGHT)
    monkeypatch.setattr(G, "fuse", lambda *a, **k: o)
    monkeypatch.setattr(G, "load_imu", lambda _p: (None, None, None))
    with caplog.at_level(logging.INFO, logger="cutpoint.pano.geometry"):
        G.orientation(src, Path("i00"), FPS, len(o.t))
        G.orientation(src, Path("i00"), FPS, len(o.t))
    assert len(caplog.records) == 1  # the second call is silent
    assert G.cached_forward_axis(src) == "+z"
    assert G.load_orientation(src).meta["forward_axis"] == "+z"


# --------------------------------------------------------------------------- caches
@pytest.mark.parametrize(
    ("up", "meta", "valid"),
    [
        ([0.0, 0.0, -1.0], {}, True),  # helmet: as before
        ([0.0, -1.0, 0.0], {}, False),  # upright, computed before the fix
        ([0.0, -1.0, 0.0], {"forward_axis": "+z"}, False),  # d10b5c0: equirect in +z yaw
        ([0.0, -1.0, 0.0], {"forward_axis": "+z", "det_frame": "-z"}, True),
        ([0.0, 0.0, -1.0], {"forward_axis": "+z"}, False),
        ([0.0, -1.0, 0.0], {"forward_axis": "-z"}, False),  # this branch's first sign
    ],
)
def test_views_cache_follows_the_forward_axis(home, tmp_path, up, meta, valid) -> None:
    src = tmp_path / "clip.lrv"
    src.write_bytes(b"x")
    if up is not None:
        write_orient(src, up)
    write_views(src, {"source": str(src)} | meta)
    assert (VW.load_views(src) is not None) is valid


def test_a_views_meta_written_for_other_arrays_is_not_used(home, tmp_path) -> None:
    """A211: the arrays and their meta carry one token; an interrupted save leaves them
    apart, and the pair is not used. Files from before the token have it in neither."""
    src = tmp_path / "clip.lrv"
    src.write_bytes(b"x")
    path = VW.cache_path(src)
    write_views(src, {"source": str(src)})
    assert VW.load_views(src) is not None
    z = np.zeros(0)
    d = np.zeros((0, len(VW.DET_COLS)), np.float32)
    np.savez(path, t=z, emb=z, det=d, box_emb=z, box_idx=z, pair=np.array("new"))
    for meta_pair, ok in (("old", False), (None, False), ("new", True)):
        meta = {"source": str(src)} | ({"pair": meta_pair} if meta_pair else {})
        cache.save_json(path.with_suffix(".json"), meta)
        assert (VW.load_views(src) is not None) is ok


def test_a_stale_views_cache_is_logged_once(home, tmp_path, caplog) -> None:
    src = tmp_path / "stale.lrv"
    src.write_bytes(b"x")
    write_orient(src, [0.0, -1.0, 0.0])
    write_views(src, {"source": str(src)})
    with caplog.at_level(logging.INFO, logger="cutpoint.pano.views"):
        assert VW.load_views(src) is None and VW.load_views(src) is None
    assert len(caplog.records) == 1  # the second load is silent


def test_a_cached_track_records_its_axis_in_memory_only_when_upright(home, tmp_path) -> None:
    """Tracks cached before the fix have no ``forward_axis`` in their meta: it is added on
    load for an upright clip, never for a helmet one (whose meta stays as it was)."""
    for up, want in (([0.0, -1.0, 0.0], "+z"), ([0.0, 0.0, -1.0], None)):
        src = tmp_path / f"{want}.lrv"
        src.write_bytes(b"x")
        path = G._orient_path(src)
        path.parent.mkdir(parents=True, exist_ok=True)
        z = np.zeros(10, np.float32)
        ups = np.tile(np.float32(up), (10, 1))
        meta = np.array(json.dumps({"fps": 30.0}))
        np.savez(path, t=z, up=ups, heading=z, heading_smooth=z, q=z, meta=meta)
        assert G.load_orientation(src).meta.get("forward_axis") == want
        with np.load(path) as stored:
            assert json.loads(str(stored["meta"])) == {"fps": 30.0}


KEYS = [{"t": 1.0, "yaw": 0.0, "pitch": -5.0, "fov": 86.0}]  # a usable path (A215)


def test_director_reuse_key_has_the_axis_only_when_upright(tmp_path) -> None:
    seg = ("001", 1.0, 5.0, "people", False, {"a": 1})
    from cutpoint.pano.director import PATH_VERSION  # noqa: PLC0415

    base = {
        "version": PATH_VERSION,
        "clip": "001",
        "segment": {"start": 1.0, "end": 5.0},
        "kind": "people",
        "vertical": False,
        "params": {"a": 1},
        "keys": KEYS,
    }
    cache.save_json(tmp_path / "001_a.json", base)
    upright = {
        "clip": "002",
        "forward_axis": "+z",
        "det_frame": VW.DET_FRAME,
        "wearer_model": G.UPRIGHT_WEARER,
    }
    cache.save_json(tmp_path / "002_b.json", base | upright)
    cache.save_json(tmp_path / "003_c.json", base | upright | {"clip": "003", "det_frame": None})
    found = PC._reusable(tmp_path)
    assert PC._plan_key(*seg) in found
    now = PC._axis_key("+z", VW.DET_FRAME, G.UPRIGHT_WEARER)
    assert PC._plan_key("002", *seg[1:], now) in found
    assert PC._plan_key("002", *seg[1:]) not in found  # planned upright, now default: re-plan
    assert PC._plan_key("003", *seg[1:], now) not in found  # d10b5c0's +z views: re-plan


def test_director_reuse_key_follows_the_layers_the_path_was_planned_on(tmp_path) -> None:
    """A212: a path records the identity of the view, voice and orientation caches; once
    one is recomputed the key differs and the path is planned anew."""
    from cutpoint.pano.director import PATH_VERSION  # noqa: PLC0415

    seg = ("001", 1.0, 5.0, "people", False, {"a": 1})
    planned_on = {"dir": "d", "views": ["v.npz", 10, 1], "voice": None, "orient": None}
    path = {
        "version": PATH_VERSION,
        "clip": "001",
        "segment": {"start": 1.0, "end": 5.0},
        "kind": "people",
        "vertical": False,
        "params": {"a": 1},
        "keys": KEYS,
        "inputs": planned_on,
    }
    cache.save_json(tmp_path / "001_a.json", path)
    found = PC._reusable(tmp_path)
    assert PC._plan_key(*seg, None, None, planned_on) in found
    recomputed = planned_on | {"views": ["v.npz", 10, 2]}
    assert PC._plan_key(*seg, None, None, recomputed) not in found
    assert PC._plan_key(*seg) not in found


def test_layer_inputs_of_an_unreachable_source_is_none(tmp_path) -> None:
    assert PC._layer_inputs(str(tmp_path / "gone.lrv")) is None


# --------------------------------------------------------------------------- yaw 0 and meta
def test_upright_yaw_0_follows_the_10_lens() -> None:
    """Upright, still or turning: display yaw 0 is the ``_10_`` lens (camera +z), the
    ``_00_`` lens (-z) is behind."""
    for o in (swaying(UPRIGHT, sway_deg=8.0), swaying(UPRIGHT, turn_dps=30.0, sway_deg=5.0)):
        for t in IDX[::40] / FPS:
            r = o.r_vw(float(t))
            front, _ = G.ray_yaw_pitch(r.T @ np.array([0.0, 0.0, 1.0]))
            back, _ = G.ray_yaw_pitch(r.T @ np.array([0.0, 0.0, -1.0]))
            assert abs(front) < 10.0 and abs(abs(back) - 180.0) < 10.0


def test_a_stale_axis_in_the_meta_is_replaced() -> None:
    o = G.Orientation(
        t=np.zeros(3), up=np.tile([0.0, -1.0, 0.0], (3, 1)), heading=np.zeros(3),
        heading_smooth=np.zeros(3), meta={"fps": FPS, "forward_axis": "-z"},
    )  # fmt: skip
    assert o.meta["forward_axis"] == "+z"


# --------------------------------------------------------------------------- detector frame
def _upright_pair(n: int = 600) -> tuple[G.Orientation, G.Orientation]:
    """An upright, tipped and turning track (``+z``) and the same track levelled by the
    ``_00_`` lens (``-z``, the frame of 4b3894b; reported as the default axis)."""
    k = np.arange(n)
    up = np.stack([0.1 * np.sin(k / 50), -np.ones(n), 0.15 + 0.05 * np.cos(k / 70)], 1)
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    o = G.Orientation(t=k / FPS, up=up, heading=0.7 * k, heading_smooth=0.6 * k, meta={"fps": FPS})
    minus = G.Orientation(
        t=o.t, up=up, heading=o.heading, heading_smooth=o.heading_smooth, meta={"fps": FPS}
    )
    minus.fwd = -G.LENS_V
    assert o.forward_axis == "+z" and minus.forward_axis == "-y"
    return o, minus


def test_flip_is_exact() -> None:
    o, minus = _upright_pair()
    for t in o.t[::37]:
        assert (o.r_vw(t) @ VW.DET_FLIP).tobytes() == minus.r_vw(t).tobytes()


def test_upright_views_are_the_minus_z_views_relabelled(home, tmp_path, monkeypatch) -> None:
    """``compute_views`` on an upright clip: the -z frame's detections, embeddings and crops
    (4b3894b), with yaw + 180 and +z view k = -z view k + 4; the selfie pass too."""
    rec = SimpleNamespace(built=[], closed=0, det_calls=0, embeds=[])
    monkeypatch.setitem(sys.modules, "torch", FAKE_TORCH)
    monkeypatch.setattr(G, "FisheyeSampler", Sampler)
    monkeypatch.setattr(V, "iter_frames", frames)
    lrv = tmp_path / "LRV_001.lrv"
    lrv.write_bytes(b"x")
    clip = X3Clip("001", "2026-01-01", "5.7k", str(lrv), [], frames=600, duration=20.0)
    p = VW.ViewParams(view_px=4, eq_width=32)
    models = Models(rec)
    o, minus = _upright_pair()
    plus = VW.compute_views(clip, models, p, o)
    ref = VW.compute_views(clip, models, p, minus)
    assert len(ref.det) and len(ref.box_idx)
    assert plus.emb.tobytes() == ref.emb[:, VW.DET_PERM].tobytes()
    assert plus.emb[:, 0].tobytes() == ref.emb[:, 4].tobytes()  # +z yaw 0 = -z yaw 180
    assert plus.emb[:, VW.SELFIE].tobytes() == ref.emb[:, VW.SELFIE].tobytes()
    yaw = 3
    rest = [c for c in range(len(VW.DET_COLS)) if c != yaw]
    assert plus.det[:, rest].tobytes() == ref.det[:, rest].tobytes()
    assert np.allclose(plus.det[:, yaw], G.wrap_deg(ref.det[:, yaw] + 180.0), atol=1e-4)
    for k in ("t", "box_emb", "box_idx", "rays"):
        assert getattr(plus, k).tobytes() == getattr(ref, k).tobytes(), k
    assert plus.meta["forward_axis"] == "+z" and plus.meta["det_frame"] == "-z"
    assert "det_frame" not in ref.meta
    # the selfie pass reuses the relabelled rows and permutes its views the same way
    axis = np.array([0.3, 0.7, 0.65]) / np.linalg.norm([0.3, 0.7, 0.65])
    plus2 = VW.compute_views(clip, models, p, o, wearer_v=axis, reuse=plus)
    ref2 = VW.compute_views(clip, models, p, minus, wearer_v=axis, reuse=ref)
    assert plus2.emb.tobytes() == ref2.emb[:, VW.DET_PERM].tobytes()
    assert plus2.det[:, rest].tobytes() == ref2.det[:, rest].tobytes()
    assert plus2.meta["det_frame"] == "-z"
    # the standard visual layer gets 4b3894b's crops (same camera directions)
    vis = {}
    for name, pv in (("plus", plus), ("ref", ref)):
        src = tmp_path / f"{name}.lrv"
        src.write_bytes(b"x")
        with np.load(VW.write_standard_visual(src, pv)) as z:
            vis[name] = {f: z[f].tobytes() for f in z.files}
    assert vis["plus"] == vis["ref"]


def test_upright_viterbi_ties_break_as_in_the_minus_z_frame() -> None:
    """Two equal vistas: +z (bins rolled by n_bins / 2) keeps -z's camera direction."""
    from cutpoint.pano import director as D  # noqa: PLC0415

    p = D.DirectorParams()
    nb = p.n_bins
    ys = np.zeros((20, nb))
    ys[:, [5, 30]] = 1.0  # an exact tie
    sg = {
        "yaw_score": ys,
        "selfie": np.full(20, -9.0),
        "centres": np.arange(nb) * 360.0 / nb,
    }
    ref = D._viterbi(sg, p)
    plus = D._viterbi(sg | {"yaw_score": np.roll(ys, nb // 2, 1)}, p, roll=nb // 2)
    assert np.array_equal((plus + nb // 2) % nb, ref)
    assert np.array_equal(D._viterbi(sg, p, roll=0), ref)
