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

FPS, DT, DUR = 30.0, 0.001, 12.0


def _rot(axis: int, deg: np.ndarray) -> np.ndarray:
    """``(K, 3, 3)`` rotations by ``deg`` about coordinate ``axis``."""
    a = np.radians(np.asarray(deg, float))
    c, s = np.cos(a), np.sin(a)
    i, j = [k for k in range(3) if k != axis]
    r = np.zeros((len(a), 3, 3))
    r[:, axis, axis] = 1.0
    r[:, i, i], r[:, j, j], r[:, i, j], r[:, j, i] = c, c, -s, s
    return r


# camera -> world at rest. Helmet: sky lens (-z) up, FORWARD_V along world +z.
HELMET = _rot(0, [-90.0])[0]
# Upright, lenses horizontal: camera -y up, tipped 6 deg forward (|horizontal| of FORWARD_V
# ~0.1, as on the cruise footage).
UPRIGHT = _rot(0, [6.0])[0]


def _fuse(mount: np.ndarray, turn_dps: float = 0.0, sway_deg: float = 0.0) -> G.Orientation:
    """Fused track of a camera on ``mount`` turning steadily about the vertical and swaying
    (roll and pitch about world axes, a few Hz)."""
    t = np.arange(0, DUR, DT)
    psi = turn_dps * t
    roll = sway_deg * np.sin(2 * np.pi * 1.3 * t)
    pitch = sway_deg * np.sin(2 * np.pi * 0.7 * t + 1.0)
    rm = _rot(1, psi) @ _rot(2, roll) @ _rot(0, pitch) @ mount
    rel = np.einsum("kji,kjl->kil", rm[:-1], rm[1:])  # small body-frame steps
    w = np.stack(
        [rel[:, 2, 1] - rel[:, 1, 2], rel[:, 0, 2] - rel[:, 2, 0], rel[:, 1, 0] - rel[:, 0, 1]], 1
    ) / (2 * DT)
    w = np.vstack([np.zeros(3), w])
    a_v = np.einsum("kji,j->ki", rm, [0.0, -G.G, 0.0])
    return G.fuse(t, a_v @ G.M_IMU_TO_V, np.degrees(w) @ G.M_IMU_TO_V, FPS, int(DUR * FPS) - 1)


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
    o = _fuse(mount, turn_dps=20.0, sway_deg=25.0)
    assert o.forward_axis == axis
    for part in (slice(0, 120), slice(100, 220), slice(200, None)):
        assert G.axis_name(G.forward_axis(o.up[part])) == axis


# --------------------------------------------------------------------------- upright frames
def test_upright_turn_has_a_smooth_display_heading_in_both_frames() -> None:
    """Turning 30 deg/s while swaying: the display yaw follows the turn without jumps in the
    view layer's ``r_vw`` and the renderer's frame, and the two frames agree."""
    o = _fuse(UPRIGHT, turn_dps=30.0, sway_deg=5.0)
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
    o = _fuse(UPRIGHT, sway_deg=8.0)
    view = np.stack([o.r_vw(i / FPS) for i in IDX])
    # the view layer's gyro heading wobbles a few degrees with sway (3 deg on the helmet mount)
    assert np.ptp(_display_yaw(o, view, IDX)) < 6.0
    assert np.ptp(_display_yaw(o, render.stabilised_r_vw(o, IDX, 0.7, 0.5), IDX)) < 1.0


def test_the_helmet_axis_was_noisy_on_upright_footage() -> None:
    """The bug: levelled by ``FORWARD_V``, the same still upright camera swings its view."""
    o = _fuse(UPRIGHT, sway_deg=8.0)
    old = np.stack([G.level_matrix(o.up[i]) for i in IDX])
    assert np.ptp(_display_yaw(o, old, IDX)) > 20.0


# --------------------------------------------------------------------------- helmet unchanged
def test_helmet_mount_matrices_are_unchanged() -> None:
    o = _fuse(HELMET, turn_dps=30.0, sway_deg=15.0)
    assert o.fwd is G.FORWARD_V and "forward_axis" not in o.meta
    for i in IDX[::17]:
        before = G.level_matrix(o.up[i]) @ G.rot_y(o.heading[i] - o.heading_smooth[i])
        assert np.array_equal(o.r_vw(i / FPS), before)


def test_helmet_renderer_heading_uses_forward_v() -> None:
    o = _fuse(HELMET, turn_dps=30.0, sway_deg=15.0)
    new = render.stabilised_r_vw(o, IDX, 0.7, 0.5)
    o.fwd = np.array([0.0, -1.0, 0.0])  # a copy: what the code used before
    assert np.array_equal(render.stabilised_r_vw(o, IDX, 0.7, 0.5), new)


def test_upright_axis_logged_once(home, tmp_path, monkeypatch, caplog) -> None:
    src = tmp_path / "up.lrv"
    src.write_bytes(b"x")
    o = _fuse(UPRIGHT)
    monkeypatch.setattr(G, "fuse", lambda *a, **k: o)
    monkeypatch.setattr(G, "load_imu", lambda _p: (None, None, None))
    with caplog.at_level(logging.INFO, logger="cutpoint.pano.geometry"):
        G.orientation(src, Path("i00"), FPS, len(o.t))
        G.orientation(src, Path("i00"), FPS, len(o.t))
    assert sum("_10_ lens axis (+z)" in r.message for r in caplog.records) == 1
    assert G.cached_forward_axis(src) == "+z"
    assert cache.load_json(G._orient_path(src).with_suffix(".json"))["forward_axis"] == "+z"


# --------------------------------------------------------------------------- caches
def _write_orient(src: Path, up: list[float]) -> None:
    path = G._orient_path(src)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, up=np.tile(np.asarray(up, np.float32), (10, 1)))


def _write_views(src: Path, meta: dict) -> None:
    path = VW.cache_path(src)
    path.parent.mkdir(parents=True, exist_ok=True)
    z = np.zeros(0)
    np.savez(path, t=z, emb=z, det=z, box_emb=z, box_idx=z)
    cache.save_json(path.with_suffix(".json"), meta)


@pytest.mark.parametrize(
    ("up", "meta", "valid"),
    [
        ([0.0, 0.0, -1.0], {}, True),  # helmet: as before
        (None, {}, True),  # no orientation cached: taken as it is
        ([0.0, -1.0, 0.0], {}, False),  # upright, computed before the fix
        ([0.0, -1.0, 0.0], {"forward_axis": "+z"}, False),  # d10b5c0: equirect in +z yaw
        ([0.0, -1.0, 0.0], {"forward_axis": "+z", "det_frame": "-z"}, True),
        ([0.0, 0.0, -1.0], {"det_frame": "-z"}, True),  # helmet: no det_frame check
        ([0.0, 0.0, -1.0], {"forward_axis": "+z"}, False),
        ([0.0, -1.0, 0.0], {"forward_axis": "-z"}, False),  # this branch's first sign
    ],
)
def test_views_cache_follows_the_forward_axis(home, tmp_path, up, meta, valid) -> None:
    src = tmp_path / "clip.lrv"
    src.write_bytes(b"x")
    if up is not None:
        _write_orient(src, up)
    _write_views(src, {"source": str(src)} | meta)
    assert (VW.load_views(src) is not None) is valid


def test_a_stale_views_cache_is_logged_once(home, tmp_path, caplog) -> None:
    src = tmp_path / "stale.lrv"
    src.write_bytes(b"x")
    _write_orient(src, [0.0, -1.0, 0.0])
    _write_views(src, {"source": str(src)})
    with caplog.at_level(logging.INFO, logger="cutpoint.pano.views"):
        assert VW.load_views(src) is None and VW.load_views(src) is None
    assert sum("cutpoint 360 analyze" in r.message for r in caplog.records) == 1


def test_a_cached_track_records_its_axis_in_memory_only_when_upright(home, tmp_path) -> None:
    """Tracks cached before the fix have no ``forward_axis`` in their meta: it is added on
    load for an upright clip, never for a helmet one (whose meta stays as it was)."""
    for up, want in (([0.0, -1.0, 0.0], "+z"), ([0.0, 0.0, -1.0], None)):
        src = tmp_path / f"{want}.lrv"
        src.write_bytes(b"x")
        path = G._orient_path(src)
        path.parent.mkdir(parents=True, exist_ok=True)
        z = np.zeros(10, np.float32)
        np.savez(path, t=z, up=np.tile(np.float32(up), (10, 1)), heading=z, heading_smooth=z, q=z)
        cache.save_json(path.with_suffix(".json"), {"fps": 30.0})
        assert G.load_orientation(src).meta.get("forward_axis") == want
        assert cache.load_json(path.with_suffix(".json")) == {"fps": 30.0}


def test_director_reuse_key_has_the_axis_only_when_upright(tmp_path) -> None:
    seg = ("001", 1.0, 5.0, "people", False, {"a": 1})
    assert PC._plan_key(*seg) == json.dumps([["001", 1.0, 5.0, "people", False], {"a": 1}])
    assert PC._plan_key(*seg, "+z") != PC._plan_key(*seg)
    from cutpoint.pano.director import PATH_VERSION  # noqa: PLC0415

    base = {
        "version": PATH_VERSION,
        "clip": "001",
        "segment": {"start": 1.0, "end": 5.0},
        "kind": "people",
        "vertical": False,
        "params": {"a": 1},
    }
    cache.save_json(tmp_path / "001_a.json", base)
    upright = {"clip": "002", "forward_axis": "+z", "det_frame": VW.DET_FRAME}
    cache.save_json(tmp_path / "002_b.json", base | upright)
    cache.save_json(tmp_path / "003_c.json", base | upright | {"clip": "003", "det_frame": None})
    found = PC._reusable(tmp_path)
    assert PC._plan_key(*seg) in found
    now = PC._axis_key("+z", VW.DET_FRAME)
    assert PC._plan_key("002", *seg[1:], now) in found
    assert PC._plan_key("002", *seg[1:]) not in found  # planned upright, now default: re-plan
    assert PC._plan_key("003", *seg[1:], now) not in found  # d10b5c0's +z views: re-plan


# --------------------------------------------------------------------------- yaw 0 and meta
def test_upright_yaw_0_follows_the_10_lens() -> None:
    """Upright, still or turning: display yaw 0 is the ``_10_`` lens (camera +z), the
    ``_00_`` lens (-z) is behind."""
    for o in (_fuse(UPRIGHT, sway_deg=8.0), _fuse(UPRIGHT, turn_dps=30.0, sway_deg=5.0)):
        for t in IDX[::40] / FPS:
            r = o.r_vw(float(t))
            front, _ = G.ray_yaw_pitch(r.T @ np.array([0.0, 0.0, 1.0]))
            back, _ = G.ray_yaw_pitch(r.T @ np.array([0.0, 0.0, -1.0]))
            assert abs(front) < 10.0 and abs(abs(back) - 180.0) < 10.0


def test_upright_wearer_prior_is_the_helmet_one() -> None:
    """The wearer prior does not depend on the mount (the upright one is left to #573): too
    few large person boxes give ``WEARER_V`` on a helmet and an upright clip alike."""
    for name, meta in (("helmet-wearer.lrv", {}), ("upright-wearer.lrv", {"forward_axis": "+z"})):
        pv = VW.PanoViews(
            np.zeros(1), np.zeros((1, 9, 4)), np.zeros((0, 11)), np.zeros(0), np.zeros(0)
        )
        pv.meta = {"source": name} | meta
        assert VW.wearer_axis(pv) is G.WEARER_V


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
    import test_pano_selfie_recompute as SR  # noqa: PLC0415

    rec = SimpleNamespace(built=[], closed=0, det_calls=0, embeds=[])
    monkeypatch.setitem(sys.modules, "torch", SR.FAKE_TORCH)
    monkeypatch.setattr(G, "FisheyeSampler", SR._Sampler)
    monkeypatch.setattr(V, "iter_frames", SR._frames)
    lrv = tmp_path / "LRV_001.lrv"
    lrv.write_bytes(b"x")
    clip = X3Clip("001", "2026-01-01", "5.7k", str(lrv), [], frames=600, duration=20.0)
    p = VW.ViewParams(view_px=4, eq_width=32)
    models = SR._Models(rec)
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
