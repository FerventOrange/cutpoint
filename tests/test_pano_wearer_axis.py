"""The 360 wearer direction per clip (#573): :func:`geometry.wearer_direction` from the
forward axis and the ``up`` track, and its use by the view layer (``wearer_axis``, the
first selfie pass, the cache check) and the director. Helmet clips keep ``WEARER_V``."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint import cache, insta360
from cutpoint.events import visual as V
from cutpoint.pano import director as D
from cutpoint.pano import geometry as G
from cutpoint.pano import views as VW
from cutpoint.pano.x3 import X3Clip
from pano_fakes import FAKE_TORCH, Models, Sampler, frames

HELMET_UP = [0.0, 0.0, -1.0]  # sky lens up
UPRIGHT_UP = [0.0, -1.0, 0.0]  # lenses horizontal, camera -y up (the cruise footage)


def _deg(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(
        math.acos(float(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
    )


def _orient(up: list[float], n: int = 600, fps: float = 30.0) -> G.Orientation:
    return G.Orientation(
        t=np.arange(n) / fps,
        up=np.tile(np.asarray(up, float), (n, 1)),
        heading=np.zeros(n),
        heading_smooth=np.zeros(n),
        meta={"fps": fps},
    )


def _write_orient(src: Path, up: list[float]) -> None:
    path = G._orient_path(src)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, up=np.tile(np.asarray(up, np.float32), (10, 1)))


def _write_views(src: Path, meta: dict, det: np.ndarray | None = None) -> None:
    path = VW.cache_path(src)
    path.parent.mkdir(parents=True, exist_ok=True)
    z = np.zeros(0)
    d = np.zeros((0, len(VW.DET_COLS)), np.float32) if det is None else det
    np.savez(path, t=z, emb=z, det=d, box_emb=z, box_idx=z)
    cache.save_json(path.with_suffix(".json"), meta)


def _boxes(v: list[float], n: int = 20) -> np.ndarray:
    """``n`` confident person rows 60 deg tall centred on camera direction ``v``."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    row = [0, 1, 0.95, 0.0, 0.0, 30.0, 60.0, 0.0, *v]
    return np.tile(np.asarray(row, np.float32), (n, 1))


UPRIGHT_META = {"forward_axis": "+z", "det_frame": VW.DET_FRAME}


# --------------------------------------------------------------------------- geometry
def test_helmet_keeps_the_measured_mount() -> None:
    for up in (HELMET_UP, [0.3, 0.2, -0.93], [-1.0, 0.0, 0.0]):
        assert G.wearer_direction(np.tile(up, (5, 1))) is G.WEARER_V
    assert G.wearer_direction(np.zeros((0, 3))) is G.WEARER_V
    assert _orient(HELMET_UP).wearer is G.WEARER_V


def test_upright_wearer_is_below_the_camera_behind_the_travel_lens() -> None:
    w = G.wearer_direction(np.tile(UPRIGHT_UP, (5, 1)))
    a = math.radians(G.UPRIGHT_WEARER_TILT_DEG)
    assert np.allclose(w, [0.0, math.cos(a), -math.sin(a)])
    assert np.linalg.norm(w) == pytest.approx(1.0)
    assert _orient(UPRIGHT_UP).wearer.tobytes() == w.tobytes()


def test_upright_wearer_follows_the_median_down_of_a_tipped_camera() -> None:
    """A camera tipped 20 deg about x and swaying: the wearer stays the tilt from the
    median down, on the side away from the _10_ lens."""
    k = np.arange(400)
    tip = math.radians(20.0)
    up = np.stack(
        [0.05 * np.sin(k / 9), -math.cos(tip) * np.ones(len(k)), math.sin(tip) + 0 * k], 1
    )
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    assert G.axis_name(G.forward_axis(up)) == "+z"
    w = G.wearer_direction(up)
    down = -np.median(up, 0)
    assert _deg(w, down) == pytest.approx(G.UPRIGHT_WEARER_TILT_DEG, abs=1e-6)
    assert w @ G.LENS_V < down @ G.LENS_V / np.linalg.norm(down)  # leaning away from +z
    # far from the helmet default, which points 47 deg toward the travel lens here
    assert _deg(w, G.WEARER_V) > 45.0


def test_cached_wearer_direction(home, tmp_path) -> None:
    src = tmp_path / "a.lrv"
    src.write_bytes(b"x")
    assert G.cached_wearer_direction(src) is None
    _write_orient(src, UPRIGHT_UP)
    assert np.allclose(G.cached_wearer_direction(src), _orient(UPRIGHT_UP).wearer)
    _write_orient(src, HELMET_UP)
    assert G.cached_wearer_direction(src) is G.WEARER_V


# --------------------------------------------------------------------------- view layer
def _pv(src: Path, meta: dict, det: np.ndarray) -> VW.PanoViews:
    z = np.zeros(0)
    return VW.PanoViews(t=z, emb=z, det=det, box_emb=z, box_idx=z, meta={"source": str(src)} | meta)


def test_wearer_axis_of_an_upright_clip_comes_from_its_track(home, tmp_path) -> None:
    """Close people beside the pole (the cruise neighbours) do not move the wearer."""
    src = tmp_path / "up.lrv"
    src.write_bytes(b"x")
    _write_orient(src, UPRIGHT_UP)
    VW._AXES.clear()
    axis = VW.wearer_axis(_pv(src, UPRIGHT_META, _boxes([0.52, 0.75, -0.41])))
    assert np.allclose(axis, _orient(UPRIGHT_UP).wearer)


def test_wearer_axis_of_a_helmet_clip_is_unchanged(home, tmp_path) -> None:
    src = tmp_path / "helmet.lrv"
    src.write_bytes(b"x")
    _write_orient(src, HELMET_UP)
    VW._AXES.clear()
    v = [0.3, 0.7, 0.65]
    assert np.allclose(VW.wearer_axis(_pv(src, {}, _boxes(v))), np.asarray(v) / np.linalg.norm(v))
    VW._AXES.clear()
    assert VW.wearer_axis(_pv(src, {}, _boxes(v, n=3))) is G.WEARER_V  # too few boxes


def test_wearer_axis_of_an_upright_clip_without_a_track_uses_the_boxes(home, tmp_path) -> None:
    src = tmp_path / "notrack.lrv"
    src.write_bytes(b"x")
    VW._AXES.clear()
    v = [0.3, 0.7, 0.65]
    assert np.allclose(
        VW.wearer_axis(_pv(src, UPRIGHT_META, _boxes(v))), np.asarray(v) / np.linalg.norm(v)
    )


def test_wearer_cone_is_narrower_only_on_an_upright_clip_with_a_track(home, tmp_path) -> None:
    up, helmet, notrack = (tmp_path / f"{n}.lrv" for n in ("up", "helmet", "notrack"))
    for src in (up, helmet, notrack):
        src.write_bytes(b"x")
    _write_orient(up, UPRIGHT_UP)
    _write_orient(helmet, HELMET_UP)
    no_det = np.zeros((0, len(VW.DET_COLS)), np.float32)
    assert VW.wearer_cone(_pv(up, UPRIGHT_META, no_det)) == G.UPRIGHT_WEARER_CONE_DEG
    assert VW.wearer_cone(_pv(up, UPRIGHT_META, no_det), 40.0) == G.UPRIGHT_WEARER_CONE_DEG
    assert VW.wearer_cone(_pv(helmet, {}, no_det)) == G.WEARER_CONE_DEG
    assert VW.wearer_cone(_pv(helmet, {}, no_det), 40.0) == 40.0
    assert VW.wearer_cone(_pv(notrack, UPRIGHT_META, no_det)) == G.WEARER_CONE_DEG
    assert G.UPRIGHT_WEARER_CONE_DEG < G.WEARER_CONE_DEG


def test_upright_neighbours_are_people_not_the_wearer(home, tmp_path) -> None:
    """A paddler seated beside the pole (40 deg off the nadir, sideways) is another person;
    the wearer's own box below the camera is not; the --paths QA's self tags agree."""
    from cutpoint.pano import qa_paths  # noqa: PLC0415

    src = tmp_path / "raft.lrv"
    src.write_bytes(b"x")
    _write_orient(src, UPRIGHT_UP)
    a = math.radians(40.0)
    det = np.concatenate([_boxes([math.sin(a), math.cos(a), 0.0], 1), _boxes([0.0, 1.0, -0.1], 1)])
    det[:, 6] = 40.0  # under the 70 deg height rule
    pv = _pv(src, UPRIGHT_META, det)
    VW._AXES.clear()
    assert VW.other_people(pv).tolist() == [True, False]
    assert qa_paths.person_rows(pv)[1].tolist() == [False, True]


@pytest.mark.parametrize(
    ("up", "meta", "valid"),
    [
        (UPRIGHT_UP, UPRIGHT_META, True),  # no wearer_v recorded: kept
        (
            UPRIGHT_UP,
            UPRIGHT_META | {"wearer_v": list(G.WEARER_V)},
            False,
        ),  # selfie at the helmet axis
        (UPRIGHT_UP, UPRIGHT_META | {"wearer_v": [0.52, 0.75, -0.41]}, False),  # at the neighbours
        (UPRIGHT_UP, UPRIGHT_META | {"wearer_v": [0.0, 0.97, -0.26]}, True),  # the new direction
        (UPRIGHT_UP, UPRIGHT_META | {"wearer_v": [0.0, 1.0, 0.0]}, True),  # within 25 deg
        (HELMET_UP, {"wearer_v": [0.52, 0.75, -0.41]}, True),  # helmet: never checked
    ],
)
def test_views_cache_follows_the_upright_wearer(home, tmp_path, up, meta, valid) -> None:
    src = tmp_path / "clip.lrv"
    src.write_bytes(b"x")
    _write_orient(src, up)
    _write_views(src, {"source": str(src)} | meta)
    assert (VW.load_views(src) is not None) is valid


def test_a_stale_upright_selfie_is_logged_once(home, tmp_path, caplog) -> None:
    src = tmp_path / "stale_selfie.lrv"
    src.write_bytes(b"x")
    _write_orient(src, UPRIGHT_UP)
    _write_views(src, {"source": str(src), "wearer_v": list(G.WEARER_V)} | UPRIGHT_META)
    with caplog.at_level(logging.INFO, logger="cutpoint.pano.views"):
        assert VW.load_views(src) is None and VW.load_views(src) is None
    assert sum("another wearer direction" in r.message for r in caplog.records) == 1


@pytest.fixture
def pano(home, tmp_path: Path, monkeypatch):
    lrv = tmp_path / "LRV_001.lrv"
    lrv.write_bytes(b"x")
    clip = X3Clip("001", "2026-01-01", "5.7k", str(lrv), [], frames=600, duration=20.0)
    rec = SimpleNamespace(built=[], closed=0, det_calls=0, embeds=[], clip=clip, lrv=lrv)
    monkeypatch.setitem(sys.modules, "torch", FAKE_TORCH)
    monkeypatch.setattr(G, "FisheyeSampler", Sampler)
    monkeypatch.setattr(V, "iter_frames", frames)
    monkeypatch.setattr(V, "VisualModels", lambda **kw: Models(rec, **kw))
    monkeypatch.setattr(V, "text_embeddings", lambda prompts, models=None: None)
    monkeypatch.setattr(insta360, "imu_source", lambda src: src)
    rec.p = VW.ViewParams(view_px=4, eq_width=32)
    return rec


def test_run_layer_looks_at_the_upright_wearer_in_one_pass(pano, monkeypatch) -> None:
    """The first selfie pass of an upright clip already looks below the camera, so the
    box median cannot trigger a second (selfie-only) pass."""
    o = _orient(UPRIGHT_UP)
    monkeypatch.setattr(V, "require_detector", lambda: None)  # the fake models need no file
    monkeypatch.setattr(G, "orientation", lambda *a: o)
    _write_orient(pano.lrv, UPRIGHT_UP)  # what G.orientation caches in a real run
    seen = []
    real = VW.compute_views

    def spy(clip, models, p, orient, wearer_v=G.WEARER_V, reuse=None):
        seen.append(np.asarray(wearer_v).copy())
        return real(clip, models, p, orient, wearer_v=wearer_v, reuse=reuse)

    monkeypatch.setattr(VW, "compute_views", spy)
    VW._AXES.clear()
    assert VW.run_layer([pano.clip], pano.p)["computed"] == 1
    assert len(seen) == 1 and seen[0].tobytes() == o.wearer.tobytes()
    pv = VW.load_views(pano.lrv, pano.p)
    assert pv is not None and _deg(pv.meta["wearer_v"], o.wearer) < 0.01
    assert np.allclose(VW.wearer_axis(pv), o.wearer)


def test_helmet_run_layer_starts_at_the_measured_mount(pano, monkeypatch) -> None:
    o = _orient(HELMET_UP)
    monkeypatch.setattr(V, "require_detector", lambda: None)  # the fake models need no file
    monkeypatch.setattr(G, "orientation", lambda *a: o)
    seen = []
    real = VW.compute_views

    def spy(clip, models, p, orient, wearer_v=G.WEARER_V, reuse=None):
        seen.append(wearer_v)
        return real(clip, models, p, orient, wearer_v=wearer_v, reuse=reuse)

    monkeypatch.setattr(VW, "compute_views", spy)
    VW.run_layer([pano.clip], pano.p)
    assert seen[0] is G.WEARER_V


# --------------------------------------------------------------------------- director
@pytest.mark.parametrize("up", [HELMET_UP, UPRIGHT_UP])
def test_director_without_views_places_the_wearer_from_the_track(tmp_path, monkeypatch, up) -> None:
    """No view layer: the wearer penalty uses the clip's wearer direction (the helmet
    mount's WEARER_V on a helmet clip, as before)."""
    o = _orient(up)
    got = []

    def wiv(orient, axis, *a, **k):
        got.append(axis)
        return np.zeros((len(a[0]), len(a[1])))

    monkeypatch.setattr(D, "wearer_in_view", wiv)
    monkeypatch.setattr(D, "_voice_at", lambda src, times: np.zeros(len(times)))
    clip = SimpleNamespace(source=str(tmp_path / "x.lrv"))
    D.score_grid(clip, 0.0, 2.0, D.DirectorParams(), None, orient=o)
    assert got and got[0] is o.wearer
    if up == HELMET_UP:
        assert got[0] is G.WEARER_V


# ------------------------------------------------------------------ director path reuse
def _plan_twice(tmp_path, monkeypatch, axis: str, edit=None, views=True) -> tuple[int, dict]:
    """Run ``360 director`` (no --force) twice on one segment of a clip whose forward axis is
    ``axis``; ``edit`` changes the saved path in between; ``views``: the clip's view layer is
    current in the first run. Returns the plans of the second run and the path saved by the
    first."""
    import argparse  # noqa: PLC0415
    from dataclasses import asdict  # noqa: PLC0415

    from cutpoint.pano import cli as PC  # noqa: PLC0415
    from cutpoint.pano import project as P  # noqa: PLC0415

    proj = tmp_path / "proj"
    proj.mkdir(parents=True)
    clip = SimpleNamespace(clip_id="001", lrv="x.lrv", source="x.lrv")
    seg = {"source": "x.lrv", "start": 1.0, "end": 4.0, "label": "a"}
    monkeypatch.setattr(PC, "_project", lambda arg: proj)
    monkeypatch.setattr(P, "load", lambda project: {"x.lrv": clip})
    monkeypatch.setattr(P, "clip_for", lambda clips, src: clip)
    monkeypatch.setattr(PC, "_selected_segments", lambda project, a: [seg])
    monkeypatch.setattr(G, "cached_forward_axis", lambda src: axis)
    monkeypatch.setattr(VW, "load_views", lambda src: SimpleNamespace() if views else None)
    planned: list[str] = []

    def compute(clip, start, end, *, vertical, kind, **kw):
        planned.append(clip.clip_id)
        path = {
            "version": D.PATH_VERSION,
            "clip": clip.clip_id,
            "segment": {"start": start, "end": end},
            "kind": kind,
            "vertical": vertical,
            "params": asdict(D.params_for(kind)),
            "keys": [],
            "cuts": [],
        }
        if axis != G.axis_name(G.FORWARD_V):  # as compute_path writes an upright path
            path |= {"forward_axis": axis, "det_frame": VW.DET_FRAME}
        return path

    monkeypatch.setattr(D, "compute_path", compute)
    a = argparse.Namespace(
        project="proj", vertical=False, preview=False, param=[], out_name=None, force=False
    )
    PC.cmd_director(a)
    f = proj / "pano_paths" / "director" / "001_a.json"
    saved = cache.load_json(f)
    if edit is not None:
        cache.save_json(f, edit(dict(saved)))
    planned.clear()
    views = True  # `360 analyze` has run since
    PC.cmd_director(a)
    return len(planned), saved


def test_upright_director_paths_record_the_wearer_model_and_are_reused(tmp_path, monkeypatch):
    plans, saved = _plan_twice(tmp_path, monkeypatch, "+z")
    assert saved["wearer_model"] == G.UPRIGHT_WEARER
    assert plans == 0


def test_upright_paths_of_another_wearer_model_are_planned_again(tmp_path, monkeypatch):
    def old(path):  # planned before the upright wearer model (#573)
        del path["wearer_model"]
        return path

    assert _plan_twice(tmp_path, monkeypatch, "+z", old)[0] == 1
    assert _plan_twice(tmp_path / "b", monkeypatch, "+z", lambda p: p | {"wearer_model": "x"})[0]


def test_helmet_director_paths_have_no_wearer_model_and_stay_reused(tmp_path, monkeypatch):
    plans, saved = _plan_twice(tmp_path, monkeypatch, G.axis_name(G.FORWARD_V))
    assert "wearer_model" not in saved and "forward_axis" not in saved
    assert plans == 0


def test_upright_paths_planned_without_current_views_are_planned_again(
    tmp_path, monkeypatch, caplog
):
    with caplog.at_level(logging.WARNING, logger="cutpoint"):
        plans, saved = _plan_twice(tmp_path, monkeypatch, "+z", views=False)
    assert "wearer_model" not in saved  # planned without the view layer: not final
    assert plans == 1
    assert sum("360 analyze" in r.getMessage() for r in caplog.records) == 1
