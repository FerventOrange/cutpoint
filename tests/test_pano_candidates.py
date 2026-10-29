"""Director paths of the analysis candidates (#517): cache hit / miss / invalidation, reuse in
``clips`` (equal to a fresh path), diagnostics and flags on crafted paths, and the read API.

No models or footage: the view layer is synthetic and its CLIP scores are stubbed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest

import pano_helpers as helpers
from conftest import set_config
from cutpoint import analyze as analyze_mod
from cutpoint import cache, jsonout
from cutpoint.events import pipeline
from cutpoint.pano import candidates as C
from cutpoint.pano import cli
from cutpoint.pano import director as D
from cutpoint.pano import geometry as G
from cutpoint.pano import render as R
from cutpoint.pano import views as VW

N = 20  # view frames, every 1.2 s
FWD_V = [0.0, -1.0, 0.0]
FRIEND = [(0.95, 40.0 + 2 * f, -10.0, 10.0, 30.0, FWD_V) for f in range(N)]


@pytest.fixture
def stub_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the CLIP view scores, voice and the orientation track."""
    group = np.zeros((N, 9), np.float32)
    monkeypatch.setattr(VW, "view_scores", lambda pv: {"group_zs": group, "beauty": group})
    monkeypatch.setattr(
        VW,
        "occlusion_scores",
        lambda pv: {"occl_zs": np.zeros((N, 8), np.float32), "selfie_ok": np.ones(N)},
        raising=False,
    )
    monkeypatch.setattr(VW, "forward_yaw", lambda pv: 0.0)
    monkeypatch.setattr(D, "_voice_at", lambda src, times: np.zeros(len(times)))
    monkeypatch.setattr(G, "orientation", lambda *a, **k: helpers.orientation(n=900))


@pytest.fixture
def project(tmp_path: Path, stub_scores: None) -> tuple[Path, object]:
    """A 360 project with clip 001 (views cached) and highlights: two 360 candidates and
    one regular video."""
    lrv = tmp_path / "cam" / helpers.LRV
    lrv.parent.mkdir()
    lrv.write_bytes(b"x")
    clip = helpers.x3clip(lrv=str(lrv), frames=720, duration=24.0)
    proj = tmp_path / "proj"
    proj.mkdir()
    cache.save_json(proj / "pano.json", {"clips": [clip.to_json()]})
    VW.save_views(
        lrv, helpers.make_views({f: [FRIEND[f]] for f in range(N)}, str(lrv), n=N), VW.ViewParams()
    )
    cands = [
        {
            "id": "h001",
            "source": str(lrv),
            "start": 4.0,
            "end": 12.5,
            "kind": "people",
            "label": "a",
        },
        {"id": "h002", "source": str(lrv), "start": 14.0, "end": 20.0, "label": "b"},
        {"id": "h003", "source": str(tmp_path / "flat.mp4"), "start": 1.0, "end": 5.0},
    ]
    cache.save_json(proj / "highlights.json", {"style": "ski", "candidates": cands})
    return proj, clip


def _fresh(clip, start, end, kind=None, overrides=None):
    return D.compute_path(clip, start, end, kind=kind, overrides=overrides)


def _dump(x) -> str:
    return json.dumps(x, sort_keys=True)


# --------------------------------------------------------------------------- cache
def test_analyze_plans_caches_and_clips_reuses_an_equal_path(project, monkeypatch) -> None:
    proj, clip = project
    st = C.cache_candidate_paths(proj)
    assert (st["candidates"], st["computed"], st["cached"], st["failed"]) == (2, 2, 0, 0)
    files = sorted((proj / C.CACHE_DIR).glob("*.json"))
    assert len(files) == 2 and st["dir"] == proj / C.CACHE_DIR
    entry = json.loads(files[0].read_text(encoding="utf-8"))
    assert {"version", "key", "candidate", "diagnostics", "path"} <= entry.keys()
    assert {"sky", "peak_yaw_rate", "cuts_per_min", "wearer_in_view", "off_target"} <= set(
        entry["diagnostics"]
    )
    # the paths equal fresh ones, as clips would plan them
    fresh = _fresh(clip, 4.0, 12.5, "people")
    got = C.reusable_path(proj, clip, 4.0, 12.5, "people", None)
    assert got is not None and _dump(got) == _dump(fresh)

    with monkeypatch.context() as m:
        m.setattr(D, "compute_path", lambda *a, **k: pytest.fail("re-planned"))
        st = C.cache_candidate_paths(proj)  # a second analysis keeps them
        assert (st["computed"], st["cached"]) == (0, 2)
        # clips: path_for takes the cached path
        path, pfile = R.PanoExporter(proj, proj / "clips").path_for(
            clip, "001_a.mp4", 4.0, 12.5, "people"
        )
    assert _dump(path) == _dump(fresh)
    # the clip's path file is the one a run without the cache writes
    for f in files:
        f.unlink()
    _, pfile2 = R.PanoExporter(proj, proj / "clips_fresh").path_for(
        clip, "001_a.mp4", 4.0, 12.5, "people"
    )
    assert pfile2.parent != pfile.parent and pfile.read_bytes() == pfile2.read_bytes()


def test_clips_recomputes_on_any_mismatch(project, home, monkeypatch) -> None:
    proj, clip = project
    C.cache_candidate_paths(proj)
    calls = []
    real = D.compute_path

    def spy(*a, **k):
        calls.append(a[1:3])
        return real(*a, **k)

    monkeypatch.setattr(D, "compute_path", spy)
    ex = R.PanoExporter(proj, proj / "clips")
    ex.path_for(clip, "001_a.mp4", 4.0, 12.5, "people")
    assert calls == []  # hit
    ex.path_for(clip, "001_b.mp4", 4.0, 13.1, "people")  # tail / speech pass moved the end
    ex.path_for(clip, "001_c.mp4", 4.0, 12.5, "speed")  # another kind
    assert calls == [(4.0, 13.1), (4.0, 12.5)]
    assert C.reusable_path(proj, clip, 4.0, 12.5, "people", None, vertical=True) is None
    # other director settings (config.json pano.director)
    set_config(home, pano={"director": {"max_speed": 30}})
    assert C.reusable_path(proj, clip, 4.0, 12.5, "people", {"max_speed": 30}) is None
    calls.clear()
    R.PanoExporter(proj, proj / "clips2").path_for(clip, "001_a.mp4", 4.0, 12.5, "people")
    assert calls == [(4.0, 12.5)]


def test_clips_logs_why_it_re_plans(project, home, monkeypatch, caplog) -> None:
    proj, clip = project
    ex = R.PanoExporter(proj, proj / "clips")
    with caplog.at_level(logging.INFO):
        ex.path_for(clip, "001_a.mp4", 4.0, 12.5, "people")  # no analysis cache: silent
    assert "re-planned" not in caplog.text
    C.cache_candidate_paths(proj)

    def lines(*args, overrides=None) -> list[str]:
        caplog.clear()
        ex.overrides = overrides
        with caplog.at_level(logging.INFO):
            ex.path_for(clip, *args)
        return [r.getMessage() for r in caplog.records if "director path" in r.getMessage()]

    assert lines("001_a.mp4", 4.0, 12.5, "people") == [
        "  001_a.mp4: director path planned by 360 analyze"
    ]
    assert lines("001_b.mp4", 4.0, 13.1, "people") == [  # a smart tail moved the end
        "  001_b.mp4: director path re-planned (span differs from 360 analyze)"
    ]
    assert lines("001_c.mp4", 4.0, 12.5, "speed") == [
        "  001_c.mp4: director path re-planned (settings changed)"
    ]
    assert lines("001_d.mp4", 4.0, 12.5, "people", overrides={"max_speed": 30}) == [
        "  001_d.mp4: director path re-planned (settings changed)"
    ]
    caplog.clear()
    with caplog.at_level(logging.INFO):
        R.PanoExporter(proj, proj / "clips", vertical=True).path_for(
            clip, "001_v.mp4", 4.0, 12.5, "people"
        )
    assert [r.getMessage() for r in caplog.records if "director path" in r.getMessage()] == [
        "  001_v.mp4: director path re-planned (9:16 not planned by 360 analyze)"
    ]
    views = VW.cache_path(Path(clip.lrv))
    st = views.stat()
    os.utime(views, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # 360 analyze --force-views
    assert lines("001_e.mp4", 4.0, 12.5, "people") == [
        "  001_e.mp4: director path re-planned (layers changed)"
    ]
    monkeypatch.setattr(C, "code_id", lambda: "0123456789ab")
    assert C.cached_path(proj, clip, 14.0, 20.0, None, None) == (
        None,
        "layers changed, planning code changed",
    )


def test_changed_views_layer_invalidates(project, monkeypatch) -> None:
    proj, clip = project
    C.cache_candidate_paths(proj)
    assert C.reusable_path(proj, clip, 14.0, 20.0, None, None) is not None
    lrv = Path(clip.lrv)
    views = VW.cache_path(lrv)
    st = views.stat()
    os.utime(views, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # 360 analyze --force-views
    assert C.reusable_path(proj, clip, 14.0, 20.0, None, None) is None
    st = C.cache_candidate_paths(proj)
    assert (st["computed"], st["cached"]) == (2, 0)
    assert C.reusable_path(proj, clip, 14.0, 20.0, None, None) is not None


def test_changed_planning_code_invalidates(project, monkeypatch) -> None:
    proj, clip = project
    C.cache_candidate_paths(proj)
    assert C.reusable_path(proj, clip, 14.0, 20.0, None, None) is not None
    assert len(C.code_id()) == 12 and C.code_id() == C.code_id()
    monkeypatch.setattr(C, "code_id", lambda: "0123456789ab")  # director code changed
    assert C.reusable_path(proj, clip, 14.0, 20.0, None, None) is None
    assert C.candidate_path(proj, "h002")["current"] is False
    st = C.cache_candidate_paths(proj)
    assert (st["computed"], st["cached"]) == (2, 0)


def test_code_id_covers_the_modules_the_path_reads() -> None:
    import importlib.util  # noqa: PLC0415

    assert {"cutpoint.pano.director", "cutpoint.pano.views", "cutpoint.events.voice"} <= set(
        C.CODE_MODULES
    )
    assert all(importlib.util.find_spec(m) is not None for m in C.CODE_MODULES)


def test_stale_entries_are_pruned_and_the_folder_removed(project) -> None:
    proj, clip = project
    C.cache_candidate_paths(proj)
    hl = cache.load_json(proj / "highlights.json")
    hl["candidates"] = hl["candidates"][1:]
    cache.save_json(proj / "highlights.json", hl)
    st = C.cache_candidate_paths(proj)
    assert (st["computed"], st["cached"]) == (0, 1)
    assert [f.name for f in (proj / C.CACHE_DIR).iterdir()] == [C.slot_name(clip, 14.0, 20.0)]
    hl["candidates"] = []
    cache.save_json(proj / "highlights.json", hl)
    st = C.cache_candidate_paths(proj)
    assert not (proj / C.CACHE_DIR).exists() and st["dir"] is None


def test_offline_clip_keeps_its_entries(project, monkeypatch) -> None:
    proj, _ = project
    C.cache_candidate_paths(proj)
    monkeypatch.setattr(VW, "load_views", lambda *a, **k: None)
    st = C.cache_candidate_paths(proj)
    assert (st["skipped"], st["computed"]) == (2, 0)
    assert len(list((proj / C.CACHE_DIR).glob("*.json"))) == 2


def test_a_failing_path_warns_and_the_rest_is_cached(project, monkeypatch, caplog) -> None:
    proj, _ = project
    real = D.compute_path

    def flaky(clip, start, end, **k):
        if start == 4.0:
            raise RuntimeError("boom")
        return real(clip, start, end, **k)

    monkeypatch.setattr(D, "compute_path", flaky)
    with caplog.at_level(logging.WARNING):
        st = C.cache_candidate_paths(proj)
    assert (st["computed"], st["failed"]) == (1, 1)
    assert "h001: director path not planned (RuntimeError: boom)" in caplog.text


def test_plan_paths_never_fails_the_analysis(tmp_path, monkeypatch, caplog) -> None:
    def boom(project):
        raise OSError("pano.json unreadable")

    monkeypatch.setattr(C, "cache_candidate_paths", boom)
    with caplog.at_level(logging.WARNING):
        cli._plan_paths(tmp_path)
    assert "Director paths not planned: OSError" in caplog.text


@pytest.mark.parametrize("no_events", [True, False])
def test_360_analyze_plans_paths_into_the_envelope_and_timings(
    monkeypatch, tmp_path: Path, no_events: bool
) -> None:
    order: list[str] = []
    monkeypatch.setattr(cli, "_project", lambda arg: tmp_path)
    monkeypatch.setattr(cli, "_clips", lambda project: [argparse.Namespace(lrv="a.lrv")])
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "run_events", lambda *a, **k: order.append("events"))
    monkeypatch.setattr("cutpoint.crash.dense_needs_run", lambda sources: False)
    monkeypatch.setattr(VW, "run_layer", lambda clips, force=False, handoff=None: {})
    monkeypatch.setattr(analyze_mod, "analyze", lambda *a, **k: order.append("analyze"))
    monkeypatch.setattr(analyze_mod, "reanalyze_if_reachable", lambda project: None)
    monkeypatch.setattr("cutpoint.pano.project.default_style", lambda: "ski")
    monkeypatch.setattr("cutpoint.styles.get_style", lambda name: name)
    stats = {"candidates": 2, "computed": 2, "flags": {"sky": 1}, "dir": None, "seconds": 1.5}

    def plan(project):
        order.append("paths")
        return stats

    monkeypatch.setattr(C, "cache_candidate_paths", plan)
    args = argparse.Namespace(
        project="p",
        only=None,
        scan=None,
        no_events=no_events,
        views_only=False,
        force_views=False,
        style=None,
        model="m",
        language=None,
    )
    with jsonout.run("360 analyze") as res:
        cli.cmd_analyze(args)
    assert order == (["analyze", "paths"] if no_events else ["analyze", "events", "paths"])
    assert res.outputs["candidate_paths"] == stats
    timings = json.loads((tmp_path / "pano_timings.json").read_text(encoding="utf-8"))
    assert timings["paths_s"] == 1.5


# --------------------------------------------------------------------------- diagnostics
def keys_path(yaw, pitch=-8.0, fov=100.0, dt=0.5, cuts=(), vertical=False) -> dict:
    yaw = np.broadcast_to(np.asarray(yaw, float), (41,)) if np.ndim(yaw) == 0 else yaw
    keys = [
        {
            "t": round(i * dt, 3),
            "yaw": float(y),
            "pitch": pitch,
            "fov": fov,
            "target": "view",
            **({"cut": True} if round(i * dt, 3) in cuts else {}),
        }
        for i, y in enumerate(yaw)
    ]
    return {"keys": keys, "cuts": list(cuts), "vertical": vertical, "wearer_axis": list(G.WEARER_V)}


def test_clean_path_has_no_flags() -> None:
    d = C.diagnostics(keys_path(0.0), 2.0, 18.0)
    assert d["flags"] == [] and d["sky"] == d["above_horizon"] == 0.0
    assert d["peak_yaw_rate"] == 0.0 and d["cuts"] == 0
    assert d["off_target"] is None and d["wearer_in_view"] is None and d["quality"] is None


def test_sky_staring_path_is_flagged() -> None:
    d = C.diagnostics(keys_path(0.0, pitch=30.0), 2.0, 18.0)
    assert d["sky"] == 1.0 and d["above_horizon"] == 1.0 and d["flags"] == ["sky"]
    # a little above the horizon is not the sky: the horizon stays above the lower third
    d = C.diagnostics(keys_path(0.0, pitch=4.0), 2.0, 18.0)
    assert d["above_horizon"] == 1.0 and d["sky"] == 0.0 and d["flags"] == []


def test_whip_pan_and_cut_heavy_paths_are_flagged() -> None:
    yaw = np.r_[np.zeros(16), np.arange(1, 9) * 45.0, np.full(17, 360.0)]  # 90 deg/s for 4 s
    d = C.diagnostics(keys_path(yaw), 2.0, 18.0)
    assert d["peak_yaw_rate"] > C.WHIP_DEG_S and "whip" in d["flags"]
    # a cut is a jump, not a pan: never a whip
    yaw = np.r_[np.zeros(20), np.full(21, 180.0)]
    d = C.diagnostics(keys_path(yaw, cuts=(10.0,)), 2.0, 18.0)
    assert d["peak_yaw_rate"] == 0.0 and d["cuts"] == 1 and d["flags"] == []
    # one cut in a short span is over the rate, but no cut-heavy path
    d = C.diagnostics(keys_path(yaw, cuts=(10.0,)), 6.0, 14.0)
    assert d["cuts"] == 1 and d["cuts_per_min"] > C.CUTS_MAX_PER_MIN and d["flags"] == []
    cuts = (4.0, 6.0, 8.0, 10.0, 12.0, 14.0)
    yaw = np.array([90.0 * (sum(t >= c for c in cuts) % 2) for t in np.arange(41) * 0.5])
    d = C.diagnostics(keys_path(yaw, cuts=cuts), 2.0, 18.0)
    assert d["cuts_per_min"] > C.CUTS_MAX_PER_MIN and d["flags"] == ["cuts"]


def test_wearer_in_view_is_measured_from_the_orientation() -> None:
    orient = helpers.orientation()
    wy, wp = G.ray_yaw_pitch(orient.r_vw(0.0).T @ G.WEARER_V)
    at = C.diagnostics(keys_path(float(wy), pitch=float(wp)), 2.0, 18.0, orient=orient)
    assert at["wearer_in_view"] == 1.0 and "wearer" in at["flags"]
    away = C.diagnostics(keys_path(float(wy) + 180.0, pitch=0.0), 2.0, 18.0, orient=orient)
    assert away["wearer_in_view"] == 0.0 and away["flags"] == []
    # a deliberate selfie shot aimed at the wearer does not count
    selfie = keys_path(float(wy), pitch=float(wp))
    for k in selfie["keys"]:
        k["target"] = "selfie"
    d = C.diagnostics(selfie, 2.0, 18.0, orient=orient)
    assert d["wearer_in_view"] == 0.0 and d["flags"] == []


def test_off_target_compares_with_the_best_bin() -> None:
    times = np.arange(41) * 0.5
    centres = np.arange(36) * 10.0
    ys = np.zeros((41, 36))
    ys[:, 18] = 1.0  # yaw 180 scores best throughout
    grid = {"times": times, "centres": centres, "yaw_score": ys}
    d = C.diagnostics(keys_path(0.0), 2.0, 18.0, grid=grid)
    assert d["off_target"] == 1.0 and d["flags"] == ["off_target"]
    d = C.diagnostics(keys_path(170.0), 2.0, 18.0, grid=grid)  # 180 is in the frame
    assert d["off_target"] == 0.0
    ys[:, 18] = 0.1  # only a little better: within the margin
    assert C.diagnostics(keys_path(0.0), 2.0, 18.0, grid=grid)["off_target"] == 0.0


def test_failed_quality_is_flagged_and_flags_follow_the_thresholds(monkeypatch) -> None:
    path = keys_path(0.0)
    path["quality"] = {"score": 40.0, "pass": False, "fails": ["edge_cut"], "frames": 3}
    d = C.diagnostics(path, 2.0, 18.0)
    assert d["quality"] == {"score": 40.0, "pass": False, "fails": ["edge_cut"]}
    assert d["flags"] == ["qa_fail"]
    monkeypatch.setattr(C, "SKY_MAX_SHARE", -1.0)
    assert C.path_flags(d) == ["sky", "qa_fail"]
    assert set(C.FLAGS) == {"sky", "whip", "cuts", "wearer", "off_target", "qa_fail"}


# --------------------------------------------------------------------------- read API
def test_read_api_returns_the_cached_path_and_diagnostics(project, home) -> None:
    proj, clip = project
    assert C.candidate_path(proj, "h001") is None  # not planned yet
    assert C.candidate_flags(proj) == {}
    C.cache_candidate_paths(proj)
    got = C.candidate_path(proj, "h001")
    assert got is not None and got["current"] is True
    assert got["file"] == proj / C.CACHE_DIR / C.slot_name(clip, 4.0, 12.5)
    assert _dump(got["path"]) == _dump(_fresh(clip, 4.0, 12.5, "people"))
    assert got["diagnostics"]["reasons"] == {f: C.FLAGS[f] for f in got["diagnostics"]["flags"]}
    assert C.candidate_path(proj, "h003") is None  # not a 360 clip
    assert C.candidate_path(proj, "h999") is None
    assert set(C.candidate_flags(proj)) == {"h001", "h002"}
    # a re-ranked analysis: the ids change, the spans stay
    hl = cache.load_json(proj / "highlights.json")
    hl["candidates"][0]["id"], hl["candidates"][1]["id"] = "h002", "h001"
    cache.save_json(proj / "highlights.json", hl)
    assert C.candidate_path(proj, "h002")["path"]["segment"] == {"start": 4.0, "end": 12.5}
    # other settings: still readable for a preview, but clips would plan it anew
    set_config(home, pano={"director": {"max_speed": 30}})
    assert C.candidate_path(proj, "h002")["current"] is False
