"""Director paths of the analysis candidates (#517): cache hit / miss / invalidation, reuse in
``clips`` (equal to a fresh path), diagnostics and flags on crafted paths, and the read API.

No models or footage: the view layer is synthetic and its CLIP scores are stubbed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
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
from cutpoint.pano import project as P
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
        cli._plan_paths(tmp_path, {})
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


TIMING_RUNS = {
    "scan": ({"scan": ["cam"]}, {"scan_s", "views", "analyze_s", "events_s", "paths_s"}),
    "no_scan": ({}, {"views", "analyze_s", "events_s", "paths_s"}),
    "no_events": ({"no_events": True}, {"views", "analyze_s", "paths_s"}),
    # no earlier file to keep the analysis keys of (see the views_only_keeps test)
    "views_only": ({"views_only": True}, {"views"}),
    "views_only_scan": ({"views_only": True, "scan": ["cam"]}, {"views", "scan_s"}),
}
STALE_TIMINGS = {"scan_s": 99.0, "events_s": 99.0, "analyze_total_s": 99.0, "views": {"old": 1}}


def _rewrite_highlights(*a, **k) -> None:
    """Stub of analyze(): rewrites highlights.json, atomically as the real one does."""
    cache.save_json(Path(a[2]) / "highlights.json", {"candidates": [], "run": "this"})


def _timing_run(monkeypatch, tmp_path: Path) -> list[dict | None]:
    """Stub the steps of ``360 analyze`` for the pano_timings.json tests; seed the
    highlights.json and pano_timings.json of an earlier run. Returns the timings dicts the
    scan got."""
    clock = iter(range(0, 1000, 5))  # every perf_counter call: 5 s later
    monkeypatch.setattr(cli.time, "perf_counter", lambda: float(next(clock)))
    monkeypatch.setattr(cli, "_project", lambda arg: tmp_path)
    monkeypatch.setattr(cli, "_clips", lambda project: [argparse.Namespace(lrv="a.lrv")])
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "run_events", lambda *a, **k: None)
    monkeypatch.setattr("cutpoint.crash.dense_needs_run", lambda sources: False)
    monkeypatch.setattr(VW, "run_layer", lambda clips, force=False, handoff=None: {"computed": 1})
    monkeypatch.setattr(analyze_mod, "analyze", _rewrite_highlights)
    monkeypatch.setattr(analyze_mod, "reanalyze_if_reachable", lambda project: None)
    monkeypatch.setattr("cutpoint.pano.project.default_style", lambda: "ski")
    monkeypatch.setattr("cutpoint.styles.get_style", lambda name: name)
    monkeypatch.setattr(C, "cache_candidate_paths", lambda project: {"seconds": 1.5})
    scanned: list[dict | None] = []

    def scan(project, folders, force=False, only=None, allow_empty=True, timings=None):
        scanned.append(timings)
        timings["scan_s"] = 2.0
        return 1

    monkeypatch.setattr(cli, "_scan", scan)
    cache.save_json(tmp_path / "highlights.json", {"candidates": [], "run": "earlier"})
    cache.save_json(tmp_path / "pano_timings.json", STALE_TIMINGS)
    return scanned


def _run_analyze(**over: object) -> None:
    base = {"project": "p", "only": None, "scan": None, "no_events": False}
    base |= {"views_only": False, "force_views": False, "style": None, "model": "m"}
    cli.cmd_analyze(argparse.Namespace(**{**base, "language": None, **over}))


@pytest.mark.parametrize(("over", "keys"), list(TIMING_RUNS.values()), ids=list(TIMING_RUNS))
def test_360_analyze_timings_hold_this_run_only(
    monkeypatch, tmp_path: Path, over: dict, keys: set[str]
) -> None:
    """pano_timings.json is one dict per run, saved once on every return path: no keys of an
    earlier run (#11), no analyze_total_s any more; total_s times the whole run, scan
    included. --views-only runs here have no earlier file to keep."""
    scanned = _timing_run(monkeypatch, tmp_path)
    if over.get("views_only"):
        (tmp_path / "pano_timings.json").unlink()
    _run_analyze(**over)
    timings = json.loads((tmp_path / "pano_timings.json").read_text(encoding="utf-8"))
    assert set(timings) == keys | {"total_s"}
    assert set(timings["views"]) == {"computed", "total_s"}  # not the earlier run's "old"
    assert timings["views"]["total_s"] > 0
    assert timings.get("scan_s", 2.0) == 2.0
    assert timings.get("paths_s", 1.5) == 1.5
    # total_s: from the run's start (before the scan) to the save
    skip = ("total_s", "paths_s")
    parts = [v for k, v in timings.items() if k.endswith("_s") and k not in skip]
    assert timings["total_s"] > sum(parts) + timings["views"]["total_s"] - 1e-9
    assert len(scanned) == ("scan_s" in keys)


@pytest.mark.parametrize(
    ("over", "step", "exc"),
    [
        ({}, "run_events", RuntimeError),
        ({}, "run_events", KeyboardInterrupt),
        ({}, "reanalyze_if_reachable", RuntimeError),
        ({}, "cache_candidate_paths", KeyboardInterrupt),  # an Exception only warns
        ({"no_events": True}, "cache_candidate_paths", KeyboardInterrupt),
        ({}, "analyze", KeyboardInterrupt),  # held until analyze's commit, then raised
    ],
)
def test_360_analyze_failure_after_highlights_removes_stale_timings(
    monkeypatch, tmp_path: Path, over: dict, step: str, exc: type[BaseException]
) -> None:
    """A step failing after analyze() rewrote highlights.json (Ctrl+C included) removes the
    earlier run's pano_timings.json, which describes another analysis, and re-raises."""
    _timing_run(monkeypatch, tmp_path)

    def fail(*a, **k):
        if step == "analyze":
            _rewrite_highlights(*a, **k)
        raise exc("boom")

    owners = {
        "cache_candidate_paths": C,
        "reanalyze_if_reachable": analyze_mod,
        "analyze": analyze_mod,
    }
    monkeypatch.setattr(owners.get(step, pipeline), step, fail)
    with pytest.raises(exc):
        _run_analyze(**over)
    assert not (tmp_path / "pano_timings.json").exists()


def test_360_analyze_failure_after_highlights_without_timings_file(
    monkeypatch, tmp_path: Path
) -> None:
    """No pano_timings.json to remove: the step's own error comes through."""
    _timing_run(monkeypatch, tmp_path)
    (tmp_path / "pano_timings.json").unlink()

    def fail(*a, **k):
        raise OSError("boom")

    monkeypatch.setattr(pipeline, "run_events", fail)
    with pytest.raises(OSError, match="boom"):
        _run_analyze()
    assert not (tmp_path / "pano_timings.json").exists()


@pytest.mark.parametrize(
    ("over", "target", "exc"),
    [
        ({}, (analyze_mod, "analyze"), RuntimeError),
        ({}, (analyze_mod, "analyze"), KeyboardInterrupt),
        ({}, (VW, "run_layer"), RuntimeError),
        ({"views_only": True}, (VW, "run_layer"), KeyboardInterrupt),
    ],
)
def test_360_analyze_failure_before_highlights_keeps_timings(
    monkeypatch, tmp_path: Path, over: dict, target: tuple, exc: type[BaseException]
) -> None:
    """A failure before highlights.json was rewritten leaves the earlier run's file alone:
    it still matches the analysis in the project."""
    _timing_run(monkeypatch, tmp_path)
    before = (tmp_path / "pano_timings.json").read_bytes()

    def fail(*a, **k):
        raise exc("boom")

    monkeypatch.setattr(*target, fail)
    with pytest.raises(exc):
        _run_analyze(**over)
    assert (tmp_path / "pano_timings.json").read_bytes() == before


def test_360_analyze_locked_stale_timings_keep_the_runs_own_error(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    """A pano_timings.json that cannot be removed (held open on Windows) only warns: the
    run's own error (here Ctrl+C) still comes through, not the PermissionError."""
    from cutpoint import ffmpeg_utils  # noqa: PLC0415

    _timing_run(monkeypatch, tmp_path)
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", lambda s: None)  # retry_locked's backoff
    calls: list[Path] = []

    def locked(self, missing_ok=False):
        calls.append(self)
        raise PermissionError(13, "in use", str(self))

    def interrupt(*a, **k):
        monkeypatch.setattr(Path, "unlink", locked)
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "run_events", interrupt)
    with caplog.at_level(logging.WARNING, logger="cutpoint"), pytest.raises(KeyboardInterrupt):
        _run_analyze()
    monkeypatch.undo()
    assert calls and set(calls) == {tmp_path / "pano_timings.json"}
    assert "not removed" in caplog.text
    timings = json.loads((tmp_path / "pano_timings.json").read_text(encoding="utf-8"))
    assert timings == STALE_TIMINGS


@pytest.mark.parametrize("locked", [False, True], ids=["removed", "kept"])
def test_360_analyze_failure_json_does_not_name_stale_timings(
    monkeypatch, tmp_path: Path, locked: bool
) -> None:
    """#11: once a failure dropped the earlier run's pano_timings.json, or could not (held
    open), the failure envelope's outputs.timings is null; a kept file is named by a
    stale_timings_kept warning."""
    from cutpoint import ffmpeg_utils  # noqa: PLC0415

    _timing_run(monkeypatch, tmp_path)
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", lambda s: None)  # retry_locked's backoff

    def locked_unlink(self, missing_ok=False):
        raise PermissionError(13, "in use", str(self))

    def fail(*a, **k):
        if locked:
            monkeypatch.setattr(Path, "unlink", locked_unlink)
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "run_events", fail)
    with pytest.raises(RuntimeError), jsonout.run("360 analyze") as res:
        _run_analyze()
    monkeypatch.undo()
    assert "timings" in res.outputs and res.outputs["timings"] is None
    assert res.outputs["views"] == {"cached": None, "computed": 1}  # earlier outputs stay
    timings = tmp_path / "pano_timings.json"
    assert timings.exists() is locked
    warned = [(w["code"], w["files"]) for w in res.warnings]
    assert warned == ([(cli.STALE_TIMINGS_KEPT, [str(timings)])] if locked else [])


@pytest.mark.parametrize("scan", [None, ["cam"]], ids=["no_scan", "scan"])
def test_360_analyze_views_only_keeps_the_analysis_timings(
    monkeypatch, tmp_path: Path, scan: list[str] | None
) -> None:
    """--views-only leaves highlights.json as it was, so the earlier run's timings still
    describe it: their keys stay (total_s and scan_s included), views is this run's and
    views_only_s times this run, its scan included."""
    _timing_run(monkeypatch, tmp_path)
    earlier = {"views": {"old": 1}, "analyze_s": 90.0, "events_s": 30.0, "paths_s": 1.5}
    earlier |= {"scan_s": 12.0, "total_s": 140.0}
    cache.save_json(tmp_path / "pano_timings.json", earlier)
    highlights = (tmp_path / "highlights.json").read_bytes()
    _run_analyze(views_only=True, scan=scan)
    timings = json.loads((tmp_path / "pano_timings.json").read_text(encoding="utf-8"))
    assert set(timings) == set(earlier) | {"views_only_s"}
    assert {k: timings[k] for k in earlier if k != "views"} == {
        k: v for k, v in earlier.items() if k != "views"
    }
    assert set(timings["views"]) == {"computed", "total_s"}
    assert timings["views_only_s"] > timings["views"]["total_s"]
    assert (tmp_path / "highlights.json").read_bytes() == highlights


@pytest.mark.parametrize(
    "earlier",
    [{"scan_s": 5.0}, {"views": {"old": 1}, "total_s": 40.0}, ["not", "a", "dict"]],
    ids=["scan_only", "views_only", "not_a_dict"],
)
def test_360_analyze_views_only_without_an_earlier_analysis(
    monkeypatch, tmp_path: Path, earlier: object
) -> None:
    """An earlier file that times no analysis (a standalone 360 scan's, a views-only run's)
    is not kept: --views-only writes its own keys, as with no earlier file."""
    _timing_run(monkeypatch, tmp_path)
    cache.save_json(tmp_path / "pano_timings.json", earlier)
    _run_analyze(views_only=True)
    timings = json.loads((tmp_path / "pano_timings.json").read_text(encoding="utf-8"))
    assert set(timings) == {"views", "total_s"}
    assert set(timings["views"]) == {"computed", "total_s"}


def test_360_analyze_views_only_keeps_an_older_files_analysis(monkeypatch, tmp_path: Path) -> None:
    """A file from before total_s times its analysis with analyze_total_s: kept too."""
    _timing_run(monkeypatch, tmp_path)
    earlier = {"views": {"old": 1}, "scan_s": 12.5, "analyze_s": 90.0, "analyze_total_s": 300.0}
    cache.save_json(tmp_path / "pano_timings.json", earlier)
    _run_analyze(views_only=True)
    timings = json.loads((tmp_path / "pano_timings.json").read_text(encoding="utf-8"))
    assert set(timings) == set(earlier) | {"views_only_s"}
    assert timings["analyze_total_s"] == 300.0


def test_standalone_360_scan_merges_scan_s(monkeypatch, tmp_path: Path) -> None:
    from cutpoint.pano import x3  # noqa: PLC0415

    monkeypatch.setattr(x3, "scan", lambda folder: [helpers.x3clip()])
    monkeypatch.setattr(x3, "imu_source", lambda source: None)
    cache.save_json(tmp_path / "pano_timings.json", {"views": {"computed": 1}, "total_s": 9.0})
    cli._scan(tmp_path, [str(tmp_path)])
    timings = json.loads((tmp_path / "pano_timings.json").read_text(encoding="utf-8"))
    assert set(timings) == {"views", "total_s", "scan_s"}
    got: dict = {}
    cli._scan(tmp_path, [str(tmp_path)], timings=got)  # 360 analyze --scan: the run's dict
    assert set(got) == {"scan_s"}


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


def test_steady_rotation_is_spin_not_whip() -> None:
    t = np.arange(81) * 0.5  # 40 s
    path = keys_path(20.0 * t)  # 20 deg/s, the whole time
    d = C.diagnostics(path, 2.0, 32.0)  # 30 s of it: 600 deg
    assert d["peak_yaw_rate"] < C.WHIP_DEG_S
    assert d["net_turns"] == pytest.approx(600 / 360, abs=0.02)
    assert d["yaw_travel_per_min"] == pytest.approx(1200.0, rel=0.01)
    assert d["flags"] == ["spin"] and C.FLAGS["spin"] == "keeps turning"
    # one full turn in a window is enough, whichever way
    d = C.diagnostics(keys_path(-20.0 * t), 2.0, 22.0)  # 20 s, -400 deg
    assert d["net_turns"] == pytest.approx(400 / 360, abs=0.02) and d["flags"] == ["spin"]


def test_back_and_forth_pans_are_no_spin() -> None:
    t = np.arange(81) * 0.5
    d = C.diagnostics(keys_path(40.0 * np.sin(2 * np.pi * t / 20.0)), 0.0, 40.0)  # +-40 deg
    assert d["net_turns"] == pytest.approx(80 / 360, abs=0.02)
    assert d["yaw_travel_per_min"] < C.SPIN_TRAVEL_DEG_MIN and d["flags"] == []


def test_spin_travel_needs_a_full_turn_and_ignores_cuts() -> None:
    # one quick deliberate pan: over the rate, but under a full turn of travel
    yaw = np.r_[np.zeros(4), np.arange(1, 13) * 7.5, np.full(25, 90.0)]  # 90 deg in 6 s
    d = C.diagnostics(keys_path(yaw), 1.0, 9.0)
    assert d["yaw_travel_per_min"] > C.SPIN_TRAVEL_DEG_MIN and d["flags"] == []
    # cuts add no rotation: a path cutting 90 deg every 2 s does not spin
    cuts = tuple(float(c) for c in range(2, 20, 2))
    yaw = np.array([90.0 * sum(t >= c for c in cuts) for t in np.arange(41) * 0.5])
    d = C.diagnostics(keys_path(yaw, cuts=cuts), 0.0, 20.0)
    assert d["net_turns"] == 0.0 and d["yaw_travel"] == 0.0 and "spin" not in d["flags"]
    # an older record without the spin measures is not flagged
    assert C.path_flags({"peak_yaw_rate": 10.0}) == []


def test_gentle_sweeps_travel_far_but_do_not_spin() -> None:
    t = np.arange(121) * 0.5  # 60 s
    d = C.diagnostics(keys_path(60.0 * np.sin(2 * np.pi * t / 20.0)), 0.0, 60.0)  # +-60 deg
    assert d["yaw_travel_per_min"] > C.SPIN_TRAVEL_DEG_MIN
    assert d["yaw_travel"] >= C.SPIN_MIN_TRAVEL_DEG
    assert d["net_turns"] < C.SPIN_TRAVEL_MIN_TURNS and d["flags"] == []
    # a steady 10 deg/s: under a full turn per 30 s, but it keeps going one way
    d = C.diagnostics(keys_path(10.0 * t), 0.0, 60.0)
    assert C.SPIN_TRAVEL_MIN_TURNS <= d["net_turns"] < C.SPIN_TURNS
    assert d["yaw_travel_per_min"] > C.SPIN_TRAVEL_DEG_MIN and d["flags"] == ["spin"]


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
    assert set(C.FLAGS) == {"sky", "whip", "spin", "cuts", "wearer", "off_target", "qa_fail"}


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


def test_version_1_entries_are_planned_again(project) -> None:
    proj, clip = project
    C.cache_candidate_paths(proj)
    file = proj / C.CACHE_DIR / C.slot_name(clip, 4.0, 12.5)
    old = cache.load_json(file)
    old["version"] = 1
    for k in ("yaw_travel", "yaw_travel_per_min", "net_turns"):
        old["diagnostics"].pop(k)
    cache.save_json(file, old, indent=1)
    assert C.cached_path(proj, clip, 4.0, 12.5, "people", None) == (None, "older cache")
    assert set(C.candidate_flags(proj)) == {"h002"}
    st = C.cache_candidate_paths(proj)
    assert (st["computed"], st["cached"]) == (1, 1)
    new = cache.load_json(file)
    assert new["version"] == C.CACHE_VERSION == 2 and "net_turns" in new["diagnostics"]
    assert set(C.candidate_flags(proj)) == {"h001", "h002"}


def test_candidate_flags_has_reasons_diagnostics_and_a_path_key(project) -> None:
    proj, _ = project
    C.cache_candidate_paths(proj)
    got = C.candidate_flags(proj)["h001"]
    assert set(got) == {"file", "flags", "reasons", "diagnostics", "key"}
    entry = cache.load_json(got["file"])
    assert got["reasons"] == {f: C.FLAGS[f] for f in got["flags"]}
    assert got["diagnostics"] == entry["diagnostics"] | {"flags": got["flags"]}
    assert len(got["key"]) == 12 and got["key"] == C.path_id(entry["path"])
    assert got["key"] != C.candidate_flags(proj)["h002"]["key"]
    assert C.candidate_flags(proj)["h001"]["key"] == got["key"]  # stable
    # the cached path changes (re-planned with other settings): so does the key
    entry["path"]["keys"][0]["yaw"] += 1.0
    cache.save_json(got["file"], entry, indent=1)
    assert C.candidate_flags(proj)["h001"]["key"] != got["key"]


# --------------------------------------------------------------------------- preview API
def test_preview_is_the_cached_path_for_an_analyzed_span(project, monkeypatch) -> None:
    proj, clip = project
    C.cache_candidate_paths(proj)
    monkeypatch.setattr(D, "compute_path", lambda *a, **k: pytest.fail("re-planned"))
    got = C.preview_path(proj, clip.source, 4.0, 12.5)
    assert got["kind"] == "cached" and got["reason"] is None
    entry = cache.load_json(C.candidate_flags(proj)["h001"]["file"])
    assert _dump(got["path"]) == _dump(entry["path"])


def test_preview_plans_a_trimmed_or_vertical_span_as_clips_would(project) -> None:
    proj, clip = project
    C.cache_candidate_paths(proj)
    before = sorted((proj / C.CACHE_DIR).iterdir())
    got = C.preview_path(proj, clip.source, 5.0, 11.0)  # h001 trimmed: its kind applies
    assert got["kind"] == "planned" and got["reason"] == "span differs from 360 analyze"
    assert _dump(got["path"]) == _dump(D.compute_path(clip, 5.0, 11.0, kind="people"))
    got = C.preview_path(proj, clip.source, 4.0, 12.5, vertical=True)
    assert got["kind"] == "planned" and got["reason"] == "9:16 not planned by 360 analyze"
    assert got["path"]["vertical"] is True
    assert sorted((proj / C.CACHE_DIR).iterdir()) == before  # nothing written
    shutil.rmtree(proj / C.CACHE_DIR)  # no analysis cache at all
    got = C.preview_path(proj, clip.source, 4.0, 12.5)
    assert got["kind"] == "planned" and got["reason"] == "not planned by 360 analyze"
    assert not (proj / C.CACHE_DIR).exists()


def _world_pitch(clip, path, orient) -> float:
    """World pitch (deg, + up) of the output centre of a render's first frame."""
    src = P.RenderSource("dfisheye", ("a.mp4",), "dfisheye", False, True, "test")
    _, init = R.command_lines(clip, path, src, 4.0, 1.0, R.DEFAULT_FORMAT, 1, orient)
    r = G.rot_r(init["roll"]) @ G.rot_y(init["yaw"]) @ G.rot_p(init["pitch"])
    cam = r @ np.array([0.0, 0.0, 1.0])  # output centre ray in camera axes
    world = orient.cam_to_world(np.array([orient.index(4.0)]))[0] @ cam
    return float(np.degrees(np.arcsin(-world[1])))  # world -y is up


def test_preview_falls_back_to_a_levelled_forward_view(project, monkeypatch) -> None:
    proj, clip = project
    cam = Path(clip.source).parent
    (cam / helpers.INSV[0]).write_bytes(b"x")  # the clip's IMU file

    def offline(*a, **k):
        raise FileNotFoundError(2, "No such file or directory")

    # footage offline: levelled by the IMU once it is back, never the upright guess
    cam.rename(cam.with_name("cam-offline"))
    with monkeypatch.context() as m:
        m.setattr(G, "orientation", offline)
        got = C.preview_path(proj, clip.source, 4.0, 12.5)
    assert got["kind"] == "level_forward" and got["path"]["level"] == "imu"
    assert got["reason"] == "footage offline (No such file or directory)"
    assert got["path"]["forward_yaw"] == 0.0  # no view layer reachable
    cam.with_name("cam-offline").rename(cam)

    # back online, but the director fails: forward view at the travel direction
    monkeypatch.setattr(D, "compute_path", lambda *a, **k: (_ for _ in ()).throw(KeyError("x")))
    monkeypatch.setattr(VW, "forward_yaw", lambda pv: 30.0)
    got = C.preview_path(proj, clip.source, 4.0, 12.5)
    assert got["kind"] == "level_forward"
    assert got["reason"] == "director could not plan (KeyError: 'x')"
    path = got["path"]
    p = D.params_for("people")
    assert path["level"] == "imu" and path["forward_yaw"] == 30.0 and path["cuts"] == []
    assert [k["t"] for k in path["keys"]] == [0.0, 22.5]  # with the director's margin
    assert {(k["yaw"], k["pitch"], k["fov"]) for k in path["keys"]} == {
        (30.0, p.pitch_default, p.fov_wide)
    }
    yaw, pitch, _ = D.evaluate(path, np.array([4.0, 8.0, 12.5]))
    assert np.allclose(yaw, 30.0) and np.allclose(pitch, p.pitch_default)
    assert C.diagnostics(path, 4.0, 12.5)["flags"] == []
    got = C.preview_path(proj, clip.source, 4.0, 12.5, vertical=True)
    assert {k["fov"] for k in got["path"]["keys"]} == {p.vertical_fov_wide}
    (cam / helpers.INSV[0]).unlink()

    # no IMU at all: levelled for the upright mount, never the raw fisheye
    def no_imu(*a, **k):
        raise ValueError("no IMU samples")

    monkeypatch.setattr(G, "orientation", no_imu)
    monkeypatch.setattr(D, "compute_path", lambda *a, **k: no_imu())
    got = C.preview_path(proj, clip.source, 4.0, 12.5)
    assert got["kind"] == "level_forward" and got["path"]["level"] == R.UPRIGHT
    assert got["reason"].startswith("no IMU (no _00_.insv next to the clip)")
    assert "levelled for the upright mount" in got["reason"]
    up = R.upright_orientation(clip.fps, 900)  # what render_segment levels it with
    pitch = _world_pitch(clip, got["path"], up)
    assert pitch == pytest.approx(D.params_for("people").pitch_default, abs=0.01)


def test_bad_director_settings_fall_back_and_bad_arguments_raise(project, home) -> None:
    proj, clip = project
    set_config(home, pano={"director": {"no_such_param": 1}})
    got = C.preview_path(proj, clip.source, 4.0, 12.5)
    assert got["kind"] == "level_forward" and got["reason"].startswith("settings: ")
    with pytest.raises(ValueError, match="not a 360 clip"):
        C.preview_path(proj, proj / "other.mp4", 1.0, 2.0)
    with pytest.raises(ValueError, match="empty"):
        C.preview_path(proj, clip.source, 5.0, 5.0)


def test_preview_path_is_consistent_across_threads(project) -> None:
    proj, clip = project
    C.cache_candidate_paths(proj)
    spans = [(4.0, 12.5), (5.0, 11.0), (14.0, 20.0), (4.0, 12.5)]
    want = [_dump(C.preview_path(proj, clip.source, a, b)) for a, b in spans]
    kinds = [json.loads(w)["kind"] for w in want]
    assert kinds == ["cached", "planned", "cached", "cached"]
    for _ in range(3):
        start = threading.Barrier(len(spans), timeout=10)

        def call(span, start=start):
            start.wait()  # all four at once
            return _dump(C.preview_path(proj, clip.source, *span))

        with ThreadPoolExecutor(len(spans)) as pool:
            got = list(pool.map(call, spans))
        assert got == want
