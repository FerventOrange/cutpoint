"""The commands use the caption set of the project's ``activity.json`` (review F3/F4): the
crash, mix, events, 360 director, QA and vision-check paths see it, an old record is upgraded
as documented, and the 360 plan keys change only for another profile.

No models or footage: every consumer of the prompts is a recording stub.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cutpoint import activity, crash, ffmpeg_utils, mix
from cutpoint.events import pipeline
from cutpoint.pano import candidates as C
from cutpoint.pano import cli, qa, views, vlcheck
from cutpoint.pano import director as D
from cutpoint.pano import project as P


def _write(project: Path, name: str) -> None:
    rec = {
        "version": activity.PROFILE_VERSION,
        "name": name,
        "source": "library",
        "profile": activity.full(activity.builtin(name)),
    }
    project.mkdir(parents=True, exist_ok=True)
    (project / activity.ACTIVITY_FILE).write_text(json.dumps(rec), encoding="utf-8")


def _mtb() -> activity.Prompts:
    return activity.prompts(activity.full(activity.builtin("mtb")))


def _mtb_key() -> str:
    with activity.use_prompts(_mtb()):
        key = C.prompts_key()
    assert key
    return key


class _Stop(Exception):
    """Raised by a stub to end a run once the prompts were recorded."""


# ----------------------------------------------------------------- crash.detect_project
def _crash_project(tmp_path: Path) -> Path:
    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    project = tmp_path / "proj"
    project.mkdir()
    hl = {"sources": [{"path": str(src), "duration": 60.0}]}
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    return project


def _stub_load_inputs(monkeypatch) -> list[tuple[str, bool]]:
    """load_inputs finds the lift rides as the real one does (``mix.lift_rides`` ->
    ``views.lift_grid``); returns what it saw: (prompt set name, has lift captions)."""
    seen: list[tuple[str, bool]] = []
    monkeypatch.setattr(views, "_grid_views", lambda src: None)  # lift_grid: no 360 cache

    def load_inputs(src, dur, p, tmp, lang):
        seen.append((activity.active_prompts().name, bool(activity.active_prompts().lift)))
        assert mix.lift_rides(src, 10) is None
        return SimpleNamespace(imu=object(), visual=object(), path=src)

    monkeypatch.setattr(crash, "load_inputs", load_inputs)
    monkeypatch.setattr(crash, "candidate_events", lambda inp, p: [])
    return seen


@pytest.mark.parametrize(("name", "lift"), [("mtb", False), ("ski", True)])
def test_detect_project_loads_inputs_with_the_projects_prompts(tmp_path, monkeypatch, name, lift):
    project = _crash_project(tmp_path)
    _write(project, name)
    seen = _stub_load_inputs(monkeypatch)
    own_words = replace(crash.CrashParams(), pos=("a",), neg=("b",), words=())  # no profile
    crash.detect_project(project, params=own_words, dense=False, sheets=0)
    assert seen == [(name, lift)]
    assert activity.active_prompts() == activity.default_prompts()  # restored afterwards


def test_detect_project_uses_the_profile_it_resolved_for_this_run(tmp_path, monkeypatch):
    project = _crash_project(tmp_path)  # no activity.json yet: --activity mtb writes it
    seen = _stub_load_inputs(monkeypatch)
    crash.detect_project(project, activity_name="mtb", llm=False, dense=False, sheets=0)
    assert seen == [("mtb", False)]


def test_lift_grid_is_none_for_a_profile_without_lift_captions(monkeypatch):
    called = []
    monkeypatch.setattr(views, "_grid_views", lambda src: called.append(src))
    with activity.use_prompts(_mtb()):
        assert views.lift_grid(Path("x.lrv"), 10) is None
    assert called == []  # not even the view layer is read
    views.lift_grid(Path("x.lrv"), 10)  # ski (default) looks for the lifts
    assert called == [Path("x.lrv")]


def test_the_snowboard_profile_has_lift_captions():
    p = activity.prompts(activity.full(activity.builtin("snowboard")))
    assert p.lift and p.lift_pos and p.lift_neg


# ----------------------------------------------------------------- mix.run_mix
@pytest.mark.parametrize(("name", "want"), [("mtb", "mtb"), (None, "ski")])
def test_run_mix_finds_candidates_with_the_projects_prompts(tmp_path, monkeypatch, name, want):
    if name:
        _write(tmp_path, name)
    seen: list[str] = []

    def part_candidates(_d):
        seen.append(activity.active_prompts().name)
        return [mix.Cand("run", "a", 47.0, 53.0, 50.0, 0.9, "run", "run")]

    monkeypatch.setattr(mix.ski, "load_project_features", lambda _p: [{}])
    monkeypatch.setattr(mix, "part_candidates", part_candidates)
    monkeypatch.setattr(mix, "_runs_index", lambda _p, _k: ([], {}))
    monkeypatch.setattr(mix, "_embed", lambda _c: None)
    mix.run_mix(tmp_path, target_s=60, sheet=False, exclude=[], out=tmp_path / "mix_segments.json")
    assert seen == [want]


# ----------------------------------------------------------------- events.pipeline.run_events
class _Mon:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def summary(self):
        return {}


@pytest.mark.parametrize(("name", "want"), [("mtb", "mtb"), (None, "ski")])
def test_run_events_visual_layer_and_sheets_use_the_projects_prompts(
    home, tmp_path, monkeypatch, name, want
):
    from cutpoint.events import sheets  # noqa: PLC0415
    from cutpoint.events import visual as visual_layer  # noqa: PLC0415

    if name:
        _write(tmp_path, name)
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    s = SimpleNamespace(path=src, name="a.mp4")
    hl = {"sources": [{"path": str(src)}]}
    monkeypatch.setattr(pipeline, "load_highlights", lambda project: hl)
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "load_source", lambda path, dur: s)
    monkeypatch.setattr(pipeline, "bad_ranges", lambda *a: {})
    monkeypatch.setattr(pipeline, "load_transcript_lines", lambda project: [])
    monkeypatch.setattr(
        pipeline, "VisionRater", lambda *a, **k: SimpleNamespace(provider_name="ollama", model="m")
    )
    monkeypatch.setattr(pipeline, "VramMonitor", _Mon)
    monkeypatch.setattr(pipeline, "detect_audio_events", lambda *a, **k: [])
    seen: list[tuple[str, str]] = []

    def run_layer(*a, **k):
        seen.append(("visual", activity.active_prompts().name))
        return {}

    def write_sheets(project, sources):
        seen.append(("sheets", activity.active_prompts().name))
        raise _Stop  # the rest of the run is not needed

    monkeypatch.setattr(visual_layer, "run_layer", run_layer)
    monkeypatch.setattr(sheets, "write_visual_sheets", write_sheets)
    with pytest.raises(_Stop):
        pipeline.run_events(tmp_path, layers=("visual",), sheets=True, cached_ocr=False)
    assert seen == [("visual", want), ("sheets", want)]


# ----------------------------------------------------------------- activity.record_profile
@pytest.mark.parametrize("source", ["context", "synthesized"])
def test_an_old_llm_record_keeps_incident_and_identity_and_takes_the_rest_from_the_library(
    source,
):
    mtb = activity.full(activity.builtin("mtb"))
    old = {
        "name": "mtb",
        "description": "my words",
        "detect": ["x"],
        "incident": {"pos": ["a rider crashing hard"], "neg": ["a calm trail"]},
        "highlight": {"pos": ["an old ski-era highlight"], "neg": []},
        "low_info": ["an old low-info caption"],
    }
    got = activity.record_profile({"name": "mtb", "source": source, "profile": old})
    assert got["incident"] == old["incident"] and got["description"] == "my words"
    assert got["detect"] == ["x"]
    assert got["highlight"] == mtb["highlight"] and got["low_info"] == mtb["low_info"]
    assert all(k in got for k in (*activity.PAIR_SECTIONS, *activity.LIST_SECTIONS, "vl"))


def test_an_old_record_of_an_unknown_name_takes_the_floor():
    old = {"name": "unicycle", "incident": {"pos": ["a fall"], "neg": ["a ride"]}}
    got = activity.record_profile({"name": "unicycle", "source": "synthesized", "profile": old})
    floor = activity.base()
    assert got["incident"] == old["incident"] and got["name"] == "unicycle"
    assert got["highlight"] == floor["highlight"] and got["low_info"] == floor["low_info"]


# ----------------------------------------------------------------- 360 plan keys
SEG = ("001", 1.0, 5.0, "people", False, {"a": 1})


def test_plan_key_is_unchanged_for_the_default_prompts_and_differs_for_another():
    old = json.dumps([["001", 1.0, 5.0, "people", False], {"a": 1}], sort_keys=True)
    assert C.prompts_key() is None
    assert cli._plan_key(*SEG, None, C.prompts_key()) == old == cli._plan_key(*SEG)
    key = _mtb_key()
    assert cli._plan_key(*SEG, None, key) != old


def test_reusable_does_not_reuse_a_path_saved_with_other_prompts(tmp_path):
    base = {
        "version": D.PATH_VERSION,
        "clip": "001",
        "segment": {"start": 1.0, "end": 5.0},
        "kind": "people",
        "vertical": False,
        "params": {"a": 1},
    }
    (tmp_path / "001_a.json").write_text(json.dumps(base), encoding="utf-8")
    key = _mtb_key()
    other = base | {"clip": "002", "prompts": key}
    (tmp_path / "002_b.json").write_text(json.dumps(other), encoding="utf-8")
    found = cli._reusable(tmp_path)
    assert cli._plan_key(*SEG) in found  # saved with the default prompts
    assert cli._plan_key("002", *SEG[1:], None, key) in found
    assert cli._plan_key("002", *SEG[1:]) not in found  # now ski: re-plan


# ----------------------------------------------------------------- 360 qa
def test_project_of_is_the_project_of_a_clips_or_a_director_folder(tmp_path):
    assert qa._project_of(tmp_path / "clips") == tmp_path
    assert qa._project_of(tmp_path / "pano_paths" / "mix") == tmp_path


@pytest.mark.parametrize(("name", "want"), [("mtb", "mtb"), (None, "ski")])
@pytest.mark.parametrize("sub", ["clips", "pano_paths/director"])
def test_run_qa_scores_with_the_projects_prompts(tmp_path, monkeypatch, name, want, sub):
    from cutpoint.events import visual as V  # noqa: PLC0415

    if name:
        _write(tmp_path, name)
    d = tmp_path / sub
    d.mkdir(parents=True)
    seen: list[str] = []

    def qa_clip(item, models, fps, size, th):
        seen.append(activity.active_prompts().name)
        raise _Stop

    monkeypatch.setattr(qa, "find_inputs", lambda p: [{"file": "001_a.mp4"}])
    monkeypatch.setattr(qa, "qa_clip", qa_clip)
    monkeypatch.setattr(V, "VisualModels", lambda: SimpleNamespace(close=lambda: None))
    with pytest.raises(_Stop):
        qa.run_qa(d, json_only=True)
    assert seen == [want]


# ----------------------------------------------------------------- 360 director + vision check
def _director(tmp_path: Path, monkeypatch) -> tuple[Path, dict[str, Any]]:
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    seen: dict[str, Any] = {"compute": [], "wording": []}
    clip = SimpleNamespace(clip_id="001", lrv="x.lrv", source="x.lrv")
    seg = {"source": "x.lrv", "start": 1.0, "end": 4.0, "label": "a"}
    monkeypatch.setattr(cli, "_project", lambda arg: proj)
    monkeypatch.setattr(P, "load", lambda project: {"x.lrv": clip})
    monkeypatch.setattr(P, "clip_for", lambda clips, src: clip)
    monkeypatch.setattr(cli, "_selected_segments", lambda project, a: [seg])

    def compute(clip, start, end, **kw):
        seen["compute"].append(activity.active_prompts().name)
        flags = dict.fromkeys(
            ("occluded", "too_close", "edge_cut", "subject_present", "size_ok", "safe_area"), 0.0
        )
        quality = {**flags, "score": 100.0, "pass": True, "fails": []}
        return {"keys": [], "cuts": [], "quality": quality}

    def available(self):
        seen["wording"].append(self.wording)
        return False

    monkeypatch.setattr(D, "compute_path", compute)
    monkeypatch.setattr(vlcheck.VLChecker, "available", available)
    a = argparse.Namespace(
        project="proj", vertical=False, preview=False, param=[], out_name=None, force=True,
        vl_check=True,
    )  # fmt: skip
    cli.cmd_director(a)
    return proj / "pano_paths" / "director", seen


def test_cmd_director_plans_and_checks_with_the_projects_prompts(tmp_path, monkeypatch):
    _write(tmp_path / "proj", "mtb")
    out, seen = _director(tmp_path, monkeypatch)
    assert seen["compute"] == ["mtb"]
    assert seen["wording"] == [vlcheck.wording_id(_mtb())] and seen["wording"][0]
    saved = json.loads((out / "001_a.json").read_text(encoding="utf-8"))
    assert saved["prompts"] == _mtb_key()  # read back by _reusable


def test_cmd_director_with_the_default_prompts_saves_the_paths_as_before(tmp_path, monkeypatch):
    out, seen = _director(tmp_path, monkeypatch)
    assert seen["compute"] == ["ski"] and seen["wording"] == [""]
    assert "prompts" not in json.loads((out / "001_a.json").read_text(encoding="utf-8"))


def test_check_preview_does_not_reuse_replies_given_to_another_wording(tmp_path, monkeypatch):
    monkeypatch.setattr(ffmpeg_utils, "grab_images", lambda video, t, **kw: [b"jpeg"])
    keys = [{"t": t / 2, "yaw": 30.0, "pitch": -8.0, "fov": 86.0} for t in range(25)]
    path = {"clip": "c1", "segment": {"start": 1.0, "end": 10.0}, "keys": keys, "vl_suspect": []}

    class Fake:
        model = "fake-vl"

        def __init__(self, wording: str) -> None:
            self.wording, self.calls = wording, 0

        def ask(self, jpeg):
            self.calls += 1
            return {"blocked": False, "where": "none", "what": "", "low_info": False}

    def check(checker: Fake) -> None:
        vlcheck._MEMO.clear()  # read from the disk cache
        vlcheck.check_preview(checker, tmp_path / "p.mp4", path)  # type: ignore[arg-type]

    ski = Fake("")
    check(ski)
    again = Fake("")
    check(again)
    assert ski.calls > 0 and again.calls == 0  # same wording: all from the cache
    other = Fake(vlcheck.wording_id(_mtb()))
    check(other)
    assert other.calls == ski.calls  # another wording: asked again
    vlcheck._MEMO.clear()
