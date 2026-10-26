"""``--json`` on capabilities, mix, stage, setup and the ``360`` subcommands (#4, phase 3).

Everything heavy is replaced: no download, no model, no GPU work and never the Insta360 SDK.
"""

from __future__ import annotations

import json
import sys
from collections import namedtuple
from pathlib import Path

import pytest

from conftest import envelope, run_json, warning_codes, write
from cutpoint import cache, capabilities, cli, mix, stage
from cutpoint.events import paths as event_paths
from cutpoint.events import setup
from cutpoint.mix import Cand
from cutpoint.pano import check as pano_check
from cutpoint.pano import director as D
from cutpoint.pano import project as P
from cutpoint.pano import qa as pano_qa
from cutpoint.pano import render as pano_render
from cutpoint.pano import stitch as pano_stitch
from cutpoint.pano import views as VW
from cutpoint.pano import vlcheck, x3
from pano_helpers import x3clip


# ---------------------------------------------------------------------------- capabilities
def test_capabilities_json_keeps_the_report_keys_and_warns(monkeypatch, capsys):
    data = {
        "platform": "linux",
        "torch": {"warning": "torch is a CPU-only build", "fix": "reinstall"},
        "insta360_sdk": {"found": False, "warning": capabilities.SDK_NEEDS_NVIDIA},
    }
    monkeypatch.setattr(capabilities, "report", lambda: data)
    code, doc = run_json(capsys, ["capabilities"])
    assert code == 0 and doc["ok"] is True and doc["command"] == "capabilities"
    assert doc["outputs"] == data and doc["project"] is None
    assert warning_codes(doc) == ["torch_no_cuda", "sdk_needs_nvidia"]
    assert "reinstall" in doc["warnings"][0]["message"]


def test_capabilities_json_error(monkeypatch, capsys):
    def boom():
        raise RuntimeError("probe failed")

    monkeypatch.setattr(capabilities, "report", boom)
    code, doc = run_json(capsys, ["capabilities"])
    assert code == 1 and doc["ok"] is False
    assert doc["error"] == {"code": "error", "message": "probe failed"}


def test_capabilities_text_unchanged_without_json(monkeypatch, capsys):
    monkeypatch.setattr(capabilities, "report", lambda: {"x": 1})
    monkeypatch.setattr(capabilities, "format_report", lambda d: "the report")
    assert cli.main(["capabilities"]) == 0
    assert capsys.readouterr().out == "the report\n"


# ------------------------------------------------------------------------------------- mix
@pytest.fixture
def mix_project(home, tmp_path, monkeypatch) -> Path:
    """A one-source ski project (candidates stubbed, as in test_pano_repair)."""
    project = home / "projects" / "ski"
    project.mkdir(parents=True)
    src = str(tmp_path / "a.mp4")

    def cands(_d: object) -> list[Cand]:
        return [
            Cand("lift", src, 100, 105, 102, 1.0, "beauty", "lift"),
            Cand("social", src, 300, 309, 304, 0.9, "laughter", "laugh"),
            Cand("run", src, 500, 506, 503, 1.0, "carve", "run"),
        ]

    def sheet(segs, out: Path) -> Path:
        out.write_bytes(b"jpg")
        return out

    monkeypatch.setattr(mix.ski, "load_project_features", lambda _p: [{}])
    monkeypatch.setattr(mix, "part_candidates", cands)
    monkeypatch.setattr(mix, "load_crashes", lambda _p: None)
    monkeypatch.setattr(mix, "_runs_index", lambda _p, _k: ([], {}))
    monkeypatch.setattr(mix, "_embed", lambda _c: None)
    monkeypatch.setattr(mix, "_write_sheet", sheet)
    return project


def test_mix_json(mix_project, capsys):
    code, doc = run_json(capsys, ["mix", "ski", "--target", "25"])
    assert code == 0 and doc["ok"] is True and doc["command"] == "mix"
    assert doc["project"] == str(mix_project)
    out = doc["outputs"]
    assert out["segments"] == str(mix_project / "mix_segments.json")
    assert out["report"] == str(mix_project / "mix_report.json")
    assert out["sheet"] == str(mix_project / "mix_sheet.jpg")
    assert all(Path(out[k]).is_file() for k in ("segments", "report", "sheet"))
    segs = cache.load_json(mix_project / "mix_segments.json")
    assert out["count"] == len(segs) == len(out["moments"]) > 0
    for m, s in zip(out["moments"], segs, strict=True):
        assert (m["id"], m["start"], m["end"]) == (s["id"], s["start"], s["end"])
        assert m["duration"] == pytest.approx(s["end"] - s["start"])
    report = cache.load_json(mix_project / "mix_report.json")
    assert out["total_seconds"] == report["total_s"] and out["target_seconds"] == 25
    assert out["content_qa_pass"] is report["content_qa"]["pass"]
    assert set(warning_codes(doc)) <= {"content_qa_failed"}
    assert ("content_qa_failed" in warning_codes(doc)) is not report["content_qa"]["pass"]


def test_mix_json_sheet_is_none_when_this_run_wrote_none(mix_project, monkeypatch, capsys):
    write(mix_project / "mix_sheet.jpg", "an earlier run's sheet")
    monkeypatch.setattr(mix, "_write_sheet", lambda segs, out: None)  # e.g. no frame decoded
    code, doc = run_json(capsys, ["mix", "ski", "--target", "25"])
    assert code == 0 and doc["outputs"]["sheet"] is None
    assert "sheet" not in cache.load_json(mix_project / "mix_report.json")


@pytest.mark.skipif(sys.platform != "win32", reason="drive-relative paths are Windows only")
def test_mix_json_warns_about_ignored_exclusions(mix_project, capsys):
    cache.save_json(mix_project / "mix_exclude.json", [{"source": "D:a.mp4", "start": 0, "end": 1}])
    code, doc = run_json(capsys, ["mix", "ski", "--target", "25"])
    assert code == 0 and "exclusions_ignored" in warning_codes(doc)
    warning = doc["warnings"][warning_codes(doc).index("exclusions_ignored")]
    assert warning["files"] == [str(mix_project / "mix_exclude.json")]


def test_mix_json_bad_quotas_is_bad_args(mix_project, capsys):
    code, doc = run_json(capsys, ["mix", "ski", "--quotas", "run=x"])
    assert code == 1 and doc["ok"] is False
    assert doc["error"]["code"] == "bad_args" and "--quotas" in doc["error"]["message"]
    assert not (mix_project / "mix_segments.json").exists()


# ----------------------------------------------------------------------------------- stage
@pytest.fixture
def plenty_of_space(monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(stage.shutil, "disk_usage", lambda _p: usage(10**13, 0, 10**13))


def test_stage_json_lists_copied_then_skipped_files(home, tmp_path, plenty_of_space, capsys):
    a = write(tmp_path / "footage" / "a.mp4", "aaaa")
    b = write(tmp_path / "footage" / "b.mp4", "bb")
    stage_dir = home / "stage"
    code, doc = run_json(capsys, ["stage", str(tmp_path / "footage")])
    assert code == 0 and doc["ok"] is True and doc["command"] == "stage"
    out = doc["outputs"]
    assert out["stage_dir"] == str(stage_dir)
    assert [(f["source"], f["bytes"], f["status"]) for f in out["files"]] == [
        (str(a), 4, "copied"),
        (str(b), 2, "copied"),
    ]
    for f in out["files"]:
        assert Path(f["staged"]).parent == stage_dir and Path(f["staged"]).is_file()
    assert (out["copied"], out["skipped"], out["bytes"]) == (2, 0, 6)

    code, doc = run_json(capsys, ["stage", str(tmp_path / "footage")])
    out = doc["outputs"]
    assert code == 0 and [f["status"] for f in out["files"]] == ["skipped", "skipped"]
    assert (out["copied"], out["skipped"], out["bytes"], out["mb_per_s"]) == (0, 2, 0, None)

    code, doc = run_json(capsys, ["stage", "--list"])
    assert code == 0 and sorted(f["source"] for f in doc["outputs"]["files"]) == [str(a), str(b)]
    assert all(Path(f["staged"]).is_file() for f in doc["outputs"]["files"])

    code, doc = run_json(capsys, ["stage", "--clear"])
    assert code == 0 and doc["outputs"]["removed"] == 2


def test_stage_json_clear_warns_about_an_input_with_no_copies(home, tmp_path, capsys):
    gone = tmp_path / "gone"
    code, doc = run_json(capsys, ["stage", "--clear", str(gone)])
    assert code == 0 and doc["outputs"]["removed"] == 0
    assert warning_codes(doc) == ["nothing_staged_under_input"]
    assert doc["warnings"][0]["files"] == [str(gone)]


def test_stage_json_without_inputs_is_bad_args(home, capsys):
    code, doc = run_json(capsys, ["stage"])
    assert code == 1 and doc["error"]["code"] == "bad_args"
    assert doc["outputs"] == {}


def test_stage_json_missing_input(home, tmp_path, plenty_of_space, capsys):
    code, doc = run_json(capsys, ["stage", str(tmp_path / "nope.mp4")])
    assert code == 1 and doc["error"]["code"] == "not_found"


def test_stage_json_relative_input_is_reported_absolute(
    home, tmp_path, plenty_of_space, monkeypatch, capsys
):
    a = write(tmp_path / "footage" / "a.mp4", "aaaa")
    monkeypatch.chdir(tmp_path)
    code, doc = run_json(capsys, ["stage", "footage"])
    assert code == 0 and [f["source"] for f in doc["outputs"]["files"]] == [str(a)]


# ----------------------------------------------------------------------------------- setup
@pytest.fixture
def fake_installs(monkeypatch) -> list[str]:
    """PANNs and Silero "download" small files; nothing touches the network."""
    calls: list[str] = []

    def panns() -> None:
        calls.append("panns")
        write(event_paths.panns_ckpt(), "checkpoint")

    def voice() -> None:
        calls.append("voice")
        write(event_paths.silero_vad(), "onnx")

    monkeypatch.setattr(setup, "install_panns", panns)
    monkeypatch.setattr(setup, "install_voice", voice)
    return calls


def test_setup_json_reports_downloaded_and_found_files(home, fake_installs, capsys):
    labels = write(event_paths.panns_labels(), "labels")  # there before: found
    code, doc = run_json(capsys, ["setup", "--only", "pip", "panns", "voice"])
    assert code == 0 and doc["ok"] is True and doc["command"] == "setup"
    assert fake_installs == ["panns", "voice"]
    comps = doc["outputs"]["components"]
    assert set(comps) == {"pip", "panns", "voice"}
    assert comps["panns"]["files"] == [
        {"path": str(labels), "bytes": 6, "status": "found"},
        {"path": str(event_paths.panns_ckpt()), "bytes": 10, "status": "downloaded"},
    ]
    assert comps["voice"]["files"] == [
        {"path": str(event_paths.silero_vad()), "bytes": 4, "status": "downloaded"}
    ]
    pip = comps["pip"]
    assert pip["installed"] is (not pip["missing"])
    assert ("events_extra_missing" in warning_codes(doc)) is bool(pip["missing"])


def test_setup_json_failed_step_still_reports_what_is_there(
    home, fake_installs, monkeypatch, capsys
):
    def offline() -> None:
        raise RuntimeError("Could not download silero_vad.onnx")

    monkeypatch.setattr(setup, "install_voice", offline)
    code, doc = run_json(capsys, ["setup", "--only", "panns", "voice"])
    assert code == 1 and doc["ok"] is False
    assert doc["error"] == {"code": "error", "message": "Could not download silero_vad.onnx"}
    comps = doc["outputs"]["components"]
    assert comps["panns"]["files"][1]["status"] == "downloaded"
    assert comps["voice"]["files"][0] == {
        "path": str(event_paths.silero_vad()),
        "bytes": None,
        "status": "missing",
    }


def test_setup_json_bad_component_is_bad_args(home, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["setup", "--only", "bogus", "--json"])
    assert exc.value.code == 2
    doc = envelope(capsys.readouterr().out)
    assert doc["command"] == "setup" and doc["error"]["code"] == "bad_args"


def test_setup_json_visual_without_events_extra_is_missing_extra(home, monkeypatch, capsys):
    real = setup.importlib.util.find_spec
    monkeypatch.setattr(setup, "download", lambda url, dst, *_a, **_k: None)
    monkeypatch.setattr(
        setup.importlib.util, "find_spec", lambda m, *a: None if m == "torch" else real(m, *a)
    )
    code, doc = run_json(capsys, ["setup", "--only", "visual"])
    assert code == 1 and doc["ok"] is False
    assert doc["error"]["code"] == "missing_extra"
    assert doc["error"]["message"].startswith("torch is not installed; it comes with the 'events'")


# ------------------------------------------------------------------------------ 360: helpers
@pytest.fixture
def pano_project(home, tmp_path) -> tuple[Path, x3.X3Clip]:
    """A project with one scanned X3 clip whose files exist (empty)."""
    cam = tmp_path / "cam"
    insv = [write(cam / n, "x") for n in ("VID_20260101_100000_00_001.insv",)]
    lrv = write(cam / "LRV_20260101_100000_11_001.lrv", "x")
    clip = x3clip(insv=[str(p) for p in insv], lrv=str(lrv), duration=12.0, frames=360)
    project = home / "projects" / "trip"
    P.save(project, [clip], [cam])
    return project, clip


# ------------------------------------------------------------------------------- 360 scan
def test_360_scan_json(home, tmp_path, monkeypatch, capsys):
    cam = tmp_path / "cam"
    insv = write(cam / "VID_20260101_100000_00_001.insv", "x" * 7)
    clip = x3clip(insv=[str(insv)], duration=12.5, frames=375, bytes_insv=7, mode="4k")
    monkeypatch.setattr(x3, "scan", lambda folder: [clip])
    monkeypatch.setattr(x3, "imu_source", lambda src: None)
    code, doc = run_json(capsys, ["360", "scan", str(cam), "--out", "trip"])
    project = home / "projects" / "trip"
    assert code == 0 and doc["ok"] is True and doc["command"] == "360 scan"
    assert doc["project"] == str(project)
    out = doc["outputs"]
    assert out["pano"] == str(project / "pano.json") and Path(out["pano"]).is_file()
    assert out["timings"] == str(project / "pano_timings.json")
    assert Path(out["timings"]).is_file()
    assert out["clips"] == [
        {
            "id": "001",
            "mode": "4k",
            "duration": 12.5,
            "frames": 375,
            "insv": [str(insv)],
            "insv_bytes": 7,
            "lrv": None,
        }
    ]
    assert (out["count"], out["total_seconds"]) == (1, 12.5)


def test_360_scan_json_missing_folder(home, tmp_path, capsys):
    code, doc = run_json(capsys, ["360", "scan", str(tmp_path / "nope"), "--out", "trip"])
    assert code == 1 and doc["error"]["code"] == "not_found"


# ---------------------------------------------------------------------------- 360 analyze
def test_360_analyze_json_views_only(pano_project, monkeypatch, capsys):
    project, _ = pano_project
    monkeypatch.setattr(VW, "run_layer", lambda clips, force=False: {"cached": 0, "computed": 1})
    code, doc = run_json(capsys, ["360", "analyze", "trip", "--views-only"])
    assert code == 0 and doc["ok"] is True and doc["command"] == "360 analyze"
    assert doc["project"] == str(project)
    out = doc["outputs"]
    assert out == {
        "timings": str(project / "pano_timings.json"),
        "views": {"cached": 0, "computed": 1},
    }
    assert Path(out["timings"]).is_file()


def test_360_analyze_json_without_pano(home, capsys):
    code, doc = run_json(capsys, ["360", "analyze", "nothing", "--views-only"])
    assert code == 1 and doc["error"]["code"] == "not_found"


# ----------------------------------------------------------------------------- 360 status
@pytest.fixture
def no_sdk(monkeypatch):
    status = {"exe": None, "origin": "none", "found": False, "runs": False, "version": None}
    monkeypatch.setattr(pano_stitch, "sdk_status", lambda: dict(status))


def test_360_status_json(pano_project, no_sdk, monkeypatch, capsys):
    project, _ = pano_project
    monkeypatch.setattr(P, "find_studio_export", lambda c: (None, "no Studio export"))
    monkeypatch.setattr(VW, "load_views", lambda src: None)
    code, doc = run_json(capsys, ["360", "status", "trip"])
    assert code == 0 and doc["ok"] is True and doc["command"] == "360 status"
    assert doc["project"] == str(project)
    out = doc["outputs"]
    assert out["sdk"] == {"exe": None, "origin": "none", "found": False, "runs": False,
                          "version": None}  # fmt: skip
    assert out["clips"] == [
        {
            "id": "001",
            "mode": "5.7k",
            "duration": 12.0,
            "views": False,
            "render": "dfisheye",
            "render_file": None,
        }
    ]


def test_360_status_json_unknown_project(home, no_sdk, capsys):
    code, doc = run_json(capsys, ["360", "status", "nothing"])
    assert code == 1 and doc["error"]["code"] == "not_found"
    assert doc["outputs"]["sdk"]["found"] is False  # the SDK part ran first


# --------------------------------------------------------------------------- 360 director
def test_360_director_json(pano_project, monkeypatch, capsys):
    project, clip = pano_project
    segments = project / "picks.json"
    cache.save_json(segments, [{"source": clip.lrv, "start": 1.0, "end": 4.0, "label": "jump"}])
    monkeypatch.setattr(
        D, "compute_path", lambda *a, **k: {"version": D.PATH_VERSION, "keys": [], "cuts": []}
    )
    code, doc = run_json(capsys, ["360", "director", "trip", "--segments", str(segments)])
    out_dir = project / "pano_paths" / "director"
    assert code == 0 and doc["ok"] is True and doc["command"] == "360 director"
    assert doc["project"] == str(project)
    out = doc["outputs"]
    assert out["paths_dir"] == str(out_dir)
    assert out["paths"] == [{"path": str(out_dir / "001_jump.json"), "preview": None}]
    assert Path(out["paths"][0]["path"]).is_file()
    assert (out["count"], out["vl_check"]) == (1, None)


def test_360_director_json_vl_check_only_when_this_run_saved_it(pano_project, monkeypatch, capsys):
    project, clip = pano_project
    segments = project / "picks.json"
    cache.save_json(segments, [{"source": clip.lrv, "start": 1.0, "end": 4.0, "label": "jump"}])
    out_dir = project / "pano_paths" / "director"
    write(out_dir / "vl_check.json", "{}")  # an earlier run's
    monkeypatch.setattr(
        D, "compute_path", lambda *a, **k: {"version": D.PATH_VERSION, "keys": [], "cuts": []}
    )
    argv = ["360", "director", "trip", "--segments", str(segments), "--force"]
    code, doc = run_json(capsys, argv)
    assert code == 0 and doc["outputs"]["vl_check"] is None

    unloads: list[int] = []

    class Checker:
        model, dead = "m", False

        def available(self) -> bool:
            return True

        def unload(self) -> None:
            unloads.append(1)

    def render_segment(clip, path, start, end, preview, **_k) -> dict:
        write(preview, "mp4")
        return {"seconds": 0.1}

    vl = {"bad": 0, "frames": 1, "flags": [], "seconds": 0.1}
    monkeypatch.setattr(vlcheck, "VLChecker", Checker)
    monkeypatch.setattr(vlcheck, "check_preview", lambda *a, **k: ([], vl))
    monkeypatch.setattr(pano_render, "render_segment", render_segment)
    code, doc = run_json(capsys, [*argv, "--vl-check"])
    assert code == 0 and doc["outputs"]["vl_check"] == str(out_dir / "vl_check.json")
    assert cache.load_json(out_dir / "vl_check.json")["segments"][0]["stem"] == "001_jump"
    assert unloads == [1]  # the vision model is freed once, after the last check


@pytest.mark.parametrize("ending", ["dead", "error"])
def test_360_director_vl_check_unloads_the_model_on_any_exit(
    pano_project, monkeypatch, capsys, ending
):
    project, clip = pano_project
    segments = project / "picks.json"
    picks = [
        {"source": clip.lrv, "start": 1.0, "end": 4.0, "label": "jump"},
        {"source": clip.lrv, "start": 5.0, "end": 8.0, "label": "land"},
    ]
    cache.save_json(segments, picks)
    monkeypatch.setattr(
        D, "compute_path", lambda *a, **k: {"version": D.PATH_VERSION, "keys": [], "cuts": []}
    )
    unloads: list[int] = []
    checks: list[int] = []

    class Checker:
        model, dead = "m", False

        def available(self) -> bool:
            return True

        def unload(self) -> None:
            unloads.append(1)

    def check_preview(checker, *a, **k):
        checks.append(1)
        if ending == "error":
            raise RuntimeError("boom")
        checker.dead = True  # stops answering after the first segment
        return [], {"bad": 0, "frames": 1, "flags": [], "seconds": 0.1}

    def render_segment(clip, path, start, end, preview, **_k) -> dict:
        write(preview, "mp4")
        return {"seconds": 0.1}

    monkeypatch.setattr(vlcheck, "VLChecker", Checker)
    monkeypatch.setattr(vlcheck, "check_preview", check_preview)
    monkeypatch.setattr(pano_render, "render_segment", render_segment)
    argv = ["360", "director", "trip", "--segments", str(segments), "--force", "--vl-check"]
    code, doc = run_json(capsys, argv)
    assert code == (1 if ending == "error" else 0)
    assert ("vl_check_unavailable" in warning_codes(doc)) == (ending == "dead")
    assert checks == [1]  # a dead model is not asked again; an error ends the run
    assert unloads == [1]  # freed anyway, once


def test_360_director_json_bad_out_name(pano_project, capsys):
    code, doc = run_json(capsys, ["360", "director", "trip", "--out-name", "../x"])
    assert code == 1 and doc["error"]["code"] == "bad_args"


# ----------------------------------------------------------------------------- 360 stitch
def test_360_stitch_json_never_reports_sdk_output(pano_project, monkeypatch, capsys):
    project, _ = pano_project
    log = project / "stitch.log"
    recs = [
        {"clip": "001", "output": "a.mp4", "status": "dry-run",
         "command": ["sdk", "-x"], "estimated_gb": 1.5},
        {"clip": "002", "output": str(project / "b.mp4"), "status": "failed", "seconds": 3.0,
         "log_file": str(log), "error": "sdk said no", "log": ["sdk line"], "sdk": {"x": 1}},
    ]  # fmt: skip
    monkeypatch.setattr(pano_stitch, "stitch_clips", lambda clips, force, dry_run: recs)
    monkeypatch.setattr(capabilities, "nvidia_present", lambda: False)
    code, doc = run_json(capsys, ["360", "stitch", "trip", "--clips", "001", "--dry-run"])
    assert code == 0 and doc["ok"] is True and doc["command"] == "360 stitch"
    out = doc["outputs"]
    assert out["dry_run"] is True
    assert out["clips"] == [
        {"clip": "001", "output": str(Path("a.mp4").absolute()), "status": "dry-run",
         "estimated_gb": 1.5},
        {"clip": "002", "output": str(project / "b.mp4"), "status": "failed", "seconds": 3.0,
         "log_file": str(log)},
    ]  # fmt: skip
    assert warning_codes(doc) == ["sdk_needs_nvidia", "stitch_failed"]
    assert doc["warnings"][1]["files"] == [str(log)]
    assert "sdk said no" not in json.dumps(doc) and "sdk line" not in json.dumps(doc)


def test_360_stitch_json_nothing_selected(pano_project, capsys):
    code, doc = run_json(capsys, ["360", "stitch", "trip", "--clips", "999", "--dry-run"])
    assert code == 1 and doc["error"]["code"] == "error"
    assert "No 360 clips selected" in doc["error"]["message"]


# ------------------------------------------------------------------------------ 360 check
def test_360_check_json(home, monkeypatch, capsys):
    clips_dir = home / "projects" / "trip" / "clips"
    clips_dir.mkdir(parents=True)
    check_dir = clips_dir.with_name("clips_check")

    def sheets(d: Path, per_clip: int) -> list[Path]:
        return [write(check_dir / "sheet_01.jpg", "jpg")]

    monkeypatch.setattr(pano_check, "clip_sheets", sheets)
    code, doc = run_json(capsys, ["360", "check", "projects/trip/clips"])
    assert code == 0 and doc["ok"] is True and doc["command"] == "360 check"
    assert doc["project"] == str(home / "projects" / "trip")
    assert doc["outputs"] == {
        "check_dir": str(check_dir),
        "sheets": [str(check_dir / "sheet_01.jpg")],
    }


def test_360_check_json_missing_folder(home, capsys):
    code, doc = run_json(capsys, ["360", "check", "projects/trip/clips"])
    assert code == 1 and doc["error"]["code"] == "not_found"


# --------------------------------------------------------------------------------- 360 qa
def test_360_qa_json(home, monkeypatch, capsys):
    qa_dir = home / "projects" / "trip" / "pano_paths" / "mix"
    qa_dir.mkdir(parents=True)
    out_dir = qa_dir.with_name("mix_qa")
    seen = {}

    def run_qa(d: Path, fps: float, size, json_only: bool) -> dict:
        seen["json_only"] = json_only
        write(out_dir / "qa.json", "{}")
        return {
            "mode": "rendered",
            "out_dir": str(out_dir),
            "sheets": [],
            "clips": [
                {"file": "001_a.mp4", "score": 90.0, "pass": True, "fails": []},
                {"file": "002_b.mp4", "score": 40.0, "pass": False, "fails": ["occluded"]},
            ],
            "skipped": [{"file": "003_c.mp4", "error": "no frames decoded"}],
            "edit": {"score": 65.0},
        }

    monkeypatch.setattr(pano_qa, "run_qa", run_qa)
    monkeypatch.setattr(pano_qa, "log_report", lambda report: None)
    code, doc = run_json(capsys, ["360", "qa", "projects/trip/pano_paths/mix", "--json-only"])
    assert seen == {"json_only": True}
    assert code == 0 and doc["ok"] is True and doc["command"] == "360 qa"
    assert doc["project"] == str(home / "projects" / "trip")
    out = doc["outputs"]
    assert out["qa_dir"] == str(out_dir) and out["report"] == str(out_dir / "qa.json")
    assert Path(out["report"]).is_file()
    assert (out["mode"], out["sheets"], out["count"], out["passed"], out["score"]) == (
        "rendered", [], 2, 1, 65.0,
    )  # fmt: skip
    assert out["clips"][1] == {
        "file": str(qa_dir / "002_b.mp4"),
        "score": 40.0,
        "pass": False,
        "fails": ["occluded"],
    }
    assert warning_codes(doc) == ["qa_clips_skipped"]
    assert doc["warnings"][0]["files"] == [str(qa_dir / "003_c.mp4")]


def test_360_qa_json_missing_folder(home, capsys):
    code, doc = run_json(capsys, ["360", "qa", "projects/trip/clips"])
    assert code == 1 and doc["error"]["code"] == "not_found"


# ----------------------------------------------------------------------------- 360 repair
# The envelope of a repair run: test_pano_repair.test_repair_json_output_shape.
def test_360_repair_json_missing_qa(home, capsys):
    project = home / "projects" / "trip"
    project.mkdir(parents=True)
    cache.save_json(project / "mix_segments.json", [{"source": "a.mp4", "start": 0, "end": 1}])
    code, doc = run_json(capsys, ["360", "repair", "trip", "--qa", "nope/qa.json"])
    assert code == 1 and doc["ok"] is False and doc["error"]["code"] == "not_found"
    assert doc["project"] == str(project)
