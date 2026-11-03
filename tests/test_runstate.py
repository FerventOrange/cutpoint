"""progress.json and the project run lock of the long project commands (#109 phase 2)."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

import cutpoint.analyze
from conftest import envelope
from cutpoint import cache, cli, folders, procs, progress, runstate

KEYS = {
    "version", "generator", "command", "argv", "cwd", "options_key", "status", "pid", "host",
    "started", "updated", "exit_code", "error", "sources", "steps",
}  # fmt: skip
LOCK = ".analysis.run.cutpoint-lock"


@pytest.fixture
def project(home: Path) -> Path:
    p = home / "projects" / "p"
    p.mkdir(parents=True)
    cache.save_json(p / "highlights.json", {"sources": [{"path": "/videos/a.mp4"}]})
    return p


def _read(project: Path) -> dict:
    return json.loads((project / runstate.FILE).read_text(encoding="utf-8"))


def _fake_events(monkeypatch, body=None) -> list[dict]:
    """``cutpoint events`` as a fake that runs two progress steps, then ``body``; returns
    the progress.json records seen at the start of the command and at the second step."""
    seen: list[dict] = []

    def cmd(a):
        project = runstate._ACTIVE.project
        seen.append(_read(project))
        with progress.run(("one", "two")) as steps:
            with steps.step("one"):
                pass
            with steps.step("two"):
                seen.append(_read(project))
                if body is not None:
                    body(project)

    monkeypatch.setattr(cli, "_cmd_events", cmd)
    return seen


def test_events_writes_the_schema_and_ends_done(project, monkeypatch):
    seen = _fake_events(monkeypatch)
    assert cli.main(["events", "p"]) == 0
    start, second = seen
    assert set(start) == KEYS and start["version"] == 1
    assert start["status"] == "running" and start["command"] == "events"
    assert start["argv"] == ["events", "p"] and start["cwd"] == str(Path.cwd())
    assert start["generator"].startswith("cutpoint ")
    assert start["pid"] == os.getpid() and start["host"] == socket.gethostname()
    assert start["sources"] == ["/videos/a.mp4"]  # from highlights.json
    assert start["exit_code"] is None and start["error"] is None and start["steps"] == {}
    assert second["steps"]["one"] == {"state": "done", "done": None, "total": None, "pct": 100}
    assert second["steps"]["two"]["state"] == "running"
    end = _read(project)
    assert end["status"] == "done" and end["exit_code"] == 0 and end["error"] is None
    assert end["steps"]["two"]["state"] == "done"
    assert end["options_key"] == start["options_key"]
    assert not (project / LOCK).exists()  # the lock is given back


def test_options_key_follows_the_options_not_json_or_verbose(project, monkeypatch):
    _fake_events(monkeypatch)
    cli.main(["events", "p"])
    plain = _read(project)["options_key"]
    cli.main(["-v", "events", "p", "--json"])
    assert _read(project)["options_key"] == plain
    cli.main(["events", "p", "--no-vlm"])
    assert _read(project)["options_key"] != plain


@pytest.mark.parametrize(
    ("error", "code", "status", "err_code"),
    [
        (RuntimeError("boom"), 1, "failed", "error"),
        (FileNotFoundError("gone"), 1, "failed", "not_found"),
        (KeyboardInterrupt(), 130, "interrupted", "interrupted"),
    ],
)
def test_status_per_exit_code(project, monkeypatch, error, code, status, err_code):
    def body(project):
        raise error

    _fake_events(monkeypatch, body)
    assert cli.main(["events", "p"]) == code
    rec = _read(project)
    assert rec["status"] == status and rec["exit_code"] == code
    assert rec["error"]["code"] == err_code
    if status == "interrupted":
        assert rec["error"] == {"code": "interrupted", "message": "interrupted"}  # as --json
    assert rec["steps"]["two"]["state"] == "failed"
    assert not (project / LOCK).exists()


def test_a_failed_write_never_stops_the_run_and_keeps_the_last_file(project, monkeypatch, caplog):
    """Written atomically (cache.save_json): a write that fails leaves the previous file
    whole and no temporary file; the run goes on, with one WARNING."""

    denied = []
    replace = os.replace

    def flaky(src, dst):
        if denied:
            raise OSError(28, "No space left on device")
        replace(src, dst)

    monkeypatch.setattr(cache.os, "replace", flaky)
    _fake_events(monkeypatch, lambda project: denied.append(True))
    assert cli.main(["events", "p"]) == 0
    rec = _read(project)
    assert rec["status"] == "running" and rec["steps"]["two"]["state"] == "running"
    assert sorted(p.name for p in project.iterdir()) == ["highlights.json", runstate.FILE]
    assert sum("Could not write" in r.message for r in caplog.records) == 1


def test_read_reports_a_running_record_of_a_dead_process_as_interrupted(project, monkeypatch):
    rec = {"version": 1, "status": "running", "pid": 4242, "host": socket.gethostname()}
    cache.save_json(project / runstate.FILE, rec)
    monkeypatch.setattr(procs, "pid_alive", lambda pid, not_after=None: True)
    assert runstate.read(project)["status"] == "running"
    monkeypatch.setattr(procs, "pid_alive", lambda pid, not_after=None: False)
    assert runstate.read(project)["status"] == "interrupted"
    assert _read(project)["status"] == "running"  # read never rewrites the file
    cache.save_json(project / runstate.FILE, {**rec, "host": "elsewhere"})
    assert runstate.read(project)["status"] == "running"  # another computer: cannot tell
    cache.save_json(project / runstate.FILE, {**rec, "pid": os.getpid()})
    assert runstate.read(project)["status"] == "interrupted"  # this process records no run
    cache.save_json(project / runstate.FILE, {**rec, "status": "done"})
    assert runstate.read(project)["status"] == "done"
    assert runstate.read(project / "missing") is None


def _other_runs_lock(project: Path, pid: int) -> Path:
    lock = project / LOCK
    rec = {"pid": pid, "host": socket.gethostname(), "started": "2026-11-03 10:00:00"}
    lock.write_text(json.dumps(rec), encoding="utf-8")
    return lock


@pytest.mark.parametrize("command", [["events", "p"], ["crashes", "p"], ["profile", "p"]])
def test_a_second_run_on_the_project_gets_folder_busy(project, monkeypatch, capsys, command):
    lock = _other_runs_lock(project, 4242)
    monkeypatch.setattr(procs, "pid_alive", lambda pid, not_after=None: True)
    ran = []
    for name in ("_cmd_events", "_cmd_crashes", "_cmd_profile"):
        monkeypatch.setattr(cli, name, ran.append)
    assert cli.main([*command, "--json"]) == 1
    doc = envelope(capsys.readouterr().out)
    assert doc["error"]["code"] == "folder_busy" and str(lock) in doc["error"]["message"]
    assert ran == [] and lock.exists()
    assert not (project / runstate.FILE).exists()  # the other run's record is not touched


def test_a_stale_lock_of_a_dead_process_is_taken_over(project, monkeypatch):
    _other_runs_lock(project, 4242)
    monkeypatch.setattr(procs, "pid_alive", lambda pid, not_after=None: False)
    _fake_events(monkeypatch)
    assert cli.main(["events", "p"]) == 0
    assert _read(project)["status"] == "done" and not (project / LOCK).exists()


def test_nested_records_and_hooks_do_not_lock_the_run_out(project, monkeypatch):
    """edit runs analyze, 360 analyze runs analyze and the events: one record and one lock
    per process; analyze's hooks inside it change nothing."""

    def body(project):
        with runstate.record("analyze", ["analyze"], project) as inner:
            assert inner is None
        with runstate.project_ready(project):
            runstate.transcribed(project, [Path("/videos/b.mp4")])
        assert (project / LOCK).exists()

    _fake_events(monkeypatch, body)
    assert cli.main(["events", "p"]) == 0
    rec = _read(project)
    assert rec["command"] == "events" and rec["status"] == "done"
    assert rec["sources"] == ["/videos/a.mp4"]  # written from the start: not replaced


def test_other_commands_keep_no_record(project, monkeypatch):
    monkeypatch.setattr(cli, "_cmd_paths", lambda a: None)
    assert cli.main(["paths"]) == 0
    assert not (project / runstate.FILE).exists()


# -- analyze and edit: progress.json once the first source is transcribed (Q1 c) ----------


def _two_sources(tmp_path: Path, monkeypatch, fail_at: int, error: BaseException) -> list[Path]:
    videos = tmp_path / "videos"
    videos.mkdir()
    sources = [videos / "a.mp4", videos / "b.mp4"]
    for s in sources:
        s.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cutpoint.analyze, "Transcriber", lambda **kw: None)
    calls: list[Path] = []

    def analyze_source(src, *a, **kw):
        calls.append(src)
        if len(calls) == fail_at:
            raise error
        return SimpleNamespace(segments=[], read_error=None)

    monkeypatch.setattr(cutpoint.analyze, "analyze_source", analyze_source)
    return sources


@pytest.mark.parametrize("command", ["analyze", "edit"])
@pytest.mark.parametrize(
    ("error", "code", "status"),
    [(KeyboardInterrupt(), 130, "interrupted"), (RuntimeError("bad"), 1, "failed")],
)
def test_first_run_keeps_its_folder_once_a_source_is_transcribed(
    home, tmp_path, monkeypatch, command, error, code, status
):
    sources = _two_sources(tmp_path, monkeypatch, 2, error)
    argv = [command, str(sources[0].parent), "--no-vision", "--out", "new"]
    assert cli.main(argv) == code
    project = home / "projects" / "new"
    assert sorted(p.name for p in project.iterdir()) == [runstate.FILE]  # no lock left
    rec = _read(project)
    assert rec["command"] == command and rec["status"] == status
    assert rec["exit_code"] == code and rec["argv"] == argv
    assert rec["sources"] == [str(s) for s in sources]
    assert rec["steps"]["transcribe"]["state"] == "failed"


@pytest.mark.parametrize("command", ["analyze", "edit"])
def test_first_run_stopped_before_a_transcript_leaves_no_folder(
    home, tmp_path, monkeypatch, command
):
    sources = _two_sources(tmp_path, monkeypatch, 1, KeyboardInterrupt())
    assert cli.main([command, str(sources[0]), "--no-vision", "--out", "new"]) == 130
    projects = home / "projects"
    assert not projects.exists() or list(projects.iterdir()) == []


def test_a_rerun_of_analyze_records_and_locks_at_once(project, tmp_path, monkeypatch):
    cache.save_json(project / runstate.FILE, {"version": 1, "status": "done", "run": "earlier"})
    sources = _two_sources(tmp_path, monkeypatch, 1, RuntimeError("bad"))
    lock_seen = []
    real = cutpoint.analyze.analyze_source

    def analyze_source(src, *a, **kw):
        lock_seen.append((project / LOCK).exists())
        return real(src, *a, **kw)

    monkeypatch.setattr(cutpoint.analyze, "analyze_source", analyze_source)
    assert cli.main(["analyze", str(sources[0]), "--no-vision", "--out", "p"]) == 1
    assert lock_seen == [True]
    rec = _read(project)  # nothing transcribed, yet this run's record replaced the old one
    assert "run" not in rec and rec["command"] == "analyze" and rec["status"] == "failed"
    assert rec["sources"] == ["/videos/a.mp4"]  # from highlights.json until a transcript
    assert not (project / LOCK).exists()


def test_a_rerun_of_analyze_records_its_sources_once_transcribed(project, tmp_path, monkeypatch):
    sources = _two_sources(tmp_path, monkeypatch, 2, RuntimeError("bad"))
    assert cli.main(["analyze", str(sources[0].parent), "--no-vision", "--out", "p"]) == 1
    rec = _read(project)
    assert rec["command"] == "analyze" and rec["status"] == "failed"
    assert rec["sources"] == [str(s) for s in sources]


@pytest.mark.parametrize("command", ["events", "profile"])
def test_a_folder_that_is_no_project_is_left_as_it_was(home, monkeypatch, capsys, command):
    """A typo or the footage folder: not_found, and no progress.json or lock in it."""
    folder = home / "projects" / "videos"
    folder.mkdir(parents=True)
    (folder / "video.mp4").write_bytes(b"x")
    assert cli.main([command, "videos", "--json"]) == 1
    assert envelope(capsys.readouterr().out)["error"]["code"] == "not_found"
    assert sorted(p.name for p in folder.iterdir()) == ["video.mp4"]


def test_first_analyze_on_a_busy_new_project_gets_folder_busy(home, tmp_path, monkeypatch):
    """Two first runs on one new project: the second stops once analyze made the folder,
    before any transcription."""
    sources = _two_sources(tmp_path, monkeypatch, 99, RuntimeError("unused"))
    project = home / "projects" / "new"
    reached = []

    def make_dirs(path, made):
        path.mkdir(parents=True, exist_ok=True)  # the other run made the folder meanwhile
        _other_runs_lock(project, 4242)

    monkeypatch.setattr(folders, "make_dirs", make_dirs)
    monkeypatch.setattr(procs, "pid_alive", lambda pid, not_after=None: True)
    monkeypatch.setattr(cutpoint.analyze, "analyze_source", lambda *a, **k: reached.append(a))
    assert cli.main(["analyze", str(sources[0]), "--no-vision", "--out", "new"]) == 1
    assert reached == [] and (project / LOCK).exists()
    assert not (project / runstate.FILE).exists()


# -- percent throttle -----------------------------------------------------------------------


def test_pct_is_written_at_most_every_five_seconds(tmp_path, monkeypatch):
    rec = runstate.Recorder("events", ["events", "p"], tmp_path, {})
    rec.armed = True
    writes: list[dict] = []
    monkeypatch.setattr(cache, "save_json", lambda path, data: writes.append(json.loads(
        json.dumps(data))))  # fmt: skip
    now = [100.0]
    monkeypatch.setattr(runstate.time, "monotonic", lambda: now[0])
    rec.observe("one", "running", None, None, None)  # a state change: at once
    assert len(writes) == 1
    for t, pct in ((101.0, 10), (104.9, 20)):
        now[0] = t
        rec.observe("one", "running", pct, None, None)
    assert len(writes) == 1
    now[0] = 105.0
    rec.observe("one", "running", 30, None, None)
    assert len(writes) == 2 and writes[-1]["steps"]["one"]["pct"] == 30
    now[0] = 105.1
    rec.observe("one", "running", 30, 3, 8)  # counts at the step's start: at once
    rec.observe("one", "done", 100, None, None)  # a state change: at once
    assert len(writes) == 4
    assert writes[-1]["steps"]["one"] == {"state": "done", "done": 8, "total": 8, "pct": 100}


def test_progress_observers_get_updates_without_cutpoint_progress(monkeypatch):
    monkeypatch.delenv(progress.ENV, raising=False)
    got: list[tuple] = []
    monkeypatch.setattr(progress.Run, "observers", [lambda *a: got.append(a)])
    with progress.run(("one", "two")) as steps:
        steps.start("one")
        steps.update(40)
        progress.continuing("one", 2, 5)
        with progress.run(("x",), use=False) as quiet:  # reanalyze inside events: not seen
            quiet.start("x")
    assert got == [
        ("one", "running", None, None, None),
        ("one", "running", 40, None, None),
        ("one", "running", 40, 2, 5),
        ("one", "done", 100, None, None),
        ("two", "skipped", None, None, None),
    ]


def test_360_analyze_records_once_and_does_not_lock_itself_out(project, monkeypatch):
    """``360 analyze`` runs analyze and the events in one process under one lock."""
    from cutpoint.events import pipeline  # noqa: PLC0415
    from cutpoint.pano import candidates, views  # noqa: PLC0415
    from cutpoint.pano import cli as pano_cli  # noqa: PLC0415

    order: list[str] = []

    def analyze(sources, style, out_dir, **kw):
        with folders.output_dir(out_dir), runstate.project_ready(out_dir):
            runstate.transcribed(out_dir, sources)  # replaces the highlights.json sources
            order.append("analyze")

    def run_events(project, **kw):
        with runstate.record("events", ["events"], project) as inner:
            assert inner is None and (project / LOCK).exists()
            order.append("events")

    monkeypatch.setattr(pano_cli, "_clips", lambda p: [argparse.Namespace(lrv="/v/a.lrv")])
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "run_events", run_events)
    monkeypatch.setattr("cutpoint.crash.dense_needs_run", lambda sources: False)
    monkeypatch.setattr(views, "run_layer", lambda clips, force=False, handoff=None: {})
    monkeypatch.setattr(cutpoint.analyze, "analyze", analyze)
    monkeypatch.setattr(cutpoint.analyze, "reanalyze_if_reachable", lambda p: None)
    monkeypatch.setattr("cutpoint.pano.project.default_style", lambda: "action-cam")
    monkeypatch.setattr("cutpoint.styles.get_style", lambda name: name)
    monkeypatch.setattr(candidates, "cache_candidate_paths", lambda p: {"seconds": 0.0})
    assert cli.main(["360", "analyze", "p"]) == 0
    assert order == ["analyze", "events"]
    rec = _read(project)
    assert rec["command"] == "360 analyze" and rec["status"] == "done"
    # written from the start, then this run's sources
    assert rec["sources"] == [str(Path("/v/a.lrv"))]
    assert [rec["steps"][s]["state"] for s in progress.PANO_STEPS] == [
        "skipped", "done", "done", "done",
    ]  # fmt: skip
    assert not (project / LOCK).exists()


def test_360_analyze_scan_of_a_new_project_locks_and_records_once_scanned(
    home, tmp_path, monkeypatch
):
    """``360 analyze new --scan F --views-only``: the folder is made by the scan; the lock is
    held from then on (views included) and progress.json ends done."""
    from cutpoint.pano import cli as pano_cli  # noqa: PLC0415
    from cutpoint.pano import views  # noqa: PLC0415

    project = home / "projects" / "new"
    footage = tmp_path / "x3"
    footage.mkdir()

    def scan(project, root, **kw):
        project.mkdir(parents=True, exist_ok=True)
        cache.save_json(project / "pano.json", {"clips": []})
        return 1

    locked = []

    def run_layer(clips, force=False, handoff=None):
        locked.append((project / LOCK).exists())
        return {}

    monkeypatch.setattr(pano_cli, "_scan", scan)
    monkeypatch.setattr(pano_cli, "_require_detector", lambda: None)
    monkeypatch.setattr(pano_cli, "_clips", lambda p: [argparse.Namespace(lrv="/v/a.lrv")])
    monkeypatch.setattr(views, "run_layer", run_layer)
    assert cli.main(["360", "analyze", "new", "--scan", str(footage), "--views-only"]) == 0
    assert locked == [True]
    rec = _read(project)
    assert rec["command"] == "360 analyze" and rec["status"] == "done"
    assert not (project / LOCK).exists()


def test_360_analyze_scan_of_a_new_project_records_the_transcribed_sources(
    home, tmp_path, monkeypatch
):
    """``360 analyze new --scan F``: no highlights.json, so the record starts with no sources;
    the nested analyze's transcribed sources replace them."""
    from cutpoint.events import pipeline  # noqa: PLC0415
    from cutpoint.pano import candidates, views  # noqa: PLC0415
    from cutpoint.pano import cli as pano_cli  # noqa: PLC0415

    project = home / "projects" / "new"
    footage = tmp_path / "x3"
    footage.mkdir()

    def scan(project, root, **kw):
        project.mkdir(parents=True, exist_ok=True)
        cache.save_json(project / "pano.json", {"clips": []})
        return 1

    seen = []

    def analyze(sources, style, out_dir, **kw):
        with folders.output_dir(out_dir), runstate.project_ready(out_dir):
            runstate.transcribed(out_dir, sources)
            seen.append(_read(out_dir)["sources"])

    monkeypatch.setattr(pano_cli, "_scan", scan)
    monkeypatch.setattr(pano_cli, "_require_detector", lambda: None)
    monkeypatch.setattr(pano_cli, "_clips", lambda p: [argparse.Namespace(lrv="/v/a.lrv")])
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "run_events", lambda project, **kw: None)
    monkeypatch.setattr("cutpoint.crash.dense_needs_run", lambda sources: False)
    monkeypatch.setattr(views, "run_layer", lambda clips, force=False, handoff=None: {})
    monkeypatch.setattr(cutpoint.analyze, "analyze", analyze)
    monkeypatch.setattr(cutpoint.analyze, "reanalyze_if_reachable", lambda p: None)
    monkeypatch.setattr("cutpoint.pano.project.default_style", lambda: "action-cam")
    monkeypatch.setattr("cutpoint.styles.get_style", lambda name: name)
    monkeypatch.setattr(candidates, "cache_candidate_paths", lambda p: {"seconds": 0.0})
    assert cli.main(["360", "analyze", "new", "--scan", str(footage)]) == 0
    assert seen == [[str(Path("/v/a.lrv"))]]  # written at once, not only at the end
    rec = _read(project)
    assert rec["command"] == "360 analyze" and rec["status"] == "done"
    assert rec["sources"] == [str(Path("/v/a.lrv"))]
    assert not (project / LOCK).exists()
