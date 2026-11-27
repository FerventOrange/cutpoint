"""Jobs of ``cutpoint ui``: one at a time, cancel and kill, kept assemblies, resume (#109)."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from conftest import needs_ffmpeg, patch_ui
from cutpoint import cache, cli, folders, procs, runstate, ui
from ui_helpers import (
    PROC_WAIT_S,
    JobSpy,
    _pid_alive,
    cli_project,
    make_project,
    peaks_job,
    progress_record,
    record_export,
    resume_job,
    touch,
    wait_job,
    wait_until,
)

# the encoder probe and the keyring stay stubbed in every test (ui_helpers fixtures)
pytestmark = pytest.mark.usefixtures("no_nvenc", "no_keyring")


@pytest.mark.parametrize(
    ("event_sent", "exits_in_time", "job", "expected"),
    [
        (1, False, 777, ["ctrl_break", "wait", "terminate", "taskkill"]),  # job object first
        (1, False, None, ["ctrl_break", "wait", "taskkill"]),  # ignores the event
        (1, True, None, ["ctrl_break", "wait", "taskkill"]),  # taskkill still follows
        (0, False, None, ["ctrl_break", "taskkill"]),  # no console: nothing to wait for
    ],
    ids=["job-object", "ignores-the-event", "soft-stop-works", "no-console"],
)
def test_windows_cancel_tries_ctrl_break_then_the_job_object_then_taskkill(
    monkeypatch: pytest.MonkeyPatch,
    event_sent: int,
    exits_in_time: bool,
    job: int | None,
    expected: list[str],
) -> None:
    calls: list[tuple[str, object]] = []

    def wait(timeout: float) -> int:
        calls.append(("wait", timeout))
        if exits_in_time:
            return 130
        raise subprocess.TimeoutExpired("job", timeout)

    kernel32 = SimpleNamespace(
        GenerateConsoleCtrlEvent=lambda event, pid: (
            calls.append(("ctrl_break", (event, pid))) or event_sent
        ),
        TerminateJobObject=lambda handle, code: calls.append(("terminate", (handle, code))) or 1,
    )
    fake_ctypes = SimpleNamespace(WinDLL=lambda name: kernel32, c_void_p=lambda value: value)
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    monkeypatch.setattr(procs, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(procs, "signal", SimpleNamespace(CTRL_BREAK_EVENT=1))
    monkeypatch.setattr(
        procs,
        "subprocess",
        SimpleNamespace(
            run=lambda args, **kw: calls.append(("taskkill", args)),
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    proc = SimpleNamespace(pid=4242, wait=wait, tree_job=job)
    procs.kill_tree(proc, 0.01)  # type: ignore[arg-type]
    # the job object reaches descendants that outlive the leader, which taskkill /T cannot
    assert [name for name, _ in calls] == expected
    got = dict(calls)
    assert got["ctrl_break"] == (1, 4242)
    assert got["taskkill"] == ["taskkill", "/PID", "4242", "/T", "/F"]
    assert got.get("wait", 0.01) == 0.01
    assert got.get("terminate", (777, 1)) == (777, 1)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_posix_kill_tree_waits_job_stop_s_between_term_and_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []
    waits: list[float | None] = []
    monkeypatch.setattr(procs.os, "killpg", lambda pid, sig: signals.append(sig))

    def wait(timeout: float | None = None) -> int:
        waits.append(timeout)
        raise subprocess.TimeoutExpired("job", timeout or 0)

    procs.kill_tree(SimpleNamespace(pid=4242, wait=wait))  # type: ignore[arg-type]
    # a job's own soft stop (two ffmpeg waits plus cleanup) fits before the hard kill (#257)
    assert waits == [procs.JOB_STOP_S]
    assert procs.JOB_STOP_S > 2 * procs.SOFT_STOP_S
    assert signals == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_posix_kill_tree_without_term_sends_only_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []
    monkeypatch.setattr(procs.os, "killpg", lambda pid, sig: signals.append(sig))

    def wait(timeout: float | None = None) -> int:
        raise AssertionError("no soft stop to wait for")

    procs.kill_tree(SimpleNamespace(pid=4242, wait=wait), term=False)  # type: ignore[arg-type]
    assert signals == [signal.SIGKILL]

    def gone(pid: int, sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(procs.os, "killpg", gone)  # the group already exited: nothing to report
    procs.kill_tree(SimpleNamespace(pid=4242, wait=wait), term=False)  # type: ignore[arg-type]


def test_jobs_list_sends_the_log_of_the_running_and_the_newest_job_only(
    client: TestClient,
) -> None:
    """A744: the page polls the jobs every second while one runs and shows the log of the
    running job, else of the newest; the other kept jobs (up to 50) come without their log."""
    jobs = client.app.state.jobs

    def keep(status: str, cancelled: bool = False) -> ui.Job:
        job = ui.Job("p", "analyze")
        job.status, job.cancelled = status, cancelled
        job.note("INFO", f"{status} job")
        jobs.jobs[job.id] = job
        return job

    old, failed = keep("completed"), keep("failed")
    running = keep("running", cancelled=True)  # Cancel asked, the process not gone yet
    try:
        got = client.get("/api/jobs").json()["jobs"]
        assert [j["id"] for j in got] == [old.id, failed.id, running.id]
        assert ["logs" in j for j in got] == [False, False, True]
        assert got[2]["status"] == "cancelling"
        assert got[2]["logs"][0].endswith("INFO running job")
        assert got[1]["status"] == "failed" and got[1]["progress"] == {"total": None, "steps": []}
        running.status = "cancelled"  # it ended: the newest keeps its log, for the page to show
        got = client.get("/api/jobs").json()["jobs"]
        assert ["logs" in j for j in got] == [False, False, True]
        assert got[2]["logs"][0].endswith("INFO running job")
        assert running.view()["logs"] == got[2]["logs"]  # cancel and start still send it
    finally:
        running.status = "cancelled"


@pytest.mark.parametrize(
    ("done", "cancelled", "expected"),
    [
        (True, True, []),  # exited by itself while stopping: no kill at all
        (False, True, [False]),  # still stuck after JOB_STOP_S: KILL only, no second TERM
        (False, False, [True]),  # not yet stopped: the full TERM then KILL, as before
    ],
)
def test_jobs_shutdown_does_not_term_a_cancelling_job_twice(
    monkeypatch: pytest.MonkeyPatch, done: bool, cancelled: bool, expected: list[bool]
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(procs, "kill_tree", lambda proc, wait=0, *, term=True: calls.append(term))
    monkeypatch.setattr(procs, "JOB_STOP_S", 0.01)
    jobs = ui.Jobs()
    job = ui.Job("p", "analyze")
    job.proc = SimpleNamespace(pid=4242)  # type: ignore[assignment]
    job.cancelled = cancelled
    if done:
        job.done.set()
    jobs.jobs[job.id] = job
    jobs.shutdown()
    assert calls == expected
    assert job.cancelled


@pytest.mark.slow
@pytest.mark.skipif(os.name != "nt", reason="Windows job objects and CTRL_BREAK")
def test_windows_cancel_ends_a_grandchild_that_ignores_ctrl_break() -> None:
    grandchild = (
        "import signal, time\n"
        "signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        # longer than PROC_WAIT_S: a child left alive cannot pass the death check by exiting
        "time.sleep(600)\n"
    )
    leader = (
        "import subprocess, sys, time\n"
        f"c = subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
        "print('child', c.pid, flush=True)\n"
        "time.sleep(600)\n"
    )
    jobs = ui.Jobs()
    job = jobs.start("p", "analyze", [sys.executable, "-c", leader])
    child = 0
    try:
        # two interpreter starts can be slow under a loaded suite
        wait_until(
            lambda: job.status != "running" or "ready" in list(job.logs),
            PROC_WAIT_S,
            "the grandchild's ready line",
        )
        logs = list(job.logs)
        assert "ready" in logs, (job.status, job.error, logs)
        child = int(next(line for line in logs if line.startswith("child")).split()[1])
        cancelled = jobs.cancel(job.id)
        # the leader dies on the event at once (only the grandchild ignores it), so the job
        # object kill follows promptly; JOB_STOP_S bounds the wait, plus time for the watcher
        # to see EOF
        wait_until(
            lambda: cancelled.status == "cancelled",
            procs.JOB_STOP_S + PROC_WAIT_S,
            "the job to be cancelled",
        )
        wait_until(lambda: not _pid_alive(child), PROC_WAIT_S, f"the grandchild {child} to die")
    finally:
        if child and _pid_alive(child):
            subprocess.run(["taskkill", "/PID", str(child), "/T", "/F"], capture_output=True)
        if job.proc is not None and job.proc.poll() is None:
            procs.kill_tree(job.proc)


def test_assemble_ignores_a_hidden_part_file_left_by_a_killed_export(
    client: TestClient, tmp_path: Path
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    (p / "clips_ui_1").mkdir()
    cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": []})
    (p / "clips_ui_1" / ".001_a.part.mp4").write_bytes(b"x")
    cache.save_json(p / "ui" / "exports.json", {"exports": [{"dir": "clips_ui_1"}]})
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "assemble"})
    assert res.status_code == 400
    assert "export clips before assembling" in res.text


@pytest.mark.slow
@needs_ffmpeg
def test_a_timeline_only_assembly_succeeds_without_a_video_and_is_listed(
    client: TestClient, tmp_path: Path
) -> None:
    """#519: Assemble with an fcpxml format writes only the timeline, so the job completes
    without a final video and the timeline is listed as the export's own final."""
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    record_export(p, None)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=d=2:s=320x240:r=25",
         "-f", "lavfi", "-i", "sine=d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(p / "clips_ui_1" / "001_a.mp4")],
        check=True,
    )  # fmt: skip
    body = {"action": "assemble", "format": "fcpxml", "loudnorm": False}
    res = client.post(f"/api/projects/{pid}/jobs", json=body)
    assert res.status_code == 200, res.text
    job = wait_job(client)
    assert job["status"] == "completed", job
    got = client.get(f"/api/projects/{pid}").json()
    assert [o["name"] for o in got["outputs"]] == ["final_ui_1.fcpxml"]
    assert got["final_current"] is True
    assert not list(p.glob("*.mp4"))


@pytest.mark.parametrize("body", [{"loudnorm": True}, {"captions": True}])
@pytest.mark.parametrize("fmt", ["fcpxml", "otio"])
def test_assemble_job_refuses_audio_processing_and_captions_with_a_timeline_format(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: dict, fmt: str
) -> None:
    """#519: as the CLI does (exit 2), a timeline format refuses Normalize loudness and Burn in
    captions instead of starting a job that would fail."""
    body = {"action": "assemble", "format": fmt, "loudnorm": False, **body}
    res, captured = peaks_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 400, res.text
    assert not captured


def test_a_timeline_only_final_is_listed_on_its_own_and_a_timeline_beside_its_video_is_not(
    client: TestClient, tmp_path: Path
) -> None:
    """#519: a timeline assembled without a video (``--format fcpxml``) is a final of its own,
    newest first; a timeline next to its .mp4 is listed after that video, not as another final."""
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    for d in ("clips_ui_1", "clips_ui_2"):
        (p / d).mkdir()
        cache.save_json(p / d / "manifest.json", {"clips": []})
    exports = [{"dir": "clips_ui_1", "final": "final_ui_1.fcpxml"}, {"dir": "clips_ui_2"}]
    cache.save_json(p / "ui" / "exports.json", {"exports": exports})  # clips_ui_2: not assembled
    touch(p / "final.mp4", 300)
    touch(p / "final_ui_1.fcpxml", 100)
    touch(p / "final.fcpxml", 10)  # newest, but it belongs to final.mp4
    got = client.get(f"/api/projects/{pid}").json()
    assert [o["name"] for o in got["outputs"]] == ["final_ui_1.fcpxml", "final.mp4", "final.fcpxml"]
    assert got["final_current"] is False


@pytest.mark.slow
@needs_ffmpeg
def test_export_keeps_saved_order_and_each_assembly_is_kept(
    client: TestClient, tmp_path: Path
) -> None:
    """Review order is the clip order; re-exporting or re-assembling never overwrites."""
    video = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=d=8:s=320x240:r=25",
         "-f", "lavfi", "-i", "sine=d=8", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(video)],
        check=True,
    )  # fmt: skip
    pid, folder = make_project(client, tmp_path, {"src.mp4": video.read_bytes()})
    src = str((folder / "src.mp4").resolve())
    segments = [
        {"id": "s-2", "source": src, "start": 5, "end": 7, "label": "late"},
        {"id": "s-1", "source": src, "start": 0.5, "end": 2.5, "label": "early"},
    ]
    assert client.put(f"/api/projects/{pid}/review", json={"segments": segments}).status_code == 200

    def run(body: dict) -> dict:
        res = client.post(f"/api/projects/{pid}/jobs", json=body)
        assert res.status_code == 200, res.text
        job = wait_job(client)
        assert job["status"] == "completed", job
        return client.get(f"/api/projects/{pid}").json()

    project = run({"action": "clips"})
    assert [(c["id"], c["start"], c["file"][:3]) for c in project["clips"]["clips"]] == [
        ("s-2", 5.0, "001"),
        ("s-1", 0.5, "002"),
    ]
    first_dir = project["clips"]["dir"]
    assert project["outputs"] == []

    taken = ui.project_dir(pid) / f"final{first_dir[5:]}.ass"  # a caption file already there
    taken.write_text("keep me", encoding="utf-8")
    first = run({"action": "assemble"})["outputs"]
    assert taken.read_text(encoding="utf-8") == "keep me"
    assert first[0]["name"] != f"{taken.stem}.mp4"
    second = run({"action": "assemble", "transition": "cut"})["outputs"]
    assert first[0]["name"].endswith(".mp4") and first[0]["name"] != second[0]["name"]
    new = [o["name"] for o in second if o["name"] not in {x["name"] for x in first}]
    assert new[0].endswith(".mp4")  # assemble writes no timelines by default
    assert {n.rsplit(".", 1)[1] for n in new} <= {"mp4", "srt"}
    assert first[0]["name"] in [o["name"] for o in second]  # the earlier final stays listed
    assert (ui.project_dir(pid) / first[0]["name"]).is_file()  # the earlier final survives
    assert client.get(second[0]["url"]).status_code == 200

    again = run({"action": "clips"})  # fresh folder; the earlier export stays untouched
    assert again["clips"]["dir"] != first_dir
    # Every final stays listed (newest first), but none is the new export's own yet.
    assert again["outputs"] == second and again["final_current"] is False
    assert (ui.project_dir(pid) / first_dir / "manifest.json").is_file()


@pytest.fixture
def scratch_appdata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """APPDATA too (conftest covers CUTPOINT_HOME and LOCALAPPDATA): the CLI children of these
    tests never see the user's real folders."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))


def test_a_project_with_only_progress_json_is_listed_as_interrupted_and_opens(
    client: TestClient, scratch_appdata: None
) -> None:
    p = ui.projects_dir() / "stopped"
    progress_record(p)
    (ui.projects_dir() / "empty").mkdir()  # no record, no analysis: still not a project
    listed = client.get("/api/projects").json()["projects"]
    assert listed == [{"id": "stopped", "name": "stopped", "interrupted": True}]
    res = client.get("/api/projects/stopped")
    assert res.status_code == 200, res.text
    got = res.json()
    assert got["highlights"] is None and got["sources"] == []  # progress.json grants no media
    assert got["run"] == {"command": "analyze", "status": "interrupted", "resumable": True}


@pytest.mark.parametrize(
    ("status", "analyzed", "resumable"),
    [
        ("interrupted", True, True),
        ("low_memory", True, True),
        ("failed", False, True),  # nothing but Resume can finish a first analysis
        ("failed", True, False),
        ("done", False, False),
    ],
)
def test_resume_is_offered_for_a_stopped_run(
    tmp_path: Path, status: str, analyzed: bool, resumable: bool
) -> None:
    p = tmp_path / "p"
    progress_record(p, status=status)
    assert ui.run_state(p, analyzed) == {
        "command": "analyze",
        "status": status,
        "resumable": resumable,
    }
    assert ui.run_state(tmp_path / "none", analyzed) is None


@pytest.mark.slow
def test_a_running_record_of_a_dead_process_reads_interrupted(tmp_path: Path) -> None:
    p = tmp_path / "p"
    proc = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    progress_record(p, status="running", pid=int(proc.stdout), host=socket.gethostname())
    assert ui.run_state(p, False)["status"] == "interrupted"


@pytest.mark.slow
def test_resume_preview_runs_the_dry_run_and_reports_its_steps(
    client: TestClient, scratch_appdata: None
) -> None:
    p = ui.projects_dir() / "stopped"
    progress_record(p, sources=[str(p.parent / "clip.mp4")])  # not there: the CLI warns
    res = client.get("/api/projects/stopped/resume")
    assert res.status_code == 200, res.text
    got = res.json()
    assert (got["command"], got["status"]) == ("analyze", "interrupted")
    assert got["steps"] == {"transcribe": {"state": "failed", "done": 1, "total": 3}}
    assert got["warnings"] == ["1 source(s) of the recorded run are not there now"]


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        ({"code": "folder_busy", "message": "... delete the lock"}, 409, ui.BUSY_TEXT),
        ({"code": "not_found", "message": "No progress.json"}, 404, "no run to resume"),
        ({"code": "error", "message": "names no command"}, 409, "names no command"),
    ],
)
def test_resume_preview_turns_cli_errors_into_plain_answers(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, error: dict, status: int, detail: str
) -> None:
    progress_record(ui.projects_dir() / "stopped")
    seen: list[dict[str, object]] = []
    envelope = {"ok": False, "command": "resume", "schema": 1, "project": None}
    envelope |= {"outputs": {}, "warnings": [], "error": error}

    def run(args: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
        seen.append({"args": args, **kw})
        return subprocess.CompletedProcess(args, 1, "log line\n" + json.dumps(envelope) + "\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    res = client.get("/api/projects/stopped/resume")
    assert res.status_code == status and detail in res.json()["detail"]
    p = ui.projects_dir() / "stopped"
    assert seen[0]["args"] == ui.cutpoint_cmd("resume", str(p), "--dry-run", "--json")
    assert seen[0]["cwd"] == p.parent  # the projects folder, not the server's own


@pytest.mark.slow
def test_a_folder_busy_job_shows_the_plain_state_without_the_delete_advice() -> None:
    message = "Another cutpoint run (process 9) is using X. If no other run is active, delete L"
    envelope = {"ok": False, "command": "analyze", "schema": 1, "project": None, "outputs": {}}
    envelope |= {"warnings": [], "error": {"code": "folder_busy", "message": message}}
    script = (
        "import sys\n"
        "print('12:00:00 INFO starting', flush=True)\n"
        f"print('12:00:01 ERROR ' + {message!r}, file=sys.stderr, flush=True)\n"
        f"print({json.dumps(envelope)!r}, flush=True)\n"
        "raise SystemExit(1)\n"
    )
    job = ui.Jobs().start("p", "analyze", [sys.executable, "-c", script])
    assert job.done.wait(30)
    view = job.view()
    assert (view["status"], view["error"], view["error_code"]) == (
        "failed",
        ui.BUSY_TEXT,
        "folder_busy",
    )
    assert not any("delete" in ln or '"schema"' in ln for ln in view["logs"]), view["logs"]
    assert view["logs"][0].endswith("INFO starting")


@pytest.mark.slow
def test_another_failure_keeps_its_error_line_and_reports_its_code() -> None:
    envelope = {"ok": False, "command": "clips", "schema": 1, "project": None, "outputs": {}}
    envelope |= {"warnings": [], "error": {"code": "not_found", "message": "boom"}}
    script = f"print('12:00:00 ERROR boom')\nprint({json.dumps(envelope)!r})\nraise SystemExit(1)"
    job = ui.Jobs().start("p", "clips", [sys.executable, "-c", script])
    assert job.done.wait(30)
    assert (job.status, job.error, job.error_code) == ("failed", "boom", "not_found")
    assert job.result is not None and job.result["error"]["code"] == "not_found"


@pytest.mark.slow
def test_a_failed_job_takes_its_error_from_the_envelope_before_any_error_line() -> None:
    """A270: a later log line that holds "ERROR " (transcript text, a path) does not replace
    the reason the command's --json envelope gives."""
    envelope = {"ok": False, "command": "clips", "schema": 1, "project": None, "outputs": {}}
    envelope |= {"warnings": [], "error": {"code": "error", "message": "the real reason"}}
    script = (
        "print('12:00:00 ERROR the real reason')\n"
        "print('12:00:01 INFO said: ERROR 404 in the transcript')\n"
        f"print({json.dumps(envelope)!r})\nraise SystemExit(1)"
    )
    job = ui.Jobs().start("p", "clips", [sys.executable, "-c", script])
    assert job.done.wait(30)
    assert (job.status, job.error) == ("failed", "the real reason")


@pytest.mark.parametrize(
    ("envelope_error", "expected"),
    [
        ({"code": "error", "message": "from the envelope"}, "from the envelope"),
        # an ffmpeg failure's message goes on with its stderr: its first line only
        (
            {"code": "error", "message": "ffmpeg failed (exit 1):\nline a\nline b"},
            "ffmpeg failed (exit 1):",
        ),
        ({"code": "error", "message": "\n  \nfrom the envelope\nmore"}, "from the envelope"),
        ({"code": "error", "message": " \n "}, "last line"),  # only blanks: the log's line
        ({"code": "error", "message": ""}, "last line"),  # nothing to say: the log's line
        ("not an object", "last line"),
        (None, "last line"),  # stopped before its envelope
    ],
)
def test_a_job_error_text_prefers_the_envelope_message(envelope_error: Any, expected: str) -> None:
    job = ui.Job("p", "clips")
    job.add_output("12:00:00 ERROR first line")
    job.add_output("12:00:01 ERROR last line")
    if envelope_error is not None:
        job.result = {"error": envelope_error}
    assert ui.Jobs._error_text(job, 0) == expected
    assert ui.Jobs._error_text(job, job.logged) == (
        None if expected == "last line" else expected
    )  # lines before the mark belong to the step before


@pytest.mark.slow
def test_jobs_run_in_the_folder_they_are_given(tmp_path: Path) -> None:
    job = ui.Jobs().start(
        "p", "analyze", [sys.executable, "-c", "import os; print(os.getcwd())"], cwd=tmp_path
    )
    assert job.done.wait(30) and job.status == "completed", job.error
    assert Path(job.logs[-1]).resolve() == tmp_path.resolve()


def test_every_project_job_runs_from_the_projects_folder_with_json(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, seg = cli_project(tmp_path)
    cache.save_json(
        p / "highlights.json", {**cache.load_json(p / "highlights.json"), "style": "funny"}
    )
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    seen = JobSpy(monkeypatch, check=False).calls
    for action in ("analyze", "clips"):
        assert client.post("/api/projects/cli/jobs", json={"action": action}).status_code == 200
    assert [s["cwd"] for s in seen] == [ui.projects_dir(), ui.projects_dir()]
    for s in seen:  # --json last, and the CLI takes it there
        assert s["raw"][-1] == "--json"
        cli.build_parser().parse_args(s["raw"][3:])


def test_the_resume_job_runs_cutpoint_resume_on_the_project(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = ui.projects_dir() / "stopped"
    progress_record(p)
    res, seen = resume_job(client, monkeypatch, "stopped")
    assert res.status_code == 200, res.text
    assert seen[0]["action"] == "resume"
    assert seen[0]["raw"] == [*ui.cutpoint_cmd("resume", str(p)), "--json"]
    assert seen[0]["cwd"] == ui.projects_dir()


@pytest.mark.parametrize("status", ["done", "running", None])
def test_the_resume_job_is_refused_without_a_stopped_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str | None
) -> None:
    p, _ = cli_project(tmp_path)
    if status:
        progress_record(p, status=status)
    res, seen = resume_job(client, monkeypatch, "cli")
    assert res.status_code == 409 and seen == []


def test_a_resumed_export_becomes_the_latest_export(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, _ = cli_project(tmp_path)
    out = p / "clips_ui_20260101-000000"
    argv = ["clips", str(p), "--segments", "s.json", "--out", str(out), "--vertical", "--json"]
    progress_record(p, command="clips", argv=argv)
    res, seen = resume_job(client, monkeypatch, "cli")
    assert res.status_code == 200, res.text
    on_result = seen[0]["on_result"]
    out.mkdir()
    cache.save_json(out / "manifest.json", {"clips": []})
    # a record that had finished meanwhile resumes nothing: nothing is listed
    on_result(SimpleNamespace(result={"outputs": {"resumed": False, "clips_dir": str(out)}}))
    assert not (p / "ui" / "exports.json").exists()
    on_result(SimpleNamespace(result={"outputs": {"resumed": True, "clips_dir": str(out)}}))
    items = cache.load_json(p / "ui" / "exports.json")["exports"]
    assert [(e["dir"], e["vertical"], e["final"]) for e in items] == [(out.name, True, None)]
    # a clips folder outside the project is never listed
    other = tmp_path / "elsewhere"
    other.mkdir()
    cache.save_json(other / "manifest.json", {"clips": []})
    on_result(SimpleNamespace(result={"outputs": {"resumed": True, "clips_dir": str(other)}}))
    assert len(cache.load_json(p / "ui" / "exports.json")["exports"]) == 1


def test_a_resumed_edit_lists_its_clips_and_final_and_a_resumed_analysis_keeps_its_skip(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, _ = cli_project(tmp_path)
    argv = ["edit", "clip.mp4", "--kind", "montage", "--out", str(p), "--skip", "crashes"]
    progress_record(p, command="edit", argv=[*argv, "keyframes", "--json"])
    (p / "clips").mkdir()
    cache.save_json(p / "clips" / "manifest.json", {"clips": []})
    (p / "final.mp4").write_bytes(b"x")
    res, seen = resume_job(client, monkeypatch, "cli")
    assert res.status_code == 200, res.text
    outputs = {"resumed": True, "clips_dir": str(p / "clips"), "final": str(p / "final.mp4")}
    seen[0]["on_result"](SimpleNamespace(result={"outputs": outputs}))
    items = cache.load_json(p / "ui" / "exports.json")["exports"]
    assert [(e["dir"], e["vertical"], e["final"]) for e in items] == [("clips", False, "final.mp4")]
    assert cache.load_json(p / ui.ANALYSIS_FILE)["skip"] == ["keyframes", "crashes"]


def test_record_export_replaces_appends_and_updates_as_the_jobs_wrote_it(tmp_path: Path) -> None:
    """record_export: ``replace`` drops the earlier entry of the folder and appends it as the
    latest (edit, a resumed run), ``append`` keeps a duplicate (clips, extend), ``update`` sets
    fields in place (assemble); the file is what cache.save_json writes."""
    from cutpoint.ui.projects import record_export as record  # noqa: PLC0415

    p = tmp_path / "p"
    (p / "ui").mkdir(parents=True)
    old = {"dir": "clips", "created": "1", "vertical": False, "final": "final.mp4"}
    other = {"dir": "clips_ui_1", "created": "2", "vertical": False, "final": None}
    cache.save_json(p / "ui" / "exports.json", {"exports": [old, other, {"dir": "../x"}]})
    record(p, "clips", created="3", vertical=True, final="final_vertical.mp4")
    new = {"dir": "clips", "created": "3", "vertical": True, "final": "final_vertical.mp4"}
    assert cache.load_json(p / "ui" / "exports.json")["exports"] == [other, new]
    record(p, "clips_ui_1", "append", created="4", vertical=False, final=None)
    again = {"dir": "clips_ui_1", "created": "4", "vertical": False, "final": None}
    record(p, "clips_ui_1", "update", final="final.mp4", final_joins=[])
    done = [{**e, "final": "final.mp4", "final_joins": []} for e in (other, again)]
    want = {"exports": [done[0], new, done[1]]}
    raw = (p / "ui" / "exports.json").read_bytes()
    assert raw == cache.json_bytes(want)
    assert list(json.loads(raw)["exports"][0]) == [
        "dir",
        "created",
        "vertical",
        "final",
        "final_joins",
    ]


@pytest.mark.slow
def test_a_project_another_process_holds_reads_busy(client: TestClient, tmp_path: Path) -> None:
    p, _ = cli_project(tmp_path)
    assert client.get("/api/projects/cli").json()["busy"] is False
    lock = folders.lock_path(p / "analysis.run")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        lock.write_text(json.dumps({"pid": child.pid, "host": socket.gethostname()}))
        assert client.get("/api/projects/cli").json()["busy"] is True
    finally:
        child.kill()
        child.wait()
    assert client.get("/api/projects/cli").json()["busy"] is False  # its process is gone
    lock.unlink()
    progress_record(p, status="running")  # a run on another host still going
    assert client.get("/api/projects/cli").json()["busy"] is True


@pytest.mark.slow
def test_a_clips_job_reports_the_clips_it_took_over() -> None:
    def job(outputs: dict[str, object]) -> dict[str, object]:
        envelope = {"ok": True, "command": "clips", "schema": 1, "project": None}
        envelope |= {"outputs": outputs, "warnings": [], "error": None}
        j = ui.Jobs().start(
            "p", "clips", [sys.executable, "-c", f"print({json.dumps(envelope)!r})"]
        )
        assert j.done.wait(30) and j.status == "completed", j.error
        return j.view()

    assert job({"taken_over": 2})["taken_over"] == 2
    assert "taken_over" not in job({"taken_over": 0})
    assert "taken_over" not in job({})
    assert "taken_over" not in job({"taken_over": True})
    assert "taken_over" not in job({"taken_over": 2.0})


@pytest.mark.slow
def test_jobs_never_import_a_module_from_the_projects_folder(tmp_path: Path) -> None:
    """``python -m cutpoint`` runs there: PYTHONSAFEPATH keeps a stray json.py out of reach."""
    (tmp_path / "json.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
    assert ui.job_env()["PYTHONSAFEPATH"] == "1"
    job = ui.Jobs().start(
        "p", "analyze", [sys.executable, "-c", "import json; print('real json')"], cwd=tmp_path
    )
    assert job.done.wait(30) and job.status == "completed", (job.error, list(job.logs))
    assert list(job.logs) == ["real json"]


def test_a_finished_record_alone_is_no_project(client: TestClient) -> None:
    progress_record(ui.projects_dir() / "finished", status="done")
    assert client.get("/api/projects").json()["projects"] == []
    assert client.get("/api/projects/finished").status_code == 404


def test_resume_preview_returns_the_command_line_it_replays(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress_record(ui.projects_dir() / "stopped")
    argv = ["clips", "p", "--force", "--out", "D:/elsewhere"]
    envelope = {"ok": True, "command": "clips", "schema": 1, "project": None, "warnings": []}
    envelope |= {"error": None, "outputs": {"resume": {"argv": argv, "status_before": "x"}}}
    out = json.dumps(envelope) + "\n"
    monkeypatch.setattr(
        subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(args, 0, out, "")
    )
    assert client.get("/api/projects/stopped/resume").json()["argv"] == argv
    envelope["outputs"]["resume"]["argv"] = ["clips", 3]  # never anything but strings
    out = json.dumps(envelope) + "\n"
    assert client.get("/api/projects/stopped/resume").json()["argv"] == []


def test_resuming_an_export_with_a_ui_file_in_the_way_is_refused_before_it_starts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, _ = cli_project(tmp_path)
    progress_record(p, command="clips")
    (p / "ui").write_text("not a folder", encoding="utf-8")
    res, _ = resume_job(client, monkeypatch, "cli")
    assert res.status_code == 409 and "is not a folder" in res.json()["detail"]


@pytest.mark.parametrize("command", ["analyze", "edit"])
@pytest.mark.parametrize("skip", [True, False])
def test_resuming_an_analysis_with_skip_and_a_ui_file_in_the_way_is_refused_before_it_starts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: str, skip: bool
) -> None:
    """resumed() saves a recorded --skip into ui/analysis.json: a ui file refuses the job first."""
    p, _ = cli_project(tmp_path)
    argv = [command, "clip.mp4", "--out", str(p), *(["--skip", "reactions"] if skip else [])]
    progress_record(p, command=command, argv=argv)
    (p / "ui").write_text("not a folder", encoding="utf-8")
    res, _ = resume_job(client, monkeypatch, "cli")
    if skip or command == "edit":  # edit registers its export in ui/exports.json too
        assert res.status_code == 409 and "is not a folder" in res.json()["detail"]
    else:  # nothing goes into ui: the analysis may resume
        assert res.status_code == 200, res.text


def test_resuming_an_analysis_with_skip_makes_the_ui_folder_first(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, _ = cli_project(tmp_path)
    argv = ["analyze", "clip.mp4", "--out", str(p), "--skip", "reactions"]
    progress_record(p, argv=argv)
    res, _ = resume_job(client, monkeypatch, "cli")
    assert res.status_code == 200, res.text
    assert (p / "ui").is_dir()


@pytest.mark.parametrize(
    ("argv", "seg", "picks"),
    [
        (["clips", "p", "--segments", "ui/s.json", "--out", "o"], "ui/s.json", {}),
        (
            ["clips", "p", "--top", "3", "--max-total", "60", "--out", "o"],
            None,
            {"top": 3, "ids": None, "max_total": 60.0},
        ),
        (["clips", "p", "--ids", "h001", "h004", "--json"], None, {"ids": ["h001", "h004"]}),
    ],
)
def test_a_resumed_360_export_stitches_what_its_command_line_exports(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argv: list[str],
    seg: str | None,
    picks: dict[str, object],
) -> None:
    p, _ = cli_project(tmp_path)
    (p / "ui").mkdir()
    cache.save_json(p / "ui" / "meta.json", {"name": "cli", "kind": "360"})
    progress_record(p, command="clips", argv=argv)
    stitched: list[tuple[object, ...]] = []
    patch_ui(monkeypatch, "stitch_first", lambda job, *a: stitched.append(a) and None)
    res, seen = resume_job(client, monkeypatch, "cli")
    assert res.status_code == 200, res.text
    before = seen[0]["before"]  # Jobs.start(..., plan, before)
    assert before(SimpleNamespace()) is None
    project, seg_file, chosen = stitched[0]
    assert project == p
    assert seg_file == (None if seg is None else p.parent / seg)
    assert chosen == ({"top": None, "ids": None, "max_total": None} | picks if seg is None else {})


def stopped_export(p: Path, name: str, *, marker: bool = True) -> Path:
    """The hidden resume folder a stopped export of the clips folder ``name`` leaves."""
    res = p / f".{name}.cutpoint-resume"
    res.mkdir(parents=True)
    (res / "001_a.mp4").write_bytes(b"x")
    if marker:
        cache.save_json(res / "resume.json", {"version": 1, "clips": {}})
    return res


def test_a_later_export_removes_the_resume_folders_that_can_never_be_resumed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = tmp_path / "proj"
    old = stopped_export(p, "clips_ui_20260101-101010")
    older = stopped_export(p, "clips_ui_20260101-090909-2")
    current = stopped_export(p, "clips_ui_20260102-101010")
    progress_record(
        p, "clips", argv=["clips", str(p), "--out", str(p / "clips_ui_20260102-101010")]
    )
    with caplog.at_level("INFO", logger=ui.logger.name):
        ui.prune_resume_folders(p)
    assert not old.exists() and not older.exists()
    assert current.is_dir()  # Resume still continues it
    assert caplog.text.count("Removed") == 2


def test_every_resume_folder_goes_once_the_last_run_is_done(tmp_path: Path) -> None:
    p = tmp_path / "proj"
    old = stopped_export(p, "clips_ui_20260101-101010")
    progress_record(
        p,
        "clips",
        status="done",
        argv=["clips", str(p), "--out", str(old.parent / "clips_ui_20260101-101010")],
    )
    ui.prune_resume_folders(p)
    assert not old.exists()


def test_a_resume_folder_of_a_locked_clips_folder_or_project_stays(tmp_path: Path) -> None:
    p = tmp_path / "proj"
    busy = stopped_export(p, "clips_ui_20260101-101010")
    free = stopped_export(p, "clips_ui_20260101-111111")
    with folders.swap_lock(p / "clips_ui_20260101-101010"):
        ui.prune_resume_folders(p)
        assert busy.is_dir() and not free.exists()
        free = stopped_export(p, "clips_ui_20260101-111111")
    with folders.swap_lock(p / runstate.LOCK_NAME):
        ui.prune_resume_folders(p)
    assert free.is_dir()


def test_folders_cutpoint_did_not_write_are_never_removed(tmp_path: Path) -> None:
    p = tmp_path / "proj"
    keep = [
        stopped_export(p, "clips_ui_20260101-101010", marker=False),  # no resume.json
        stopped_export(p, "my_clips"),  # not a UI export folder
        stopped_export(p, "clips_ui_notes"),
    ]
    (p / ".clips_ui_20260101-121212.cutpoint-resume").write_text("a file", encoding="utf-8")
    (p / "clips_ui_20260101-131313").mkdir()  # the clips folder itself
    ui.prune_resume_folders(p)
    assert all(f.is_dir() for f in keep)
    assert (p / ".clips_ui_20260101-121212.cutpoint-resume").is_file()
    assert (p / "clips_ui_20260101-131313").is_dir()


@pytest.mark.slow
def test_the_ui_prunes_when_an_export_job_ends(tmp_path: Path) -> None:
    p = tmp_path / "proj"
    old = stopped_export(p, "clips_ui_20260101-101010")
    jobs = ui.Jobs()
    job = jobs.start(
        "proj",
        "clips",
        [sys.executable, "-c", "pass"],
        on_end=lambda: ui.prune_resume_folders(p),
    )
    assert job.done.wait(30)
    assert not old.exists()


def test_a_linked_resume_folder_is_left_alone(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = tmp_path / "proj"
    p.mkdir()
    target = stopped_export(tmp_path / "elsewhere", "target")
    link = p / ".clips_ui_20260101-101010.cutpoint-resume"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted")
    with caplog.at_level("INFO", logger=ui.logger.name):
        ui.prune_resume_folders(p)
    assert link.is_symlink() and (target / "resume.json").is_file()
    assert "WARNING" not in caplog.text


def test_an_unreadable_progress_record_keeps_every_resume_folder(tmp_path: Path) -> None:
    p = tmp_path / "proj"
    keep = stopped_export(p, "clips_ui_20260101-101010")
    (p / "progress.json").write_text("{", encoding="utf-8")
    ui.prune_resume_folders(p)
    assert keep.is_dir()


@pytest.mark.slow
def test_a_failed_job_still_ends_with_on_end_and_a_failing_on_end_changes_nothing() -> None:
    calls: list[int] = []
    jobs = ui.Jobs()
    job = jobs.start(
        "proj",
        "clips",
        [sys.executable, "-c", "raise SystemExit(3)"],
        on_end=lambda: calls.append(1),
    )
    assert job.done.wait(30) and calls == [1]

    def boom() -> None:
        raise RuntimeError("cleanup")

    job = jobs.start("proj", "clips", [sys.executable, "-c", "raise SystemExit(3)"], on_end=boom)
    assert job.done.wait(30)
    assert job.status == "failed" and job.error == "exit code 3"
