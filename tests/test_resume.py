"""``cutpoint resume <project>`` (#109 phase 3): replays the run recorded in progress.json."""

from __future__ import annotations

import json
import socket
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

import cutpoint.analyze
from conftest import envelope, write
from cutpoint import cache, cli, folders, procs, progress, runstate
from cutpoint.errors import FolderBusyError
from cutpoint.ffmpeg_utils import MediaInfo

LOCK = ".analysis.run.cutpoint-lock"


@pytest.fixture
def project(home: Path, tmp_path: Path, monkeypatch) -> Path:
    """A project of ``videos/a.mp4``, run from ``tmp_path`` (the recorded cwd)."""
    monkeypatch.chdir(tmp_path)
    video = write(tmp_path / "videos" / "a.mp4")
    p = home / "projects" / "p"
    p.mkdir(parents=True)
    cache.save_json(p / "highlights.json", {"sources": [{"path": str(video)}]})
    return p


def _read(project: Path) -> dict:
    return json.loads((project / runstate.FILE).read_text(encoding="utf-8"))


def _record(project: Path, **over: Any) -> dict:
    """A progress.json of an interrupted ``events p`` run, as runstate writes it."""
    rec = runstate.Recorder("events", ["events", "p"], project, {})
    rec.data.update({"status": "interrupted", "exit_code": 130, "options_key": _events_key()})
    rec.data.update(over)
    cache.save_json(project / runstate.FILE, rec.data)
    return rec.data


def _events_key(*argv: str) -> str:
    return runstate.options_key(vars(cli._parse(cli.build_parser(), ["events", "p", *argv])))


def _fake_events(monkeypatch, body=None) -> list[dict]:
    """``cutpoint events`` as a fake with one progress step; returns what each call saw
    (cwd, its argv in progress.json, the parsed namespace)."""
    calls: list[dict] = []

    def cmd(a):
        project = runstate._ACTIVE.project
        calls.append({"cwd": Path.cwd(), "record": _read(project), "args": a})
        with progress.run(("vlm",)) as steps, steps.step("vlm"):
            if body is not None:
                body(project)

    monkeypatch.setattr(cli, "_cmd_events", cmd)
    return calls


def _resume(capsys, *argv: str) -> tuple[int, dict]:
    code = cli.main(["resume", "p", *argv, "--json"])
    return code, envelope(capsys.readouterr().out)


# -- the replay -------------------------------------------------------------------------------


def test_resume_replays_the_recorded_command_from_its_cwd(project, tmp_path, monkeypatch, capsys):
    calls = _fake_events(monkeypatch)
    _record(project, steps={"vlm": {"state": "failed", "done": 3, "total": 8, "pct": 40}})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    code, doc = _resume(capsys)
    assert code == 0 and doc["ok"] and doc["command"] == "events"  # the replayed command
    assert doc["project"] == str(project)
    assert doc["outputs"]["resumed"] is True
    assert doc["outputs"]["resume"] == {
        "argv": ["events", "p"],
        "status_before": "interrupted",
        "done_before": {"vlm": {"state": "failed", "done": 3, "total": 8}},
    }
    (call,) = calls
    assert call["cwd"] == tmp_path and Path.cwd() == elsewhere  # cwd restored
    # the replayed run records as the command it replays, not as resume
    assert call["record"]["argv"] == ["events", "p"] and call["record"]["command"] == "events"
    rec = _read(project)
    assert rec["status"] == "done" and rec["command"] == "events"
    assert rec["options_key"] == _events_key()
    assert rec["steps"]["vlm"]["state"] == "done"
    assert not (project / LOCK).exists()


def test_resume_replays_a_recorded_force_and_json(project, monkeypatch, capsys):
    """resume replays the command line as recorded (its --force too); --json is resume's."""
    calls = _fake_events(monkeypatch)
    _record(project, argv=["-v", "events", "p", "--force", "vlm", "--json"])
    assert cli.main(["resume", "p"]) == 0
    assert capsys.readouterr().out == ""  # no --json on resume: no envelope
    (call,) = calls
    assert call["args"].force == ["vlm"] and call["args"].json_result is False
    assert _read(project)["argv"] == ["-v", "events", "p", "--force", "vlm", "--json"]


def test_resume_has_no_force(project, monkeypatch, capsys):
    calls = _fake_events(monkeypatch)
    _record(project)
    with pytest.raises(SystemExit) as exc:
        cli.main(["resume", "p", "--force", "--json"])
    assert exc.value.code == 2
    doc = envelope(capsys.readouterr().out)
    assert doc["command"] == "resume" and doc["error"]["code"] == "bad_args"
    assert calls == [] and _read(project)["status"] == "interrupted"


def test_a_failing_replay_keeps_its_exit_code_and_error(project, monkeypatch, capsys):
    def body(project):
        raise KeyboardInterrupt

    _fake_events(monkeypatch, body)
    _record(project)
    code, doc = _resume(capsys)
    assert code == 130 and doc["error"]["code"] == "interrupted"
    assert doc["command"] == "events" and doc["outputs"]["resumed"] is True
    assert _read(project)["status"] == "interrupted"  # resumable again


# -- nothing to resume ------------------------------------------------------------------------


def test_a_finished_run_is_not_resumed(project, monkeypatch, capsys):
    calls = _fake_events(monkeypatch)
    _record(project, status="done", exit_code=0, error=None)
    code, doc = _resume(capsys)
    assert code == 0 and doc["ok"] and doc["command"] == "events"
    assert doc["outputs"]["resumed"] is False
    assert doc["outputs"]["resume"]["status_before"] == "done"
    assert calls == []


@pytest.mark.parametrize("content", [None, "not json"], ids=["missing", "unreadable"])
def test_no_record_is_not_found(project, monkeypatch, capsys, content):
    calls = _fake_events(monkeypatch)
    if content is not None:
        (project / runstate.FILE).write_text(content, encoding="utf-8")
    code, doc = _resume(capsys)
    assert code == 1 and doc["command"] == "resume"
    assert doc["error"]["code"] == "not_found" and runstate.FILE in doc["error"]["message"]
    assert calls == []


def test_a_missing_recorded_cwd_is_not_found_and_not_resumed(
    project, tmp_path, monkeypatch, capsys
):
    calls = _fake_events(monkeypatch)
    _record(project, cwd=str(tmp_path / "gone"))
    code, doc = _resume(capsys)
    assert code == 1 and doc["error"]["code"] == "not_found"
    assert doc["outputs"]["resumed"] is False and calls == []


@pytest.mark.parametrize("json_flag", [True, False], ids=["json", "plain"])
def test_a_copied_project_does_not_replay_on_the_original(
    project, monkeypatch, capsys, caplog, json_flag
):
    """After copying p to p2, p2's record names p: resume p2 stops instead of running on p."""
    calls = _fake_events(monkeypatch)
    _record(project, argv=["events", "p", "--force", "vlm"])
    copy = project.parent / "p2"
    copy.mkdir()
    (copy / runstate.FILE).write_bytes((project / runstate.FILE).read_bytes())
    code = cli.main(["resume", "p2", *(["--json"] if json_flag else [])])
    out = capsys.readouterr().out
    assert code == 1 and calls == []
    if json_flag:
        doc = envelope(out)
        assert doc["error"]["code"] == "error"
        assert f"records a run on {project}" in doc["error"]["message"]
        assert doc["outputs"]["resumed"] is False
    else:
        assert out == ""
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert errors == [f"{copy / runstate.FILE} records a run on {project}"]
    assert not any("Resuming in" in r.getMessage() for r in caplog.records)
    assert _read(project)["status"] == "interrupted" and not (project / LOCK).exists()


def test_resume_of_a_relative_project_path_from_the_projects_folder(project, monkeypatch, capsys):
    """``resume p`` inside the projects folder names the same project: it is replayed."""
    calls = _fake_events(monkeypatch)
    _record(project)
    monkeypatch.chdir(project.parent)
    code, doc = _resume(capsys)
    assert code == 0 and doc["outputs"]["resumed"] is True
    assert doc["project"] == str(project)
    assert len(calls) == 1 and Path.cwd() == project.parent


def test_a_relative_copy_does_not_replay_on_the_original(project, tmp_path, monkeypatch, capsys):
    """A copy ``./p`` of a project whose run was recorded from the projects folder: its
    record's ``p`` is the original there, so resume stops instead of running on it."""
    calls = _fake_events(monkeypatch)
    _record(project, cwd=str(project.parent))
    elsewhere = tmp_path / "elsewhere"
    copy = elsewhere / "p"
    copy.mkdir(parents=True)
    (copy / runstate.FILE).write_bytes((project / runstate.FILE).read_bytes())
    monkeypatch.chdir(elsewhere)
    code, doc = _resume(capsys)
    assert code == 1 and doc["error"]["code"] == "error"
    assert f"records a run on {project}" in doc["error"]["message"]
    assert doc["project"] == str(copy)
    assert doc["outputs"]["resumed"] is False and calls == []
    assert _read(project)["status"] == "interrupted" and not (project / LOCK).exists()


def test_a_missing_project_is_not_found(home, capsys):
    code, doc = _resume(capsys)
    assert code == 1 and doc["error"]["code"] == "not_found" and doc["command"] == "resume"
    assert not (home / "projects" / "p").exists()


def test_a_live_run_is_folder_busy(project, monkeypatch, capsys):
    """A record still running (its process alive), or a live lock: folder_busy, nothing run."""
    calls = _fake_events(monkeypatch)
    monkeypatch.setattr(procs, "pid_alive", lambda pid, not_after=None: True)
    _record(project, status="running", pid=4242, exit_code=None)
    code, doc = _resume(capsys)
    assert code == 1 and doc["error"]["code"] == "folder_busy" and doc["command"] == "events"
    assert "4242" in doc["error"]["message"] and str(project) in doc["error"]["message"]
    # an interrupted record, but another run holds the project's lock
    _record(project)
    lock = project / LOCK
    lock.write_text(
        json.dumps({"pid": 4242, "host": socket.gethostname(), "started": "now"}),
        encoding="utf-8",
    )
    code, doc = _resume(capsys)
    assert code == 1 and doc["error"]["code"] == "folder_busy"
    assert calls == [] and lock.exists() and _read(project)["status"] == "interrupted"


def test_a_running_record_of_a_dead_process_is_resumed(project, monkeypatch, capsys):
    calls = _fake_events(monkeypatch)
    monkeypatch.setattr(procs, "pid_alive", lambda pid, not_after=None: False)
    _record(project, status="running", pid=4242, exit_code=None)
    code, doc = _resume(capsys)
    assert code == 0 and doc["outputs"]["resume"]["status_before"] == "interrupted"
    assert len(calls) == 1


# -- --dry-run --------------------------------------------------------------------------------


def test_dry_run_prints_the_command_and_counts(project, monkeypatch, capsys):
    calls = _fake_events(monkeypatch)
    steps = {
        "transcribe": {"state": "failed", "done": 1, "total": 3, "pct": 40},
        "reactions": {"state": "pending", "done": None, "total": None, "pct": None},
    }
    _record(project, argv=["events", "p", "--window", "12"], steps=steps)
    assert cli.main(["resume", "p", "--dry-run"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "cutpoint events p --window 12"
    assert "interrupted" in out[1] and str(Path.cwd()) in out[1]
    assert out[2:] == ["  transcribe: failed, 1/3 done", "  reactions: pending"]
    code, doc = _resume(capsys, "--dry-run")
    assert code == 0 and doc["command"] == "events" and doc["outputs"]["resumed"] is False
    assert doc["outputs"]["resume"]["done_before"]["transcribe"] == {
        "state": "failed",
        "done": 1,
        "total": 3,
    }
    assert calls == [] and _read(project)["status"] == "interrupted"


# -- warnings ---------------------------------------------------------------------------------


def test_resume_warns_about_sources_that_are_gone(project, tmp_path, monkeypatch, capsys):
    calls = _fake_events(monkeypatch)
    there = tmp_path / "videos" / "a.mp4"
    gone = tmp_path / "videos" / "b.mp4"
    _record(project, sources=[str(there), str(gone), "videos/c.mp4", "videos/a.mp4"])
    monkeypatch.chdir(tmp_path / "videos")  # relative sources are from the recorded cwd
    code, doc = _resume(capsys)
    assert code == 0 and len(calls) == 1  # the run goes on (events skips unreachable footage)
    (w,) = [w for w in doc["warnings"] if w["code"] == "resume_sources_changed"]
    assert w["files"] == [str(gone), "videos/c.mp4"]


def test_a_rerun_with_other_options_warns_resume_args_changed(project, monkeypatch, capsys, caplog):
    _fake_events(monkeypatch)
    _record(project)
    with caplog.at_level("INFO"):
        assert cli.main(["events", "p", "--window", "12", "--json"]) == 0
    doc = envelope(capsys.readouterr().out)
    (w,) = [w for w in doc["warnings"] if w["code"] == "resume_args_changed"]
    assert w["files"] == [str(project / runstate.FILE)]
    assert any("this run with other options replaces it" in m for m in caplog.messages)
    assert _read(project)["argv"] == ["events", "p", "--window", "12", "--json"]


def test_a_run_of_another_command_names_it_in_resume_args_changed(project, monkeypatch, capsys):
    _fake_events(monkeypatch)
    _record(project, command="analyze", argv=["analyze", "videos/a.mp4", "--out", "p"])
    assert cli.main(["events", "p", "--json"]) == 0
    doc = envelope(capsys.readouterr().out)
    (w,) = [w for w in doc["warnings"] if w["code"] == "resume_args_changed"]
    assert "interrupted 'analyze' run; this 'events' run replaces it" in w["message"]


@pytest.mark.parametrize(
    ("status", "argv"),
    [("interrupted", ["events", "p"]), ("done", ["events", "p", "--window", "12"])],
    ids=["same-options", "finished-record"],
)
def test_no_resume_args_changed_otherwise(project, monkeypatch, capsys, status, argv):
    """The same options continue the record; a finished record is no loss."""
    _fake_events(monkeypatch)
    _record(project, status=status)
    assert cli.main([*argv, "--json"]) == 0
    assert "resume_args_changed" not in [
        w["code"] for w in envelope(capsys.readouterr().out)["warnings"]
    ]


# -- Windows review of #702 -------------------------------------------------------------------


def test_folder_busy_names_the_project_not_the_lock_subject(project, monkeypatch):
    """W-F1: the message names the project (and the lock file), not <project>/analysis.run."""
    lock = project / LOCK
    lock.write_text(
        json.dumps({"pid": 4242, "host": socket.gethostname(), "started": "now"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(procs, "pid_alive", lambda pid, not_after=None: True)
    with pytest.raises(FolderBusyError) as exc, runstate.record("events", ["events"], project):
        pass
    message = str(exc.value)
    assert f"is using {project}. " in message and str(lock) in message
    with pytest.raises(FolderBusyError) as exc, folders.swap_lock(project / runstate.LOCK_NAME):
        pass
    assert f"is using {project / runstate.LOCK_NAME}." in str(exc.value)  # default: dst


@pytest.mark.parametrize("command", ["events", "crashes", "profile"])
def test_a_record_alone_is_no_project_for_the_later_commands(home, monkeypatch, capsys, command):
    """W-F2: events, crashes or profile on a folder with only the progress.json of an
    interrupted first analyze stop with not_found and leave that record as it was."""
    project = home / "projects" / "p"
    project.mkdir(parents=True)
    rec = runstate.Recorder("analyze", ["analyze", "/videos", "--out", "p"], project, {})
    rec.data.update(status="interrupted", sources=["/videos/a.mp4"])
    cache.save_json(project / runstate.FILE, rec.data)
    before = (project / runstate.FILE).read_bytes()
    for name in ("_cmd_events", "_cmd_crashes", "_cmd_profile"):
        monkeypatch.setattr(cli, name, _no_project)
    assert cli.main([command, "p", "--json"]) == 1
    assert envelope(capsys.readouterr().out)["error"]["code"] == "not_found"
    assert (project / runstate.FILE).read_bytes() == before
    assert sorted(p.name for p in project.iterdir()) == [runstate.FILE]


def _no_project(a) -> None:
    raise FileNotFoundError("highlights.json not found")


# -- events: the VLM layer resumes to the events.json of an uninterrupted run -----------------


class _Ollama:
    """Fake Ollama whose reply names the window it was asked about (the frames carry the
    source and start, see ``_frames``); ``stop`` raises KeyboardInterrupt on that request."""

    def __init__(self) -> None:
        self.asked: list[str] = []
        self.stop: int | None = None

    def get(self, url: str, **_kw: Any) -> Any:
        return _Resp({"models": [{"name": "qwen3.5:9b", "digest": "d1"}]})

    def post(self, url: str, json: dict[str, Any], **_kw: Any) -> Any:
        if url.endswith("/api/generate"):  # unload
            return _Resp({})
        import base64  # noqa: PLC0415

        what = base64.b64decode(json["messages"][0]["images"][0]).decode()
        self.asked.append(what)
        if self.stop is not None and len(self.asked) == self.stop:
            raise KeyboardInterrupt
        from cutpoint.events import vlm_windows as vw  # noqa: PLC0415

        reply = {
            "events": [{"t": 2.0, "what": f"moment {what}", "funny": 1, "hype": 6}],
            "summary": f"window {what}",
            "funny_score": 2,
            "highlight_score": 6,
            "label": "Big Moment",
            "tags": dict.fromkeys(vw.SKI_TAGS, 2),
        }
        return _Resp({"message": {"content": json_dumps(reply)}, "prompt_eval_count": 99})


def json_dumps(data: Any) -> str:
    return json.dumps(data)


class _Resp:
    def __init__(self, body: Any) -> None:
        self.body, self.status_code, self.is_redirect = body, 200, False

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return self.body


@pytest.fixture
def vlm_project(home, tmp_path, monkeypatch):
    """Project ``p`` of two 300 s sources with three candidates each (six VLM windows), the
    events extra "installed", every layer but the VLM's faked; ``state.cache`` names the cache
    folder of the sources (one per run, so a reference run shares nothing)."""
    from types import SimpleNamespace  # noqa: PLC0415

    from cutpoint import vision  # noqa: PLC0415
    from cutpoint.events import common, pipeline  # noqa: PLC0415
    from cutpoint.events import vlm_windows as vw  # noqa: PLC0415
    from cutpoint.events.common import Source  # noqa: PLC0415

    monkeypatch.chdir(tmp_path)
    videos = [write(tmp_path / "videos" / n) for n in ("a.mp4", "b.mp4")]
    project = home / "projects" / "p"
    project.mkdir(parents=True)
    state = SimpleNamespace(project=project, cache="ref", ollama=_Ollama(), stop_frames=None)
    frames: list[str] = []

    def highlights(kind: str | None) -> None:
        cands = [
            {"id": f"h{i}{j}", "source": str(v), "start": t, "end": t + 4.0, "score": 60 + j}
            for i, v in enumerate(videos)
            for j, t in enumerate((20.0, 100.0, 200.0))
        ]
        hl = {"sources": [{"path": str(v), "duration": 300.0} for v in videos], "candidates": cands}
        cache.save_json(project / "highlights.json", hl | ({"style_kind": kind} if kind else {}))

    def load_source(path, dur):
        cdir = tmp_path / "cache" / state.cache / path.name
        cdir.mkdir(parents=True, exist_ok=True)
        return Source(path, 300.0, cdir, [])

    def grab(path, start, dur, n, height):
        frames.append(f"{Path(path).name}@{start:g}")
        if state.stop_frames is not None and len(frames) == state.stop_frames:
            raise KeyboardInterrupt  # between two windows: none of this one is done
        return [f"{Path(path).name}@{start:g}".encode()]

    monkeypatch.setattr(pipeline.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(pipeline, "load_source", load_source)
    monkeypatch.setattr(pipeline, "bad_ranges", lambda *a: {})
    monkeypatch.setattr(pipeline, "load_transcript_lines", lambda project: [])
    monkeypatch.setattr(pipeline, "_telemetry_layer", lambda *a: {})
    monkeypatch.setattr(common, "gpu_memory_used_mib", lambda: None)
    monkeypatch.setattr(vw, "grab_frames", grab)
    monkeypatch.setattr(vision.requests, "get", state.ollama.get)
    monkeypatch.setattr(vision.requests, "post", state.ollama.post)
    state.highlights, state.frames = highlights, frames
    return state


def _events_json(project: Path) -> dict:
    data = cache.load_json(project / "events.json")
    return {k: v for k, v in data.items() if k not in ("created", "timings", "stats")}


@pytest.mark.parametrize("kind", [None, "action-cam"], ids=["game", "action-cam-incremental"])
@pytest.mark.parametrize("where", ["boundary", "inside"])
def test_events_resumed_after_a_vlm_window_equal_an_uninterrupted_run(
    vlm_project, kind, where, capsys
):
    s = vlm_project
    s.highlights(kind)
    argv = ["events", "p", "--layers", "vlm", "--no-rescore"]
    assert cli.main(argv) == 0
    reference = _events_json(s.project)
    assert len(reference["windows"]) == 6 and len(s.ollama.asked) == 6
    for name in ("events.json", "events.txt"):
        (s.project / name).unlink()

    s.cache = "run"  # nothing cached: the same run again, stopped after two windows
    s.ollama.asked.clear()
    s.frames.clear()
    if where == "boundary":
        s.stop_frames = 3  # before the third window's frames
    else:
        s.ollama.stop = 3  # the third window's request is in flight
    assert cli.main(argv) == 130
    assert _read(s.project)["status"] == "interrupted"
    assert not (s.project / "events.json").exists()
    s.stop_frames = s.ollama.stop = None
    asked = len(s.ollama.asked)
    s.ollama.asked.clear()
    assert cli.main(["resume", "p"]) == 0
    assert len(s.ollama.asked) == 4  # the two described windows are not asked again
    assert asked == (2 if where == "boundary" else 3)
    assert _events_json(s.project) == reference  # the action-cam path: reasons kept too
    assert _read(s.project)["status"] == "done"


# -- analyze: interrupted while transcribing source 2 of 3 -----------------------------------


def test_analyze_resumed_mid_transcribe_transcribes_only_the_rest(home, tmp_path, monkeypatch):
    """The transcripts are the record of what is done: resume runs Whisper on the source it
    was stopped in and the one after, not on the first."""
    monkeypatch.chdir(tmp_path)
    videos = [write(tmp_path / "videos" / f"{n}.mp4") for n in "abc"]
    whisper: list[str] = []
    stop = {"at": 2}

    class Whisper:
        model_name, language, device = "fake", None, "cpu"

        def __init__(self, **_kw: Any) -> None:
            pass

        def transcribe(self, wav: Path, on_progress=None) -> list[dict]:
            whisper.append(wav.parent.name)
            if len(whisper) == stop["at"]:
                raise KeyboardInterrupt
            return [{"start": 1.0, "end": 2.0, "text": "hi", "words": []}]

        def close(self) -> None:
            pass

    def analyze_source(src, style, tr, use_scenes, max_cand, timings, **kw):
        cdir = cache.source_dir(src)
        segments = cutpoint.analyze._transcript(cdir / "audio16k.wav", cdir, tr)
        info = MediaInfo(
            src, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 1, None, None, None
        )
        cand = cutpoint.analyze.Candidate(src, 10.0, 20.0, 15.0, 1.0, {"loudness": 1.0}, 50.0)
        return cutpoint.analyze.SourceResult(src, info, 0.0, 60.0, [cand], segments, None)

    monkeypatch.setattr(cutpoint.analyze, "Transcriber", Whisper)
    monkeypatch.setattr(cutpoint.analyze, "analyze_source", analyze_source)
    monkeypatch.setattr(cutpoint.analyze, "plan_layers", lambda *a: cutpoint.analyze.LayerPlan())
    argv = ["analyze", "videos", "--style", "funny", "--no-vision", "--out", "p"]
    assert cli.main(argv) == 130
    project = home / "projects" / "p"
    rec = _read(project)
    assert rec["status"] == "interrupted" and rec["argv"] == argv
    assert rec["steps"]["transcribe"]["state"] == "failed"
    assert not (project / "highlights.json").exists()
    keys = [cache.source_dir(v).name for v in videos]
    assert whisper == keys[:2]
    whisper.clear()
    stop["at"] = 0
    monkeypatch.chdir(home)  # resume runs from the recorded cwd
    assert cli.main(["resume", "p"]) == 0
    assert whisper == keys[1:]  # the first source's transcript is cached
    assert Path.cwd() == home
    assert [s["path"] for s in cache.load_json(project / "highlights.json")["sources"]] == [
        str(v) for v in videos
    ]
    rec = _read(project)
    assert rec["status"] == "done" and rec["command"] == "analyze" and rec["argv"] == argv
    assert rec["sources"] == [str(v) for v in videos]


# -- edit: a resume after the clips step keeps the clips --------------------------------------


@pytest.fixture
def edit_run(project, tmp_path, monkeypatch):
    """``edit videos/a.mp4 --out p`` with analyze, clips and assemble faked; the first
    assemble is interrupted. Returns the calls."""
    import cutpoint.edit  # noqa: PLC0415

    calls: dict[str, list] = {"clips": [], "assemble": []}

    def export_clips(project, **kw):
        calls["clips"].append(kw)
        write(project / "clips" / "001_moment.mp4", b"clip")
        clip = {"file": "001_moment.mp4", "source": "videos/a.mp4", "start": 1.0, "end": 4.0}
        cache.save_json(project / "clips" / "manifest.json", {"clips": [clip]})

    def assemble(work, out, **kw):
        calls["assemble"].append(work)
        if len(calls["assemble"]) == 1:
            raise KeyboardInterrupt
        return write(project / "final.mp4", b"final")

    monkeypatch.setattr(cutpoint.analyze, "analyze", lambda *a, **k: None)
    monkeypatch.setattr(cutpoint.edit, "export_clips", export_clips)
    monkeypatch.setattr(cutpoint.edit, "assemble", assemble)
    assert cli.main(["edit", "videos/a.mp4", "--out", "p", "--no-vision"]) == 130
    rec = _read(project)
    assert rec["status"] == "interrupted" and rec["steps"]["clips"]["state"] == "done"
    assert rec["steps"]["clips"]["manifest"]["path"] == str(project / "clips" / "manifest.json")
    return calls


def test_edit_resume_skips_the_finished_clips(edit_run, project, capsys, caplog):
    with caplog.at_level("INFO"):
        code, doc = _resume(capsys)
    assert code == 0 and doc["command"] == "edit" and doc["outputs"]["resumed"] is True
    assert len(edit_run["clips"]) == 1 and len(edit_run["assemble"]) == 2  # clips not redone
    assert doc["outputs"]["clips_dir"] == str(project / "clips")  # the kept clips reported
    assert doc["outputs"]["count"] == 1
    assert any("clips of the interrupted run are kept" in m for m in caplog.messages)
    assert (project / "a-final.mp4").read_bytes() == b"final"
    rec = _read(project)
    assert rec["status"] == "done" and rec["steps"]["clips"]["state"] == "done"
    assert "manifest" in rec["steps"]["clips"]  # a later resume can keep them again


@pytest.mark.parametrize("how", ["manifest-changed", "plain-rerun"])
def test_edit_keeps_clips_only_on_resume_of_the_same_manifest(edit_run, project, capsys, how):
    """The clips folder stays protected: a manifest changed since the interrupted run, or a
    plain re-run (not resume), stops on the non-empty clips folder as before."""
    if how == "manifest-changed":
        cache.save_json(project / "clips" / "manifest.json", {"clips": [], "edited": True})
        code, doc = _resume(capsys)
    else:
        code = cli.main(["edit", "videos/a.mp4", "--out", "p", "--no-vision", "--json"])
        doc = envelope(capsys.readouterr().out)
    assert code == 1 and doc["error"]["code"] == "output_not_empty"
    assert len(edit_run["clips"]) == 1 and len(edit_run["assemble"]) == 1
