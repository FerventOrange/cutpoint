"""ffmpeg_utils: tools are never found in the current folder (A664); inputs open only local
files and pipes (A665)."""

from __future__ import annotations

import os
import stat

import pytest

from conftest import write
from cutpoint import ffmpeg_utils as fu

WL = ["-protocol_whitelist", fu.INPUT_PROTOCOLS]


def test_every_input_but_lavfi_is_limited_to_files_and_pipes() -> None:
    cmd = ["ffmpeg", "-y", "-ss", "1", "-i", "a.mp4", "-f", "lavfi", "-i", "anullsrc"]
    cmd += ["-f", "concat", "-safe", "0", "-i", "l.txt", "-f", "mp4", "out.mp4"]
    assert fu.limit_protocols(cmd) == [
        "ffmpeg", "-y", "-ss", "1", *WL, "-i", "a.mp4", "-f", "lavfi", "-i", "anullsrc",
        "-f", "concat", "-safe", "0", *WL, "-i", "l.txt", "-f", "mp4", "out.mp4",
    ]  # fmt: skip
    own = ["ffmpeg", "-protocol_whitelist", "file", "-i", "a.mp4", "o.mp4"]
    assert fu.limit_protocols(own) == own


def test_command_limits_the_protocols(monkeypatch) -> None:
    monkeypatch.setattr(fu, "find_tool", lambda name: name)
    assert fu._command("ffmpeg", ["-i", "a.mp4", "o.mp4"]) == [
        "ffmpeg",
        *WL,
        "-i",
        "a.mp4",
        "o.mp4",
    ]


def _exe(path):
    """An executable ``path`` (``.exe`` added on Windows, where ``shutil.which`` needs it)."""
    if os.name == "nt":
        path = path.with_name(path.name + ".exe")
    write(path, "#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def fresh_find_tool(monkeypatch):
    monkeypatch.delenv("CUTPOINT_FFMPEG", raising=False)
    fu.find_tool.cache_clear()
    yield
    fu.find_tool.cache_clear()


def test_a_tool_in_a_relative_path_folder_is_not_used(tmp_path, monkeypatch, fresh_find_tool):
    monkeypatch.chdir(tmp_path)
    _exe(tmp_path / "ffmpeg")  # planted in the current folder, which PATH names as "."
    real = _exe(tmp_path / "bin" / "ffmpeg")
    monkeypatch.setenv("PATH", os.pathsep.join([".", "", str(real.parent)]))
    assert os.path.normcase(fu._on_path("ffmpeg") or "") == os.path.normcase(str(real))
    monkeypatch.setenv("PATH", ".")
    assert fu._on_path("ffmpeg") is None


def test_no_localappdata_searches_no_winget_folder_below_the_current_one(
    tmp_path, monkeypatch, fresh_find_tool
):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("LOCALAPPDATA", "")
    monkeypatch.setattr(fu.sys, "platform", "win32")
    with pytest.raises(fu.FFmpegNotFoundError):
        fu.find_tool("ffmpeg")


def test_the_peak_tag_probe_names_its_input_with_i(monkeypatch, tmp_path) -> None:
    """Review F4: ffprobe's input is given with -i, so it gets the protocol whitelist."""
    from cutpoint import assemble  # noqa: PLC0415

    seen: list[list[str]] = []
    monkeypatch.setattr(fu, "find_tool", lambda name: name)
    monkeypatch.setattr(
        assemble, "run", lambda args: seen.append(fu._command(args[0], args[1:])) or ""
    )
    clip = tmp_path / "c.mp4"
    assemble._peak_tagged(clip)
    (cmd,) = seen
    assert cmd[-4:] == [*WL, "-i", str(clip)]


# --------------------------------------------------------------------------- A400


class _Proc:
    """A ``Popen`` stand-in: ``communicate`` answers ``outcome`` (raised when an exception)."""

    def __init__(self, cmd, outcome=(b"out", b"err"), returncode=0, **kwargs) -> None:
        self.cmd, self.kwargs, self.outcome, self.returncode = cmd, kwargs, outcome, returncode
        self.pid, self.killed, self.calls = 7, False, 0

    def communicate(self, timeout=None):
        self.calls += 1
        if self.calls == 1 and isinstance(self.outcome, BaseException):
            raise self.outcome
        return (b"", b"") if self.calls > 1 else self.outcome

    def kill(self) -> None:
        self.killed = True

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


def test_spawn_isolates_and_registers_the_child(monkeypatch) -> None:
    made, jobs = [], []

    def popen(cmd, **kw):
        made.append(_Proc(cmd, **kw))
        return made[-1]

    monkeypatch.setattr(fu.subprocess, "Popen", popen)
    monkeypatch.setattr(fu.procs, "kill_with_parent", jobs.append)
    scope = fu.ChildScope()
    proc = fu._spawn(["ffmpeg"], scope, isolate=True, stdout=fu.subprocess.PIPE)
    assert proc.kwargs == {"stdout": fu.subprocess.PIPE, **fu.procs.isolated_kwargs()}
    assert jobs == [proc] and proc in scope._procs
    plain = fu._spawn(["ffmpeg"], None, bufsize=4)
    assert plain.kwargs == {"bufsize": 4} and jobs == [proc]
    scope.cancel()
    with pytest.raises(KeyboardInterrupt):
        fu._spawn(["ffmpeg"], scope)
    assert len(made) == 2  # a cancelled scope starts no child


@pytest.mark.parametrize("step", ["job", "scope"])
def test_spawn_stops_the_child_when_a_ctrl_c_hits_right_after_the_start(monkeypatch, step):
    """Review F1: a KeyboardInterrupt in the job-object or scope step stops the child (it is
    waited for) and leaves it out of the scope."""
    made, stopped = [], []

    def popen(cmd, **kw):
        made.append(_Proc(cmd, **kw))
        return made[-1]

    def interrupt(*_a):
        raise KeyboardInterrupt

    monkeypatch.setattr(fu.subprocess, "Popen", popen)
    monkeypatch.setattr(fu.procs, "stop_child", lambda proc, err: stopped.append((proc, err)))
    scope = fu.ChildScope()
    if step == "job":
        monkeypatch.setattr(fu.procs, "kill_with_parent", interrupt)
    else:
        real_add = scope.add

        def add(proc):
            real_add(proc)
            raise KeyboardInterrupt

        monkeypatch.setattr(scope, "add", add)
    with pytest.raises(KeyboardInterrupt):
        fu._spawn(["ffmpeg"], scope, isolate=True)
    (proc,) = made
    assert [p for p, _ in stopped] == [proc] and isinstance(stopped[0][1], KeyboardInterrupt)
    assert proc not in scope._procs


def test_communicate_times_out_with_the_callers_words() -> None:
    scope = fu.ChildScope()
    proc = _Proc(["ffmpeg"], outcome=fu.subprocess.TimeoutExpired("ffmpeg", 5))
    scope.add(proc)
    with pytest.raises(fu.FFmpegError, match=r"^ffprobe timed out after 5s on x\.mp4$"):
        fu._communicate(proc, scope, 5, "ffprobe", " on x.mp4")
    assert proc.killed and proc.calls == 2 and proc not in scope._procs


def test_communicate_returns_the_output_and_leaves_the_exit_code_to_the_caller() -> None:
    proc = _Proc(["ffmpeg"], returncode=1)
    assert fu._communicate(proc, None, None, "ffmpeg") == (b"out", b"err")
    scope = fu.ChildScope()
    assert fu._communicate(_Proc(["ffmpeg"], returncode=1), scope, None, "ffmpeg")[0] == b"out"


def test_communicate_a_failed_exit_in_a_cancelled_scope_is_a_ctrl_c(monkeypatch) -> None:
    scope = fu.ChildScope()
    monkeypatch.setattr(scope, "wait_cancelled", lambda t: t == fu.STOP_GRACE_S)
    with pytest.raises(KeyboardInterrupt):
        fu._communicate(_Proc(["ffmpeg"], returncode=1), scope, None, "ffmpeg")
    assert fu._communicate(_Proc(["ffmpeg"]), scope, None, "ffmpeg") == (b"out", b"err")
    scope.cancel()
    with pytest.raises(KeyboardInterrupt):  # stopped from outside: whatever its exit
        fu._communicate(_Proc(["ffmpeg"]), scope, None, "ffmpeg")
