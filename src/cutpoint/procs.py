"""Child processes: isolate, lower the priority of, stop and kill them, and tie them to Python.

On Windows a child goes in a job object (:func:`kill_with_parent`, :func:`join_tree_job`); on
Linux it gets SIGTERM when Python exits (:func:`pdeathsig_hook`). The Windows-only modules
(``ctypes``) are imported inside the functions that need them, never at import time.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import weakref
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

SOFT_STOP_S = 5.0  # a stopped child (ffmpeg; a UI job after CTRL_BREAK) gets this long
# A stopped job gets this long before the hard kill: its own soft stop waits SOFT_STOP_S for
# ffmpeg, then kills it and waits SOFT_STOP_S again, and only then removes the partial file;
# the rest is for that removal and the interpreter's exit (#257).
JOB_STOP_S = 2 * SOFT_STOP_S + 2.0


def isolated_kwargs() -> dict[str, Any]:
    """``Popen`` options that start a child in its own process group (Windows) or session
    (POSIX), so the console's Ctrl+C (and the terminal's SIGINT) does not reach it: the
    caller stops it. No stdin, so it never reads the console."""
    if os.name == "nt":
        return {"stdin": subprocess.DEVNULL, "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"stdin": subprocess.DEVNULL, "start_new_session": True}


# Isolated children also go in a Windows job object (:func:`kill_with_parent`):
# out of the console's group, a ``taskkill /F`` of Python would leave ffmpeg running until its
# next write to the closed pipe; the job (KILL_ON_JOB_CLOSE) ends it at once. POSIX does nothing:
# the pipe closes with the parent and ffmpeg 7+ exits on EPIPE (6.x finishes its pass first);
# either way the next run's sweep removes the spool and the tmp proxy.


def nice_prefix(args: list[str]) -> list[str]:
    """``args`` run through ``nice -n 10`` when there is one (POSIX; it execs, so the pid and
    group stay the same), else ``args`` as they are."""
    nice = shutil.which("nice")
    return [nice, "-n", "10", *args] if nice else args


def stop_child(
    proc: subprocess.Popen[Any], err: BaseException | None, *, label: str | None = None
) -> None:
    """Let ``proc`` exit before the caller removes its partial output (#199, #229).

    On Windows a running ffmpeg keeps its output open, so removing a ``.part`` or
    ``.tmp.wav`` file right after Ctrl+C fails. After a KeyboardInterrupt (Ctrl+C,
    Ctrl+Break, the UI's cancel) the child got the same event, so it is given
    ``SOFT_STOP_S`` to exit; after any other error, ``err`` None (a timeout), or when it is
    still running then, it is killed and waited for (again at most ``SOFT_STOP_S``). The
    caller re-raises ``err``.

    ``label``: when ``proc`` is still running after the kill, log a warning naming it (the
    360 stitcher's MediaSDKTest). The UI's cancel also gives a job ``SOFT_STOP_S`` before
    ending it: a child that is slow to exit can use it up before the partial file is removed.
    """
    try:
        if isinstance(err, KeyboardInterrupt):
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(SOFT_STOP_S)
    finally:  # a second Ctrl+C during the wait still kills it
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(SOFT_STOP_S)
            except subprocess.TimeoutExpired:
                if label:
                    logger.warning("%s (pid %s) did not exit after kill", label, proc.pid)


def join_tree_job(proc: subprocess.Popen[str]) -> None:
    """Windows: put ``proc`` in a job object of its own, so its descendants join it too.

    ``kill_tree`` ends the job, which reaches every descendant even when the leader is gone:
    ``taskkill /T`` walks the tree from a live leader only, and a soft stop that worked has ended
    the leader. A descendant can miss CTRL_BREAK: a venv's python.exe launcher ignores the event,
    so an interpreter it starts just after the event keeps running and holds the job's stdout.
    """
    import ctypes  # noqa: PLC0415 - Windows only

    k32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    k32.CreateJobObjectW.restype = ctypes.c_void_p
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return
    if k32.AssignProcessToJobObject(ctypes.c_void_p(job), ctypes.c_void_p(int(proc._handle))):  # type: ignore[attr-defined]
        proc.tree_job = job  # type: ignore[attr-defined]
        weakref.finalize(proc, k32.CloseHandle, ctypes.c_void_p(job))
    else:  # e.g. the leader already exited: taskkill alone, as before
        k32.CloseHandle(ctypes.c_void_p(job))


def kill_tree(proc: subprocess.Popen[str], wait: float = JOB_STOP_S, *, term: bool = True) -> None:
    """Stop ``proc`` and everything it started (ffmpeg, workers): TERM, wait ``wait``, then KILL.

    ``term=False``: the KILL alone, for a job that was already sent its TERM.

    Windows: CTRL_BREAK_EVENT to the job's process group first (cutpoint turns it into
    KeyboardInterrupt: partial files removed, exit 130), then the job object ``spawn`` made
    and ``taskkill /T /F``. The event
    needs a console shared with the job; without one (pythonw) only the hard kill runs.
    GenerateConsoleCtrlEvent is called directly: when it fails, ``os.kill`` falls through to
    TerminateProcess on the leader alone, which orphans its ffmpeg before taskkill can see it.
    """
    if os.name == "nt":
        import ctypes  # noqa: PLC0415 - Windows only

        k32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
        if term and k32.GenerateConsoleCtrlEvent(signal.CTRL_BREAK_EVENT, proc.pid):
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(wait)
        job = getattr(proc, "tree_job", None)
        if job:  # every descendant, including ones the leader's exit left out of /T
            k32.TerminateJobObject(ctypes.c_void_p(job), 1)
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False
        )
        return
    if term:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(wait)
    with contextlib.suppress(ProcessLookupError):  # sweeps members that outlived the leader
        os.killpg(proc.pid, signal.SIGKILL)


_JOB_HANDLE: Any = None
_LOCK = threading.Lock()  # ffmpeg is started from worker threads too: create one job only


def kill_with_parent(proc: subprocess.Popen[Any]) -> None:
    """Windows: put ``proc`` in a job object that is killed when Python exits.

    No-op on other platforms or if the API is unavailable or fails (logged at DEBUG).
    """
    global _JOB_HANDLE
    if os.name != "nt":
        return
    try:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        with _LOCK:
            if _JOB_HANDLE is None:

                class _Basic(ctypes.Structure):
                    _fields_ = [
                        ("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD),
                    ]

                class _Ext(ctypes.Structure):
                    _fields_ = [
                        ("BasicLimitInformation", _Basic),
                        ("IoInfo", ctypes.c_uint64 * 6),  # IO_COUNTERS
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t),
                    ]

                k32.CreateJobObjectW.restype = wintypes.HANDLE
                job = k32.CreateJobObjectW(None, None)
                if not job:
                    logger.debug("job object not created: error %d", ctypes.get_last_error())
                    return
                info = _Ext()
                info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
                if not k32.SetInformationJobObject(
                    wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)
                ):
                    logger.debug("job object limits not set: error %d", ctypes.get_last_error())
                    k32.CloseHandle(wintypes.HANDLE(job))
                    return
                _JOB_HANDLE = job
        if not k32.AssignProcessToJobObject(
            wintypes.HANDLE(_JOB_HANDLE),
            wintypes.HANDLE(int(proc._handle)),  # type: ignore[attr-defined]
        ):
            logger.debug(
                "process %d not put in the job object: error %d", proc.pid, ctypes.get_last_error()
            )
    except (OSError, AttributeError, ValueError) as err:
        logger.debug("job object not available: %s", err)


def pdeathsig_hook() -> Callable[[], None] | None:
    """Linux ``preexec_fn`` that makes the child get SIGTERM when Python exits.

    The Linux counterpart of :func:`kill_with_parent`: without either, a crashed or killed
    driver leaves the child running. libc is loaded here, in the parent, so the child only
    calls ``prctl`` between fork and exec.
    None on other platforms or if libc is unavailable.
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes  # noqa: PLC0415

        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError) as err:
        logger.debug("prctl not available: %s", err)
        return None
    parent = os.getpid()

    def hook() -> None:
        try:
            prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
            if os.getppid() != parent:  # parent already gone
                os.kill(os.getpid(), signal.SIGTERM)
        except Exception:  # noqa: BLE001 - best effort, must not break the launch
            pass

    return hook


def _win_created(k32: Any, handle: Any) -> float | None:
    """Creation time (epoch seconds) of the Windows process ``handle``, or None."""
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    times = [wintypes.FILETIME() for _ in range(4)]  # creation, exit, kernel, user
    k32.GetProcessTimes.argtypes = (wintypes.HANDLE, *[ctypes.POINTER(wintypes.FILETIME)] * 4)
    if not k32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
        return None
    ft = times[0].dwHighDateTime << 32 | times[0].dwLowDateTime  # 100 ns since 1601
    return (ft - 116444736000000000) / 1e7


# Windows file times (the lock's mtime) and process creation times come from clocks of
# different granularity, so a lock written right after its process started can read a little
# earlier than that process's creation time. Real id reuse comes long after the lock.
_CREATED_SLACK = 2.0


def pid_alive(pid: int, not_after: float | None = None) -> bool:
    """Whether the process ``pid`` is running (True when that cannot be told).

    ``not_after``: on Windows, which reuses process ids quickly, a process created after
    this time (epoch seconds, the lock file's mtime, plus a small slack)
    is another one that got the id: False.
    """
    if pid <= 0 or pid > 0xFFFFFFFF:
        return False
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            # 87 (ERROR_INVALID_PARAMETER): no such process; anything else (5, access
            # denied, for a protected or another user's process): it exists
            return ctypes.get_last_error() != 87
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            if code.value != 259:  # STILL_ACTIVE
                return False
            created = _win_created(k32, handle) if not_after is not None else None
            return created is None or not_after is None or created <= not_after + _CREATED_SLACK
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # signal 0 only checks (on Windows os.kill would terminate it)
    except ProcessLookupError:
        return False
    except OSError:  # PermissionError: it exists, owned by another user
        return True
    return True
