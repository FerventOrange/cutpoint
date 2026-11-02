"""End child processes when Python exits: a Windows job object and a Linux parent-death signal.

:func:`kill_with_parent` puts a child in one job object per Python process, created with
KILL_ON_JOB_CLOSE: when Python exits, even by ``taskkill /F`` or a crash, Windows closes the
job's handle and kills every process in it. Used for isolated ffmpeg processes, which leave
the console's group, and for the CLIProxyAPI guardian. :func:`pdeathsig_hook` is the Linux
counterpart, a ``preexec_fn`` for ``subprocess.Popen``.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_JOB_HANDLE: Any = None
_LOCK = threading.Lock()  # ffmpeg is started from worker threads too: create one job only


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
