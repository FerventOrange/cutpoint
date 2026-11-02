"""Supervise CLIProxyAPI independently of its foreground launcher."""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from cutpoint.assistant.keys import KeyringError
from cutpoint.assistant.proxy_credentials import CredentialStore
from cutpoint.events.llama_server import _pdeathsig_hook
from cutpoint.ffmpeg_utils import FolderBusyError, _pid_alive, swap_lock
from cutpoint.winjob import kill_with_parent

_TERMINATE_TIMEOUT = 3.0
_KILL_TIMEOUT = 2.0
_POLL_INTERVAL = 0.5

_SUPERVISION_ERRORS = (
    KeyringError,
    OSError,
    ValueError,
    subprocess.SubprocessError,
    FolderBusyError,
)


def _parent_alive(pid: int) -> bool:
    """Check process liveness without signaling it (including on Windows)."""
    if pid <= 0 or (sys.platform != "win32" and os.getppid() != pid):
        return False
    return _pid_alive(pid)


def _cleanup_child(child: subprocess.Popen[Any]) -> int | None:
    """Bound child termination, escalate to kill, and always attempt to reap."""
    if child.poll() is None:
        with contextlib.suppress(OSError):
            child.terminate()
        try:
            child.wait(timeout=_TERMINATE_TIMEOUT)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                child.kill()
            try:
                child.wait(timeout=_KILL_TIMEOUT)
            except subprocess.TimeoutExpired:
                # The OS/backend refused or could not complete termination; retain the failure.
                return None
        except OSError:
            pass
    try:
        return child.wait(timeout=_KILL_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return child.poll()


def _watch_stdin(fd: int, eof: threading.Event, started: threading.Event) -> None:
    started.set()
    try:
        with contextlib.suppress(OSError):
            while os.read(fd, 4096):
                pass
    finally:
        eof.set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CLIProxyAPI daemon supervisor")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not command:
        return 1

    parent_pid = args.parent_pid

    eof = threading.Event()
    reader_started = threading.Event()
    threading.Thread(
        target=_watch_stdin, args=(sys.stdin.fileno(), eof, reader_started), daemon=True
    ).start()

    stopping = threading.Event()

    def request_stop(signum: int, frame: Any) -> None:
        stopping.set()

    previous: dict[int, Any] = {}
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            with contextlib.suppress(ValueError, OSError):
                previous[sig] = signal.signal(sig, request_stop)

    def parent_gone() -> bool:
        return eof.is_set() or not _parent_alive(parent_pid)

    # Start the portable blocking reader before making the pre-prepare liveness decision.
    reader_started.wait(1.0)
    eof.wait(0.05)
    if parent_gone() or stopping.is_set():
        return 0

    store = CredentialStore(args.root)
    child: subprocess.Popen[Any] | None = None
    child_status: int | None = None
    failure = False
    prepare_attempted = False
    try:
        with swap_lock(args.root / "daemon.run"):
            if parent_gone() or stopping.is_set():
                return 0
            prepare_attempted = True
            try:
                store.prepare()
                if not (parent_gone() or stopping.is_set()):
                    child = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        close_fds=True,
                        preexec_fn=_pdeathsig_hook(),
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    kill_with_parent(child)
                    while child.poll() is None and not (parent_gone() or stopping.is_set()):
                        try:
                            child.wait(timeout=_POLL_INTERVAL)
                        except subprocess.TimeoutExpired:
                            store.sync()
                    child_status = _cleanup_child(child) if child.poll() is None else child.wait()
            except _SUPERVISION_ERRORS as err:
                failure = True
                detail = str(err) if isinstance(err, KeyringError) else type(err).__name__
                print(
                    "proxy daemon supervision or credential preparation failed: " + detail,
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                if child is not None and child.poll() is None:
                    child_status = _cleanup_child(child)
                if child is not None and child.poll() is not None:
                    child_status = child.returncode
                if prepare_attempted:
                    if child is None or child.poll() is not None:
                        try:
                            store.finish()
                        except _SUPERVISION_ERRORS as err:
                            failure = True
                            detail = (
                                str(err) if isinstance(err, KeyringError) else type(err).__name__
                            )
                            print(
                                f"proxy credential finalization failed: {detail}",
                                file=sys.stderr,
                                flush=True,
                            )
                    else:
                        failure = True
                        print(
                            "daemon could not be reaped; credential finalization was skipped",
                            file=sys.stderr,
                            flush=True,
                        )
    except _SUPERVISION_ERRORS as err:
        failure = True
        detail = str(err) if isinstance(err, KeyringError) else type(err).__name__
        print(f"proxy guardian failed: {detail}", file=sys.stderr, flush=True)
    finally:
        for sig, handler in previous.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(sig, handler)

    if failure or (child is not None and child_status is None):
        return 1
    if child_status is None or eof.is_set():
        return 0
    return child_status if child_status >= 0 else 128 - child_status


if __name__ == "__main__":
    sys.exit(main())
