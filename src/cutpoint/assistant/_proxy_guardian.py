"""Supervise CLIProxyAPI independently of its foreground launcher."""

from __future__ import annotations

import argparse
import collections
import contextlib
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any

from cutpoint import folders, procs
from cutpoint.assistant import keys
from cutpoint.assistant.keys import KeyringError
from cutpoint.assistant.private_files import private_directory, private_file
from cutpoint.assistant.proxy_credentials import CredentialStore
from cutpoint.errors import FolderBusyError

_TERMINATE_TIMEOUT = 3.0
_KILL_TIMEOUT = 2.0
_POLL_INTERVAL = 0.5  # how often the daemon, the parent and stop requests are checked
# How often runtime credentials are compared with the last commit. sync() skips the keyring and
# permission changes when nothing changed, but still checks the modes: a loosened one makes the
# probe return None, so the full sync runs and tightens it. finish() always saves on stop.
_SYNC_INTERVAL = 5.0

_SUPERVISION_ERRORS = (
    KeyringError,
    OSError,
    ValueError,
    subprocess.SubprocessError,
    FolderBusyError,
)
# A periodic sync that fails (say a keyring locked after screen lock) warns once per failure
# streak and is tried again later; prepare() and finish() still fail closed. A failed sync reads
# the keyring in full, so during a streak the wait doubles per failure, up to _SYNC_BACKOFF_MAX
# (a locked keyring that prompts is not asked every 5 s), and is back to _SYNC_INTERVAL after a
# success.
_SYNC_RETRY_ERRORS = (KeyringError, OSError, ValueError)
_SYNC_BACKOFF_MAX = 60.0

# CLIProxyAPI's stdout and stderr go to one pipe that a thread drains into a ring of its last
# lines, so a daemon that runs for days neither stalls on a full pipe nor fills a log file.
# When the daemon exits on its own, the redacted tail goes to OUTPUT_FILE in the proxy folder
# (owner-only, removed at each start) for the launcher's error message.
OUTPUT_FILE = "daemon-output.txt"
_TAIL_LINES = 10
_TAIL_CHARS = 2000
_LINE_BYTES = 8192  # a longer line is kept in pieces
_DRAIN_JOIN = 1.0  # a grandchild may hold the pipe open: never wait longer for the reader
# The proxy's 64-hex management and client keys; the launcher remembers them, this process
# does not, so they are masked by shape too.
_HEX_KEY = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])")
# Daemon output is logs, not prose, so these mask with no length minimum and may mask too
# much: a value after a secret's name (JSON, key=value or a header: OAuth access, refresh and
# id tokens, passwords, API keys), an OAuth callback's ?code=, a Basic login and a bare
# refresh token (rt_...).
_DAEMON_PARAM = re.compile(
    r"((?:access|refresh|id)?[-_]?token|secret|password|passwd|(?:secret|api)[-_]?key"
    r"|x-goog-api-key)(\\*[\"']?\s*[:=]\s*\\*[\"']?)[^\s\"'&,;}\\]+",
    re.IGNORECASE,
)
_DAEMON_CODE = re.compile(r"([?&]code=)[^\s\"'&#]+", re.IGNORECASE)
_DAEMON_BASIC = re.compile(r"(\bBasic\s+)[A-Za-z0-9+/=]{8,}", re.IGNORECASE)
_DAEMON_RT = re.compile(r"\brt_[A-Za-z0-9_\-.]{8,}")
# A bare JWT (three base64url parts, the first a JSON header) and a bare Google API key. The
# lookbehind lets only the first eyJ of a run start a match (linear, not quadratic).
_DAEMON_JWT = re.compile(r"(?<![A-Za-z0-9_\-])eyJ[A-Za-z0-9_\-]*\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]*")
_DAEMON_GOOGLE = re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}")


class _Tail:
    """The last lines a stream wrote, bounded in lines and bytes."""

    def __init__(self) -> None:
        self._lines: collections.deque[bytes] = collections.deque()
        self._partial = False  # the oldest piece kept continues a dropped one
        self._lock = threading.Lock()

    def drain(self, stream: IO[bytes]) -> None:
        with contextlib.suppress(OSError, ValueError):
            for piece in iter(lambda: stream.readline(_LINE_BYTES), b""):
                with self._lock:
                    if len(self._lines) == _TAIL_LINES:
                        self._partial = not self._lines.popleft().endswith(b"\n")
                    self._lines.append(piece)

    def text(self) -> str:
        with self._lock:
            raw, partial = b"".join(self._lines), self._partial
        text = raw.decode("utf-8", errors="replace")
        if partial:  # the start of that line is gone: a secret in it could be cut in two
            text = text.partition("\n")[2]
        return redact_output(text)


def redact_output(text: str) -> str:
    """``text`` with the keys :func:`keys.redact` knows, any 64-hex key and the daemon's
    token, password, API key, OAuth code and Basic login values, bare JWTs and Google keys
    masked, cut to its last ``_TAIL_CHARS`` characters (after redacting, so no secret is cut
    in two)."""
    text = _DAEMON_PARAM.sub(rf"\g<1>\g<2>{keys.REDACTED}", text)
    text = _DAEMON_CODE.sub(rf"\g<1>{keys.REDACTED}", text)
    text = _DAEMON_BASIC.sub(rf"\g<1>{keys.REDACTED}", text)
    text = _DAEMON_RT.sub(keys.REDACTED, text)
    text = _DAEMON_JWT.sub(keys.REDACTED, text)
    text = _DAEMON_GOOGLE.sub(keys.REDACTED, text)
    text = _HEX_KEY.sub(keys.REDACTED, keys.redact(text)).strip()
    return text[-_TAIL_CHARS:].lstrip()


def _save_output(root: Path, text: str) -> None:
    if not text:
        return
    private_directory(root)
    path = root / OUTPUT_FILE
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    private_file(path)


def output_tail(root: Path) -> str:
    """What CLIProxyAPI printed last before it exited on its own, redacted, as a suffix for
    an error message ("" when it printed nothing or the file cannot be read safely)."""
    path = Path(root) / OUTPUT_FILE
    try:
        private_file(path)
        with path.open(encoding="utf-8", errors="replace") as f:
            text = redact_output(f.read(4 * _TAIL_CHARS))
    except (OSError, ValueError):
        return ""
    return f"\nCLIProxyAPI output (last lines):\n{text}" if text else ""


def _parent_alive(pid: int) -> bool:
    """Check process liveness without signaling it (including on Windows)."""
    if pid <= 0 or (sys.platform != "win32" and os.getppid() != pid):
        return False
    return procs.pid_alive(pid)


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


def _periodic_sync(store: CredentialStore, failing: bool) -> bool:
    """Run one periodic sync and return whether it failed. The first failure of a streak is
    printed as a WARNING; the others are silent until a sync succeeds again."""
    try:
        store.sync()
    except _SYNC_RETRY_ERRORS as err:
        if not failing:
            detail = str(err) if isinstance(err, KeyringError) else type(err).__name__
            print(
                f"WARNING proxy credential sync failed, retrying: {detail}",
                file=sys.stderr,
                flush=True,
            )
        return True
    if failing:
        print("proxy credential sync recovered", file=sys.stderr, flush=True)
    return False


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
    with contextlib.suppress(OSError):
        (args.root / OUTPUT_FILE).unlink()  # an earlier run's output

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
    tail = _Tail()
    drain: threading.Thread | None = None
    exited_itself = False
    failure = False
    prepare_attempted = False
    try:
        with folders.swap_lock(args.root / "daemon.run"):
            if parent_gone() or stopping.is_set():
                return 0
            prepare_attempted = True
            try:
                store.prepare()
                if not (parent_gone() or stopping.is_set()):
                    child = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        close_fds=True,
                        preexec_fn=procs.pdeathsig_hook(),
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    procs.kill_with_parent(child)
                    drain = threading.Thread(target=tail.drain, args=(child.stdout,), daemon=True)
                    drain.start()
                    next_sync = time.monotonic() + _SYNC_INTERVAL
                    sync_failing = False
                    sync_delay = _SYNC_INTERVAL
                    while child.poll() is None and not (parent_gone() or stopping.is_set()):
                        try:
                            child.wait(timeout=_POLL_INTERVAL)
                        except subprocess.TimeoutExpired:
                            if time.monotonic() >= next_sync:
                                sync_failing = _periodic_sync(store, sync_failing)
                                sync_delay = (
                                    min(sync_delay * 2, _SYNC_BACKOFF_MAX)
                                    if sync_failing
                                    else _SYNC_INTERVAL
                                )
                                next_sync = time.monotonic() + sync_delay
                    exited_itself = child.poll() is not None
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
                if exited_itself and drain is not None:
                    drain.join(_DRAIN_JOIN)
                    try:
                        _save_output(args.root, tail.text())
                    except OSError as err:
                        print(
                            f"cannot save the proxy daemon's output: {type(err).__name__}",
                            file=sys.stderr,
                            flush=True,
                        )
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
