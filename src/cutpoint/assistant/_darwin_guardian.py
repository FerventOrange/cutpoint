"""Private stdlib-only guardian process for supervising daemon lifetime on Darwin (macOS).

Monitors a parent-owned pipe; on pipe EOF (indicating parent termination or SIGKILL),
terminates, reaps, and cleans up the child daemon process to prevent orphaned listeners.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import select
import signal
import subprocess
import sys
from typing import Any


def _cleanup_child(child: subprocess.Popen[Any], timeout: float = 3.0) -> int:
    """Terminate, then if needed kill and reap child process."""
    if child.poll() is None:
        try:
            child.terminate()
            child.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            try:
                child.kill()
                child.wait(timeout=2.0)
            except OSError:
                pass
    return child.poll() if child.poll() is not None else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Darwin child daemon supervisor")
    parser.add_argument("--read-fd", type=int, required=True, help="Parent pipe read descriptor")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Daemon command to execute")
    args = parser.parse_args(argv)

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        return 1
    read_fd = args.read_fd
    child: subprocess.Popen[Any] | None = None
    interrupted = False

    def _handle_signal(signum: int, frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        if child is not None and child.poll() is None:
            with contextlib.suppress(OSError):
                child.send_signal(signum)

    orig_sigint = signal.signal(signal.SIGINT, _handle_signal)
    orig_sigterm = signal.signal(signal.SIGTERM, _handle_signal)

    # Race check: if parent already died before guardian started, read_fd will be EOF
    try:
        rlist, _, _ = select.select([read_fd], [], [], 0.0)
        if read_fd in rlist:
            data = os.read(read_fd, 1024)
            if not data:
                return 0
    except OSError:
        return 0

    try:
        child = subprocess.Popen(cmd)
        while not interrupted:
            ret = child.poll()
            if ret is not None:
                return ret

            try:
                rlist, _, _ = select.select([read_fd], [], [], 0.2)
            except (OSError, InterruptedError):
                continue

            if read_fd in rlist:
                try:
                    data = os.read(read_fd, 1024)
                except OSError:
                    data = b""

                if not data:
                    break
    finally:
        signal.signal(signal.SIGINT, orig_sigint)
        signal.signal(signal.SIGTERM, orig_sigterm)
        with contextlib.suppress(OSError):
            os.close(read_fd)
        if child is not None:
            _cleanup_child(child, timeout=3.0)

    return child.poll() if (child is not None and child.poll() is not None) else 0


if __name__ == "__main__":
    sys.exit(main())
