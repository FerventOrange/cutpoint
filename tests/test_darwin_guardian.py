"""Tests for Darwin parent-death guardian (_darwin_guardian.py).

Exercised across POSIX environments to verify pipe EOF detection, bounded child
termination/kill, signal forwarding, and race handling.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX guardian tests")
def test_guardian_terminates_child_on_parent_pipe_eof(tmp_path: Path) -> None:
    r_fd, w_fd = os.pipe()
    os.set_inheritable(r_fd, True)

    marker = tmp_path / "child_alive.txt"
    # Child writes marker, then sleeps
    child_code = (
        f"from pathlib import Path; import time; Path('{marker}').write_text('1'); time.sleep(60)"
    )

    guardian_cmd = [
        sys.executable,
        "-m",
        "cutpoint.assistant._darwin_guardian",
        "--read-fd",
        str(r_fd),
        "--",
        sys.executable,
        "-c",
        child_code,
    ]

    proc = subprocess.Popen(guardian_cmd, pass_fds=(r_fd,))
    os.close(r_fd)

    # Wait for child to start
    deadline = time.time() + 3.0
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert marker.exists()

    # Simulate parent sudden death (SIGKILL) by closing parent's write-end of pipe
    os.close(w_fd)

    # Guardian must terminate child and exit promptly, well before 60s
    proc.wait(timeout=5.0)
    # On SIGTERM, child returncode on POSIX is negative signal or 0/1 depending on shell/reap
    assert proc.poll() is not None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX guardian tests")
def test_guardian_forwards_child_exit_code() -> None:
    r_fd, w_fd = os.pipe()
    os.set_inheritable(r_fd, True)

    guardian_cmd = [
        sys.executable,
        "-m",
        "cutpoint.assistant._darwin_guardian",
        "--read-fd",
        str(r_fd),
        "--",
        sys.executable,
        "-c",
        "import sys; sys.exit(42)",
    ]

    proc = subprocess.Popen(guardian_cmd, pass_fds=(r_fd,))
    os.close(r_fd)
    try:
        rc = proc.wait(timeout=5.0)
        assert rc == 42
    finally:
        os.close(w_fd)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX guardian tests")
def test_guardian_race_parent_dead_before_child_starts(tmp_path: Path) -> None:
    r_fd, w_fd = os.pipe()
    os.set_inheritable(r_fd, True)
    # Close write end immediately before guardian even runs
    os.close(w_fd)

    marker = tmp_path / "should_not_run.txt"
    guardian_cmd = [
        sys.executable,
        "-m",
        "cutpoint.assistant._darwin_guardian",
        "--read-fd",
        str(r_fd),
        "--",
        sys.executable,
        "-c",
        f"from pathlib import Path; Path('{marker}').write_text('ran')",
    ]

    proc = subprocess.Popen(guardian_cmd, pass_fds=(r_fd,))
    os.close(r_fd)

    rc = proc.wait(timeout=5.0)
    assert rc == 0
    # Child was never executed because parent was already dead
    assert not marker.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX guardian tests")
def test_guardian_signal_forwarding() -> None:
    r_fd, w_fd = os.pipe()
    os.set_inheritable(r_fd, True)

    # Child exits with code 13 upon SIGTERM
    child_code = (
        "import signal, sys, time; "
        "signal.signal(signal.SIGTERM, lambda s, f: sys.exit(13)); "
        "print('READY', flush=True); "
        "time.sleep(30)"
    )

    guardian_cmd = [
        sys.executable,
        "-m",
        "cutpoint.assistant._darwin_guardian",
        "--read-fd",
        str(r_fd),
        "--",
        sys.executable,
        "-c",
        child_code,
    ]

    proc = subprocess.Popen(
        guardian_cmd,
        stdout=subprocess.PIPE,
        pass_fds=(r_fd,),
    )
    os.close(r_fd)

    try:
        # Wait until child prints READY
        assert proc.stdout is not None
        proc.stdout.readline()

        # Send SIGTERM to guardian
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=5.0)
        assert rc == 13
    finally:
        os.close(w_fd)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX guardian tests")
def test_guardian_escalates_to_sigkill_when_sigterm_ignored() -> None:
    r_fd, w_fd = os.pipe()
    os.set_inheritable(r_fd, True)

    # Child ignores SIGTERM
    child_code = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('READY', flush=True); "
        "time.sleep(60)"
    )

    guardian_cmd = [
        sys.executable,
        "-m",
        "cutpoint.assistant._darwin_guardian",
        "--read-fd",
        str(r_fd),
        "--",
        sys.executable,
        "-c",
        child_code,
    ]

    proc = subprocess.Popen(
        guardian_cmd,
        stdout=subprocess.PIPE,
        pass_fds=(r_fd,),
    )
    os.close(r_fd)

    try:
        assert proc.stdout is not None
        proc.stdout.readline()

        # Close write end to simulate parent SIGKILL
        os.close(w_fd)
        # Guardian terminates (ignored), waits timeout (3s), then sends SIGKILL and reaps
        proc.wait(timeout=7.0)
        assert proc.poll() is not None
    finally:
        with contextlib.suppress(OSError):
            os.close(w_fd)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX guardian tests")
def test_guardian_sigterm_escalates_when_child_ignores_term() -> None:
    r_fd, w_fd = os.pipe()
    os.set_inheritable(r_fd, True)

    child_code = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('READY', flush=True); "
        "time.sleep(60)"
    )

    guardian_cmd = [
        sys.executable,
        "-m",
        "cutpoint.assistant._darwin_guardian",
        "--read-fd",
        str(r_fd),
        "--",
        sys.executable,
        "-c",
        child_code,
    ]

    proc = subprocess.Popen(
        guardian_cmd,
        stdout=subprocess.PIPE,
        pass_fds=(r_fd,),
    )
    os.close(r_fd)

    try:
        assert proc.stdout is not None
        proc.stdout.readline()

        # Send SIGTERM to guardian while keeping pipe open
        proc.send_signal(signal.SIGTERM)
        # Guardian must not hang waiting forever: it attempts term, times out (3s), kills and reaps
        proc.wait(timeout=7.0)
        assert proc.poll() is not None
    finally:
        with contextlib.suppress(OSError):
            os.close(w_fd)
