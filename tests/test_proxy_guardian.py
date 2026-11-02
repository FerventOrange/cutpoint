"""Real-process tests for the platform-independent proxy credential guardian."""

from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_BACKEND = """
import json, os
from pathlib import Path
from keyring.backend import KeyringBackend

class SyntheticBackend(KeyringBackend):
    @property
    def priority(self):
        return 1

    def _read(self):
        path = Path(os.environ["SYNTHETIC_KEYRING_FILE"])
        return json.loads(path.read_text()) if path.exists() else {}

    def _write(self, values):
        path = Path(os.environ["SYNTHETIC_KEYRING_FILE"])
        path.write_text(json.dumps(values))

    def get_password(self, service, username):
        return self._read().get(f"{service}:{username}")

    def set_password(self, service, username, password):
        if os.environ.get("SYNTHETIC_KEYRING_FAIL"):
            raise RuntimeError("synthetic backend failure")
        values = self._read()
        values[f"{service}:{username}"] = password
        self._write(values)

    def delete_password(self, service, username):
        values = self._read()
        values.pop(f"{service}:{username}", None)
        self._write(values)
"""


@pytest.fixture
def guardian_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    backend = tmp_path / "synthetic_keyring.py"
    backend.write_text(_BACKEND, encoding="utf-8")
    (tmp_path / "sitecustomize.py").write_text(
        "from cutpoint.assistant.proxy_credentials import CredentialStore\n"
        "CredentialStore._select_os_backend = staticmethod(lambda module: module.get_keyring())\n",
        encoding="utf-8",
    )

    state = tmp_path / "keyring.json"
    source = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHON_KEYRING_BACKEND"] = "synthetic_keyring.SyntheticBackend"
    env["SYNTHETIC_KEYRING_FILE"] = str(state)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(tmp_path), str(source), env.get("PYTHONPATH", "")) if part
    )
    return env, state


def _seed_runtime(root: Path, token: str = "initial-token") -> Path:
    auth = root / "auths" / "claude.json"
    auth.parent.mkdir(parents=True)
    auth.write_text(json.dumps({"token": token}), encoding="utf-8")
    return auth


def _launch(
    root: Path,
    tmp_path: Path,
    guardian_env: tuple[dict[str, str], Path],
    command: list[str],
) -> tuple[subprocess.Popen[bytes], Path]:
    env, state = guardian_env
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cutpoint.assistant._proxy_guardian",
            "--root",
            str(root),
            "--parent-pid",
            str(os.getpid()),
            "--",
            *command,
        ],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process, state


def _wait_for(path: Path, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 8
    while not path.exists() and time.monotonic() < deadline:
        if process.poll() is not None:
            break
        time.sleep(0.02)
    assert path.exists(), "daemon fixture did not reach its ready point"


def _stored_snapshot(state_path: Path) -> dict[str, str]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    key, manifest_raw = next(
        (key, value)
        for key, value in state.items()
        if key.startswith("cutpoint:") and key.endswith("-manifest")
    )
    manifest = json.loads(manifest_raw)
    prefix = key[: -len("manifest")]
    encoded = "".join(
        state[f"{prefix}generation-{manifest['generation']}-{index}"]
        for index in range(manifest["count"])
    )
    return json.loads(base64.b64decode(encoded).decode("utf-8"))


@pytest.mark.slow
def test_startup_eof_does_not_prepare_or_launch_daemon(
    tmp_path: Path, guardian_env: tuple[dict[str, str], Path]
) -> None:
    root = tmp_path / "proxy"
    marker = tmp_path / "launched"
    process, _ = _launch(
        root,
        tmp_path,
        guardian_env,
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path(__import__('sys').argv[1]).touch()",
            str(marker),
        ],
    )
    assert process.stdin is not None
    process.stdin.close()
    assert process.wait(timeout=8) == 0
    assert not marker.exists()
    assert not root.exists()


@pytest.mark.slow
@pytest.mark.parametrize("status", [0, 42])
def test_guardian_propagates_normal_and_abnormal_daemon_exit(
    tmp_path: Path,
    guardian_env: tuple[dict[str, str], Path],
    status: int,
) -> None:
    root = tmp_path / "proxy"
    _seed_runtime(root)
    process, _ = _launch(
        root,
        tmp_path,
        guardian_env,
        [sys.executable, "-c", "import sys; sys.exit(int(sys.argv[1]))", str(status)],
    )
    assert process.wait(timeout=12) == status
    assert not list((root / "auths").glob("*.json"))


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="Windows terminate is forceful, not ignorable")
def test_guardian_escalates_when_daemon_ignores_termination(
    tmp_path: Path, guardian_env: tuple[dict[str, str], Path]
) -> None:
    root = tmp_path / "proxy"
    _seed_runtime(root)
    ready = tmp_path / "ready"
    code = (
        "import signal,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "Path(sys.argv[1]).touch(); time.sleep(60)"
    )
    process, _ = _launch(root, tmp_path, guardian_env, [sys.executable, "-c", code, str(ready)])
    try:
        _wait_for(ready, process)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=9) == 137
        assert not list((root / "auths").glob("*.json"))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows terminates rather than signals processes"
)
def test_guardian_signal_stop_preserves_daemon_exit_code(
    tmp_path: Path, guardian_env: tuple[dict[str, str], Path]
) -> None:
    root = tmp_path / "proxy"
    _seed_runtime(root)
    ready = tmp_path / "ready"
    code = (
        "import signal,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(13)); "
        "Path(sys.argv[1]).touch(); time.sleep(60)"
    )
    process, _ = _launch(root, tmp_path, guardian_env, [sys.executable, "-c", code, str(ready)])
    try:
        _wait_for(ready, process)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=8) == 13
        assert not list((root / "auths").glob("*.json"))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.slow
def test_parent_already_gone_skips_prepare_and_daemon_launch(
    tmp_path: Path, guardian_env: tuple[dict[str, str], Path]
) -> None:
    root = tmp_path / "proxy"
    marker = tmp_path / "launched"
    env, state = guardian_env
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cutpoint.assistant._proxy_guardian",
            "--root",
            str(root),
            "--parent-pid",
            "4294967295",
            "--",
            sys.executable,
            "-c",
            "from pathlib import Path; Path(__import__('sys').argv[1]).touch()",
            str(marker),
        ],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    process.stdin.close()
    assert process.wait(timeout=8) == 0
    assert not marker.exists()
    assert not root.exists()
    assert not state.exists()


@pytest.mark.slow
def test_store_failures_are_sanitized_and_nonzero(
    tmp_path: Path, guardian_env: tuple[dict[str, str], Path]
) -> None:
    root = tmp_path / "proxy"
    _seed_runtime(root, "private-secret-value")
    env, state = guardian_env
    env["SYNTHETIC_KEYRING_FAIL"] = "1"
    process, _ = _launch(root, tmp_path, (env, state), [sys.executable, "-c", "pass"])
    assert process.wait(timeout=12) == 1
    assert process.stderr is not None
    stderr = process.stderr.read()
    assert b"keyring" in stderr
    assert b"private-secret-value" not in stderr
    assert b"synthetic backend failure" not in stderr
    if process.stdin is not None:
        process.stdin.close()
