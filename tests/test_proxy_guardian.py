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

from cutpoint.ffmpeg_utils import _pid_alive

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
        "import os\n"
        "from pathlib import Path\n"
        "from cutpoint.assistant import proxy_credentials\n"
        "from cutpoint.assistant.proxy_credentials import CredentialStore\n"
        "CredentialStore._select_os_backend = staticmethod(lambda module: module.get_keyring())\n"
        "_private_file = proxy_credentials.private_file\n"
        "def _race_private_file(path):\n"
        "    _private_file(path)\n"
        "    target = os.environ.get('GUARDIAN_DISAPPEAR_PATH')\n"
        "    ready = os.environ.get('GUARDIAN_READY_PATH')\n"
        "    disappeared = os.environ.get('GUARDIAN_DISAPPEARED_PATH')\n"
        "    if (\n"
        "        target and ready and disappeared and str(path) == target\n"
        "        and Path(ready).exists() and not Path(disappeared).exists()\n"
        "    ):\n"
        "        path.unlink()\n"
        "        Path(disappeared).touch()\n"
        "proxy_credentials.private_file = _race_private_file\n"
        "_read_file = proxy_credentials._read_file\n"
        "def _observe_read_file(path):\n"
        "    try:\n"
        "        return _read_file(path)\n"
        "    except UnicodeDecodeError:\n"
        "        target = os.environ.get('GUARDIAN_DECODE_PATH')\n"
        "        observed = os.environ.get('GUARDIAN_DECODE_ERROR_PATH')\n"
        "        if target and observed and str(path) == target:\n"
        "            Path(observed).touch()\n"
        "        raise\n"
        "proxy_credentials._read_file = _observe_read_file\n",
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
def test_stdin_eof_reaps_daemon_then_commits_refreshed_credentials(
    tmp_path: Path, guardian_env: tuple[dict[str, str], Path]
) -> None:
    root = tmp_path / "proxy"
    auth = _seed_runtime(root)
    ready = tmp_path / "ready"
    context = tmp_path / "inherited-context"
    env, state = guardian_env
    env["GUARDIAN_TEST_VALUE"] = "preserved"
    code = (
        "from pathlib import Path; import os,sys,time; "
        'Path(sys.argv[1]).write_text(\'{\\"token\\":\\"rotated-token\\"}\'); '
        "Path(sys.argv[2]).write_text('ready'); "
        "Path(sys.argv[3]).write_text(os.getcwd() + '|' + os.environ['GUARDIAN_TEST_VALUE']); "
        "time.sleep(60)"
    )
    process, _ = _launch(
        root,
        tmp_path,
        (env, state),
        [sys.executable, "-c", code, str(auth), str(ready), str(context)],
    )
    try:
        _wait_for(ready, process)
        assert (root / ".daemon.run.cutpoint-lock").exists()
        assert context.read_text(encoding="utf-8") == f"{tmp_path}|preserved"
        assert process.stdin is not None
        process.stdin.close()
        assert process.wait(timeout=12) == 0
        assert _stored_snapshot(state)["claude.json"] == '{"token":"rotated-token"}'
        assert not auth.exists()
        assert not (root / ".daemon.run.cutpoint-lock").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()


@pytest.mark.slow
def test_foreground_hardkill_closes_guardian_pipe_and_finishes_store(
    tmp_path: Path, guardian_env: tuple[dict[str, str], Path]
) -> None:
    root = tmp_path / "proxy"
    auth = _seed_runtime(root)
    ready = tmp_path / "ready"
    stopped = tmp_path / "stopped"
    daemon_code = (
        "import signal,sys,time; from pathlib import Path; "
        'Path(sys.argv[1]).write_text(\'{\\"token\\":\\"hardkill-refresh\\"}\'); '
        "Path(sys.argv[2]).write_text(str(__import__('os').getpid())); "
        "Path(sys.argv[3]).touch(); "
        "signal.signal(signal.SIGTERM, lambda signum, frame: "
        "Path(sys.argv[4]).touch()); time.sleep(60)"
    )
    foreground_exit = tmp_path / "foreground-started"
    parent_code = (
        "import os,subprocess,sys,time; from pathlib import Path; "
        "cmd=[sys.executable,'-m','cutpoint.assistant._proxy_guardian',"
        "'--root',sys.argv[1],'--parent-pid',str(os.getpid()),'--',"
        "sys.executable,'-c',sys.argv[5],sys.argv[2],sys.argv[4],sys.argv[3],sys.argv[6]]; "
        "guardian=subprocess.Popen(cmd,stdin=subprocess.PIPE,cwd=os.getcwd(),env=os.environ.copy(),"
        "start_new_session=os.name!='nt',"
        "creationflags=getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0) if os.name=='nt' else 0); "
        "deadline=time.monotonic()+8; "
        "exec('while not Path(sys.argv[3]).exists() and "
        "time.monotonic()<deadline: time.sleep(.02)'); "
        "Path(sys.argv[7]).touch(); time.sleep(60)"
    )
    env, state = guardian_env
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            parent_code,
            str(root),
            str(auth),
            str(ready),
            str(tmp_path / "daemon-pid"),
            daemon_code,
            str(stopped),
            str(foreground_exit),
        ],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )
    try:
        _wait_for(foreground_exit, parent)
        parent.kill()
        assert parent.wait(timeout=8) is not None
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
    pid = int((tmp_path / "daemon-pid").read_text(encoding="utf-8"))
    deadline = time.monotonic() + 12
    while (auth.exists() or _pid_alive(pid)) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not auth.exists()
    assert not _pid_alive(pid)
    assert _stored_snapshot(state)["claude.json"] == '{"token":"hardkill-refresh"}'
    if sys.platform != "win32":
        assert stopped.exists()


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


@pytest.mark.slow
def test_guardian_survives_disappearing_credential_during_periodic_sync(
    tmp_path: Path, guardian_env: tuple[dict[str, str], Path]
) -> None:
    root = tmp_path / "proxy"
    auth = _seed_runtime(root)
    ready = tmp_path / "ready"
    disappeared = tmp_path / "disappeared"
    env, state = guardian_env
    env["GUARDIAN_DISAPPEAR_PATH"] = str(auth)
    env["GUARDIAN_READY_PATH"] = str(ready)
    env["GUARDIAN_DISAPPEARED_PATH"] = str(disappeared)
    code = "from pathlib import Path; import sys,time; Path(sys.argv[1]).touch(); time.sleep(60)"
    process, _ = _launch(root, tmp_path, (env, state), [sys.executable, "-c", code, str(ready)])
    try:
        _wait_for(ready, process)
        _wait_for(disappeared, process)
        assert process.poll() is None
        assert not auth.exists()
        assert process.stdin is not None
        process.stdin.close()
        assert process.wait(timeout=12) == 0
        assert _stored_snapshot(state) == {}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()


@pytest.mark.slow
def test_guardian_defers_snapshot_during_truncated_utf8_credential_write(
    tmp_path: Path, guardian_env: tuple[dict[str, str], Path]
) -> None:
    root = tmp_path / "proxy"
    auth = _seed_runtime(root)
    ready = tmp_path / "ready"
    allow_completion = tmp_path / "allow-completion"
    completed = tmp_path / "completed"
    decode_error = tmp_path / "decode-error"
    env, state = guardian_env
    env["GUARDIAN_DECODE_PATH"] = str(auth)
    env["GUARDIAN_DECODE_ERROR_PATH"] = str(decode_error)
    code = (
        "from pathlib import Path; import sys,time; "
        "auth=Path(sys.argv[1]); ready=Path(sys.argv[2]); "
        "allow=Path(sys.argv[3]); complete=Path(sys.argv[4]); "
        'auth.write_bytes(b\'{"token":"partial-\' + bytes((0xE2, 0x82))); '
        "ready.touch(); "
        "deadline=time.monotonic()+8; "
        "exec('while not Path(sys.argv[3]).exists() and "
        "time.monotonic()<deadline: time.sleep(.02)'); "
        "auth.write_text('{\"token\":\"finished-雪\"}', encoding='utf-8'); "
        "complete.touch(); time.sleep(60)"
    )
    process, _ = _launch(
        root,
        tmp_path,
        (env, state),
        [
            sys.executable,
            "-c",
            code,
            str(auth),
            str(ready),
            str(allow_completion),
            str(completed),
        ],
    )
    try:
        _wait_for(ready, process)
        _wait_for(decode_error, process)
        assert process.poll() is None, "guardian exited after decoding an incomplete credential"
        allow_completion.touch()
        _wait_for(completed, process)
        assert process.stdin is not None
        process.stdin.close()
        assert process.wait(timeout=12) == 0
        assert _stored_snapshot(state)["claude.json"] == '{"token":"finished-雪"}'
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
