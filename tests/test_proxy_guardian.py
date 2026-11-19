"""Real-process tests for the platform-independent proxy credential guardian."""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cutpoint.assistant import _proxy_guardian as guardian
from cutpoint.assistant.keys import KeyringError

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
        "from cutpoint.assistant import keys\n"
        "keys.os_backend = lambda module: module.get_keyring()\n",
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


# Synthetic secrets: a proxy-style 64-hex key and an OpenAI-shaped key.
_HEX_SECRET = "0123456789abcdef" * 4
_SK_SECRET = "sk-" + "SyntheticSecret" * 2


@pytest.mark.slow
def test_a_failing_daemon_s_redacted_output_tail_is_kept_for_the_error(
    tmp_path: Path, guardian_env: tuple[dict[str, str], Path]
) -> None:
    root = tmp_path / "proxy"
    _seed_runtime(root)
    script = (
        "import sys\n"
        f"print('loaded key {_HEX_SECRET} and {_SK_SECRET}', flush=True)\n"
        "print('fatal: synthetic listen error', file=sys.stderr, flush=True)\n"
        "sys.exit(3)\n"
    )
    process, _ = _launch(root, tmp_path, guardian_env, [sys.executable, "-c", script])
    assert process.wait(timeout=12) == 3
    tail = guardian.output_tail(root)
    assert "fatal: synthetic listen error" in tail
    assert "CLIProxyAPI output" in tail
    assert _HEX_SECRET not in tail and _SK_SECRET not in tail
    saved = (root / guardian.OUTPUT_FILE).read_text(encoding="utf-8")
    assert _HEX_SECRET not in saved and _SK_SECRET not in saved
    assert saved.count("[redacted]") == 2
    if os.name != "nt":
        assert (root / guardian.OUTPUT_FILE).stat().st_mode & 0o777 == 0o600


@pytest.mark.slow
def test_a_daemon_that_writes_crlf_leaves_clean_lines(
    tmp_path: Path, guardian_env: tuple[dict[str, str], Path]
) -> None:
    """A child that ends its lines with CRLF (as on Windows) leaves no \\r and no blank lines
    between the tail's lines, in the saved file or in the error."""
    root = tmp_path / "proxy"
    _seed_runtime(root)
    script = (
        "import sys\n"
        "for i in range(3):\n"
        "    sys.stdout.buffer.write(b'synthetic line %d\\r\\n' % i)\n"
        "sys.stdout.buffer.flush()\n"
        "sys.exit(3)\n"
    )
    process, _ = _launch(root, tmp_path, guardian_env, [sys.executable, "-c", script])
    assert process.wait(timeout=12) == 3
    want = [f"synthetic line {i}" for i in range(3)]
    saved = (root / guardian.OUTPUT_FILE).read_bytes()
    assert b"\r" not in saved and saved.decode("utf-8").split("\n") == want
    tail = guardian.output_tail(root)
    assert "\r" not in tail and tail.split("\n")[2:] == want


def test_crlf_and_a_stray_cr_never_make_blank_tail_lines(tmp_path: Path) -> None:
    tail = guardian._Tail()
    tail.drain(io.BytesIO(b"first\r\nsecond\r\r\nthird\r\n"))
    assert tail.text() == "first\nsecond\nthird"
    # a file saved before line ends were cleaned (Windows: \r\r\n)
    saved = tmp_path / guardian.OUTPUT_FILE
    saved.write_bytes(b"first\r\r\nsecond\r\r\n")
    if os.name != "nt":
        saved.chmod(0o600)
    assert guardian.output_tail(tmp_path).split("\n")[2:] == ["first", "second"]


@pytest.mark.slow
def test_a_chatty_daemon_starts_without_blocking_and_leaves_no_output_file(
    tmp_path: Path, guardian_env: tuple[dict[str, str], Path]
) -> None:
    root = tmp_path / "proxy"
    _seed_runtime(root)
    ready = tmp_path / "ready"
    (root / guardian.OUTPUT_FILE).write_text("an earlier run's output", encoding="utf-8")
    script = (
        "import sys, time\n"
        "line = 'x' * 1023 + '\\n'\n"
        "for _ in range(4096):\n"  # 4 MB on each stream: far beyond any pipe buffer
        "    sys.stdout.write(line); sys.stderr.write(line)\n"
        "sys.stdout.flush(); sys.stderr.flush()\n"
        "open(sys.argv[1], 'w').close()\n"
        "time.sleep(30)\n"
    )
    process, _ = _launch(root, tmp_path, guardian_env, [sys.executable, "-c", script, str(ready)])
    _wait_for(ready, process)
    assert not (root / guardian.OUTPUT_FILE).exists()  # removed at start
    assert process.stdin is not None
    process.stdin.close()
    process.wait(timeout=12)
    assert not (root / guardian.OUTPUT_FILE).exists()  # stopped, not failed: nothing kept
    assert guardian.output_tail(root) == ""


def test_the_output_tail_is_bounded_and_starts_at_a_whole_line() -> None:
    tail = guardian._Tail()
    tail.drain(io.BytesIO(b"".join(b"line %d\n" % i for i in range(1000))))
    text = tail.text()
    assert text.splitlines() == [f"line {i}" for i in range(990, 1000)]

    tail = guardian._Tail()
    long_line = b"y" * (guardian._TAIL_CHARS * 3) + b"\n"
    tail.drain(io.BytesIO(long_line * 5))
    assert len(tail.text()) <= guardian._TAIL_CHARS

    # a line longer than the ring holds: its kept end could be half a key, so it is dropped
    tail = guardian._Tail()
    pieces = guardian._TAIL_LINES + 2
    tail.drain(io.BytesIO(b"z" * (guardian._LINE_BYTES * pieces) + b"\nlast line\n"))
    assert tail.text() == "last line"


def test_the_output_is_redacted_before_it_is_cut(monkeypatch) -> None:
    monkeypatch.setattr(guardian.keys, "_SEEN", set())
    remembered = "synthetic-remembered-secret"
    guardian.keys._remember(remembered)
    text = "a" * guardian._TAIL_CHARS + f" {remembered} {_HEX_SECRET.upper()} done"
    clean = guardian.redact_output(text)
    assert len(clean) <= guardian._TAIL_CHARS
    assert remembered not in clean and _HEX_SECRET.upper() not in clean
    assert clean.endswith("[redacted] [redacted] done")


# One synthetic daemon log line per secret shape, and the secret that must not survive.
_DAEMON_SECRETS = [
    ('{"access_token":"eyJhbGciOiJTeW50aGV0aWMifQ.eyJzdWIiOiJ4In0.c3lu"}', "eyJhbGciOiJ"),
    ('{"refresh_token": "1//0gSyntheticRefresh-Value_01"}', "1//0gSynthetic"),
    ("x-goog-api-key: AIzaSyntheticGoogleKeyValue0001", "AIzaSynthetic"),
    ("GET /oauth/callback?code=4/0SyntheticCode-01&state=st8", "4/0SyntheticCode"),
    ("Authorization: Basic c3ludGhldGljOnVzZXI=", "c3ludGhldGljOnVzZXI="),
    ("password: shortpw1", "shortpw1"),
    ("refreshing with token=abc123def", "abc123def"),
    ('body={\\"refresh_token\\":\\"1//0gSyntheticEscaped-02\\"}', "1//0gSyntheticEscaped"),
    ("saved rt_SyntheticRefresh01 for account", "rt_SyntheticRefresh01"),
    ("session eyJhbGciOiJTeW50aCJ9.eyJzdWIiOiJzeW50aCJ9.U3ludGhTaWc-01 ok", "U3ludGhTaWc"),
    ("using key AIzaSyntheticGoogleKey_0123456789-abcdef ok", "AIzaSyntheticGoogleKey"),
]


@pytest.mark.parametrize(("line", "secret"), _DAEMON_SECRETS)
def test_daemon_output_masks_named_tokens_codes_and_logins(line: str, secret: str) -> None:
    clean = guardian.redact_output(f"time=1 level=info msg={line}")
    assert "[redacted]" in clean
    for part in (secret, secret[:6], secret[-6:]):
        assert part not in clean, clean


def test_daemon_output_masking_stays_linear_on_jwt_like_runs() -> None:
    text = "eyJ-" * (80 * 1024 // 4)
    started = time.perf_counter()
    guardian.redact_output(text)
    assert time.perf_counter() - started < 0.5


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


# A297: a failed periodic sync warns once per failure streak and is retried, the wait doubling
# per failure up to a minute; prepare() and finish() still fail closed. In-process, with a fake
# store, a fake daemon and a fake clock that each poll of the daemon moves on.


class _FakeStore:
    def __init__(self, sync_results: list[Exception | None], clock: list[float], **errors):
        self.sync_results = sync_results
        self.clock = clock
        self.times: list[float] = []
        self.prepare_error = errors.get("prepare")
        self.finish_error = errors.get("finish")
        self.finished = False

    def prepare(self) -> None:
        if self.prepare_error is not None:
            raise self.prepare_error

    def sync(self) -> None:
        n = len(self.times)
        self.times.append(self.clock[0])
        result = self.sync_results[n] if n < len(self.sync_results) else None
        if result is not None:
            raise result

    def finish(self) -> None:
        self.finished = True
        if self.finish_error is not None:
            raise self.finish_error


class _FakeDaemon:
    """Runs for ``seconds`` of the fake clock, then exits with 0."""

    def __init__(self, seconds: float, clock: list[float]) -> None:
        self.seconds = seconds
        self.clock = clock
        self.returncode: int | None = None
        self.terminated = False
        self.stdout = io.BytesIO()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None and self.clock[0] < self.seconds:
            self.clock[0] += timeout or 0
            raise subprocess.TimeoutExpired("daemon", timeout or 0)
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    kill = terminate


def _run_in_process(
    monkeypatch, tmp_path: Path, results: list[Exception | None], seconds: float, **errors
) -> tuple:
    clock = [0.0]
    store = _FakeStore(results, clock, **errors)
    daemon = _FakeDaemon(seconds, clock)

    def watch(fd, eof, started):
        started.set()

    monkeypatch.setattr(guardian, "_watch_stdin", watch)
    monkeypatch.setattr(guardian, "_parent_alive", lambda pid: True)
    monkeypatch.setattr(guardian, "CredentialStore", lambda root: store)
    monkeypatch.setattr(guardian, "time", type("T", (), {"monotonic": lambda: clock[0]}))
    monkeypatch.setattr(guardian.procs, "kill_with_parent", lambda child: None)
    monkeypatch.setattr(guardian.subprocess, "Popen", lambda *a, **k: daemon)
    monkeypatch.setattr(guardian.sys, "stdin", type("In", (), {"fileno": lambda self: 0})())
    code = guardian.main(["--root", str(tmp_path / "proxy"), "--parent-pid", "1", "--", "x"])
    return code, daemon, store


LOCKED = KeyringError("the keyring is locked")


def test_failed_periodic_sync_keeps_the_proxy_running_and_retries(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    code, daemon, store = _run_in_process(monkeypatch, tmp_path, [LOCKED] * 4, seconds=150)
    assert code == 0
    assert not daemon.terminated  # the daemon ran until it exited on its own
    assert store.times[:5] == [5, 15, 35, 75, 135]  # retried, waiting longer each time
    assert store.times[5:] == [140, 145, 150]  # every 5 s again after the success
    assert store.finished
    err = capsys.readouterr().err
    assert err.count("WARNING proxy credential sync failed") == 1  # once per streak, no spam
    assert "the keyring is locked" in err
    assert err.count("proxy credential sync recovered") == 1


def test_a_failure_streak_backs_off_to_one_sync_a_minute(monkeypatch, tmp_path: Path) -> None:
    code, daemon, store = _run_in_process(monkeypatch, tmp_path, [LOCKED] * 99, seconds=300)
    assert code == 0 and not daemon.terminated
    assert store.times == [5, 15, 35, 75, 135, 195, 255]  # not 60 keyring reads, one per 5 s


def test_each_new_failure_streak_warns_once(monkeypatch, tmp_path: Path, capsys) -> None:
    results = [LOCKED, LOCKED, None, OSError("x"), OSError("x")]
    code, daemon, store = _run_in_process(monkeypatch, tmp_path, results, seconds=80)
    assert code == 0 and not daemon.terminated
    assert store.times[:6] == [5, 15, 35, 40, 50, 70]  # the wait starts again at 5 s
    err = capsys.readouterr().err
    assert err.count("WARNING proxy credential sync failed") == 2
    assert err.count("proxy credential sync recovered") == 2


@pytest.mark.parametrize("step", ["prepare", "finish"])
def test_prepare_and_finish_still_fail_closed(monkeypatch, tmp_path: Path, capsys, step) -> None:
    code, daemon, _ = _run_in_process(monkeypatch, tmp_path, [], seconds=1, **{step: LOCKED})
    assert code == 1
    if step == "prepare":
        assert daemon.returncode is None  # never launched
    assert "WARNING" not in capsys.readouterr().err
