"""Tests for CLIProxyAPI daemon integration (cutpoint.assistant.cliproxy)."""

from __future__ import annotations

import ctypes
import hashlib
import http.server
import io
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path

import pytest

from cutpoint import cli
from cutpoint.assistant import cliproxy, keys
from cutpoint.assistant.proxy_credentials import CredentialStore


def _install_child_runtime(tmp_path: Path) -> None:
    module_dir = tmp_path / "test_child_modules"
    module_dir.mkdir(exist_ok=True)
    (module_dir / "synthetic_keyring.py").write_text(
        """import json, os
from pathlib import Path
from keyring.backend import KeyringBackend
class Backend(KeyringBackend):
    @property
    def priority(self):
        return 1
    def _read(self):
        path = Path(os.environ["CUTPOINT_TEST_KEYRING_FILE"])
        return json.loads(path.read_text()) if path.exists() else {}
    def _write(self, data):
        Path(os.environ["CUTPOINT_TEST_KEYRING_FILE"]).write_text(json.dumps(data))
    def get_password(self, service, username):
        return self._read().get(service + "::" + username)
    def set_password(self, service, username, value):
        if os.environ.get("CUTPOINT_TEST_KEYRING_FAIL_WRITE") == "1":
            raise RuntimeError("synthetic keyring write failure")
        data = self._read()
        data[service + "::" + username] = value
        self._write(data)
    def delete_password(self, service, username):
        data = self._read()
        data.pop(service + "::" + username, None)
        self._write(data)
""",
        encoding="utf-8",
    )
    (module_dir / "sitecustomize.py").write_text(
        """import os, subprocess, sys
from cutpoint.assistant.proxy_credentials import CredentialStore
CredentialStore._select_os_backend = staticmethod(lambda module: module.get_keyring())
_popen = subprocess.Popen
def _daemon_boundary(args, *pos, **kw):
    if (
        isinstance(args, (tuple, list))
        and len(args) > 1
        and os.path.samefile(args[0], sys.executable)
        and args[1] == "-config"
        and os.environ.get("CUTPOINT_TEST_DAEMON_SCRIPT")
    ):
        args = [sys.executable, os.environ["CUTPOINT_TEST_DAEMON_SCRIPT"], *args[1:]]
    return _popen(args, *pos, **kw)
subprocess.Popen = _daemon_boundary
""",
        encoding="utf-8",
    )


@pytest.fixture
def temp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "cutpoint_home"
    home.mkdir(parents=True)
    _install_child_runtime(tmp_path)
    monkeypatch.setenv("CUTPOINT_TEST_KEYRING_FILE", str(tmp_path / "keyring.json"))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "synthetic_keyring.Backend")
    source = str(Path(__file__).parents[1] / "src")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            filter(
                None,
                (
                    str(tmp_path / "test_child_modules"),
                    source,
                    os.environ.get("PYTHONPATH", ""),
                ),
            )
        ),
    )
    monkeypatch.setattr("cutpoint.paths.home", lambda: home)
    monkeypatch.setattr(
        CredentialStore, "_select_os_backend", staticmethod(lambda backend: backend)
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        monkeypatch.setenv("CUTPOINT_CLIPROXY_PORT", str(probe.getsockname()[1]))
    monkeypatch.delenv("CUTPOINT_CLIPROXY_BINARY", raising=False)
    monkeypatch.delenv("CUTPOINT_TEST_DAEMON_SCRIPT", raising=False)
    monkeypatch.delenv("CUTPOINT_TEST_KEYRING_FAIL_WRITE", raising=False)
    monkeypatch.setattr(cliproxy, "checkpoint", lambda root: None)
    return home


class FakeCredentialRing:
    def __init__(self) -> None:
        self.path = Path(os.environ["CUTPOINT_TEST_KEYRING_FILE"])

    @property
    def fail_new_writes(self) -> bool:
        return os.environ.get("CUTPOINT_TEST_KEYRING_FAIL_WRITE") == "1"

    @fail_new_writes.setter
    def fail_new_writes(self, enabled: bool) -> None:
        if enabled:
            os.environ["CUTPOINT_TEST_KEYRING_FAIL_WRITE"] = "1"
        else:
            os.environ.pop("CUTPOINT_TEST_KEYRING_FAIL_WRITE", None)

    def _read(self) -> dict[str, str]:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    @staticmethod
    def _key(service: str, username: str) -> str:
        return f"{service}::{username}"

    def get_password(self, service: str, username: str) -> str | None:
        return self._read().get(self._key(service, username))

    def set_password(self, service: str, username: str, value: str) -> None:
        data = self._read()
        if self.fail_new_writes and self._key(service, username) not in data:
            raise RuntimeError("synthetic keyring write failure")
        data[self._key(service, username)] = value
        self.path.write_text(json.dumps(data))

    def delete_password(self, service: str, username: str) -> None:
        data = self._read()
        data.pop(self._key(service, username), None)
        self.path.write_text(json.dumps(data))


@pytest.fixture
def credential_ring(monkeypatch: pytest.MonkeyPatch) -> FakeCredentialRing:
    ring = FakeCredentialRing()
    monkeypatch.setattr(keys, "_keyring", lambda: ring)
    return ring


def _seed_proxy_auth(root: Path, name: str, token: str) -> Path:
    path = root / "auths" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"refresh_token": token}), encoding="utf-8")
    store = CredentialStore(root)
    store.prepare()
    store.finish()
    return path


def _run_proxy_helper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script_text: str) -> Path:
    script = tmp_path / "proxy_helper.py"
    script.write_text(script_text, encoding="utf-8")
    monkeypatch.setenv("CUTPOINT_TEST_DAEMON_SCRIPT", str(script))
    monkeypatch.setenv("CUTPOINT_CLIPROXY_BINARY", sys.executable)
    return script


def test_port_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    assert cliproxy._validate_port(8317) == 8317
    assert cliproxy._validate_port("8317") == 8317

    with pytest.raises(cliproxy.ProxyError, match="must be between 1 and 65535"):
        cliproxy._validate_port(0)

    with pytest.raises(cliproxy.ProxyError, match="must be between 1 and 65535"):
        cliproxy._validate_port(70000)

    with pytest.raises(cliproxy.ProxyError, match="invalid port"):
        cliproxy._validate_port("not-a-number")

    monkeypatch.setenv("CUTPOINT_CLIPROXY_PORT", "invalid-port")
    with pytest.raises(cliproxy.ProxyError, match="invalid port"):
        cliproxy.port()


def test_key_file_permission_failure_is_not_treated_as_missing(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cliproxy._ensure_keys(target_port=8317)
    key_path = cliproxy.keys_path()
    original = key_path.read_text(encoding="utf-8")
    monkeypatch.setenv("CUTPOINT_CLIPROXY_PORT", "")

    def deny_file(_path: Path) -> None:
        raise PermissionError("cannot protect private Windows file")

    monkeypatch.setattr(cliproxy, "private_file", deny_file)
    with pytest.raises(PermissionError, match="cannot protect private Windows file"):
        cliproxy.port()
    with pytest.raises(PermissionError, match="cannot protect private Windows file"):
        cliproxy._ensure_keys(target_port=8317)
    assert key_path.read_text(encoding="utf-8") == original


def test_explicit_start_key_refresh_invalidates_previous_tokens(temp_home: Path) -> None:
    old_keys = cliproxy._ensure_keys(target_port=8317)
    fresh_keys = cliproxy._ensure_keys(target_port=8317, refresh=True)
    assert fresh_keys != old_keys
    assert cliproxy._read_keys_file() == (*fresh_keys, 8317)


def test_redact_secrets(temp_home: Path) -> None:
    mgmt, client = cliproxy._ensure_keys(target_port=8317)
    raw = (
        f"Error connecting with Bearer {mgmt} or Bearer my_super_secret_token_12345 "
        f"and client key {client} and token sk-12345678901234567890"
    )
    scrubbed = cliproxy.redact(raw)
    assert mgmt not in scrubbed
    assert client not in scrubbed
    assert "my_super_secret_token_12345" not in scrubbed
    assert "sk-12345678901234567890" not in scrubbed
    assert "[REDACTED]" in scrubbed


def test_status_offline_no_raise(temp_home: Path) -> None:
    st = cliproxy.status("claude-code")
    assert st["provider"] == "claude-code"
    assert st["logged_in"] is False
    assert st["email"] is None
    assert st["org"] is None
    assert "cutpoint proxy not running" in (st["error"] or "")

    all_st = cliproxy.statuses()
    assert len(all_st) == 3
    for s in all_st:
        assert s["logged_in"] is False
        assert "cutpoint proxy not running" in (s["error"] or "")


def test_status_token_free_allowlist(temp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cliproxy._ensure_keys(target_port=8317)

    fake_credentials = {
        "files": [
            {
                "id": "antigravity-user.json",
                "name": "antigravity-user.json",
                "provider": "antigravity",
                "type": "antigravity",
                "prefix": "antigravity",
                "status": "active",
                "disabled": False,
                "email": "user@example.com",
                "org": "MyOrg",
                "expires_at": 1800000000000,
                # Sensitive tokens that MUST NOT leak:
                "access_token": "ya29.secret_access_token_1234567890",
                "refresh_token": "secret_refresh_token_9876543210",
            }
        ]
    }

    def fake_mgmt_request(method: str, path: str, **kwargs: object) -> dict:
        if path == "/v8/management/credentials":
            return fake_credentials
        return {}

    monkeypatch.setattr(cliproxy, "_management_request", fake_mgmt_request)

    st = cliproxy.status("antigravity")
    assert st["provider"] == "antigravity"
    assert st["logged_in"] is True
    assert st["email"] == "user@example.com"
    assert st["org"] == "MyOrg"
    assert st["expired"] is False
    assert st["expires_ms"] == 1800000000000
    assert st["error"] is None

    st_json = json.dumps(st)
    assert "ya29" not in st_json
    assert "secret_access_token" not in st_json
    assert "secret_refresh_token" not in st_json
    assert "access_token" not in st


def test_status_multi_account_healthy_preferred(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cliproxy._ensure_keys(target_port=8317)

    fake_credentials = {
        "files": [
            {
                "id": "claude-expired.json",
                "name": "claude-expired.json",
                "provider": "claude",
                "status": "expired",
                "disabled": False,
                "email": "expired@example.com",
            },
            {
                "id": "claude-active.json",
                "name": "claude-active.json",
                "provider": "claude",
                "status": "active",
                "disabled": False,
                "email": "healthy@example.com",
            },
        ]
    }

    monkeypatch.setattr(
        cliproxy,
        "_management_request",
        lambda method, path, **kwargs: (
            fake_credentials if path == "/v8/management/credentials" else {}
        ),
    )

    st = cliproxy.status("claude-code")
    assert st["logged_in"] is True
    assert st["email"] == "healthy@example.com"
    assert st["expired"] is False


def test_status_invalid_provider() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        cliproxy.status("unknown-provider")


def test_start_login_poll_and_cancel(temp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cliproxy._ensure_keys(target_port=8317)

    deleted_sessions = []

    def fake_mgmt_request(
        method: str,
        path: str,
        json_data: dict | None = None,
        params: dict | None = None,
        **kwargs: object,
    ) -> dict:
        if path == "/v8/management/oauth/auth-url":
            assert json_data is not None
            assert json_data.get("provider") == "claude"
            assert json_data.get("is_webui") is True
            return {
                "status": "ok",
                "url": "https://claude.ai/oauth/authorize?state=xyz",
                "state": "xyz",
            }
        elif path == "/v8/management/oauth/status":
            assert params == {"state": "xyz"}
            return {"status": "ok"}
        elif path == "/v8/management/oauth/session" and method == "DELETE":
            deleted_sessions.append(params.get("state") if params else None)
            return {"status": "ok"}
        elif path == "/v8/management/credentials":
            return {"files": []}
        return {}

    monkeypatch.setattr(cliproxy, "_management_request", fake_mgmt_request)

    login_data = cliproxy.start_login("claude-code")
    assert login_data["url"] == "https://claude.ai/oauth/authorize?state=xyz"
    assert login_data["state"] == "xyz"

    poll_data = cliproxy.poll_login("xyz")
    assert poll_data["state"] == "done"
    assert poll_data["error"] is None

    cliproxy.cancel_login("xyz")
    assert deleted_sessions == ["xyz"]


def test_poll_login_prefix_failure_reports_error(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cliproxy._ensure_keys(target_port=8317)

    def fake_mgmt_request(method: str, path: str, **kwargs: object) -> dict:
        if path == "/v8/management/oauth/status":
            return {"status": "ok"}
        elif path == "/v8/management/credentials":
            return {"files": [{"name": "claude.json", "provider": "claude", "prefix": "wrong"}]}
        elif path == "/v8/management/credentials/fields":
            raise cliproxy.ProxyError("daemon refused prefix patch")
        return {}

    monkeypatch.setattr(cliproxy, "_management_request", fake_mgmt_request)

    poll_data = cliproxy.poll_login("xyz")
    assert poll_data["state"] == "error"
    assert poll_data["error"] is not None
    assert "daemon refused prefix patch" in poll_data["error"]


def test_logout_deletes_only_matching_credentials(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cliproxy._ensure_keys(target_port=8317)

    deleted_names = []

    def fake_mgmt_request(
        method: str, path: str, params: dict | None = None, **kwargs: object
    ) -> dict:
        if path == "/v8/management/credentials":
            if method == "GET":
                return {
                    "files": [
                        {"name": "claude-user.json", "provider": "claude"},
                        {"name": "codex-user.json", "provider": "codex"},
                    ]
                }
            elif method == "DELETE":
                assert params is not None
                deleted_names.append(params.get("name"))
                return {"status": "ok"}
        return {}

    monkeypatch.setattr(cliproxy, "_management_request", fake_mgmt_request)

    cliproxy.logout("claude-code")
    assert deleted_names == ["claude-user.json"]
    assert "codex-user.json" not in deleted_names


def test_list_models_unions_and_strips_prefix(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cliproxy._ensure_keys(target_port=8317)

    def fake_mgmt_request(
        method: str, path: str, params: dict | None = None, **kwargs: object
    ) -> dict:
        if path == "/v8/management/credentials":
            return {
                "files": [
                    {
                        "name": "antigravity-user1.json",
                        "provider": "antigravity",
                        "prefix": "antigravity",
                    },
                    {
                        "name": "antigravity-user2.json",
                        "provider": "antigravity",
                        "prefix": "antigravity",
                    },
                ]
            }
        elif path == "/v8/management/credentials/models":
            fname = params.get("name") if params else ""
            if fname == "antigravity-user1.json":
                return {
                    "models": [
                        {
                            "id": "antigravity/claude-3-5-sonnet",
                            "display_name": "Claude 3.5 Sonnet",
                        },
                        {"id": "gemini-1.5-pro", "display_name": "Gemini 1.5 Pro"},
                    ]
                }
            elif fname == "antigravity-user2.json":
                return {
                    "models": [
                        {"id": "gemini-1.5-pro", "display_name": "Gemini 1.5 Pro"},
                        {"id": "gemini-1.5-flash", "display_name": "Gemini 1.5 Flash"},
                    ]
                }
        return {}

    monkeypatch.setattr(cliproxy, "_management_request", fake_mgmt_request)

    models = cliproxy.list_models("antigravity")
    # Union of models without duplicates
    assert len(models) == 3
    ids = [m["id"] for m in models]
    assert ids == ["claude-3-5-sonnet", "gemini-1.5-pro", "gemini-1.5-flash"]


def test_list_models_error_propagates(temp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cliproxy._ensure_keys(target_port=8317)

    def fake_mgmt_request(method: str, path: str, **kwargs: object) -> dict:
        if path == "/v8/management/credentials":
            return {"files": [{"name": "claude.json", "provider": "claude", "prefix": "claude"}]}
        elif path == "/v8/management/credentials/models":
            raise cliproxy.ProxyError("daemon model registry error")
        return {}

    monkeypatch.setattr(cliproxy, "_management_request", fake_mgmt_request)

    with pytest.raises(
        cliproxy.ProxyError, match="failed to load models for Claude Code subscription"
    ):
        cliproxy.list_models("claude-code")


def test_prefix_sync_failure_propagates(temp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cliproxy._ensure_keys(target_port=8317)

    def fake_mgmt_request(method: str, path: str, **kwargs: object) -> dict:
        if path == "/v8/management/credentials":
            return {"files": [{"name": "claude.json", "provider": "claude", "prefix": "wrong"}]}
        elif path == "/v8/management/credentials/fields":
            raise cliproxy.ProxyError("patch failure")
        return {}

    monkeypatch.setattr(cliproxy, "_management_request", fake_mgmt_request)

    with pytest.raises(cliproxy.ProxyError, match="patch failure"):
        cliproxy._ensure_credential_prefixes()


def test_chat_provider_isolation_model_prefix(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cliproxy._ensure_keys(target_port=8317)
    monkeypatch.setattr(cliproxy, "_ensure_credential_prefixes", lambda: None)

    intercepted_payload = {}

    def fake_client_request(
        method: str, path: str, json_data: dict | None = None, **kwargs: object
    ) -> dict:
        assert path == "/v1/chat/completions"
        assert json_data is not None
        intercepted_payload.update(json_data)
        return {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}

    monkeypatch.setattr(cliproxy, "_client_request", fake_client_request)

    cliproxy.chat(
        "antigravity",
        {"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert intercepted_payload["model"] == "antigravity/claude-3-5-sonnet"

    cliproxy.chat(
        "claude-code",
        {"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert intercepted_payload["model"] == "claude/claude-3-5-sonnet"


def test_generate_config_and_permissions(temp_home: Path) -> None:
    mgmt, client = cliproxy._ensure_keys(target_port=8317)
    tls_cert, tls_key = cliproxy.proxy_tls.ensure_identity(cliproxy.cliproxy_dir())
    cfg = cliproxy._generate_config(
        port=8317,
        auth_dir_path=cliproxy.auth_dir(),
        mgmt_key=mgmt,
        client_key=client,
        tls_cert_path=tls_cert,
        tls_key_path=tls_key,
    )
    assert cfg["server"]["tls"] == {
        "enable": True,
        "cert": str(tls_cert),
        "key": str(tls_key),
    }
    assert cfg["config-version"] == 8
    assert cfg["server"]["host"] == "127.0.0.1"
    assert cfg["server"]["port"] == 8317
    assert cfg["server"]["commercial-mode"] is True
    assert cfg["server"]["discovery"]["enabled"] is False
    assert cfg["management"]["allow-remote"] is False
    assert cfg["management"]["disable-control-panel"] is True
    assert cfg["management"]["disable-auto-update-panel"] is True
    assert cfg["access"]["api-keys"] == [client]
    assert cfg["plugins"]["enabled"] is False
    assert cfg["routing"]["force-model-prefix"] is True
    assert cfg["upstream"]["claude"]["disable-claude-cloak-mode"] is True
    assert cfg["upstream"]["codex"]["disable-codex-cloaking"] is True
    assert cfg["observability"]["logs"]["request-log"] is False

    kp = cliproxy.keys_path()
    assert kp.exists()
    if os.name != "nt":
        # Windows does not support POSIX permission bits.
        assert stat.S_IMODE(kp.stat().st_mode) == 0o600
        assert stat.S_IMODE(kp.parent.stat().st_mode) == 0o700


def test_live_daemon_lock_refuses_before_secret_materialization(temp_home: Path) -> None:
    root = cliproxy.cliproxy_dir()
    root.mkdir(parents=True)
    with (
        cliproxy.swap_lock(root / "daemon.run"),
        pytest.raises(cliproxy.ProxyError, match="another cutpoint proxy is already running"),
    ):
        cliproxy.run()
    assert not cliproxy.keys_path().exists()
    assert not cliproxy.config_path().exists()
    assert not (root / "proxy-cert.pem").exists()
    assert not (root / "proxy-key.pem").exists()


def test_run_missing_custom_binary_fails(temp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUTPOINT_CLIPROXY_BINARY", "/nonexistent/cli-proxy-api")
    with pytest.raises(
        cliproxy.ProxyError, match="configured CUTPOINT_CLIPROXY_BINARY does not exist"
    ):
        cliproxy.run()


def test_run_nonzero_exit_fails(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _run_proxy_helper(monkeypatch, tmp_path, "raise SystemExit(2)\n")
    monkeypatch.setattr(cliproxy, "_ensure_credential_prefixes", lambda: None)
    with pytest.raises(cliproxy.ProxyError, match="exited with non-zero code"):
        cliproxy.run()


def test_empty_port_env_ignored(temp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUTPOINT_CLIPROXY_PORT", "")
    assert cliproxy.port() == cliproxy.DEFAULT_PORT


def test_unowned_port_selection_prefers_environment_over_saved(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cliproxy._ensure_keys(target_port=19318)
    monkeypatch.setenv("CUTPOINT_CLIPROXY_PORT", "19317")
    assert not cliproxy._is_daemon_owned()
    assert cliproxy.port() == 19317
    monkeypatch.delenv("CUTPOINT_CLIPROXY_PORT")
    assert cliproxy.port() == 19318


@pytest.mark.slow
def test_owned_daemon_uses_explicit_port_over_environment(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An environment override must not redirect an owned daemon's selected endpoint."""
    with socket.socket() as conflicting, socket.socket() as listener:
        conflicting.bind(("127.0.0.1", 0))
        listener.bind(("127.0.0.1", 0))
        explicit_port = listener.getsockname()[1]
        listener.close()
        monkeypatch.setenv("CUTPOINT_CLIPROXY_PORT", str(conflicting.getsockname()[1]))
        _run_proxy_helper(monkeypatch, tmp_path, "import time\ntime.sleep(60)\n")

        def check_endpoint() -> None:
            if not cliproxy._is_daemon_owned():
                raise cliproxy.ProxyConnectionError("waiting for guardian ownership")
            assert cliproxy.port() == explicit_port
            raise KeyboardInterrupt

        monkeypatch.setattr(cliproxy, "_ensure_credential_prefixes", check_endpoint)
        with pytest.raises(KeyboardInterrupt):
            cliproxy.run(port=explicit_port)
        assert cliproxy._read_keys_file()[2] == explicit_port
        config = json.loads(cliproxy.config_path().read_text(encoding="utf-8"))
        assert config["server"]["port"] == explicit_port


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX executable bit; Windows os.access(X_OK) only checks existence",
)
def test_run_nonexecutable_binary_fails(temp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    non_exec = temp_home / "non_exec"
    non_exec.write_text("#!/bin/sh\necho hi\n")
    non_exec.chmod(0o644)
    monkeypatch.setenv("CUTPOINT_CLIPROXY_BINARY", str(non_exec))

    with pytest.raises(
        cliproxy.ProxyError, match="configured CUTPOINT_CLIPROXY_BINARY is not executable"
    ):
        cliproxy.run()


def test_install_binary_download_error_wrapped(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_download(*args: object, **kwargs: object) -> Path:
        raise RuntimeError("network failure during download")

    monkeypatch.setattr("cutpoint.events.setup.download", failing_download)
    with pytest.raises(cliproxy.ProxyError, match="failed to download CLIProxyAPI release"):
        cliproxy._install_binary(temp_home / "tools" / "cliproxy")


def test_run_subprocess_isolates_cwd_and_env(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Simulate caller working directory with an untrusted .env
    caller_dir = tmp_path / "caller_project"
    caller_dir.mkdir()
    (caller_dir / ".env").write_text("PGSTORE_DSN=postgres://attacker@remote:5432/db\n")
    monkeypatch.chdir(caller_dir)

    monkeypatch.setenv("PGSTORE_DSN", "postgres://inherited-override")
    monkeypatch.setenv("HOME_JWT", "inherited-jwt")
    monkeypatch.setenv("MANAGEMENT_PASSWORD", "inherited-management-override")
    monkeypatch.setenv("HTTP_PROXY", "http://corporate-proxy:8080")

    report_file = tmp_path / "child_report.json"
    _run_proxy_helper(
        monkeypatch,
        tmp_path,
        f"""import json, os
report = {{
    "cwd": os.getcwd(),
    "env": {{k: os.environ[k] for k in ("PGSTORE_DSN", "HOME_JWT", "MANAGEMENT_PASSWORD", "HTTP_PROXY") if k in os.environ}},
}}
with open(r"{report_file}", "w") as f:
    json.dump(report, f)
""",
    )
    monkeypatch.setattr(cliproxy, "_ensure_credential_prefixes", lambda: None)
    cliproxy.run()

    child_data = json.loads(report_file.read_text())

    expected_cwd = (cliproxy.cliproxy_dir() / "run").resolve()
    assert Path(child_data["cwd"]).resolve() == expected_cwd
    assert Path(child_data["cwd"]).resolve() != caller_dir.resolve()

    child_env = child_data["env"]
    assert "PGSTORE_DSN" not in child_env
    assert "HOME_JWT" not in child_env
    assert "MANAGEMENT_PASSWORD" not in child_env
    assert child_env.get("HTTP_PROXY") == "http://corporate-proxy:8080"


def test_run_fails_closed_if_run_dir_has_untrusted_dotenv(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin = temp_home / "fake_bin"
    fake_bin.write_text("#!/bin/sh\nexit 0\n")
    fake_bin.chmod(0o755)
    monkeypatch.setenv("CUTPOINT_CLIPROXY_BINARY", str(fake_bin))

    run_dir = cliproxy.cliproxy_dir() / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".env").write_text("SOME_OVERRIDE=1\n")

    with pytest.raises(
        cliproxy.ProxyError, match=r"untrusted \.env found in daemon working directory"
    ):
        cliproxy.run()


@pytest.mark.parametrize(
    (
        "files",
        "expect_logged_in",
        "expect_expired",
    ),
    [
        # Cooldown: status=error keeps logged_in=True so preflight passes to proxy
        (
            [
                {
                    "name": "c.json",
                    "provider": "claude",
                    "status": "error",
                    "disabled": False,
                    "email": "cool@example.com",
                }
            ],
            True,
            False,
        ),
        # Multi-account: earlier expired, later cooldown -> selects cooldown
        (
            [
                {"name": "c1.json", "provider": "claude", "status": "expired", "disabled": False},
                {
                    "name": "c2.json",
                    "provider": "claude",
                    "status": "error",
                    "disabled": False,
                    "email": "cool2@example.com",
                },
            ],
            True,
            False,
        ),
        # Expired credentials remain disconnected.
        (
            [{"name": "c.json", "provider": "claude", "status": "expired", "disabled": False}],
            False,
            True,
        ),
        # Disabled credentials remain disconnected.
        (
            [{"name": "c.json", "provider": "claude", "status": "active", "disabled": True}],
            False,
            False,
        ),
    ],
)
def test_subscription_status_distinguishes_cooldown_from_logout(
    files: list[dict],
    expect_logged_in: bool,
    expect_expired: bool,
    temp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient cooldowns must not turn valid saved accounts into a logout."""
    cliproxy._ensure_keys(target_port=8317)
    monkeypatch.setattr(
        cliproxy,
        "_management_request",
        lambda method, path, **kwargs: (
            {"files": files} if path == "/v8/management/credentials" else {}
        ),
    )
    st = cliproxy.status("claude-code")
    assert st["logged_in"] is expect_logged_in
    assert st["expired"] is expect_expired


def test_install_binary_synthetic_tar_gz_valid(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    tar_path = tmp_path / "valid.tar.gz"
    bin_name = "cli-proxy-api.exe" if sys.platform == "win32" else "cli-proxy-api"

    with tarfile.open(tar_path, "w:gz") as tf:
        bin_data = b"synthetic binary content"
        ti = tarfile.TarInfo(name=bin_name)
        ti.size = len(bin_data)
        tf.addfile(ti, io.BytesIO(bin_data))

        lic_data = b"MIT License"
        ti_lic = tarfile.TarInfo(name="LICENSE")
        ti_lic.size = len(lic_data)
        tf.addfile(ti_lic, io.BytesIO(lic_data))

    actual_sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        cliproxy,
        "_detect_platform_asset",
        lambda: ("fake.tar.gz", actual_sha),
    )
    monkeypatch.setattr(
        "cutpoint.events.setup.download",
        lambda url, dst, sha256: dst.write_bytes(tar_path.read_bytes()),
    )

    installed = cliproxy._install_binary(temp_home / "tools" / "cliproxy")
    assert installed.exists()
    assert installed.name == bin_name
    assert installed.read_bytes() == b"synthetic binary content"
    if os.name != "nt":
        assert os.access(installed, os.X_OK)
    assert (temp_home / "tools" / "cliproxy" / "LICENSE").read_bytes() == b"MIT License"


def test_install_binary_synthetic_zip_valid(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    zip_path = tmp_path / "valid.zip"
    bin_name = "cli-proxy-api.exe" if sys.platform == "win32" else "cli-proxy-api"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(bin_name, b"synthetic zip binary")
        zf.writestr("LICENSE", b"synthetic license")

    actual_sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        cliproxy,
        "_detect_platform_asset",
        lambda: ("fake.zip", actual_sha),
    )
    monkeypatch.setattr(
        "cutpoint.events.setup.download",
        lambda url, dst, sha256: dst.write_bytes(zip_path.read_bytes()),
    )

    installed = cliproxy._install_binary(temp_home / "tools" / "cliproxy")
    assert installed.exists()
    assert installed.name == bin_name
    assert installed.read_bytes() == b"synthetic zip binary"


@pytest.mark.parametrize(
    ("archive_suffix", "write_archive"),
    [
        (
            ".tar.gz",
            lambda path: _write_tar(path, [("other.txt", b"not the binary")]),
        ),
        (
            ".zip",
            lambda path: _write_zip(path, [("other.txt", b"not the binary")]),
        ),
    ],
)
def test_install_binary_missing_member_fails(
    temp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    archive_suffix: str,
    write_archive,
) -> None:
    archive = tmp_path / f"missing{archive_suffix}"
    write_archive(archive)
    monkeypatch.setattr(
        cliproxy,
        "_detect_platform_asset",
        lambda: (f"fake{archive_suffix}", "any_hash"),
    )
    monkeypatch.setattr(
        "cutpoint.events.setup.download",
        lambda url, dst, sha256: dst.write_bytes(archive.read_bytes()),
    )

    with pytest.raises(cliproxy.ProxyError, match="member missing in archive"):
        cliproxy._install_binary(temp_home / "tools" / "cliproxy")


def _write_tar(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members:
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))


def _write_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members:
            zf.writestr(name, data)


def test_install_binary_path_traversal_tar_fails(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tar_path = tmp_path / "traversal.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        data = b"evil payload"
        ti = tarfile.TarInfo(name="../cli-proxy-api")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))

    monkeypatch.setattr(
        cliproxy,
        "_detect_platform_asset",
        lambda: ("fake.tar.gz", "any_hash"),
    )
    monkeypatch.setattr(
        "cutpoint.events.setup.download",
        lambda url, dst, sha256: dst.write_bytes(tar_path.read_bytes()),
    )

    with pytest.raises(cliproxy.ProxyError, match="unsafe member path"):
        cliproxy._install_binary(temp_home / "tools" / "cliproxy")


def test_install_binary_path_traversal_zip_fails(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    zip_path = tmp_path / "traversal.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../cli-proxy-api", b"evil payload")

    monkeypatch.setattr(
        cliproxy,
        "_detect_platform_asset",
        lambda: ("fake.zip", "any_hash"),
    )
    monkeypatch.setattr(
        "cutpoint.events.setup.download",
        lambda url, dst, sha256: dst.write_bytes(zip_path.read_bytes()),
    )

    with pytest.raises(cliproxy.ProxyError, match="unsafe member path"):
        cliproxy._install_binary(temp_home / "tools" / "cliproxy")


def test_run_interrupt_cleans_and_reaps_child(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pid_file = tmp_path / "daemon.pid"
    _run_proxy_helper(
        monkeypatch,
        tmp_path,
        f"import os, time\nPath = __import__('pathlib').Path\nPath({str(pid_file)!r}).write_text(str(os.getpid()))\ntime.sleep(60)\n",
    )

    def interrupt_after_spawn() -> None:
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists(), "guardian did not launch the daemon fixture"
        raise KeyboardInterrupt

    monkeypatch.setattr(cliproxy, "_ensure_credential_prefixes", interrupt_after_spawn)
    with pytest.raises(KeyboardInterrupt):
        cliproxy.run()

    child_pid = int(pid_file.read_text())
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel32.GetExitCodeProcess.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        )
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel32.OpenProcess(0x1000, False, child_pid)
        if handle:
            try:
                exit_code = ctypes.c_ulong()
                assert kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                assert exit_code.value != 259
            finally:
                kernel32.CloseHandle(handle)
        else:
            assert ctypes.get_last_error() == 87
    else:
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)


def test_run_startup_failure_checkpoints_before_clearing_runtime_files(
    temp_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    credential_ring: FakeCredentialRing,
) -> None:
    root = cliproxy.cliproxy_dir()
    auth = _seed_proxy_auth(root, "codex-user.json", "durable-before-startup")
    assert not auth.exists()
    monkeypatch.setenv("CUTPOINT_CLIPROXY_BINARY", str(temp_home / "missing-proxy"))

    with pytest.raises(cliproxy.ProxyError, match="does not exist"):
        cliproxy.run()

    assert not auth.exists()
    store = CredentialStore(root)
    store.prepare()
    assert json.loads(auth.read_text(encoding="utf-8"))["refresh_token"] == "durable-before-startup"
    store.finish()


def test_run_persists_daemon_refresh_and_cleans_runtime_files(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch, credential_ring: FakeCredentialRing
) -> None:
    root = cliproxy.cliproxy_dir()
    auth = _seed_proxy_auth(root, "codex-user.json", "old-token")
    monkeypatch.setattr(cliproxy, "_ensure_credential_prefixes", lambda: None)
    _run_proxy_helper(
        monkeypatch,
        temp_home,
        "\n".join(
            (
                "from pathlib import Path",
                "import json,time",
                f"Path({str(auth)!r}).write_text(json.dumps({{'refresh_token':'rotated-token'}}))",
                "time.sleep(0.8)",
            )
        ),
    )

    cliproxy.run()

    assert not auth.exists()
    store = CredentialStore(root)
    store.prepare()
    assert json.loads(auth.read_text(encoding="utf-8"))["refresh_token"] == "rotated-token"
    store.finish()


@pytest.mark.parametrize("operation", ["login", "logout"])
def test_credential_checkpoint_failure_is_safe(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    def fail_checkpoint(root: Path) -> None:
        raise cliproxy.KeyringError(
            "the keyring could not checkpoint proxy credentials (RuntimeError)"
        )

    monkeypatch.setattr(cliproxy, "checkpoint", fail_checkpoint)
    if operation == "login":
        monkeypatch.setattr(
            cliproxy,
            "_management_request",
            lambda *args, **kwargs: {"status": "ok"},
        )
        monkeypatch.setattr(cliproxy, "_ensure_credential_prefixes", lambda: None)
        with pytest.raises(cliproxy.ProxyError) as exc:
            cliproxy.poll_login("state")
    else:
        monkeypatch.setattr(
            cliproxy,
            "_management_request",
            lambda *args, **kwargs: {"files": []},
        )
        with pytest.raises(cliproxy.ProxyError) as exc:
            cliproxy.logout("codex")

    assert str(exc.value) == (
        "secure credential storage failed: "
        "the keyring could not checkpoint proxy credentials (RuntimeError)"
    )


def test_cli_proxy_interrupt_exit_130_and_child_cleanup(tmp_path: Path) -> None:
    home = tmp_path / "home"
    mock_bin = tmp_path / "waiting_proxy.py"
    pid_file = tmp_path / "child.pid"
    mock_bin.write_text(
        f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n"
    )
    _install_child_runtime(tmp_path)
    env = os.environ.copy()
    env.pop("CUTPOINT_CLIPROXY_PORT", None)
    env["CUTPOINT_HOME"] = str(home)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        env["CUTPOINT_CLIPROXY_PORT"] = str(probe.getsockname()[1])
    env["CUTPOINT_CLIPROXY_BINARY"] = sys.executable
    env["CUTPOINT_TEST_DAEMON_SCRIPT"] = str(mock_bin)
    env["CUTPOINT_TEST_KEYRING_FILE"] = str(tmp_path / "keyring.json")
    env["PYTHON_KEYRING_BACKEND"] = "synthetic_keyring.Backend"
    env["PYTHONPATH"] = os.pathsep.join(
        (str(tmp_path / "test_child_modules"), str(Path(__file__).parents[1] / "src"))
    )
    bootstrap = (
        "import cutpoint.assistant.cliproxy as cp\n"
        "cp._ensure_credential_prefixes=lambda:None\n"
        "from cutpoint.cli import main; raise SystemExit(main(['proxy']))\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", bootstrap],
        env=env,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0,
    )
    try:
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists()
        child_pid = int(pid_file.read_text())
        parent.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        assert parent.wait(timeout=10) == 130
        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
            kernel32.GetExitCodeProcess.argtypes = (
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_ulong),
            )
            kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
            handle = kernel32.OpenProcess(0x1000, False, child_pid)
            if handle:
                try:
                    exit_code = ctypes.c_ulong()
                    assert kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                    assert exit_code.value != 259
                finally:
                    kernel32.CloseHandle(handle)
            else:
                assert ctypes.get_last_error() == 87
        else:
            with pytest.raises(ProcessLookupError):
                os.kill(child_pid, 0)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()


def test_parent_hardkill_leaves_no_plaintext_after_guardian_finalizes(
    temp_home: Path,
    credential_ring: FakeCredentialRing,
    tmp_path: Path,
) -> None:
    root = cliproxy.cliproxy_dir()
    auth = _seed_proxy_auth(root, "codex-user.json", "old-token")
    port = cliproxy.port()
    pid_file = tmp_path / "daemon.pid"
    script = tmp_path / "hardkill_daemon.py"
    script.write_text(
        "\n".join(
            (
                "import json, socket, sys, time",
                "from pathlib import Path",
                "cfg = json.loads(Path(sys.argv[2]).read_text())",
                "auth = Path(cfg['oauth']['auth-dir']) / 'codex-user.json'",
                "auth.write_text(json.dumps({'refresh_token':'rotated-before-hardkill'}))",
                "listener = socket.socket()",
                "listener.bind(('127.0.0.1', cfg['server']['port']))",
                "listener.listen()",
                f"Path({str(pid_file)!r}).write_text(str(__import__('os').getpid()))",
                "time.sleep(60)",
            )
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["CUTPOINT_HOME"] = str(temp_home)
    env["CUTPOINT_CLIPROXY_BINARY"] = sys.executable
    env["CUTPOINT_TEST_DAEMON_SCRIPT"] = str(script)
    bootstrap = (
        "import cutpoint.assistant.cliproxy as cp\n"
        "cp._ensure_credential_prefixes=lambda:None\n"
        "cp.run()\n"
    )
    parent = subprocess.Popen([sys.executable, "-c", bootstrap], env=env)
    try:
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists()
        parent.kill()
        parent.wait(timeout=10)

        deadline = time.monotonic() + 20
        while auth.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not auth.exists()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))

        store = CredentialStore(root)
        store.prepare()
        assert json.loads(auth.read_text(encoding="utf-8"))["refresh_token"] == (
            "rotated-before-hardkill"
        )
        store.finish()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()


def test_cli_proxy_interrupt_returns_exit_130(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_interrupt(port: int | None = None) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cliproxy, "run", raise_interrupt)
    rc = cli.main(["proxy"])
    assert rc == 130


def test_custom_port_18317_status_without_starting_or_download(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUTPOINT_CLIPROXY_PORT", "18317")
    assert cliproxy.port() == 18317
    assert cliproxy.base_url() == "https://127.0.0.1:18317"

    st = cliproxy.status("codex")
    assert st["provider"] == "codex"
    assert st["logged_in"] is False
    assert "cutpoint proxy not running" in (st["error"] or "")

    tools_dir = temp_home / "tools" / "cliproxy"
    assert not tools_dir.exists()


def test_statuses_distinguishes_connection_error_vs_http_error(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cliproxy._ensure_keys(target_port=18317)
    monkeypatch.setattr(cliproxy, "_is_daemon_owned", lambda: True)

    # 1. HTTP error (e.g. 401 Unauthorized)
    def http_401_request(method: str, path: str, **kwargs: object) -> dict:
        raise cliproxy.ProxyHTTPError("proxy error 401: Unauthorized", status_code=401)

    monkeypatch.setattr(cliproxy, "_management_request", http_401_request)
    all_st = cliproxy.statuses()
    for s in all_st:
        assert s["logged_in"] is False
        assert s["error"] == "proxy error 401: Unauthorized"

    # 2. Connection error (offline / connection refused)
    def conn_error_request(method: str, path: str, **kwargs: object) -> dict:
        raise cliproxy.ProxyConnectionError("proxy connection failed: Connection refused")

    monkeypatch.setattr(cliproxy, "_management_request", conn_error_request)
    all_st2 = cliproxy.statuses()
    for s in all_st2:
        assert s["logged_in"] is False
        assert "cutpoint proxy not running" in (s["error"] or "")


def test_ownership_guard_blocks_sending_management_key_to_unrelated_server(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = None

    received_auth_headers: list[str] = []

    class DummyHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            auth = self.headers.get("Authorization", "")
            received_auth_headers.append(auth)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"files": []}')

        def log_message(self, format_str: str, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), DummyHandler)
    monkeypatch.setenv("CUTPOINT_CLIPROXY_PORT", str(server.server_port))
    cliproxy._ensure_keys(target_port=server.server_port)
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    try:
        assert not cliproxy._is_daemon_owned()

        with pytest.raises(cliproxy.ProxyError, match="cutpoint proxy not running"):
            cliproxy._management_request("GET", "/v8/management/credentials")

        with pytest.raises(cliproxy.ProxyError, match="cutpoint proxy not running"):
            cliproxy._client_request("POST", "/v1/chat/completions", json_data={})

        st = cliproxy.statuses()
        for s in st:
            assert "cutpoint proxy not running" in (s["error"] or "")

        assert len(received_auth_headers) == 0
    finally:
        server.server_close()


def test_prefix_setup_not_called_in_chat_or_list_models(
    temp_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cliproxy._ensure_keys(target_port=8317)

    prefix_called: list[bool] = []
    monkeypatch.setattr(
        cliproxy,
        "_ensure_credential_prefixes",
        lambda: prefix_called.append(True),
    )

    fake_creds = {"files": [{"name": "claude.json", "provider": "claude", "prefix": "claude"}]}
    fake_models = {
        "models": [{"id": "claude/claude-3-5-sonnet", "display_name": "Claude 3.5 Sonnet"}]
    }

    monkeypatch.setattr(
        cliproxy,
        "_management_request",
        lambda method, path, **kwargs: (
            fake_creds if path == "/v8/management/credentials" else fake_models
        ),
    )
    monkeypatch.setattr(
        cliproxy,
        "_client_request",
        lambda method, path, **kwargs: {"choices": []},
    )

    models = cliproxy.list_models("claude-code")
    assert len(models) == 1
    assert prefix_called == []

    cliproxy.chat("claude-code", {"model": "claude-3-5-sonnet", "messages": []})
    assert prefix_called == []
