"""Tests for CLIProxyAPI daemon integration (cutpoint.assistant.cliproxy)."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from cutpoint.assistant import cliproxy


@pytest.fixture
def temp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "cutpoint_home"
    home.mkdir(parents=True)
    monkeypatch.setattr("cutpoint.paths.home", lambda: home)
    monkeypatch.setattr("cutpoint.paths.tools_dir", lambda: home / "tools")
    monkeypatch.delenv("CUTPOINT_CLIPROXY_PORT", raising=False)
    monkeypatch.delenv("CUTPOINT_CLIPROXY_BINARY", raising=False)
    return home


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
    cfg = cliproxy._generate_config(
        port=8317,
        auth_dir_path=cliproxy.auth_dir(),
        mgmt_key=mgmt,
        client_key=client,
    )
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


def test_run_missing_custom_binary_fails(temp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUTPOINT_CLIPROXY_BINARY", "/nonexistent/cli-proxy-api")
    with pytest.raises(
        cliproxy.ProxyError, match="configured CUTPOINT_CLIPROXY_BINARY does not exist"
    ):
        cliproxy.run()


def test_run_nonzero_exit_fails(temp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUTPOINT_CLIPROXY_BINARY", sys.executable)
    real_popen = subprocess.Popen

    def popen_exit_2(_args, **kwargs):
        return real_popen([sys.executable, "-c", "raise SystemExit(2)"], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen_exit_2)

    with pytest.raises(cliproxy.ProxyError, match="exited with non-zero code 2"):
        cliproxy.run()


def test_empty_port_env_ignored(temp_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUTPOINT_CLIPROXY_PORT", "")
    assert cliproxy.port() == cliproxy.DEFAULT_PORT


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


@pytest.mark.skipif(os.name == "nt", reason="fixture uses a POSIX executable script")
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

    # Create mock binary that dumps observed cwd and env to JSON
    report_file = tmp_path / "child_report.json"
    mock_bin = temp_home / "record_runner"
    mock_bin.write_text(
        f"""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

report = {{
    "cwd": os.getcwd(),
    "env": {{k: os.environ[k] for k in ("PGSTORE_DSN", "HOME_JWT", "MANAGEMENT_PASSWORD", "HTTP_PROXY") if k in os.environ}},
}}
with open(r"{report_file}", "w") as f:
    json.dump(report, f)
sys.exit(0)
"""
    )
    mock_bin.chmod(0o755)
    monkeypatch.setenv("CUTPOINT_CLIPROXY_BINARY", str(mock_bin))

    cliproxy.run()

    child_data = json.loads(report_file.read_text())

    expected_cwd = str((cliproxy.cliproxy_dir() / "run").resolve())
    assert child_data["cwd"] == expected_cwd
    assert child_data["cwd"] != str(caller_dir.resolve())

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
