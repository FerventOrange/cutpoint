"""CLIProxyAPI daemon client and lifecycle management.

Connects Cutpoint assistant subscription providers (Claude Code, ChatGPT Codex,
Google Antigravity) to an external authenticated CLIProxyAPI loopback proxy.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import re
import secrets
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import webbrowser
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from cutpoint import folders, paths
from cutpoint.assistant import proxy_tls
from cutpoint.assistant.keys import KeyringError
from cutpoint.assistant.private_files import private_directory, private_file
from cutpoint.assistant.proxy_credentials import checkpoint
from cutpoint.errors import FolderBusyError

logger = logging.getLogger(__name__)

PROVIDERS = ("claude-code", "codex", "antigravity")

TITLES = {
    "claude-code": "Claude Code subscription",
    "codex": "ChatGPT Codex subscription",
    "antigravity": "Google Antigravity subscription",
}

CLI_PROVIDERS = {
    "claude-code": "claude",
    "codex": "codex",
    "antigravity": "antigravity",
}

PREFIX_BY_PROVIDER = {
    "claude-code": "claude",
    "codex": "codex",
    "antigravity": "antigravity",
}

DEFAULT_PORT = 18317
# what a request says while no Cutpoint proxy runs (``cutpoint ui`` starts one when needed)
NOT_RUNNING = "cutpoint proxy not running; run 'cutpoint proxy' to authenticate"
PINNED_VERSION = "8.0.47"

PLATFORM_ASSETS: dict[tuple[str, str], tuple[str, str]] = {
    ("linux", "x86_64"): (
        "CLIProxyAPI_8.0.47_linux_amd64_no-plugin.tar.gz",
        "805d4027a67d9c022fd738d772bd4581baac080785649385afc24d9edf83a86f",
    ),
    ("linux", "aarch64"): (
        "CLIProxyAPI_8.0.47_linux_aarch64_no-plugin.tar.gz",
        "27d798132c9e07f24f2fad2727fb6a3c276d029f4d8a402b431ec84277230cc5",
    ),
    ("darwin", "arm64"): (
        "CLIProxyAPI_8.0.47_darwin_aarch64.tar.gz",
        "a5b260ec0f843d2291488acd76e5595232da8f3b8ec2d334154f1570f5f3ca08",
    ),
    ("darwin", "x86_64"): (
        "CLIProxyAPI_8.0.47_darwin_amd64.tar.gz",
        "2e72cb7b28905b9135563e4fc01af00009ce72a444729eb95bdb634543ce7d41",
    ),
    ("win32", "amd64"): (
        "CLIProxyAPI_8.0.47_windows_amd64.zip",
        "8e23cbb2f9f695411b7df46320855820b352fb886c3c3bdf45ab4634f2f14632",
    ),
    ("win32", "arm64"): (
        "CLIProxyAPI_8.0.47_windows_aarch64.zip",
        "9a947eec7234e5a2e2ce57a9fe55c82328fc823ffd2f41cca85f9da41d11eae0",
    ),
}
_PROHIBITED_ENV_PREFIXES = (
    "pgstore_",
    "gitstore_",
    "objectstore_",
)

_PROHIBITED_ENV_EXACT = {
    "deploy",
    "home_jwt",
    "management_password",
}


def _sanitize_daemon_env() -> dict[str, str]:
    """Strip pinned upstream store, deployment and management authentication overrides."""
    cleaned: dict[str, str] = {}
    for k, v in os.environ.items():
        k_lower = k.lower()
        if (
            any(k_lower.startswith(p) for p in _PROHIBITED_ENV_PREFIXES)
            or k_lower in _PROHIBITED_ENV_EXACT
        ):
            continue
        cleaned[k] = v
    return cleaned


BASE_RELEASE_URL = (
    f"https://github.com/router-for-me/CLIProxyAPI/releases/download/v{PINNED_VERSION}"
)


class ProxyError(RuntimeError):
    """CLIProxyAPI error; safe to display (secrets and tokens redacted)."""


class ProxyConnectionError(ProxyError):
    """CLIProxyAPI connection error (daemon offline or unreachable)."""


class ProxyHTTPError(ProxyError):
    """CLIProxyAPI daemon responded with an HTTP error status."""

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


class LoginError(ProxyError):
    """A terminal login (:func:`login_interactive`) did not start, failed or timed out."""


def _credential_store_error(err: Exception) -> ProxyError:
    detail = str(err) if isinstance(err, KeyringError) else type(err).__name__
    return ProxyError(f"secure credential storage failed: {detail}")


def cliproxy_dir() -> Path:
    """Directory for cutpoint cliproxy keys, config, and auth records."""
    return paths.home() / "cliproxy"


def auth_dir() -> Path:
    """Directory holding provider credentials managed by CLIProxyAPI."""
    return cliproxy_dir() / "auths"


def keys_path() -> Path:
    """Path to private JSON keys file."""
    return cliproxy_dir() / "keys.json"


def config_path() -> Path:
    """Path to private CLIProxyAPI config JSON file."""
    return cliproxy_dir() / "cliproxy.json"


def _validate_port(val: Any) -> int:
    try:
        p = int(val)
    except (TypeError, ValueError) as err:
        raise ProxyError(f"invalid port {val!r}: must be an integer between 1 and 65535") from err
    if not (1 <= p <= 65535):
        raise ProxyError(f"invalid port {p}: must be between 1 and 65535")
    return p


def port() -> int:
    """Effective port for CLIProxyAPI loopback daemon."""
    if _is_daemon_owned():
        owned_keys = _read_keys_file()
        if owned_keys:
            return owned_keys[2]
    env_port = os.environ.get("CUTPOINT_CLIPROXY_PORT")
    if env_port:
        return _validate_port(env_port)
    kp = keys_path()
    if kp.exists() or kp.is_symlink():
        private_directory(kp.parent)
        private_file(kp)
        try:
            with kp.open(encoding="utf-8") as f:
                data = json.load(f)
            stored_port = data.get("port")
            if stored_port:
                return _validate_port(stored_port)
        except (ValueError, json.JSONDecodeError):
            pass
    return DEFAULT_PORT


def base_url() -> str:
    """Base loopback URL of the running proxy daemon."""
    return f"https://127.0.0.1:{port()}"


_BEARER_RE = re.compile(r"(Bearer\s+)[a-zA-Z0-9_\-\.\$]+", re.IGNORECASE)
_KEY_PARAM_RE = re.compile(
    r"((?:secret-key|api-key|secret_key|api_key|token|password)[\"':\s=]+)[a-zA-Z0-9_\-\.\$]+",
    re.IGNORECASE,
)
_RAW_TOKEN_RE = re.compile(r"\b(?:sk-[a-zA-Z0-9_\-]{16,}|ya29\.[a-zA-Z0-9_\-]{20,})\b")


def redact(text: str) -> str:
    """Scrub management keys, client API keys, and bearer tokens from message text."""
    if not text:
        return ""
    keys = _read_keys_file()
    if keys:
        mgmt, client = keys[0], keys[1]
        if mgmt:
            text = text.replace(mgmt, "[REDACTED]")
        if client:
            text = text.replace(client, "[REDACTED]")
    text = _BEARER_RE.sub(r"\1[REDACTED]", text)
    text = _KEY_PARAM_RE.sub(r"\1[REDACTED]", text)
    text = _RAW_TOKEN_RE.sub("[REDACTED]", text)
    return text


def _write_private_file(path: Path, content: str) -> None:
    private_directory(path.parent)
    if path.exists() or path.is_symlink():
        private_file(path)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        os.close(fd)
        fd = -1
        private_file(temp_path)
        with temp_path.open("w", encoding="utf-8") as f:
            f.write(content)
        temp_path.replace(path)
    finally:
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(OSError):
            temp_path.unlink()


def _read_keys_file() -> tuple[str, str, int] | None:
    kp = keys_path()
    if not kp.exists() and not kp.is_symlink():
        return None
    private_directory(kp.parent)
    private_file(kp)
    try:
        with kp.open(encoding="utf-8") as f:
            data = json.load(f)
        mgmt = data.get("management_key", "")
        client = data.get("client_api_key", "")
        p = int(data.get("port") or DEFAULT_PORT)
        if mgmt and client:
            return mgmt, client, p
    except (ValueError, json.JSONDecodeError):
        return None
    return None


def _ensure_keys(target_port: int | None = None) -> tuple[str, str]:
    p = target_port or port()
    mgmt = secrets.token_hex(32)
    client = secrets.token_hex(32)
    _write_private_file(
        keys_path(),
        json.dumps(
            {"management_key": mgmt, "client_api_key": client, "port": p},
            indent=2,
        ),
    )
    return mgmt, client


def _load_keys() -> tuple[str, str]:
    res = _read_keys_file()
    if not res:
        raise ProxyConnectionError(NOT_RUNNING)
    return res[0], res[1]


def _session() -> requests.Session:
    cert_path = cliproxy_dir() / "proxy-cert.pem"
    try:
        private_file(cert_path)
    except OSError as err:
        raise ProxyError("local proxy TLS identity is unavailable; restart the proxy") from err
    s = requests.Session()
    s.trust_env = False
    s.verify = str(cert_path)
    return s


def _is_daemon_owned() -> bool:
    """Whether the CLIProxyAPI daemon on the current port is owned by this Cutpoint installation."""
    lock = folders.lock_path(cliproxy_dir() / "daemon.run")
    if lock is None:
        return False
    found = folders._read_lock(lock)
    if not found or found[1] is None:
        return False
    return not folders._lock_stale(lock, found[1])


def _management_request(
    method: str,
    path: str,
    json_data: dict | None = None,
    params: dict | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    if not _is_daemon_owned():
        raise ProxyConnectionError(NOT_RUNNING)
    mgmt_key, _ = _load_keys()
    url = f"{base_url().rstrip('/')}/{path.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {mgmt_key}",
        "Content-Type": "application/json",
    }
    s = _session()
    try:
        resp = s.request(
            method,
            url,
            json=json_data,
            params=params,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
        )
    except requests.exceptions.SSLError as err:
        raise ProxyError("proxy TLS identity validation failed; refusing the request") from err
    except requests.RequestException as err:
        raise ProxyConnectionError(redact(f"proxy connection failed: {err}")) from err

    if not resp.ok:
        try:
            body = resp.json()
            err_msg = body.get("error") or body.get("message") or resp.text
        except (ValueError, json.JSONDecodeError):
            err_msg = resp.text
        clean_msg = redact(f"proxy error {resp.status_code}: {err_msg}")
        raise ProxyHTTPError(clean_msg, resp.status_code)
    try:
        return resp.json()
    except (ValueError, json.JSONDecodeError) as err:
        raise ProxyError(redact(f"proxy returned invalid JSON: {err}")) from err


def _client_request(
    method: str,
    path: str,
    json_data: dict | None = None,
    params: dict | None = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    if not _is_daemon_owned():
        raise ProxyConnectionError(NOT_RUNNING)
    _, client_key = _load_keys()
    url = f"{base_url().rstrip('/')}/{path.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {client_key}",
        "Content-Type": "application/json",
    }
    s = _session()
    try:
        resp = s.request(
            method,
            url,
            json=json_data,
            params=params,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
        )
    except requests.exceptions.SSLError as err:
        raise ProxyError("proxy TLS identity validation failed; refusing the request") from err
    except requests.RequestException as err:
        raise ProxyConnectionError(redact(f"proxy inference request failed: {err}")) from err

    if not resp.ok:
        try:
            body = resp.json()
            err_msg = body.get("error") or body.get("message") or resp.text
        except (ValueError, json.JSONDecodeError):
            err_msg = resp.text
        clean_msg = redact(f"proxy inference error {resp.status_code}: {err_msg}")
        raise ProxyHTTPError(clean_msg, resp.status_code)

    try:
        return resp.json()
    except (ValueError, json.JSONDecodeError) as err:
        raise ProxyError(redact(f"proxy returned invalid JSON: {err}")) from err


def _ensure_credential_prefixes() -> None:
    data = _management_request("GET", "/v8/management/credentials")
    files = data.get("files") or []
    for f in files:
        if not isinstance(f, dict):
            continue
        filename = f.get("name") or f.get("id")
        provider_type = f.get("provider") or f.get("type")
        current_prefix = f.get("prefix")
        if provider_type not in ("claude", "codex", "antigravity") or not filename:
            continue
        expected_prefix = provider_type
        if current_prefix != expected_prefix:
            _management_request(
                "PATCH",
                "/v8/management/credentials/fields",
                json_data={"name": filename, "prefix": expected_prefix},
            )


def statuses() -> list[dict[str, Any]]:
    """Return status dictionaries for all supported subscription providers in one call."""
    try:
        _load_keys()
        data = _management_request("GET", "/v8/management/credentials", timeout=3.0)
        files = data.get("files") or []
    except ProxyHTTPError as err:
        return [
            {
                "provider": p,
                "logged_in": False,
                "email": None,
                "org": None,
                "expired": False,
                "expires_ms": None,
                "error": str(err),
            }
            for p in PROVIDERS
        ]
    except ProxyConnectionError:
        return [
            {
                "provider": p,
                "logged_in": False,
                "email": None,
                "org": None,
                "expired": False,
                "expires_ms": None,
                "error": NOT_RUNNING,
            }
            for p in PROVIDERS
        ]
    except (ProxyError, OSError) as err:
        return [
            {
                "provider": p,
                "logged_in": False,
                "email": None,
                "org": None,
                "expired": False,
                "expires_ms": None,
                "error": redact(str(err)),
            }
            for p in PROVIDERS
        ]

    out = []
    for p in PROVIDERS:
        cli_p = CLI_PROVIDERS[p]
        matching = [
            f
            for f in files
            if isinstance(f, dict) and (f.get("provider") == cli_p or f.get("type") == cli_p)
        ]
        if not matching:
            out.append(
                {
                    "provider": p,
                    "logged_in": False,
                    "email": None,
                    "org": None,
                    "expired": False,
                    "expires_ms": None,
                    "error": None,
                }
            )
            continue

        # Select active credential first; keep cooldown (status: error) connected to let proxy report status
        enabled = [f for f in matching if not bool(f.get("disabled", False))]
        connected = [f for f in enabled if str(f.get("status", "")).lower() != "expired"]
        record = next(
            (f for f in connected if str(f.get("status", "")).lower() in ("active", "ok", "")),
            (connected or enabled or matching)[0],
        )

        email = record.get("email")
        org = record.get("org") or record.get("organization")
        raw_status = str(record.get("status", "")).lower()
        disabled = bool(record.get("disabled", False))
        expired = raw_status == "expired"
        logged_in = (not disabled) and (not expired)

        expires_ms = None
        if record.get("expires_at"):
            with contextlib.suppress(TypeError, ValueError):
                expires_ms = int(record["expires_at"])
        elif record.get("expires_in"):
            with contextlib.suppress(TypeError, ValueError):
                expires_ms = int(time.time() * 1000) + int(record["expires_in"]) * 1000

        error_msg = None
        if expired:
            error_msg = f"the {TITLES.get(p, p)} login was rejected; log in again"
        elif disabled:
            error_msg = f"{TITLES.get(p, p)} credential is disabled"

        out.append(
            {
                "provider": p,
                "logged_in": logged_in,
                "email": email,
                "org": org,
                "expired": expired,
                "expires_ms": expires_ms,
                "error": error_msg,
            }
        )
    return out


def status(provider: str) -> dict[str, Any]:
    """Return token-free status dictionary for provider.

    Does not raise when proxy daemon is offline; returns actionable guidance instead.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; must be one of {PROVIDERS}")

    for s in statuses():
        if s.get("provider") == provider:
            return s
    return {
        "provider": provider,
        "logged_in": False,
        "email": None,
        "org": None,
        "expired": False,
        "expires_ms": None,
        "error": None,
    }


def start_login(provider: str) -> dict[str, str]:
    """Initiate OAuth authorization for provider and return authorize URL and state."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; must be one of {PROVIDERS}")
    cli_p = CLI_PROVIDERS[provider]
    resp = _management_request(
        "POST",
        "/v8/management/oauth/auth-url",
        json_data={"provider": cli_p, "is_webui": True},
    )
    url = resp.get("url")
    state = resp.get("state")
    if not url or not state:
        raise ProxyError(f"failed to start login for {provider}: missing url or state from proxy")
    return {"url": url, "state": state}


def poll_login(state: str) -> dict[str, Any]:
    """Poll OAuth completion status for state; returns state ('pending', 'done', 'error') and error."""
    try:
        resp = _management_request("GET", "/v8/management/oauth/status", params={"state": state})
    except ProxyError as err:
        return {"state": "error", "error": redact(str(err))}

    status_str = str(resp.get("status", "")).lower()
    if status_str == "wait":
        return {"state": "pending", "error": None}
    elif status_str == "ok":
        try:
            _ensure_credential_prefixes()
        except ProxyError as err:
            return {
                "state": "error",
                "error": redact(f"failed to configure credential prefix: {err}"),
            }
        try:
            checkpoint(cliproxy_dir())
        except KeyringError as err:
            raise _credential_store_error(err) from None
        return {"state": "done", "error": None}
    elif status_str == "error":
        err_msg = resp.get("error") or "login failed"
        return {"state": "error", "error": redact(err_msg)}
    return {"state": "pending", "error": None}


def cancel_login(state: str) -> None:
    """Cancel pending OAuth login session on daemon."""
    if not state:
        return
    _management_request("DELETE", "/v8/management/oauth/session", params={"state": state})


LOGIN_TIMEOUT_S = 300.0  # how long login_interactive waits for the browser sign-in


def login_interactive(provider: str, *, open_browser: bool = True) -> dict[str, Any]:
    """Log in to ``provider`` from a terminal (``cutpoint login``): start the OAuth login, log
    its URL (and open it in the browser with ``open_browser``), poll once a second until it
    is done and return :func:`status`. A login that fails, times out or is interrupted is
    cancelled on the daemon.

    Raises:
        LoginError: The login could not start, failed or timed out.
        ProxyError: Polling or the status after the login failed otherwise.
    """
    try:
        res = start_login(provider)
    except ProxyError as err:
        raise LoginError(f"could not start login for {provider}: {err}") from err
    url = res.get("url")
    state = res.get("state")
    if not url or not state:
        raise LoginError(f"invalid login response from proxy for {provider}")
    logger.info("open this URL to log in: %s", url)
    success = False
    try:
        if open_browser:
            with contextlib.suppress(Exception):
                webbrowser.open(url)
        deadline = time.monotonic() + LOGIN_TIMEOUT_S
        while time.monotonic() < deadline:
            time.sleep(1.0)
            poll = poll_login(state)
            p_state = poll.get("state")
            if p_state == "done":
                st = status(provider)
                success = True
                return st
            if p_state == "error":
                raise LoginError(f"login failed: {poll.get('error') or 'unknown error'}")
        raise LoginError("login timed out")
    finally:
        if not success:
            with contextlib.suppress(Exception):
                cancel_login(state)


def logout(provider: str) -> None:
    """Delete dedicated credentials for provider from CLIProxyAPI."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; must be one of {PROVIDERS}")
    cli_p = CLI_PROVIDERS[provider]
    data = _management_request("GET", "/v8/management/credentials")
    files = data.get("files") or []
    for f in files:
        if not isinstance(f, dict):
            continue
        p = f.get("provider") or f.get("type")
        if p == cli_p:
            filename = f.get("name") or f.get("id")
            if filename:
                _management_request(
                    "DELETE",
                    "/v8/management/credentials",
                    params={"name": filename},
                )
    try:
        checkpoint(cliproxy_dir())
    except KeyringError as err:
        raise _credential_store_error(err) from None


def list_models(provider: str) -> list[dict[str, str]]:
    """List available models for provider, unioning healthy accounts and stripped of routing prefixes."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; must be one of {PROVIDERS}")
    cli_p = CLI_PROVIDERS[provider]
    prefix = PREFIX_BY_PROVIDER[provider]

    data = _management_request("GET", "/v8/management/credentials")
    files = data.get("files") or []
    matching = [
        f
        for f in files
        if isinstance(f, dict) and (f.get("provider") == cli_p or f.get("type") == cli_p)
    ]
    if not matching:
        raise ProxyError(f"{TITLES[provider]}: run cutpoint login {provider} first")

    # Union models across matching credentials
    seen_ids: set[str] = set()
    out: list[dict[str, str]] = []
    pfx = f"{prefix}/"

    for cred in matching:
        filename = cred.get("name") or cred.get("id")
        if not filename:
            continue
        try:
            models_data = _management_request(
                "GET",
                "/v8/management/credentials/models",
                params={"name": filename},
            )
        except ProxyError as err:
            raise ProxyError(
                f"failed to load models for {TITLES.get(provider, provider)}: {err}"
            ) from err
        for m in models_data.get("models") or []:
            if not isinstance(m, dict):
                continue
            raw_id = m.get("id", "")
            clean_id = raw_id[len(pfx) :] if raw_id.startswith(pfx) else raw_id
            if clean_id and clean_id not in seen_ids:
                seen_ids.add(clean_id)
                name = m.get("display_name") or clean_id
                out.append({"id": clean_id, "name": name})

    return out


def chat(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Forward OpenAI chat completions payload with provider model prefix isolation."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; must be one of {PROVIDERS}")
    prefix = PREFIX_BY_PROVIDER[provider]

    req_body = dict(payload)
    model = str(req_body.get("model", ""))
    pfx = f"{prefix}/"
    if not model.startswith(pfx):
        req_body["model"] = f"{pfx}{model}"

    return _client_request("POST", "/v1/chat/completions", json_data=req_body)


def _detect_platform_asset() -> tuple[str, str]:
    os_name = sys.platform
    machine = platform.machine().lower()
    if os_name == "linux":
        if machine in ("x86_64", "amd64"):
            key = ("linux", "x86_64")
        elif machine in ("aarch64", "arm64"):
            key = ("linux", "aarch64")
        else:
            raise ProxyError(f"unsupported linux architecture: {machine}")
    elif os_name == "darwin":
        key = ("darwin", "arm64") if machine in ("arm64", "aarch64") else ("darwin", "x86_64")
    elif os_name == "win32":
        key = ("win32", "arm64") if machine in ("arm64", "aarch64") else ("win32", "amd64")
    else:
        raise ProxyError(f"unsupported operating system: {os_name}")

    if key not in PLATFORM_ASSETS:
        raise ProxyError(f"no pinned binary for platform {key}")
    return PLATFORM_ASSETS[key]


def _install_binary(dst_dir: Path) -> Path:
    from cutpoint.events.setup import download  # noqa: PLC0415

    filename, sha256 = _detect_platform_asset()
    url = f"{BASE_RELEASE_URL}/{filename}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_dir.chmod(0o700)
    archive_dst = dst_dir / filename
    try:
        download(url=url, dst=archive_dst, sha256=sha256)
    except (RuntimeError, OSError) as err:
        raise ProxyError(f"failed to download CLIProxyAPI release: {err}") from err

    bin_name = "cli-proxy-api.exe" if sys.platform == "win32" else "cli-proxy-api"
    final_bin = dst_dir / bin_name
    final_license = dst_dir / "LICENSE"
    tmp_bin = dst_dir / f"{bin_name}.tmp"
    tmp_license = dst_dir / "LICENSE.tmp"

    found_bin = False
    found_license = False

    try:
        if filename.endswith(".zip"):
            with zipfile.ZipFile(archive_dst, "r") as zf:
                for member in zf.namelist():
                    p_member = Path(member)
                    if p_member.is_absolute() or ".." in p_member.parts:
                        raise ProxyError(
                            f"failed to extract {filename}: unsafe member path {member!r}"
                        )
                    base_name = p_member.name
                    if base_name in (bin_name, "cli-proxy-api", "cli-proxy-api.exe"):
                        with zf.open(member) as src, tmp_bin.open("wb") as dst:
                            dst.write(src.read())
                        found_bin = True
                    elif base_name == "LICENSE":
                        with zf.open(member) as src, tmp_license.open("wb") as dst:
                            dst.write(src.read())
                        found_license = True
        else:
            with tarfile.open(archive_dst, "r:*") as tf:
                for member in tf.getmembers():
                    p_member = Path(member.name)
                    if p_member.is_absolute() or ".." in p_member.parts:
                        raise ProxyError(
                            f"failed to extract {filename}: unsafe member path {member.name!r}"
                        )
                    base_name = p_member.name
                    if base_name in (bin_name, "cli-proxy-api", "cli-proxy-api.exe"):
                        f = tf.extractfile(member)
                        if f is not None:
                            with tmp_bin.open("wb") as dst:
                                dst.write(f.read())
                            found_bin = True
                    elif base_name == "LICENSE":
                        f = tf.extractfile(member)
                        if f is not None:
                            with tmp_license.open("wb") as dst:
                                dst.write(f.read())
                            found_license = True

        if not found_bin or not tmp_bin.exists():
            raise ProxyError(
                f"failed to extract {bin_name} from {filename}: member missing in archive"
            )

        tmp_bin.chmod(0o755)
        tmp_bin.replace(final_bin)
        if found_license and tmp_license.exists():
            tmp_license.replace(final_license)
    except (OSError, tarfile.TarError, zipfile.BadZipFile, ProxyError) as err:
        with contextlib.suppress(OSError):
            if tmp_bin.exists():
                tmp_bin.unlink()
            if tmp_license.exists():
                tmp_license.unlink()
        if isinstance(err, ProxyError):
            raise
        raise ProxyError(f"failed to extract CLIProxyAPI: {err}") from err

    return final_bin


def _generate_config(
    port: int,
    auth_dir_path: Path,
    mgmt_key: str,
    client_key: str,
    tls_cert_path: Path,
    tls_key_path: Path,
) -> dict[str, Any]:
    return {
        "config-version": 8,
        "server": {
            "host": "127.0.0.1",
            "port": port,
            "tls": {
                "enable": True,
                "cert": str(tls_cert_path),
                "key": str(tls_key_path),
            },
            "commercial-mode": True,
            "discovery": {
                "enabled": False,
            },
        },
        "management": {
            "allow-remote": False,
            "secret-key": mgmt_key,
            "disable-control-panel": True,
            "disable-auto-update-panel": True,
        },
        "access": {
            "api-keys": [client_key],
        },
        "oauth": {
            "auth-dir": str(auth_dir_path),
        },
        "plugins": {
            "enabled": False,
        },
        "routing": {
            "force-model-prefix": True,
            "retry": {
                "request-retry": 0,
            },
        },
        "upstream": {
            "claude": {
                "disable-claude-cloak-mode": True,
                "disable-cloaking-model-list": True,
            },
            "codex": {
                "disable-codex-cloaking": True,
            },
        },
        "observability": {
            "logs": {
                "debug": False,
                "logging-to-file": False,
                "request-log": False,
            },
            "usage": {
                "usage-statistics-enabled": False,
            },
            "pprof": {
                "enable": False,
            },
        },
    }


_STARTUP_TIMEOUT = 120.0  # covers a keyring unlock prompt, a virus scan or a slow disk
_STARTUP_NOTICE = 5.0
_STARTUP_RETRY = 0.2


def _wait_until_ready(
    proc: subprocess.Popen[Any], *, cancel: threading.Event | None = None
) -> None:
    """Wait while the supervisor is alive until the daemon answers and prefixes are set.

    Returns early when the supervisor exits with code 0 (the caller then sees that code);
    raises ProxyError at once on a non-zero exit, after ``_STARTUP_TIMEOUT`` seconds, and
    once ``cancel`` is set (the UI is stopping).
    """
    started = time.monotonic()
    noticed = False
    while True:
        if cancel is not None and cancel.is_set():
            raise ProxyError("the server is stopping")
        rc = proc.poll()
        if rc is not None:
            if rc:
                raise ProxyError(
                    f"CLIProxyAPI supervisor exited with non-zero code {rc} before the proxy "
                    "was ready"
                )
            return
        try:
            _ensure_credential_prefixes()
            return
        except ProxyConnectionError as err:
            waited = time.monotonic() - started
            if waited >= _STARTUP_TIMEOUT:
                raise ProxyConnectionError(
                    f"CLIProxyAPI did not become ready within {_STARTUP_TIMEOUT:.0f} s"
                ) from err
            if not noticed and waited >= _STARTUP_NOTICE:
                logger.info(
                    "Waiting for proxy: CLIProxyAPI is still starting (up to %.0f s)",
                    _STARTUP_TIMEOUT,
                )
                noticed = True
        # Waiting on the process (not sleeping) notices an early exit at once.
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_STARTUP_RETRY)


def _resolve_port(port: int | None) -> int:
    """The port a new proxy listens on: ``port``, else CUTPOINT_CLIPROXY_PORT, else the saved one."""
    if port is not None:
        return _validate_port(port)
    if os.environ.get("CUTPOINT_CLIPROXY_PORT"):
        return _validate_port(os.environ["CUTPOINT_CLIPROXY_PORT"])
    existing_keys = _read_keys_file()
    return existing_keys[2] if existing_keys else DEFAULT_PORT


def _prepare_launch(home_dir: Path, resolved_port: int) -> tuple[list[str], Path]:
    """Check nothing else holds the port, write fresh keys and the config, find (or download)
    the binary; return the guardian's command line and the daemon's working folder. Call it
    while holding the ``launch`` lock."""
    if _is_daemon_owned():
        raise ProxyError("another cutpoint proxy is already running")
    auths = auth_dir()
    private_directory(auths)
    run_dir = home_dir / "run"
    private_directory(run_dir)
    if (run_dir / ".env").exists():
        raise ProxyError(f"untrusted .env found in daemon working directory: {run_dir}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", resolved_port))
    except OSError as err:
        raise ProxyError(
            f"port {resolved_port} is already in use; refusing to send Cutpoint credentials"
        ) from err

    tls_cert, tls_key = proxy_tls.ensure_identity(home_dir)
    mgmt_key, client_key = _ensure_keys(target_port=resolved_port)
    cfg_path = config_path()
    cfg_dict = _generate_config(
        port=resolved_port,
        auth_dir_path=auths,
        mgmt_key=mgmt_key,
        client_key=client_key,
        tls_cert_path=tls_cert,
        tls_key_path=tls_key,
    )
    _write_private_file(cfg_path, json.dumps(cfg_dict, indent=2))

    env_bin = os.environ.get("CUTPOINT_CLIPROXY_BINARY")
    if env_bin:
        bin_file = Path(env_bin).expanduser().resolve()
        if not bin_file.exists():
            raise ProxyError(f"configured CUTPOINT_CLIPROXY_BINARY does not exist: {bin_file}")
        if not os.access(bin_file, os.X_OK):
            raise ProxyError(f"configured CUTPOINT_CLIPROXY_BINARY is not executable: {bin_file}")
    else:
        bin_file = _install_binary(paths.tools_dir() / "cliproxy")

    cmd = [str(bin_file), "-config", str(cfg_path), "-no-browser", "--local-model"]
    logger.info("Starting CLIProxyAPI on 127.0.0.1:%d", resolved_port)
    guardian_cmd = [
        sys.executable,
        "-m",
        "cutpoint.assistant._proxy_guardian",
        "--root",
        str(home_dir),
        "--parent-pid",
        str(os.getpid()),
        "--",
        *cmd,
    ]
    return guardian_cmd, run_dir


def _spawn_guardian(guardian_cmd: list[str], run_dir: Path) -> subprocess.Popen[Any]:
    """Start the guardian in its own process group (Windows) or session (POSIX), so the
    console's Ctrl+C does not reach it. It stops CLIProxyAPI and saves the credentials when
    its stdin closes, which also happens when this process dies."""
    return subprocess.Popen(
        guardian_cmd,
        cwd=run_dir,
        env=_sanitize_daemon_env(),
        stdin=subprocess.PIPE,
        close_fds=True,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0,
        start_new_session=os.name != "nt",
    )


def _stop_guardian(proc: subprocess.Popen[Any]) -> int:
    """Close the guardian's stdin (it then stops CLIProxyAPI and saves the credentials) and
    wait for it; terminate it after 30 s.

    Raises:
        ProxyError: It did not exit within 10 s of that either.
    """
    if proc.stdin is not None and not proc.stdin.closed:
        with contextlib.suppress(OSError):
            proc.stdin.close()
    try:
        return proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired as err:
            raise ProxyError("proxy supervisor did not finish credential cleanup") from err


def run(port: int | None = None) -> None:
    """Run CLIProxyAPI in the foreground as an authenticated local loopback proxy."""
    resolved_port = _resolve_port(port)
    home_dir = cliproxy_dir()
    private_directory(home_dir)
    try:
        with folders.swap_lock(home_dir / "launch"):
            guardian_cmd, run_dir = _prepare_launch(home_dir, resolved_port)
            proc: subprocess.Popen[Any] | None = None
            orig_handlers: dict[int, Any] = {}

            def _handle_signal(signum: int, frame: Any) -> None:
                raise KeyboardInterrupt

            if threading.current_thread() is threading.main_thread():
                signals_to_hook = [signal.SIGINT, signal.SIGTERM]
                if hasattr(signal, "SIGBREAK"):
                    signals_to_hook.append(signal.SIGBREAK)
                for sig in signals_to_hook:
                    with contextlib.suppress(ValueError, OSError):
                        orig_handlers[sig] = signal.signal(sig, _handle_signal)

            completed = False
            try:
                proc = _spawn_guardian(guardian_cmd, run_dir)
                _wait_until_ready(proc)

                while True:
                    try:
                        rc = proc.wait(timeout=0.5)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if rc != 0:
                    raise ProxyError(f"CLIProxyAPI supervisor exited with non-zero code {rc}")
                completed = True
            except KeyboardInterrupt:
                if proc is not None and _stop_guardian(proc) != 0:
                    raise ProxyError("proxy supervisor failed during interrupt cleanup") from None
                raise
            finally:
                # Restore the handlers first: _stop_guardian() raises when the supervisor is
                # stuck in credential cleanup, and the handlers must not outlive run().
                try:
                    for sig, orig in orig_handlers.items():
                        with contextlib.suppress(ValueError, OSError):
                            signal.signal(sig, orig)
                finally:
                    if not completed:
                        # Startup or the run already failed: keep that error as the main
                        # one and only log a cleanup failure (#577).
                        try:
                            if (
                                proc is not None
                                and proc.poll() is None
                                and _stop_guardian(proc) != 0
                            ):
                                logger.warning("proxy supervisor failed during cleanup")
                        except ProxyError as err:
                            logger.warning("proxy supervisor cleanup failed: %s", err)
                    elif proc is not None and proc.poll() is None and _stop_guardian(proc) != 0:
                        raise ProxyError("proxy supervisor failed during cleanup") from None
    except FolderBusyError as err:
        raise ProxyError(f"another cutpoint proxy is already running: {err}") from err


class ProxyProcess:
    """A proxy :func:`start` started in the background: its guardian, and the ``launch`` lock
    it holds until :meth:`stop`."""

    def __init__(self, proc: subprocess.Popen[Any], lock: contextlib.ExitStack) -> None:
        self.proc = proc
        self._lock = lock

    def poll(self) -> int | None:
        return self.proc.poll()

    def stop(self) -> None:
        """Stop the guardian (it stops CLIProxyAPI and saves the credentials) and free the lock.

        Raises:
            ProxyError: The guardian did not finish its credential cleanup.
        """
        try:
            if self.proc.poll() is None and _stop_guardian(self.proc) != 0:
                raise ProxyError("proxy supervisor failed during cleanup")
        finally:
            self._lock.close()


def start(port: int | None = None, *, cancel: threading.Event | None = None) -> ProxyProcess:
    """Start the proxy in the background (what ``cutpoint proxy`` runs) and wait until it
    answers (at most ``_STARTUP_TIMEOUT``; ``cancel`` set ends the wait with a ProxyError and
    stops it). The guardian stops it when this process exits, also when it is killed: its
    stdin then closes. Stop it sooner with :meth:`ProxyProcess.stop`.

    Raises:
        ProxyError: Another proxy runs, the port is taken, or it did not start in time.
    """
    resolved_port = _resolve_port(port)
    home_dir = cliproxy_dir()
    private_directory(home_dir)
    lock = contextlib.ExitStack()
    proc: subprocess.Popen[Any] | None = None
    try:
        try:
            lock.enter_context(folders.swap_lock(home_dir / "launch"))
        except FolderBusyError as err:
            raise ProxyError(f"another cutpoint proxy is already running: {err}") from err
        guardian_cmd, run_dir = _prepare_launch(home_dir, resolved_port)
        proc = _spawn_guardian(guardian_cmd, run_dir)
        if cancel is None:
            _wait_until_ready(proc)
        else:
            _wait_until_ready(proc, cancel=cancel)
        if proc.poll() is not None:  # _wait_until_ready returns on a code 0 exit too
            raise ProxyError(f"CLIProxyAPI supervisor exited with code {proc.returncode}")
    except BaseException:
        # keep the startup error as the main one; only log a cleanup failure (#577)
        try:
            if proc is not None and proc.poll() is None and _stop_guardian(proc) != 0:
                logger.warning("proxy supervisor failed during cleanup")
        except ProxyError as err:
            logger.warning("proxy supervisor cleanup failed: %s", err)
        finally:
            lock.close()
        raise
    return ProxyProcess(proc, lock)


def has_saved_login() -> bool:
    """Whether an earlier proxy run kept a subscription login: the credentials marker names
    its files (never reads the keyring or a token). After the last logout it stays true until
    the proxy has run once more."""
    marker = cliproxy_dir() / "credentials-runtime.json"
    try:
        if marker.is_symlink() or not marker.is_file():
            return False
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    managed = data.get("managed") if isinstance(data, dict) else None
    return isinstance(managed, list) and bool(managed)


class ProxyLauncher:
    """Starts the proxy for ``cutpoint ui`` when it is needed and stops the one it started when
    the UI exits (#821). A proxy that was already running (``cutpoint proxy`` in a terminal,
    another UI) is used as it is and never stopped. One start runs at a time; requests that
    come meanwhile wait for it and then use it.

    ``start``, ``running`` and ``saved_login`` default to :func:`start`, whether a Cutpoint
    proxy runs (:func:`_is_daemon_owned`) and :func:`has_saved_login`; tests pass their own.
    """

    def __init__(
        self,
        *,
        start: Callable[[], ProxyProcess] | None = None,
        running: Callable[[], bool] | None = None,
        saved_login: Callable[[], bool] | None = None,
    ) -> None:
        self._start = start
        self._running = running
        self._saved_login = saved_login
        self._starting = threading.Lock()  # one start at a time
        self._state = threading.Lock()  # guards _proc and _closed; never held while starting
        self._proc: ProxyProcess | None = None
        self._closed = False
        self._stopping = threading.Event()  # set by shutdown: ends a start that waits

    def running(self) -> bool:
        """Whether a Cutpoint proxy runs, this launcher's or another one. When ours exited on
        its own, its ``launch`` lock is freed here, so ``cutpoint proxy`` can start again."""
        with self._state:
            exited = self._proc
            if exited is not None and exited.poll() is not None:
                self._proc = None
            else:
                exited = None
        if exited is not None:
            with contextlib.suppress(ProxyError):
                exited.stop()
        return (self._running or _is_daemon_owned)()

    def saved_login(self) -> bool:
        return (self._saved_login or has_saved_login)()

    def wait_for_start(self) -> None:
        """Wait while this launcher starts the proxy: it holds its lock (and so counts as
        running) before it answers."""
        if self._starting.locked():
            with self._starting:
                pass

    def ensure(self, *, logged_in_only: bool = False) -> bool:
        """Start the proxy unless one runs; True when this call started it. With
        ``logged_in_only`` only when a subscription login is saved (else False).

        Raises:
            ProxyError: It did not start (the message says why), or the UI is stopping.
        """
        self.wait_for_start()
        if self.running():
            return False
        if logged_in_only and not self.saved_login():
            return False
        with self._starting:
            if self.running():  # started by the request this one waited for
                return False
            with self._state:
                if self._closed:
                    raise ProxyError("the server is stopping")
                gone, self._proc = self._proc, None
            if gone is not None:  # ours, but it exited: free its lock before starting again
                with contextlib.suppress(ProxyError):
                    gone.stop()
            proc = self._start() if self._start else start(cancel=self._stopping)
            with self._state:
                if not self._closed:
                    self._proc = proc
                    return True
            proc.stop()  # the UI began to stop while it started
            raise ProxyError("the server is stopping")

    def shutdown(self) -> None:
        """Stop the proxy this launcher started, if any; a cleanup failure is logged."""
        with self._state:
            self._closed = True
            proc, self._proc = self._proc, None
        self._stopping.set()
        if self._starting.locked():
            logger.info("Stopping the subscription proxy that is still starting")
        if proc is None:
            return
        try:
            proc.stop()
        except ProxyError as err:
            logger.warning("could not stop the subscription proxy: %s", err)
