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
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

from cutpoint import paths
from cutpoint.events.llama_server import _pdeathsig_hook
from cutpoint.ffmpeg_utils import FolderBusyError, swap_lock
from cutpoint.winjob import kill_with_parent

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

DEFAULT_PORT = 8317
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
    env_port = os.environ.get("CUTPOINT_CLIPROXY_PORT")
    if env_port:
        return _validate_port(env_port)
    kp = keys_path()
    if kp.exists():
        with contextlib.suppress(OSError, ValueError, json.JSONDecodeError):
            with kp.open(encoding="utf-8") as f:
                data = json.load(f)
            stored_port = data.get("port")
            if stored_port:
                return _validate_port(stored_port)
    return DEFAULT_PORT


def base_url() -> str:
    """Base loopback URL of the running proxy daemon."""
    return f"http://127.0.0.1:{port()}"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    path.chmod(0o600)


def _read_keys_file() -> tuple[str, str, int] | None:
    kp = keys_path()
    if not kp.exists():
        return None
    try:
        with kp.open(encoding="utf-8") as f:
            data = json.load(f)
        mgmt = data.get("management_key", "")
        client = data.get("client_api_key", "")
        p = int(data.get("port") or DEFAULT_PORT)
        if mgmt and client:
            return mgmt, client, p
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return None


def _ensure_keys(target_port: int | None = None) -> tuple[str, str]:
    existing = _read_keys_file()
    p = target_port or port()
    if existing:
        mgmt, client, curr_p = existing
        if target_port and curr_p != target_port:
            _write_private_file(
                keys_path(),
                json.dumps(
                    {"management_key": mgmt, "client_api_key": client, "port": target_port},
                    indent=2,
                ),
            )
        return mgmt, client
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
        raise ProxyError("cutpoint proxy not running; run 'cutpoint proxy' to authenticate")
    return res[0], res[1]


def _session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False
    return s


def _management_request(
    method: str,
    path: str,
    json_data: dict | None = None,
    params: dict | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
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
    except requests.RequestException as err:
        raise ProxyError(redact(f"proxy connection failed: {err}")) from err

    if not resp.ok:
        try:
            body = resp.json()
            err_msg = body.get("error") or body.get("message") or resp.text
        except (ValueError, json.JSONDecodeError):
            err_msg = resp.text
        raise ProxyError(redact(f"proxy error {resp.status_code}: {err_msg}"))

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
    except requests.RequestException as err:
        raise ProxyError(redact(f"proxy inference request failed: {err}")) from err

    if not resp.ok:
        try:
            body = resp.json()
            err_msg = body.get("error") or body.get("message") or resp.text
        except (ValueError, json.JSONDecodeError):
            err_msg = resp.text
        raise ProxyError(redact(f"proxy inference error {resp.status_code}: {err_msg}"))

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
    except (ProxyError, OSError):
        return [
            {
                "provider": p,
                "logged_in": False,
                "email": None,
                "org": None,
                "expired": False,
                "expires_ms": None,
                "error": "cutpoint proxy not running; run 'cutpoint proxy' to authenticate",
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


def list_models(provider: str) -> list[dict[str, str]]:
    """List available models for provider, unioning healthy accounts and stripped of routing prefixes."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; must be one of {PROVIDERS}")
    cli_p = CLI_PROVIDERS[provider]
    prefix = PREFIX_BY_PROVIDER[provider]
    _ensure_credential_prefixes()

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
    _ensure_credential_prefixes()

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
                    base_name = Path(member).name
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
                    base_name = Path(member.name).name
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
) -> dict[str, Any]:
    return {
        "config-version": 8,
        "server": {
            "host": "127.0.0.1",
            "port": port,
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


def run(port: int | None = None) -> None:
    """Run CLIProxyAPI in the foreground as an authenticated local loopback proxy."""
    if port is not None:
        resolved_port = _validate_port(port)
    elif os.environ.get("CUTPOINT_CLIPROXY_PORT"):
        resolved_port = _validate_port(os.environ["CUTPOINT_CLIPROXY_PORT"])
    else:
        existing_keys = _read_keys_file()
        resolved_port = existing_keys[2] if existing_keys else DEFAULT_PORT

    home_dir = cliproxy_dir()
    home_dir.mkdir(parents=True, exist_ok=True)
    home_dir.chmod(0o700)

    auths = auth_dir()
    auths.mkdir(parents=True, exist_ok=True)
    auths.chmod(0o700)

    lock_target = home_dir / "daemon.run"
    try:
        with swap_lock(lock_target):
            mgmt_key, client_key = _ensure_keys(target_port=resolved_port)
            cfg_path = config_path()
            cfg_dict = _generate_config(
                port=resolved_port,
                auth_dir_path=auths,
                mgmt_key=mgmt_key,
                client_key=client_key,
            )
            _write_private_file(cfg_path, json.dumps(cfg_dict, indent=2))

            env_bin = os.environ.get("CUTPOINT_CLIPROXY_BINARY")
            if env_bin:
                bin_file = Path(env_bin).expanduser().resolve()
                if not bin_file.exists():
                    raise ProxyError(
                        f"configured CUTPOINT_CLIPROXY_BINARY does not exist: {bin_file}"
                    )
                if not os.access(bin_file, os.X_OK):
                    raise ProxyError(
                        f"configured CUTPOINT_CLIPROXY_BINARY is not executable: {bin_file}"
                    )
            else:
                bin_file = _install_binary(paths.tools_dir() / "cliproxy")

            cmd = [str(bin_file), "-config", str(cfg_path), "-no-browser", "--local-model"]
            logger.info("Starting CLIProxyAPI on 127.0.0.1:%d", resolved_port)
            run_dir = home_dir / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_dir.chmod(0o700)
            if (run_dir / ".env").exists():
                raise ProxyError(f"untrusted .env found in daemon working directory: {run_dir}")

            child_env = _sanitize_daemon_env()
            proc: subprocess.Popen[Any] | None = None
            w_fd: int | None = None
            orig_sigint = None
            orig_sigterm = None

            def _handle_signal(signum: int, frame: Any) -> None:
                logger.info("Stopping CLIProxyAPI...")
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=6.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                sys.exit(0)

            orig_sigint = signal.signal(signal.SIGINT, _handle_signal)
            orig_sigterm = signal.signal(signal.SIGTERM, _handle_signal)

            try:
                if sys.platform == "darwin":
                    r_fd, w_fd = os.pipe()
                    os.set_inheritable(r_fd, True)
                    os.set_inheritable(w_fd, False)
                    guardian_cmd = [
                        sys.executable,
                        "-m",
                        "cutpoint.assistant._darwin_guardian",
                        "--read-fd",
                        str(r_fd),
                        "--",
                        *cmd,
                    ]
                    try:
                        proc = subprocess.Popen(
                            guardian_cmd,
                            cwd=run_dir,
                            env=child_env,
                            pass_fds=(r_fd,),
                        )
                    finally:
                        os.close(r_fd)
                else:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=run_dir,
                        env=child_env,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        preexec_fn=_pdeathsig_hook(),
                    )
                kill_with_parent(proc)

                rc = proc.wait()
                if rc != 0:
                    raise ProxyError(f"CLIProxyAPI process exited with non-zero code {rc}")
            finally:
                if orig_sigint is not None:
                    signal.signal(signal.SIGINT, orig_sigint)
                if orig_sigterm is not None:
                    signal.signal(signal.SIGTERM, orig_sigterm)
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=6.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                if w_fd is not None:
                    with contextlib.suppress(OSError):
                        os.close(w_fd)
    except FolderBusyError as err:
        raise ProxyError(f"another cutpoint proxy is already running: {err}") from err
