"""Audit #1085 hardening of API keys and the subscription proxy: the proxy's child gets no key
variable, keys live only in an OS keyring, the proxy's secrets are redacted from logs, a server
on this machine is reached without the environment's proxy, ``~/.netrc`` never replaces a key,
and a proxy answer that is not a JSON object is a :class:`~cutpoint.assistant.cliproxy.ProxyError`.

All keys are fake; no request leaves the machine (the HTTP adapter is stubbed).
"""

from __future__ import annotations

import io
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import keyring.core
import pytest
import requests

from cutpoint.assistant import cliproxy, keys
from cutpoint.assistant.providers import OllamaProvider, OpenAIProvider

FAKE_KEY = "sk-test-fake-0123456789abcdef"  # never a real key


@pytest.fixture(autouse=True)
def clean_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in keys.ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(keys, "_SEEN", set())
    monkeypatch.setattr(keys, "_CAPTURED", {})


# A676: the guardian (and so CLIProxyAPI) inherits no API key variable


def test_the_proxy_guardian_gets_no_api_key_variable(monkeypatch, tmp_path: Path) -> None:
    for name in keys.ENV_VARS:
        monkeypatch.setenv(name, FAKE_KEY)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("PGSTORE_DSN", "postgres://example.invalid/db")
    seen: dict[str, Any] = {}

    def popen(cmd: list[str], **kw: Any) -> Any:
        seen.update(kw)
        return None

    monkeypatch.setattr(subprocess, "Popen", popen)
    cliproxy._spawn_guardian(["guardian"], tmp_path)
    env = seen["env"]
    assert env["PATH"] == "/usr/bin"
    assert not set(keys.ENV_VARS) & set(env)
    assert "PGSTORE_DSN" not in env and FAKE_KEY not in env.values()


# A677: keys are kept only in an operating-system keyring


class _Store:
    def __init__(self) -> None:
        self.saved: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, user: str) -> str | None:
        return self.saved.get((service, user))

    def set_password(self, service: str, user: str, value: str) -> None:
        self.saved[(service, user)] = value

    def delete_password(self, service: str, user: str) -> None:
        del self.saved[(service, user)]


PlaintextKeyring = type("PlaintextKeyring", (_Store,), {"__module__": "keyrings.alt.file"})
SecretServiceKeyring = type("Keyring", (_Store,), {"__module__": "keyring.backends.SecretService"})


def test_a_plaintext_keyring_backend_is_never_used_for_keys(monkeypatch) -> None:
    plaintext = PlaintextKeyring()
    plaintext.saved[(keys.SERVICE, "openai")] = FAKE_KEY  # put there by another program
    monkeypatch.setattr(keyring.core, "_keyring_backend", plaintext)
    assert keys.lookup("openai", "api.openai.com") == (None, None)
    assert not keys.in_keyring("openai", "api.openai.com")
    with pytest.raises(keys.KeyringError, match=r"no OS keyring.*OPENAI_API_KEY") as err:
        keys.save("openai", "sk-test-fake-other-0123456789", "api.openai.com")
    assert "sk-test-fake-other" not in str(err.value)
    assert plaintext.saved == {(keys.SERVICE, "openai"): FAKE_KEY}  # nothing written


def test_an_os_keyring_backend_keeps_keys(monkeypatch) -> None:
    store = SecretServiceKeyring()
    monkeypatch.setattr(keyring.core, "_keyring_backend", store)
    keys.save("openai", FAKE_KEY, "api.openai.com")
    assert keys.lookup("openai", "api.openai.com") == (FAKE_KEY, "keyring")
    keys.clear("openai", "api.openai.com")
    assert store.saved == {}


# A681: the proxy's keys and OAuth-shaped tokens are redacted from log records


def test_log_records_never_carry_the_proxy_keys(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("cutpoint.paths.home", lambda: tmp_path)
    mgmt, client = cliproxy._ensure_keys(target_port=cliproxy.DEFAULT_PORT)
    out = io.StringIO()
    handler = logging.StreamHandler(out)
    handler.addFilter(keys._FILTER)
    log = logging.getLogger("cutpoint.test.proxykeys")
    log.addHandler(handler)
    try:
        try:
            raise RuntimeError(f"sent {mgmt} and {client}")
        except RuntimeError:
            log.exception("proxy call failed")
    finally:
        log.removeHandler(handler)
    text = out.getvalue()
    assert "RuntimeError: sent [redacted] and [redacted]" in text
    assert mgmt not in text and client not in text

    monkeypatch.setattr(keys, "_SEEN", set())  # another process: it reads the keys file
    cliproxy._read_keys_file()
    assert keys.redact(f"x {mgmt}") == "x [redacted]"


def test_bearer_and_google_oauth_tokens_are_redacted() -> None:
    token = "ya29.a0Fake-OAuth-token_0123456789"
    assert keys.redact(f"got {token}") == "got [redacted]"
    header = "Authorization: Bearer fakeBearer0123456789abc"
    assert keys.redact(header) == "Authorization: Bearer [redacted]"
    assert keys.redact("a Bearer token header") == "a Bearer token header"


# A680: on-machine servers skip the environment's proxy; ~/.netrc never replaces the key


class _Adapter:
    """``HTTPAdapter.send``: records what requests would send, answers ``{"models": []}``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.sent: list[tuple[requests.PreparedRequest, dict[str, Any]]] = []
        monkeypatch.setattr(
            requests.adapters.HTTPAdapter, "send", lambda _adapter, req, **kw: self._send(req, **kw)
        )

    def _send(self, request: requests.PreparedRequest, **kw: Any) -> Any:
        self.sent.append((request, kw.get("proxies") or {}))
        resp = requests.Response()
        resp.status_code = 200
        resp._content = b'{"models": [], "data": []}'
        resp.request, resp.url = request, request.url
        return resp


@pytest.fixture
def proxied(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Adapter:
    for name in ("NO_PROXY", "no_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, "http://proxy.invalid:3128")
    netrc = tmp_path / "netrc"
    netrc.write_text(
        "machine api.openai.com login someone password fake-netrc-password\n"
        "machine 127.0.0.1 login someone password fake-netrc-password\n",
        encoding="utf-8",
    )
    netrc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc))
    return _Adapter(monkeypatch)


def test_ollama_on_this_machine_is_not_sent_through_the_environment_proxy(proxied) -> None:
    OllamaProvider("http://127.0.0.1:11434", "m", 0).list_models()
    ((_, proxies),) = proxied.sent
    assert not any(proxies.values())


def test_an_on_machine_openai_server_skips_the_proxy_and_netrc(proxied) -> None:
    OpenAIProvider("http://127.0.0.1:1234/v1", "m", 0, None)._send(
        "GET", "http://127.0.0.1:1234/v1/models", None, 5
    )
    ((request, proxies),) = proxied.sent
    assert not any(proxies.values())
    assert "Authorization" not in request.headers


def test_a_hosted_key_is_never_replaced_by_netrc_and_keeps_the_proxy(proxied) -> None:
    OpenAIProvider("", "m", 0, FAKE_KEY)._send("GET", "https://api.openai.com/v1/models", None, 5)
    ((request, proxies),) = proxied.sent
    assert request.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert proxies.get("https") == "http://proxy.invalid:3128"  # hosted: the user's proxy


# A294: a proxy answer that is not a JSON object is a ProxyError


class _Answer:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code, self.ok = status, status < 400
        self._body = body
        self.text = json.dumps(body)

    def json(self) -> Any:
        return self._body


@pytest.fixture
def daemon(monkeypatch: pytest.MonkeyPatch) -> list[_Answer]:
    answers: list[_Answer] = []
    session = type("S", (), {"request": lambda self, *a, **k: answers.pop(0)})()
    monkeypatch.setattr(cliproxy, "_is_daemon_owned", lambda: True)
    monkeypatch.setattr(cliproxy, "_load_keys", lambda: ("fake-mgmt", "fake-client"))
    monkeypatch.setattr(cliproxy, "_session", lambda: session)
    monkeypatch.setattr(cliproxy, "port", lambda: cliproxy.DEFAULT_PORT)
    return answers


@pytest.mark.parametrize("ask", [cliproxy._management_request, cliproxy._client_request])
@pytest.mark.parametrize("body", [["a", "list"], "text", 3])
def test_a_proxy_answer_that_is_no_object_is_a_proxy_error(daemon, ask, body) -> None:
    daemon.append(_Answer(200, body))
    with pytest.raises(cliproxy.ProxyError, match="not an object"):
        ask("GET", "/x")
    daemon.append(_Answer(500, body))
    with pytest.raises(cliproxy.ProxyHTTPError, match="500") as err:
        ask("GET", "/x")
    assert err.value.status == 500
    daemon.append(_Answer(200, {"files": []}))
    assert ask("GET", "/x") == {"files": []}


# A683: the binary's folder is made private with the module's helper (also on Windows)


def test_the_binary_folder_is_made_private(monkeypatch, tmp_path: Path) -> None:
    made: list[Path] = []
    monkeypatch.setattr(cliproxy, "private_directory", made.append)
    monkeypatch.setattr(cliproxy, "_detect_platform_asset", lambda: ("x.tar.gz", "0" * 64))

    def no_download(**kw: Any) -> None:
        raise RuntimeError("no network in tests")

    monkeypatch.setattr("cutpoint.events.setup.download", no_download)
    with pytest.raises(cliproxy.ProxyError, match="failed to download"):
        cliproxy._install_binary(tmp_path / "tools" / "cliproxy")
    assert made == [tmp_path / "tools" / "cliproxy"]
