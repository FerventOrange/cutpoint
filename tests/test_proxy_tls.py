"""Local TLS authentication protects both CLIProxyAPI request surfaces."""

from __future__ import annotations

import http.server
import ipaddress
import os
import socket
import ssl
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest
from cryptography import x509

from cutpoint.assistant import cliproxy, proxy_tls


class RecordingHandler(http.server.BaseHTTPRequestHandler):
    records: ClassVar[list[tuple[str, str, bytes]]] = []

    def _respond(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        type(self).records.append((self.command, self.headers.get("Authorization", ""), body))
        response = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    do_GET = _respond
    do_POST = _respond

    def log_message(self, format_str: str, *args: Any) -> None:
        pass


def test_start_identity_is_fresh_and_contains_loopback_ip_san(tmp_path: Path) -> None:
    cert_path, key_path = proxy_tls.ensure_identity(tmp_path / "identity")
    first_cert, first_key = cert_path.read_bytes(), key_path.read_bytes()
    proxy_tls.ensure_identity(tmp_path / "identity")
    second_cert, second_key = cert_path.read_bytes(), key_path.read_bytes()

    assert first_cert != second_cert
    assert first_key != second_key
    cert = x509.load_pem_x509_certificate(second_cert)
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert san.get_values_for_type(x509.IPAddress) == [ipaddress.ip_address("127.0.0.1")]
    assert cert.not_valid_after_utc > datetime.now(UTC)


@pytest.mark.skipif(
    os.name == "nt", reason="Windows replacement semantics are covered by integration"
)
def test_existing_open_tls_key_handle_never_observes_fresh_key(tmp_path: Path) -> None:
    _, key_path = proxy_tls.ensure_identity(tmp_path / "identity")
    old_key = key_path.read_bytes()
    with key_path.open("rb") as old_handle:
        old_inode = key_path.stat().st_ino
        proxy_tls.ensure_identity(tmp_path / "identity")
        old_handle.seek(0)
        assert old_handle.read() == old_key
    assert key_path.read_bytes() != old_key
    assert key_path.stat().st_ino != old_inode


def _server(cert: Path | None = None, key: Path | None = None):
    RecordingHandler.records = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RecordingHandler)
    if cert is not None and key is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=str(cert), keyfile=str(key))
        server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _configure_client(root: Path, monkeypatch: pytest.MonkeyPatch, port: int) -> None:
    monkeypatch.setattr("cutpoint.paths.home", lambda: root.parent)
    monkeypatch.setenv("CUTPOINT_CLIPROXY_PORT", str(port))
    cliproxy._ensure_keys(target_port=port)
    monkeypatch.setattr(cliproxy, "_is_daemon_owned", lambda: True)


def test_management_and_inference_verify_the_actual_listener_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "home" / "cliproxy"
    root.mkdir(parents=True)
    cert, key = proxy_tls.ensure_identity(root)
    server, thread = _server(cert, key)
    _configure_client(root, monkeypatch, server.server_port)
    conflicting = socket.socket()
    conflicting.bind(("127.0.0.1", 0))
    monkeypatch.setenv("CUTPOINT_CLIPROXY_PORT", str(conflicting.getsockname()[1]))
    try:
        assert cliproxy._management_request("GET", "/v8/management/credentials") == {"ok": True}
        assert cliproxy._client_request(
            "POST", "/v1/chat/completions", json_data={"prompt": "private-inference-body"}
        ) == {"ok": True}
        assert RecordingHandler.records == [
            ("GET", f"Bearer {cliproxy._load_keys()[0]}", b""),
            (
                "POST",
                f"Bearer {cliproxy._load_keys()[1]}",
                b'{"prompt": "private-inference-body"}',
            ),
        ]
    finally:
        conflicting.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_wrong_tls_listener_gets_no_management_or_inference_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "home" / "cliproxy"
    root.mkdir(parents=True)
    trusted_cert, _ = proxy_tls.ensure_identity(root)
    attacker_root = tmp_path / "attacker"
    attacker_root.mkdir()
    attacker_cert, attacker_key = proxy_tls.ensure_identity(attacker_root)
    server, thread = _server(attacker_cert, attacker_key)
    _configure_client(root, monkeypatch, server.server_port)
    try:
        with pytest.raises(cliproxy.ProxyError, match="TLS identity validation failed"):
            cliproxy._management_request(
                "POST", "/v8/management/credentials", json_data={"secret": "must-not-leak"}
            )
        with pytest.raises(cliproxy.ProxyError, match="TLS identity validation failed"):
            cliproxy._client_request(
                "POST", "/v1/chat/completions", json_data={"prompt": "must-not-leak"}
            )
        assert trusted_cert.read_bytes() != attacker_cert.read_bytes()
        assert RecordingHandler.records == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_missing_tls_identity_fails_closed_without_generating_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setattr("cutpoint.paths.home", lambda: root)
    with pytest.raises(cliproxy.ProxyError, match="TLS identity is unavailable"):
        cliproxy._session()
    assert not (root / "cliproxy" / "proxy-cert.pem").exists()


def test_plain_http_listener_is_not_contacted_with_http_request_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "home" / "cliproxy"
    root.mkdir(parents=True)
    proxy_tls.ensure_identity(root)
    server, thread = _server()
    _configure_client(root, monkeypatch, server.server_port)
    try:
        with pytest.raises(cliproxy.ProxyError, match="TLS identity validation failed"):
            cliproxy._client_request(
                "POST", "/v1/chat/completions", json_data={"prompt": "must-not-leak"}
            )
        assert RecordingHandler.records == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
