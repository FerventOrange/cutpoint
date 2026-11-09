"""Where Cutpoint's local model servers are, resolved at call time."""

from __future__ import annotations

import os
from urllib.parse import SplitResult, urlsplit, urlunsplit

from cutpoint.config import get_config

OLLAMA_FALLBACK_URL = "http://127.0.0.1:11434"
OLLAMA_PORT = 11434
_ANY_HOST = ("0.0.0.0", "::", "")  # bind-all addresses: replaced below


def split_ollama_url(raw: str, what: str) -> tuple[SplitResult, int | None]:
    """``raw`` split as ``(parts, port)``: ``http://`` is added to a value without a scheme
    (a bare IPv6 address is bracketed), and a bare value without a port gets Ollama's.

    Raises:
        ValueError: ``<what> '<raw>': invalid port`` if the port is not a number in 0-65535.
    """
    url = raw
    bare = "://" not in url
    if bare:
        if url.count(":") > 1 and not url.startswith("["):
            url = f"[{url}]"
        url = "http://" + url
    parts = urlsplit(url)
    try:
        port = parts.port or (OLLAMA_PORT if bare else None)
    except ValueError:
        raise ValueError(f"{what} {raw!r}: invalid port") from None
    return parts, port


def normalize_ollama_url(raw: str, what: str) -> str:
    """``raw`` as an Ollama base URL, without a trailing slash, normalised like Ollama's own
    client does: a value without a scheme (``myhost``, ``0.0.0.0:11434``, ``::1``) gets
    ``http://`` and, if it has no port, ``:11434``; a bare IPv6 address is bracketed; a
    bind-all host (``0.0.0.0``, ``::`` or empty, as set for ``ollama serve``) becomes
    ``127.0.0.1`` with the port kept. ``user:pw@`` is kept.

    Used for ``$OLLAMA_HOST``, ``ollama_url`` in ``config.json`` and ``--ollama-url`` alike.

    Raises:
        ValueError: ``<what> '<raw>': invalid port`` if the port is not a number in 0-65535.
    """
    parts, port = split_ollama_url(raw.strip(), what)
    host = parts.hostname or ""
    if host in _ANY_HOST:
        host = "127.0.0.1"
    elif ":" in host:
        host = f"[{host}]"
    auth = parts.netloc.rpartition("@")[0]
    netloc = (f"{auth}@" if auth else "") + host + (f":{port}" if port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment)).rstrip("/")


def ollama_url() -> str:
    """Ollama base URL, without a trailing slash.

    Taken from ``$OLLAMA_HOST``, else ``ollama_url`` in ``config.json``, else
    ``http://127.0.0.1:11434``, and normalised by :func:`normalize_ollama_url`.

    Raises:
        ValueError: If the port is not a number in 0-65535.
    """
    raw = (
        (os.environ.get("OLLAMA_HOST") or "").strip()
        or get_config().ollama_url.strip()
        or OLLAMA_FALLBACK_URL
    )
    return normalize_ollama_url(raw, "OLLAMA_HOST/ollama_url")
