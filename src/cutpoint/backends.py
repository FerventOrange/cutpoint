"""Where Cutpoint's local model servers are, resolved at call time."""

from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit

from cutpoint.config import get_config

OLLAMA_FALLBACK_URL = "http://localhost:11434"
OLLAMA_PORT = 11434
_ANY_HOST = ("0.0.0.0", "::", "")  # bind-all addresses: replaced below


def ollama_url() -> str:
    """Ollama base URL, without a trailing slash.

    Taken from ``$OLLAMA_HOST``, else ``ollama_url`` in ``config.json``, else
    ``http://localhost:11434``, and normalised like Ollama's own client does:
    a value without a scheme (``myhost``, ``0.0.0.0:11434``, ``::1``) gets
    ``http://`` and, if it has no port, ``:11434``; a bare IPv6 address is
    bracketed; a bind-all host (``0.0.0.0``, ``::`` or empty, as set for
    ``ollama serve``) becomes ``127.0.0.1`` with the port kept. ``user:pw@``
    is kept.

    Raises:
        ValueError: If the port is not a number in 0-65535.
    """
    raw = (
        (os.environ.get("OLLAMA_HOST") or "").strip()
        or get_config().ollama_url.strip()
        or OLLAMA_FALLBACK_URL
    )
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
        raise ValueError(f"OLLAMA_HOST/ollama_url {raw!r}: invalid port") from None
    host = parts.hostname or ""
    if host in _ANY_HOST:
        host = "127.0.0.1"
    elif ":" in host:
        host = f"[{host}]"
    auth = parts.netloc.rpartition("@")[0]
    netloc = (f"{auth}@" if auth else "") + host + (f":{port}" if port else "")
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment)).rstrip("/")
