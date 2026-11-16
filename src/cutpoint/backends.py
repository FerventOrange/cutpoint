"""Where Cutpoint's local model servers are, resolved at call time."""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

import requests
from requests.auth import HTTPBasicAuth
from requests.utils import get_auth_from_url

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
        raise ValueError(f"{what} {strip_userinfo(raw)!r}: invalid port") from None
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


# --------------------------------------------------------------------------- Ollama requests

# Up to the LAST "@" of the URL (before any "?" or "#"): a password may hold an unencoded "/" or "@".
_USERINFO = re.compile(r"(?<=://)[^\s'\"?#]*@")


def strip_userinfo(text: str) -> str:
    """``text`` with the ``user:password@`` of every URL in it left out (A641)."""
    return _USERINFO.sub("", text)


class OllamaReplyError(ValueError):
    """Ollama answered, but not with the JSON shape asked for (A116)."""


def headers_only(request: Any) -> Any:
    """An explicit ``auth`` that changes nothing: ``requests`` then reads no ``~/.netrc``, which
    would replace the ``Authorization`` header (the key) with that file's login, or send that
    login to an Ollama server."""
    return request


def _auth(url: str) -> Any:
    """The URL's own ``user:password`` as basic auth (what ``requests`` sends for it), else
    :func:`headers_only`: never a ``~/.netrc`` login."""
    user, password = get_auth_from_url(url)
    return HTTPBasicAuth(user, password) if user or password else headers_only


def _redacted(err: requests.RequestException) -> requests.RequestException:
    """``err``, or a copy of it of the same type without the URL's ``user:password``."""
    text = str(err)
    clean = strip_userinfo(text)
    if clean == text:
        return err
    try:
        return type(err)(clean, request=err.request, response=err.response)
    except TypeError:  # a subclass with other arguments
        return requests.RequestException(clean, request=err.request, response=err.response)


def ollama_send(
    method: str,
    url: str,
    path: str,
    *,
    json: Any = None,
    timeout: Any,
    stream: bool = False,
) -> requests.Response:
    """One request to the Ollama at ``url`` (``path``: ``/api/...``), its answer unchecked.

    A redirect is never followed (the caller refuses it: what was sent never goes to a host
    the user did not allow); ``~/.netrc`` is never read; a server on this machine is reached
    without the environment's proxy (A680). Raises ``requests``' errors without the URL's
    ``user:password`` (A641).
    """
    from cutpoint.assistant.providers import is_local  # noqa: PLC0415 (it imports this module)

    kw: dict[str, Any] = {"timeout": timeout, "allow_redirects": False, "auth": _auth(url)}
    if is_local(url):
        kw["proxies"] = {"http": None, "https": None, "all": None}
    if json is not None:
        kw["json"] = json
    if stream:
        kw["stream"] = True
    send = requests.post if method == "POST" else requests.get
    try:
        return send(f"{url}{path}", **kw)
    except requests.RequestException as err:
        clean = _redacted(err)
        if clean is err:
            raise
        raise clean from None


def refuse_redirect(resp: Any) -> None:
    """Raise :class:`requests.HTTPError` when Ollama answered with a redirect: requests is
    called with ``allow_redirects=False``, so what was sent (frames) never goes to a host the
    user did not allow."""
    if resp.is_redirect:
        raise requests.HTTPError(
            f"Ollama answered with a redirect (HTTP {resp.status_code}); not followed"
        )


def _check(resp: Any) -> None:
    """Refuse a redirect and an error status, the error without the URL's ``user:password``."""
    refuse_redirect(resp)
    try:
        resp.raise_for_status()
    except requests.HTTPError as err:
        clean = _redacted(err)
        if clean is err:
            raise
        raise clean from None


def _read(resp: Any) -> dict[str, Any]:
    _check(resp)
    body = resp.json()
    if not isinstance(body, dict):
        raise OllamaReplyError(f"Ollama answered {type(body).__name__}, not a JSON object")
    return body


def ollama_get(url: str, path: str, *, timeout: Any) -> dict[str, Any]:
    """GET ``path`` of the Ollama at ``url`` (as :func:`ollama_send`): the answer's JSON object.

    Raises:
        requests.RequestException: The request failed, was redirected or got an error status.
        ValueError: The answer is not JSON (:class:`requests.JSONDecodeError`, also a
            ``RequestException``) or not a JSON object (:class:`OllamaReplyError`).
    """
    return _read(ollama_send("GET", url, path, timeout=timeout))


def ollama_post(
    url: str, path: str, json: dict[str, Any], *, timeout: Any, read: bool = True
) -> dict[str, Any]:
    """POST ``json`` to ``path`` of the Ollama at ``url``, as :func:`ollama_get`; ``read``
    False: only the status is checked and {} returned (a load or unload)."""
    resp = ollama_send("POST", url, path, json=json, timeout=timeout)
    if not read:
        _check(resp)
        return {}
    return _read(resp)


def ollama_reply(body: dict[str, Any], generate: bool = False) -> str:
    """The text of an ``/api/chat`` answer (``message.content``), or of an ``/api/generate``
    one (``response``).

    Raises:
        OllamaReplyError: ``body`` has no such text.
    """
    if generate:
        text = body.get("response")
    else:
        msg = body.get("message")
        text = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(text, str):
        raise OllamaReplyError("Ollama's answer has no reply text")
    return text
