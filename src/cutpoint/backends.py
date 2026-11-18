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
# The connect timeout of a request to a model server (Ollama, a hosted API): one that does
# not answer is reported in seconds, not after the whole read timeout (A298)
CONNECT_TIMEOUT_S = 10.0
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
        login, at, host = url.rpartition("@")  # a login (user:pw@host:port) is no IPv6 address
        if host.count(":") > 1 and not host.startswith("["):
            url = f"{login}{at}[{host}]"
        url = "http://" + url
    parts = urlsplit(url)
    try:
        port = parts.port or (OLLAMA_PORT if bare else None)
    except ValueError:
        raise ValueError(f"{what} {without_login(raw)!r}: invalid port") from None
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


# host[:port] as an authority without a login: a name or IPv4, a bracketed or bare IPv6 address
_HOST_PORT = re.compile(
    r"(\[[^\]]*\]|[^:\[\]@]*)(:\d*)?|[0-9A-Fa-f.]*:[0-9A-Fa-f.]*:[0-9A-Fa-f:.]*"
)


def without_login(raw: str) -> str:
    """``raw``, an Ollama URL as given (with or without a scheme, as ``--ollama-url`` takes
    it), without its ``user:password@``: what Cutpoint writes down (``settings.ollama_url`` of
    ``highlights.json``, the ``argv`` of ``progress.json``) and shows in errors. A login is
    never saved; a later run takes it from :func:`with_login` (A639).

    The login is the authority's (up to its last ``@``, as a password may hold one); an
    ``@`` in the path (``http://h/p@x``) is no login. An authority that is no ``host[:port]``
    while an ``@`` follows (a password with an unencoded ``/``, ``?`` or ``#``) is cut up to
    the last ``@``."""
    head, sep, rest = raw.partition("://")
    if not sep:
        head, rest = "", raw
    ends = [i for i in (rest.find(c) for c in "/?#") if i >= 0]
    end = min(ends, default=len(rest))
    authority = rest[:end]
    if "@" in authority:
        rest = authority.rpartition("@")[2] + rest[end:]
    elif "@" in rest and not _HOST_PORT.fullmatch(authority):
        rest = rest[rest.rfind("@") + 1 :]
    return head + sep + rest


def saved_settings(hl: Any) -> Any:
    """``hl`` (a ``highlights.json`` object) as it is written: ``settings.ollama_url`` without
    its login (:func:`without_login`), changed in place. Every writer of ``highlights.json``
    calls it, so an older file that kept a login loses it on its next write (A639)."""
    st = hl.get("settings") if isinstance(hl, dict) else None
    if isinstance(st, dict) and isinstance(st.get("ollama_url"), str):
        st["ollama_url"] = without_login(st["ollama_url"])
    return hl


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin(url: str) -> tuple[str, str, int | None] | None:
    """``(scheme, host, port)`` of an Ollama URL (normalised as :func:`normalize_ollama_url`),
    the scheme's port when it names none; None for a value that is no usable URL."""
    try:
        parts = urlsplit(normalize_ollama_url(url, "ollama_url"))
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    return scheme, (parts.hostname or "").lower(), port or _DEFAULT_PORTS.get(scheme)


def _configured_ollama_urls() -> list[str]:
    """The Ollama URLs the user set: ``$OLLAMA_HOST``, then ``ollama_url`` and the URLs of the
    Vision and AI Chat roles on Ollama in ``config.json`` (an invalid ``config.json``: none of
    it)."""
    urls = [(os.environ.get("OLLAMA_HOST") or "").strip()]
    try:
        cfg = get_config()
    except ValueError:
        return [u for u in urls if u]
    urls.append(cfg.ollama_url.strip())
    urls += [s.url.strip() for s in (cfg.vision_assistant, cfg.assistant) if s.provider == "ollama"]
    return [u for u in urls if u]


def with_login(url: str) -> str:
    """``url`` (an Ollama base URL) with the ``user:password@`` of ``$OLLAMA_HOST`` or of an
    Ollama URL in ``config.json`` that names the same scheme, host and port, when it has no
    login of its own; else ``url`` as it is. A recorded ``--ollama-url`` keeps no login
    (:func:`without_login`), so a replayed run (``reanalyze``, ``resume``, ``events``,
    ``rerank --analyze``, a job of ``cutpoint ui``) gets it here (A639)."""
    if any(get_auth_from_url(url)):
        return url
    want = _origin(url)
    if want is None:
        return url
    for other in _configured_ollama_urls():
        user, password = get_auth_from_url(_with_scheme(other))
        if (user or password) and _origin(other) == want:
            parts = urlsplit(url)
            login = urlsplit(_with_scheme(other)).netloc.rpartition("@")[0]
            return urlunsplit(parts._replace(netloc=f"{login}@{parts.netloc}"))
    return url


def _with_scheme(url: str) -> str:
    """``url`` with ``http://`` when it has no scheme (as :func:`split_ollama_url` adds it)."""
    return url if "://" in url else "http://" + url


class OllamaReplyError(ValueError):
    """Ollama answered, but not with the JSON shape asked for (A116)."""


def headers_only(request: Any) -> Any:
    """An explicit ``auth`` that changes nothing: ``requests`` then reads no ``~/.netrc``, which
    would replace the ``Authorization`` header (the key) with that file's login, or send that
    login to an Ollama server."""
    return request


def _auth(url: str) -> Any:
    """The URL's own ``user:password`` as basic auth (what ``requests`` sends for it), else the
    configured login of the same server (:func:`with_login`), else :func:`headers_only`: never
    a ``~/.netrc`` login."""
    user, password = get_auth_from_url(with_login(url))
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
    ``user:password`` (A641). ``timeout``: a ``(connect, read)`` pair, or one number for the
    read, whose connection then gets :data:`CONNECT_TIMEOUT_S` (never more than that number),
    so an Ollama that does not answer fails in seconds (A298).
    """
    from cutpoint.assistant.providers import is_local  # noqa: PLC0415 (it imports this module)

    if isinstance(timeout, int | float):
        timeout = (min(CONNECT_TIMEOUT_S, timeout), timeout)
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
