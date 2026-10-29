"""API keys of the hosted assistant providers: looked up, saved in the keyring, never shown.

A key belongs to a provider *and the host it is sent to*. It is read, in order, from the
provider's own variable (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``XAI_API_KEY``; only for the
provider's official host, :data:`OFFICIAL_HOST`, so an OpenAI key never goes to OpenRouter or a
LAN server, nor an xAI key anywhere but ``api.x.ai``), the system keyring (service ``cutpoint``,
user ``openai`` / ``anthropic`` / ``xai`` for the official host, else ``<provider>@<host>``;
``cutpoint ui`` saves a key there from Settings) and ``CUTPOINT_ASSISTANT_API_KEY``, which is
only for custom and self-hosted base URLs: it never goes to an official host (#504).

A key is never written to ``config.json`` or project files, never sent to the page (only where
it comes from: :func:`status`) and never put in a job's arguments. ``cutpoint ui`` moves the
variables above out of its environment at startup (:func:`capture`), and its jobs get an
environment without them (:func:`child_env`), so no child process sees a key. Every key read in this
process is replaced by ``[redacted]`` in error messages (:func:`redact`) and in log records
(:class:`RedactFilter`, installed by :func:`redact_logs`).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

ENV_KEY = "CUTPOINT_ASSISTANT_API_KEY"  # a custom host's key; never for OFFICIAL_HOST (#504)
PROVIDER_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "xai": "XAI_API_KEY"}
ENV_VARS = (ENV_KEY, *PROVIDER_ENV.values())  # removed from every child process's environment
KEYED = tuple(PROVIDER_ENV)  # providers that take a key
# The only host a provider's own variable (and its plain keyring entry) is sent to.
OFFICIAL_HOST = {"openai": "api.openai.com", "anthropic": "api.anthropic.com", "xai": "api.x.ai"}
SERVICE = "cutpoint"  # keyring service name
MAX_KEY = 512
REDACTED = "[redacted]"
MIN_SECRET = 8  # shorter values are not searched for (they would match ordinary text)
# Key-like tokens (OpenAI "sk-...", Anthropic "sk-ant-...", xAI "xai-..."), redacted even when
# not read here.
_KEY_LIKE = re.compile(r"\b(?:sk|xai)-[A-Za-z0-9_\-]{16,}")

_SEEN: set[str] = set()  # every key read or saved in this process
_LOCK = threading.Lock()


class KeyringError(RuntimeError):
    """The keyring refused (or has no backend); the message is for the user (never the key)."""


def _keyring() -> Any:
    """The ``keyring`` module (a core dependency, imported when first needed)."""
    import keyring  # noqa: PLC0415

    return keyring


def _remember(key: str) -> None:
    if len(key) >= MIN_SECRET:
        with _LOCK:
            _SEEN.add(key)


_CAPTURED: dict[str, str] = {}  # the key variables taken out of os.environ by capture()


def capture() -> None:
    """Move the key variables out of ``os.environ`` into this module, so no child process of
    this one (jobs, ffmpeg, ffprobe, file dialogs) inherits them. ``cutpoint ui`` calls it once
    at startup; :func:`lookup` reads them from here afterwards."""
    for name in ENV_VARS:
        value = os.environ.pop(name, None)
        if value is not None and value.strip():
            _CAPTURED[name] = value.strip()
            _remember(value.strip())


def _env(name: str) -> str:
    return (os.environ.get(name) or _CAPTURED.get(name) or "").strip()


def official(provider: str, host: str) -> bool:
    """Whether ``host`` is ``provider``'s own API host (:data:`OFFICIAL_HOST`)."""
    return bool(host) and OFFICIAL_HOST.get(provider) == host.lower()


def ring_user(provider: str, host: str) -> str:
    """The keyring user name of ``provider``'s key for ``host``: ``openai`` / ``anthropic`` /
    ``xai`` for the official host, else ``<provider>@<host>``."""
    return provider if official(provider, host) else f"{provider}@{host.lower()}"


def shared_for(host: str) -> bool:
    """Whether :data:`ENV_KEY` may be sent to ``host``: any host but the official ones
    (:data:`OFFICIAL_HOST`, whichever provider is pointed at it; #504)."""
    h = host.lower().rstrip(".")  # api.openai.com. is api.openai.com
    return bool(h) and h not in OFFICIAL_HOST.values()


def env_names(provider: str, host: str) -> list[str]:
    """The variables that may hold ``provider``'s key for ``host``: its own for its official
    host, :data:`ENV_KEY` for a host that is not official, else none."""
    if official(provider, host) and provider in PROVIDER_ENV:
        return [PROVIDER_ENV[provider]]
    return [ENV_KEY] if shared_for(host) else []


def has_shared() -> bool:
    """Whether :data:`ENV_KEY` is set (never its value)."""
    return bool(_env(ENV_KEY))


def lookup(provider: str, host: str, *, shared: bool = True) -> tuple[str | None, str | None]:
    """``(key, where)`` of ``provider``'s key for ``host``: ``where`` is ``"env"`` or
    ``"keyring"``; ``(None, None)`` when there is none. In order: the provider's own variable
    (only for its official host), its keyring entry for ``host``, then :data:`ENV_KEY`, only for
    a host that is not official (:func:`shared_for`, #504) and unless ``shared`` is False (a
    provider that is not the saved one: the shared key may be meant for another host)."""
    own = PROVIDER_ENV.get(provider) if official(provider, host) else None
    if own and (value := _env(own)):
        _remember(value)
        return value, "env"
    try:
        value = (_keyring().get_password(SERVICE, ring_user(provider, host)) or "").strip()
    except Exception as err:  # noqa: BLE001 - no backend, a locked store: as if no key
        logger.debug("keyring: could not read the %s key (%s)", provider, type(err).__name__)
        value = ""
    if value:
        _remember(value)
        return value, "keyring"
    if shared and shared_for(host) and (value := _env(ENV_KEY)):
        _remember(value)
        return value, "env"
    return None, None


def status(provider: str, host: str) -> str | None:
    """Where ``provider``'s key for ``host`` comes from (``"env"``, ``"keyring"``), or None:
    never the key."""
    return lookup(provider, host)[1]


def _check(provider: str) -> Any:
    if provider not in KEYED:
        raise ValueError(f"provider must be one of {', '.join(KEYED)}")
    return _keyring()


def save(provider: str, key: str, host: str) -> None:
    """Save ``key`` for ``provider`` at ``host`` in the system keyring (:func:`ring_user`).

    Raises:
        ValueError: An unknown provider, or a key that is empty, too long or has spaces.
        KeyringError: The keyring refused (or has no backend).
    """
    ring = _check(provider)
    key = key.strip() if isinstance(key, str) else ""
    if not key or len(key) > MAX_KEY or any(c.isspace() or not c.isprintable() for c in key):
        raise ValueError(f"the key must be 1-{MAX_KEY} printable characters without spaces")
    _remember(key)
    try:
        ring.set_password(SERVICE, ring_user(provider, host), key)
    except Exception as err:  # noqa: BLE001 - any backend error, named without the key
        names = env_names(provider, host)
        hint = f"; or set {names[0]} before starting cutpoint ui" if names else ""
        raise KeyringError(
            f"the keyring did not save the key ({type(err).__name__}){hint}"
        ) from None


def clear(provider: str, host: str) -> None:
    """Remove ``provider``'s key for ``host`` from the keyring (nothing to remove is fine).

    Raises:
        ValueError: An unknown provider.
        KeyringError: The keyring refused (or has no backend).
    """
    ring = _check(provider)
    user = ring_user(provider, host)
    try:
        if ring.get_password(SERVICE, user):
            ring.delete_password(SERVICE, user)
    except Exception as err:  # noqa: BLE001 - any backend error, named without the key
        raise KeyringError(f"the keyring did not remove the key ({type(err).__name__})") from None


def child_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """``env`` (default: this process's environment) without the API key variables."""
    env = os.environ if env is None else env
    drop = {name.upper() for name in ENV_VARS}
    return {k: v for k, v in env.items() if k.upper() not in drop}


def redact(text: str) -> str:
    """``text`` with every key this process read (or one in the key variables) and anything
    shaped like a key replaced by ``[redacted]``."""
    with _LOCK:
        secrets = set(_SEEN)
    secrets.update(v for name in ENV_VARS if len(v := _env(name)) >= MIN_SECRET)
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    return _KEY_LIKE.sub(REDACTED, text)


class RedactFilter(logging.Filter):
    """Redact keys from a record's message and traceback before any handler formats it."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad format string: the handler reports it
            return True
        clean = redact(message)
        if clean != message:
            record.msg, record.args = clean, None
        if record.exc_info and not record.exc_text:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
        elif record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


_FILTER = RedactFilter()


LOGGERS = ("", "uvicorn", "uvicorn.error", "uvicorn.access")  # whose handlers are filtered


def redact_logs() -> None:
    """Add :class:`RedactFilter` to the handlers of the root logger and uvicorn's (once each).
    uvicorn sets up its handlers when its ``Config`` is made: call this again after that."""
    for name in LOGGERS:
        for handler in logging.getLogger(name).handlers:
            if _FILTER not in handler.filters:
                handler.addFilter(_FILTER)
