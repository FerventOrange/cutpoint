"""Subscription OAuth grants (Claude Code, Codex, Antigravity): stored in the keyring, refreshed
on demand, never written to ``config.json`` or shown.

A grant is the OAuth result for one provider: the access token sent to the model host, the refresh
token that mints new ones, when the access token expires, and the account/project metadata the
provider needs at request time. Grants live in the system keyring (service ``cutpoint``, user
``oauth:<provider>``) as JSON; :func:`cutpoint.assistant.keys.redact` already scrubs anything this
process reads from logs and errors, and every token read here is registered with it.

Refresh is owned here: :func:`access_token` returns a token that is fresh for the next request,
refreshing (single-flight per provider) when it is within the skew of expiry. The provider flow
definitions and the token exchange/refresh HTTP live in :mod:`cutpoint.assistant.oauth`.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, replace

from cutpoint.assistant import keys

logger = logging.getLogger(__name__)

# User-facing provider ids (no upstream-CLI branding). Each maps to a flow in oauth.FLOWS.
PROVIDERS = ("claude-code", "codex", "antigravity")
TITLES = {
    "claude-code": "Claude Code subscription",
    "codex": "Codex subscription",
    "antigravity": "Antigravity",
}
SERVICE = keys.SERVICE  # keyring service, shared with the API-key store
# Refresh this many ms before the stored expiry so refresh starts early.
REFRESH_SKEW_MS = 60_000
# Anthropic expires the whole grant family ~30 days after the interactive login, regardless of
# refresh rotation; only a fresh login recovers it.
ANTHROPIC_GRANT_TTL_MS = 30 * 24 * 60 * 60 * 1000


@dataclass(frozen=True)
class Grant:
    """One provider's OAuth result. ``access``/``refresh`` are secrets; the rest is metadata."""

    provider: str
    access: str
    refresh: str
    expires_ms: int  # epoch ms the access token stops being usable (already minus the login skew)
    authorized_at_ms: int = 0  # epoch ms of the interactive login (Anthropic 30-day deadline)
    email: str = ""
    account_id: str = ""
    org_id: str = ""
    org_name: str = ""
    project_id: str = ""  # Antigravity Cloud Code project
    api_endpoint: str = ""


def _ring_user(provider: str) -> str:
    return f"oauth:{provider}"


def _remember(grant: Grant) -> None:
    keys._remember(grant.access)
    keys._remember(grant.refresh)


def load(provider: str) -> Grant | None:
    """The stored grant for ``provider``, or ``None`` when there is none or the keyring refuses."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown subscription provider {provider!r}")
    try:
        raw = keys._keyring().get_password(SERVICE, _ring_user(provider))
    except Exception as err:  # noqa: BLE001 - no backend or a locked store: as if no grant
        logger.debug("keyring: could not read the %s grant (%s)", provider, type(err).__name__)
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
        grant = Grant(provider=provider, **{k: data[k] for k in data if k != "provider"})
    except (ValueError, TypeError) as err:
        logger.warning("ignoring a malformed %s grant in the keyring (%s)", provider, err)
        return None
    _remember(grant)
    return grant


def save(grant: Grant) -> None:
    """Store ``grant`` in the keyring (replaces any existing one for its provider).

    Raises:
        keys.KeyringError: The keyring refused (or has no backend).
    """
    _remember(grant)
    try:
        keys._keyring().set_password(SERVICE, _ring_user(grant.provider), json.dumps(asdict(grant)))
    except Exception as err:  # noqa: BLE001 - any backend error, named without the token
        raise keys.KeyringError(
            f"the keyring did not save the {grant.provider} login ({type(err).__name__})"
        ) from None


def clear(provider: str) -> None:
    """Remove ``provider``'s grant from the keyring (nothing to remove is fine)."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown subscription provider {provider!r}")
    with _LOCKS[provider]:
        try:
            if keys._keyring().get_password(SERVICE, _ring_user(provider)):
                keys._keyring().delete_password(SERVICE, _ring_user(provider))
        except Exception as err:  # noqa: BLE001
            raise keys.KeyringError(
                f"the keyring did not remove the {provider} login ({type(err).__name__})"
            ) from None


def deadline_ms(grant: Grant) -> int | None:
    """Epoch ms the grant family dies and a fresh login is required, or ``None`` when the provider
    has no absolute lifetime. Only Anthropic (Claude Code) does."""
    if getattr(grant, "provider", None) == "claude-code" and getattr(grant, "authorized_at_ms", 0):
        return grant.authorized_at_ms + ANTHROPIC_GRANT_TTL_MS
    return None


def status(provider: str) -> dict:
    """What the UI/CLI show for ``provider``: logged in, which account, when it expires and when a
    re-login is needed. Never a token."""
    grant = load(provider)
    if grant is None:
        return {"provider": provider, "logged_in": False}
    out = {
        "provider": provider,
        "logged_in": True,
        "email": grant.email,
        "org": grant.org_name,
        "expires_ms": grant.expires_ms,
        "expired": time.time() * 1000 >= grant.expires_ms,
    }
    end = deadline_ms(grant)
    if end is not None:
        out["relogin_by_ms"] = end
    return out


class GrantExpiredError(RuntimeError):
    """The grant cannot be refreshed; the user must log in again."""


_LOCKS: dict[str, threading.Lock] = {p: threading.Lock() for p in PROVIDERS}


def access_token(provider: str) -> str:
    """A token fresh for the next request to ``provider``, refreshing it first when it is within
    :data:`REFRESH_SKEW_MS` of expiry. Single-flight per provider.

    Raises:
        GrantExpiredError: No grant, or the grant is dead (refresh rejected / past the 30-day
            Anthropic lifetime); the user must log in again.
    """
    return _fresh(provider).access


def fresh_grant(provider: str) -> Grant:
    """Like :func:`access_token` but returns the whole grant (project id, account id, ...), for
    providers whose request shaping needs the metadata (Antigravity, Codex)."""
    return _fresh(provider)


def _fresh(provider: str) -> Grant:
    from cutpoint.assistant import oauth  # noqa: PLC0415 - avoid an import cycle

    grant = load(provider)
    if grant is None:
        raise GrantExpiredError(f"not logged in to {TITLES.get(provider, provider)}")
    now_ms = time.time() * 1000
    end = deadline_ms(grant)
    if end is not None and now_ms >= end:
        raise GrantExpiredError(
            f"the {TITLES.get(provider, provider)} login expired (30-day limit); log in again"
        )
    if now_ms + REFRESH_SKEW_MS < grant.expires_ms:
        return grant
    with _LOCKS[provider]:
        # Re-read under the lock: another thread may have just refreshed and saved, or logged out.
        grant = load(provider)
        if grant is None:
            raise GrantExpiredError(f"not logged in to {TITLES.get(provider, provider)}")
        if time.time() * 1000 + REFRESH_SKEW_MS < grant.expires_ms:
            return grant
        if not grant.refresh:
            raise GrantExpiredError(
                f"the {TITLES.get(provider, provider)} login has no refresh token"
            )
        try:
            refreshed = oauth.refresh(grant)
        except oauth.OAuthDeadError as err:
            raise GrantExpiredError(
                f"the {TITLES.get(provider, provider)} login was rejected; log in again ({err})"
            ) from err
        try:
            save(refreshed)
        except Exception as err:
            raise oauth.OAuthError(f"could not save refreshed {provider} grant: {err}") from err
        return refreshed


def merge_refresh(
    current: Grant, access: str, refresh: str, expires_ms: int, **optional: str
) -> Grant:
    """Merge a refresh response onto the stored grant: new token/expiry win, metadata falls back to
    the stored value, and the login stamp (Anthropic deadline) survives."""
    kept = {
        k: (optional.get(k) or getattr(current, k))
        for k in ("email", "account_id", "org_id", "org_name", "project_id", "api_endpoint")
    }
    return replace(
        current, access=access, refresh=refresh or current.refresh, expires_ms=expires_ms, **kept
    )
