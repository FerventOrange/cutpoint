"""Settings, assistant, key and consent views of ``cutpoint ui``."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

from cutpoint import paths
from cutpoint.assistant import cliproxy
from cutpoint.assistant import keys as ai_keys
from cutpoint.assistant import providers as ai_providers
from cutpoint.backends import ollama_url
from cutpoint.config import (
    ASSISTANT_PROVIDERS,
    COLOR_MODES,
    COMPRESS_PEAKS,
    VISION_PROVIDERS,
    AssistantSettings,
    assistant_settings,
    get_config,
    load_config_data,
    resolve_projects_dir,
    vision_assistant_settings,
)
from cutpoint.ui.common import bad
from cutpoint.ui.files import _typed_path
from cutpoint.vision import default_model


# --------------------------------------------------------------------------- settings
def config_data() -> dict[str, Any]:
    """The raw ``config.json`` object.

    Raises:
        HTTPException: 409 if the file is not a valid JSON object (it is never overwritten).
    """
    try:
        return load_config_data()
    except ValueError as err:
        raise bad(str(err), 409) from err


def require_config() -> None:
    """Dependency of every route that reads the projects folder or remaps sources.

    Raises:
        HTTPException: 409 naming the rejected setting's source (``config.json`` or a
            ``CUTPOINT_`` variable) and what is wrong with it, when
            :func:`~cutpoint.config.get_config` rejects it (Settings stay usable to fix it).
    """
    try:
        get_config()
    except ValueError as err:  # its message names the file or variable
        raise bad(str(err), 409) from err
    except TypeError as err:  # a wrongly typed value (e.g. a number for a path)
        raise bad(f"Invalid config file {paths.config_file()}: {err}", 409) from err


def _vision_provider_from_data(data: dict[str, Any]) -> ai_providers.Provider | None:
    section = data.get("vision_assistant")
    if not isinstance(section, dict):
        section = {"provider": "ollama"}
    provider_name = (section.get("provider") or "ollama").strip().lower()
    if provider_name not in VISION_PROVIDERS:
        provider_name = "ollama"
    model = (section.get("model") or "").strip()
    url = (section.get("url") or "").strip()
    chat_name, chat_url = _provider_url(data, "assistant")
    try:
        return ai_providers.make_vision(
            AssistantSettings(provider=provider_name, model=model, url=url),
            AssistantSettings(provider=chat_name or "ollama", url=chat_url),
        )
    except (ValueError, TypeError, KeyError):  # an invalid value in config.json: no provider
        return None


def _tuple_of_str(val: Any) -> tuple[str, ...]:
    """A tuple of string items when ``val`` is a list, else empty tuple."""
    if not isinstance(val, list):
        return ()
    return tuple(x for x in val if isinstance(x, str))


def settings_view(data: dict[str, Any]) -> dict[str, Any]:
    """The settings for config ``data``: the projects folder, the vision default and the export
    defaults of Compress peaks and Color, so other invalid keys in ``config.json`` do not hide
    them; a wrongly typed ``projects_dir`` is shown as is, to replace, a ``vision`` that is not
    ``true`` as off and an unknown ``compress_peaks`` or ``color`` as off."""
    peaks = data.get("compress_peaks")
    color = data.get("color")
    try:
        folder = str(resolve_projects_dir(data))
    except TypeError:
        folder = str(data.get("projects_dir"))
    vis_section = data.get("vision_assistant")
    vis_consent = _tuple_of_str(vis_section.get("consent")) if isinstance(vis_section, dict) else ()
    vp = _vision_provider_from_data(data)
    # an Ollama on this machine needs no consent; one on another machine does, as for AI Chat
    if vp is not None and (vp.name != "ollama" or not vp.local):
        v_consent_info = consent_view(vp, consent=vis_consent, role="vision")
    else:
        v_consent_info = {
            "needed": False,
            "key": None,
            "host": getattr(vp, "host", "") if vp is not None else "",
            "missing_key": None,
            "missing_kind": None,
        }
    return {
        "projects_dir": folder,
        "projects_dir_locked": paths.env("PROJECTS_DIR") is not None,
        "config_file": str(paths.config_file()),
        "vision": data.get("vision") is True,
        "visual_index": data.get("visual_index") is True,
        "vision_model": default_model(),
        "compress_peaks": peaks if peaks in COMPRESS_PEAKS else "off",
        # whether config.json sets it: the page's Compress peaks then starts from it, else Light
        "compress_peaks_set": peaks in COMPRESS_PEAKS,
        # the page's Color starts from it (Off when unset), never with a grade on gaming footage
        "color": color if isinstance(color, str) and color in COLOR_MODES else "off",
        "assistant": assistant_view(data, section_name="assistant"),
        "vision_assistant": assistant_view(data, section_name="vision_assistant"),
        "vision_consent": v_consent_info,
    }


MAX_ASSISTANT_TEXT = 500  # characters of the assistant's model or base URL
LOGIN_TIMEOUT_S = 300.0  # seconds before a pending login expires


def subscription_presets() -> dict[str, str]:
    """The picker's providers: the keyed and local presets, then each subscription login a
    provider is built for (Antigravity only once :data:`~cutpoint.assistant.providers.PROVIDERS`
    has it)."""
    subscriptions = {p: cliproxy.TITLES[p] for p in cliproxy.PROVIDERS}
    return {**ai_providers.PRESETS, **subscriptions}


def assistant_view(data: dict[str, Any], section_name: str = "assistant") -> dict[str, Any]:
    """The assistant or vision_assistant settings of config ``data`` for Settings, never a key:
    the provider, model and base URL as saved (an invalid value shown as the default), whether that
    URL is custom (not the preset's, :func:`cutpoint.assistant.providers.is_preset`), what each
    preset offers, and where each provider's key comes from (``"env"``, ``"keyring"`` or None)."""
    section = data.get(section_name)
    if not isinstance(section, dict):
        section = {}
    name = section.get("provider")
    name = name.strip().lower() if isinstance(name, str) else ""

    def text(key: str) -> str:
        value = section.get(key)
        return value.strip() if isinstance(value, str) else ""

    try:
        ollama = ai_providers.public(ollama_url())  # OLLAMA_HOST may carry user:password@
    except (ValueError, TypeError):  # a bad port or config.json: Settings still opens
        ollama = ""
    # provider -> (default URL, default model, models offered, Model placeholder, model needed)
    is_vision = section_name == "vision_assistant"
    dm = default_model() if is_vision else ai_providers.DEFAULT_OLLAMA_MODEL
    offers = {
        "ollama": (ollama, dm if is_vision else ai_providers.DEFAULT_OLLAMA_MODEL, [dm], dm, False),
        "openai": (
            ai_providers.OPENAI_URL,
            "",
            list(ai_providers.OPENAI_MODELS),
            "OpenAI model name",
            True,
        ),
        "anthropic": (
            ai_providers.ANTHROPIC_URL,
            ai_providers.DEFAULT_ANTHROPIC_MODEL,
            list(ai_providers.ANTHROPIC_MODELS),
            ai_providers.DEFAULT_ANTHROPIC_MODEL,
            False,
        ),
        "xai": (ai_providers.XAI_URL, "", list(ai_providers.XAI_MODELS), "xAI model name", True),
        "claude-code": (
            "",
            ai_providers.DEFAULT_ANTHROPIC_MODEL,
            list(ai_providers.ANTHROPIC_MODELS),
            ai_providers.DEFAULT_ANTHROPIC_MODEL,
            False,
        ),
        "codex": (
            "",
            ai_providers.DEFAULT_CODEX_MODEL,
            [ai_providers.DEFAULT_CODEX_MODEL],
            ai_providers.DEFAULT_CODEX_MODEL,
            False,
        ),
        "antigravity": ("", "", [], "model name", False),
    }
    hosts = key_hosts(data, section_name=section_name)

    def key_env(p: str) -> list[str]:
        if p not in ai_keys.KEYED:
            return []
        names = ai_keys.env_names(p, hosts[p])
        if is_vision and not _vision_key_shared(data, p):  # never read for this host
            names = [n for n in names if n != ai_keys.ENV_KEY]
        return names

    allowed = VISION_PROVIDERS if is_vision else ASSISTANT_PROVIDERS
    name = name if name in allowed else "ollama"
    all_presets = subscription_presets()
    presets_offered = {p: title for p, title in all_presets.items() if p in allowed}
    try:
        local = ai_providers.make(AssistantSettings(provider=name, url=text("url"))).local
    except (ValueError, TypeError):
        local = name == "ollama"
    return {
        "provider": name,
        "local": local,
        "model": text("model"),
        "url": ai_providers.public(text("url")),  # a hand-edited one too; Settings refuses it
        # a base URL no preset has: shown as "Custom (from config.json)", kept as it is
        "custom": not ai_providers.is_preset(name, text("url")),
        "providers": [
            {
                "name": p,
                "title": title,
                "default_url": offers[p][0],
                "default_model": offers[p][1],
                "models": offers[p][2],
                "placeholder": offers[p][3],
                "model_required": offers[p][4],
                "key_env": key_env(p),
                "subscription": p in cliproxy.PROVIDERS,  # logs in (Settings > Subscriptions)
            }
            for p, title in presets_offered.items()
        ],
        **key_status(data, section_name=section_name),
    }


def _provider_url(data: dict[str, Any], section_name: str) -> tuple[str, str]:
    """``(provider, url)`` of config ``data``'s ``section_name`` section ("" when unset)."""
    section = data.get(section_name)
    section = section if isinstance(section, dict) else {}
    name, url = section.get("provider"), section.get("url")
    name = name.strip().lower() if isinstance(name, str) else ""
    url = url.strip() if isinstance(url, str) else ""
    return name, url


def key_hosts(data: dict[str, Any], section_name: str = "assistant") -> dict[str, str]:
    """The host each keyed provider's key is for, by config ``data``: the configured base URL's
    for the configured provider, the official API host for the others."""
    name, url = _provider_url(data, section_name)
    return {p: ai_providers.key_host(p, url if p == name else "") for p in ai_keys.KEYED}


def _vision_key_shared(data: dict[str, Any], p: str) -> bool:
    """Whether the Vision role's provider ``p`` may use ``CUTPOINT_ASSISTANT_API_KEY`` by config
    ``data`` (:func:`cutpoint.assistant.providers.vision_key_shared`)."""
    name, url = _provider_url(data, "vision_assistant")
    chat_name, chat_url = _provider_url(data, "assistant")
    return ai_providers.vision_key_shared(p, url if p == name else "", chat_name, chat_url)


def key_status(data: dict[str, Any], section_name: str = "assistant") -> dict[str, Any]:
    """``{keys, key_hosts}``: where each keyed provider's key for its host comes from
    (``"env"``, ``"keyring"`` or None), and that host. Never a key. The Vision role's keys come
    from ``CUTPOINT_ASSISTANT_API_KEY`` only for the assistant's key host
    (:func:`cutpoint.assistant.providers.vision_key_shared`)."""
    hosts = key_hosts(data, section_name=section_name)
    vision = section_name == "vision_assistant"

    def shared(p: str) -> bool:
        return not vision or _vision_key_shared(data, p)

    def where(p: str) -> str | None:
        # UI jobs read the keyring first for the Vision role, even when a variable is set
        if vision and ai_keys.in_keyring(p, hosts[p]):
            return "keyring"
        return ai_keys.status(p, hosts[p], shared=shared(p))

    return {
        "keys": {p: where(p) for p in ai_keys.KEYED},
        "key_hosts": hosts,
    }


def check_assistant(
    value: Any, section_name: str = "assistant", existing_provider: str | None = None
) -> dict[str, str]:
    """The ``assistant`` or ``vision_assistant`` object of a Settings change: ``provider``,
    ``model`` and ``url`` (an http(s) base URL; "" for the default).

    Raises:
        HTTPException: 400 for anything else.
    """
    is_vision = section_name == "vision_assistant"
    label = "vision_assistant" if is_vision else "assistant"
    allowed = VISION_PROVIDERS if is_vision else ASSISTANT_PROVIDERS
    if not isinstance(value, dict) or not value or set(value) - {"provider", "model", "url"}:
        raise bad(f"{label} must be an object with provider, model and/or url")
    out: dict[str, str] = {}
    for key, raw in value.items():
        if not isinstance(raw, str) or len(raw) > MAX_ASSISTANT_TEXT:
            raise bad(f"{label}.{key} must be text of at most {MAX_ASSISTANT_TEXT} characters")
        if any(not c.isprintable() for c in raw):
            raise bad(f"{label}.{key} must be printable text")
        out[key] = raw.strip()
    if "provider" in out:
        out["provider"] = out["provider"].lower()
        if out["provider"] not in allowed:
            raise bad(f"{label}.provider must be one of {', '.join(allowed)}")
    url = out.get("url")
    provider = out.get("provider") or existing_provider or ""
    if url and (provider in cliproxy.PROVIDERS or ai_providers.is_subscription(provider)):
        raise bad(f"{label}.url cannot be customized for subscription providers")
    if url:
        scheme = url.split("://", 1)[0].lower() if "://" in url else "http"
        if scheme not in ("http", "https") or " " in url or not ai_providers.host_of(url):
            raise bad(f"{label}.url must be an http:// or https:// base URL")
        if "\\" in url:  # requests and urlsplit read different hosts from some of these
            raise bad(f"{label}.url must not have a backslash")
        if ai_providers.has_userinfo(url):  # https://api.openai.com@evil.example goes to evil
            raise bad(f"{label}.url must not contain a user name or password")
        if not ai_providers.unambiguous(url):  # the host a request goes to must be the one shown
            raise bad(f"{label}.url must not have an unclear host")
    return out


def consent_view(
    provider: ai_providers.Provider,
    consent: tuple[str, ...] | None = None,
    role: str = "assistant",
) -> dict[str, Any]:
    """Whether the user still has to agree before project data goes to ``provider`` for
    ``role`` (``"assistant"`` or ``"vision"``): ``{needed, key, host, missing_key,
    missing_kind}``; ``key`` (:func:`cutpoint.assistant.providers.consent_key`) is None for a
    local provider; ``missing_key``: how to add the API key (or log in) it needs, which comes
    before agreeing (for Vision, a key only in an environment variable is missing: UI jobs get
    none); ``missing_kind``: ``"login"`` for a subscription provider (no key: a proxy
    login), ``"key"`` otherwise, None when nothing is missing. ``consent`` defaults to the
    role's own list in ``config.json``: one role's consent never covers the other's."""
    key = ai_providers.consent_key(provider, role)
    if consent is None:
        try:
            consent = (
                vision_assistant_settings() if role == "vision" else assistant_settings()
            ).consent
        except (ValueError, TypeError):  # config.json is invalid: no consent is known
            consent = ()
    # a Vision job started here gets no key variables: only a keyring key counts for it
    missing = (
        ai_providers.missing_ui_job_key(provider)
        if role == "vision"
        else ai_providers.missing_key(provider)
    )
    kind = "login" if ai_providers.is_subscription(provider) else "key"
    return {
        "needed": key is not None and key not in consent,
        "key": key,
        "host": provider.host,
        "missing_key": missing,
        "missing_kind": None if missing is None else kind,
    }


def check_projects_dir(value: Any) -> tuple[Path, bool]:
    """A folder the user may make the projects folder: absolute, a folder (the last part is created
    if only it is missing) and writable; and whether it was created here.

    Raises:
        HTTPException: 400 naming what is wrong.
    """
    if not isinstance(value, str) or not value.strip():
        raise bad("projects_dir must be a folder path")
    folder = Path(_typed_path(value))  # pasted with Copy as path's quotes too
    if not folder.is_absolute():
        raise bad("the projects folder must be an absolute path")
    folder = Path(os.path.abspath(folder))  # noqa: PTH100 - drops '..', keeps mapped drive letters
    created = False
    try:
        if not folder.exists():
            if not folder.parent.is_dir():
                raise bad(f"{folder.parent} does not exist")
            folder.mkdir()
            created = True
        if not folder.is_dir():
            raise bad(f"{folder} is not a folder")
        with tempfile.TemporaryFile(dir=folder):
            pass
    except (OSError, ValueError) as err:
        if created:
            with contextlib.suppress(OSError):
                folder.rmdir()
        raise bad(f"cannot use {folder}: {getattr(err, 'strerror', None) or err}") from err
    return folder, created
