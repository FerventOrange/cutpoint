"""Session, pick, settings, assistant key, consent, model and login routes."""

from __future__ import annotations

import contextlib
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from cutpoint import paths
from cutpoint.assistant import cliproxy
from cutpoint.assistant import keys as ai_keys
from cutpoint.assistant import providers as ai_providers
from cutpoint.config import (
    VISION_PROVIDERS,
    AssistantSettings,
    assistant_settings,
    update_config,
    update_section,
    vision_assistant_settings,
)
from cutpoint.ui.analysis import _flag, ui_styles
from cutpoint.ui.common import bad
from cutpoint.ui.picker import folder_opener, path_picker, pick_start
from cutpoint.ui.routes.assistant import assistant_provider, vision_provider
from cutpoint.ui.routes.shared import proxy_error
from cutpoint.ui.settings import (
    LOGIN_TIMEOUT_S,
    _tuple_of_str,
    _vision_key_shared,
    check_assistant,
    check_projects_dir,
    config_data,
    consent_view,
    key_hosts,
    key_status,
    settings_view,
)


# -- assistant provider: key and consent (#346); config.json and the keyring only -------
def keyed_provider(body: dict[str, Any]) -> str:
    name = body.get("provider")
    if name not in ai_keys.KEYED:
        raise bad(f"provider must be one of {', '.join(ai_keys.KEYED)}")
    return name


def role_of(value: Any) -> str:
    """A request's ``role``: ``assistant`` (also when not sent) or ``vision``; else 400."""
    role = value or "assistant"
    if role not in ("assistant", "vision"):
        raise bad("role must be assistant or vision")
    return role


def role_section(body: dict[str, Any]) -> str:
    """The config section of the request's ``role`` (:func:`role_of`): ``assistant`` or
    ``vision_assistant``; its base URL names a key's host, its ``consent`` the role's."""
    return "vision_assistant" if role_of(body.get("role")) == "vision" else "assistant"


# -- subscription logins (Claude Code, Codex, Antigravity): managed via CLIProxyAPI ----
def login_provider(value: Any) -> str:
    if value not in cliproxy.PROVIDERS:
        raise bad(f"provider must be one of {', '.join(cliproxy.PROVIDERS)}")
    return value


secure = APIRouter()


@secure.get("/session")
def session():
    return {
        "styles": ui_styles(),
        "can_reveal": folder_opener() is not None,
        "can_pick": path_picker() is not None,
    }


@secure.post("/pick")
def pick_path(request: Request, kind: str, start: str = ""):
    """Ask for a video (``kind=file``) or a folder in this machine's own open dialog; waits
    until the user closes it. ``{"path": null}`` when they cancel."""
    picker = request.app.state.picker
    if kind not in ("file", "folder"):
        raise bad("kind must be 'file' or 'folder'")
    return {"path": picker.ask(kind, pick_start(start))}


@secure.get("/settings")
def get_settings():
    return settings_view(config_data())


@secure.put("/settings")
def put_settings(request: Request, body: dict[str, Any]):
    """Change the projects folder, the vision and visual index defaults and/or the assistant's provider, model
    and base URL in ``config.json``. Existing projects stay where they are. A changed
    assistant is checked at once (reachable, key accepted, tool calls or JSON output):
    ``assistant_check`` is why it cannot answer, or None. Nothing (a key least of all) is
    sent to a hosted provider the user has not agreed to in AI Chat (#504): its check waits
    for the consent, and ``assistant_note`` says so."""
    jobs = request.app.state.jobs
    if not {"projects_dir", "vision", "visual_index", "assistant", "vision_assistant"} & set(body):
        raise bad("send projects_dir, vision, visual_index, assistant and/or vision_assistant")
    updates: dict[str, Any] = {}
    removals: tuple[str, ...] = ()
    for flag in ("vision", "visual_index"):
        if flag in body:
            updates[flag] = _flag(body, flag)
    if updates.get("vision") is False:  # unticked: back to the default, not "vision": false
        del updates["vision"]
        removals = ("vision",)
    data = config_data()  # an unreadable config.json: refuse before creating the folder
    existing_ast = data.get("assistant")
    existing_ast_prov = existing_ast.get("provider") if isinstance(existing_ast, dict) else None
    existing_v_ast = data.get("vision_assistant")
    existing_v_ast_prov = (
        existing_v_ast.get("provider") if isinstance(existing_v_ast, dict) else None
    )
    assistant = (
        check_assistant(body["assistant"], "assistant", existing_provider=existing_ast_prov)
        if "assistant" in body
        else None
    )
    vision_ast = (
        check_assistant(
            body["vision_assistant"],
            section_name="vision_assistant",
            existing_provider=existing_v_ast_prov,
        )
        if "vision_assistant" in body
        else None
    )
    moving = "projects_dir" in body
    if moving and paths.env("PROJECTS_DIR") is not None:
        raise bad("CUTPOINT_PROJECTS_DIR sets the projects folder; change or unset it there", 409)
    busy = "wait for the running job to finish before changing the projects folder"
    if moving and any(j.status == "running" for j in list(jobs.jobs.values())):
        raise bad(busy, 409)
    folder, created = None, False
    if moving:
        folder, created = check_projects_dir(body.get("projects_dir"))
        updates["projects_dir"] = str(folder)

    def merge(now: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
        # the sections as they are in the file now: a change saved meanwhile stays
        values = dict(updates)
        for name, given in (("assistant", assistant), ("vision_assistant", vision_ast)):
            if given is not None:
                raw = now.get(name)
                section = raw if isinstance(raw, dict) else {}
                # checked again against the provider saved meanwhile (e.g. a subscription
                # one, which takes no url): a refusal writes nothing
                check_assistant(given, name, existing_provider=section.get("provider"))
                values[name] = {**section, **given}
        return values, removals

    # The jobs lock makes the running-job check and the write atomic against Jobs.start,
    # so no job starts against the old folder while the setting changes.
    try:
        with jobs.lock:
            if moving and any(j.status == "running" for j in jobs.jobs.values()):
                raise bad(busy, 409)
            data = update_config(merge)
    except (ValueError, OSError, HTTPException) as err:
        if created and folder is not None:
            with contextlib.suppress(OSError):
                folder.rmdir()
        if isinstance(err, HTTPException):
            raise
        if isinstance(err, ValueError):  # config.json is not valid JSON: never overwrite it
            raise bad(str(err), 409) from err
        raise bad(f"could not save the settings: {err.strerror or err}", 500) from err
    out = settings_view(data)
    if assistant is not None:
        out["assistant_note"] = None
        try:
            provider = ai_providers.make()
            # as the model list and AI Chat: a start in flight is waited for and a saved
            # login starts the proxy (none without one), so a Save right after a restart is
            # not "not running", with or without the consent
            if ai_providers.is_subscription(provider) and (
                why := proxy_error(request.app, logged_in_only=True)
            ):
                out["assistant_check"] = why
                return out
            if consent_view(provider)["needed"]:
                # nothing goes to a host not agreed to yet, a key least of all (#504);
                # a missing key is known without asking
                out["assistant_check"] = ai_providers.missing_key(provider)
                if out["assistant_check"] is None:
                    out["assistant_note"] = ai_providers.consent_note(provider)
                return out
            out["assistant_check"] = ai_providers.check(provider)
            if out["assistant_check"] is None:  # reached; maybe without using the key
                out["assistant_note"] = ai_providers.check_note(provider)
        except (ValueError, TypeError) as err:  # another invalid value in config.json
            out["assistant_check"] = str(err)
    return out


@secure.post("/assistant/key")
def save_key(body: dict[str, Any]):
    """Save ``key`` for ``provider`` in the system keyring, for the host
    its configured base URL names (``key_hosts`` of ``role``'s section): a key saved for
    OpenRouter is never sent to OpenAI. The answer says only where each key now comes from,
    never a key."""
    name = keyed_provider(body)
    section = role_section(body)
    data = config_data()
    try:
        vision = section == "vision_assistant"
        ai_keys.save(
            name,
            body.get("key"),
            key_hosts(data, section)[name],
            vision=vision,
            shared=not vision or _vision_key_shared(data, name),
        )
    except ValueError as err:
        raise bad(str(err)) from None
    except ai_keys.KeyringError as err:
        raise bad(str(err), 500) from None
    return key_status(data, section)


@secure.post("/assistant/key/clear")
def clear_key(body: dict[str, Any]):
    """Remove ``provider``'s keyring key for its configured host (of ``role``'s section);
    a key from the environment stays."""
    name = keyed_provider(body)
    section = role_section(body)
    data = config_data()
    try:
        ai_keys.clear(name, key_hosts(data, section)[name])
    except ai_keys.KeyringError as err:
        raise bad(str(err), 500) from None
    return key_status(data, section)


@secure.post("/assistant/consent")
def give_consent(body: dict[str, Any]):
    """Record that the user agreed to send project data to the provider in use for ``role``
    (``assistant``, the default, or ``vision``), named by its ``key``, in that role's own
    ``consent`` list; 409 when the provider in use is another one by now, or is local."""
    role, section_name = role_of(body.get("role")), role_section(body)
    provider = vision_provider() if role == "vision" else assistant_provider()
    want = ai_providers.consent_key(provider, role)
    if want is None or body.get("key") != want:
        raise bad(f"the {role} provider changed: check it and agree again", 409)

    def agree(section: dict[str, Any]) -> dict[str, Any] | None:
        given = list(_tuple_of_str(section.get("consent")))
        return None if want in given else {**section, "consent": [*given, want]}

    try:
        section = update_section(section_name, agree)  # re-read under the config lock
    except ValueError as err:
        raise bad(str(err), 409) from err
    except OSError as err:
        raise bad(f"could not save the consent: {err.strerror or err}", 500) from err
    consent_list = _tuple_of_str(section.get("consent"))
    return {"consent": consent_view(provider, consent=consent_list, role=role)}


@secure.get("/assistant/models")
def assistant_models(
    request: Request,
    provider: str = "ollama",
    role: str = "assistant",
    refresh: bool = False,
):
    """The models a provider offers, for Settings' Model field: ``{state, models, message}``
    (:func:`cutpoint.assistant.providers.model_list`). Accepts role=assistant|vision."""
    app = request.app
    role = role_of(role)
    try:
        if role == "vision":
            if provider not in VISION_PROVIDERS and provider != "custom":
                raise bad(f"provider must be custom or one of {', '.join(VISION_PROVIDERS)}")
            settings = vision_assistant_settings()
        else:
            if (
                provider not in ai_providers.PRESETS
                and provider not in cliproxy.PROVIDERS
                and provider != "custom"
            ):
                raise bad(f"provider must be custom or one of {', '.join(ai_providers.PRESETS)}")
            settings = assistant_settings()
        saved = settings.provider == provider and ai_providers.is_preset(
            settings.provider, settings.url
        )
        if provider == "custom" or saved:
            p = (ai_providers.make_vision if role == "vision" else ai_providers.make)(settings)
        elif provider in ai_providers.PRESETS:
            p = ai_providers.preset(provider, settings.context_tokens)
        else:  # a subscription provider, checked above
            p = ai_providers.make(
                AssistantSettings(provider=provider, context_tokens=settings.context_tokens)
            )
    except (ValueError, TypeError) as err:  # a bad Ollama port, an invalid config.json
        raise bad(str(err), 409) from err
    if ai_providers.is_subscription(p) and (why := proxy_error(app, logged_in_only=True)):
        return {"state": "error", "models": [], "message": why}
    keep = settings.model if provider in (settings.provider, "custom") else ""
    return ai_providers.model_list(p, settings.consent, role=role, refresh=refresh, keep=keep)


@secure.get("/assistant/logins")
def assistant_logins(request: Request):
    """Each subscription's login (cliproxy.status: never a token) with its title and
    whether the assistant can use it as a provider; ``proxy_running`` and ``saved_login``
    (a login an earlier proxy run kept): the page starts the proxy when both say so."""
    app = request.app
    proxies = app.state.proxies
    running = proxies.running()
    st_map = {}
    with contextlib.suppress(cliproxy.ProxyError, OSError):
        for s in cliproxy.statuses():
            if isinstance(s, dict) and "provider" in s:
                st_map[s["provider"]] = s
    out = []
    for p in cliproxy.PROVIDERS:
        st = st_map.get(p)
        if not st:
            try:
                st = cliproxy.status(p)
            except (cliproxy.ProxyError, OSError) as err:
                st = {
                    "provider": p,
                    "logged_in": False,
                    "email": None,
                    "org": None,
                    "expired": False,
                    "expires_ms": None,
                    "error": str(err),
                }
        if st.get("error") == cliproxy.NOT_RUNNING:  # this server starts it when needed
            st = {**st, "error": None}
        out.append(
            {
                "provider": p,
                **st,
                "title": cliproxy.TITLES.get(p, p),
            }
        )
    saved = not running and proxies.saved_login()
    return {"logins": out, "proxy_running": running, "saved_login": saved}


@secure.post("/assistant/proxy")
def start_proxy(request: Request):
    """Start the subscription proxy when a login is saved and none runs (#821); 503 with
    the reason when it does not start. ``{running}``."""
    app = request.app
    if why := proxy_error(app, logged_in_only=True):
        raise bad(why, 503)
    return {"running": app.state.proxies.running()}


@secure.post("/assistant/login")
def start_login(request: Request, body: dict[str, Any]):
    """Start ``provider``'s login via CLIProxyAPI and answer with the sign-in ``url`` for
    the page to open; ``GET .../login/status`` tells when it is done. 409 while one waits."""
    app, login_lock, login_states = (
        request.app,
        request.app.state.login_lock,
        request.app.state.login_states,
    )
    name = login_provider(body.get("provider"))
    now = time.monotonic()
    old_state_to_cancel = None
    with login_lock:
        cur = login_states.get(name)
        if cur and cur.get("state") == "pending":
            if now - cur.get("started_at", 0) < LOGIN_TIMEOUT_S:
                raise bad(f"a {name} login is already waiting for the browser", 409)
            old_state_to_cancel = cur.get("proxy_state")
        login_states[name] = {"state": "pending", "started_at": now, "error": ""}
    if old_state_to_cancel:
        with contextlib.suppress(Exception):
            cliproxy.cancel_login(old_state_to_cancel)
    app.state.proxies.wait_for_start()  # a start in flight answers only once it is done
    try:
        try:
            res = cliproxy.start_login(name)
        except cliproxy.ProxyConnectionError as first:
            # no proxy ran: a login always starts one (#821); one that ran and failed stands
            try:
                started = app.state.proxies.ensure()
            except (cliproxy.ProxyError, OSError) as err:
                why = ai_keys.redact(f"the subscription proxy did not start: {err}")
                with login_lock:
                    login_states[name] = {"state": "error", "error": why}
                raise bad(why, 503) from None
            if not started and str(first) != cliproxy.NOT_RUNNING:
                raise
            res = cliproxy.start_login(name)  # ours, or one another request just started
    except (cliproxy.ProxyError, OSError) as err:
        with login_lock:
            login_states[name] = {
                "state": "error",
                "error": ai_keys.redact(str(err)) or type(err).__name__,
            }
        # proxy not running or unreachable: expected, so 503 rather than a server error
        code = 503 if isinstance(err, cliproxy.ProxyConnectionError) else 500
        raise bad(ai_keys.redact(str(err)) or f"the {name} login did not start", code) from None

    url = res.get("url")
    proxy_state = res.get("state")
    if not url or not proxy_state:
        with login_lock:
            login_states[name] = {"state": "error", "error": f"the {name} login did not start"}
        raise bad(f"the {name} login did not start", 500)

    with login_lock:
        login_states[name] = {
            "state": "pending",
            "proxy_state": proxy_state,
            "started_at": time.monotonic(),  # the browser gets the whole time after a start
            "error": "",
        }
    return {"url": url}


@secure.get("/assistant/login/status")
def login_status(request: Request, provider: str):
    """``{state, error}`` of ``provider``'s last login: ``pending``, ``done`` or ``error``
    (``idle`` before any)."""
    login_lock, login_states = request.app.state.login_lock, request.app.state.login_states
    name = login_provider(provider)
    with login_lock:
        info = login_states.get(name)
        if not info:
            return {"state": "idle", "error": ""}
        if info.get("state") != "pending":
            return {"state": info.get("state", "idle"), "error": info.get("error", "")}

        proxy_state = info.get("proxy_state")
        started_at = info.get("started_at", 0)
        timed_out = time.monotonic() - started_at > LOGIN_TIMEOUT_S
        if timed_out:
            info["state"] = "error"
            info["error"] = "login timed out"
        elif not proxy_state:  # start_login has not had the proxy's answer yet: nothing to poll
            return {"state": "pending", "error": ""}
    if timed_out:
        if proxy_state:
            with contextlib.suppress(Exception):
                cliproxy.cancel_login(proxy_state)
        return {"state": "error", "error": "login timed out"}
    try:
        poll_res = cliproxy.poll_login(proxy_state)
    except (cliproxy.ProxyError, OSError) as err:
        poll_res = {"state": "error", "error": ai_keys.redact(str(err)) or type(err).__name__}

    with login_lock:
        cur = login_states.get(name)
        if cur and cur.get("proxy_state") == proxy_state:
            p_state = poll_res.get("state", "pending")
            if p_state == "done":
                ai_providers.clear_model_cache(name)
                cur["state"] = "done"
                cur["error"] = ""
            elif p_state == "error":
                cur["state"] = "error"
                cur["error"] = poll_res.get("error") or "login failed"
            elif p_state == "pending":
                cur["state"] = "pending"
            return {"state": cur.get("state", "pending"), "error": cur.get("error", "")}
        return {
            "state": (cur or {}).get("state", "idle"),
            "error": (cur or {}).get("error", ""),
        }


@secure.post("/assistant/logout")
def logout(body: dict[str, Any]):
    """Remove ``provider``'s login from CLIProxyAPI."""
    name = login_provider(body.get("provider"))
    try:
        cliproxy.logout(name)
    except (cliproxy.ProxyError, OSError) as err:
        raise bad(str(err), 500) from None
    ai_providers.clear_model_cache(name)
    st = cliproxy.status(name)
    return {"provider": name, **st, "title": cliproxy.TITLES.get(name, name)}
