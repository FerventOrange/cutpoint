"""Helpers more than one route group calls, reading the app's shared objects from ``app.state``."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException

from cutpoint.assistant import cliproxy
from cutpoint.assistant import keys as ai_keys
from cutpoint.assistant import ops as ai_ops
from cutpoint.assistant import store as ai_store
from cutpoint.config import remap_source
from cutpoint.ui.common import bad
from cutpoint.ui.files import _highlights
from cutpoint.ui.projects import moments_view, validate_segments


def proxy_error(app: FastAPI, *, logged_in_only: bool) -> str | None:
    """Start the subscription proxy unless one runs (:class:`cliproxy.ProxyLauncher`;
    ``logged_in_only``: only when a login is saved); why it did not start, else None."""
    try:
        app.state.proxies.ensure(logged_in_only=logged_in_only)
    except (cliproxy.ProxyError, OSError) as err:
        return ai_keys.redact(f"the subscription proxy did not start: {err}")
    return None


def assistant_state(
    app: FastAPI, p: Path, segments: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], ai_ops.Durations, str]:
    """``(moments, source durations, base_key)`` of the project with ``segments``."""
    media, sources = app.state.media, app.state.sources
    hl = _highlights(p)
    durations = ai_ops.Durations(
        {s["path"]: s["duration"] for s in sources.entries(p)},
        lambda path: media.duration(remap_source(path)),
    )
    moments = (moments_view(hl) or {}).get("candidates") or []
    return moments, durations, ai_ops.base_key(segments, hl)


def assistant_revert(
    app: FastAPI, p: Path, hid: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``(history entry, its "before" checked again)``: what Revert puts back."""
    media, sources = app.state.media, app.state.sources
    entry = next((e for e in ai_store.history(p) if e.get("id") == hid), None)
    if entry is None:
        raise bad("unknown change", 404)
    try:
        return entry, validate_segments(entry.get("before"), sources.entries(p), media)
    except HTTPException as err:
        raise bad(f"this change can no longer be reverted: {err.detail}", 409) from err


def url(app: FastAPI, pid: str, kind: str, name: str) -> str:
    media_t = app.state.media_token
    return f"/api/projects/{quote(pid, safe='')}/{kind}/{quote(name, safe='')}?t={media_t}"


def window_url(app: FastAPI, pid: str, sid: str, lo: int, hi: int) -> str:
    return url(app, pid, "sources", sid).replace("?", f"/window/{lo}-{hi}?", 1)


def director_url(app: FastAPI, pid: str, sid: str, name: str) -> str:
    return url(app, pid, "sources", sid).replace("?", f"/director/{name}?", 1)


def pano_views(
    app: FastAPI,
    pid: str,
    p: Path,
    listing: list[dict[str, Any]],
    only: str | None = None,
    hl: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """The camera path flags and director preview of each candidate (by CLI id) of a 360
    clip, a ready preview with its url (under the source the page plays it as). ``hl``:
    the project's highlights.json when the caller has read it already."""
    previews = app.state.previews
    previews.prune(p)
    views = previews.listing(p)
    if only is not None:
        views = {only: views[only]} if only in views else {}
    hl = (hl if hl is not None else _highlights(p)) or {}
    where = {str(c.get("id")): c.get("source") for c in hl.get("candidates") or []}
    sids = {s["path"]: s["id"] for s in listing}
    for cid, view in views.items():
        pv = view["preview"]
        if pv is not None and pv["state"] == "ready":
            sid = sids.get(where.get(cid))
            if sid is None:  # not one of the project's sources: nothing may serve it
                view["preview"] = None
            else:
                pv["url"] = director_url(app, pid, sid, pv.pop("name"))
    return views
