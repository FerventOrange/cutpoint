"""Helpers more than one route group calls, reading the app's shared objects from ``app.state``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
