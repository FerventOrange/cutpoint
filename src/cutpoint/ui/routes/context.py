"""What ``create_app`` shares with the route groups it registers."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI

from cutpoint.assistant import ops as ai_ops
from cutpoint.assistant import session as ai_session
from cutpoint.ui.jobs import Jobs
from cutpoint.ui.media import Media
from cutpoint.ui.pano_previews import PanoPreviews
from cutpoint.ui.picker import Picker
from cutpoint.ui.projects import MediaLookups, Sources


@dataclass
class AppContext:
    jobs: Jobs
    media: Media
    picker: Picker
    sources: Sources
    lookups: MediaLookups
    previews: PanoPreviews
    login_lock: threading.Lock
    login_states: dict[str, dict[str, Any]]
    app: FastAPI
    secure: APIRouter
    owned: APIRouter
    served: APIRouter
    url: Callable[[str, str, str], str]
    window_url: Callable[[str, str, int, int], str]
    pano_views: Callable[..., dict[str, dict[str, Any]]]
    proxy_error: Callable[..., str | None]
    turns: ai_session.Turns
    assistant_state: Callable[
        [Path, list[dict[str, Any]]], tuple[list[dict[str, Any]], ai_ops.Durations, str]
    ]
    assistant_revert: Callable[[Path, str], tuple[dict[str, Any], list[dict[str, Any]]]]
