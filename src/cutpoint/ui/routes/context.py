"""What ``create_app`` shares with the route groups it registers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter

from cutpoint.assistant import session as ai_session
from cutpoint.ui.jobs import Jobs
from cutpoint.ui.media import Media
from cutpoint.ui.pano_previews import PanoPreviews
from cutpoint.ui.projects import MediaLookups, Sources


@dataclass
class AppContext:
    jobs: Jobs
    media: Media
    sources: Sources
    lookups: MediaLookups
    previews: PanoPreviews
    secure: APIRouter
    owned: APIRouter
    served: APIRouter
    url: Callable[[str, str, str], str]
    window_url: Callable[[str, str, int, int], str]
    pano_views: Callable[..., dict[str, dict[str, Any]]]
    proxy_error: Callable[..., str | None]
    turns: ai_session.Turns
