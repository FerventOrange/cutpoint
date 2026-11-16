"""What ``create_app`` shares with the route groups it registers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fastapi import APIRouter

from cutpoint.ui.jobs import Jobs
from cutpoint.ui.media import Media
from cutpoint.ui.projects import Sources


@dataclass
class AppContext:
    jobs: Jobs
    media: Media
    sources: Sources
    secure: APIRouter
    owned: APIRouter
    proxy_error: Callable[..., str | None]
