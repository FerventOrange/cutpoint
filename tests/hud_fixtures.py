"""Synthetic HUD presets for tests (invented games: no real game content)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def boxfeed(**over: Any) -> dict[str, Any]:
    """A ``"builtin": true`` preset with a title, view and hype (invented game ``boxfeed``)."""
    data: dict[str, Any] = {
        "game": "boxfeed",
        "builtin": True,
        "title": "Box Feed",
        "view": "top-down",
        "hype": "  hype: 0-1 box stacking\n",
        "player_tags": ["BOXER"],
        "regions": [
            {"name": "feed", "x": 1400, "y": 40, "w": 500, "h": 160, "kind": "feed", "every": 1},
            {"name": "banner", "x": 600, "y": 100, "w": 700, "h": 120, "kind": "banner"},
        ],
        "banner_phrases": ["STACK COMPLETE"],
        "feed": {"kill_words": {"toppled": "kill"}},
    }
    return data | over


def verbfeed(**over: Any) -> dict[str, Any]:
    """A plain (not builtin) preset with a title, view and hype (game ``verbfeed``)."""
    data = boxfeed(game="verbfeed", title="Verb Feed", banner_phrases=["VERB DONE"])
    del data["builtin"]
    return data | over


def write_preset(folder: Path, data: dict[str, Any]) -> Path:
    """Write ``data`` as ``<folder>/<game>.json`` (folder created)."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{data['game']}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path
