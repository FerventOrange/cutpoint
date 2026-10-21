"""Per-game HUD regions for OCR, in 1920x1080 reference pixels.

Built in, for the optional Rainbow Six Siege module (calibrated on full frames of a
1080p stream):

R6 Siege (2025 HUD):
    * kill feed: right-aligned rows at x~1590-1830, newest row at y~296, older
      rows stacking upward ~32 px apart. Blue name boxes = own team, orange =
      enemy. "Reverse Friendly Fire has been activated for X" appears here too.
    * "5 VS 1" counter: centre, x~880-1040, y~230-290.
    * round banner ("WON ROUND 5", "VICTORY"): centre, x~650-1250, y~430-540.
    * notices ("OPFOR last op standing."): centre, y~765-795.

Other games use JSON presets, looked up in this order: the project's
``<project>/hud_presets/<game>.json``, the user presets in
``<CUTPOINT_HOME>/presets`` (where region discovery writes by default), then
the presets bundled with the package (``cutpoint/events/presets``). They are
written by region discovery (:mod:`cutpoint.events.discover`) and calibrated
by hand. War Thunder's bundled preset is ``presets/wt.json``. R6 stays built in, so its OCR cache keys and
results are unchanged.

Preset JSON::

    {"game": "wt", "title": "War Thunder", "match": ["war thunder"],
     "player_tags": [],
     "regions": [{"name": "killfeed", "x": 1380, "y": 560, "w": 540, "h": 130,
                  "kind": "feed", "gate": "det", "every": 1}, ...],
     "banner_phrases": ["TARGET DESTROYED", ...],
     "feed": {"kill_words": {"destroyed": "kill", ...}, "colors": {...}},
     "bigtext": true}
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from cutpoint import cache, paths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Region:
    """A HUD rectangle to OCR.

    Attributes:
        name: Region id ("killfeed", "banner", ...).
        x, y, w, h: Rectangle in 1920x1080 reference pixels.
        kind: How OCR text is interpreted: "killfeed" (R6 coloured name
            boxes), "feed" (generic stacked text lines: "A destroyed B" or
            coloured names), "banner", "bigtext" (generic large centred
            text), "counter", "notice" or "log".
        min_change: Fraction of the downscaled HUD-pixel mask that must change
            before the region is OCR'd again ("mask" gate).
        min_fill: Minimum fraction of HUD pixels for the region to be OCR'd at
            all (big banner text fills ~3-8% of the banner region).
        gate: When to run recognition: "mask" (the R6 HUD-colour pixel mask
            changed) or "det" (game-agnostic: RapidOCR text *detection* on the
            crop; recognition only when the text-box layout changed).
        every: Process the region on every n-th sampled frame only (banners
            need ~1 fps, kill feeds the full OCR rate).
        min_text_h: Generic banners: only text lines at least this tall
            (reference px) count.
        det_scale: "det" gate: run text detection on the crop downscaled by
            this factor (big text survives 0.35x; feeds need ~0.6x).
    """

    name: str
    x: int
    y: int
    w: int
    h: int
    kind: str
    min_change: float = 0.015
    min_fill: float = 0.002  # skip OCR when fewer HUD pixels than this fraction
    gate: str = "mask"
    every: int = 1
    min_text_h: int = 0
    det_scale: float = 1.0

    def key(self) -> dict[str, Any]:
        """Cache-key dict. ``gate``/``every``/``min_text_h`` are left out at their
        defaults, so keys of the built-in R6 regions stay identical."""
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in _OPTIONAL_KEY_FIELDS and v == f.default:
                continue
            out[f.name] = v
        return out


_OPTIONAL_KEY_FIELDS = ("gate", "every", "min_text_h", "det_scale")

PRESETS: dict[str, tuple[Region, ...]] = {
    "r6": (
        # min_change is low: the grey "Reverse Friendly Fire" row changes the
        # mask by only ~1.4% of the region (measured).
        Region("killfeed", 1300, 150, 560, 185, "killfeed", min_change=0.008),
        Region("counter", 760, 200, 400, 110, "counter"),
        Region("banner", 520, 380, 880, 220, "banner", min_change=0.03, min_fill=0.012),
        Region("notice", 600, 740, 720, 80, "notice", min_change=0.03, min_fill=0.006),
    ),
}

# Substrings of the source path that identify a game (lower-case). JSON presets
# add their own "match" hints.
GAME_HINTS: dict[str, tuple[str, ...]] = {
    "r6": ("rainbow six", "r6", "siege"),
    "wt": ("war thunder", "warthunder"),
}

# Phrases looked for in banner/notice/killfeed text of the built-in presets,
# mapped to an event type. Matching is fuzzy (see killfeed.fuzzy_contains).
BANNER_PHRASES: dict[str, tuple[str, ...]] = {
    "r6": (
        "WON ROUND",
        "ROUND WON",
        "LOST ROUND",
        "ROUND LOST",
        "VICTORY",
        "DEFEAT",
        "REVERSE FRIENDLY FIRE",
        "ACE",
        "CLUTCH",
        "OVERTIME",
        "MATCH POINT",
        "LAST OP STANDING",
        "DEFUSER PLANTED",
        "DEFUSER DISABLED",
        "BOMB DEFUSED",
        "ELIMINATED",
        "TEAM KILL",
        "TEAMKILL",
        "DRAW",
    ),
}

# Game-agnostic banner vocabulary for the generic big-text check, merged into
# every JSON preset's phrases. Matching is fuzzy.
GENERIC_BANNERS: tuple[str, ...] = (
    "VICTORY",
    "DEFEAT",
    "MISSION COMPLETE",
    "MISSION ACCOMPLISHED",
    "MISSION FAILED",
    "YOU DIED",
    "YOU WIN",
    "YOU LOSE",
    "WASTED",
    "ELIMINATED",
    "DESTROYED",
    "KILLED",
    "ACE",
    "ROUND WON",
    "ROUND LOST",
    "WON ROUND",
    "LOST ROUND",
    "LEVEL UP",
    "ACHIEVEMENT",
    "ACHIEVEMENT UNLOCKED",
    "DOUBLE KILL",
    "TRIPLE KILL",
    "MULTI KILL",
    "KILLING SPREE",
    "HEADSHOT",
    "FIRST BLOOD",
    "WINNER",
    "GAME OVER",
    "MATCH POINT",
    "OVERTIME",
    "DRAW",
    "SHOT DOWN",
    "CHAMPION",
    "RANK UP",
    "NEW RECORD",
)


@dataclass
class HudPreset:
    """Regions and vocabulary for one game.

    Attributes:
        game: Preset key ("r6", "wt", or a slug of the game's name).
        regions: HUD rectangles to OCR.
        phrases: Banner phrases (built-in R6: its own list only; JSON presets:
            their list plus :data:`GENERIC_BANNERS`).
        builtin: True for the built-in R6 preset (original interpretation).
        player_tags: Extra in-game names listed in the preset.
        feed: Feed parsing options (``kill_words``, ``colors``, ...).
        bigtext: Add the generic big-centred-text check.
        path: JSON file the preset came from (None for built-in).
        title: Human-readable game name (VLM prompt).
    """

    game: str
    regions: tuple[Region, ...]
    phrases: tuple[str, ...]
    builtin: bool = False
    player_tags: tuple[str, ...] = ()
    feed: dict[str, Any] = field(default_factory=dict)
    bigtext: bool = True
    path: Path | None = None
    title: str = ""

    def all_regions(self) -> tuple[Region, ...]:
        """Preset regions plus the generic big-text region when enabled."""
        if self.bigtext and not any(r.kind == "bigtext" for r in self.regions):
            return (*self.regions, bigtext_region())
        return self.regions

    def key(self) -> dict[str, Any]:
        """Extra cache-key fields of JSON presets (empty for built-in R6)."""
        if self.builtin:
            return {}
        return {
            "phrases": list(self.phrases),
            "feed": self.feed,
            "bigtext": self.bigtext,
            "preset_tags": list(self.player_tags),
        }


def bigtext_region() -> Region:
    """The generic big-banner check: screen centre, every 6th frame (0.5 fps at the
    default 3 fps), tall text only, detected on a 0.35x downscale."""
    return Region(
        "bigtext",
        300,
        110,
        1320,
        700,
        "bigtext",
        gate="det",
        every=6,
        min_text_h=36,
        det_scale=0.35,
    )


def preset_files(project: Path | None = None) -> list[Path]:
    """JSON preset files: project-local first, then user presets, then bundled ones."""
    dirs = ([project / "hud_presets"] if project else []) + [paths.presets_dir(), paths.PRESETS_DIR]
    return [p for d in dirs if d.is_dir() for p in sorted(d.glob("*.json"))]


def _read_json(path: Path) -> dict[str, Any] | None:
    data = cache.load_json(path)
    if not isinstance(data, dict):
        logger.warning("Ignoring invalid HUD preset %s", path)
        return None
    return data


def preset_from_dict(data: dict[str, Any], path: Path | None = None) -> HudPreset:
    """Build a :class:`HudPreset` from preset JSON (unknown keys are ignored)."""
    names = {f.name for f in fields(Region)}
    regions = tuple(
        Region(**{k: v for k, v in r.items() if k in names}) for r in data.get("regions", [])
    )
    phrases = tuple(dict.fromkeys([*map(str, data.get("banner_phrases", [])), *GENERIC_BANNERS]))
    return HudPreset(
        game=str(data.get("game") or (path.stem if path else "game")),
        regions=regions,
        phrases=phrases,
        player_tags=tuple(data.get("player_tags", [])),
        feed=dict(data.get("feed") or {}),
        bigtext=bool(data.get("bigtext", True)),
        path=path,
        title=str(data.get("title", "")),
    )


def load_preset(game: str | None, project: Path | None = None) -> HudPreset | None:
    """The HUD preset for ``game``.

    Lookup order: project, user (``<CUTPOINT_HOME>/presets``), bundled, then built-in.

    Returns:
        The preset, or None when ``game`` has none (discovery can create one).
    """
    if not game:
        return None
    for path in preset_files(project):
        data = _read_json(path)
        if data and (data.get("game") or path.stem) == game:
            return preset_from_dict(data, path)
    if game in PRESETS:
        return HudPreset(
            game, PRESETS[game], BANNER_PHRASES.get(game, ()), builtin=True, bigtext=False
        )
    return None


def preset_for_path(
    path: str, override: str | None = None, project: Path | None = None
) -> str | None:
    """Game key for ``path`` (or ``override`` when not "auto").

    Built-in hints first (R6, WT), then the ``match`` hints of JSON presets.
    Returns None when nothing matches (see :func:`game_key_for_path`).
    """
    if override and override != "auto":
        return override
    low = path.lower()
    for game, hints in GAME_HINTS.items():
        if any(h in low for h in hints):
            return game
    for p in preset_files(project):
        data = _read_json(p) or {}
        if any(str(h).lower() in low for h in data.get("match", [])):
            return str(data.get("game") or p.stem)
    return None


_DATE_RE = re.compile(r"\s+\d{4}[.\-_]\d{2}[.\-_]\d{2}.*$")


def game_key_for_path(path: str, override: str | None = None, project: Path | None = None) -> str:
    """Like :func:`preset_for_path` but never None.

    Falls back to a slug of the game name, taken from a recorder-style file name
    ("Some Game 2024.11.14 - 19.01.53.09.DVR.mp4" -> "some-game") or else the
    parent folder's name.
    """
    known = preset_for_path(path, override, project)
    if known:
        return known
    p = Path(path)
    stem = _DATE_RE.sub("", p.stem)
    return _game_slug(stem if stem != p.stem else (p.parent.name or p.stem))


def _game_slug(name: str) -> str:
    """Preset key for a game name ("Counter-strike 2" -> "counter-strike-2").

    The key is persisted (preset file stem and OCR cache key), so keep these
    exact semantics: no Unicode folding, non-ASCII characters become "-".
    """
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "game"


def scaled(region: Region, width: int, height: int) -> tuple[int, int, int, int]:
    """Region rectangle scaled to a ``width`` x ``height`` frame (even values)."""
    sx, sy = width / 1920, height / 1080
    x, y = int(region.x * sx) // 2 * 2, int(region.y * sy) // 2 * 2
    w = min(int(region.w * sx) // 2 * 2, width - x)
    h = min(int(region.h * sy) // 2 * 2, height - y)
    return x, y, w, h
