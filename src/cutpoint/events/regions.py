"""Per-game HUD regions for OCR, in 1920x1080 reference pixels.

Games use JSON presets, looked up in this order: the project's
``<project>/hud_presets/<game>.json``, the user presets in
``<CUTPOINT_HOME>/presets`` (where region discovery writes by default), then
the presets bundled with the package (``cutpoint/events/presets``). They are
written by region discovery (:mod:`cutpoint.events.discover`) and calibrated
by hand. No game presets ship with the package; the bundled folder is kept
for generic ones.

An installed preset marked ``"builtin": true`` keeps the original R6
interpretation (coloured name boxes, its own banner list), so its OCR cache
keys and results stay those of the former built-in R6 preset.

The user and bundled folders are the installed, trusted tiers: a preset copied
into ``<CUTPOINT_HOME>/presets`` behaves exactly like the same file bundled with
the package (same ``"builtin"`` handling, same OCR and VLM cache keys). Project
presets are per-project overrides and are never trusted.

Preset JSON::

    {"game": "my-game", "title": "My Game", "match": ["my game"],
     "player_tags": [],
     "regions": [{"name": "killfeed", "x": 1400, "y": 40, "w": 500, "h": 160,
                  "kind": "feed", "gate": "det", "every": 1}, ...],
     "banner_phrases": ["ROUND WON", ...],
     "feed": {"kill_words": {"eliminated": "kill", ...}, "colors": {...}},
     "bigtext": true,
     "view": "first-person", "hype": "  hype: 0-1 ...\\n",
     "events": [{"name": "objective", "when": [...], "score": 6}, ...]}

``view`` and ``hype`` adjust the VLM prompt; ``events`` defines event types
(:mod:`cutpoint.events.eventtypes`), whose banner ``prefix`` phrases are
added to the phrases of JSON presets.

Region entries take any :class:`Region` field (``min_change``, ``min_fill``,
...); missing ones use the defaults. ``"builtin": true`` selects the original R6
interpretation: ``banner_phrases`` are used as listed, without
:data:`GENERIC_BANNERS`, :meth:`HudPreset.key` adds no fields and the preset
records no ``path``. It is honoured only in installed presets (user or
bundled); a project preset that sets it is loaded as a normal JSON preset
(with a warning).
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


def _regions(data: dict[str, Any]) -> tuple[Region, ...]:
    """Regions of preset JSON (unknown keys are ignored)."""
    names = {f.name for f in fields(Region)}
    return tuple(
        Region(**{k: v for k, v in r.items() if k in names}) for r in data.get("regions", [])
    )


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
        game: Preset key (the preset's ``game``, or a slug of the game's name).
        regions: HUD rectangles to OCR.
        phrases: Banner phrases (built-in R6: its own list only; JSON presets:
            their list plus :data:`GENERIC_BANNERS`).
        builtin: True for ``"builtin": true`` presets (R6; original
            interpretation).
        player_tags: Extra in-game names listed in the preset.
        feed: Feed parsing options (``kill_words``, ``colors``, ...).
        bigtext: Add the generic big-centred-text check.
        path: JSON file the preset came from (None for built-in ones).
        title: Human-readable game name (VLM prompt).
        view: Camera view for the VLM prompt ("first-person"; empty: none).
        hype: The VLM prompt's hype rubric line (empty: the generic one).
        events: Event types (raw ``events`` entries, see
            :mod:`cutpoint.events.eventtypes`).
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
    view: str = ""
    hype: str = ""
    events: tuple[dict[str, Any], ...] = ()

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


_warned_invalid: set[Path] = set()  # invalid presets already warned about (once per path)


def _read_json(path: Path) -> dict[str, Any] | None:
    data = cache.load_json(path)
    if not isinstance(data, dict):
        if path not in _warned_invalid:
            _warned_invalid.add(path)
            logger.warning("Ignoring invalid HUD preset %s", path)
        return None
    return data


_warned_builtin: set[Path] = set()


def is_installed_path(path: Path | None) -> bool:
    """Whether a preset at ``path`` is installed: bundled (``path`` None, or in
    :data:`~cutpoint.paths.PRESETS_DIR`) or a user preset (in
    :func:`~cutpoint.paths.presets_dir`), not a project one."""
    return path is None or path.parent in (paths.PRESETS_DIR, paths.presets_dir())


def _is_builtin(data: dict[str, Any], path: Path | None) -> bool:
    """``"builtin": true`` counts only for installed presets: bundled (``path`` None
    or in :data:`~cutpoint.paths.PRESETS_DIR`) or user presets (in
    :func:`~cutpoint.paths.presets_dir`)."""
    if not data.get("builtin"):
        return False
    if is_installed_path(path):
        return True
    if path not in _warned_builtin:
        _warned_builtin.add(path)
        logger.warning(
            'Ignoring "builtin" in HUD preset %s: only user or bundled presets can set it', path
        )
    return False


def preset_from_dict(data: dict[str, Any], path: Path | None = None) -> HudPreset:
    """Build a :class:`HudPreset` from preset JSON (unknown keys are ignored)."""
    builtin = _is_builtin(data, path)
    phrases = tuple(map(str, data.get("banner_phrases", [])))
    events = tuple(e for e in data.get("events") or [] if isinstance(e, dict))
    if not builtin:
        phrases = tuple(dict.fromkeys([*phrases, *GENERIC_BANNERS, *_event_phrases(events)]))
    return HudPreset(
        game=str(data.get("game") or (path.stem if path else "game")),
        regions=_regions(data),
        phrases=phrases,
        builtin=builtin,
        player_tags=tuple(data.get("player_tags", [])),
        feed=dict(data.get("feed") or {}),
        bigtext=bool(data.get("bigtext", True)),
        path=None if builtin else path,
        title=str(data.get("title", "")),
        view=str(data.get("view", "")),
        hype=str(data.get("hype", "")),
        events=events,
    )


def _event_phrases(events: tuple[dict[str, Any], ...]) -> list[str]:
    """Banner ``prefix`` phrases of event types, so OCR looks for them too."""
    out = []
    for e in events:
        for m in e.get("when") or []:
            if isinstance(m, dict) and m.get("type") == "banner":
                p = m.get("prefix")
                out += [str(x).upper() for x in (p if isinstance(p, list) else [p] if p else [])]
    return out


def load_preset(game: str | None, project: Path | None = None) -> HudPreset | None:
    """The HUD preset for ``game``.

    Lookup order: project, user (``<CUTPOINT_HOME>/presets``), then bundled.

    Returns:
        The preset, or None when ``game`` has none (discovery can create one).
    """
    if not game:
        return None
    for path in preset_files(project):
        data = _read_json(path)
        if data and (data.get("game") or path.stem) == game:
            return preset_from_dict(data, path)
    return None


def preset_for_path(
    path: str, override: str | None = None, project: Path | None = None
) -> str | None:
    """Game key for ``path`` (or ``override`` when not "auto").

    The ``match`` hints of JSON presets, in lookup order (project, user, bundled).
    Returns None when nothing matches (see :func:`game_key_for_path`).
    """
    if override and override != "auto":
        return override
    low = path.lower()
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
