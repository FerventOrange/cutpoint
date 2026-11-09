"""Per-game HUD regions for OCR, in 1920x1080 reference pixels.

Games use JSON presets, looked up in this order: the project's
``<project>/hud_presets/<game>.json``, the user presets in
``<CUTPOINT_HOME>/presets`` (where region discovery writes by default), then
the presets bundled with the package (``cutpoint/events/presets``). They are
written by region discovery (:mod:`cutpoint.events.discover`) and calibrated
by hand. No game presets ship with the package; the bundled folder is kept
for generic ones.

The user and bundled folders are the installed, trusted tiers: a preset copied
into ``<CUTPOINT_HOME>/presets`` behaves exactly like the same file bundled with
the package (same OCR and VLM cache keys). Project presets are per-project
overrides and are never trusted.

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
...); missing ones use the defaults. Other keys are ignored: region discovery
marks the extra notice regions it adds where text shows outside every other
region with ``"origin": "outside"`` and the vision model's name for a discovered
region as ``"label"`` (``kill_feed``, ``chat``, ...); neither changes the region or its
cache key.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
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
        kind: How OCR text is interpreted: "feed" (stacked text lines: "A
            destroyed B" or coloured names), "banner", "bigtext" (large
            centred text), "notice" or "score" (a small on-screen number such as a
            round score or a counter; :class:`cutpoint.events.killfeed.ScoreTracker`).
        min_change: Fraction of the downscaled HUD-pixel mask that must change
            before the region is OCR'd again ("mask" gate).
        min_fill: Minimum fraction of HUD pixels for the region to be OCR'd at
            all (big banner text fills ~3-8% of the banner region).
        gate: When to run recognition: "mask" (the HUD-colour pixel mask
            changed), "det" (game-agnostic: RapidOCR text *detection* on the
            crop; recognition only when the text-box layout changed) or "read"
            (detection and recognition on every frame the region is due).
        every: Process the region on every n-th sampled frame only (banners
            need ~1 fps, kill feeds the full OCR rate).
        min_text_h: Generic banners: only text lines at least this tall
            (reference px) count.
        det_scale: "det" gate: run text detection on the crop downscaled by
            this factor (big text survives 0.35x; feeds need ~0.6x).
        upscale: "mask" and "read" gates: OCR the crop enlarged to this factor of its
            1920x1080 reference size (small, thin digits read far better at 2.5x); 1:
            as decoded. Left out of the cache key at 1, so older keys stay.
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
    upscale: float = 1.0

    def key(self) -> dict[str, Any]:
        """Cache-key dict (also the region's preset JSON entry): every field, ``upscale``
        only when it is not 1."""
        d = asdict(self)
        if self.upscale == 1.0:
            del d["upscale"]
        return d


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
        phrases: Banner phrases: the preset's list plus :data:`GENERIC_BANNERS`.
        player_tags: Extra in-game names listed in the preset.
        feed: Feed parsing options (``kill_words``, ``colors``, ...).
        bigtext: Add the generic big-centred-text check.
        path: JSON file the preset came from.
        title: Human-readable game name (VLM prompt).
        view: Camera view for the VLM prompt ("first-person"; empty: none).
        hype: The VLM prompt's hype rubric line (empty: the generic one).
        events: Event types (raw ``events`` entries, see
            :mod:`cutpoint.events.eventtypes`).
        source: The preset's ``source`` ("discovered" for one made by discovery).
    """

    game: str
    regions: tuple[Region, ...]
    phrases: tuple[str, ...]
    player_tags: tuple[str, ...] = ()
    feed: dict[str, Any] = field(default_factory=dict)
    bigtext: bool = True
    path: Path | None = None
    title: str = ""
    view: str = ""
    hype: str = ""
    events: tuple[dict[str, Any], ...] = ()
    source: str = ""

    def all_regions(self) -> tuple[Region, ...]:
        """Preset regions plus the generic big-text region when enabled."""
        if self.bigtext and not any(r.kind == "bigtext" for r in self.regions):
            return (*self.regions, bigtext_region())
        return self.regions

    def key(self) -> dict[str, Any]:
        """Extra cache-key fields of the preset."""
        return {
            "phrases": list(self.phrases),
            "feed": self.feed,
            "bigtext": self.bigtext,
            "preset_tags": list(self.player_tags),
            "source": self.source,
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


def is_installed_path(path: Path | None) -> bool:
    """Whether a preset at ``path`` is installed: bundled (``path`` None, or in
    :data:`~cutpoint.paths.PRESETS_DIR`) or a user preset (in
    :func:`~cutpoint.paths.presets_dir`), not a project one."""
    return path is None or path.parent in (paths.PRESETS_DIR, paths.presets_dir())


def preset_from_dict(data: dict[str, Any], path: Path | None = None) -> HudPreset:
    """Build a :class:`HudPreset` from preset JSON (unknown keys are ignored)."""
    phrases = tuple(map(str, data.get("banner_phrases", [])))
    events = tuple(e for e in data.get("events") or [] if isinstance(e, dict))
    phrases = tuple(dict.fromkeys([*phrases, *GENERIC_BANNERS, *_event_phrases(events)]))
    return HudPreset(
        game=str(data.get("game") or (path.stem if path else "game")),
        regions=_regions(data),
        phrases=phrases,
        player_tags=tuple(data.get("player_tags", [])),
        feed=dict(data.get("feed") or {}),
        bigtext=bool(data.get("bigtext", True)),
        path=path,
        title=str(data.get("title", "")),
        view=str(data.get("view", "")),
        hype=str(data.get("hype", "")),
        events=events,
        source=str(data.get("source", "")),
    )


def as_list(v: Any) -> list[Any]:
    """A preset value that may be one item or a list, as a list (falsy: empty)."""
    return v if isinstance(v, list) else [v] if v else []


def _event_phrases(events: tuple[dict[str, Any], ...]) -> list[str]:
    """Banner ``prefix`` phrases of event types, so OCR looks for them too."""
    out = []
    for e in events:
        for m in e.get("when") or []:
            if isinstance(m, dict) and m.get("type") == "banner":
                p = m.get("prefix")
                out += [str(x).upper() for x in as_list(p)]
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

# folder names that say nothing about the game: footage of many games lands in them, so
# their name would give unrelated games one preset (#822)
GENERIC_FOLDERS = frozenset(
    {
        "videos",
        "vods",
        "captures",
        "recordings",
        "clips",
        "gameplay",
        "footage",
        "desktop",
        "downloads",
        "src",
        "replay",
    }
)
_DIGITS_RE = re.compile(r"[0-9-]+")


def _is_generic(slug: str) -> bool:
    """Whether a slug names no game: one of :data:`GENERIC_FOLDERS`, "game" (also the slug
    of a name without letters or digits, such as "🎮" or "___"), at most two characters
    ("a", "b2") or only digits and dashes (a date or a number)."""
    return (
        slug in GENERIC_FOLDERS
        or slug == "game"
        or len(slug) <= 2
        or bool(_DIGITS_RE.fullmatch(slug))
    )


def game_key_for_path(path: str, override: str | None = None, project: Path | None = None) -> str:
    """Like :func:`preset_for_path` but never None.

    Order: ``override`` and the presets' ``match`` hints (:func:`preset_for_path`), then
    the decision the HUD fit check of an earlier run recorded for this file in
    ``<project>/discovery/game_keys.json`` (:func:`decided_entry`; only with ``project``),
    then the first guess (:func:`first_guess`). Lookup only: the fit check runs in
    :func:`cutpoint.events.pipeline._discover`.
    """
    known = preset_for_path(path, override, project)
    if known:
        return known
    guess = first_guess(path, project)
    entry = decided_entry(path, project, guess) if project is not None else None
    return str(entry["key"]) if entry else guess


def first_guess(path: str, project: Path | None = None) -> str:
    """The game key a file's name and folder suggest.

    A slug of the game name, taken from a recorder-style file name
    ("Some Game 2024.11.14 - 19.01.53.09.DVR.mp4" -> "some-game") unless that name is
    generic (see :func:`_is_generic`; "Desktop 2024.11.14 ... .DVR.mp4" of a desktop
    capture), else the parent folder's name unless it is generic, else the file name
    without extension. In a generic folder, files named by a date or number only (OBS:
    "2026-11-08 19-01-53.mkv"), or by a generic recorder name and a date ("Replay
    2026-11-08 19-01-53.mkv", "Desktop 2024.11.14 ... .DVR.mp4"), share the key of the first such video in the folder, by
    name (:func:`date_group`), so one game recorded there gets one discovery; the
    fit check splits other games off. A key that :func:`_legacy_slug` spells differently
    (a non-ASCII name) and that has no preset yet while the legacy key has one gives the
    legacy key, so presets written before keep matching; not the legacy "game", which
    every name without ASCII letters or digits had.
    """
    p = Path(path)
    stem = _DATE_RE.sub("", p.stem)
    if stem != p.stem and not _is_generic(_game_slug(stem)):
        name = stem
    elif p.parent.name and not _is_generic(_game_slug(p.parent.name)):
        name = p.parent.name
    else:
        name = date_group(path, project)[1] or p.stem
    key, legacy = _game_slug(name), _legacy_slug(name)
    if (
        key != legacy
        and legacy != "game"
        and load_preset(key, project) is None
        and load_preset(legacy, project) is not None
    ):
        return legacy
    return key


def _date_only(stem: str) -> bool:
    """Whether a file name says only when it was recorded: digits and separators, or a
    recorder-style name whose game part is generic ("Replay 2026-11-08 19-01-53",
    "Desktop 2024.11.14 - 19.01.53.09.DVR")."""
    if _DIGITS_RE.fullmatch(_game_slug(stem)):
        return True
    game = _DATE_RE.sub("", stem)
    return game != stem and _is_generic(_game_slug(game))


_listings: dict[tuple[str, int], list[str]] = {}  # (folder, mtime_ns) -> sorted date stems


def _date_stems(folder: Path) -> list[str]:
    """The names (without extension) of the date-named videos in ``folder``, sorted; cached
    per folder and modification time. Raises OSError when it cannot be listed."""
    from cutpoint.utils import VIDEO_EXTS  # noqa: PLC0415

    key = (str(folder), folder.stat().st_mtime_ns)
    if key not in _listings:
        _listings[key] = sorted(
            q.stem
            for q in folder.iterdir()
            if q.suffix.lower() in VIDEO_EXTS and _date_only(q.stem) and q.is_file()
        )
    return _listings[key]


def date_group(path: str, project: Path | None = None) -> tuple[str, str | None]:
    """For a file named by a date or number only, or by a generic recorder name and a date
    (:func:`_date_only`), in a named folder: the folder's
    :func:`source_id` and the name (without extension) of the first such video in it,
    sorted by name. With ``project``, the name recorded in ``game_keys.json`` for the folder
    wins (deleting the oldest recording, or a folder offline, changes no key); otherwise the
    folder is listed, or the folder of its ``source_remap`` copy when it cannot be. The name
    is None for other files and when no folder can be listed."""
    p = Path(path)
    folder = source_id(p.parent)
    if not p.parent.name or not _date_only(p.stem):
        return folder, None
    if project is not None:
        stored = load_game_groups(project).get(folder)
        if isinstance(stored, str) and stored:
            return folder, stored
    try:
        stems = _date_stems(p.parent)
    except OSError:
        from cutpoint.config import remap_source  # noqa: PLC0415

        try:
            stems = _date_stems(remap_source(path).parent)
        except OSError:
            return folder, None
    return folder, stems[0] if stems else None


# --------------------------------------------------------------------------- fit decisions
# The HUD fit check (cutpoint.events.game_fit) records per source which key it decided in
# <project>/discovery/game_keys.json, so later runs and every reader of the key agree:
# {"version": 1, "sources": {"<path>": {"tag", "guess", "key", "fit"}}}
GAME_KEYS_VERSION = 1


def game_keys_path(project: Path) -> Path:
    """The fit check's decisions of ``project``."""
    return project / "discovery" / "game_keys.json"


def source_id(path: str | Path) -> str:
    """A source's entry name in ``game_keys.json``: its absolute path, case-folded where
    the file system ignores case."""
    return os.path.normcase(Path(path).absolute())


def source_tag(path: str | Path, io_path: Path | None = None) -> str:
    """Short digest of a source's path and the size and mtime of the file read for it
    (``io_path``, by default its ``source_remap`` copy or the path itself), like the key of
    :func:`cutpoint.cache.source_dir`."""
    key = str(Path(path))
    if io_path is None:
        from cutpoint.config import remap_source  # noqa: PLC0415

        io_path = remap_source(path)
    try:
        st = io_path.stat()
        key += f"|{st.st_size}|{st.st_mtime_ns}"
    except OSError:  # gone: the path alone still tells files apart
        pass
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def _game_keys_part(project: Path, part: str) -> dict[str, Any]:
    data = cache.load_json(game_keys_path(project))
    if not isinstance(data, dict) or data.get("version") != GAME_KEYS_VERSION:
        return {}
    got = data.get(part)
    return {str(k): v for k, v in got.items()} if isinstance(got, dict) else {}


def load_game_keys(project: Path) -> dict[str, dict[str, Any]]:
    """The recorded decisions per :func:`source_id`; a missing or unreadable file, or one
    of another version, gives none."""
    return {k: v for k, v in _game_keys_part(project, "sources").items() if isinstance(v, dict)}


def load_game_groups(project: Path) -> dict[str, str]:
    """The recorded first file name per folder of date-named videos (:func:`date_group`)."""
    return {k: v for k, v in _game_keys_part(project, "groups").items() if isinstance(v, str)}


def save_game_keys(
    project: Path, entries: dict[str, dict[str, Any]], groups: dict[str, str] | None = None
) -> None:
    """Write the decisions and the date groups (atomically); ``groups`` None keeps the
    recorded ones."""
    groups = load_game_groups(project) if groups is None else groups
    game_keys_path(project).parent.mkdir(parents=True, exist_ok=True)
    cache.save_json(
        game_keys_path(project),
        {"version": GAME_KEYS_VERSION, "sources": entries, "groups": groups},
    )


def decided_entry(path: str, project: Path, guess: str | None = None) -> dict[str, Any] | None:
    """The recorded decision for ``path`` while it still holds: same file (its
    :func:`source_tag`) and same first guess (``guess``, by default :func:`first_guess`).
    None when there is none or it is stale."""
    entry = load_game_keys(project).get(source_id(path))
    if not entry or not isinstance(entry.get("key"), str) or not entry["key"]:
        return None
    if entry.get("guess") != (guess if guess is not None else first_guess(path, project)):
        return None
    return entry if entry.get("tag") == source_tag(path) else None


def misfit_key(path: str, taken: Callable[[str], bool]) -> str:
    """The key for a source that does not fit its first guess's HUD preset: a slug of its
    file name (the recorder-style game name when there is one), or
    ``<folder key>-<6 hex digits of its path>`` when that name is generic or ``taken``
    (the key of another group or of a preset)."""
    p = Path(path)
    stem = _DATE_RE.sub("", p.stem)
    name = stem if stem != p.stem and not _is_generic(_game_slug(stem)) else p.stem
    key = _game_slug(name)
    if _is_generic(key) or taken(key):
        digest = hashlib.sha1(source_id(path).encode("utf-8")).hexdigest()[:6]
        key = f"{_game_slug(p.parent.name)}-{digest}"
    return key


def _game_slug(name: str) -> str:
    """Preset key for a game name ("Counter-strike 2" -> "counter-strike-2",
    "vidéo ünï 日本" -> "vidéo-ünï-日本").

    The key is persisted (preset file stem and OCR cache key). The name is NFC-normalised
    and lower-cased; each character that is not a letter or digit (``str.isalnum``) or a
    combining mark (Unicode category M, such as the vowel signs of "हिन्दी") becomes "-",
    runs of "-" collapse to one and leading or trailing ones go; an empty result is
    "game". For ASCII names this is exactly :func:`_legacy_slug`. Non-ASCII letters,
    digits and marks stay as they are, without folding: "²" and "Ⅻ" stay (NFKC would
    make them "2" and "xii", but also turn "™" into "tm").
    """
    text = unicodedata.normalize("NFC", name).lower()
    slug = "".join(
        c if c.isalnum() or unicodedata.category(c).startswith("M") else "-" for c in text
    )
    return re.sub(r"-+", "-", slug).strip("-") or "game"


def _legacy_slug(name: str) -> str:
    """The key :func:`_game_slug` gave before non-ASCII letters were kept: lower-case
    ASCII letters and digits, every other run of characters "-" ("Café Racer" ->
    "caf-racer"). Only for finding presets written under it."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "game"


def scaled(region: Region, width: int, height: int) -> tuple[int, int, int, int]:
    """Region rectangle scaled to a ``width`` x ``height`` frame (even values)."""
    sx, sy = width / 1920, height / 1080
    x, y = int(region.x * sx) // 2 * 2, int(region.y * sy) // 2 * 2
    w = min(int(region.w * sx) // 2 * 2, width - x)
    h = min(int(region.h * sy) // 2 * 2, height - y)
    return x, y, w, h
