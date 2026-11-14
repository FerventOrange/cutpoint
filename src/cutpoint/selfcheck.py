"""Self-checks of a finished analysis (#1073): cheap sanity checks over ``highlights.json`` and
``events.json`` that warn when a detection silently failed (a kill feed that reads almost
nothing, kills never credited to the player, one signal or label behind most of the moments).

The checks only diagnose: they never change a score, a candidate or the ranking. A check
whose inputs an older file lacks is skipped, so an old project never fails here.

Each finding is ``{"code", "message", "hint", "files"}``. :func:`apply` records them in
``highlights.json``'s ``settings.self_checks``, logs one WARNING per finding and adds each to the
``--json`` envelope's ``warnings`` (the message followed by the hint).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import PureWindowsPath
from typing import Any

from cutpoint import jsonout
from cutpoint.utils import slugify

logger = logging.getLogger(__name__)

CODES = ("feed_quiet", "feed_missing", "kills_uncredited", "signal_dominates", "label_dominates")

# ---- feed_quiet / feed_missing -------------------------------------------------------------
# A kill feed shows a distinct line for about every kill and death. Measured on this project's
# real footage (distinct lines per hour of the whole footage): a feed that works reads 150 lines/h
# or more (Ready or Not 152, Rainbow Six Siege 441-574 once its region was right, War Thunder
# 760-2300); the bogus box that read 67 lines in 3 h of Siege was 22 lines/h. The lines/h are
# counted over the gameplay hours (footage without the menu frames, which show no feed), so a
# long menu does not hide a quiet feed: that stream was 31 lines/h of gameplay.
FEED_QUIET_LINES_PER_HOUR = 100  # under the 150 of the lowest working feed, with margin
# Less gameplay than this is too short for a rate to mean anything (a few kills at most).
FEED_MIN_GAMEPLAY_H = 0.5

# ---- kills_uncredited ------------------------------------------------------------------------
# The feed read this many kills, or the vision model's descriptions of the moments mention a kill
# this many times, so a player who set ``player_tags`` should have had some credited. Fewer is
# plausible for a session without fights (a 3 h stream gave 338 descriptions, a few dozen of kills).
KILLS_MIN_READ = 5
KILLS_MIN_VLM_MENTIONS = 10
KILL_WORDS = re.compile(r"\b(?:kill(?:s|ed|ing)?|eliminat\w*|headshots?|takedowns?)\b", re.I)
KILL_TYPES = ("kill", "multikill")  # OCR event types of the feed

# ---- signal_dominates ------------------------------------------------------------------------
# The moments checked: the ``suggested_top`` best (what the default edit takes), at least
# SIGNAL_MIN_MOMENTS (fewer make "most" meaningless), else SIGNAL_DEFAULT_MOMENTS.
SIGNAL_DEFAULT_MOMENTS = 20
SIGNAL_MIN_MOMENTS = 10
# One OCR text or kind of event evidence behind more than this share of those moments
# is the ranking's main driver, not a spread of different reasons; and in at least this many.
# Keyword phrases are no reason here: a phrase said often ("oh my god") is what a talker says,
# not a failed detection.
SIGNAL_SHARE = 0.5
SIGNAL_MIN_COUNT = 4
OCR_TEXT_MIN_CHARS = 3  # shorter OCR texts are noise
# ---- label_dominates -------------------------------------------------------------------------
# One label named by a detection (an OCR or VLM one: ``banners``, ``explosions``, a VLM title) on
# more than this share of the candidates: that detection is probably firing on everything.
LABEL_SHARE = 0.15
LABEL_MIN_CANDIDATES = 40  # fewer candidates: a share says little (a short clip has 20 to 30)
# Labels that are no detection: what analysis names a moment when nothing does (``<style>-moment``,
# ``loud-moment``, ``vlm``) and the sounds of a reaction. A keyword phrase (``oh-my-god``) is
# skipped per candidate, from its own keywords.
GENERIC_LABEL = re.compile(r"(?:[\w-]+-moment|vlm)?")
REACTION_LABELS = frozenset(
    {"laughter", "laugh", "laughing", "cheering", "cheer", "scream", "screaming", "shout", "yell"}
)


def _finding(code: str, message: str, hint: str, files: Iterable[str] = ()) -> dict[str, Any]:
    return {"code": code, "message": message, "hint": hint, "files": list(files)}


def _name(path: str) -> str:
    """A source's file name, from a Windows or a POSIX path (events.json keys stats by it)."""
    return PureWindowsPath(path).name


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _gaming(hl: dict[str, Any]) -> bool:
    return (hl.get("style_base") or hl.get("style")) == "gaming"


def _tags(ev: dict[str, Any]) -> list[str]:
    tags = _dict(_dict(ev.get("settings")).get("ocr")).get("tags")
    return [t for t in tags if isinstance(t, str)] if isinstance(tags, list) else []


def _ocr_stats(ev: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        n: s for n, s in _dict(_dict(ev.get("stats")).get("ocr")).items() if isinstance(s, dict)
    }


def _source_paths(hl: dict[str, Any], ev: dict[str, Any], names: Iterable[str]) -> list[str]:
    """The paths of the sources (of highlights.json, else events.json) with these file names."""
    want = set(names)
    paths = [str(s.get("path")) for s in hl.get("sources") or [] if isinstance(s, dict)]
    paths = paths or [s for s in ev.get("sources") or [] if isinstance(s, str)]
    return [p for p in paths if _name(p) in want]


def _gameplay_hours(stat: dict[str, Any], region: dict[str, Any]) -> float | None:
    hours = region.get("hours")
    if not isinstance(hours, int | float) or isinstance(hours, bool):
        return None
    frames, menu = stat.get("frames"), stat.get("menu_frames")
    if isinstance(frames, int) and frames > 0 and isinstance(menu, int) and 0 <= menu <= frames:
        hours *= 1 - menu / frames
    return float(hours)


def feed_quiet(hl: dict[str, Any], ev: dict[str, Any]) -> list[dict[str, Any]]:
    """A gameplay feed region that reads far fewer lines per gameplay hour than a working one."""
    if not _gaming(hl):
        return []
    found: list[str] = []
    layers = _dict(ev.get("settings")).get("layers")
    told = _vlm_kills(ev) if isinstance(layers, list) and "vlm" in layers else None
    for name, stat in _ocr_stats(ev).items():
        if told is not None and told.get(name, 0) < KILLS_MIN_VLM_MENTIONS:
            continue  # the vision model saw no kills here: a game without a kill feed
        for region in _dict(stat.get("feed_regions")).values():
            lines = _dict(region).get("lines")
            gameplay = _gameplay_hours(stat, region)
            if (
                _dict(region).get("label") != "feed"
                or not isinstance(lines, int)
                or gameplay is None
                or gameplay < FEED_MIN_GAMEPLAY_H
            ):
                continue
            rate = lines / gameplay
            if rate < FEED_QUIET_LINES_PER_HOUR:
                found.append(f"{name}: {lines} lines in {gameplay:.1f} h of gameplay")
                logger.debug("%s: feed %.0f lines/h of gameplay", name, rate)
    if not found:
        return []
    names = [f.split(":")[0] for f in found]
    return [
        _finding(
            "feed_quiet",
            "The kill feed reads almost nothing ("
            + "; ".join(found)
            + f"; a working one reads over {FEED_QUIET_LINES_PER_HOUR} lines per hour), so kills "
            "and deaths are probably missed",
            "Check the feed region the analysis discovered (its box may be wrong), or ignore "
            "this if the game has no kill feed",
            _source_paths(hl, ev, names),
        )
    ]


def _vlm_kills(ev: dict[str, Any]) -> Counter[str]:
    """The VLM descriptions that mention a kill, per source file name."""
    return Counter(
        _name(str(e.get("source")))
        for e in ev.get("events") or []
        if isinstance(e, dict)
        and e.get("kind") == "vlm"
        and KILL_WORDS.search(str(e.get("description") or ""))
    )


def feed_missing(hl: dict[str, Any], ev: dict[str, Any]) -> list[dict[str, Any]]:
    """The player set ``player_tags`` and the vision model describes kills in a source, but the
    HUD read found no feed region in it (``player_tags`` is a global setting: a game without
    kills stays quiet)."""
    if not _gaming(hl) or not _tags(ev):
        return []
    told = _vlm_kills(ev)
    names = []
    for name, stat in _ocr_stats(ev).items():
        regions = stat.get("feed_regions")
        frames, menu = stat.get("frames"), stat.get("menu_frames") or 0
        if (
            isinstance(regions, dict)  # an older file has none: skipped
            and not regions
            and isinstance(frames, int)
            and frames > 0
            and isinstance(menu, int)
            and menu < frames
            and told.get(name, 0) >= KILLS_MIN_VLM_MENTIONS
        ):
            names.append(name)
    if not names:
        return []
    return [
        _finding(
            "feed_missing",
            "No kill feed region was found in " + ", ".join(names) + ", so no kills are read "
            "although player_tags is set",
            "Move or rename the game's HUD preset (`<presets>/<game>.json` or the project's "
            "`hud_presets/<game>.json`) and run `cutpoint events <project>` again so discovery "
            "makes a new one, or copy the `feed` region from the `proposal.json` of `events "
            "--discover-regions` into the preset; ignore this if the game has no kill feed",
            _source_paths(hl, ev, names),
        )
    ]


def _flagged(e: dict[str, Any], flag: str) -> bool:
    return bool(e.get(flag) or _dict(e.get("details")).get(flag))


def kills_uncredited(hl: dict[str, Any], ev: dict[str, Any]) -> list[dict[str, Any]]:
    """``player_tags`` is set and kills were seen (read by the feed, or told by the vision
    model), but none is credited to the player anywhere in the project."""
    tags = _tags(ev)
    if not _gaming(hl) or not tags:
        return []
    events = [e for e in ev.get("events") or [] if isinstance(e, dict)]
    read = [e for e in events if e.get("kind") == "ocr" and e.get("type") in KILL_TYPES]
    credited = sum(_flagged(e, "player_kill") for e in read)
    for c in hl.get("candidates") or []:
        n = _dict(_dict(c.get("signals")).get("events")).get("player_kills")
        credited += n if isinstance(n, int) and not isinstance(n, bool) else 0
    if credited:
        return []
    told = sum(_vlm_kills(ev).values())
    if len(read) < KILLS_MIN_READ and told < KILLS_MIN_VLM_MENTIONS:
        return []
    seen = (
        f"the feed read {len(read)} kill(s)"
        if len(read) >= KILLS_MIN_READ
        else f"the vision model's descriptions mention {told} kill(s)"
    )
    return [
        _finding(
            "kills_uncredited",
            f"No kill is credited to {', '.join(tags)} although {seen}",
            "Check player_tags in config.json against the name in the kill feed, and that the "
            "kill feed region reads the lines (see feed_quiet)",
            _source_paths(hl, ev, [_name(str(s)) for s in ev.get("sources") or []]),
        )
    ]


def top_moments(hl: dict[str, Any]) -> list[dict[str, Any]]:
    """The best ``suggested_top`` candidates (at least :data:`SIGNAL_MIN_MOMENTS`, else
    :data:`SIGNAL_DEFAULT_MOMENTS`), in rank order."""
    cands = [c for c in hl.get("candidates") or [] if isinstance(c, dict)]
    top = hl.get("suggested_top")
    ok = isinstance(top, int) and not isinstance(top, bool) and top > 0
    return cands[: max(top, SIGNAL_MIN_MOMENTS) if ok else SIGNAL_DEFAULT_MOMENTS]


def _reasons(c: dict[str, Any], ocr: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """What a moment was ranked for: the evidence kinds of its events
    (counted above zero, plus the nominating ones) and the OCR texts near it."""
    from cutpoint.events.scoring import EVIDENCE_KEYS, LOOKAHEAD_S  # noqa: PLC0415

    out: set[tuple[str, str]] = set()
    ev = _dict(_dict(c.get("signals")).get("events"))
    for k in EVIDENCE_KEYS:
        v = ev.get(k)
        if isinstance(v, int | float) and not isinstance(v, bool) and v > 0:
            out.add(("event", k))
    for k in ev.get("nominated") or []:
        if isinstance(k, str):
            out.add(("event", k))
    try:
        start, end = float(c["start"]), float(c["end"]) + LOOKAHEAD_S
    except (KeyError, TypeError, ValueError):
        return out
    for e in ocr:
        text = " ".join(str(e.get("label") or "").casefold().split())
        if (
            e.get("source") == c.get("source")
            and len(text) >= OCR_TEXT_MIN_CHARS
            and isinstance(e.get("start"), int | float)
            and start <= e["start"] <= end
        ):
            out.add(("OCR text", text))
    return out


def signal_dominates(hl: dict[str, Any], ev: dict[str, Any] | None) -> list[dict[str, Any]]:
    """One OCR text or kind of event evidence behind most of the top moments."""
    top = top_moments(hl)
    if len(top) < SIGNAL_MIN_MOMENTS:
        return []
    ocr = [
        e for e in _dict(ev).get("events") or [] if isinstance(e, dict) and e.get("kind") == "ocr"
    ]
    counts: Counter[tuple[str, str]] = Counter()
    for c in top:
        counts.update(_reasons(c, ocr))
    need = max(SIGNAL_MIN_COUNT, int(SIGNAL_SHARE * len(top)) + 1)  # more than the share
    over = [(kind, key, n) for (kind, key), n in counts.most_common() if n >= need]
    if not over:
        return []
    what = "; ".join(f"{kind} {key!r} in {n}" for kind, key, n in over[:5])
    return [
        _finding(
            "signal_dominates",
            f"One signal is behind most of the top {len(top)} moments ({what})",
            "Check it is real and not a misread (a caption read as an announcement, a common "
            "word as a keyword); the ranking was not changed",
        )
    ]


def _detected_label(c: dict[str, Any]) -> bool:
    """The label of ``c`` is named by an OCR or VLM detection: not a fallback, a reaction or one
    of the candidate's own keyword phrases."""
    label = c.get("label")
    if not isinstance(label, str) or not label:
        return False
    if GENERIC_LABEL.fullmatch(label) or label in REACTION_LABELS:
        return False
    words = _dict(c.get("signals")).get("keywords") or []
    return label not in {slugify(k) for k in words if isinstance(k, str)}


def label_dominates(hl: dict[str, Any]) -> list[dict[str, Any]]:
    """One detection-named label (:func:`_detected_label`) on more than :data:`LABEL_SHARE` of the
    candidates."""
    if (hl.get("style_base") or hl.get("style")) == "action-cam":
        return []  # its labels name telemetry (``fast``, ``carve``), not a detection
    cands = [c for c in hl.get("candidates") or [] if isinstance(c, dict)]
    if len(cands) < LABEL_MIN_CANDIDATES:
        return []
    counts = Counter(c["label"] for c in cands if _detected_label(c))
    over = [
        (label, n)
        for label, n in counts.most_common()
        if n > LABEL_SHARE * len(cands) and not GENERIC_LABEL.fullmatch(label)
    ]
    if not over:
        return []
    what = ", ".join(f"{label!r} on {n} of {len(cands)}" for label, n in over)
    return [
        _finding(
            "label_dominates",
            f"One label names a large share of the moments ({what})",
            "That detection may be firing on everything; check what it reads in events.json",
        )
    ]


def run(hl: dict[str, Any], ev: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Every check over the analysis ``hl`` and its events.json object ``ev`` (None: without
    events); the findings, in a fixed order. Reads only; never raises on a malformed file."""
    ev = ev if isinstance(ev, dict) else {}
    found: list[dict[str, Any]] = []
    checks = (
        lambda: feed_quiet(hl, ev),
        lambda: feed_missing(hl, ev),
        lambda: kills_uncredited(hl, ev),
        lambda: signal_dominates(hl, ev),
        lambda: label_dominates(hl),
    )
    for check in checks:
        try:
            found += check()
        except (AttributeError, KeyError, TypeError, ValueError):  # a file of an odd shape
            logger.debug("self-check skipped", exc_info=True)
    return found


def apply(hl: dict[str, Any], ev: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Run :func:`run` and report it: ``hl["settings"]["self_checks"]`` (in place; the caller
    saves it), one WARNING per finding and a ``--json`` warning each. Changes nothing else."""
    found = run(hl, ev)
    jsonout.drop_warnings(CODES)  # edit: the events step's findings replace the analysis's
    settings = hl.setdefault("settings", {})
    if found:
        settings["self_checks"] = found
    else:  # a finding of an earlier run is stale
        settings.pop("self_checks", None)
    for f in found:
        text = f"{f['message']}. {f['hint']}"
        logger.warning("%s", text)
        jsonout.warn(f["code"], text, f["files"])
    return found
