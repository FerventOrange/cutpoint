"""Feed events.json back into highlights.json candidate scores.

Each candidate keeps its analysis score as ``score_pre_events``; the new
``score`` adds event boosts for ``[start, end + lookahead]`` (so a payoff
right after the banter counts for the banter clip) and blends in the best
overlapping VLM window (boost-only: it can raise but never lower a score).
Event types with ``reaction`` points (by default the player's own deaths, in
the funny style only) add them once per candidate when a reaction follows one
of their events: laughter or a scream, or speech after a pause in the cached
transcript (see :func:`has_reaction`). Re-running is idempotent because the boost is always
computed from ``score_pre_events``. Candidate ids are kept; the list is
re-sorted by the new score and ``suggested_top`` is recomputed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cutpoint import cache, sources
from cutpoint.events.eventtypes import DEFAULT_EVENT_TYPES, EventType, first_match, resolve
from cutpoint.styles import get_style

logger = logging.getLogger(__name__)

LOOKAHEAD_S = 10.0  # events this long after a candidate still count for it
VLM_WEIGHT = 0.35  # share of the final score taken from the VLM window score
MAX_BOOST = 30.0
# Evidence keys of the default event types, always present and summed in this order.
EVIDENCE_KEYS = ("player_kills", "teamkills", "banners", "self_kills")
REACTION_WINDOW_S = 6.0  # a reaction starts at most this long after the event
REACTION_PAUSE_S = 1.5  # speech is a reaction only after this much silence up to the event
REACTION_SOUNDS = ("Laughter", "Scream")


def has_reaction(
    t: float, events: list[dict[str, Any]], segments: list[dict[str, Any]] | None = None
) -> bool:
    """Whether a reaction follows an event at ``t`` (seconds, source time).

    A reaction is a Laughter or Scream audio event starting in
    ``(t, t + REACTION_WINDOW_S]`` (speech or not), or a transcript segment
    starting in that window while none overlaps ``[t - REACTION_PAUSE_S, t]``:
    a pause, then speech, so ongoing banter does not count. Without
    ``segments`` only the sounds count.
    """
    end = t + REACTION_WINDOW_S
    for e in events:
        if e.get("kind") == "audio" and e.get("label") in REACTION_SOUNDS and t < e["start"] <= end:
            return True
    if not segments:
        return False
    if any(s["end"] >= t - REACTION_PAUSE_S and s["start"] <= t for s in segments):
        return False
    return any(t < s["start"] <= end for s in segments)


def event_boost(
    cand: dict[str, Any],
    events: list[dict[str, Any]],
    types: tuple[EventType, ...] = DEFAULT_EVENT_TYPES,
    *,
    style: str | tuple[str, ...] | None = None,
    segments: list[dict[str, Any]] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Additive boost (points on the 0-100 scale) and the evidence behind it.

    Each OCR event counts for the first of ``types`` it matches (see
    :mod:`cutpoint.events.eventtypes`), whatever the style; types without a
    ``score`` add nothing. A type's ``score`` and ``reaction`` count only when
    it applies to ``style`` (a name, or the style's name and base; a type
    limited to some styles never applies to ``style=None``). A type's ``reaction`` is added once if any of its events
    in the window is followed by a reaction (:func:`has_reaction`, with the
    source's transcript ``segments`` in source time); its evidence key is
    ``<name>_reaction`` (0 or 1), present only when the reaction applies.
    """
    a, b = cand["start"], cand["end"] + LOOKAHEAD_S
    laugh = scream = 0.0
    names = (style,) if isinstance(style, str) else style or ()
    scored = {t.name: t for t in types if t.score is not None and t.applies(*names)}
    reacting = {t.name: t for t in types if t.reaction is not None and t.applies(*names)}
    counts = dict.fromkeys([*EVIDENCE_KEYS, *scored], 0)
    reactions = dict.fromkeys(reacting, 0)
    for e in events:
        if e["end"] < a or e["start"] > b:
            continue
        if e["kind"] == "audio":
            if e["label"] == "Laughter":
                laugh = max(laugh, e["confidence"] or 0)
            elif e["label"] in ("Scream", "Cheering"):
                scream = max(scream, e["confidence"] or 0)
        elif e["kind"] == "ocr":
            t = first_match(types, e)
            if t and t.name in scored:
                counts[t.name] += 1
            if (
                t
                and t.name in reacting
                and not reactions[t.name]
                and has_reaction(e["start"], events, segments)
            ):
                reactions[t.name] = 1
    boost = min(12.0, 60 * laugh) + min(6.0, 40 * scream)
    for name, n in counts.items():  # defaults first, in the original order
        if name in scored:
            boost += scored[name].score * min(n, scored[name].max)
    for name, n in reactions.items():
        boost += reacting[name].reaction * n
    evidence = {"laughter": round(laugh, 3), "scream": round(scream, 3), **counts}
    evidence.update({f"{name}_reaction": n for name, n in reactions.items()})
    return min(MAX_BOOST, boost), evidence


def best_window(cand: dict[str, Any], windows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The VLM window overlapping the candidate the most (None if none)."""
    best, best_ov = None, 0.0
    for w in windows:
        if w["source"] != cand["source"]:
            continue
        ov = min(w["end"], cand["end"] + LOOKAHEAD_S / 2) - max(w["start"], cand["start"])
        if ov > best_ov:
            best, best_ov = w, ov
    return best if best_ov >= 2.0 else None


def _segments(source: str, language: str | None = None) -> list[dict[str, Any]]:
    """:func:`cutpoint.sources.source_segments` (of the transcript ``language``), or []
    when the source has none or cannot be read."""
    try:
        return sources.source_segments(Path(source), language) or []
    except OSError as err:  # the source is gone (moved, drive disconnected)
        logger.debug("No transcript for %s (%s)", source, err)
        return []


def style_and_base(name: str | None) -> tuple[str, ...]:
    """The project style's name and, for a learned style, its base (just the name
    when the style is unknown or its file is missing)."""
    if not name:
        return ()
    try:
        base = get_style(name).base
    except ValueError:
        return (name,)
    return (name,) if base == name else (name, base)


def rescore_highlights(project: Path) -> int:
    """Rescore ``<project>/highlights.json`` from ``events.json``.

    The project's ``style`` (its name or, for a learned style, its base) gates
    event types limited to some styles. For a source with an event type whose
    ``reaction`` applies, its cached transcript segments are loaded once
    (:func:`cutpoint.sources.source_segments`).

    Returns:
        Number of candidates whose score changed (0 if either file is missing).
    """
    hl = cache.load_json(project / "highlights.json")
    ev = cache.load_json(project / "events.json")
    if not hl or not ev:
        return 0
    backup = project / "highlights.pre-events.json"
    if not backup.exists() and not hl.get("settings", {}).get("events_rescored"):
        cache.save_json(backup, hl)  # one-time copy of the untouched analysis
    changed = rescore(hl, ev, project)
    cache.save_json(project / "highlights.json", hl)
    logger.info(
        "Rescored %d of %d candidates from events.json", changed, len(hl.get("candidates", []))
    )
    return changed


def rescore(hl: dict[str, Any], ev: dict[str, Any], project: Path) -> int:
    """:func:`rescore_highlights` in memory: rescore the analysis ``hl`` of ``project``
    from the events ``ev`` (an events.json object); returns the candidates changed."""
    by_src: dict[str, list[dict[str, Any]]] = {}
    for e in ev.get("events", []):
        by_src.setdefault(e["source"], []).append(e)
    windows = ev.get("windows", [])
    types = resolve(project, ev)
    style = style_and_base(hl.get("style"))
    language = (hl.get("settings") or {}).get("language")
    segments: dict[str, list[dict[str, Any]] | None] = {}
    changed = 0
    for c in hl.get("candidates", []):
        base = c.get("score_pre_events", c["score"])
        src = c["source"]
        src_types = types.get(src, DEFAULT_EVENT_TYPES)
        if src not in segments:
            reacts = any(t.reaction is not None and t.applies(*style) for t in src_types)
            segments[src] = _segments(src, language) if reacts else None
        boost, evidence = event_boost(
            c, by_src.get(src, []), src_types, style=style, segments=segments[src]
        )
        score = base + boost
        win = best_window(c, windows)
        if win:
            vlm = max(win.get("highlight_score", 0), win.get("funny_score", 0)) * 10
            # Boost-only: Omni-7B often scores real action 0, so it may raise a
            # candidate toward its window score but never pull it down.
            score += VLM_WEIGHT * max(0.0, vlm - score)
            evidence["vlm"] = {
                "highlight": win.get("highlight_score"),
                "funny": win.get("funny_score"),
                "label": win.get("label"),
                "window": [win["start"], win["end"]],
            }
        score = round(min(100.0, score), 1)
        c["score_pre_events"] = base
        c.setdefault("signals", {})["events"] = {"boost": round(boost, 1), **evidence}
        if score != c["score"]:
            changed += 1
        c["score"] = score
    hl["candidates"].sort(key=lambda c: -c["score"])
    target = (hl.get("settings") or {}).get("target_total")
    if target:
        total, n = 0.0, 0
        for c in hl["candidates"]:
            if total >= target:
                break
            total += c["end"] - c["start"]
            n += 1
        hl["suggested_top"] = n
    hl.setdefault("settings", {})["events_rescored"] = ev.get("created")
    return changed
