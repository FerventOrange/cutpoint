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

In styles with an open list (``gaming``), events that can score and that no candidate covers
nominate candidates of their own first (:func:`nominate`, marked ``signals.events.nominated``), so a game with little talk still gets
moments from what happens on screen; they are rescored like the others. A stretch of talk the
dialogue reader marked (:mod:`cutpoint.events.dialogue`, a ``text`` event of type ``dialogue``)
nominates a candidate of its own, as marked, and adds a few points to any candidate it overlaps
(:data:`DIALOGUE_POINTS`).

A whole dialogue stretch longer than :data:`DIALOGUE_WIDEN_MAX_S` (:func:`is_long_dialogue`)
has more room for events than a short moment, so its laughter and on-screen events count per
minute (see :func:`event_boost`). It also does not take the score of the best moment inside it: its analysis score is the :data:`NOMINEE_FLOOR` share of the source's best (see
:func:`nominate`).

This module collects the evidence; the points are combined by the scorer's bonus stage
(:func:`cutpoint.scorer.rescore_events`, the ``bonus`` features of :mod:`cutpoint.features`).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cutpoint import cache, scorer, sources
from cutpoint.events.eventtypes import DEFAULT_EVENT_TYPES, EventType, first_match, resolve
from cutpoint.events.menu_spans import drop_menu_time, source_menu_spans
from cutpoint.events.tails import OCR_HOLD_S
from cutpoint.styles import get_style
from cutpoint.utils import title_slug

if TYPE_CHECKING:
    from cutpoint.analyze import GridPart

logger = logging.getLogger(__name__)

LOOKAHEAD_S = 10.0  # events this long after a candidate still count for it
MAX_BOOST = 30.0
# Evidence keys of the default event types, always present and summed in this order.
EVIDENCE_KEYS = ("player_kills", "teamkills", "banners", "self_kills")
REACTION_WINDOW_S = 6.0  # a reaction starts at most this long after the event
REACTION_PAUSE_S = 1.5  # speech is a reaction only after this much silence up to the event
REACTION_SOUNDS = ("Laughter", "Scream")
# A nominee's analysis score is at least this share of its source's best. A whole dialogue
# stretch over DIALOGUE_WIDEN_MAX_S has exactly this score, whatever its grids say.
NOMINEE_FLOOR = 0.6
LAUGH_SOUNDS = ("Laughter",)  # audio events that score laughter points
# A heard laugh with a confidence below this (the detector's own absolute threshold,
# ``audio_events.DEFAULT_GROUPS``: weaker peaks pass only its relative rule, and at 0.04-0.1 they
# include speech and game sounds) makes no moment of its own and gives it no label unless something
# nearby backs it (:func:`laugh_backed`); a stronger one does as before.
LAUGH_STRONG = 0.12
BACKUP_S = 5.0  # the backup of a weak laugh is this close to it
LOUD_SPEECH = 0.6  # loudness (0-1 grid) that, with speech, backs a weak laugh
SCREAM_SOUNDS = ("Scream", "Cheering")  # audio events that score scream points
DIALOGUE_TYPE = "dialogue"  # ``type`` of the text events of the dialogue reader
# Points of a dialogue event at confidence 1. Goals and funny situations score more: a kill 6
# (a teamkill 12, a banner 6), laughter up to 12, a scream up to 6. A stretch of talk is
# the weakest evidence of the three, so at most half a kill, scaled by the reader's confidence;
# laughter inside it counts on its own through the usual laughter points.
DIALOGUE_POINTS = 3.0
# A candidate a dialogue stretch overlaps widens to cover it only when the whole result (the
# stretch and every candidate it hits) stays this long at most. A longer stretch is its own moment.
# It is also the length above which a whole dialogue stretch is scored at the floor and its
# laughter and on-screen events count per minute.
DIALOGUE_WIDEN_MAX_S = 90.0
# A widened candidate does not keep a silence this long (no transcript speech) next to its
# original moment: it starts after the one before it and ends before the one after it.
WIDEN_SILENCE_S = 8.0


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

    A whole dialogue stretch longer than :data:`DIALOGUE_WIDEN_MAX_S` (:func:`is_long_dialogue`,
    read from ``cand["signals"]["events"]["nominated"]``) collects events by length, so for it
    the laughter points are scaled by the rate of laughter events per minute over
    ``[start, end + LOOKAHEAD_S]``, from nothing up to 1 at
    :data:`~cutpoint.events.dialogue.DENSITY_FULL` a minute, and a type's ``score`` is
    ``score * min(max, n / minutes)`` (n events over the same minutes) instead of
    ``score * min(max, n)``. Reaction, scream and dialogue points and the evidence counts are
    the same for every length.
    """
    a, b = cand["start"], cand["end"] + LOOKAHEAD_S
    laugh = scream = dialogue = 0.0
    laughs = 0  # laughter events with a confidence in the window (long candidates)
    names = (style,) if isinstance(style, str) else style or ()
    talk: bool | None = None  # whether the style has an open list (asked at the first stretch)
    scored = {t.name: t for t in types if t.score is not None and t.applies(*names)}
    reacting = {t.name: t for t in types if t.reaction is not None and t.applies(*names)}
    counts = dict.fromkeys([*EVIDENCE_KEYS, *scored], 0)
    reactions = dict.fromkeys(reacting, 0)
    for e in events:
        if e["end"] < a or e["start"] > b:
            continue
        if e["kind"] == "audio":
            if e["label"] in LAUGH_SOUNDS:
                laugh = max(laugh, e["confidence"] or 0)
                laughs += bool(e["confidence"])
            elif e["label"] in SCREAM_SOUNDS:
                scream = max(scream, e["confidence"] or 0)
        elif e["kind"] == "text":
            if e.get("type") == DIALOGUE_TYPE:
                if talk is None:  # stretches left in events.json by an earlier style do not count
                    talk = _open_list(names)
                if talk:
                    dialogue = max(dialogue, e["confidence"] or 0)
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
    laugh_points = min(12.0, 60 * laugh)
    minutes = (b - a) / 60
    long = is_long_dialogue(cand)
    if long:
        from cutpoint.events.dialogue import DENSITY_FULL  # noqa: PLC0415 - imports this module

        laugh_points *= min(1.0, laughs / minutes / DENSITY_FULL)
    boost = laugh_points + min(6.0, 40 * scream) + DIALOGUE_POINTS * dialogue
    for name, n in counts.items():  # defaults first, in the original order
        if name in scored:
            boost += scored[name].score * min(scored[name].max, n / minutes if long else n)
    for name, n in reactions.items():
        boost += reacting[name].reaction * n
    evidence = {"laughter": round(laugh, 3), "scream": round(scream, 3), **counts}
    if dialogue:  # only where the dialogue reader ran
        evidence["dialogue"] = round(dialogue, 3)
    evidence.update({f"{name}_reaction": n for name, n in reactions.items()})
    return min(MAX_BOOST, boost), evidence


def is_long(c: dict[str, Any]) -> bool:
    """Whether the candidate ``c`` is longer than :data:`DIALOGUE_WIDEN_MAX_S`."""
    return c["end"] - c["start"] > DIALOGUE_WIDEN_MAX_S


def is_long_dialogue(c: dict[str, Any]) -> bool:
    """Whether ``c`` is a whole dialogue stretch (nominated by one) longer than
    :data:`DIALOGUE_WIDEN_MAX_S`."""
    nominated = ((c.get("signals") or {}).get("events") or {}).get("nominated") or []
    return is_long(c) and any(str(n).startswith("dialogue-") for n in nominated)


def _open_list(names: tuple[str, ...]) -> bool:
    """Whether a project whose style has one of ``names`` (its name and base) has an open list."""
    for name in names:
        try:
            if get_style(name).open_list:
                return True
        except ValueError:
            continue
    return False


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


@dataclass(frozen=True)
class Seed:
    """An event that can score, as the start of a candidate (see :func:`seeds`).

    Attributes:
        name: The event type's name, or the sound's (``laughter``, ``cheering``).
        start: Event start (source seconds); ``end``: its end.
        hold: How long it counts as lasting for the clip: a sound its length, an on-screen
            event at most :data:`~cutpoint.events.tails.OCR_HOLD_S`.
        tail_s: The event type's reaction kept after it (None: the style's ``post_roll``).
        whole: A dialogue stretch: its candidate keeps the span as marked (``hold`` long, no
            ``max_len`` cap; see :func:`cutpoint.analyze.nominate_candidates`).
        over: A ``whole`` stretch too long to widen the candidates it overlaps
            (:data:`DIALOGUE_WIDEN_MAX_S`): it stays whole over them (set by :func:`nominate`).
    """

    name: str
    start: float
    end: float
    hold: float
    tail_s: float | None = None
    whole: bool = False
    over: bool = False


def seeds(
    events: list[dict[str, Any]],
    types: tuple[EventType, ...],
    style: tuple[str, ...],
    menu: list[list[float]] | None = None,
    backup: Callable[[dict[str, Any]], bool] | None = None,
) -> list[Seed]:
    """The events of one source that can score in a project of ``style`` (its name and base):
    those of an event type with a ``score`` that applies, and the laughter, scream and
    cheering sounds :func:`event_boost` counts, and the dialogue stretches (``whole`` seeds).
    Events starting in a ``menu`` span are left out, and so is a weak heard laugh nothing backs
    (:func:`laugh_backed`, with ``backup``)."""
    out = []
    for e in events:
        if any(a <= e["start"] < b for a, b in menu or ()):
            continue
        length = max(0.0, e["end"] - e["start"])
        if e["kind"] == "audio":
            if (e["label"] in LAUGH_SOUNDS or e["label"] in SCREAM_SOUNDS) and e["confidence"]:
                if e["label"] in LAUGH_SOUNDS and not laugh_backed(e, events, backup):
                    continue
                out.append(Seed(e["label"].lower(), e["start"], e["end"], length))
        elif e["kind"] == "ocr":
            t = first_match(types, e)
            if t and t.score and t.applies(*style):
                out.append(Seed(t.name, e["start"], e["end"], min(length, OCR_HOLD_S), t.tail_s))
        elif e["kind"] == "text" and e.get("type") == DIALOGUE_TYPE:
            # the whole stretch as the reader marked it, with no reaction kept after it
            out.append(Seed(f"dialogue-{e['label']}", e["start"], e["end"], length, 0.0, True))
    return out


def sound_backed(e: dict[str, Any], events: list[dict[str, Any]]) -> bool:
    """Whether another laughter, scream or cheering sound (with a confidence) of ``events`` is
    within :data:`BACKUP_S` of the audio event ``e``."""
    a, b = e["start"] - BACKUP_S, e["end"] + BACKUP_S
    return any(
        o is not e
        and o["kind"] == "audio"
        and o["label"] in (*LAUGH_SOUNDS, *SCREAM_SOUNDS)
        and o["confidence"]
        and o["end"] >= a
        and o["start"] <= b
        for o in events
    )


def laugh_backed(
    e: dict[str, Any],
    events: list[dict[str, Any]],
    backup: Callable[[dict[str, Any]], bool] | None = None,
) -> bool:
    """Whether the heard laugh ``e`` may make a moment of its own: its confidence is at least
    :data:`LAUGH_STRONG`, or another laughter, scream or cheering sound is near it
    (:func:`sound_backed`), or ``backup(e)`` (the transcript has a laughter word or marker
    near it, or loud speech covers it; asked last, as it reads the caches)."""
    if (e["confidence"] or 0) >= LAUGH_STRONG or sound_backed(e, events):
        return True
    return bool(backup and backup(e))


def heard_laughter(c: dict[str, Any]) -> bool:
    """Whether laughter was heard in the candidate ``c``: its audio laughter evidence
    (``signals.events.laughter``) is at least :data:`LAUGH_STRONG`."""
    laughter = ((c.get("signals") or {}).get("events") or {}).get("laughter") or 0
    return laughter >= LAUGH_STRONG


def is_nominated(c: dict[str, Any]) -> bool:
    """Whether the candidate ``c`` was nominated by events (see :func:`nominate`)."""
    return bool(((c.get("signals") or {}).get("events") or {}).get("nominated"))


def _set_span(c: dict[str, Any], start: float, end: float) -> None:
    """Give the candidate ``c`` the span ``[start, end]`` (and its ``duration``)."""
    c["start"], c["end"] = start, end
    c["duration"] = round(end - start, 3)


def _set_title(c: dict[str, Any], kept: dict[str, Any], title: str) -> None:
    """Give the widened candidate ``c`` the dialogue stretch's ``title`` (a readable
    ``dialogue_title`` and, as its label, the slug of it); its earlier label is kept in
    ``kept`` (its ``widened_from``) for :func:`restore_widened`. An empty title takes it back."""
    if not title:
        _put_title(c, kept)
        return
    c["dialogue_title"] = title
    if slug := title_slug(title, default=""):  # a title with no Latin letters keeps the label
        kept.setdefault("label", c.get("label"))
        c["label"] = kept["title_label"] = slug


def _put_title(c: dict[str, Any], was: dict[str, Any]) -> None:
    """Take back the title :func:`_set_title` gave ``c``: its label returns unless something
    else (the vision rating) has renamed it since."""
    c.pop("dialogue_title", None)
    if (
        "title_label" in was
        and was.get("label") is not None
        and c.get("label") == was["title_label"]
    ):
        c["label"] = was["label"]
    was.pop("title_label", None)


def stretch_title(events: list[dict[str, Any]], source: str, start: float, end: float) -> str:
    """The title of the dialogue stretch of ``events`` inside ``[start, end]`` of ``source``
    that has one, the most confident (the earlier on a tie); "" for none."""
    best, conf = "", -1.0
    for e in sorted(events, key=lambda e: e["start"]):
        if (
            e.get("kind") == "text"
            and e.get("type") == DIALOGUE_TYPE
            and e.get("source") == source
            and e.get("title")
            and e["start"] >= start - 0.01
            and e["end"] <= end + 0.01
            and (e.get("confidence") or 0) > conf
        ):
            best, conf = str(e["title"]), e.get("confidence") or 0
    return best


def _put_text(c: dict[str, Any], was: dict[str, Any]) -> None:
    """Give ``c`` back the transcript snippet it had before it was widened (``was``)."""
    c.pop("dialogue_description", None)
    if "transcript" in was:
        c["transcript"] = was["transcript"]
        if was.get("transcript_unsure"):
            c["transcript_unsure"] = was["transcript_unsure"]
        else:
            c.pop("transcript_unsure", None)


def restore_widened(cands: list[dict[str, Any]]) -> int:
    """Undo :func:`widen_for_dialogue` on ``cands`` (in place): each widened candidate gets its
    own span back (``widened_from``) and the candidates merged into it return. Returns how many
    were widened."""
    back: list[dict[str, Any]] = []
    n = 0
    for c in cands:
        was = c.pop("widened_from", None)
        if isinstance(was, dict):
            _set_span(c, was["start"], was["end"])
            _put_text(c, was)
            _put_title(c, was)
            back += was.get("absorbed") or []
            n += 1
    cands.extend(back)
    return n


def trim_silence(
    spoken: list[tuple[float, float]],
    start: float,
    end: float,
    core: tuple[float, float],
    min_gap: float = WIDEN_SILENCE_S,
) -> tuple[float, float]:
    """``[start, end]`` without the long silences next to ``core`` (a ``(start, end)`` inside it).

    A silence is a stretch of at least ``min_gap`` seconds with none of the ``spoken`` word
    spans (``(start, end)``, in time order). The span starts after the last silence that
    begins before ``core`` and ends before the first silence that ends after it; never inside
    ``core``, so a silence under the moment itself changes nothing."""
    gaps, edge = [], start
    for a, b in spoken:
        if b <= start or a >= end:
            continue
        gaps.append((edge, max(edge, a)))
        edge = max(edge, b)
    gaps.append((edge, end))
    new_start, new_end = start, end
    for a, b in gaps:
        if b - a < min_gap:
            continue
        if a < core[0]:
            new_start = max(new_start, min(b, core[0]))
        if b > core[1]:
            new_end = min(new_end, max(a, core[1]))
    return new_start, new_end


def widen_for_dialogue(
    cands: list[dict[str, Any]],
    by_src: dict[str, list[dict[str, Any]]],
    menu: dict[str, list[list[float]]] | None = None,
    snippet: Callable[[str, float, float], tuple[str, Any] | None] | None = None,
    spoken: Callable[[str], list[tuple[float, float]]] | None = None,
    handled: set[tuple[str, float, float]] | None = None,
) -> int:
    """Widen the candidates of ``cands`` (in place) that a dialogue stretch overlaps to cover
    the whole stretch, in place of a near-duplicate nominee: the union of the candidate's span
    and the stretch's, not cut to ``max_len``. Candidates a stretch overlaps together become one,
    the best scoring (its id and signals kept); the others are kept inside it. This happens only
    when the union of the stretch and all the candidates it overlaps is at most
    :data:`DIALOGUE_WIDEN_MAX_S` long; otherwise the candidates stay as they are (no partial
    widening) and :func:`nominate` makes the stretch a moment of its own. The earlier span
    is kept in ``widened_from`` (with the merged ones, ``absorbed``), so :func:`restore_widened`
    undoes it and a re-run widens from the same bounds. Visual pool candidates are left alone.
    The widened span keeps only the part of the stretch that leads into the original moments:
    with ``spoken(source)`` (the ``(start, end)`` of the transcript's words; none: no
    trimming) it starts after a silence of :data:`WIDEN_SILENCE_S` or more before the moment
    and ends before one after it (:func:`trim_silence`). The 90 s limit is checked before
    trimming. The ``(source, start, end)`` of each stretch used is added to ``handled``.
    A stretch that starts in menu time, or no candidate overlaps, is left to :func:`nominate`.
    The widened candidate's transcript is rebuilt over its new span with ``snippet(source,
    start, end)`` (``(text, unsure)``, or None without a transcript); without one it gets the
    stretch's description in ``dialogue_description``. The widened candidate takes the title of
    the most confident stretch that widened it (``dialogue_title``, and its slug as the label;
    :func:`restore_widened` gives the label back). Returns how many candidates were widened."""
    from cutpoint.visual_pool import is_pool  # noqa: PLC0415 - imports ski, which imports us

    widened: set[int] = set()
    for src, evs in by_src.items():
        stretches = sorted(
            (e for e in evs if e.get("kind") == "text" and e.get("type") == DIALOGUE_TYPE),
            key=lambda e: e["start"],
        )
        for e in stretches:
            if any(a <= e["start"] < b for a, b in (menu or {}).get(src, ())):
                continue
            hits = [
                c
                for c in cands
                if c["source"] == src
                and not is_pool(c)
                and not is_nominated(c)
                and c["end"] >= e["start"]
                and c["start"] <= e["end"]
            ]
            if not hits:
                continue
            best = max(hits, key=lambda c: c.get("score_pre_events", c["score"]))
            start = min(e["start"], *(c["start"] for c in hits))
            end = max(e["end"], *(c["end"] for c in hits))
            if end - start > DIALOGUE_WIDEN_MAX_S:
                continue
            if spoken and (words := spoken(src)):
                core = (  # where the moments were before any widening
                    min((c.get("widened_from") or c)["start"] for c in hits),
                    max((c.get("widened_from") or c)["end"] for c in hits),
                )
                start, end = trim_silence(words, start, end, core)
            if handled is not None:
                handled.add((src, e["start"], e["end"]))
            if len(hits) == 1 and (start, end) == (best["start"], best["end"]):
                continue
            kept = best.setdefault("widened_from", {"start": best["start"], "end": best["end"]})
            if "transcript" not in kept:  # the first widening keeps the snippet it had
                kept["transcript"] = best.get("transcript", "")
                kept["transcript_unsure"] = best.get("transcript_unsure")
            for c in hits:
                if c is best:
                    continue
                was = c.pop("widened_from", None)  # merged earlier in this pass: as it was
                carry = None
                if isinstance(was, dict):
                    if c.get("dialogue_title") and was.get("title_conf", -1) > kept.get(
                        "title_conf", -1
                    ):  # its stretch's title may be the most confident one
                        carry = (c["dialogue_title"], was["title_conf"])
                    _set_span(c, was["start"], was["end"])
                    _put_text(c, was)
                    _put_title(c, was)
                    kept.setdefault("absorbed", []).extend(was.get("absorbed") or [])
                    if carry:
                        kept["title_conf"] = carry[1]
                        _set_title(best, kept, carry[0])
                kept.setdefault("absorbed", []).append(c)
                cands.remove(c)
            _set_span(best, start, end)
            if e.get("title") and (e.get("confidence") or 0) > kept.get("title_conf", -1):
                kept["title_conf"] = e.get("confidence") or 0  # the most confident title wins
                _set_title(best, kept, str(e["title"]))
            said = snippet(src, start, end) if snippet else None
            if said is not None:
                best["transcript"] = said[0]
                if said[1]:
                    best["transcript_unsure"] = said[1]
                else:
                    best.pop("transcript_unsure", None)
                best.pop("dialogue_description", None)
            else:
                best["dialogue_description"] = e.get("description") or ""
            widened.add(id(best))
    return len(widened)


def nominate(
    hl: dict[str, Any], ev: dict[str, Any], project: Path, parts: list[GridPart] | None = None
) -> int:
    """Nominate candidates for the events of ``ev`` that can score (:func:`seeds`) and that no
    candidate of ``hl`` covers: none with the event inside ``[start, end]``. An event just
    after a clip counts for that clip too (:func:`event_boost`, ``LOOKAHEAD_S``), but gets a
    window of its own after it. Returns how many nominated candidates ``hl`` has now.

    Open-list styles only (``Style.open_list``): any other style gets none. A nominee is
    scored from the grids of its part (``parts``; None:
    from the caches of the analysis, see :func:`cutpoint.analyze.cached_grid_parts`, so a
    source whose caches are gone gets none), as a picked window there would be, and is then
    rescored with the events like the rest. Its analysis score is at least
    :data:`NOMINEE_FLOOR` of the best other candidate's of its source (an event is evidence
    enough that a quiet window is a moment); the grid score it had stays as ``grid_score``,
    for ties. A whole dialogue stretch over :data:`DIALOGUE_WIDEN_MAX_S` (:func:`is_long_dialogue`)
    has exactly that floor as its analysis score, not the peak of the grids inside it: the best
    short moment in minutes of talk would otherwise rank the whole stretch first. The geometry is :func:`cutpoint.analyze.nominate_candidates`'s. Menu time is
    dropped: the events in it, and the windows mostly in it
    (:func:`~cutpoint.events.menu_spans.drop_menu_time`). A part gets at most as many
    candidates in all as its cap (``max_candidates``, else the style's, see
    :func:`cutpoint.analyze.candidate_cap`), the nominees with the best score and boost.

    A dialogue stretch (:func:`seeds`) that a candidate overlaps widens that candidate to cover it
    instead (:func:`widen_for_dialogue`), the candidate's other signals counting as before, when
    the result stays within :data:`DIALOGUE_WIDEN_MAX_S`. A longer stretch that no candidate
    contains is nominated whole with its full length, over the candidates it overlaps (a pool
    candidate it overlaps still covers it).

    Re-running gives the same nominees from the same events and candidates: those of an
    earlier run are replaced, and kept (id, vision rating) when their window is the same.
    A new one has no id (:func:`number_candidates`).
    """
    from cutpoint import analyze  # noqa: PLC0415 - imports this module
    from cutpoint.visual_pool import is_pool  # noqa: PLC0415 - imports ski, which imports us

    try:
        style = get_style(hl.get("style") or "")
    except ValueError:
        return 0
    if not style.open_list:
        return 0
    names = style_and_base(hl.get("style"))
    cands = hl.get("candidates", [])
    types, menu = resolve(project, ev), source_menu_spans(ev)
    by_src: dict[str, list[dict[str, Any]]] = {}
    for e in ev.get("events", []):
        by_src.setdefault(e["source"], []).append(e)
    restore_widened(cands)  # from the same bounds as the first time, whatever the events now
    language = (hl.get("settings") or {}).get("language")
    heard: dict[str, list[dict[str, Any]]] = {}

    def segs(src: str) -> list[dict[str, Any]]:
        if src not in heard:
            heard[src] = _segments(src, language)
        return heard[src]

    def snippet(src: str, a: float, b: float) -> tuple[str, Any] | None:
        """The transcript snippet of ``[a, b]`` as a candidate record has it (None: no transcript)."""
        if not segs(src):
            return None
        said = [w for w in analyze.flat_words(segs(src)) if a <= w["start"] <= b]
        return analyze._transcript_snippet(said)

    grids = parts

    def grid_parts() -> list[GridPart]:
        nonlocal grids
        if grids is None:
            grids = analyze.cached_grid_parts(hl, style)
        return grids

    def backed_by_talk(src: str) -> Callable[[dict[str, Any]], bool]:
        """Whether the transcript of ``src`` has a laughter word or marker near a weak laugh, or
        speech under a loud stretch there (:func:`laugh_backed`)."""
        words: list[float] | None = None

        def backed(e: dict[str, Any]) -> bool:
            nonlocal words
            if words is None:
                words = [t for t, _, _ in analyze.laughter_hits(analyze.flat_words(segs(src)))]
            if any(e["start"] - BACKUP_S <= t <= e["end"] + BACKUP_S for t in words):
                return True
            a, b = e["start"] - 1.0, e["end"] + 1.0
            if not any(s["end"] >= a and s["start"] <= b for s in segs(src)):
                return False
            for p in grid_parts():
                if str(p.source) == src and p.offset <= a < p.offset + p.duration:
                    loud = p.signals.sig["loudness"][
                        int((a - p.offset) / analyze.DT) : int((b - p.offset) / analyze.DT) + 1
                    ]
                    return bool(loud.size and loud.max() >= LOUD_SPEECH)
            return False

        return backed

    said: dict[str, list[tuple[float, float]]] = {}

    def spoken(src: str) -> list[tuple[float, float]]:
        """The ``(start, end)`` of the words of ``src``'s transcript."""
        if src not in said:
            said[src] = [
                (float(w["start"]), float(w.get("end", w["start"])))
                for w in analyze.flat_words(segs(src))
            ]
        return said[src]

    handled: set[tuple[str, float, float]] = set()
    widen_for_dialogue(cands, by_src, menu, snippet, spoken, handled)
    old = [c for c in cands if is_nominated(c)]
    kept = [c for c in cands if not is_nominated(c)]
    own = kept
    todo: dict[str, list[Seed]] = {}
    for src, evs in by_src.items():
        mine = [c for c in own if c["source"] == src]
        left = []
        for s in seeds(
            evs, types.get(src, DEFAULT_EVENT_TYPES), names, menu.get(src), backed_by_talk(src)
        ):
            hit = [c for c in mine if s.end >= c["start"] and s.start <= c["end"]]
            if not hit:
                left.append(s)
            elif (
                s.whole
                and (src, s.start, s.end) not in handled
                and not any(
                    is_pool(c) or (c["start"] <= s.start + 0.01 and c["end"] >= s.end - 0.01)
                    for c in hit
                )
            ):  # too long to widen what it overlaps (and not inside one): a moment of its own
                left.append(replace(s, over=True))
        if left:
            todo[src] = left
    new: list[dict[str, Any]] = []
    done: set[str] = set()
    if todo:
        explicit = (hl.get("settings") or {}).get("max_candidates")
        for part in grid_parts():
            src = str(part.source)
            lo, hi = part.offset, part.offset + part.duration
            here = [s for s in todo.get(src, []) if lo <= s.start < hi]
            if not here:
                continue
            done.add(src)
            taken = [(c["start"], c["end"]) for c in kept if c["source"] == src]  # pool too
            recs = []
            for c in analyze.nominate_candidates(part, style, here, taken):
                rec = analyze._candidate_record(c, part.index, style)
                rec["signals"]["events"] = {"nominated": c.nominated}
                if any(str(n).startswith("dialogue-") for n in c.nominated) and (
                    title := stretch_title(by_src[src], src, rec["start"], rec["end"])
                ):
                    rec["dialogue_title"] = title
                    rec["label"] = title_slug(title, default=rec["label"])
                recs.append(rec)
            menu_hl = {"style": hl.get("style"), "settings": hl.get("settings"), "candidates": recs}
            drop_menu_time(menu_hl, ev)
            recs = menu_hl["candidates"]
            # An event is evidence of a moment where nothing is said: its window scores at least
            # a share of the best picked one's, whatever the length of the list, so the boost
            # decides, not the grids.
            base = [c.get("score_pre_events", c["score"]) for c in own if c["source"] == src]
            floor = NOMINEE_FLOOR * max(base) if base else 0.0
            for r in recs:
                r["grid_score"] = r["score"]
                # a long stretch has the floor itself, not the best moment inside it
                r["score"] = floor if is_long_dialogue(r) else max(r["score"], floor)
            room = analyze.candidate_cap(style, explicit) - sum(
                c["source"] == src and lo <= c["start"] < hi for c in own
            )
            cap = max(0, room)
            if len(recs) > cap:
                src_types = types.get(src, DEFAULT_EVENT_TYPES)
                worth = {
                    id(r): r["score"] + event_boost(r, by_src[src], src_types, style=names)[0]
                    for r in recs
                }
                recs = sorted(recs, key=lambda r: (-worth[id(r)], -r["grid_score"]))[:cap]
            new += recs
        logger.info(
            "Nominated %d candidate(s) for %d event(s) no candidate covers",
            len(new),
            sum(len(v) for v in todo.values()),
        )
    new += [c for c in old if c["source"] in todo and c["source"] not in done]  # no grids now
    same = {(c["source"], round(c["start"], 2), round(c["end"], 2)): c for c in old}
    new = [same.get((r["source"], round(r["start"], 2), round(r["end"], 2)), r) for r in new]
    cands[:] = [*kept, *new]  # in place: callers hold the list
    return len(new)


def number_candidates(hl: dict[str, Any]) -> None:
    """Give the candidates of the analysis ``hl`` that have no ``id`` (nominees) the next ``h``
    ids, after the highest id there is or ever was (``settings.last_candidate_id``, so an id of
    a nominee that went away is not given to another window). The ids that exist stay."""
    cands = hl.get("candidates", [])
    if all("id" in c for c in cands):
        return
    settings = hl.setdefault("settings", {})
    ids = [int(str(c["id"])[1:]) for c in cands if re.fullmatch(r"h\d+", str(c.get("id", "")))]
    last = max([settings.get("last_candidate_id") or 0, *ids])
    for i, c in enumerate(cands):
        if "id" not in c:
            last += 1
            cands[i] = {"id": f"h{last:03d}", **c}
    settings["last_candidate_id"] = last


def rescore_highlights(project: Path) -> int:
    """Rescore ``<project>/highlights.json`` from ``events.json``, after the candidates in
    menu time are dropped or trimmed (:func:`~cutpoint.events.menu_spans.drop_menu_time`).

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
    # widened spans back to their own first: the menu check below sees the real candidates,
    # and nominate widens them again
    restore_widened(hl.get("candidates", []))
    backup = project / "highlights.pre-events.json"
    if not backup.exists() and not hl.get("settings", {}).get("events_rescored"):
        cache.save_json(backup, hl)  # one-time copy of the untouched analysis
    drop_menu_time(hl, ev)
    changed = rescore(hl, ev, project)
    number_candidates(hl)
    cache.save_json(project / "highlights.json", hl)
    logger.info(
        "Rescored %d of %d candidates from events.json", changed, len(hl.get("candidates", []))
    )
    return changed


def rescore(
    hl: dict[str, Any], ev: dict[str, Any], project: Path, parts: list[GridPart] | None = None
) -> int:
    """:func:`rescore_highlights` in memory: rescore the analysis ``hl`` of ``project``
    from the events ``ev`` (an events.json object); returns the candidates changed.

    The scorer's bonus stage (:func:`cutpoint.scorer.rescore_events`), with the evidence of
    this module. ``parts``: the grids of the analysed sources for :func:`nominate` (None:
    from their caches)."""
    return scorer.rescore_events(hl, ev, project, parts)
