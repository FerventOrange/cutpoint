"""Crash moments outside action-cam: the crashes of crashes.json made candidates of an analysis
(:func:`add_crash_candidates`) and rebuilt for a re-run of ``cutpoint events``
(:func:`rebuild_crash_moments`).
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from cutpoint import backends, cache, projectfiles, scorer
from cutpoint.config import path_key
from cutpoint.events.menu_spans import drop_menu_time
from cutpoint.events.scoring import is_nominated, number_candidates, restore_widened
from cutpoint.pool_record import is_pool
from cutpoint.projectfiles import EVENTS_JSON, HIGHLIGHTS
from cutpoint.reanalyze import _recorded_style
from cutpoint.styles import ACTION_CAM, Style

if TYPE_CHECKING:
    from cutpoint.analyze import Candidate, PartSignals

logger = logging.getLogger(__name__)

CRASH_KIND = (
    "crash"  # the ``kind`` of a crash moment outside action-cam (action-cam's: ``wipeout``)
)


def _crash_event(c: dict[str, Any]) -> dict[str, Any]:
    """The ``crash`` record of a candidate for the crashes.json crash ``c`` (as action-cam's)."""
    return {
        "type": "crash",
        "id": c.get("id"),
        "t": c["t0"],
        "end": c["t1"],
        "confidence": c["confidence"],
    }


def _crash_entries(crashes: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The usable crashes of a crashes.json object: a source, a finite confidence, ``t0`` and
    ``t1``, and a planned clip (with a reaction that has a start and end, or none)."""

    def num(x: Any) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    raw = crashes.get("crashes") if isinstance(crashes, dict) else None
    return [
        c
        for c in (raw if isinstance(raw, list) else [])
        if isinstance(c, dict)
        and isinstance(c.get("source"), str)
        and all(num(c.get(k)) for k in ("confidence", "t0", "t1"))
        and isinstance(c.get("clip"), dict)
        and num(c["clip"].get("start"))
        and num(c["clip"].get("end"))
        and (
            (r := c["clip"].get("reaction")) is None
            or (isinstance(r, dict) and num(r.get("start")) and num(r.get("end")))
        )
    ]


class CrashPart(Protocol):
    """What :func:`add_crash_candidates` reads and changes of a part: a
    :class:`~cutpoint.analyze.SourceResult` of this run, or a :class:`HighlightsPart` of an earlier
    one."""

    source: Path
    offset: float
    duration: float
    candidates: list[Candidate]
    signals: PartSignals | None


@dataclass
class HighlightsPart:
    """A part of an earlier analysis, for :func:`rebuild_crash_moments`: its cached grids and
    the candidates of its highlights.json records."""

    source: Path
    offset: float
    duration: float
    candidates: list[Candidate]
    signals: PartSignals | None


def _crash_windows(r: CrashPart, c: dict[str, Any], style: Style, forced: bool) -> list[Candidate]:
    """The candidates of the crash ``c`` in the part ``r`` (absolute source time): its planned
    clip (approach, impact, aftermath) and, when ``forced``, its reaction. A forced one scores
    its confidence (:func:`cutpoint.scorer.score`); any other is scored from the part's grids
    over its span, as a picked window there would be. Word edges only (no scene snapping)."""
    from cutpoint.analyze import DT, Candidate, _snap, describe_candidates  # noqa: PLC0415 - analyze imports this module

    ps, off, span = r.signals, r.offset, r.duration
    if ps is None:
        return []
    spans = [("crash", c["clip"]["start"], c["clip"]["end"], c["t0"])]
    reaction = c["clip"].get("reaction")
    if forced and isinstance(reaction, dict):
        spans.append(("crash reaction", reaction["start"], reaction["end"], reaction["start"]))
    words_only = replace(style, snap_sentences=False)
    combined = scorer.combined(ps.sig, style)
    out = []
    for label, a, b, peak_t in spans:
        a, b = max(0.0, a - off), min(span, b - off)
        if b - a < 1.0:
            continue  # in another part of a recovered source
        cand = Candidate(
            r.source,
            a,
            b,
            peak_t - off,
            0.0,
            {},
            label_hint=label,
            kind=CRASH_KIND,
            must_include=forced,
            event=_crash_event(c),
        )
        _snap(cand, [], ps.segments, span, words_only)
        i = max(0, min(int(cand.start / DT), ps.n - 1))
        j = max(i + 1, int(cand.end / DT))
        cand.signals = {k: round(float(v[i:j].max()), 3) for k, v in ps.sig.items()}
        cand.peak = float(c["confidence"]) if forced else float(combined[i:j].max())
        out.append(cand)
    return describe_candidates(out, style, ps, off)


def add_crash_candidates(
    results: Sequence[CrashPart], style: Style, crashes: dict[str, Any] | None
) -> None:
    """Make the crashes of ``crashes`` (a crashes.json object) moments of ``results`` (any
    style but action-cam, whose own scorer reads crashes.json).

    Only crashes at or above the project's floor (:func:`cutpoint.crash.crash_floor`) count.
    One at or above ``style.must_include_crash`` (0: none) is forced, as in action-cam: its
    clip and its reaction become must-include candidates (``kind`` crash) that replace the
    candidates they overlap. A weaker one marks the candidate it overlaps most (its ``event``;
    span and score stay), or else becomes a candidate of its own, scored and ranked like any
    other. A crash counts in the part (of a recovered source) its ``t0`` falls in."""
    from cutpoint.crash import crash_floor  # noqa: PLC0415

    entries = _crash_entries(crashes)
    if not entries:
        return
    floor = crash_floor(c["confidence"] for c in entries)
    at = style.must_include_crash
    n_forced = n_new = n_marked = 0
    for r in results:
        if r.signals is None:
            continue
        key = path_key(str(r.source))
        mine = [
            c
            for c in entries
            if c["confidence"] >= floor
            and path_key(c["source"]) == key
            and r.offset <= c["t0"] < r.offset + r.duration
        ]
        forced = [c for c in mine if at > 0 and c["confidence"] >= at]
        made = [x for c in forced for x in _crash_windows(r, c, style, forced=True)]
        if made:
            r.candidates = made + [
                x
                for x in r.candidates
                if not any(x.start < f.end and f.start < x.end for f in made)
            ]
            n_forced += len(forced)
        weak = sorted((c for c in mine if c not in forced), key=lambda c: -c["confidence"])
        for c in weak:
            a, b = c["clip"]["start"], c["clip"]["end"]
            hits = [x for x in r.candidates if x.start < b and a < x.end]
            if hits:
                host = max(hits, key=lambda x: min(b, x.end) - max(a, x.start))
                if host.event is None:  # a crash moment shows its own crash
                    host.event = _crash_event(c)
                    n_marked += 1
                continue
            new = _crash_windows(r, c, style, forced=False)
            r.candidates += new
            n_new += len(new)
    if n_forced or n_new or n_marked:
        logger.info(
            "Crashes: %d must-include (confidence >= %.2f), %d new moment(s), %d marked "
            "(floor %.2f)",
            n_forced,
            at,
            n_new,
            n_marked,
            floor,
        )


CRASH_MARKS = ("crash", "crash_confidence", "crash_forced")  # the keys of a candidate's crash


def _crash_keys(recs: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    """The crash moments ``recs`` by what names one across runs: its part and crash (id and
    impact span, not its confidence) and its place among that crash's windows in time order
    (0: its clip, 1: its reaction). Not its span, which menu time may have trimmed, nor its
    label."""
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for rec in sorted(recs, key=lambda r: (r["source_index"], r["start"])):
        crash = rec.get("crash") or {}
        base = (rec["source_index"], crash.get("id"), crash.get("t"), crash.get("end"))
        out[(*base, sum(k[:4] == base for k in out))] = rec
    return out


def _canonical(hl: dict[str, Any]) -> dict[str, Any]:
    """``hl`` with its candidates in a fixed order, to compare two analyses whatever the order."""
    cands = sorted(hl.get("candidates", []), key=lambda c: str(c.get("id", "")))
    return {**hl, "candidates": cands}


def rebuild_crash_moments(project: Path) -> bool:
    """Rebuild the crash moments of ``<project>/highlights.json`` from the current crashes.json, as
    :func:`~cutpoint.analyze.analyze` and :func:`~cutpoint.reanalyze.reanalyze` build them
    (:func:`add_crash_candidates`), for a re-run of ``cutpoint events`` that does not re-analyse.
    Returns whether it wrote the file.

    The crash moments and marks of the analysis are taken off and made again over the grids its
    caches give (:func:`~cutpoint.analyze.cached_grid_parts`), so a moment whose crash is unchanged
    stays as it is (id, vision rating, a span menu time trimmed) and repeat runs change nothing. A
    moment is matched by its crash, not its span. As in :func:`~cutpoint.analyze.analyze`, the
    candidates events nominated host no crash (they come after the crash moments; the re-rank
    nominates again) and menu time drops or trims a new crash moment (:func:`drop_menu_time`) before
    it takes an id. New ones get the next ids (``settings.last_candidate_id``). Left alone:
    action-cam (its scorer reads crashes.json), 360 projects, ``settings.crash_moments`` false, and
    a source whose grids are not cached (recovered, offline). A candidate that a forced crash moment
    replaced is not brought back when that crash goes away: only a re-analysis does. A file without
    ``style`` takes the fallback style of :func:`~cutpoint.reanalyze._recorded_style`.

    Raises:
        ValueError: If the style is not in the library and the file records no built-in base
            (:func:`~cutpoint.reanalyze._recorded_style`).
    """
    from cutpoint.analyze import (  # noqa: PLC0415 - analyze imports this module
        Candidate,
        _candidate_record,
        cached_grid_parts,
    )
    from cutpoint.crash import load_crashes  # noqa: PLC0415
    from cutpoint.pano.project import PANO_FILE  # noqa: PLC0415

    path = project / HIGHLIGHTS
    hl = projectfiles.load_highlights(project)
    if not isinstance(hl, dict) or "sources" not in hl:
        return False
    if (hl.get("settings") or {}).get("crash_moments") is False or (project / PANO_FILE).is_file():
        return False
    style = _recorded_style(hl)
    if style.kind == ACTION_CAM:
        return False
    parts = {
        p.index: HighlightsPart(p.source, p.offset, p.duration, [], p.signals)
        for p in cached_grid_parts(hl, style)
    }
    if not parts:
        return False
    cands = hl.get("candidates", [])
    restore_widened(cands)  # as analyze made the crash moments, before any widening
    before = copy.deepcopy(hl)
    rest: list[dict[str, Any]] = []
    rec_of: dict[int, dict[str, Any]] = {}  # id() of a candidate made from a record -> record
    olds: list[dict[str, Any]] = []  # the crash moments there were
    for rec in cands:
        i = rec.get("source_index")
        if i not in parts or is_pool(rec):
            rest.append(rec)
        elif is_nominated(rec):  # analyze builds crash moments before events nominate
            for k in CRASH_MARKS:
                rec.pop(k, None)
            rest.append(rec)
        elif rec.get("kind") == CRASH_KIND:
            olds.append(rec)
        else:
            c = Candidate(parts[i].source, rec["start"], rec["end"], rec["start"], 0.0, {})
            parts[i].candidates.append(c)
            rec_of[id(c)] = rec
    add_crash_candidates(list(parts.values()), style, load_crashes(project))
    kept: list[dict[str, Any]] = []
    built: list[dict[str, Any]] = []  # the crash moments made now
    for i, part in parts.items():
        for c in part.candidates:
            rec = rec_of.get(id(c))
            if rec is not None:  # a candidate of the analysis: with its crash mark, or none
                for k in CRASH_MARKS:
                    rec.pop(k, None)
                if c.event:
                    rec["crash"] = c.event
                    rec["crash_confidence"] = round(float(c.event["confidence"]), 3)
                    rec["crash_forced"] = c.must_include
                kept.append(rec)
                continue
            built.append(_candidate_record(c, i, style))
    was = _crash_keys(olds)
    fresh: list[dict[str, Any]] = []  # crash moments with no earlier record
    for key, new in _crash_keys(built).items():
        old = was.get(key)
        if old is None:
            fresh.append(new)
            continue
        # the same crash: as it was (a span menu time trimmed included) if its marks are the
        # same, else the new moment with its id
        same = all(old.get(k) == new.get(k) for k in (*CRASH_MARKS, "must_include"))
        kept.append(old if same else ({"id": old["id"]} if "id" in old else {}) | new)
    ev = cache.load_json(project / EVENTS_JSON)
    if fresh and ev:  # menu time drops or trims a new one as analyze does, so it takes no id
        menu_hl = {
            "style": hl.get("style") or style.name,
            "style_base": hl.get("style_base"),
            "settings": hl.get("settings"),
            "candidates": fresh,
        }
        drop_menu_time(menu_hl, ev)
        fresh = menu_hl["candidates"]
    pool = [c for c in rest if is_pool(c)]
    kept = sorted(
        [c for c in rest if not is_pool(c)] + kept + fresh,
        key=lambda c: (not c.get("must_include"), *scorer.rank_key(c), *scorer.tie_key(c)),
    )
    hl["candidates"] = kept + pool
    number_candidates(hl)
    if _canonical(hl) == _canonical(before):  # the same records, maybe in another order
        return False
    cache.save_json(path, backends.saved_settings(hl), durable=True)
    logger.info("Crash moments rebuilt from crashes.json in %s", path.name)
    return True
