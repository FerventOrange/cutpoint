"""Re-analysis of a project with its recorded inputs and style (:func:`reanalyze`), and the ids
of its candidates a re-rank or a fresh analysis keeps (:func:`earlier_ids`, #902, #1005, #1009).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from cutpoint import cache, projectfiles
from cutpoint.projectfiles import HIGHLIGHTS, RERANK_BACKUP
from cutpoint.styles import AUTO, Style, analysis_style, style_name

logger = logging.getLogger(__name__)


def span_key(c: dict[str, Any]) -> tuple[str, float, float]:
    """What names a candidate's span across runs: its source and its start and end to the
    hundredth of a second (as :func:`~cutpoint.events.scoring.nominate` matches nominees)."""
    return (str(c["source"]), round(float(c["start"]), 2), round(float(c["end"]), 2))


def _h_ids(hl: dict[str, Any]) -> tuple[dict[tuple[str, float, float], str], int]:
    """The ``h`` ids of the candidates of ``hl`` by :func:`span_key`, and the highest id it
    ever gave (``settings.last_candidate_id``, at least its highest ``h`` id)."""
    ids = {
        span_key(c): str(c["id"])
        for c in hl.get("candidates") or []
        if re.fullmatch(r"h\d+", str(c.get("id", "")))
    }
    last = (hl.get("settings") or {}).get("last_candidate_id")
    return ids, max([int(last or 0), *(int(i[1:]) for i in ids.values())])


def backup_digest(raw: bytes) -> str:
    """What names a re-rank's backup file by its bytes (``settings.rerank_backup``, #1005)."""
    return hashlib.sha256(raw).hexdigest()[:16]


def earlier_ids(
    hl: dict[str, Any], backup: bytes | None = None
) -> tuple[dict[tuple[str, float, float], str], int]:
    """What a re-rank of the analysis ``hl`` keeps (#902): the ``h`` ids of its candidates by
    :func:`span_key` (``ids`` of :func:`~cutpoint.analyze.rank_parts`), and the highest id it ever
    gave (``settings.last_candidate_id``, at least its highest ``h`` id).

    ``backup``: the bytes of ``highlights.before-rerank.json`` (#1005), used only when they are
    the backup ``hl`` is linked to (:func:`backup_digest` equal to ``settings.rerank_backup``,
    which a re-rank records; a re-analysis and a fresh analysis keep it while it still names the
    backup, :func:`project_backup_link`). A span ``hl``
    lacks gets its id there, unless ``hl`` gives that id to a span (``hl``'s ids come first,
    so no id names another span); its highest id counts too. A backup that is not an analysis,
    whose candidates cannot be read, or that gives a span both have another id than ``hl``
    (edited by hand) is ignored."""
    ids, last = _h_ids(hl)
    link = (hl.get("settings") or {}).get("rerank_backup")
    if backup is None or not link or backup_digest(backup) != link:
        return ids, last  # none, or not the one hl was re-ranked from: stale
    try:
        old, old_last = _h_ids(json.loads(backup.decode("utf-8-sig")))
    except (AttributeError, KeyError, TypeError, ValueError):  # not an analysis: ignored
        logger.debug("the backup's ids cannot be read; ignored", exc_info=True)
        return ids, last
    if any(ids.get(k, i) != i for k, i in old.items()):  # edited since: stale
        logger.debug("the backup's ids differ from highlights.json's; ignored")
        return ids, last
    taken = set(ids.values())
    for key, i in old.items():
        if key not in ids and i not in taken:
            ids[key] = i
            taken.add(i)
    return ids, max(last, old_last)


def project_ids(
    project: Path,
) -> tuple[dict[tuple[str, float, float], str] | None, int]:
    """The ids a fresh analysis into ``project`` keeps (#1009): :func:`earlier_ids` of its
    highlights.json and of its re-rank backup, or ``(None, 0)`` (number from ``h001``) when
    there is no highlights.json or it is not a readable analysis."""
    raw = cache.load_bytes(project / HIGHLIGHTS)
    if raw is None:
        return None, 0
    try:
        hl = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(hl, dict) or not isinstance(hl.get("candidates"), list):
            return None, 0
        return earlier_ids(hl, cache.load_bytes(project / RERANK_BACKUP))
    except (AttributeError, KeyError, TypeError, ValueError):  # not an analysis
        logger.debug(
            "the earlier highlights.json's ids cannot be read; numbering anew", exc_info=True
        )
        return None, 0


def project_backup_link(project: Path) -> str | None:
    """The ``settings.rerank_backup`` of ``project``'s highlights.json when it still names the
    re-rank backup file next to it (:func:`backup_digest` equal), else None (#1009): a fresh
    analysis records it again, so the backup keeps giving ids back after it."""
    raw = cache.load_bytes(project / HIGHLIGHTS)
    backup = cache.load_bytes(project / RERANK_BACKUP)
    if raw is None or backup is None:
        return None
    try:
        link = (json.loads(raw.decode("utf-8-sig")).get("settings") or {}).get("rerank_backup")
    except (AttributeError, TypeError, ValueError):
        return None
    return link if isinstance(link, str) and backup_digest(backup) == link else None


def _recorded_style(hl: dict[str, Any]) -> Style:
    """The style of the analysis ``hl``, with the ``--must-include-crash`` it recorded.

    A style no longer in the library (a deleted user style) degrades to its recorded built-in
    base; ``hl`` without ``style`` has the built-in style :func:`cutpoint.styles.style_name`
    falls back to (:func:`cutpoint.styles.analysis_style`).

    Raises:
        ValueError: If the style is not in the library and ``hl`` records no built-in base.
    """
    style = analysis_style(hl)
    at = (hl.get("settings") or {}).get("must_include_crash")
    if at is not None:  # an `analyze --must-include-crash`
        style = replace(style, must_include_crash=float(at))
    return style


def restyled(hl: dict[str, Any], style: Style) -> Style:
    """``style`` for a re-rank of the analysis ``hl``, with the ``--must-include-crash`` it
    recorded when that was not its own style's (an override of ``analyze``). A file without
    ``style`` has the fallback style of :func:`cutpoint.styles.analysis_style`."""
    at = (hl.get("settings") or {}).get("must_include_crash")
    if not isinstance(at, int | float) or isinstance(at, bool):
        return style
    try:
        own = analysis_style(hl, logging.DEBUG).must_include_crash
    except ValueError:  # a deleted style without a recorded base: no override to tell
        return style
    return style if float(at) == own else replace(style, must_include_crash=float(at))


def rerank_record(hl: dict[str, Any], style: Style) -> dict[str, Any] | None:
    """The ``settings.reranked`` record of the analysis ``hl`` re-ranked to ``style``: ``from``
    the style it had and ``requested`` the style asked for then (``auto``, an action-cam style
    that fell back, or the style of an earlier re-rank; None for a plain one). The same style:
    the record ``hl`` has (None without one)."""
    st = hl.get("settings") or {}
    if style.name == style_name(hl):
        kept = st.get("reranked")
        return kept if isinstance(kept, dict) else None
    requested = st.get("style_requested")
    return {"from": style_name(hl), "requested": requested if isinstance(requested, str) else None}


def reanalyze(
    project: Path,
    style: Style | None = None,
    *,
    backup: bytes | None = None,
    rerank_backup: str | None = None,
) -> Path:
    """Re-run ``analyze`` with the inputs and style recorded in highlights.json.

    Everything expensive is cached (audio, transcript, telemetry), so this is
    quick. ``cutpoint events`` calls it for action-cam projects, whose events feed the
    feature grid itself rather than a score boost. ``timings.total_s`` stays the earlier
    analysis's; the re-rank's own time is ``timings.rerank_s``.

    ``style``: analyze with this style instead (``cutpoint rerank --analyze``), with the
    event layers it plans (made where their caches miss) and the dialogue step; another
    style than the recorded one is the user's choice (:func:`rerank_record`, an auto
    analysis's ``detection`` block kept). None: the recorded style, the layers as they are. A
    project re-ranked before keeps its style and record either way.

    Either way the candidates whose span is unchanged keep their ids and the new spans get ids
    above ``settings.last_candidate_id``, which never goes down (:func:`earlier_ids`, #1006);
    ``backup`` (``cutpoint rerank --analyze``: the bytes of the backup before its own) gives a
    span the current analysis lacks its id there (#1005). ``rerank_backup``: the
    ``settings.rerank_backup`` to record (``rerank --analyze``: its own backup's); None keeps
    the recorded one.

    Raises:
        FileNotFoundError: If the project has no highlights.json.
        ValueError: If highlights.json is unreadable, not valid JSON or records
            no analysis, or its style is not in the library and it records no built-in
            base (:func:`_recorded_style`).
    """
    from cutpoint.analyze import analyze  # noqa: PLC0415 - analyze imports this module
    from cutpoint.pano.project import PANO_FILE  # noqa: PLC0415

    # e.g. an empty {} or a list: a clear error, not a KeyError or AttributeError traceback
    hl = projectfiles.load_highlights(
        project, strict=True, invalid="{path} records no analysis; run 'cutpoint analyze' again"
    )
    st = hl.get("settings") or {}
    inputs = st.get("inputs") or [
        s["path"] for s in hl["sources"] if isinstance(s, dict) and isinstance(s.get("path"), str)
    ]
    pano = (project / PANO_FILE).is_file()
    given = style
    reranked = None if given is None else rerank_record(hl, given)
    if reranked is None and isinstance(st.get("reranked"), dict):
        reranked = st["reranked"]  # re-ranked before: the user's style stays
    style = _recorded_style(hl) if given is None else restyled(hl, given)
    requested = st.get("style_requested")
    if given is None and isinstance(st.get("style_fallback"), dict) and isinstance(requested, str):
        # an action-cam style that fell back to action (#939): asked for again, so the
        # telemetry probe decides as it did (or picks it up once the sources have it)
        from cutpoint.styles import get_style, learned_styles  # noqa: PLC0415

        try:
            style = replace(get_style(requested), must_include_crash=style.must_include_crash)
        except ValueError:  # a deleted user style: the recorded one, without the probe
            if requested in learned_styles():  # there but unusable (broken file): its error
                raise
            logger.info(
                "Style %r is not in the style library (deleted?); using %r", requested, style.name
            )
    total = (hl.get("timings") or {}).get("total_s")
    # --style auto: the style it chose, and its decision kept (never detected again); for a
    # re-ranked project the block only, as recorded
    auto = st.get("style_requested") == AUTO
    detection = hl.get("detection") if auto or reranked is not None else None
    ids, last_id = earlier_ids(hl, backup)
    link = st.get("rerank_backup")
    if rerank_backup is None and isinstance(link, str):
        rerank_backup = link  # the backup still gives ids back after this re-analysis
    return analyze(
        [Path(p) for p in inputs],
        style,
        project,
        # opt-in: only if the earlier run asked (older projects record no request: off)
        vision=st.get("vision_requested") is True,
        vision_top=st.get("vision_top"),
        ollama_url=st.get("ollama_url"),
        ollama_model=st.get("ollama_model"),
        model=st.get("whisper_model"),
        language=st.get("language"),
        use_scenes=bool(st.get("scenes", True)),
        max_candidates=st.get("max_candidates"),
        exclude=st.get("exclude") or [],
        recursive=bool(st.get("recursive")),
        recover=bool(st.get("recover")),
        # the analysis time stays the first run's; this re-rank's is timings.rerank_s
        earlier_total_s=float(total) if isinstance(total, int | float) else None,
        detection=detection if isinstance(detection, dict) else None,
        ocr_det_skip=st.get("ocr_det_skip") is True,
        # off for 360 projects (pano.json: also one analysed before crash_moments was recorded)
        crash_moments=st.get("crash_moments") is not False and not pano,
        telemetry_fallback=not pano,  # 360 projects stay action-cam
        # another style: its own event layers (360 analyze runs its layers itself)
        layers=given is not None and not pano,
        reranked=reranked,
        rerank_backup=rerank_backup,
        ids=ids,
        last_candidate_id=last_id,
        context=st.get("context"),  # #1068: as the analysis was told (footage_context checks it)
        context_recorded=True,
        # #1069: recorded again (nothing is detected again; cutpoint ui reads it back)
        vision_detect_all=st.get("vision_detect_all") is True,
    )


def reanalyze_if_reachable(project: Path) -> Path | None:
    """:func:`reanalyze`, unless a source of the project is offline.

    ``analyze`` refuses a missing input, and without it the source would drop out
    of highlights.json; ``crashes`` and ``events`` already skipped it, so they
    leave highlights.json unchanged instead of failing. Likewise when analyze would
    not find an input, or a recorded source under its recorded path, even through
    ``source_remap``: re-analysing would fail, or drop or re-key the source.

    Returns:
        Path of highlights.json, or None if skipped.
    """
    from cutpoint.analyze import _find_input  # noqa: PLC0415 - analyze imports this module

    hl = projectfiles.load_highlights(project)
    if not isinstance(hl, dict):  # missing or not an object: reanalyze below reports it
        hl = {}
    offline = 0
    recorded = [
        s["path"]
        for s in hl.get("sources") or []
        if isinstance(s, dict) and isinstance(s.get("path"), str)
    ]
    for path in dict.fromkeys(recorded):
        try:
            cache.source_dir(Path(path))
        except OSError:
            offline += 1
    if offline:
        logger.warning(
            "%d source(s) not reachable; highlights.json left unchanged, "
            "rerun once the footage is back",
            offline,
        )
        return None
    # Run analyze's own input search (through ``source_remap`` for a moved input): an
    # input it cannot find would fail it, and a recorded source it does not find under
    # the same spelling would drop out (e.g. moved out of a folder input that still
    # exists) or be re-keyed (an older project that recorded the remapped copy).
    st = hl.get("settings") or {}
    srcs = recorded
    found: set[str] = set()
    missing = 0
    for p in dict.fromkeys(st.get("inputs") or srcs):
        try:
            batch = _find_input(Path(p), st.get("exclude") or [], bool(st.get("recursive")))
        except FileNotFoundError:
            missing += 1
            continue
        found.update(os.path.normcase(str(f)) for f in batch)
    missing += sum(os.path.normcase(str(Path(p))) not in found for p in dict.fromkeys(srcs))
    if missing:
        logger.warning(
            "%d path(s) not found where analyze recorded them, even through source_remap; "
            "highlights.json left unchanged, rerun once the footage is back",
            missing,
        )
        return None
    return reanalyze(project)
