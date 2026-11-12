"""Re-rank an analysed project with another style from its caches, without analyze (#902).

``cutpoint rerank <project> --style NAME`` rebuilds the grids of every source from the caches
of the analysis (loudness, transcript, scene cuts: :func:`cutpoint.analyze.cached_grid_parts`),
picks the style's candidates from them and runs the ranking tail of an analysis
(:func:`cutpoint.analyze.rank_parts`: crash moments, the events' bonus stage, cached vision
ratings only, the final order). It takes seconds, where analyze takes minutes to hours.

:func:`readiness` says whether the caches hold what the style needs; when they do not, the
re-rank is refused (:class:`~cutpoint.errors.NeedsAnalyzeError`, ``needs_analyze``) and
``--analyze`` runs :func:`cutpoint.analyze.reanalyze` with the style instead.

The preview (``--dry-run``) is the re-rank itself, written nowhere: the save writes the same
ranking. Saving replaces ``highlights.json`` and keeps the one before as
:data:`BACKUP` (one Undo). Candidates whose span is unchanged keep their ids; new spans get
ids above ``settings.last_candidate_id``.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cutpoint import analyze, cache, category, features, folders, progress, runstate
from cutpoint.config import projects_dir, remap_source
from cutpoint.errors import NeedsAnalyzeError
from cutpoint.ffmpeg_utils import probe
from cutpoint.styles import ACTION_CAM, Style, style_hash
from cutpoint.utils import command_line

logger = logging.getLogger("cutpoint")

HIGHLIGHTS = "highlights.json"
BACKUP = "highlights.before-rerank.json"  # the highlights.json a save replaced (one Undo)
GENERIC = "generic"  # the style kind of every style but action-cam

# The event layers a footage category plans (:func:`cutpoint.analyze.plan_layers`) before the
# install's own limits; when the project's events.json has them all, the plan is not needed.
_CATEGORY_LAYERS = {
    category.GAMEPLAY: frozenset({"audio", "ocr"}),
    category.TALK: frozenset({"audio"}),
    category.ACTION: frozenset({"audio", "voice", "crash", "visual"}),
    category.GENERAL: frozenset({"audio", "voice", "crash", "visual"}),
    category.ACTION_CAM: frozenset({"audio", "voice", "crash", "visual"}),
}


@dataclass(frozen=True)
class Missing:
    """What keeps a re-rank from the fast path: a ``feature`` code (``360``, ``action_cam``,
    ``style_kind``, ``recovered``, ``offline``, ``loudness``, ``transcript``, ``scenes`` or
    ``layer:<name>``), the ``detail`` the ERROR line gives and its ``source`` (None: the
    project)."""

    feature: str
    detail: str
    source: str | None = None

    def record(self) -> dict[str, str | None]:
        return {"feature": self.feature, "detail": self.detail, "source": self.source}


@dataclass
class Readiness:
    """:func:`readiness`: what is ``missing`` (empty: the fast path runs) and ``notes`` that
    do not block it (logged, and in ``--json`` ``notes``)."""

    missing: list[Missing] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.missing


@dataclass
class Result:
    """A re-rank of ``project``: the analysis ``before`` and ``after`` it (``after`` is what
    was saved, or with ``saved`` False what a save writes), its ``notes``, whether it
    ``analyzed`` instead (``--analyze``), the ``backup`` written and the ``seconds`` taken."""

    project: Path
    before: dict[str, Any]
    after: dict[str, Any]
    notes: list[str]
    saved: bool = False
    analyzed: bool = False
    backup: Path | None = None
    seconds: float = 0.0


def load(project: Path) -> tuple[bytes, dict[str, Any]]:
    """The bytes and the object of ``<project>/highlights.json``.

    Raises:
        FileNotFoundError: If the project has no highlights.json.
        ValueError: If it is not valid JSON or records no analysis.
    """
    path = project / HIGHLIGHTS
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise FileNotFoundError(f"{path} not found; run 'cutpoint analyze' first") from None
    hl = cache.read_json(path, hint="; re-run `cutpoint analyze` to rewrite it")
    if not isinstance(hl, dict) or "style" not in hl or not isinstance(hl.get("sources"), list):
        raise ValueError(f"{path} records no analysis; run 'cutpoint analyze' again")
    return raw, hl


def _has_audio(src: Path) -> bool:
    """Whether ``src`` has an audio stream (True when it cannot be probed: then its missing
    caches count as missing)."""
    try:
        return bool(probe(remap_source(src)).audio_streams)
    except Exception:  # any probe failure: the caches count as missing
        logger.debug("probe of %s failed", src, exc_info=True)
        return True


def _events_layers(project: Path) -> tuple[set[str], list[dict[str, Any]]]:
    """The layers the project's events.json was made with, and its events (none without)."""
    ev = cache.load_json(project / "events.json")
    if not isinstance(ev, dict):
        return set(), []
    layers = (ev.get("settings") or {}).get("layers") or []
    events = [e for e in ev.get("events") or [] if isinstance(e, dict)]
    return {str(x) for x in layers}, events


def _events_extra() -> bool:
    """Whether the events extra is installed (Cutpoint full): the event layers can run."""
    return all(importlib.util.find_spec(m) is not None for m in analyze.EVENTS_MODULES)


def _missing_layers(style: Style, ran: set[str]) -> list[str]:
    """The event layers ``analyze`` with ``style`` plans on this install that are not in
    ``ran`` (the layers of the project's events.json), in the order they run."""
    cat = analyze.layer_category(style)
    if _CATEGORY_LAYERS.get(cat, frozenset()) <= ran:
        return []  # all there: no plan needed (it may load torch for its CUDA check)
    if not _events_extra():
        return []  # lite: analyze would run none of them either
    plan = analyze.plan_layers(style, [], (), cat)  # no sources: no telemetry probe
    return [x for x in plan.all_layers() if x not in ran]


def readiness(project: Path, hl: dict[str, Any], style: Style) -> Readiness:
    """Whether the analysis ``hl`` of ``project`` can be re-ranked with ``style`` from its
    caches alone, as ``analyze`` with ``style`` would rank it.

    Refused (each a :class:`Missing`): a 360 project; an action-cam ``style`` (its feature grid
    is built by analyze) or another style kind than the project's; a source read in parts
    (``offset``, ``read_errors``) or not reachable; a source with audio but no cached loudness
    or transcript; a source without a full scene pass (:data:`~cutpoint.analyze.SCENES_FILE`)
    when scene detection is on and ``style`` weighs scenes, or the source is too long for
    edge windows (:func:`~cutpoint.analyze.edge_windows`); an event layer ``style``'s category
    plans that events.json lacks. Dialogue never read (``style`` reads it, the project's did
    not) is a note only. Reads only: no cache folder is made.
    """
    from cutpoint.pano.project import PANO_FILE  # noqa: PLC0415

    out = Readiness()
    miss = out.missing.append
    if (project / PANO_FILE).is_file():
        miss(Missing("360", "a 360 project is ranked by '360 analyze'"))
    kind = hl.get("style_kind") or GENERIC
    if style.kind == ACTION_CAM:
        miss(
            Missing("action_cam", "an action-cam style scores from the feature grid analyze builds")
        )
    elif kind != style.kind:
        miss(Missing("style_kind", f"the project was analyzed with a style of the {kind} kind"))
    settings = hl.get("settings") or {}
    scenes = bool(settings.get("scenes", True))
    seen: set[str] = set()
    for e in hl.get("sources") or []:
        path = str(e.get("path"))
        if path in seen:  # a recovered source's other parts
            continue
        seen.add(path)
        src = Path(path)
        name = src.name
        if e.get("offset") or e.get("read_errors"):
            miss(Missing("recovered", f"{name} was read in parts (read errors)", path))
            continue
        try:
            cdir = cache.source_dir(src, create=False)
        except OSError:
            miss(Missing("offline", f"{name} is not reachable", path))
            continue
        loud = (cdir / "loudness.json").is_file()
        heard = cdir.is_dir() and any(cdir.glob("transcript-*.json"))
        if not (loud and heard) and _has_audio(src):
            if not loud:
                miss(Missing("loudness", f"{name} has no cached loudness", path))
            if not heard:
                miss(Missing("transcript", f"{name} has no cached transcript", path))
        if scenes and not (cdir / analyze.SCENES_FILE).is_file():
            span = float(e.get("analyzed_s") or e.get("duration") or 0.0)
            if analyze.scores_scenes(style) or not analyze.edge_windows(style, span):
                miss(Missing("scenes", f"{name} has no full scene pass", path))
    ran, events = _events_layers(project)
    for layer in _missing_layers(style, ran):
        miss(Missing(f"layer:{layer}", f"the {layer} layer has not run for this project"))
    timings = hl.get("timings") or {}
    if (
        style.open_list
        and "dialogue_s" not in timings
        and not any(e.get("kind") == "text" and e.get("type") == "dialogue" for e in events)
    ):
        out.notes.append(
            f"Dialogue was never read for this project; with the {style.name} style analyze "
            "reads it, and its talk stretches are not candidates until it does"
        )
    return out


def analyze_command(project: Path, style: Style) -> str:
    """The command that analyzes ``project`` again with ``style`` (its recorded inputs and
    settings): ``cutpoint rerank <project> --style NAME --analyze``."""
    here = project.absolute()
    where = here.name if here.parent == projects_dir().absolute() else str(here)
    return "cutpoint " + command_line(["rerank", where, "--style", style.name, "--analyze"])


def _refused(project: Path, style: Style, missing: list[Missing]) -> NeedsAnalyzeError:
    command = analyze_command(project, style)
    shown = "; ".join(m.detail for m in missing[:5])
    more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
    message = (
        f"Re-ranking {project.name} with the {style.name} style needs analyze: "
        f"{shown}{more}. Run: {command}"
    )
    return NeedsAnalyzeError(message, [m.record() for m in missing], command)


def ranked(project: Path, hl: dict[str, Any], style: Style) -> dict[str, Any]:
    """The analysis ``hl`` of ``project`` re-ranked with ``style`` from its caches (see
    :func:`readiness`), as a new highlights.json object; writes nothing to the project.

    The style's fields and settings replace the earlier style's (its ``--must-include-crash``
    override kept, :func:`~cutpoint.analyze.restyled`); another style is the user's choice
    (``settings.style_requested``, ``settings.reranked``; an auto analysis's ``detection``
    stays). Vision takes cached ratings only. ``timings`` keep the analysis's, with this
    re-rank's ``rerank_s``.
    """
    t0 = time.perf_counter()
    style = analyze.restyled(hl, style)
    new = copy.deepcopy({k: v for k, v in hl.items() if k != "candidates"})
    new.update(style=style.name, style_kind=style.kind, style_base=style.base)
    st = new.setdefault("settings", {})
    reranked = analyze.rerank_record(hl, style)
    if reranked is not None:
        st["style_requested"] = style.name
        st["reranked"] = reranked
        st.pop("style_fallback", None)
    st.pop("visual_pool", None)  # set again for a style with a visual pool
    st.update(
        must_include_crash=style.must_include_crash,
        target_total=style.target_total,
        weights=style.weights,
        style_hash=style_hash(style),
        vision_used=False,
        vision=None,
    )
    old = hl.get("candidates") or []
    ids = {
        analyze.span_key(c): str(c["id"])
        for c in old
        if re.fullmatch(r"h\d+", str(c.get("id", "")))
    }
    st["last_candidate_id"] = max(
        [int(st.get("last_candidate_id") or 0), *(int(i[1:]) for i in ids.values())]
    )
    timings = new.setdefault("timings", {})
    timings.pop("vision_s", None)
    grids = {p.index: p for p in analyze.cached_grid_parts(new, style)}
    parts: list[analyze.HighlightsPart] = []
    by_source: dict[str, list[str]] = {}
    for i, e in enumerate(new.get("sources") or []):
        g = grids.get(i)
        if g is None:  # not reachable through readiness; its candidates go
            span = float(e.get("analyzed_s") or e["duration"])
            parts.append(
                analyze.HighlightsPart(
                    Path(e["path"]), float(e.get("offset") or 0.0), span, [], None
                )
            )
            by_source[str(e["path"])] = sorted(features.available())
            continue
        picked = analyze.pick_candidates(g, style, st.get("max_candidates"), {})
        parts.append(analyze.HighlightsPart(g.source, g.offset, g.duration, picked, g.signals))
        by_source[str(g.source)] = sorted(features.available() if g.avail is None else g.avail)
    st["features_by_source"] = by_source
    total = (hl.get("timings") or {}).get("total_s")
    with progress.run(progress.ANALYZE_STEPS, use=False) as steps:
        steps.start("ranking")
        analyze.rank_parts(
            new,
            parts,
            style,
            project,
            steps,
            t_start=t0,
            crash_moments=st.get("crash_moments") is not False,
            vision=st.get("vision_requested") is True,
            vision_top=st.get("vision_top"),
            ollama_url=st.get("ollama_url"),
            ollama_model=st.get("ollama_model"),
            batch_size=0,
            vision_time_share=None,
            earlier_total_s=float(total) if isinstance(total, int | float) else None,
            ids=ids,
            vision_cache_only=True,
        )
    timings["rerank_s"] = round(time.perf_counter() - t0, 2)
    return new


def rerank(
    project: Path, style: Style, *, dry_run: bool = False, analyze_missing: bool = False
) -> Result:
    """Re-rank ``project`` with ``style`` (:func:`ranked`) and, unless ``dry_run``, save it:
    highlights.json replaced and the one before kept as :data:`BACKUP`, in one commit under the
    project's run lock (``folder_busy`` while another run holds it).

    When :func:`readiness` refuses the fast path: with ``analyze_missing`` (not with
    ``dry_run``), :func:`cutpoint.analyze.reanalyze` with ``style`` instead (the backup kept as
    well); else :class:`~cutpoint.errors.NeedsAnalyzeError`.

    Raises:
        NeedsAnalyzeError: If the caches lack what ``style`` needs (see above).
        FileNotFoundError: If the project has no highlights.json.
        ValueError: If highlights.json records no analysis.
        FolderBusyError: If another run holds the project's lock (not with ``dry_run``).
    """
    t0 = time.perf_counter()
    lock = (
        contextlib.nullcontext()
        if dry_run
        else folders.swap_lock(project / runstate.LOCK_NAME, project)
    )
    with lock:
        raw, hl = load(project)
        ready = readiness(project, hl, style)
        for note in ready.notes:
            logger.info("%s", note)
        if not ready.ready:
            if dry_run or not analyze_missing:
                raise _refused(project, style, ready.missing)
            logger.info(
                "Re-ranking with the %s style needs analyze (%s); analyzing again",
                style.name,
                "; ".join(m.detail for m in ready.missing),
            )
            analyze.reanalyze(project, style)
            cache.save_bytes(project / BACKUP, raw)
            _, after = load(project)
            return Result(
                project,
                hl,
                after,
                ready.notes,
                saved=True,
                analyzed=True,
                backup=project / BACKUP,
                seconds=round(time.perf_counter() - t0, 2),
            )
        after = ranked(project, hl, style)
        result = Result(project, hl, after, ready.notes)
        if not dry_run:
            with cache.hold_outputs(project, (HIGHLIGHTS, BACKUP)) as held:
                cache.save_bytes(project / BACKUP, raw)
                cache.save_json(project / HIGHLIGHTS, after)
                held.commit()
            result.saved, result.backup = True, project / BACKUP
            logger.info(
                "Re-ranked %s with the %s style; the earlier ranking is in %s",
                project / HIGHLIGHTS,
                style.name,
                BACKUP,
            )
        result.seconds = round(time.perf_counter() - t0, 2)
        return result


def _ranks(hl: dict[str, Any]) -> dict[str, int]:
    """The rank (1 = first) of each scored candidate's id in ``hl`` (its file order)."""
    from cutpoint.visual_pool import is_pool  # noqa: PLC0415

    cands = [c for c in hl.get("candidates") or [] if not is_pool(c) and "id" in c]
    return {str(c["id"]): i for i, c in enumerate(cands, 1)}


def rows(hl: dict[str, Any], top: int, other: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The first ``top`` scored candidates of ``hl``: ``rank``, ``id``, ``source``, ``start``,
    ``end``, ``score``, ``label`` and, with ``other`` (the ranking compared with), ``was``: the
    id's rank there, or None for a span it does not have."""
    from cutpoint.visual_pool import is_pool  # noqa: PLC0415

    there = _ranks(other) if other is not None else None
    cands = [c for c in hl.get("candidates") or [] if not is_pool(c)][:top]
    out = []
    for rank, c in enumerate(cands, 1):
        row = {
            "rank": rank,
            "id": c.get("id"),
            "source": c.get("source"),
            "start": c.get("start"),
            "end": c.get("end"),
            "score": c.get("score"),
            "label": c.get("label"),
        }
        if there is not None:
            row["was"] = there.get(str(c.get("id")))
        out.append(row)
    return out


def changes(before: dict[str, Any], after: dict[str, Any]) -> tuple[list[str], list[str]]:
    """The ids ``after`` has that ``before`` has not (new spans), and those it lost."""
    a, b = _ranks(after), _ranks(before)
    return [i for i in a if i not in b], [i for i in b if i not in a]
