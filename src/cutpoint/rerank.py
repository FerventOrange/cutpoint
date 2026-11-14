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
:data:`BACKUP` (one Undo). Candidates whose span is unchanged keep their ids, a span only the
backup has gets its id there (#1005: a switch of style and back restores the ids), and new
spans get ids above ``settings.last_candidate_id``. A saved ranking is sealed
(``settings.rerank_undo``, :func:`seal`): ``cutpoint ui`` offers its Undo while highlights.json
is still the file the save wrote.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cutpoint import analyze, cache, category, features, folders, progress, runstate
from cutpoint.config import projects_dir, remap_source
from cutpoint.errors import NeedsAnalyzeError, StaleRankingError
from cutpoint.ffmpeg_utils import probe
from cutpoint.styles import ACTION_CAM, Style, style_hash
from cutpoint.utils import command_line

logger = logging.getLogger("cutpoint")

HIGHLIGHTS = "highlights.json"
BACKUP = "highlights.before-rerank.json"  # the highlights.json a save replaced (one Undo)
GENERIC = "generic"  # the style kind of every style but action-cam
# settings key of a ranking a save (or cutpoint ui's Undo) wrote: {"seal": seal(hl), "undone": bool}
UNDO = "rerank_undo"

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
    ``style_kind``, ``recovered``, ``offline``, ``loudness``, ``transcript``, ``scenes``,
    ``caches`` or ``layer:<name>``), the ``detail`` the ERROR line gives and its ``source``
    (None: the project)."""

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


def _backup(project: Path) -> bytes | None:
    """The bytes of the project's :data:`BACKUP` (the ranking before the last re-rank), or
    None without one: :func:`cutpoint.analyze.earlier_ids` uses it only when highlights.json
    is linked to it (``settings.rerank_backup``)."""
    return cache.load_bytes(project / BACKUP)


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
    if not isinstance(hl, dict) or not isinstance(hl.get("sources"), list):
        raise ValueError(f"{path} records no analysis; run 'cutpoint analyze' again")
    return raw, hl


def key(raw: bytes) -> str:
    """The ranking key of the highlights.json bytes ``raw`` (sha256, its first 16 hex digits):
    it changes whenever an analysis or a re-rank rewrites the file. Same digest as
    :func:`analyze.backup_digest`."""
    return analyze.backup_digest(raw)


def seal(hl: dict[str, Any]) -> str:
    """The digest of the analysis ``hl`` as a file holds it, its ``settings.rerank_undo`` (:data:`UNDO`)
    left out: equal to the recorded seal only while no analysis, ``events`` run or hand edit
    has rewritten the file since a re-rank save (or ``cutpoint ui``'s Undo) wrote it (a
    re-analysis keeps ``settings.rerank_backup`` but records no seal)."""
    data = json.loads(json.dumps(hl))  # as the file reads back: string keys, lists
    st = data.get("settings")
    if isinstance(st, dict):
        st.pop(UNDO, None)
    text = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return analyze.backup_digest(text.encode("utf-8"))


def sealed(hl: dict[str, Any], *, undone: bool = False) -> dict[str, Any]:
    """``hl`` with its ``settings.rerank_undo`` (:data:`UNDO`) set: its :func:`seal` and whether
    it is the ranking an Undo went back to (``undone``: the next swap is a Redo)."""
    hl.setdefault("settings", {})[UNDO] = {"seal": seal(hl), "undone": undone}
    return hl


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


def _missing_layers(
    style: Style, ran: set[str], cuda: Callable[[], bool] | None = None
) -> list[str]:
    """The event layers ``analyze`` with ``style`` plans on this install that are not in
    ``ran`` (the layers of the project's events.json), in the order they run. ``cuda``: the
    plan's CUDA check (:func:`cutpoint.analyze.plan_layers`)."""
    cat = analyze.layer_category(style)
    if _CATEGORY_LAYERS.get(cat, frozenset()) <= ran:
        return []  # all there: no plan needed (it may load torch for its CUDA check)
    if not _events_extra():
        return []  # lite: analyze would run none of them either
    plan = analyze.plan_layers(style, [], (), cat, cuda=cuda)  # no sources: no telemetry probe
    return [x for x in plan.all_layers() if x not in ran]


def readiness(
    project: Path, hl: dict[str, Any], style: Style, cuda: Callable[[], bool] | None = None
) -> Readiness:
    """Whether the analysis ``hl`` of ``project`` can be re-ranked with ``style`` from its
    caches alone, as ``analyze`` with ``style`` would rank it.

    Refused (each a :class:`Missing`): a 360 project; an action-cam ``style`` (its feature grid
    is built by analyze) or another style kind than the project's; a source read in parts
    (``offset``, ``read_errors``) or not reachable; a source with audio but no cached loudness
    or transcript; a source without a full scene pass (:data:`~cutpoint.analyze.SCENES_FILE`)
    when scene detection is on and ``style`` weighs scenes, or the source is too long for
    edge windows (:func:`~cutpoint.analyze.edge_windows`); a source with nothing cached to
    build its grid from (no audio and no scene cuts); an event layer ``style``'s category
    plans that events.json lacks. Dialogue never read (``style`` reads it, the project's did
    not) is a note only. Reads only: no cache folder is made. ``cuda``: whether a CUDA GPU is
    there, for the layer plan (None: torch's own check in :func:`cutpoint.analyze.plan_layers`;
    ``cutpoint ui`` asks a child process, never importing torch itself).
    """
    out = Readiness()
    miss = out.missing.append
    if _is_360(project):
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
        span = float(e.get("analyzed_s") or e.get("duration") or 0.0)
        if not span:
            miss(Missing("length", _no_length(name), path))
            continue
        found = (cdir / analyze.SCENES_FILE).is_file()
        windowed = scenes and not found and analyze.edge_windows(style, span)
        if scenes and not found and (analyze.scores_scenes(style) or not windowed):
            miss(Missing("scenes", f"{name} has no full scene pass", path))
            continue
        # as cached_grid_parts: no loudness, transcript, scene cuts or edge windows to decode
        # (scenes off) leaves the source no grid to rank
        if not (loud or heard or found or windowed):
            miss(Missing("caches", _nothing_cached(name), path))
    ran, events = _events_layers(project)
    for layer in _missing_layers(style, ran, cuda):
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


def _is_360(project: Path) -> bool:
    from cutpoint.pano.project import PANO_FILE  # noqa: PLC0415

    return (project / PANO_FILE).is_file()


def analyze_command(project: Path, style: Style) -> str:
    """The command that analyzes ``project`` again with ``style`` (its recorded inputs and
    settings): ``cutpoint rerank <project> --style NAME --analyze``, for a 360 project
    ``cutpoint 360 analyze <project> --style NAME``."""
    here = project.absolute()
    where = here.name if here.parent == projects_dir().absolute() else str(here)
    if _is_360(project):
        return "cutpoint " + command_line(["360", "analyze", where, "--style", style.name])
    return "cutpoint " + command_line(["rerank", where, "--style", style.name, "--analyze"])


def _nothing_cached(name: str) -> str:
    return f"{name} has no cached loudness, transcript or scene cuts to build a ranking from"


def _no_length(name: str) -> str:
    return f"{name} records no length (duration) in highlights.json"


def _refused(project: Path, style: Style, missing: list[Missing]) -> NeedsAnalyzeError:
    command = analyze_command(project, style)
    shown = "; ".join(m.detail for m in missing[:5])
    more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
    offline = [m for m in missing if m.feature == "offline"]
    if offline and len(offline) == len(missing):  # analyze can't read it either
        then = "Reconnect the footage and run the rerank again"
    elif offline:
        then = f"Reconnect the footage, then run: {command}"
    else:
        then = f"Run: {command}"
    message = (
        f"Re-ranking {project.name} with the {style.name} style needs analyze: "
        f"{shown}{more}. {then}"
    )
    return NeedsAnalyzeError(message, [m.record() for m in missing], command)


def ranked(
    project: Path, hl: dict[str, Any], style: Style, raw: bytes | None = None
) -> dict[str, Any]:
    """The analysis ``hl`` of ``project`` re-ranked with ``style`` from its caches (see
    :func:`readiness`), as a new highlights.json object; writes nothing to the project.

    The style's fields and settings replace the earlier style's (its ``--must-include-crash``
    override kept, :func:`~cutpoint.analyze.restyled`); another style is the user's choice
    (``settings.style_requested``, ``settings.reranked``; an auto analysis's ``detection``
    stays). Vision takes cached ratings only. ``timings`` keep the analysis's, with this
    re-rank's ``rerank_s``. ``settings.rerank_backup`` links it to the backup a save keeps:
    :func:`~cutpoint.analyze.backup_digest` of ``raw``, the bytes of the project's
    highlights.json (read when None); ``settings.rerank_undo`` seals it (:func:`sealed`).
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
    # read before a save replaces the backup: a span only it has gets its id back (#1005)
    ids, st["last_candidate_id"] = analyze.earlier_ids(hl, _backup(project))
    if raw is None:
        raw = (project / HIGHLIGHTS).read_bytes()
    st["rerank_backup"] = analyze.backup_digest(raw)  # the backup a save writes
    timings = new.setdefault("timings", {})
    timings.pop("vision_s", None)
    grids = {p.index: p for p in analyze.cached_grid_parts(new, style)}
    parts: list[analyze.HighlightsPart] = []
    by_source: dict[str, list[str]] = {}
    for i, e in enumerate(new.get("sources") or []):
        g = grids.get(i)
        if g is None:  # readiness refuses such a source: never drop its candidates quietly
            path = str(e["path"])
            name = Path(path).name
            if not (e.get("analyzed_s") or e.get("duration")):
                raise _refused(project, style, [Missing("length", _no_length(name), path)])
            raise _refused(project, style, [Missing("caches", _nothing_cached(name), path)])
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
    return sealed(new)  # last: the seal covers everything a save writes


def rerank(
    project: Path,
    style: Style,
    *,
    dry_run: bool = False,
    analyze_missing: bool = False,
    expect: str | None = None,
    cuda: Callable[[], bool] | None = None,
) -> Result:
    """Re-rank ``project`` with ``style`` (:func:`ranked`) and, unless ``dry_run``, save it:
    highlights.json replaced and the one before kept as :data:`BACKUP`, in one commit under the
    project's run lock (``folder_busy`` while another run holds it).

    When :func:`readiness` refuses the fast path: with ``analyze_missing`` (not with
    ``dry_run``, not for a 360 project), :func:`cutpoint.analyze.reanalyze` with ``style``
    instead (the backup written before it; unchanged spans keep their ids there too); else
    :class:`~cutpoint.errors.NeedsAnalyzeError`, naming :func:`analyze_command`.

    ``expect``: the :func:`key` of the highlights.json the caller previewed (``cutpoint ui``'s
    apply); when the file under the lock has another, nothing is written. ``cuda``: as
    :func:`readiness` takes it.

    Raises:
        StaleRankingError: If highlights.json no longer has the key ``expect``.
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
        if expect is not None and key(raw) != expect:
            raise StaleRankingError(
                f"{project / HIGHLIGHTS} changed since the ranking was previewed; preview it again"
            )
        ready = readiness(project, hl, style, cuda)
        for note in ready.notes:
            logger.info("%s", note)
        if not ready.ready:
            # a 360 project is analyzed by `360 analyze` (the command the error names)
            if dry_run or not analyze_missing or _is_360(project):
                raise _refused(project, style, ready.missing)
            logger.info(
                "Re-ranking with the %s style needs analyze (%s); analyzing again",
                style.name,
                "; ".join(m.detail for m in ready.missing),
            )
            # the backup first: reanalyze replaces highlights.json when it ends; the one it
            # replaces still gives ids back (#1005)
            earlier = _backup(project)
            cache.save_bytes(project / BACKUP, raw)
            analyze.reanalyze(
                project, style, backup=earlier, rerank_backup=analyze.backup_digest(raw)
            )
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
        after = ranked(project, hl, style, raw)
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
