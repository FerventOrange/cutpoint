"""Re-rank an analysed project with another style from its caches, without analyze (#902).

``cutpoint rerank <project> --style NAME`` rebuilds the grids of every source from the caches
of the analysis (loudness, transcript, scene cuts: :func:`cutpoint.analyze.cached_grid_parts`),
picks the style's candidates from them and runs the ranking tail of an analysis
(:func:`cutpoint.analyze.rank_parts`: crash moments, the events' bonus stage, cached vision
ratings only, the final order). It takes seconds, where analyze takes minutes to hours.

:func:`readiness` says whether the caches hold what the style needs; when they do not, the
re-rank is refused (:class:`~cutpoint.errors.NeedsAnalyzeError`, ``needs_analyze``) and
``--analyze`` runs :func:`cutpoint.reanalyze.reanalyze` with the style instead.

The preview (``--dry-run``) is the re-rank itself, written nowhere: the save writes the same
ranking. Saving replaces ``highlights.json`` and keeps the one before as
:data:`~cutpoint.projectfiles.RERANK_BACKUP` (one Undo). Candidates whose span is unchanged keep
their ids, a span only the backup has gets its id there (#1005: a switch of style and back
restores the ids), and new spans get ids above ``settings.last_candidate_id``. A saved ranking
is sealed (``settings.rerank_undo``, :func:`seal`): ``cutpoint ui`` offers its Undo while
highlights.json is still the file the save wrote.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cutpoint import (
    analyze,
    cache,
    crash_moments,
    features,
    folders,
    progress,
    reanalyze,
    runstate,
    selfcheck,
)
from cutpoint.config import projects_dir, remap_source
from cutpoint.errors import NeedsAnalyzeError, StaleRankingError
from cutpoint.ffmpeg_utils import probe
from cutpoint.pool_record import is_pool
from cutpoint.projectfiles import EVENTS_JSON, HIGHLIGHTS, RERANK_BACKUP, check_candidates
from cutpoint.scenes import scores_scenes
from cutpoint.styles import ACTION_CAM, Style, style_hash
from cutpoint.utils import command_line

logger = logging.getLogger("cutpoint")

GENERIC = "generic"  # the style kind of every style but action-cam
# settings key of a ranking a save (or cutpoint ui's Undo) wrote: {"seal": seal(hl), "undone": bool}
UNDO = "rerank_undo"

CHAT_ONLY_LAYER = "visual_light"  # planned by analyze without a CUDA GPU; ranks nothing


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
    """The bytes of the project's :data:`RERANK_BACKUP` (the ranking before the last
    re-rank), or None without one: :func:`cutpoint.reanalyze.earlier_ids` uses it only when
    highlights.json is linked to it (``settings.rerank_backup``)."""
    return cache.load_bytes(project / RERANK_BACKUP)


def load(project: Path) -> tuple[bytes, dict[str, Any]]:
    """The bytes and the object of ``<project>/highlights.json``.

    Raises:
        FileNotFoundError: If the project has no highlights.json.
        ValueError: If it is not valid JSON, records no analysis or has a candidate that is
            not an object.
    """
    path = project / HIGHLIGHTS
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise FileNotFoundError(f"{path} not found; run 'cutpoint analyze' first") from None
    hl = cache.read_json(path, hint="; re-run `cutpoint analyze` to rewrite it")
    if not isinstance(hl, dict) or not isinstance(hl.get("sources"), list):
        raise ValueError(f"{path} records no analysis; run 'cutpoint analyze' again")
    check_candidates(hl, path)
    return raw, hl


def backup_bytes(raw: bytes) -> bytes:
    """What :data:`RERANK_BACKUP` keeps of the highlights.json bytes ``raw``: ``raw`` itself
    (so ``settings.rerank_backup`` names it byte for byte), or, when an older file kept an Ollama
    login in ``settings.ollama_url``, the file without it (A639)."""
    from cutpoint.backends import saved_settings, without_login  # noqa: PLC0415 (requests: off the CLI import)

    try:
        data = json.loads(raw)
    except ValueError:
        return raw
    st = data.get("settings") if isinstance(data, dict) else None
    url = st.get("ollama_url") if isinstance(st, dict) else None
    if not isinstance(url, str) or without_login(url) == url:
        return raw
    return json.dumps(saved_settings(data), indent=2, ensure_ascii=False).encode("utf-8")


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
    return reanalyze.backup_digest(text.encode("utf-8"))


def sealed(hl: dict[str, Any], *, undone: bool = False) -> dict[str, Any]:
    """``hl`` with its ``settings.rerank_undo`` (:data:`UNDO`) set: its :func:`seal` and whether
    it is the ranking an Undo went back to (``undone``: the next swap is a Redo)."""
    hl.setdefault("settings", {})[UNDO] = {"seal": seal(hl), "undone": undone}
    return hl


def _inside(project: Path, name: str) -> bool:
    """``project / name`` still lies inside ``project`` once links are resolved."""
    try:
        return (project / name).resolve().is_relative_to(project.resolve())
    except (OSError, RuntimeError):  # RuntimeError: a symlink loop (Python 3.12)
        return False


def _source_paths(hl: dict[str, Any]) -> list[str]:
    return [str(s.get("path")) for s in hl.get("sources") or [] if isinstance(s, dict)]


def undo_target(project: Path, hl: dict[str, Any]) -> tuple[dict[str, Any], bool] | None:
    """The ranking an Undo (or Redo) of ``project`` swaps in, given its highlights.json ``hl``:
    the backup's object and whether ``hl`` is the ranking an Undo went back to (the swap is a
    Redo then). None when nothing matches (no Undo):

    - ``hl`` is sealed (``settings.rerank_undo``, :func:`seal`): no analysis, ``events`` run or
      hand edit rewrote highlights.json since a re-rank save or a swap wrote it (a
      re-analysis keeps ``settings.rerank_backup``, but not the seal);
    - :data:`RERANK_BACKUP` is in the project (not a link out of it) and its key is
      ``settings.rerank_backup``: the very file that ranking replaced, unchanged;
    - the backup is an analysis of the same sources.
    """
    st = hl.get("settings")
    mark = st.get(UNDO) if isinstance(st, dict) else None
    link = st.get("rerank_backup") if isinstance(st, dict) else None
    if (
        not isinstance(mark, dict)
        or not isinstance(link, str)
        or not _inside(project, RERANK_BACKUP)
    ):
        return None
    back_raw = cache.load_bytes(project / RERANK_BACKUP)
    if (
        back_raw is None
        or reanalyze.backup_digest(back_raw) != link
        or mark.get("seal") != seal(hl)
    ):
        return None
    try:
        back = json.loads(back_raw.decode("utf-8-sig"))
    except ValueError:  # UnicodeDecodeError too
        return None
    if (
        not isinstance(back, dict)
        or "style" not in back
        or not isinstance(back.get("sources"), list)
        or not isinstance(back.get("settings", {}), dict)
        or _source_paths(back) != _source_paths(hl)
    ):
        return None
    return back, mark.get("undone") is True


def _swap_files(project: Path, restored: bytes, kept: bytes, current: bytes) -> None:
    """Swap the project's rankings: highlights.json gets ``restored`` (made from the backup)
    and :data:`RERANK_BACKUP` ``kept`` (:func:`backup_bytes` of ``current``, highlights.json's
    bytes now). Both are staged as temp files first, then renamed: highlights.json first, while the
    backup still holds the ranking it gets. A failed second rename puts ``current`` back into
    highlights.json; should that fail too, the temp file holding ``kept`` is kept (its path
    logged), as one a hard kill leaves.
    Durable as ``cache.save_bytes(..., durable=True)`` (A667): each temp file is flushed to
    disk before its rename and, on POSIX, the folder after the renames."""
    hl_path, back_path = project / HIGHLIGHTS, project / RERANK_BACKUP
    staged: list[Path] = []
    keep = False
    try:
        for path, data in ((hl_path, restored), (back_path, kept)):
            fd, name = tempfile.mkstemp(dir=project, prefix=path.name + ".", suffix=".tmp")
            os.close(fd)
            staged.append(Path(name))
            staged[-1].write_bytes(data)
            with staged[-1].open("r+b") as f:  # as cache.save_bytes(durable=True)
                os.fsync(f.fileno())
            cache.match_mode(staged[-1], path)
        cache.retry_locked(lambda: staged[0].replace(hl_path), hl_path, "replace")
        try:
            cache.retry_locked(lambda: staged[1].replace(back_path), back_path, "replace")
        except BaseException:
            try:
                cache.save_bytes(hl_path, current, durable=True)  # as it was: all or nothing
            except BaseException:
                keep = True  # the only copy of the current ranking now
                logger.error("The ranking before the swap is kept in %s", staged[1])
                raise
            raise
        cache._fsync_dir(project)
    finally:
        for tmp in staged if not keep else ():
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)


def swap(project: Path, base: str) -> tuple[dict[str, Any], dict[str, Any], bool] | None:
    """Undo (or Redo) the last ranking change of ``project``, under its run lock: swap
    highlights.json and :data:`RERANK_BACKUP` (:func:`undo_target`). The ranking swapped in
    keeps its moments and ids as they are; only its settings change: ``rerank_backup`` links it
    to the file it replaced, ``last_candidate_id`` is the higher of the two (never down), and it
    is sealed again (``undone`` flips, so the next swap is the other way). Returns the ranking
    before, the one now and whether this was an Undo; None when there is nothing to swap in
    (:func:`undo_target` is None).

    Raises:
        StaleRankingError: If highlights.json's key is not ``base`` any more.
        FolderBusyError: If another run holds the project's lock.
        FileNotFoundError, ValueError: As :func:`load`.
    """
    from cutpoint import backends  # noqa: PLC0415 (requests: off the CLI import)

    with folders.swap_lock(project / runstate.LOCK_NAME, project):
        raw, hl = load(project)
        if reanalyze.backup_digest(raw) != base:
            raise StaleRankingError(f"{project / HIGHLIGHTS} changed since the page loaded")
        found = undo_target(project, hl)
        if found is None:
            return None
        back, undone = found
        # neither file keeps an Ollama login an older one held (A639): the ranking swapped in
        # as every highlights.json is written, the one it replaces as a re-rank keeps it
        backends.saved_settings(back)
        kept = backup_bytes(raw)
        st = back.setdefault("settings", {})
        st["rerank_backup"] = reanalyze.backup_digest(kept)
        last = [(x.get("settings") or {}).get("last_candidate_id") for x in (hl, back)]
        if ints := [n for n in last if isinstance(n, int) and not isinstance(n, bool)]:
            st["last_candidate_id"] = max(ints)
        sealed(back, undone=not undone)
        _swap_files(project, cache.json_bytes(back), kept, raw)
        return hl, back, not undone


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
    ev = cache.load_json(project / EVENTS_JSON)
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
    if not _events_extra():
        return []  # lite: analyze would run none of them either
    # with a CUDA GPU assumed the plan needs no torch check; when the project has all of its
    # layers (it asks for no more than the real plan), the real plan is not needed
    best = analyze.plan_layers(
        style, [], (), cat, cuda=lambda: True, quiet=True
    )  # no sources: no probe
    if all(x in ran or x == CHAT_ONLY_LAYER for x in best.all_layers()):
        return []
    plan = analyze.plan_layers(style, [], (), cat, cuda=cuda, quiet=True)
    # the light frame index is the AI Chat's alone: no ranking reads it (#989)
    return [x for x in plan.all_layers() if x not in ran and x != CHAT_ONLY_LAYER]


def readiness(
    project: Path, hl: dict[str, Any], style: Style, cuda: Callable[[], bool] | None = None
) -> Readiness:
    """Whether the analysis ``hl`` of ``project`` can be re-ranked with ``style`` from its
    caches alone, as ``analyze`` with ``style`` would rank it.

    Refused (each a :class:`Missing`): a 360 project; an action-cam ``style`` (its feature grid
    is built by analyze) or another style kind than the project's; a source read in parts
    (``offset``, ``read_errors``) or not reachable; a source with audio but no cached loudness
    or transcript; a source without a full scene pass (:data:`~cutpoint.scenes.SCENES_FILE`)
    when scene detection is on and ``style`` weighs scenes, or the source is too long for
    edge windows (:func:`~cutpoint.scenes.edge_windows`); a source with nothing cached to
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
    scenes = bool((hl.get("settings") or {}).get("scenes", True))
    entries = hl.get("sources") or []
    seen: set[str] = set()
    # the gaps (recovered, offline, length, caches) as cached_grid_parts sees them
    for c in analyze.cached_sources(hl, style, create=False):
        path = str(entries[c.index].get("path"))
        if path in seen:  # a recovered source's other parts
            continue
        seen.add(path)
        name = c.source.name
        if c.gap in ("recovered", "offline"):
            miss(Missing(c.gap, _gap_detail(c.gap, name), path))
            continue
        if not (c.loudness and c.transcript) and _has_audio(c.source):
            if not c.loudness:
                miss(Missing("loudness", f"{name} has no cached loudness", path))
            if not c.transcript:
                miss(Missing("transcript", f"{name} has no cached transcript", path))
        if c.gap == "length":
            miss(Missing(c.gap, _gap_detail(c.gap, name), path))
            continue
        if scenes and not c.scenes and (scores_scenes(style) or not c.windowed):
            miss(Missing("scenes", f"{name} has no full scene pass", path))
            continue
        if c.gap is not None:
            miss(Missing(c.gap, _gap_detail(c.gap, name), path))
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


def _gap_detail(gap: str, name: str) -> str:
    """The ERROR line's words for a source of ``name`` that gets no grid for ``gap`` (a
    :class:`~cutpoint.analyze.CachedSource` gap)."""
    return {
        "recovered": f"{name} was read in parts (read errors)",
        "offline": f"{name} is not reachable",
        "length": f"{name} records no length (duration) in highlights.json",
        "caches": (
            f"{name} has no cached loudness, transcript or scene cuts to build a ranking from"
        ),
    }[gap]


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
    project: Path,
    hl: dict[str, Any],
    style: Style,
    raw: bytes | None = None,
    *,
    check: bool = True,
) -> dict[str, Any]:
    """The analysis ``hl`` of ``project`` re-ranked with ``style`` from its caches (see
    :func:`readiness`), as a new highlights.json object; writes nothing to the project.

    The style's fields and settings replace the earlier style's (its ``--must-include-crash``
    override kept, :func:`~cutpoint.reanalyze.restyled`); another style is the user's choice
    (``settings.style_requested``, ``settings.reranked``; an auto analysis's ``detection``
    stays). Vision takes cached ratings only. ``timings`` keep the analysis's, with this
    re-rank's ``rerank_s``. ``settings.rerank_backup`` links it to the backup a save keeps:
    :func:`~cutpoint.reanalyze.backup_digest` of ``raw``, the bytes of the project's
    highlights.json (read when None); ``settings.rerank_undo`` seals it (:func:`sealed`).
    ``check``: run the analysis self-checks (:mod:`cutpoint.selfcheck`; they warn and record
    ``settings.self_checks``).
    """
    t0 = time.perf_counter()
    style = reanalyze.restyled(hl, style)
    new = copy.deepcopy({k: v for k, v in hl.items() if k != "candidates"})
    new.update(style=style.name, style_kind=style.kind, style_base=style.base)
    st = new.setdefault("settings", {})
    reranked = reanalyze.rerank_record(hl, style)
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
    ids, st["last_candidate_id"] = reanalyze.earlier_ids(hl, _backup(project))
    if raw is None:
        raw = (project / HIGHLIGHTS).read_bytes()
    st["rerank_backup"] = reanalyze.backup_digest(raw)  # the backup a save writes
    timings = new.setdefault("timings", {})
    timings.pop("vision_s", None)
    grids = {p.index: p for p in analyze.cached_grid_parts(new, style)}
    parts: list[crash_moments.HighlightsPart] = []
    by_source: dict[str, list[str]] = {}
    for i, e in enumerate(new.get("sources") or []):
        g = grids.get(i)
        if g is None:  # never drop its candidates quietly: the gap readiness names, else
            # (cache files gone since, or there but unreadable) nothing cached to build from
            path = str(e["path"])
            gap = analyze.cached_sources(new, style, create=False)[i].gap or "caches"
            raise _refused(project, style, [Missing(gap, _gap_detail(gap, Path(path).name), path)])
        picked = analyze.pick_candidates(g, style, st.get("max_candidates"), {})
        parts.append(
            crash_moments.HighlightsPart(g.source, g.offset, g.duration, picked, g.signals)
        )
        by_source[str(g.source)] = sorted(features.available() if g.avail is None else g.avail)
    st["features_by_source"] = by_source
    total = (hl.get("timings") or {}).get("total_s")
    with progress.run(progress.ANALYZE_STEPS, use=False) as steps:
        steps.start("ranking")
        ev = analyze.rank_parts(
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
    if check:  # a UI preview records and warns about nothing
        selfcheck.apply(new, ev)  # warns; changes no score (#1073)
    timings["rerank_s"] = round(time.perf_counter() - t0, 2)
    from cutpoint.backends import saved_settings  # noqa: PLC0415 (requests: off the CLI import)

    # an older file's Ollama login, which a save drops, would leave the seal unmatched (A639)
    saved_settings(new)
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
    highlights.json replaced and the one before kept as :data:`RERANK_BACKUP`, in one commit
    under the project's run lock (``folder_busy`` while another run holds it).

    When :func:`readiness` refuses the fast path: with ``analyze_missing`` (not with
    ``dry_run``, not for a 360 project), :func:`cutpoint.reanalyze.reanalyze` with ``style``
    instead (the backup written before it; unchanged spans keep their ids there too); else
    :class:`~cutpoint.errors.NeedsAnalyzeError`, naming :func:`analyze_command`.

    ``expect``: the :func:`~cutpoint.reanalyze.backup_digest` of the highlights.json the caller
    previewed (``cutpoint ui``'s apply); when the file under the lock has another, nothing is
    written. ``cuda``: as :func:`readiness` takes it.

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
        if expect is not None and reanalyze.backup_digest(raw) != expect:
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
            kept = backup_bytes(raw)
            cache.save_bytes(project / RERANK_BACKUP, kept, durable=True)
            try:
                reanalyze.reanalyze(
                    project, style, backup=earlier, rerank_backup=reanalyze.backup_digest(kept)
                )
            except BaseException:
                # a failed or interrupted analysis keeps the earlier Undo, unless it already
                # replaced highlights.json (its rerank_backup then points at the new RERANK_BACKUP)
                if cache.load_bytes(project / HIGHLIGHTS) == raw:
                    if earlier is None:
                        (project / RERANK_BACKUP).unlink(missing_ok=True)
                    else:
                        cache.save_bytes(project / RERANK_BACKUP, earlier, durable=True)
                raise
            _, after = load(project)
            return Result(
                project,
                hl,
                after,
                ready.notes,
                saved=True,
                analyzed=True,
                backup=project / RERANK_BACKUP,
                seconds=round(time.perf_counter() - t0, 2),
            )
        from cutpoint import backends  # noqa: PLC0415 (requests: off the CLI import)

        kept = backup_bytes(raw)
        after = ranked(project, hl, style, kept)
        result = Result(project, hl, after, ready.notes)
        if not dry_run:
            with cache.hold_outputs(project, (HIGHLIGHTS, RERANK_BACKUP)) as held:
                cache.save_bytes(project / RERANK_BACKUP, kept, durable=True)
                cache.save_json(project / HIGHLIGHTS, backends.saved_settings(after), durable=True)
                held.commit()
            result.saved, result.backup = True, project / RERANK_BACKUP
            logger.info(
                "Re-ranked %s with the %s style; the earlier ranking is in %s",
                project / HIGHLIGHTS,
                style.name,
                RERANK_BACKUP,
            )
        result.seconds = round(time.perf_counter() - t0, 2)
        return result


def _ranks(hl: dict[str, Any]) -> dict[str, int]:
    """The rank (1 = first) of each scored candidate's id in ``hl`` (its file order)."""
    cands = [c for c in hl.get("candidates") or [] if not is_pool(c) and "id" in c]
    return {str(c["id"]): i for i, c in enumerate(cands, 1)}


def rows(hl: dict[str, Any], top: int, other: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The first ``top`` scored candidates of ``hl``: ``rank``, ``id``, ``source``, ``start``,
    ``end``, ``score``, ``label`` and, with ``other`` (the ranking compared with), ``was``: the
    id's rank there, or None for a span it does not have."""
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
