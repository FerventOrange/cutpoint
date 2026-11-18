"""The routes that preview, apply and undo a ranking in another style."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from cutpoint import jsonout
from cutpoint.errors import FolderBusyError
from cutpoint.styles import AUTO
from cutpoint.ui.analysis import _analysis_inputs, set_style
from cutpoint.ui.common import BUSY_TEXT, META_FILE, bad
from cutpoint.ui.files import _own, _ui_folder, read_meta
from cutpoint.ui.projects import RERANKING, project_dir
from cutpoint.ui.ranking import (
    NO_UNDO_TEXT,
    RERANK_BUSY_TEXT,
    STALE_TEXT,
    UNDO_STALE_TEXT,
    RankingRefused,
    _needs_analyze,
    _ranking_check,
    _ranking_load,
    _ranking_style,
    cuda_probe,
    ranking_preview,
    ranking_swap,
)

owned = APIRouter()


@contextlib.contextmanager
def _reranking(
    request: Request, pid: str, p: Path, names: tuple[str, ...], *, load: bool = False
) -> Iterator[tuple[Path, list[Path]]]:
    """The project ``pid`` (folder ``p``, :func:`project_dir`) marked in :data:`RERANKING`
    while its ranking is replaced (no job starts on it meanwhile): yields its folder, looked up
    again under the jobs' lock, and the list of ui folders the body creates
    (:func:`_ui_folder`'s ``made``). On the way out, also on an error, the folders still in
    that list are removed and the mark discarded.

    Before the mark, the files ``names`` must be the project's own (409 otherwise, refused now,
    not after the write) and, with ``load``, highlights.json an analysis (:func:`_ranking_load`).

    Raises:
        RankingRefused: ``folder_busy`` (:func:`_ranking_check`; a ranking or a job on the
            project).
    """
    jobs = request.app.state.jobs
    _ranking_check(p)
    for name in names:
        _own(p, name)
    if load:
        _ranking_load(p)  # 409 without an analysis (and for a highlights.json link)
    with jobs.lock:
        p = project_dir(pid)
        if p in RERANKING:
            raise RankingRefused(RERANK_BUSY_TEXT, jsonout.FOLDER_BUSY)
        if any(j.status == "running" and j.project_id == pid for j in jobs.jobs.values()):
            raise RankingRefused(BUSY_TEXT, jsonout.FOLDER_BUSY)
        RERANKING.add(p)
    made: list[Path] = []
    try:
        yield p, made
    finally:
        for folder in made:  # the ui folder this created for a ranking it did not save
            with contextlib.suppress(OSError):
                folder.rmdir()
        with jobs.lock:
            RERANKING.discard(p)


# -- ranking in another style (#902) ---------------------------------------------
@owned.post("/projects/{pid}/ranking/preview")
def preview_ranking(pid: str, body: dict[str, Any]):
    """The project re-ranked in the library style ``style`` (:func:`ranking_preview`), for
    the before/after card; writes nothing. 409 ``folder_busy`` while a run holds it."""
    p = project_dir(pid)
    try:
        return ranking_preview(p, body.get("style"))
    except RankingRefused as err:
        return err.response()


@owned.post("/projects/{pid}/ranking/apply")
def apply_ranking(request: Request, pid: str, body: dict[str, Any]):
    """Re-rank the project in the library style ``style`` and save it, after the user
    clicked to use the preview made against the ranking key ``base``: highlights.json
    replaced (the one before kept, :data:`cutpoint.rerank.BACKUP`) and ``style`` the project's
    chosen style (a detection kept when it leaves Auto). Redone, not stored: the same
    caches give the ranking the preview showed. No job starts on the project meanwhile.

    409 with ``code``: ``stale`` (highlights.json is not the previewed one any more),
    ``needs_analyze`` (with :func:`_needs_analyze`'s ``missing`` and ``can_analyze``) or
    ``folder_busy``; 400 for auto or a style not in the library."""
    from cutpoint import rerank  # noqa: PLC0415
    from cutpoint.errors import NeedsAnalyzeError, StaleRankingError  # noqa: PLC0415

    p = project_dir(pid)
    style = _ranking_style(body.get("style"))
    base = body.get("base")
    if not isinstance(base, str) or not base:
        raise bad("base must be the ranking key of the preview")
    try:
        with _reranking(request, pid, p, (rerank.BACKUP, META_FILE), load=True) as (p, made):
            meta = read_meta(p)
            # a command-line project has no ui folder (nor meta.json) yet; 409 for a ui that
            # is no folder, before the ranking is saved
            _ui_folder(p, made=made)
            try:
                result = rerank.rerank(p, style, expect=base, cuda=cuda_probe)
            except StaleRankingError:
                return RankingRefused(STALE_TEXT, jsonout.STALE).response()
            except NeedsAnalyzeError as err:
                need = _needs_analyze(style, err.missing)
                return RankingRefused(need.pop("message"), jsonout.NEEDS_ANALYZE, **need).response()
            except FolderBusyError:
                return RankingRefused(BUSY_TEXT, jsonout.FOLDER_BUSY).response()
            except FileNotFoundError:  # removed since the check above
                raise bad(
                    "This project has no analysis to rank yet; analyze it first", 409
                ) from None
            before = result.before
            # left Auto (here or in an earlier re-rank on the command line): the detection is
            # kept as the "Detected earlier" copy, as the header does
            reranked = isinstance((before.get("settings") or {}).get("reranked"), dict)
            auto = _analysis_inputs(meta, before)[1] == AUTO or reranked
            kept = before.get("detection") if auto else None
            set_style(p, style.name, kept if isinstance(kept, dict) else None)
            made.clear()  # the ui folder holds meta.json now
            new_ids, _ = rerank.changes(before, result.after)
            return {"new_ids": new_ids}
    except RankingRefused as err:
        return err.response()


@owned.post("/projects/{pid}/ranking/undo")
def undo_ranking(request: Request, pid: str, body: dict[str, Any]):
    """Undo the last ranking change, or Redo it after an Undo (:func:`ranking_swap`), from
    the page that loaded the ranking key ``base``; the style of the ranking swapped in is the
    project's chosen style again. No job starts on the project meanwhile.

    409 with ``code``: ``stale`` (highlights.json is not the loaded one any more) or
    ``folder_busy``; 409 for a 360 project and when there is no earlier ranking that
    matches this analysis."""
    from cutpoint import rerank  # noqa: PLC0415
    from cutpoint.errors import StaleRankingError  # noqa: PLC0415

    p = project_dir(pid)
    base = body.get("base")
    if not isinstance(base, str) or not base:
        raise bad("base must be the ranking key of the loaded project")
    try:
        names = (rerank.HIGHLIGHTS, rerank.BACKUP, META_FILE)
        with _reranking(request, pid, p, names) as (p, made):
            _ui_folder(p, made=made)  # a command-line project has no ui folder (meta.json) yet
            try:
                swapped = ranking_swap(p, base)
            except StaleRankingError:
                return RankingRefused(UNDO_STALE_TEXT, jsonout.STALE).response()
            except FolderBusyError:
                return RankingRefused(BUSY_TEXT, jsonout.FOLDER_BUSY).response()
            except (FileNotFoundError, ValueError):  # no analysis (any more)
                raise bad(NO_UNDO_TEXT, 409) from None
            if swapped is None:
                raise bad(NO_UNDO_TEXT, 409)
            _, after, undone = swapped
            # the style the ranking swapped in runs on (auto for an auto analysis), as the
            # header shows it and the next analysis passes it; a detection kept when not Auto
            style = _analysis_inputs({}, after)[1]
            if isinstance(style, str) and style:
                det = after.get("detection") if style != AUTO else None
                set_style(p, style, det if isinstance(det, dict) else None)
            made.clear()  # the ui folder holds meta.json now
            return {"action": "undo" if undone else "redo"}
    except RankingRefused as err:
        return err.response()
