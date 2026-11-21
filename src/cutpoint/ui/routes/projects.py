"""Project list, create, view, review, target, reveal and delete routes."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from cutpoint import cache, folders, insta360, projectdelete, runstate
from cutpoint.config import projects_dir, remap_source
from cutpoint.errors import FolderBusyError
from cutpoint.pano import x3
from cutpoint.pano.project import default_style as pano_style
from cutpoint.projectfiles import HIGHLIGHTS
from cutpoint.styles import AUTO
from cutpoint.ui.analysis import (
    ProjectStyle,
    _action_cam_style,
    _flag,
    _gaming_style,
    _style_crash,
    analyze_options,
    detected_view,
    new_project,
    preset_problem,
    preset_taken,
    refine_state,
    ui_styles,
)
from cutpoint.ui.common import BUSY_TEXT, MAX_KEPT_NAMES, UI_DIR, bad, logger
from cutpoint.ui.files import _highlights, _load, _own, _read_locked, _ui_folder, read_meta
from cutpoint.ui.picker import folder_opener
from cutpoint.ui.projects import (
    DELETING,
    RERANKING,
    TARGET_MAX_S,
    _body_inputs,
    _events_sig,
    _exclude_globs,
    _joins_of,
    _target_of,
    _usable,
    _valid_target,
    analysis_stats,
    analysis_took,
    current_export,
    forget_network,
    inputs_problem,
    inspect_source,
    moments_view,
    on_network,
    output_names,
    project_busy,
    project_dir,
    protected_sources,
    resume_check,
    run_state,
    source_folder,
    source_problem,
    validate_joins,
    validate_segments,
)
from cutpoint.ui.ranking import PlainJSON, forget_undo_view, ranking_key, ranking_undo_view
from cutpoint.ui.routes.shared import pano_views, url, window_url

owned = APIRouter()
owned_delete = APIRouter()


@owned.get("/projects")
def list_projects():
    root = projects_dir()
    found = []
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        try:
            # a command-line 360 project (pano.json, no UI meta) is not listed
            if not d.is_dir() or (
                (d / "pano.json").exists() and not (d / UI_DIR / "meta.json").is_file()
            ):
                continue
            project_dir(d.name)  # listed only if it opens: the same name, folder and link rules
        except HTTPException:
            continue
        except OSError as err:  # one unreadable entry must not fail the whole list
            logger.debug("skipping %s: %s", d.name, err)
            continue
        run = run_state(d, (d / HIGHLIGHTS).is_file())
        found.append(
            {
                "id": d.name,
                "name": read_meta(d).get("name") or d.name,
                # a run that stopped and Resume continues (#109): the list marks it
                "interrupted": bool(run and run["resumable"]),
            }
        )
    return {"projects": found}


@owned.post("/projects")
def create_project(body: dict[str, Any]):
    name, style = body.get("name"), body.get("style", AUTO)
    if not isinstance(name, str) or not name.strip() or len(name) > 100:
        raise bad("name must be 1-100 characters")
    inputs = _body_inputs(body)
    recursive, exclude = _flag(body, "recursive"), _exclude_globs(body)
    is_cam = isinstance(style, str) and _action_cam_style(style)
    if style != AUTO and style not in ui_styles() and not is_cam:
        raise bad(f"unknown style {style!r}")
    if len(inputs) > 1 or recursive or exclude:  # analyze's inputs, --recursive, --exclude
        try:
            problem = inputs_problem([Path(x) for x in inputs], exclude, recursive)
        except OSError as err:
            raise bad(f"source is not readable: {err}") from err
        if problem:
            raise bad(problem)
        full = [os.path.abspath(x) for x in inputs]  # noqa: PTH100
        extra: dict[str, Any] = {"inputs": full} if len(full) > 1 else {}
        extra |= ({"recursive": True} if recursive else {}) | (
            {"exclude": exclude} if exclude else {}
        )
        return new_project(name.strip(), full[0], style, **extra)
    source = inputs[0]
    src = Path(source)
    try:
        problem = source_problem(src)
        usable = None if problem else _usable(inspect_source(src)[1])
    except OSError as err:
        raise bad(f"source is not readable: {err}") from err
    if problem:
        raise bad(problem)
    if not usable:
        return new_project(name.strip(), os.path.abspath(source), style)  # noqa: PTH100
    # 360: the action-cam styles as given, any other the 360 default (pano.default_style)
    try:
        style = style if is_cam else pano_style()
    except ValueError as err:  # config.json's pano section is not an object
        raise bad(str(err), 409) from err
    f = insta360.parse_name(src) if src.is_file() else None  # one file: just that clip
    clips = {"clips": [x3.clip_key(f["date"], f["time"], f["num"])]} if f else {}
    full = os.path.abspath(source)  # noqa: PTH100
    return new_project(name.strip(), full, style, kind="360", **clips)


def source_views(
    request: Request, pid: str, p: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """The project's sources as the page reads them (status, proxy, windows): the listing, its
    view and the listing's error. The project view and the light sources poll share it."""
    app, media, sources = request.app, request.app.state.media, request.app.state.sources
    found = []
    listing, source_error = sources.listing(p)
    for s in listing:
        # Never probes: 200 sources took 11 s of ffprobe. The page prepares the one it plays.
        key = media.key(remap_source(s["path"]))  # one stat of the source for all three
        if key is None:  # missing: what the three say, without checking it again
            status, proxied, done = "failed", False, None
            ready, pending, failed = [], [], []
        else:
            status, proxied, done = media.cached_view(p, s["path"], key)
            ready, pending = media.windows_of(p, s["path"], key)
            failed = media.failed_windows(p, s["path"], key)
        duration = s["duration"]
        if duration is None and (ready or pending):  # the page needs it to play windows
            duration = media.probed_duration(s["path"], key)
        found.append(
            {
                "id": s["id"],
                "name": s["name"],
                "path": s["path"],
                "url": url(app, pid, "sources", s["id"]),
                "duration": duration,
                # "ready": the whole source plays (as is or from its proxy)
                "status": status,
                "proxy": proxied,  # the browser plays it from a proxy, not as is
                # a fragmented MP4 played as is opens slowly (#1014); null until probed
                "fragmented": media.fragmented.get(str(remap_source(s["path"]))),
                # share of the whole proxy encoded (0-1) while it encodes, else null
                "progress": None if done is None else round(done, 3),
                # window proxies (#323) that play while the whole proxy builds
                "windows": [
                    {"lo": lo / 100, "hi": hi / 100, "url": window_url(app, pid, s["id"], lo, hi)}
                    for lo, hi in ready
                ],
                "pending": [
                    {"lo": lo / 100, "hi": hi / 100, "state": state} for lo, hi, state in pending
                ],
                # window encodes that failed (until asked for again): the page stops waiting
                "failed": [
                    {"lo": lo / 100, "hi": hi / 100, "reason": reason} for lo, hi, reason in failed
                ],
            }
        )
    return listing, found, source_error


@owned.get("/projects/{pid}")
def get_project(request: Request, pid: str):
    app = request.app
    p = project_dir(pid)
    meta = read_meta(p)
    hl = _highlights(p)
    review = _load(p, f"{UI_DIR}/review.json")
    exp = current_export(p)
    manifest = cache.load_json(p / exp["dir"] / "manifest.json") if exp else None
    outputs = output_names(p)
    listing, found, source_error = source_views(request, pid, p)
    highlights = moments_view(hl, _load(p, "events.json"))
    last_run = run_state(p, hl is not None)
    if highlights is not None and meta.get("kind") == "360":
        # #517: flags of each moment's cached camera path and its director preview (null:
        # not a moment of a 360 clip of the project, the page plays the raw preview)
        views = pano_views(app, pid, p, listing, hl=hl)
        for c in highlights["candidates"]:
            c["pano"] = views.get(str(c.get("cli_id")))
    ps = ProjectStyle.of(meta, hl)
    view = {
        "id": pid,
        "name": meta.get("name") or pid,
        "style": ps.chosen or "",
        # the page's Color starts at Off for gaming footage, as config.json's color does
        "gaming": _gaming_style(ps.resolved, ps),
        # the header's style button (#897): the detected category and the chosen style
        "detected": detected_view(meta, hl),
        "kind": "360" if meta.get("kind") == "360" else "video",
        "source": meta.get("source"),
        "highlights": highlights,
        # the project menu's "Analyzed X of video in Y"
        "analysis": analysis_stats(hl, analysis_took(p, meta, hl)),
        "events_sig": _events_sig(p),
        "review": review if isinstance(review, dict) else None,
        "clips": {**manifest, "dir": exp["dir"]} if isinstance(manifest, dict) and exp else None,
        "outputs": [{"name": n, "url": url(app, pid, "outputs", n)} for n in outputs],
        # The first listed final is the current export's own (or, with no export, the
        # newest); false while that export is not assembled yet and an earlier final leads.
        # One-click presets (edit --kind): why they are off, else null; the folders whose
        # earlier clips a run would replace (the page asks first).
        "preset_problem": preset_problem(meta, hl),
        # Analyze options (#483): what the analysis used, which the page shows; the style's
        # --must-include-crash default, null for 360 projects or no known style (no option)
        "analyze_options": analyze_options(p, meta, hl),
        "must_include_crash_default": None
        if meta.get("kind") == "360"
        else _style_crash(ps.reranked or meta.get("style") or ps.recorded, ps),
        # "Describe the footage" (crashes --context) and "Look closer with vision" (events)
        "refine": refine_state(p, meta, hl),
        "preset_taken": preset_taken(p),
        "final_current": exp is None or (bool(outputs) and outputs[0] == exp.get("final")),
        # The transition each join of that final came out with, for the page's "Assemble
        # again" note; null when not known, as for a One-Click Edit's final.
        "final_joins": (exp or {}).get("final_joins"),
        "sources": found,
        # The picked folder could not be listed (gone, unreadable): a field, so Settings
        # (source remap), Reveal and Delete still work.
        "source_error": source_error,
        # the project folder is likely on a network drive: the page names it when the
        # player buffers long (#1014)
        "network": on_network(p),
        # The last recorded run (progress.json, #109): Resume when it stopped, null without one
        "run": last_run,
        # a run holds the project now (this server's own job too): the page says to wait
        "busy": project_busy(p, last_run),
        # the key of highlights.json a ranking preview is applied against (#902)
        "ranking_key": ranking_key(p),
        # Undo (or Redo) of the last ranking change: {action, style}, null when the backup
        # does not match the current analysis (ranking_undo_view)
        "ranking_undo": ranking_undo_view(p, meta),
    }
    return PlainJSON(view)


@owned.get("/projects/{pid}/sources")
def get_sources(request: Request, pid: str):
    """The ``sources`` of the project view alone, for the page's proxy poll (A754): the whole
    view carries every moment and its transcript."""
    _, found, _ = source_views(request, pid, project_dir(pid))
    return PlainJSON({"sources": found})


@owned.get("/projects/{pid}/resume")
def resume_preview(pid: str):
    """What Resume would continue (``cutpoint resume --dry-run``), for the page to confirm:
    the command, how its last run ended, its steps' counts and the CLI's warnings."""
    p = project_dir(pid)
    result = resume_check(p)
    rec = result["outputs"].get("resume")
    rec = rec if isinstance(rec, dict) else {}
    steps, argv = rec.get("done_before"), rec.get("argv")
    return {
        "command": result.get("command"),
        # the command line Resume replays (a recorded --force or --out too), shown to confirm
        "argv": argv if isinstance(argv, list) and all(isinstance(a, str) for a in argv) else [],
        "status": rec.get("status_before"),
        "steps": steps if isinstance(steps, dict) else {},
        "warnings": [
            w["message"]
            for w in result.get("warnings") or []
            if isinstance(w, dict) and isinstance(w.get("message"), str)
        ],
    }


@owned.put("/projects/{pid}/review")
def put_review(request: Request, pid: str, body: dict[str, Any]):
    media, review_lock, sources = (
        request.app.state.media,
        request.app.state.review_lock,
        request.app.state.sources,
    )
    p = project_dir(pid)
    segments = validate_segments(body.get("segments"), sources.entries(p), media)
    joins = validate_joins(body["joins"]) if "joins" in body else None
    review = {"segments": segments, "updated": datetime.now().isoformat(timespec="seconds")}
    file = _own(p, f"{UI_DIR}/review.json")
    _ui_folder(p)  # a command-line project has no ui folder yet
    with review_lock:
        # the sequence's target length is saved by its own route: Save keeps it; a short
        # lock must not drop it, and a file that cannot be read is never written over
        try:
            saved = cache.retry_locked(lambda: _read_locked(file), file, "read")
        except FileNotFoundError:
            saved = None
        except (ValueError, OSError) as err:
            # the reason, naming the file as the page does (never its absolute path)
            if isinstance(err, OSError) and err.strerror:
                why = f"ui/review.json cannot be read: {err.strerror}"
            else:  # read_json's ValueError, retry_locked's PermissionError on Windows
                why = str(err).replace(str(file), "ui/review.json")
            raise bad(
                f"{why.rstrip('.')}. Fix or delete ui/review.json (it keeps the "
                "sequence's target length and joins), then save again; the sequence was "
                "not saved",
                409,
            ) from err
        kept = _target_of(saved)
        if kept is not None:
            review["target"] = kept
        # a page that sends no joins (the AI Chat's Apply included) keeps them too
        joins = _joins_of(saved) if joins is None else joins
        if joins:
            review["joins"] = joins
        cache.save_json(file, review)
    return review


@owned.put("/projects/{pid}/target")
def put_target(request: Request, pid: str, body: dict[str, Any]):
    """The sequence's target length in ``seconds`` (null clears it), kept in ``review.json``
    beside the segments; only a guide for the page's progress bar."""
    review_lock = request.app.state.review_lock
    p = project_dir(pid)
    if "seconds" not in body:
        raise bad("seconds is required (a number, or null to clear)")
    secs = body["seconds"]
    if secs is not None and not _valid_target(secs):
        raise bad(f"seconds must be a number above 0 and at most {TARGET_MAX_S}, or null")
    file = _own(p, f"{UI_DIR}/review.json")
    with review_lock:
        try:
            review = cache.read_json(file)
        except FileNotFoundError:
            review = {}
        except (ValueError, OSError):
            review = None
        if not isinstance(review, dict):  # never write over a sequence we could not read
            raise bad("ui/review.json could not be read; the target was not saved", 409)
        review = dict(review)
        if secs is None:
            if "target" not in review:
                return {"target": None}
            review.pop("target")
        else:
            review["target"] = float(secs)
        review.setdefault("segments", [])
        _ui_folder(p)  # a command-line project has no ui folder yet
        cache.save_json(file, review)
    return {"target": review.get("target")}


@owned.post("/projects/{pid}/reveal")
def reveal_folder(pid: str, target: str = "sources"):
    """Show a project folder in the file manager: its videos, latest clips, final or the
    project folder itself (path from the project; ``target`` only picks which)."""
    p = project_dir(pid)
    if target == "sources":
        folder = source_folder(p)
        if folder is None:
            raise bad("the folder with this project's videos no longer exists", 404)
    elif target == "clips":
        exp = current_export(p)
        if exp is None:
            raise bad("nothing has been exported yet", 404)
        folder = p / exp["dir"]
    elif target == "outputs":
        if not output_names(p):
            raise bad("nothing has been assembled yet", 404)
        folder = p
    elif target == "project":  # AI Chat's link next to a reply naming a project file
        folder = p
    else:
        raise bad("unknown folder")
    opener = folder_opener()
    if opener is None:
        logger.info("cannot open %s: no file manager is available here", folder)
        raise bad("Opening folders isn't available on this machine", 501)
    try:
        opener(folder)
    except OSError as err:
        logger.warning("opening %s failed: %s", folder, err)
        raise bad(f"could not open the folder: {err.strerror or err}", 500) from err
    return {"path": str(folder)}


# -- delete a project (Settings) ---------------------------------------------------
def delete_blocker(app: FastAPI, pid: str, p: Path) -> str | None:
    """Why ``p`` cannot be deleted now (a job, a run, an encode or a chat turn on it)."""
    jobs, media, previews, turns = (
        app.state.jobs,
        app.state.media,
        app.state.previews,
        app.state.assistant,
    )
    if any(j.status == "running" and j.project_id == pid for j in list(jobs.jobs.values())):
        return "A job is running on this project; wait for it to finish or cancel it"
    if project_busy(p, run_state(p, True)) or projectdelete.locked(p):
        return "A Cutpoint run holds this project; wait for it to finish"
    if p in RERANKING:
        return "A ranking is being applied to this project; wait a moment"
    with previews.lock:
        lane = previews.lane
        if any(v is not None and v.project == p for v in (lane.running, *lane.queue)):
            return "A 360 preview is rendering for this project; wait for it to finish"
    if media.encoding_for(p) or any(t.status == "running" for t in turns.of(str(p))):
        return "The project is still preparing videos or answering in AI Chat; wait a moment"
    return None


def forget_project(app: FastAPI, p: Path) -> None:
    """Drop everything this run keeps per project for the folder ``p`` (it was deleted), so a
    project made later under the same name starts clean: the media lookups and source lists,
    proxies and 360 previews, the AI Chat's turns and edge planner, the remembered Undo view
    and the network-drive check. Its running chat turn, if any, is cancelled."""
    st = app.state
    st.lookups.forget(p)
    st.sources.forget(p)
    st.media.forget(p)
    st.previews.forget(p)
    st.planners.pop(p, None)
    st.assistant.clear(str(p))
    forget_undo_view(p)
    forget_network(p)


@owned_delete.get("/projects/{pid}/delete-plan")
def delete_plan(pid: str):
    """What deleting the project removes (``size`` bytes in ``files`` files; ``deleted``, the
    finals and timelines among them) and the names of what stays (``kept``); each list holds
    at most :data:`MAX_KEPT_NAMES` names of its ``*_count``."""
    p = project_dir(pid)
    plan = projectdelete.make_plan(p, protected_sources())
    return {
        "name": read_meta(p).get("name") or pid,
        "size": plan.size,
        "files": plan.files,
        "deleted": plan.deleted[:MAX_KEPT_NAMES],
        "deleted_count": len(plan.deleted),
        "kept": plan.kept[:MAX_KEPT_NAMES],
        "kept_count": len(plan.kept),
    }


@owned_delete.delete("/projects/{pid}")
def delete_project(request: Request, pid: str, body: dict[str, Any] | None = None):
    """Delete the project's own files (:mod:`cutpoint.projectdelete`) to the Recycle Bin /
    Trash, or for good with ``permanent``. 409 with ``code`` ``no_trash`` when the trash
    cannot be trusted with them, for the page to ask before it sends ``permanent``."""
    app, jobs, lookups = request.app, request.app.state.jobs, request.app.state.lookups
    permanent = _flag(body or {}, "permanent")
    project_dir(pid)
    protected = protected_sources()  # before the lock: a slow path must not hold it
    # The mark and the check run under the jobs lock, the move (it can take long) after it:
    # once marked, project_dir answers 409 and start_job refuses a request that resolved the
    # folder before, so nothing starts on the project while its files go.
    with jobs.lock:
        p = project_dir(pid)  # again: a delete that ran first may have removed it
        DELETING.add(p)  # every other route on the project answers 409 meanwhile
    try:
        with jobs.lock:
            if why := delete_blocker(app, pid, p):
                raise bad(why, 409)
        plan = projectdelete.make_plan(p, protected)
        try:
            with folders.swap_lock(p / runstate.LOCK_NAME, p):
                projectdelete.remove_items(plan, permanent=permanent)
        except FolderBusyError as err:
            raise bad(BUSY_TEXT, 409) from err
        except projectdelete.NoTrash as err:
            logger.warning("could not move %s to the trash: %s", p, err)
            return JSONResponse(
                {
                    "detail": f"The Recycle Bin / Trash cannot be used: {err}.",
                    "code": "no_trash",
                },
                status_code=409,
            )
        except PermissionError as err:
            logger.warning("deleting %s failed: %s", p, err)
            f = err.filename
            name = (
                Path(f[0] if isinstance(f, (list, tuple)) and f else str(f)).name if f else "a file"
            )
            raise bad(f"A file is in use ({name}); close it and try again.", 409) from err
        except OSError as err:
            logger.warning("deleting %s failed: %s", p, err)
            raise bad(f"could not delete the project: {err.strerror or err}", 500) from err
        projectdelete.remove_folder(p)
    finally:
        lookups.forget(p)  # before the 409 ends: the media routes check it again
        with jobs.lock:
            DELETING.discard(p)
    forget_project(app, p)
    return {"deleted": pid, "kept": len(plan.kept)}
