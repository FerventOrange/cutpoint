"""Source media, frames, envelopes, events, director and output routes."""

from __future__ import annotations

import contextlib
import math
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from cutpoint.errors import FFmpegError
from cutpoint.ui.common import FRAME_MAX_W, FRAME_MIN_W, FRAME_WIDTH, UI_DIR, bad, logger
from cutpoint.ui.files import _highlights, read_meta
from cutpoint.ui.media import _mark_used
from cutpoint.ui.projects import _rail_events, output_names, project_dir
from cutpoint.ui.routes.shared import pano_views


def send(
    file: Path,
    request: Request,
    name: str | None = None,
    *,
    proxy: bool = False,
    stat_result: os.stat_result | None = None,
) -> FileResponse:
    """Range-capable file response; ``?download=1`` makes the browser save it.

    ``proxy``: a proxy, marked used (see :func:`_mark_used`), with an ETag from its name
    (which holds its key) and size, so touching it does not turn a seek's ranged request
    (If-Range) into a whole response. ``stat_result``: the file's stat, taken by the caller."""
    download = request.query_params.get("download") == "1"
    # mimetypes knows .srt only where the OS registry or mime.types lists it
    kind = "text/plain" if file.suffix.lower() == ".srt" else mimetypes.guess_type(file.name)[0]
    headers = None
    if proxy:
        _mark_used(file)
        with contextlib.suppress(OSError):
            headers = {"etag": f'"{file.stem}-{file.stat().st_size}"'}
    return FileResponse(
        file,
        media_type=kind or "application/octet-stream",
        headers=headers,
        filename=name or file.name if download else None,
        content_disposition_type="attachment" if download else "inline",
        stat_result=stat_result,
    )


def pano_project_dir(pid: str) -> Path:
    p = project_dir(pid)
    if read_meta(p).get("kind") != "360":
        raise bad("not a 360 project", 404)
    return p


owned = APIRouter()
served = APIRouter()


def find_source(app: FastAPI, pid: str, sid: str) -> tuple[Path, dict[str, Any]]:
    sources = app.state.sources
    p = project_dir(pid)
    entry = next((s for s in sources.entries(p) if s["id"] == sid), None)
    if entry is None:
        raise bad("unknown source", 404)
    return p, entry


@served.api_route("/projects/{pid}/sources/{sid}", methods=["GET", "HEAD"])
def source_media(pid: str, sid: str, request: Request):
    lookups, media = request.app.state.lookups, request.app.state.media
    p, entry = lookups.find(pid, sid)  # one Range request of many (#1014)
    status, file = media.status(p, entry["path"], lookups.content_key(pid, entry["path"]))
    if file is None:
        raise bad(f"source is {status}", 404 if status == "failed" else 409)
    is_proxy = file.parent == p / UI_DIR / "proxy"
    st = None
    if not is_proxy:  # the source itself: gone since the kept key is a 404, not a 500
        try:
            st = file.stat()
        except OSError:
            raise bad("source is failed", 404) from None
    # a proxy is marked used; never the source itself
    return send(file, request, entry["name"], proxy=is_proxy, stat_result=st)


@served.api_route("/projects/{pid}/sources/{sid}/window/{span}", methods=["GET", "HEAD"])
def source_window(pid: str, sid: str, span: str, request: Request):
    """A ready window proxy of a source; ``span`` is ``<lo>-<hi>`` in centiseconds."""
    lookups, media = request.app.state.lookups, request.app.state.media
    m = re.fullmatch(r"(\d{1,9})-(\d{1,9})", span)
    if m is None:
        raise bad("unknown window", 404)
    p, entry = lookups.find(pid, sid)
    key = lookups.content_key(pid, entry["path"])  # no stat per request (#1030)
    file = media.window(p, entry["path"], int(m[1]), int(m[2]), key)
    if file is None:
        status = media.status(p, entry["path"], key)[0]
        raise bad(f"window is not ready (source is {status})", 404 if status == "failed" else 409)
    return send(file, request, proxy=True)


@served.get("/projects/{pid}/sources/{sid}/frame")
def source_frame(
    request: Request, pid: str, sid: str, at: float = 0.0, w: int = FRAME_WIDTH, key: bool = False
):
    """One still of a source (``at`` seconds in, ``w`` px wide) for thumbnails; ``key=1``:
    the keyframe at or before ``at`` (the rail's filmstrip)."""
    app, media, sources = request.app, request.app.state.media, request.app.state.sources
    if not math.isfinite(at) or at < 0:
        raise bad("at must be a time in seconds, 0 or more")
    if media.closing:
        raise bad("the server is stopping", 503)
    p, entry = find_source(app, pid, sid)
    if p not in media.frames_pruned:
        media.prune_frames(p, [s["path"] for s in sources.entries(p)])
    try:
        file = media.frame(p, entry["path"], at, min(max(w, FRAME_MIN_W), FRAME_MAX_W), key=key)
    except (FFmpegError, OSError) as err:
        logger.warning("still of %s at %.2fs failed: %s", entry["name"], at, err)
        raise bad("could not extract a frame from this source", 500) from err
    if file is None:
        raise bad("source cannot be read", 404)
    return FileResponse(
        file, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"}
    )


@owned.get("/projects/{pid}/sources/{sid}/envelope")
def source_envelope(request: Request, pid: str, sid: str):
    """The audio envelope of a source for the trim rail (``step`` seconds per value in
    ``peaks``, 0-1; empty without audio)."""
    app, media, sources = request.app, request.app.state.media, request.app.state.sources
    if media.closing:
        raise bad("the server is stopping", 503)
    p, entry = find_source(app, pid, sid)
    media.prune_envelopes(p, [s["path"] for s in sources.entries(p)])
    try:
        data = media.envelope(p, entry["path"])
    except (FFmpegError, OSError) as err:
        logger.warning("envelope of %s failed: %s", entry["name"], err)
        raise bad("could not read the audio of this source", 500) from err
    if data is None:
        raise bad("source cannot be read", 404)
    if data.get("pending"):  # decoding: the page asks again in a few seconds
        return JSONResponse(data, status_code=202)
    return data


@owned.get("/projects/{pid}/events")
def project_events(pid: str):
    """The detected events the trim rail marks (see :func:`_rail_events`)."""
    return {"events": _rail_events(project_dir(pid))}


@owned.post("/projects/{pid}/sources/{sid}/prepare")
def prepare_source(request: Request, pid: str, sid: str, body: dict[str, Any] | None = None):
    """Queue the source's proxy; with ``{lo, hi}`` (seconds), a window proxy around that
    range first, which plays while the whole proxy builds. ``queue: true``: that window waits
    behind the queued ones instead of superseding them."""
    app, media, sources = request.app, request.app.state.media, request.app.state.sources
    window = None
    queue = (body or {}).get("queue", False)
    if not isinstance(queue, bool):
        raise bad("queue must be true or false")
    if body and ("lo" in body or "hi" in body):
        span = [body.get("lo"), body.get("hi")]
        times = [float(x) for x in span if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if len(times) != 2 or not all(map(math.isfinite, times)) or not 0 <= times[0] <= times[1]:
            raise bad("lo and hi must be times in seconds with 0 <= lo <= hi")
        window = (times[0], times[1])
    if queue and window is None:
        raise bad("queue needs lo and hi")
    p, entry = find_source(app, pid, sid)
    if p not in media.proxies_pruned:
        media.prune_proxies(p, [s["path"] for s in sources.entries(p)])
    return {"status": media.prepare(p, entry["path"], window, queue=queue)}


@served.api_route("/projects/{pid}/sources/{sid}/director/{name}", methods=["GET", "HEAD"])
def source_director(pid: str, sid: str, name: str, request: Request):
    """A rendered director preview of a 360 moment of this source (#517)."""
    lookups, previews = request.app.state.lookups, request.app.state.previews
    p, entry = lookups.find(pid, sid)  # one Range request of many, as source_media (#1020)
    # highlights.json and the previews listing are read when they change (#1029)
    ready = previews.ready(p).values()
    file = None
    if any(src == entry["path"] and n == name for src, n in ready):
        file = previews.file(p, name)
    if file is None:
        raise bad("director preview is not ready", 404)
    return send(file, request, proxy=True)


@owned.post("/projects/{pid}/director/{cid}")
def prepare_director(request: Request, pid: str, cid: str):
    """The page shows 360 candidate ``cid`` (its CLI id): plan its camera path
    (``candidates.preview_path``) and render its director preview in the background.
    ``{"pano": null}``: it is no moment of a 360 clip of the project (the page plays the raw
    preview)."""
    app, media, previews, sources = (
        request.app,
        request.app.state.media,
        request.app.state.previews,
        request.app.state.sources,
    )
    p = pano_project_dir(pid)
    if media.closing:
        raise bad("the server is stopping", 503)
    listing = sources.entries(p)
    hl = _highlights(p) or {}
    where = {str(c.get("id")): c.get("source") for c in hl.get("candidates") or []}
    if where.get(cid) not in {s["path"] for s in listing}:  # nothing may serve it
        return {"pano": None}
    if previews.request(p, cid) is None:
        return {"pano": None}
    return {"pano": pano_views(app, pid, p, listing, cid).get(cid)}


@owned.post("/projects/{pid}/director/{cid}/cancel")
def cancel_director(request: Request, pid: str, cid: str):
    """Stop the director preview render of ``cid``; the page plays the raw preview."""
    app, previews, sources = request.app, request.app.state.previews, request.app.state.sources
    p = pano_project_dir(pid)
    previews.cancel(p, cid)
    return {"pano": pano_views(app, pid, p, sources.entries(p), cid).get(cid)}


@served.api_route("/projects/{pid}/outputs/{name}", methods=["GET", "HEAD"])
def output_file(pid: str, name: str, request: Request):
    p = project_dir(pid)
    if name not in output_names(p):
        raise bad("unknown output", 404)
    return send(p / name, request)
