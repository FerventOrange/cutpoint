"""``cutpoint ui``: local browser workspace for the standard-video highlight workflow.

A loopback-only FastAPI app that serves ``cutpoint/web`` and a small JSON
API over the projects folder. Heavy work (analyze, clips, extend, assemble, resume) runs as
ordinary ``python -m cutpoint ... --json`` child processes in the projects folder, one at a time,
so the browser and the CLI share one implementation. Source videos are only ever read.

Trust boundaries:

* Every request needs a loopback ``Host`` (and ``Origin``, when sent) on the bound port.
* Every ``/api`` call needs the per-run token (``X-Cutpoint-Token``), which only the URL printed at
  startup carries (``#t=<token>``; there is no way to fetch it from the server). Media/output URLs
  carry ``?t=`` because ``<video>`` cannot send headers: a read-only media token derived from it
  (:func:`media_token`), which only the media routes accept, so a copied video or download
  address grants no other call. A request body over :data:`MAX_BODY_BYTES` is refused.
* The page's Content-Security-Policy (:func:`page_policy`) allows only its own files, inline
  styles, ``data:`` images and the inline theme script of ``index.html`` by its hash.
* Source stills (``.../sources/<id>/frame?at=<seconds>``) are media URLs too; they are cached in
  the project's ``ui/frames`` folder under a name built from the source's content key.
* The audio envelope of a source (``.../sources/<id>/envelope``, a JSON call) is cached in ``ui/envelope``
  under the same content key; the source is always one of the project's, by id.
* Director previews of 360 moments (``.../sources/<id>/director/<name>``) are media URLs too:
  renders in the project's ``ui/pano`` folder, made from a clip's ``.lrv`` in ``pano.json`` along
  the camera path ``pano.candidates.preview_path`` gives (``POST .../director/<candidate>`` asks
  for one).
* A project id is a direct child folder of the projects folder. A project's media are its
  recorded source files, and only those under the source the user picked for it (or, for projects
  made on the command line, the inputs of its analysis). For a project the page made,
  ``highlights.json`` alone never grants access to a file; for a command-line project both the
  inputs and the source list come from it, so whoever can write that file into the projects
  folder can have any video on disk served (the trust of local file access).
* Resume (``GET .../resume``, the ``resume`` job) replays the command line the project's
  ``progress.json`` recorded, from its recorded working folder (``cutpoint resume``): a
  recorded ``--force`` or an ``--out`` outside the project too. The page shows that command
  line in its confirm dialog first; the server never changes it, and a clips folder it wrote
  outside the project is never listed as an export.
* "Show folder" (``.../reveal?target=``) opens, in the file manager and without a shell, one of
  four folders the server resolves from the project: its own sources (the default), the clips
  folder of its latest export, the project folder holding its assembled final, or the project
  folder itself (AI Chat's Open project folder, next to a reply that names a project file). The
  request names only the target, never a path.
* Choosing a source or folder (``POST /api/pick?kind=file|folder``) opens the machine's own open
  dialog (Windows: Tk's system dialog; macOS: ``osascript``; Linux: zenity, else kdialog) in a
  child process without a shell, one at a time; the server never lists folders itself. With no
  dialog (no display, Docker) the page hides those buttons and the path is typed or pasted.
* Settings (``GET``/``PUT /api/settings``) change only ``projects_dir``, ``vision``,
  ``visual_index`` and the assistant's ``provider``, ``model`` and ``url`` in ``config.json``. The projects folder never changes while a job runs or when
  ``CUTPOINT_PROJECTS_DIR`` sets it; existing projects are not moved. Every analysis the page
  starts (first or again) passes ``--vision`` or ``--no-vision`` as ``vision`` says when it
  starts (neither while ``config.json`` has no ``vision`` key: the CLI default asks the vision
  model about unsure sources and camera-tagged sources without telemetry); the page has no per-run vision box. The same goes for
  ``--visual-index`` or ``--no-visual-index`` as ``visual_index`` says. The vision box rates
  frames only: ``vision: false`` in a PUT removes the key (the default) and never writes false;
  a hand-set ``"vision": false`` stays until then.
  While ``config.json`` is unreadable or holds a value Cutpoint rejects, the project routes answer
  409 naming the file and the error; session, pick and settings keep working.
* The assistant (``/api/projects/<id>/assistant/...``) takes the token header only, never
  ``?t=``. It only proposes sequence changes, which the page applies to its working sequence; it
  writes only ``ui/assistant/chat.json``, ``ui/assistant/history.json`` and the cached timeline
  sheets in ``ui/assistant/sheets``, never ``review.json``, and runs no commands (frame grabs
  aside). A ranking in another library style (#902) is previewed without a write
  (``POST /api/projects/<id>/ranking/preview``); only ``.../ranking/apply``, which the page
  sends when the user clicks to use it, replaces ``highlights.json`` (the one before kept as
  ``highlights.before-rerank.json``) and the chosen style in ``ui/meta.json``, in this process,
  and only when ``highlights.json`` is still the one previewed (409 ``stale`` otherwise). Its
  CUDA check runs in a child process: the UI process never imports torch. Project text (labels, transcripts, file names, never
  paths; only a model on this machine is given the project folder's path) reaches the model only
  inside marked data blocks. Its GET routes only read.
* A hosted assistant provider (any whose host is not this machine) gets no project text until
  the user agrees once for it (``POST /api/assistant/consent``, an ``assistant-v3:`` key kept in
  ``config.json``); nor is its model list asked for (``GET /api/assistant/models``) before that
  and without a key. That consent also covers the whole transcripts ``cutpoint analyze`` sends to
  a hosted chat model for its dialogue reader (:mod:`cutpoint.events.dialogue`). The Vision role's
  hosted provider has its own consent (``role: vision``, a
  ``vision-v2:`` key): agreeing for one role never covers the other, and an unprefixed key from an
  older version covers neither.
  Video frames (``view_moment``, :mod:`cutpoint.assistant.watch`) go to a hosted chat provider
  only once the user allowed that provider and host for that project
  (``POST /api/projects/<id>/assistant/frames-consent``, a ``frames-v1:`` key kept per project
  folder in ``assistant.frames_consent`` of ``config.json``); a provider on this machine needs
  none. The sheets are cached in ``ui/assistant/sheets``.
  API keys come from the environment or the system keyring (``POST /api/assistant/key``,
  ``.../key/clear``); no route returns one, only where it comes from. ``cutpoint ui`` takes the
  key variables out of its environment at startup, jobs get an environment without them, and
  log records have keys redacted (:mod:`cutpoint.assistant.keys`).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import errno
import hashlib
import hmac
import math
import mimetypes
import os
import re
import secrets
import signal
import socket
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cutpoint import cache, folders, insta360, jsonout, paths, progress, projectdelete, runstate
from cutpoint.assemble import (
    AUDIO_TRANSITIONS,
    OUTPUT_FORMATS,
    default_audio_transition,
    exact_clips,
)
from cutpoint.assistant import cliproxy
from cutpoint.assistant import context as ai_context
from cutpoint.assistant import edges as ai_edges
from cutpoint.assistant import keys as ai_keys
from cutpoint.assistant import lookfor as ai_lookfor
from cutpoint.assistant import ops as ai_ops
from cutpoint.assistant import providers as ai_providers
from cutpoint.assistant import session as ai_session
from cutpoint.assistant import store as ai_store
from cutpoint.assistant import watch as ai_watch
from cutpoint.config import (
    COLOR_MODES,
    VISION_PROVIDERS,
    AssistantSettings,
    assistant_settings,
    load_config_data,
    projects_dir,
    remap_source,
    update_config,
    update_section,
    vision_assistant_settings,
)
from cutpoint.edit import KINDS
from cutpoint.encode import BT709_TAGS
from cutpoint.errors import FFmpegError, FolderBusyError
from cutpoint.ffmpeg_utils import VideoEncoder, grab_images, probe, video_encoder
from cutpoint.pano import project as pano_project
from cutpoint.pano import x3
from cutpoint.pano.project import default_style as pano_style
from cutpoint.segments import MAX_SEGMENTS
from cutpoint.sources import TRANSCRIPT_MODELS
from cutpoint.styles import AUTO, get_style
from cutpoint.styles import style_name as fallback_style_name
from cutpoint.ui.analysis import (
    ANALYZE_KEYS,
    REFINE_WARNINGS,
    _action_cam_style,
    _analysis_inputs,
    _analyze_args,
    _auto_picks,
    _color_after,
    _color_args,
    _color_state,
    _compress_peaks,
    _flag,
    _gaming_style,
    _keep_style,
    _library_style,
    _no_options,
    _note_warnings,
    _opt,
    _picture_args,
    _reranked_style,
    _resolved_style,
    _saved_skip,
    _seconds,
    _speech_args,
    _style_crash,
    _subscription_roles,
    _vision_args,
    _vision_detect_all,
    _vision_detect_all_arg,
    analyze_options,
    crash_step_runs,
    detected_view,
    events_unavailable,
    new_project,
    preset_problem,
    preset_taken,
    refine_state,
    set_style,
    stitch_first,
    suggest_style,
    ui_styles,
)
from cutpoint.ui.common import (
    ANALYSIS_FILE,
    BUSY_TEXT,
    CLIP_ORDERS,
    ENVELOPE_STALE_S,
    ENVELOPE_TIMEOUT_S,
    ENVELOPE_VERSION,
    EVENT_TAILS,
    FINAL_EXTS,
    FRAME_CACHE_MAX,
    FRAME_MAX_AT,
    FRAME_MAX_W,
    FRAME_MIN_W,
    FRAME_WIDTH,
    LOOPBACK,
    MAX_BODY_BYTES,
    MAX_EXTEND_S,
    MAX_FADE_S,
    MAX_KEPT_NAMES,
    MAX_TIMING_S,
    MAX_TOTAL_S,
    MEDIA_LOOKUP_S,
    META_FILE,
    REVERT,
    SOURCES_CACHE_MAX,
    SOURCES_CACHE_S,
    STYLE_FOR,
    TELEMETRY_PROBE_TIMEOUT,
    UI_DIR,
    WEB_DIR,
    _suggest_lock,
    bad,
    cutpoint_cmd,
    job_env,
    logger,
    spawn,
)
from cutpoint.ui.files import (
    _highlights,
    _inside,
    _load,
    _own,
    _read_locked,
    _ui_folder,
    _values,
    picked_filters,
    picked_inputs,
    read_meta,
    save_analysis_skip,
)
from cutpoint.ui.jobs import Job, Jobs, parse_progress
from cutpoint.ui.media import (
    PROXY_PRUNE_GRACE_S,
    PROXY_WINDOWS_QUEUED,
    Media,
    ProxyWindow,
    _covers,
    _exit_reason,
    _mark_used,
    fragmented_mp4,
    proxy_args,
)
from cutpoint.ui.pano_previews import PANO_PREVIEW_W, PanoPreviews, _file_ident, _pano_reasons
from cutpoint.ui.picker import Picker, folder_opener, path_picker, pick_start
from cutpoint.ui.projects import (
    _REMOTE,
    DELETING,
    RERANKING,
    TARGET_MAX_S,
    MediaLookups,
    Sources,
    _body_inputs,
    _crash_events,
    _events_sig,
    _exclude_globs,
    _joins_of,
    _pano_folder,
    _rail_events,
    _recorded_picks,
    _skip_after,
    _target_of,
    _usable,
    _valid_target,
    analysis_stats,
    analysis_took,
    chat_files,
    clip_joins,
    current_export,
    export_joins,
    fresh_name,
    inputs_problem,
    inspect_source,
    load_exports,
    moment_id,
    moments_view,
    on_network,
    output_names,
    project_busy,
    project_dir,
    protected_sources,
    prune_resume_folders,
    resume_check,
    resumed,
    run_state,
    source_entries,
    source_folder,
    source_problem,
    validate_joins,
    validate_segments,
)
from cutpoint.ui.ranking import (
    CUDA_PROBE_S,
    NO_UNDO_TEXT,
    PREVIEW_TOP,
    RERANK_BACKUP,
    RERANK_BUSY_TEXT,
    STALE_TEXT,
    UNDO_STALE_TEXT,
    CudaProbe,
    PlainJSON,
    RankingHook,
    RankingRefused,
    _needs_analyze,
    _ranking_check,
    _ranking_load,
    _ranking_style,
    cuda_probe,
    ranking_key,
    ranking_preview,
    ranking_swap,
    ranking_undo_view,
)
from cutpoint.ui.settings import (
    LOGIN_TIMEOUT_S,
    _tuple_of_str,
    _vision_key_shared,
    _vision_provider_from_data,
    assistant_view,
    check_assistant,
    check_projects_dir,
    config_data,
    consent_view,
    key_hosts,
    key_status,
    require_config,
    settings_view,
)
from cutpoint.utils import clip_files, find_videos


# --------------------------------------------------------------------------- projects
def _loopback_url(value: str, port: int, *, origin: bool) -> bool:
    try:
        parts = urlsplit(value if origin else f"//{value}")
        return (
            (not origin or parts.scheme == "http")
            and parts.hostname in LOOPBACK
            and parts.port == port
        )
    except ValueError:
        return False


def page_policy() -> str:
    """The page's Content-Security-Policy: its own files only, inline styles, ``data:`` images
    (the icon) and the one inline script of ``index.html`` (the theme) by its hash."""
    script = "script-src 'self'"
    index = WEB_DIR / "index.html"
    if index.is_file():
        inline = re.findall(r"<script>(.*?)</script>", index.read_text(encoding="utf-8"), re.S)
        for body in inline:
            digest = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
            script += f" 'sha256-{digest}'"
    return (
        f"default-src 'self'; {script}; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; media-src 'self'; frame-ancestors 'none'"
    )


def media_token(token: str) -> str:
    """The read-only token of media URLs (``?t=``): accepted by the media routes only, so a
    copied video or download address cannot call the rest of the API."""
    return hmac.new(token.encode(), b"media", hashlib.sha256).hexdigest()


def create_app(port: int) -> FastAPI:
    token = secrets.token_urlsafe(32)
    media_t = media_token(token)
    policy = page_policy()
    jobs, media, picker, sources = Jobs(), Media(), Picker(), Sources()
    lookups = MediaLookups(sources)
    previews = PanoPreviews()
    # the subscription proxy, started when a login or a subscription model needs it (#821)
    proxies = cliproxy.ProxyLauncher()
    login_lock = threading.Lock()
    login_states: dict[str, dict[str, Any]] = {}
    ai_keys.redact_logs()  # an assistant API key never reaches a log line

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        asyncio.get_running_loop().set_exception_handler(quiet_resets)
        yield
        jobs.shutdown()
        media.shutdown()
        previews.shutdown()
        picker.shutdown()
        app.state.assistant.shutdown()
        app.state.proxies.shutdown()

    app = FastAPI(
        title="Cutpoint", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )

    app.state.token, app.state.jobs, app.state.media = token, jobs, media
    app.state.picker, app.state.sources, app.state.lookups = picker, sources, lookups
    app.state.previews = previews
    app.state.proxies = proxies  # routes look it up here, so a test can swap it

    class BodyCap:
        """Pure ASGI wrapper: a body read past MAX_BODY_BYTES, whatever its headers say, is 413."""

        def __init__(self, inner) -> None:
            self.inner = inner

        async def __call__(self, scope, receive, send) -> None:
            if scope["type"] != "http":
                await self.inner(scope, receive, send)
                return
            got = 0

            async def capped():
                nonlocal got
                msg = await receive()
                if msg["type"] == "http.request":
                    got += len(msg.get("body", b""))
                    if got > MAX_BODY_BYTES:
                        raise bad("request body too large", 413)
                return msg

            await self.inner(scope, capped, send)

    app.add_middleware(BodyCap)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        origin = request.headers.get("origin")
        if not _loopback_url(request.headers.get("host", ""), port, origin=False) or (
            origin is not None and not _loopback_url(origin, port, origin=True)
        ):
            return JSONResponse({"detail": "forbidden host or origin"}, status_code=403)
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > MAX_BODY_BYTES:  # refused before it is read
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = policy
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        else:
            # The page's own files: revalidated on each load (a 304 by ETag). Without it the
            # browser may reuse a copy it judges fresh by its age, so an updated app.css or
            # app.js is not used until that runs out (#415).
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    def supplied(request: Request, *, query: bool = False) -> bool:
        got = request.headers.get("x-cutpoint-token") or (
            request.query_params.get("t") if query else ""
        )
        got = (got or "").encode()
        # media URLs carry the read-only media token; the session token works there too
        return secrets.compare_digest(got, token.encode()) or (
            query and secrets.compare_digest(got, media_t.encode())
        )

    def auth(request: Request) -> None:
        if not supplied(request):
            raise bad("missing or wrong session token", 403)

    def auth_media(request: Request) -> None:
        if not supplied(request, query=True):
            raise bad("missing or wrong session token", 403)

    api = APIRouter(prefix="/api")
    secure = APIRouter(dependencies=[Depends(auth)])
    # Project routes need a config.json get_config() accepts; session, pick and settings do not.
    owned = APIRouter(dependencies=[Depends(auth), Depends(require_config)])
    served = APIRouter(dependencies=[Depends(auth_media), Depends(require_config)])

    def url(pid: str, kind: str, name: str) -> str:
        return f"/api/projects/{quote(pid, safe='')}/{kind}/{quote(name, safe='')}?t={media_t}"

    def window_url(pid: str, sid: str, lo: int, hi: int) -> str:
        return url(pid, "sources", sid).replace("?", f"/window/{lo}-{hi}?", 1)

    def director_url(pid: str, sid: str, name: str) -> str:
        return url(pid, "sources", sid).replace("?", f"/director/{name}?", 1)

    def pano_views(
        pid: str,
        p: Path,
        listing: list[dict[str, Any]],
        only: str | None = None,
        hl: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """The camera path flags and director preview of each candidate (by CLI id) of a 360
        clip, a ready preview with its url (under the source the page plays it as). ``hl``:
        the project's highlights.json when the caller has read it already."""
        previews.prune(p)
        views = previews.listing(p)
        if only is not None:
            views = {only: views[only]} if only in views else {}
        hl = (hl if hl is not None else _highlights(p)) or {}
        where = {str(c.get("id")): c.get("source") for c in hl.get("candidates") or []}
        sids = {s["path"]: s["id"] for s in listing}
        for cid, view in views.items():
            pv = view["preview"]
            if pv is not None and pv["state"] == "ready":
                sid = sids.get(where.get(cid))
                if sid is None:  # not one of the project's sources: nothing may serve it
                    view["preview"] = None
                else:
                    pv["url"] = director_url(pid, sid, pv.pop("name"))
        return views

    @secure.get("/session")
    def session():
        return {
            "styles": ui_styles(),
            "can_reveal": folder_opener() is not None,
            "can_pick": path_picker() is not None,
        }

    @secure.post("/pick")
    def pick_path(kind: str, start: str = ""):
        """Ask for a video (``kind=file``) or a folder in this machine's own open dialog; waits
        until the user closes it. ``{"path": null}`` when they cancel."""
        if kind not in ("file", "folder"):
            raise bad("kind must be 'file' or 'folder'")
        return {"path": picker.ask(kind, pick_start(start))}

    @secure.get("/settings")
    def get_settings():
        return settings_view(config_data())

    @secure.put("/settings")
    def put_settings(body: dict[str, Any]):
        """Change the projects folder, the vision and visual index defaults and/or the assistant's provider, model
        and base URL in ``config.json``. Existing projects stay where they are. A changed
        assistant is checked at once (reachable, key accepted, tool calls or JSON output):
        ``assistant_check`` is why it cannot answer, or None. Nothing (a key least of all) is
        sent to a hosted provider the user has not agreed to in AI Chat (#504): its check waits
        for the consent, and ``assistant_note`` says so."""
        if not {"projects_dir", "vision", "visual_index", "assistant", "vision_assistant"} & set(
            body
        ):
            raise bad("send projects_dir, vision, visual_index, assistant and/or vision_assistant")
        updates: dict[str, Any] = {}
        removals: tuple[str, ...] = ()
        for flag in ("vision", "visual_index"):
            if flag in body:
                updates[flag] = _flag(body, flag)
        if updates.get("vision") is False:  # unticked: back to the default, not "vision": false
            del updates["vision"]
            removals = ("vision",)
        data = config_data()  # an unreadable config.json: refuse before creating the folder
        existing_ast = data.get("assistant")
        existing_ast_prov = existing_ast.get("provider") if isinstance(existing_ast, dict) else None
        existing_v_ast = data.get("vision_assistant")
        existing_v_ast_prov = (
            existing_v_ast.get("provider") if isinstance(existing_v_ast, dict) else None
        )
        assistant = (
            check_assistant(body["assistant"], "assistant", existing_provider=existing_ast_prov)
            if "assistant" in body
            else None
        )
        vision_ast = (
            check_assistant(
                body["vision_assistant"],
                section_name="vision_assistant",
                existing_provider=existing_v_ast_prov,
            )
            if "vision_assistant" in body
            else None
        )
        moving = "projects_dir" in body
        if moving and paths.env("PROJECTS_DIR") is not None:
            raise bad(
                "CUTPOINT_PROJECTS_DIR sets the projects folder; change or unset it there", 409
            )
        busy = "wait for the running job to finish before changing the projects folder"
        if moving and any(j.status == "running" for j in list(jobs.jobs.values())):
            raise bad(busy, 409)
        folder, created = None, False
        if moving:
            folder, created = check_projects_dir(body.get("projects_dir"))
            updates["projects_dir"] = str(folder)

        def merge(now: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
            # the sections as they are in the file now: a change saved meanwhile stays
            values = dict(updates)
            for name, given in (("assistant", assistant), ("vision_assistant", vision_ast)):
                if given is not None:
                    raw = now.get(name)
                    section = raw if isinstance(raw, dict) else {}
                    # checked again against the provider saved meanwhile (e.g. a subscription
                    # one, which takes no url): a refusal writes nothing
                    check_assistant(given, name, existing_provider=section.get("provider"))
                    values[name] = {**section, **given}
            return values, removals

        # The jobs lock makes the running-job check and the write atomic against Jobs.start,
        # so no job starts against the old folder while the setting changes.
        try:
            with jobs.lock:
                if moving and any(j.status == "running" for j in jobs.jobs.values()):
                    raise bad(busy, 409)
                data = update_config(merge)
        except (ValueError, OSError, HTTPException) as err:
            if created and folder is not None:
                with contextlib.suppress(OSError):
                    folder.rmdir()
            if isinstance(err, HTTPException):
                raise
            if isinstance(err, ValueError):  # config.json is not valid JSON: never overwrite it
                raise bad(str(err), 409) from err
            raise bad(f"could not save the settings: {err.strerror or err}", 500) from err
        out = settings_view(data)
        if assistant is not None:
            out["assistant_note"] = None
            try:
                provider = ai_providers.make()
                if consent_view(provider)["needed"]:
                    # nothing goes to a host not agreed to yet, a key least of all (#504);
                    # a missing key is known without asking
                    out["assistant_check"] = ai_providers.missing_key(provider)
                    if out["assistant_check"] is None:
                        out["assistant_note"] = ai_providers.consent_note(provider)
                    return out
                out["assistant_check"] = ai_providers.check(provider)
                if out["assistant_check"] is None:  # reached; maybe without using the key
                    out["assistant_note"] = ai_providers.check_note(provider)
            except (ValueError, TypeError) as err:  # another invalid value in config.json
                out["assistant_check"] = str(err)
        if vision_ast is not None:
            out["vision_note"] = None
            out["vision_check"] = None
        return out

    # -- assistant provider: key and consent (#346); config.json and the keyring only -------
    def keyed_provider(body: dict[str, Any]) -> str:
        name = body.get("provider")
        if name not in ai_keys.KEYED:
            raise bad(f"provider must be one of {', '.join(ai_keys.KEYED)}")
        return name

    def key_section(body: dict[str, Any]) -> str:
        """The config section whose base URL names a key's host: ``role`` ``assistant`` (the
        default) or ``vision``."""
        role = body.get("role") or "assistant"
        if role not in ("assistant", "vision"):
            raise bad("role must be assistant or vision")
        return "vision_assistant" if role == "vision" else "assistant"

    @secure.post("/assistant/key")
    def save_key(body: dict[str, Any]):
        """Save ``key`` for ``provider`` in the system keyring, for the host
        its configured base URL names (``key_hosts`` of ``role``'s section): a key saved for
        OpenRouter is never sent to OpenAI. The answer says only where each key now comes from,
        never a key."""
        name = keyed_provider(body)
        section = key_section(body)
        data = config_data()
        try:
            vision = section == "vision_assistant"
            ai_keys.save(
                name,
                body.get("key"),
                key_hosts(data, section)[name],
                vision=vision,
                shared=not vision or _vision_key_shared(data, name),
            )
        except ValueError as err:
            raise bad(str(err)) from None
        except ai_keys.KeyringError as err:
            raise bad(str(err), 500) from None
        return key_status(data, section)

    @secure.post("/assistant/key/clear")
    def clear_key(body: dict[str, Any]):
        """Remove ``provider``'s keyring key for its configured host (of ``role``'s section);
        a key from the environment stays."""
        name = keyed_provider(body)
        section = key_section(body)
        data = config_data()
        try:
            ai_keys.clear(name, key_hosts(data, section)[name])
        except ai_keys.KeyringError as err:
            raise bad(str(err), 500) from None
        return key_status(data, section)

    @secure.post("/assistant/consent")
    def give_consent(body: dict[str, Any]):
        """Record that the user agreed to send project data to the provider in use for ``role``
        (``assistant``, the default, or ``vision``), named by its ``key``, in that role's own
        ``consent`` list; 409 when the provider in use is another one by now, or is local."""
        role = body.get("role") or "assistant"
        if role not in ("assistant", "vision"):
            raise bad("role must be assistant or vision")
        section_name = "vision_assistant" if role == "vision" else "assistant"
        provider = vision_provider() if role == "vision" else assistant_provider()
        want = ai_providers.consent_key(provider, role)
        if want is None or body.get("key") != want:
            raise bad(f"the {role} provider changed: check it and agree again", 409)

        def agree(section: dict[str, Any]) -> dict[str, Any] | None:
            given = list(_tuple_of_str(section.get("consent")))
            return None if want in given else {**section, "consent": [*given, want]}

        try:
            section = update_section(section_name, agree)  # re-read under the config lock
        except ValueError as err:
            raise bad(str(err), 409) from err
        except OSError as err:
            raise bad(f"could not save the consent: {err.strerror or err}", 500) from err
        consent_list = _tuple_of_str(section.get("consent"))
        return {"consent": consent_view(provider, consent=consent_list, role=role)}

    @secure.get("/assistant/models")
    def assistant_models(
        provider: str = "ollama",
        role: str = "assistant",
        refresh: bool = False,
    ):
        """The models a provider offers, for Settings' Model field: ``{state, models, message}``
        (:func:`cutpoint.assistant.providers.model_list`). Accepts role=assistant|vision."""
        if role not in ("assistant", "vision"):
            raise bad("role must be assistant or vision")
        try:
            if role == "vision":
                if provider not in VISION_PROVIDERS and provider != "custom":
                    raise bad(f"provider must be custom or one of {', '.join(VISION_PROVIDERS)}")
                settings = vision_assistant_settings()
            else:
                if (
                    provider not in ai_providers.PRESETS
                    and provider not in cliproxy.PROVIDERS
                    and provider != "custom"
                ):
                    raise bad(
                        f"provider must be custom or one of {', '.join(ai_providers.PRESETS)}"
                    )
                settings = assistant_settings()
            saved = settings.provider == provider and ai_providers.is_preset(
                settings.provider, settings.url
            )
            if provider == "custom" or saved:
                p = (ai_providers.make_vision if role == "vision" else ai_providers.make)(settings)
            elif provider in ai_providers.PRESETS:
                p = ai_providers.preset(provider, settings.context_tokens)
            elif provider in cliproxy.PROVIDERS:
                p = ai_providers.make(
                    AssistantSettings(provider=provider, context_tokens=settings.context_tokens)
                )
            else:
                raise bad(f"unknown provider {provider}")
        except (ValueError, TypeError) as err:  # a bad Ollama port, an invalid config.json
            raise bad(str(err), 409) from err
        if ai_providers.is_subscription(p) and (why := proxy_error(logged_in_only=True)):
            return {"state": "error", "models": [], "message": why}
        keep = settings.model if provider in (settings.provider, "custom") else ""
        return ai_providers.model_list(p, settings.consent, role=role, refresh=refresh, keep=keep)

    # -- subscription logins (Claude Code, Codex, Antigravity): managed via CLIProxyAPI ----
    def proxy_error(*, logged_in_only: bool) -> str | None:
        """Start the subscription proxy unless one runs (:class:`cliproxy.ProxyLauncher`;
        ``logged_in_only``: only when a login is saved); why it did not start, else None."""
        try:
            app.state.proxies.ensure(logged_in_only=logged_in_only)
        except (cliproxy.ProxyError, OSError) as err:
            return ai_keys.redact(f"the subscription proxy did not start: {err}")
        return None

    def login_provider(value: Any) -> str:
        if value not in cliproxy.PROVIDERS:
            raise bad(f"provider must be one of {', '.join(cliproxy.PROVIDERS)}")
        return value

    @secure.get("/assistant/logins")
    def assistant_logins():
        """Each subscription's login (cliproxy.status: never a token) with its title and
        whether the assistant can use it as a provider; ``proxy_running`` and ``saved_login``
        (a login an earlier proxy run kept): the page starts the proxy when both say so."""
        proxies = app.state.proxies
        running = proxies.running()
        st_map = {}
        with contextlib.suppress(cliproxy.ProxyError, OSError):
            for s in cliproxy.statuses():
                if isinstance(s, dict) and "provider" in s:
                    st_map[s["provider"]] = s
        out = []
        for p in cliproxy.PROVIDERS:
            st = st_map.get(p)
            if not st:
                try:
                    st = cliproxy.status(p)
                except (cliproxy.ProxyError, OSError) as err:
                    st = {
                        "provider": p,
                        "logged_in": False,
                        "email": None,
                        "org": None,
                        "expired": False,
                        "expires_ms": None,
                        "error": str(err),
                    }
            if st.get("error") == cliproxy.NOT_RUNNING:  # this server starts it when needed
                st = {**st, "error": None}
            out.append(
                {
                    "provider": p,
                    **st,
                    "title": cliproxy.TITLES.get(p, p),
                }
            )
        saved = not running and proxies.saved_login()
        return {"logins": out, "proxy_running": running, "saved_login": saved}

    @secure.post("/assistant/proxy")
    def start_proxy():
        """Start the subscription proxy when a login is saved and none runs (#821); 503 with
        the reason when it does not start. ``{running}``."""
        if why := proxy_error(logged_in_only=True):
            raise bad(why, 503)
        return {"running": app.state.proxies.running()}

    @secure.post("/assistant/login")
    def start_login(body: dict[str, Any]):
        """Start ``provider``'s login via CLIProxyAPI and answer with the sign-in ``url`` for
        the page to open; ``GET .../login/status`` tells when it is done. 409 while one waits."""
        name = login_provider(body.get("provider"))
        now = time.monotonic()
        old_state_to_cancel = None
        with login_lock:
            cur = login_states.get(name)
            if cur and cur.get("state") == "pending":
                if now - cur.get("started_at", 0) < LOGIN_TIMEOUT_S:
                    raise bad(f"a {name} login is already waiting for the browser", 409)
                old_state_to_cancel = cur.get("proxy_state")
            login_states[name] = {"state": "pending", "started_at": now, "error": ""}
        if old_state_to_cancel:
            with contextlib.suppress(Exception):
                cliproxy.cancel_login(old_state_to_cancel)
        app.state.proxies.wait_for_start()  # a start in flight answers only once it is done
        try:
            try:
                res = cliproxy.start_login(name)
            except cliproxy.ProxyConnectionError as first:
                # no proxy ran: a login always starts one (#821); one that ran and failed stands
                try:
                    started = app.state.proxies.ensure()
                except (cliproxy.ProxyError, OSError) as err:
                    why = ai_keys.redact(f"the subscription proxy did not start: {err}")
                    with login_lock:
                        login_states[name] = {"state": "error", "error": why}
                    raise bad(why, 503) from None
                if not started and str(first) != cliproxy.NOT_RUNNING:
                    raise
                res = cliproxy.start_login(name)  # ours, or one another request just started
        except (cliproxy.ProxyError, OSError) as err:
            with login_lock:
                login_states[name] = {
                    "state": "error",
                    "error": ai_keys.redact(str(err)) or type(err).__name__,
                }
            # proxy not running or unreachable: expected, so 503 rather than a server error
            code = 503 if isinstance(err, cliproxy.ProxyConnectionError) else 500
            raise bad(ai_keys.redact(str(err)) or f"the {name} login did not start", code) from None

        url = res.get("url")
        proxy_state = res.get("state")
        if not url or not proxy_state:
            with login_lock:
                login_states[name] = {"state": "error", "error": f"the {name} login did not start"}
            raise bad(f"the {name} login did not start", 500)

        with login_lock:
            login_states[name] = {
                "state": "pending",
                "proxy_state": proxy_state,
                "started_at": time.monotonic(),  # the browser gets the whole time after a start
                "error": "",
            }
        return {"url": url}

    @secure.get("/assistant/login/status")
    def login_status(provider: str):
        """``{state, error}`` of ``provider``'s last login: ``pending``, ``done`` or ``error``
        (``idle`` before any)."""
        name = login_provider(provider)
        with login_lock:
            info = login_states.get(name)
            if not info:
                return {"state": "idle", "error": ""}
            if info.get("state") != "pending":
                return {"state": info.get("state", "idle"), "error": info.get("error", "")}

            proxy_state = info.get("proxy_state")
            started_at = info.get("started_at", 0)
            if time.monotonic() - started_at > LOGIN_TIMEOUT_S:
                info["state"] = "error"
                info["error"] = "login timed out"
                to_cancel = proxy_state
                proxy_state = None
            else:
                to_cancel = None
        if to_cancel:
            with contextlib.suppress(Exception):
                cliproxy.cancel_login(to_cancel)
            return {"state": "error", "error": "login timed out"}
        try:
            poll_res = cliproxy.poll_login(proxy_state)
        except (cliproxy.ProxyError, OSError) as err:
            poll_res = {"state": "error", "error": ai_keys.redact(str(err)) or type(err).__name__}

        with login_lock:
            cur = login_states.get(name)
            if cur and cur.get("proxy_state") == proxy_state:
                p_state = poll_res.get("state", "pending")
                if p_state == "done":
                    ai_providers.clear_model_cache(name)
                    cur["state"] = "done"
                    cur["error"] = ""
                elif p_state == "error":
                    cur["state"] = "error"
                    cur["error"] = poll_res.get("error") or "login failed"
                elif p_state == "pending":
                    cur["state"] = "pending"
                return {"state": cur.get("state", "pending"), "error": cur.get("error", "")}
            return {
                "state": (cur or {}).get("state", "idle"),
                "error": (cur or {}).get("error", ""),
            }

    @secure.post("/assistant/logout")
    def logout(body: dict[str, Any]):
        """Remove ``provider``'s login from CLIProxyAPI."""
        name = login_provider(body.get("provider"))
        try:
            cliproxy.logout(name)
        except (cliproxy.ProxyError, OSError) as err:
            raise bad(str(err), 500) from None
        ai_providers.clear_model_cache(name)
        st = cliproxy.status(name)
        return {"provider": name, **st, "title": cliproxy.TITLES.get(name, name)}

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
            run = run_state(d, (d / "highlights.json").is_file())
            found.append(
                {
                    "id": d.name,
                    "name": read_meta(d).get("name") or d.name,
                    # a run that stopped and Resume continues (#109): the list marks it
                    "interrupted": bool(run and run["resumable"]),
                }
            )
        return {"projects": found}

    @owned.post("/projects/suggest-style")
    def suggest_project_style(body: dict[str, Any]):
        """The style the New project dialog preselects for these sources (null: no suggestion)."""
        inputs = _body_inputs(body)
        recursive, exclude = _flag(body, "recursive"), _exclude_globs(body)
        if not _suggest_lock.acquire(blocking=False):  # a suggestion is optional: never queue
            return {"style": None, "busy": True}  # not "no telemetry": the page asks again
        try:
            style = suggest_style(inputs, exclude, recursive)
        finally:
            _suggest_lock.release()
        return {"style": style if style in ui_styles() else None}

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

    @owned.get("/projects/{pid}")
    def get_project(pid: str):
        p = project_dir(pid)
        meta = read_meta(p)
        hl = _highlights(p)
        review = _load(p, f"{UI_DIR}/review.json")
        exp = current_export(p)
        manifest = cache.load_json(p / exp["dir"] / "manifest.json") if exp else None
        outputs = output_names(p)
        found = []
        listing, source_error = sources.listing(p)
        for s in listing:
            # Never probes: 200 sources took 11 s of ffprobe. The page prepares the one it plays.
            status, proxied, done = media.cached_view(p, s["path"])
            ready, pending = media.windows_of(p, s["path"])
            failed = media.failed_windows(p, s["path"])
            duration = s["duration"]
            if duration is None and (ready or pending):  # the page needs it to play windows
                duration = media.probed_duration(s["path"])
            found.append(
                {
                    "id": s["id"],
                    "name": s["name"],
                    "path": s["path"],
                    "url": url(pid, "sources", s["id"]),
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
                        {"lo": lo / 100, "hi": hi / 100, "url": window_url(pid, s["id"], lo, hi)}
                        for lo, hi in ready
                    ],
                    "pending": [
                        {"lo": lo / 100, "hi": hi / 100, "state": state}
                        for lo, hi, state in pending
                    ],
                    # window encodes that failed (until asked for again): the page stops waiting
                    "failed": [
                        {"lo": lo / 100, "hi": hi / 100, "reason": reason}
                        for lo, hi, reason in failed
                    ],
                }
            )
        highlights = moments_view(hl, _load(p, "events.json"))
        last_run = run_state(p, hl is not None)
        if highlights is not None and meta.get("kind") == "360":
            # #517: flags of each moment's cached camera path and its director preview (null:
            # not a moment of a 360 clip of the project, the page plays the raw preview)
            views = pano_views(pid, p, listing, hl=hl)
            for c in highlights["candidates"]:
                c["pano"] = views.get(str(c.get("cli_id")))
        view = {
            "id": pid,
            "name": meta.get("name") or pid,
            "style": _analysis_inputs(meta, hl)[1] or "",
            # the page's Color starts at Off for gaming footage, as config.json's color does
            "gaming": _gaming_style(_resolved_style(meta, hl), hl),
            # the header's style button (#897): the detected category and the chosen style
            "detected": detected_view(meta, hl),
            "kind": "360" if meta.get("kind") == "360" else "video",
            "source": meta.get("source"),
            "highlights": highlights,
            # the project menu's "Analyzed X of video in Y"
            "analysis": analysis_stats(hl, analysis_took(p, meta, hl)),
            "events_sig": _events_sig(p),
            "review": review if isinstance(review, dict) else None,
            "clips": {**manifest, "dir": exp["dir"]}
            if isinstance(manifest, dict) and exp
            else None,
            "outputs": [{"name": n, "url": url(pid, "outputs", n)} for n in outputs],
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
            else _style_crash(
                _reranked_style(hl, meta) or meta.get("style") or (hl or {}).get("style"), hl
            ),
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
            # the key of highlights.json a ranking preview is applied against (#902), and
            # whether a ranking applied here or by `cutpoint rerank` kept the one before
            "ranking_key": ranking_key(p),
            "rerank_backup": _inside(p, RERANK_BACKUP) and (p / RERANK_BACKUP).is_file(),
            # Undo (or Redo) of the last ranking change: {action, style}, null when the backup
            # does not match the current analysis (ranking_undo_view)
            "ranking_undo": ranking_undo_view(p, meta),
        }
        return PlainJSON(view)

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
            "argv": argv
            if isinstance(argv, list) and all(isinstance(a, str) for a in argv)
            else [],
            "status": rec.get("status_before"),
            "steps": steps if isinstance(steps, dict) else {},
            "warnings": [
                w["message"]
                for w in result.get("warnings") or []
                if isinstance(w, dict) and isinstance(w.get("message"), str)
            ],
        }

    review_lock = threading.Lock()  # review.json is read and rewritten by two routes

    @owned.put("/projects/{pid}/review")
    def put_review(pid: str, body: dict[str, Any]):
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
    def put_target(pid: str, body: dict[str, Any]):
        """The sequence's target length in ``seconds`` (null clears it), kept in ``review.json``
        beside the segments; only a guide for the page's progress bar."""
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

    @owned.post("/projects/{pid}/jobs")
    def start_job(pid: str, body: dict[str, Any]):
        p = project_dir(pid)
        action = body.get("action")
        meta = read_meta(p)
        hl = _highlights(p)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        on_success: Callable[[], None] | None = None
        prepare: Callable[[], None] | None = None
        before: Callable[[Job], list[str] | None] | None = None  # a step ahead of ``args``
        on_result: Callable[[Job], None] | None = None  # after on_success, with the envelope
        written: list[Path] = []  # files ``prepare`` wrote, removed if the job does not start
        undo: list[Callable[[], None]] = []  # what ``prepare`` changed, put back the same way
        plan: list[dict[str, str]] | None = None
        if action == "analyze" and meta.get("kind") == "360":
            # The CLI's 360 flow: scan, views, analysis of the .lrv, events (no vision rating)
            style = meta.get("style")
            if not meta.get("source") or not style:
                raise bad("project has no recorded source to analyze")
            if any(
                body.get(k) not in (None, "", []) and body[k] is not False for k in ANALYZE_KEYS
            ):
                raise bad("Analyze options are not available for 360 projects")
            if body.get("style") is not None:
                raise bad("the style of a 360 project cannot be changed")
            _no_options([meta["source"]], "the source")
            only = [c for c in meta.get("clips") or [] if isinstance(c, str)]
            if not all(re.fullmatch(r"\d{3}|\d{8}_\d{6}_\d{3}", c) for c in only):  # X3 clip ids
                raise bad("the clips in the project's ui/meta.json are not clip ids")
            f = insta360.parse_name(Path(meta["source"]))
            if f:  # a project made before A207 holds the bare number of its one picked file
                only = [
                    x3.clip_key(f["date"], f["time"], f["num"]) if c == f["num"] else c
                    for c in only
                ]
            live = _library_style(style, hl)  # a deleted style: its recorded base
            _own(p, "highlights.json")  # the analysis rewrites it
            _own(p, "pano.json")  # and the scan this
            args = cutpoint_cmd(
                "360", "analyze", str(p), "--scan", str(_pano_folder(meta["source"]))
            )
            args += ["--style", str(live)]
            if live != style:
                _own(p, META_FILE)  # refuse now, not when the job starts

                def prepare() -> None:
                    _keep_style(p, str(live), undo)

            if only:
                args += ["--only", *only]
            plan = [{"step": s, "label": progress.LABELS[s]} for s in progress.PANO_STEPS]
        elif action == "analyze":
            inputs, style = _analysis_inputs(meta, hl)
            if not inputs or not style:
                raise bad("project has no recorded source to analyze")
            _no_options(inputs, "an input")
            # The header's style button (#897): analyze again in another style (auto: detected)
            new = body.get("style")
            if new is not None and (
                not isinstance(new, str)
                or (new != AUTO and new not in ui_styles() and not _action_cam_style(new))
            ):
                raise bad(f"unknown style {new!r}")
            if new is None and (live := _library_style(style, hl)) != style:
                new = str(live)  # a deleted style: analyze again in its base, and keep that
            if new is None and style == _reranked_style(hl, meta) != meta.get("style"):
                # a re-rank's style: kept in meta.json, as the analysis records no re-rank and
                # the next one would go back to the chosen style (#902)
                new = str(style)
            # the options as the earlier analysis used them (inferred against its own style)
            recorded = analyze_options(p, meta, hl)
            # a re-rank (on the command line) left Auto too: its detection is kept as well
            reranked = isinstance(((hl or {}).get("settings") or {}).get("reranked"), dict)
            leaving_auto = new is not None and (style == AUTO or reranked) and new != AUTO
            style = new if new is not None else style
            _own(p, "highlights.json")  # the analysis rewrites it
            args = cutpoint_cmd(
                "analyze", *map(str, inputs), "--style", str(style), "--out", str(p)
            )
            # What the project was made with, else (made, or re-analyzed with these options, on
            # the command line) what the earlier analysis recorded, or the source set would change.
            st = (hl or {}).get("settings") or {}
            exclude, recursive = picked_filters(meta)
            if not picked_inputs(meta) or not (exclude or recursive):
                ex = st.get("exclude")
                exclude = [g for g in ex if isinstance(g, str)] if isinstance(ex, list) else []
                recursive = st.get("recursive") is True
            args += [f"--exclude={g}" for g in exclude]
            if recursive:
                args.append("--recursive")
            # The Analyze options; a recorded --recover is kept unless the page turns it off, or
            # the parts past a read error drop out.
            opts, skip = _analyze_args(body, recorded, style == AUTO)
            args += opts + _vision_args(st)
            if skip:  # last: its steps follow it. Saved (an older file never matches the new
                # analysis's created stamp, so without --skip there is nothing to save)
                args += ["--skip", *skip]
                _own(p, ANALYSIS_FILE)  # refuse now, not after the analysis

                def prepare() -> None:
                    _ui_folder(p)  # a command-line project has no ui folder yet

                def on_success() -> None:
                    save_analysis_skip(p, skip)

            if new is not None:
                _own(p, META_FILE)  # refuse now, not when the job starts
                # left Auto: the detection is kept as the "Detected earlier" copy
                kept = (hl or {}).get("detection") if leaving_auto else None

                def prepare() -> None:
                    # once the job may start, so a refused one leaves the style as it was; not
                    # while a run elsewhere holds the project (its analyze would stop at once)
                    if project_busy(p, run_state(p, hl is not None)):
                        raise bad(BUSY_TEXT, 409)
                    made: list[Path] = []
                    _ui_folder(
                        p, made=made
                    )  # a command-line project has no ui folder (nor meta.json) yet
                    back = set_style(p, new, kept if isinstance(kept, dict) else None)

                    def put_back() -> None:
                        try:
                            back()
                        finally:
                            for folder in made:  # the ui folder this created, if still empty
                                with contextlib.suppress(OSError):
                                    folder.rmdir()

                    undo.append(put_back)

        elif action == "edit":
            # A one-click preset: analyze (cached), clips and assemble as `edit --kind` says, in
            # the project's own style so its analysis and moments stay as they are.
            kind = body.get("kind")
            if not isinstance(kind, str) or kind not in KINDS:
                raise bad(f"kind must be one of {', '.join(KINDS)}")
            if why := preset_problem(meta, hl):
                raise bad(why)
            inputs, style = _analysis_inputs(meta, hl)
            _no_options(inputs or [], "an input")
            live = _library_style(style, hl)  # a deleted style: the edit runs in its base
            vertical = _flag(body, "vertical")
            force = _flag(body, "force")
            clips_name = "clips_vertical" if vertical else "clips"
            final_name = "final_vertical.mp4" if vertical else "final.mp4"
            _own(p, "highlights.json")  # the analysis rewrites it
            _own(p, clips_name)
            _own(p, f"{UI_DIR}/exports.json")  # refuse now, not after the edit
            if clips_name in preset_taken(p) and not force:
                raise bad(
                    f"{clips_name} holds the clips of an earlier run; confirm to replace", 409
                )
            args = cutpoint_cmd(
                "edit", *map(str, inputs), "--kind", kind, "--style", str(live), "--out", str(p)
            )
            if vertical:
                args.append("--vertical")
            if force:  # the user confirmed replacing the earlier run's clips
                args.append("--force")
            args += (
                _compress_peaks(body)
                + _color_args(body)
                + _vision_args((hl or {}).get("settings") or {})
            )
            if live == AUTO:  # the analysis's --vision-detect-all, so the detection repeats
                args += _vision_detect_all_arg(_vision_detect_all(hl))
            # edit analyzes again: with the Analyze options' --skip, or it would run those steps
            if skip := _saved_skip(p, hl):
                args += ["--skip", *skip]
                _own(p, ANALYSIS_FILE)

            # a re-rank's style is kept in meta.json too: the edit's analysis records no
            # re-rank, and the next job would go back to the chosen style (#902)
            from_rerank = style == _reranked_style(hl, meta) != meta.get("style")
            keep = live != style or from_rerank
            if keep:
                _own(p, META_FILE)  # refuse now, not when the job starts

            def prepare() -> None:
                _ui_folder(p)  # a command-line project has no ui folder for exports.json yet
                if keep:  # the style (a deleted one: the base the edit records) is kept
                    det = (hl or {}).get("detection") if from_rerank else None
                    _keep_style(p, str(live), undo, det if isinstance(det, dict) else None)

            def on_success() -> None:
                if (
                    not (p / clips_name / "manifest.json").is_file()
                    or not (p / final_name).is_file()
                ):
                    raise RuntimeError("edit finished without its clips and final video")
                if skip:  # the analysis edit rewrote ran with it too
                    save_analysis_skip(p, skip)
                items = [e for e in load_exports(p) if e["dir"] != clips_name]  # replaced
                items.append(
                    {"dir": clips_name, "created": stamp, "vertical": vertical, "final": final_name}
                )
                cache.save_json(_own(p, f"{UI_DIR}/exports.json"), {"exports": items})

        elif action == "clips":
            picks = body.get("picks", "sequence")
            if picks not in ("sequence", "auto"):
                raise bad("picks must be 'sequence' or 'auto'")
            if picks == "auto":  # the CLI picks from highlights.json (clips --top/--ids)
                selection = _auto_picks(body, hl)
                segments = []
            else:
                if any(body.get(k) is not None for k in ("top", "ids", "max_total", "order")):
                    raise bad("top, ids, max_total and order need automatic picks")
                raw = _load(p, f"{UI_DIR}/review.json")
                segments = validate_segments(
                    (raw.get("segments") or []) if isinstance(raw, dict) else [],
                    sources.entries(p),
                    media,
                )
                if not segments:
                    raise bad("choose at least one moment before exporting clips")
            extend = body.get("extend", 0)
            if (
                isinstance(extend, bool)
                or not isinstance(extend, (int, float))
                or not 0 <= extend <= MAX_EXTEND_S
            ):
                raise bad(f"extend must be 0-{MAX_EXTEND_S:g} seconds")
            extend_start = _seconds(body, "extend_start", MAX_EXTEND_S)
            extend_end = _seconds(body, "extend_end", MAX_EXTEND_S)
            vertical = _flag(body, "vertical")
            # Smart edges (older pages) is both; each of its halves can be sent on its own.
            smart = _flag(body, "smart_edges")
            speech = _flag({"speech_aware": smart, **body}, "speech_aware")
            tails = _flag({"smart_tail": smart, **body}, "smart_tail")
            event_tails = body.get("event_tails", "auto")
            if not isinstance(event_tails, str) or event_tails not in EVENT_TAILS:
                raise bad("event_tails must be 'auto', 'on' or 'off'")
            if segments and not tails and event_tails == "on":
                # each segment's "smart_tail": false turns event tails off too
                raise bad("event tails need smart tails for the sequence's moments")
            timing = (
                _speech_args(body)
                + _opt("--tail-min", _seconds(body, "tail_min", MAX_TIMING_S))
                + _opt("--tail-max", _seconds(body, "tail_max", MAX_TIMING_S))
                + _opt(
                    "--event-window", _seconds(body, "event_window", MAX_TIMING_S, positive=True)
                )
            )
            peaks = _compress_peaks(body) + _color_args(body)
            picture = _picture_args(body) + _opt(
                "--audio-fade", _seconds(body, "audio_fade", MAX_FADE_S)
            )
            out_name = fresh_name(p, f"clips_ui_{stamp}")
            seg_file = _own(p, f"{UI_DIR}/segments-{out_name[len('clips_ui_') :]}.json")
            _own(p, f"{UI_DIR}/exports.json")  # refuse now, not after the export

            def prepare() -> None:
                _ui_folder(p)  # a command-line project has no ui folder yet
                if not segments:  # automatic picks: no segments file
                    return
                # The user's trims, in their order. Exactly them unless Speech-aware cuts or Smart
                # tails is on: then the segments leave that to the flags below.
                off = {} if speech else {"speech_aware": False}
                if not tails:
                    off = {"smart_tail": False, **off}
                cache.save_json(seg_file, [{**s, **off} for s in segments])
                written.append(seg_file)

            if segments:
                args = cutpoint_cmd(
                    "clips", str(p), "--segments", str(seg_file), "--out", str(p / out_name)
                )
            else:
                args = cutpoint_cmd("clips", str(p), *selection, "--out", str(p / out_name))
            if vertical:
                args.append("--vertical")
            if segments:  # an exact trim is a per-segment setting, not a flag
                args += ["--smart-tail"] if tails else []
                args += ["--speech-aware"] if speech else []
            else:  # explicit, so the style's defaults do not override what the page showed
                args.append("--smart-tail" if tails else "--no-smart-tail")
                args.append("--speech-aware" if speech else "--no-speech-aware")
            args += EVENT_TAILS[event_tails] + timing
            if extend:
                args += ["--extend", f"{float(extend):g}"]
            args += _opt("--extend-start", extend_start) + _opt("--extend-end", extend_end)
            args += picture + peaks
            if meta.get("kind") == "360":  # stitch the recordings first (pano.auto_stitch)
                chosen = {k: body.get(k) for k in ("top", "ids", "max_total")}

                def before(job: Job) -> list[str] | None:
                    return stitch_first(job, p, seg_file if segments else None, chosen)

            def on_success() -> None:
                if not (p / out_name / "manifest.json").is_file():
                    raise RuntimeError("clips finished without a manifest")
                items = load_exports(p)
                items.append(
                    {"dir": out_name, "created": stamp, "vertical": vertical, "final": None}
                )
                cache.save_json(_own(p, f"{UI_DIR}/exports.json"), {"exports": items})

        elif action == "extend":
            exp = current_export(p)
            if exp is None or not clip_files(p / exp["dir"]):
                raise bad("export clips before extending them")
            src_dir = exp["dir"]
            vertical = exp.get("vertical") is True
            pad = _opt("--start", _seconds(body, "extend_start", MAX_EXTEND_S)) + _opt(
                "--end", _seconds(body, "extend_end", MAX_EXTEND_S)
            )
            out_name = fresh_name(p, f"clips_ui_{stamp}")
            _own(p, f"{UI_DIR}/exports.json")  # refuse now, not after the export
            args = cutpoint_cmd("extend", str(p / src_dir), "--out", str(p / out_name), *pad)
            # explicit: extend snaps to speech by default, the page may have it off
            args.append("--speech-aware" if _flag(body, "speech_aware") else "--no-speech-aware")
            args += _speech_args(body)

            def on_success() -> None:
                if not (p / out_name / "manifest.json").is_file():
                    raise RuntimeError("extend finished without a manifest")
                items = load_exports(p)
                items.append(
                    {"dir": out_name, "created": stamp, "vertical": vertical, "final": None}
                )
                cache.save_json(_own(p, f"{UI_DIR}/exports.json"), {"exports": items})

        elif action == "assemble":
            exp = current_export(p)
            if exp is None or not clip_files(p / exp["dir"]):
                raise bad("export clips before assembling")
            transition = body.get("transition", "cut")
            if transition not in ("cut", "fade"):
                raise bad("transition must be 'cut' or 'fade'")
            # Old pages send none: the audio follows the video, as before
            audio_transition = body.get("audio_transition", default_audio_transition(transition))
            if audio_transition not in AUDIO_TRANSITIONS:
                raise bad("audio_transition must be 'cut' or 'crossfade'")
            if transition == "cut" and audio_transition == "crossfade":
                raise bad("an audio crossfade needs the fade video transition")
            exp_dir = exp["dir"]
            saved = _joins_of(_load(p, f"{UI_DIR}/review.json"))
            joins = export_joins(p / exp_dir, saved)
            files = clip_files(p / exp_dir)
            hard = [a or b for a, b in pairwise(exact_clips(p / exp_dir, files))]
            final_joins = [
                {
                    "a": j["a"],
                    "b": j["b"],
                    # a join next to a word splice is always a cut (assemble.hard_cuts)
                    "transition": "cut" if cut else j["transition"] or transition,
                }
                for j, cut in zip(clip_joins(p / exp_dir, saved), hard, strict=True)
                if j["a"] and j["b"]
            ]
            fade_dur = _seconds(body, "fade_dur", MAX_FADE_S, positive=True)
            # the saved choices, not ``joins``: the page offers Fade length from the sequence's
            # order, and a Fade pair there may not be neighbours in this export's clips
            fading = transition == "fade" or any(j["transition"] == "fade" for j in saved)
            if fade_dur is not None and not fading:
                raise bad("fade_dur needs the fade video transition")
            captions = _flag(body, "captions")
            model = body.get("caption_model")
            if model is not None and (not isinstance(model, str) or model not in TRANSCRIPT_MODELS):
                raise bad(f"caption_model must be one of {', '.join(TRANSCRIPT_MODELS)}")
            if model is not None and not captions:
                raise bad("caption_model needs captions")
            fmt = body.get("format", "mp4")
            if not isinstance(fmt, str) or fmt not in OUTPUT_FORMATS:
                raise bad(f"format must be one of {', '.join(OUTPUT_FORMATS)}")
            loudnorm = _flag(body, "loudnorm")
            if fmt != "mp4" and (loudnorm or captions):  # as the CLI refuses them (exit 2)
                raise bad(
                    f"an {fmt} timeline has no audio processing or burned-in captions; "
                    "turn off Normalize loudness and Burn in captions, or choose mp4"
                )
            stem = fresh_name(
                p,
                "final" + exp_dir[5:] if exp_dir.startswith("clips") else f"final_{exp_dir}",
                FINAL_EXTS,
            )
            args = cutpoint_cmd(
                "assemble",
                str(p / exp_dir),
                "--out",
                str(p / f"{stem}.{fmt}"),
                "--format",
                fmt,
                "--transition",
                transition,
                "--audio-transition",
                audio_transition,
                *_compress_peaks(body),
                *_opt("--fade-dur", fade_dur),
                *_opt("--audio-fade", _seconds(body, "audio_fade", MAX_FADE_S)),
            )
            if loudnorm:
                args.append("--loudnorm")
            if captions:
                args.append("--captions")
            if model is not None:
                args += ["--model", model]
            if joins:
                joins_file = _own(p, f"{UI_DIR}/joins-{stem}.json")
                args += ["--joins", str(joins_file)]

                def prepare() -> None:
                    _ui_folder(p)  # a command-line project has no ui folder yet
                    cache.save_json(joins_file, {"joins": joins})
                    written.append(joins_file)

            def on_success() -> None:
                # a timeline export joins and encodes nothing: its timeline is the result
                if not (p / f"{stem}.{fmt}").is_file():
                    what = "a final video" if fmt == "mp4" else f"its {fmt} timeline"
                    raise RuntimeError(f"assemble finished without {what}")
                items = load_exports(p)
                for e in items:
                    if e["dir"] == exp_dir:
                        e["final"] = f"{stem}.{fmt}"
                        e["final_joins"] = final_joins
                cache.save_json(_own(p, f"{UI_DIR}/exports.json"), {"exports": items})

        elif action in ("crashes", "events"):
            # The analysis refined after the fact, one CLI command each: "Describe the footage"
            # is `crashes --context` (profiles, redoes the crash detection, re-ranks), "Look
            # closer with vision" is `events` (the VLM layer, re-ranks).
            if hl is None:
                raise bad("analyze the footage first")
            if meta.get("kind") == "360":
                raise bad(f"{action} does not apply to 360 projects")
            if why := events_unavailable():
                raise bad(why, 409)
            _own(p, "highlights.json")  # the re-rank rewrites it
            if action == "crashes":
                context = body.get("context")
                if not isinstance(context, str) or not context.strip():
                    raise bad("describe the footage first")
                from cutpoint.activity import MAX_CONTEXT_CHARS  # noqa: PLC0415

                if len(context.strip()) > MAX_CONTEXT_CHARS:
                    raise bad(f"the description is at most {MAX_CONTEXT_CHARS} characters")
                if not crash_step_runs(p, meta, hl):
                    raise bad("crash detection does not run for this footage")
                _own(p, "activity.json")
                _own(p, "crashes.json")
                args = cutpoint_cmd("crashes", str(p), f"--context={context.strip()}")
            else:
                _own(p, "events.json")
                args = cutpoint_cmd("events", str(p))

            def on_result(job: Job) -> None:
                _note_warnings(job)

        elif action == "resume":
            # cutpoint resume replays the stopped run of progress.json (#109) from its own cwd
            rec = runstate.read(p) if _inside(p, runstate.FILE) else None
            run = run_state(p, hl is not None)
            if rec is None or run is None or not run["resumable"]:
                raise bad("this project has no stopped run to resume", 409)
            skips = run["command"] in ("analyze", "edit") and _skip_after(
                [str(a) for a in rec.get("argv") or []]
            )
            if skips:
                _own(p, ANALYSIS_FILE)  # resumed() saves the --skip: refuse now, not after
            if run["command"] in ("clips", "edit"):
                _own(p, f"{UI_DIR}/exports.json")  # refuse now, not after the run
            if skips or run["command"] in ("clips", "edit"):

                def prepare() -> None:
                    _ui_folder(p)  # as the edit's: a ui file refuses the job before it starts

            if run["command"] == "clips" and meta.get("kind") == "360":
                # stitch the recordings first, as the export it continues did
                seg_file, chosen = _recorded_picks(rec)

                def before(job: Job) -> list[str] | None:
                    return stitch_first(job, p, seg_file, chosen)

            if run["command"] != "clips":
                _own(p, "highlights.json")  # the analysis rewrites it
            if run["command"] == "360 analyze":
                plan = [{"step": s, "label": progress.LABELS[s]} for s in progress.PANO_STEPS]
            args = cutpoint_cmd("resume", str(p))

            def on_result(job: Job) -> None:
                outputs = (job.result or {}).get("outputs") or {}
                if outputs.get("resumed") is True:  # not when the record had finished meanwhile
                    resumed(p, rec, outputs, stamp)

        else:
            raise bad(
                "action must be 'analyze', 'edit', 'clips', 'extend', 'assemble', 'crashes', "
                "'events' or 'resume'"
            )

        def same_folder() -> None:
            # Under the jobs lock, like the settings change: a folder changed since ``p`` was
            # resolved would run the job against the old one.
            if projects_dir().resolve() != p.resolve().parent:
                raise bad("the projects folder changed; reload the page", 409)
            if p in RERANKING:  # its highlights.json is being replaced now
                raise bad(RERANK_BUSY_TEXT, 409)
            if prepare:
                prepare()  # only once the job may start: a refused one leaves no file behind

        # --json: the envelope's error.code tells a busy project apart (Jobs._busy) and a resume
        # names what it wrote. The jobs run in the projects folder, not where the server started.
        if action in ("analyze", "edit") and meta.get("kind") != "360":
            needs_proxy = _subscription_roles(args, next(iter(_values(args, "--style")), style))
        elif action == "resume" and run["command"] in ("analyze", "edit"):
            argv = [str(a) for a in rec.get("argv") or []]
            # no --style: auto (edit: its kind's default, funny for funny-moments)
            kind_arg = next(iter(_values(argv, "--kind")), "highlights")
            default = (
                KINDS[kind_arg].style if run["command"] == "edit" and kind_arg in KINDS else AUTO
            )
            style_arg = next(iter(_values(argv, "--style")), default)
            needs_proxy = _subscription_roles(argv, style_arg)
        elif action in ("crashes", "events") or (
            action == "resume" and run["command"] in ("crashes", "events")
        ):  # the profile's and the VLM layer's Vision model
            try:
                needs_proxy = ai_providers.is_subscription(vision_assistant_settings().provider)
            except (ValueError, TypeError, AttributeError):  # an invalid config.json: the job says
                needs_proxy = False
        else:
            needs_proxy = False
        if needs_proxy and before is None:

            def before(job: Job) -> list[str] | None:
                # a Vision or dialogue role on a subscription needs the proxy (#832); the step
                # is skipped with its usual warning when it does not start
                if why := proxy_error(logged_in_only=True):
                    job.note("WARNING", why)
                return None

        args = [*args, "--json"]

        def put_back_all() -> None:  # under the jobs lock, so no newer write is overwritten
            for back in undo:
                with contextlib.suppress(OSError):
                    back()

        try:
            return jobs.start(
                pid,
                action,
                args,
                on_success,
                same_folder,
                plan,
                before,
                cwd=p.parent,
                on_result=on_result,
                on_end=(lambda: prune_resume_folders(p))
                if action in ("clips", "extend", "resume")
                else None,
                on_refused=put_back_all,
            ).view()
        except HTTPException:  # for example the command could not be started
            for f in written:
                with contextlib.suppress(OSError):
                    f.unlink(missing_ok=True)
            raise

    @secure.get("/jobs")
    def list_jobs():
        """The kept jobs, oldest first. Only the running job and the newest carry their log: the
        page shows the log of the running job, else of the newest, and polls this every second
        while a job runs (A744: 50 kept logs of 300 lines each were sent every time)."""
        kept = list(jobs.jobs.values())
        return {
            "jobs": [
                j.view(logs=j.status == "running" or i == len(kept) - 1) for i, j in enumerate(kept)
            ]
        }

    @secure.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        return jobs.cancel(job_id).view()

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

    def find_source(pid: str, sid: str) -> tuple[Path, dict[str, Any]]:
        p = project_dir(pid)
        entry = next((s for s in sources.entries(p) if s["id"] == sid), None)
        if entry is None:
            raise bad("unknown source", 404)
        return p, entry

    @served.api_route("/projects/{pid}/sources/{sid}", methods=["GET", "HEAD"])
    def source_media(pid: str, sid: str, request: Request):
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
        m = re.fullmatch(r"(\d{1,9})-(\d{1,9})", span)
        if m is None:
            raise bad("unknown window", 404)
        p, entry = lookups.find(pid, sid)
        key = lookups.content_key(pid, entry["path"])  # no stat per request (#1030)
        file = media.window(p, entry["path"], int(m[1]), int(m[2]), key)
        if file is None:
            status = media.status(p, entry["path"], key)[0]
            raise bad(
                f"window is not ready (source is {status})", 404 if status == "failed" else 409
            )
        return send(file, request, proxy=True)

    @served.get("/projects/{pid}/sources/{sid}/frame")
    def source_frame(pid: str, sid: str, at: float = 0.0, w: int = FRAME_WIDTH, key: bool = False):
        """One still of a source (``at`` seconds in, ``w`` px wide) for thumbnails; ``key=1``:
        the keyframe at or before ``at`` (the rail's filmstrip)."""
        if not math.isfinite(at) or at < 0:
            raise bad("at must be a time in seconds, 0 or more")
        if media.closing:
            raise bad("the server is stopping", 503)
        p, entry = find_source(pid, sid)
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
    def source_envelope(pid: str, sid: str):
        """The audio envelope of a source for the trim rail (``step`` seconds per value in
        ``peaks``, 0-1; empty without audio)."""
        if media.closing:
            raise bad("the server is stopping", 503)
        p, entry = find_source(pid, sid)
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
    def prepare_source(pid: str, sid: str, body: dict[str, Any] | None = None):
        """Queue the source's proxy; with ``{lo, hi}`` (seconds), a window proxy around that
        range first, which plays while the whole proxy builds. ``queue: true``: that window waits
        behind the queued ones instead of superseding them."""
        window = None
        queue = (body or {}).get("queue", False)
        if not isinstance(queue, bool):
            raise bad("queue must be true or false")
        if body and ("lo" in body or "hi" in body):
            span = [body.get("lo"), body.get("hi")]
            times = [
                float(x) for x in span if isinstance(x, (int, float)) and not isinstance(x, bool)
            ]
            if (
                len(times) != 2
                or not all(map(math.isfinite, times))
                or not 0 <= times[0] <= times[1]
            ):
                raise bad("lo and hi must be times in seconds with 0 <= lo <= hi")
            window = (times[0], times[1])
        if queue and window is None:
            raise bad("queue needs lo and hi")
        p, entry = find_source(pid, sid)
        if p not in media.proxies_pruned:
            media.prune_proxies(p, [s["path"] for s in sources.entries(p)])
        return {"status": media.prepare(p, entry["path"], window, queue=queue)}

    @served.api_route("/projects/{pid}/sources/{sid}/director/{name}", methods=["GET", "HEAD"])
    def source_director(pid: str, sid: str, name: str, request: Request):
        """A rendered director preview of a 360 moment of this source (#517)."""
        p, entry = lookups.find(pid, sid)  # one Range request of many, as source_media (#1020)
        # highlights.json and the previews listing are read when they change (#1029)
        ready = previews.ready(p).values()
        file = None
        if any(src == entry["path"] and n == name for src, n in ready):
            file = previews.file(p, name)
        if file is None:
            raise bad("director preview is not ready", 404)
        return send(file, request, proxy=True)

    def pano_project_dir(pid: str) -> Path:
        p = project_dir(pid)
        if read_meta(p).get("kind") != "360":
            raise bad("not a 360 project", 404)
        return p

    @owned.post("/projects/{pid}/director/{cid}")
    def prepare_director(pid: str, cid: str):
        """The page shows 360 candidate ``cid`` (its CLI id): plan its camera path
        (``candidates.preview_path``) and render its director preview in the background.
        ``{"pano": null}``: it is no moment of a 360 clip of the project (the page plays the raw
        preview)."""
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
        return {"pano": pano_views(pid, p, listing, cid).get(cid)}

    @owned.post("/projects/{pid}/director/{cid}/cancel")
    def cancel_director(pid: str, cid: str):
        """Stop the director preview render of ``cid``; the page plays the raw preview."""
        p = pano_project_dir(pid)
        previews.cancel(p, cid)
        return {"pano": pano_views(pid, p, sources.entries(p), cid).get(cid)}

    @served.api_route("/projects/{pid}/outputs/{name}", methods=["GET", "HEAD"])
    def output_file(pid: str, name: str, request: Request):
        p = project_dir(pid)
        if name not in output_names(p):
            raise bad("unknown output", 404)
        return send(p / name, request)

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
    def apply_ranking(pid: str, body: dict[str, Any]):
        """Re-rank the project in the library style ``style`` and save it, after the user
        clicked to use the preview made against the ranking key ``base``: highlights.json
        replaced (the one before kept, :data:`RERANK_BACKUP`) and ``style`` the project's
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
            _ranking_check(p)
        except RankingRefused as err:
            return err.response()
        for name in (rerank.BACKUP, META_FILE):
            _own(p, name)  # refuse now, not after the ranking is saved
        _ranking_load(p)  # 409 without an analysis (and for a highlights.json link)
        with jobs.lock:  # no job starts on the project while its ranking is replaced
            p = project_dir(pid)
            if p in RERANKING:
                return RankingRefused(RERANK_BUSY_TEXT, jsonout.FOLDER_BUSY).response()
            if any(j.status == "running" and j.project_id == pid for j in jobs.jobs.values()):
                return RankingRefused(BUSY_TEXT, jsonout.FOLDER_BUSY).response()
            RERANKING.add(p)
        made: list[Path] = []
        try:
            meta = read_meta(p)
            # a command-line project has no ui folder (nor meta.json) yet; 409 for a ui that is
            # no folder, before the ranking is saved
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
            new_ids, dropped = rerank.changes(before, result.after)
            return {
                "style": style.name,
                "from": fallback_style_name(before),
                "key": ranking_key(p),
                "backup": True,
                "new_ids": new_ids,
                "dropped_ids": dropped,
                "seconds": result.seconds,
            }
        finally:
            for folder in made:  # the ui folder this created for a ranking it did not save
                with contextlib.suppress(OSError):
                    folder.rmdir()
            with jobs.lock:
                RERANKING.discard(p)

    @owned.post("/projects/{pid}/ranking/undo")
    def undo_ranking(pid: str, body: dict[str, Any]):
        """Undo the last ranking change, or Redo it after an Undo (:func:`ranking_swap`), from
        the page that loaded the ranking key ``base``; the style of the ranking swapped in is the
        project's chosen style again. No job starts on the project meanwhile.

        409 with ``code``: ``stale`` (highlights.json is not the loaded one any more) or
        ``folder_busy``; 409 for a 360 project and when there is no earlier ranking that
        matches this analysis."""
        from cutpoint.errors import StaleRankingError  # noqa: PLC0415

        p = project_dir(pid)
        base = body.get("base")
        if not isinstance(base, str) or not base:
            raise bad("base must be the ranking key of the loaded project")
        try:
            _ranking_check(p)
        except RankingRefused as err:
            return err.response()
        for name in ("highlights.json", RERANK_BACKUP, META_FILE):
            _own(p, name)  # refuse now, not after the swap
        with jobs.lock:  # no job starts on the project while its ranking is swapped
            p = project_dir(pid)
            if p in RERANKING:
                return RankingRefused(RERANK_BUSY_TEXT, jsonout.FOLDER_BUSY).response()
            if any(j.status == "running" and j.project_id == pid for j in jobs.jobs.values()):
                return RankingRefused(BUSY_TEXT, jsonout.FOLDER_BUSY).response()
            RERANKING.add(p)
        made: list[Path] = []
        try:
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
            before, after, undone = swapped
            # the style the ranking swapped in runs on (auto for an auto analysis), as the
            # header shows it and the next analysis passes it; a detection kept when not Auto
            style = _analysis_inputs({}, after)[1]
            if isinstance(style, str) and style:
                det = after.get("detection") if style != AUTO else None
                set_style(p, style, det if isinstance(det, dict) else None)
            made.clear()  # the ui folder holds meta.json now
            return {
                "action": "undo" if undone else "redo",
                "style": fallback_style_name(after),
                "from": fallback_style_name(before),
                "key": ranking_key(p),
            }
        finally:
            for folder in made:  # the ui folder this created for a swap it did not make
                with contextlib.suppress(OSError):
                    folder.rmdir()
            with jobs.lock:
                RERANKING.discard(p)

    # -- assistant (#346) --------------------------------------------------------------
    # A chat turn only proposes sequence edits: the page applies one to its working sequence and
    # reports it here for the history. Nothing below writes review.json or highlights.json; GET
    # routes only read. A ranking it previews is saved only by the apply route above, which the
    # page sends when the user clicks to use it.
    turns = ai_session.Turns()
    app.state.assistant = turns

    def assistant_provider() -> ai_providers.Provider:
        try:
            return ai_providers.make()
        except (ValueError, TypeError) as err:  # a bad Ollama port, an invalid config.json
            raise bad(str(err), 409) from err

    def vision_provider() -> ai_providers.Provider:
        try:
            return ai_providers.make_vision(vision_assistant_settings())
        except (ValueError, TypeError) as err:  # a bad Ollama port, an invalid config.json
            raise bad(str(err), 409) from err

    def assistant_state(
        p: Path, segments: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], ai_ops.Durations, str]:
        """``(moments, source durations, base_key)`` of the project with ``segments``."""
        hl = _highlights(p)
        durations = ai_ops.Durations(
            {s["path"]: s["duration"] for s in sources.entries(p)},
            lambda path: media.duration(remap_source(path)),
        )
        moments = (moments_view(hl) or {}).get("candidates") or []
        return moments, durations, ai_ops.base_key(segments, hl)

    planners: dict[Path, ai_edges.EdgePlanner] = {}  # one per project: it caches what it planned

    def edge_planner(p: Path) -> ai_edges.EdgePlanner:
        return planners.setdefault(p, ai_edges.EdgePlanner(p, remap_source))

    @owned.post("/projects/{pid}/edges")
    def planned_edges(pid: str, body: dict[str, Any]):
        """Where the clips of the working sequence ``segments`` start and end once exported
        (:meth:`cutpoint.assistant.edges.EdgePlanner.plan`: Smart tails and Speech-aware cuts,
        both on unless ``smart_tail`` / ``speech_aware`` say false) and the other export settings
        the page sends as Export clips does (``event_tails``, ``event_window``, ``tail_min``,
        ``tail_max``, ``speech_max_extend``, ``speech_max_lead``, ``extend_start``,
        ``extend_end``), as ``{id: {start, end, reason}}`` (``reason`` only for a range that moved:
        ``event tail``, ``smart tail``, ``speech-aware`` or ``extend``); the page previews and
        totals the sequence with them. Writes nothing."""
        p = project_dir(pid)
        flags = {"smart_tail": True, "speech_aware": True, **body}
        segments = validate_segments(body.get("segments"), sources.entries(p), media)
        event_tails = body.get("event_tails", "auto")
        if not isinstance(event_tails, str) or event_tails not in EVENT_TAILS:
            raise bad("event_tails must be 'auto', 'on' or 'off'")
        options = ai_edges.ExportOptions(
            event_tails={"auto": None, "on": True, "off": False}[event_tails],
            event_window=_seconds(body, "event_window", MAX_TIMING_S, positive=True),
            tail_min=_seconds(body, "tail_min", MAX_TIMING_S),
            tail_max=_seconds(body, "tail_max", MAX_TIMING_S),
            speech_max_extend=_seconds(body, "speech_max_extend", MAX_TIMING_S),
            speech_max_lead=_seconds(body, "speech_max_lead", MAX_TIMING_S),
            extend_start=_seconds(body, "extend_start", MAX_EXTEND_S),
            extend_end=_seconds(body, "extend_end", MAX_EXTEND_S),
        )
        planned, reasons = edge_planner(p).explain(
            segments,
            smart_tail=_flag(flags, "smart_tail"),
            speech_aware=_flag(flags, "speech_aware"),
            options=options,
        )
        return {
            "edges": {
                k: {"start": a, "end": b, **({"reason": reasons[k]} if k in reasons else {})}
                for k, (a, b) in planned.items()
            }
        }

    def assistant_save(p: Path, name: str) -> None:
        """Check ``ui/assistant/<name>`` may be written and create its folder."""
        _own(p, f"{ai_store.FOLDER}/{name}")
        _ui_folder(p, "assistant")

    def assistant_turn(pid: str, tid: str) -> ai_session.Turn:
        turn = turns.get(str(project_dir(pid)), tid)
        if turn is None:
            raise bad("unknown turn", 404)
        return turn

    def assistant_revert(p: Path, hid: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """``(history entry, its "before" checked again)``: what Revert puts back."""
        entry = next((e for e in ai_store.history(p) if e.get("id") == hid), None)
        if entry is None:
            raise bad("unknown change", 404)
        try:
            return entry, validate_segments(entry.get("before"), sources.entries(p), media)
        except HTTPException as err:
            raise bad(f"this change can no longer be reverted: {err.detail}", 409) from err

    @owned.get("/projects/{pid}/assistant")
    def get_assistant(pid: str):
        """The provider in use, whether the user must agree before it gets project text
        (``consent``), whether its visual search can run (``frame_index``: ``ok``, else ``why``
        and ``fix``, :func:`cutpoint.assistant.lookfor.index_state`), the saved chat and the
        turn still running, if any."""
        p = project_dir(pid)
        running = [t.view() for t in turns.of(str(p)) if t.status == "running"]
        provider = assistant_provider()
        return {
            "provider": ai_providers.describe(provider),
            "consent": consent_view(provider),
            "frames": frames_state(p, provider),
            "frame_index": ai_lookfor.index_state([s["path"] for s in sources.entries(p)]),
            "turns": ai_store.turns(p),
            "running": running,
            # the files the chat may name: the page links a reply naming one (Open folder)
            "files": [{"name": f["name"], "target": f["target"]} for f in chat_files(p)],
        }

    def frames_state(p: Path, provider: ai_providers.Provider) -> dict[str, Any]:
        """Whether video frames of ``p`` may go to ``provider`` (:func:`ai_watch.frames_view`);
        an unreadable config.json allows none."""
        try:
            data = load_config_data()
        except ValueError:
            data = {}
        return ai_watch.frames_view(provider, p, data)

    def frames_given(p: Path) -> tuple[str, ...]:
        try:
            return ai_watch.frames_consent(load_config_data(), p)
        except ValueError:  # config.json became unreadable: no frames go out
            return ()

    @owned.post("/projects/{pid}/assistant/frames-consent")
    def give_frames_consent(pid: str, body: dict[str, Any]):
        """Allow video frames of this project to go to the assistant's provider, named by its
        ``key`` (``frames-v1:<name>@<host>``), kept in ``assistant.frames_consent`` of
        ``config.json`` under the project folder; 409 when the provider in use is another one
        by now or is on this machine, or the user has not agreed to the assistant itself."""
        p = project_dir(pid)
        provider = assistant_provider()
        want = ai_watch.frames_key(provider)
        if want is None or body.get("key") != want:
            raise bad("the assistant's provider changed: check it and allow again", 409)
        if consent_view(provider)["needed"]:
            raise bad(f"agree first to use {provider.host} in AI Chat", 409)
        where = ai_watch.project_key(p)

        def allow(section: dict[str, Any]) -> dict[str, Any] | None:
            given = section.get("frames_consent", {})
            if not isinstance(given, dict):
                raise bad("assistant.frames_consent in config.json must be an object", 409)
            keys_now = list(ai_watch.frames_consent({"assistant": section}, p))
            if want in keys_now:
                return None
            return {**section, "frames_consent": {**given, where: [*keys_now, want]}}

        try:
            update_section("assistant", allow)  # re-read under the config lock
        except ValueError as err:
            raise bad(str(err), 409) from err
        except OSError as err:
            raise bad(f"could not save the consent: {err.strerror or err}", 500) from err
        return {"frames": frames_state(p, provider)}

    @owned.post("/projects/{pid}/assistant/turns")
    def start_turn(pid: str, body: dict[str, Any]):
        """Ask the assistant ``text`` about the working sequence ``segments`` (default: the
        saved review). Poll the turn for its events. A hosted provider the user has not agreed
        to send project text to (``POST /api/assistant/consent``) gets nothing: 409."""
        p = project_dir(pid)
        provider = assistant_provider()
        if consent_view(provider)["needed"]:
            raise bad(
                f"agree first to send this project's analysis results (moment labels, scores "
                f"and signals, vision descriptions, transcript snippets, file names and times) "
                f"and, when analyzing, the whole transcripts of its footage to {provider.host}",
                409,
            )
        text = body.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > ai_session.MAX_TEXT:
            raise bad(f"text must be 1-{ai_session.MAX_TEXT} characters")
        raw = body.get("segments")
        if raw is None:
            review = _load(p, f"{UI_DIR}/review.json")
            raw = (review.get("segments") if isinstance(review, dict) else None) or []
        segments = validate_segments(raw, sources.entries(p), media)
        moments, durations, key = assistant_state(p, segments)
        transcript = ""
        if _inside(p, "transcript.txt"):
            with contextlib.suppress(OSError, UnicodeDecodeError):
                transcript = (p / "transcript.txt").read_text(encoding="utf-8-sig")
        # the events VLM windows: what the vision model saw, which find_moments searches
        windows = ai_context.vlm_windows(_load(p, "events.json"))
        # preview_ranking: an analyzed project that is not 360, ranked in its recorded style
        meta, hl = read_meta(p), _highlights(p)
        ranking = None
        if hl and meta.get("kind") != "360" and not (p / pano_project.PANO_FILE).is_file():
            current = hl.get("style") or _analysis_inputs(meta, hl)[1] or ""
            ranking = RankingHook(p, str(current))
        project = ai_session.Project(
            moments,
            segments,
            durations,
            transcript,
            key,
            windows,
            edge_planner(p),
            frame_index=ai_lookfor.indexed(list(durations)),  # no frames, no look_for
            index_state=ai_lookfor.index_state(list(durations)),  # why not, in the rules
            words=edge_planner(p).words,  # find_words: the cached transcript words
            ranking=ranking,
            ranking_key=ranking_key(p),  # kept in the turn: later turns know when ids changed
            files=chat_files(p),  # names only
            folder=str(p),  # only a model on this machine is given it (Turn.folder)
        )
        if ai_providers.is_subscription(provider) and (why := proxy_error(logged_in_only=True)):
            raise bad(why, 503)
        assistant_save(p, ai_store.CHAT)
        viewer = ai_watch.Viewer(p, provider, resolve=remap_source, consent=lambda: frames_given(p))
        turn = ai_session.Turn(provider, project, text.strip(), ai_store.turns(p), viewer=viewer)
        try:
            turns.start(str(p), turn, lambda t: ai_store.add_turn(p, t.record()))
        except ai_session.BusyError as err:
            raise bad(str(err), 409) from err
        return turn.view()

    @owned.get("/projects/{pid}/assistant/turns/{tid}")
    def poll_turn(pid: str, tid: str, after: int = 0):
        """The turn with its events from number ``after`` on."""
        return assistant_turn(pid, tid).view(after)

    @owned.post("/projects/{pid}/assistant/turns/{tid}/cancel")
    def cancel_turn(pid: str, tid: str):
        turn = assistant_turn(pid, tid)
        turn.cancel()
        return turn.view()

    @owned.post("/projects/{pid}/assistant/proposals/{prop}/applied")
    def proposal_applied(pid: str, prop: str, body: dict[str, Any]):
        """Record that the page applied proposal ``prop`` to its working sequence ``segments``
        (as it was before). When those or the analysis changed since the proposal was made, its
        ops are applied again to them; the result's ``after`` is the sequence to show.
        ``revert-<id>`` (from ``history/<id>/revert``) puts back the "before" of applied change
        ``<id>``; ``base_key`` (from that preview) must still match ``segments``, else 409.
        ``color`` is the page's Color now (``{mode, strength}``); a proposal with a color op
        (or a revert of one) records it as ``before`` with the Color it sets as ``after``."""
        p = project_dir(pid)
        now = _color_state(body.get("color"))
        if prop.startswith(REVERT):
            undone, back = assistant_revert(p, prop.removeprefix(REVERT))
            found = (undone.get("turn"), {"summary": f"Revert: {undone.get('summary', '')}"})
            color = (undone.get("color") or {}).get("before")
        else:
            back = None
            found = next(
                ((t.id, hit) for t in turns.of(str(p)) if (hit := t.proposal(prop)) is not None),
                None,
            )
            if found is None and (saved := ai_store.find_proposal(p, prop)) is not None:
                found = (saved[0].get("id"), saved[1])
            color = (found[1] if found else {}).get("color")
        if found is None:
            raise bad("unknown proposal", 404)
        tid, proposal = found
        before = validate_segments(body.get("segments"), sources.entries(p), media)
        moments, durations, key = assistant_state(p, before)
        after, changes, rebased = proposal.get("after"), proposal.get("diff"), False
        if back is not None:
            if body.get("base_key") != key:
                raise bad(
                    "the sequence changed since this revert was previewed: preview it again", 409
                )
            after, changes = back, ai_ops.diff(before, back)
        elif key != proposal.get("base_key"):
            # look_for moments are not in highlights.json: the proposal carries them
            known = [*moments, *ai_ops.extra_moments(proposal.get("moments"))]
            out = ai_ops.apply_ops(
                before, proposal.get("ops"), known, durations, snap=edge_planner(p).phrase_end
            )
            if out.error:
                raise bad(f"the proposal no longer applies to this sequence: {out.error}", 409)
            after, changes, rebased = out.after, out.diff, True
        entry = {
            "id": secrets.token_hex(6),
            "proposal": prop,
            "turn": tid,
            "summary": proposal.get("summary", ""),
            "applied": datetime.now().isoformat(timespec="seconds"),
            "base_key": key,
            "rebased": rebased,
            "before": before,
            "after": after,
            "diff": changes,
        }
        if isinstance(color, dict) and color.get("mode") in COLOR_MODES:
            entry["color"] = {"before": now, "after": _color_after(color, now)}
        assistant_save(p, ai_store.HISTORY)
        ai_store.add_applied(p, entry)
        return entry

    @owned.delete("/projects/{pid}/assistant/chat")
    def clear_chat(pid: str):
        """Forget the chat and stop a running turn; the history of applied proposals stays."""
        p = project_dir(pid)
        turns.clear(str(p))
        if (p / ai_store.FOLDER / ai_store.CHAT).exists():
            assistant_save(p, ai_store.CHAT)
            ai_store.clear_chat(p)
        return {"turns": []}

    @owned.get("/projects/{pid}/assistant/history")
    def get_history(pid: str):
        return {"applied": ai_store.history(project_dir(pid))}

    # -- delete a project (Settings) ---------------------------------------------------
    def delete_blocker(pid: str, p: Path) -> str | None:
        """Why ``p`` cannot be deleted now (a job, a run, an encode or a chat turn on it)."""
        if any(j.status == "running" and j.project_id == pid for j in list(jobs.jobs.values())):
            return "A job is running on this project; wait for it to finish or cancel it"
        if project_busy(p, run_state(p, True)) or projectdelete.locked(p):
            return "A Cutpoint run holds this project; wait for it to finish"
        if p in RERANKING:
            return "A ranking is being applied to this project; wait a moment"
        with previews.lock:
            if any(v is not None and v.project == p for v in (previews.running, *previews.queue)):
                return "A 360 preview is rendering for this project; wait for it to finish"
        if media.encoding_for(p) or any(t.status == "running" for t in turns.of(str(p))):
            return "The project is still preparing videos or answering in AI Chat; wait a moment"
        return None

    @owned.get("/projects/{pid}/delete-plan")
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

    @owned.delete("/projects/{pid}")
    def delete_project(pid: str, body: dict[str, Any] | None = None):
        """Delete the project's own files (:mod:`cutpoint.projectdelete`) to the Recycle Bin /
        Trash, or for good with ``permanent``. 409 with ``code`` ``no_trash`` when the trash
        cannot be trusted with them, for the page to ask before it sends ``permanent``."""
        permanent = _flag(body or {}, "permanent")
        project_dir(pid)
        protected = protected_sources()  # before the lock: a slow path must not hold it
        with jobs.lock:  # no job starts on the project while it is deleted
            p = project_dir(pid)  # again: a delete that ran first may have removed it
            DELETING.add(p)  # every other route on the project answers 409 meanwhile
            try:
                if why := delete_blocker(pid, p):
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
                        Path(f[0] if isinstance(f, (list, tuple)) and f else str(f)).name
                        if f
                        else "a file"
                    )
                    raise bad(f"A file is in use ({name}); close it and try again.", 409) from err
                except OSError as err:
                    logger.warning("deleting %s failed: %s", p, err)
                    raise bad(f"could not delete the project: {err.strerror or err}", 500) from err
                projectdelete.remove_folder(p)
            finally:
                lookups.forget(p)  # before the 409 ends: the media routes check it again
                DELETING.discard(p)
        media.forget(p)
        turns.clear(str(p))
        return {"deleted": pid, "kept": len(plan.kept)}

    @owned.post("/projects/{pid}/assistant/history/{hid}/revert")
    def revert_change(pid: str, hid: str, body: dict[str, Any]):
        """A proposal that puts the working sequence ``segments`` back as it was before applied
        change ``hid``, to preview; applying it goes through ``proposals/<id>/applied`` with the
        ``base_key`` returned here. Writes nothing."""
        p = project_dir(pid)
        entry, back = assistant_revert(p, hid)
        now = validate_segments(body.get("segments"), sources.entries(p), media)
        changes = ai_ops.diff(now, back)
        color = (entry.get("color") or {}).get("before")
        if color is not None and color == _color_state(body.get("color")):
            color = None  # the Color is already as it was
        if color is None and not any(row["changes"] for row in changes):
            raise bad("the sequence is already as it was before this change", 409)
        return {
            "id": f"{REVERT}{hid}",
            "summary": f"Revert: {entry.get('summary', '')}",
            "after": back,
            "diff": changes,
            **({"color": color} if color is not None else {}),
            "base_key": assistant_state(p, now)[2],
        }

    api.include_router(secure)
    api.include_router(owned)
    api.include_router(served)
    app.include_router(api)
    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def quiet_resets(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    """Event loop exception handler: drop connection resets, pass the rest to the default.

    A browser that cancels a video range request (a seek, another source) resets the connection;
    on Windows the proactor then logged an ERROR traceback (WinError 10054) for each one.
    """
    if isinstance(context.get("exception"), ConnectionResetError | ConnectionAbortedError):
        return
    loop.default_exception_handler(context)


class _Hangup:
    """SIGHUP handler (the terminal closed): stop the server as Ctrl+C does, then exit 1.

    Raising SystemExit in the handler while uvicorn handled a request did not stop it: uvicorn
    caught it as an app error and kept serving, its jobs and ffmpeg children too. A running
    server is only asked to stop; run() raises SystemExit(1) once it has.
    """

    def __init__(self) -> None:
        self.server: Any = None  # the uvicorn.Server, once there is one
        self.received = False

    def __call__(self, *_: object) -> None:
        self.received = True
        if self.server is None:
            raise SystemExit(1)  # unwinds through run()'s finally, which stops the children
        self.server.should_exit = True


def listen(port: int) -> socket.socket:
    """A listening socket on 127.0.0.1:``port`` that no other program shares.

    Raises:
        RuntimeError: If the port is already in use.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt":  # SO_REUSEADDR there would let two servers share the port
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:  # as uvicorn does: restart at once despite TIME_WAIT; a listener still blocks it
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(2048)
    except OSError as err:
        sock.close()
        # Windows reports a port held exclusively (or reserved) as WSAEACCES.
        if err.errno in {errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", None)} or (
            os.name == "nt" and err.errno == getattr(errno, "WSAEACCES", None)
        ):
            other = port + 1 if port < 65535 else port - 1
            raise RuntimeError(
                f"Port {port} is already in use on 127.0.0.1; pass --port <n> "
                f"(for example --port {other})"
            ) from None
        raise
    return sock


def run(port: int = 8765) -> None:
    """Serve the workspace on 127.0.0.1:``port`` until interrupted."""
    import uvicorn  # noqa: PLC0415

    ai_keys.capture()  # no child process of this one inherits an API key variable
    app = create_app(port)
    sock = listen(port)
    # Printed only once the port is ours, so the URL always works.
    # The fragment never reaches the network; the page keeps the token in sessionStorage.
    print(f"Cutpoint UI: http://127.0.0.1:{port}/#t={app.state.token} (Ctrl+C to stop)", flush=True)
    hangup = _Hangup()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, hangup)
    try:
        conf = uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="warning", timeout_graceful_shutdown=5
        )
        ai_keys.redact_logs()  # again: the Config set up uvicorn's own log handlers
        server = uvicorn.Server(conf)
        hangup.server = server
        server.run(sockets=[sock])
        if hangup.received:
            raise SystemExit(1)  # through the finally below, which stops the children
    finally:  # jobs and ffmpeg run in their own sessions: nothing else would stop them
        sock.close()
        app.state.jobs.shutdown()
        app.state.media.shutdown()
        app.state.picker.shutdown()
        app.state.assistant.shutdown()
        app.state.proxies.shutdown()  # only the proxy this UI started; a user's stays
