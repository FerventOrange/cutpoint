"""The FastAPI app of ``cutpoint ui`` and the server that runs it."""

from __future__ import annotations

import asyncio
import base64
import errno
import hashlib
import hmac
import os
import re
import secrets
import signal
import socket
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from cutpoint.assistant import cliproxy
from cutpoint.assistant import keys as ai_keys
from cutpoint.assistant import ops as ai_ops
from cutpoint.assistant import session as ai_session
from cutpoint.assistant import store as ai_store
from cutpoint.config import remap_source
from cutpoint.ui import routes
from cutpoint.ui.common import LOOPBACK, MAX_BODY_BYTES, WEB_DIR, bad
from cutpoint.ui.files import _highlights
from cutpoint.ui.jobs import Jobs
from cutpoint.ui.media import Media
from cutpoint.ui.pano_previews import PanoPreviews
from cutpoint.ui.picker import Picker
from cutpoint.ui.projects import MediaLookups, Sources, moments_view, validate_segments
from cutpoint.ui.routes.context import AppContext
from cutpoint.ui.settings import require_config


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

    def proxy_error(*, logged_in_only: bool) -> str | None:
        """Start the subscription proxy unless one runs (:class:`cliproxy.ProxyLauncher`;
        ``logged_in_only``: only when a login is saved); why it did not start, else None."""
        try:
            app.state.proxies.ensure(logged_in_only=logged_in_only)
        except (cliproxy.ProxyError, OSError) as err:
            return ai_keys.redact(f"the subscription proxy did not start: {err}")
        return None

    turns = ai_session.Turns()
    app.state.assistant = turns

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

    def assistant_revert(p: Path, hid: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """``(history entry, its "before" checked again)``: what Revert puts back."""
        entry = next((e for e in ai_store.history(p) if e.get("id") == hid), None)
        if entry is None:
            raise bad("unknown change", 404)
        try:
            return entry, validate_segments(entry.get("before"), sources.entries(p), media)
        except HTTPException as err:
            raise bad(f"this change can no longer be reverted: {err.detail}", 409) from err

    ctx = AppContext(
        jobs=jobs,
        media=media,
        picker=picker,
        sources=sources,
        lookups=lookups,
        previews=previews,
        login_lock=login_lock,
        login_states=login_states,
        app=app,
        secure=secure,
        owned=owned,
        served=served,
        url=url,
        window_url=window_url,
        pano_views=pano_views,
        proxy_error=proxy_error,
        turns=turns,
        assistant_state=assistant_state,
        assistant_revert=assistant_revert,
    )
    routes.settings.register(ctx)
    routes.projects.register(ctx)
    routes.jobs.register(ctx)
    routes.media.register(ctx)
    routes.ranking.register(ctx)
    routes.assistant.register(ctx)
    routes.projects.register_delete(ctx)
    routes.assistant.register_revert(ctx)

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
