"""Director previews of 360 moments for ``cutpoint ui``."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from cutpoint import cache, folders
from cutpoint.errors import FFmpegError
from cutpoint.ffmpeg_utils import ChildScope, child_scope
from cutpoint.pano import project as pano_project
from cutpoint.ui.common import SOURCES_CACHE_MAX, SOURCES_CACHE_S, UI_DIR, finite_number, logger
from cutpoint.ui.files import _inside, _own, _ui_folder
from cutpoint.ui.lanes import Lane
from cutpoint.ui.media import PROXY_PRUNE_GRACE_S, Media, _idle, _mark_used
from cutpoint.utils import file_stamp

# --------------------------------------------------------------------------- 360 director previews
# A 360 moment previews through the camera path cutpoint.pano.candidates.preview_path gives for
# it (#517): the one ``360 analyze`` cached, one planned now as ``clips`` would, or a level
# forward view when the director cannot plan (never the raw fisheye). A small flat render of the
# .lrv along that path, as ``360 director --preview`` makes, lands in
# ui/pano/<candidate id>-<key>.mp4. The key is ``candidates.preview_key``: it covers the path,
# the render format and the pano settings the render reads (``imu_offset_s`` and the levelling
# ones), so a re-planned path or a changed render is never served from an older file. The render
# spans what the path was planned for: the moment and up to 10 s on either side.
PANO_PREVIEW_W, PANO_PREVIEW_H = 640, 360
PANO_PREVIEW_NAME = re.compile(r"[A-Za-z0-9_]{1,40}-[0-9a-f]{16}")
# a preview, its temporary (folders.part_path) or the sendcmd file render_segment leaves
PANO_PREVIEW_FILE = re.compile(r"\.?([A-Za-z0-9_]{1,40}-[0-9a-f]{16})(?:\.part)?\.(?:mp4|cmd)")


def _pano_format() -> Any:
    from cutpoint.pano import render as pano_render  # noqa: PLC0415

    return pano_render.Format360(PANO_PREVIEW_W, PANO_PREVIEW_H)


def _pano_settings() -> dict[str, Any]:
    """The pano settings of config.json (``pano_project.pano_config``), {} when unreadable."""
    try:
        return pano_project.pano_config()
    except ValueError:  # config.json unreadable: the render fails and says so
        return {}


def _pano_key(path: dict[str, Any], settings: dict[str, Any]) -> str:
    """``candidates.preview_key`` of a 16:9 preview along ``path`` in the UI's format with
    ``settings`` (:func:`_pano_settings`). Of the empty path ``{}``: what a preview renders
    with besides its path."""
    from cutpoint.pano.candidates import preview_key  # noqa: PLC0415

    return preview_key(path, vertical=False, fmt=_pano_format(), settings=settings)


def _pano_name(cid: str, key: str) -> str:
    """File stem of the preview of candidate ``cid`` whose :func:`_pano_key` is ``key``."""
    return f"{_pano_safe(cid)}-{key}"


def _pano_safe(cid: str) -> str:
    """The file-name prefix of candidate ``cid``'s previews."""
    return re.sub(r"[^A-Za-z0-9_]", "_", cid)[:40] or "c"


def _pano_span(path: dict[str, Any], clip: Any) -> tuple[float, float] | None:
    """The source range a preview of ``path`` covers: its planned span, starting on the frame at
    or before its start (``render_segment`` keeps a start on a frame, so the file's 0 is ``lo``;
    a path holds its first view before its planned span) and ending with the clip at the latest.
    A director path and a level forward view are both planned with the director's margin
    (``director.planned_span``): a start rounded up would miss the path's first frame."""
    planned = path.get("planned") or path.get("segment") or {}
    try:
        lo, hi = float(planned["start"]), float(planned["end"])
    except (KeyError, TypeError, ValueError):
        return None
    fps = clip.fps or 30000 / 1001
    lo = math.floor(max(0.0, lo) * fps + 1e-6) / fps
    if clip.duration > 0:
        hi = min(hi, float(clip.duration))
    return (lo, hi) if hi - lo >= 0.1 else None  # unrounded: render_segment rounds lo again


def _pano_reasons(
    flags: list[str], reasons: dict[str, str], diag: dict[str, Any]
) -> list[dict[str, str]]:
    """Each flag with its reason (``candidate_flags``' ``reasons``) and the measure behind it
    (its ``diagnostics``)."""
    from cutpoint.pano.candidates import SPIN_WINDOW_S  # noqa: PLC0415

    def num(key: str) -> float | None:
        return finite_number(diag.get(key))

    def share(key: str) -> str:
        v = num(key)
        return f" ({round(v * 100)}% of the moment)" if v is not None else ""

    out = []
    for f in flags:
        text = reasons.get(f, f)
        if f == "sky":
            text += share("sky")
        elif f == "wearer":
            text += share("wearer_in_view")
        elif f == "off_target":
            text += share("off_target")
        elif f == "whip" and (rate := num("peak_yaw_rate")) is not None:
            text += f" (up to {rate:g} deg/s)"
        elif f == "spin" and (turns := num("net_turns")) is not None:
            per_min = num("yaw_travel_per_min")
            pan = f", {per_min:g} deg of pan per minute" if per_min is not None else ""
            win = min(SPIN_WINDOW_S, num("seconds") or SPIN_WINDOW_S)  # a short moment: all of it
            text += f" (up to {turns:g} turns one way in {win:g} s{pan})"
        elif f == "cuts" and (n := num("cuts")) is not None:
            text += f" ({n:g} in the moment)"
        elif f == "qa_fail":
            fails = (diag.get("quality") or {}).get("fails")
            if isinstance(fails, list) and fails:
                text += ": " + ", ".join(str(x) for x in fails[:4])
        out.append({"flag": f, "text": text})
    return out


def _file_ident(f: Path) -> tuple[str, int, int] | None:
    stamp = file_stamp(f)
    return None if stamp is None else (f.name, stamp[1], stamp[0])


class PanoPreview:
    """One queued or running director preview: plan its path, then render it."""

    def __init__(self, project: Path, cid: str, info: dict[str, Any]) -> None:
        self.project, self.cid, self.ident = project, cid, info["ident"]
        self.clip, self.source = info["clip"], info["source"]
        self.start, self.end = info["start"], info["end"]
        self.scope = ChildScope()  # kill() stops its ffmpeg; the render removes its .part
        self.cancelled = False  # superseded, cancelled or shutting down (under PanoPreviews.lock)
        self.started: float | None = None
        self.name: str | None = None  # once its path is planned (under PanoPreviews.lock)
        self.lo = self.hi = 0.0
        self.progress: float | None = None  # rendered share, 0-1, once it renders

    @property
    def key(self) -> tuple[Path, str]:
        return self.project, self.cid


class PanoPreviews:
    """Director previews of 360 moments (#517): flags of the cached camera paths for the
    listing, and on request a render along ``candidates.preview_path`` in one background lane.
    A new request supersedes the queued ones and stops a running render of another moment (as
    window proxies do); a stopped render leaves no partial file."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # the one render worker; ``run`` looks ``_render`` up per job, and the lane outlives
        # a render that raises past its own handlers
        self.lane: Lane[PanoPreview] = Lane(
            self.lock,
            lambda job: self._render(job),
            name="cutpoint-pano",
            started=self._started,
            ended=self._ended,
            failed=lambda job: logger.exception("director preview lane: %s failed", job.cid),
        )
        self.closing = False
        # (project, cid) -> {"state": failed | cancelled, "reason", "ident"}
        self.outcomes: dict[tuple[Path, str], dict[str, Any]] = {}
        # (project, cid) -> {"ident", "name", "lo", "hi", "kind", "reason"} of the preview made
        # (or found on disk) this run: only that one is listed ready, and only for its ident
        self.done: dict[tuple[Path, str], dict[str, Any]] = {}
        # project -> (signature of its candidate cache and render settings, {cid: entry info})
        self.memo: dict[Path, tuple[Any, dict[str, dict[str, Any]]]] = {}
        self.pruned: set[Path] = set()
        # bumped (under the lock) by each change of the queue, the running render, ``done`` or
        # ``outcomes``: what :meth:`ready` keeps is valid for one value
        self.rev = 0
        # project -> (built at, rev and file stamps it was built from, {cid: (source, name)}),
        # least recently used first
        self.ready_memo: dict[Path, tuple[float, Any, dict[str, tuple[str, str]]]] = {}

    # -- the listing ------------------------------------------------------------------
    @staticmethod
    def _signature(project: Path, settings: str) -> Any:
        from cutpoint.pano.candidates import CACHE_DIR  # noqa: PLC0415

        files = [i for f in (project / CACHE_DIR).glob("*.json") if (i := _file_ident(f))]
        return (
            _file_ident(project / "highlights.json"),
            _file_ident(project / pano_project.PANO_FILE),
            tuple(sorted(files)),
            settings,
        )

    def entries(self, project: Path) -> dict[str, dict[str, Any]]:
        """``{cid: {flags, reasons, name, ident, clip, source, start, end}}`` of the candidates
        of the project's 360 clips. ``flags`` and ``reasons`` come from
        ``candidates.candidate_flags``; ``name`` is the preview file stem of the cached path
        (:func:`_pano_name`), else None; ``ident`` names what a preview is made of (the
        candidate's span, its cached path, the render settings): when it changes, the page asks
        again. Rebuilt only when ``highlights.json``,
        ``pano.json``, a cache file or the render settings changed. No freshness check
        (``preview_path`` decides that)."""
        from cutpoint.pano import candidates as PC  # noqa: PLC0415

        settings = _pano_settings()
        render = _pano_key({}, settings)
        sig = self._signature(project, render)
        with self.lock:
            hit = self.memo.get(project)
        if hit is not None and hit[0] == sig:
            return hit[1]
        out: dict[str, dict[str, Any]] = {}
        try:
            flagged = PC.candidate_flags(project)
            clips = pano_project.load(project)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as err:
            logger.warning("camera paths of %s unreadable: %s", project.name, err)
            flagged, clips = {}, {}
        hl = cache.load_json(project / "highlights.json")
        for c in hl.get("candidates") or [] if isinstance(hl, dict) else []:
            if not isinstance(c, dict) or not isinstance(c.get("source"), str):
                continue
            try:
                start, end = float(c["start"]), float(c["end"])
            except (KeyError, TypeError, ValueError):
                continue
            clip = pano_project.clip_for(clips, c["source"]) if clips else None
            if clip is None:
                continue
            cid = str(c.get("id"))
            info = flagged.get(cid) or {}
            flags = list(info.get("flags") or [])
            entry = cache.load_json(info["file"]) if info.get("file") else None
            cached = entry.get("path") if isinstance(entry, dict) else None
            key = _pano_key(cached, settings) if isinstance(cached, dict) else None
            name = _pano_name(cid, key) if key else None
            blob = json.dumps([c["source"], start, end, name, render])
            out[cid] = {
                "flags": flags,
                "reasons": _pano_reasons(
                    flags, info.get("reasons") or {}, info.get("diagnostics") or {}
                ),
                "name": name,
                "ident": f"{cid}-{hashlib.sha1(blob.encode()).hexdigest()[:16]}",
                "clip": clip,
                "source": c["source"],
                "start": start,
                "end": end,
            }
        with self.lock:
            self.memo[project] = (sig, out)
        return out

    def listing(self, project: Path) -> dict[str, dict[str, Any]]:
        """Per candidate id of a 360 clip: ``flags``, ``reasons``, ``path`` (what its preview
        is made of: a new value is asked for again) and ``preview`` (None until asked for; else
        ``state`` ready (``name``, ``lo``, ``hi``, and ``preview_path``'s ``kind`` and
        ``reason``) | rendering (``elapsed``; ``seconds`` and ``progress`` 0-1, None while the
        path is planned) | queued | failed | cancelled, with a ``reason`` for the last two)."""
        return {
            cid: {
                "flags": info["flags"],
                "reasons": info["reasons"],
                "path": info["ident"],
                "preview": self._state(project, cid, info),
            }
            for cid, info in self.entries(project).items()
        }

    def ready(self, project: Path) -> dict[str, tuple[str, str]]:
        """``{cid: (source path, preview name)}`` of the previews :meth:`listing` has ready, for
        the director route's many Range requests (#1029). Kept while ``highlights.json``,
        ``pano.json``, the camera path cache folder and the previews folder (a preview written or
        removed changes it) keep their stamps, the render settings and :attr:`rev` stay, and for
        at most ``SOURCES_CACHE_S`` (what a stamp cannot see); at most ``SOURCES_CACHE_MAX``
        projects. Callers must not change the answer."""
        from cutpoint.pano.candidates import CACHE_DIR  # noqa: PLC0415

        # stamped (and rev read) before the listing is built: a change made meanwhile shows next time
        with self.lock:
            rev, hit = self.rev, self.ready_memo.get(project)
        sig = (
            rev,
            _pano_key({}, _pano_settings()),
            file_stamp(project / "highlights.json"),
            file_stamp(project / pano_project.PANO_FILE),
            file_stamp(project / CACHE_DIR),
            file_stamp(project / UI_DIR / "pano"),
        )
        now = time.monotonic()
        if hit is not None and hit[1] == sig and now - hit[0] < SOURCES_CACHE_S:
            with self.lock:
                if self.ready_memo.get(project) is hit:
                    self.ready_memo[project] = self.ready_memo.pop(project)
            return hit[2]
        entries = self.entries(project)  # one snapshot: states come from these entries
        out = {
            cid: (info["source"], st["name"])
            for cid, info in entries.items()
            if (st := self._state(project, cid, info)) is not None and st["state"] == "ready"
        }
        with self.lock:
            self.ready_memo.pop(project, None)
            self.ready_memo[project] = (now, sig, out)
            while len(self.ready_memo) > SOURCES_CACHE_MAX:
                del self.ready_memo[next(iter(self.ready_memo))]
        return out

    def _state(self, project: Path, cid: str, info: dict[str, Any]) -> dict[str, Any] | None:
        key = (project, cid)
        if not info["clip"].lrv:
            return {"state": "failed", "reason": "no .lrv proxy to render this camera path from"}
        with self.lock:
            run = self.lane.running
            if run is not None and run.key == key and not run.cancelled:
                elapsed = round(time.time() - run.started, 1) if run.started else 0.0
                planned = run.name is not None
                return {
                    "state": "rendering",
                    "elapsed": elapsed,
                    "seconds": round(run.hi - run.lo, 1) if planned else None,
                    "progress": None if run.progress is None else round(run.progress, 3),
                }
            if any(j.key == key and not j.cancelled for j in self.lane.queue):
                return {"state": "queued"}
            failed = self.outcomes.get(key)
            done = self.done.get(key)
        if failed is not None and failed["ident"] == info["ident"]:
            return {"state": failed["state"], "reason": failed["reason"]}
        if (
            done is not None
            and done["ident"] == info["ident"]
            and self.file(project, done["name"], mark=False) is not None
        ):
            return {
                "state": "ready",
                "name": done["name"],
                "lo": round(done["lo"], 6),
                "hi": round(done["hi"], 6),
                "kind": done["kind"],
                "reason": done["reason"],
            }
        return None

    def file(self, project: Path, name: str, *, mark: bool = True) -> Path | None:
        """The rendered preview ``name`` of ``project``, else None; ``mark``: it is being served."""
        if not PANO_PREVIEW_NAME.fullmatch(name):
            return None
        rel = f"{UI_DIR}/pano/{name}.mp4"
        f = project / rel
        if not f.is_file() or not _inside(project, rel):
            return None
        if mark:
            _mark_used(f)
        return f

    # -- requests ---------------------------------------------------------------------
    def request(self, project: Path, cid: str) -> dict[str, Any] | None:
        """The page shows candidate ``cid``: queue its preview (plan the path with
        ``candidates.preview_path``, then render it unless that file exists) unless one is
        ready or on its way; a failed or cancelled one is tried again. Returns the candidate's
        listing entry, or None when it is no moment of a 360 clip of the project."""
        info = self.entries(project).get(cid)
        if info is None:
            return None
        view = self.listing(project).get(cid)  # None: highlights.json changed meanwhile
        if view is None:
            return None
        state = (view["preview"] or {}).get("state")
        if info["clip"].lrv and state not in ("ready", "rendering", "queued"):
            with self.lock:
                self.outcomes.pop((project, cid), None)
                self.rev += 1
            self._queue(PanoPreview(project, cid, info))
            view = self.listing(project).get(cid)
        return view

    def _queue(self, job: PanoPreview) -> None:
        stop = None
        with self.lock:
            if self.closing:
                return
            for queued in self.lane.queue:
                queued.cancelled = True  # superseded: the user moved on (no outcome to show)
            self.lane.queue.clear()
            self.lane.queue.append(job)
            self.rev += 1
            run = self.lane.running
            if run is not None and not run.cancelled:
                run.cancelled = True
                stop = run.scope
            self.lane.wake()
        if stop is not None:
            stop.kill()

    def cancel(self, project: Path, cid: str) -> bool:
        """Stop the queued or running preview of ``cid`` (its partial file goes); False if none."""
        key, stop, ident = (project, cid), None, None
        with self.lock:
            for job in [j for j in self.lane.queue if j.key == key]:
                job.cancelled = True
                self.lane.queue.remove(job)
                ident = job.ident
            run = self.lane.running
            if run is not None and run.key == key and not run.cancelled:
                run.cancelled = True
                stop, ident = run.scope, run.ident
            if ident is not None:
                self.outcomes[key] = {"state": "cancelled", "reason": "", "ident": ident}
            self.rev += 1
        if stop is not None:
            stop.kill()
        return ident is not None

    def _started(self, job: PanoPreview) -> None:  # the lane took ``job``, under the lock
        self.rev += 1
        job.started = time.time()

    def _ended(self, _job: PanoPreview) -> None:  # ``job`` ended, under the lock
        self.rev += 1

    def _render(self, job: PanoPreview) -> None:
        """Plan ``job``'s path (``preview_path``) and render it from the clip's .lrv
        (``render_segment``, as ``360 director --preview``) unless that preview exists; on
        failure or a stop no file is left but the reason (unless stopped)."""
        from cutpoint.pano import candidates as PC  # noqa: PLC0415
        from cutpoint.pano import render as pano_render  # noqa: PLC0415

        p = job.project
        what = f"{job.cid} ({job.start:.2f}-{job.end:.2f} s)"
        out: Path | None = None
        plan: dict[str, Any] | None = None
        failed = None

        def report(frac: float) -> None:  # render_segment's progress: rising, then one 1.0
            with self.lock:
                job.progress = max(job.progress or 0.0, min(1.0, frac))

        try:
            _ui_folder(p, "pano")  # ui or ui/pano a file: a clear refusal, not a traceback
            with child_scope(job.scope):
                job.scope.check()  # cancelled before it started
                plan = PC.preview_path(p, job.source, job.start, job.end)
                job.scope.check()  # cancelled while its path was planned
                name = _pano_name(job.cid, _pano_key(plan["path"], _pano_settings()))
                span = _pano_span(plan["path"], job.clip)
                if span is None:
                    raise ValueError("the camera path spans no video")
                with self.lock:
                    job.name, (job.lo, job.hi) = name, span
                if plan["kind"] != "cached":
                    logger.info(
                        "director preview of %s: %s, %s", what, plan["kind"], plan["reason"]
                    )
                out = p / UI_DIR / "pano" / f"{name}.mp4"
                if self.file(p, name) is None:
                    _own(p, f"{UI_DIR}/pano/{out.name}")  # never rendered through a link
                    _own(p, f"{UI_DIR}/pano/{folders.part_path(out).name}")
                    _own(p, f"{UI_DIR}/pano/{name}.cmd")  # render_segment writes it beside it
                    src = pano_project.RenderSource(
                        "proxy", (job.clip.lrv,), "dfisheye", False, True, "lrv proxy"
                    )
                    t0 = time.perf_counter()
                    pano_render.render_segment(
                        job.clip, plan["path"], span[0], span[1], out,
                        fmt=_pano_format(), src=src, cmd_every=1, work_dir=out.parent,
                        progress=report,
                    )  # fmt: skip
                    logger.info("director preview of %s in %.1fs", what, time.perf_counter() - t0)
            with self.lock:
                self.rev += 1
                if not job.cancelled:
                    self.done[job.key] = {
                        "ident": job.ident,
                        "name": name,
                        "lo": span[0],
                        "hi": span[1],
                        "kind": plan["kind"],
                        "reason": plan["reason"],
                    }
        except KeyboardInterrupt:  # its scope was stopped: cancelled, superseded or shutdown
            if not job.cancelled and not self.closing:
                failed = "stopped"
        except HTTPException as err:
            failed = str(err.detail)
        except Exception as err:  # a failed preview never stops the lane
            lines = str(err).strip().splitlines()
            failed = (lines[-1] if lines else type(err).__name__)[:200]
            if plan is not None and plan["kind"] == "level_forward" and plan["reason"]:
                failed = f"{plan['reason']}: {failed}"  # why no director path (footage offline?)
            if not isinstance(err, (FFmpegError, OSError, ValueError)):
                logger.exception("director preview of %s failed", what)
        finally:
            if out is not None:
                # .cmd: only `360 qa` reads it
                for f in (folders.part_path(out), out.with_suffix(".cmd")):
                    with contextlib.suppress(OSError):
                        f.unlink(missing_ok=True)
        if failed is not None:
            logger.warning("director preview of %s failed: %s", what, failed)
            with self.lock:
                self.rev += 1
                if not job.cancelled:
                    self.outcomes[job.key] = {
                        "state": "failed",
                        "reason": failed,
                        "ident": job.ident,
                    }

    # -- housekeeping -----------------------------------------------------------------
    def prune(self, project: Path) -> None:
        """Once per run and project: delete previews that no candidate's cached path makes
        any more (none made this run, and none that is a candidate's newest: a path planned now
        or a level forward view is named only once planned) and leftovers of stopped renders,
        unless written or used in the last PROXY_PRUNE_GRACE_S (another run may be rendering or
        serving one)."""
        with self.lock:
            if project in self.pruned:
                return
            self.pruned.add(project)
        folder = project / UI_DIR / "pano"
        if not folder.is_dir() or not _inside(project, f"{UI_DIR}/pano"):
            return  # never delete in a folder a link takes outside the project
        entries = self.entries(project)
        keep = {i["name"] for i in entries.values() if i["name"]}
        prefixes = {_pano_safe(cid) for cid in entries}
        newest: dict[str, tuple[int, str]] = {}  # candidate prefix -> (mtime, name)
        for f in folder.glob("*.mp4"):
            prefix = f.stem.rsplit("-", 1)[0]
            if not PANO_PREVIEW_NAME.fullmatch(f.stem) or prefix not in prefixes:
                continue
            if (ident := _file_ident(f)) is not None and ident[2] > newest.get(prefix, (-1, ""))[0]:
                newest[prefix] = (ident[2], f.stem)
        keep |= {name for _, name in newest.values()}
        with self.lock:
            jobs = (self.lane.running, *self.lane.queue)
            busy = {j.name for j in jobs if j is not None and j.name}
            keep |= {d["name"] for k, d in self.done.items() if k[0] == project}
        cutoff = time.time_ns() - int(PROXY_PRUNE_GRACE_S * 1e9)
        drop = []
        for f in folder.iterdir():
            m = PANO_PREVIEW_FILE.fullmatch(f.name)
            if m is None or m[1] in busy:
                continue
            if (f.name != f"{m[1]}.mp4" or m[1] not in keep) and _idle(f, cutoff):
                drop.append(f)
        Media._delete(drop)

    def forget(self, project: Path) -> None:
        """Drop what this run knows of the project folder ``project`` (it was deleted), so a
        project made later under the same name starts clean: no failed outcome of the old one,
        and its leftover previews are pruned again."""
        with self.lock:
            self.rev += 1
            for table in (self.outcomes, self.done):
                for k in [k for k in table if k[0] == project]:
                    del table[k]
            self.memo.pop(project, None)
            self.ready_memo.pop(project, None)
            self.pruned.discard(project)

    def shutdown(self) -> None:
        with self.lock:
            self.closing = True
            jobs = [j for j in (self.lane.running, *self.lane.queue) if j is not None]
            self.lane.close()
        for job in jobs:
            job.cancelled = True
            job.scope.kill()
