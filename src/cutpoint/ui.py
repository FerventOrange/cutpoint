"""``cutpoint ui``: local browser workspace for the standard-video highlight workflow.

A loopback-only FastAPI app (optional ``ui`` extra) that serves ``cutpoint/web`` and a small JSON
API over the projects folder. Heavy work (analyze, clips, assemble) runs as ordinary
``python -m cutpoint ...`` child processes, one at a time, so the browser and the CLI share one
implementation. Source videos are only ever read.

Trust boundaries:

* Every request needs a loopback ``Host`` (and ``Origin``, when sent) on the bound port.
* Every ``/api`` call needs the per-run token (``X-Cutpoint-Token``), which only the URL printed at
  startup carries (``#t=<token>``; there is no way to fetch it from the server). Media/output URLs
  carry it as ``?t=`` because ``<video>`` cannot send headers.
* Source stills (``.../sources/<id>/frame?at=<seconds>``) are media URLs too; they are cached in
  the project's ``ui/frames`` folder under a name built from the source's content key.
* A project id is a direct child folder of the projects folder. A project's media are its
  recorded source files, and only those under the source the user picked for it (or, for projects
  made on the command line, the inputs of its analysis); ``highlights.json`` alone never
  grants access to a file.
* "Show folder" (``.../reveal?target=``) opens, in the file manager and without a shell, one of
  three folders the server resolves from the project: its own sources (the default), the clips
  folder of its latest export, or the project folder holding its assembled final. The request
  names only the target, never a path.
* Choosing a source or folder (``POST /api/pick?kind=file|folder``) opens the machine's own open
  dialog (Windows: Tk's system dialog; macOS: ``osascript``; Linux: zenity, else kdialog) in a
  child process without a shell, one at a time; the server never lists folders itself. With no
  dialog (no display, Docker) the page hides those buttons and the path is typed or pasted.
* Settings (``GET``/``PUT /api/settings``) change only ``projects_dir`` in ``config.json``, never
  while a job runs or when ``CUTPOINT_PROJECTS_DIR`` sets it; existing projects are not moved.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import importlib.util
import logging
import math
import mimetypes
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cutpoint import cache, paths
from cutpoint.config import (
    load_config_data,
    projects_dir,
    remap_source,
    resolve_projects_dir,
    set_config_value,
)
from cutpoint.ffmpeg_utils import FFmpegError, audio_mix_filter, grab_images, probe
from cutpoint.styles import get_style, style_names
from cutpoint.utils import VIDEO_EXTS, find_videos, slugify

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
UI_DIR = "ui"  # per-project state: meta.json, review.json, exports.json, proxy/, frames/
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
MAX_SEGMENTS = 500
MAX_LOG_LINES = 300
MAX_JOBS = 50
MIN_CLIP_S = 0.1  # export_clips rejects shorter clips
MAX_EXTEND_S = 30.0
BROWSER_CONTAINERS = {".mp4", ".m4v", ".mov", ".webm"}
BROWSER_VIDEO = {"h264", "vp8", "vp9", "av1"}
BROWSER_AUDIO = {None, "aac", "mp3", "opus", "vorbis"}
SAFE_NAME = re.compile(r"[\w.-]+")
FRAME_WIDTH = 160  # default still width (px) for sequence thumbnails
FRAME_MIN_W, FRAME_MAX_W = 32, 640
FRAME_TIMEOUT_S = 30.0
FRAME_SLOTS = 2  # ffmpeg still grabs at once; the rest wait their turn
FRAME_CACHE_MAX = 500  # stills kept per project; the least recently used go first


def bad(detail: str, status: int = 400) -> HTTPException:
    return HTTPException(status, detail)


# --------------------------------------------------------------------------- processes
def spawn(args: list[str]) -> subprocess.Popen[str]:
    """Start ``args`` (no shell) in its own process group, stdout+stderr merged."""
    group = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    return subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        **group,
    )


def kill_tree(proc: subprocess.Popen[str]) -> None:
    """Stop ``proc`` and everything it started (ffmpeg, workers): TERM, then KILL."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False
        )
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(5)
    with contextlib.suppress(ProcessLookupError):  # sweeps members that outlived the leader
        os.killpg(proc.pid, signal.SIGKILL)


def cutpoint_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "cutpoint", *args]


def folder_opener() -> Callable[[Path], None] | None:
    """How this machine shows a folder in its file manager (no shell, no waiting), else None.

    None where nobody could see the window: no ``xdg-open`` or no display (servers, Docker).
    """
    if sys.platform == "win32":
        start = getattr(os, "startfile", None)
        return None if start is None else lambda folder: start(str(folder))
    if sys.platform == "darwin":
        exe = shutil.which("open")
    elif os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        exe = shutil.which("xdg-open")
    else:
        exe = None
    if exe is None:
        return None

    def run(folder: Path) -> None:
        proc = subprocess.Popen(
            [exe, str(folder)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        threading.Thread(target=proc.wait, daemon=True).start()  # reap it; never block the request

    return run


# Windows: Tk 8.6's dialogs are the system's IFileDialog (breadcrumbs, search, quick access; the
# folder one too, unlike WinForms' tree FolderBrowserDialog in Windows PowerShell). A hidden topmost
# owner keeps them above the browser. Runs as ``python -I -c`` (no current folder on sys.path).
TK_PICK = """\
import sys
kind, start, exts = sys.argv[1], sys.argv[2] or None, sys.argv[3]
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # sharp text on scaled displays
except Exception:
    pass
try:
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    if kind == "folder":
        out = filedialog.askdirectory(
            parent=root, initialdir=start, mustexist=True, title="Choose a folder"
        )
    else:
        out = filedialog.askopenfilename(
            parent=root, initialdir=start, title="Choose a video",
            filetypes=[("Videos", exts), ("All files", "*.*")],
        )
    root.destroy()
except Exception as err:
    sys.stderr.write(f"{err}\\n")
    sys.exit(3)
sys.stdout.buffer.write((out or "").encode("utf-8"))
"""

# macOS: ``choose file`` has no "All files" switch, so it lists videos only. A cancel prints nothing.
OSA_PICK = """\
on run argv
  activate
  try
    if item 1 of argv is "folder" then
      if (count of argv) > 1 then
        set f to choose folder with prompt "Choose a folder" default location (POSIX file (item 2 of argv))
      else
        set f to choose folder with prompt "Choose a folder"
      end if
    else
      if (count of argv) > 1 then
        set f to choose file with prompt "Choose a video" of type {%s} default location (POSIX file (item 2 of argv))
      else
        set f to choose file with prompt "Choose a video" of type {%s}
      end if
    end if
  on error number -128
    return ""
  end try
  return POSIX path of f
end run
"""


def path_picker() -> Callable[[str, Path | None], tuple[list[str], int | None]] | None:
    """How this machine asks the user for a video (``kind`` "file") or a folder, else None.

    The result maps ``(kind, start folder)`` to the dialog's argv (no shell) and the exit code that
    means "cancelled" (None: a cancel exits 0 and prints nothing). None where nobody could see the
    window: no display, or neither zenity nor kdialog (Linux), no ``osascript`` (macOS), no
    tkinter (Windows), or an SSH session (Windows, macOS: the dialog would open on the remote
    desktop, if any, and keep the one-dialog lock until someone closes it there).
    """
    exts = sorted(e.lstrip(".") for e in VIDEO_EXTS)
    if sys.platform in ("win32", "darwin") and (
        os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY")
    ):
        return None
    if sys.platform == "win32":
        if importlib.util.find_spec("_tkinter") is None:
            return None
        patterns = " ".join(f"*.{e}" for e in exts)
        return lambda kind, start: (
            [sys.executable, "-I", "-c", TK_PICK, kind, str(start or ""), patterns],
            None,
        )
    if sys.platform == "darwin":
        osa = shutil.which("osascript")
        if osa is None:
            return None
        script = OSA_PICK % ((", ".join(f'"{e}"' for e in exts),) * 2)
        return lambda kind, start: (
            [osa, "-e", script, kind, *([str(start)] if start else [])],
            None,
        )
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    if zenity := shutil.which("zenity"):
        globs = " ".join(f"*.{x}" for e in exts for x in (e, e.upper()))

        def zenity_args(kind: str, start: Path | None) -> tuple[list[str], int | None]:
            args = [zenity, "--file-selection"]
            args += (
                ["--directory", "--title=Choose a folder"]
                if kind == "folder"
                else [
                    "--title=Choose a video",
                    f"--file-filter=Videos | {globs}",
                    "--file-filter=All files | *",
                ]
            )
            if start:
                args.append(f"--filename={start}{os.sep}")
            return args, 1

        return zenity_args
    if kdialog := shutil.which("kdialog"):
        kfilter = f"{' '.join(f'*.{e}' for e in exts)}|Videos\n*|All files"

        def kdialog_args(kind: str, start: Path | None) -> tuple[list[str], int | None]:
            where = str(start or Path.home())
            if kind == "folder":
                return [kdialog, "--title", "Choose a folder", "--getexistingdirectory", where], 1
            return [kdialog, "--title", "Choose a video", "--getopenfilename", where, kfilter], 1

        return kdialog_args
    return None


class Picker:
    """The one native open dialog allowed at a time."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen[str] | None = None

    def ask(self, kind: str, start: Path | None) -> str | None:
        """The path the user chose, or None if they cancelled; waits as long as the dialog is open.

        Raises:
            HTTPException: 501 without a dialog here, 409 while one is open, 500 if it failed.
        """
        build = path_picker()
        if build is None:
            raise bad("File dialogs aren't available on this machine; type or paste the path", 501)
        if not self.lock.acquire(blocking=False):
            raise bad("a file dialog is already open", 409)
        try:
            args, cancel_code = build(kind, start)
            try:
                self.proc = subprocess.Popen(
                    args,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                out, err = self.proc.communicate()
            except OSError as e:
                raise bad(f"could not open the file dialog: {e.strerror or e}", 500) from e
            code = self.proc.returncode
            if cancel_code is not None and code == cancel_code:
                return None
            if code != 0:
                logger.warning("file dialog exited with %s: %s", code, err.strip()[-500:])
                raise bad("the file dialog failed", 500)
            path = out.rstrip("\r\n")
            if not path:
                return None
            if not Path(path).is_absolute():
                raise bad("the file dialog returned no usable path", 500)
            return os.path.normpath(path)
        finally:
            self.proc = None
            self.lock.release()

    def shutdown(self) -> None:
        if (proc := self.proc) is not None:
            with contextlib.suppress(OSError):
                proc.kill()


def pick_start(start: str) -> Path | None:
    """Where a dialog opens: ``start``'s folder, or its nearest existing ancestor (None if none)."""
    if not start:
        return None
    try:
        p = Path(start)
        if not p.is_absolute():
            return None
        while not p.is_dir():
            if p.parent == p:
                return None
            p = p.parent
    except (OSError, ValueError):  # ValueError: NUL byte in the path
        return None
    return p


# --------------------------------------------------------------------------- settings
def config_data() -> dict[str, Any]:
    """The raw ``config.json`` object.

    Raises:
        HTTPException: 409 if the file is not a valid JSON object (it is never overwritten).
    """
    try:
        return load_config_data()
    except ValueError as err:
        raise bad(str(err), 409) from err


def settings_view(data: dict[str, Any]) -> dict[str, Any]:
    """The settings for config ``data``; only the projects folder, so other invalid keys in
    ``config.json`` do not hide it."""
    return {
        "projects_dir": str(resolve_projects_dir(data)),
        "projects_dir_locked": paths.env("PROJECTS_DIR") is not None,
        "config_file": str(paths.config_file()),
    }


def check_projects_dir(value: Any) -> tuple[Path, bool]:
    """A folder the user may make the projects folder: absolute, a folder (the last part is created
    if only it is missing) and writable; and whether it was created here.

    Raises:
        HTTPException: 400 naming what is wrong.
    """
    if not isinstance(value, str) or not value.strip():
        raise bad("projects_dir must be a folder path")
    folder = Path(value.strip())
    if not folder.is_absolute():
        raise bad("the projects folder must be an absolute path")
    folder = Path(os.path.abspath(folder))  # noqa: PTH100 - drops '..', keeps mapped drive letters
    created = False
    try:
        if not folder.exists():
            if not folder.parent.is_dir():
                raise bad(f"{folder.parent} does not exist")
            folder.mkdir()
            created = True
        if not folder.is_dir():
            raise bad(f"{folder} is not a folder")
        with tempfile.TemporaryFile(dir=folder):
            pass
    except (OSError, ValueError) as err:
        if created:
            with contextlib.suppress(OSError):
                folder.rmdir()
        raise bad(f"cannot use {folder}: {getattr(err, 'strerror', None) or err}") from err
    return folder, created


# --------------------------------------------------------------------------- jobs
class Job:
    """One child process and what it said."""

    def __init__(self, project_id: str, action: str) -> None:
        self.id = secrets.token_hex(6)
        self.project_id = project_id
        self.action = action
        self.status = "running"
        self.logs: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self.error: str | None = None
        self.started = datetime.now().isoformat(timespec="seconds")
        self.proc: subprocess.Popen[str] | None = None
        self.cancelled = False
        self.done = threading.Event()

    def view(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "project_id": self.project_id,
            "action": self.action,
            "status": self.status,
            "logs": list(self.logs),
            "started": self.started,
        }
        if self.error:
            out["error"] = self.error
        return out


class Jobs:
    """At most one running job serverwide."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}

    def start(
        self,
        project_id: str,
        action: str,
        args: list[str],
        on_success: Callable[[], None] | None = None,
    ) -> Job:
        with self.lock:
            if any(j.status == "running" for j in self.jobs.values()):
                raise bad("another job is already running", 409)
            job = Job(project_id, action)
            try:
                job.proc = spawn(args)
            except OSError as err:
                raise bad(f"could not start job: {err}", 500) from err
            for old in [k for k, j in self.jobs.items() if j.status != "running"][
                : max(0, len(self.jobs) + 1 - MAX_JOBS)
            ]:
                del self.jobs[old]
            self.jobs[job.id] = job
        threading.Thread(target=self._watch, args=(job, on_success), daemon=True).start()
        return job

    @staticmethod
    def _watch(job: Job, on_success: Callable[[], None] | None) -> None:
        proc = job.proc
        assert proc is not None and proc.stdout is not None
        try:
            for line in proc.stdout:
                if line.strip():
                    job.logs.append(line.rstrip())
            code = proc.wait()
            if job.cancelled:
                job.status = "cancelled"
            elif code != 0:
                last = next(
                    (
                        m.group(1)
                        for ln in reversed(job.logs)
                        if (m := re.search(r"ERROR\s+(.*)", ln))
                    ),
                    None,
                )
                job.error, job.status = last or f"exit code {code}", "failed"
            else:
                try:
                    if on_success:
                        on_success()
                    job.status = "completed"
                except Exception as err:  # surfaced to the user as the job error
                    logger.exception("job %s post-processing failed", job.id)
                    job.error, job.status = str(err), "failed"
        finally:
            job.done.set()

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise bad("unknown job", 404)
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status == "running" and job.proc is not None:
            job.cancelled = True
            kill_tree(job.proc)
            job.done.wait(5)
        return job

    def shutdown(self) -> None:
        for job in list(self.jobs.values()):
            if job.status == "running" and job.proc is not None:
                job.cancelled = True
                kill_tree(job.proc)


def proxy_args(real: Path, out: Path, audio_streams: int) -> list[str]:
    """ffmpeg command for a 720p H.264/AAC preview; several audio tracks are mixed as in exports."""
    mix = audio_mix_filter(audio_streams)
    audio = ["-filter_complex", f"{mix}[a]", "-map", "[a]"] if mix else ["-map", "0:a:0?"]
    return [
        "ffmpeg", "-y", "-nostdin", "-v", "error", "-i", str(real), "-map", "0:v:0", *audio,
        "-vf", "scale=-2:min(720\\,ih)",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out),
    ]  # fmt: skip


# --------------------------------------------------------------------------- playable media
class Media:
    """Probe cache and on-demand H.264/AAC proxies for sources a browser cannot play."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.probed: dict[str, tuple[bool, float, int] | None] = {}
        self.proxies: dict[str, str] = {}  # key -> "preparing" | "failed"
        self.procs: set[subprocess.Popen[str]] = set()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cutpoint-proxy")
        self.closing = False
        # still file name -> (its builder lock, requests using it); dropped when the last one leaves
        self.frame_locks: dict[str, tuple[threading.Lock, int]] = {}
        self.frame_slots = threading.BoundedSemaphore(FRAME_SLOTS)
        self.frames_pruned: set[Path] = set()  # projects whose stale stills went this run

    @staticmethod
    def key(real: Path) -> str | None:
        try:
            st = real.stat()
        except OSError:
            return None
        return hashlib.sha1(f"{real}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]

    def info(self, real: Path, key: str) -> tuple[bool, float, int] | None:
        """``(browser plays it as is, duration, audio tracks)``, or None if unreadable."""
        if key not in self.probed:
            try:
                i = probe(real)
                ok = (
                    real.suffix.lower() in BROWSER_CONTAINERS
                    and i.vcodec in BROWSER_VIDEO
                    and (i.vcodec != "h264" or i.pix_fmt in {"yuv420p", "yuvj420p"})
                    and i.acodec in BROWSER_AUDIO
                    and i.audio_streams <= 1  # a browser plays one track; exports mix them all
                )
                self.probed[key] = (ok, i.duration, i.audio_streams)
            except (FFmpegError, OSError, ValueError, KeyError):
                self.probed[key] = None
        return self.probed[key]

    def duration(self, real: Path) -> float | None:
        key = self.key(real)
        res = self.info(real, key) if key else None
        return res[1] if res else None

    def status(self, project: Path, path: str) -> tuple[str, Path | None]:
        """``(status, file to serve)``: ready | needs-proxy | preparing | failed. No side effects."""
        real = remap_source(path)
        key = self.key(real)
        res = self.info(real, key) if key else None
        if res is None or key is None:
            return "failed", None
        if res[0]:
            return "ready", real
        proxy = project / UI_DIR / "proxy" / f"{key}.mp4"
        if proxy.is_file():
            return "ready", proxy
        return {"preparing": "preparing", "failed": "failed"}.get(
            self.proxies.get(key, ""), "needs-proxy"
        ), None

    def prepare(self, project: Path, path: str) -> str:
        """Queue the proxy build of one source (also retries a failed build); returns its status."""
        real = remap_source(path)
        key = self.key(real)
        res = self.info(real, key) if key else None
        if res is not None and key is not None and not res[0] and not self.closing:
            proxy = project / UI_DIR / "proxy" / f"{key}.mp4"
            with self.lock:
                if not proxy.is_file() and self.proxies.get(key) != "preparing":
                    self.proxies[key] = "preparing"
                    self.pool.submit(self._build, real, proxy, key, res[2])
        return self.status(project, path)[0]

    def _build(self, real: Path, proxy: Path, key: str, audio_streams: int) -> None:
        tmp = proxy.with_name(f"{key}.part.mp4")
        state = "failed"
        try:
            proxy.parent.mkdir(parents=True, exist_ok=True)
            proc = spawn(proxy_args(real, tmp, audio_streams))
            self.procs.add(proc)
            try:
                tail = (proc.communicate()[0] or "").strip()[-500:]
            finally:
                self.procs.discard(proc)
            if proc.returncode == 0 and not self.closing:
                tmp.replace(proxy)
                state = ""
            else:
                logger.warning("proxy of %s failed: %s", real.name, tail)
        except OSError:
            logger.exception("proxy of %s failed", real.name)
        finally:
            tmp.unlink(missing_ok=True)
            with self.lock:
                if state:
                    self.proxies[key] = state
                else:
                    self.proxies.pop(key, None)

    def frame(self, project: Path, path: str, at: float, width: int) -> Path | None:
        """A JPEG still of ``path`` at ``at`` s (clamped into the source), ``width`` px wide.

        Cached as ``ui/frames/<content key>-<centiseconds>-<width>.jpg`` and built once even when
        requested concurrently. None if the source cannot be read.

        Raises:
            FFmpegError: If ffmpeg decodes no image.
        """
        real = remap_source(path)
        key = self.key(real)
        res = self.info(real, key) if key else None
        if res is None or key is None:
            return None
        if res[1] > 0:
            at = min(at, max(0.0, res[1] - 0.1))  # the last instants may hold no decodable frame
        cs = round(at * 100)
        out = project / UI_DIR / "frames" / f"{key}-{cs}-{width}.jpg"
        if out.is_file():
            with contextlib.suppress(OSError):
                os.utime(out)  # recently used: evicted last
            return out
        with self.lock:
            lock, users = self.frame_locks.get(out.name, (threading.Lock(), 0))
            self.frame_locks[out.name] = (lock, users + 1)
        try:
            with lock:
                if not out.is_file():
                    with self.frame_slots:
                        image = grab_images(
                            real, cs / 100, width=width, quality=5, timeout=FRAME_TIMEOUT_S
                        )[0]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_name(f"{out.stem}.{threading.get_ident()}.part")
                    try:
                        tmp.write_bytes(image)
                        tmp.replace(out)
                    finally:
                        tmp.unlink(missing_ok=True)
                    self._evict_frames(out.parent)
        finally:
            with self.lock:
                lock, users = self.frame_locks[out.name]
                if users > 1:
                    self.frame_locks[out.name] = (lock, users - 1)
                else:
                    del self.frame_locks[out.name]
        return out

    def _drop_frames(self, files: list[Path]) -> None:
        """Delete cached stills (call with ``self.lock`` held), skipping any being built."""
        for f in files:
            if f"{f.name.split('.')[0]}.jpg" in self.frame_locks:
                continue
            try:
                f.unlink(missing_ok=True)
            except OSError as err:  # e.g. still being sent on Windows; next time
                logger.debug("could not delete %s: %s", f.name, err)

    def _evict_frames(self, folder: Path) -> None:
        """Keep at most ``FRAME_CACHE_MAX`` stills in ``folder``, dropping the least recently used."""
        with self.lock:
            stills: list[tuple[int, Path]] = []
            for f in folder.glob("*.jpg"):
                with contextlib.suppress(OSError):
                    stills.append((f.stat().st_mtime_ns, f))
            if len(stills) > FRAME_CACHE_MAX:
                stills.sort()
                self._drop_frames([f for _, f in stills[: len(stills) - FRAME_CACHE_MAX]])

    def prune_frames(self, project: Path, paths: list[str]) -> None:
        """Once per run and project: delete stills of sources that changed or left, and leftovers."""
        folder = project / UI_DIR / "frames"
        if project in self.frames_pruned:
            return
        keys = {self.key(remap_source(path)) for path in paths} - {None}
        with self.lock:
            self.frames_pruned.add(project)
            if not folder.is_dir():
                return
            stale = [f for f in folder.glob("*.jpg") if f.name.split("-")[0] not in keys]
            self._drop_frames([*stale, *folder.glob("*.part")])

    def shutdown(self) -> None:
        self.closing = True
        self.pool.shutdown(wait=False, cancel_futures=True)
        for proc in list(self.procs):
            kill_tree(proc)


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


def project_dir(pid: str) -> Path:
    """The project folder ``pid`` (a direct child of the projects folder), else 404."""
    root = projects_dir()
    if not pid or pid.startswith(".") or any(c in pid for c in "/\\\0") or pid != pid.strip():
        raise bad("unknown project", 404)
    p = root / pid
    try:
        ok = p.is_dir() and p.resolve().parent == root.resolve()
    except OSError:
        ok = False
    if not ok or not ((p / UI_DIR / "meta.json").is_file() or (p / "highlights.json").is_file()):
        raise bad("unknown project", 404)
    return p


def read_meta(p: Path) -> dict[str, Any]:
    meta = cache.load_json(p / UI_DIR / "meta.json")
    return meta if isinstance(meta, dict) else {}


def _under(path: str, roots: list[str]) -> bool:
    """``path`` is one of ``roots`` or inside one (checked on both spelled and real paths)."""

    def forms(x: str) -> set[str]:
        return {os.path.normcase(os.path.abspath(x)), os.path.normcase(os.path.realpath(x))}  # noqa: PTH100

    for root in roots:
        for r in forms(root):
            if any(c == r or c.startswith(r.rstrip(os.sep) + os.sep) for c in forms(path)):
                return True
    return False


def source_entries(p: Path, hl: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The project's playable-in-principle sources: ``[{path, id, name, duration}]``."""
    meta = read_meta(p)
    if meta.get("source"):
        roots = [str(meta["source"])]
    else:  # command-line project: the inputs its analysis recorded
        roots = [str(x) for x in ((hl or {}).get("settings") or {}).get("inputs") or []]
    paths: dict[str, float | None] = {}
    for s in (hl or {}).get("sources") or []:
        if isinstance(s, dict) and isinstance(s.get("path"), str):
            dur = s.get("duration")
            paths.setdefault(s["path"], dur if isinstance(dur, (int, float)) else None)
    if not paths and meta.get("source"):
        with contextlib.suppress(FileNotFoundError):
            paths = dict.fromkeys(map(str, find_videos([Path(meta["source"])])))
    return [
        {
            "path": path,
            "id": hashlib.sha1(path.encode()).hexdigest()[:12],
            "name": Path(path).name,
            "duration": dur,
        }
        for path, dur in paths.items()
        if Path(path).is_absolute()
        and Path(path).suffix.lower() in VIDEO_EXTS
        and _under(path, roots)
    ]


def source_folder(p: Path) -> Path | None:
    """The folder holding the project's videos: its picked source (a file's parent), or for a
    command-line project the deepest folder shared by its recorded sources. Never client input."""
    source = read_meta(p).get("source")
    try:
        if source:
            # A picked folder that is gone must not fall back to its parent; only files step up.
            x = remap_source(Path(str(source)))
            is_file = x.is_file() or (not x.exists() and x.suffix.lower() in VIDEO_EXTS)
            folders = [x.parent if is_file else x]
        else:
            hl = cache.load_json(p / "highlights.json")
            entries = source_entries(p, hl if isinstance(hl, dict) else None)
            folders = [remap_source(Path(s["path"])).parent for s in entries]
        try:
            folder = Path(os.path.commonpath(folders)) if folders else None
        except ValueError:  # on different drives: there is no one folder to show
            return None
        return folder if folder is not None and folder.is_dir() else None
    except OSError:  # unreadable or stale (permissions, network share): treat as gone
        return None


def moments_view(hl: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep saved trims attached to source intervals, not mutable CLI ranks."""
    if hl is None:
        return None
    return {
        **hl,
        "candidates": [
            {
                **c,
                "id": hashlib.sha1(
                    f"{os.path.realpath(c['source'])}|{float(c['start'])}|{float(c['end'])}".encode()
                ).hexdigest()[:12],
            }
            for c in hl.get("candidates") or []
        ],
    }


def _num(item: dict[str, Any], key: str, i: int) -> float:
    v = item.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise bad(f"segment {i}: {key} must be a number")
    return float(v)


def validate_segments(
    raw: Any, sources: list[dict[str, Any]], media: Media
) -> list[dict[str, Any]]:
    """Check a review against the project's sources; return the cleaned segments."""
    if not isinstance(raw, list):
        raise bad("segments must be a list")
    if len(raw) > MAX_SEGMENTS:
        raise bad(f"at most {MAX_SEGMENTS} segments")
    by_path = {s["path"]: s for s in sources}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise bad(f"segment {i}: object expected")
        sid, label = item.get("id"), item.get("label", "segment")
        if not isinstance(sid, str) or not 0 < len(sid) <= 64 or sid in seen:
            raise bad(f"segment {i}: id must be a unique string of 1-64 characters")
        seen.add(sid)
        src = by_path.get(item.get("source")) if isinstance(item.get("source"), str) else None
        if src is None:
            raise bad(f"segment {i}: source is not part of this project")
        start, end = _num(item, "start", i), _num(item, "end", i)
        if start < 0 or end - start < MIN_CLIP_S:
            raise bad(f"segment {i}: needs 0 <= start and end at least {MIN_CLIP_S}s after start")
        duration = src["duration"] or media.duration(remap_source(src["path"]))
        if duration is None:
            raise bad(f"segment {i}: source cannot be read")
        if end > duration + 0.01:
            raise bad(f"segment {i}: end {end:g}s is past the source's {duration:g}s")
        if not isinstance(label, str) or len(label) > 200:
            raise bad(f"segment {i}: label must be text of at most 200 characters")
        out.append(
            {
                "id": sid,
                "source": src["path"],
                "start": round(start, 3),
                "end": round(end, 3),
                "label": "".join(c for c in label if c.isprintable()).strip() or "segment",
            }
        )
    return out


def load_exports(p: Path) -> list[dict[str, Any]]:
    data = cache.load_json(p / UI_DIR / "exports.json")
    items = data.get("exports") if isinstance(data, dict) else None
    return [
        e
        for e in items or []
        if isinstance(e, dict) and isinstance(e.get("dir"), str) and SAFE_NAME.fullmatch(e["dir"])
    ]


def current_export(p: Path) -> dict[str, Any] | None:
    """The latest export whose clips are still on disk."""
    return next(
        (e for e in reversed(load_exports(p)) if (p / e["dir"] / "manifest.json").is_file()), None
    )


def output_names(p: Path) -> list[str]:
    """Files of the current final (video first, then timelines) that exist."""
    exp = current_export(p)
    final = (exp or {}).get("final")
    if not isinstance(final, str) or not SAFE_NAME.fullmatch(final) or not (p / final).is_file():
        return []
    stem = Path(final).stem
    return [final, *(f"{stem}{e}" for e in (".fcpxml", ".otio") if (p / f"{stem}{e}").is_file())]


def fresh_name(parent: Path, stem: str, exts: tuple[str, ...] = ("",)) -> str:
    """``stem`` or ``stem-N``: the first name no file ``name + ext`` already uses."""
    n, name = 1, stem
    while any((parent / f"{name}{e}").exists() for e in exts):
        n += 1
        name = f"{stem}-{n}"
    return name


def new_project(name: str, source: str, style: str) -> dict[str, str]:
    root = projects_dir()
    root.mkdir(parents=True, exist_ok=True)
    base, n = slugify(name, 60, "project"), 1
    while True:
        pid = base if n == 1 else f"{base}-{n}"
        try:
            (root / pid).mkdir()
            break
        except FileExistsError:
            n += 1
    (root / pid / UI_DIR).mkdir()
    cache.save_json(
        root / pid / UI_DIR / "meta.json",
        {"name": name, "source": source, "style": style, "created": datetime.now().isoformat()},
    )
    return {"id": pid, "name": name}


def ui_styles() -> list[str]:
    """Styles for standard video (telemetry-driven action-cam styles are command-line only)."""
    return [n for n in style_names() if get_style(n).kind != "ski"]


# --------------------------------------------------------------------------- app
def _flag(body: dict[str, Any], key: str) -> bool:
    v = body.get(key, False)
    if not isinstance(v, bool):
        raise bad(f"{key} must be true or false")
    return v


def create_app(port: int) -> FastAPI:
    token = secrets.token_urlsafe(32)
    jobs, media, picker = Jobs(), Media(), Picker()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        jobs.shutdown()
        media.shutdown()
        picker.shutdown()

    app = FastAPI(
        title="Cutpoint", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )

    app.state.token, app.state.jobs, app.state.media = token, jobs, media
    app.state.picker = picker

    @app.middleware("http")
    async def guard(request: Request, call_next):
        origin = request.headers.get("origin")
        if not _loopback_url(request.headers.get("host", ""), port, origin=False) or (
            origin is not None and not _loopback_url(origin, port, origin=True)
        ):
            return JSONResponse({"detail": "forbidden host or origin"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    def supplied(request: Request, *, query: bool = False) -> bool:
        got = request.headers.get("x-cutpoint-token") or (
            request.query_params.get("t") if query else ""
        )
        return secrets.compare_digest((got or "").encode(), token.encode())

    def auth(request: Request) -> None:
        if not supplied(request):
            raise bad("missing or wrong session token", 403)

    def auth_media(request: Request) -> None:
        if not supplied(request, query=True):
            raise bad("missing or wrong session token", 403)

    api = APIRouter(prefix="/api")
    secure = APIRouter(dependencies=[Depends(auth)])
    served = APIRouter(dependencies=[Depends(auth_media)])

    def url(pid: str, kind: str, name: str) -> str:
        return f"/api/projects/{quote(pid, safe='')}/{kind}/{quote(name, safe='')}?t={token}"

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
        """Change the projects folder in ``config.json``. Existing projects stay where they are."""
        if paths.env("PROJECTS_DIR") is not None:
            raise bad(
                "CUTPOINT_PROJECTS_DIR sets the projects folder; change or unset it there", 409
            )
        if any(j.status == "running" for j in list(jobs.jobs.values())):
            raise bad("wait for the running job to finish before changing the projects folder", 409)
        config_data()  # an unreadable config.json: refuse before creating the folder
        folder, created = check_projects_dir(body.get("projects_dir"))
        try:
            set_config_value("projects_dir", str(folder))
        except (ValueError, OSError) as err:
            if created:
                with contextlib.suppress(OSError):
                    folder.rmdir()
            if isinstance(err, ValueError):  # config.json is not valid JSON: never overwrite it
                raise bad(str(err), 409) from err
            raise bad(f"could not save the settings: {err.strerror or err}", 500) from err
        return settings_view({"projects_dir": str(folder)})

    @secure.get("/projects")
    def list_projects():
        root = projects_dir()
        found = []
        for d in sorted(root.iterdir()) if root.is_dir() else []:
            if d.name.startswith(".") or not d.is_dir() or (d / "pano.json").exists():
                continue
            if (d / UI_DIR / "meta.json").is_file() or (d / "highlights.json").is_file():
                found.append({"id": d.name, "name": read_meta(d).get("name") or d.name})
        return {"projects": found}

    @secure.post("/projects")
    def create_project(body: dict[str, Any]):
        name, source, style = body.get("name"), body.get("source"), body.get("style", "gaming")
        if not isinstance(name, str) or not name.strip() or len(name) > 100:
            raise bad("name must be 1-100 characters")
        if not isinstance(source, str) or not Path(source).is_absolute():
            raise bad("source must be an absolute path")
        if style not in ui_styles():
            raise bad(f"unknown style {style!r}")
        src = Path(source)
        try:
            if src.is_file():
                if src.suffix.lower() not in VIDEO_EXTS:
                    raise bad("source is not a video file")
            elif not src.is_dir() or not find_videos([src]):
                raise bad("source must be a video file or a folder containing videos")
        except (OSError, FileNotFoundError) as err:
            raise bad(f"source is not readable: {err}") from err
        return new_project(name.strip(), os.path.abspath(source), style)  # noqa: PTH100

    @secure.get("/projects/{pid}")
    def get_project(pid: str):
        p = project_dir(pid)
        meta = read_meta(p)
        hl = cache.load_json(p / "highlights.json")
        hl = hl if isinstance(hl, dict) else None
        review = cache.load_json(p / UI_DIR / "review.json")
        exp = current_export(p)
        manifest = cache.load_json(p / exp["dir"] / "manifest.json") if exp else None
        sources = []
        for s in source_entries(p, hl):
            status, _ = media.status(p, s["path"])
            sources.append(
                {
                    "id": s["id"],
                    "name": s["name"],
                    "path": s["path"],
                    "url": url(pid, "sources", s["id"]),
                    "duration": s["duration"],
                    "status": status,
                }
            )
        return {
            "id": pid,
            "name": meta.get("name") or pid,
            "style": meta.get("style") or (hl or {}).get("style") or "",
            "source": meta.get("source"),
            "highlights": moments_view(hl),
            "review": review if isinstance(review, dict) else None,
            "clips": {**manifest, "dir": exp["dir"]}
            if isinstance(manifest, dict) and exp
            else None,
            "outputs": [{"name": n, "url": url(pid, "outputs", n)} for n in output_names(p)],
            "sources": sources,
        }

    @secure.put("/projects/{pid}/review")
    def put_review(pid: str, body: dict[str, Any]):
        p = project_dir(pid)
        hl = cache.load_json(p / "highlights.json")
        segments = validate_segments(
            body.get("segments"), source_entries(p, hl if isinstance(hl, dict) else None), media
        )
        review = {"segments": segments, "updated": datetime.now().isoformat(timespec="seconds")}
        cache.save_json(p / UI_DIR / "review.json", review)
        return review

    @secure.post("/projects/{pid}/reveal")
    def reveal_folder(pid: str, target: str = "sources"):
        """Show a project folder in the file manager: its videos, latest clips or final (path from
        the project; ``target`` only picks which)."""
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

    @secure.post("/projects/{pid}/jobs")
    def start_job(pid: str, body: dict[str, Any]):
        p = project_dir(pid)
        action = body.get("action")
        meta = read_meta(p)
        hl = cache.load_json(p / "highlights.json")
        hl = hl if isinstance(hl, dict) else None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        on_success: Callable[[], None] | None = None
        if action == "analyze":
            inputs = (
                [meta["source"]]
                if meta.get("source")
                else (hl or {}).get("settings", {}).get("inputs")
            )
            style = meta.get("style") or (hl or {}).get("style")
            if not inputs or not style:
                raise bad("project has no recorded source to analyze")
            args = cutpoint_cmd(
                "analyze", *map(str, inputs), "--style", str(style), "--out", str(p)
            )
            # Keep what the earlier analysis recorded, or the source set would change.
            st = (hl or {}).get("settings") or {}
            if isinstance(st.get("exclude"), list):
                args += [f"--exclude={g}" for g in st["exclude"] if isinstance(g, str)]
            if st.get("recursive") is True:
                args.append("--recursive")
        elif action == "clips":
            raw = cache.load_json(p / UI_DIR / "review.json")
            segments = validate_segments(
                (raw.get("segments") or []) if isinstance(raw, dict) else [],
                source_entries(p, hl),
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
            vertical = _flag(body, "vertical")
            out_name = fresh_name(p, f"clips_ui_{stamp}")
            seg_file = p / UI_DIR / f"segments-{out_name[len('clips_ui_') :]}.json"
            # Exactly the trims the user made, in their order: no smart tails or speech snapping.
            cache.save_json(
                seg_file, [{**s, "smart_tail": False, "speech_aware": False} for s in segments]
            )
            args = cutpoint_cmd(
                "clips", str(p), "--segments", str(seg_file), "--out", str(p / out_name)
            )
            if vertical:
                args.append("--vertical")
            if extend:
                args += ["--extend", f"{float(extend):g}"]

            def on_success() -> None:
                if not (p / out_name / "manifest.json").is_file():
                    raise RuntimeError("clips finished without a manifest")
                items = load_exports(p)
                items.append(
                    {"dir": out_name, "created": stamp, "vertical": vertical, "final": None}
                )
                cache.save_json(p / UI_DIR / "exports.json", {"exports": items})

        elif action == "assemble":
            exp = current_export(p)
            if exp is None or not any((p / exp["dir"]).glob("*.mp4")):
                raise bad("export clips before assembling")
            transition = body.get("transition", "cut")
            if transition not in ("cut", "fade"):
                raise bad("transition must be 'cut' or 'fade'")
            exp_dir = exp["dir"]
            stem = fresh_name(
                p,
                "final" + exp_dir[5:] if exp_dir.startswith("clips") else f"final_{exp_dir}",
                (".mp4", ".fcpxml", ".otio", ".ass"),
            )
            args = cutpoint_cmd(
                "assemble",
                str(p / exp_dir),
                "--out",
                str(p / f"{stem}.mp4"),
                "--transition",
                transition,
            )
            if _flag(body, "loudnorm"):
                args.append("--loudnorm")
            if _flag(body, "captions"):
                args.append("--captions")

            def on_success() -> None:
                if not (p / f"{stem}.mp4").is_file():
                    raise RuntimeError("assemble finished without a final video")
                items = load_exports(p)
                for e in items:
                    if e["dir"] == exp_dir:
                        e["final"] = f"{stem}.mp4"
                cache.save_json(p / UI_DIR / "exports.json", {"exports": items})

        else:
            raise bad("action must be 'analyze', 'clips' or 'assemble'")
        return jobs.start(pid, action, args, on_success).view()

    @secure.get("/jobs")
    def list_jobs():
        return {"jobs": [j.view() for j in list(jobs.jobs.values())]}

    @secure.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        return jobs.cancel(job_id).view()

    def send(file: Path, request: Request, name: str | None = None) -> FileResponse:
        """Range-capable file response; ``?download=1`` makes the browser save it."""
        download = request.query_params.get("download") == "1"
        return FileResponse(
            file,
            media_type=mimetypes.guess_type(file.name)[0] or "application/octet-stream",
            filename=name or file.name if download else None,
            content_disposition_type="attachment" if download else "inline",
        )

    def find_source(pid: str, sid: str) -> tuple[Path, dict[str, Any]]:
        p = project_dir(pid)
        hl = cache.load_json(p / "highlights.json")
        entry = next(
            (s for s in source_entries(p, hl if isinstance(hl, dict) else None) if s["id"] == sid),
            None,
        )
        if entry is None:
            raise bad("unknown source", 404)
        return p, entry

    @served.api_route("/projects/{pid}/sources/{sid}", methods=["GET", "HEAD"])
    def source_media(pid: str, sid: str, request: Request):
        p, entry = find_source(pid, sid)
        status, file = media.status(p, entry["path"])
        if file is None:
            raise bad(f"source is {status}", 404 if status == "failed" else 409)
        return send(file, request, entry["name"])

    @served.get("/projects/{pid}/sources/{sid}/frame")
    def source_frame(pid: str, sid: str, at: float = 0.0, w: int = FRAME_WIDTH):
        """One still of a source (``at`` seconds in, ``w`` px wide) for thumbnails."""
        if not math.isfinite(at) or at < 0:
            raise bad("at must be a time in seconds, 0 or more")
        if media.closing:
            raise bad("the server is stopping", 503)
        p, entry = find_source(pid, sid)
        if p not in media.frames_pruned:
            hl = cache.load_json(p / "highlights.json")
            sources = source_entries(p, hl if isinstance(hl, dict) else None)
            media.prune_frames(p, [s["path"] for s in sources])
        try:
            file = media.frame(p, entry["path"], at, min(max(w, FRAME_MIN_W), FRAME_MAX_W))
        except (FFmpegError, OSError) as err:
            logger.warning("still of %s at %.2fs failed: %s", entry["name"], at, err)
            raise bad("could not extract a frame from this source", 500) from err
        if file is None:
            raise bad("source cannot be read", 404)
        return FileResponse(
            file, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"}
        )

    @secure.post("/projects/{pid}/sources/{sid}/prepare")
    def prepare_source(pid: str, sid: str):
        p, entry = find_source(pid, sid)
        return {"status": media.prepare(p, entry["path"])}

    @served.api_route("/projects/{pid}/outputs/{name}", methods=["GET", "HEAD"])
    def output_file(pid: str, name: str, request: Request):
        p = project_dir(pid)
        if name not in output_names(p):
            raise bad("unknown output", 404)
        return send(p / name, request)

    api.include_router(secure)
    api.include_router(served)
    app.include_router(api)
    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def _hangup(*_: object) -> None:
    raise SystemExit(1)  # unwinds through run()'s finally, which stops the children


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

    app = create_app(port)
    sock = listen(port)
    # Printed only once the port is ours, so the URL always works.
    # The fragment never reaches the network; the page keeps the token in sessionStorage.
    print(f"Cutpoint UI: http://127.0.0.1:{port}/#t={app.state.token} (Ctrl+C to stop)", flush=True)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _hangup)
    try:
        server = uvicorn.Server(
            uvicorn.Config(
                app, host="127.0.0.1", port=port, log_level="warning", timeout_graceful_shutdown=5
            )
        )
        server.run(sockets=[sock])
    finally:  # jobs and ffmpeg run in their own sessions: nothing else would stop them
        sock.close()
        app.state.jobs.shutdown()
        app.state.media.shutdown()
        app.state.picker.shutdown()
