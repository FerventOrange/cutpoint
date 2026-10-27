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
* Settings (``GET``/``PUT /api/settings``) change only ``projects_dir`` and ``vision`` in
  ``config.json``. The projects folder never changes while a job runs or when
  ``CUTPOINT_PROJECTS_DIR`` sets it; existing projects are not moved. Every analysis the page
  starts (first or again) passes ``--vision`` or ``--no-vision`` as ``vision`` says when it
  starts; the page has no per-run vision box.
  While ``config.json`` is unreadable or holds a value Cutpoint rejects, the project routes answer
  409 naming the file and the error; session, pick and settings keep working.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import importlib.util
import json
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
import time
import weakref
from collections import Counter, deque
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

from cutpoint import cache, paths, progress
from cutpoint.clips import MIN_EXPORT_S
from cutpoint.config import (
    COMPRESS_PEAKS,
    get_config,
    load_config_data,
    projects_dir,
    remap_source,
    resolve_projects_dir,
    set_config_values,
)
from cutpoint.encode import BT709_TAGS, sdr_filter
from cutpoint.ffmpeg_utils import (
    SOFT_STOP_S,
    EncoderUnavailableError,
    FFmpegError,
    VideoEncoder,
    audio_mix_filter,
    grab_images,
    probe,
    video_encoder,
)
from cutpoint.pano import x3
from cutpoint.pano.project import default_style as pano_style
from cutpoint.styles import get_style, learned_styles, style_names
from cutpoint.utils import (
    PANO_EXTS,
    VIDEO_EXTS,
    clip_files,
    file_stamp,
    find_videos,
    safe_is_file,
    slugify,
)
from cutpoint.vision import default_model

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
UI_DIR = "ui"  # per-project state: meta.json, review.json, exports.json, proxy/, frames/
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
MAX_SEGMENTS = 500
MAX_LOG_LINES = 300
MAX_JOBS = 50
MAX_EXTEND_S = 30.0
# A stopped job gets this long before the hard kill: its own soft stop waits SOFT_STOP_S for
# ffmpeg, then kills it and waits SOFT_STOP_S again, and only then removes the partial file;
# the rest is for that removal and the interpreter's exit (#257).
JOB_STOP_S = 2 * SOFT_STOP_S + 2.0
BROWSER_CONTAINERS = {".mp4", ".m4v", ".mov", ".webm"}
BROWSER_VIDEO = {"h264", "vp8", "vp9", "av1"}
BROWSER_AUDIO = {None, "aac", "mp3", "opus", "vorbis"}
SAFE_NAME = re.compile(r"\w[\w.-]*")  # a name inside the project: no ".", ".." or ".x"
# What ``assemble`` writes next to ``<stem>.mp4`` (.ass: the --captions subtitles it burns in).
FINAL_EXTS = (".mp4", ".fcpxml", ".otio", ".srt", ".ass")
# A final in the project folder: ``assemble``'s final.mp4 / final_<x>.mp4 (UI exports included)
# and ``edit``'s <name>-final.mp4 / <name>-final_vertical.mp4. Partial and staging files start
# with "." and never match.
FINAL_VIDEO = re.compile(r"(?:[^.].*-)?final(?:_.*)?\.mp4")
# A step update the CLI writes when CUTPOINT_PROGRESS=1 (``spawn`` sets it), bare or after a
# ``HH:MM:SS LEVEL`` log prefix: ``PROGRESS {"step": ..., "index": ..., "state": ...}``.
PROGRESS_LINE = re.compile(r"(?:\d{1,2}:\d{2}:\d{2}\s+[A-Z]+\s+)?PROGRESS\s+(\{.*\})")
STEP_STATES = {"running", "done", "skipped", "failed"}
FRAME_WIDTH = 160  # default still width (px) for sequence thumbnails
FRAME_MIN_W, FRAME_MAX_W = 32, 640
FRAME_TIMEOUT_S = 30.0
FRAME_SLOTS = 2  # ffmpeg still grabs at once; the rest wait their turn
FRAME_CACHE_MAX = 500  # stills kept per project; the least recently used go first
SOURCES_CACHE_MAX = 32  # projects whose source lists are kept; the least recently used go first
SOURCES_CACHE_S = 60.0  # rebuild a kept source list at least this often (what stats miss)


def bad(detail: str, status: int = 400) -> HTTPException:
    return HTTPException(status, detail)


# --------------------------------------------------------------------------- processes
def spawn(args: list[str], *, low_priority: bool = True) -> subprocess.Popen[str]:
    """Start ``args`` (no shell) in its own process group at low priority, stdout+stderr merged.

    Low priority keeps the desktop and the browser responsive while a job or proxy encodes; the
    children it starts (ffmpeg, workers) inherit it. Windows: BELOW_NORMAL_PRIORITY_CLASS;
    elsewhere ``nice -n 10`` when there is one (it execs, so the pid and group stay the same).
    ``low_priority=False``: normal priority, for the short window proxy encodes the user waits on.
    """
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        if low_priority:
            flags |= subprocess.BELOW_NORMAL_PRIORITY_CLASS
        group: dict[str, Any] = {"creationflags": flags}
    else:
        group = {"start_new_session": True}
        nice = shutil.which("nice") if low_priority else None
        args = [nice, "-n", "10", *args] if nice else args
    proc = subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        # CUTPOINT_PROGRESS: analyze and edit write PROGRESS step lines (``parse_progress``); the
        # other commands and ffmpeg ignore it.
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "CUTPOINT_PROGRESS": "1",
        },
        **group,
    )
    if os.name == "nt":
        _join_tree_job(proc)
    return proc


def _join_tree_job(proc: subprocess.Popen[str]) -> None:
    """Windows: put ``proc`` in a job object of its own, so its descendants join it too.

    ``kill_tree`` ends the job, which reaches every descendant even when the leader is gone:
    ``taskkill /T`` walks the tree from a live leader only, and a soft stop that worked has ended
    the leader. A descendant can miss CTRL_BREAK: a venv's python.exe launcher ignores the event,
    so an interpreter it starts just after the event keeps running and holds the job's stdout.
    """
    import ctypes  # noqa: PLC0415 - Windows only

    k32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    k32.CreateJobObjectW.restype = ctypes.c_void_p
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return
    if k32.AssignProcessToJobObject(ctypes.c_void_p(job), ctypes.c_void_p(int(proc._handle))):  # type: ignore[attr-defined]
        proc.tree_job = job  # type: ignore[attr-defined]
        weakref.finalize(proc, k32.CloseHandle, ctypes.c_void_p(job))
    else:  # e.g. the leader already exited: taskkill alone, as before
        k32.CloseHandle(ctypes.c_void_p(job))


def kill_tree(proc: subprocess.Popen[str], wait: float = JOB_STOP_S, *, term: bool = True) -> None:
    """Stop ``proc`` and everything it started (ffmpeg, workers): TERM, wait ``wait``, then KILL.

    ``term=False``: the KILL alone, for a job that was already sent its TERM.

    Windows: CTRL_BREAK_EVENT to the job's process group first (cutpoint turns it into
    KeyboardInterrupt: partial files removed, exit 130), then the job object ``spawn`` made
    and ``taskkill /T /F``. The event
    needs a console shared with the job; without one (pythonw) only the hard kill runs.
    GenerateConsoleCtrlEvent is called directly: when it fails, ``os.kill`` falls through to
    TerminateProcess on the leader alone, which orphans its ffmpeg before taskkill can see it.
    """
    if os.name == "nt":
        import ctypes  # noqa: PLC0415 - Windows only

        k32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
        if term and k32.GenerateConsoleCtrlEvent(signal.CTRL_BREAK_EVENT, proc.pid):
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(wait)
        job = getattr(proc, "tree_job", None)
        if job:  # every descendant, including ones the leader's exit left out of /T
            k32.TerminateJobObject(ctypes.c_void_p(job), 1)
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False
        )
        return
    if term:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(wait)
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
kind, start, exts, pano = sys.argv[1], sys.argv[2] or None, sys.argv[3], sys.argv[4]
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
            filetypes=[("Videos", exts), ("Insta360 360 video", pano), ("All files", "*.*")],
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
    pano = sorted(e.lstrip(".") for e in PANO_EXTS)  # a second filter: VIDEO_EXTS stays flat-only
    if sys.platform in ("win32", "darwin") and (
        os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY")
    ):
        return None
    if sys.platform == "win32":
        if importlib.util.find_spec("_tkinter") is None:
            return None
        patterns, pano_patterns = (" ".join(f"*.{e}" for e in x) for x in (exts, pano))
        return lambda kind, start: (
            [sys.executable, "-I", "-c", TK_PICK, kind, str(start or ""), patterns, pano_patterns],
            None,
        )
    if sys.platform == "darwin":
        osa = shutil.which("osascript")
        if osa is None:
            return None
        script = OSA_PICK % ((", ".join(f'"{e}"' for e in exts + pano),) * 2)
        return lambda kind, start: (
            [osa, "-e", script, kind, *([str(start)] if start else [])],
            None,
        )
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    if zenity := shutil.which("zenity"):
        globs, pano_globs = (
            " ".join(f"*.{x}" for e in g for x in (e, e.upper())) for g in (exts, pano)
        )

        def zenity_args(kind: str, start: Path | None) -> tuple[list[str], int | None]:
            args = [zenity, "--file-selection"]
            args += (
                ["--directory", "--title=Choose a folder"]
                if kind == "folder"
                else [
                    "--title=Choose a video",
                    f"--file-filter=Videos | {globs}",
                    f"--file-filter=Insta360 360 video | {pano_globs}",
                    "--file-filter=All files | *",
                ]
            )
            if start:
                args.append(f"--filename={start}{os.sep}")
            return args, 1

        return zenity_args
    if kdialog := shutil.which("kdialog"):
        kfilter = "\n".join(
            f"{' '.join(f'*.{e}' for e in x)}|{label}"
            for x, label in ((exts, "Videos"), (pano, "Insta360 360 video"))
        )
        kfilter += "\n*|All files"

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


def require_config() -> None:
    """Dependency of every route that reads the projects folder or remaps sources.

    Raises:
        HTTPException: 409 naming the rejected setting's source (``config.json`` or a
            ``CUTPOINT_`` variable) and what is wrong with it, when
            :func:`~cutpoint.config.get_config` rejects it (Settings stay usable to fix it).
    """
    try:
        get_config()
    except ValueError as err:  # its message names the file or variable
        raise bad(str(err), 409) from err
    except TypeError as err:  # a wrongly typed value (e.g. a number for a path)
        raise bad(f"Invalid config file {paths.config_file()}: {err}", 409) from err


def settings_view(data: dict[str, Any]) -> dict[str, Any]:
    """The settings for config ``data``: the projects folder, the vision default and the export
    default of Compress peaks, so other invalid keys in ``config.json`` do not hide them; a wrongly
    typed ``projects_dir`` is shown as is, to replace, a ``vision`` that is not ``true`` as off and
    an unknown ``compress_peaks`` as off."""
    peaks = data.get("compress_peaks")
    try:
        folder = str(resolve_projects_dir(data))
    except TypeError:
        folder = str(data.get("projects_dir"))
    return {
        "projects_dir": folder,
        "projects_dir_locked": paths.env("PROJECTS_DIR") is not None,
        "config_file": str(paths.config_file()),
        "vision": data.get("vision") is True,
        "vision_model": default_model(),
        "compress_peaks": peaks if peaks in COMPRESS_PEAKS else "off",
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
def parse_progress(line: str) -> dict[str, Any] | None:
    """The step update a ``PROGRESS {...}`` output line carries, or None for any other line.

    A line that starts like one but is not a valid update (bad JSON, missing or wrong fields) is
    None as well, so it stays in the log as text.
    """
    m = PROGRESS_LINE.fullmatch(line.strip())
    if not m:
        return None
    try:
        raw = json.loads(m.group(1))
    except (ValueError, RecursionError):  # RecursionError: nested too deeply
        return None
    if not isinstance(raw, dict):
        return None
    step, index, label, state = (raw.get(k) for k in ("step", "index", "label", "state"))
    total, pct, detail = raw.get("total"), raw.get("pct"), raw.get("detail")
    if (
        not isinstance(step, str)
        or not step
        or not isinstance(index, int)
        or isinstance(index, bool)
        or state not in STEP_STATES
    ):
        return None
    number = isinstance(pct, (int, float)) and not isinstance(pct, bool) and math.isfinite(pct)
    return {
        "step": step,
        "index": index,
        "total": total if isinstance(total, int) and not isinstance(total, bool) else None,
        "label": label if isinstance(label, str) and label else step,
        "state": state,
        "pct": min(100.0, max(0.0, float(pct))) if number else None,
        "detail": detail if isinstance(detail, str) and detail else None,
    }


class Job:
    """One child process and what it said."""

    def __init__(
        self, project_id: str, action: str, plan: list[dict[str, str]] | None = None
    ) -> None:
        self.id = secrets.token_hex(6)
        self.project_id = project_id
        self.action = action
        self.plan = plan  # the steps the command will report (``{step, label}``), if known
        self.status = "running"
        self.logs: deque[str] = deque(maxlen=MAX_LOG_LINES)
        # Latest update per step (``parse_progress``), keyed by step: the CLI's PROGRESS lines.
        self.steps: dict[str, dict[str, Any]] = {}
        self.steps_total: int | None = None
        self.error: str | None = None
        self.started = datetime.now().isoformat(timespec="seconds")
        self.proc: subprocess.Popen[str] | None = None
        self.cancelled = False
        self.done = threading.Event()

    def add_output(self, line: str) -> None:
        """One output line: a step update goes to the steps, any other text to the log."""
        step = parse_progress(line)
        if step is None:
            if line.strip():
                self.logs.append(line.rstrip())
            return
        total = step.pop("total")
        if total is not None:
            self.steps_total = total
        self.steps[step["step"]] = step  # replaced, never changed in place: ``view`` copies it

    def interrupt_steps(self) -> None:
        """The job failed or was stopped: a step still running will not finish."""
        for name, step in list(self.steps.items()):
            if step["state"] == "running":
                self.steps[name] = {**step, "state": "interrupted"}

    def view(self) -> dict[str, Any]:
        # Ordered by the CLI's step index; a step sent again later keeps its place.
        steps = sorted(self.steps.values(), key=lambda s: s["index"])
        out: dict[str, Any] = {
            "id": self.id,
            "project_id": self.project_id,
            "action": self.action,
            # Cancelled but the watcher has not seen the exit yet (Windows can lag a poll or two).
            "status": "cancelling" if self.cancelled and self.status == "running" else self.status,
            "logs": list(self.logs),
            "progress": {"total": self.steps_total, "steps": steps},
            "started": self.started,
        }
        if self.plan:
            out["progress"]["plan"] = self.plan
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
        check: Callable[[], None] | None = None,
        plan: list[dict[str, str]] | None = None,
    ) -> Job:
        """Start ``args`` as a job; ``check`` runs under the lock first and may refuse it.
        ``plan``: the steps it will report, shown as waiting before their first update."""
        with self.lock:
            if any(j.status == "running" for j in self.jobs.values()):
                raise bad("another job is already running", 409)
            if check:
                check()
            job = Job(project_id, action, plan)
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
                try:
                    job.add_output(line)
                except Exception:  # one odd line must not end the watcher: keep it as text
                    logger.exception("job %s could not read an output line", job.id)
                    job.logs.append(line.rstrip())
            code = proc.wait()
            if job.cancelled or code != 0:
                job.interrupt_steps()
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
                except HTTPException as err:  # a refused write: its message is the error
                    job.error, job.status = err.detail, "failed"
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
        """Stop the job in the background and return at once; it reads "cancelling" until it exits."""
        job = self.get(job_id)
        with self.lock:  # one killer per job, however often Cancel is pressed
            start = job.status == "running" and job.proc is not None and not job.cancelled
            if start:
                job.cancelled = True
        if start:
            threading.Thread(target=kill_tree, args=(job.proc,), daemon=True).start()
        return job

    def shutdown(self) -> None:
        # One by one: at most one job runs at a time (``start``), so this waits JOB_STOP_S at most.
        for job in list(self.jobs.values()):
            if job.status != "running" or job.proc is None:
                continue
            with self.lock:  # as in ``cancel``: did a killer already send the TERM?
                stopping, job.cancelled = job.cancelled, True
            if not stopping:
                kill_tree(job.proc)
            elif not job.done.wait(JOB_STOP_S):  # a second TERM could cut short its cleanup
                kill_tree(job.proc, term=False)


# Proxies: a keyframe every PROXY_GOP frames and at most PROXY_FPS_MAX fps keep seeks quick (the
# encoder default GOP made them take up to ~0.4 s) and the encode light.
PROXY_GOP = 30
PROXY_FPS_MAX = 30
# NVENC at CQ 28 alone was up to four times libx264's size at CRF 27. At CQ 32 with this cap, a
# 1080p60 recording's proxy was about 1.2 times libx264's (scoring a higher SSIM) and needed
# about half the CPU time of libx264.
# VAAPI at QP 32 (AMD, 1080p60, 15-20 s of four synthetic contents) came out 0.6 to 1.0 times
# libx264's size, so it needs no cap. AMF was not measured (no Windows machine): AMF proxies
# stay on libx264 (Media._gpu_encoder).
PROXY_GPU_QUALITY = 32  # NVENC CQ; the same number as QP for VAAPI
PROXY_NVENC_RATE = ["-maxrate", "1500k", "-bufsize", "3000k"]
# Window proxies (#323): while the whole proxy builds, the focused moment plays from a short
# proxy of [start - BEFORE, end + AFTER], its edges rounded outward to STEP so nearby requests
# share one. At most WINDOWS_MAX per source are kept; the least recently used go first.
PROXY_WINDOW_BEFORE = 10.0
PROXY_WINDOW_AFTER = 20.0
PROXY_WINDOW_STEP = 5.0
PROXY_WINDOWS_MAX = 8
# "Play sequence" queues the windows of its next clips behind the focus (``queue``); at most this
# many wait, so a long sequence cannot evict the windows it is about to play
PROXY_WINDOWS_QUEUED = 4
# The page plays a window up to this many centiseconds before its end (app.js ``covers``, FRAME)
PROXY_WINDOW_FRAME_CS = 4
# ui/proxy/<proxy key>.w<lo>-<hi>.mp4 (see Media._proxy_key), the range in centiseconds of
# the source
PROXY_WINDOW_FILE = re.compile(r"([0-9a-f]{16})\.w(\d+)-(\d+)\.mp4")
# a whole proxy ui/proxy/<proxy key>.mp4 or a window
PROXY_FILE = re.compile(r"([0-9a-f]{16})(?:\.w\d+-\d+)?\.mp4")
# prune_proxies leaves files written or used this recently: another run on the project may be
# encoding one (a temporary grows as it encodes), or about to serve or serving one
# (Media.window and the proxy routes touch what they serve, see _mark_used)
PROXY_PRUNE_GRACE_S = 3600.0
# a ``key=value`` line of ffmpeg's ``-progress`` output (frame=, out_time_us=, progress=...)
PROXY_PROGRESS_LINE = re.compile(r"([a-z0-9_]+)= *(\S*)")  # bitrate= 4.1kbits/s


def proxy_args(
    real: Path,
    out: Path,
    audio_streams: int,
    *,
    encoder: VideoEncoder | None = None,
    color: str = "",
    lo: float | None = None,
    hi: float | None = None,
) -> list[str]:
    """ffmpeg command for a 720p H.264/AAC preview; several audio tracks are mixed as in exports.

    ``encoder``: a GPU encoder (NVENC, VAAPI or AMF), else libx264. Decoding stays on the CPU:
    ``-hwaccel cuda`` halved the CPU time but made the build two to three times slower on an RTX 3080.
    ``color``: the source's :func:`~cutpoint.encode.sdr_filter` (HDR and BT.2020 become SDR
    Rec.709, tagged so). ``lo``/``hi``: only that part of the source, in seconds (a window
    proxy, whose 0 is ``lo``).
    """
    mix = audio_mix_filter(audio_streams)
    audio = ["-filter_complex", f"{mix}[a]", "-map", "[a]"] if mix else ["-map", "0:a:0?"]
    video = (
        [*encoder.args(PROXY_GPU_QUALITY), *(PROXY_NVENC_RATE if encoder.kind == "nvenc" else [])]
        if encoder
        else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p"]
    )
    scale = "scale=-2:min(720\\,ih)"
    vf = f"{scale},{color},format=yuv420p" if color else scale
    if encoder:
        vf = encoder.chain(vf)
    start = ["-ss", f"{lo:.3f}"] if lo is not None else []
    length = ["-t", f"{hi - (lo or 0.0):.3f}"] if hi is not None else []
    return [
        "ffmpeg", "-y", "-nostdin", "-v", "error", *start, "-i", str(real), "-map", "0:v:0",
        *audio, "-vf", vf, *video,
        *(BT709_TAGS if color else []), *length,
        "-g", str(PROXY_GOP), "-fpsmax", str(PROXY_FPS_MAX),
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out),
    ]  # fmt: skip


def _tail(output: str, lines: int = 5) -> str:
    """The last ``lines`` non-empty lines of an encode's output, trimmed: why it failed."""
    kept = [line.strip() for line in output.splitlines() if line.strip()]
    return "\n".join(kept[-lines:])[-500:]


def _exit_reason(code: int, tail: str) -> str:
    """``tail``, or the exit code when the encode printed nothing (e.g. it was killed)."""
    if tail:
        return tail
    hexa = f" (0x{code & 0xFFFFFFFF:08X})" if code < 0 or code > 255 else ""  # Windows NTSTATUS
    return f"ffmpeg exited with code {code}{hexa}"


def _idle(file: Path, cutoff_ns: int) -> bool:
    """``file`` was last written or touched before ``cutoff_ns``; False if it is gone."""
    try:
        return file.stat().st_mtime_ns < cutoff_ns
    except OSError:
        return False


def _mark_used(file: Path) -> None:
    """A proxy is being served: touch it so another run's prune_proxies leaves it. Only once a
    quarter of the grace period has passed, so its Last-Modified rarely changes mid-playback."""
    with contextlib.suppress(OSError):
        if time.time() - file.stat().st_mtime > PROXY_PRUNE_GRACE_S / 4:
            os.utime(file)


def _covers(span: tuple[int, int], want: tuple[int, int], dur_cs: int) -> bool:
    """The window ``span`` plays all of ``want`` (centiseconds) as the page sees it (app.js
    ``covers``): not its last frame, unless it ends with the source (``dur_cs``, 0 if unknown)."""
    lo, hi = span
    return want[0] >= lo and (
        want[1] < hi - PROXY_WINDOW_FRAME_CS
        or (dur_cs > 0 and hi >= dur_cs - PROXY_WINDOW_FRAME_CS and want[1] <= hi)
    )


class ProxyWindow:
    """One queued or running window proxy encode: ``lo``/``hi`` in centiseconds of the source."""

    def __init__(
        self, project: Path, real: Path, key: str, lo: int, hi: int, audio_streams: int, color: str
    ) -> None:
        self.project, self.real, self.key = project, real, key
        self.lo, self.hi = lo, hi
        self.audio_streams, self.color = audio_streams, color
        self.proc: subprocess.Popen[str] | None = None  # set and cleared under ``Media.lock``
        self.cancelled = False  # superseded or no longer needed: never starts, or is killed

    @property
    def name(self) -> str:
        return f"{self.key}.w{self.lo}-{self.hi}"


# --------------------------------------------------------------------------- playable media
class Media:
    """Probe cache and on-demand H.264/AAC proxies for sources a browser cannot play.

    Two lanes, one worker each: whole-source proxies (``pool``, first in, first out) and window
    proxies of the focused moment (``windows_queue``, the latest focus request only), so a
    moment plays within seconds while its source's whole proxy still builds.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # key -> (browser plays it as is, duration, audio tracks, sdr_filter of a proxy) or None
        self.probed: dict[str, tuple[bool, float, int, str] | None] = {}
        # (project, key) -> "preparing" | "failed": each project builds its own proxy file
        self.proxies: dict[tuple[Path, str], str] = {}
        # (project, key) -> share of its whole proxy encoded, 0-1
        self.progress: dict[tuple[Path, str], float] = {}
        self.procs: set[subprocess.Popen[str]] = set()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cutpoint-proxy")
        # (project, key) -> {(lo, hi) centiseconds: file} of ready windows, least recently used first
        self.windows: dict[tuple[Path, str], dict[tuple[int, int], Path]] = {}
        # (project, key) -> windows of a source whose whole proxy landed: no longer listed, but
        # still served (a page may be playing one) until the next run's prune_proxies deletes them
        self.retired: dict[tuple[Path, str], dict[tuple[int, int], Path]] = {}
        # (project, key) -> {(lo, hi) centiseconds: short reason} of window encodes that failed,
        # oldest first; listed so the page stops waiting, cleared when one is asked for again
        self.window_failures: dict[tuple[Path, str], dict[tuple[int, int], str]] = {}
        self.windows_queue: deque[ProxyWindow] = deque()
        self.window_running: ProxyWindow | None = None
        self.window_wake = threading.Condition(self.lock)
        self.window_lane: threading.Thread | None = None  # started with the first window
        self.proxies_pruned: set[Path] = set()  # projects whose proxy leftovers went this run
        # (project, content key) of HDR sources whose proxies of an earlier tone map went
        self.tone_maps_pruned: set[tuple[Path, str]] = set()
        self.closing = False
        self.gpu_failed = False  # a proxy failed with the GPU encoder and worked with libx264
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

    def info(self, real: Path, key: str) -> tuple[bool, float, int, str] | None:
        """``(browser plays it as is, duration, audio tracks, colour filter of its proxy)``, or
        None if unreadable. The filter is :func:`~cutpoint.encode.sdr_filter`'s, "" for a source
        that plays as is."""
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
                self.probed[key] = (ok, i.duration, i.audio_streams, "" if ok else sdr_filter(i))
            except (FFmpegError, OSError, ValueError, KeyError):
                self.probed[key] = None
        return self.probed[key]

    def _proxy_key(self, key: str | None) -> str | None:
        """The key naming a source's proxies (whole, windows, builds): its content key, mixed
        with the colour filter of a probed source that has one, so a changed tone map rebuilds
        HDR proxies while the others keep their names. Never runs ffprobe."""
        res = self.probed.get(key) if key else None
        if res is None or not res[3]:
            return key
        return hashlib.sha1(f"{key}|{res[3]}".encode()).hexdigest()[:16]

    def _lookup(self, real: Path) -> tuple[str | None, tuple[bool, float, int, str] | None]:
        """``(content key, info)`` of ``real``; both None if it is missing."""
        key = self.key(real)
        return key, self.info(real, key) if key else None

    def duration(self, real: Path) -> float | None:
        res = self._lookup(real)[1]
        return res[1] if res else None

    def probed_duration(self, path: str) -> float | None:
        """The duration of ``path`` if it is probed already; never runs ffprobe."""
        key = self.key(remap_source(path))
        res = self.probed.get(key) if key else None
        return res[1] if res else None

    def status(self, project: Path, path: str) -> tuple[str, Path | None]:
        """``(status, file to serve)``: ready | needs-proxy | preparing | failed. No side effects."""
        real = remap_source(path)
        key, res = self._lookup(real)
        if res is None or key is None:
            return "failed", None
        if res[0]:
            return "ready", real
        key = self._proxy_key(key)
        proxy = project / UI_DIR / "proxy" / f"{key}.mp4"
        if proxy.is_file() and _inside(project, f"{UI_DIR}/proxy/{key}.mp4"):
            return "ready", proxy
        return {"preparing": "preparing", "failed": "failed"}.get(
            self.proxies.get((project, key), ""), "needs-proxy"
        ), None

    def cached_status(self, project: Path, path: str) -> str:
        """As :meth:`status`, or "unknown" while ``path`` is not probed yet; never runs ffprobe."""
        return self.cached_view(project, path)[0]

    def cached_view(self, project: Path, path: str) -> tuple[str, bool, float | None]:
        """``(cached_status, the source plays from a proxy, share of its whole proxy encoded or
        None while that is not encoding)``; never runs ffprobe."""
        key = self.key(remap_source(path))
        if key is not None and key not in self.probed:
            return "unknown", False, None
        status = self.status(project, path)[0]
        res = self.probed.get(key) if key else None
        pkey = self._proxy_key(key)
        with self.lock:
            done = self.progress.get((project, pkey)) if pkey and status == "preparing" else None
        return status, res is not None and not res[0], done

    def prepare(
        self,
        project: Path,
        path: str,
        window: tuple[float, float] | None = None,
        *,
        queue: bool = False,
    ) -> str:
        """Queue the proxy build of one source (also retries a failed build); returns its status.

        ``window``: a ``(lo, hi)`` range in seconds the user is looking at; a window proxy around
        it is queued first, in its own lane (see :meth:`_focus`). ``queue``: it waits behind the
        windows already queued instead of superseding them ("Play sequence" asks for its next
        clips so).
        """
        real = remap_source(path)
        content, res = self._lookup(real)
        key = self._proxy_key(content)
        if content is not None and key != content:
            self._drop_earlier_tone_map(project, content)
        if res is not None and key is not None and not res[0] and not self.closing:
            proxy = _own(project, f"{UI_DIR}/proxy/{key}.mp4")  # never built through a link
            _own(project, f"{UI_DIR}/proxy/{key}.part.mp4")  # nor its temporary
            if window is not None and not proxy.is_file():
                self._focus(project, real, key, res, *window, queue=queue)
            with self.lock:
                if not proxy.is_file() and self.proxies.get((project, key)) != "preparing":
                    self.proxies[project, key] = "preparing"
                    self.pool.submit(self._build, project, real, proxy, key, res[2], res[3], res[1])
        return self.status(project, path)[0]

    def _focus(
        self,
        project: Path,
        real: Path,
        key: str,
        res: tuple[bool, float, int, str],
        lo: float,
        hi: float,
        *,
        queue: bool = False,
    ) -> None:
        """The user looks at ``[lo, hi]`` s of a source: queue a window proxy around it unless a
        ready or running one covers it. The request goes ahead of (and drops) earlier queued
        ones, and a running window encode that does not overlap it is killed.

        ``queue``: it goes behind the queued ones instead, unless one of them covers it or
        PROXY_WINDOWS_QUEUED wait already (the page asks again later), and kills nothing."""
        dur = res[1]
        if dur > 0:
            lo, hi = min(lo, dur), min(hi, dur)
        step = PROXY_WINDOW_STEP
        a = max(0.0, math.floor((lo - PROXY_WINDOW_BEFORE) / step) * step)
        b = math.ceil((hi + PROXY_WINDOW_AFTER) / step) * step
        dur_cs = math.floor(dur * 100) if dur > 0 else 0
        span = (round(a * 100), min(math.floor(b * 100), dur_cs) if dur_cs else math.floor(b * 100))
        if span[1] <= span[0]:
            return
        want = (min(math.floor(lo * 100), span[1]), min(math.ceil(hi * 100), span[1]))
        win = ProxyWindow(project, real, key, *span, res[2], res[3])
        _own(project, f"{UI_DIR}/proxy/{win.name}.mp4")  # never built through a link
        _own(project, f"{UI_DIR}/proxy/{win.name}.part.mp4")  # nor its temporary
        with self.lock:
            ready = self.windows.get((project, key), {})
            covered = {s: f for s, f in ready.items() if _covers(s, want, dur_cs)}
        gone = [s for s, f in covered.items() if not f.is_file()]  # deleted by hand meanwhile
        kill = None
        with self.lock:
            if self.closing:
                return
            for s in gone:
                ready.pop(s, None)
            if not queue:
                self.windows_queue.clear()  # superseded: the user moved on
            run = self.window_running
            mine = run is not None and (run.project, run.key) == (project, key)
            waits = queue and (
                len(self.windows_queue) >= PROXY_WINDOWS_QUEUED
                or any(
                    (w.project, w.key) == (project, key) and _covers((w.lo, w.hi), want, dur_cs)
                    for w in self.windows_queue
                )
            )
            if mine and not run.cancelled and _covers((run.lo, run.hi), want, dur_cs):
                pass  # the running encode covers it
            elif waits:
                pass  # a queued one covers it, or the queue is full
            elif len(covered) == len(gone):
                self.windows_queue.append(win)
                failed = self.window_failures.get((project, key), {})
                for sp in [sp for sp in failed if sp[0] < win.hi and win.lo < sp[1]]:
                    del failed[sp]  # asked for again: no longer a failure to show
                if self.window_lane is None:
                    self.window_lane = threading.Thread(
                        target=self._window_lane, name="cutpoint-proxy-window", daemon=True
                    )
                    self.window_lane.start()
                self.window_wake.notify()
            if (
                not queue
                and run is not None
                and not (mine and run.lo < span[1] and span[0] < run.hi)
            ):
                run.cancelled = True  # does not start if it has not yet
                kill, run.proc = run.proc, None
        if kill is not None:
            threading.Thread(target=kill_tree, args=(kill, SOFT_STOP_S), daemon=True).start()

    def _window_lane(self) -> None:
        """The window proxies' one worker, beside the whole-source lane."""
        while True:
            with self.lock:
                while not self.windows_queue and not self.closing:
                    self.window_wake.wait()
                if self.closing:
                    return
                win = self.window_running = self.windows_queue.popleft()
            try:
                self._build_window(win)
            finally:
                with self.lock:
                    self.window_running = None

    def _build_window(self, win: ProxyWindow) -> None:
        p, name = win.project, f"{UI_DIR}/proxy/{win.name}"
        tmp, out = p / f"{name}.part.mp4", p / f"{name}.mp4"
        what = f"{win.real.name} {win.lo / 100:.2f}-{win.hi / 100:.2f} s"
        drop: list[Path] = []
        try:
            _ui_folder(p, "proxy")  # ui or ui/proxy a file: a clear refusal, not a traceback
            _own(p, f"{name}.part.mp4")  # a link made while this window was queued
            gpu = self._gpu_encoder()

            def encode(enc: VideoEncoder | None) -> tuple[int, str]:
                args = proxy_args(
                    win.real, tmp, win.audio_streams, encoder=enc, color=win.color,
                    lo=win.lo / 100, hi=win.hi / 100,
                )  # fmt: skip
                return self._encode(args, win)

            code, tail = encode(gpu)
            if code != 0 and gpu and not self.closing and not win.cancelled:
                logger.warning(
                    "window proxy of %s: %s failed (%s); retrying with libx264",
                    what, gpu.codec, tail,
                )  # fmt: skip
                code, tail = encode(None)
                self.gpu_failed = code == 0
            if code == 0 and not self.closing and not win.cancelled:
                _own(p, f"{name}.mp4")  # a link made during the encode
                with self.lock:
                    # the whole proxy landed meanwhile: this window is not needed
                    if not (p / UI_DIR / "proxy" / f"{win.key}.mp4").is_file():
                        tmp.replace(out)
                        drop = self._keep_window((p, win.key), (win.lo, win.hi), out)
            elif not win.cancelled and not self.closing:
                reason = _exit_reason(code, tail)
                logger.warning("window proxy of %s failed: %s", what, reason)
                self._window_failed(win, reason)
        except HTTPException as err:  # refused: the temporary is removed below
            logger.warning("window proxy of %s failed: %s", what, err.detail)
            self._window_failed(win, str(err.detail))
        except OSError as err:
            logger.exception("window proxy of %s failed", what)
            self._window_failed(win, str(err))
        finally:
            with contextlib.suppress(OSError):  # e.g. NotADirectoryError when ui is a file
                tmp.unlink(missing_ok=True)
            self._delete(drop)

    def _window_failed(self, win: ProxyWindow, reason: str) -> None:
        """Record a failed window encode for the listing: its last line is the reason shown."""
        last = reason.splitlines()[-1] if reason else ""
        with self.lock:
            failed = self.window_failures.setdefault((win.project, win.key), {})
            failed.pop((win.lo, win.hi), None)
            failed[(win.lo, win.hi)] = last[:200]
            while len(failed) > PROXY_WINDOWS_MAX:
                del failed[next(iter(failed))]

    def failed_windows(self, project: Path, path: str) -> list[tuple[int, int, str]]:
        """``(lo, hi, reason)`` of a source's failed window encodes, in centiseconds; no stat
        while the project has none."""
        with self.lock:
            if not any(p == project and f for (p, _), f in self.window_failures.items()):
                return []
        key = self._proxy_key(self.key(remap_source(path)))
        with self.lock:
            return [
                (lo, hi, r) for (lo, hi), r in self.window_failures.get((project, key), {}).items()
            ]

    def _keep_window(self, pid: tuple[Path, str], span: tuple[int, int], file: Path) -> list[Path]:
        """Record a ready window (call with ``self.lock`` held); returns the windows evicted for
        it, least recently used first, for the caller to delete."""
        ready = self.windows.setdefault(pid, {})
        ready.pop(span, None)
        ready[span] = file
        drop = []
        while len(ready) > PROXY_WINDOWS_MAX:
            drop.append(ready.pop(next(iter(ready))))
        return drop

    @staticmethod
    def _delete(files: list[Path]) -> None:
        for f in files:
            try:
                f.unlink(missing_ok=True)
            except OSError as err:  # e.g. still being sent on Windows; pruned next run
                logger.debug("could not delete %s: %s", f.name, err)

    def _drop_earlier_tone_map(self, project: Path, content: str) -> None:
        """Once per run: delete an HDR source's proxies named by its content key alone (built
        with an earlier tone map), unless written or used in the last PROXY_PRUNE_GRACE_S."""
        folder = project / UI_DIR / "proxy"
        cutoff = time.time_ns() - int(PROXY_PRUNE_GRACE_S * 1e9)
        with self.lock:
            if (project, content) in self.tone_maps_pruned:
                return
            self.tone_maps_pruned.add((project, content))
            self.windows.pop((project, content), None)  # adopted before the source was probed
            if not folder.is_dir() or not _inside(project, f"{UI_DIR}/proxy"):
                return  # never delete in a folder a link takes outside the project
            stale = [f for f in folder.glob(f"{content}*.mp4") if PROXY_FILE.fullmatch(f.name)]
        self._delete([f for f in stale if _idle(f, cutoff)])

    def _drop_windows(self, project: Path, key: str) -> None:
        """The whole proxy of ``key`` landed: stop that source's window encodes and retire its
        windows (still served, no longer listed; the next run's :meth:`prune_proxies` deletes
        them, so a page playing one is not cut off)."""
        kill = None
        with self.lock:
            done = self.windows.pop((project, key), None)
            if done:
                self.retired.setdefault((project, key), {}).update(done)
            self.window_failures.pop((project, key), None)
            for win in [w for w in self.windows_queue if (w.project, w.key) == (project, key)]:
                self.windows_queue.remove(win)
            run = self.window_running
            if run is not None and (run.project, run.key) == (project, key):
                run.cancelled = True
                kill, run.proc = run.proc, None
        if kill is not None:
            threading.Thread(target=kill_tree, args=(kill, SOFT_STOP_S), daemon=True).start()

    def window(self, project: Path, path: str, lo: int, hi: int) -> Path | None:
        """The ready window ``lo``-``hi`` (centiseconds) of a source, else None; marks it used."""
        key = self._proxy_key(self.key(remap_source(path)))
        if key is None:
            return None
        with self.lock:
            ready = self.windows.get((project, key))
            file = ready.pop((lo, hi), None) if ready else None
            if ready is not None and file is not None:
                ready[(lo, hi)] = file  # recently used: evicted last
            elif file is None:
                file = self.retired.get((project, key), {}).get((lo, hi))
        if (
            file is None
            or not file.is_file()
            or not _inside(project, f"{UI_DIR}/proxy/{file.name}")
        ):
            return None
        _mark_used(file)
        return file

    def windows_of(
        self, project: Path, path: str
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int, str]]]:
        """``(ready windows, pending ones with "encoding" | "queued")`` of a source, in
        centiseconds, from memory only; no stat at all while the project has none."""
        with self.lock:
            busy = [
                w for w in (self.window_running, *self.windows_queue)
                if w is not None and w.project == project and not w.cancelled
            ]  # fmt: skip
            any_ready = any(p == project and r for (p, _), r in self.windows.items())
        if not busy and not any_ready:
            return [], []
        key = self._proxy_key(self.key(remap_source(path)))
        with self.lock:
            ready = sorted(self.windows.get((project, key), {}))
            run = self.window_running
        pending = [(w.lo, w.hi, "encoding" if w is run else "queued") for w in busy if w.key == key]
        return ready, pending

    def prune_proxies(self, project: Path, paths: list[str]) -> None:
        """Once per run and project: delete leftover temporaries, the whole proxies and windows
        of sources that changed or left or whose tone map changed, and windows whose whole proxy
        is built; adopt the other windows on disk. A file written or used in the last
        PROXY_PRUNE_GRACE_S stays (another run may be encoding or serving it), as does one that
        cannot be deleted (open on Windows); a later run deletes it. While a source cannot be
        read (an offline drive), no proxy goes for naming no current source, and no source is
        probed.

        A proxy no known key names may be of an HDR source not probed yet (its proxy key holds
        the colour filter): sources are then probed, in order, until every such proxy is
        matched, so this may run ffprobe. Proxies an HDR source's content key names (an earlier
        tone map) go in :meth:`prepare` instead, so no other source is probed for them."""
        folder = project / UI_DIR / "proxy"
        if project in self.proxies_pruned:
            return
        cutoff = time.time_ns() - int(PROXY_PRUNE_GRACE_S * 1e9)
        reals = [remap_source(path) for path in paths]
        keys = {self._proxy_key(self.key(real)) for real in reals}
        unsure = None in keys  # a source is unreachable: its proxies' names are unknown
        keys.discard(None)
        named = {m[1] for m in (PROXY_FILE.fullmatch(f.name) for f in folder.glob("*.mp4")) if m}
        left = set() if unsure else named - keys
        for real in reals:
            if not left:
                break
            key, res = self._lookup(real)
            if res is not None and res[3]:  # only a colour filter changes the key
                pkey = self._proxy_key(key)
                keys.discard(key)  # its windows from an earlier tone map go
                keys.add(pkey)
                left.discard(pkey)
        drop: list[Path] = []
        with self.lock:
            self.proxies_pruned.add(project)
            if not folder.is_dir() or not _inside(project, f"{UI_DIR}/proxy"):
                return  # never delete in a folder a link takes outside the project
            busy = {
                f"{k}.part.mp4"
                for (q, k), st in self.proxies.items()
                if q == project and st == "preparing"
            }
            busy |= {
                f"{w.name}.part.mp4"
                for w in (self.window_running, *self.windows_queue)
                if w is not None and w.project == project
            }
            drop += [f for f in folder.glob("*.part.mp4") if f.name not in busy]
            if not unsure:
                drop += [
                    f
                    for f in folder.glob("*.mp4")
                    if (m := PROXY_FILE.fullmatch(f.name))
                    and ".w" not in f.name
                    and m[1] not in keys
                ]  # whole proxies
            found: list[tuple[int, Path, str, tuple[int, int]]] = []
            for f in folder.glob("*.w*.mp4"):
                m = PROXY_WINDOW_FILE.fullmatch(f.name)
                if m is None:
                    continue
                k = m.group(1)
                if (folder / f"{k}.mp4").is_file():
                    drop.append(f)
                elif k not in keys:
                    if not unsure:
                        drop.append(f)
                else:
                    with contextlib.suppress(OSError):
                        found.append((f.stat().st_mtime_ns, f, k, (int(m[2]), int(m[3]))))
            for _, f, k, span in sorted(found):  # oldest first: evicted first
                if span not in self.windows.get((project, k), {}):
                    drop += self._keep_window((project, k), span, f)
        self._delete([f for f in drop if _idle(f, cutoff)])

    def _build(
        self,
        project: Path,
        real: Path,
        proxy: Path,
        key: str,
        audio_streams: int,
        color: str = "",
        duration: float = 0.0,
    ) -> None:
        tmp = proxy.with_name(f"{key}.part.mp4")
        state = "failed"
        made: list[Path] = []  # ui folders this build created
        try:
            # ui or ui/proxy a file: a clear refusal, not a traceback
            _ui_folder(project, "proxy", made)
            # a link made while this build was queued
            _own(project, f"{UI_DIR}/proxy/{key}.part.mp4")
            gpu = self._gpu_encoder()
            track = ((project, key), duration)  # the page shows how far the whole proxy got
            code, tail = self._encode(
                proxy_args(real, tmp, audio_streams, encoder=gpu, color=color), progress=track
            )
            if code != 0 and gpu and not self.closing:
                # The probe encodes 0.2 s of black: a driver can still fail on real footage.
                logger.warning(
                    "proxy of %s: %s failed (%s); retrying with libx264", real.name, gpu.codec, tail
                )
                code, tail = self._encode(
                    proxy_args(real, tmp, audio_streams, color=color), progress=track
                )
                self.gpu_failed = code == 0
            if code == 0 and not self.closing:
                _own(project, f"{UI_DIR}/proxy/{key}.mp4")  # a link made during the encode
                tmp.replace(proxy)
                state = ""
                self._drop_windows(project, key)  # the whole proxy replaces them
            else:
                logger.warning("proxy of %s failed: %s", real.name, tail)
        except HTTPException as err:  # refused: the temporary is removed below
            logger.warning("proxy of %s failed: %s", real.name, err.detail)
        except OSError:
            logger.exception("proxy of %s failed", real.name)
        finally:
            with contextlib.suppress(OSError):  # e.g. NotADirectoryError when ui is a file
                tmp.unlink(missing_ok=True)
            with self.lock:
                self.progress.pop((project, key), None)
                if state:
                    self.proxies[project, key] = state
                else:
                    self.proxies.pop((project, key), None)
                # Accepted narrow race: a request writing into a just-made ui/ right now gets one
                # retryable error. Queued builds (one worker) recreate the folders themselves.
                if state:
                    for folder in made:  # failed: the folders it created, while still empty
                        with contextlib.suppress(OSError):
                            folder.rmdir()

    def _gpu_encoder(self) -> VideoEncoder | None:
        """The GPU encoder for proxies: the one exports use (``video_encoder``, probed once),
        or None for libx264, also after a proxy failed with it and worked with libx264. AMF
        proxies stay on libx264: their size was never measured (VAAPI's was, see below)."""
        if self.gpu_failed:
            return None
        try:
            enc = video_encoder()
        except (ValueError, EncoderUnavailableError):
            return None
        return None if enc.kind in ("cpu", "amf") else enc

    def _encode(
        self,
        args: list[str],
        window: ProxyWindow | None = None,
        progress: tuple[tuple[Path, str], float] | None = None,
    ) -> tuple[int, str]:
        """Run one proxy encode; ``(exit code, end of its output)``.

        Spawned and registered under ``self.lock`` once ``closing`` (and the ``window``'s
        ``cancelled``) is checked, so ``shutdown`` (or :meth:`_focus`) either kills it or it
        never starts. ``window.proc`` holds the process while it runs; a window encodes at normal
        priority (the user waits on it, and the whole proxy beside it would starve it).
        ``progress``: ``((project, key), source duration)``; ffmpeg reports how far it got, kept
        in ``self.progress[(project, key)]``.
        """
        if progress is not None:
            args = [args[0], "-progress", "pipe:1", "-nostats", *args[1:]]
        with self.lock:
            if self.closing:
                return 1, "the server is shutting down"
            if window is not None and window.cancelled:
                return 1, "no longer needed"
            proc = spawn(args, low_priority=window is None)
            self.procs.add(proc)
            if window is not None:
                window.proc = proc
            if progress is not None:
                self.progress[progress[0]] = 0.0
        try:
            if progress is None:
                tail = _tail(proc.communicate()[0] or "")  # stderr is merged into it
            else:
                tail = self._read_progress(proc, *progress)
        finally:
            with self.lock:
                self.procs.discard(proc)
                if window is not None and window.proc is proc:
                    window.proc = None
        return proc.returncode, tail

    def _read_progress(
        self, proc: subprocess.Popen[str], key: tuple[Path, str], duration: float
    ) -> str:
        """Read an encode's output (``-progress pipe:1`` merged with its errors) until it exits:
        ``out_time_us`` updates ``self.progress[key]``; returns the end of the other lines."""
        rest: deque[str] = deque(maxlen=20)
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.strip()
            m = PROXY_PROGRESS_LINE.fullmatch(line)
            if m is None:
                if line:
                    rest.append(line)
            elif m[1] == "out_time_us" and m[2].isdigit() and duration > 0:  # "N/A" at first
                with self.lock:
                    self.progress[key] = min(1.0, int(m[2]) / 1e6 / duration)
        proc.wait()
        return "\n".join(rest)[-500:]

    def frame(self, project: Path, path: str, at: float, width: int) -> Path | None:
        """A JPEG still of ``path`` at ``at`` s (clamped into the source), ``width`` px wide.

        Cached as ``ui/frames/<content key>-<centiseconds>-<width>.jpg`` and built once even when
        requested concurrently. None if the source cannot be read.

        Raises:
            FFmpegError: If ffmpeg decodes no image.
        """
        real = remap_source(path)
        key, res = self._lookup(real)
        if res is None or key is None:
            return None
        if res[1] > 0:
            at = min(at, max(0.0, res[1] - 0.1))  # the last instants may hold no decodable frame
        cs = round(at * 100)
        out = _own(project, f"{UI_DIR}/frames/{key}-{cs}-{width}.jpg")  # nor cached through one
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
                    _ui_folder(project, "frames")  # ui or ui/frames a file: a clear 409
                    tmp = _own(project, f"{UI_DIR}/frames/{out.stem}.{threading.get_ident()}.part")
                    try:
                        tmp.write_bytes(image)
                        tmp.replace(out)
                    finally:
                        with contextlib.suppress(OSError):  # keeps the write's own error
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
            if not folder.is_dir() or not _inside(project, f"{UI_DIR}/frames"):
                return  # never delete in a folder a link takes outside the project
            stale = [f for f in folder.glob("*.jpg") if f.name.split("-")[0] not in keys]
            self._drop_frames([*stale, *folder.glob("*.part")])

    def shutdown(self) -> None:
        with self.lock:  # as in ``_encode``: no proxy ffmpeg starts after this
            self.closing = True
            procs = list(self.procs)
            self.windows_queue.clear()
            self.window_wake.notify_all()  # the window lane ends
        self.pool.shutdown(wait=False, cancel_futures=True)
        for proc in procs:
            kill_tree(proc, SOFT_STOP_S)  # a bare ffmpeg: no soft stop of its own to wait for


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
    if not ok or not _is_project(p):
        raise bad("unknown project", 404)
    return p


def read_meta(p: Path) -> dict[str, Any]:
    meta = _load(p, f"{UI_DIR}/meta.json")
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


def _listed_folder(folder: Path) -> Path:
    """Where a picked folder is listed: in place, else at its ``source_remap`` target, as
    ``analyze`` lists it."""
    return folder if os.path.exists(folder) else remap_source(folder)  # noqa: PTH110 - any OSError: missing


def _pano_folder(source: str | Path) -> Path:
    """The camera folder of a 360 project's picked source: the folder, or a picked file's."""
    src = Path(str(source))
    return src.parent if src.suffix.lower() in PANO_EXTS else src


def source_entries(
    p: Path, hl: dict[str, Any] | None, listed: list[Path | str] | None = None
) -> list[dict[str, Any]]:
    """The project's playable-in-principle sources: ``[{path, id, name, duration}]``. The source
    it lists itself (a picked folder before its analysis) is added to ``listed``, followed by
    the error message when that listing failed."""
    meta = read_meta(p)
    pano = meta.get("kind") == "360"
    if meta.get("source"):
        roots = [str(_pano_folder(meta["source"]) if pano else meta["source"])]
    else:  # command-line project: the inputs its analysis recorded
        roots = [str(x) for x in ((hl or {}).get("settings") or {}).get("inputs") or []]
    paths: dict[str, float | None] = {}
    for s in (hl or {}).get("sources") or []:
        if isinstance(s, dict) and isinstance(s.get("path"), str):
            dur = s.get("duration")
            paths.setdefault(s["path"], dur if isinstance(dur, (int, float)) else None)
    if not paths and meta.get("source"):
        folder = Path(str(meta["source"]))
        if listed is not None:
            listed.append(folder)
        try:
            if pano:  # one .lrv per clip 360 analysis can use (those of meta.clips, if set)
                only = meta.get("clips") or None
                _, clips, _ = inspect_source(folder)
                found = [clips[n]["lrv"] for n in _usable(clips) if not only or n in only]
            else:
                io = _listed_folder(folder)
                found = find_videos([io])
                if io != folder:  # read at its remap target: videos keep the original spelling
                    base, orig = io.resolve(), folder.resolve()
                    found = [
                        orig / f.relative_to(base) if f.is_relative_to(base) else f for f in found
                    ]
                    # source_remap_exclude: they would be read at the missing original
                    found = [f for f in found if remap_source(f) != f]
            paths = dict.fromkeys(map(str, found))
        except OSError as err:  # find_videos: missing, unreadable, stale mount
            if listed is not None:
                listed.append(str(err))
    return [
        {
            "path": path,
            "id": hashlib.sha1(path.encode()).hexdigest()[:12],
            "name": Path(path).name,
            "duration": dur,
        }
        for path, dur in paths.items()
        if Path(path).is_absolute()
        and Path(path).suffix.lower() in (VIDEO_EXTS | {".lrv"} if pano else VIDEO_EXTS)
        and _under(path, roots)
    ]


class Sources:
    """Each project's :func:`source_entries`, kept while its inputs are unchanged, so media
    requests (each Range request, each still) do not reread and rescan a network share.

    The list depends only on ``ui/meta.json``, ``highlights.json`` and, before an analysis, the
    entries directly inside the picked folder (``find_videos`` does not recurse); a folder's mtime
    changes when one is added, removed or renamed. So a hit costs two or three stats, and any
    change to them rebuilds it, as does a change to ``source_remap`` or ``source_remap_exclude``
    (``get_config`` rereads an edited ``config.json``). ``SOURCES_CACHE_S`` bounds what stats
    cannot see (a link under a source retargeted, a filesystem with coarse mtimes). Callers must
    not change the entries."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # project folder -> (built at, files stamped, their stamps, remap settings, entries);
        # oldest use first
        self.kept: dict[str, tuple[float, list[Path], list[Any], Any, list[dict[str, Any]]]] = {}

    def entries(self, p: Path) -> list[dict[str, Any]]:
        return self.listing(p)[0]

    def listing(self, p: Path) -> tuple[list[dict[str, Any]], str | None]:
        """``(entries, why the picked folder could not be listed or None)``."""
        key = str(p)
        with self.lock:
            hit = self.kept.get(key)
        cfg = get_config()
        remap = (cfg.source_remap, cfg.source_remap_exclude)
        if (
            hit is not None
            and time.monotonic() - hit[0] < SOURCES_CACHE_S
            and hit[3] == remap
            and [file_stamp(f) for f in hit[1]] == hit[2]
        ):
            with self.lock:
                if self.kept.get(key) is hit:
                    self.kept[key] = self.kept.pop(key)  # recently used: evicted last
            return hit[4], None
        # Each file is stamped before it is read: a change made meanwhile shows on the next request.
        files = [p / f"{UI_DIR}/meta.json", p / "highlights.json"]
        at, stamps = time.monotonic(), [file_stamp(f) for f in files]
        meta = read_meta(p)
        source = meta.get("source")
        if source and meta.get("kind") == "360":
            source = _pano_folder(source)  # its .lrv files are listed, a picked file's too
        files += [_listed_folder(Path(str(source)))] if source else []
        stamps += [file_stamp(files[2])] if source else []
        listed: list[Path | str] = []
        entries = source_entries(p, _highlights(p), listed)
        n = 3 if listed else 2  # the picked source counts only while its folder is listed
        with self.lock:
            self.kept.pop(key, None)
            if len(listed) > 1:  # the folder could not be listed: the next request lists again
                return entries, str(listed[1])
            self.kept[key] = (at, files[:n], stamps[:n], remap, entries)
            while len(self.kept) > SOURCES_CACHE_MAX:
                del self.kept[next(iter(self.kept))]
        return entries, None


def source_folder(p: Path) -> Path | None:
    """The folder holding the project's videos: its picked source (a file's parent), or for a
    command-line project the deepest folder shared by its recorded sources. Never client input."""
    source = read_meta(p).get("source")
    try:
        if source:
            # A picked folder that is gone must not fall back to its parent; only files step up.
            x = remap_source(Path(str(source)))
            missing_file = not x.exists() and x.suffix.lower() in VIDEO_EXTS | PANO_EXTS
            is_file = x.is_file() or missing_file
            folders = [x.parent if is_file else x]
        else:
            entries = source_entries(p, _highlights(p))
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
        if start < 0 or end - start < MIN_EXPORT_S:
            raise bad(f"segment {i}: needs 0 <= start and end at least {MIN_EXPORT_S}s after start")
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
    data = _load(p, f"{UI_DIR}/exports.json")
    items = data.get("exports") if isinstance(data, dict) else None
    return [
        e
        for e in items or []
        if isinstance(e, dict) and isinstance(e.get("dir"), str) and SAFE_NAME.fullmatch(e["dir"])
    ]


def _inside(p: Path, name: str) -> bool:
    """``p / name`` still lies inside ``p`` once symlinks and junctions are resolved."""
    try:
        return (p / name).resolve().is_relative_to(p.resolve())
    except (OSError, RuntimeError):  # RuntimeError: a symlink loop (Python 3.12)
        return False


def _load(p: Path, name: str) -> Any | None:
    """The project file ``p / name`` as JSON; None (as if missing) when a link takes it outside."""
    return cache.load_json(p / name) if _inside(p, name) else None


def _highlights(p: Path) -> dict[str, Any] | None:
    """The project's ``highlights.json``; None when missing, outside or not an object."""
    hl = _load(p, "highlights.json")
    return hl if isinstance(hl, dict) else None


def _own(p: Path, name: str) -> Path:
    """``p / name`` to write, else 409 when a symlink or junction takes it outside the project."""
    if not _inside(p, name):
        raise bad(f"{name} resolves outside the project folder; remove the link", 409)
    return p / name


def _ui_folder(p: Path, sub: str = "", made: list[Path] | None = None) -> None:
    """Create ``p/ui`` (and ``p/ui/<sub>``) if missing; 409 when one is a file or a dangling link
    (never followed). The levels created here are added to ``made``, deepest first."""
    for name in (UI_DIR, f"{UI_DIR}/{sub}") if sub else (UI_DIR,):
        _own(p, name)
        folder = p / name
        try:
            folder.mkdir()
        except OSError as err:
            if folder.is_dir():  # already there (as mkdir(exist_ok=True))
                continue
            if isinstance(err, (FileExistsError, NotADirectoryError)):
                raise bad(f"{p.name}/{name} is not a folder; remove or rename it", 409) from None
            raise
        if made is not None:
            made.insert(0, folder)


def _is_project(d: Path) -> bool:
    """``d`` holds UI metadata or an analysis of its own (not one a link points to)."""
    return any(
        (d / n).is_file() and _inside(d, n) for n in (f"{UI_DIR}/meta.json", "highlights.json")
    )


def current_export(p: Path) -> dict[str, Any] | None:
    """The latest export whose clips are still on disk, inside the project."""
    return next(
        (
            e
            for e in reversed(load_exports(p))
            if (p / e["dir"] / "manifest.json").is_file()
            and _inside(p, e["dir"])
            and _inside(p, f"{e['dir']}/manifest.json")
        ),
        None,
    )


def _final_names(p: Path) -> list[str]:
    """Finals in the project folder: the current export's first, then the newest first."""
    exp = current_export(p)
    final = (exp or {}).get("final")
    first = [final] if isinstance(final, str) and SAFE_NAME.fullmatch(final) else []
    found: list[tuple[int, str]] = []
    with contextlib.suppress(OSError), os.scandir(p) as it:
        for e in it:
            if FINAL_VIDEO.fullmatch(e.name):
                with contextlib.suppress(OSError):
                    found.append((e.stat().st_mtime_ns, e.name))
    found.sort(reverse=True)
    names = dict.fromkeys([*first, *(n for _, n in found)])  # in order, no duplicates
    return [n for n in names if (p / n).is_file() and _inside(p, n)]


def output_names(p: Path) -> list[str]:
    """Files of the finals (each video, then its timelines and subtitles) inside the project."""
    out: list[str] = []
    for final in _final_names(p):
        stem = Path(final).stem
        out += [
            final,
            *(
                f"{stem}{e}"
                for e in (".fcpxml", ".otio", ".srt")
                if (p / f"{stem}{e}").is_file() and _inside(p, f"{stem}{e}")
            ),
        ]
    return out


def fresh_name(parent: Path, stem: str, exts: tuple[str, ...] = ("",)) -> str:
    """``stem`` or ``stem-N``: the first name no file ``name + ext`` already uses.

    A broken symlink counts as taken, so an output is never written through one.
    """
    n, name = 1, stem
    while any(os.path.lexists(parent / f"{name}{e}") for e in exts):
        n += 1
        name = f"{stem}-{n}"
    return name


def inspect_source(src: Path) -> tuple[list[Path], dict[str, dict[str, Any]], Counter[str]]:
    """Split a picked source into flat videos, Insta360 X3 clips and the suffixes of other files.

    One non-recursive listing of regular files (sub-folders are skipped); nothing is read or
    written. A folder gives its files; a file gives itself or, when it has an X3 camera name, the
    files of its clip next to it (one 360 file means that clip). Clips come from
    ``x3.group_files``; ``other`` counts the remaining suffixes, lower-cased ("no extension" when
    there is none).
    """
    f = x3.parse_name(src) if src.is_file() else None
    if src.is_file() and f is None:
        files = [src]
    else:
        listed = sorted(p for p in (src.parent if f else src).iterdir() if safe_is_file(p))
        files = [p for p in listed if not f or (x3.parse_name(p) or {}).get("num") == f["num"]]
    videos = [p for p in files if p.suffix.lower() in VIDEO_EXTS]
    other = Counter(
        p.suffix.lower() or "no extension"
        for p in files
        if p.suffix.lower() not in VIDEO_EXTS and not x3.NAME_RE.match(p.name)
    )
    return videos, x3.group_files(files), other


def _usable(clips: dict[str, dict[str, Any]]) -> list[str]:
    """Numbers of the X3 clips 360 analysis can use: those with a ``_00_.insv`` and a ``.lrv``."""
    return sorted(n for n, g in clips.items() if "lens00" in g and "lrv" in g)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _suffix_counts(counts: Counter[str], limit: int = 5) -> str:
    """Counts such as "12 .insv, 3 .lrv": most first, then by name; at most ``limit`` types."""
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    text = ", ".join(
        f"{n} {'without extension' if k == 'no extension' else k}" for k, n in items[:limit]
    )
    rest = len(items) - limit
    return f"{text} and {_plural(rest, 'more type')}" if rest > 0 else text


def source_problem(src: Path) -> str | None:
    """Why ``src`` cannot become a UI project, or None when it can: flat videos, or Insta360 X3
    footage with at least one clip 360 analysis can use (never both in one folder).

    Raises:
        OSError: If the source cannot be listed.
    """
    if src.is_file() and src.suffix.lower() not in VIDEO_EXTS | PANO_EXTS:
        return "source is not a video file"
    if not src.is_file() and not src.is_dir():
        return "source must be a video file or a folder containing videos"
    videos, clips, other = inspect_source(src)
    names = "Insta360 files must keep their camera names (VID_..._00_NNN.insv)."
    if not clips and not videos:
        if not other:
            return "source must be a video file or a folder containing videos"
        if src.is_file():
            return names
        hint = f". {names}" if PANO_EXTS & other.keys() else ""
        return f"no supported videos found (found {_suffix_counts(other)}){hint}"
    if not clips:
        return None
    if videos:
        flat = _suffix_counts(Counter(p.suffix.lower() for p in videos))
        return (
            f"This folder mixes 360 footage ({_plural(len(clips), 'Insta360 clip')}) and regular "
            f"videos ({flat}). Put them in separate folders and make one project for each."
        )
    if not _usable(clips):
        parts = Counter(k for g in clips.values() for k in ("lens00", "lens10", "lrv") if k in g)
        label = {"lens00": "_00_.insv", "lens10": "_10_.insv", "lrv": ".lrv"}
        found = " and ".join(f"{parts[k]} {label[k]}" for k in label if parts[k])
        missing = " or ".join(
            label[k] for k in ("lens00", "lrv") if any(k not in g for g in clips.values())
        )
        nums = sorted(clips)
        named = ", ".join(nums[:5]) + (f" and {len(nums) - 5} more" if len(nums) > 5 else "")
        return (
            f"Found {found} but no matching {missing} (clip{'s' if len(nums) > 1 else ''} "
            f"{named}): 360 analysis needs both files of each clip."
        )
    return None


def new_project(name: str, source: str, style: str, **extra: Any) -> dict[str, str]:
    """Create the project folder and its ``ui/meta.json`` (``extra``: e.g. ``kind`` and
    ``clips`` of a 360 project)."""
    root = projects_dir()
    with cache.created_dirs() as made:  # a failed request leaves no empty folder behind
        cache.make_dirs(root, made)
        base, n = slugify(name, 60, "project"), 1
        while True:
            pid = base if n == 1 else f"{base}-{n}"
            try:
                (root / pid).mkdir()
                break
            except FileExistsError:
                n += 1
        made.insert(0, root / pid)
        (root / pid / UI_DIR).mkdir()
        made.insert(0, root / pid / UI_DIR)
        cache.save_json(
            root / pid / UI_DIR / "meta.json",
            {
                "name": name,
                "source": source,
                "style": style,
                "created": datetime.now().isoformat(),
                **extra,
            },
        )
    return {"id": pid, "name": name}


def ui_styles() -> list[str]:
    """Styles for standard video (telemetry-driven action-cam styles are command-line only).

    A learned style that fails to load is left out with a warning, so one bad file does not
    break the UI.
    """
    names = []
    for n in style_names():
        try:
            style = get_style(n)
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("skipping learned style %s: %s", learned_styles().get(n, n), e)
            continue
        if style.kind != "ski":
            names.append(n)
    return names


def _ski_style(name: str) -> bool:
    """``name`` is a telemetry-driven action-cam style (``kind`` ski), accepted for 360 sources."""
    try:
        return get_style(name).kind == "ski"
    except (ValueError, TypeError, AttributeError):
        return False


# --------------------------------------------------------------------------- app
def _flag(body: dict[str, Any], key: str) -> bool:
    v = body.get(key, False)
    if not isinstance(v, bool):
        raise bad(f"{key} must be true or false")
    return v


def _compress_peaks(body: dict[str, Any]) -> list[str]:
    """``--compress-peaks`` with the page's choice, else ``compress_peaks`` in config.json;
    explicit, so the job does what the page showed even if config.json changes before it runs."""
    v = body.get("compress_peaks", get_config().compress_peaks)
    if v not in COMPRESS_PEAKS:
        raise bad(f"compress_peaks must be one of {', '.join(COMPRESS_PEAKS)}")
    return ["--compress-peaks", v]


def create_app(port: int) -> FastAPI:
    token = secrets.token_urlsafe(32)
    jobs, media, picker, sources = Jobs(), Media(), Picker(), Sources()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        asyncio.get_running_loop().set_exception_handler(quiet_resets)
        yield
        jobs.shutdown()
        media.shutdown()
        picker.shutdown()

    app = FastAPI(
        title="Cutpoint", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )

    app.state.token, app.state.jobs, app.state.media = token, jobs, media
    app.state.picker, app.state.sources = picker, sources

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
        return secrets.compare_digest((got or "").encode(), token.encode())

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
        return f"/api/projects/{quote(pid, safe='')}/{kind}/{quote(name, safe='')}?t={token}"

    def window_url(pid: str, sid: str, lo: int, hi: int) -> str:
        return url(pid, "sources", sid).replace("?", f"/window/{lo}-{hi}?", 1)

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
        """Change the projects folder and/or the vision default in ``config.json``. Existing
        projects stay where they are."""
        if "projects_dir" not in body and "vision" not in body:
            raise bad("send projects_dir and/or vision")
        updates: dict[str, Any] = {}
        if "vision" in body:
            updates["vision"] = _flag(body, "vision")
        moving = "projects_dir" in body
        if moving and paths.env("PROJECTS_DIR") is not None:
            raise bad(
                "CUTPOINT_PROJECTS_DIR sets the projects folder; change or unset it there", 409
            )
        busy = "wait for the running job to finish before changing the projects folder"
        if moving and any(j.status == "running" for j in list(jobs.jobs.values())):
            raise bad(busy, 409)
        data = config_data()  # an unreadable config.json: refuse before creating the folder
        folder, created = None, False
        if moving:
            folder, created = check_projects_dir(body.get("projects_dir"))
            updates["projects_dir"] = str(folder)
        # The jobs lock makes the running-job check and the write atomic against Jobs.start,
        # so no job starts against the old folder while the setting changes.
        try:
            with jobs.lock:
                if moving and any(j.status == "running" for j in jobs.jobs.values()):
                    raise bad(busy, 409)
                set_config_values(updates)
        except (ValueError, OSError, HTTPException) as err:
            if created and folder is not None:
                with contextlib.suppress(OSError):
                    folder.rmdir()
            if isinstance(err, HTTPException):
                raise
            if isinstance(err, ValueError):  # config.json is not valid JSON: never overwrite it
                raise bad(str(err), 409) from err
            raise bad(f"could not save the settings: {err.strerror or err}", 500) from err
        return settings_view({**data, **updates})

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
            found.append({"id": d.name, "name": read_meta(d).get("name") or d.name})
        return {"projects": found}

    @owned.post("/projects")
    def create_project(body: dict[str, Any]):
        name, source, style = body.get("name"), body.get("source"), body.get("style", "gaming")
        if not isinstance(name, str) or not name.strip() or len(name) > 100:
            raise bad("name must be 1-100 characters")
        if not isinstance(source, str) or not Path(source).is_absolute():
            raise bad("source must be an absolute path")
        ski = isinstance(style, str) and _ski_style(style)
        if style not in ui_styles() and not ski:
            raise bad(f"unknown style {style!r}")
        src = Path(source)
        try:
            problem = source_problem(src)
            usable = None if problem else _usable(inspect_source(src)[1])
        except OSError as err:
            raise bad(f"source is not readable: {err}") from err
        if problem:
            raise bad(problem)
        if not usable:
            if ski:  # action-cam styles need telemetry: 360 footage only in the UI
                raise bad(f"unknown style {style!r}")
            return new_project(name.strip(), os.path.abspath(source), style)  # noqa: PTH100
        # 360: the ski styles as given, any other the 360 default (pano.default_style)
        try:
            style = style if ski else pano_style()
        except ValueError as err:  # config.json's pano section is not an object
            raise bad(str(err), 409) from err
        clips = {"clips": usable} if src.is_file() else {}  # one file: just that clip
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
        return {
            "id": pid,
            "name": meta.get("name") or pid,
            "style": meta.get("style") or (hl or {}).get("style") or "",
            "kind": "360" if meta.get("kind") == "360" else "video",
            "source": meta.get("source"),
            "highlights": moments_view(hl),
            "review": review if isinstance(review, dict) else None,
            "clips": {**manifest, "dir": exp["dir"]}
            if isinstance(manifest, dict) and exp
            else None,
            "outputs": [{"name": n, "url": url(pid, "outputs", n)} for n in outputs],
            # The first listed final is the current export's own (or, with no export, the
            # newest); false while that export is not assembled yet and an earlier final leads.
            "final_current": exp is None or (bool(outputs) and outputs[0] == exp.get("final")),
            "sources": found,
            # The picked folder could not be listed (gone, unreadable): a field, so Settings
            # (source remap), Reveal and Delete still work.
            "source_error": source_error,
        }

    @owned.put("/projects/{pid}/review")
    def put_review(pid: str, body: dict[str, Any]):
        p = project_dir(pid)
        segments = validate_segments(body.get("segments"), sources.entries(p), media)
        review = {"segments": segments, "updated": datetime.now().isoformat(timespec="seconds")}
        file = _own(p, f"{UI_DIR}/review.json")
        _ui_folder(p)  # a command-line project has no ui folder yet
        cache.save_json(file, review)
        return review

    @owned.post("/projects/{pid}/reveal")
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

    @owned.post("/projects/{pid}/jobs")
    def start_job(pid: str, body: dict[str, Any]):
        p = project_dir(pid)
        action = body.get("action")
        meta = read_meta(p)
        hl = _highlights(p)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        on_success: Callable[[], None] | None = None
        prepare: Callable[[], None] | None = None
        written: list[Path] = []  # files ``prepare`` wrote, removed if the job does not start
        plan: list[dict[str, str]] | None = None
        if action == "analyze" and meta.get("kind") == "360":
            # The CLI's 360 flow: scan, views, analysis of the .lrv, events (no vision rating)
            style = meta.get("style")
            if not meta.get("source") or not style:
                raise bad("project has no recorded source to analyze")
            _own(p, "highlights.json")  # the analysis rewrites it
            _own(p, "pano.json")  # and the scan this
            args = cutpoint_cmd(
                "360", "analyze", str(p), "--scan", str(_pano_folder(meta["source"]))
            )
            args += ["--style", str(style)]
            if only := [c for c in meta.get("clips") or [] if isinstance(c, str)]:
                args += ["--only", *only]
            plan = [{"step": s, "label": progress.LABELS[s]} for s in progress.PANO_STEPS]
        elif action == "analyze":
            inputs = (
                [meta["source"]]
                if meta.get("source")
                else (hl or {}).get("settings", {}).get("inputs")
            )
            style = meta.get("style") or (hl or {}).get("style")
            if not inputs or not style:
                raise bad("project has no recorded source to analyze")
            _own(p, "highlights.json")  # the analysis rewrites it
            args = cutpoint_cmd(
                "analyze", *map(str, inputs), "--style", str(style), "--out", str(p)
            )
            # Keep what the earlier analysis recorded, or the source set would change.
            st = (hl or {}).get("settings") or {}
            if isinstance(st.get("exclude"), list):
                args += [f"--exclude={g}" for g in st["exclude"] if isinstance(g, str)]
            if st.get("recursive") is True:
                args.append("--recursive")
            # The Settings value now, first analysis or again; explicit, so the job and the
            # page agree even if config.json changes before the CLI reads it.
            args.append("--vision" if get_config().vision else "--no-vision")
            vision_top = st.get("vision_top")
            if isinstance(vision_top, int) and not isinstance(vision_top, bool) and vision_top > 0:
                args.append(f"--vision-top={vision_top}")
            for key, opt in (("ollama_url", "--ollama-url"), ("ollama_model", "--ollama-model")):
                if isinstance(st.get(key), str) and st[key]:
                    args.append(f"{opt}={st[key]}")
        elif action == "clips":
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
            vertical = _flag(body, "vertical")
            peaks = _compress_peaks(body)
            out_name = fresh_name(p, f"clips_ui_{stamp}")
            seg_file = _own(p, f"{UI_DIR}/segments-{out_name[len('clips_ui_') :]}.json")
            _own(p, f"{UI_DIR}/exports.json")  # refuse now, not after the export

            def prepare() -> None:
                _ui_folder(p)  # a command-line project has no ui folder yet
                # Exactly the trims the user made, in their order: no smart tails or speech snapping.
                cache.save_json(
                    seg_file, [{**s, "smart_tail": False, "speech_aware": False} for s in segments]
                )
                written.append(seg_file)

            args = cutpoint_cmd(
                "clips", str(p), "--segments", str(seg_file), "--out", str(p / out_name)
            )
            if vertical:
                args.append("--vertical")
            if extend:
                args += ["--extend", f"{float(extend):g}"]
            args += peaks

            def on_success() -> None:
                if not (p / out_name / "manifest.json").is_file():
                    raise RuntimeError("clips finished without a manifest")
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
            exp_dir = exp["dir"]
            stem = fresh_name(
                p,
                "final" + exp_dir[5:] if exp_dir.startswith("clips") else f"final_{exp_dir}",
                FINAL_EXTS,
            )
            args = cutpoint_cmd(
                "assemble",
                str(p / exp_dir),
                "--out",
                str(p / f"{stem}.mp4"),
                "--transition",
                transition,
                *_compress_peaks(body),
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
                cache.save_json(_own(p, f"{UI_DIR}/exports.json"), {"exports": items})

        else:
            raise bad("action must be 'analyze', 'clips' or 'assemble'")

        def same_folder() -> None:
            # Under the jobs lock, like the settings change: a folder changed since ``p`` was
            # resolved would run the job against the old one.
            if projects_dir().resolve() != p.resolve().parent:
                raise bad("the projects folder changed; reload the page", 409)
            if prepare:
                prepare()  # only once the job may start: a refused one leaves no file behind

        try:
            return jobs.start(pid, action, args, on_success, same_folder, plan).view()
        except HTTPException:  # for example the command could not be started
            for f in written:
                with contextlib.suppress(OSError):
                    f.unlink(missing_ok=True)
            raise

    @secure.get("/jobs")
    def list_jobs():
        return {"jobs": [j.view() for j in list(jobs.jobs.values())]}

    @secure.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        return jobs.cancel(job_id).view()

    def send(
        file: Path, request: Request, name: str | None = None, *, proxy: bool = False
    ) -> FileResponse:
        """Range-capable file response; ``?download=1`` makes the browser save it.

        ``proxy``: a proxy, marked used (see :func:`_mark_used`), with an ETag from its name
        (which holds its key) and size, so touching it does not turn a seek's ranged request
        (If-Range) into a whole response."""
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
        )

    def find_source(pid: str, sid: str) -> tuple[Path, dict[str, Any]]:
        p = project_dir(pid)
        entry = next((s for s in sources.entries(p) if s["id"] == sid), None)
        if entry is None:
            raise bad("unknown source", 404)
        return p, entry

    @served.api_route("/projects/{pid}/sources/{sid}", methods=["GET", "HEAD"])
    def source_media(pid: str, sid: str, request: Request):
        p, entry = find_source(pid, sid)
        status, file = media.status(p, entry["path"])
        if file is None:
            raise bad(f"source is {status}", 404 if status == "failed" else 409)
        # a proxy is marked used; never the source itself
        return send(file, request, entry["name"], proxy=file.parent == p / UI_DIR / "proxy")

    @served.api_route("/projects/{pid}/sources/{sid}/window/{span}", methods=["GET", "HEAD"])
    def source_window(pid: str, sid: str, span: str, request: Request):
        """A ready window proxy of a source; ``span`` is ``<lo>-<hi>`` in centiseconds."""
        m = re.fullmatch(r"(\d{1,9})-(\d{1,9})", span)
        if m is None:
            raise bad("unknown window", 404)
        p, entry = find_source(pid, sid)
        file = media.window(p, entry["path"], int(m[1]), int(m[2]))
        if file is None:
            status = media.status(p, entry["path"])[0]
            raise bad(
                f"window is not ready (source is {status})", 404 if status == "failed" else 409
            )
        return send(file, request, proxy=True)

    @served.get("/projects/{pid}/sources/{sid}/frame")
    def source_frame(pid: str, sid: str, at: float = 0.0, w: int = FRAME_WIDTH):
        """One still of a source (``at`` seconds in, ``w`` px wide) for thumbnails."""
        if not math.isfinite(at) or at < 0:
            raise bad("at must be a time in seconds, 0 or more")
        if media.closing:
            raise bad("the server is stopping", 503)
        p, entry = find_source(pid, sid)
        if p not in media.frames_pruned:
            media.prune_frames(p, [s["path"] for s in sources.entries(p)])
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

    @served.api_route("/projects/{pid}/outputs/{name}", methods=["GET", "HEAD"])
    def output_file(pid: str, name: str, request: Request):
        p = project_dir(pid)
        if name not in output_names(p):
            raise bad("unknown output", 404)
        return send(p / name, request)

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
