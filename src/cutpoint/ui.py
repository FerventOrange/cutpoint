"""``cutpoint ui``: local browser workspace for the standard-video highlight workflow.

A loopback-only FastAPI app that serves ``cutpoint/web`` and a small JSON
API over the projects folder. Heavy work (analyze, clips, extend, assemble, resume) runs as
ordinary ``python -m cutpoint ... --json`` child processes in the projects folder, one at a time,
so the browser and the CLI share one implementation. Source videos are only ever read.

Trust boundaries:

* Every request needs a loopback ``Host`` (and ``Origin``, when sent) on the bound port.
* Every ``/api`` call needs the per-run token (``X-Cutpoint-Token``), which only the URL printed at
  startup carries (``#t=<token>``; there is no way to fetch it from the server). Media/output URLs
  carry it as ``?t=`` because ``<video>`` cannot send headers.
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
  made on the command line, the inputs of its analysis); ``highlights.json`` alone never
  grants access to a file.
* Resume (``GET .../resume``, the ``resume`` job) replays the command line the project's
  ``progress.json`` recorded, from its recorded working folder (``cutpoint resume``): a
  recorded ``--force`` or an ``--out`` outside the project too. The page shows that command
  line in its confirm dialog first; the server never changes it, and a clips folder it wrote
  outside the project is never listed as an export.
* "Show folder" (``.../reveal?target=``) opens, in the file manager and without a shell, one of
  three folders the server resolves from the project: its own sources (the default), the clips
  folder of its latest export, or the project folder holding its assembled final. The request
  names only the target, never a path.
* Choosing a source or folder (``POST /api/pick?kind=file|folder``) opens the machine's own open
  dialog (Windows: Tk's system dialog; macOS: ``osascript``; Linux: zenity, else kdialog) in a
  child process without a shell, one at a time; the server never lists folders itself. With no
  dialog (no display, Docker) the page hides those buttons and the path is typed or pasted.
* Settings (``GET``/``PUT /api/settings``) change only ``projects_dir``, ``vision`` and the
  assistant's ``provider``, ``model`` and ``url`` in ``config.json``. The projects folder never changes while a job runs or when
  ``CUTPOINT_PROJECTS_DIR`` sets it; existing projects are not moved. Every analysis the page
  starts (first or again) passes ``--vision`` or ``--no-vision`` as ``vision`` says when it
  starts; the page has no per-run vision box.
  While ``config.json`` is unreadable or holds a value Cutpoint rejects, the project routes answer
  409 naming the file and the error; session, pick and settings keep working.
* The assistant (``/api/projects/<id>/assistant/...``) takes the token header only, never
  ``?t=``. It only proposes sequence changes, which the page applies to its working sequence; it
  writes only ``ui/assistant/chat.json``, ``ui/assistant/history.json`` and the cached timeline
  sheets in ``ui/assistant/sheets``, never ``review.json``, and runs no commands (frame grabs
  aside). Project text (labels, transcripts, file names, never
  paths) reaches the model only inside marked data blocks. Its GET routes only read.
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
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cutpoint import (
    cache,
    folders,
    insta360,
    jsonout,
    paths,
    procs,
    progress,
    projectdelete,
    runstate,
)
from cutpoint import envelope as audio_envelope
from cutpoint.assemble import (
    AUDIO_TRANSITIONS,
    JOIN_TRANSITIONS,
    OUTPUT_FORMATS,
    default_audio_transition,
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
from cutpoint.backends import ollama_url
from cutpoint.clips import MIN_EXPORT_S
from cutpoint.config import (
    ASSISTANT_PROVIDERS,
    COLOR_MODES,
    COMPRESS_PEAKS,
    VISION_PROVIDERS,
    AssistantSettings,
    analysis_skip,
    assistant_settings,
    get_config,
    load_config_data,
    projects_dir,
    remap_source,
    resolve_projects_dir,
    set_config_values,
    vision_assistant_settings,
)
from cutpoint.edit import KINDS
from cutpoint.encode import BT709_TAGS, TV_RANGE, preview_filter, still_filter
from cutpoint.errors import EncoderUnavailableError, FFmpegError, FolderBusyError
from cutpoint.events import scoring
from cutpoint.ffmpeg_utils import (
    ChildScope,
    VideoEncoder,
    audio_mix_filter,
    child_scope,
    grab_images,
    probe,
    video_encoder,
)
from cutpoint.pano import project as pano_project
from cutpoint.pano import stitch as pano_stitch
from cutpoint.pano import x3
from cutpoint.pano.project import default_style as pano_style
from cutpoint.sources import TRANSCRIPT_MODELS
from cutpoint.styles import (
    ACTION_CAM,
    AUTO,
    Style,
    get_style,
    learned_styles,
    renamed_error,
    style_names,
)
from cutpoint.utils import (
    PANO_EXTS,
    VIDEO_EXTS,
    clip_files,
    file_stamp,
    find_videos,
    matches_glob,
    safe_is_file,
    slugify,
)
from cutpoint.vision import default_model

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
UI_DIR = "ui"  # per-project state: meta.json, review.json, exports.json, proxy/, frames/, pano/
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
MAX_SEGMENTS = ai_ops.MAX_SEGMENTS
MAX_LOG_LINES = 300
MAX_JOBS = 50
MAX_KEPT_NAMES = 100  # names of kept files the delete plan lists
MAX_EXTEND_S = 30.0
MAX_TIMING_S = 60.0  # speech caps, tail lengths and the event window the page may send
MAX_TOTAL_S = 36000.0  # clips --max-total
CLIP_ORDERS = ("story", "score", "chronological")  # clips --order without --mix
EVENT_TAILS = {"auto": [], "on": ["--event-tails"], "off": ["--no-event-tails"]}
MAX_FADE_S = 10.0  # clips / assemble --audio-fade and assemble --fade-dur the page may send
CROP_BOX = re.compile(r"[1-9]\d{0,4}:[1-9]\d{0,4}:\d{1,5}:\d{1,5}")  # clips --crop W:H:X:Y
EVEN_PX = r"(?:[1-9]\d{0,3}[02468]|[2468])"  # H.264 needs even sizes
CLIP_SIZE = re.compile(rf"{EVEN_PX}x{EVEN_PX}")  # clips --size WxH
REVERT = "revert-"  # proposal id prefix: put back the "before" of an applied assistant change
BROWSER_CONTAINERS = {".mp4", ".m4v", ".mov", ".webm"}
BROWSER_VIDEO = {"h264", "vp8", "vp9", "av1"}
BROWSER_AUDIO = {None, "aac", "mp3", "opus", "vorbis"}
SAFE_NAME = re.compile(r"\w[\w.-]*")  # a name inside the project: no ".", ".." or ".x"
# What ``assemble`` writes next to ``<stem>.mp4`` (.ass: the --captions subtitles it burns in).
FINAL_EXTS = (".mp4", ".fcpxml", ".otio", ".srt", ".ass")
# A final in the project folder: ``assemble``'s final.mp4 / final_<x>.mp4 (UI exports included)
# and ``edit``'s <name>-final.mp4 / <name>-final_vertical.mp4, or the .fcpxml / .otio timeline
# that ``--format`` writes instead. Partial and staging files, ``assemble <project>``'s
# .assemble-tmp-<id> folders and their lock files start with "." and never match.
FINAL_FILE = re.compile(r"(?:[^.].*-)?final(?:_.*)?\.(?:mp4|fcpxml|otio)")
# A step update the CLI writes when CUTPOINT_PROGRESS=1 (``spawn`` sets it), bare or after a
# ``HH:MM:SS LEVEL`` log prefix: ``PROGRESS {"step": ..., "index": ..., "state": ...}``.
PROGRESS_LINE = re.compile(r"(?:\d{1,2}:\d{2}:\d{2}\s+[A-Z]+\s+)?PROGRESS\s+(\{.*\})")
STEP_STATES = {"running", "done", "skipped", "failed"}
FRAME_WIDTH = 160  # default still width (px) for sequence thumbnails
FRAME_MIN_W, FRAME_MAX_W = 32, 640
FRAME_TIMEOUT_S = 30.0
FRAME_MAX_AT = 1e7  # seconds (~115 days): cap when the length is unknown
FRAME_SLOTS = 2  # ffmpeg still grabs at once; the rest wait their turn
FRAME_CACHE_MAX = 1500  # stills kept per project; the least recently used go first
ENVELOPE_VERSION = 1  # part of a cached envelope's name: bump it when the envelope changes
ENVELOPE_TIMEOUT_S = 1800.0  # a decode running longer is killed (a multi-hour source on a NAS)
ENVELOPE_STALE_S = (
    10.0  # a queued decode nobody asked for in this long (2 x the page's 5 s) is dropped
)
EVENT_KINDS = ("audio", "ocr", "telemetry")  # events.json kinds the rail marks (not the VLM blobs)
EVENTS_MAX = 5000  # events sent per project
SOURCES_CACHE_MAX = 32  # projects whose source lists are kept; the least recently used go first
SOURCES_CACHE_S = 60.0  # rebuild a kept source list at least this often (what stats miss)
_suggest_lock = threading.Lock()  # one style suggestion probes at a time; another gets no answer
TELEMETRY_PROBE_TIMEOUT = 10  # seconds per ffprobe of that suggestion
MAX_TELEMETRY_PROBES = 12  # files one style suggestion probes, so a big folder or a NAS stays quick
MAX_INPUTS = 50  # files and folders one New project takes (analyze's positional inputs)
MAX_EXCLUDES = 20  # analyze --exclude patterns one New project takes
MAX_GLOB = 200  # characters of one --exclude pattern
# What the page says when a job stops with folder_busy (#109): another run holds the project's run
# lock or the clips folder's; never the CLI's message, which tells the user to delete the lock.
BUSY_TEXT = "Analysis is running on this project; wait for it to finish"
RESUME_CHECK_S = 60.0  # seconds ``cutpoint resume --dry-run`` may take (it starts the CLI)


def bad(detail: str, status: int = 400) -> HTTPException:
    return HTTPException(status, detail)


# --------------------------------------------------------------------------- processes
def job_env() -> dict[str, str]:
    """The environment of a child ``cutpoint`` command (``spawn``, ``resume_check``)."""
    # CUTPOINT_PROGRESS: analyze and edit write PROGRESS step lines (``parse_progress``); the
    # other commands and ffmpeg ignore it.
    # Never the assistant's API key variables (ai_keys.child_env): no job uses them.
    return {
        **ai_keys.child_env(),
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "CUTPOINT_PROGRESS": "1",
        # ``python -m cutpoint`` runs in the projects folder: a stray json.py there must not
        # shadow a module (no current folder on sys.path)
        "PYTHONSAFEPATH": "1",
    }


def spawn(
    args: list[str], *, low_priority: bool = True, cwd: Path | None = None, binary: bool = False
) -> subprocess.Popen[Any]:
    """Start ``args`` (no shell) in its own process group at low priority, stdout+stderr merged.

    Low priority keeps the desktop and the browser responsive while a job or proxy encodes; the
    children it starts (ffmpeg, workers) inherit it. Windows: BELOW_NORMAL_PRIORITY_CLASS;
    elsewhere ``nice -n 10`` when there is one (it execs, so the pid and group stay the same).
    ``low_priority=False``: normal priority, for the short window proxy encodes the user waits on.
    ``cwd``: the folder it runs in (jobs: the projects folder), else the server's own.
    ``binary``: stdout is raw bytes and stderr is dropped, not merged into it.
    """
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        if low_priority:
            flags |= subprocess.BELOW_NORMAL_PRIORITY_CLASS
        group: dict[str, Any] = {"creationflags": flags}
    else:
        group = {"start_new_session": True}
        args = procs.nice_prefix(args) if low_priority else args
    text: dict[str, Any] = (
        {"stderr": subprocess.DEVNULL}
        if binary
        else {"stderr": subprocess.STDOUT, "text": True, "encoding": "utf-8", "errors": "replace"}
    )
    proc = subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        **text,
        env=job_env(),
        cwd=cwd,
        **group,
    )
    if os.name == "nt":
        procs.join_tree_job(proc)
    return proc


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
            env=ai_keys.child_env(),
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
                    env=ai_keys.child_env(),
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


def _vision_provider_from_data(data: dict[str, Any]) -> ai_providers.Provider | None:
    section = data.get("vision_assistant")
    if not isinstance(section, dict):
        section = {"provider": "ollama"}
    provider_name = (section.get("provider") or "ollama").strip().lower()
    if provider_name not in VISION_PROVIDERS:
        provider_name = "ollama"
    model = (section.get("model") or "").strip()
    url = (section.get("url") or "").strip()
    chat_name, chat_url = _provider_url(data, "assistant")
    try:
        return ai_providers.make_vision(
            AssistantSettings(provider=provider_name, model=model, url=url),
            AssistantSettings(provider=chat_name or "ollama", url=chat_url),
        )
    except (ValueError, TypeError, KeyError):  # an invalid value in config.json: no provider
        return None


def _tuple_of_str(val: Any) -> tuple[str, ...]:
    """A tuple of string items when ``val`` is a list, else empty tuple."""
    if not isinstance(val, list):
        return ()
    return tuple(x for x in val if isinstance(x, str))


def settings_view(data: dict[str, Any]) -> dict[str, Any]:
    """The settings for config ``data``: the projects folder, the vision default and the export
    defaults of Compress peaks and Color, so other invalid keys in ``config.json`` do not hide
    them; a wrongly typed ``projects_dir`` is shown as is, to replace, a ``vision`` that is not
    ``true`` as off and an unknown ``compress_peaks`` or ``color`` as off."""
    peaks = data.get("compress_peaks")
    color = data.get("color")
    try:
        folder = str(resolve_projects_dir(data))
    except TypeError:
        folder = str(data.get("projects_dir"))
    vis_section = data.get("vision_assistant")
    vis_consent = _tuple_of_str(vis_section.get("consent")) if isinstance(vis_section, dict) else ()
    vp = _vision_provider_from_data(data)
    # an Ollama on this machine needs no consent; one on another machine does, as for AI Chat
    if vp is not None and (vp.name != "ollama" or not vp.local):
        v_consent_info = consent_view(vp, consent=vis_consent, role="vision")
    else:
        v_consent_info = {
            "needed": False,
            "key": None,
            "host": getattr(vp, "host", "") if vp is not None else "",
            "missing_key": None,
            "missing_kind": None,
        }
    return {
        "projects_dir": folder,
        "projects_dir_locked": paths.env("PROJECTS_DIR") is not None,
        "config_file": str(paths.config_file()),
        "vision": data.get("vision") is True,
        "vision_model": default_model(),
        "compress_peaks": peaks if peaks in COMPRESS_PEAKS else "off",
        # whether config.json sets it: the page's Compress peaks then starts from it, else Light
        "compress_peaks_set": peaks in COMPRESS_PEAKS,
        # the page's Color starts from it (Off when unset), never with a grade on gaming footage
        "color": color if isinstance(color, str) and color in COLOR_MODES else "off",
        "assistant": assistant_view(data, section_name="assistant"),
        "vision_assistant": assistant_view(data, section_name="vision_assistant"),
        "vision_consent": v_consent_info,
    }


MAX_ASSISTANT_TEXT = 500  # characters of the assistant's model or base URL
LOGIN_TIMEOUT_S = 300.0  # seconds before a pending login expires


def subscription_presets() -> dict[str, str]:
    """The picker's providers: the keyed and local presets, then each subscription login a
    provider is built for (Antigravity only once :data:`~cutpoint.assistant.providers.PROVIDERS`
    has it)."""
    subscriptions = {p: cliproxy.TITLES[p] for p in cliproxy.PROVIDERS}
    return {**ai_providers.PRESETS, **subscriptions}


def assistant_view(data: dict[str, Any], section_name: str = "assistant") -> dict[str, Any]:
    """The assistant or vision_assistant settings of config ``data`` for Settings, never a key:
    the provider, model and base URL as saved (an invalid value shown as the default), whether that
    URL is custom (not the preset's, :func:`cutpoint.assistant.providers.is_preset`), what each
    preset offers, and where each provider's key comes from (``"env"``, ``"keyring"`` or None)."""
    section = data.get(section_name)
    if not isinstance(section, dict):
        section = {}
    name = section.get("provider")
    name = name.strip().lower() if isinstance(name, str) else ""

    def text(key: str) -> str:
        value = section.get(key)
        return value.strip() if isinstance(value, str) else ""

    try:
        ollama = ai_providers.public(ollama_url())  # OLLAMA_HOST may carry user:password@
    except (ValueError, TypeError):  # a bad port or config.json: Settings still opens
        ollama = ""
    # provider -> (default URL, default model, models offered, Model placeholder, model needed)
    is_vision = section_name == "vision_assistant"
    dm = default_model() if is_vision else ai_providers.DEFAULT_OLLAMA_MODEL
    offers = {
        "ollama": (ollama, dm if is_vision else ai_providers.AUTO_MODEL, [dm], dm, False),
        "openai": (
            ai_providers.OPENAI_URL,
            "",
            list(ai_providers.OPENAI_MODELS),
            "OpenAI model name",
            True,
        ),
        "anthropic": (
            ai_providers.ANTHROPIC_URL,
            ai_providers.DEFAULT_ANTHROPIC_MODEL,
            list(ai_providers.ANTHROPIC_MODELS),
            ai_providers.DEFAULT_ANTHROPIC_MODEL,
            False,
        ),
        "xai": (ai_providers.XAI_URL, "", list(ai_providers.XAI_MODELS), "xAI model name", True),
        "claude-code": (
            "",
            ai_providers.DEFAULT_ANTHROPIC_MODEL,
            list(ai_providers.ANTHROPIC_MODELS),
            ai_providers.DEFAULT_ANTHROPIC_MODEL,
            False,
        ),
        "codex": (
            "",
            ai_providers.DEFAULT_CODEX_MODEL,
            [ai_providers.DEFAULT_CODEX_MODEL],
            ai_providers.DEFAULT_CODEX_MODEL,
            False,
        ),
        "antigravity": ("", "", [], "model name", False),
    }
    hosts = key_hosts(data, section_name=section_name)

    def key_env(p: str) -> list[str]:
        if p not in ai_keys.KEYED:
            return []
        names = ai_keys.env_names(p, hosts[p])
        if is_vision and not _vision_key_shared(data, p):  # never read for this host
            names = [n for n in names if n != ai_keys.ENV_KEY]
        return names

    allowed = VISION_PROVIDERS if is_vision else ASSISTANT_PROVIDERS
    name = name if name in allowed else "ollama"
    all_presets = subscription_presets()
    presets_offered = {p: title for p, title in all_presets.items() if p in allowed}
    try:
        local = ai_providers.make(AssistantSettings(provider=name, url=text("url"))).local
    except (ValueError, TypeError):
        local = name == "ollama"
    return {
        "provider": name,
        "local": local,
        "model": text("model"),
        "url": ai_providers.public(text("url")),  # a hand-edited one too; Settings refuses it
        # a base URL no preset has: shown as "Custom (from config.json)", kept as it is
        "custom": not ai_providers.is_preset(name, text("url")),
        "providers": [
            {
                "name": p,
                "title": title,
                "default_url": offers[p][0],
                "default_model": offers[p][1],
                "models": offers[p][2],
                "placeholder": offers[p][3],
                "model_required": offers[p][4],
                "key_env": key_env(p),
                "subscription": p in cliproxy.PROVIDERS,  # logs in (Settings > Subscriptions)
            }
            for p, title in presets_offered.items()
        ],
        **key_status(data, section_name=section_name),
    }


def _provider_url(data: dict[str, Any], section_name: str) -> tuple[str, str]:
    """``(provider, url)`` of config ``data``'s ``section_name`` section ("" when unset)."""
    section = data.get(section_name)
    section = section if isinstance(section, dict) else {}
    name, url = section.get("provider"), section.get("url")
    name = name.strip().lower() if isinstance(name, str) else ""
    url = url.strip() if isinstance(url, str) else ""
    return name, url


def key_hosts(data: dict[str, Any], section_name: str = "assistant") -> dict[str, str]:
    """The host each keyed provider's key is for, by config ``data``: the configured base URL's
    for the configured provider, the official API host for the others."""
    name, url = _provider_url(data, section_name)
    return {p: ai_providers.key_host(p, url if p == name else "") for p in ai_keys.KEYED}


def _vision_key_shared(data: dict[str, Any], p: str) -> bool:
    """Whether the Vision role's provider ``p`` may use ``CUTPOINT_ASSISTANT_API_KEY`` by config
    ``data`` (:func:`cutpoint.assistant.providers.vision_key_shared`)."""
    name, url = _provider_url(data, "vision_assistant")
    chat_name, chat_url = _provider_url(data, "assistant")
    return ai_providers.vision_key_shared(p, url if p == name else "", chat_name, chat_url)


def key_status(data: dict[str, Any], section_name: str = "assistant") -> dict[str, Any]:
    """``{keys, key_hosts}``: where each keyed provider's key for its host comes from
    (``"env"``, ``"keyring"`` or None), and that host. Never a key. The Vision role's keys come
    from ``CUTPOINT_ASSISTANT_API_KEY`` only for the assistant's key host
    (:func:`cutpoint.assistant.providers.vision_key_shared`)."""
    hosts = key_hosts(data, section_name=section_name)
    vision = section_name == "vision_assistant"

    def shared(p: str) -> bool:
        return not vision or _vision_key_shared(data, p)

    def where(p: str) -> str | None:
        # UI jobs read the keyring first for the Vision role, even when a variable is set
        if vision and ai_keys.in_keyring(p, hosts[p]):
            return "keyring"
        return ai_keys.status(p, hosts[p], shared=shared(p))

    return {
        "keys": {p: where(p) for p in ai_keys.KEYED},
        "key_hosts": hosts,
    }


def check_assistant(
    value: Any, section_name: str = "assistant", existing_provider: str | None = None
) -> dict[str, str]:
    """The ``assistant`` or ``vision_assistant`` object of a Settings change: ``provider``,
    ``model`` and ``url`` (an http(s) base URL; "" for the default).

    Raises:
        HTTPException: 400 for anything else.
    """
    is_vision = section_name == "vision_assistant"
    label = "vision_assistant" if is_vision else "assistant"
    allowed = VISION_PROVIDERS if is_vision else ASSISTANT_PROVIDERS
    if not isinstance(value, dict) or not value or set(value) - {"provider", "model", "url"}:
        raise bad(f"{label} must be an object with provider, model and/or url")
    out: dict[str, str] = {}
    for key, raw in value.items():
        if not isinstance(raw, str) or len(raw) > MAX_ASSISTANT_TEXT:
            raise bad(f"{label}.{key} must be text of at most {MAX_ASSISTANT_TEXT} characters")
        if any(not c.isprintable() for c in raw):
            raise bad(f"{label}.{key} must be printable text")
        out[key] = raw.strip()
    if "provider" in out:
        out["provider"] = out["provider"].lower()
        if out["provider"] not in allowed:
            raise bad(f"{label}.provider must be one of {', '.join(allowed)}")
    url = out.get("url")
    provider = out.get("provider") or existing_provider or ""
    if url and (provider in cliproxy.PROVIDERS or ai_providers.is_subscription(provider)):
        raise bad(f"{label}.url cannot be customized for subscription providers")
    if url:
        scheme = url.split("://", 1)[0].lower() if "://" in url else "http"
        if scheme not in ("http", "https") or " " in url or not ai_providers.host_of(url):
            raise bad(f"{label}.url must be an http:// or https:// base URL")
        if "\\" in url:  # requests and urlsplit read different hosts from some of these
            raise bad(f"{label}.url must not have a backslash")
        if ai_providers.has_userinfo(url):  # https://api.openai.com@evil.example goes to evil
            raise bad(f"{label}.url must not contain a user name or password")
        if not ai_providers.unambiguous(url):  # the host a request goes to must be the one shown
            raise bad(f"{label}.url must not have an unclear host")
    return out


def consent_view(
    provider: ai_providers.Provider,
    consent: tuple[str, ...] | None = None,
    role: str = "assistant",
) -> dict[str, Any]:
    """Whether the user still has to agree before project data goes to ``provider`` for
    ``role`` (``"assistant"`` or ``"vision"``): ``{needed, key, host, missing_key,
    missing_kind}``; ``key`` (:func:`cutpoint.assistant.providers.consent_key`) is None for a
    local provider; ``missing_key``: how to add the API key (or log in) it needs, which comes
    before agreeing (for Vision, a key only in an environment variable is missing: UI jobs get
    none); ``missing_kind``: ``"login"`` for a subscription provider (no key: a proxy
    login), ``"key"`` otherwise, None when nothing is missing. ``consent`` defaults to the
    role's own list in ``config.json``: one role's consent never covers the other's."""
    key = ai_providers.consent_key(provider, role)
    if consent is None:
        try:
            consent = (
                vision_assistant_settings() if role == "vision" else assistant_settings()
            ).consent
        except (ValueError, TypeError):  # config.json is invalid: no consent is known
            consent = ()
    # a Vision job started here gets no key variables: only a keyring key counts for it
    missing = (
        ai_providers.missing_ui_job_key(provider)
        if role == "vision"
        else ai_providers.missing_key(provider)
    )
    kind = "login" if ai_providers.is_subscription(provider) else "key"
    return {
        "needed": key is not None and key not in consent,
        "key": key,
        "host": provider.host,
        "missing_key": missing,
        "missing_kind": None if missing is None else kind,
    }


def check_projects_dir(value: Any) -> tuple[Path, bool]:
    """A folder the user may make the projects folder: absolute, a folder (the last part is created
    if only it is missing) and writable; and whether it was created here.

    Raises:
        HTTPException: 400 naming what is wrong.
    """
    if not isinstance(value, str) or not value.strip():
        raise bad("projects_dir must be a folder path")
    folder = Path(_typed_path(value))  # pasted with Copy as path's quotes too
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


def parse_result(line: str) -> dict[str, Any] | None:
    """The ``--json`` envelope (:mod:`cutpoint.jsonout`) an output line carries, else None.

    The envelope is one JSON object on stdout; jobs merge stdout into their output."""
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        raw = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != jsonout.SCHEMA
        or not isinstance(raw.get("ok"), bool)
        or not isinstance(raw.get("outputs"), dict)
    ):
        return None
    return raw


class Job:
    """One child process and what it said."""

    def __init__(
        self,
        project_id: str,
        action: str,
        plan: list[dict[str, str]] | None = None,
        cwd: Path | None = None,
    ) -> None:
        self.id = secrets.token_hex(6)
        self.project_id = project_id
        self.action = action
        self.plan = plan  # the steps the command will report (``{step, label}``), if known
        self.cwd = cwd  # the folder its commands run in (the projects folder)
        self.result: dict[str, Any] | None = None  # its command's ``--json`` envelope
        self.error_code: str | None = None  # the envelope's ``error.code`` of a failed job
        self.status = "running"
        self.logs: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self.logged = 0  # lines ever added to ``logs`` (it drops the oldest past its size)
        # Latest update per step (``parse_progress``), keyed by step: the CLI's PROGRESS lines.
        self.steps: dict[str, dict[str, Any]] = {}
        self.steps_total: int | None = None
        self.error: str | None = None
        self.started = datetime.now().isoformat(timespec="seconds")
        self.proc: subprocess.Popen[str] | None = None
        self.cancelled = False
        self.done = threading.Event()

    def add_output(self, line: str) -> None:
        """One output line: a step update goes to the steps, the ``--json`` envelope to
        ``result``, any other text to the log."""
        step = parse_progress(line)
        if step is None and (result := parse_result(line)) is not None:
            self.result = result
            return
        if step is None:
            if line.strip():
                self.logs.append(line.rstrip())
                self.logged += 1
            return
        total = step.pop("total")
        if total is not None:
            self.steps_total = total
        self.steps[step["step"]] = step  # replaced, never changed in place: ``view`` copies it

    def note(self, level: str, message: str) -> None:
        """A log line of the server's own, in the CLI's ``HH:MM:SS LEVEL message`` form."""
        self.logs.append(f"{datetime.now():%H:%M:%S} {level} {message}")
        self.logged += 1

    def drop_lines(self, text: str) -> None:
        """Take the log lines holding ``text`` out of the log."""
        self.logs = deque((ln for ln in self.logs if text not in ln), maxlen=MAX_LOG_LINES)

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
        if self.error_code:
            out["error_code"] = self.error_code
        # clips, extend or a resumed export (#109): finished clips of a stopped export it kept
        taken = (self.result or {}).get("outputs", {}).get("taken_over")
        if isinstance(taken, int) and not isinstance(taken, bool) and taken > 0:
            out["taken_over"] = taken
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
        before: Callable[[Job], list[str] | None] | None = None,
        *,
        cwd: Path | None = None,
        on_result: Callable[[Job], None] | None = None,
        on_end: Callable[[], None] | None = None,
    ) -> Job:
        """Start ``args`` as a job; ``check`` runs under the lock first and may refuse it.
        ``plan``: the steps it will report, shown as waiting before their first update.
        ``before``: runs in the job's thread first and returns a command to run ahead of
        ``args``, or None; that command failing (or ``before`` raising) only warns, then
        ``args`` runs. Cancel stops both. ``cwd``: the folder the commands run in.
        ``on_result``: runs after ``on_success``, given the job (its ``--json`` envelope in
        ``result``). ``on_end``: runs when the job ended, however it did (a failure is only
        logged)."""
        with self.lock:
            if any(j.status == "running" for j in self.jobs.values()):
                raise bad("another job is already running", 409)
            if check:
                check()
            job = Job(project_id, action, plan, cwd)
            if before is None:
                try:
                    job.proc = self._popen(job, args)
                except OSError as err:
                    raise bad(f"could not start job: {err}", 500) from err
            for old in [k for k, j in self.jobs.items() if j.status != "running"][
                : max(0, len(self.jobs) + 1 - MAX_JOBS)
            ]:
                del self.jobs[old]
            self.jobs[job.id] = job
        threading.Thread(
            target=self._watch, args=(job, args, on_success, before, on_result, on_end), daemon=True
        ).start()
        return job

    @staticmethod
    def _popen(job: Job, args: list[str]) -> subprocess.Popen[str]:
        return spawn(args) if job.cwd is None else spawn(args, cwd=job.cwd)

    def _spawn(self, job: Job, args: list[str]) -> bool:
        """Start ``args`` as the job's process, unless it was cancelled; False if it was."""
        with self.lock:  # a Cancel either sees this process or stops the job before it starts
            if job.cancelled:
                return False
            job.proc = self._popen(job, args)
        return True

    @staticmethod
    def _follow(job: Job) -> int:
        """Read the job's process output into the job until it exits; its exit code."""
        proc = job.proc
        assert proc is not None and proc.stdout is not None
        for line in proc.stdout:
            try:
                job.add_output(line)
            except Exception:  # one odd line must not end the watcher: keep it as text
                logger.exception("job %s could not read an output line", job.id)
                job.logs.append(line.rstrip())
                job.logged += 1
        return proc.wait()

    def _first(self, job: Job, before: Callable[[Job], list[str] | None]) -> None:
        """Run the command ``before`` asks for ahead of the job's own; a failure only warns."""
        try:
            first = before(job)
        except Exception as err:
            logger.exception("job %s: the step before it failed", job.id)
            job.note("WARNING", f"skipped the step before {job.action}: {err}")
            return
        if first is None:
            return
        try:
            if not self._spawn(job, first):
                return
        except OSError as err:
            job.note("WARNING", f"could not start the step before {job.action}: {err}")
            return
        start = job.logged
        code = self._follow(job)
        with self.lock:
            job.proc = None  # exited: a Cancel from now on has nothing of it to stop
        if code != 0 and not job.cancelled:
            job.interrupt_steps()
            job.note("WARNING", f"the step before {job.action} failed (exit code {code})")
        elif code == 0 and job.logged > start:
            # e.g. 360 stitch exits 0 when the SDK failed on a recording: it logs an ERROR
            said = list(job.logs)[-(job.logged - start) :]
            if any(re.search(r"\bERROR\s", ln) for ln in said):
                job.note(
                    "WARNING",
                    f"the step before {job.action} reported errors: those recordings export "
                    "from the 360 preview",
                )

    def _watch(
        self,
        job: Job,
        args: list[str],
        on_success: Callable[[], None] | None,
        before: Callable[[Job], list[str] | None] | None = None,
        on_result: Callable[[Job], None] | None = None,
        on_end: Callable[[], None] | None = None,
    ) -> None:
        try:
            mark = 0  # the job's own error is an ERROR line of its own command
            if before is not None:
                self._first(job, before)
                mark = job.logged
                try:
                    started = self._spawn(job, args)
                except OSError as err:
                    job.error, job.status = f"could not start job: {err}", "failed"
                    return
                if not started:
                    job.interrupt_steps()
                    job.status = "cancelled"
                    return
            code = self._follow(job)
            if job.cancelled or code != 0:
                job.interrupt_steps()
            if job.cancelled:
                job.status = "cancelled"
            elif code != 0 and self._busy(job):
                job.status = "failed"
            elif code != 0:
                own = list(job.logs)[-(job.logged - mark) :] if job.logged > mark else []
                last = next(
                    (m.group(1) for ln in reversed(own) if (m := re.search(r"ERROR\s+(.*)", ln))),
                    None,
                )
                job.error, job.status = last or f"exit code {code}", "failed"
            else:
                try:
                    if on_success:
                        on_success()
                    if on_result:
                        on_result(job)
                    job.status = "completed"
                except HTTPException as err:  # a refused write: its message is the error
                    job.error, job.status = err.detail, "failed"
                except Exception as err:  # surfaced to the user as the job error
                    logger.exception("job %s post-processing failed", job.id)
                    job.error, job.status = str(err), "failed"
        finally:
            if on_end:
                try:
                    on_end()
                except Exception:
                    logger.exception("job %s: the cleanup after it failed", job.id)
            job.done.set()

    @staticmethod
    def _busy(job: Job) -> bool:
        """The job's command stopped with ``folder_busy`` (another run holds the project's run
        lock or a clips folder's): the error is :data:`BUSY_TEXT`, and the CLI's ERROR line,
        which tells the user to delete the lock file, leaves the log."""
        err = (job.result or {}).get("error")
        code = err.get("code") if isinstance(err, dict) else None
        if isinstance(code, str):
            job.error_code = code
        if code != jsonout.FOLDER_BUSY:
            return False
        if isinstance(err.get("message"), str) and err["message"]:
            job.drop_lines(err["message"])
        job.note("ERROR", BUSY_TEXT)
        job.error = BUSY_TEXT
        return True

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise bad("unknown job", 404)
        return job

    def cancel(self, job_id: str) -> Job:
        """Stop the job in the background and return at once; it reads "cancelling" until it exits."""
        job = self.get(job_id)
        with self.lock:  # one killer per job, however often Cancel is pressed
            start = job.status == "running" and not job.cancelled
            if start:
                job.cancelled = True  # also between two steps: the next one never starts
            proc = job.proc
        if start and proc is not None:
            threading.Thread(target=procs.kill_tree, args=(proc,), daemon=True).start()
        return job

    def shutdown(self) -> None:
        # One by one: at most one job runs at a time (``start``), so this waits JOB_STOP_S at most.
        for job in list(self.jobs.values()):
            if job.status != "running":
                continue
            with self.lock:  # as in ``cancel``: did a killer already send the TERM?
                stopping, job.cancelled = job.cancelled, True
                proc = job.proc
            if proc is None:  # between two steps: the next one never starts
                continue
            if not stopping:
                procs.kill_tree(proc)
            elif not job.done.wait(procs.JOB_STOP_S):  # a second TERM could cut short its cleanup
                procs.kill_tree(proc, term=False)


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
# while a source cannot be read, a proxy or window that names no other source goes only once
# it has been idle this long (a drive that stays offline must not keep them forever)
PROXY_OFFLINE_PRUNE_S = 30 * 86400.0
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
    ``color``: the source's :func:`~cutpoint.encode.preview_filter` (HDR and BT.2020 become SDR
    Rec.709, tagged so; full-range BT.601 becomes TV range, its tags kept). ``lo``/``hi``: only
    that part of the source, in seconds (a window proxy, whose 0 is ``lo``).
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
        *(BT709_TAGS if color and color != TV_RANGE else []), *length,
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
        # key -> (browser plays it as is, duration, audio tracks, preview_filter of a proxy)
        # or None
        self.probed: dict[str, tuple[bool, float, int, str] | None] = {}
        # (project, key) -> "preparing" | "failed": each project builds its own proxy file
        self.proxies: dict[tuple[Path, str], str] = {}
        # (project, key) -> share of its whole proxy encoded, 0-1
        self.progress: dict[tuple[Path, str], float] = {}
        self.procs: set[subprocess.Popen[Any]] = set()
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
        # (project, content key) of colour-filtered sources whose unfiltered proxies went
        self.tone_maps_pruned: set[tuple[Path, str]] = set()
        self.closing = False
        self.gpu_failed = False  # a proxy failed with the GPU encoder and worked with libx264
        # still file name -> (its builder lock, requests using it); dropped when the last one leaves
        self.frame_locks: dict[str, tuple[threading.Lock, int]] = {}
        self.frame_slots = threading.BoundedSemaphore(FRAME_SLOTS)
        self.frames_pruned: set[Path] = set()  # projects whose stale stills went this run
        # one envelope decodes at a time, on its own worker; content keys queued or decoding, and
        # the failure of a finished decode, until the next request reports it
        self.envelope_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cutpoint-wave")
        self.envelopes_running: set[str] = set()
        self.envelope_asked: dict[str, float] = {}  # content key -> monotonic time of the last ask
        self.envelope_errors: dict[str, str] = {}
        self.envelopes_pruned: set[Path] = set()  # projects whose stale envelopes went this run
        # key -> still_filter of its stills (#412, #441): also for a source the browser plays
        # as is
        self.still_color: dict[str, str] = {}

    @staticmethod
    def key(real: Path) -> str | None:
        try:
            st = real.stat()
        except OSError:
            return None
        return hashlib.sha1(f"{real}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]

    def info(self, real: Path, key: str) -> tuple[bool, float, int, str] | None:
        """``(browser plays it as is, duration, audio tracks, colour filter of its proxy)``, or
        None if unreadable. The filter is :func:`~cutpoint.encode.preview_filter`'s, "" for a
        source that plays as is."""
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
                color = preview_filter(i)
                self.still_color[key] = still_filter(i)
                self.probed[key] = (ok, i.duration, i.audio_streams, "" if ok else color)
            except (FFmpegError, OSError, ValueError, KeyError):
                self.probed[key] = None
        return self.probed[key]

    def _proxy_key(self, key: str | None) -> str | None:
        """The key naming a source's proxies (whole, windows, builds): its content key, mixed
        with the colour filter of a probed source that has one (HDR, BT.2020, full-range
        BT.601), so a changed filter rebuilds its proxies while the others keep their names.
        Never runs ffprobe."""
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
            threading.Thread(
                target=procs.kill_tree, args=(kill, procs.SOFT_STOP_S), daemon=True
            ).start()

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
        """Once per run: delete the proxies of a source with a colour filter (HDR, BT.2020,
        full-range BT.601) named by its content key alone (built before it had one), unless
        written or used in the last PROXY_PRUNE_GRACE_S."""
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
            threading.Thread(
                target=procs.kill_tree, args=(kill, procs.SOFT_STOP_S), daemon=True
            ).start()

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
        read (an offline drive), a proxy or window naming no current source goes only once idle
        for PROXY_OFFLINE_PRUNE_S (it may be the offline source's), and the readable sources are
        probed only when such a proxy has been idle that long.

        A proxy no known key names may be of a source with a colour filter (HDR, BT.2020,
        full-range BT.601) not probed yet (its proxy key holds the filter): sources are then
        probed, in order, until every such proxy is matched, so this may run ffprobe. Proxies
        such a source's content key names (built before it had a filter) go in :meth:`prepare`
        instead, so no other source is probed for them."""
        folder = project / UI_DIR / "proxy"
        if project in self.proxies_pruned:
            return
        cutoff = time.time_ns() - int(PROXY_PRUNE_GRACE_S * 1e9)
        offline_cutoff = time.time_ns() - int(PROXY_OFFLINE_PRUNE_S * 1e9)
        reals = [remap_source(path) for path in paths]
        keys = {self._proxy_key(self.key(real)) for real in reals}
        unsure = None in keys  # a source is unreachable: its proxies' names are unknown
        keys.discard(None)
        if unsure:  # probe only for a proxy idle long enough to go if no source names it
            left = {
                m[1]
                for f in folder.glob("*.mp4")
                if (m := PROXY_FILE.fullmatch(f.name))
                and m[1] not in keys
                and _idle(f, offline_cutoff)
            }
        else:
            named = {
                m[1] for m in (PROXY_FILE.fullmatch(f.name) for f in folder.glob("*.mp4")) if m
            }
            left = named - keys
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
            drop += [
                f
                for f in folder.glob("*.mp4")
                if (m := PROXY_FILE.fullmatch(f.name))
                and ".w" not in f.name
                and m[1] not in keys
                and (not unsure or _idle(f, offline_cutoff))
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
                    if not unsure or _idle(f, offline_cutoff):
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
                # Never lower: ffmpeg reports the furthest stream not yet finished, so it steps
                # back when audio ends ahead of a video encode that is still catching up.
                with self.lock:
                    done = min(1.0, int(m[2]) / 1e6 / duration)
                    self.progress[key] = max(self.progress.get(key, 0.0), done)
        proc.wait()
        return "\n".join(rest)[-500:]

    def frame(
        self, project: Path, path: str, at: float, width: int, *, key: bool = False
    ) -> Path | None:
        """A JPEG still of ``path`` at ``at`` s (clamped into the source), ``width`` px wide.

        Cached as ``ui/frames/<content key>-<centiseconds>-<width>.jpg`` and built once even when
        requested concurrently. A still goes through its source's
        :func:`~cutpoint.encode.still_filter` (HDR, BT.2020 and BT.709 sources), so it shows what
        exports show, and then its name ends in ``-<filter hash>``, so stills cached before the
        conversion (or a changed one) are not reused. ``key``: the keyframe at or before ``at``
        (fast, for the filmstrip), cached with ``-k`` after the width, so exact stills never get one.
        None if the source cannot be read.

        Raises:
            FFmpegError: If ffmpeg decodes no image.
        """
        real = remap_source(path)
        content, res = self._lookup(real)
        if res is None or content is None:
            return None
        at = min(at, FRAME_MAX_AT)  # a huge time (1e308) would overflow its centiseconds
        if res[1] > 0:
            at = min(at, max(0.0, res[1] - 0.1))  # the last instants may hold no decodable frame
        cs = round(at * 100)
        color = self.still_color.get(content, "")
        name = f"{content}-{cs}-{width}" + ("-k" if key else "")
        if color:  # a converted still never reuses one cached before (or with another filter)
            name += f"-{hashlib.sha1(color.encode()).hexdigest()[:8]}"
        out = _own(project, f"{UI_DIR}/frames/{name}.jpg")  # nor cached through one
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
                            real,
                            cs / 100,
                            width=width,
                            quality=5,
                            timeout=FRAME_TIMEOUT_S,
                            color=color,
                            keyframe=key,
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
        """Keep at most ``FRAME_CACHE_MAX`` stills in ``folder``, dropping the least recently used.

        The folder is listed without ``self.lock`` (1500 stats would hold up every other request);
        two evictions at once pick the same oldest stills, and ``_drop_frames`` (under the lock)
        still skips any being built."""
        stills: list[tuple[int, Path]] = []
        for f in folder.glob("*.jpg"):
            with contextlib.suppress(OSError):
                stills.append((f.stat().st_mtime_ns, f))
        if len(stills) > FRAME_CACHE_MAX:
            stills.sort()
            with self.lock:
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

    def envelope(self, project: Path, path: str) -> dict[str, Any] | None:
        """The audio envelope of ``path``: ``{"step": seconds per value, "peaks": [0-1, ...]}``,
        ``peaks`` empty for a source without audio. None if the source cannot be read.
        ``{"pending": True}`` while its decode runs: the first call queues it, later ones
        (the page asks again) find it cached.

        Cached as ``ui/envelope/<content key>-<version>.json``; decoded one source at a time on
        its own worker, once even when requested concurrently (see :mod:`cutpoint.envelope`), at
        low priority and streamed, so a long source costs no memory. A cached source answers at
        once, whatever decodes.

        Raises:
            FFmpegError: If the decode of an earlier call failed or ran longer than
                ``ENVELOPE_TIMEOUT_S`` (reported once), or the server is shutting down.
        """
        real = remap_source(path)
        content, res = self._lookup(real)
        if res is None or content is None:
            return None
        if res[2] == 0:
            return {"step": audio_envelope.BASE_S, "peaks": []}
        out = _own(project, f"{UI_DIR}/envelope/{content}-{ENVELOPE_VERSION}.json")
        hit = cache.load_json(out)
        if isinstance(hit, dict) and isinstance(hit.get("peaks"), list):
            return hit
        with self.lock:
            if content in self.envelope_errors:
                raise FFmpegError(self.envelope_errors.pop(content))
            if self.closing:
                raise FFmpegError("the server is shutting down")
            self.envelope_asked[content] = time.monotonic()
            if content not in self.envelopes_running:
                self.envelopes_running.add(content)
                self.envelope_pool.submit(
                    self._decode_envelope, project, real, content, res[2], out
                )
        return {"pending": True}

    def _decode_envelope(self, project: Path, real: Path, content: str, tracks: int, out: Path):
        """Decode one source's envelope into ``out`` (on the envelope worker); a failure is kept
        in ``envelope_errors`` for the next ``envelope`` call. A job nobody asked for in the last
        ``ENVELOPE_STALE_S`` (the page moved on to another source) is dropped unrun."""
        try:
            with self.lock:
                asked = self.envelope_asked.get(content, 0.0)
            if time.monotonic() - asked > ENVELOPE_STALE_S:
                return
            hit = cache.load_json(out)
            if not (isinstance(hit, dict) and isinstance(hit.get("peaks"), list)):
                self._write_envelope(project, real, tracks, out)
        except Exception as err:
            logger.exception("envelope decode of %s failed", real.name)
            with self.lock:
                self.envelope_errors[content] = str(getattr(err, "detail", "") or err)
        finally:
            with self.lock:
                self.envelopes_running.discard(content)
                self.envelope_asked.pop(content, None)

    def _write_envelope(self, project: Path, real: Path, tracks: int, out: Path) -> None:
        _ui_folder(project, "envelope")
        with self.lock:  # as in ``_encode``: shutdown sees the process, or none starts
            if self.closing:
                raise FFmpegError("the server is shutting down")
            proc = spawn(audio_envelope.ffmpeg_args(str(real), tracks), binary=True)
            self.procs.add(proc)
        timed_out = threading.Event()

        def expire() -> None:
            timed_out.set()
            procs.kill_tree(proc, 0.0)

        timer = threading.Timer(ENVELOPE_TIMEOUT_S, expire)
        timer.daemon = True
        timer.start()
        try:
            assert proc.stdout is not None
            peaks = audio_envelope.read_peaks(proc.stdout)
            proc.wait()
        finally:
            timer.cancel()
            if proc.poll() is None:
                procs.kill_tree(proc, 0.0)
                proc.wait()
            if proc.stdout is not None:
                proc.stdout.close()
            with self.lock:
                self.procs.discard(proc)
        if timed_out.is_set():
            raise FFmpegError(f"decoding the audio took longer than {ENVELOPE_TIMEOUT_S:.0f} s")
        if proc.returncode != 0:
            raise FFmpegError(f"ffmpeg could not decode the audio (exit {proc.returncode})")
        step, values = audio_envelope.shrink(peaks)
        cache.save_bytes(
            out, json.dumps({"step": step, "peaks": values}, separators=(",", ":")).encode()
        )

    def prune_envelopes(self, project: Path, paths: list[str]) -> None:
        """Once per run and project: delete the envelopes of sources that changed or left."""
        folder = project / UI_DIR / "envelope"
        if project in self.envelopes_pruned:
            return
        self.envelopes_pruned.add(project)
        if not folder.is_dir() or not _inside(project, f"{UI_DIR}/envelope"):
            return
        keep = {
            f"{key}-{ENVELOPE_VERSION}.json"
            for key in {self.key(remap_source(path)) for path in paths} - {None}
        }
        for f in folder.glob("*.json"):
            if f.name not in keep:
                with contextlib.suppress(OSError):
                    f.unlink()

    def encoding_for(self, project: Path) -> bool:
        """A whole proxy or a window proxy of ``project`` is queued or encoding now."""
        with self.lock:
            return any(
                q == project and st == "preparing" for (q, _), st in self.proxies.items()
            ) or any(
                w is not None and w.project == project
                for w in (self.window_running, *self.windows_queue)
            )

    def forget(self, project: Path) -> None:
        """Drop what is remembered of ``project`` (it was deleted): a project made later under
        the same name starts clean."""
        with self.lock:
            for table in (
                self.proxies,
                self.progress,
                self.windows,
                self.retired,
                self.window_failures,
            ):
                for k in [k for k in table if k[0] == project]:
                    del table[k]
            self.proxies_pruned.discard(project)
            self.frames_pruned.discard(project)
            self.envelopes_pruned.discard(project)
            self.tone_maps_pruned -= {t for t in self.tone_maps_pruned if t[0] == project}

    def shutdown(self) -> None:
        with self.lock:  # as in ``_encode``: no proxy ffmpeg starts after this
            self.closing = True
            running = list(self.procs)
            self.windows_queue.clear()
            self.window_wake.notify_all()  # the window lane ends
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.envelope_pool.shutdown(wait=False, cancel_futures=True)
        for proc in running:
            # a bare ffmpeg: no soft stop of its own to wait for
            procs.kill_tree(proc, procs.SOFT_STOP_S)


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
        v = diag.get(key)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

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
    try:
        st = f.stat()
    except OSError:
        return None
    return f.name, st.st_size, st.st_mtime_ns


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
        self.wake = threading.Condition(self.lock)
        self.queue: deque[PanoPreview] = deque()
        self.running: PanoPreview | None = None
        self.lane: threading.Thread | None = None
        self.closing = False
        # (project, cid) -> {"state": failed | cancelled, "reason", "ident"}
        self.outcomes: dict[tuple[Path, str], dict[str, Any]] = {}
        # (project, cid) -> {"ident", "name", "lo", "hi", "kind", "reason"} of the preview made
        # (or found on disk) this run: only that one is listed ready, and only for its ident
        self.done: dict[tuple[Path, str], dict[str, Any]] = {}
        # project -> (signature of its candidate cache and render settings, {cid: entry info})
        self.memo: dict[Path, tuple[Any, dict[str, dict[str, Any]]]] = {}
        self.pruned: set[Path] = set()

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

    def _state(self, project: Path, cid: str, info: dict[str, Any]) -> dict[str, Any] | None:
        key = (project, cid)
        if not info["clip"].lrv:
            return {"state": "failed", "reason": "no .lrv proxy to render this camera path from"}
        with self.lock:
            run = self.running
            if run is not None and run.key == key and not run.cancelled:
                elapsed = round(time.time() - run.started, 1) if run.started else 0.0
                planned = run.name is not None
                return {
                    "state": "rendering",
                    "elapsed": elapsed,
                    "seconds": round(run.hi - run.lo, 1) if planned else None,
                    "progress": None if run.progress is None else round(run.progress, 3),
                }
            if any(j.key == key and not j.cancelled for j in self.queue):
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
            self._queue(PanoPreview(project, cid, info))
            view = self.listing(project).get(cid)
        return view

    def _queue(self, job: PanoPreview) -> None:
        stop = None
        with self.lock:
            if self.closing:
                return
            for queued in self.queue:
                queued.cancelled = True  # superseded: the user moved on (no outcome to show)
            self.queue.clear()
            self.queue.append(job)
            run = self.running
            if run is not None and not run.cancelled:
                run.cancelled = True
                stop = run.scope
            if self.lane is None:
                self.lane = threading.Thread(target=self._lane, name="cutpoint-pano", daemon=True)
                self.lane.start()
            self.wake.notify()
        if stop is not None:
            stop.kill()

    def cancel(self, project: Path, cid: str) -> bool:
        """Stop the queued or running preview of ``cid`` (its partial file goes); False if none."""
        key, stop, ident = (project, cid), None, None
        with self.lock:
            for job in [j for j in self.queue if j.key == key]:
                job.cancelled = True
                self.queue.remove(job)
                ident = job.ident
            run = self.running
            if run is not None and run.key == key and not run.cancelled:
                run.cancelled = True
                stop, ident = run.scope, run.ident
            if ident is not None:
                self.outcomes[key] = {"state": "cancelled", "reason": "", "ident": ident}
        if stop is not None:
            stop.kill()
        return ident is not None

    def _lane(self) -> None:
        while True:
            with self.lock:
                while not self.queue and not self.closing:
                    self.wake.wait()
                if self.closing:
                    return
                job = self.running = self.queue.popleft()
                job.started = time.time()
            try:
                self._render(job)
            finally:
                with self.lock:
                    self.running = None

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
            busy = {j.name for j in (self.running, *self.queue) if j is not None and j.name}
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

    def shutdown(self) -> None:
        with self.lock:
            self.closing = True
            jobs = [j for j in (self.running, *self.queue) if j is not None]
            self.queue.clear()
            self.wake.notify_all()
        for job in jobs:
            job.cancelled = True
            job.scope.kill()


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
    """The project folder ``pid`` (a direct child of the projects folder), else 404; 409 while
    it is being deleted."""
    root = projects_dir()
    if not pid or pid.startswith(".") or any(c in pid for c in "/\\\0") or pid != pid.strip():
        raise bad("unknown project", 404)
    p = root / pid
    if p in DELETING:
        raise bad("this project is being deleted", 409)
    try:
        ok = p.is_dir() and p.resolve().parent == root.resolve()
    except OSError:
        ok = False
    if not ok or not _is_project(p):
        raise bad("unknown project", 404)
    return p


DELETING: set[Path] = set()  # projects being deleted now (under ``Jobs.lock``): see project_dir


def protected_sources() -> dict[str, str]:
    """``{normcased real path: project name}`` of every source video or folder that a project
    under the projects folder reads (its picked inputs and the sources of its analysis): deleting
    a project never removes one."""
    found: dict[str, str] = {}
    try:
        projects = sorted(d for d in projects_dir().iterdir() if d.is_dir())
    except OSError:
        return found
    for d in projects:
        meta, hl = read_meta(d), _highlights(d) or {}
        raw = [
            *picked_inputs(meta),
            *((hl.get("settings") or {}).get("inputs") or []),
            *(s.get("path") for s in hl.get("sources") or [] if isinstance(s, dict)),
        ]
        for path in raw:
            if isinstance(path, str) and path:
                with contextlib.suppress(OSError, ValueError):
                    key = projectdelete.source_key(path)
                    found.setdefault(key, str(meta.get("name") or d.name))
    return found


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


def picked_inputs(meta: dict[str, Any]) -> list[str]:
    """The files and folders a UI project was made from (``analyze``'s inputs), else []. One
    input is ``source``; several are ``inputs`` too, ``source`` being the first."""
    inputs = meta.get("inputs")
    if isinstance(inputs, list) and inputs and all(isinstance(x, str) and x for x in inputs):
        return list(inputs)
    return [str(meta["source"])] if meta.get("source") else []


def picked_filters(meta: dict[str, Any]) -> tuple[list[str], bool]:
    """``(analyze --exclude patterns, --recursive)`` a UI project was made with."""
    ex = meta.get("exclude")
    globs = [g for g in ex if isinstance(g, str)] if isinstance(ex, list) else []
    return globs, meta.get("recursive") is True


def find_picked(inputs: list[str], exclude: list[str], recursive: bool) -> list[Path]:
    """The videos ``analyze`` finds in ``inputs``, de-duplicated in input order. As ``analyze``
    does, a missing input is listed at its ``source_remap`` target: the videos keep the original
    spelling, and ``exclude`` is matched against it.

    Raises:
        OSError: If an input cannot be listed (missing, unreadable, stale mount).
    """
    found: list[Path] = []
    for x in inputs:
        target = Path(x)
        io = _listed_folder(target)
        if io == target:
            found += find_videos([target], exclude=exclude, recursive=recursive)
            continue
        base, orig = io.resolve(), target.resolve()
        for read in find_videos([io], recursive=recursive):
            f = orig / read.relative_to(base) if read.is_relative_to(base) else read
            # source_remap_exclude: it would be read at the missing original
            if not matches_glob(f, exclude) and remap_source(f) != f:
                found.append(f)
    return list(dict.fromkeys(found))


def source_entries(
    p: Path, hl: dict[str, Any] | None, listed: list[Path | str] | None = None
) -> list[dict[str, Any]]:
    """The project's playable-in-principle sources: ``[{path, id, name, duration}]``. The picked
    inputs it lists itself (before their analysis) are added to ``listed``, followed by the error
    message when that listing failed."""
    meta = read_meta(p)
    pano = meta.get("kind") == "360"
    picked = picked_inputs(meta)
    if picked:
        roots = [str(_pano_folder(x)) for x in picked] if pano else picked
    else:  # command-line project: the inputs its analysis recorded
        roots = [str(x) for x in ((hl or {}).get("settings") or {}).get("inputs") or []]
    paths: dict[str, float | None] = {}
    for s in (hl or {}).get("sources") or []:
        if isinstance(s, dict) and isinstance(s.get("path"), str):
            dur = s.get("duration")
            paths.setdefault(s["path"], dur if isinstance(dur, (int, float)) else None)
    if not paths and picked:
        if listed is not None:
            listed.extend(map(Path, picked))
        try:
            if pano:  # one .lrv per clip 360 analysis can use (those of meta.clips, if set)
                only = meta.get("clips") or None
                _, clips, _ = inspect_source(Path(picked[0]))
                found = [clips[n]["lrv"] for n in _usable(clips) if not only or n in only]
            else:
                found = find_picked(picked, *picked_filters(meta))
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
    entries of the picked inputs; a folder's mtime changes when an entry directly inside it is
    added, removed or renamed. So a hit costs a stat per file and input, and any change to them
    rebuilds it, as does a change to ``source_remap`` or ``source_remap_exclude`` (``get_config``
    rereads an edited ``config.json``). ``SOURCES_CACHE_S`` bounds what stats cannot see (a
    sub-folder of a ``--recursive`` project changed, a link under a source retargeted, a
    filesystem with coarse mtimes). Callers must not change the entries."""

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
        picked = picked_inputs(meta)
        if picked and meta.get("kind") == "360":
            picked = [str(_pano_folder(picked[0]))]  # its .lrv files are listed, a file's too
        files += [_listed_folder(Path(x)) for x in picked]
        stamps += [file_stamp(f) for f in files[2:]]
        listed: list[Path | str] = []
        entries = source_entries(p, _highlights(p), listed)
        n = 2 + len(picked) if listed else 2  # the picked inputs count only while listed
        error = next((x for x in listed if isinstance(x, str)), None)
        with self.lock:
            self.kept.pop(key, None)
            if error is not None:  # an input could not be listed: the next request lists again
                return entries, error
            self.kept[key] = (at, files[:n], stamps[:n], remap, entries)
            while len(self.kept) > SOURCES_CACHE_MAX:
                del self.kept[next(iter(self.kept))]
        return entries, None


def source_folder(p: Path) -> Path | None:
    """The folder holding the project's videos: the deepest folder shared by its picked inputs
    (a file's parent), or for a command-line project by its recorded sources. Never client
    input."""
    picked = picked_inputs(read_meta(p))
    try:
        if picked:
            # A picked folder that is gone must not fall back to its parent; only files step up.
            folders = []
            for source in picked:
                x = remap_source(Path(source))
                missing_file = not x.exists() and x.suffix.lower() in VIDEO_EXTS | PANO_EXTS
                is_file = x.is_file() or missing_file
                folders.append(x.parent if is_file else x)
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


def bit_description(c: dict[str, Any], events: Any) -> str:
    """What the dialogue reader said happens in the moment ``c`` if it is a dialogue moment (one a
    talk stretch nominated or widened, or that has a ``dialogue_title``): the ``description`` of
    the most confident dialogue event of ``events`` (the events.json object) inside it, else the
    ``dialogue_description`` highlights.json keeps; "" for any other moment."""
    nominated = ((c.get("signals") or {}).get("events") or {}).get("nominated") or []
    if not (
        any(str(n).startswith("dialogue-") for n in nominated)
        or c.get("widened_from")
        or c.get("dialogue_title")
        or c.get("dialogue_description")
    ):
        return ""
    best, conf = "", -1.0
    for e in (events.get("events") or []) if isinstance(events, dict) else []:
        if (
            isinstance(e, dict)
            and e.get("kind") == "text"
            and e.get("type") == scoring.DIALOGUE_TYPE
            and e.get("source") == c.get("source")
            and isinstance(e.get("start"), int | float)
            and isinstance(e.get("end"), int | float)
            and e["start"] >= c["start"] - 0.01
            and e["end"] <= c["end"] + 0.01
            and e.get("description")
            and (e.get("confidence") or 0) > conf
        ):
            best, conf = str(e["description"]), e.get("confidence") or 0
    return best or str(c.get("dialogue_description") or "")


def moments_view(hl: dict[str, Any] | None, events: Any = None) -> dict[str, Any] | None:
    """Keep saved trims attached to source intervals, not mutable CLI ranks; ``cli_id`` keeps the
    CLI id (h001) that the cards show and ``clips --ids`` takes. ``display_label`` is the label
    the moments list shows: a generic one (``action-moment``) gives way to what the vision model
    saw (:func:`cutpoint.assistant.context.shown_label`, as in AI Chat); ``label`` stays the
    analysis label, which a moment added to the sequence takes. ``long_bit`` marks a long
    dialogue bit (the "Bit" badge and "Long bits" filter), with ``bit_description`` what
    happens in a dialogue moment (:func:`bit_description`; ``events``: the project's events.json
    object). A
    dialogue moment's ``display_label`` is its readable ``dialogue_title``, not the slug label."""
    if hl is None:
        return None
    return {
        **hl,
        "candidates": [
            {
                **c,
                "cli_id": c.get("id"),
                "display_label": c.get("dialogue_title")
                or ai_context.shown_label(c.get("label"), c),
                "long_bit": scoring.is_long_dialogue(c),
                "bit_description": bit_description(c, events),
                "id": hashlib.sha1(
                    f"{os.path.realpath(c['source'])}|{float(c['start'])}|{float(c['end'])}".encode()
                ).hexdigest()[:12],
            }
            for c in hl.get("candidates") or []
        ],
    }


def _secs(v: Any) -> float | None:
    """``v`` as a positive, finite number of seconds, else None."""
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    return float(v) if ok and v > 0 else None


def analysis_took(p: Path, meta: dict[str, Any], hl: dict[str, Any] | None) -> float | None:
    """Wall-clock seconds of the last analysis. ``timings.total_s`` covers the flat analysis
    only (a re-rank keeps it and adds ``timings.rerank_s``), not the scan, events and camera
    paths of ``360 analyze``, so a 360 project's time is ``total_s`` in
    ``pano_timings.json`` (every analyze run since #564; a ``--views-only`` run, which leaves
    ``highlights.json`` alone, keeps the earlier analysis keys and time and adds
    ``views_only_s``), or for an older file without it the scan and analyze totals; None
    when they are missing."""
    if meta.get("kind") == "360":
        pano = _load(p, "pano_timings.json")
        pano = pano if isinstance(pano, dict) else {}
        if _secs(pano.get("analyze_s")) is not None and (whole := _secs(pano.get("total_s"))):
            return whole
        scan, total = _secs(pano.get("scan_s")), _secs(pano.get("analyze_total_s"))
        return None if scan is None or total is None else round(scan + total, 2)
    timings = (hl or {}).get("timings")
    return _secs(timings.get("total_s")) if isinstance(timings, dict) else None


def analysis_stats(hl: dict[str, Any] | None, took: float | None) -> dict[str, float] | None:
    """Seconds of video the last analysis read, from highlights.json, and the seconds it
    ``took`` (#459, see ``analysis_took``); None when either is unknown."""
    # analyzed_s: the part a recovered or cut-short read covered, else the whole source
    parts = [
        _secs(s.get("analyzed_s", s.get("duration")))
        for s in (hl or {}).get("sources") or []
        if isinstance(s, dict)
    ]
    video = [v for v in parts if v is not None]
    if took is None or not video or len(video) < len(parts):
        return None
    return {"video_s": round(sum(video), 3), "took_s": took}


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
        if not isinstance(label, str) or len(label) > ai_ops.LABEL_MAX:
            raise bad(f"segment {i}: label must be text of at most {ai_ops.LABEL_MAX} characters")
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


def validate_joins(raw: Any) -> list[dict[str, str]]:
    """The cleaned ``joins`` of a review: a clip pair by segment id (``a`` then ``b``) and its
    ``transition``. The ids need not be in the sequence now: a pair a reorder separated keeps
    its choice for when the two are next to each other again."""
    if not isinstance(raw, list):
        raise bad("joins must be a list")
    if len(raw) > MAX_SEGMENTS:
        raise bad(f"at most {MAX_SEGMENTS} joins")
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise bad(f"join {i}: object expected")
        a, b, kind = item.get("a"), item.get("b"), item.get("transition")
        if not all(isinstance(s, str) and 0 < len(s) <= 64 for s in (a, b)):
            raise bad(f"join {i}: a and b must be segment ids of 1-64 characters")
        if (a, b) in seen:
            raise bad(f"join {i}: the pair {a}, {b} is already set")
        seen.add((a, b))
        if kind not in JOIN_TRANSITIONS:
            raise bad(f"join {i}: transition must be 'cut' or 'fade'")
        out.append({"a": a, "b": b, "transition": kind})
    return out


def _joins_of(review: Any) -> list[dict[str, str]]:
    """The joins saved in a ``review.json`` document; [] when it has none or they are invalid."""
    try:
        return validate_joins(review.get("joins", []) if isinstance(review, dict) else [])
    except HTTPException:
        return []


def clip_joins(folder: Path, joins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    r"""Each join of the clips in ``folder``, in clip order: the two files (``from``, ``to``),
    their manifest ``id``\ s (``a``, ``b``; None for a clip it does not list) and the review's
    ``transition`` for the pair, None when ``joins`` has no choice for it."""
    from cutpoint.planning import manifest_entries  # noqa: PLC0415

    files = clip_files(folder)
    entries = manifest_entries(folder, files)
    chosen = {(j["a"], j["b"]): j["transition"] for j in joins}
    out = []
    for x, y in pairwise(files):
        a, b = entries.get(x.name, {}).get("id"), entries.get(y.name, {}).get("id")
        out.append({"from": x.name, "to": y.name, "a": a, "b": b, "transition": chosen.get((a, b))})
    return out


def export_joins(folder: Path, joins: list[dict[str, str]]) -> list[dict[str, str]]:
    r"""The review's ``joins`` as ``assemble --joins`` entries for the clips in ``folder``,
    through its manifest's ``id``\ s. Only pairs next to each other in the clip order: the
    others use the global transition, which is no mismatch to warn about."""
    return [
        {"from": j["from"], "to": j["to"], "transition": j["transition"]}
        for j in clip_joins(folder, joins)
        if j["transition"]
    ]


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


def _events_sig(p: Path) -> str | None:
    """``<mtime_ns>-<size>`` of the project's ``events.json``, which changes when a run rewrites
    it; None when there is none."""
    try:
        st = (p / "events.json").stat()
    except OSError:
        return None
    return f"{st.st_mtime_ns}-{st.st_size}"


def _rail_events(p: Path) -> list[dict[str, Any]]:
    """The time-located events of ``events.json`` the trim rail marks, cut to ``{source, start,
    end, kind, type, label, confidence}``: sounds (``audio``), on-screen text (``ocr``) and
    camera telemetry such as crashes. No model descriptions and no coverage spans. At most
    ``EVENTS_MAX`` (the most confident), by start. [] when the file is missing or unreadable."""
    doc = _load(p, "events.json")
    raw = doc.get("events") if isinstance(doc, dict) else None
    out = []
    for e in raw if isinstance(raw, list) else []:
        if not isinstance(e, dict) or not isinstance(e.get("source"), str):
            continue
        if e.get("kind") not in EVENT_KINDS or str(e.get("type", "")).startswith("coverage"):
            continue
        start, end, conf = e.get("start"), e.get("end"), e.get("confidence")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (start, end)):
            continue
        label = e.get("type") if e["kind"] == "telemetry" else e.get("label")
        out.append(
            {
                "source": e["source"],
                "start": start,
                "end": max(start, end),
                "kind": e["kind"],
                "type": str(e.get("type", ""))[:40],
                "label": str(label or e.get("type") or e["kind"])[:60],
                "confidence": round(conf, 3)
                if isinstance(conf, (int, float)) and math.isfinite(conf)
                else None,
            }
        )
    if len(out) > EVENTS_MAX:
        out = sorted(out, key=lambda e: -(e["confidence"] or 0.0))[:EVENTS_MAX]
    return sorted(out, key=lambda e: (e["source"], e["start"]))


TARGET_MAX_S = 36000  # the sequence's target length: up to 10 hours


def _valid_target(v: Any) -> bool:
    """Whether ``v`` is a usable target length (a number above 0, at most TARGET_MAX_S)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v <= TARGET_MAX_S


def _target_of(review: Any) -> float | None:
    """The target length saved in a ``review.json`` document, else None."""
    v = review.get("target") if isinstance(review, dict) else None
    return float(v) if _valid_target(v) else None


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
    """``d`` holds UI metadata or an analysis of its own (not one a link points to), or the
    ``progress.json`` of a first analysis that stopped before writing one (#109; one of a run
    that finished has nothing to show or resume)."""
    if any((d / n).is_file() and _inside(d, n) for n in (f"{UI_DIR}/meta.json", "highlights.json")):
        return True
    rec = runstate.read(d) if _inside(d, runstate.FILE) else None
    return rec is not None and rec.get("status") != "done"


def run_state(p: Path, analyzed: bool) -> dict[str, Any] | None:
    """The project's last recorded run (``progress.json``, :func:`cutpoint.runstate.read`) as
    the page shows it: ``command``, ``status`` and whether Resume continues it (``resumable``:
    interrupted or stopped for memory; also failed while there is no analysis, ``analyzed``
    False, which only Resume can finish). None without a readable record."""
    rec = runstate.read(p) if _inside(p, runstate.FILE) else None
    command = (rec or {}).get("command")
    if rec is None or command not in runstate.COMMANDS:
        return None
    status = rec.get("status")
    resumable = status in runstate.RESUMABLE or (status == "failed" and not analyzed)
    return {"command": command, "status": status, "resumable": resumable}


CLIPS_UI_RESUME = re.compile(r"\.(clips_ui_\d{8}-\d{6}(?:-\d+)?)\.cutpoint-resume")


def prune_resume_folders(p: Path) -> None:
    """Remove the hidden resume folders (:mod:`cutpoint.clipresume`) of stopped UI exports that
    can never be resumed (#716). Every UI export writes a new ``clips_ui_<stamp>`` folder and
    Resume replays only the project's last run, so once a later export has superseded a stopped
    one its ``.clips_ui_<stamp>.cutpoint-resume`` only takes disk space.

    Only a folder with Cutpoint's naming and its ``resume.json`` goes. It stays when
    ``progress.json`` still names its clips folder in the recorded command line of a run that
    has not finished, when a run holds the project's lock or the clips folder's, and when it
    cannot be told apart from what Cutpoint wrote. Never raises."""
    try:
        if folders.lock_held(p / runstate.LOCK_NAME):
            return
        rec = runstate.read(p) if _inside(p, runstate.FILE) else None
        if rec is None and os.path.lexists(p / runstate.FILE):
            return  # a record we cannot read may point at a folder that is resumable
        named: set[str] = set()
        if rec is not None and rec.get("status") != "done":
            named = {Path(a).name for a in rec.get("argv") or [] if isinstance(a, str)}
        for f in sorted(p.iterdir()):
            m = CLIPS_UI_RESUME.fullmatch(f.name)
            if (
                m is None
                or f.is_symlink()
                or f.is_junction()
                or not f.is_dir()
                or not (f / folders.RESUME_FILE).is_file()
                or m.group(1) in named
                or folders.lock_held(p / m.group(1))
            ):
                continue
            try:
                folders._remove_leftover(f, "the clips of a stopped export that was not resumed")
            except OSError as err:
                logger.warning(
                    "Could not remove %s (%s); it is removed after a later export", f, err
                )
                continue
            logger.info("Removed %s: a later export replaced the stopped one", f)
    except OSError as err:
        logger.warning("Could not look for old resume folders in %s: %s", p, err)


def project_busy(p: Path, run: dict[str, Any] | None) -> bool:
    """A run is going on the project: another process holds its run lock
    (``.analysis.run.cutpoint-lock``, :mod:`cutpoint.runstate`; one whose process is gone does
    not count), or ``run`` (:func:`run_state`) reads ``running``."""
    if run is not None and run["status"] == "running":
        return True
    return folders.lock_held(p / runstate.LOCK_NAME)


def resume_check(p: Path) -> dict[str, Any]:
    """What ``cutpoint resume <p>`` would continue: the ``--json`` envelope of its ``--dry-run``,
    run from the projects folder.

    Raises:
        HTTPException: 409 :data:`BUSY_TEXT` while the run is still going, 404 without a run to
            resume, 409 with the CLI's message when it refuses the record, 500 or 504 when the
            check could not run.
    """
    try:
        done = subprocess.run(
            cutpoint_cmd("resume", str(p), "--dry-run", "--json"),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=job_env(),
            cwd=p.parent,
            timeout=RESUME_CHECK_S,
            check=False,
        )
    except subprocess.TimeoutExpired as err:
        raise bad("checking the run to resume took too long", 504) from err
    except OSError as err:
        raise bad(f"could not check the run to resume: {err}", 500) from err
    result = next((r for ln in reversed(done.stdout.splitlines()) if (r := parse_result(ln))), None)
    if result is None:
        raise bad(f"could not check the run to resume (exit code {done.returncode})", 500)
    err = result.get("error")
    if isinstance(err, dict):
        if err.get("code") == jsonout.FOLDER_BUSY:
            raise bad(BUSY_TEXT, 409)
        if err.get("code") == jsonout.NOT_FOUND:
            raise bad("this project has no run to resume", 404)
        raise bad(str(err.get("message") or "the run cannot be resumed"), 409)
    return result


def _values(argv: list[str], flag: str) -> list[str]:
    """The tokens a recorded command line gives ``flag`` (after it, up to the next option)."""
    if flag not in argv:
        return []
    rest = argv[argv.index(flag) + 1 :]
    return rest[: next((i for i, t in enumerate(rest) if t.startswith("-")), len(rest))]


def _skip_after(argv: list[str]) -> list[str]:
    """The steps a recorded command line gives ``--skip``."""
    steps = _values(argv, "--skip")
    return [s for s in progress.SKIPPABLE if s in steps]


def _recorded_picks(rec: dict[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    """What a recorded ``clips`` command line exports, as :func:`stitch_first` takes it: its
    ``--segments`` file (relative to the record's ``cwd``), else its ``--top``, ``--ids`` and
    ``--max-total``."""
    argv = [str(a) for a in rec.get("argv") or []]
    if seg := _values(argv, "--segments"):
        return Path(str(rec.get("cwd") or ".")) / seg[0], {}
    top, total = _values(argv, "--top"), _values(argv, "--max-total")
    picks: dict[str, Any] = {"top": None, "ids": _values(argv, "--ids") or None, "max_total": None}
    with contextlib.suppress(ValueError):
        picks["top"] = int(top[0]) if top else None
    with contextlib.suppress(ValueError):
        picks["max_total"] = float(total[0]) if total else None
    return None, picks


def resumed(p: Path, rec: dict[str, Any], outputs: dict[str, Any], stamp: str) -> None:
    """A resumed run finished: record what its own job would have (the run that stopped never
    got there). An analysis keeps its ``--skip`` (:func:`save_analysis_skip`); a clips folder
    of the project that ``clips`` or ``edit`` wrote (``outputs.clips_dir`` of the ``--json``
    envelope) becomes the latest export, with ``edit``'s final video."""
    command, argv = rec.get("command"), [str(a) for a in rec.get("argv") or []]
    if command in ("analyze", "edit") and (skip := _skip_after(argv)):
        _ui_folder(p)
        save_analysis_skip(p, skip)
    clips_dir = outputs.get("clips_dir")
    if command not in ("clips", "edit") or not isinstance(clips_dir, str):
        return
    name = Path(clips_dir).name
    if (
        not SAFE_NAME.fullmatch(name)
        or not _inside(p, name)
        or (p / name).resolve() != Path(clips_dir).resolve()
        or not (p / name / "manifest.json").is_file()
    ):
        return  # not a clips folder of this project (an --out elsewhere)
    final = outputs.get("final") if command == "edit" else None
    final_name = Path(final).name if isinstance(final, str) else None
    if (
        final_name is None
        or not FINAL_FILE.fullmatch(final_name)
        or not (p / final_name).is_file()
        or (p / final_name).resolve() != Path(str(final)).resolve()
    ):
        final_name = None
    _ui_folder(p)
    items = [e for e in load_exports(p) if e["dir"] != name]  # replaced, as edit's
    items.append(
        {"dir": name, "created": stamp, "vertical": "--vertical" in argv, "final": final_name}
    )
    cache.save_json(_own(p, f"{UI_DIR}/exports.json"), {"exports": items})


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
            # a timeline next to its video is listed with it, not as a final of its own
            if FINAL_FILE.fullmatch(e.name) and (
                e.name.endswith(".mp4") or not (p / f"{Path(e.name).stem}.mp4").is_file()
            ):
                with contextlib.suppress(OSError):
                    found.append((e.stat().st_mtime_ns, e.name))
    found.sort(reverse=True)
    names = dict.fromkeys([*first, *(n for _, n in found)])  # in order, no duplicates
    return [n for n in names if (p / n).is_file() and _inside(p, n)]


def output_names(p: Path) -> list[str]:
    """Files of the finals inside the project: each video or timeline-only export (``assemble
    --format fcpxml|otio``), then the timelines and subtitles of the same name."""
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
    # a timeline-only final is also its own timeline; a .fcpxml and .otio of one stem share their .srt
    return list(dict.fromkeys(out))


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
    f = insta360.parse_name(src) if src.is_file() else None
    if src.is_file() and f is None:
        files = [src]
    else:
        listed = sorted(p for p in (src.parent if f else src).iterdir() if safe_is_file(p))
        files = [
            p for p in listed if not f or (insta360.parse_name(p) or {}).get("num") == f["num"]
        ]
    videos = [p for p in files if p.suffix.lower() in VIDEO_EXTS]
    other = Counter(
        p.suffix.lower() or "no extension"
        for p in files
        if p.suffix.lower() not in VIDEO_EXTS and not insta360.NAME_RE.match(p.name)
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


def inputs_problem(srcs: list[Path], exclude: list[str], recursive: bool) -> str | None:
    """Why several inputs, or one with ``--exclude`` or ``--recursive``, cannot become a UI
    project, or None when they can: each is a video file or a folder, none holds Insta360 footage
    (a 360 project takes one folder or file and neither option, as ``360 analyze``), and
    ``analyze`` would find at least one video in them.

    Raises:
        OSError: If an input cannot be listed.
    """
    for src in srcs:
        if src.is_file() or src.is_dir():
            clips = inspect_source(src)[1]
            # a folder may hold its videos in sub-folders only (--recursive)
            problem = source_problem(src) if src.is_file() or clips else None
        else:
            clips, problem = {}, "source must be a video file or a folder containing videos"
        if problem:
            return f"{src}: {problem}" if len(srcs) > 1 else problem
        if clips:
            return (
                f"{src} holds 360 footage (Insta360), which makes a project of its own: choose "
                "just its folder or file, without sub-folders or exclude patterns."
            )
    if find_picked([str(s) for s in srcs], exclude, recursive):
        return None
    left = " that the exclude patterns leave" if exclude else ""
    deeper = "" if recursive else " (sub-folders are searched only with --recursive)"
    return f"no videos found{left} in the chosen sources{deeper}"


def _typed_path(x: object) -> object:
    """A source as typed or pasted: without surrounding blanks (a CRLF line's ``\\r`` too) and
    one pair of surrounding double quotes (Windows' Copy as path), then a leading ``~`` as
    :func:`os.path.expanduser` reads it on this OS (left as typed when it cannot)."""
    if isinstance(x, str):
        x = x.strip()
        if len(x) > 1 and x[0] == x[-1] == '"':
            x = x[1:-1]
        if x.startswith("~"):
            # a NUL or an unencodable ~user is the callers' error; unlike Path's, a missing
            # home folder leaves the path as typed
            with contextlib.suppress(ValueError):
                x = os.path.expanduser(x)  # noqa: PTH111
    return x


def _body_inputs(body: dict[str, Any]) -> list[str]:
    """``source`` (one path) or ``sources`` (1 to ``MAX_INPUTS``, blank lines left out): absolute
    paths, each once."""
    source, sources = body.get("source"), body.get("sources")
    if sources is None:
        sources = [source]
    elif source is not None:
        raise bad("send source or sources, not both")
    elif isinstance(sources, list):
        sources = [x for x in sources if not isinstance(x, str) or x.strip()]
    if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_INPUTS:
        raise bad(f"sources must list 1-{MAX_INPUTS} paths")
    sources = [_typed_path(x) for x in sources]
    for x in sources:
        if not (isinstance(x, str) and "\0" not in x and Path(x).is_absolute()):
            name = f"{x}: " if len(sources) > 1 and isinstance(x, str) else ""
            raise bad(f"{name}source must be an absolute path")
    if len({os.path.normcase(os.path.abspath(x)) for x in sources}) < len(sources):  # noqa: PTH100
        raise bad("each source can be chosen once")
    return sources


def _exclude_globs(body: dict[str, Any]) -> list[str]:
    """``exclude``: up to ``MAX_EXCLUDES`` ``analyze --exclude`` patterns, one line each."""
    v = body.get("exclude", [])
    if (
        not isinstance(v, list)
        or len(v) > MAX_EXCLUDES
        or not all(
            isinstance(g, str) and g.strip() and len(g) <= MAX_GLOB and not set(g) & set("\0\r\n")
            for g in v
        )
    ):
        raise bad(f"exclude must be up to {MAX_EXCLUDES} patterns of 1-{MAX_GLOB} characters")
    return v


def new_project(name: str, source: str, style: str, **extra: Any) -> dict[str, str]:
    """Create the project folder and its ``ui/meta.json`` (``extra``: e.g. ``kind`` and
    ``clips`` of a 360 project; ``inputs``, ``recursive`` and ``exclude`` of a project made from
    several inputs or with those ``analyze`` options)."""
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
    """Styles for standard video, the telemetry-driven action-cam styles included (#633).

    A learned style that fails to load is left out with a warning, so one bad file does not
    break the UI.
    """
    names = []
    for n in style_names():
        try:
            get_style(n)
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("skipping learned style %s: %s", learned_styles().get(n, n), e)
            continue
        names.append(n)
    return names


def _gaming_style(name: object) -> bool:
    """``name`` is a ``gaming`` style (or one based on it): config.json's ``color`` never grades
    it, so the page's Color starts at Off for it."""
    try:
        return get_style(str(name)).base == "gaming" if name else False
    except (ValueError, TypeError, AttributeError):
        return False


def suggest_style(
    inputs: list[str], exclude: list[str] = (), recursive: bool = False
) -> str | None:
    """``action-cam`` when an input carries camera telemetry, else None (#633).

    Only the container's stream list is probed (``analyze.has_telemetry``: a GoPro GPMF track or an
    Insta360 IMU trailer), never the telemetry itself, within ``TELEMETRY_PROBE_TIMEOUT`` s per file
    and stopping at the first hit. The files are those ``analyze`` would find (``find_picked``:
    ``recursive``, ``exclude`` and ``source_remap`` apply), the first ``MAX_TELEMETRY_PROBES`` of
    them. Unreadable inputs count as none. Usable 360 footage gets no suggestion: the server picks
    its 360 style (``pano.default_style``) whatever is chosen.
    """
    from cutpoint.analyze import has_telemetry  # noqa: PLC0415

    try:
        for x in inputs:
            src = Path(x)
            if (src.is_file() or src.is_dir()) and _usable(inspect_source(src)[1]):
                return None
        files = find_picked(inputs, exclude, recursive)[:MAX_TELEMETRY_PROBES]
    except OSError:
        return None
    return "action-cam" if any(has_telemetry(f, TELEMETRY_PROBE_TIMEOUT) for f in files) else None


def _ski_style(name: str) -> bool:
    """``name`` is a telemetry-driven action-cam style (``kind`` action-cam): kept as given for 360 sources."""
    try:
        return get_style(name).kind == ACTION_CAM
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


def _color_state(raw: Any) -> dict[str, Any] | None:
    """The page's Color as ``{mode, strength}`` (strength 0-1) from an assistant request, or
    None when it did not send one (or sent nonsense)."""
    if not isinstance(raw, dict) or raw.get("mode") not in COLOR_MODES:
        return None
    strength = raw.get("strength")
    ok = isinstance(strength, int | float) and not isinstance(strength, bool)
    if not ok or not 0 <= strength <= 1:
        return None
    return {"mode": raw["mode"], "strength": float(strength)}


def _color_after(target: dict[str, Any], now: dict[str, Any] | None) -> dict[str, Any]:
    """The Color a proposal's color op gives: its ``target`` ``{mode, strength}``, where a
    strength it leaves out (None) stays what the page has ``now``."""
    strength = target.get("strength")
    if strength is None:
        strength = now["strength"] if now else 1.0
    return {"mode": target["mode"], "strength": strength}


def _color_args(body: dict[str, Any]) -> list[str]:
    """``--color`` and ``--color-strength`` with the page's choice. Not sent: no flag, so the CLI
    applies ``color`` in config.json itself (never to gaming footage, which an explicit flag
    would grade). The page has no LUT, but a ``lut`` with a grade is refused as the CLI refuses
    ``--color`` with ``--lut``."""
    color = body.get("color")
    strength = body.get("color_strength")
    if color is not None and (not isinstance(color, str) or color not in COLOR_MODES):
        raise bad(f"color must be one of {', '.join(COLOR_MODES)}")
    if body.get("lut") is not None and (color not in (None, "off") or strength is not None):
        raise bad("color cannot be combined with lut; use one")
    if strength is None:
        return [] if color is None else ["--color", color]
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not 0 <= strength <= 1
    ):
        raise bad("color_strength must be a number from 0 to 1")
    if color not in ("auto", "underwater"):
        raise bad("color_strength needs color auto or underwater")
    return ["--color", color, "--color-strength", f"{float(strength):g}"]


def _analysis_inputs(meta: dict[str, Any], hl: dict[str, Any] | None) -> tuple[Any, Any]:
    """The inputs and style a project's analysis runs on: the inputs the user picked, else (made
    on the command line) the recorded inputs; either may be missing."""
    inputs = picked_inputs(meta) or (hl or {}).get("settings", {}).get("inputs")
    return inputs, meta.get("style") or (hl or {}).get("style")


def _subscription_roles(args: list[str], style: object) -> bool:
    """Whether an analyze/edit job with ``args`` on ``style`` has a role on a subscription
    provider (#832): Vision (rating on) or the AI Chat model of the ``dialogue`` step (open-list
    styles; ``auto`` may pick one; not when ``--skip`` names it, ``analysis.skip`` when there is
    none). An explicit ``--ollama-url`` or ``--ollama-model`` wins over both."""
    if any(a.startswith(("--ollama-url", "--ollama-model")) for a in args):
        return False
    try:
        rating = "--vision" in args or ("--no-vision" not in args and get_config().vision)
        if rating and ai_providers.is_subscription(vision_assistant_settings().provider):
            return True
        skipped = _values(args, "--skip") if "--skip" in args else analysis_skip()
        if "dialogue" in skipped or not ai_providers.is_subscription(assistant_settings().provider):
            return False
        return style == AUTO or get_style(str(style)).open_list
    except (ValueError, TypeError, AttributeError):  # an invalid config.json or style: the job says
        return False


def _vision_args(st: dict[str, Any]) -> list[str]:
    """Vision options for an analysis: the Settings value now (explicit, so the job and the page
    agree even if config.json changes before the CLI reads it) and the recorded Ollama options."""
    args = ["--vision" if get_config().vision else "--no-vision"]
    vision_top = st.get("vision_top")
    if isinstance(vision_top, int) and not isinstance(vision_top, bool) and vision_top > 0:
        args.append(f"--vision-top={vision_top}")
    for key, opt in (("ollama_url", "--ollama-url"), ("ollama_model", "--ollama-model")):
        if isinstance(st.get(key), str) and st[key]:
            args.append(f"{opt}={st[key]}")
    return args


# The Analyze options (#483), each one analyze flag: the request's key and its flag.
ANALYZE_TEXT = {"model": "--model", "language": "--language"}
ANALYZE_TEXT_FORMS = {  # what the page may send for them: a model name, a language code
    "model": (
        re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"),
        "a whisper model name such as small or large-v3",
    ),
    "language": (re.compile(r"[A-Za-z]{2,3}"), "a language code such as en"),
}
ANALYZE_FLAGS = {"no_scenes": "--no-scenes", "recover": "--recover"}
ANALYZE_KEYS = ("model", "language", "max_candidates", "no_scenes", "skip", "must_include_crash",
                "recover")  # fmt: skip
ANALYSIS_FILE = f"{UI_DIR}/analysis.json"  # the --skip of the project's analysis (not recorded)


def _default_model(st: dict[str, Any]) -> str:
    """The whisper model ``analyze`` picks without ``--model`` on the device the analysis ran on
    (Transcriber's default)."""
    return "medium" if st.get("whisper_device") == "cuda" else "small"


def _style(name: object) -> Style | None:
    try:
        return get_style(str(name)) if name else None
    except (ValueError, TypeError, AttributeError):
        return None


def _style_crash(style: object) -> float | None:
    """The ``must_include_crash`` of an action-cam ``style`` (what ``--must-include-crash``
    overrides), else None: the flag only changes action-cam analysis."""
    s = _style(style)
    return s.must_include_crash if s is not None and s.kind == ACTION_CAM else None


def _no_scenes(st: dict[str, Any], style: object) -> bool:
    """The analysis ran with ``--no-scenes``: it recorded no scene detection although its style
    detects scenes (action-cam never does, so it records none without the flag)."""
    s = _style(style)
    return st.get("scenes") is False and (s is None or s.use_scenes)


def _saved_skip(p: Path, hl: dict[str, Any] | None) -> list[str]:
    """The ``--skip`` steps the UI ran the project's current analysis with; none when that file
    belongs to an earlier analysis (``created`` differs), as after a command-line re-run."""
    saved = _load(p, ANALYSIS_FILE)
    if not isinstance(saved, dict) or not hl or saved.get("created") != hl.get("created"):
        return []
    skip = saved.get("skip")
    return [s for s in progress.SKIPPABLE if isinstance(skip, list) and s in skip]


def save_analysis_skip(p: Path, skip: list[str]) -> None:
    """Remember the ``--skip`` of the analysis just written, with its ``created`` stamp."""
    hl = _highlights(p) or {}
    cache.save_json(_own(p, ANALYSIS_FILE), {"skip": skip, "created": hl.get("created")})


def analyze_options(p: Path, meta: dict[str, Any], hl: dict[str, Any] | None) -> dict[str, Any]:
    """The Analyze options the project's analysis used, as the page sends them: what
    ``highlights.json`` recorded (a model only when not the default) and the saved ``--skip``."""
    st = (hl or {}).get("settings") or {}
    model = st.get("whisper_model")
    mc = st.get("max_candidates")
    crash = st.get("must_include_crash")
    style = _analysis_inputs(meta, hl)[1]
    default_crash = _style_crash(style)
    return {
        "model": model
        if isinstance(model, str) and model and model != _default_model(st)
        else None,
        "language": st["language"]
        if isinstance(st.get("language"), str) and st["language"]
        else None,
        "max_candidates": mc
        if isinstance(mc, int) and not isinstance(mc, bool) and mc > 0
        else None,
        "no_scenes": _no_scenes(st, style),
        "skip": _saved_skip(p, hl),
        "must_include_crash": float(crash)
        if default_crash is not None
        and isinstance(crash, (int, float))
        and not isinstance(crash, bool)
        and 0 <= crash <= 1
        and crash != default_crash
        else None,
        "recover": st.get("recover") is True,
    }


def _analyze_args(body: dict[str, Any], recorded: dict[str, Any]) -> tuple[list[str], list[str]]:
    """The Analyze options' flags and the ``--skip`` steps: each from the request, else (a key it
    does not send) what the project's analysis used, so Analyze again repeats it."""
    opts = {k: body.get(k, v) for k, v in recorded.items()}
    args = []
    for key, flag in ANALYZE_TEXT.items():
        v = opts[key]
        if v is not None and not isinstance(v, str):
            raise bad(f"{key} must be text")
        v = (v or "").strip()
        if not v:
            continue
        pattern, what = ANALYZE_TEXT_FORMS[key]
        if not pattern.fullmatch(v):
            raise bad(f"{key} must be {what}")
        args.append(f"{flag}={v}")
    mc = opts["max_candidates"]
    if mc is not None:
        if isinstance(mc, bool) or not isinstance(mc, int) or mc < 1:
            raise bad("max_candidates must be a whole number above 0")
        args.append(f"--max-candidates={mc}")
    crash = opts["must_include_crash"]
    if crash is not None:
        if (
            isinstance(crash, bool)
            or not isinstance(crash, (int, float))
            or not 0 <= crash <= 1  # NaN fails this too
        ):
            raise bad("must_include_crash must be a number from 0 to 1")
        args.append(f"--must-include-crash={float(crash):g}")
    for key, flag in ANALYZE_FLAGS.items():
        if _flag(opts, key):
            args.append(flag)
    skip = opts["skip"]
    if skip is None:
        skip = []
    if not isinstance(skip, list) or any(
        not isinstance(s, str) or s not in progress.SKIPPABLE for s in skip
    ):
        raise bad(f"skip must be a list of {', '.join(progress.SKIPPABLE)}")
    return args, [s for s in progress.SKIPPABLE if s in skip]


def preset_problem(meta: dict[str, Any], hl: dict[str, Any] | None) -> str | None:
    """Why ``cutpoint edit`` cannot remake this project's analysis (None when it can)."""
    if meta.get("kind") == "360":
        return "Presets are not available for 360 projects (cutpoint edit has no 360 mode)."
    st = (hl or {}).get("settings") or {}
    exclude, recursive = picked_filters(meta)
    if st.get("exclude") or st.get("recursive") is True or exclude or recursive:
        return (
            "Presets are not available here: this project's analysis used --exclude or "
            "--recursive, which cutpoint edit does not take."
        )
    # edit has no --model or --no-scenes: it would transcribe again with the default model or
    # detect scenes, and rewrite highlights.json. The default model is Transcriber's for the
    # device the analysis ran on.
    model = st.get("whisper_model")
    if model is not None and model != ("medium" if st.get("whisper_device") == "cuda" else "small"):
        return (
            f"Presets are not available here: this project was transcribed with the {model} "
            "model (analyze --model), which cutpoint edit does not take."
        )
    inputs, style = _analysis_inputs(meta, hl)
    if _no_scenes(st, style):
        return (
            "Presets are not available here: this project's analysis used --no-scenes, which "
            "cutpoint edit does not take."
        )
    if st.get("recover") is True:
        return (
            "Presets are not available here: this project's analysis used --recover, which "
            "cutpoint edit does not take."
        )
    # nor --language or --max-candidates (recorded as None when not given)
    for key, opt in (("language", "--language"), ("max_candidates", "--max-candidates")):
        if st.get(key) is not None:
            return (
                f"Presets are not available here: this project's analysis used {opt}, which "
                "cutpoint edit does not take."
            )
    # nor --must-include-crash: edit would analyze with the style's own confidence
    crash, default_crash = st.get("must_include_crash"), _style_crash(style)
    if default_crash is not None and crash is not None and crash != default_crash:
        return (
            "Presets are not available here: this project's analysis used --must-include-crash, "
            "which cutpoint edit does not take."
        )
    if not inputs or not style:
        return "Presets need a recorded source to analyze."
    return None


def preset_taken(p: Path) -> list[str]:
    """The folders a preset writes (``clips``, ``clips_vertical``) that hold files already, so
    ``edit`` needs ``--force``."""
    taken = []
    for name in ("clips", "clips_vertical"):
        with contextlib.suppress(OSError):
            if (p / name).is_dir() and any((p / name).iterdir()):
                taken.append(name)
    return taken


def _seconds(body: dict[str, Any], key: str, hi: float, *, positive: bool = False) -> float | None:
    """``body[key]`` as seconds, 0 (or above 0 with ``positive``) to ``hi``; None when not sent."""
    v = body.get(key)
    if v is None:
        return None
    if (
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        or v > hi
        or v < 0
        or (positive and v == 0)
    ):
        raise bad(f"{key} must be {'above 0' if positive else '0'}-{hi:g} seconds")
    return float(v)


def _opt(flag: str, v: float | None) -> list[str]:
    return [] if v is None else [flag, f"{v:g}"]


def _picture_args(body: dict[str, Any]) -> list[str]:
    """``clips --crop`` and ``--size`` as the page sent them; not sent: the CLI's defaults (the
    project's per-source crop, else none; the largest source's size, capped at 1080p)."""
    args = []
    crop, size = body.get("crop"), body.get("size")
    if crop is not None:
        if not isinstance(crop, str) or not (crop in ("auto", "none") or CROP_BOX.fullmatch(crop)):
            raise bad("crop must be 'auto', 'none' or W:H:X:Y in pixels")
        args += ["--crop", crop]
    if size is not None:
        if not isinstance(size, str) or not CLIP_SIZE.fullmatch(size):
            raise bad("size must be WxH in even pixels, such as 1920x1080")
        args += ["--size", size]
    return args


def _speech_args(body: dict[str, Any]) -> list[str]:
    """``--speech-max-extend`` / ``--speech-max-lead`` when the page sent them."""
    return _opt("--speech-max-extend", _seconds(body, "speech_max_extend", MAX_TIMING_S)) + _opt(
        "--speech-max-lead", _seconds(body, "speech_max_lead", MAX_TIMING_S)
    )


def _auto_picks(body: dict[str, Any], hl: dict[str, Any] | None) -> list[str]:
    """``clips`` selection flags for automatic picks: ``--top`` or ``--ids``, ``--max-total`` and
    ``--order``, each only when sent (else the CLI's defaults: the style's count and order)."""
    cands = (hl or {}).get("candidates")
    if not isinstance(cands, list) or not cands:
        raise bad("analyze the footage before exporting automatic picks")
    top, ids, order = body.get("top"), body.get("ids"), body.get("order")
    if top is not None and ids is not None:
        raise bad("give top or ids, not both")
    args: list[str] = []
    if top is not None:
        if isinstance(top, bool) or not isinstance(top, int) or not 1 <= top <= MAX_SEGMENTS:
            raise bad(f"top must be a whole number from 1 to {MAX_SEGMENTS}")
        args += ["--top", str(top)]
    if ids is not None:
        known = {c.get("id") for c in cands if isinstance(c, dict)}
        if (
            not isinstance(ids, list)
            or not 0 < len(ids) <= MAX_SEGMENTS
            or len(set(map(repr, ids))) != len(ids)
            or not all(isinstance(i, str) and SAFE_NAME.fullmatch(i) and i in known for i in ids)
        ):
            raise bad("ids must be different candidate ids of this analysis, such as h001")
        args += ["--ids", *ids]
    args += _opt("--max-total", _seconds(body, "max_total", MAX_TOTAL_S, positive=True))
    if order is not None:
        if order not in CLIP_ORDERS:
            raise bad(f"order must be one of {', '.join(CLIP_ORDERS)}")
        args += ["--order", order]
    return args


PREVIEW_NOTE = "exporting from the 360 preview (not stitched)"


def stitch_first(
    job: Job, project: Path, seg_file: Path | None, picks: dict[str, Any]
) -> list[str] | None:
    """``cutpoint 360 stitch`` for the recordings a 360 export needs that are not stitched yet,
    or None (``pano.auto_stitch`` off, no working SDK or NVIDIA GPU, all stitched, too little
    free space).

    ``seg_file``: the export's segments file; else ``picks`` (``top``, ``ids``, ``max_total``)
    choose the moments as ``clips`` will. Says why in the job log; never raises for a reason
    to skip (``Jobs.start`` turns anything else into a warning too).
    """
    try:
        on = pano_project.auto_stitch()
    except ValueError as err:
        job.note("WARNING", f"{err}; {PREVIEW_NOTE}")
        return None
    if not on:
        job.note("INFO", f"pano.auto_stitch is off: {PREVIEW_NOTE}")
        return None
    sdk = pano_stitch.sdk_status()
    if not sdk.get("runs"):
        why = "does not run" if sdk.get("found") else "was not found"
        job.note("WARNING", f"the Insta360 SDK {why} (see cutpoint 360 status): {PREVIEW_NOTE}")
        return None
    from cutpoint import capabilities  # noqa: PLC0415

    if not capabilities.nvidia_present():  # the SDK needs one: the stitch would likely fail
        job.note("WARNING", "no NVIDIA GPU; exporting the 360 preview")
        return None
    if seg_file is not None:
        sources = [m["source"] for m in cache.read_json(seg_file)]
    else:
        from cutpoint.planning import select_moments  # noqa: PLC0415

        sources = [m["source"] for m in select_moments(project, **picks)[0]]
    recorded = pano_project.load(project)
    chosen = {c.clip_id: c for s in sources if (c := pano_project.clip_for(recorded, s))}
    missing = [c for _, c in sorted(chosen.items()) if not pano_project.sdk_output(c).is_file()]
    if not missing:
        if chosen:
            job.note("INFO", f"the {len(chosen)} recording(s) of this export are stitched")
        return None
    need = pano_stitch.estimate_gb(missing)
    out_dir = pano_project.stitched_dir()
    free = pano_stitch.free_gb(out_dir)
    room = "unknown" if free is None else f"{free:.0f} GB"
    job.note(
        "INFO",
        f"stitching {len(missing)} recording(s) with the Insta360 SDK first: about "
        f"{need:.0f} GB in {out_dir} ({room} free)",
    )
    if free is None:
        job.note("WARNING", f"cannot read the free space in {out_dir}: {PREVIEW_NOTE}")
        return None
    if free - need < pano_stitch.MIN_FREE_GB:
        job.note(
            "WARNING",
            f"not enough free space in {out_dir} (needs about {need:.0f} GB plus "
            f"{pano_stitch.MIN_FREE_GB:.0f} GB to spare): {PREVIEW_NOTE}",
        )
        return None
    if seg_file is not None:  # the stitch skips the recordings already stitched
        return cutpoint_cmd("360", "stitch", str(project), "--segments", str(seg_file))
    return cutpoint_cmd("360", "stitch", str(project), "--clips", *(c.clip_id for c in missing))


def create_app(port: int) -> FastAPI:
    token = secrets.token_urlsafe(32)
    jobs, media, picker, sources = Jobs(), Media(), Picker(), Sources()
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
    app.state.picker, app.state.sources = picker, sources
    app.state.previews = previews
    app.state.proxies = proxies  # routes look it up here, so a test can swap it

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

    def director_url(pid: str, sid: str, name: str) -> str:
        return url(pid, "sources", sid).replace("?", f"/director/{name}?", 1)

    def pano_views(
        pid: str, p: Path, listing: list[dict[str, Any]], only: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """The camera path flags and director preview of each candidate (by CLI id) of a 360
        clip, a ready preview with its url (under the source the page plays it as)."""
        previews.prune(p)
        views = previews.listing(p)
        if only is not None:
            views = {only: views[only]} if only in views else {}
        hl = _highlights(p) or {}
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
        """Change the projects folder, the vision default and/or the assistant's provider, model
        and base URL in ``config.json``. Existing projects stay where they are. A changed
        assistant is checked at once (reachable, key accepted, tool calls or JSON output):
        ``assistant_check`` is why it cannot answer, or None. Nothing (a key least of all) is
        sent to a hosted provider the user has not agreed to in AI Chat (#504): its check waits
        for the consent, and ``assistant_note`` says so."""
        if not {"projects_dir", "vision", "assistant", "vision_assistant"} & set(body):
            raise bad("send projects_dir, vision, assistant and/or vision_assistant")
        updates: dict[str, Any] = {}
        if "vision" in body:
            updates["vision"] = _flag(body, "vision")
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
        if assistant is not None:
            section = data.get("assistant")
            updates["assistant"] = {**(section if isinstance(section, dict) else {}), **assistant}
        if vision_ast is not None:
            raw_v = data.get("vision_assistant")
            v_section = dict(raw_v) if isinstance(raw_v, dict) else {}
            updates["vision_assistant"] = {
                **v_section,
                **vision_ast,
            }
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
        out = settings_view({**data, **updates})
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
        data = config_data()
        raw_section = data.get(section_name)
        section = dict(raw_section) if isinstance(raw_section, dict) else {}
        given = list(_tuple_of_str(section.get("consent")))
        if want not in given:
            try:
                set_config_values({section_name: {**section, "consent": [*given, want]}})
            except ValueError as err:
                raise bad(str(err), 409) from err
            except OSError as err:
                raise bad(f"could not save the consent: {err.strerror or err}", 500) from err
        fresh_data = config_data()
        fresh_section = fresh_data.get(section_name)
        fresh_dict = fresh_section if isinstance(fresh_section, dict) else {}
        consent_list = _tuple_of_str(fresh_dict.get("consent"))
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
        name, style = body.get("name"), body.get("style", "gaming")
        if msg := renamed_error(style):  # ski: an error naming action-cam
            raise bad(msg)
        if not isinstance(name, str) or not name.strip() or len(name) > 100:
            raise bad("name must be 1-100 characters")
        inputs = _body_inputs(body)
        recursive, exclude = _flag(body, "recursive"), _exclude_globs(body)
        ski = isinstance(style, str) and _ski_style(style)
        if style not in ui_styles() and not ski:
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
        highlights = moments_view(hl, _load(p, "events.json"))
        last_run = run_state(p, hl is not None)
        if highlights is not None and meta.get("kind") == "360":
            # #517: flags of each moment's cached camera path and its director preview (null:
            # not a moment of a 360 clip of the project, the page plays the raw preview)
            views = pano_views(pid, p, listing)
            for c in highlights["candidates"]:
                c["pano"] = views.get(str(c.get("cli_id")))
        return {
            "id": pid,
            "name": meta.get("name") or pid,
            "style": meta.get("style") or (hl or {}).get("style") or "",
            # the page's Color starts at Off for gaming footage, as config.json's color does
            "gaming": _gaming_style(meta.get("style") or (hl or {}).get("style")),
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
            # --must-include-crash default, null when the style is not action-cam (no such option)
            "analyze_options": analyze_options(p, meta, hl),
            "must_include_crash_default": _style_crash(
                meta.get("style") or (hl or {}).get("style")
            ),
            "preset_taken": preset_taken(p),
            "final_current": exp is None or (bool(outputs) and outputs[0] == exp.get("final")),
            # The transition each join of that final came out with, for the page's "Assemble
            # again" note; null when not known, as for a One-Click Edit's final.
            "final_joins": (exp or {}).get("final_joins"),
            "sources": found,
            # The picked folder could not be listed (gone, unreadable): a field, so Settings
            # (source remap), Reveal and Delete still work.
            "source_error": source_error,
            # The last recorded run (progress.json, #109): Resume when it stopped, null without one
            "run": last_run,
            # a run holds the project now (this server's own job too): the page says to wait
            "busy": project_busy(p, last_run),
        }

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
            # the sequence's target length is saved by its own route: Save keeps it
            # a short lock must not drop the target; missing or still locked: load_json reads None
            with contextlib.suppress(OSError):
                cache.retry_locked(file.read_bytes, file, "read")
            saved = cache.load_json(file)
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
        before: Callable[[Job], list[str] | None] | None = None  # a step ahead of ``args``
        on_result: Callable[[Job], None] | None = None  # after on_success, with the envelope
        written: list[Path] = []  # files ``prepare`` wrote, removed if the job does not start
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
            inputs, style = _analysis_inputs(meta, hl)
            if not inputs or not style:
                raise bad("project has no recorded source to analyze")
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
            opts, skip = _analyze_args(body, analyze_options(p, meta, hl))
            args += opts + _vision_args(st)
            if skip:  # last: its steps follow it. Saved (an older file never matches the new
                # analysis's created stamp, so without --skip there is nothing to save)
                args += ["--skip", *skip]
                _own(p, ANALYSIS_FILE)  # refuse now, not after the analysis

                def prepare() -> None:
                    _ui_folder(p)  # a command-line project has no ui folder yet

                def on_success() -> None:
                    save_analysis_skip(p, skip)

        elif action == "edit":
            # A one-click preset: analyze (cached), clips and assemble as `edit --kind` says, in
            # the project's own style so its analysis and moments stay as they are.
            kind = body.get("kind")
            if not isinstance(kind, str) or kind not in KINDS:
                raise bad(f"kind must be one of {', '.join(KINDS)}")
            if why := preset_problem(meta, hl):
                raise bad(why)
            inputs, style = _analysis_inputs(meta, hl)
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
                "edit", *map(str, inputs), "--kind", kind, "--style", str(style), "--out", str(p)
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
            # edit analyzes again: with the Analyze options' --skip, or it would run those steps
            if skip := _saved_skip(p, hl):
                args += ["--skip", *skip]
                _own(p, ANALYSIS_FILE)

            def prepare() -> None:
                _ui_folder(p)  # a command-line project has no ui folder for exports.json yet

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
            final_joins = [
                {"a": j["a"], "b": j["b"], "transition": j["transition"] or transition}
                for j in clip_joins(p / exp_dir, saved)
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
            raise bad("action must be 'analyze', 'edit', 'clips', 'extend', 'assemble' or 'resume'")

        def same_folder() -> None:
            # Under the jobs lock, like the settings change: a folder changed since ``p`` was
            # resolved would run the job against the old one.
            if projects_dir().resolve() != p.resolve().parent:
                raise bad("the projects folder changed; reload the page", 409)
            if prepare:
                prepare()  # only once the job may start: a refused one leaves no file behind

        # --json: the envelope's error.code tells a busy project apart (Jobs._busy) and a resume
        # names what it wrote. The jobs run in the projects folder, not where the server started.
        if action in ("analyze", "edit") and meta.get("kind") != "360":
            needs_proxy = _subscription_roles(args, style)
        elif action == "resume" and run["command"] in ("analyze", "edit"):
            argv = [str(a) for a in rec.get("argv") or []]
            needs_proxy = _subscription_roles(argv, next(iter(_values(argv, "--style")), None))
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
            ).view()
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
        p, entry = find_source(pid, sid)
        hl = _highlights(p) or {}
        mine = {
            str(c.get("id")) for c in hl.get("candidates") or [] if c.get("source") == entry["path"]
        }
        listed = {
            view["preview"].get("name")
            for cid, view in previews.listing(p).items()
            if cid in mine and view["preview"] is not None and view["preview"]["state"] == "ready"
        }
        file = previews.file(p, name) if name in listed else None
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

    # -- assistant (#346) --------------------------------------------------------------
    # A chat turn only proposes sequence edits: the page applies one to its working sequence and
    # reports it here for the history. Nothing below writes review.json; GET routes only read.
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
        (``consent``), the saved chat and the turn still running, if any."""
        p = project_dir(pid)
        running = [t.view() for t in turns.of(str(p)) if t.status == "running"]
        provider = assistant_provider()
        return {
            "provider": ai_providers.describe(provider),
            "consent": consent_view(provider),
            "frames": frames_state(p, provider),
            "turns": ai_store.turns(p),
            "running": running,
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
        data = config_data()
        raw = data.get("assistant")
        section = dict(raw) if isinstance(raw, dict) else {}
        given = section.get("frames_consent", {})
        if not isinstance(given, dict):
            raise bad("assistant.frames_consent in config.json must be an object", 409)
        where = ai_watch.project_key(p)
        keys_now = list(ai_watch.frames_consent(data, p))
        if want not in keys_now:
            frames = {**given, where: [*keys_now, want]}
            try:
                set_config_values({"assistant": {**section, "frames_consent": frames}})
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
        project = ai_session.Project(
            moments,
            segments,
            durations,
            transcript,
            key,
            windows,
            edge_planner(p),
            frame_index=ai_lookfor.indexed(list(durations)),  # no frames, no look_for
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
