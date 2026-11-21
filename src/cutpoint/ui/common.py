"""Constants, the logger and the child-process helpers of ``cutpoint ui``."""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from cutpoint import procs
from cutpoint.assistant import keys as ai_keys
from cutpoint.projectfiles import UI_DIR

logger = logging.getLogger("cutpoint.ui")

WEB_DIR = Path(__file__).parent.parent / "web"
# UI_DIR (projectfiles): per-project state: meta.json, review.json, exports.json, proxy/, ...
META_FILE = f"{UI_DIR}/meta.json"  # the project's name, sources and chosen style
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
MAX_LOG_LINES = 300
MAX_JOBS = 50
MAX_KEPT_NAMES = 100  # names of kept files the delete plan lists
MAX_EXTEND_S = 30.0
MAX_BODY_BYTES = 16 * 1024 * 1024  # of one request (JSON only: the page uploads no files)
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
ENVELOPE_VERSION = 2  # part of a cached envelope's name: bump it when the envelope changes
ENVELOPE_TIMEOUT_S = 1800.0  # a decode running longer is killed (a multi-hour source on a NAS)
ENVELOPE_STALE_S = (
    10.0  # a queued decode nobody asked for in this long (2 x the page's 5 s) is dropped
)
EVENT_KINDS = (
    "audio",
    "ocr",
    "telemetry",
    "text",
)  # events.json kinds the rail marks (not the VLM blobs)
EVENTS_MAX = 5000  # events sent per project
SOURCES_CACHE_MAX = 32  # projects whose source lists are kept; the least recently used go first
SOURCES_CACHE_S = 60.0  # rebuild a kept source list at least this often (what stats miss)
# media routes reuse a project's folder check and source list this long (#1014): a browser opens
# a fragmented MP4 with hundreds of Range requests
MEDIA_LOOKUP_S = 2.0
FRAGMENT_SCAN_BOXES = 16  # top-level MP4 boxes read to tell a fragmented file (#1014)
MAX_INPUTS = 50  # files and folders one New project takes (analyze's positional inputs)
MAX_EXCLUDES = 20  # analyze --exclude patterns one New project takes
MAX_GLOB = 200  # characters of one --exclude pattern
# What the page says when a job stops with folder_busy (#109): another run holds the project's run
# lock or the clips folder's; never the CLI's message, which tells the user to delete the lock.
BUSY_TEXT = "Analysis is running on this project; wait for it to finish"
RESUME_CHECK_S = 60.0  # seconds ``cutpoint resume --dry-run`` may take (it starts the CLI)


def bad(detail: str, status: int = 400) -> HTTPException:
    return HTTPException(status, detail)


def finite_number(v: Any) -> float | None:
    """``v`` as a float when it is a finite int or float (not a bool), else None (an int too
    large for a float too)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except OverflowError:
        return None
    return f if math.isfinite(f) else None


def number_in(v: Any, lo: float, hi: float = math.inf, *, integer: bool = False) -> float | None:
    """``v`` when it is a number from ``lo`` to ``hi`` (finite; an int, not a bool, with
    ``integer``), else None; as a float unless ``integer``. The one check of a number a request
    or a saved file gives; each caller words its own error."""
    if integer:
        ok = isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi
        return v if ok else None
    f = finite_number(v)
    return f if f is not None and lo <= f <= hi else None


# --------------------------------------------------------------------------- processes
def job_env() -> dict[str, str]:
    """The environment of a child ``cutpoint`` command (``spawn``, ``resume_check``)."""
    # CUTPOINT_PROGRESS: analyze and edit write PROGRESS step lines (``parse_progress``); the
    # other commands and ffmpeg ignore it.
    # Never the assistant's API key variables (ai_keys.child_env): no job uses them. Other
    # secret-named variables stay (HF_TOKEN: the job's model downloads).
    return {
        **ai_keys.child_env(keep_secret_names=True),
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


STYLE_FOR = "style_for"  # meta.json: the re-rank (_rerank_mark) the chosen style was picked after
ANALYSIS_FILE = f"{UI_DIR}/analysis.json"  # the --skip of the project's analysis (not recorded)
