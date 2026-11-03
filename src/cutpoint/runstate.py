"""Run state of the long project commands (#109): ``<project>/progress.json`` and the project
run lock.

``analyze``, ``edit``, ``events``, ``crashes``, ``profile`` and ``360 analyze`` run inside
:func:`record`, entered once per process by the CLI (``edit`` running ``analyze``, or
``360 analyze`` running ``analyze`` and the events, record and lock once). It holds the
project's run lock ``<project>/.analysis.run.cutpoint-lock`` (:func:`cutpoint.folders.swap_lock`,
so a second run on the project stops with :class:`~cutpoint.errors.FolderBusyError` and one left
by a dead process is taken over) and writes ``progress.json`` atomically
(:func:`cutpoint.cache.save_json`) at the start, on each step state change of the run's
:mod:`cutpoint.progress` steps, with their ``pct`` at most every :data:`PCT_INTERVAL_S`, and at
the end. Schema (``version`` 1)::

    {"version": 1, "generator": "cutpoint X.Y.Z", "command": "analyze",
     "argv": ["analyze", "clip.mp4", ...], "cwd": "/home/me", "options_key": "<params_hash of the options>",
     "status": "running", "pid": 123, "host": "pc", "started": "...", "updated": "...",
     "exit_code": null, "error": null, "sources": ["/videos/clip.mp4"],
     "steps": {"transcribe": {"state": "running", "done": 1, "total": 3, "pct": 40}}}

``status`` is ``running``, then ``done`` (exit 0), ``interrupted`` (130) or ``failed`` (1, with
``error`` ``{"code", "message"}`` as the ``--json`` envelope's); ``low_memory`` is reserved.
:func:`read` reports a ``running`` record whose process is gone as ``interrupted``.

Only a project folder (``highlights.json``, ``pano.json`` or ``progress.json`` in it) is recorded
from the start; a folder that is no project gets no record and, but from ``analyze`` and
``edit`` that make it one, no lock, so a command that stops on it with ``not_found`` leaves it
as it was. A first ``analyze`` or ``edit`` creates its
project folder only for the run and removes it again when the run fails or is cancelled while
it is empty (:func:`cutpoint.folders.output_dir`). So the lock of a project that is not there
yet is taken by :func:`project_ready`, once ``analyze`` made the folder, and given back on a
failure before ``progress.json`` exists; and a first ``analyze`` or ``edit`` writes
``progress.json`` only once its first source is transcribed (:func:`transcribed`). A re-run
writes it from the start, with the sources of ``highlights.json`` until the first transcript
names this run's. ``360 analyze --scan`` on a new project locks and records once the scan made
the folder (:func:`project_made`). It is never one of the held
outputs (``highlights.json`` and the rest of :data:`cutpoint.analyze.HELD_OUTPUTS` stay one
commit).
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from cutpoint import __version__, cache, folders, jsonout, procs
from cutpoint.config import path_key
from cutpoint.progress import Run

logger = logging.getLogger(__name__)

VERSION = 1
FILE = "progress.json"
LOCK_NAME = "analysis.run"  # swap_lock(<project>/analysis.run): .analysis.run.cutpoint-lock
COMMANDS = frozenset({"analyze", "edit", "events", "crashes", "profile", "360 analyze"})
AFTER_TRANSCRIPT = frozenset({"analyze", "edit"})  # progress.json once a source is transcribed
STATUSES = ("running", "done", "interrupted", "failed", "low_memory")
PCT_INTERVAL_S = 5.0
# Parsed arguments that are no options of the run (callables are left out too).
_NOT_OPTIONS = frozenset({"json_command", "json_result", "verbose"})

_ACTIVE: Recorder | None = None


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def options_key(options: dict[str, Any]) -> str:
    """:func:`cutpoint.cache.params_hash` of the parsed options (no callables, ``--json`` or
    ``-v``; paths as strings)."""
    return cache.params_hash(
        {k: v for k, v in options.items() if k not in _NOT_OPTIONS and not callable(v)}
    )


def _error(err: BaseException) -> dict[str, str]:
    """The ``--json`` envelope's ``error`` for ``err`` (as ``cli._run`` reports it)."""
    from cutpoint.extras import missing_module_message  # noqa: PLC0415

    message = str(err) or type(err).__name__
    extra = missing_module_message(err.name or "") if isinstance(err, ModuleNotFoundError) else None
    if extra is not None:
        return {"code": jsonout.MISSING_EXTRA, "message": extra}
    return {"code": jsonout.error_code(err), "message": message}


class Recorder:
    """``progress.json`` and the run lock of one command; see the module doc."""

    def __init__(
        self, command: str, argv: Sequence[str], project: Path, options: dict[str, Any]
    ) -> None:
        self.project = project
        self.path = project / FILE
        self.command = command
        self.locked = False
        self.armed = False  # progress.json is written
        self.ended = False
        self._stack = ExitStack()
        self._lock = threading.Lock()
        self._last = 0.0  # time of the last write
        self._warned = False
        self.data: dict[str, Any] = {
            "version": VERSION,
            "generator": f"cutpoint {__version__}",
            "command": command,
            "argv": [str(a) for a in argv],
            "cwd": str(Path.cwd()),
            "options_key": options_key(options),
            "status": "running",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started": _now(),
            "updated": None,
            "exit_code": None,
            "error": None,
            "sources": [],
            "steps": {},
        }

    def is_project(self) -> bool:
        """The folder is a project: ``highlights.json``, ``pano.json`` or ``progress.json``."""
        return any((self.project / n).is_file() for n in ("highlights.json", "pano.json", FILE))

    def begin(self) -> None:
        """Lock an existing project and write the record (sources from ``highlights.json``).
        ``analyze`` and ``edit`` also lock an existing folder that is no project yet (they make
        it one), and write the record once a source is transcribed."""
        if not self.is_project():
            if self.command in AFTER_TRANSCRIPT and self.project.is_dir():
                self.lock()
            return  # project_ready / project_made lock a new project
        self.lock()
        hl = cache.load_json(self.project / "highlights.json")
        srcs = hl.get("sources") if isinstance(hl, dict) else None
        if isinstance(srcs, list):
            self.data["sources"] = [s["path"] for s in srcs if isinstance(s, dict)]
        self.arm()

    def lock(self) -> None:
        if not self.locked:
            self._stack.enter_context(folders.swap_lock(self.project / LOCK_NAME))
            self.locked = True

    def unlock(self) -> None:
        self._stack.close()
        self.locked = False

    def arm(self) -> None:
        with self._lock:
            if not self.armed:
                self.armed = True
                self._write()

    def _write(self) -> None:
        """Write the record (the caller holds ``_lock``); a failure is a WARNING, once."""
        if not self.armed:
            return
        self.data["updated"] = _now()
        self._last = time.monotonic()
        try:
            cache.save_json(self.path, self.data)
        except OSError as err:
            log = logger.debug if self._warned else logger.warning
            self._warned = True
            log("Could not write %s: %s", self.path, err)

    def observe(
        self, step: str, state: str, pct: int | None, done: int | None, total: int | None
    ) -> None:
        """:attr:`cutpoint.progress.Run.observers` entry: a state change or new counts are
        written at once, a ``pct`` at most every :data:`PCT_INTERVAL_S`."""
        with self._lock:
            entry = self.data["steps"].setdefault(
                step, {"state": None, "done": None, "total": None, "pct": None}
            )
            now = entry["state"] != state or done is not None
            entry["state"] = state
            if pct is not None:
                entry["pct"] = pct
            if done is not None:
                entry["done"], entry["total"] = done, total
            if state == "done" and entry["total"] is not None:
                entry["done"] = entry["total"]
            if now or time.monotonic() - self._last >= PCT_INTERVAL_S:
                self._write()

    def end(self, err: BaseException | None) -> None:
        """Write the final status (once): done, interrupted or failed."""
        if self.ended:
            return
        self.ended = True
        if err is None:
            status, code, error = "done", 0, None
        elif isinstance(err, KeyboardInterrupt):
            status, code = "interrupted", 130
            error = {"code": jsonout.INTERRUPTED, "message": "interrupted"}  # as the envelope
        else:
            status, code, error = "failed", 1, _error(err)
        with self._lock:
            self.data.update(status=status, exit_code=code, error=error)
            if self.armed and self.project.is_dir():
                self._write()


@contextmanager
def record(
    command: str, argv: Sequence[str], project: Path, options: dict[str, Any] | None = None
) -> Iterator[Recorder | None]:
    """Record the run of ``command`` (one of :data:`COMMANDS`) on ``project`` and hold its
    run lock; a no-op inside another :func:`record` (once per process) or for other commands.

    Raises:
        FolderBusyError: If another run holds the project's lock.
    """
    global _ACTIVE
    if _ACTIVE is not None or command not in COMMANDS:
        yield None
        return
    rec = Recorder(command, argv, project, options or {})
    try:
        rec.begin()
        _ACTIVE = rec
        Run.observers.append(rec.observe)
        try:
            yield rec
        except BaseException as err:
            rec.end(err)
            raise
        rec.end(None)
    finally:
        if rec.observe in Run.observers:
            Run.observers.remove(rec.observe)
        _ACTIVE = None
        rec.unlock()


def _active(project: Path) -> Recorder | None:
    rec = _ACTIVE
    if rec is None or path_key(str(rec.project.absolute())) != path_key(str(project.absolute())):
        return None
    return rec


@contextmanager
def project_ready(project: Path) -> Iterator[None]:
    """``analyze``'s project folder exists: take the run lock of a project that did not
    exist when :func:`record` started. A failure before ``progress.json`` is written gives it
    back at once, so the run can remove the folder it created while it is empty.

    Raises:
        FolderBusyError: If another run holds the project's lock.
    """
    rec = _active(project)
    if rec is None or rec.locked:
        yield
        return
    rec.lock()
    try:
        yield
    except BaseException:
        if not rec.armed:
            rec.unlock()
        raise


def project_made(project: Path) -> None:
    """A command made its project folder (``360 analyze --scan`` on a new project): take the
    run lock and, but for ``analyze`` and ``edit``, write the record (no-op when locked
    already, or outside :func:`record`).

    Raises:
        FolderBusyError: If another run holds the project's lock.
    """
    rec = _active(project)
    if rec is None or rec.locked or not project.is_dir():
        return
    rec.lock()
    if rec.command not in AFTER_TRANSCRIPT:
        rec.arm()


def transcribed(project: Path, sources: Sequence[Path]) -> None:
    """A source of ``analyze`` is transcribed: record ``sources`` and write ``progress.json``
    from now on (no-op outside :func:`record`). On a re-run of ``analyze`` or ``edit``, written
    from the start, the sources of ``highlights.json`` are replaced by this run's, once."""
    rec = _active(project)
    if rec is None or not rec.locked:
        return
    names = [str(s) for s in sources]
    if not rec.armed:
        rec.data["sources"] = names
        rec.arm()
        return
    if rec.command in AFTER_TRANSCRIPT:
        with rec._lock:
            if rec.data["sources"] != names:
                rec.data["sources"] = names
                rec._write()


def read(project: Path) -> dict[str, Any] | None:
    """``progress.json`` of ``project`` (None if missing or unreadable); a ``running`` record
    whose process is gone (same host, the rule of the folder locks) reads ``interrupted``."""
    path = project / FILE
    data = cache.load_json(path)
    if not isinstance(data, dict):
        return None
    pid = data.get("pid")
    if data.get("status") != "running" or not isinstance(pid, int):
        return data
    if data.get("host") != socket.gethostname():  # a run on another computer: cannot tell
        return data
    if pid == os.getpid():
        alive = _active(project) is not None
    else:
        try:
            written = path.stat().st_mtime
        except OSError:
            written = None
        alive = procs.pid_alive(pid, written)
    return data if alive else {**data, "status": "interrupted"}
