"""Run state of the long project commands (#109): ``<project>/progress.json`` and the project
run lock.

``analyze``, ``edit``, ``events``, ``crashes``, ``profile``, ``360 analyze`` and ``clips`` run inside
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

``status`` is ``running``, then ``done`` (exit 0), ``interrupted`` (130), ``low_memory`` (1:
the memory guard stopped it, :mod:`cutpoint.memguard`) or ``failed`` (1); the last three with
``error`` ``{"code", "message"}`` as the ``--json`` envelope's. The step a stopped run was in
reads ``failed`` (the step states are those of the ``PROGRESS`` lines); ``status`` says why.
:func:`read` reports a ``running`` record whose process is gone as ``interrupted``.

Only a project folder (``highlights.json`` or ``pano.json`` in it; for ``analyze``, ``edit`` and
``360 analyze`` also ``progress.json`` alone, :meth:`Recorder.is_project`) is recorded from the
start; a folder that is no project gets no record and, but from ``analyze`` and
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

``cutpoint resume <project>`` (:func:`resumable`, :func:`resuming`) replays a record's ``argv``
from its ``cwd``; the replayed command records as usual. A run that replaces an interrupted
record of another command or other options warns ``resume_args_changed``. ``edit`` keeps the
stamp of the clips' ``manifest.json`` under ``steps.clips.manifest`` (:func:`clips_written`), so
its resume skips clips it finished (:func:`clips_kept`).
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
from cutpoint.errors import LowMemoryError
from cutpoint.progress import Run
from cutpoint.utils import command_line, file_stamp

logger = logging.getLogger(__name__)

VERSION = 1
FILE = "progress.json"
LOCK_NAME = "analysis.run"  # swap_lock(<project>/analysis.run): .analysis.run.cutpoint-lock
# clips: so cutpoint resume replays an interrupted export (cutpoint.clipresume)
COMMANDS = frozenset({"analyze", "edit", "events", "crashes", "profile", "360 analyze", "clips"})
AFTER_TRANSCRIPT = frozenset({"analyze", "edit"})  # progress.json once a source is transcribed
STATUSES = ("running", "done", "interrupted", "failed", "low_memory")
PCT_INTERVAL_S = 5.0
# Parsed arguments that are no options of the run (callables are left out too); --batch-size
# changes no result, so a re-run with it (to use less memory) continues the record.
_NOT_OPTIONS = frozenset({"json_command", "json_result", "verbose", "batch_size"})

RESUMABLE = frozenset({"interrupted", "low_memory"})  # a record that resume continues
_ACTIVE: Recorder | None = None
_RESUMING: dict[str, Any] | None = None  # the record cutpoint resume replays


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
        self.previous: dict[str, Any] | None = None  # the record this run replaces
        self.resuming = _RESUMING is not None  # replayed by cutpoint resume
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
        """The folder is a project: ``highlights.json`` or ``pano.json``; ``progress.json``
        alone only for ``analyze``, ``edit`` and ``360 analyze``, which resume it, so that
        ``events``, ``crashes`` or ``profile`` stop with ``not_found`` and leave the record of
        an interrupted first ``analyze`` as it was."""
        names = ["highlights.json", "pano.json"]
        if self.command in AFTER_TRANSCRIPT or self.command == "360 analyze":
            names.append(FILE)
        return any((self.project / n).is_file() for n in names)

    def begin(self) -> None:
        """Lock an existing project and write the record (sources from ``highlights.json``).
        ``analyze`` and ``edit`` also lock an existing folder that is no project yet (they make
        it one), and write the record once a source is transcribed."""
        if not self.is_project():
            if self.command in AFTER_TRANSCRIPT and self.project.is_dir():
                self.lock()
            return  # project_ready / project_made lock a new project
        self.lock()
        self.previous = read(self.project)
        prev = self.previous or {}
        if (
            self.command == "clips"
            and prev.get("status") in RESUMABLE
            and prev.get("command") not in (None, "clips")
        ):
            # an export never replaces the record of an interrupted analysis (or edit)
            logger.info(
                "progress.json keeps the interrupted '%s' run; cutpoint resume continues that one",
                prev.get("command"),
            )
            return
        hl = cache.load_json(self.project / "highlights.json")
        srcs = hl.get("sources") if isinstance(hl, dict) else None
        if self.same_run() and self.previous and isinstance(self.previous.get("sources"), list):
            srcs = self.previous["sources"]  # resumed: the sources the run had, until replaced
        elif isinstance(srcs, list):
            srcs = [s["path"] for s in srcs if isinstance(s, dict)]
        if isinstance(srcs, list):
            self.data["sources"] = [str(s) for s in srcs]
        self.arm()

    def same_run(self) -> bool:
        """The record this run replaces is of the same command and options."""
        prev = self.previous
        return bool(prev) and (prev.get("command"), prev.get("options_key")) == (
            self.command,
            self.data["options_key"],
        )

    def _warn_replaced(self) -> None:
        """INFO and ``resume_args_changed``: this run replaces an interrupted record of
        another command or other options, which ``cutpoint resume`` can no longer continue."""
        prev = self.previous
        if not prev or prev.get("status") not in RESUMABLE or self.same_run():
            return
        this = (
            "this run with other options"
            if prev.get("command") == self.command
            else f"this '{self.command}' run"
        )
        message = (
            f"{self.path} recorded an interrupted '{prev.get('command')}' run; {this} "
            f"replaces it, so cutpoint resume continues this one"
        )
        logger.info("%s", message)
        jsonout.warn("resume_args_changed", message, [self.path])

    def lock(self) -> None:
        if not self.locked:
            self._stack.enter_context(folders.swap_lock(self.project / LOCK_NAME, self.project))
            self.locked = True

    def unlock(self) -> None:
        self._stack.close()
        self.locked = False

    def arm(self) -> None:
        with self._lock:
            if not self.armed:
                self.armed = True
                self._warn_replaced()
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
        """Write the final status (once): done, interrupted, low_memory or failed."""
        if self.ended:
            return
        self.ended = True
        if err is None:
            status, code, error = "done", 0, None
        elif isinstance(err, KeyboardInterrupt):
            status, code = "interrupted", 130
            error = {"code": jsonout.INTERRUPTED, "message": "interrupted"}  # as the envelope
        elif isinstance(err, LowMemoryError):
            status, code, error = "low_memory", 1, _error(err)
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


def resume_project() -> Path | None:
    """The project ``cutpoint resume`` continues if this run stops now: the active record's,
    once ``progress.json`` is written (None before, or outside :func:`record`)."""
    rec = _ACTIVE
    return rec.project if rec is not None and rec.armed else None


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
    from now on (no-op outside :func:`record`). On a re-run of ``analyze``, ``edit`` or
    ``360 analyze``, written from the start, the sources of ``highlights.json`` are replaced by
    this run's, once."""
    rec = _active(project)
    if rec is None or not rec.locked:
        return
    names = [str(s) for s in sources]
    if not rec.armed:
        rec.data["sources"] = names
        rec.arm()
        return
    if rec.command in AFTER_TRANSCRIPT or rec.command == "360 analyze":
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


@contextmanager
def resuming(rec: dict[str, Any]) -> Iterator[None]:
    """``cutpoint resume`` replays ``rec`` in the block: the :class:`Recorder` of the replayed
    command knows it resumes (:func:`clips_kept`)."""
    global _RESUMING
    _RESUMING = rec
    try:
        yield
    finally:
        _RESUMING = None


def force_hint() -> str:
    """How to replace the clips of a non-empty clips folder: ``--force``; in a run ``cutpoint
    resume`` replays (it takes no ``--force``, #1003), the recorded command with ``--force``
    and the folder to run it in."""
    if _RESUMING is None:
        return "use --force to replace its clips"
    argv: list[str] = list(_RESUMING["argv"])
    cut = argv.index("--") if "--" in argv else len(argv)
    again = command_line(["cutpoint", *argv[:cut], "--force", *argv[cut:]])
    return (
        f"cutpoint resume takes no --force; to replace its clips, run {again} in {_RESUMING['cwd']}"
    )


def resumable(project: Path) -> dict[str, Any]:
    """The record ``cutpoint resume`` replays: ``progress.json`` of ``project``.

    Raises:
        FileNotFoundError: If there is none (or it cannot be read).
        ValueError: If it names no command line of :data:`COMMANDS`.
    """
    rec = read(project)
    if rec is None:
        raise FileNotFoundError(f"No {FILE} in {project}: no run of this project to resume")
    argv = rec.get("argv")
    if (
        rec.get("command") not in COMMANDS
        or not isinstance(argv, list)
        or not all(isinstance(a, str) for a in argv)
        or not isinstance(rec.get("cwd"), str)
    ):
        raise ValueError(f"{project / FILE} names no command to resume")
    return rec


def missing_sources(rec: dict[str, Any]) -> list[str]:
    """The recorded ``sources`` that are not there now (relative ones from the record's
    ``cwd``)."""
    base = Path(rec.get("cwd") or ".")
    sources = rec.get("sources") if isinstance(rec.get("sources"), list) else []
    return [str(s) for s in sources if not (base / str(s)).exists()]


def step_counts(rec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{step: {state, done, total}}`` of the record (``outputs.resume.done_before``)."""
    steps = rec.get("steps") if isinstance(rec.get("steps"), dict) else {}
    return {
        name: {k: s.get(k) for k in ("state", "done", "total")}
        for name, s in steps.items()
        if isinstance(s, dict)
    }


def _stamp(path: Path) -> list[int] | None:
    stamp = file_stamp(path)
    return list(stamp) if stamp is not None else None


def clips_written(project: Path, clips_dir: Path) -> None:
    """``edit`` exported (or kept) the clips of ``clips_dir``: record the stamp of their
    ``manifest.json`` (no-op outside :func:`record` or before ``progress.json``)."""
    rec = _active(project)
    stamp = _stamp(clips_dir / "manifest.json")
    if rec is None or stamp is None:
        return
    with rec._lock:
        entry = rec.data["steps"].setdefault(
            "clips", {"state": None, "done": None, "total": None, "pct": None}
        )
        entry["manifest"] = {"path": str(clips_dir / "manifest.json"), "stamp": stamp}
        rec._write()


def clips_kept(project: Path, clips_dir: Path) -> bool:
    """``cutpoint resume`` of an ``edit`` whose clips step finished: the replaced record has
    the clips step done and the stamp of ``clips_dir``'s ``manifest.json`` as it is now (the
    same command line made it, and nothing changed it since), so the clips are kept."""
    rec = _active(project)
    if rec is None or not rec.resuming or not rec.same_run() or rec.previous is None:
        return False
    steps = rec.previous.get("steps")
    entry = steps.get("clips") if isinstance(steps, dict) else None
    if not isinstance(entry, dict) or entry.get("state") != "done":
        return False
    manifest = entry.get("manifest")
    if not isinstance(manifest, dict):
        return False
    path = clips_dir / "manifest.json"
    return path_key(str(manifest.get("path"))) == path_key(str(path)) and manifest.get(
        "stamp"
    ) == _stamp(path)
