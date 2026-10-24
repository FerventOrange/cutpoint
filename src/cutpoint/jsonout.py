"""The ``--json`` result envelope: exactly one JSON object on stdout per command run.

::

    {"ok": bool, "command": "analyze", "schema": 1, "project": "<abs path>" | null,
     "outputs": {...}, "warnings": [{"code", "message", "files"}],
     "error": null | {"code", "message"}}

:func:`run` wraps a command: while it is active, ``sys.stdout`` and file descriptor 1 point
at stderr, so stray prints, tqdm and child processes (ffmpeg, ...) that inherit stdout write
to stderr; on exit the envelope is written to the real stdout. Commands fill it with
:func:`set_project`, :func:`output` and :func:`warn`, which do nothing without ``--json``.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePath
from typing import Any, TextIO

SCHEMA = 1

# Error codes (``error.code``).
BAD_ARGS = "bad_args"  # argparse rejected the command line (exit 2); SystemExit("msg") (exit 1)
NOT_FOUND = "not_found"  # FileNotFoundError: an input, project or file is missing
FFMPEG_MISSING = "ffmpeg_missing"  # FFmpegNotFoundError: ffmpeg / ffprobe not found
MISSING_EXTRA = "missing_extra"  # an optional extra is not installed
OUTPUT_NOT_EMPTY = "output_not_empty"  # FileExistsError: an output is in the way (--force)
INTERRUPTED = "interrupted"  # Ctrl+C, Ctrl+Break, SIGTERM (exit 130)
ERROR = "error"  # anything else


class Result:
    """What one command reports; :meth:`envelope` is the JSON object."""

    def __init__(self, command: str) -> None:
        self.command = command
        self.project: str | None = None
        self.outputs: dict[str, Any] = {}
        self.warnings: list[dict[str, Any]] = []
        self.error: dict[str, str] | None = None

    def envelope(self) -> dict[str, Any]:
        return {
            "ok": self.error is None,
            "command": self.command,
            "schema": SCHEMA,
            "project": self.project,
            "outputs": self.outputs,
            "warnings": self.warnings,
            "error": self.error,
        }


_active: Result | None = None
_lock = threading.Lock()


def _value(value: Any) -> Any:
    """``value`` as JSON: paths absolute strings, tuples and sets lists, inside lists and dicts
    too. Anything else that is not JSON is written as ``str(value)`` (see :func:`_write`)."""
    if isinstance(value, Path):
        return str(value.absolute())
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _value(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_value(v) for v in value]
    return value


def active() -> bool:
    """True while a ``--json`` command runs."""
    return _active is not None


def set_project(path: Path) -> None:
    """Record the project folder (``project``)."""
    with _lock:
        if _active is not None:
            _active.project = str(path.absolute())


def output(name: str, value: Any) -> None:
    """Record ``outputs[name]``; a Path becomes an absolute path string."""
    with _lock:
        if _active is not None:
            _active.outputs[name] = _value(value)


def warn(code: str, message: str, files: Iterable[Path | str] = ()) -> None:
    """Append a warning, e.g. ``warn("kept_clips", "...", [path, ...])``."""
    with _lock:
        if _active is not None:
            _active.warnings.append(
                {"code": code, "message": message, "files": [_value(f) for f in files]}
            )


def fail(code: str, message: str) -> None:
    """Record the error (the first one wins); the envelope then has ``ok`` false."""
    with _lock:
        if _active is not None and _active.error is None:
            _active.error = {"code": code, "message": message}


def error_code(err: BaseException) -> str:
    """The ``error.code`` for an exception the CLI turns into an exit code."""
    from cutpoint.ffmpeg_utils import FFmpegNotFoundError  # noqa: PLC0415

    if isinstance(err, KeyboardInterrupt):
        return INTERRUPTED
    if isinstance(err, FFmpegNotFoundError):
        return FFMPEG_MISSING
    if isinstance(err, FileExistsError):
        return OUTPUT_NOT_EMPTY
    if isinstance(err, FileNotFoundError):
        return NOT_FOUND
    return ERROR


def _write(stream: TextIO | None, result: Result) -> None:
    if stream is None:  # no stdout at all (closed, or pythonw)
        return
    # ASCII only, so no console encoding can fail on it.
    stream.write(json.dumps(result.envelope(), default=str) + "\n")
    stream.flush()


def emit_error(command: str, code: str, message: str) -> None:
    """Write an ``ok: false`` envelope for a command that did not start (e.g. bad arguments)."""
    result = Result(command)
    result.error = {"code": code, "message": message}
    _write(sys.stdout, result)


def _stderr_usable() -> bool:
    if sys.stderr is None:
        return False
    try:
        os.fstat(2)
    except OSError:
        return False
    return True


@contextlib.contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """Point ``sys.stdout`` and fd 1 at stderr until the block ends.

    The fd is redirected too, so child processes that inherit stdout write to stderr (on
    Windows ``dup2`` also sets the process's standard output handle, which subprocess
    passes on). Without a usable stderr (closed, or pythonw) both go to the null device;
    without a usable fd 1 only ``sys.stdout`` is swapped.
    """
    real = sys.stdout
    to_stderr = _stderr_usable()
    sink = sys.stderr if to_stderr else Path(os.devnull).open("w", encoding="utf-8")  # noqa: SIM115
    saved = None
    with contextlib.suppress(OSError, ValueError, AttributeError):
        if real is not None:
            real.flush()
    with contextlib.suppress(OSError):
        target = 2 if to_stderr else os.open(os.devnull, os.O_WRONLY)
        try:
            saved = os.dup(1)
            os.dup2(target, 1)
        except OSError:
            if saved is not None:
                os.close(saved)
                saved = None
        finally:
            if not to_stderr:
                os.close(target)
    sys.stdout = sink
    try:
        yield
    finally:
        try:
            with contextlib.suppress(OSError, ValueError, AttributeError):
                sink.flush()
                if real is not None:
                    real.flush()  # writes made to it meanwhile still go to stderr
        finally:
            try:
                if saved is not None:
                    os.dup2(saved, 1)
                    os.close(saved)
            finally:
                sys.stdout = real
                if not to_stderr:
                    sink.close()


@contextlib.contextmanager
def run(command: str) -> Iterator[Result]:
    """Run a ``--json`` command: guard stdout, then write the envelope to the real stdout.

    Errors are recorded with :func:`fail` by the caller; an exception that leaves the block
    is recorded here (code from :func:`error_code`) and re-raised after the envelope is
    written.
    """
    global _active
    result = Result(command)
    real = sys.stdout
    with _lock:
        _active = result
    try:
        with _stdout_to_stderr():
            yield result
    except SystemExit as err:
        if isinstance(err.code, str):  # sys.exit("message"): exit 1, the message on stderr
            fail(BAD_ARGS, err.code)
        elif err.code not in (0, None):
            fail(ERROR, f"exit code {err.code}")
        raise
    except BaseException as err:
        fail(error_code(err), str(err) or type(err).__name__)
        raise
    finally:
        with _lock:
            _active = None
        _write(real, result)
