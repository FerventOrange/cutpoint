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
from pathlib import Path
from typing import Any, TextIO

SCHEMA = 1

# Error codes (``error.code``).
BAD_ARGS = "bad_args"  # argparse rejected the command line (exit 2)
NOT_FOUND = "not_found"  # FileNotFoundError: an input, project or file is missing
FFMPEG_MISSING = "ffmpeg_missing"  # ffmpeg / ffprobe not found
MISSING_EXTRA = "missing_extra"  # an optional extra is not installed
OUTPUT_NOT_EMPTY = "output_not_empty"  # FileExistsError: an output is in the way (--force)
INTERRUPTED = "interrupted"  # Ctrl+C, Ctrl+Break, SIGTERM (exit 130)
ERROR = "error"  # anything else

_FFMPEG_MISSING_PREFIXES = ("ffmpeg not found", "ffprobe not found")  # find_tool's messages


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
    """``value`` as JSON: paths absolute strings, inside lists and dicts too."""
    if isinstance(value, Path):
        return str(value.absolute())
    if isinstance(value, dict):
        return {str(k): _value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
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
    from cutpoint.ffmpeg_utils import FFmpegError  # noqa: PLC0415

    if isinstance(err, KeyboardInterrupt):
        return INTERRUPTED
    if isinstance(err, FFmpegError) and str(err).startswith(_FFMPEG_MISSING_PREFIXES):
        return FFMPEG_MISSING
    if isinstance(err, FileExistsError):
        return OUTPUT_NOT_EMPTY
    if isinstance(err, FileNotFoundError):
        return NOT_FOUND
    return ERROR


def _write(stream: TextIO, result: Result) -> None:
    # ASCII only, so no console encoding can fail on it.
    stream.write(json.dumps(result.envelope()) + "\n")
    stream.flush()


def emit_error(command: str, code: str, message: str) -> None:
    """Write an ``ok: false`` envelope for a command that did not start (e.g. bad arguments)."""
    result = Result(command)
    result.error = {"code": code, "message": message}
    _write(sys.stdout, result)


@contextlib.contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """Point ``sys.stdout`` and fd 1 at stderr until the block ends.

    The fd is redirected too, so child processes that inherit stdout write to stderr (on
    Windows ``dup2`` also sets the process's standard output handle, which subprocess
    passes on). Without a usable fd 1 or 2 (e.g. pythonw) only ``sys.stdout`` is swapped.
    """
    real = sys.stdout
    saved = None
    with contextlib.suppress(OSError, ValueError):
        real.flush()
        saved = os.dup(1)
        try:
            os.dup2(2, 1)
        except OSError:
            os.close(saved)
            saved = None
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        with contextlib.suppress(OSError, ValueError):
            sys.stderr.flush()
            real.flush()  # writes made to it meanwhile still go to stderr
        if saved is not None:
            os.dup2(saved, 1)
            os.close(saved)
        sys.stdout = real


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
        if err.code not in (0, None):
            fail(ERROR, f"exit code {err.code}")
        raise
    except BaseException as err:
        fail(error_code(err), str(err) or type(err).__name__)
        raise
    finally:
        with _lock:
            _active = None
        _write(real, result)
