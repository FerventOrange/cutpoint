"""Reading and writing a project's files for ``cutpoint ui``."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

from cutpoint import cache, runstate
from cutpoint.ui.common import ANALYSIS_FILE, UI_DIR, bad


def read_meta(p: Path) -> dict[str, Any]:
    meta = _load(p, f"{UI_DIR}/meta.json")
    return meta if isinstance(meta, dict) else {}


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


def _read_locked(file: Path) -> Any:
    """:func:`cache.read_json` that raises a lock's PermissionError as is, so
    :func:`cache.retry_locked` waits it out."""
    try:
        return cache.read_json(file)
    except ValueError as err:
        if isinstance(err.__cause__, PermissionError):
            raise err.__cause__ from None
        raise


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


def _values(argv: list[str], flag: str) -> list[str]:
    """The tokens a recorded command line gives ``flag`` (after it, up to the next option)."""
    if flag not in argv:
        return []
    rest = argv[argv.index(flag) + 1 :]
    return rest[: next((i for i, t in enumerate(rest) if t.startswith("-")), len(rest))]


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


def save_analysis_skip(p: Path, skip: list[str]) -> None:
    """Remember the ``--skip`` of the analysis just written, with its ``created`` stamp."""
    hl = _highlights(p) or {}
    cache.save_json(_own(p, ANALYSIS_FILE), {"skip": skip, "created": hl.get("created")})
