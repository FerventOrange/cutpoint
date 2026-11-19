"""The parts the user's libraries share: style files (:mod:`cutpoint.stylefiles`) and activity
profiles (:mod:`cutpoint.activity`) are both ``<name>.json`` files in a folder of the user's
Cutpoint home, with the same name rule, case-clash check, lookup and delete.

Every function returns the message or raises the error the caller's verb reports; the callers
keep their own wording for what a name cannot be (a built-in style or profile).
"""

from __future__ import annotations

import re
from pathlib import Path

NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
# File names Windows refuses whatever the extension
RESERVED_STEMS = frozenset(
    ["con", "prn", "aux", "nul"] + [f"{p}{i}" for p in ("com", "lpt") for i in range(1, 10)]
)


def format_problem(name: object, kind: str) -> str | None:
    """Why ``name`` is not a safe file name for a ``kind`` ("style", "profile"), or None."""
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        return (
            f"{name!r} is not a usable {kind} name: use up to 64 letters, digits, '-' and '_', "
            "starting with a letter or digit"
        )
    return None


def reserved_problem(name: str) -> str | None:
    """Why ``name`` cannot be a file name on Windows, or None (compared ignoring case)."""
    if name.lower() in RESERVED_STEMS:
        return f"{name!r} cannot be a file name on Windows; choose another name"
    return None


def case_clash_problem(files: dict[str, Path], name: str, kind: str) -> str | None:
    """The message for a file of ``files`` whose name differs from ``name`` only in case."""
    clash = next((s for s in files if s.lower() == name.lower() and s != name), None)
    if clash is None:
        return None
    return (
        f"a {kind} named {clash!r} exists; names that differ only in case clash on "
        "Windows and macOS"
    )


def find(files: dict[str, Path], name: str, missing: str) -> Path:
    """The file of ``name`` (exact name) in ``files``.

    Raises:
        FileNotFoundError: With the message ``missing`` if there is none.
    """
    path = files.get(name)
    if path is None:
        raise FileNotFoundError(missing)
    return path


def delete(files: dict[str, Path], name: str, missing: str) -> Path:
    """Delete the file of ``name`` in ``files`` and return it (:func:`find` raises if none)."""
    path = find(files, name, missing)
    path.unlink()
    return path


def looks_like_path(target: str) -> bool:
    """Whether a ``name or file`` target is written as a file: ``.json`` or a separator."""
    return target.endswith(".json") or "/" in target or "\\" in target


def file_target(target: str) -> Path:
    """The existing file ``target`` names.

    Raises:
        FileNotFoundError: If it is not a file.
    """
    path = Path(target)
    if not path.is_file():
        raise FileNotFoundError(f"{target}: file not found")
    return path
