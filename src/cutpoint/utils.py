"""Small shared helpers."""

from __future__ import annotations

import argparse
import errno
import fnmatch
import logging
import math
import re
import stat
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path, PurePath
from typing import Any

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm"}


def slugify(text: str, max_len: int | None = 40, default: str = "moment") -> str:
    """Lowercase ASCII slug safe for Windows filenames.

    Args:
        text: Any text; accents are folded to ASCII, other runs become ``-``.
        max_len: Truncate to this many characters (None = no limit).
        default: Returned when nothing is left.
    """
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if max_len is not None:
        slug = slug[:max_len].rstrip("-")
    return slug or default


def clock(t: float) -> str:
    """Format seconds as ``HH:MM:SS.s`` (same style as transcript.txt).

    Rounds to tenths first so carries propagate (``59.96`` -> ``00:01:00.0``);
    negative values get a leading ``-``.
    """
    tenths = round(abs(t) * 10)
    sign = "-" if t < 0 and tenths else ""
    h, rem = divmod(tenths, 36000)
    m, rem = divmod(rem, 600)
    return f"{sign}{h:02d}:{m:02d}:{rem // 10:02d}.{rem % 10}"


def parse_size(text: str) -> tuple[int, int]:
    """argparse type: ``WxH`` (e.g. ``1920x1080``) as ``(width, height)``."""
    try:
        w, h = (int(x) for x in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r}: expected WxH, e.g. 1920x1080") from exc
    return w, h


def _number[N: (int, float)](
    text: str, kind: Callable[[str], N], what: str, ok: Callable[[N], bool], rule: str
) -> N:
    """``kind(text)`` if it passes ``ok``; else ``ArgumentTypeError`` saying ``"<text>: expected
    <what>"`` (not a number) or ``"<text>: <rule>"`` (out of range)."""
    try:
        value = kind(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r}: expected {what}") from exc
    if not ok(value):  # also rejects NaN
        raise argparse.ArgumentTypeError(f"{text!r}: {rule}")
    return value


def positive_float(text: str) -> float:
    """argparse type: a number greater than 0."""
    return _number(text, float, "a number", lambda v: v > 0, "must be greater than 0")


def positive_int(text: str) -> int:
    """argparse type: a whole number greater than 0."""
    return _number(text, int, "a whole number", lambda v: v > 0, "must be greater than 0")


def unit_float(text: str) -> float:
    """argparse type: a number from 0 to 1."""
    return _number(text, float, "a number", lambda v: 0 <= v <= 1, "must be from 0 to 1")


def seconds_keys(
    item: Mapping[str, Any], keys: Iterable[str], where: str, *, allow_negative: bool = False
) -> dict[str, float]:
    """The ``keys`` of ``item`` that are set (present and not null), as seconds.

    Raises:
        ValueError: ``"<where>: <key> must be a number of seconds"`` for a value that is not a
            number or is NaN, and unless ``allow_negative``, ``"<where>: <key> must not be
            negative"`` for one below 0.
    """
    out: dict[str, float] = {}
    for key in keys:
        if item.get(key) is None:
            continue
        try:
            val = float(item[key])
        except (TypeError, ValueError):
            val = math.nan
        if math.isnan(val):
            raise ValueError(f"{where}: {key} must be a number of seconds")
        if not allow_negative and val < 0:
            raise ValueError(f"{where}: {key} must not be negative")
        out[key] = val
    return out


def file_stamp(path: Path) -> tuple[int, int, int] | None:
    """``(mtime_ns, size, inode)`` of ``path`` (one stat), or None when it cannot be read:
    changes when the file is rewritten or replaced."""
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size, st.st_ino


def input_name(target: Path, default: str) -> str:
    """Name a project after an input: a file's stem, a folder's full name (``.`` and a drive
    root resolved first), else ``default`` (a drive root has no name)."""
    p = target.resolve()
    return (p.stem if p.is_file() else p.name) or default


def natural_key(path: Path) -> list[object]:
    """Sort key so that ``2_x`` < ``10_x`` (case-insensitive)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def clip_files(folder: Path) -> list[Path]:
    """The .mp4 clips directly in ``folder`` in natural filename order (none if missing)."""
    # Dot files are temporaries (an interrupted export's ``.NNN_x.part.mp4``), not clips.
    return sorted((f for f in folder.glob("*.mp4") if not f.name.startswith(".")), key=natural_key)


def matches_glob(path: str | PurePath, patterns: Iterable[str]) -> bool:
    """True if ``path``'s file name or full path matches any glob.

    Case-insensitive; backslashes and forward slashes are treated alike, so
    ``"*/Captures/*"`` also matches Windows paths.
    """
    full = str(path).replace("\\", "/").rstrip("/").lower()
    name = full.rsplit("/", 1)[-1]
    for pat in patterns:
        pat = pat.lower().replace("\\", "/")
        if fnmatch.fnmatchcase(name, pat) or fnmatch.fnmatchcase(full, pat):
            return True
    return False


def find_videos(
    targets: Path | Sequence[Path], *, exclude: Sequence[str] = (), recursive: bool = False
) -> list[Path]:
    """Resolve files and/or folders into a de-duplicated list of video files.

    Args:
        targets: One path or several. Files are taken as given (any extension);
            folders contribute the videos directly inside them.
        exclude: Glob patterns matched against each file's name and its full
            path, e.g. ``"*DVR*"`` or ``"*/Captures/*"``. With ``recursive``, a
            sub-folder whose full path plus a trailing ``/`` matches is skipped
            without being listed (folder inputs themselves are never skipped);
            files inside it are dropped even if they link to files elsewhere.
        recursive: Also search sub-folders of folder inputs.

    Returns:
        Videos in input order; files inside one folder are naturally sorted.

    Raises:
        FileNotFoundError: If an input path does not exist or cannot be read
            (permission denied, stale mount, I/O error, name too long). Unreadable
            files and sub-folders inside an input folder are skipped with a WARNING.
    """
    if isinstance(targets, (str, Path)):
        targets = [Path(targets)]
    found: list[Path] = []
    for target in map(Path, targets):
        try:
            mode = _mode(target)
            is_file = mode is not None and stat.S_ISREG(mode)
            is_dir = mode is not None and stat.S_ISDIR(mode)
            entries = list(_walk(target, recursive, exclude)) if is_dir else []
        except OSError as err:
            raise FileNotFoundError(f"Input not readable: {target} ({err.strerror})") from err
        if is_file:
            batch = [target.resolve()]
        elif is_dir:
            batch = sorted(
                (p.resolve() for p in entries if p.suffix.lower() in VIDEO_EXTS and _is_file(p)),
                key=lambda p: [natural_key(Path(part)) for part in p.parts],
            )
        else:
            raise FileNotFoundError(f"Input not found: {target}")
        found.extend(p for p in batch if not matches_glob(p, exclude))
    return list(dict.fromkeys(found))


# Errors that ``Path.is_file()``/``is_dir()`` treat as "does not exist" (pathlib's list).
_MISSING_ERRNOS = (errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP)
_MISSING_WINERRORS = (21, 123, 1921)  # not ready, invalid name, can't resolve name


def _mode(p: Path, *, follow_symlinks: bool = True) -> int | None:
    """``st_mode`` of ``p``, or None where ``is_file()``/``is_dir()`` would say False.

    Unlike ``is_file()``, which on Python 3.13+ returns False for every OSError,
    this raises OSError for real failures (permission denied, stale mount, I/O
    error, name too long) on every version.
    """
    try:
        return p.stat(follow_symlinks=follow_symlinks).st_mode
    except OSError as err:
        if err.errno in _MISSING_ERRNOS or getattr(err, "winerror", None) in _MISSING_WINERRORS:
            return None
        raise
    except ValueError:  # embedded NUL / not encodable
        return None


def _walk(folder: Path, recursive: bool, exclude: Sequence[str] = ()) -> Iterator[Path]:
    """Entries of ``folder`` and, if ``recursive``, of its sub-folders (symlinked folders
    are not followed, as with ``rglob``). Iterative, so depth is not limited by the
    recursion limit; entries come in no particular order.

    A sub-folder whose resolved full path plus ``/`` matches an ``exclude`` glob
    (case-insensitive, e.g. ``"*/Captures/*"``) is neither yielded nor listed, and
    gives no WARNING even if unreadable. ``folder`` itself is never excluded.

    Raises:
        OSError: If ``folder`` itself cannot be listed. Entries and sub-folders below
            it that cannot be read are skipped with a WARNING.
    """
    pats = [pat.lower().replace("\\", "/") for pat in exclude]
    root = folder.resolve() if pats and recursive else folder
    stack = [folder]
    while stack:
        d = stack.pop()
        try:
            children = list(d.iterdir())
        except OSError as err:
            if d is folder:
                raise
            logger.warning("Skipping unreadable folder: %s (%s)", d, err.strerror)
            continue
        for p in children:
            try:
                mode = _mode(p, follow_symlinks=False) if recursive else None
            except OSError as err:
                logger.warning("Skipping unreadable path: %s (%s)", p, err.strerror)
                continue
            descend = mode is not None and stat.S_ISDIR(mode)
            if descend and pats:
                full = (root / p.relative_to(folder)).as_posix().lower() + "/"
                if any(fnmatch.fnmatchcase(full, pat) for pat in pats):
                    continue
            yield p
            if descend:
                stack.append(p)


def _is_file(p: Path) -> bool:
    """``p.is_file()`` (symlinks followed), but False with a WARNING when ``p`` cannot
    be checked (also on Python 3.13+, where ``is_file()`` hides such errors)."""
    try:
        mode = _mode(p)
    except OSError as err:
        logger.warning("Skipping unreadable file: %s (%s)", p, err.strerror)
        return False
    return mode is not None and stat.S_ISREG(mode)
