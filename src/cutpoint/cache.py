"""Per-source-file cache (audio, loudness, transcripts, scene cuts)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Collection, Iterator
from functools import partial
from pathlib import Path
from typing import Any

from cutpoint import paths


def cache_root() -> Path:
    """Cache directory (see :func:`cutpoint.paths.cache_dir`); created if missing."""
    root = paths.cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def source_dir(src: Path, offset: float = 0.0) -> Path:
    """Cache folder for one source, keyed by absolute path + size + mtime.

    The path in the key is canonicalised through ``source_remap`` (see
    :mod:`cutpoint.config`), so a remapped copy keys like its original.

    Args:
        src: Source video.
        offset: Start of a virtual source (``analyze --recover`` resumes a
            damaged file here); 0 for the file itself. Each offset gets its own
            folder, so the cached audio/transcript of a part starts at 0.
    """
    from cutpoint.config import canonical_source, remap_source  # noqa: PLC0415

    # Stat the file actually read (a remapped copy at the footage's new location), but key
    # it by its original path so an identical copy (same size + mtime) keeps hits.
    io_path = remap_source(src)
    st = io_path.stat()
    key = f"{canonical_source(io_path.resolve())}|{st.st_size}|{st.st_mtime_ns}"
    tag = ""
    if offset > 0:
        key += f"|offset={offset:.3f}"
        tag = f"@{offset:.0f}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    folder = cache_root() / f"{src.stem[:40]}{tag}-{digest}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


class HeldOutputs:
    """Files of a project that a run in progress holds back (see :func:`hold_outputs`):
    ``files`` maps each resolved path to the bytes it will get, or None to delete it."""

    def __init__(self, folder: Path, names: Collection[str]) -> None:
        self.folder = folder.resolve()
        self.names = frozenset(names)
        self.files: dict[Path, bytes | None] = {}

    def key(self, path: Path) -> Path | None:
        """The resolved ``path`` if this hold covers it, else None."""
        p = path.resolve()
        try:
            rel = p.relative_to(self.folder)
        except ValueError:
            return None
        return p if rel.parts and rel.parts[0] in self.names else None

    def commit(self) -> None:
        """Stop holding and apply the held changes in two phases: every file is staged
        first (a temp file in its folder, missing folders created, the mode of
        :func:`match_mode`); if any of that fails, the temp files and the new folders are
        removed and the project is left untouched. Only then are the deletions made and the
        temp files renamed onto their files (retried while locked, as :func:`save_bytes`)."""
        from cutpoint.ffmpeg_utils import retry_locked  # noqa: PLC0415 - it imports this module

        with contextlib.suppress(ValueError):
            _HOLDS.remove(self)
        files, self.files = self.files, {}
        staged: list[tuple[Path, Path]] = []  # (temp file, its file)
        try:
            with created_dirs() as made:  # new folders removed again on failure
                try:
                    for path, data in files.items():
                        if data is None:
                            continue
                        make_dirs(path.parent, made)
                        fd, name = tempfile.mkstemp(
                            dir=path.parent, prefix=path.name + ".", suffix=".tmp"
                        )
                        os.close(fd)
                        tmp = Path(name)
                        staged.append((tmp, path))
                        tmp.write_bytes(data)
                        match_mode(tmp, path)
                except BaseException:
                    for tmp, _ in staged:  # before created_dirs removes the empty folders
                        tmp.unlink(missing_ok=True)
                    staged = []
                    raise
            for path, data in files.items():
                if data is None:
                    path.unlink(missing_ok=True)
            for tmp, path in staged:
                # os.replace, not Path.replace: tests patch cache.os.replace
                retry_locked(partial(os.replace, tmp, path), path, "replace")
        finally:
            for tmp, _ in staged:  # those not renamed (a rename failed or was interrupted)
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)


# Holds in force (cutpoint analyze holds its result files until it completes).
_HOLDS: list[HeldOutputs] = []
_MISSING = object()  # not in a hold's files


def _held(path: Path) -> tuple[HeldOutputs, Path] | None:
    """The hold covering ``path`` and its key, or None (the file itself is used)."""
    for hold in _HOLDS:
        if (key := hold.key(Path(path))) is not None:
            return hold, key
    return None


def _held_bytes(path: Path) -> tuple[bool, bytes | None]:
    """``(True, bytes or None for deleted)`` if a hold has ``path`` in memory, else
    ``(False, None)``: read the file."""
    if not _HOLDS or (got := _held(path)) is None:
        return False, None
    hold, key = got
    data = hold.files.get(key, _MISSING)  # one lookup: commit() swaps the dict
    if data is _MISSING:
        return False, None
    return True, data  # type: ignore[return-value]


@contextlib.contextmanager
def hold_outputs(folder: Path, names: Collection[str]) -> Iterator[HeldOutputs]:
    """Hold back the writes and deletions of the files and folders of ``folder`` named in
    ``names`` in the block, so a failed or cancelled run leaves them as they were.

    :func:`save_bytes`, :func:`save_json` and :func:`remove` keep them in memory and
    :func:`load_json` and :func:`read_json` read them from there; nothing of them reaches the
    disk until :meth:`HeldOutputs.commit` (the yielded object). Leaving the block without a
    commit drops them.
    """
    hold = HeldOutputs(folder, names)
    _HOLDS.append(hold)
    try:
        yield hold
    finally:
        with contextlib.suppress(ValueError):
            _HOLDS.remove(hold)
        hold.files = {}


def is_held(path: Path) -> bool:
    """Whether a hold (:func:`hold_outputs`) keeps ``path``'s writes back."""
    return bool(_HOLDS) and _held(path) is not None


def remove(path: Path) -> None:
    """Delete ``path`` (if it exists), or mark it deleted in a hold (:func:`hold_outputs`)."""
    if _HOLDS and (got := _held(path)) is not None:
        hold, key = got
        hold.files[key] = None
        return
    path.unlink(missing_ok=True)


def load_json(path: Path) -> Any | None:
    """Load JSON from ``path`` or return None if missing/corrupt (a UTF-8 BOM is allowed)."""
    try:
        found, data = _held_bytes(path)
        if found:
            return None if data is None else json.loads(data.decode("utf-8-sig"))
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):  # ValueError: JSONDecodeError, UnicodeDecodeError
        return None


def read_json(path: Path, hint: str = "") -> Any:
    """Load JSON from ``path`` like :func:`load_json`, but raise instead of returning None.

    Args:
        path: JSON file (a UTF-8 BOM is allowed).
        hint: Appended to the unreadable-file message, e.g. how to rewrite the file.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If it exists but cannot be read or is not valid UTF-8 JSON; the
            message names the path and the reason.
    """
    try:
        found, data = _held_bytes(path)
        if found and data is None:
            raise FileNotFoundError(f"{path} not found")
        text = data.decode("utf-8-sig") if found else path.read_text(encoding="utf-8-sig")
        return json.loads(text)  # BOM: PowerShell 5.1 UTF8
    except FileNotFoundError:
        raise
    except OSError as err:  # e.g. permission denied, a directory
        reason = err.strerror or str(err)
        raise ValueError(f"{path} cannot be read: {reason}{hint}") from err
    except UnicodeDecodeError as err:  # e.g. UTF-16 from > / Out-File
        raise ValueError(
            f"{path} is not valid JSON: not UTF-8 text (save it as UTF-8){hint}"
        ) from err
    except ValueError as err:  # JSONDecodeError
        raise ValueError(f"{path} is not valid JSON: {err}{hint}") from err


REPLACE_ATTEMPTS = 5


def _umask() -> int:
    """The process umask.

    Linux: the ``Umask:`` line of ``/proc/self/status``. Elsewhere (or without
    it) by setting it and restoring it at once.
    """
    with contextlib.suppress(OSError, ValueError):
        for line in (
            Path("/proc/self/status").read_text(encoding="utf-8", errors="replace").splitlines()
        ):
            if line.startswith("Umask:"):
                return int(line.split()[1], 8)
    mask = os.umask(0)
    os.umask(mask)
    return mask


def match_mode(tmp: Path, target: Path) -> None:
    """Give a ``tempfile`` file (created 0600) the mode a plain write of
    ``target`` would get, before it is renamed onto ``target``.

    POSIX only: the existing target's mode if there is one, else ``0o666``
    minus the umask. No-op on Windows.
    """
    if os.name != "posix":
        return
    try:
        mode = target.stat().st_mode & 0o7777
    except OSError:
        mode = 0o666 & ~_umask()
    with contextlib.suppress(OSError):  # some SMB/CIFS/FUSE mounts reject chmod
        tmp.chmod(mode)


def save_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """Atomically write ``data`` as JSON, UTF-8, with the platform's line endings (see
    :func:`save_bytes`)."""
    text = json.dumps(data, indent=indent, ensure_ascii=False)
    # strings escape their newlines, so these are only the indent's line breaks
    save_bytes(path, text.replace("\n", os.linesep).encode("utf-8"))


def save_bytes(path: Path, data: bytes, *, parents: bool = False) -> None:
    """Atomically write ``data`` (unique temp file, then rename), with the mode a plain
    write would give (:func:`match_mode`); ``parents``: create the missing folders first.
    A file held back by :func:`hold_outputs` is kept in memory instead (no folder made).

    On Windows the rename fails with PermissionError while another process has
    ``path`` open; it is retried a few times with a short backoff
    (:func:`cutpoint.ffmpeg_utils.retry_locked`). The temp file is removed if the
    rename never succeeds.

    Raises:
        PermissionError: If ``path`` stays locked through all attempts (naming it and
            the program that may hold it, see :func:`~cutpoint.ffmpeg_utils.retry_locked`).
    """
    from cutpoint.ffmpeg_utils import retry_locked  # noqa: PLC0415 - it imports this module

    if _HOLDS and (got := _held(path)) is not None:
        hold, key = got
        hold.files[key] = bytes(data)
        return
    if parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    tmp = Path(name)
    try:
        tmp.write_bytes(data)
        match_mode(tmp, path)
        # os.replace, not Path.replace: tests patch cache.os.replace
        retry_locked(partial(os.replace, tmp, path), path, "replace")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def make_dirs(folder: Path, made: list[Path]) -> None:
    """Create ``folder`` and its missing parents (``mkdir(parents=True, exist_ok=True)``); the
    levels this call's own mkdir created are added to ``made``, deepest first (see
    :func:`created_dirs`).

    Raises:
        OSError: As ``mkdir``, e.g. FileExistsError or NotADirectoryError for a file in the way.
    """
    missing = []
    for level in (folder, *folder.parents):
        if os.path.lexists(level):
            break
        missing.append(level)
    # top down, recording only what this call's own mkdir created: a level that exists by
    # then (another process, or ``x/..``, which is an existing folder) is never recorded
    for level in reversed(missing):
        try:
            level.mkdir()
        except FileExistsError:
            continue
        except OSError:  # a file in the way or no permission: the mkdir below decides
            break
        made.insert(0, level)
    folder.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def created_dirs() -> Iterator[list[Path]]:
    """Yield a list for :func:`make_dirs` to record the folders a run creates; on failure (Ctrl+C
    included) those still empty are removed, deepest first, so a failed run leaves no empty
    folder behind. A folder that existed before, or that holds anything, is kept."""
    made: list[Path] = []
    try:
        yield made
    except BaseException:
        for folder in made:
            try:
                folder.rmdir()
            except OSError:  # not empty (something was written into it): keep it
                continue
        raise


def params_hash(params: dict[str, Any], length: int = 10) -> str:
    """Short stable hash of a parameter dict (cache keys and file names).

    ``json.dumps(params, sort_keys=True, default=str)`` hashed with SHA-1; for
    JSON-serialisable dicts these are the same bytes the modules hashed before
    this helper existed, so existing cache entries keep matching.
    """
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]
