"""Per-source-file cache (audio, loudness, transcripts, scene cuts)."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import signal
import tempfile
import time
import zipfile
import zlib
from collections.abc import Callable, Collection, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import paths, state

logger = logging.getLogger(__name__)


def cache_root() -> Path:
    """Cache directory (see :func:`cutpoint.paths.cache_dir`); created if missing."""
    root = paths.cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def source_dir(src: Path, offset: float = 0.0, *, create: bool = True) -> Path:
    """Cache folder for one source, keyed by absolute path + size + mtime.

    The path in the key is canonicalised through ``source_remap`` (see
    :mod:`cutpoint.config`), so a remapped copy keys like its original. An existing
    folder is marked used (:func:`mark_used`).

    Args:
        src: Source video.
        offset: Start of a virtual source (``analyze --recover`` resumes a
            damaged file here); 0 for the file itself. Each offset gets its own
            folder, so the cached audio/transcript of a part starts at 0.
        create: Make the folder when it is missing; False only names it (a check that
            must leave no folder behind, such as ``cutpoint rerank``'s readiness).
    """
    folder = source_folder(src, offset)
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    mark_used(folder)
    return folder


def source_folder(src: Path, offset: float = 0.0) -> Path:
    """The cache folder :func:`source_dir` gives ``src``, only named: neither created nor
    marked used (``cutpoint cache --prune`` maps projects to their folders with it).

    Raises:
        OSError: If ``src`` (or its ``source_remap`` copy) cannot be read.
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
    return cache_root() / f"{src.stem[:40]}{tag}-{digest}"


# ``cutpoint cache --prune`` reads when a source's cache folder was last used from its
# modification time: source_dir sets it to now, once per process per this many seconds.
USED_INTERVAL_S = 3600.0
# path -> time.monotonic() of the last mark_used in this process
_USED: dict[str, float] = state.register_container(__name__, "_USED", {})


def mark_used(folder: Path) -> None:
    """Set the modification time of the cache folder ``folder`` to now (it is in use, for
    ``cutpoint cache --prune``), at most once per :data:`USED_INTERVAL_S` in this process.
    A missing folder or one that cannot be changed (a read-only cache) is left as it is."""
    key = str(folder)
    now = time.monotonic()
    last = _USED.get(key)
    if last is not None and now - last < USED_INTERVAL_S:
        return
    try:
        os.utime(folder)
    except OSError:
        return
    _USED[key] = now


class HeldOutputs:
    """Files of a project that a run in progress holds back (see :func:`hold_outputs`):
    ``files`` maps each resolved path to the bytes it will get, or None to delete it;
    ``durable`` holds the paths a held write asked to flush to disk (``save_bytes(...,
    durable=True)``), which :meth:`commit` flushes before their rename."""

    def __init__(self, folder: Path, names: Collection[str]) -> None:
        self.folder = folder.resolve()
        self.names = frozenset(names)
        self.files: dict[Path, bytes | None] = {}
        self.durable: set[Path] = set()

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
        temp files renamed onto their files (retried while locked, as :func:`save_bytes`), with
        Ctrl+C, Ctrl+Break and SIGTERM deferred (:func:`_deferred_interrupts`). A file in
        ``durable`` is flushed to disk (``fsync``) when staged and, on POSIX, its folder after
        the renames, as :func:`save_bytes` does with ``durable``."""
        with contextlib.suppress(ValueError):
            _HOLDS.remove(self)
        files, self.files = self.files, {}
        durable, self.durable = self.durable, set()
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
                        if path in durable:
                            with tmp.open("r+b") as f:  # as save_bytes(durable=True)
                                os.fsync(f.fileno())
                        match_mode(tmp, path)
                except BaseException:
                    for tmp, _ in staged:  # before created_dirs removes the empty folders
                        tmp.unlink(missing_ok=True)
                    staged = []
                    raise
            # A cancel in here takes effect once every file is in place: the project gets the
            # whole new set, never part of it next to the old files.
            with _deferred_interrupts():
                for path, data in files.items():
                    if data is None:
                        path.unlink(missing_ok=True)
                for tmp, path in staged:
                    # os.replace, not Path.replace: tests patch cache.os.replace
                    retry_locked(partial(os.replace, tmp, path), path, "replace")
            for folder in dict.fromkeys(path.parent for _, path in staged if path in durable):
                _fsync_dir(folder)
        finally:
            for tmp, _ in staged:  # those not renamed (a rename failed or was interrupted)
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)


# Ctrl+C, and the stop signal the CLI raises KeyboardInterrupt for while a command runs:
# Ctrl+Break on Windows, SIGTERM (a UI cancel) on POSIX
_INTERRUPTS = (signal.SIGINT, signal.SIGBREAK if os.name == "nt" else signal.SIGTERM)


@contextlib.contextmanager
def _deferred_interrupts() -> Iterator[None]:
    """Hold back Ctrl+C and Ctrl+Break or SIGTERM in the block: their handlers only note the
    signal, and on leaving the block the previous handlers are put back and called for the
    signals that arrived (the CLI's and Python's default SIGINT handler raise
    KeyboardInterrupt; so does a default SIGBREAK here, instead of ending the process; a
    default SIGTERM is sent again and ends it). Outside the main thread ``signal.signal``
    raises ValueError, so nothing is deferred there."""
    arrived: list[int] = []
    previous: dict[int, Any] = {}
    try:
        with contextlib.suppress(ValueError):  # not the main thread
            for signum in _INTERRUPTS:
                if signal.getsignal(signum) is None:  # set outside Python: cannot be put back
                    continue
                previous[signum] = signal.signal(signum, lambda s, frame: arrived.append(s))
        yield
    finally:
        error: BaseException | None = None
        for signum, handler in previous.items():  # every one, even if one fails
            for _ in range(2):  # once more if interrupted (another signal's handler)
                try:
                    signal.signal(signum, handler)
                    break
                except BaseException as err:  # noqa: BLE001 - raised below
                    error = error or err
        if error is not None:
            raise error
        for signum in dict.fromkeys(arrived):
            handler = previous[signum]
            if callable(handler):
                handler(signum, None)
            elif handler == signal.SIG_DFL and signum == signal.SIGTERM:
                signal.raise_signal(signum)
            elif handler == signal.SIG_DFL:
                raise KeyboardInterrupt


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


def load_bytes(path: Path) -> bytes | None:
    """The bytes of ``path`` (from a hold when it has them, :func:`hold_outputs`), or None
    if it is missing, deleted in a hold or cannot be read."""
    found, data = _held_bytes(path)
    if found:
        return data
    try:
        return path.read_bytes()
    except OSError:
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

# Windows "Access is denied" and "being used by another process": the file is open elsewhere
# (error 5 can also mean a read-only file or folder).
_LOCKED_WINERRORS = (5, 32)
_LOCK_HINT = "it is open in another program (a player, an editor or a virus scanner)"


def retry_locked[T](
    op: Callable[[], T], path: Path, action: str, locked: Callable[[OSError], bool] | None = None
) -> T:
    """Run ``op`` (replacing or deleting ``path``), retrying briefly on PermissionError.

    On Windows another process (a player, an editor, a virus scanner) holding ``path``
    open makes a rename or delete fail for a moment. ``locked`` picks the errors to retry
    instead; the last one is then re-raised as is, for the caller to word.

    Raises:
        PermissionError: Naming ``path`` and the program holding it, if it is still locked
            (Windows error 5 or 32) after the retries (for error 5 also a read-only file or
            folder); any other error as raised by ``op``.
    """
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            return op()
        except OSError as err:
            if not (locked(err) if locked else isinstance(err, PermissionError)):
                raise
            if attempt < REPLACE_ATTEMPTS - 1:
                time.sleep(min(0.05 * 2**attempt, 0.5))
                continue
            if locked or getattr(err, "winerror", None) not in _LOCKED_WINERRORS:
                raise
            read_only = (
                " If no program has it open, the file or its folder may be read-only: clear the"
                " read-only attribute and re-run."
                if getattr(err, "winerror", None) == 5
                else ""
            )
            raise PermissionError(
                f"Cannot {action} {path}: {_LOCK_HINT}. Close it and re-run.{read_only} "
                f"({err.strerror})"
            ) from err
    raise AssertionError("unreachable")


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


def json_bytes(data: Any, *, indent: int = 2) -> bytes:
    """``data`` as :func:`save_json` writes it: JSON, UTF-8, with the platform's line endings."""
    text = json.dumps(data, indent=indent, ensure_ascii=False)
    # strings escape their newlines, so these are only the indent's line breaks
    return text.replace("\n", os.linesep).encode("utf-8")


def save_json(path: Path, data: Any, *, indent: int = 2, durable: bool = False) -> None:
    """Atomically write ``data`` as JSON, UTF-8, with the platform's line endings
    (:func:`json_bytes`; see :func:`save_bytes`, also for ``durable``)."""
    save_bytes(path, json_bytes(data, indent=indent), durable=durable)


def _fsync_dir(folder: Path) -> None:
    """Flush ``folder``'s entries (a rename into it) to disk: POSIX only, best effort
    (Windows cannot open a folder for it, and some file systems refuse)."""
    if os.name != "posix":
        return
    with contextlib.suppress(OSError):
        fd = os.open(folder, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def save_bytes(path: Path, data: bytes, *, parents: bool = False, durable: bool = False) -> None:
    """Atomically write ``data`` (unique temp file, then rename), with the mode a plain
    write would give (:func:`match_mode`); ``parents``: create the missing folders first.
    A file held back by :func:`hold_outputs` is kept in memory instead (no folder made).

    ``durable``: flush the temp file to disk (``fsync``) before the rename and, on POSIX,
    the folder after it, so a power cut leaves the old or the new file, not an empty one.
    For the files that matter (``config.json``, ``highlights.json``, the swap journal of
    :mod:`cutpoint.folders`); caches leave it off, as each flush costs a disk write (A667).

    On Windows the rename fails with PermissionError while another process has
    ``path`` open; it is retried a few times with a short backoff
    (:func:`retry_locked`). The temp file is removed if the
    rename never succeeds.

    Raises:
        PermissionError: If ``path`` stays locked through all attempts (naming it and
            the program that may hold it, see :func:`retry_locked`).
    """
    if _HOLDS and (got := _held(path)) is not None:
        hold, key = got
        hold.files[key] = bytes(data)
        if durable:
            hold.durable.add(key)
        return
    if parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    tmp = Path(name)
    try:
        tmp.write_bytes(data)
        if durable:
            with tmp.open("r+b") as f:  # Windows flushes only a handle open for writing
                os.fsync(f.fileno())
        match_mode(tmp, path)
        # os.replace, not Path.replace: tests patch cache.os.replace
        retry_locked(partial(os.replace, tmp, path), path, "replace")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if durable:
        _fsync_dir(path.parent)


def save_npz(path: Path, *, compressed: bool = True, parents: bool = False, **arrays: Any) -> None:
    """Atomically write ``arrays`` as an ``.npz`` (:func:`save_bytes`), so an interrupted
    write never leaves a half-written archive at ``path``."""
    buf = io.BytesIO()  # a file object: numpy adds no suffix
    (np.savez_compressed if compressed else np.savez)(buf, **arrays)
    save_bytes(path, buf.getvalue(), parents=parents)


#: What reading a damaged or half-written ``.npz`` raises.
NPZ_ERRORS = (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile, zlib.error)


def load_npz(path: Path) -> dict[str, np.ndarray] | None:
    """All arrays of the ``.npz`` at ``path`` (read eagerly, no file handle kept), or None
    if it is missing or unreadable (logged at DEBUG: the caller treats it as a cache miss)."""
    data = load_bytes(path)
    if data is None:
        return None
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except NPZ_ERRORS as err:
        logger.debug("%s: cache unreadable (%s); treated as missing", path, err)
        return None


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
