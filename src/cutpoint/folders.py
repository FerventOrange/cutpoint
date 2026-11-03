"""Output files and folders: atomic writes, all-or-nothing folder replaces and folder locks.

:func:`output_dir` creates a folder that a failed run removes again while empty;
:func:`atomic_output` writes a file under a temporary name (:func:`part_path`) renamed onto
it on success; :func:`staged_dir` and :func:`swap_in` replace a folder's files all or
nothing (:func:`recover_swap` finishes one a hard kill interrupted); :func:`check_not_sources`
refuses to overwrite a file the run reads; :func:`swap_lock` lets one run at a time work on a
folder. None of it runs ffmpeg.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import socket
import tempfile
import time
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from functools import partial
from pathlib import Path
from typing import Any

from cutpoint import cache, procs
from cutpoint.cache import REPLACE_ATTEMPTS, created_dirs, load_json, make_dirs, save_json
from cutpoint.config import path_key
from cutpoint.errors import FolderBusyError

logger = logging.getLogger(__name__)


@contextmanager
def output_dir(path: Path) -> Iterator[Path]:
    """Create the folder ``path``; on failure (Ctrl+C included) remove it again if this call
    created it and it is still empty, so a failed run leaves no empty folder behind.

    Missing parents are created too and removed the same way. A folder that existed before,
    or that holds anything, is always kept (:func:`cutpoint.cache.created_dirs`).
    """
    with created_dirs() as made:
        make_dirs(path, made)
        yield path


def part_path(out: Path) -> Path:
    """Temporary name :func:`atomic_output` writes ``out`` to: ``.<stem>.part<suffix>``.

    The suffix is kept so ffmpeg picks the muxer; the leading dot keeps it out of the
    clip listings (``assemble``, ``extend``).
    """
    return out.with_name(f".{out.stem}.part{out.suffix}")


@contextmanager
def atomic_output(out: Path) -> Iterator[Path]:
    """Yield :func:`part_path` of ``out`` to write to, renamed onto ``out`` on success.

    On any failure (Ctrl+C included) the partial file is removed and ``out`` is left as
    it was, so a truncated file never appears under the final name. On Windows the rename
    is retried briefly while another process (a virus scanner, a player) holds a file open.
    """
    part = part_path(out)
    try:
        yield part
        cache.retry_locked(lambda: part.replace(out), out, "replace")
    except BaseException:
        # A still-locked part (Windows, right after Ctrl+C) must not replace the real error.
        with suppress(OSError):
            part.unlink(missing_ok=True)
        raise


# --- all-or-nothing folder replace (``clips --force``, ``extend --out <clips_dir> --force``,
# ``extend --out DIR --force`` into a non-empty DIR)
#
# A run exports into the hidden staging folder ``.<name>.cutpoint-tmp`` next to the output,
# then :func:`swap_in` moves the originals it replaces into ``.<name>.cutpoint-bak`` and the
# new files in. The backup holds a journal (the old and new names) until every new file is
# in; on any failure the originals are put back. A hard kill leaves the journal, so the next
# run's :func:`recover_swap` finishes the job: journal present, put the originals back;
# backup without journal, the new set was complete, drop the backup.
# Both folders sit on the parent's file system, so :func:`staged_dir` refuses a ``dst`` that
# is a drive root (no parent to hold them), a mount point or a bind mount before any export,
# unless an empty one is exported into directly (``direct_if_empty``).

_SWAP_JOURNAL = ".cutpoint-swap.json"
_SWAP_PROBE = ".cutpoint-swap-probe"
_SWAP_FIX = "use --out with a folder on that drive"


def _root_error(dst: Path) -> ValueError:
    return ValueError(f"{dst} is a drive root; {_SWAP_FIX}")


def staging_path(dst: Path) -> Path:
    """Hidden staging folder of ``dst``: ``.<name>.cutpoint-tmp`` next to it.

    Raises:
        ValueError: If ``dst`` is a drive root.
    """
    res = dst.resolve()  # "." and ".." have no usable name; a symlink's target is the folder
    if not res.name:
        raise _root_error(dst)
    return res.with_name(f".{res.name}.cutpoint-tmp")


def backup_path(dst: Path) -> Path:
    """Hidden folder :func:`swap_in` moves the replaced originals of ``dst`` to.

    Raises:
        ValueError: If ``dst`` is a drive root.
    """
    res = dst.resolve()
    if not res.name:
        raise _root_error(dst)
    return res.with_name(f".{res.name}.cutpoint-bak")


def check_swap_target(dst: Path) -> None:
    """Fail before any export if ``dst`` cannot be replaced through :func:`staged_dir`.

    Raises:
        ValueError: If ``dst`` is a drive root or a mount point, saying what to do.
    """
    res = dst.resolve()
    if not res.name:
        # A volume mounted into a folder (C:\mnt\vol) resolves to the volume's own root on
        # Windows, so only a folder that is a root before resolving is a drive root.
        own = Path(os.path.abspath(dst))  # noqa: PTH100 - normalises "..", no resolving
        if not own.name or not own.is_mount():
            raise _root_error(dst)
    if not res.name or res.is_mount():
        raise ValueError(
            f"{dst} is a mount point, so the clips cannot be swapped in safely; {_SWAP_FIX}"
        )


def _check_rename(tmp: Path, dst: Path) -> None:
    """Fail if a file cannot be renamed from ``tmp`` into ``dst``.

    Catches what :func:`check_swap_target` cannot see: a Linux bind mount of a folder on the
    same device, where ``is_mount()`` is False but a rename across it fails with EXDEV.
    Any other error is left to :func:`swap_in`, which retries and names the file.

    Raises:
        ValueError: If ``dst`` is on another file system than ``tmp``.
    """
    probe = tmp / _SWAP_PROBE
    try:
        probe.touch()
        probe.rename(dst / _SWAP_PROBE)  # a leftover there is removed by recover_swap
    except OSError as err:
        with suppress(OSError):  # else swap_in would move it into dst
            probe.unlink(missing_ok=True)
        if err.errno == errno.EXDEV:
            raise ValueError(
                f"{dst} is on another file system than its parent folder (a mount point or "
                f"bind mount), so the clips cannot be swapped in safely; {_SWAP_FIX}"
            ) from err
        return
    with suppress(OSError):
        (dst / _SWAP_PROBE).unlink()


def _restore(dst: Path, bak: Path, journal: dict[str, Any]) -> None:
    """Put the originals in ``bak`` back into ``dst`` and drop the new-only files there.

    Both steps are best effort: one locked file does not stop the others. The journal is
    removed last and only if every file went back, so a restore that fails half-way is
    redone by :func:`recover_swap`. A ``dst`` the swap created is removed when empty.

    Raises:
        OSError: The first error, after every file was tried.
    """
    errors: list[OSError] = []
    old = {path_key(n, dst) for n in journal["old"]}
    for name in journal["old"]:
        src = bak / name
        if src.exists() or src.is_symlink():  # else it never left dst
            try:
                cache.retry_locked(partial(src.replace, dst / name), dst / name, "restore")
            except OSError as err:
                errors.append(err)
    for name in journal["new"]:
        # never a user file: swap_in records every existing name as old (once per path_key,
        # so a case variant of a new name is the same file and is not deleted)
        if path_key(name, dst) not in old:
            f = dst / name
            try:
                cache.retry_locked(partial(f.unlink, missing_ok=True), f, "delete")
            except OSError as err:
                errors.append(err)
    if errors:
        raise errors[0]
    (bak / _SWAP_JOURNAL).unlink()
    bak.rmdir()
    if journal.get("created"):
        with suppress(OSError):  # not empty: something else was put there meanwhile
            dst.rmdir()


def recover_swap(dst: Path) -> None:
    """Clean up after a :func:`swap_in` or staged export killed hard (``taskkill /F``).

    With a journal the originals are put back; a backup without one is from a completed
    swap and is removed. A stale staging folder (only this tool's output) is removed.

    Raises:
        RuntimeError: If the journal is unreadable or malformed; the backup is left as is.
        PermissionError: If a leftover backup or staging folder cannot be removed (a file
            in it is still open); the next swap could not start with it in place.
    """
    if not dst.resolve().name:  # a drive root: nothing is ever staged next to it
        return
    with suppress(OSError):  # a probe of _check_rename left by a hard kill
        (dst / _SWAP_PROBE).unlink(missing_ok=True)
    bak = backup_path(dst)
    if bak.exists():
        path = bak / _SWAP_JOURNAL
        if not (path.exists() or path.is_symlink()):
            # Nothing in it is needed, but it must go: the next swap_in creates it anew.
            _remove_leftover(bak, f"left over from a finished replace of {dst}")
        else:
            journal = load_json(path)
            if not (
                isinstance(journal, dict)
                and all(
                    isinstance(journal.get(k), list) and all(isinstance(n, str) for n in journal[k])
                    for k in ("old", "new")
                )
            ):
                raise RuntimeError(
                    f"An earlier replace of {dst} was interrupted and its record {path} cannot "
                    f"be read. The original files are in {bak}: move them back into {dst} by "
                    f"hand, then delete {bak} and re-run."
                )
            _restore(dst, bak, journal)
            logger.warning("Put back the original files of %s from an interrupted replace", dst)
    tmp = staging_path(dst)
    if tmp.exists():
        _remove_leftover(tmp, f"left over from an interrupted export into {dst}")


def _remove_leftover(path: Path, why: str) -> None:
    """Remove the folder ``path`` (retried briefly while a file in it is locked).

    Raises:
        PermissionError: Saying ``why`` it is there and how to fix it, if it stays locked
            (on Windows: a file in it is open in a player or a virus scanner, or a scanner
            briefly keeps it non-empty, error 145).
        OSError: Saying ``why`` it is there, for any other failure.
    """

    def locked(err: OSError) -> bool:
        return isinstance(err, PermissionError) or getattr(err, "winerror", None) == 145

    try:
        cache.retry_locked(partial(shutil.rmtree, path), path, "remove", locked)
    except OSError as err:
        if locked(err):
            raise PermissionError(
                f"Cannot remove {path}, {why}: {cache._LOCK_HINT}. Close it, or delete the folder, "
                f"and re-run. ({err.strerror or err})"
            ) from err
        raise OSError(
            f"Cannot remove {path}, {why}: {err.strerror or err}. Delete the folder and re-run."
        ) from err


@contextmanager
def staged_dir(dst: Path, *, direct_if_empty: bool = False) -> Iterator[Path]:
    """Yield a fresh :func:`staging_path` of ``dst`` to export into, removed on failure.

    The caller moves the result in with :func:`swap_in`, unless ``dst`` itself was yielded:
    with ``direct_if_empty``, an existing empty ``dst`` that cannot be staged (a drive root,
    a mount point or a bind mount) is exported into directly and kept on failure. Call
    :func:`recover_swap` first when ``dst`` is checked for existing files before this.

    Raises:
        ValueError: If ``dst`` is a drive root, a mount point or a bind mount that is not
            exported into directly, before the caller exports anything (see
            :func:`check_swap_target`).
    """
    direct = direct_if_empty and dst.exists() and not any(dst.iterdir())
    try:
        check_swap_target(dst)
        stageable = True
    except ValueError:
        if not direct:
            raise
        stageable = False  # a root or mount point: no staging folder next to it
    if not stageable:
        yield dst
        return
    recover_swap(dst)
    tmp = staging_path(dst)
    tmp.mkdir()
    work = tmp
    try:
        if dst.exists():
            try:
                _check_rename(tmp, dst)
            except ValueError:
                if not direct:
                    raise
                work = dst
                with suppress(OSError):  # still open (a virus scanner): recover_swap removes it
                    shutil.rmtree(tmp)
        yield work
    except BaseException:
        # A still-locked part (Windows, right after Ctrl+C) must not replace the real error.
        try:
            if work is tmp:  # a direct export keeps its finished clips
                shutil.rmtree(tmp)
        except OSError as err:
            logger.warning("Could not remove %s (%s); the next run removes it", tmp, err)
        raise


def file_keys(path: Path) -> set[Any]:
    """Identities of the file ``path`` for :func:`same_file`: its resolved path (case-folded
    where names are case-insensitive) and, where the file system has them, its device and
    inode numbers, so a symlink, a hard link or another spelling of the same file matches.
    Empty if missing."""
    keys: set[Any] = set()
    try:
        st = path.stat()
    except OSError:
        return keys
    keys.add(path_key(str(path.resolve())))
    if st.st_ino:  # 0: no usable inode numbers (some network shares)
        keys.add((st.st_dev, st.st_ino))
    return keys


def same_file(path: Path, keys: set[Any]) -> bool:
    """True if ``path`` is one of the files whose :func:`file_keys` are in ``keys``."""
    return bool(file_keys(path) & keys)


def source_keys(reads: Iterable[tuple[str, Path, Iterable[Path]]]) -> set[Any]:
    """:func:`file_keys` of every file a run reads, given per source ``(source as recorded,
    path read (source_remap), files of a 360 source (PanoExporter.inputs))``."""
    keys: set[Any] = set()
    for s, p, pano_files in reads:
        for f in (Path(s), p, *pano_files):
            keys |= file_keys(f)
    return keys


def check_not_sources(dst: Path, names: Iterable[str], sources: set[Any]) -> None:
    """Fail before any export if writing ``names`` into ``dst`` would replace a source.

    ``sources`` holds the :func:`file_keys` of every file the run reads.

    Raises:
        ValueError: Naming the first such file.
    """
    for n in names:
        if same_file(dst / n, sources):
            raise ValueError(
                f"Cannot write {dst / n}: this run reads it as a source. Move the source out "
                f"of {dst}, or export into another folder (--out)."
            )


def swap_in(new: Path, dst: Path, old: Iterable[str] = ()) -> None:
    """Move every file of ``new`` into ``dst``, all or nothing, and remove ``new``.

    The files ``old`` (names in ``dst``) and those the new files replace are moved to
    :func:`backup_path` first and deleted only after every new file is in. On any failure
    (a locked file, Ctrl+C) the originals are put back and the error is re-raised. If even
    that fails, the backup is kept and the next run puts them back (:func:`recover_swap`).
    A missing ``dst`` is created, and removed again on failure.

    Raises:
        PermissionError: Naming the file, if one stays locked (see
            :func:`cutpoint.cache.retry_locked`).
    """
    created = not dst.exists()
    # manifest.json last in, first out: it never lists a clip that is not there
    names = sorted((p.name for p in new.iterdir()), key=lambda n: (n == "manifest.json", n))
    # one file per path_key (the first spelling): a case variant would move it twice
    present = {
        path_key(n, dst): n
        for n in reversed([*old, *names])
        if (dst / n).exists() or (dst / n).is_symlink()
    }
    olds = [] if created else sorted(present.values(), key=lambda n: (n != "manifest.json", n))
    journal = {"old": olds, "new": names, "created": created}
    bak = backup_path(dst)
    bak.mkdir()
    try:
        save_json(bak / _SWAP_JOURNAL, journal)
        if created:
            dst.mkdir()
        for n in olds:
            cache.retry_locked(partial((dst / n).replace, bak / n), dst / n, "replace")
        for n in names:
            cache.retry_locked(partial((new / n).replace, dst / n), dst / n, "replace")
    except BaseException:
        try:
            if (bak / _SWAP_JOURNAL).exists():
                _restore(dst, bak, journal)
            else:
                bak.rmdir()
        except BaseException as err:  # a second Ctrl+C too: say where the originals are
            logger.error(
                "Could not put back every original file of %s (%s): the rest are in %s. "
                "The next cutpoint run on this folder puts them back.",
                dst,
                err or type(err).__name__,
                bak,
            )
            if not isinstance(err, Exception):
                raise
        raise
    # The new set is complete: the backup is no longer needed.
    journal_path = bak / _SWAP_JOURNAL
    cache.retry_locked(journal_path.unlink, journal_path, "delete")
    try:
        shutil.rmtree(bak)
    except OSError as err:  # still open (a virus scanner): recover_swap removes it later
        logger.warning("Could not remove %s (%s); the next run removes it", bak, err)
    with suppress(OSError):  # still open (a virus scanner): recover_swap removes it later
        new.rmdir()


# --- one run at a time per folder (``clips``, ``extend``, ``assemble``)
#
# Each run holds the lock file ``.<name>.cutpoint-lock`` next to the folder (beside its
# staging and backup folders) while it recovers, exports and swaps, so a second run never
# removes the first one's staging folder. The file is created atomically (O_EXCL) and holds
# the run's process id, host and a token; it is removed on exit, Ctrl+C included. One left by
# a hard kill is stale once its process is gone, and the next run on the same host takes it.

_LOCK_UNWRITTEN_S = 30.0  # an empty lock older than this was never written (a hard kill)
_HELD_LOCKS: set[str] = set()  # path_key of the locks this process holds


def lock_path(dst: Path) -> Path | None:
    """Lock file of ``dst``: ``.<name>.cutpoint-lock`` next to it (None for a drive root,
    which nothing is ever staged next to)."""
    res = dst.resolve()
    return res.with_name(f".{res.name}.cutpoint-lock") if res.name else None


def _read_lock(lock: Path) -> tuple[bytes, dict[str, Any] | None] | None:
    """``(raw bytes, parsed record or None)`` of ``lock``, or None if it is gone."""
    try:
        raw = lock.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:  # being written or removed right now (Windows): treat as unreadable
        return b"", None
    try:
        rec = json.loads(raw.decode("utf-8"))
    except ValueError:
        return raw, None
    return raw, rec if isinstance(rec, dict) and isinstance(rec.get("pid"), int) else None


def _lock_stale(lock: Path, rec: dict[str, Any] | None) -> bool:
    """Whether the lock ``lock`` with the record ``rec`` belongs to no running process."""
    if rec is None:  # empty or garbled: a run is writing it, or one was killed before it did
        try:
            return time.time() - lock.stat().st_mtime > _LOCK_UNWRITTEN_S
        except OSError:
            return False
    if rec.get("host") != socket.gethostname():  # a run on another computer: cannot tell
        return False
    if rec["pid"] == os.getpid():  # this process: held by a run still going, else left over
        return path_key(str(lock)) not in _HELD_LOCKS
    try:
        written = lock.stat().st_mtime
    except OSError:
        written = None
    return not procs.pid_alive(rec["pid"], written)


def _busy_error(dst: Path, lock: Path, rec: dict[str, Any] | None) -> FolderBusyError:
    """``dst``: what the message names as in use (the folder, or ``swap_lock``'s subject)."""
    if rec is None:
        who = "Another cutpoint run is starting on"
    else:
        host = rec.get("host")
        on = f" on {host}" if host and host != socket.gethostname() else ""
        since = f", started {rec['started']}" if rec.get("started") else ""
        who = f"Another cutpoint run (process {rec['pid']}{on}{since}) is using"
    return FolderBusyError(
        f"{who} {dst}. Wait until it has finished, or stop it, and run again. If no other run "
        f"is active, delete {lock} and run again."
    )


def _take_stale(lock: Path, raw: bytes) -> None:
    """Remove the stale lock ``lock`` whose content was ``raw``.

    It is first renamed to a name of this process, so of two runs that both found it stale,
    only one removes it; a lock that another run created meanwhile is put back.
    """
    moved = lock.with_name(f"{lock.name}.{os.getpid()}.stale")
    try:
        lock.replace(moved)
    except OSError:  # another run took it first, or has it open (Windows): try again
        return
    try:
        if moved.read_bytes() == raw:
            moved.unlink()
            logger.info("Removed %s, left by a cutpoint run that is no longer running", lock)
            return
    except OSError:
        pass
    with suppress(OSError):  # a fresh lock of another run: give it back
        moved.rename(lock)


def _dir_writable(folder: Path) -> bool:
    """Whether a file can be created in ``folder`` (a probe file, removed again)."""
    try:
        fd, name = tempfile.mkstemp(dir=folder, prefix=".cutpoint-probe.")
    except OSError:
        return False
    os.close(fd)
    with suppress(OSError):
        Path(name).unlink()
    return True


def _open_lock(lock: Path) -> int | None:
    """Create ``lock`` exclusively and return its descriptor, or None if its folder is
    read-only (nothing can be staged there either, so there is nothing to protect).

    A PermissionError is retried briefly: on Windows a lock another run is removing is
    "delete pending" for a moment.

    Raises:
        FileExistsError: If another run's lock is there.
        PermissionError: If the folder is writable but the lock still cannot be created.
        OSError: Any other error.
    """
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            return os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            raise
        except OSError as err:
            if err.errno == errno.EROFS:
                return None
            if err.errno not in (errno.EACCES, errno.EPERM):
                raise
            if attempt < REPLACE_ATTEMPTS - 1:
                time.sleep(min(0.05 * 2**attempt, 0.5))
                continue
            if not _dir_writable(lock.parent):
                return None
            raise PermissionError(
                f"Cannot create the lock file {lock}: {err.strerror or err}. If no other "
                "cutpoint run is active, delete it and run again."
            ) from err
    raise AssertionError("unreachable")


@contextmanager
def swap_lock(dst: Path, subject: Path | None = None) -> Iterator[None]:
    """Hold the lock of the folder ``dst`` (:func:`lock_path`) for one run. ``subject``: what
    the busy message names as in use instead of ``dst`` (e.g. the project of a run lock).

    Missing parent folders are created (removed again on failure while empty, as
    :func:`output_dir`). The lock is removed on exit, Ctrl+C included; one left by a run that
    was killed hard is taken over once its process is gone (same host only).

    Raises:
        FolderBusyError: If another run holds it, naming its process and the lock file.
    """
    lock = lock_path(dst)
    if lock is None:
        yield
        return
    with output_dir(lock.parent):
        token = uuid.uuid4().hex
        data = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "token": token,
            }
        ).encode("utf-8")
        for _ in range(3):
            try:
                fd = _open_lock(lock)
            except FileExistsError:
                found = _read_lock(lock)
                if found is None:  # removed meanwhile: try again
                    continue
                raw, rec = found
                if not _lock_stale(lock, rec):
                    raise _busy_error(subject or dst, lock, rec) from None
                _take_stale(lock, raw)
                continue
            if fd is None:
                logger.info("Not locking %s: %s is read-only", dst, lock.parent)
                yield
                return
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
            break
        else:
            found = _read_lock(lock)
            raise _busy_error(subject or dst, lock, found[1] if found else None)
        key = path_key(str(lock))
        _HELD_LOCKS.add(key)
        try:
            yield
        finally:
            _HELD_LOCKS.discard(key)
            found = _read_lock(lock)
            if found is not None and found[0] == data:  # never another run's lock
                try:
                    cache.retry_locked(lock.unlink, lock, "delete")
                except OSError as err:  # a stale lock: the next run takes it over
                    logger.warning("Could not remove %s (%s)", lock, err)
