"""``cutpoint stage``: copy sources on slow storage to a local folder once.

Slow storage means a NAS, a USB drive, a cloud-synced folder or any other
network share.

Every later read of a staged source (``analyze``, ``events``, ``clips``,
region discovery) is redirected to the local copy by
:func:`cutpoint.ffmpeg_utils.stage_inputs`, so a 3 GB recording crosses
the network exactly once instead of once per pass (audio extraction, scene
proxy, OCR decode, VLM frames, clip export). Projects keep recording the
original path; the stage folder is disposable.

Layout: ``<stage_dir>/<digest>-<file name>`` plus ``index.json`` mapping the
canonical source path (:func:`stage_key`) to ``{"file", "size", "mtime_ns", "source",
"read_s"}``, plus ``"rel"`` (the exact path parts below a Windows-form ``source_remap``
prefix, on POSIX only) and, for case-twins below such a prefix, extra keys (see :func:`_find`).
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path, PureWindowsPath
from typing import Any, NamedTuple

from cutpoint import cache, state
from cutpoint.config import (
    canonical_source,
    get_config,
    is_foreign_windows,
    norm_path,
    remap_source,
    unc_path,
    windows_norm,
)
from cutpoint.utils import matches_glob

logger = logging.getLogger(__name__)

CHUNK = 16 * 1024 * 1024
STAGE_INDEX = "index.json"
# A case-twin's index key is <key>|<its "rel" joined by "/">. A POSIX file name may contain "|",
# so only that exact suffix is stripped (_base_key).
_TWIN_SEP = "|"
_NETWORK_FS = (
    "nfs",
    "nfs4",
    "cifs",
    "smb3",
    "smbfs",
    "sshfs",
    "fuse.sshfs",
    "fuse.gvfsd-fuse",
    "fuse.rclone",
    "davfs",
    "9p",
    "ceph",
    "fuse.glusterfs",
)


def _mount(path: Path) -> tuple[str, str]:
    """``(mount point, file-system type)`` holding ``path`` (from ``/proc/mounts``;
    ``("", "")`` if unknown)."""
    try:
        target = str(path.resolve())
        lines = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "", ""
    best, fstype = "", ""
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        # /proc/mounts escapes spaces and the like as octal (\040).
        mnt = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), fields[1])
        inside = target == mnt or target.startswith(mnt.rstrip("/") + "/")
        if inside and len(mnt) >= len(best):  # deepest mount; the later one of equals
            best, fstype = mnt, fields[2]
    return best, fstype


def mount_fstype(path: Path) -> str:
    """File-system type of the mount holding ``path`` (from ``/proc/mounts``; "" if unknown)."""
    return _mount(path)[1]


def is_remote(path: Path) -> bool:
    """Whether ``path`` is likely on slow or network storage.

    Windows: any drive but ``C:``, ``D:`` and ``E:``. Elsewhere: a network file
    system mount (NFS, SMB/CIFS, SSHFS, GVfs, rclone, WebDAV, 9p, Ceph,
    GlusterFS); anything else counts as local.
    """
    if sys.platform == "win32":
        return path.drive.upper() not in ("C:", "D:", "E:")
    return mount_fstype(path) in _NETWORK_FS


def storage_label(path: Path) -> str:
    """Where ``path`` is stored, for log messages: its drive on Windows, otherwise
    its mount point and file system (``/mnt/nas (nfs4)``), or its folder if unknown."""
    if sys.platform == "win32":
        return path.drive
    mnt, fstype = _mount(path)
    return f"{mnt} ({fstype})" if mnt else str(path.parent)


def stage_key(path: str | Path) -> str:
    """Index key of a source: its canonical (pre-remap) path, normalised
    (:func:`cutpoint.config.norm_path`: case-folded on Windows only).

    The path is resolved first, so a symlinked folder or, on Windows, a mapped
    network drive (``N:\\Footage\\x.mp4``) and its UNC form
    (``\\\\server\\share\\Footage\\x.mp4``, what ``find_videos`` returns) give
    the same key.

    On POSIX, a Windows path (``D:\\Footage\\x.mp4``, as recorded on Windows or
    as the canonical form under a Windows ``source_remap`` prefix) is keyed the
    Windows way (:func:`cutpoint.config.windows_norm`: case-folded, backslashes),
    without resolving it against the current folder. Files that differ only in
    case below that prefix share the key; see :func:`_find`.
    """
    return _identity(path)[0]


def _identity(path: str | Path) -> tuple[str, tuple[str, ...]]:
    """``(stage key, exact parts of the canonical path)``; the parts only for a Windows-form
    key on POSIX (``()`` otherwise), the one kind of key that folds case on a case-sensitive
    file system."""
    text = _canonical(path)
    if is_foreign_windows(text):
        return windows_norm(text), PureWindowsPath(text).parts
    return norm_path(text), ()


def _canonical(path: str | Path) -> str:
    """``path`` resolved (unless a Windows path on POSIX) and in its pre-remap spelling."""
    text = str(path)
    if not is_foreign_windows(text):
        text = os.path.abspath(text)  # noqa: PTH100 - normalises '..', keeps drive letters
        with contextlib.suppress(OSError):
            text = str(Path(text).resolve())
    return canonical_source(text)


def _below_prefix(key: str, parts: tuple[str, ...]) -> tuple[str, ...] | None:
    """The exact ``parts`` below the deepest Windows-form ``source_remap`` prefix holding
    ``key`` (``("T", "x.mp4")`` for ``D:\\Videos\\T\\x.mp4`` under ``D:\\Videos``), or None
    when no such prefix holds it (any key that is not Windows-form on POSIX)."""
    if not parts:
        return None
    depth = None
    for old, _new in get_config().source_remap:
        form = windows_norm(old) if is_foreign_windows(old) else None
        if form and (key == form or key.startswith(form + "\\")):
            depth = max(depth or 0, len(PureWindowsPath(old).parts))
    return None if depth is None else parts[depth:]


def _same_file(entry: dict[str, Any], parts: tuple[str, ...], rel: tuple[str, ...] | None) -> bool:
    """``entry`` (under the same Windows-form key) is the copy of the file with ``parts``.

    The key compares the Windows way, but the files below a Windows ``source_remap`` prefix
    may sit on a case-sensitive file system (``T/x.mp4`` and ``T/X.MP4``): there the spelling
    must match exactly. ``entry["rel"]`` is that spelling, recorded when it was staged. An
    entry staged before it was recorded (#163) is compared through its ``source`` (the path
    it was read from, under the remap target): its last ``len(rel)`` parts, with ``rel`` the
    asked file's parts below the prefix in the current ``source_remap``. Without either, the
    key decides.
    """
    stored = entry.get("rel")
    if isinstance(stored, list):
        want = tuple(stored)
    elif rel is not None and isinstance(entry.get("source"), str):
        source = entry["source"]
        read = (PureWindowsPath(source) if is_foreign_windows(source) else Path(source)).parts
        if len(rel) > len(read):
            return False
        want = read[len(read) - len(rel) :]
    else:
        return True
    return len(want) <= len(parts) and parts[len(parts) - len(want) :] == want


def _find(index: dict[str, dict[str, Any]], key: str, parts: tuple[str, ...]) -> str | None:
    """Index key of the entry for the file with stage key ``key`` and ``parts``, or None.

    Usually ``key`` itself. Case-twins under a Windows-form key on POSIX (``T/x.mp4`` and
    ``T/X.MP4``) share ``key``: the first one staged has it, every other one
    ``<key>|<exact path below the prefix>``; each entry is told apart by :func:`_same_file`
    (which also rejects a different file whose own key merely starts with ``<key>|``).
    """
    if not parts:
        return key if key in index else None
    rel = _below_prefix(key, parts)
    twins = (k for k in index if k.startswith(key + _TWIN_SEP))
    for slot in (key, *twins):
        entry = index.get(slot)
        if entry and _same_file(entry, parts, rel):
            return slot
    return None


def _base_key(slot: str, entry: dict[str, Any]) -> str:
    """Stage key of an index key: a case-twin's key without its ``|<rel>`` suffix (a ``|`` in
    a file name stays)."""
    rel = entry.get("rel")
    suffix = _TWIN_SEP + "/".join(rel) if isinstance(rel, list) else None
    return slot[: -len(suffix)] if suffix and slot.endswith(suffix) else slot


class _CacheInfo(NamedTuple):
    currsize: int


# The last loaded stage index and the last staged_path results, each keyed by what it was built
# from, so a long-lived process (`cutpoint ui`) sees copies staged by another process and an
# edited stage_dir or source_remap without reload_stage_index().
_INDEX: dict[tuple, dict[str, dict]] = {}
# staged_path results: path -> (result, size of the copy; None when result is path).
_STAGED: dict[tuple, dict[str, tuple[str, int | None]]] = {}


def _index_stamp(stage_dir: Path) -> tuple:
    """``(stage_dir, (mtime_ns, size, inode) of its index or None when missing)``: what a loaded
    index depends on (one stat; the index is replaced on every write, so its inode changes)."""
    try:
        st = (stage_dir / STAGE_INDEX).stat()
        stamp: tuple | None = (st.st_mtime_ns, st.st_size, st.st_ino)
    except OSError:
        stamp = None
    return (stage_dir, stamp)


def _stage_index() -> dict[str, dict]:
    """The stage index ``{key: {"file", "size", "mtime_ns", ...}}``, cached until
    ``stage_dir`` or the index file changes."""
    stamp = _index_stamp(get_config().stage_dir)
    index = _INDEX.get(stamp)
    if index is None:
        index = _load_index(stamp[0])
        _INDEX.clear()
        _INDEX[stamp] = index
    return index


_stage_index.cache_clear = _INDEX.clear  # type: ignore[attr-defined]
_stage_index.cache_info = lambda: _CacheInfo(len(_INDEX))  # type: ignore[attr-defined]
state.register_cache(_stage_index)  # type: ignore[arg-type]


def reload_stage_index() -> None:
    """Forget the cached stage index and staged paths (they also reload by themselves when
    ``stage_dir``, ``source_remap`` or the index file changes)."""
    _stage_index.cache_clear()  # type: ignore[attr-defined]
    staged_path.cache_clear()  # type: ignore[attr-defined]
    _EXPANDED.clear()
    _SUBST.clear()


def staged_path(path: str) -> str:
    """Local staged copy of a source if one exists, else ``path`` unchanged.

    A copy counts only if its size matches the size recorded when it was
    staged. Non-video arguments (pipes, lavfi specs, outputs) never match the
    index, so this is safe to apply to any ffmpeg ``-i`` argument.

    Results are cached until ``stage_dir``, ``source_remap`` or the stage index
    file changes; a cached copy is checked again (one stat), so one deleted or
    truncated by hand is no longer returned.
    """
    cfg = get_config()
    deps = (cfg.source_remap, _index_stamp(cfg.stage_dir))
    memo = _STAGED.get(deps)
    if memo is None:
        _STAGED.clear()
        memo = _STAGED[deps] = {}
    hit = memo.get(path)
    if hit is not None and hit[1] is not None:
        try:
            if Path(hit[0]).stat().st_size != hit[1]:
                hit = None
        except OSError:
            hit = None
    if hit is None:
        hit = memo[path] = _staged_path(path)
    return hit[0]


staged_path.cache_clear = _STAGED.clear  # type: ignore[attr-defined]
staged_path.cache_info = lambda: _CacheInfo(  # type: ignore[attr-defined]
    sum(len(memo) for memo in _STAGED.values())
)
state.register_cache(staged_path)  # type: ignore[arg-type]


def _staged_path(path: str) -> tuple[str, int | None]:
    """:func:`staged_path` without the cache: ``(result, size of the copy or None)``."""
    if not path or path.startswith(("-", "pipe:")) or "=" in path:
        return path, None
    index = _stage_index()
    if not index:
        return path, None
    key, parts = _identity(path)
    slot = _find(index, key, parts)
    if slot is None:
        slot = _find_subst(index, key)
    if slot is None:
        return path, None
    entry = index[slot]
    copy = get_config().stage_dir / entry["file"]
    try:
        if copy.stat().st_size != entry["size"]:
            return path, None
    except OSError:
        return path, None
    logger.debug("staged read %s -> %s", path, copy)
    return str(copy), entry["size"]


def _query_dos_device(drive: str) -> str | None:
    """Windows: what ``QueryDosDeviceW`` names drive ``drive`` (``Q:``): ``\\??\\C:\\real`` for
    a ``subst`` drive, ``\\Device\\...`` for a disk or a network drive. A local kernel call (no
    network access). Never raises; None on failure and on other platforms.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes  # noqa: PLC0415 - Windows only
    except ImportError:
        return None
    try:
        query = ctypes.WinDLL("kernel32").QueryDosDeviceW  # type: ignore[attr-defined]
        buf = ctypes.create_unicode_buffer(1024)
        if query(drive, buf, len(buf)):
            return buf.value
    except (OSError, AttributeError, ValueError, ctypes.ArgumentError):
        pass
    return None


# _subst_target results by drive letter.
_SUBST: dict[str, str | None] = state.register_container(__name__, "_SUBST", {})


def _subst_target(drive: str) -> str | None:
    """Folder the ``subst`` drive ``drive`` (``Q:``) maps to (``C:\\real``, or
    ``\\\\server\\share\\x`` for one on a share), or None for any other drive: a disk or a
    network-mapped drive (``\\Device\\LanmanRedirector\\...``, ``\\Device\\Mup\\...``).
    Cached per drive letter until the stage index reloads."""
    if drive not in _SUBST:
        device = _query_dos_device(drive)
        target = device[4:] if device and device.startswith("\\??\\") else None
        if target is not None and target[:4].upper() == "UNC\\":
            target = "\\\\" + target[4:]
        _SUBST[drive] = target
    return _SUBST[drive]


# A subst drive may map into another one; follow at most this many.
_SUBST_DEPTH = 8


def _expand_subst(path: str) -> str:
    """``path`` with a leading ``subst`` drive letter replaced by the folder it maps to, again
    while that folder is on one. Lexical: the file system is not touched; any other path
    (another drive, UNC) is returned unchanged."""
    for _ in range(_SUBST_DEPTH):
        drive = PureWindowsPath(path).drive
        target = _subst_target(drive.upper()) if len(drive) == 2 and drive[1] == ":" else None
        if target is None:
            break
        path = target.rstrip("\\") + "\\" + path[2:].lstrip("\\/")
    return path


# Subst-expanded forms of the last index's keys: id(index) -> (index, {form: index key}).
_EXPANDED: dict[int, tuple[dict[str, dict], dict[str, str]]] = state.register_container(
    __name__, "_EXPANDED", {}
)


def _find_subst(index: dict[str, dict[str, Any]], key: str) -> str | None:
    """Windows: index key matching stage key ``key`` once ``subst`` drive letters are expanded
    on both sides, or None (#186).

    For a miss of the exact spelling: an index key keeps the canonical (pre-remap) spelling,
    which may be on a ``subst`` drive (``Q:\\Footage``) while the file is asked for through the
    folder it maps to (``D:\\Footage``), or the reverse. Network-free: keys are never resolved,
    and network-mapped drives and UNC paths stay as they are. Keys are expanded once per loaded
    index; stored keys and copy names stay as they are.
    """
    if sys.platform != "win32":
        return None
    cached = _EXPANDED.get(id(index))
    if cached is None or cached[0] is not index:
        forms: dict[str, str] = {}
        _SUBST.clear()  # subst mappings live as long as the expanded forms built from them
        for slot, entry in index.items():
            forms.setdefault(norm_path(_expand_subst(_base_key(slot, entry))), slot)
        _EXPANDED.clear()
        cached = _EXPANDED[id(index)] = (index, forms)
    return cached[1].get(norm_path(_expand_subst(key)))


def _load_index(stage_dir: Path) -> dict[str, dict[str, Any]]:
    data = cache.load_json(stage_dir / STAGE_INDEX)
    return data if isinstance(data, dict) else {}


def _save_index(stage_dir: Path, index: dict[str, dict[str, Any]]) -> None:
    cache.save_json(stage_dir / STAGE_INDEX, index)


def copy_once(src: Path, dst: Path, label: str) -> float:
    """Sequentially copy ``src`` to ``dst`` (via ``.partial``), keeping mtime.

    Returns:
        Seconds spent reading and writing.
    """
    st = src.stat()
    size = st.st_size
    tmp = dst.with_name(dst.name + ".partial")
    t0 = time.perf_counter()
    done, next_log = 0, 0.1
    with src.open("rb") as fi, tmp.open("wb") as fo:
        while chunk := fi.read(CHUNK):
            fo.write(chunk)
            done += len(chunk)
            if size and done / size >= next_log:
                el = time.perf_counter() - t0
                logger.info(
                    "  %s: %3d%% (%.0f MB/s)",
                    label,
                    int(done / size * 100),
                    done / 1e6 / max(el, 1e-6),
                )
                next_log += 0.1
    os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
    tmp.replace(dst)
    return time.perf_counter() - t0


def _stage_files(targets: Sequence[Path], exclude: Sequence[str], recursive: bool) -> list[Path]:
    """Videos under ``targets`` at the path they are read from, de-duplicated.

    Listed and excluded as ``analyze`` does (original spelling), then read
    through ``source_remap``. Shared by ``stage`` and ``stage --clear``.

    Raises:
        FileNotFoundError: As ``find_videos``, when an input cannot be listed.
    """
    # Lazy: analyze is heavy and stage is imported by the CLI at startup.
    from cutpoint.analyze import _find_inputs  # noqa: PLC0415

    found = _find_inputs(list(targets), exclude, recursive)
    return list(dict.fromkeys(remap_source(p) for p in found))


def stage_sources(
    targets: Sequence[Path],
    *,
    exclude: Sequence[str] = (),
    recursive: bool = False,
    min_free_gb: float = 5.0,
) -> dict[str, Any]:
    """Stage every video under ``targets`` (already-staged copies are kept).

    Args:
        targets: Files and/or folders (``source_remap`` is applied, so
            a source under a remapped prefix is read from its new location).
        exclude: Glob patterns to skip, matched against the original
            (pre-remap) spelling, as for ``analyze``.
        recursive: Search sub-folders of folder inputs.
        min_free_gb: Refuse to start a copy that would leave less free space.

    Returns:
        ``{"files", "bytes", "read_s", "mb_per_s", "skipped"}``.

    Raises:
        OSError: If the stage folder lacks space for a file.
    """
    stage_dir = get_config().stage_dir
    with cache.created_dirs() as made:  # a failed run leaves no empty stage folder behind
        cache.make_dirs(stage_dir, made)
        index = _load_index(stage_dir)
        files = _stage_files(targets, exclude, recursive)
        total_b, total_s, copied, skipped = 0, 0.0, 0, 0
        for src in files:
            key, parts = _identity(src)
            rel = _below_prefix(key, parts)
            st = src.stat()
            slot = _find(index, key, parts)
            entry = index.get(slot) if slot else None
            if entry:
                dst = stage_dir / entry["file"]
                if (
                    dst.is_file()
                    and dst.stat().st_size == st.st_size
                    and entry.get("mtime_ns") == st.st_mtime_ns
                ):
                    logger.info("Already staged: %s", src.name)
                    skipped += 1
                    if rel is not None and "rel" not in entry:  # staged before #163: record it now
                        entry["rel"] = list(rel)
                        _save_index(stage_dir, index)
                    continue
            if slot is None:  # a new file; a case-twin of a staged one gets its own key
                slot = key
                if key in index:
                    slot += _TWIN_SEP + "/".join(rel if rel is not None else parts)
            free = shutil.disk_usage(stage_dir).free
            if free - st.st_size < min_free_gb * 1e9:
                raise OSError(
                    f"Not enough space in {stage_dir} for {src.name} "
                    f"({st.st_size / 1e9:.1f} GB, {free / 1e9:.1f} GB free)"
                )
            digest = hashlib.sha1(slot.encode("utf-8")).hexdigest()[:8]
            dst = stage_dir / f"{digest}-{src.name}"
            logger.info("Staging %s (%.2f GB) -> %s", src, st.st_size / 1e9, dst)
            secs = copy_once(src, dst, src.name)
            logger.info("  staged in %.0fs (%.0f MB/s)", secs, st.st_size / 1e6 / secs)
            index[slot] = {
                "file": dst.name,
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
                "source": str(src),
                "read_s": round(secs, 1),
            }
            if rel is not None:
                index[slot]["rel"] = list(rel)
            _save_index(stage_dir, index)
            total_b += st.st_size
            total_s += secs
            copied += 1
        reload_stage_index()
        summary = {
            "files": copied,
            "skipped": skipped,
            "bytes": total_b,
            "read_s": round(total_s, 1),
            "mb_per_s": round(total_b / 1e6 / total_s, 1) if total_s else None,
        }
        logger.info(
            "Staged %d file(s), %.1f GB in %.0fs (%s MB/s); %d already staged",
            copied,
            total_b / 1e9,
            total_s,
            summary["mb_per_s"],
            skipped,
        )
        return summary


def _recorded_under(
    index: dict[str, dict[str, Any]], target: Path, exclude: Sequence[str], recursive: bool
) -> set[str]:
    """Index keys of ``target`` itself or of the files directly below it (any
    depth with ``recursive``), minus those matching ``exclude``.

    For an input that can no longer be listed. Keys are compared normalised, on
    a separator boundary (``old`` does not take ``older``); ``exclude`` is
    matched as ``find_videos`` does, against the key (the original spelling,
    case-folded on Windows, where globs ignore case anyway). A case-twin's key
    (:func:`_find`) is compared without its suffix, so twins are selected together.
    """
    absolute = os.path.abspath(str(target))  # noqa: PTH100 - normalises '..', keeps drive letters
    # The UNC form too: a disconnected mapped drive does not resolve to it.
    prefixes = {stage_key(target), stage_key(unc_path(absolute))}
    keys = set()
    for slot in index:
        key = _base_key(slot, index[slot])
        for prefix in prefixes:
            if key == prefix:
                break
            # A Windows-recorded key on POSIX (canonical form of a remap target) uses '\\'.
            sep = "\\" if sys.platform == "win32" or is_foreign_windows(prefix) else "/"
            if key.startswith(prefix + sep) and (recursive or sep not in key[len(prefix) + 1 :]):
                break
        else:
            continue
        if not matches_glob(key, exclude):
            keys.add(slot)
    return keys


def clear_stage(
    targets: Sequence[Path] = (), *, exclude: Sequence[str] = (), recursive: bool = False
) -> int:
    """Delete staged copies (all, or only those of ``targets``); returns the count.

    Each of ``targets`` is selected on its own. One that can be listed (directly
    or through ``source_remap``) selects, with ``exclude`` and ``recursive``,
    the same files :func:`stage_sources` would copy. One that cannot (gone
    together with its remap target) selects the copies recorded under it in the
    stage index (see :func:`_recorded_under`), or nothing with a WARNING; the
    other targets are still cleared. Everything is selected before anything is
    deleted.
    """
    stage_dir = get_config().stage_dir
    index = _load_index(stage_dir)
    keys = set(index)
    if targets:
        wanted: set[str] = set()
        for target in targets:
            try:
                files = _stage_files([target], exclude, recursive)
            except FileNotFoundError as err:
                recorded = _recorded_under(index, target, exclude, recursive)
                if recorded:
                    logger.info(
                        "%s; clearing the %d staged copy(ies) recorded under it",
                        err,
                        len(recorded),
                    )
                else:
                    logger.warning("%s; no staged copy recorded under it", err)
                wanted |= recorded
                continue
            found = (_find(index, *_identity(p)) for p in files)
            wanted |= {slot for slot in found if slot is not None}
        keys &= wanted
    for key in keys:
        (stage_dir / index[key]["file"]).unlink(missing_ok=True)
        del index[key]
    if stage_dir.is_dir():
        _save_index(stage_dir, index)
    reload_stage_index()
    logger.info("Removed %d staged file(s) from %s", len(keys), stage_dir)
    return len(keys)


def list_stage() -> list[dict[str, Any]]:
    """Index entries of the stage folder (for ``cutpoint stage --list``)."""
    return list(_load_index(get_config().stage_dir).values())
