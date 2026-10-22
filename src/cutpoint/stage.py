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
canonical source path to ``{"file", "size", "mtime_ns", "source", "read_s"}``.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
import os
import re
import shutil
import sys
import time
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from cutpoint import cache, state
from cutpoint.config import canonical_source, get_config, norm_path, remap_source
from cutpoint.utils import find_videos

logger = logging.getLogger(__name__)

CHUNK = 16 * 1024 * 1024
STAGE_INDEX = "index.json"
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
    """
    text = os.path.abspath(str(path))  # noqa: PTH100 - normalises '..', keeps drive letters
    with contextlib.suppress(OSError):
        text = str(Path(text).resolve())
    return norm_path(canonical_source(text))


@state.register_cache
@lru_cache(maxsize=1)
def _stage_index() -> dict[str, dict]:
    """The stage index ``{key: {"file", "size", "mtime_ns"}}`` (cached per process)."""
    data = cache.load_json(get_config().stage_dir / STAGE_INDEX)
    return data if isinstance(data, dict) else {}


def reload_stage_index() -> None:
    """Forget the cached stage index (after ``cutpoint stage`` wrote new copies)."""
    _stage_index.cache_clear()
    staged_path.cache_clear()


@state.register_cache
@functools.cache  # not `cache`: that name is the cutpoint.cache module here
def staged_path(path: str) -> str:
    """Local staged copy of a source if one exists, else ``path`` unchanged.

    A copy counts only if its size matches the size recorded when it was
    staged. Non-video arguments (pipes, lavfi specs, outputs) never match the
    index, so this is safe to apply to any ffmpeg ``-i`` argument.
    """
    if not path or path.startswith(("-", "pipe:")) or "=" in path:
        return path
    index = _stage_index()
    if not index:
        return path
    entry = index.get(stage_key(path))
    if not entry:
        return path
    copy = get_config().stage_dir / entry["file"]
    try:
        if copy.stat().st_size != entry["size"]:
            return path
    except OSError:
        return path
    logger.debug("staged read %s -> %s", path, copy)
    return str(copy)


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
    size = src.stat().st_size
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
    st = src.stat()
    os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
    tmp.replace(dst)
    return time.perf_counter() - t0


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
        exclude: Glob patterns to skip.
        recursive: Search sub-folders of folder inputs.
        min_free_gb: Refuse to start a copy that would leave less free space.

    Returns:
        ``{"files", "bytes", "read_s", "mb_per_s", "skipped"}``.

    Raises:
        OSError: If the stage folder lacks space for a file.
    """
    stage_dir = get_config().stage_dir
    stage_dir.mkdir(parents=True, exist_ok=True)
    index = _load_index(stage_dir)
    files = find_videos([remap_source(t) for t in targets], exclude=exclude, recursive=recursive)
    total_b, total_s, copied, skipped = 0, 0.0, 0, 0
    for src in files:
        key = stage_key(src)
        st = src.stat()
        entry = index.get(key)
        if entry:
            dst = stage_dir / entry["file"]
            if (
                dst.is_file()
                and dst.stat().st_size == st.st_size
                and entry.get("mtime_ns") == st.st_mtime_ns
            ):
                logger.info("Already staged: %s", src.name)
                skipped += 1
                continue
        free = shutil.disk_usage(stage_dir).free
        if free - st.st_size < min_free_gb * 1e9:
            raise OSError(
                f"Not enough space in {stage_dir} for {src.name} "
                f"({st.st_size / 1e9:.1f} GB, {free / 1e9:.1f} GB free)"
            )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
        dst = stage_dir / f"{digest}-{src.name}"
        logger.info("Staging %s (%.2f GB) -> %s", src, st.st_size / 1e9, dst)
        secs = copy_once(src, dst, src.name)
        logger.info("  staged in %.0fs (%.0f MB/s)", secs, st.st_size / 1e6 / secs)
        index[key] = {
            "file": dst.name,
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "source": str(src),
            "read_s": round(secs, 1),
        }
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


def clear_stage(targets: Sequence[Path] = ()) -> int:
    """Delete staged copies (all, or only those of ``targets``); returns the count."""
    stage_dir = get_config().stage_dir
    index = _load_index(stage_dir)
    keys = set(index)
    if targets:
        wanted = {stage_key(p) for p in find_videos([remap_source(t) for t in targets])}
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
