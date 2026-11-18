"""``cutpoint cache --prune DAYS``: remove the source cache folders not used for DAYS days.

Only the per-source folders of :func:`cutpoint.cache.source_dir` (``<name>[@offset]-<16 hex>``)
are removed, least recently used first. Their last use is the newest modification time of the
folder and of everything in it: :func:`cutpoint.cache.source_dir` sets the folder's to now when
a run uses it, and every write into it changes it too. Everything else in the cache folder
stays (the profile discovery and 360 check caches, the timeline index, symlinks, unknown
files). The profile discovery (``activity/``) and 360 check (``pano_vl/``) caches are a few KB
per footage or clip and stand for model calls, so they are not pruned. After a prune that
removed folders, the timeline index (:data:`cutpoint.stage.TIMELINE_INDEX`) drops the entries
of the copies that were in them.

A folder is kept when it was used within DAYS days, when it belongs to a project in the
projects folder (unless ``include_projects``), and always while it is in use: its project's
run lock (``<project>/.analysis.run.cutpoint-lock``) or a lock in the folder (tsrepair's
``.timeline.cutpoint-lock``) is held, or it cannot be read in full. A project's folders are
those of the sources its ``highlights.json``, ``progress.json``, ``pano.json``,
``ui/meta.json`` and ``clips*/manifest.json`` name, with their recovered parts
(``read_errors.json``); for a source that cannot be read, every folder named after it is kept.

The checks are made again just before a folder is removed, and once more after its rename
(a folder used meanwhile is renamed back), as a run can mark it used at any moment.

A folder is first renamed to ``.<name>.<8 hex>.cutpoint-prune`` (atomic, so no run sees it half
removed; on Windows a folder with an open file cannot be renamed and stays), then removed. A
removal that fails is a warning, and what is left of it goes on the next prune. One prune at a
time holds ``<cache>/.prune.cutpoint-lock``. Nothing outside the cache folder is touched and no
symlink is followed.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import stat
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from cutpoint import cache, folders, jsonout, paths, runstate, stage
from cutpoint.config import projects_dir
from cutpoint.errors import FolderBusyError

logger = logging.getLogger(__name__)

DAY_S = 86400.0
FOLDER_NAME = re.compile(r".+-[0-9a-f]{16}")  # cache.source_dir: <stem[:40]>[@offset]-<sha16>
LEFTOVER = re.compile(r"\..+\.[0-9a-f]{8}\.cutpoint-prune")  # a folder a prune did not finish
LOCK_SUFFIX = ".cutpoint-lock"  # folders.lock_path: .<name>.cutpoint-lock
PRUNE_LOCK = "prune"  # swap_lock(<cache>/prune): .prune.cutpoint-lock
# Why a folder was kept (``kept`` of the report)
RECENT, PROJECT, IN_USE = "recent", "project", "in_use"


@dataclass
class Removed:
    """A folder removed (or, with ``dry_run``, that would be)."""

    path: Path
    bytes: int
    last_used: float  # seconds since the epoch

    def used_at(self) -> str:
        """``last_used`` as a local ISO time (``2026-11-18T09:30:00+01:00``)."""
        return datetime.fromtimestamp(self.last_used).astimezone().isoformat(timespec="seconds")


@dataclass
class Report:
    """What :func:`prune` did."""

    cache: Path
    days: int
    include_projects: bool
    dry_run: bool
    size: int | None = None  # bytes in the cache folder before (None when it is missing)
    removed: list[Removed] = field(default_factory=list)
    kept: dict[str, int] = field(default_factory=lambda: {RECENT: 0, PROJECT: 0, IN_USE: 0})
    failed: list[Path] = field(default_factory=list)
    leftovers: int = 0  # bytes of unfinished earlier prunes removed

    @property
    def freed(self) -> int:
        return sum(r.bytes for r in self.removed) + self.leftovers


def human_size(n: int) -> str:
    """``5.6 GB``, ``120 MB`` below 1 GB, ``0 MB`` for an empty folder."""
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _is_link(entry: os.DirEntry[str]) -> bool:
    """A symlink, or a junction on Windows: never followed."""
    return entry.is_symlink() or entry.is_junction()


def tree_stats(folder: Path) -> tuple[int, float, bool]:
    """``(bytes, newest modification time, read in full)`` of ``folder`` and everything in
    it; symlinks and junctions count as themselves and are not followed."""
    try:
        st = folder.lstat()
    except OSError:
        return 0, 0.0, False
    total, newest, complete = 0, st.st_mtime, True
    todo = [folder]
    while todo:
        try:
            with os.scandir(todo.pop()) as it:
                entries = list(it)
        except OSError:
            complete = False
            continue
        for e in entries:
            try:
                st = e.stat(follow_symlinks=False)
            except OSError:
                complete = False
                continue
            newest = max(newest, st.st_mtime)
            if e.is_dir(follow_symlinks=False) and not _is_link(e):
                todo.append(Path(e.path))
            else:
                total += st.st_size
    return total, newest, complete


def cache_size() -> int | None:
    """Bytes in the cache folder (:func:`cutpoint.paths.cache_dir`), or None when it is
    missing. Creates nothing."""
    root = paths.cache_dir()
    if not root.is_dir():
        return None
    return tree_stats(root)[0]


def _project_sources(project: Path) -> list[str]:
    """The source paths a project's files name (shapes they do not have are skipped)."""
    out: list[str] = []
    hl = cache.load_json(project / "highlights.json")
    if isinstance(hl, dict) and isinstance(hl.get("sources"), list):
        out += [s.get("path") for s in hl["sources"] if isinstance(s, dict)]
    rec = cache.load_json(project / runstate.FILE)
    if isinstance(rec, dict) and isinstance(rec.get("sources"), list):
        out += rec["sources"]
    pano = cache.load_json(project / "pano.json")
    if isinstance(pano, dict) and isinstance(pano.get("clips"), list):
        for c in pano["clips"]:
            if isinstance(c, dict):
                out.append(c.get("lrv"))
                out += c.get("insv") if isinstance(c.get("insv"), list) else []
    meta = cache.load_json(project / "ui" / "meta.json")  # cutpoint.ui.files.picked_inputs
    if isinstance(meta, dict):
        out += meta["inputs"] if isinstance(meta.get("inputs"), list) else [meta.get("source")]
    for manifest in sorted(project.glob("clips*/manifest.json")):  # clips --segments picks too
        data = cache.load_json(manifest)
        if isinstance(data, dict) and isinstance(data.get("clips"), list):
            out += [c.get("source") for c in data["clips"] if isinstance(c, dict)]
    return [s for s in dict.fromkeys(out) if isinstance(s, str) and s]


def _source_folders(src: str) -> tuple[set[str], str | None]:
    """``(cache folder names of src and its recovered parts, None)``, or ``(set(), the name
    stem its folders start with)`` when ``src`` cannot be read (every such folder is kept)."""
    from cutpoint.sources import MAX_PARTS, READ_ERRORS_FILE  # noqa: PLC0415

    path = Path(src)
    names: set[str] = set()
    offset = 0.0
    try:
        for _ in range(MAX_PARTS):  # as cutpoint.sources.parts
            folder = cache.source_folder(path, offset)
            names.add(folder.name)
            nxt = (cache.load_json(folder / READ_ERRORS_FILE) or {}).get("resumed_at")
            if not isinstance(nxt, int | float) or nxt <= offset:
                break
            offset = float(nxt)
    except (OSError, ValueError):  # missing or unreadable (ValueError: a bad path)
        return names, path.stem[:40]
    return names, None


@dataclass
class _Owners:
    names: dict[str, set[Path]] = field(default_factory=dict)  # folder name -> its projects
    stems: dict[str, set[Path]] = field(default_factory=dict)  # stem of unreadable sources
    busy: set[Path] = field(default_factory=set)  # projects with a run going when listed

    def projects(self, name: str) -> set[Path]:
        """The projects owning the cache folder ``name``."""
        out = set(self.names.get(name, ()))
        for stem, projects in self.stems.items():
            if re.fullmatch(re.escape(stem) + r"(@\d+)?-[0-9a-f]{16}", name):
                out |= projects
        return out

    def owns(self, name: str, busy: bool = False) -> bool:
        projects = self.projects(name)
        return bool(projects & self.busy) if busy else bool(projects)

    def running(self, name: str) -> bool:
        """Whether a run of a project owning ``name`` holds its run lock now."""
        return any(folders.lock_held(p / runstate.LOCK_NAME) for p in self.projects(name))


def _owners() -> _Owners:
    """The cache folders of the projects in the projects folder."""
    out = _Owners()
    try:
        projects = sorted(d for d in projects_dir().iterdir() if d.is_dir())
    except OSError:
        return out
    for d in projects:
        for src in _project_sources(d):
            got, stem = _source_folders(src)
            for name in got:
                out.names.setdefault(name, set()).add(d)
            if stem:
                out.stems.setdefault(stem, set()).add(d)
        if folders.lock_held(d / runstate.LOCK_NAME):
            out.busy.add(d)
    return out


def _locked(folder: Path) -> bool:
    """Whether a run holds a lock in ``folder`` (``.<name>.cutpoint-lock``)."""
    try:
        with os.scandir(folder) as it:
            locks = [e.name for e in it if e.name.startswith(".") and e.name.endswith(LOCK_SUFFIX)]
    except OSError:
        return True  # cannot tell: in use
    return any(folders.lock_held(folder / n[1 : -len(LOCK_SUFFIX)]) for n in locks)


def _rmtree(folder: Path) -> list[str]:
    """Remove ``folder`` (no symlink followed); the errors, empty when all of it went. A
    read-only file (Windows) is made writable and tried again, and a locked one retried
    briefly (:func:`cutpoint.cache.retry_locked`)."""
    errors: list[str] = []

    def onexc(func: Any, path: str, err: BaseException) -> None:
        if isinstance(err, FileNotFoundError):
            return
        if func not in (os.unlink, os.rmdir) or not isinstance(err, PermissionError):
            errors.append(f"{path}: {getattr(err, 'strerror', None) or err}")
            return
        try:
            if os.name == "nt":  # a read-only file cannot be deleted there
                with contextlib.suppress(OSError):
                    os.chmod(path, stat.S_IWRITE)  # noqa: PTH101
            cache.retry_locked(lambda: func(path), Path(path), "delete")
        except OSError as again:
            errors.append(f"{path}: {again.strerror or again}")

    shutil.rmtree(folder, onexc=onexc)
    return errors


def _used_since(folder: Path, cutoff: float) -> bool:
    """Whether ``folder`` itself was marked used (:func:`cutpoint.cache.mark_used`) or
    written into since ``cutoff``, or a lock in it is held; True if it cannot be read."""
    try:
        return folder.lstat().st_mtime > cutoff or _locked(folder)
    except OSError:
        return True


def _remove(folder: Path, report: Report, cutoff: float, owners: _Owners) -> Removed | None:
    """Remove the cache folder ``folder`` if it is still unused since ``cutoff``; the
    removed folder, or None (it was used meanwhile, or a warning says why it stays).

    A run can mark it used at any moment, so the cheap checks are made again last thing
    before the rename, and once more after it: a folder used meanwhile is renamed back."""
    size, newest, complete = tree_stats(folder)
    if (
        not complete
        or newest > cutoff
        or _locked(folder)
        or _used_since(folder, cutoff)  # the walk above takes a while
        or owners.running(folder.name)
    ):
        report.kept[IN_USE] += 1
        return None
    trash = folder.with_name(f".{folder.name}.{uuid.uuid4().hex[:8]}.cutpoint-prune")
    try:
        cache.retry_locked(lambda: folder.rename(trash), folder, "rename")
    except OSError as err:
        _failed(report, folder, f"it stays ({err.strerror or err})")
        return None
    if _used_since(trash, cutoff) or owners.running(folder.name):
        try:  # marked used just before the rename: give it back
            cache.retry_locked(lambda: trash.rename(folder), trash, "rename")
        except OSError as err:  # a run made the folder again meanwhile: it starts afresh
            _failed(
                report,
                folder,
                f"it was in use, but {trash.name} cannot go back ({err.strerror or err})",
            )
            return None
        report.kept[IN_USE] += 1
        return None
    errors = _rmtree(trash)
    if errors:
        _failed(report, folder, f"part of it is left in {trash.name} ({errors[0]})")
    return Removed(folder, size, newest)


def _failed(report: Report, folder: Path, why: str) -> None:
    message = f"Could not remove the cache folder {folder}: {why}"
    logger.warning("%s", message)
    report.failed.append(folder)
    jsonout.warn("cache_remove_failed", message, [folder])


@contextlib.contextmanager
def _prune_lock(root: Path, dry_run: bool) -> Iterator[None]:
    if dry_run:
        yield
        return
    with folders.swap_lock(root / PRUNE_LOCK):
        yield


def prune(
    days: int, *, include_projects: bool = False, dry_run: bool = False, now: float | None = None
) -> Report:
    """Remove the source cache folders not used for ``days`` days, least recently used first
    (see the module docstring for what is kept).

    Args:
        days: Days without use (1 or more).
        include_projects: Also remove folders of projects in the projects folder (never one
            in use).
        dry_run: Only report what would be removed.
        now: The current time (seconds since the epoch; tests).

    Raises:
        ValueError: If ``days`` is below 1.
        FolderBusyError: If another prune is running.
    """
    if days < 1:
        raise ValueError(f"days must be 1 or more, not {days}")
    root = paths.cache_dir()
    report = Report(root, days, include_projects, dry_run)
    if not root.is_dir():
        return report
    cutoff = (time.time() if now is None else now) - days * DAY_S
    with _prune_lock(root, dry_run):
        owners = _owners()
        report.size = tree_stats(root)[0]
        found: list[tuple[float, Path]] = []
        with os.scandir(root) as it:
            entries = sorted(it, key=lambda e: e.name)
        for e in entries:
            if not e.is_dir(follow_symlinks=False) or _is_link(e):
                continue
            path = Path(e.path)
            if LEFTOVER.fullmatch(e.name):
                if not dry_run:
                    size = tree_stats(path)[0]
                    if errors := _rmtree(path):
                        _failed(report, path, errors[0])
                    else:
                        report.leftovers += size
                continue
            if not FOLDER_NAME.fullmatch(e.name):
                continue
            _, newest, complete = tree_stats(path)
            if not complete or owners.owns(e.name, busy=True) or _locked(path):
                report.kept[IN_USE] += 1
            elif newest > cutoff:
                report.kept[RECENT] += 1
            elif not include_projects and owners.owns(e.name):
                report.kept[PROJECT] += 1
            else:
                found.append((newest, path))
        for newest, path in sorted(found, key=lambda f: f[0]):  # least recently used first
            if dry_run:
                report.removed.append(Removed(path, tree_stats(path)[0], newest))
            elif (done := _remove(path, report, cutoff, owners)) is not None:
                report.removed.append(done)
        if not dry_run and report.removed:
            _drop_timeline_entries(root, {os.path.normcase(r.path.name) for r in report.removed})
    return report


def _drop_timeline_entries(root: Path, removed: set[str]) -> None:
    """Drop the entries of tsrepair's timeline index (:data:`cutpoint.stage.TIMELINE_INDEX`)
    whose copy was in one of the ``removed`` cache folders (names as ``os.path.normcase``
    gives them: an entry's case can differ from the disk's on Windows) and is still missing (a run may
    have made it again meanwhile). Under the index's lock; when another run holds it, the
    index stays as it is (readers reject an entry whose copy is missing), as when it cannot be
    written."""
    path = root / stage.TIMELINE_INDEX
    if not path.is_file():
        return
    try:
        with stage.index_lock(path):
            data = cache.load_json(path)
            if not isinstance(data, dict):
                return
            keep = {
                k: v
                for k, v in data.items()
                if not (
                    isinstance(v, dict)
                    and isinstance(file := v.get("file"), str)
                    # as tsrepair writes it (posix)
                    and os.path.normcase(file.split("/", 1)[0]) in removed
                    and not os.path.lexists(root / file)
                )
            }
            if len(keep) < len(data):
                cache.save_json(path, keep)
                logger.debug("timeline index: %d entries dropped", len(data) - len(keep))
    except (FolderBusyError, OSError) as err:
        logger.debug("timeline index not cleaned (%s)", err)
