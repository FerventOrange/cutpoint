"""Delete a project's own files (``cutpoint ui``, Settings > Delete this project).

Only what Cutpoint wrote goes: the result files and folders of a project folder (:data:`OWN_FILES`,
:data:`OWN_DIRS`, the finals and their subtitles and timelines, the hidden temporary, backup,
resume and lock names of a run) and, in each clips folder whose ``manifest.json`` Cutpoint wrote,
the clips it lists, their ``.srt`` and the manifest; in each ``<clips>_qa`` folder the QA report
and sheets. Everything else stays: staged footage copies, clips added by hand, HUD presets
(``hud_presets``, which a user may have calibrated), unknown files and folders, and anything that
is, or holds, a source video of a project. A symlink or junction is never followed; it stays (and
with it the folder that holds it). The project folder itself goes only once it is empty.

The files go to the Recycle Bin / Trash. ``send2trash`` deletes for good on Windows when an item
cannot be recycled (network, removable and ``subst`` drives, a bin set to delete at once, an item over the
bin's size limit) and in a container there is no trash: :func:`trash_blocker` finds those cases
first, and :func:`remove_items` then raises :class:`NoTrash` instead, so the page can ask.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import send2trash

from cutpoint import cache, folders, planning, runstate

UI_DIR = "ui"  # cutpoint.ui.UI_DIR
OWN_FILES = frozenset(
    {
        "highlights.json",
        "highlights.pre-events.json",
        "transcript.txt",
        "events.json",
        "events.txt",
        "progress.json",
        "activity.json",
        "crashes.json",
        "telemetry.json",
        "mix_segments.json",
        "mix_report.json",
        "mix_sheet.jpg",
        "mix_exclude.json",
        "pano.json",
        "pano_timings.json",
    }
)
OWN_DIRS = frozenset(
    {
        UI_DIR,
        "crash_check",
        "visual_check",
        "action_cam_features",
        "discovery",
        "pano_paths",
        "pano_candidates",
    }
)
# Kept with a reason: discovered presets (``cutpoint events``) that the user is told to edit
KEPT_DIRS = {"hud_presets": "HUD presets you may have edited"}
# ``assemble``'s final.* / final_<x>.* and ``edit``'s <name>-final.*, with their subtitles
# (.srt, .ass) and timelines (.fcpxml, .otio) of the same name
FINAL_FILE = re.compile(r"(?:[^.].*-)?final(?:_.*)?\.(?:mp4|fcpxml|otio|srt|ass)")
FINAL_VIDEO = re.compile(r".*\.(?:mp4|fcpxml|otio)")
# Hidden names a run leaves in the project folder: .assemble-tmp-<id>, the staging, backup,
# resume and lock names of folders.py (.<name>.cutpoint-tmp|bak|resume|lock) and .<stem>.part.<ext>
HIDDEN_OWN = re.compile(
    r"\.assemble-tmp-[^/\\]+|\.[^/\\]+\.cutpoint-(?:tmp|bak|resume|lock)|\.[^/\\]+\.part\.\w+"
)
LOCK_FILE = re.compile(r"\.(.+)\.cutpoint-lock")
RUN_LOCK = f".{runstate.LOCK_NAME}.cutpoint-lock"  # held by the delete itself, never in the plan
# What ``360 qa`` writes into ``<clips>_qa`` (cutpoint.pano.qa, qa_compare)
QA_FILE = re.compile(r"qa\.json|edge_gap\.json|(?:flagged|overview)_\d+\.jpg")
DRIVE_FIXED = 3  # GetDriveTypeW


class NoTrash(OSError):
    """The Recycle Bin / Trash cannot take the files (none on this drive or here, say); nothing
    is lost: the files are still there, or in the trash."""


@dataclass
class Plan:
    """What deleting a project removes and keeps."""

    items: list[Path] = field(default_factory=list)  # files and folders to remove, each as one
    sizes: list[int] = field(default_factory=list)  # bytes inside each of ``items``
    files: int = 0  # files inside ``items``
    kept: list[str] = field(default_factory=list)  # what stays, with its reason when it has one
    deleted: list[str] = field(default_factory=list)  # names of the finals and timelines that go

    @property
    def size(self) -> int:
        return sum(self.sizes)


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _walk(path: Path) -> tuple[list[Path], list[Path]]:
    """``(paths to remove, paths kept)`` for ``path``.

    A folder with nothing kept inside goes as one path; otherwise its removable children go
    one by one. A link is kept, never followed."""
    if _is_link(path):
        return [], [path]
    try:
        if path.is_file():
            return [path], []
        if not path.is_dir():
            return [], [path]
        children = sorted(path.iterdir())
    except OSError:
        return [], [path]
    items: list[Path] = []
    kept: list[Path] = []
    for child in children:
        i, k = _walk(child)
        items += i
        kept += k
    return ([path] if not kept else items), kept


def _measure(path: Path) -> tuple[int, int]:
    """``(files, bytes)`` inside ``path`` (a file, or a folder with no link in it)."""
    try:
        if path.is_file():
            return 1, path.lstat().st_size
        children = list(path.iterdir())
    except OSError:
        return 0, 0
    files = size = 0
    for child in children:
        f, s = _measure(child)
        files += f
        size += s
    return files, size


def _plain(name: object) -> str | None:
    """``name`` when it is a bare file name (no folder part), else None."""
    return name if isinstance(name, str) and name and Path(name).name == name else None


def _only(folder: Path, names: set[str], pattern: re.Pattern[str] | None = None):
    """:func:`_walk` for a folder of which only the files in ``names`` (or matching ``pattern``)
    go; the folder goes as one path when nothing else is in it."""
    try:
        children = sorted(folder.iterdir())
    except OSError:
        return [], [folder]
    items: list[Path] = []
    kept: list[Path] = []
    for child in children:
        if child.name in names or (pattern is not None and pattern.fullmatch(child.name)):
            i, k = _walk(child)
            items += i
            kept += k
        else:
            kept.append(child)
    return ([folder] if not kept else items), kept


def _clips_folder(folder: Path) -> tuple[list[Path], list[Path]]:
    """:func:`_walk` for a folder that may hold clips: only the files its ``manifest.json``
    lists (and their ``.srt``) and the manifest go, and only when Cutpoint wrote the manifest. A
    ``<clips>_qa`` folder of ``360 qa`` loses its report and sheets."""
    manifest = folder / "manifest.json"
    if folder.name.endswith("_qa") and (folder / "qa.json").is_file():
        return _only(folder, set(), QA_FILE)
    if (
        not manifest.is_file()
        or _is_link(manifest)
        or planning.foreign_manifest(folder) is not None
    ):
        return [], [folder]
    data = cache.load_json(manifest)
    listed = data.get("clips") if isinstance(data, dict) else None
    names = {"manifest.json"}
    for entry in listed if isinstance(listed, list) else []:
        if not isinstance(entry, dict) or (file := _plain(entry.get("file"))) is None:
            continue
        names |= {file, Path(file).with_suffix(".srt").name}
        if (srt := _plain(entry.get("srt"))) is not None:
            names.add(srt)
    return _only(folder, names)


def _owner(item: Path, protected: dict[str, str]) -> str | None:
    """The project that reads a source video that ``item`` is or holds (keys of ``protected``
    are normcased real paths), else None."""
    try:
        key = source_key(str(item))
    except OSError:
        return None
    prefix = key.rstrip(os.sep) + os.sep
    return next(
        (who for src, who in protected.items() if src == key or src.startswith(prefix)), None
    )


def make_plan(project: Path, protected: dict[str, str] | None = None) -> Plan:
    """What deleting ``project`` removes and what stays (nothing is touched).

    ``protected`` maps the normcased real path of each source video or folder that a project
    reads to that project's name: an entry that is or holds one of them stays."""
    plan = Plan()
    kept: list[tuple[Path, str | None]] = []
    for entry in sorted(project.iterdir()):
        if entry.name == RUN_LOCK:
            continue
        if _is_link(entry):
            items, left = [], [entry]
        elif entry.is_dir():
            if entry.name in KEPT_DIRS:
                kept.append((entry, KEPT_DIRS[entry.name]))
                continue
            items, left = (
                _walk(entry)
                if entry.name in OWN_DIRS or HIDDEN_OWN.fullmatch(entry.name)
                else _clips_folder(entry)
            )
        elif (
            entry.name in OWN_FILES
            or FINAL_FILE.fullmatch(entry.name)
            or HIDDEN_OWN.fullmatch(entry.name)
        ):
            items, left = _walk(entry)
        else:
            items, left = [], [entry]
        kept += [(k, None) for k in left]
        for item in items:
            if who := _owner(item, protected or {}):
                kept.append((item, f"used as a source by project {who}"))
            else:
                plan.items.append(item)
    for item in plan.items:
        files, size = _measure(item)
        plan.files += files
        plan.sizes.append(size)
        if item.parent == project and FINAL_VIDEO.fullmatch(item.name):
            plan.deleted.append(item.name)
    for path, why in kept:
        name = path.relative_to(project).as_posix() + ("/" if path.is_dir() else "")
        plan.kept.append(f"{name} ({why})" if why else name)
    return plan


def locked(project: Path) -> bool:
    """A run holds the project's lock or one of its clips folders' locks (a stale lock of a
    process that is gone does not count)."""
    if folders.lock_held(project / runstate.LOCK_NAME):
        return True
    try:
        names = [e.name for e in project.iterdir()]
    except OSError:
        return False
    return any(
        (m := LOCK_FILE.fullmatch(n)) is not None and folders.lock_held(project / m.group(1))
        for n in names
    )


# --- is there a trash that keeps what it is given?


def in_container() -> bool:
    """Running in a container (Docker, Podman): no desktop trash there."""
    return (
        sys.platform != "win32"
        and (Path("/.dockerenv").exists() or Path("/run/.containerenv").exists())
    ) or bool(os.environ.get("container"))  # noqa: SIM112 - the name podman and systemd set


def _drive_type(root: str) -> int:
    import ctypes  # noqa: PLC0415

    return ctypes.windll.kernel32.GetDriveTypeW(root)  # type: ignore[attr-defined]


def _dos_target(drive: str) -> str:
    """Where the drive letter ``drive`` (``X:``) points: ``\\Device\\HarddiskVolumeN`` for a
    volume, ``\\??\\...`` for a ``subst`` drive. Empty when Windows cannot say."""
    import ctypes  # noqa: PLC0415

    buf = ctypes.create_unicode_buffer(1024)
    if not ctypes.windll.kernel32.QueryDosDeviceW(drive, buf, 1024):  # type: ignore[attr-defined]
        return ""
    return buf.value


def _mount_root(path: str) -> str:
    """The mount point of the volume holding ``path`` (``C:\\``, or a folder a volume is mounted
    on)."""
    import ctypes  # noqa: PLC0415

    buf = ctypes.create_unicode_buffer(1024)
    if not ctypes.windll.kernel32.GetVolumePathNameW(path, buf, 1024):  # type: ignore[attr-defined]
        raise OSError(f"no volume path for {path}")
    return buf.value


def _volume_guid(root: str) -> str:
    """``{GUID}`` of the volume mounted at ``root`` (``C:\\``)."""
    import ctypes  # noqa: PLC0415

    buf = ctypes.create_unicode_buffer(64)
    if not ctypes.windll.kernel32.GetVolumeNameForVolumeMountPointW(root, buf, 64):  # type: ignore[attr-defined]
        raise OSError(f"no volume name for {root}")
    m = re.search(r"\{[0-9A-Fa-f-]+\}", buf.value)
    if m is None:
        raise OSError(f"unexpected volume name {buf.value!r}")
    return m.group(0)


def _volume_total_mb(root: str) -> int:
    """Size of the volume at ``root`` in MB."""
    import ctypes  # noqa: PLC0415

    total = ctypes.c_ulonglong()
    if not ctypes.windll.kernel32.GetDiskFreeSpaceExW(root, None, ctypes.byref(total), None):  # type: ignore[attr-defined]
        raise OSError(f"cannot read the size of {root}")
    return total.value // 2**20


def _bin_used(root: str) -> int:
    """Bytes the Recycle Bin of the volume at ``root`` already holds."""
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    class Info(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("i64Size", ctypes.c_longlong),
            ("i64NumItems", ctypes.c_longlong),
        ]

    info = Info(ctypes.sizeof(Info))
    if ctypes.windll.shell32.SHQueryRecycleBinW(root, ctypes.byref(info)) != 0:  # type: ignore[attr-defined]
        raise OSError(f"cannot read the Recycle Bin of {root}")
    return info.i64Size


def _bin_settings(guid: str) -> tuple[int, int | None]:
    """``(NukeOnDelete, MaxCapacity in MB or None)`` the Recycle Bin of the volume ``guid`` is
    set to (None: no capacity of its own, Windows' default applies).

    Raises:
        FileNotFoundError: The volume has no settings key of its own: Windows' defaults apply.
    """
    import winreg  # noqa: PLC0415

    key = rf"Software\Microsoft\Windows\CurrentVersion\Explorer\BitBucket\Volume\{guid}"
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as k:  # type: ignore[attr-defined]
        try:
            nuke = int(winreg.QueryValueEx(k, "NukeOnDelete")[0])  # type: ignore[attr-defined]
        except FileNotFoundError:
            nuke = 0
        try:
            return nuke, int(winreg.QueryValueEx(k, "MaxCapacity")[0])  # type: ignore[attr-defined]
        except FileNotFoundError:
            return nuke, None


def _policy() -> tuple[bool, int | None]:
    """``(NoRecycleFiles, RecycleBinSize in percent or None)`` set by Group Policy (HKCU and HKLM
    ``...\\Policies\\Explorer``)."""
    import winreg  # noqa: PLC0415

    no_recycle, percent = False, None
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):  # type: ignore[attr-defined]
        try:
            with winreg.OpenKey(
                hive, r"Software\Microsoft\Windows\CurrentVersion\Policies\Explorer"
            ) as k:  # type: ignore[attr-defined]
                for name in ("NoRecycleFiles", "RecycleBinSize"):
                    try:
                        value = int(winreg.QueryValueEx(k, name)[0])  # type: ignore[attr-defined]
                    except FileNotFoundError:
                        continue
                    if name == "NoRecycleFiles":
                        no_recycle = no_recycle or value == 1
                    else:
                        percent = value
        except FileNotFoundError:
            continue
    return no_recycle, percent


def _default_capacity_mb(total_mb: int) -> int:
    """Windows' default Recycle Bin size for a volume of ``total_mb``: 10% up to 40 GB, then 5%
    of the rest, from 4 GB."""
    return total_mb // 10 if total_mb <= 40960 else 4096 + (total_mb - 40960) * 5 // 100


def windows_blocker(items: list[tuple[Path, int]]) -> str | None:
    """Why the Recycle Bin might not keep ``items`` (path, bytes), else None.

    Windows deletes for good what it cannot recycle: on a network, removable or ``subst`` drive, with
    ``NukeOnDelete`` or a Group Policy that turns the bin off, and when one operation holds more
    than the bin's size limit (it empties the oldest entries). Per volume the sum must fit in
    90% of the limit, less what the bin already holds. Unknown counts as a reason."""
    try:
        sizes: dict[str, int] = {}
        for path, size in items:
            drive = os.path.splitdrive(os.path.abspath(path))[0]  # noqa: PTH100
            if len(drive) == 2 and drive[1] == ":" and _dos_target(drive).startswith("\\??\\"):
                # realpath would show the real folder, but the Shell gets the X: paths and
                # recycles nothing there
                return "it is on a substituted drive (subst), which has no Recycle Bin"
            real = os.path.realpath(path)
            if real.startswith("\\\\"):
                return "it is on a network share, which has no Recycle Bin"
            root = _mount_root(real)
            sizes[root] = sizes.get(root, 0) + size
        no_recycle, percent = _policy()
        for root, total in sizes.items():
            if _drive_type(root) != DRIVE_FIXED:
                return f"{root} is not a fixed drive, so it has no Recycle Bin"
            if no_recycle:
                return "a Group Policy turns the Recycle Bin off"
            guid = _volume_guid(root)
            try:
                nuke, capacity_mb = _bin_settings(guid)
            except FileNotFoundError:
                nuke, capacity_mb = 0, None
            if nuke:
                return f"the Recycle Bin of {root} is set to delete files at once"
            if capacity_mb is None:
                capacity_mb = _default_capacity_mb(_volume_total_mb(root))
            if percent is not None:
                capacity_mb = _volume_total_mb(root) * percent // 100
            room = capacity_mb * 2**20 * 9 // 10 - _bin_used(root)
            if total > room:
                return f"the files are bigger than the room in the Recycle Bin of {root}"
    except (OSError, ValueError, AttributeError) as err:
        return f"Cutpoint cannot tell whether the Recycle Bin will keep the files ({err})"
    return None


def source_key(path: str) -> str:
    """Normcased path for comparing ``path`` with project items: resolved (``realpath``) only on
    a local fixed drive; a network or removable path is never resolved, which could hang."""
    absolute = os.path.abspath(path)  # noqa: PTH100 - Path.resolve() would follow links
    if sys.platform == "win32":
        drive = os.path.splitdrive(absolute)[0]
        try:
            local = (
                bool(drive)
                and not drive.startswith("\\\\")
                and _drive_type(drive + "\\") == DRIVE_FIXED
            )
        except (OSError, AttributeError):
            local = False
        if not local:
            return os.path.normcase(absolute)
    return os.path.normcase(os.path.realpath(absolute))


def trash_blocker(items: list[tuple[Path, int]]) -> str | None:
    """Why ``items`` (path, bytes) would not reach a trash that keeps them here, else None."""
    if in_container():
        return "there is no trash in a container"
    return windows_blocker(items) if sys.platform == "win32" else None


def remove_items(plan: Plan, *, permanent: bool) -> None:
    """Remove ``plan``'s paths to the Recycle Bin / Trash, or for good with ``permanent``.

    The trash is one ``send2trash`` call for all of them, guarded first by
    :func:`trash_blocker`, which sums the sizes per volume.

    Raises:
        NoTrash: The trash cannot be trusted with them (:func:`trash_blocker`) or refused them
            (``TrashPermissionError``; on macOS the plain error for a volume without a trash);
            checked before any is moved, so nothing is lost.
        OSError: A path could not be deleted (a file in use, say); other items may be gone.
    """
    items = [(p, s) for p, s in zip(plan.items, plan.sizes, strict=True) if os.path.lexists(p)]
    if permanent:
        for item, _ in items:
            if item.is_dir() and not _is_link(item):
                folders._remove_leftover(item, "part of the project being deleted")
            else:
                item.unlink()
        return
    if why := trash_blocker(items):
        raise NoTrash(why)
    if not items:
        return
    try:
        send2trash.send2trash([str(p) for p, _ in items])
    except send2trash.TrashPermissionError as err:
        raise NoTrash(f"the trash refused {err.filename}") from err
    except OSError as err:
        if (
            sys.platform == "darwin"
            and not isinstance(err, PermissionError)
            and "trash" in str(err).lower()
        ):
            raise NoTrash(str(err)) from err  # a volume without a trash
        raise


def remove_folder(project: Path) -> None:
    """Remove the project folder if nothing is left in it."""
    with contextlib.suppress(OSError):
        project.rmdir()
