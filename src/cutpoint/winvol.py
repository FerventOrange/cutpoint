"""Windows drive and volume queries (ctypes), in one place.

Every function is Windows only: it needs ``ctypes.windll`` and raises ``AttributeError`` on
other platforms (callers check ``sys.platform`` first, as before). The module itself imports
cleanly everywhere: ``ctypes`` is loaded inside the functions.
"""

from __future__ import annotations

import re
import sys

DRIVE_NO_ROOT = 1  # GetDriveTypeW; 0 is unknown
DRIVE_REMOVABLE = 2
DRIVE_FIXED = 3
DRIVE_REMOTE = 4

ERROR_MORE_DATA = 234
ERROR_CONNECTION_UNAVAIL = 1201  # remembered mapping, not connected; the name is still returned
ERROR_NOT_CONNECTED = 2250  # the letter is not a network drive


def drive_type(root: str) -> int:
    r"""``GetDriveTypeW`` of ``root`` (``C:\``)."""
    import ctypes  # noqa: PLC0415

    return ctypes.windll.kernel32.GetDriveTypeW(root)  # type: ignore[attr-defined]


def dos_target(drive: str) -> str:
    r"""Where the drive letter ``drive`` (``X:``) points: ``\Device\HarddiskVolumeN`` for a
    volume, ``\??\...`` for a ``subst`` drive. Empty when Windows cannot say."""
    import ctypes  # noqa: PLC0415

    buf = ctypes.create_unicode_buffer(1024)
    if not ctypes.windll.kernel32.QueryDosDeviceW(drive, buf, 1024):  # type: ignore[attr-defined]
        return ""
    return buf.value


def query_dos_device(drive: str) -> str | None:
    """:func:`dos_target`, but never raises; None on failure and on other platforms."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes  # noqa: PLC0415 - Windows only
    except ImportError:
        return None
    try:
        return dos_target(drive) or None
    except (OSError, AttributeError, ValueError, ctypes.ArgumentError):
        return None


def mount_root(path: str) -> str:
    r"""The mount point of the volume holding ``path`` (``C:\``, or a folder a volume is mounted
    on)."""
    import ctypes  # noqa: PLC0415

    buf = ctypes.create_unicode_buffer(1024)
    if not ctypes.windll.kernel32.GetVolumePathNameW(path, buf, 1024):  # type: ignore[attr-defined]
        raise OSError(f"no volume path for {path}")
    return buf.value


def volume_guid(root: str) -> str:
    r"""``{GUID}`` of the volume mounted at ``root`` (``C:\``)."""
    import ctypes  # noqa: PLC0415

    buf = ctypes.create_unicode_buffer(64)
    if not ctypes.windll.kernel32.GetVolumeNameForVolumeMountPointW(root, buf, 64):  # type: ignore[attr-defined]
        raise OSError(f"no volume name for {root}")
    m = re.search(r"\{[0-9A-Fa-f-]+\}", buf.value)
    if m is None:
        raise OSError(f"unexpected volume name {buf.value!r}")
    return m.group(0)


def volume_total_mb(root: str) -> int:
    """Size of the volume at ``root`` in MB."""
    import ctypes  # noqa: PLC0415

    total = ctypes.c_ulonglong()
    if not ctypes.windll.kernel32.GetDiskFreeSpaceExW(root, None, ctypes.byref(total), None):  # type: ignore[attr-defined]
        raise OSError(f"cannot read the size of {root}")
    return total.value // 2**20


def bin_used(root: str) -> int:
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


def network_connection(drive: str) -> tuple[int | None, str | None]:
    r"""``WNetGetConnectionW`` of ``drive`` (``Z:``): ``(rc, unc)``. ``unc`` is the
    ``\\server\share`` it names (also for a remembered, disconnected mapping), else None.
    ``rc`` is the last Windows error code, or None when the call could not be made.
    Never raises; ``(None, None)`` on other platforms."""
    if sys.platform != "win32":
        return None, None
    try:
        import ctypes  # noqa: PLC0415 - Windows only
    except ImportError:
        return None, None
    rc = None
    try:
        get_connection = ctypes.WinDLL("mpr").WNetGetConnectionW  # type: ignore[attr-defined]
        size = ctypes.c_ulong(1024)
        for _ in range(2):  # retry once with the size ERROR_MORE_DATA asked for
            buf = ctypes.create_unicode_buffer(size.value)
            rc = get_connection(drive, buf, ctypes.byref(size))
            if rc in (0, ERROR_CONNECTION_UNAVAIL) and buf.value.startswith("\\\\"):
                return rc, buf.value
            if rc != ERROR_MORE_DATA:
                break
    except (OSError, AttributeError, ValueError, ctypes.ArgumentError):
        pass
    return rc, None
