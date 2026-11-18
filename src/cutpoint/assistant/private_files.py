"""Private filesystem permissions for proxy credentials and TLS identity files."""

from __future__ import annotations

import contextlib
import os
import stat
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path


def _posix_path(path: Path) -> Path:
    path = Path(path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise PermissionError("unsafe private path")
    return path


def _windows_api():
    """Load the Win32 APIs lazily so POSIX imports need no Windows modules."""
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    kernel32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    kernel32.OpenProcessToken.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL

    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    advapi32.ConvertStringSidToSidW.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
    advapi32.GetSecurityDescriptorControl.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.WORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetSecurityDescriptorControl.restype = wintypes.BOOL
    advapi32.GetSecurityDescriptorDacl.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.BOOL),
    ]
    advapi32.GetSecurityDescriptorDacl.restype = wintypes.BOOL
    advapi32.GetSecurityInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetSecurityInfo.restype = wintypes.DWORD
    advapi32.SetSecurityInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    advapi32.SetSecurityInfo.restype = wintypes.DWORD
    return ctypes, wintypes, kernel32, advapi32


def _win_token_sids(ctypes, wintypes, kernel32, advapi32) -> tuple[str, str]:
    token = wintypes.HANDLE()
    if not kernel32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        raise PermissionError("cannot inspect current Windows user")
    try:
        sid_texts = []
        for info_class in (1, 4):  # TokenUser and TokenOwner
            buffer = ctypes.create_string_buffer(4096)
            needed = wintypes.DWORD()
            if not advapi32.GetTokenInformation(
                token, info_class, buffer, len(buffer), ctypes.byref(needed)
            ):
                raise PermissionError("cannot inspect current Windows user")
            sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
            sid_text = wintypes.LPWSTR()
            if not sid or not advapi32.ConvertSidToStringSidW(sid, ctypes.byref(sid_text)):
                raise PermissionError("cannot identify current Windows user")
            try:
                sid_texts.append(sid_text.value)
            finally:
                kernel32.LocalFree(ctypes.cast(sid_text, ctypes.c_void_p))
        return sid_texts[0], sid_texts[1]
    finally:
        kernel32.CloseHandle(token)


def _win_owner_needs_normalization(owner_sid: str, user_sid: str, token_owner_sid: str) -> bool:
    if owner_sid == user_sid:
        return False
    if owner_sid == token_owner_sid:
        return True
    raise PermissionError("private Windows path has a different owner")


def _win_verify_dacl(
    handle, directory: bool, sid: str, ctypes, wintypes, kernel32, advapi32
) -> None:
    dacl, descriptor = ctypes.c_void_p(), ctypes.c_void_p()
    status = advapi32.GetSecurityInfo(
        handle,
        1,
        4,
        None,
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(descriptor),
    )
    if status or not descriptor:
        raise PermissionError("cannot verify private Windows permissions")
    try:
        control, revision = wintypes.WORD(), wintypes.DWORD()
        present, defaulted = wintypes.BOOL(), wintypes.BOOL()
        if (
            not advapi32.GetSecurityDescriptorControl(
                descriptor,
                ctypes.byref(control),
                ctypes.byref(revision),
            )
            or not control.value & 0x1000
            or not advapi32.GetSecurityDescriptorDacl(
                descriptor,
                ctypes.byref(present),
                ctypes.byref(dacl),
                ctypes.byref(defaulted),
            )
            or not present.value
            or not dacl.value
        ):
            raise PermissionError("private Windows permissions were not applied")
        base = dacl.value
        acl_size = ctypes.cast(base + 2, ctypes.POINTER(wintypes.WORD))[0]
        if ctypes.cast(base + 4, ctypes.POINTER(wintypes.WORD))[0] != 1 or acl_size < 16:
            raise PermissionError("private Windows permissions were not applied")
        offset = 8
        expected_flags = 3 if directory else 0
        flags = ctypes.cast(base + offset + 1, ctypes.POINTER(ctypes.c_ubyte))[0]
        size = ctypes.cast(base + offset + 2, ctypes.POINTER(wintypes.WORD))[0]
        mask = ctypes.cast(base + offset + 4, ctypes.POINTER(wintypes.DWORD))[0]
        ace_sid = wintypes.LPWSTR()
        if (
            flags != expected_flags
            or mask != 0x1F01FF
            or ctypes.cast(base + offset, ctypes.POINTER(ctypes.c_ubyte))[0] != 0
            or size < 16
            or offset + size > acl_size
            or not advapi32.ConvertSidToStringSidW(base + offset + 8, ctypes.byref(ace_sid))
        ):
            raise PermissionError("private Windows permissions were not applied")
        try:
            if ace_sid.value != sid:
                raise PermissionError("private Windows permissions were not applied")
        finally:
            kernel32.LocalFree(ctypes.cast(ace_sid, ctypes.c_void_p))
    finally:
        kernel32.LocalFree(descriptor)


def _win_check_ancestors(path: Path) -> None:
    ctypes, wintypes, kernel32, _ = _windows_api()
    current = Path(path.anchor)
    for part in path.parts[1:-1]:
        current /= part
        handle = kernel32.CreateFileW(
            str(current),
            0x00000080,
            0x1 | 0x2 | 0x4,
            None,
            3,
            0x02000000 | 0x00200000,
            None,
        )
        if handle == wintypes.HANDLE(-1).value:
            if ctypes.get_last_error() in (2, 3):
                break
            raise PermissionError("cannot safely inspect private path")

        class _Info(ctypes.Structure):
            _fields_ = (
                ("attributes", wintypes.DWORD),
                ("creation_low", wintypes.DWORD),
                ("creation_high", wintypes.DWORD),
                ("access_low", wintypes.DWORD),
                ("access_high", wintypes.DWORD),
                ("write_low", wintypes.DWORD),
                ("write_high", wintypes.DWORD),
                ("volume", wintypes.DWORD),
                ("size_high", wintypes.DWORD),
                ("size_low", wintypes.DWORD),
                ("links", wintypes.DWORD),
                ("index_high", wintypes.DWORD),
                ("index_low", wintypes.DWORD),
            )

        try:
            info = _Info()
            if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(info)):
                raise PermissionError("cannot safely inspect private path")
            if info.attributes & 0x400 or not info.attributes & 0x10:
                raise PermissionError("unsafe private path")
        finally:
            kernel32.CloseHandle(handle)


def _win_open(path: Path, directory: bool):
    ctypes, wintypes, kernel32, advapi32 = _windows_api()
    # WRITE_OWNER keeps normalization on the same object whose type and owner are verified.
    handle = kernel32.CreateFileW(
        str(path),
        0x00020000 | 0x00040000 | 0x00080000 | 0x00000080,
        0x1 | 0x2 | 0x4,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        if ctypes.get_last_error() in (2, 3):
            raise FileNotFoundError("private path no longer exists")
        raise PermissionError("cannot safely open private path")

    class _Info(ctypes.Structure):
        _fields_ = (
            ("attributes", wintypes.DWORD),
            ("creation_low", wintypes.DWORD),
            ("creation_high", wintypes.DWORD),
            ("access_low", wintypes.DWORD),
            ("access_high", wintypes.DWORD),
            ("write_low", wintypes.DWORD),
            ("write_high", wintypes.DWORD),
            ("volume", wintypes.DWORD),
            ("size_high", wintypes.DWORD),
            ("size_low", wintypes.DWORD),
            ("links", wintypes.DWORD),
            ("index_high", wintypes.DWORD),
            ("index_low", wintypes.DWORD),
        )

    try:
        info = _Info()
        if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(info)):
            raise PermissionError("cannot inspect private path")
        if info.attributes & 0x400 or bool(info.attributes & 0x10) != directory:
            raise PermissionError("unsafe private path type")
        if not directory and info.links != 1:
            raise PermissionError("unsafe private file link count")
        sid, token_owner_sid = _win_token_sids(ctypes, wintypes, kernel32, advapi32)
        descriptor = ctypes.c_void_p()
        if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            f"D:P(A;OICI;FA;;;{sid})" if directory else f"D:P(A;;FA;;;{sid})",
            1,
            ctypes.byref(descriptor),
            None,
        ):
            raise PermissionError("cannot create private Windows permissions")
        try:
            dacl = ctypes.c_void_p()
            present, defaulted = wintypes.BOOL(), wintypes.BOOL()
            if (
                not advapi32.GetSecurityDescriptorDacl(
                    descriptor, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)
                )
                or not present.value
            ):
                raise PermissionError("cannot create private Windows permissions")

            current_owner = ctypes.c_void_p()
            sd = ctypes.c_void_p()
            status = advapi32.GetSecurityInfo(
                handle, 1, 1, ctypes.byref(current_owner), None, None, None, ctypes.byref(sd)
            )
            if status:
                raise PermissionError("cannot inspect private Windows owner")
            try:
                owner_string = wintypes.LPWSTR()
                if not advapi32.ConvertSidToStringSidW(current_owner, ctypes.byref(owner_string)):
                    raise PermissionError("cannot inspect private Windows owner")
                try:
                    normalize_owner = _win_owner_needs_normalization(
                        owner_string.value, sid, token_owner_sid
                    )
                finally:
                    kernel32.LocalFree(ctypes.cast(owner_string, ctypes.c_void_p))
            finally:
                kernel32.LocalFree(sd)

            if normalize_owner:
                owner_sid = ctypes.c_void_p()
                if not advapi32.ConvertStringSidToSidW(sid, ctypes.byref(owner_sid)):
                    raise PermissionError("cannot identify current Windows user")
                try:
                    status = advapi32.SetSecurityInfo(
                        handle, 1, 0x00000001, owner_sid, None, None, None
                    )
                    if status:
                        raise PermissionError("cannot set private Windows owner")
                finally:
                    kernel32.LocalFree(owner_sid)
                final_owner = ctypes.c_void_p()
                sd = ctypes.c_void_p()
                status = advapi32.GetSecurityInfo(
                    handle, 1, 1, ctypes.byref(final_owner), None, None, None, ctypes.byref(sd)
                )
                if status:
                    raise PermissionError("cannot verify private Windows owner")
                try:
                    final_owner_string = wintypes.LPWSTR()
                    if not advapi32.ConvertSidToStringSidW(
                        final_owner, ctypes.byref(final_owner_string)
                    ):
                        raise PermissionError("cannot verify private Windows owner")
                    try:
                        if final_owner_string.value != sid:
                            raise PermissionError("private Windows path has a different owner")
                    finally:
                        kernel32.LocalFree(ctypes.cast(final_owner_string, ctypes.c_void_p))
                finally:
                    kernel32.LocalFree(sd)

            status = advapi32.SetSecurityInfo(
                handle,
                1,
                0x00000004 | 0x80000000,
                None,
                None,
                dacl,
                None,
            )
            if status:
                raise PermissionError("cannot protect private Windows path")
            _win_verify_dacl(handle, directory, sid, ctypes, wintypes, kernel32, advapi32)
        finally:
            kernel32.LocalFree(descriptor)
        return handle, kernel32
    except BaseException:
        kernel32.CloseHandle(handle)
        raise


def _win_checked(path: Path, directory: bool) -> os.stat_result:
    """:func:`_win_open` ``path`` and stat it while the checked handle is still open."""
    handle, kernel32 = _win_open(path, directory)
    try:
        return os.lstat(path)
    finally:
        kernel32.CloseHandle(handle)


def private_directory(path: Path) -> os.stat_result:
    """Create ``path`` if needed and restrict its access to the current user; returns the stat
    of the directory checked (after its permissions were set)."""
    path = Path(path)
    if os.name == "nt":
        try:
            path = Path(path).absolute()
            _win_check_ancestors(path)
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            _win_check_ancestors(path)
            return _win_checked(path, True)
        except OSError:
            raise
        except Exception:  # noqa: BLE001 - sanitize unexpected native API failures.
            raise PermissionError("cannot protect private Windows directory") from None
    path = _posix_path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise PermissionError("safe private paths are unsupported")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise PermissionError("unsafe private directory")
        os.fchmod(fd, 0o700)
        return os.fstat(fd)
    finally:
        os.close(fd)


def private_file(path: Path) -> os.stat_result:
    """Require an existing regular file and restrict its access to the current user; returns
    the stat of the file checked (after its permissions were set)."""
    path = Path(path)
    if os.name == "nt":
        try:
            path = Path(path).absolute()
            _win_check_ancestors(path)
            return _win_checked(path, False)
        except OSError:
            raise
        except Exception:  # noqa: BLE001 - sanitize unexpected native API failures.
            raise PermissionError("cannot protect private Windows file") from None
    path = _posix_path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise PermissionError("safe private paths are unsupported")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise PermissionError("unsafe private file")
        os.fchmod(fd, 0o600)
        return os.fstat(fd)
    finally:
        os.close(fd)


def write_private(
    path: Path,
    data: str | bytes,
    *,
    directory: Callable[[Path], object] = private_directory,
    file: Callable[[Path], object] = private_file,
) -> None:
    """Write ``data`` (text as UTF-8, newlines as given) to the private file ``path``
    atomically: its folder is made private (``directory``) and an existing ``path`` is checked
    (``file``) first, then a temporary file in the same folder is made owner-only (``file``: mode
    0o600 on POSIX, the owner-only DACL on Windows) before any byte is written, flushed and
    fsync-ed, and replaces ``path`` (``Path.replace``), which is checked again; on POSIX the folder
    is fsync-ed too. The temporary file is removed when it can be, also when a step fails.

    ``directory`` / ``file``: the checks to run (the callers pass their module's names, so a
    test that replaces them is honoured, as for :class:`Checked`)."""
    path = Path(path)
    directory(path.parent)
    if path.exists() or path.is_symlink():
        file(path)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            file(temp)
            stream.write(data.encode("utf-8") if isinstance(data, str) else data)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
        file(path)
        if os.name != "nt":
            parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    finally:
        with contextlib.suppress(OSError):
            temp.unlink()


def stamp(info: os.stat_result) -> tuple[int, ...]:
    """What identifies a checked file or folder and changes when it is replaced, rewritten,
    linked or (on POSIX) has its mode or owner changed: device, inode (the file index on
    Windows), type and mode, owner, link count, size, modification time and, on POSIX, the
    status change time (a chmod or chown sets it; on Windows ``st_ctime`` is the creation
    time, and a DACL change sets no stat field)."""
    ctime = 0 if os.name == "nt" else info.st_ctime_ns
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        ctime,
    )


class Checked:
    """:func:`private_directory` and :func:`private_file` for reads, once per path while it
    stays the object that was checked (:func:`stamp`), for at most :attr:`MAX_AGE_S`. A path
    that is replaced, rewritten, re-linked or (on POSIX) chmod-ed or chown-ed since, or that
    cannot be stat-ed, is checked again before it is read; a check that fails is forgotten, so
    it runs again next time. The age limit bounds what no stat field shows (a Windows DACL
    widened by hand): it is made private again within five minutes.

    ``directory`` / ``file``: the checks to run (the callers pass their module's names, so a
    test that replaces them is honoured). A check that returns no ``os.stat_result`` (a fake)
    is never remembered. Thread-safe; the check itself runs outside the lock."""

    MAX_AGE_S = 300.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: dict[tuple[str, bool], tuple[tuple[int, ...], float]] = {}
        self._clock = time.monotonic  # tests pass a fake clock

    def check(self, path: Path, run: Callable[[Path], object], *, directory: bool) -> None:
        key = (str(path), directory)
        try:
            now = stamp(os.lstat(path))
        except OSError:
            now = None
        with self._lock:
            held = self._seen.get(key)
            if (
                now is not None
                and held is not None
                and held[0] == now
                and self._clock() - held[1] < self.MAX_AGE_S
            ):
                return
            self._seen.pop(key, None)
        info = run(path)
        if isinstance(info, os.stat_result):
            with self._lock:
                self._seen[key] = (stamp(info), self._clock())

    def clear(self) -> None:
        with self._lock:
            self._seen.clear()

    def __len__(self) -> int:
        return len(self._seen)
