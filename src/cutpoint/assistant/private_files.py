"""Private filesystem permissions for proxy credentials and TLS identity files."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def _error(message: str) -> OSError:
    return PermissionError(message)


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
            raise _error("unsafe private path")
    return path


def _windows_api():
    """Load the Win32 APIs lazily so POSIX imports need no Windows modules."""
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
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
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
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


def _win_user_sid(ctypes, wintypes, kernel32, advapi32) -> str:
    token = wintypes.HANDLE()
    if not kernel32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        raise _error("cannot inspect current Windows user")
    token_buffer = ctypes.create_string_buffer(4096)
    needed = wintypes.DWORD()
    sid_text = wintypes.LPWSTR()
    try:
        if not advapi32.GetTokenInformation(
            token, 1, token_buffer, len(token_buffer), ctypes.byref(needed)
        ):
            raise _error("cannot inspect current Windows user")
        # TOKEN_USER begins with SID_AND_ATTRIBUTES; its first field is the SID pointer.
        sid = ctypes.cast(token_buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        if not sid or not advapi32.ConvertSidToStringSidW(sid, ctypes.byref(sid_text)):
            raise _error("cannot identify current Windows user")
        return sid_text.value
    finally:
        if sid_text:
            kernel32.LocalFree(ctypes.cast(sid_text, ctypes.c_void_p))
        kernel32.CloseHandle(token)


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
        raise _error("cannot verify private Windows permissions")
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
            raise _error("private Windows permissions were not applied")
        base = dacl.value
        acl_size = ctypes.cast(base + 2, ctypes.POINTER(wintypes.WORD))[0]
        if ctypes.cast(base + 4, ctypes.POINTER(wintypes.WORD))[0] != 1 or acl_size < 16:
            raise _error("private Windows permissions were not applied")
        offset = 8
        for expected_flags in (3 if directory else 0,):
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
                raise _error("private Windows permissions were not applied")
            try:
                if ace_sid.value != sid:
                    raise _error("private Windows permissions were not applied")
            finally:
                kernel32.LocalFree(ctypes.cast(ace_sid, ctypes.c_void_p))
            offset += size
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
            raise _error("cannot safely inspect private path")

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
                raise _error("cannot safely inspect private path")
            if info.attributes & 0x400 or not info.attributes & 0x10:
                raise _error("unsafe private path")
        finally:
            kernel32.CloseHandle(handle)


def _win_open(path: Path, directory: bool):
    ctypes, wintypes, kernel32, advapi32 = _windows_api()
    handle = kernel32.CreateFileW(
        str(path),
        0x00020000 | 0x00040000 | 0x00000080,
        0x1 | 0x2 | 0x4,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        if ctypes.get_last_error() in (2, 3):
            raise FileNotFoundError("private path no longer exists")
        raise _error("cannot safely open private path")

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
            raise _error("cannot inspect private path")
        if info.attributes & 0x400 or bool(info.attributes & 0x10) != directory:
            raise _error("unsafe private path type")
        if not directory and info.links != 1:
            raise _error("unsafe private file link count")
        sid = _win_user_sid(ctypes, wintypes, kernel32, advapi32)
        descriptor = ctypes.c_void_p()
        if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            f"D:P(A;OICI;FA;;;{sid})" if directory else f"D:P(A;;FA;;;{sid})",
            1,
            ctypes.byref(descriptor),
            None,
        ):
            raise _error("cannot create private Windows permissions")
        try:
            dacl = ctypes.c_void_p()
            present, defaulted = wintypes.BOOL(), wintypes.BOOL()
            if (
                not advapi32.GetSecurityDescriptorDacl(
                    descriptor, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)
                )
                or not present.value
            ):
                raise _error("cannot create private Windows permissions")

            current_owner = ctypes.c_void_p()
            sd = ctypes.c_void_p()
            status = advapi32.GetSecurityInfo(
                handle, 1, 1, ctypes.byref(current_owner), None, None, None, ctypes.byref(sd)
            )
            if status:
                raise _error("cannot inspect private Windows owner")
            try:
                owner_string = wintypes.LPWSTR()
                if not advapi32.ConvertSidToStringSidW(current_owner, ctypes.byref(owner_string)):
                    raise _error("cannot inspect private Windows owner")
                try:
                    if owner_string.value != sid:
                        raise _error("private Windows path has a different owner")
                finally:
                    kernel32.LocalFree(ctypes.cast(owner_string, ctypes.c_void_p))
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
                raise _error("cannot protect private Windows path")
            _win_verify_dacl(handle, directory, sid, ctypes, wintypes, kernel32, advapi32)
        finally:
            kernel32.LocalFree(descriptor)
        return handle, kernel32
    except BaseException:
        kernel32.CloseHandle(handle)
        raise


def private_directory(path: Path) -> None:
    """Create ``path`` if needed and restrict its access to the current user."""
    path = Path(path)
    if os.name == "nt":
        try:
            path = Path(path).absolute()
            _win_check_ancestors(path)
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            _win_check_ancestors(path)
            handle, kernel32 = _win_open(path, True)
            kernel32.CloseHandle(handle)
        except OSError:
            raise
        except Exception:  # noqa: BLE001 - sanitize unexpected native API failures.
            raise _error("cannot protect private Windows directory") from None
        return
    path = _posix_path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise _error("safe private paths are unsupported")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise _error("unsafe private directory")
        os.fchmod(fd, 0o700)
    finally:
        os.close(fd)


def private_file(path: Path) -> None:
    """Require an existing regular file and restrict its access to the current user."""
    path = Path(path)
    if os.name == "nt":
        try:
            path = Path(path).absolute()
            _win_check_ancestors(path)
            handle, kernel32 = _win_open(path, False)
            kernel32.CloseHandle(handle)
        except OSError:
            raise
        except Exception:  # noqa: BLE001 - sanitize unexpected native API failures.
            raise _error("cannot protect private Windows file") from None
        return
    path = _posix_path(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise _error("safe private paths are unsupported")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise _error("unsafe private file")
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)
