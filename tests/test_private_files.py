"""Private proxy paths enforce current-user-only access on each platform."""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutpoint.assistant import private_files
from cutpoint.assistant.private_files import private_directory, private_file, write_private


@pytest.mark.parametrize(
    ("error_code", "expected"),
    [(2, FileNotFoundError), (3, FileNotFoundError), (5, PermissionError)],
)
def test_windows_open_distinguishes_missing_credentials_from_denied_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_code: int, expected: type[OSError]
) -> None:
    """Concurrent deletion must be distinguishable without hiding denied credential access."""
    invalid_handle = ctypes.c_void_p(-1).value
    monkeypatch.setattr(
        private_files,
        "_windows_api",
        lambda: (
            SimpleNamespace(get_last_error=lambda: error_code),
            SimpleNamespace(HANDLE=ctypes.c_void_p),
            SimpleNamespace(CreateFileW=lambda *args: invalid_handle),
            None,
        ),
    )
    with pytest.raises(expected):
        private_files._win_open(tmp_path / "credential.json", False)


def _fake_ancestor_api(monkeypatch: pytest.MonkeyPatch, refused: Path, failure: str | None) -> None:
    """Every ancestor opens as a folder, except `refused`, which fails at the chosen step."""
    opened: dict[int, str] = {}

    def create_file(path: str, *args: object) -> int:
        if failure == "create" and path == str(refused):
            return ctypes.c_void_p(-1).value
        opened[len(opened) + 1] = path
        return len(opened)

    def get_file_information(handle: int, info: object) -> int:
        if failure == "read" and opened[handle] == str(refused):
            return 0
        attributes = 0x10
        if opened[handle] == str(refused):
            attributes = {"reparse": 0x410, "file": 0x20}.get(failure or "", 0x10)
        info._obj.attributes = attributes  # type: ignore[attr-defined]
        return 1

    monkeypatch.setattr(
        private_files,
        "_windows_api",
        lambda: (
            SimpleNamespace(
                get_last_error=lambda: 5, byref=ctypes.byref, Structure=ctypes.Structure
            ),
            SimpleNamespace(HANDLE=ctypes.c_void_p, DWORD=ctypes.c_ulong),
            SimpleNamespace(
                CreateFileW=create_file,
                GetFileInformationByHandle=get_file_information,
                CloseHandle=lambda handle: 1,
            ),
            None,
        ),
    )


def test_windows_ancestor_error_names_the_refused_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Bug #1283: a refused ancestor (such as on a scratch drive) must be named in the error."""
    target = tmp_path / "scratch" / "credential.json"
    _fake_ancestor_api(monkeypatch, target.parent, "create")
    with (
        caplog.at_level(logging.DEBUG, logger="cutpoint.assistant.private_files"),
        pytest.raises(PermissionError, match="cannot safely inspect") as excinfo,
    ):
        private_files._win_check_ancestors(target)
    assert str(target.parent) in str(excinfo.value)
    assert str(excinfo.value) == (
        f'cannot safely inspect private path: cannot open "{target.parent}"'
    )
    assert "Windows error 5" in caplog.text


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("read", 'cannot safely inspect private path: cannot read "{folder}"'),
        ("reparse", 'unsafe private path: "{folder}" is a reparse point'),
        ("file", 'unsafe private path: "{folder}" is not a folder'),
    ],
)
def test_windows_ancestor_checks_name_the_refused_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, message: str
) -> None:
    """Bug #1283: each refused ancestor check names the folder that failed."""
    target = tmp_path / "scratch" / "credential.json"
    _fake_ancestor_api(monkeypatch, target.parent, failure)
    with pytest.raises(PermissionError) as excinfo:
        private_files._win_check_ancestors(target)
    assert str(excinfo.value) == message.format(folder=target.parent)


@pytest.mark.parametrize(
    ("owner_sid", "expected"),
    [("user", False), ("token-owner", True)],
)
def test_windows_normalizes_only_token_default_owner(owner_sid: str, expected: bool) -> None:
    """Only the current token's default owner group may be normalized to TokenUser."""
    assert (
        private_files._win_owner_needs_normalization(owner_sid, "user", "token-owner") is expected
    )


def test_windows_rejects_foreign_owner_without_takeover() -> None:
    with pytest.raises(PermissionError, match="different owner"):
        private_files._win_owner_needs_normalization("foreign", "user", "token-owner")


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_posix_directory_and_file_modes_are_private(tmp_path: Path) -> None:
    root = tmp_path / "private"
    private_directory(root)
    assert root.stat().st_mode & 0o777 == 0o700

    secret = root / "secret"
    secret.write_text("synthetic", encoding="utf-8")
    private_file(secret)
    assert secret.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode and hard-link semantics")
def test_posix_rejects_symlinks_and_hardlinks_before_chmod(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("synthetic", encoding="utf-8")
    linked = root / "linked"
    linked.symlink_to(outside)
    with pytest.raises(OSError):
        private_file(linked)

    alias = root / "alias"
    os.link(outside, alias)
    before = outside.stat().st_mode & 0o777
    with pytest.raises(OSError):
        private_file(alias)
    assert outside.stat().st_mode & 0o777 == before
    target_dir = tmp_path / "outside-dir"
    target_dir.mkdir()
    directory_link = root / "directory-link"
    directory_link.symlink_to(target_dir, target_is_directory=True)
    with pytest.raises(OSError):
        private_directory(directory_link)


def _windows_security(
    path: Path, *, check_protected: bool = True
) -> tuple[str, str, str, list[tuple[str, int, int]]]:
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    kernel.OpenProcessToken.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    advapi.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.GetTokenInformation.restype = wintypes.BOOL
    advapi.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    advapi.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi.GetSecurityInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi.GetSecurityInfo.restype = wintypes.DWORD
    advapi.GetSecurityDescriptorControl.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.WORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.GetSecurityDescriptorControl.restype = wintypes.BOOL
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    token = wintypes.HANDLE()
    assert kernel.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token))
    token_sids = []
    try:
        for info_class in (1, 4):
            buffer = ctypes.create_string_buffer(4096)
            needed = wintypes.DWORD()
            assert advapi.GetTokenInformation(
                token, info_class, buffer, len(buffer), ctypes.byref(needed)
            )
            sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
            current = wintypes.LPWSTR()
            assert advapi.ConvertSidToStringSidW(sid, ctypes.byref(current))
            try:
                token_sids.append(current.value)
            finally:
                kernel.LocalFree(ctypes.cast(current, ctypes.c_void_p))
        current_sid, token_owner_sid = token_sids
    finally:
        kernel.CloseHandle(token)
    handle = kernel.CreateFileW(str(path), 0x00020000, 0x7, None, 3, 0x02000000 | 0x00200000, None)
    assert handle != wintypes.HANDLE(-1).value
    owner, dacl, descriptor = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    try:
        assert (
            advapi.GetSecurityInfo(
                handle,
                1,
                1 | 4,
                ctypes.byref(owner),
                None,
                ctypes.byref(dacl),
                None,
                ctypes.byref(descriptor),
            )
            == 0
        )
        control, revision = wintypes.WORD(), wintypes.DWORD()
        assert advapi.GetSecurityDescriptorControl(
            descriptor, ctypes.byref(control), ctypes.byref(revision)
        )
        if check_protected:
            assert control.value & 0x1000
        owner_text = wintypes.LPWSTR()
        assert advapi.ConvertSidToStringSidW(owner, ctypes.byref(owner_text))
        try:
            owner_sid = owner_text.value
        finally:
            kernel.LocalFree(ctypes.cast(owner_text, ctypes.c_void_p))
        base = dacl.value
        ace_count = ctypes.cast(base + 4, ctypes.POINTER(ctypes.c_ushort))[0]
        aces = []
        offset = 8
        for _ in range(ace_count):
            ace_type = ctypes.cast(base + offset, ctypes.POINTER(ctypes.c_ubyte))[0]
            ace_flags = ctypes.cast(base + offset + 1, ctypes.POINTER(ctypes.c_ubyte))[0]
            ace_size = ctypes.cast(base + offset + 2, ctypes.POINTER(ctypes.c_ushort))[0]
            assert ace_type == 0
            mask = ctypes.cast(base + offset + 4, ctypes.POINTER(ctypes.c_ulong))[0]
            ace_sid = wintypes.LPWSTR()
            assert advapi.ConvertSidToStringSidW(base + offset + 8, ctypes.byref(ace_sid))
            try:
                aces.append((ace_sid.value, ace_flags, mask))
            finally:
                kernel.LocalFree(ctypes.cast(ace_sid, ctypes.c_void_p))
            offset += ace_size
        return current_sid, token_owner_sid, owner_sid, aces
    finally:
        if descriptor:
            kernel.LocalFree(descriptor)
        kernel.CloseHandle(handle)


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows ACLs")
def test_windows_replaces_broad_dacl_and_children_inherit_user_only(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir()
    original_owner = _windows_security(root, check_protected=False)[2]
    private_directory(root)
    current_sid, token_owner_sid, owner_sid, directory_aces = _windows_security(root)
    assert original_owner in {current_sid, token_owner_sid}
    assert owner_sid == current_sid
    assert directory_aces == [(current_sid, 0x03, 0x1F01FF)]
    child = root / "child"
    child.mkdir()
    _, token_owner_sid, child_owner, inherited_child_aces = _windows_security(
        child, check_protected=False
    )
    assert child_owner in {current_sid, token_owner_sid}
    assert len(inherited_child_aces) == 1
    assert inherited_child_aces[0][0] == current_sid
    assert inherited_child_aces[0][1] & 0x10

    private_directory(child)
    _, _, child_owner, child_aces = _windows_security(child)
    assert child_owner == current_sid
    assert child_aces == [(current_sid, 0x03, 0x1F01FF)]

    inherited_file = child / "inherited"
    inherited_file.write_text("synthetic", encoding="utf-8")
    _, token_owner_sid, inherited_owner, inherited_file_aces = _windows_security(
        inherited_file, check_protected=False
    )
    assert inherited_owner in {current_sid, token_owner_sid}
    assert len(inherited_file_aces) == 1
    assert inherited_file_aces[0][0] == current_sid
    assert inherited_file_aces[0][1] & 0x10

    private_file(inherited_file)
    _, _, protected_file_owner, protected_file_aces = _windows_security(inherited_file)
    assert protected_file_owner == current_sid
    assert protected_file_aces == [(current_sid, 0, 0x1F01FF)]


@pytest.mark.slow
@pytest.mark.skipif(os.name != "nt", reason="requires native Windows junctions")
def test_windows_rejects_junction_paths(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    junction = tmp_path / "junction"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target)], capture_output=True
    )
    if result.returncode:
        pytest.skip("Windows junction creation unavailable")
    with pytest.raises(OSError):
        private_directory(junction)


# A433: one atomic writer for the proxy's keys, config, TLS identity and stored credentials


def _entries(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_write_private_gives_owner_only_modes(tmp_path: Path) -> None:
    folder = tmp_path / "proxy"
    target = folder / "keys.json"
    write_private(target, '{"k": "fake-value"}\n')
    assert target.read_bytes() == b'{"k": "fake-value"}\n'
    assert target.stat().st_mode & 0o777 == 0o600
    assert folder.stat().st_mode & 0o777 == 0o700

    target.chmod(0o644)
    folder.chmod(0o755)
    write_private(target, b"\x00fake-bytes")
    assert target.read_bytes() == b"\x00fake-bytes"
    assert target.stat().st_mode & 0o777 == 0o600
    assert folder.stat().st_mode & 0o777 == 0o700
    assert _entries(folder) == ["keys.json"]


def test_write_private_makes_the_temp_file_private_before_any_byte(tmp_path: Path) -> None:
    target = tmp_path / "proxy-key.pem"
    seen: list[tuple[str, int, int]] = []

    def check(path: Path) -> os.stat_result:
        info = private_file(path)
        seen.append((Path(path).name, info.st_size, info.st_mode & 0o777))
        return info

    write_private(target, "fake key material", file=check)
    temp_name, temp_size, temp_mode = seen[0]
    assert temp_name.startswith(".proxy-key.pem.") and temp_size == 0
    if os.name != "nt":
        assert temp_mode == 0o600
    assert seen[-1][0] == "proxy-key.pem" and seen[-1][1] == len("fake key material")


@pytest.mark.parametrize("step", ["check", "fsync", "replace"])
def test_write_private_leaves_no_temp_file_and_the_old_file_when_it_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    target = tmp_path / "config.json"
    write_private(target, "old")

    def deny(path: Path) -> os.stat_result:
        if Path(path).name != "config.json":
            raise PermissionError("cannot protect private Windows file")
        return private_file(path)

    def fail(*_args: object) -> None:
        raise OSError("disk full")

    checks = {"file": deny} if step == "check" else {}
    if step == "fsync":
        monkeypatch.setattr(private_files.os, "fsync", fail)
    if step == "replace":
        monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError):
        write_private(target, "new", **checks)
    assert target.read_text(encoding="utf-8") == "old"
    assert _entries(tmp_path) == ["config.json"]


def test_write_private_replaces_the_file_whole_after_syncing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "credentials-runtime.json"
    write_private(target, "old")
    old_inode = target.stat().st_ino
    synced: list[int] = []
    real_fsync, real_replace = os.fsync, Path.replace

    def fsync(fd: int) -> None:
        synced.append(fd)
        real_fsync(fd)

    def replace(source: Path, dest: Path) -> Path:
        assert synced, "the data is synced before it replaces the file"
        assert Path(dest).read_text(encoding="utf-8") == "old"  # never partly written
        assert source.read_text(encoding="utf-8") == "new content"
        return real_replace(source, dest)

    monkeypatch.setattr(private_files.os, "fsync", fsync)
    monkeypatch.setattr(Path, "replace", replace)
    write_private(target, "new content")
    assert target.read_text(encoding="utf-8") == "new content"
    if os.name != "nt":
        assert target.stat().st_ino != old_inode  # a new file, not rewritten in place
        assert len(synced) == 2  # the file, then its folder
    assert _entries(tmp_path) == ["credentials-runtime.json"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")
def test_write_private_refuses_a_symlinked_target(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text("theirs", encoding="utf-8")
    link = tmp_path / "keys.json"
    link.symlink_to(elsewhere)
    with pytest.raises(OSError):
        write_private(link, "new")
    assert elsewhere.read_text(encoding="utf-8") == "theirs"
    assert _entries(tmp_path) == ["elsewhere.json", "keys.json"]
