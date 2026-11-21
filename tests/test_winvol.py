"""cutpoint.winvol: the one home of the Windows drive and volume queries."""

from __future__ import annotations

import importlib
import os
import stat
import sys

import pytest

from cutpoint import config, stage, winvol

NAS = chr(92) * 2 + "nas" + chr(92) + "share"
_REAL_DRIVE_REMOTE = config._drive_remote  # conftest stubs the attribute per test
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="needs Windows")


def test_the_module_imports_without_ctypes(monkeypatch):
    """No ctypes at import time: the functions load it when called."""
    monkeypatch.setitem(sys.modules, "ctypes", None)
    mod = importlib.reload(winvol)
    assert mod.DRIVE_FIXED == 3


def test_the_never_raising_queries_are_quiet_off_windows(monkeypatch):
    monkeypatch.setattr(winvol.sys, "platform", "linux")
    assert winvol.query_dos_device("C:") is None
    assert winvol.network_connection("Z:") == (None, None)


def test_query_dos_device_swallows_api_errors(monkeypatch):
    monkeypatch.setattr(winvol.sys, "platform", "win32")

    def boom(drive):
        raise AttributeError("no windll")

    monkeypatch.setattr(winvol, "dos_target", boom)
    assert winvol.query_dos_device("C:") is None
    monkeypatch.setattr(winvol, "dos_target", lambda d: "")
    assert winvol.query_dos_device("C:") is None
    monkeypatch.setattr(winvol, "dos_target", lambda d: r"\??\C:\x")
    assert winvol.query_dos_device("C:") == r"\??\C:\x"


def test_drive_remote_uses_the_connection_name_then_the_registry(monkeypatch):
    monkeypatch.setattr(config.sys, "platform", "win32")
    real = _REAL_DRIVE_REMOTE
    real.cache_clear()
    monkeypatch.setattr(winvol, "network_connection", lambda d: (0, NAS))
    assert real("Z:") == NAS
    monkeypatch.setattr(winvol, "network_connection", lambda d: (winvol.ERROR_NOT_CONNECTED, None))
    real.cache_clear()
    assert real("Z:") is None  # not a network drive: the registry is not asked


def test_copy_once_keeps_mtime_and_leaves_no_temp(tmp_path):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"footage")
    os.utime(src, ns=(10**9, 2 * 10**9))
    dst = tmp_path / "out" / "copy.mp4"
    dst.parent.mkdir()
    stage.copy_once(src, dst, "a")
    assert dst.read_bytes() == b"footage"
    assert dst.stat().st_mtime_ns == 2 * 10**9
    assert [p.name for p in dst.parent.iterdir()] == ["copy.mp4"]
    assert stat.S_IMODE(dst.stat().st_mode) & stat.S_IRUSR


@windows_only
def test_the_windows_queries_answer_for_the_system_drive():
    drive = os.path.splitdrive(os.environ["SYSTEMROOT"])[0]
    assert winvol.drive_type(drive + chr(92)) == winvol.DRIVE_FIXED
    assert winvol.dos_target(drive).startswith(chr(92) + "Device")
    root = winvol.mount_root(os.environ["SYSTEMROOT"])
    assert winvol.volume_total_mb(root) > 0
    assert winvol.volume_guid(root).startswith("{")
    assert winvol.bin_used(root) >= 0
