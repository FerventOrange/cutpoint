"""cutpoint.winjob: the Windows job object that ends children with Python, on a fake kernel32."""

from __future__ import annotations

import ctypes
import logging
from types import SimpleNamespace

import pytest

from cutpoint import winjob


class _Kernel32:
    """Records the job-object calls; each ``*_ok`` sets that call's return value."""

    def __init__(self, job: int = 77, set_ok: int = 1, assign_ok: int = 1) -> None:
        self.calls: list[tuple] = []
        self._set_ok, self._assign_ok = set_ok, assign_ok

        def create(attrs, name):  # a plain function: the code under test sets its restype
            self.calls.append(("create",))
            return job

        self.CreateJobObjectW = create

    def SetInformationJobObject(self, job, klass, info, size):
        flags = ctypes.cast(info, ctypes.POINTER(ctypes.c_int64 * 3)).contents[2] & 0xFFFFFFFF
        self.calls.append(("set", job.value, klass, flags))
        return self._set_ok

    def AssignProcessToJobObject(self, job, proc):
        self.calls.append(("assign", job.value, proc.value))
        return self._assign_ok

    def CloseHandle(self, handle):
        self.calls.append(("close", handle.value))
        return 1


@pytest.fixture
def k32(monkeypatch):
    def install(**kw) -> _Kernel32:
        fake = _Kernel32(**kw)
        monkeypatch.setattr(ctypes, "WinDLL", lambda name, **_: fake, raising=False)
        monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
        monkeypatch.setattr(winjob, "os", SimpleNamespace(name="nt"))
        monkeypatch.setattr(winjob, "_JOB_HANDLE", None)
        return fake

    return install


def _proc(handle: int) -> SimpleNamespace:
    return SimpleNamespace(_handle=handle, pid=handle + 1000)


def test_noop_off_windows(monkeypatch):
    monkeypatch.setattr(winjob, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: pytest.fail("no kernel32"), raising=False)
    winjob.kill_with_parent(_proc(1))  # type: ignore[arg-type]


def test_one_kill_on_close_job_for_every_child(k32):
    fake = k32()
    winjob.kill_with_parent(_proc(11))  # type: ignore[arg-type]
    winjob.kill_with_parent(_proc(12))  # type: ignore[arg-type]
    assert fake.calls == [
        ("create",),
        ("set", 77, 9, 0x2000),  # JobObjectExtendedLimitInformation, KILL_ON_JOB_CLOSE
        ("assign", 77, 11),
        ("assign", 77, 12),
    ]


def test_job_not_created_assigns_nothing_and_retries(k32):
    fake = k32(job=0)
    winjob.kill_with_parent(_proc(11))  # type: ignore[arg-type]
    winjob.kill_with_parent(_proc(12))  # type: ignore[arg-type]
    assert fake.calls == [("create",), ("create",)]
    assert winjob._JOB_HANDLE is None


def test_limits_not_set_closes_the_job_and_assigns_nothing(k32):
    fake = k32(set_ok=0)
    winjob.kill_with_parent(_proc(11))  # type: ignore[arg-type]
    assert [c[0] for c in fake.calls] == ["create", "set", "close"]
    assert fake.calls[-1] == ("close", 77)
    assert winjob._JOB_HANDLE is None  # a job without KILL_ON_JOB_CLOSE is never used


def test_assign_failure_is_logged_not_raised(k32, caplog):
    k32(assign_ok=0)
    with caplog.at_level(logging.DEBUG, logger="cutpoint.winjob"):
        winjob.kill_with_parent(_proc(11))  # type: ignore[arg-type]
    assert "process 1011 not put in the job object: error 5" in caplog.text


def test_api_unavailable_is_a_noop(k32, monkeypatch):
    k32()

    def boom(name, **_):
        raise OSError("no kernel32")

    monkeypatch.setattr(ctypes, "WinDLL", boom, raising=False)
    winjob.kill_with_parent(_proc(11))  # type: ignore[arg-type]
    assert winjob._JOB_HANDLE is None
