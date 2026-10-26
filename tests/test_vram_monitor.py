"""VramMonitor enters and leaves without waiting on nvidia-smi (#372)."""

from __future__ import annotations

import subprocess
import threading
import time

from cutpoint.events import common
from cutpoint.events.common import VramMonitor


def test_enter_and_exit_do_not_wait_on_a_slow_poll(monkeypatch):
    started, release = threading.Event(), threading.Event()

    def slow() -> int:
        started.set()
        release.wait(5)
        return 4000

    monkeypatch.setattr(common, "gpu_memory_used_mib", slow)
    t0 = time.perf_counter()
    with VramMonitor() as vm:
        assert started.wait(2)  # the poll is in flight
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0  # neither a synchronous baseline nor a join on the poll
    release.set()  # the late poll returns after exit
    vm._thread.join(2)
    assert not vm._thread.is_alive()
    assert vm.summary() == {"baseline_mib": None, "peak_mib": None}  # its result dropped


def test_baseline_is_the_first_reading_and_peak_the_max(monkeypatch):
    readings = iter([1000, 3000, 2000])
    seen = threading.Event()
    n = [0]

    def poll() -> int | None:
        n[0] += 1
        if n[0] >= 3:
            seen.set()
        return next(readings, None)

    monkeypatch.setattr(common, "gpu_memory_used_mib", poll)
    with VramMonitor(interval_s=0.01) as vm:
        assert seen.wait(2)
    vm._thread.join(2)
    assert vm.summary() == {"baseline_mib": 1000, "peak_mib": 3000}


def test_no_nvidia_smi_gives_none(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing)
    with VramMonitor(interval_s=0.01) as vm:
        time.sleep(0.05)
    vm._thread.join(2)
    assert vm.summary() == {"baseline_mib": None, "peak_mib": None}
