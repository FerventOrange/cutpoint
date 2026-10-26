"""Discovery OCR in worker processes (#374): the same ocr.jsonl, records and progress as
the serial run, no pool with one worker, the thread split and error handling (fake OCR
engine and frames; one spawn test is marked slow)."""

from __future__ import annotations

import json
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint.events import discover, killfeed


class FakeEngine:
    """Reads the frame number from the image: one box per frame, placed by it."""

    def __init__(self, det_side: int = 960, threads: int = 1) -> None:
        pass

    def __call__(self, img: np.ndarray, min_score: float) -> list[tuple[list, str, float]]:
        n = int(img[0, 0, 0])
        if n == 255:
            raise ValueError("bad frame")
        x, y = 3.3 * n, 1.7 * n
        return [([[x, y], [x + 40, y], [x + 40, y + 12], [x, y + 12]], f"kill {n}", 0.912345)]


def fake_worker(img: np.ndarray, min_score: float) -> list[dict]:
    """``_ocr_worker`` with :class:`FakeEngine` (top-level: a spawned worker imports it)."""
    time.sleep(random.uniform(0, 0.01))  # results finish out of order
    return discover._ocr_boxes(FakeEngine()(img, min_score), discover.REF_W / img.shape[1])


def sleepy_worker(img: np.ndarray, min_score: float) -> list[dict]:
    """A fake OCR that returns at once for the first frame and takes 30 s for the others."""
    if int(img[0, 0, 0]) != 0:
        time.sleep(30)
    return []


def _frames(path, t, n, step, width):
    """``n`` tiny frames whose first pixel numbers them by time (255 at t = 99)."""
    out = []
    for k in range(n):
        img = np.zeros((9, 16, 3), np.uint8)
        img[0, 0, 0] = 255 if t == 99.0 else int(t / 10) * 4 + k
        out.append(img)
    return out


@pytest.fixture
def fake_ocr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(killfeed, "OcrEngine", FakeEngine)
    monkeypatch.setattr(discover, "_burst_frames", _frames)
    monkeypatch.setattr(discover, "_ocr_worker", fake_worker)
    src = SimpleNamespace(name="rec.mp4", io_path=tmp_path / "rec.mp4")

    def run(
        name: str,
        workers: int,
        times=(0.0, 10.0, 20.0, 30.0, 40.0),
        cached=(),
        on_frame=None,
    ):
        out = tmp_path / name
        out.mkdir()
        cache_path = out / "ocr.jsonl"
        if cached:
            cache_path.write_text("".join(json.dumps(r) + "\n" for r in cached), "utf-8")
        calls: list[tuple[int, int]] = []
        params = discover.DiscoverParams(ocr_workers=workers)
        recs = discover._ocr_frames(
            [(src, t) for t in times],
            params,
            out / "frames",
            cache_path,
            on_frame or (lambda d, n: calls.append((d, n))),
        )
        return SimpleNamespace(recs=recs, calls=calls, out=out, cache=cache_path)

    return run


def _thread_pool(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Run the workers in threads; returns the shutdown calls of each pool made."""
    shutdowns: list[dict] = []

    class Pool(ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            shutdowns.append({"cancel_futures": cancel_futures})
            super().shutdown(wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(discover, "_ocr_pool", lambda workers, threads, side: Pool(workers))
    return shutdowns


def _same(a, b) -> None:
    assert a.cache.read_bytes() == b.cache.read_bytes()
    assert a.recs == b.recs
    assert a.calls == b.calls
    assert sorted(p.name for p in (a.out / "frames").iterdir()) == sorted(
        p.name for p in (b.out / "frames").iterdir()
    )


def test_parallel_writes_the_serial_records_in_order(fake_ocr, monkeypatch):
    shutdowns = _thread_pool(monkeypatch)
    serial = fake_ocr("serial", 1)
    parallel = fake_ocr("parallel", 3)
    _same(serial, parallel)
    assert [r["burst"] * 10 + r["k"] for r in parallel.recs] == [
        b * 10 + k for b in range(5) for k in range(4)
    ]
    assert shutdowns == [{"cancel_futures": False}]


def test_parallel_resumes_from_the_cache_like_serial(fake_ocr, monkeypatch):
    _thread_pool(monkeypatch)
    first = fake_ocr("first", 1, times=(0.0, 10.0))
    lines = first.cache.read_text("utf-8").splitlines()
    # burst 0 whole, burst 1 half: one burst skipped, one with two frames to OCR
    cached = [json.loads(x) for x in lines[:6]]
    serial = fake_ocr("serial", 1, cached=cached)
    parallel = fake_ocr("parallel", 2, cached=cached)
    _same(serial, parallel)
    assert len(parallel.recs) == 20


def test_one_worker_runs_in_process_without_a_pool(fake_ocr, monkeypatch):
    def no_pool(*a):
        raise AssertionError("no pool with one worker")

    monkeypatch.setattr(discover, "_ocr_pool", no_pool)
    monkeypatch.setattr(discover.os, "cpu_count", lambda: 4)  # default: 1 worker
    assert len(fake_ocr("auto", 0).recs) == 20
    assert len(fake_ocr("one", 1).recs) == 20


@pytest.mark.parametrize(
    ("cpus", "requested", "expected"),
    [
        (4, 0, (1, 4)),  # the default on 4 CPUs: the serial run
        (8, 0, (2, 4)),
        (32, 0, (2, 4)),  # the default stays at 2
        (1, 0, (1, 1)),
        (4, 2, (2, 2)),  # threads split: workers x threads <= CPUs
        (6, 4, (4, 1)),
        (2, 8, (2, 1)),  # workers capped at the CPUs
        (16, 3, (3, 4)),  # threads capped at OCR_THREADS
    ],
)
def test_worker_count_and_thread_split(monkeypatch, cpus, requested, expected):
    monkeypatch.setattr(discover.os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(killfeed, "OCR_THREADS", min(4, cpus))
    workers, threads = discover.ocr_workers(requested)
    assert (workers, threads) == expected
    assert workers * threads <= cpus


def test_a_worker_error_propagates_and_stops_the_pool(fake_ocr, monkeypatch, tmp_path):
    shutdowns = _thread_pool(monkeypatch)
    times = (0.0, 10.0, 99.0, 30.0)  # burst 2 fails
    for name, workers in (("serial", 1), ("parallel", 2)):
        with pytest.raises(ValueError, match="bad frame"):
            fake_ocr(name, workers, times=times)
    assert shutdowns == [{"cancel_futures": True}]
    # the records before the failing frame, whole and in order, as in the serial run
    serial, parallel = (tmp_path / n / "ocr.jsonl" for n in ("serial", "parallel"))
    assert parallel.read_bytes() == serial.read_bytes()
    bursts = [json.loads(x)["burst"] for x in parallel.read_text("utf-8").splitlines()]
    assert bursts == [0] * 4 + [1] * 4


def test_stop_pool_terminates_the_worker_processes():
    stopped: list[str] = []
    proc = SimpleNamespace(terminate=lambda: stopped.append("terminate"))
    pool = SimpleNamespace(
        _processes={1: proc, 2: proc},
        shutdown=lambda wait, cancel_futures: stopped.append(f"shutdown {cancel_futures}"),
    )
    discover._stop_pool(pool)
    assert stopped == ["terminate", "terminate", "shutdown True"]


@pytest.mark.slow
def test_spawned_workers_match_serial_and_raise(fake_ocr):
    serial = fake_ocr("serial", 1)
    parallel = fake_ocr("parallel", 2)
    _same(serial, parallel)
    with pytest.raises(ValueError, match="bad frame"):
        fake_ocr("err", 2, times=(0.0, 99.0))


def test_frames_waiting_for_ocr_stay_bounded(fake_ocr, monkeypatch):
    gate = threading.Event()  # the workers wait until the parent has queued all it may
    live: set = set()
    peak = [0]
    lock = threading.Lock()

    def gated(img, min_score):
        gate.wait(5)
        return fake_worker(img, min_score)

    class Pool(ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            fut = super().submit(fn, *args, **kwargs)
            with lock:
                live.add(fut)
                peak[0] = max(peak[0], len(live))
            fut.add_done_callback(lambda f: (lock.acquire(), live.discard(f), lock.release()))
            return fut

    monkeypatch.setattr(discover, "_ocr_pool", lambda workers, threads, side: Pool(workers))
    monkeypatch.setattr(discover, "_ocr_worker", gated)
    threading.Timer(0.3, gate.set).start()
    workers, frames = 2, discover.DiscoverParams().burst_frames
    run = fake_ocr("bounded", workers, times=tuple(10.0 * b for b in range(10)))
    assert len(run.recs) == 40
    assert 2 * workers < peak[0] <= 2 * workers + frames


@pytest.mark.slow
def test_ctrl_c_terminates_spawned_workers_at_once(fake_ocr, monkeypatch, tmp_path):
    monkeypatch.setattr(discover, "_ocr_worker", sleepy_worker)

    def interrupt(done, total):  # after the first record
        raise KeyboardInterrupt

    t0 = time.perf_counter()
    with pytest.raises(KeyboardInterrupt):
        fake_ocr("ctrl-c", 2, on_frame=interrupt)
    assert time.perf_counter() - t0 < 10  # not the 30 s of a running OCR call
    assert multiprocessing.active_children() == []
    text = (tmp_path / "ctrl-c" / "ocr.jsonl").read_text("utf-8")
    assert text.endswith("\n")
    assert [json.loads(x)["id"] for x in text.splitlines()] == ["rec.mp4@0.0"]


_STAND_IN = """
import time
from cutpoint.events import discover
if __name__ == "__main__":
    pool = discover._ocr_pool(2, 1, 960)
    for f in [pool.submit(time.sleep, 0.2) for _ in range(4)]:  # both workers set up
        f.result()
    print(" ".join(map(str, pool._processes)), flush=True)
    for _ in range(2):
        pool.submit(time.sleep, 60)
    time.sleep(60)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = Path(f"/proc/{pid}/status")  # a zombie waits for its new parent to reap it
    return not (status.exists() and "\nState:\tZ" in status.read_text())


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="checks POSIX process ids")
def test_workers_exit_when_the_parent_is_killed():
    parent = subprocess.Popen([sys.executable, "-c", _STAND_IN], stdout=subprocess.PIPE, text=True)
    try:
        pids = [int(p) for p in parent.stdout.readline().split()]
        assert len(pids) == 2
        assert all(_alive(p) for p in pids)
    finally:
        parent.kill()  # SIGKILL: no cleanup in the parent
        parent.wait()
    deadline = time.monotonic() + 10
    while any(_alive(p) for p in pids) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not any(_alive(p) for p in pids)
