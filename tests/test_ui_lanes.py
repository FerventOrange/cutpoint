"""The UI's job lane (:mod:`cutpoint.ui.lanes`): one worker, first in first out, hooks under
the owner's lock, errors and closing."""

from __future__ import annotations

import threading

from cutpoint.ui.lanes import Lane

WAIT = 10  # seconds; only a safety net, every wait below ends on an event


def test_the_thread_starts_with_the_first_job_and_runs_jobs_one_at_a_time_in_order() -> None:
    lock = threading.Lock()
    gate = threading.Event()
    done = threading.Event()
    ran: list[int] = []
    active = [0]
    most = [0]

    def run(job: int) -> None:
        with lock:
            active[0] += 1
            most[0] = max(most[0], active[0])
        if job == 1:
            assert gate.wait(WAIT)  # the others wait behind it
        ran.append(job)
        with lock:
            active[0] -= 1
        if job == 3:
            done.set()

    lane: Lane[int] = Lane(lock, run, name="test-lane")
    assert lane.thread is None
    with lock:
        lane.queue.extend([1, 2, 3])
        lane.wake()
    thread = lane.thread
    assert thread is not None
    assert thread.name == "test-lane"
    assert thread.daemon
    gate.set()
    assert done.wait(WAIT)
    assert ran == [1, 2, 3]
    assert most[0] == 1
    with lock:
        lane.queue.append(4)
        lane.wake()
    assert lane.thread is thread  # one thread for the lane's life
    with lock:
        lane.close()
    thread.join(WAIT)
    assert not thread.is_alive()


def test_hooks_run_under_the_lock_around_each_job() -> None:
    lock = threading.Lock()
    seen: list[tuple[str, str, object]] = []
    done = threading.Event()
    lane: Lane[str] = Lane(lock, lambda job: seen.append(("run", job, lane.running)), name="t")

    def started(job: str) -> None:
        assert lock.locked()
        seen.append(("started", job, lane.running))

    def ended(job: str) -> None:
        assert lock.locked()
        seen.append(("ended", job, lane.running))
        done.set()

    lane.started, lane.ended = started, ended
    with lock:
        lane.queue.append("a")
        lane.wake()
    assert done.wait(WAIT)
    assert seen == [("started", "a", "a"), ("run", "a", "a"), ("ended", "a", None)]
    with lock:
        lane.close()


def test_a_job_that_raises_does_not_stop_the_lane_given_failed() -> None:
    lock = threading.Lock()
    failed: list[int] = []
    second = threading.Event()

    def run(job: int) -> None:
        if job == 1:
            raise RuntimeError("unexpected")
        second.set()

    lane: Lane[int] = Lane(lock, run, name="t", failed=failed.append)
    with lock:
        lane.queue.extend([1, 2])
        lane.wake()
    assert second.wait(WAIT)
    assert failed == [1]
    assert lane.thread is not None
    assert lane.thread.is_alive()
    with lock:
        lane.close()
    lane.thread.join(WAIT)
    assert not lane.thread.is_alive()


def test_without_failed_an_error_ends_the_thread_and_still_clears_running() -> None:
    lock = threading.Lock()
    ended = threading.Event()
    lane: Lane[int] = Lane(lock, lambda _job: 1 / 0, name="t", ended=lambda _job: ended.set())
    hook = threading.excepthook
    threading.excepthook = lambda _args: None  # the thread's ZeroDivisionError is expected
    try:
        with lock:
            lane.queue.extend([1, 2])
            lane.wake()
        assert ended.wait(WAIT)
        assert lane.thread is not None
        lane.thread.join(WAIT)
    finally:
        threading.excepthook = hook
    assert not lane.thread.is_alive()
    assert lane.running is None
    assert list(lane.queue) == [2]


def test_close_drops_queued_jobs_and_ends_after_the_running_one() -> None:
    lock = threading.Lock()
    started = threading.Event()
    release = threading.Event()
    ran: list[int] = []

    def run(job: int) -> None:
        ran.append(job)
        started.set()
        assert release.wait(WAIT)

    lane: Lane[int] = Lane(lock, run, name="t")
    with lock:
        lane.queue.extend([1, 2])
        lane.wake()
    assert started.wait(WAIT)
    with lock:
        lane.close()
        assert not lane.queue
    release.set()
    assert lane.thread is not None
    lane.thread.join(WAIT)
    assert not lane.thread.is_alive()
    assert ran == [1]
    assert lane.running is None
