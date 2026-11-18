"""One background worker over a queue of jobs: the lane of the UI's window proxies
(:class:`~cutpoint.ui.media.Media`) and of its 360 director previews
(:class:`~cutpoint.ui.pano_previews.PanoPreviews`).

A lane runs its jobs one at a time, first in, first out, on one daemon thread started with the
first job. ``queue`` and ``running`` belong to the owner's lock: the owner decides under it what
to queue, supersede or cancel (each lane has its own rules), then calls :meth:`Lane.wake`.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable


class Lane[T]:
    """One worker thread over ``queue``, guarded by the owner's ``lock``.

    ``run(job)`` runs without the lock. ``started(job)`` and ``ended(job)``, when given, run
    under the lock right after ``running`` is set and right after it is cleared. ``failed(job)``,
    when given, is called inside the ``except Exception`` of a job that raised, and the lane
    goes on with the next job; without it the error ends the thread (a lane whose ``run``
    catches its own errors).
    """

    def __init__(
        self,
        lock: threading.Lock,
        run: Callable[[T], None],
        *,
        name: str,
        started: Callable[[T], None] | None = None,
        ended: Callable[[T], None] | None = None,
        failed: Callable[[T], None] | None = None,
    ) -> None:
        self.lock = lock
        self.run = run
        self.name = name
        self.started = started
        self.ended = ended
        self.failed = failed
        self.cond = threading.Condition(lock)
        self.queue: deque[T] = deque()
        self.running: T | None = None
        self.thread: threading.Thread | None = None  # started with the first job
        self.closed = False

    def wake(self) -> None:
        """Under the lock, after queueing: start the thread on the first call, else wake it."""
        if self.thread is None:
            self.thread = threading.Thread(target=self._loop, name=self.name, daemon=True)
            self.thread.start()
        self.cond.notify()

    def close(self) -> None:
        """Under the lock: drop the queued jobs and end the thread once its job (if any) ends."""
        self.closed = True
        self.queue.clear()
        self.cond.notify_all()

    def _loop(self) -> None:
        while True:
            with self.lock:
                while not self.queue and not self.closed:
                    self.cond.wait()
                if self.closed:
                    return
                job = self.running = self.queue.popleft()
                if self.started is not None:
                    self.started(job)
            try:
                self.run(job)
            except Exception:
                if self.failed is None:
                    raise
                self.failed(job)
            finally:
                with self.lock:
                    self.running = None
                    if self.ended is not None:
                        self.ended(job)
