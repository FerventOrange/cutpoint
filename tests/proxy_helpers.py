"""A stand-in for ``cliproxy.ProxyLauncher`` on ``app.state.proxies`` (#821): the routes' tests
decide what the proxy does without starting one."""

from __future__ import annotations

import threading

from cutpoint.assistant import cliproxy


class FakeLauncher:
    """Records each ``ensure``; ``start`` makes the proxy "run" or raises ``error``.

    ``running`` and ``saved`` are what the launcher reports. Like the real one, ``ensure``
    does nothing when the proxy runs, or with ``logged_in_only`` and no saved login.
    """

    def __init__(self, *, running: bool = False, saved: bool = False) -> None:
        self.is_running = running
        self.saved = saved
        self.error: Exception | None = None
        self.calls: list[bool] = []  # the logged_in_only of each ensure
        self.starts = 0
        self.shutdowns = 0
        # a start in flight: ``wait_for_start`` blocks until the test sets ``gate``
        self.gate: threading.Event | None = None
        self.waiting = threading.Event()

    def running(self) -> bool:
        return self.is_running

    def saved_login(self) -> bool:
        return self.saved

    def wait_for_start(self) -> None:
        if self.gate is not None:
            self.waiting.set()
            assert self.gate.wait(5)

    def ensure(self, *, logged_in_only: bool = False) -> bool:
        self.calls.append(logged_in_only)
        if self.is_running or (logged_in_only and not self.saved):
            return False
        if self.error is not None:
            raise self.error
        self.starts += 1
        self.is_running = True
        return True

    def shutdown(self) -> None:
        self.shutdowns += 1


def not_running() -> cliproxy.ProxyConnectionError:
    return cliproxy.ProxyConnectionError(cliproxy.NOT_RUNNING)
