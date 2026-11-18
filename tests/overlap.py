"""Proof that fake calls ran at once, without hoping a sleep is long enough (#1085 A476, A528)."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

# How long the first call of a run that must overlap waits for a second one to start: a parallel
# run starts it at once, so only a run that never overlaps waits this long (and then fails).
MUST_S = 10.0
# How long a call of a run that must not overlap holds, so a second call would show if one ran.
# It can only miss a wrong overlap, never fail a right run.
HOLD_S = 0.05


class Overlap:
    """Counts the fake calls in flight (``max`` is the most at once).

    With ``expect=True`` the first call waits (up to :data:`MUST_S`) for a second one to start,
    so a parallel run shows ``max >= 2`` on a loaded machine too; once two have met, the calls
    run free. With ``expect=False`` each call holds up to :data:`HOLD_S` for a second one, so a
    run that should go one at a time but does not shows it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.now = self.max = 0
        self.expect = False
        self.timed_out = False
        self._two = threading.Event()

    def reset(self, *, expect: bool) -> None:
        with self.lock:
            self.now = self.max = 0
            self.expect, self.timed_out = expect, False
            self._two.clear()

    @contextmanager
    def call(self) -> Iterator[None]:
        with self.lock:
            self.now += 1
            self.max = max(self.max, self.now)
            if self.now >= 2:
                self._two.set()
            wait = not self._two.is_set() and not self.timed_out
            limit = MUST_S if self.expect else HOLD_S
        try:
            if wait and not self._two.wait(limit) and self.expect:
                with self.lock:
                    self.timed_out = True  # the later calls do not wait again
            yield
        finally:
            with self.lock:
                self.now -= 1

    def hit(self) -> None:
        """One call that does nothing but count."""
        with self.call():
            pass

    def why(self) -> str:
        """The failure message of an overlap assert."""
        return f"at most {self.max} at once" + (
            f" (no second call started within {MUST_S:.0f} s)" if self.timed_out else ""
        )
