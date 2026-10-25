"""Machine-readable step progress of ``cutpoint analyze`` and ``cutpoint edit``.

Active only when ``CUTPOINT_PROGRESS=1``. Every state change of a step is one line on
stderr::

    PROGRESS {"step":"reactions","index":2,"total":6,"label":"Reactions","state":"running",
    "pct":40,"detail":null}

(one line, compact JSON). ``state`` is ``running``, ``done``, ``skipped`` or ``failed``;
``index`` counts from 1; ``pct`` is 0-100 or null; ``detail`` is a short text or null.
The steps of a run are fixed when it starts (:data:`ANALYZE_STEPS`, :data:`EDIT_STEPS`;
their names are a stable API) and run in that order. A step that does not apply is
emitted ``skipped`` with the reason in ``detail``. Updates of a running step (``pct``,
``detail``) are throttled to two per second; state changes never are. A failed write
(closed or broken stderr) turns progress off for the rest of the run: emitting never
stops the run itself.

A cancelled run (Ctrl+C, exit 130) ends its current step ``failed`` with detail
``cancelled``; an error that stops the run ends it ``failed`` with the error text.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from typing import Any

ENV = "CUTPOINT_PROGRESS"
ANALYZE_STEPS = ("transcribe", "reactions", "keyframes", "crashes", "ranking", "vision")
EDIT_STEPS = (*ANALYZE_STEPS, "clips", "assemble")
LABELS = {
    "transcribe": "Transcription",
    "reactions": "Reactions",
    "keyframes": "Keyframes",
    "crashes": "Crashes",
    "ranking": "Ranking",
    "vision": "Vision rating",
    "clips": "Clips",
    "assemble": "Assemble",
}
# Analysis steps ``analyze --skip`` and the ``analysis.skip`` setting may leave out.
SKIPPABLE = ("reactions", "keyframes", "crashes")
MIN_INTERVAL_S = 0.5  # running updates: at most two per second
CANCELLED = "cancelled"


def enabled() -> bool:
    """Whether ``CUTPOINT_PROGRESS=1`` asks for progress lines."""
    return os.environ.get(ENV, "").strip() == "1"


class Run:
    """The steps of one command and their states; see the module doc.

    Use :func:`run` to make one; the methods are safe to call when progress is off.
    """

    def __init__(self, steps: Sequence[str], *, active: bool | None = None) -> None:
        self.steps = tuple(steps)
        self.active = enabled() if active is None else active
        self.state: dict[str, str] = {}  # last emitted state per step
        self.reasons: dict[str, str] = {}  # planned skips, emitted when their turn comes
        self.current: str | None = None  # the running step
        self.stopped = False  # a step already ended failed by the error that stops the run
        self._pct: int | None = None
        self._detail: str | None = None
        self._last = 0.0  # time of the last running line of the current step
        self._pending: dict[str, Any] | None = None  # a throttled running line
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- emitting
    def _line(self, step: str, state: str, pct: int | None, detail: str | None) -> dict[str, Any]:
        return {
            "step": step,
            "index": self.steps.index(step) + 1,
            "total": len(self.steps),
            "label": LABELS.get(step, step),
            "state": state,
            "pct": pct,
            "detail": detail,
        }

    def _write(self, line: dict[str, Any]) -> None:
        """Write one line; any failure turns progress off (never raises)."""
        if not self.active:
            return
        try:
            sys.stderr.write("PROGRESS " + json.dumps(line, separators=(",", ":")) + "\n")
            sys.stderr.flush()
        except (OSError, ValueError, AttributeError):  # broken pipe, closed or no stderr
            self.active = False

    def _emit(
        self, step: str, state: str, pct: int | None = None, detail: str | None = None
    ) -> None:
        with self._lock:
            self._cancel_pending()
            self.state[step] = state
            if state == "running":
                self._last = time.monotonic()
            self._write(self._line(step, state, pct, detail))

    def _cancel_pending(self) -> None:
        self._pending = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _flush_pending(self) -> None:
        with self._lock:
            self._timer = None
            line, self._pending = self._pending, None
            if line is not None and self.current == line["step"]:
                self._last = time.monotonic()
                self._write(line)

    # ----------------------------------------------------------------- steps
    def _skip_before(self, step: str) -> None:
        """Emit the not yet emitted steps before ``step`` as skipped (planned reason or
        "not run")."""
        for name in self.steps[: self.steps.index(step)]:
            if name not in self.state:
                self._emit(name, "skipped", detail=self.reasons.get(name, "not run"))

    def start(self, step: str, detail: str | None = None) -> None:
        """Mark ``step`` running; a step still running before it is done."""
        if self.current == step:
            if detail is not None:
                self.update(detail=detail)
            return
        if self.current is not None and self.state.get(self.current) == "running":
            self.done()
        self._skip_before(step)
        self.current, self._pct, self._detail = step, None, detail
        self._emit(step, "running", None, detail)

    def update(self, pct: float | None = None, detail: str | None = None) -> None:
        """Progress of the running step (throttled); None keeps the last value."""
        step = self.current
        if step is None or self.state.get(step) != "running":
            return
        if pct is not None:
            self._pct = max(0, min(100, int(pct)))
        if detail is not None:
            self._detail = detail
        line = self._line(step, "running", self._pct, self._detail)
        with self._lock:
            wait = self._last + MIN_INTERVAL_S - time.monotonic()
            if wait <= 0:
                self._cancel_pending()
                self._last = time.monotonic()
                self._write(line)
                return
            self._pending = line  # sent when the interval is up, unless a state change comes first
            if self._timer is None and self.active:
                self._timer = threading.Timer(wait, self._flush_pending)
                self._timer.daemon = True
                self._timer.start()

    def _end(self, state: str, detail: str | None) -> None:
        step = self.current
        if step is None:
            return
        self.current = None
        self._emit(step, state, 100 if state == "done" else self._pct, detail)

    def done(self, detail: str | None = None) -> None:
        """Mark the running step done."""
        self._end("done", detail)

    def fail(self, detail: str) -> None:
        """Mark the running step failed (the next steps may still run)."""
        self._end("failed", detail)

    def skip(self, step: str, reason: str) -> None:
        """Plan ``step`` as skipped; emitted at once if it is next, else when its turn comes."""
        if step not in self.steps or step in self.state:
            return
        self.reasons[step] = reason
        index = self.steps.index(step)
        if self.current is None and all(s in self.state for s in self.steps[:index]):
            self._emit(step, "skipped", detail=reason)

    def failed(self, detail: str) -> None:
        """A run-stopping error or a cancel: the running step (else the next one) failed."""
        if self.stopped:
            return
        self.stopped = True
        step = self.current or next((s for s in self.steps if s not in self.state), None)
        if step is None:
            return
        if self.current is None:
            self._skip_before(step)
            self.current = step
        self._end("failed", detail)

    def finish(self) -> None:
        """End of a successful run: the running step is done, the rest skipped."""
        if self.current is not None and self.state.get(self.current) == "running":
            self.done()
        for name in self.steps:
            if name not in self.state:
                self._emit(name, "skipped", detail=self.reasons.get(name, "not run"))

    @contextlib.contextmanager
    def step(self, name: str, detail: str | None = None) -> Iterator[Run]:
        """Run ``name`` in the block: running, then done (or failed if the block raises,
        which propagates)."""
        self.start(name, detail)
        try:
            yield self
        except BaseException as err:
            if self.current == name:
                self.fail(_error_text(err))
                self.stopped = True
            raise
        if self.current == name:
            self.done()


_RUNS: list[Run] = []  # the active runs, innermost last


def _error_text(err: BaseException) -> str:
    if isinstance(err, KeyboardInterrupt):
        return CANCELLED
    return str(err) or type(err).__name__


@contextlib.contextmanager
def run(steps: Sequence[str], *, use: bool = True) -> Iterator[Run]:
    """The progress run of a command: a new :class:`Run` of ``steps``, or the run already
    active (``edit`` runs ``analyze`` inside its own); with ``use`` False a run that emits
    nothing (``reanalyze`` inside ``events``).

    An exception ending the block ends the current step ``failed`` (``cancelled`` for
    KeyboardInterrupt) and propagates; a normal end emits the steps not emitted yet.
    """
    if not use:
        yield Run(steps, active=False)
        return
    if _RUNS:
        yield _RUNS[-1]
        return
    r = Run(steps)
    _RUNS.append(r)
    try:
        yield r
    except BaseException as err:
        with contextlib.suppress(Exception):  # best effort, never hides the error
            r.failed(_error_text(err))
        raise
    else:
        r.finish()
    finally:
        _RUNS.remove(r)
        with r._lock:
            r._cancel_pending()


def current() -> Run | None:
    """The active run, or None."""
    return _RUNS[-1] if _RUNS else None


def update(pct: float | None = None, detail: str | None = None) -> None:
    """:meth:`Run.update` of the active run (no-op without one)."""
    if _RUNS:
        _RUNS[-1].update(pct, detail)


def fraction(done: int, total: int) -> float:
    """``done`` of ``total`` as a percentage (0 for an empty total)."""
    return 100.0 * done / total if total else 0.0
