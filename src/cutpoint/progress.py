"""Machine-readable step progress of ``cutpoint analyze``, ``edit`` and ``360 analyze``.

Active only when ``CUTPOINT_PROGRESS=1``. Every state change of a step is one line on
stderr::

    PROGRESS {"step":"reactions","index":2,"total":6,"label":"Reactions","state":"running",
    "pct":40,"detail":null}

(one line, compact JSON). ``state`` is ``running``, ``done``, ``skipped`` or ``failed``;
``index`` counts from 1; ``pct`` is 0-100 or null; ``detail`` is a short text or null.
The steps of a run are fixed when it starts (:data:`ANALYZE_STEPS`, :data:`EDIT_STEPS`,
:data:`PANO_STEPS`; their names are a stable API) and run in that order. A step that does
not apply is emitted ``skipped`` with the reason in ``detail``. Updates of a running step
(``pct``, ``detail``) are throttled to two per second and left out when nothing changed;
state changes never are. A failed write (closed or broken stderr) turns progress off for the
rest of the run: emitting never stops the run itself.

A cancelled run (Ctrl+C, exit 130) ends its current step ``failed`` with detail
``cancelled``; an error that stops the run ends it ``failed`` with the error text.

A step that starts with part, not all, of its work cached (a re-run after an interruption) logs an
INFO line ``<name>: N/M done, continuing`` and sends the same text at once as its
``detail`` (:func:`continuing`).

:attr:`Run.observers` get every state change and ``pct`` of the runs, whether or not
``CUTPOINT_PROGRESS`` is set (``progress.json``, :mod:`cutpoint.runstate`); they never stop a run.

``--batch-size N`` (#109) splits a step's sources (the VLM layer's windows) into batches of N
(:func:`batches`); its model is closed between two, and the boundary is logged and recorded as
:func:`continuing` does.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

ENV = "CUTPOINT_PROGRESS"
ANALYZE_STEPS = ("transcribe", "reactions", "keyframes", "crashes", "ranking", "vision")
EDIT_STEPS = (*ANALYZE_STEPS, "clips", "assemble")
PANO_STEPS = ("scan", "views", "analyze", "events")  # ``cutpoint 360 analyze``
# ``cutpoint events``: its layers in the order they run, recorded in progress.json only (a
# quiet run, :func:`run`: no PROGRESS lines)
EVENTS_STEPS = ("audio", "voice", "visual", "ocr", "crash", "vlm", "write")
LABELS = {
    "transcribe": "Transcription",
    "reactions": "Reactions",
    "keyframes": "Keyframes",
    "crashes": "Crashes",
    "ranking": "Ranking",
    "vision": "Vision rating",
    "clips": "Clips",
    "assemble": "Assemble",
    "scan": "Read 360 clips",
    "views": "360 views",
    "analyze": "Transcription and ranking",
    "events": "Events and re-rank",
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

    # Called as ``observer(step, state, pct, done, total)`` on each state change and ``pct``
    # update of an ``observed`` run (``done``/``total``: units cached at the step's start,
    # from :func:`continuing`, else None).
    observers: ClassVar[list[Callable[..., None]]] = []

    def __init__(
        self, steps: Sequence[str], *, active: bool | None = None, observed: bool = True
    ) -> None:
        self.steps = tuple(steps)
        self.active = enabled() if active is None else active
        self.observed = observed
        self.state: dict[str, str] = {}  # last emitted state per step
        self.reasons: dict[str, str] = {}  # planned skips, emitted when their turn comes
        self.current: str | None = None  # the running step
        self.stopped = False  # a step already ended failed by the error that stops the run
        self._pct: int | None = None
        self._share: tuple[float, float] | None = None  # update()'s 0-100 maps to it (share)
        self._detail: str | None = None
        self._last = 0.0  # time of the last running line of the current step
        self._pending: dict[str, Any] | None = None  # a throttled running line
        self._sent: dict[str, Any] | None = None  # the last line written
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
        self._sent = line
        try:
            sys.stderr.write("PROGRESS " + json.dumps(line, separators=(",", ":")) + "\n")
            sys.stderr.flush()
        except (OSError, ValueError, AttributeError):  # broken pipe, closed or no stderr
            self.active = False

    def _notify(
        self,
        step: str,
        state: str,
        pct: int | None,
        done: int | None = None,
        total: int | None = None,
    ) -> None:
        """Tell :attr:`observers` (an observer's error is logged, never raised)."""
        if not self.observed:
            return
        for observer in list(self.observers):
            try:
                observer(step, state, pct, done, total)
            except Exception:  # recording progress never stops the run
                logger.debug("progress observer failed", exc_info=True)

    def _emit(
        self, step: str, state: str, pct: int | None = None, detail: str | None = None
    ) -> None:
        with self._lock:
            self._cancel_pending()
            self.state[step] = state
            if state == "running":
                self._last = time.monotonic()
            self._write(self._line(step, state, pct, detail))
        self._notify(step, state, pct)

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
            if self._share is not None:
                lo, hi = self._share
                pct = lo + (hi - lo) * pct / 100
            self._pct = max(0, min(100, int(pct)))
        if detail is not None:
            self._detail = detail
        if pct is not None:
            self._notify(step, "running", self._pct)
        line = self._line(step, "running", self._pct, self._detail)
        with self._lock:
            if line == self._sent:  # nothing new (a pending change was undone)
                self._cancel_pending()
                return
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

    def note(self, detail: str) -> None:
        """Set the running step's ``detail`` and write it at once (not throttled), so the
        updates that follow cannot replace it before it is sent."""
        step = self.current
        if step is None or self.state.get(step) != "running":
            return
        self._detail = detail
        with self._lock:
            self._cancel_pending()
            self._last = time.monotonic()
            self._write(self._line(step, "running", self._pct, detail))

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
def run(steps: Sequence[str], *, use: bool = True, quiet: bool = False) -> Iterator[Run]:
    """The progress run of a command: a new :class:`Run` of ``steps``, or the run already
    active (``edit`` runs ``analyze`` inside its own); with ``use`` False a run that emits
    nothing (``reanalyze`` inside ``events``); with ``quiet`` one that writes no PROGRESS
    lines, but whose :attr:`Run.observers` see it (``cutpoint events``: progress.json).

    An exception ending the block ends the current step ``failed`` (``cancelled`` for
    KeyboardInterrupt) and propagates; a normal end emits the steps not emitted yet.
    """
    if not use:
        yield Run(steps, active=False, observed=False)
        return
    if _RUNS:
        yield _RUNS[-1]
        return
    r = Run(steps, active=False if quiet else None)
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


@contextlib.contextmanager
def share(lo: float, hi: float) -> Iterator[None]:
    """In the block, the active run's :meth:`Run.update` percentages (0-100) map to
    ``[lo, hi]``: one batch of a step whose work reports per batch (``--batch-size``)."""
    r = current()
    if r is None:
        yield
        return
    old, r._share = r._share, (lo, hi)
    try:
        yield
    finally:
        r._share = old


def start(step: str) -> None:
    """:meth:`Run.start` of ``step`` on the active run if it is one of its steps (the layers
    of ``cutpoint events``, :data:`EVENTS_STEPS`); a no-op otherwise."""
    if _RUNS and step in _RUNS[-1].steps:
        _RUNS[-1].start(step)


def continuing(name: str, done: int, total: int) -> None:
    """``done`` of the ``total`` units (sources, windows) of ``name`` are cached at its
    start: log ``<name>: N/M done, continuing`` (INFO) and send it as the running step's
    detail (:meth:`Run.note`). Nothing when ``done`` is 0 (a fresh run looks as before) or
    ``total`` (nothing left to continue: a re-run of finished work looks as before too).
    The counts go to the observers of the running step either way."""
    if _RUNS and (r := _RUNS[-1]).current is not None:
        r._notify(r.current, "running", r._pct, done, total)
    if not 0 < done < total:
        return
    text = f"{name}: {done}/{total} done, continuing"
    logger.info("%s", text)
    if _RUNS:
        _RUNS[-1].note(text)


def batches[T](items: Sequence[T], size: int) -> list[Sequence[T]]:
    """``items`` in batches of ``size`` (``--batch-size``); one batch for ``size`` 0."""
    if size <= 0 or len(items) <= size:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def fraction(done: float, total: int) -> float:
    """``done`` of ``total`` as a percentage (0 for an empty total)."""
    return 100.0 * done / total if total else 0.0
