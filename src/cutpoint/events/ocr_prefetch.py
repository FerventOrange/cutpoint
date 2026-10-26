"""Background HUD OCR for ``cutpoint analyze`` (#375): the OCR layer of the keyframes step
starts while Whisper, the scene cuts and PANNs use the GPU, instead of after them.

OCR (RapidOCR) runs on the CPU and decodes with NVDEC, so it overlaps the GPU steps well. One
worker thread reads the offered sources one at a time, in source order, with the same
:func:`~cutpoint.events.killfeed.detect_ocr_events` call ``run_events`` makes; ``run_events``
then takes each result in its normal turn, so the merge order, events.json and the OCR caches
are those of a serial run. The worker never reports: its log records are held and replayed,
in order, when its result is taken (the same lines in the same order, apart from rare ``-v``
debug lines of helper threads and a "resuming the OCR read" line when a read is handed back
to its turn), and its progress is forwarded then through the step's own callback.

It runs only when nothing it needs comes later: a clean read (no ``read_errors``, so no bad
ranges; a source after one is not offered either), Whisper on CUDA (the overlap pays only
against GPU steps), a HUD preset for every game (discovery stays serial) and, when
nvidia-smi answers, at least :data:`MIN_FREE_MIB` of free and :data:`MIN_TOTAL_MIB` of total
VRAM. ``CUTPOINT_ANALYZE_OVERLAP=0`` turns it off. Whatever it does not read, or a result
whose inputs differ from the ones ``run_events`` uses, is read inline as before, after the
worker is stopped: one OCR engine and one decode at most.
"""

from __future__ import annotations

import functools
import logging
import queue
import subprocess
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cutpoint import paths
from cutpoint.events.common import Source, load_source
from cutpoint.events.killfeed import OcrCancelled, OcrEngine, OcrParams, detect_ocr_events
from cutpoint.events.regions import game_key_for_path, load_preset

logger = logging.getLogger(__name__)

ENV = "ANALYZE_OVERLAP"  # CUTPOINT_ANALYZE_OVERLAP=0: OCR waits for its turn (escape hatch)
MIN_FREE_MIB = 2048  # free VRAM when the first source is offered (Whisper may load after it)
MIN_TOTAL_MIB = 8192
POLL_S = 0.5  # waits on the worker in steps this long: Ctrl+C gets through on Windows
# close(): one frame's OCR calls (ffmpeg is killed at once on cancel); a worker still busy then
# is left (daemon) and marked stuck. Below the UI's JOB_STOP_S, so a cancelled job exits by itself.
JOIN_S = 8.0
Result = tuple[list[dict[str, Any]], dict[str, Any]]


def enabled() -> bool:
    """False when ``CUTPOINT_ANALYZE_OVERLAP=0``."""
    return (paths.env(ENV) or "").strip() != "0"


def vram_ok() -> bool:
    """Whether the first GPU has :data:`MIN_FREE_MIB` free and :data:`MIN_TOTAL_MIB` in all;
    True when nvidia-smi is missing or does not answer (the overlap never needs it)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        free, total = (int(v) for v in out.strip().splitlines()[0].split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return True
    return free >= MIN_FREE_MIB and total >= MIN_TOTAL_MIB


@dataclass
class _Job:
    """One offered source: its result (or error) and what the worker held back."""

    source: Source
    done: threading.Event = field(default_factory=threading.Event)
    shares: list[float] = field(default_factory=list)  # on_progress values, in order
    records: list[tuple[logging.Handler, logging.LogRecord]] = field(default_factory=list)
    result: Result | None = None
    error: BaseException | None = None
    skipped: bool = False  # declined or cancelled: read inline


REPLAY = "cutpoint_prefetch_replay"  # set on a held record when it is replayed


class _Hold(logging.Filter):
    """On one of the root logger's handlers: holds the worker thread's records in the job
    it is on, with this handler, so :meth:`OcrPrefetch.take` replays them as each handler
    would have emitted them."""

    def __init__(self, owner: OcrPrefetch, handler: logging.Handler) -> None:
        super().__init__()
        self.owner = owner
        self.handler = handler

    def filter(self, record: logging.LogRecord) -> bool:
        if record.thread != self.owner.worker_ident or getattr(record, REPLAY, False):
            return True
        job = self.owner.current
        if job is not None:
            job.records.append((self.handler, record))
        return False


class OcrPrefetch:
    """The background OCR reader of one analysis (a context manager; leaving it stops it).

    Args:
        project: The project folder (project-local presets, ``OcrParams.project``).
        sources: Every source of the analysis, in order (their games need a preset).
        params: The OCR parameters ``run_events`` will use (its ``project`` is set here).
    """

    def __init__(self, project: Path, sources: Sequence[Path], params: OcrParams) -> None:
        params.project = project
        self.project = project
        self.sources = list(sources)
        self.params = params
        # the engine run_events uses too, so one is made at most
        self.make_engine: Callable[[], OcrEngine] = functools.cache(lambda: OcrEngine())
        self._jobs: dict[str, _Job] = {}
        self._queue: queue.Queue[_Job | None] = queue.Queue()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._stuck: threading.Thread | None = None  # a worker close() could not wait out
        self._holds: list[_Hold] = []
        self.worker_ident: int | None = None
        self.current: _Job | None = None  # the job the worker is on (its records are held)
        self._open = True  # sources may still be offered
        self._checked: bool | None = None  # presets and VRAM, checked by the worker
        self.declined: str | None = None  # why the worker read nothing

    def __enter__(self) -> OcrPrefetch:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- main thread
    def offer(self, src: Path, duration: float, ok: bool) -> None:
        """A source whose read was checked, ``duration`` as highlights.json records it.
        ``ok`` False (a read error, a recovered part, Whisper not on CUDA): neither it nor
        a later source is read in the background."""
        if not self._open or str(src) in self._jobs:
            return
        if not ok:
            self._open = False
            return
        try:
            job = _Job(load_source(src, duration))
        except Exception:  # never fails the transcription: the OCR is read in its turn
            logger.debug("OCR prefetch off: %s not loadable", src, exc_info=True)
            self._open = False
            return
        self._jobs[str(src)] = job
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="ocr-prefetch", daemon=True)
            self._thread.start()
            self.worker_ident = self._thread.ident
            self._holds = [_Hold(self, h) for h in logging.getLogger().handlers]
            for hold in self._holds:
                hold.handler.addFilter(hold)
        self._queue.put(job)

    def take(
        self, src: Source, params: OcrParams, on_progress: Callable[[float], None] | None
    ) -> Result | None:
        """The background result for ``src``, waited for; None when it was not read in the
        background or read with other inputs than ``params`` (the caller reads it then;
        the worker is stopped first). Its progress goes to ``on_progress`` and its log
        records are replayed. Re-raises the read's error."""
        job = self._jobs.get(str(src.path))
        if job is None or job.source != src or params != self.params:
            self.close()
            return None
        sent = 0
        while True:
            finished = job.done.wait(POLL_S)
            shares = job.shares[sent:]
            sent += len(shares)
            if on_progress is not None:
                for share in shares:
                    on_progress(share)
            if finished or self._thread is None or not self._thread.is_alive():
                break
        if not job.done.is_set() or job.skipped:  # a cancelled read's records are dropped
            if self.declined:
                logger.debug("OCR prefetch off: %s", self.declined)
                self.declined = None  # said once
            self.close()
            return None
        for handler, record in job.records:
            setattr(record, REPLAY, True)
            handler.handle(record)
        job.records.clear()
        if job.error is not None:
            raise job.error
        return job.result

    def close(self) -> None:
        """Stop the worker (a read in progress is cancelled: nothing after its last raw
        chunk is kept) and drop what was not taken. Idempotent."""
        self._open = False
        self._cancel.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            self._queue.put(None)
            thread.join(JOIN_S)
            if thread.is_alive():
                logger.debug("OCR prefetch still busy after %.0fs; left to exit", JOIN_S)
                self._stuck = thread
        for hold in self._holds:
            hold.handler.removeFilter(hold)
        self._holds = []
        for job in self._jobs.values():
            if not job.done.is_set() or job.skipped:  # a finished one may still be taken
                job.records.clear()

    @property
    def stuck(self) -> bool:
        """The worker outlived :meth:`close` (one long OCR call): its engine and the raw
        cache of the source it is on are not free (see :meth:`wait_for`)."""
        return self._stuck is not None and self._stuck.is_alive()

    def wait_for(self, src: Path) -> None:
        """Wait until a stuck worker is off ``src``, whose raw OCR cache it may still write
        (it stops at its next frame and starts no other source); in steps of
        :data:`POLL_S`, so Ctrl+C gets through."""
        while self.stuck and (job := self.current) is not None and job.source.path == src:
            assert self._stuck is not None
            self._stuck.join(POLL_S)

    # ---- worker thread
    def _check(self) -> bool:
        """Presets for every game and the VRAM gate (once, on the first source)."""
        for path in self.sources:
            game = game_key_for_path(str(path), self.params.game, self.project)
            if load_preset(game, self.project) is None:
                self.declined = f"no HUD preset for {game!r} (discovery runs first)"
                return False
        if not vram_ok():
            self.declined = f"less than {MIN_FREE_MIB} MiB VRAM free or {MIN_TOTAL_MIB} in all"
            return False
        return True

    def _run(self) -> None:
        while (job := self._queue.get()) is not None:
            self.current = job
            try:
                if self._cancel.is_set():
                    job.skipped = True
                    continue
                if self._checked is None:
                    self._checked = self._check()
                if not self._checked:
                    job.skipped = True
                    continue
                job.result = detect_ocr_events(
                    job.source,
                    self.params,
                    None,
                    on_progress=job.shares.append,
                    make_engine=self.make_engine,
                    cancel=self._cancel,
                )
            except OcrCancelled:  # cancelled, or a CUDA pass failed: read in its turn
                job.skipped = True
                self._cancel.set()  # the later sources too, after it
            except BaseException as err:  # noqa: BLE001 - re-raised by take()
                job.error = err
                self._cancel.set()  # a serial run stops at this source too
            finally:
                self.current = None
                job.done.set()
