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

``CUTPOINT_ANALYZE_SHARED_DECODE`` (experimental, off by default) also makes the scene proxy
of a whole, clean source of up to ``LONG_FILE_S`` in the background (:class:`SceneShare`):
``split`` decodes it in the OCR read's CUDA pass (:class:`~cutpoint.ffmpeg_utils.SplitDecode`,
the raw frames spooled to the cache folder when the free disk allows), ``proxy`` runs the
scene proxy's own pass beside the OCR read (a second NVDEC session; the control). The scene
cuts use that proxy, wait for it while it is made, and decode as before when it is not
started yet or failed: the results are the same.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import math
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cutpoint import paths
from cutpoint.events.common import Source, load_source
from cutpoint.events.killfeed import (
    DecodeHook,
    OcrCancelled,
    OcrEngine,
    OcrParams,
    detect_ocr_events,
)
from cutpoint.events.regions import game_key_for_path, load_preset
from cutpoint.ffmpeg_utils import SPOOL_DIR, SplitDecode, run_cancellable, spool_fits

logger = logging.getLogger(__name__)

ENV = "ANALYZE_OVERLAP"  # CUTPOINT_ANALYZE_OVERLAP=0: OCR waits for its turn (escape hatch)
SHARED_ENV = "ANALYZE_SHARED_DECODE"  # split | proxy | 0 (default): the scene proxy (experimental)
SHARED_MODES = ("split", "proxy")
PROXY_TMP = "proxy.tmp.mp4"  # a background scene proxy until it is complete
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


def shared_mode() -> str | None:
    """``split`` or ``proxy`` from ``CUTPOINT_ANALYZE_SHARED_DECODE``; None (off) otherwise."""
    mode = (paths.env(SHARED_ENV) or "").strip().lower()
    return mode if mode in SHARED_MODES else None


def sweep(cdir: Path) -> None:
    """Remove what an interrupted background scene proxy may have left in ``cdir``."""
    shutil.rmtree(cdir / SPOOL_DIR, ignore_errors=True)
    with contextlib.suppress(OSError):
        (cdir / PROXY_TMP).unlink(missing_ok=True)


@dataclass(frozen=True)
class ScenePlan:
    """The scene proxy :func:`cutpoint.analyze.scene_cuts` would make of a whole source:
    ``vf`` and ``enc`` are its filter chain and encoder arguments, ``proxy`` its file."""

    io: Path  # the file decoded (the source or its source_remap copy)
    cdir: Path
    duration: float
    vf: str
    enc: tuple[str, ...]

    @property
    def proxy(self) -> Path:
        return self.cdir / "proxy.mp4"


class SceneShare:
    """The background scene proxy of one source: ``pending`` until the worker claims it
    (``running``), then ``done`` or ``failed``; ``declined`` when the scene cuts came first
    or the share was refused. Thread-safe."""

    def __init__(self, plan: ScenePlan, mode: str) -> None:
        self.plan = plan
        self.mode = mode
        self.state = "pending"
        self.why = ""  # why it was declined
        self.waited = 0.0  # seconds the scene cuts waited for it
        self.spool_peak: int | None = None  # split: most spool bytes on disk at once
        self.taken = False  # the scene cuts got the proxy (else close() removes it)
        self._cond = threading.Condition()

    def claim(self) -> bool:
        """Worker side: start it unless the scene cuts came first."""
        with self._cond:
            if self.state != "pending":
                return False
            self.state = "running"
            return True

    def decline(self, why: str) -> None:
        """Worker side: it will not be made (the scene cuts decode as before)."""
        with self._cond:
            if self.state == "pending":
                self.state, self.why = "declined", why

    def finish(self, ok: bool, peak: int | None = None) -> None:
        """Worker side: ``plan.proxy`` is complete (``ok``) or nothing is left."""
        with self._cond:
            self.spool_peak = peak
            self.state = "done" if ok else "failed"
            self._cond.notify_all()

    def result(self) -> Path | None:
        """Scene cuts side: the complete proxy, waited for while it is made (in steps of
        :data:`POLL_S`, so Ctrl+C gets through); None (decode it as before) when it was not
        started, which it then never is, or failed."""
        t0 = time.perf_counter()
        with self._cond:
            if self.state == "pending":
                self.state, self.why = "declined", "not started before the scene cuts"
                return None
            while self.state == "running":
                self._cond.wait(POLL_S)
            self.waited = time.perf_counter() - t0
            self.taken = self.state == "done"
            return self.plan.proxy if self.taken else None

    def summary(self) -> str:
        """For the measurement log line."""
        text = f"{self.state}{f' ({self.why})' if self.why else ''}, waited {self.waited:.1f}s"
        if self.spool_peak is not None:
            text += f", spool peak {self.spool_peak / 2**20:.0f} MiB"
        return text


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
    share: SceneShare | None = None  # its scene proxy, made in the background


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
        self._off = ""  # the same, for the shares of every source
        self.mode = shared_mode()  # the background scene proxy (None: off)
        self._split: SplitDecode | None = None  # the worker's shared decode
        self._proxies: list[threading.Thread] = []  # proxy mode: scene proxy passes
        self._proxy_vram: bool | None = None  # proxy mode: the VRAM gate, checked once

    def __enter__(self) -> OcrPrefetch:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- main thread
    def offer(
        self, src: Path, duration: float, ok: bool, scene: ScenePlan | None = None
    ) -> SceneShare | None:
        """A source whose read was checked, ``duration`` as highlights.json records it.
        ``ok`` False (a read error, a recovered part, Whisper not on CUDA): neither it nor
        a later source is read in the background. ``scene``: its scene proxy, made in the
        background in a shared-decode mode; returns the share the scene cuts then use."""
        if not self._open or str(src) in self._jobs:
            return None
        if not ok:
            self._open = False
            return None
        try:
            job = _Job(load_source(src, duration))
        except Exception:  # never fails the transcription: the OCR is read in its turn
            logger.debug("OCR prefetch off: %s not loadable", src, exc_info=True)
            self._open = False
            return None
        sweep(job.source.cdir)  # also after a split run, when the mode is off now
        if self.mode is not None and scene is not None:
            job.share = SceneShare(scene, self.mode)
        self._jobs[str(src)] = job
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="ocr-prefetch", daemon=True)
            self._thread.start()
            self.worker_ident = self._thread.ident
            self._holds = [_Hold(self, h) for h in logging.getLogger().handlers]
            for hold in self._holds:
                hold.handler.addFilter(hold)
        self._queue.put(job)
        if job.share is not None and job.share.mode == "proxy":
            if self._proxy_vram is None:
                self._proxy_vram = vram_ok()
            if not self._proxy_vram:
                job.share.decline(
                    f"less than {MIN_FREE_MIB} MiB VRAM free or {MIN_TOTAL_MIB} in all"
                )
                return job.share
            thread = threading.Thread(
                target=self._proxy, args=(job.share,), name="scene-proxy", daemon=True
            )
            self._proxies.append(thread)
            thread.start()
        return job.share

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
        sent, t0 = 0, time.perf_counter()
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
        if self.mode is not None:  # for measuring the shared decode
            waited = time.perf_counter() - t0
            logger.info("%s: background OCR taken after a %.1fs wait", src.name, waited)
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
        for proxy in self._proxies:  # killed on cancel: only the cleanup is left
            proxy.join(JOIN_S)
        self._proxies = []
        for job in self._jobs.values():  # a proxy the scene cuts never used
            if job.share is not None and job.share.state == "done" and not job.share.taken:
                with contextlib.suppress(OSError):
                    job.share.plan.proxy.unlink(missing_ok=True)
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
            why = "the OCR read needed no fresh CUDA decode (cached, resumed or in parts)"
            try:
                if self._cancel.is_set():
                    job.skipped, why = True, "the background OCR was stopped"
                    continue
                if self._checked is None:
                    self._checked = self._check()
                    self._off = self.declined or ""  # kept: take() clears declined once said
                if not self._checked:
                    job.skipped, why = True, self._off
                    continue
                job.result = detect_ocr_events(
                    job.source,
                    self.params,
                    None,
                    on_progress=job.shares.append,
                    make_engine=self.make_engine,
                    cancel=self._cancel,
                    decode=self._decoder(job),
                )
            except OcrCancelled:  # cancelled, or a CUDA pass failed: read in its turn
                job.skipped, why = True, "the OCR read was cancelled or its CUDA pass failed"
                self._cancel.set()  # the later sources too, after it
            except BaseException as err:  # noqa: BLE001 - re-raised by take()
                job.error, why = err, "the OCR read failed"
                self._cancel.set()  # a serial run stops at this source too
            finally:
                if self._split is not None:  # its frames were not all read: ffmpeg is killed
                    self._split.close()
                    self._split = None
                if job.share is not None and job.share.mode == "split":  # unless it ran
                    job.share.decline(why)
                self.current = None
                job.done.set()

    def _decoder(self, job: _Job) -> DecodeHook | None:
        """split mode: the decode hook that reads the OCR frames from one CUDA pass that also
        makes the job's scene proxy (when the scene cuts did not come first and the disk
        holds the worst-case spool)."""
        share = job.share
        if share is None or share.mode != "split":
            return None
        plan = share.plan

        def decode(rect: tuple[int, int, int, int], vf: str, errors: list[str], exits: list[int]):
            width, height = rect[2:]
            # every frame of the file: with no backpressure the OCR read may lag the whole way
            need = width * height * 3 * math.ceil(self.params.fps * plan.duration + 1)
            if plan.io != job.source.io_path or not self.params.hwaccel:
                share.decline("another file or decoder than the OCR read")
                return None
            if not spool_fits(plan.cdir, need):
                why = f"less than {need / 2**30:.1f} GiB + 1 GiB free for the spool"
                share.decline(why)
                # said now, not held for the OCR turn (REPLAY lets it through _Hold)
                logger.info("%s: shared decode off: %s", job.source.name, why, extra={REPLAY: True})
                return None
            if not share.claim():
                return None
            try:
                self._split = SplitDecode(
                    plan.io,
                    size=(width, height),
                    raw_vf=vf,
                    stamp_fps=self.params.fps,
                    scene_vf=plan.vf,
                    scene_enc=plan.enc,
                    proxy=plan.proxy,
                    spool=plan.cdir / SPOOL_DIR,
                    cancel=self._cancel,
                    on_exit=share.finish,
                    stderr_lines=errors,
                )
            except Exception as err:  # noqa: BLE001 - not started: both decode as before
                logger.debug("shared decode not started: %s", err)
                sweep(plan.cdir)
                share.finish(False)
                return None
            return self._split.frames(exits)

        return decode

    def _proxy(self, share: SceneShare) -> None:
        """proxy mode (a thread): the scene proxy's own CUDA pass, beside the OCR read."""
        if not share.claim():
            return
        plan, ok = share.plan, False
        tmp = plan.cdir / PROXY_TMP
        try:
            args = ["-hide_banner", "-nostats", "-v", "error", "-y", "-hwaccel", "cuda"]
            args += ["-i", str(plan.io), "-an", "-vf", plan.vf, *plan.enc, str(tmp)]
            ok = run_cancellable(args, self._cancel)
            if ok:
                tmp.replace(plan.proxy)
        except OSError as err:
            logger.debug("scene proxy not made in the background: %s", err)
            ok = False
        finally:
            if not ok:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
            share.finish(ok)
