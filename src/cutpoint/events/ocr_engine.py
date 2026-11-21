"""The RapidOCR engine of the OCR layer (split from ``killfeed.py``, #1085 A347):
:class:`OcrEngine` and :class:`SharedOcr`.
"""

from __future__ import annotations

import contextlib
import os
import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from cutpoint import memguard

# --------------------------------------------------------------------------- OCR logic


# onnxruntime intra-op threads for the OCR models (det, cls, rec). CPU OCR stops scaling at
# about 4 threads: 8 / 4 / 2 / 1 threads took 4.0 / 3.3 / 6.9 / 8.7 s per 1280 px frame on an
# 8-core CPU, so 4 is ~18% faster than the default (one per physical core). Capped at the CPU
# count so small machines are not oversubscribed. The output does not depend on it (#374).
OCR_THREADS = min(4, os.cpu_count() or 1)
# OCR lanes (#403): the thread budget split over engines that run at once. Crops and row strips
# are small and use intra-op threads poorly: on 4 vCPUs, 2 engines x 2 threads read a 300 s
# clip's HUD in 286 s instead of 345 s with 1 x 4 (4 x 1 was slower). The output does not
# depend on it.
OCR_LANES = 2 if OCR_THREADS >= 4 else 1


class OcrEngine:
    """Thin wrapper around RapidOCR (onnxruntime, CPU).

    With ``lanes`` > 1 it holds that many RapidOCR engines, which share the
    :data:`OCR_THREADS` budget, and :meth:`map` and :meth:`read_rows` run OCR calls on
    them at once. Each call runs on one engine, which gives the same result on any of
    them, so the output does not depend on the lanes.

    ``rec_batch`` sets how many text boxes a full call (:meth:`__call__`) recognises in one
    batch; None keeps RapidOCR's default (6). A batch pads every box to the widest one, so
    the text of a box can depend on the others: 1 recognises each box alone.
    """

    def __init__(self, lanes: int | None = None, rec_batch: int | None = None) -> None:
        from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415

        memguard.check("ocr")
        self.lanes = OCR_LANES if lanes is None else lanes
        self.rec_batch = rec_batch
        extra = {} if rec_batch is None else {"rec_batch_num": rec_batch}

        # Default det config upscales the short side to 736 px ("limit_type: min"),
        # which makes wide HUD strips ~10x slower; cap the long side instead. RapidOCR
        # 1.4.4 then picks the limit from the image's long side (960 below 960 px, 1500
        # below 1500 px, else 2000), so crops up to 2000 px are never scaled and
        # det_limit_side_len would change nothing (#443).
        engines = [
            RapidOCR(
                det_limit_type="max",
                intra_op_num_threads=max(1, OCR_THREADS // self.lanes),
                **extra,
            )
            for _ in range(self.lanes)
        ]
        self._free: queue.SimpleQueue[Any] = queue.SimpleQueue()  # the engines not in use
        for e in engines:
            self._free.put(e)
        self._pool = ThreadPoolExecutor(self.lanes, "ocr-lane") if self.lanes > 1 else None

    @contextlib.contextmanager
    def _lane(self) -> Iterator[Any]:
        """An engine no other thread uses meanwhile."""
        e = self._free.get()
        try:
            yield e
        finally:
            self._free.put(e)

    def map(self, fn: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]:
        """``fn`` of every item, in order; on the lanes at once when there are several.
        ``fn`` may make OCR calls of this engine but not call :meth:`map` again."""
        if self._pool is None or len(items) < 2:
            return [fn(x) for x in items]
        return list(self._pool.map(fn, items))

    def detect(self, bgr: np.ndarray, scale: float = 1.0) -> list[list]:
        """Text boxes only (no recognition), used by the "det" gate.

        ``scale`` < 1 detects on a downscaled copy (much cheaper); boxes are
        returned in the coordinates of ``bgr``.
        """
        img = bgr
        if scale < 1.0:
            import cv2  # noqa: PLC0415

            img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        with self._lane() as e:
            result, _ = e(np.ascontiguousarray(img), use_cls=False, use_rec=False)
        boxes = [b.tolist() if hasattr(b, "tolist") else b for b in (result or [])]
        if scale < 1.0:
            boxes = [[[p[0] / scale, p[1] / scale] for p in b] for b in boxes]
        return boxes

    def read_rows(
        self,
        bgr: np.ndarray,
        rows: list[tuple[float, float, float, float]],
        min_score: float,
        pad: float = 6.0,
    ) -> list[tuple[list, str, float]]:
        """Recognise text rows ``(y0, x0, y1, x1)`` of ``bgr`` without a second
        detection pass (one recognition per row strip, ~25-60 ms each).

        Returns items shaped like :meth:`__call__` (box = padded row rectangle).
        """
        h, w = bgr.shape[:2]
        strips = []
        for y0, x0, y1, x1 in rows:
            a, b = int(max(0, y0 - pad / 2)), int(min(h, y1 + pad / 2))
            c, d = int(max(0, x0 - 2 * pad)), int(min(w, x1 + pad))
            if b - a >= 6 and d - c >= 6:
                strips.append((a, b, c, d))

        def recognise(strip: tuple[int, int, int, int]) -> Any:
            a, b, c, d = strip
            with self._lane() as e:
                res, _ = e(
                    np.ascontiguousarray(bgr[a:b, c:d]), use_det=False, use_cls=False, use_rec=True
                )
            return res

        out = []
        for (a, b, c, d), res in zip(strips, self.map(recognise, strips), strict=True):
            for item in res or []:
                text, score = str(item[0]), float(item[1])
                if score >= min_score and text.strip():
                    out.append(([[c, a], [d, a], [d, b], [c, b]], text, score))
        return out

    def __call__(self, bgr: np.ndarray, min_score: float) -> list[tuple[list, str, float]]:
        with self._lane() as e:
            result, _ = e(np.ascontiguousarray(bgr), use_cls=False)
        return [
            (box, text, float(score))
            for box, text, score in (result or [])
            if float(score) >= min_score and text.strip()
        ]


class SharedOcr:
    """One lazily built :class:`OcrEngine` for several reads (#1085 A694, A707): the first
    call builds it with ``kwargs`` (the memory guard checks then, as for any engine), later
    calls return the same one until :meth:`release`, which frees it for the next models.
    Thread-safe: one engine is built however many threads ask at once."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._lock = threading.Lock()
        self._engine: OcrEngine | None = None

    def __call__(self) -> OcrEngine:
        with self._lock:
            if self._engine is None:
                self._engine = OcrEngine(**self._kwargs)
            return self._engine

    def release(self) -> None:
        """Drop the engine; the next call builds a new one."""
        with self._lock:
            self._engine = None
