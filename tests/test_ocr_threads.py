"""OCR engine thread cap (#374) and lanes (#403): every OcrEngine passes the cap to
RapidOCR, split over its lanes, which run OCR calls at once with the output of one lane
(fake engine class, no model)."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cutpoint.events import discover, killfeed
from cutpoint.events.regions import Region


def _fake_rapidocr(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    made: list[dict[str, Any]] = []

    class RapidOCR:
        def __init__(self, **kwargs: Any) -> None:
            made.append(kwargs)

    module = ModuleType("rapidocr_onnxruntime")
    module.RapidOCR = RapidOCR  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", module)
    return made


def test_thread_cap_is_at_most_four_and_the_cpu_count() -> None:
    assert min(4, os.cpu_count() or 1) == killfeed.OCR_THREADS
    assert 1 <= killfeed.OCR_THREADS <= 4


@pytest.mark.parametrize("det_side", [killfeed.OCR_DET_SIDE, 1280])  # OCR layer, discovery
def test_engine_passes_the_thread_cap(monkeypatch: pytest.MonkeyPatch, det_side: int) -> None:
    made = _fake_rapidocr(monkeypatch)
    monkeypatch.setattr(killfeed, "OCR_THREADS", 3)

    killfeed.OcrEngine(det_side=det_side, lanes=1)

    assert made == [
        {"det_limit_type": "max", "det_limit_side_len": det_side, "intra_op_num_threads": 3}
    ]


def test_lanes_split_the_thread_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    made = _fake_rapidocr(monkeypatch)
    monkeypatch.setattr(killfeed, "OCR_THREADS", 4)

    killfeed.OcrEngine(lanes=2)

    assert [m["intra_op_num_threads"] for m in made] == [2, 2]
    assert (2 if killfeed.OCR_THREADS >= 4 else 1) == killfeed.OCR_LANES


def test_discovery_reads_on_one_lane_with_every_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    made = _fake_rapidocr(monkeypatch)
    monkeypatch.setattr(killfeed, "OCR_THREADS", 4)
    monkeypatch.setattr(discover, "_burst_frames", lambda *a: [])
    params = discover.DiscoverParams()
    src = SimpleNamespace(name="a.mp4", io_path=tmp_path / "a.mp4")

    discover._ocr_frames([(src, 1.0)], params, tmp_path / "frames", tmp_path / "ocr.jsonl")

    assert made == [
        {
            "det_limit_type": "max",
            "det_limit_side_len": params.width,
            "intra_op_num_threads": 4,
            "rec_batch_num": 1,  # each box recognised alone (#443)
        }
    ]


def test_the_ocr_layer_keeps_rapidocrs_rec_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only discovery recognises one box at a time (#443): the OCR layer's engines (as
    pipeline and ocr_prefetch make them) keep RapidOCR's batching, so its raw reads and
    their cache key are unchanged."""
    made = _fake_rapidocr(monkeypatch)
    monkeypatch.setattr(killfeed, "OCR_THREADS", 4)

    engine = killfeed.OcrEngine()

    assert engine.rec_batch is None
    assert len(made) == killfeed.OCR_LANES
    assert all("rec_batch_num" not in kwargs for kwargs in made)


class _Lanes:
    """Fake RapidOCR engines that track how many run at once: detection finds one box
    per bright row band, recognition reads a strip as the text of its mean pixel value.
    Each call takes ``delay(img, use_det)`` seconds; by default the darkest strip is the
    slowest, so later inputs finish first."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.busy: set[int] = set()
        self.peak = 0

    @staticmethod
    def delay(img: np.ndarray, use_det: bool) -> float:
        return 0.05 if not use_det and img.mean() < 15 else 0.01

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lanes = self

        class RapidOCR:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def __call__(self, img, use_det=True, use_cls=True, use_rec=True):
                with lanes.lock:
                    assert id(self) not in lanes.busy  # one thread per engine at a time
                    lanes.busy.add(id(self))
                    lanes.peak = max(lanes.peak, len(lanes.busy))
                time.sleep(lanes.delay(img, use_det))
                with lanes.lock:
                    lanes.busy.discard(id(self))
                if not use_det:
                    return [[f"v{int(img.mean())}", 0.9]], None
                ys = np.flatnonzero(img.max(axis=(1, 2)) > 0)
                if not len(ys):
                    return None, None
                box = np.array([[2, ys[0]], [40, ys[0]], [40, ys[-1]], [2, ys[-1]]], float)
                return [box], None

        module = ModuleType("rapidocr_onnxruntime")
        module.RapidOCR = RapidOCR  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", module)


def test_rows_recognised_at_once_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    lanes = _Lanes()
    lanes.install(monkeypatch)
    img = np.zeros((100, 60, 3), np.uint8)
    for i in range(4):
        img[i * 25 : i * 25 + 20] = 10 * (i + 1)
    rows = [(i * 25 + 2, 14, i * 25 + 18, 50) for i in range(4)]

    out = [killfeed.OcrEngine(lanes=n).read_rows(img, rows, 0.5, pad=2) for n in (1, 2)]

    assert out[0] == out[1]
    assert [text for _, text, _ in out[1]] == ["v10", "v20", "v30", "v40"]
    assert lanes.peak == 2


def test_reader_gates_regions_at_once_with_the_records_of_one_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lanes = _Lanes()
    lanes.delay = lambda img, use_det: 0.05 if img.shape[0] == 60 else 0.01  # a is slowest
    lanes.install(monkeypatch)
    regions = tuple(
        Region(name, x, 0, 100, h, "notice", gate="det")
        for name, x, h in (("a", 0, 60), ("b", 100, 50))
    )
    rects = {r.name: (r.x, r.y, r.w, r.h) for r in regions}
    frames = [np.zeros((60, 200, 3), np.uint8) for _ in range(3)]
    frames[1][10:30, :100] = 50  # region a shows text from frame 1, b from frame 2
    frames[2][10:30] = 50

    states = []

    def records(n: int) -> list[dict[str, Any]]:
        engine = killfeed.OcrEngine(lanes=n)
        size = (1920, 1080)
        reader = killfeed.OcrReader(
            regions, rects, (0, 0, 200, 60), engine, killfeed.OcrParams(), 1.0, "bottom", size
        )
        recs = [reader.read(i, i / 3, f)[0].regions for i, f in enumerate(frames)]
        states.append(reader.state()[0])
        return recs

    one, two = records(1), records(2)

    assert one == two
    assert [list(s["rows"]) for s in states] == [["a", "b"]] * 2  # keyed in region order
    assert [list(r) for r in two] == [["a", "b"]] * 3  # in region order
    assert [r["a"]["g"] for r in two] == ["det", "read", "keep"]
    assert [r["b"]["g"] for r in two] == ["det", "det", "read"]
    assert two[2]["b"]["items"]
    assert lanes.peak == 2


def test_det_side_stays_in_the_raw_cache_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """OCR_DET_SIDE has no effect on RapidOCR 1.4.4 (#443) but is part of the raw OCR cache
    key: the key of a read stays byte-identical, so existing raw caches still hit."""
    from cutpoint import cache  # noqa: PLC0415
    from cutpoint.events import ocr_raw  # noqa: PLC0415

    _fake_rapidocr(monkeypatch)

    side = killfeed.OcrEngine(lanes=1).det_side
    key = ocr_raw.raw_key(3.0, [], "bottom", 0.5, [], side) | {"engine": "1.4.4"}

    assert side == killfeed.OCR_DET_SIDE == 960
    assert cache.params_hash(key) == "2e53590a5c"  # the key before #443


@pytest.mark.slow
def test_rapidocr_ignores_the_det_side() -> None:
    """With det_limit_type="max", RapidOCR 1.4.4 picks the detection size from the image
    (#443), so det_side does not change what is read. Fails when a RapidOCR version honours
    it: then OCR_DET_SIDE changes the output and its comment is wrong."""
    pytest.importorskip("rapidocr_onnxruntime")
    import cv2  # noqa: PLC0415

    img = np.zeros((200, 1400, 3), np.uint8)  # wider than 960: scaled if det_side were used
    for i, word in enumerate(("Alpha downed Bravo", "Charlie 1234 eliminated")):
        cv2.putText(img, word, (20 + 500 * i, 60 + 80 * i), 0, 1.4, (255, 255, 255), 3)

    out = [killfeed.OcrEngine(det_side=side, lanes=1)(img, 0.5) for side in (320, 960)]

    assert [text for _, text, _ in out[1]] == ["Alpha downed Bravo", "Charlie 1234 eliminated"]
    assert repr(out[0]) == repr(out[1])
