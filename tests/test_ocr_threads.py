"""OCR engine thread cap (#374): every OcrEngine passes the cap to RapidOCR (fake engine
class, no model)."""

from __future__ import annotations

import os
import sys
from types import ModuleType
from typing import Any

import pytest

from cutpoint.events import killfeed


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

    killfeed.OcrEngine(det_side=det_side)

    assert made == [
        {"det_limit_type": "max", "det_limit_side_len": det_side, "intra_op_num_threads": 3}
    ]
