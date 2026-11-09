"""Regions outside the frame or of no size are clamped or dropped, not passed to OpenCV."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from cutpoint.events import killfeed
from cutpoint.events import regions as rg
from cutpoint.events.common import Source
from cutpoint.events.discover import _padded
from cutpoint.events.killfeed import OcrParams, OcrReader, readable_regions
from cutpoint.events.regions import HudPreset, Region, preset_from_dict


@pytest.fixture(autouse=True)
def _fresh_warnings():
    rg._WARNED.clear()


def _region(name: str, x: int, y: int, w: int, h: int) -> Region:
    return Region(name, x, y, w, h, "notice")


def test_negative_height_dropped_with_warning_others_kept(caplog):
    good = _region("a", 10, 10, 200, 100)
    bad = _region("extra_notice", 11, 3147, 149, -2067)
    with caplog.at_level(logging.WARNING):
        out = rg.guard_regions([bad, good], "src")
    assert out == [good]
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(msgs) == 1 and "extra_notice@11,3147 149x-2067" in msgs[0] and "dropped" in msgs[0]


def test_zero_width_dropped():
    assert rg.guard_regions([_region("z", 5, 5, 0, 50)], "src") == []


def test_partly_outside_is_clamped(caplog):
    with caplog.at_level(logging.WARNING):
        (out,) = rg.guard_regions([_region("edge", 1800, -20, 300, 100)], "src")
    assert (out.x, out.y, out.w, out.h) == (1800, 0, 120, 80)
    assert "clamped" in caplog.text and "edge" in caplog.text


def test_fully_outside_dropped():
    assert rg.guard_regions([_region("far", 2000, 10, 50, 50)], "src") == []
    assert rg.guard_regions([_region("low", 10, 1080, 50, 50)], "src") == []


def test_warning_once_per_region(caplog):
    bad = _region("b", 0, 0, -5, 5)
    with caplog.at_level(logging.WARNING):
        rg.guard_regions([bad], "src")
        rg.guard_regions([bad], "src")
    assert caplog.text.count("dropped") == 1


def test_discovery_padding_below_the_frame_is_dropped():
    r = _padded({"name": "n", "x": 11, "y": 3159, "w": 125, "h": 20}, "notice")
    assert r.h < 0
    assert rg.guard_regions([r], "src") == []


def test_preset_with_a_bad_region_loads_the_rest():
    data = {
        "game": "g",
        "regions": [
            {"name": "ok", "x": 0, "y": 0, "w": 100, "h": 100, "kind": "feed"},
            {"name": "bad", "x": 11, "y": 3147, "w": 149, "h": -2067, "kind": "notice"},
            {"name": "edge", "x": 1900, "y": 0, "w": 100, "h": 100, "kind": "notice"},
        ],
    }
    preset = preset_from_dict(data)
    assert [(r.name, r.w) for r in preset.regions] == [("ok", 100), ("edge", 20)]


def test_region_without_pixels_in_the_frame_is_skipped(caplog):
    # a 606x1080 frame: a sliver of 4 reference px scales to no pixel
    ok = _region("ok", 0, 0, 400, 300)
    off = _region("off", 1800, 0, 4, 100)
    with caplog.at_level(logging.WARNING):
        out = readable_regions((ok, off), 606, 1080, "clip.mp4")
    assert out == (ok,)
    assert "off" in caplog.text


def test_empty_crop_gives_no_text():
    class Engine:
        def detect(self, *_a):
            raise AssertionError("no crop, no detection")

        def __call__(self, *_a):
            raise AssertionError("no crop, no reading")

    empty = _region("empty", 10, 10, 20, 20)
    rects = {"empty": (10, 10, 20, 20)}
    # the decoded area ends before the rectangle: its crop is clipped to nothing
    reader = OcrReader(
        (empty,), rects, (0, 0, 8, 8), Engine(), OcrParams(), 1.0, "bottom", (1920, 1080)
    )
    rec, crops = reader.read(0, 0.0, np.zeros((8, 8, 3), np.uint8))
    assert rec.regions == {} and crops == {}


def test_fractional_coordinates_inside_the_frame_give_no_warning(caplog):
    entry = {"name": "frac", "x": 10.5, "y": 20, "w": "100", "h": 50}
    with caplog.at_level(logging.WARNING):
        assert rg.guard_entry(entry, "src") is entry
    assert caplog.text == ""


def test_every_region_off_frame_skips_the_read(tmp_path, monkeypatch):
    preset = preset_from_dict({"game": "g", "bigtext": False, "regions": []})
    off = _region("off", 1800, 0, 4, 100)
    monkeypatch.setattr(killfeed, "game_key_for_path", lambda *_a, **_k: "g")
    monkeypatch.setattr(killfeed, "load_preset", lambda *_a, **_k: preset)
    monkeypatch.setattr(HudPreset, "all_regions", lambda self: (off,))
    monkeypatch.setattr(
        killfeed,
        "probe",
        lambda _p: type("Info", (), {"width": 606, "height": 1080, "duration": 5})(),
    )
    src = Source(tmp_path / "clip.mp4", 5.0, tmp_path, [])
    out = killfeed.detect_ocr_events(src, OcrParams(), object())
    assert out == ([], {"skipped": "no region in the frame", "game": "g"})
