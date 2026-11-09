"""Det-gate skip of the HUD OCR (#807): a det-gated region that has not changed since its
last det call keeps that call's outcome; opt-in (``ocr_det_skip``, ``--ocr-det-skip``).
Fake OCR engine and frames, invented HUD texts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import set_config, write
from cutpoint import cli, edit
from cutpoint.events import cli as events_cli
from cutpoint.events import killfeed, ocr_raw
from cutpoint.events.killfeed import OcrParams, OcrReader, scaled
from cutpoint.events.regions import preset_from_dict
from test_ocr_raw import _dirs, _Engine, _run, _Stop

RECT = (320, 160, 320, 160)  # the notice region (reference px = frame px at 1080p)
H, W = RECT[3], RECT[2]


class _Det:
    """An OCR engine whose detection always finds one box; counts the det calls."""

    def __init__(self) -> None:
        self.dets = 0

    def detect(self, crop: np.ndarray, scale: float = 1.0) -> list[list[list[float]]]:
        self.dets += 1
        return [[[10.0, 10.0], [90.0, 10.0], [90.0, 34.0], [10.0, 34.0]]]

    def read_rows(self, crop, rows, min_score, pad=6.0) -> list[tuple[list, str, float]]:
        return [([[10, 10], [90, 10], [90, 34], [10, 34]], "Hold", 0.95)]

    def __call__(self, crop: np.ndarray, min_score: float) -> list:
        return []


def _reader(engine: _Det, det_skip: bool = True) -> OcrReader:
    x, y, w, h = RECT
    region = {"name": "notice", "x": x, "y": y, "w": w, "h": h, "kind": "notice", "gate": "det"}
    regions = preset_from_dict(
        {"game": "zonewar", "regions": [region], "bigtext": False}
    ).all_regions()
    rects = {r.name: scaled(r, 1920, 1080) for r in regions}
    union = killfeed._union(list(rects.values()))
    params = OcrParams(game="zonewar", det_skip=det_skip)
    return OcrReader(regions, rects, union, engine, params, 1.0, "bottom", (1920, 1080))


def _frame(level: int = 100) -> np.ndarray:
    return np.full((H, W, 3), level, np.uint8)


def _look(reader: OcrReader, frame: np.ndarray, n: int = 0) -> dict[str, Any]:
    rec, _ = reader.read(n, n / 3, frame)
    return rec.regions["notice"]


def test_tile_diff_sees_a_small_changed_spot() -> None:
    a = np.full((40, 40), 100, np.uint8)
    b = a.copy()
    b[8:16, 16:24] += 10  # one tile of 25
    assert np.abs(b.astype(float) - a).mean() < 1  # the global mean hardly moves
    assert killfeed._tile_diff(a, b) == pytest.approx(10)
    edge = np.full((21, 21), 100, np.uint8)
    moved = edge.copy()
    moved[16:, 16:] += 9  # the 5x5 tile at the corner
    assert killfeed._tile_diff(edge, moved) == pytest.approx(9)


def test_unchanged_region_skips_the_det_call() -> None:
    engine = _Det()
    reader = _reader(engine)
    first = _look(reader, _frame(), 0)
    assert first["g"] == "read" and "s" not in first and engine.dets == 1
    again = _look(reader, _frame(103), 1)  # within DET_SKIP_MAX_DIFF
    assert engine.dets == 1
    assert again == {"g": "keep", "s": 1, "boxes": first["boxes"]}


def test_a_changed_tile_runs_det_although_the_mean_is_small() -> None:
    engine = _Det()
    reader = _reader(engine)
    _look(reader, _frame(), 0)
    frame = _frame()
    frame[0:32, 0:32] += 8  # one tile of the 4x scaled crop: 8, the crop's mean 0.16
    rec = _look(reader, frame, 1)
    assert engine.dets == 2 and "s" not in rec


def test_the_anchor_moves_only_on_a_det_call() -> None:
    engine = _Det()
    reader = _reader(engine)
    _look(reader, _frame(100), 0)
    _look(reader, _frame(105), 1)  # 5 from the anchor: skipped
    assert engine.dets == 1
    _look(reader, _frame(110), 2)  # 5 from the last frame, 10 from the anchor: det
    assert engine.dets == 2
    _look(reader, _frame(114), 3)  # 4 from the new anchor: skipped
    assert engine.dets == 2


def test_state_round_trip_keeps_the_skip() -> None:
    engine = _Det()
    reader = _reader(engine)
    boxes = _look(reader, _frame(), 0)["boxes"]
    state, arrays = reader.state()
    assert state["boxes"] == {"notice": boxes} and set(arrays) == {"anchor0"}
    resumed = _reader(engine)
    resumed.restore(json.loads(json.dumps(state)), arrays)
    rec = _look(resumed, _frame(101), 1)
    assert engine.dets == 1 and rec == {"g": "keep", "s": 1, "boxes": boxes}


def test_off_by_default_changes_nothing() -> None:
    assert OcrParams().det_skip is False
    engine = _Det()
    reader = _reader(engine, det_skip=False)
    for n in range(3):
        assert "s" not in _look(reader, _frame(), n)
    assert engine.dets == 3
    state, arrays = reader.state()
    assert set(state) == {"layout", "rows"} and arrays == {}
    # the raw read's key is the one of a read before the option existed
    args = (3.0, [], "bottom", 0.6, [])
    assert "det_skip" not in ocr_raw.raw_key(*args)
    assert ocr_raw.raw_key(*args, [4, 8, 8.0])["det_skip"] == [4, 8, 8.0]


def test_skip_read_equals_the_plain_read_with_fewer_det_calls(tmp_path: Path, monkeypatch):
    plain_dir, skip_dir = _dirs(tmp_path, "plain", "skip")
    plain, plain_stats = _run(plain_dir, monkeypatch, _Engine())
    events, stats = _run(skip_dir, monkeypatch, _Engine(), det_skip=True)
    assert events == plain
    assert "det_skipped" not in plain_stats
    # the still menu at 8-20 s: every frame after its first skips det in both regions
    assert stats["det_skipped"] == {"notice": 35, "feed": 35}
    for name, calls in plain_stats["det_calls"].items():
        assert stats["det_calls"][name] + stats["det_skipped"][name] == calls
    (folder,) = skip_dir.glob("ocr-raw-*")
    assert json.loads((folder / "meta.json").read_text())["key"]["det_skip"] == [4, 8, 8.0]
    # the plain read keeps its own raw cache
    (plain_raw,) = plain_dir.glob("ocr-raw-*")
    assert "det_skip" not in json.loads((plain_raw / "meta.json").read_text())["key"]


def test_resumed_skip_read_equals_the_whole_one(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ocr_raw, "CHUNK_FRAMES", 30)
    whole, part = _dirs(tmp_path, "whole", "part")
    ref, ref_stats = _run(whole, monkeypatch, _Engine(), det_skip=True)
    with pytest.raises(_Stop):  # the first chunk ends at frame 30, inside the still menu
        _run(part, monkeypatch, _Engine(), det_skip=True, stop_after=45)
    events, stats = _run(part, monkeypatch, _Engine(), det_skip=True)
    assert events == ref
    assert stats["det_skipped"] == ref_stats["det_skipped"]  # the anchors were restored
    assert stats["det_calls"] == ref_stats["det_calls"]


# ----------------------------------------------------------------- config and flags


@pytest.mark.parametrize(
    ("data", "want"),
    [({}, False), ({"ocr_det_skip": None}, False), ({"ocr_det_skip": True}, True)],
)
def test_config_default_is_off_unless_true(home, data, want) -> None:
    from cutpoint import config  # noqa: PLC0415

    set_config(home, **data)
    assert config.ocr_det_skip_default() is want


def test_config_value_must_be_a_boolean(home) -> None:
    from cutpoint import config  # noqa: PLC0415

    set_config(home, ocr_det_skip="yes")
    with pytest.raises(ValueError, match=r"ocr_det_skip must be true or false"):
        config.get_config()


FLAG_CASES = [
    (None, [], False),
    (False, ["--ocr-det-skip"], True),
    (True, [], True),
    (True, ["--no-ocr-det-skip"], False),
]


@pytest.mark.parametrize(("cfg", "argv", "want"), FLAG_CASES)
def test_analyze_flag_wins_else_config(home, tmp_path, monkeypatch, cfg, argv, want) -> None:
    import cutpoint.analyze  # noqa: PLC0415

    if cfg is not None:
        set_config(home, ocr_det_skip=cfg)
    video = write(tmp_path / "a.mp4")
    seen: list = []
    monkeypatch.setattr(
        cutpoint.analyze, "analyze", lambda *a, **kw: seen.append(kw["ocr_det_skip"])
    )
    assert cli.main(["analyze", str(video), "--out", "p", *argv]) == 0
    assert seen == [want]


@pytest.mark.parametrize(("cfg", "argv", "want"), FLAG_CASES)
def test_edit_flag_wins_else_config(home, tmp_path, monkeypatch, cfg, argv, want) -> None:
    if cfg is not None:
        set_config(home, ocr_det_skip=cfg)
    video = write(tmp_path / "a.mp4")
    seen: list = []
    monkeypatch.setattr(cli, "run_edit", lambda *a, **kw: seen.append(kw["ocr_det_skip"]))
    assert cli.main(["edit", str(video), "--out", "p", *argv]) == 0
    assert seen == [want]


class _Ran(Exception):
    """run_events was reached."""


@pytest.mark.parametrize(("cfg", "argv", "want"), FLAG_CASES)
def test_events_flag_wins_else_config(home, tmp_path, monkeypatch, cfg, argv, want) -> None:
    import argparse  # noqa: PLC0415

    if cfg is not None:
        set_config(home, ocr_det_skip=cfg)
    seen: list = []

    def run_events(project, **kw):
        seen.append(kw["ocr"].det_skip)
        raise _Ran

    monkeypatch.setattr(events_cli.pipeline, "run_events", run_events)
    parser = argparse.ArgumentParser()
    events_cli.add_arguments(parser)
    with pytest.raises(_Ran):
        events_cli.run_args(parser.parse_args([str(tmp_path), *argv]))
    assert seen == [want]


@pytest.mark.parametrize("on", [True, False])
def test_edit_events_reads_ocr_as_the_analysis_did(tmp_path: Path, on: bool) -> None:
    assert edit._events_namespace(tmp_path, None, on).ocr_det_skip is on
