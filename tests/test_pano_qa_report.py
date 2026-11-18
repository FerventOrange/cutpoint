"""The 360 QA's contact sheets (``write_sheets``) and its log table (``log_report``)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from cutpoint.events import sheets
from cutpoint.pano import qa


def frame(t: float, flags: tuple[str, ...] = (), **extra: Any) -> dict[str, Any]:
    return {"t": t, "target": "person", "flags": list(flags), "subj": None, "self": [], **extra}


def clip(file: str, frames: list[dict[str, Any]], flags: list[dict], offset: float = 0.0) -> dict:
    return {"file": file, "offset": offset, "frames": frames, "flags": flags}


@pytest.fixture
def cut(monkeypatch):
    """Capture what ``write_sheets`` hands to the contact-sheet writer (no video, no JPEG)."""
    calls: list[dict[str, Any]] = []

    def contact_sheet(items, out, cols=4, width=480):
        calls.append({"items": items, "out": out, "cols": cols, "width": width})
        return out

    monkeypatch.setattr(sheets, "contact_sheet", contact_sheet)
    return calls


def test_a_flagged_span_gets_one_tile_from_its_middle_frame_and_spans_do_not_repeat(
    tmp_path: Path, cut
) -> None:
    frames = [frame(k * 0.5, ("occluded",) if 1.0 <= k * 0.5 < 2.0 else ()) for k in range(10)]
    flags = [
        {"t0": 1.0, "t1": 2.0},  # frames 1.0 and 1.5: the later of the two middles
        {"t0": 1.0, "t1": 2.0},  # the same span again: no second tile
        {"t0": 3.0, "t1": 3.0},  # no frame inside: the frame at t0
        {"t0": 7.0, "t1": 7.5},  # no frame at all: nothing
    ]
    written = qa.write_sheets(tmp_path, tmp_path / "out", [clip("001_a.mp4", frames, flags)])
    flagged = next(c for c in cut if c["out"].name == "flagged_01.jpg")
    assert [it["t"] for it in flagged["items"]] == [1.5, 3.0]
    assert (flagged["cols"], flagged["width"]) == (4, 480)
    assert all(it["path"] == str(tmp_path / "001_a.mp4") for it in flagged["items"])
    assert flagged["items"][0]["caption"] == "001_a 1.5s OCCL person"
    assert [p.name for p in written] == ["flagged_01.jpg", "overview_01.jpg"]


def test_the_overview_takes_a_frame_every_two_seconds_and_gives_the_director_view(
    tmp_path: Path, cut
) -> None:
    frames = [frame(k * 0.5, ("too_close",) if k == 4 else ()) for k in range(10)]
    path = {
        "keys": [
            {"t": 0.0, "yaw": 190.0, "pitch": 10.0, "fov": 90.0},
            {"t": 100.0, "yaw": 190.0, "pitch": 10.0, "fov": 90.0},
        ]
    }
    qa.write_sheets(
        tmp_path, tmp_path / "out", [clip("001_a.mp4", frames, [], offset=5.0)], {"001_a.mp4": path}
    )
    (overview,) = [c for c in cut if c["out"].name == "overview_01.jpg"]
    assert (overview["cols"], overview["width"]) == (8, 240)
    assert [it["t"] for it in overview["items"]] == [0.0, 2.0, 4.0]
    # the yaw is wrapped to -180..180
    assert overview["items"][0]["caption"] == "001_a 0.0s ok person y-170 p10"
    assert overview["items"][1]["caption"] == "001_a 2.0s CLOSE person y-170 p10"


def test_boxes_are_the_dropped_self_boxes_then_the_kept_subject(tmp_path: Path, cut) -> None:
    fr = frame(0.0, ("occluded",), subj=(0.1, 0.1, 0.4, 0.5), self=[(0.5, 0.5, 0.9, 0.9)])
    qa.write_sheets(tmp_path, tmp_path / "out", [clip("a.mp4", [fr], [{"t0": 0.0, "t1": 1.0}])])
    flagged = next(c for c in cut if c["out"].name == "flagged_01.jpg")
    assert flagged["items"][0]["boxes"] == [(0.5, 0.5, 0.9, 0.9, False), (0.1, 0.1, 0.4, 0.5, True)]


def test_sheets_are_split_by_tile_count_and_numbered(tmp_path: Path, cut) -> None:
    # 70 flagged spans of one frame each: 24 per flagged sheet; 70 overview tiles: 64 per sheet
    frames = [frame(float(k)) for k in range(70)]
    flags = [{"t0": float(k), "t1": k + 0.5} for k in range(70)]
    written = qa.write_sheets(tmp_path, tmp_path / "out", [clip("a.mp4", frames, flags)])
    sizes = {c["out"].name: len(c["items"]) for c in cut}
    assert sizes == {
        "flagged_01.jpg": 24,
        "flagged_02.jpg": 24,
        "flagged_03.jpg": 22,
        "overview_01.jpg": 35,  # frames are 1 s apart: every other one
    }
    assert [p.name for p in written] == list(sizes)


def test_nothing_to_show_writes_no_sheet(tmp_path: Path, cut) -> None:
    assert qa.write_sheets(tmp_path, tmp_path / "out", []) == []
    assert cut == []


def test_the_real_sheet_is_a_jpeg_of_the_tiles(tmp_path: Path, monkeypatch) -> None:
    """Through the real contact-sheet writer, with the video grab replaced by a flat frame."""
    monkeypatch.setattr(
        sheets, "grab", lambda path, t, width=480: np.full((90, width, 3), 80, np.uint8)
    )
    frames = [frame(float(k)) for k in range(3)]
    flags = [{"t0": float(k), "t1": k + 0.5} for k in range(3)]
    out = tmp_path / "out"
    written = qa.write_sheets(tmp_path, out, [clip("a.mp4", frames, flags)])
    flagged = cv2.imread(str(out / "flagged_01.jpg"))
    assert flagged.shape == (90, 4 * 480, 3)  # three tiles and a black spacer in one row
    overview = cv2.imread(str(out / "overview_01.jpg"))
    assert overview.shape == (90, 8 * 240, 3)  # two tiles (every 2 s) padded to 8 columns
    assert [p.name for p in written] == ["flagged_01.jpg", "overview_01.jpg"]


def report(**over: Any) -> dict[str, Any]:
    share = {
        "occluded": 0.1,
        "too_close": 0.0,
        "edge_cut": 0.25,
        "low_info": 0.0,
        "subject_present": 0.9,
    }
    ok = {
        "file": "001_jump.mp4",
        "duration": 12.3,
        "score": 91.5,
        "pass": True,
        "share": share,
        "pan": {"speed_p95": 31.0, "min_shot_s": 2.5},
        "horizon": {"p95": 4.2},
        "vl": {"bad": 1, "frames": 6},
        "fails": [],
    }
    bad = {
        **ok,
        "file": "002_a_very_long_clip_name_that_gets_cut.mp4",
        "pass": False,
        "pan": None,
        "horizon": None,
        "vl": None,
        "fails": ["too much occlusion", "pan too fast"],
    }
    rep = {
        "clips": [ok, bad],
        "edit": {"clips": 2, "duration": 24.6, "score": 80.0, "passed": 1, "share": share},
        "thresholds": {"max_vl_bad": 3},
        "out_dir": str(Path("proj") / "clips_qa"),
        "sheets": [Path("proj") / "clips_qa" / "flagged_01.jpg"],
    }
    rep.update(over)
    return rep


def test_the_log_has_a_row_per_clip_then_the_edit_summary_and_the_written_files(caplog) -> None:
    with caplog.at_level(logging.INFO, logger=qa.logger.name):
        qa.log_report(report())
    lines = [r.getMessage() for r in caplog.records]
    assert lines[0].split()[:3] == ["clip", "dur", "score"]
    good, bad = lines[1], lines[2]
    assert good.split()[:4] == ["001_jump.mp4", "12.3s", "91.5", "ok"]
    assert "10%" in good and "25%" in good and "90%" in good  # shares as whole percents
    assert "1/6" in good and "31" in good and "4.2" in good
    assert bad.startswith("002_a_very_long_clip_n ")  # name cut to 22 characters
    assert "that_gets_cut" not in bad
    assert " FAIL " in bad and bad.endswith("too much occlusion; pan too fast")
    assert any(
        ln.startswith("edit: 2 clips, 24.6s, score 80.0, 1/2 pass; occl 10%") for ln in lines
    )
    assert lines[-2] == f"wrote {Path('proj') / 'clips_qa' / 'qa.json'}"
    assert lines[-1] == f"wrote {Path('proj') / 'clips_qa' / 'flagged_01.jpg'}"


def test_the_vl_line_appears_only_when_clips_were_checked(caplog) -> None:
    with caplog.at_level(logging.INFO, logger=qa.logger.name):
        qa.log_report(report())
    assert not any(r.getMessage().startswith("vl check") for r in caplog.records)
    caplog.clear()
    rep = report()
    rep["edit"].update(vl_checked=2, vl_bad=4)
    with caplog.at_level(logging.INFO, logger=qa.logger.name):
        qa.log_report(rep)
    assert "vl check: 4 bad frame(s) in 2 checked clip(s); a clip fails above 3" in caplog.messages
