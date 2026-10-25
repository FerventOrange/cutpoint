"""Feed region growth in HUD discovery (synthetic OCR frames, invented names)."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from cutpoint.events.discover import FEED_GROW_MAX, DiscoverParams, cluster_regions

NAMES = ["Ash", "Boxer", "Kip", "Nova", "Quill", "Orrin", "Sable", "Tarn", "Vex", "Wren"]
NAMES += ["Yarrow", "Zeb", "Pell", "Ulric", "Dace", "Fenn"]
PITCH = 30  # feed row pitch (reference px)


def _box(text: str, x: int, y: int, w: int = 160, h: int = 20) -> dict[str, Any]:
    return {"x0": x, "y0": y, "x1": x + w, "y1": y + h, "text": text, "score": 0.9}


def _row(r: int, b: int, k: int) -> dict[str, Any]:
    """Feed row ``r`` (top = 0) at the top right; its names change every frame."""
    a, v = NAMES[(b + k + r) % 16], NAMES[(b * 5 + k * 3 + 2 * r) % 16]
    return _box(f"{a} downed {v}", 1500, 40 + r * PITCH)


def _frames(rows: int, rare: int, rare_bursts: set[int], top: str | None = None) -> list[dict]:
    """40 bursts x 4 frames of a ``rows``-row feed whose top ``rare`` rows show only in
    ``rare_bursts`` (a full feed); ``top`` is a static label one row above the feed."""
    out = []
    for b in range(40):
        for k in range(4):
            boxes = [_row(r, b, k) for r in range(rows) if r >= rare or b in rare_bursts]
            if top:
                boxes.append(_box(top, 1500, 40 - PITCH))
            out.append(
                {"id": f"rec.mp4@{b * 60 + k * 1.5:.1f}", "t": b * 60 + k * 1.5}
                | {"source": "rec.mp4", "burst": b, "k": k, "frame": f"b{b:03d}_{k}.jpg"}
                | {"boxes": boxes}
            )
    return out


def _feed(frames: list[dict]) -> dict[str, Any]:
    feeds = [p for p in cluster_regions(frames, DiscoverParams())[0] if p["kind"] == "feed"]
    assert len(feeds) == 1
    return feeds[0]


def test_rare_top_rows_join_the_feed() -> None:
    # rows 0-2 show in 3 of 40 bursts: below the dynamic-cell share (8% of the bursts)
    feed = _feed(_frames(6, 3, {0, 13, 26}))
    assert feed["evidence"]["grown_rows"] == 3
    assert feed["y"] <= 40  # the top row's text starts at y=40
    assert feed["y"] + feed["h"] >= 40 + 5 * PITCH + 20


def test_no_growth_without_rare_rows() -> None:
    feed = _feed(_frames(6, 3, set()))
    assert feed["evidence"]["grown_rows"] == 0
    assert feed["y"] >= 40 + 3 * PITCH - 24


def test_static_label_above_is_not_a_row() -> None:
    feed = _feed(_frames(3, 0, set(), top="SQUAD"))
    assert feed["evidence"]["grown_rows"] == 0
    assert feed["y"] > 40 - PITCH  # the label's text starts at y=10


def test_rows_of_one_burst_are_not_enough() -> None:
    feed = _feed(_frames(6, 3, {13}))
    assert feed["evidence"]["grown_rows"] == 0


def test_growth_is_bounded() -> None:
    # two common rows, eight rare ones above: at most double the height
    frames = _frames(10, 8, {0, 13, 26})
    plain = _feed(_frames(10, 8, set()))
    feed = _feed(frames)
    assert 0 < feed["evidence"]["grown_rows"] < 8
    assert feed["h"] <= FEED_GROW_MAX * plain["h"]


def test_split_rows_count_frames_not_boxes(caplog: pytest.LogCaptureFixture) -> None:
    # the rare rows are split into 3 boxes each (9 boxes) but show in only 3 frames, below
    # the 8 a row needs (5% of 160): they join as varied rows, not on their box count
    caplog.set_level(logging.DEBUG, logger="cutpoint.events.discover")
    frames = _frames(6, 3, set())
    for f in frames:
        if f["burst"] in (0, 13, 26) and f["k"] == 0:
            for r in range(3):
                y = 40 + r * PITCH
                f["boxes"] += [
                    _box(t, x, y, w=50)
                    for t, x in zip(
                        _row(r, f["burst"], 0)["text"].split(), (1500, 1560, 1620), strict=True
                    )
                ]
    _feed(frames)
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Feed growth"))
    assert "a row needs 8 frames" in line
    assert (
        "above: row at y 110: 3 frames (3 full), 3 bursts, variety 0.78, sighting variety 1.00, "
        "ok: varied"
    ) in line
    assert "ok: frames" not in line


def test_a_stray_box_at_the_feed_edge_does_not_stop_growth() -> None:
    # one small box (an icon read as text) a few px above the common top row, in one
    # frame: the top row is still the supported one, so the rare rows above are added
    frames = _frames(6, 3, {0, 13, 26})
    frames[50]["boxes"].append(_box("zz", 1550, 40 + 3 * PITCH - 8, w=40, h=12))
    feed = _feed(frames)
    assert feed["evidence"]["grown_rows"] == 3
    assert feed["y"] <= 40


def test_growth_logs_why_it_stopped(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="cutpoint.events.discover")
    _feed(_frames(6, 3, set()))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Feed growth")]
    assert len(lines) == 1
    line = lines[0]
    assert "pitch 30.0 px, 3 rows, full in 160/160 frames" in line
    assert (
        "above: row at y 110: 0 frames (0 full), 0 bursts, variety 0.00, sighting variety 0.00, "
        "stop: frames"
    ) in line
    assert "below: row at y 230: 0 frames" in line and line.endswith("; 0 rows added")
    assert not any(n in line for n in NAMES)  # no OCR text in the log


def _two_frame_row(frames: list[dict], texts: tuple[str, str], bursts: tuple[int, int]) -> None:
    """Add a row one pitch above a 3-row feed (rows at y 40 + 3 pitches and below) in
    frame 0 of each of ``bursts`` with ``texts``."""
    for f in frames:
        if f["k"] == 0 and f["burst"] in bursts:
            f["boxes"].append(_box(texts[bursts.index(f["burst"])], 1500, 40 + 2 * PITCH))


def test_a_varied_row_in_two_frames_of_two_bursts_is_added(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # 2 frames of 160 (a row needs 8): a full feed is rare, distinct texts admit the row
    caplog.set_level(logging.DEBUG, logger="cutpoint.events.discover")
    frames = _frames(6, 3, set())
    _two_frame_row(frames, ("Ash downed Kip", "Nova downed Wren"), (5, 21))
    feed = _feed(frames)
    assert feed["evidence"]["grown_rows"] == 1
    assert feed["y"] <= 40 + 2 * PITCH
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Feed growth"))
    assert (
        "above: row at y 110: 2 frames (2 full), 2 bursts, variety 1.00, sighting variety 1.00, "
        "ok: varied | "
    ) in line


def test_a_static_label_in_two_frames_is_not_added(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="cutpoint.events.discover")
    frames = _frames(6, 3, set())
    _two_frame_row(frames, ("RELOAD", "RELOAD"), (5, 21))
    assert _feed(frames)["evidence"]["grown_rows"] == 0
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Feed growth"))
    assert (
        "above: row at y 110: 2 frames (2 full), 2 bursts, variety 0.50, sighting variety 0.50, "
        "stop: frames"
    ) in line


def test_a_varied_row_in_two_frames_of_one_burst_is_not_added(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="cutpoint.events.discover")
    frames = _frames(6, 3, set())
    for f in frames:
        if f["burst"] == 5 and f["k"] in (0, 1):
            f["boxes"].append(_box(f"{NAMES[f['k']]} downed Kip", 1500, 40 + 2 * PITCH))
    assert _feed(frames)["evidence"]["grown_rows"] == 0
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Feed growth"))
    assert (
        "above: row at y 110: 2 frames (2 full), 1 bursts, variety 1.00, sighting variety 1.00, "
        "stop: bursts"
    ) in line


def _feed_line(caplog: pytest.LogCaptureFixture) -> str:
    return next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Feed growth"))


def test_a_row_that_stays_for_two_frames_of_two_bursts_is_added(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # a real row stays while the feed is full: the same text in frames 0 and 1 of a burst
    # (label variety 0.50 per box; one distinct text per burst)
    caplog.set_level(logging.DEBUG, logger="cutpoint.events.discover")
    frames = _frames(6, 3, set())
    texts = {5: "Ash downed Kip", 21: "Nova downed Wren"}
    for f in frames:
        if f["burst"] in texts and f["k"] in (0, 1):
            f["boxes"].append(_box(texts[f["burst"]], 1500, 40 + 2 * PITCH))
    assert _feed(frames)["evidence"]["grown_rows"] == 1
    line = _feed_line(caplog)
    assert "4 frames (4 full), 2 bursts, variety 0.50, sighting variety 1.00, ok: varied" in line


def test_split_rows_in_five_bursts_are_added() -> None:
    # 3 rare rows split into 3 boxes each, in frame 0 of 5 of 40 bursts: 5 frames, below the
    # 8 a row needs, and the repeated kill word lowers the per-box variety
    frames = _frames(6, 3, set())
    bursts = (0, 8, 16, 24, 32)
    for f in frames:
        if f["burst"] in bursts and f["k"] == 0:
            i = bursts.index(f["burst"])
            for r in range(3):
                words = [NAMES[i * 3 + r], "downed", NAMES[(i * 3 + r + 7) % 16]]
                f["boxes"] += [
                    _box(t, x, 40 + r * PITCH, w=50)
                    for t, x in zip(words, (1500, 1560, 1620), strict=True)
                ]
    feed = _feed(frames)
    assert feed["evidence"]["grown_rows"] == 3
    assert feed["y"] <= 40


def test_a_static_label_in_changing_case_is_not_added() -> None:
    frames = _frames(6, 3, set())
    _two_frame_row(frames, ("RELOAD", "Reload"), (5, 21))  # OCR case jitter
    assert _feed(frames)["evidence"]["grown_rows"] == 0


def test_notices_next_to_a_feed_that_is_not_full_are_not_added(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # two different notices one pitch above the feed, in frames whose feed lacks its top row:
    # a feed's oldest row shows only when the feed is full
    caplog.set_level(logging.DEBUG, logger="cutpoint.events.discover")
    frames = _frames(6, 3, set())
    for f in frames:
        if f["k"] == 0 and f["burst"] in (5, 21):
            f["boxes"] = [b for b in f["boxes"] if b["y0"] != 40 + 3 * PITCH]
    _two_frame_row(frames, ("Gate captured", "Supply drop incoming"), (5, 21))
    assert _feed(frames)["evidence"]["grown_rows"] == 0
    assert "row at y 110: 2 frames (0 full), 2 bursts" in _feed_line(caplog)


def test_an_oldest_row_is_added_when_ocr_missed_one_inner_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # the feed's middle row (row 4) is missing from both frames that show the oldest row:
    # they still count as full feed frames (the row next to it is there, one other missed)
    caplog.set_level(logging.DEBUG, logger="cutpoint.events.discover")
    frames = _frames(6, 3, set())
    for f in frames:
        if f["k"] == 0 and f["burst"] in (5, 21):
            f["boxes"] = [b for b in f["boxes"] if b["y0"] != 40 + 4 * PITCH]
    _two_frame_row(frames, ("Ash downed Kip", "Nova downed Wren"), (5, 21))
    assert _feed(frames)["evidence"]["grown_rows"] == 1
    assert "row at y 110: 2 frames (2 full), 2 bursts" in _feed_line(caplog)


def test_box_order_does_not_make_a_label_varied() -> None:
    frames = _frames(6, 3, set())
    for f in frames:
        if f["k"] == 0 and f["burst"] in (5, 21):
            words = ("PRESS", "RELOAD") if f["burst"] == 5 else ("RELOAD", "PRESS")
            f["boxes"] += [
                _box(t, x, 40 + 2 * PITCH, w=70) for t, x in zip(words, (1500, 1580), strict=True)
            ]
    assert _feed(frames)["evidence"]["grown_rows"] == 0
