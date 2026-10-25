"""Feed region growth in HUD discovery (synthetic OCR frames, invented names)."""

from __future__ import annotations

from typing import Any

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


def test_split_rows_count_frames_not_boxes() -> None:
    # the rare rows are split into 3 boxes each (9 boxes) but show in only 3 frames
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
    feed = _feed(frames)
    assert feed["evidence"]["grown_rows"] == 0
