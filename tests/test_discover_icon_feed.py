"""Icon kill feeds in HUD discovery (#433): a side region of "name [weapon icon] name" lines
is a feed even with one row at a time and few names (synthetic OCR frames, invented names)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cutpoint.events import discover
from cutpoint.events.discover import DiscoverParams, cluster_regions

NAMES = ["Alder", "Boxer7", "Kip_V2", "Nova", "Quill", "Orrin"]
VERBS, TARGETS = ["Capture", "Defend", "Hold"], ["Alpha", "Bravo", "Charlie", "Delta"]


def _box(text: str, x: int, y: int, w: int, h: int = 28) -> dict[str, Any]:
    return {"x0": x, "y0": y, "x1": x + w, "y1": y + h, "text": text, "score": 0.9}


def _frames(gap: int, line: str = "names") -> list[dict[str, Any]]:
    """40 bursts x 4 frames. In every other burst one kill line shows at the top right for
    3 frames (killer box, ``gap`` px, victim box; few names, so they repeat), and a
    perf overlay "FPS:NN [gap] GPU:N.N" sits at the bottom left in every frame."""
    out = []
    for b in range(40):
        for k in range(4):
            n = b * 4 + k
            boxes = [
                _box(f"FPS:{60 + n % 30}", 10, 1000, 90),
                _box(f"GPU:{n % 9}.{n % 7}", 140, 1000, 100),
            ]
            if b % 2 == 0 and k < 3:
                a, v = NAMES[b // 2 % 6], NAMES[(b // 2 + 1 + b // 12) % 6]
                if line == "labels":  # a two-label HUD notice: verbs left, targets right
                    a, v = VERBS[b // 2 % 3], TARGETS[b // 2 % 4]
                if line in ("names", "labels"):
                    boxes += [_box(a, 1550, 285, 110), _box(v, 1660 + gap, 285, 110)]
                else:  # one plain text: a notice, not an icon line
                    boxes.append(_box(f"{a} has left the game", 1550, 285, 280))
            out.append(
                {"id": f"rec.mp4@{b * 60 + k * 1.5:.1f}", "source": "rec.mp4", "burst": b}
                | {"k": k, "frame": f"b{b:03d}_{k}.jpg", "boxes": boxes}
            )
    return out


def _at_feed(p: dict[str, Any]) -> bool:
    return p["x"] < 1800 and p["x"] + p["w"] > 1600 and p["y"] < 313 and p["y"] + p["h"] > 285


def test_icon_lines_make_a_feed() -> None:
    proposals, _ = cluster_regions(_frames(gap=70), DiscoverParams())
    feed = [p for p in proposals if _at_feed(p)]
    assert [p["kind"] for p in feed] == ["feed"]
    ev = feed[0]["evidence"]
    assert ev["mean_rows"] < 1.3 and ev["label_variety"] < 0.15  # a readout before #433
    assert ev["icon_lines"] >= 0.9


def test_perf_overlay_with_a_gap_stays_a_readout() -> None:
    proposals, _ = cluster_regions(_frames(gap=70), DiscoverParams())
    perf = [p for p in proposals if p["y"] > 900 and p["x"] < 300]
    assert perf and all(p["kind"] == "readout" for p in perf)  # "FPS:61" is no name


def test_names_without_an_icon_gap_are_no_icon_lines() -> None:
    proposals, _ = cluster_regions(_frames(gap=8), DiscoverParams())
    feed = [p for p in proposals if _at_feed(p)]
    assert feed and all(p["kind"] != "feed" for p in feed)
    assert all(p["evidence"]["icon_lines"] == 0 for p in feed)


def test_plain_text_lines_are_no_icon_lines() -> None:
    proposals, _ = cluster_regions(_frames(gap=70, line="text"), DiscoverParams())
    assert all(p["kind"] != "feed" for p in proposals if _at_feed(p))


def test_two_label_notice_is_no_icon_feed() -> None:
    """ "Capture [icon] Alpha": icon lines, but no name is ever on both sides (players kill
    and die), so no feed."""
    proposals, _ = cluster_regions(_frames(gap=70, line="labels"), DiscoverParams())
    feed = [p for p in proposals if _at_feed(p)]
    assert feed and all(p["kind"] != "feed" for p in feed)
    assert feed[0]["evidence"]["icon_lines"] >= 0.9
    assert feed[0]["evidence"]["icon_names"] == [7, 0]


@pytest.mark.parametrize(("gap", "flag"), [(70, True), (8, False)])
def test_icon_feed_preset_reads_other_rows_as_notices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gap: int, flag: bool
) -> None:
    monkeypatch.setattr(discover, "_ocr_frames", lambda *_a, **_k: _frames(gap=gap))
    monkeypatch.setattr(discover, "probe", lambda _p: None)
    preset = discover.discover_regions([], tmp_path / "disc", "zonewar")
    assert ("feed" in preset) is flag
    if flag:
        assert preset["feed"] == {"row_notices": True}
