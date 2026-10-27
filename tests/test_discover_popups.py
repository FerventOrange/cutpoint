"""Score popups next to a kill feed in HUD discovery (synthetic OCR frames, invented HUD
texts)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cutpoint.events import discover
from cutpoint.events.discover import DiscoverParams, cluster_regions

NAMES = ["Alder", "Boxer", "Kip", "Nova", "Quill", "Orrin", "Sable", "Tarn", "Vex", "Wren"]
ACTIONS = ["Crate", "Relay", "Beacon", "Gate", "Drone", "Flag"]
DEEDS = ["Opened", "Spotted", "Sealed", "Tagged", "Cleared", "Held"]


def _box(text: str, x: int, y: int, w: int = 160, h: int = 24) -> dict[str, Any]:
    return {"x0": x, "y0": y, "x1": x + w, "y1": y + h, "text": text, "score": 0.9}


def _frame(b: int, k: int) -> dict[str, Any]:
    """A three-row feed at the top right over two count rows ("2/3": numbers, but no
    sign) and, in every other burst, a score popup at the right edge: "+N" over a label."""
    n = b * 4 + k
    feed = [
        _box(f"{NAMES[m % 10]} downed {NAMES[m // 10 % 10]}{NAMES[m % 7]}", 1500, 40 + r * 30)
        for r in range(3)
        for m in [n + r]
    ]
    feed += [_box(f"{(k + r) % 3 + 1}/3", 1500, 130 + r * 30, w=40) for r in range(2)]
    popup = []
    if b % 2 == 0 and k < 3:
        label = f"{ACTIONS[n % 6]} {DEEDS[n // 6 % 6]}"
        popup = [_box(f"+{10 * (k + 1)}", 1790, 570, w=40), _box(label, 1700, 596, w=130)]
    boxes = [_box("HEALTH", 20, 1040, w=100), *feed, *popup]
    return {"id": f"rec.mp4@{b * 60 + k * 1.5:.1f}", "source": "rec.mp4", "burst": b, "k": k} | {
        "frame": f"b{b:03d}_{k}.jpg",
        "boxes": boxes,
    }


FRAMES = [_frame(b, k) for b in range(40) for k in range(4)]


def _at_popup(r: dict[str, Any]) -> bool:
    return r["x"] < 1830 and r["x"] + r["w"] > 1700 and r["y"] < 620 and r["y"] + r["h"] > 570


def test_plus_value_rows_make_a_popup_not_a_feed() -> None:
    """#410: a column of score popups ("+N" over a label) beside the feed is a popup, not the
    feed."""
    proposals, _ = cluster_regions(FRAMES, DiscoverParams())
    kinds = {p["kind"] for p in proposals if _at_popup(p)}
    assert kinds == {"popup"}
    feed = [p for p in proposals if p["kind"] == "feed"]
    assert len(feed) == 1 and feed[0]["y"] < 150  # count rows ("2/3") are no scores


def test_popup_labels_are_no_regions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discover, "_ocr_frames", lambda *_a, **_k: FRAMES)
    monkeypatch.setattr(discover, "probe", lambda _p: None)
    preset = discover.discover_regions([], tmp_path / "disc", "zonewar")
    assert [r["kind"] for r in preset["regions"] if r["kind"] == "feed"] == ["feed"]
    assert not [r for r in preset["regions"] if _at_popup(r)]  # neither a feed nor a notice
