"""Menu / scoreboard frames in HUD discovery (synthetic OCR frames, invented HUD texts)."""

from __future__ import annotations

from typing import Any

from cutpoint.events import discover
from cutpoint.events.discover import DiscoverParams, cluster_regions, gameplay_frames

NAMES = ["Ash", "Boxer", "Kip", "Nova", "Quill", "Orrin", "Sable", "Tarn", "Vex", "Wren"]


def _box(text: str, x: int, y: int, w: int = 160, h: int = 20) -> dict[str, Any]:
    return {"x0": x, "y0": y, "x1": x + w, "y1": y + h, "text": text, "score": 0.9}


def _busy_hud() -> list[dict[str, Any]]:
    """60 static boxes along the left and bottom edges (labels, ammo, minimap letters)."""
    left = [_box(f"LABEL {chr(65 + i)}", 20, 150 + i * 30) for i in range(28)]
    bottom = [_box(f"SLOT {chr(65 + i)}", 220 + i * 50, 1040, w=40) for i in range(32)]
    return left + bottom


def _feed(b: int, k: int) -> list[dict[str, Any]]:
    """A three-row kill feed at the top right whose names change every frame."""
    return [
        _box(f"{NAMES[(b + k + r) % 10]} downed {NAMES[(b * 3 + k + r) % 10]}", 1500, 40 + r * 30)
        for r in range(3)
    ]


def _scoreboard() -> list[dict[str, Any]]:
    """A centred table: 11 rows x 4 columns."""
    return [
        _box(f"{NAMES[r % 10]} {c * 7 + r}", 400 + c * 300, 250 + r * 60, w=120)
        for r in range(11)
        for c in range(4)
    ]


def _frame(source: str, b: int, k: int, boxes: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": f"{source}@{b * 60 + k * 1.5:.1f}", "source": source, "burst": b, "k": k} | {
        "frame": f"b{b:03d}_{k}.jpg",
        "boxes": boxes,
    }


def test_busy_hud_stays_and_scoreboard_goes() -> None:
    frames = [
        _frame("rec.mp4", b, k, _busy_hud() + _feed(b, k)) for b in range(10) for k in range(4)
    ]
    frames += [_frame("rec.mp4", 10, k, _busy_hud() + _scoreboard()) for k in range(2)]
    assert all(len(f["boxes"]) > 28 for f in frames)  # the old fixed limit dropped them all
    kept = gameplay_frames(frames, DiscoverParams())
    assert kept == frames[:40]
    proposals, maps = cluster_regions(frames, DiscoverParams())
    assert maps["stats"]["menu_frames"] == 2
    assert maps["stats"]["gameplay_frames"] == 40
    feed = [p for p in proposals if p["kind"] == "feed"]
    assert feed and feed[0]["x"] >= 1400 and feed[0]["y"] < 150


def test_box_count_jump_is_per_source() -> None:
    quiet = [_box("HP", 20, 1040, w=40), _box("AMMO", 1800, 1040, w=60)]
    menu_list = [_box(f"OPTION {chr(65 + i)}", 60, 100 + i * 30) for i in range(30)]
    frames = [_frame("quiet.mp4", b, 0, quiet) for b in range(6)]
    frames.append(_frame("quiet.mp4", 6, 0, menu_list))  # an edge-aligned menu list
    frames += [_frame("busy.mp4", b, 0, _busy_hud()) for b in range(6)]
    kept = gameplay_frames(frames, DiscoverParams())
    # 30 boxes are a jump over the quiet source's median of 2, not over the busy one's 60
    assert [f["source"] for f in kept] == ["quiet.mp4"] * 6 + ["busy.mp4"] * 6


def test_few_boxes_never_menu() -> None:
    # centred text in most centre cells, but fewer boxes than menu_min_boxes
    spread = [_box("MARK", 300 + c * 240, 200 + r * 180, w=40) for r in range(4) for c in range(5)]
    frames = [_frame("rec.mp4", 0, 0, spread)]
    assert gameplay_frames(frames, DiscoverParams()) == frames
    assert gameplay_frames(frames, DiscoverParams(menu_min_boxes=10)) == []


def test_proposal_records_run_frames(tmp_path, monkeypatch) -> None:
    frames = [
        _frame("rec.mp4", b, k, _busy_hud() + _feed(b, k)) for b in range(10) for k in range(4)
    ]
    monkeypatch.setattr(discover, "_ocr_frames", lambda *_a, **_k: frames)
    preset = discover.discover_regions([], tmp_path / "disc", "zonewar")
    assert preset["discovery"]["frames"] == [f["id"] for f in frames]
