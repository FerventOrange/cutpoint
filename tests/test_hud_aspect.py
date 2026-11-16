"""HUD discovery on footage that is not 16:9 (#1085, A138): box y scales with the frame
height, so ``regions.scaled`` reads a stored box at the pixel row it was found at."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cutpoint import paths
from cutpoint.events import discover, feed_labels, hud_map, killfeed, regions
from cutpoint.events.regions import Region

SIZES = [(3440, 1440), (1440, 1080), (1920, 1080)]  # 21:9, 4:3, 16:9


def _engine(box: list[tuple[int, int]]):
    class Engine:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, img, min_score):
            return [(box, "KIP ELIMINATED ALDER", 0.9)]

    return Engine


def _src(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(name="a.mp4", path=tmp_path / "a.mp4", io_path=tmp_path / "a.mp4")


@pytest.mark.parametrize(("w", "h"), SIZES)
def test_discovered_box_reads_back_at_its_pixel_row(tmp_path, monkeypatch, w, h):
    # a box on the bottom edge of the frame
    x0, y0, x1, y1 = int(w * 0.70), int(h * 0.92), int(w * 0.95), int(h * 0.97)
    box = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    monkeypatch.setattr(discover, "_burst_frames", lambda *_a: [np.zeros((h, w, 3), np.uint8)] * 4)
    monkeypatch.setattr(killfeed, "OcrEngine", _engine(box))
    frames = discover._ocr_frames(
        [(_src(tmp_path), 0.0)], discover.DiscoverParams(), tmp_path / "f", tmp_path / "o.jsonl"
    )
    b = frames[0]["boxes"][0]
    r = Region("r", b["x0"], b["y0"], b["x1"] - b["x0"], b["y1"] - b["y0"], "feed")
    sx, sy, sw, sh = regions.scaled(r, w, h)
    assert abs(sx - x0) <= 4 and abs(sy - y0) <= 4
    assert abs(sw - (x1 - x0)) <= 6 and abs(sh - (y1 - y0)) <= 6
    assert b["y1"] <= discover.REF_H  # not clamped away on 4:3


@pytest.mark.parametrize(("w", "h"), SIZES)
def test_a_vision_box_lands_on_the_frame_row(w, h):
    # a box on the bottom 8% of the frame, on the 0-1000 scale of the image
    x, y, bw, bh = hud_map.to_ref([700, 900, 950, 980], (w, h))
    r = Region("r", x, y, bw, bh, "feed")
    sx, sy, _sw, sh = regions.scaled(r, w, h)
    assert abs(sy - 0.9 * h) <= 4 and abs(sh - 0.08 * h) <= 6
    assert abs(sx - 0.7 * w) <= 4


@pytest.mark.parametrize(("w", "h"), SIZES)
def test_the_label_crop_cuts_the_proposal_rows(w, h):
    img = np.zeros((h, w, 3), np.uint8)
    top = int(h * 0.9)
    img[top : int(h * 0.97), int(w * 0.7) : int(w * 0.95)] = 255
    # the proposal in reference pixels: the same band
    p = {"x": 1344, "y": 972, "w": 480, "h": 76}
    jpg = feed_labels.crop_jpeg(img, p)
    assert jpg is not None
    crop = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert crop.mean() > 100  # mostly the bright band, not the rows above it


def _preset(path: Path, res, mapping=None) -> None:
    disc: dict = {"source_resolution": res}
    if mapping:
        disc["mapping"] = mapping
    path.write_text(
        json.dumps(
            {
                "game": "gx",
                "source": "discovered",
                "regions": [{"name": "feed", "x": 1, "y": 2, "w": 300, "h": 100, "kind": "feed"}],
                "discovery": disc,
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("res", "mapping", "kept"),
    [
        ([3440, 1440], None, False),  # made with the width factor: discovered again
        ([1440, 1080], None, False),
        ([3440, 1440], 2, True),
        ([1920, 1080], None, True),  # 16:9 was never wrong
        ([1366, 768], None, True),
        (None, None, True),
    ],
)
def test_old_discovered_presets_of_other_aspects_are_not_loaded(
    tmp_path, monkeypatch, res, mapping, kept
):
    monkeypatch.setattr(regions.paths, "presets_dir", lambda: tmp_path)
    _preset(tmp_path / "gx.json", res, mapping)
    assert (regions.load_preset("gx") is not None) is kept


@pytest.mark.parametrize(("w", "h", "again"), [(3440, 1440, True), (1920, 1080, False)])
def test_cached_frames_without_a_version_are_read_again_only_when_not_16x9(
    tmp_path, monkeypatch, w, h, again
):
    reads = []

    class Engine:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, img, min_score):
            reads.append(1)
            return []

    monkeypatch.setattr(discover, "_burst_frames", lambda *_a: [np.zeros((h, w, 3), np.uint8)] * 4)
    monkeypatch.setattr(killfeed, "OcrEngine", Engine)
    bursts = [(_src(tmp_path), 0.0)]
    args = (discover.DiscoverParams(), tmp_path / "f", tmp_path / "o.jsonl")
    first = discover._ocr_frames(bursts, *args)
    assert all(f["ref_v"] == discover.REF_VERSION for f in first)
    old = [{k: v for k, v in f.items() if k != "ref_v"} for f in first]  # an older cache
    (tmp_path / "o.jsonl").write_text("".join(json.dumps(r) + "\n" for r in old), encoding="utf-8")
    reads.clear()
    redone = discover._ocr_frames(bursts, *args)
    assert bool(reads) is again
    # a frame read again carries the version; a kept 16:9 record is as it was
    assert all(("ref_v" in f) is again for f in redone)
    saved = discover.read_ocr_samples(args[2])
    assert all(("ref_v" in r) is again for r in saved)
    assert all(r["ref_v"] == discover.REF_VERSION for r in saved if again)


def test_a_stale_project_preset_does_not_hide_a_fresh_user_preset(tmp_path, monkeypatch):
    user = tmp_path / "user"
    user.mkdir()
    monkeypatch.setattr(regions.paths, "presets_dir", lambda: user)
    project = tmp_path / "proj"
    (project / "hud_presets").mkdir(parents=True)
    _preset(project / "hud_presets" / "gx.json", [3440, 1440])
    _preset(user / "gx.json", [3440, 1440], mapping=2)
    got = regions.load_preset("gx", project)
    assert got is not None and got.path == user / "gx.json"


def _map_frames(tmp_path: Path, size: tuple[int, int]):
    from test_hud_map import FEED_BOX, _feed_rows, _frames, _reply  # noqa: PLC0415

    frames = _frames(4, 2, lambda _b, n: _feed_rows(n))
    folder = tmp_path / "frames"
    folder.mkdir(exist_ok=True)
    for f in frames:
        cv2.imwrite(str(folder / f["frame"]), np.full((size[1], size[0], 3), 40, np.uint8))
    asked: list[int] = []

    def ask(*_a):
        asked.append(1)
        return _reply(FEED_BOX)

    return frames, folder, ask, asked


@pytest.mark.parametrize(("size", "again"), [((3440, 1440), True), ((1920, 1080), False)])
def test_cached_vision_boxes_without_a_version_are_asked_again_only_when_not_16x9(
    tmp_path, size, again
):
    frames, folder, ask, asked = _map_frames(tmp_path, size)
    hud_map.map_hud([], frames, folder, ask, "m")
    path = tmp_path / "hud_map.json"
    store = json.loads(path.read_text(encoding="utf-8"))
    assert all(v["ref_v"] == discover.REF_VERSION for v in store.values())
    for v in store.values():  # an older cache
        v.pop("ref_v")
    path.write_text(json.dumps(store), encoding="utf-8")
    asked.clear()
    hud_map.map_hud([], frames, folder, ask, "m")
    assert bool(asked) is again


def test_a_rediscovered_stale_preset_keeps_what_the_user_set(home, monkeypatch):
    from test_hud_map import _top_up  # noqa: PLC0415

    folder = paths.presets_dir()
    folder.mkdir(parents=True, exist_ok=True)
    old = {
        "game": "zonewar",
        "source": "discovered",
        "player_tags": ["Kip"],
        "banner_phrases": ["VICTORY"],
        "match": ["zone"],
        "title": "Zone War",
        "bigtext": False,
        "feed": {"newest": "top"},
        "regions": [{"name": "feed", "x": 1, "y": 2, "w": 300, "h": 100, "kind": "feed"}],
        "discovery": {"source_resolution": [3440, 1440]},
    }
    (folder / "zonewar.json").write_text(json.dumps(old), encoding="utf-8")
    stats, ran, _ = _top_up(home, monkeypatch, None, None)
    assert stats["zonewar"]["installed"] is True and len(ran) == 1
    backup = folder / "zonewar.json.pre-1085"
    assert json.loads(backup.read_text(encoding="utf-8")) == old
    new = json.loads((folder / "zonewar.json").read_text(encoding="utf-8"))
    assert new["player_tags"] == ["Kip"] and new["match"] == ["zone"]
    assert new["banner_phrases"] == ["VICTORY"] and new["title"] == "Zone War"
    assert new["bigtext"] is False and new["feed"] == {"newest": "top"}


def test_a_rediscovered_stale_preset_drops_the_old_row_notices(home, monkeypatch):
    """``row_notices`` is what discovery found, not the user: a new discovery that finds no
    icon feed clears it (the user's ``newest`` stays)."""
    from test_hud_map import _top_up  # noqa: PLC0415

    folder = paths.presets_dir()
    folder.mkdir(parents=True, exist_ok=True)
    old = {
        "game": "zonewar",
        "source": "discovered",
        "feed": {"newest": "top", "row_notices": True},
        "regions": [{"name": "feed", "x": 1, "y": 2, "w": 300, "h": 100, "kind": "feed"}],
        "discovery": {"source_resolution": [3440, 1440]},
    }
    (folder / "zonewar.json").write_text(json.dumps(old), encoding="utf-8")
    stats, _, _ = _top_up(home, monkeypatch, None, None)
    assert stats["zonewar"]["installed"] is True
    new = json.loads((folder / "zonewar.json").read_text(encoding="utf-8"))
    assert new["feed"] == {"newest": "top"}


def test_discovery_writes_the_mapping_version(tmp_path, monkeypatch):
    from test_hud_map import _discover  # noqa: PLC0415

    assert _discover(tmp_path, monkeypatch)["discovery"]["mapping"] == discover.REF_VERSION
