"""Extra notice regions where HUD text shows outside every discovered or preset region
(synthetic OCR frames, invented HUD texts and game ``verbfeed``)."""

from __future__ import annotations

import json
import random
import string
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cutpoint import paths
from cutpoint.events import discover, ocr_raw, pipeline
from cutpoint.events.discover import (
    DiscoverParams,
    cluster_regions,
    gameplay_frames,
    notice_proposals,
    notice_region,
    proposals_to_regions,
)
from cutpoint.events.killfeed import OcrParams
from cutpoint.events.regions import Region, bigtext_region, preset_from_dict
from hud_fixtures import verbfeed, write_preset

NAMES = ["Ash", "Boxer", "Kip", "Nova", "Quill", "Orrin", "Sable", "Tarn", "Vex", "Wren"]
BURSTS = 40


def _box(text: str, x: int, y: int, w: int = 160, h: int = 20) -> dict[str, Any]:
    return {"x0": x, "y0": y, "x1": x + w, "y1": y + h, "text": text, "score": 0.9}


def _centred(text: str, cx: int, y: int, h: int = 20) -> dict[str, Any]:
    w = 14 * len(text)
    return _box(text, cx - w // 2, y, w=w, h=h)


def _frame(source: str, b: int, k: int, boxes: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": f"{source}@{b * 60 + k * 1.5:.1f}", "source": source, "burst": b, "k": k} | {
        "frame": f"b{b:03d}_{k}.jpg",
        "boxes": boxes,
    }


def _hud(b: int, k: int) -> list[dict[str, Any]]:
    """Static labels at the bottom and a three-row feed at the top right whose names
    change every frame."""
    feed = [
        _box(f"{NAMES[n % 10]} downed {NAMES[n // 10 % 10]}", 1500, 40 + r * 30)
        for r in range(3)
        for n in [b * 4 + k + r]
    ]
    return [_box("HEALTH", 20, 1040, w=100), _box("RESERVE", 1780, 1040, w=120), *feed]


def _frames(extra: dict[tuple[int, int], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """40 bursts x 4 frames of :func:`_hud`; ``extra[(burst, k)]`` boxes are added."""
    return [
        _frame("rec.mp4", b, k, _hud(b, k) + extra.get((b, k), []))
        for b in range(BURSTS)
        for k in range(4)
    ]


def _extra(
    frames: list[dict[str, Any]], covered: tuple[Region, ...] | None = None
) -> list[dict[str, Any]]:
    """The outside-notice proposals of a discovery run on ``frames`` (covered: the
    discovered regions and the big-text region unless given)."""
    params = DiscoverParams()
    proposals, maps = cluster_regions(frames, params)
    cover = covered if covered is not None else (*proposals_to_regions(proposals), bigtext_region())
    return notice_proposals(gameplay_frames(frames, params), cover, params, maps["static"])


def _holds(p: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        p["x"] <= b["x0"]
        and b["x1"] <= p["x"] + p["w"]
        and p["y"] <= b["y0"]
        and b["y1"] <= p["y"] + p["h"]
    )


TOP = ["Sector seven captured", "Relay lost", "Supply drop inbound", "Bridge sealed"]


def _top_notice() -> dict[tuple[int, int], list[dict[str, Any]]]:
    """A centred notice at the top edge with a new text in 4 of the 40 bursts, one frame each
    (too rare for its cells to count as dynamic)."""
    return {(b, 1): [_centred(t, 960, 12, h=24)] for b, t in zip((3, 12, 25, 33), TOP, strict=True)}


def test_top_notice_with_changing_text() -> None:
    extra = _top_notice()
    frames = _frames(extra)
    proposals, _ = cluster_regions(frames, DiscoverParams())
    assert not any(p["y"] < 40 and 600 < p["x"] < 1300 for p in proposals)  # missed before
    found = _extra(frames)
    assert len(found) == 1
    p = found[0]
    assert p["name"] == "extra_notice" and p["kind"] == "notice"
    assert all(_holds(p, b) for bs in extra.values() for b in bs)
    ev = p["evidence"]
    assert ev["outside"] is True
    assert (ev["bursts"], ev["frames"], ev["distinct"], ev["anchor"]) == (4, 4, 4, "centre")
    assert sorted(ev["samples"]) == sorted(TOP)
    region = notice_region(p)
    assert region["origin"] == "outside"
    assert (region["kind"], region["gate"], region["every"], region["det_scale"]) == (
        "notice",
        "det",
        3,
        1.0,
    )
    assert p["x"] + p["w"] < 1480  # clear of the feed


def test_bottom_notice_with_one_text_and_a_low_share() -> None:
    extra = {
        (b, k): [_centred("Hold the line", 960, 870, h=22)] for b in (5, 18, 30) for k in (0, 1)
    }
    found = _extra(_frames(extra))
    assert len(found) == 1
    assert all(_holds(found[0], b) for bs in extra.values() for b in bs)
    assert found[0]["evidence"]["distinct"] == 1
    assert found[0]["evidence"]["share"] < 0.05


def test_watermark_and_clock_give_nothing() -> None:
    extra = {
        (b, k): [
            _box("Capturely Free", 1640, 990, w=180),
            _box(f"12:{(b * 4 + k) % 60:02d}", 900, 20, w=70),
        ]
        for b in range(BURSTS)
        for k in range(4)
    }
    assert _extra(_frames(extra)) == []


def test_world_text_and_text_inside_the_feed_give_nothing() -> None:
    rng = random.Random(7)
    words = ["Outpost", "Harbor gate", "Silo north", "Old mill", "Ridge camp"]
    world = {
        (b, k): [_box(words[b % 5], x, y, w=120)]
        for b in range(0, BURSTS, 2)
        for x, y in [(rng.randrange(200, 1600), rng.randrange(200, 950))]
        for k in range(4)
    }
    assert _extra(_frames(world)) == []
    inside = {(b, 2): [_box(f"Ally {NAMES[b % 10]} revived", 1520, 85)] for b in (2, 9, 20, 31)}
    assert _extra(_frames(inside)) == []


def test_small_text_inside_big_text_is_outside() -> None:
    texts = ["Reload ready", "Shield charged", "Scanner online"]
    small = {(b, 2): [_centred(t, 900, 500)] for b, t in zip((2, 9, 17), texts, strict=True)}
    found = _extra(_frames(small))
    assert len(found) == 1 and found[0]["evidence"]["distinct"] == 3
    big = {(b, 2): [_centred(t, 900, 500, h=40)] for b, t in zip((2, 9, 17), texts, strict=True)}
    assert _extra(_frames(big)) == []  # the big-text check reads it


def test_partial_scoreboard_gives_nothing() -> None:
    def board(b: int) -> list[dict[str, Any]]:
        return [_box(f"{NAMES[(b + r) % 10]}ford {r + 3}", 60, 70 + r * 80) for r in range(12)]

    extra = {(b, k): board(b) for b in (4, 14, 24) for k in (0, 1)}
    frames = _frames(extra)
    assert all(len(f["boxes"]) < DiscoverParams().menu_min_boxes for f in frames)  # no menu
    assert _extra(frames) == []
    # the box-count guard drops it: without it, the rows are notices
    params = DiscoverParams(menu_jump=99.0, notice_max_regions=20)
    cover = (Region("feed", 1452, 12, 264, 144, "feed"),)
    rows = notice_proposals(gameplay_frames(frames, params), cover, params, _no_static())
    assert len(rows) == 12 and all(p["x"] < 300 for p in rows)


def test_stacked_lines_merge() -> None:
    goals = ["Escort the convoy", "Defend the gate", "Find the key"]
    extra = {
        (b, 1): [_centred("Objective updated", 960, 760), _centred(g, 960, 790)]
        for b, g in zip((6, 16, 26), goals, strict=True)
    }
    found = _extra(_frames(extra))
    assert len(found) == 1
    assert all(_holds(found[0], b) for bs in extra.values() for b in bs)


def _five() -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Five notices one above the other, one frame per burst; (bursts, distinct texts)
    per notice: (3, 1), (3, 2), (3, 3), (4, 2), (4, 3)."""
    plan = [(3, 1), (3, 2), (3, 3), (4, 2), (4, 3)]
    extra: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for i, (n_bursts, n_texts) in enumerate(plan):
        for j in range(n_bursts):
            text = f"Notice {'ABC'[i % 3]}{'xyz'[j % n_texts]}{'pqrst'[i]}"
            extra.setdefault((3 + 7 * j + i, 1), []).append(_centred(text, 960, 250 + i * 150))
    return extra


def test_best_two_of_five() -> None:
    found = _extra(_frames(_five()))
    assert [p["name"] for p in found] == ["extra_notice", "extra_notice2"]
    assert [p["score"] for p in found] == [12.0, 9.0]
    assert [p["y"] for p in found] == [850 - 12, 550 - 12]
    frames = _frames(_five())
    cover = (Region("feed", 1452, 12, 264, 144, "feed"), bigtext_region())
    every = DiscoverParams(notice_max_regions=9)
    found = notice_proposals(gameplay_frames(frames, every), cover, every, _no_static())
    assert sorted(p["score"] for p in found) == [3.0, 6.0, 8.0, 9.0, 12.0]  # all five valid
    off = DiscoverParams(notice_max_regions=0)
    assert notice_proposals(gameplay_frames(frames, off), cover, off, _no_static()) == []


def _no_static() -> Any:
    import numpy as np  # noqa: PLC0415

    return np.zeros((discover.GH, discover.GW), bool)


def test_names_never_clash() -> None:
    frames = _frames(_top_notice())
    feed = Region("feed", 1452, 12, 264, 144, "feed")
    taken = (feed, Region("extra_notice", 0, 900, 50, 50, "counter"), bigtext_region())
    params = DiscoverParams()
    found = notice_proposals(gameplay_frames(frames, params), taken, params, _no_static())
    assert [p["name"] for p in found] == ["extra_notice2"]


def test_deterministic() -> None:
    frames = _frames(_five() | _top_notice())
    shuffled = list(frames)
    random.Random(3).shuffle(shuffled)
    assert json.dumps(_extra(frames)) == json.dumps(_extra(shuffled))


def _source(tmp_path: Path) -> SimpleNamespace:
    path = tmp_path / "rec.mp4"
    return SimpleNamespace(path=path, io_path=path, name="rec.mp4", duration=2400.0)


def _patch(monkeypatch: pytest.MonkeyPatch, frames: list[dict[str, Any]]) -> None:
    monkeypatch.setattr(discover, "_ocr_frames", lambda *_a, **_k: frames)
    monkeypatch.setattr(discover, "probe", lambda _p: None)


def test_existing_preset_gets_suggestions_only(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = write_preset(paths.presets_dir(), verbfeed())
    before = target.read_bytes()
    _patch(monkeypatch, _frames(_top_notice()))
    project = tmp_path / "proj"
    out = pipeline._discover(
        project, [_source(tmp_path)], OcrParams(game="verbfeed"), True, BURSTS, "user"
    )
    rec = out["verbfeed"]
    assert rec["installed"] is False
    assert [r["name"] for r in rec["suggested_regions"]] == ["extra_notice"]
    assert rec["suggested_regions"][0]["origin"] == "outside"
    assert target.read_bytes() == before
    proposal = json.loads((project / "discovery" / "verbfeed" / "proposal.json").read_text())
    assert proposal["suggested_regions"] == rec["suggested_regions"]
    assert not any(r.get("origin") for r in proposal["regions"])


def test_new_preset_gets_the_regions(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, _frames(_top_notice()))
    out = pipeline._discover(
        tmp_path / "proj", [_source(tmp_path)], OcrParams(game="zonewar"), None, BURSTS, "user"
    )
    rec = out["zonewar"]
    assert rec["installed"] is True
    installed = json.loads(Path(rec["preset"]).read_text(encoding="utf-8"))
    extra = [r for r in installed["regions"] if r.get("origin") == "outside"]
    assert [r["name"] for r in extra] == ["extra_notice"]
    assert "suggested_regions" not in installed
    assert any(r["kind"] == "feed" for r in installed["regions"])


def test_origin_changes_no_cache_key() -> None:
    region = {"name": "extra_notice", "x": 700, "y": 0, "w": 520, "h": 60, "kind": "notice"}
    region |= {"gate": "det", "every": 3, "det_scale": 1.0}
    plain = verbfeed(regions=[*verbfeed()["regions"], region])
    marked = verbfeed(regions=[*verbfeed()["regions"], region | {"origin": "outside"}])
    a, b = preset_from_dict(plain), preset_from_dict(marked)
    assert OcrParams().key("verbfeed", a) == OcrParams().key("verbfeed", b)

    def raw(p: Any) -> dict[str, Any]:
        return ocr_raw.raw_key(3.0, [r.key() for r in p.all_regions()], "bottom", 0.5, [], 960)

    assert raw(a) == raw(b)


def test_noise_is_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    # 160 frames of 20 random lettered boxes each: stacked-line merging once took 133 s
    rng = random.Random(1)
    frames = [
        _frame(
            "rec.mp4",
            i // 4,
            i % 4,
            [
                _box(
                    "".join(rng.choice(string.ascii_letters) for _ in range(8)),
                    rng.randrange(0, 1760),
                    rng.randrange(0, 1060),
                    w=120,
                    h=rng.randrange(12, 40),
                )
                for _ in range(20)
            ],
        )
        for i in range(160)
    ]
    calls = {"rect": 0, "clusters": 0}
    edges_rect, merge = discover._edges_rect, discover._merge_stacked

    def count_rect(e: Any) -> Any:
        calls["rect"] += 1
        return edges_rect(e)

    def count_merge(clusters: Any) -> Any:
        calls["clusters"] += len(clusters)
        return merge(clusters)

    monkeypatch.setattr(discover, "_edges_rect", count_rect)
    monkeypatch.setattr(discover, "_merge_stacked", count_merge)
    params = DiscoverParams()
    t0 = time.perf_counter()
    notice_proposals(gameplay_frames(frames, params), (bigtext_region(),), params, _no_static())
    assert time.perf_counter() - t0 < 10  # about 0.3 s
    assert calls["clusters"] > 50  # the noise forms many chance clusters
    # one rectangle per cluster, one per merge (each merge removes a cluster), the proposals
    assert calls["rect"] <= 3 * calls["clusters"]


FEED = Region("feed", 1452, 12, 264, 144, "feed")


def test_padding_trimmed_at_a_covered_region() -> None:
    # left-aligned: it grows to the right, into the feed, which keeps its own region
    extra = {
        (b, 1): [_box(t, 1150, 60, w=14 * len(t))]
        for b, t in zip((3, 12, 25, 33), TOP, strict=True)
    }
    found = _extra(_frames(extra), covered=(FEED, bigtext_region()))
    assert len(found) == 1 and found[0]["evidence"]["anchor"] == "left"
    assert found[0]["x"] + found[0]["w"] <= FEED.x
    assert all(_holds(found[0], b) for bs in extra.values() for b in bs)


def test_notice_centred_on_a_covered_region_gives_nothing() -> None:
    # two stacked lines whose boxes lie outside a thin counter between them: the merged
    # notice's centre is covered, so no cut keeps it
    extra = {
        (b, 1): [_centred("Objective updated", 960, 770), _centred(g, 960, 810)]
        for b, g in zip(
            (6, 16, 26), ["Escort the convoy", "Defend the gate", "Find it"], strict=True
        )
    }
    frames = _frames(extra)
    assert len(_extra(frames, covered=(FEED, bigtext_region()))) == 1
    counter = Region("counter", 600, 795, 720, 10, "counter")
    assert _extra(frames, covered=(FEED, counter, bigtext_region())) == []
    assert discover._trim((800, 760, 1120, 840), counter, (960, 800)) is None
