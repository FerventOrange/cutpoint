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


def test_bottom_notice_with_two_texts_in_three_bursts() -> None:
    texts = ["Hold the line", "Zone lost", "Hold the line"]
    extra = {
        (b, k): [_centred(t, 960, 870, h=22)]
        for b, t in zip((5, 18, 30), texts, strict=True)
        for k in (0, 1)
    }
    found = _extra(_frames(extra))
    assert len(found) == 1
    assert all(_holds(found[0], b) for bs in extra.values() for b in bs)
    assert found[0]["evidence"]["distinct"] == 2
    assert found[0]["evidence"]["share"] < 0.05


def _label(texts: list[str], bursts: range, x: int = 1700, y: int = 20) -> dict[Any, Any]:
    """Right-aligned text at (x, y) in two frames of each of ``bursts``, ``texts`` in turn."""
    return {
        (b, k): [_box(t, x + 140 - 14 * len(t), y, w=14 * len(t))]
        for i, b in enumerate(bursts)
        for k in (0, 1)
        for t in [texts[i % len(texts)]]
    }


def test_labels_that_come_back_give_nothing() -> None:
    # one text in 3 bursts, or two in 7 (one of them misread now and then): a map or mode
    # label, not a notice, even in few frames
    assert _extra(_frames(_label(["Harbor"], range(4, 40, 13)))) == []
    two = ["Harbor", "Casual", "Harbor", "Casual", "Harb0r", "Casual", "Harbour"]
    assert _extra(_frames(_label(two, range(2, 40, 5)))) == []
    # OCR cuts and misreads of one long line still count as one text
    cuts = ["Custom fonts enabled", "Custom fon", "Custom fonts enab1ed", "Cust"]
    assert _extra(_frames(_label(cuts * 2, range(1, 40, 5), x=200, y=80))) == []


def test_block_of_labels_with_one_changing_line_gives_nothing() -> None:
    # a menu: three stacked labels in 10 bursts and, under them, a line that changes now
    # and then; merged, the labels make up most of it
    rows = ["Server region west", "Party of four", "Mode classic"]
    news = dict(zip((4, 16, 28), ["Friend joined", "Invite sent", "Friend left"], strict=True))
    changing = {(b, 0): [_box(t, 60, 150, w=14 * len(t))] for b, t in news.items()}
    labels = {
        (b, k): [_box(t, 60, 60 + 30 * r, w=14 * len(t)) for r, t in enumerate(rows)]
        for b in range(0, 40, 4)
        for k in (0, 1)
    }
    menu = {key: labels.get(key, []) + changing.get(key, []) for key in labels | changing}
    assert _extra(_frames(menu)) == []
    assert len(_extra(_frames(changing))) == 1  # on its own, the changing line is a notice


def test_fixed_phrase_with_changing_names_stays() -> None:
    # one long phrase with another name in each of 5 bursts: 5 texts, not OCR variants of one
    names = ["Boxer", "Quill", "Orrin", "Sable", "Tarn"]
    extra = {
        (b, k): [_centred(f"Enemy convoy spotted by {n}", 960, 300)]
        for b, n in zip(range(2, 40, 8), names, strict=True)
        for k in (0, 1)
    }
    # in 2 frames per burst main discovery finds it too; the preset of a game may not have it
    found = _extra(_frames(extra), covered=(FEED, bigtext_region()))
    assert len(found) == 1 and found[0]["evidence"]["distinct"] == 5
    assert not discover._same_text("enemyconvoyspottedbyboxer", "enemyconvoyspottedbyquill")
    assert discover._same_text("enemyconvoyspottedbyboxer", "enemyconvoyspottedbyboker")


def test_brief_notice_that_recurs_stays() -> None:
    # one text in 6 bursts but a single frame each time: a notice that recurs, not a label
    brief = {(b, 1): [_centred("Hold the line", 960, 300)] for b in range(3, 40, 7)}
    assert len(_extra(_frames(brief))) == 1
    # in 3 of the 4 frames of each burst it stays: a label
    stays = {(b, k): bs for (b, _), bs in brief.items() for k in (0, 1, 2)}
    assert _extra(_frames(stays)) == []


def test_column_of_short_names_stays() -> None:
    # a killer column split off a feed by a wide icon: short camelCase or digit names are
    # readouts one by one, but a gauge repeats a few texts and these are all different
    names = ["DarkWolf", "Wren7", "TinFox", "RedOwl9", "IceBat", "SunElk3", "BluJay"]
    names += ["MossCat", "Vexo42", "OakRam"]
    column = {
        (b, 1): [_box(n, 300, 500, w=14 * len(n))]
        for b, n in zip(range(1, 40, 4), names, strict=True)
    }
    assert all(discover._readout(n, DiscoverParams().notice_readout_chars) for n in names)
    found = _extra(_frames(column))
    assert len(found) == 1 and found[0]["evidence"]["distinct"] == 10


def test_two_texts_that_keep_coming_back_give_nothing() -> None:
    # two texts in turn over 10 bursts: a menu button, not a notice
    assert _extra(_frames(_label(["Ready up", "Leave queue"], range(0, 40, 4), y=300))) == []


def test_gauge_readouts_give_nothing() -> None:
    # a gauge: short label-and-number lines that change with every burst
    texts = ["ALT40ft", "SPD9kt", "FUEL5", "TRIMup", "ALT45ft", "SPD12kt", "FUEL4"]
    gauge = {
        (b, k): [_box(texts[(b + k) % 7], 20, 960, w=100)] for b in range(0, 40, 3) for k in (0, 1)
    }
    assert not discover._steady(
        [{"text": t, "bkey": ("rec.mp4", b), "fid": str(b)} for b, t in enumerate(texts)],
        DiscoverParams(),
    )
    assert _extra(_frames(gauge)) == []
    # plain words in capitals are no readouts: a role list with changing words stays
    roles = ["PILOT", "MEDIC", "SCOUT", "PILOT"]
    found = _extra(_frames(_label(roles, range(3, 40, 10), y=300)))
    assert len(found) == 1 and found[0]["evidence"]["distinct"] == 3


def test_readout_rule() -> None:
    chars = DiscoverParams().notice_readout_chars
    assert discover._readout("RPM2100", chars) and discover._readout("ALTft", chars)
    assert not discover._readout("Gunner", chars) and not discover._readout("LOADER", chars)
    assert not discover._readout("commander", chars)
    assert not discover._readout("Vexford12 Wren", chars)  # a longer line, such as a feed
    # letters of any script count: a Cyrillic or CJK notice with a digit is no readout
    assert not discover._readout("Захвачена точка 2", chars)
    assert not discover._readout("我方正在占领据点2", chars)
    assert not discover._readout("Танк", chars)  # a capitalised word
    # scripts without case have no mixed-case words
    assert not discover._readout("据点已占领", chars) and not discover._readout("عدو", chars)


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
        names = [NAMES[(b // 10 + r) % 10] for r in range(12)]
        return [_box(f"{n}fordson {r + 3}", 60, 70 + r * 80) for r, n in enumerate(names)]

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
    words = ["Relay lost", "Gate sealed", "Drop inbound"]
    extra: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for i, (n_bursts, n_texts) in enumerate(plan):
        for j in range(n_bursts):
            text = f"{words[j % n_texts]} {['north', 'south', 'east', 'west', 'core'][i]}"
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


@pytest.mark.parametrize(("outside", "suggested"), [(1, ["extra_notice2"]), (2, [])])
def test_preset_outside_regions_count_against_the_cap(
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outside: int,
    suggested: list[str],
) -> None:
    # two notices: a preset with one outside region gets one suggestion, with two none
    own = [
        {"name": f"extra_notice{'' if i == 0 else i + 1}", "x": 0, "y": 400 + 60 * i}
        | {"w": 50, "h": 40, "kind": "notice", "gate": "det", "every": 3, "origin": "outside"}
        for i in range(outside)
    ]
    write_preset(paths.presets_dir(), verbfeed(regions=[*verbfeed()["regions"], *own]))
    _patch(monkeypatch, _frames(_five()))
    out = pipeline._discover(
        tmp_path / "proj", [_source(tmp_path)], OcrParams(game="verbfeed"), True, BURSTS, "user"
    )
    assert [r["name"] for r in out["verbfeed"]["suggested_regions"]] == suggested


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
