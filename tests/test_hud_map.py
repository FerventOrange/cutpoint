"""The vision HUD map (#1069): boxes to regions, consensus, OCR validation, the merge with
discovery's proposals, the top-up of an installed preset and the shared vision calls.

All frames, boxes and model replies are invented; the vision model is a fake (no network)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from cutpoint.events import discover, feed_labels, hud_map
from cutpoint.events.feed_labels import StopLabelling, VisionCalls
from cutpoint.events.regions import Region
from ocr_helpers import box as _box

NAMES = ["Alder", "Boxer", "Kip", "Nova", "Quill", "Orrin", "Sable", "Tarn", "Vex", "Wren"]
FEED = (1500, 100, 380, 140)  # a kill feed's rectangle in reference px


def _frames(
    bursts: int, per: int, boxes_of: Any, *, only: set[int] | None = None
) -> list[dict[str, Any]]:
    """``bursts`` x ``per`` OCR'd frames; ``boxes_of(burst, n)`` gives a frame's boxes (``n``:
    its running number), for the bursts in ``only`` when given."""
    out = []
    for b in range(bursts):
        for k in range(per):
            n = b * per + k
            out.append(
                {
                    "id": f"rec.mp4@{n}",
                    "source": "rec.mp4",
                    "burst": b,
                    "k": k,
                    "frame": f"b{b:03d}_{k}.jpg",
                    "boxes": boxes_of(b, n) if only is None or b in only else [],
                }
            )
    return out


def _feed_rows(n: int, x: int = 1500, w: int = 380, rows: int = 3) -> list[dict[str, Any]]:
    return [
        _box(f"{NAMES[(n + r) % 10]} killed {NAMES[(n + r + 3) % 10]}", x, 100 + 40 * r, w, 28)
        for r in range(rows)
    ]


def _write_images(folder: Path, frames: list[dict[str, Any]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    img = np.full((720, 1280, 3), 40, np.uint8)
    for f in frames:
        cv2.imwrite(str(folder / f["frame"]), img)


def _group(
    kind: str, box: tuple[int, int, int, int], seen: int = 2, hints: tuple[str, ...] = ()
) -> dict[str, Any]:
    return {"kind": kind, "box": box, "frames": set(range(seen)), "hints": list(hints)}


# ---------------------------------------------------------------- boxes and replies


def test_to_ref_reads_a_thousand_scale_box() -> None:
    # 1280x720 image: 0-1000 scale, then 1.5x to the 1920x1080 reference
    assert hud_map.to_ref([100, 200, 300, 400], (1280, 720)) == (192, 216, 384, 216)


def test_to_ref_reads_pixels_when_a_value_is_over_a_thousand() -> None:
    # 2560x1440 image sent: pixels, 0.75x to the reference
    assert hud_map.to_ref([1200, 100, 1400, 300], (2560, 1440)) == (900, 75, 150, 150)


def test_to_ref_clamps_and_rejects_what_is_no_box() -> None:
    assert hud_map.to_ref([-50, -50, 100, 100], (1280, 720)) == (0, 0, 192, 108)
    assert hud_map.to_ref([900, 900, 1000, 1000], (1280, 720)) == (1728, 972, 192, 108)
    assert hud_map.to_ref([2000, 10, 2100, 50], (1920, 1080)) is None  # wholly outside
    for bad in ("box", [1, 2, 3], [1, 2, 3, "x"], [1, 2, 3, float("nan")], None):
        assert hud_map.to_ref(bad, (1280, 720)) is None


def test_to_ref_takes_corners_in_either_order() -> None:
    assert hud_map.to_ref([300, 400, 100, 200], (1280, 720)) == hud_map.to_ref(
        [100, 200, 300, 400], (1280, 720)
    )


def test_parse_keeps_boxes_and_drops_tiny_ones_unknown_kinds_and_junk() -> None:
    reply = {
        "elements": [
            {"kind": "kill_feed", "box": [781, 92, 979, 222], "text_seen": "A killed B"},
            {"kind": "score", "box": [0, 0, 1, 1], "text_seen": ""},  # 2 px: tiny
            {"kind": "minimap", "box": [0, 0, 500, 500], "text_seen": ""},  # unknown kind
            {"kind": "chat", "box": "nope", "text_seen": ""},
            "junk",
        ]
    }
    got = hud_map.parse(reply, (1920, 1080))
    assert got is not None and [e["kind"] for e in got] == ["kill_feed"]
    assert "text_seen" not in got[0]  # what the model read is never kept
    assert hud_map.parse({"elements": "x"}, (1920, 1080)) is None
    assert hud_map.parse(None, (1920, 1080)) is None


# ---------------------------------------------------------------- consensus


def test_group_joins_one_kind_overlapping_boxes_one_per_frame() -> None:
    a = [
        {"kind": "kill_feed", "box": [1500, 100, 380, 140]},
        {"kind": "chat", "box": [0, 0, 300, 200]},
    ]
    b = [
        {"kind": "kill_feed", "box": [1520, 110, 380, 140]},  # same element
        {"kind": "chat", "box": [1500, 100, 380, 140]},  # another place
        {"kind": "score", "box": [1500, 100, 380, 140]},  # another kind
    ]
    got = hud_map.group([a, b])
    by = {(g["kind"], tuple(sorted(g["frames"]))): g for g in got}
    assert set(by) == {
        ("kill_feed", (0, 1)),
        ("chat", (0,)),
        ("chat", (1,)),
        ("score", (1,)),
    }
    assert by[("kill_feed", (0, 1))]["box"] == (1500, 100, 400, 150)  # their union


def test_agree_needs_a_shared_element_and_every_persistent_kind_in_both() -> None:
    feed = {"kind": "kill_feed", "box": list(FEED)}
    banner = {"kind": "objective_banner", "box": [400, 400, 600, 100]}
    assert hud_map.agree([feed, banner], [feed])  # a banner shows at times only
    assert not hud_map.agree([feed], [])  # nothing shared
    assert not hud_map.agree([feed, banner], [banner])  # the feed is on screen all the time
    assert not hud_map.agree([], [])


# ---------------------------------------------------------------- judge


def test_judge_confirms_a_kill_feed_and_makes_its_region_from_the_ocr_boxes() -> None:
    frames = _frames(4, 2, lambda _b, n: _feed_rows(n))
    rec = hud_map.judge(_group("kill_feed", FEED), frames, 2)
    assert rec["confirmed"] and rec["seen"] == 2
    x, y, w, h = rec["rect"]
    # the union of the rows (1500,100)-(1880,208), padded by PAD
    assert (x, y, x + w, y + h) == (1488, 88, 1892, 220)


def test_judge_kill_feed_thresholds() -> None:
    few_bursts = _frames(2, 2, lambda _b, n: _feed_rows(n))
    assert hud_map.judge(_group("kill_feed", FEED), few_bursts, 2)["why"].startswith("OCR text")
    loadout = _frames(4, 2, lambda _b, _n: [_box("LOADOUT", 1500, 100, 200, 28)])
    rec = hud_map.judge(_group("kill_feed", FEED), loadout, 2)
    assert not rec["confirmed"] and "too few name lines" in rec["why"]
    digits = _frames(
        4, 2, lambda _b, n: [_box(f"{n}{r}12", 1500, 100 + 40 * r, 90, 28) for r in range(3)]
    )
    assert not hud_map.judge(_group("kill_feed", FEED), digits, 2)["confirmed"]  # not letters


def test_judge_caps_a_region_at_a_share_of_the_frame_except_banners() -> None:
    wide = _frames(4, 2, lambda _b, n: _feed_rows(n, x=100, w=1800, rows=6))
    zone = (100, 100, 1800, 228)
    rec = hud_map.judge(_group("kill_feed", zone), wide, 2)
    assert not rec["confirmed"] and rec["why"] == "too large"

    banner = _frames(
        4, 2, lambda _b, _n: [_box("VICTORY ACHIEVED", 400, 400, 1100, 300)], only={0, 1}
    )
    rec = hud_map.judge(_group("objective_banner", (400, 400, 1100, 300)), banner, 2)
    _x, _y, w, h = rec["rect"]
    assert rec["confirmed"] and w * h > hud_map.MAX_AREA * 1920 * 1080


def test_judge_rejects_a_banner_whose_text_is_there_most_of_the_time() -> None:
    zone = (882, 194, 158, 56)

    def label(_b: int, _n: int) -> list[dict[str, Any]]:
        return [_box("ROUND", 890, 200, 140, 40)]

    always = _frames(8, 2, label)
    rec = hud_map.judge(_group("objective_banner", zone), always, 2)
    assert not rec["confirmed"] and "most of the time" in rec["why"]
    rare = _frames(8, 2, label, only={1, 5})
    assert hud_map.judge(_group("objective_banner", zone), rare, 2)["confirmed"]


def test_judge_rejects_a_near_full_frame_banner_even_on_strong_evidence() -> None:
    big = (12, 44, 1908, 1036)
    frames = _frames(
        8, 2, lambda _b, _n: [_box("VICTORY ACHIEVED", 20, 50, 1880, 1020)], only={1, 3, 5, 7}
    )
    rec = hud_map.judge(_group("objective_banner", big, seen=1), frames, 2)
    assert not rec["confirmed"] and rec["why"] == "too large"


def test_judge_a_strong_single_frame_counts_and_a_weak_one_does_not() -> None:
    strong = _frames(6, 2, lambda _b, n: _feed_rows(n))
    weak = _frames(4, 2, lambda _b, n: _feed_rows(n))
    assert hud_map.judge(_group("kill_feed", FEED, seen=1), strong, 3)["confirmed"]
    rec = hud_map.judge(_group("kill_feed", FEED, seen=1), weak, 3)
    assert not rec["confirmed"] and "one frame" in rec["why"]
    assert hud_map.judge(_group("kill_feed", FEED, seen=2), weak, 3)["confirmed"]


def test_judge_score_reads_small_numbers_that_change() -> None:
    zone = (900, 40, 60, 40)
    moving = _frames(4, 2, lambda b, _n: [_box(str(b + 1), 910, 45, 30, 30)])
    rec = hud_map.judge(_group("score", zone), moving, 2)
    assert rec["confirmed"] and rec["evidence"]["values"] == [1, 2, 3, 4]
    assert rec["median_text_h"] == 30
    still = _frames(4, 2, lambda _b, _n: [_box("3", 910, 45, 30, 30)])
    assert not hud_map.judge(_group("score", zone), still, 2)["confirmed"]
    clock = _frames(4, 2, lambda b, _n: [_box(f"0:{b}5", 910, 45, 30, 30)])
    assert not hud_map.judge(_group("score", zone), clock, 2)["confirmed"]


def test_judge_replay_marker_shows_only_some_of_the_time() -> None:
    zone = (800, 60, 300, 60)

    def marker(_b: int, _n: int) -> list[dict[str, Any]]:
        return [_box("INSTANT REPLAY", 820, 70, 260, 30)]

    rare = _frames(8, 2, marker, only={1, 5})
    assert hud_map.judge(_group("replay", zone, hints=("instantreplay",)), rare, 2)["confirmed"]
    always = _frames(8, 2, marker)
    rec = hud_map.judge(_group("replay", zone, hints=("instantreplay",)), always, 2)
    assert not rec["confirmed"] and "most of the time" in rec["why"]


def test_judge_exclusions_need_two_frames_and_no_ocr() -> None:
    nothing: list[dict[str, Any]] = []
    for kind in hud_map.EXCLUDE:
        assert hud_map.judge(_group(kind, (0, 0, 300, 200), seen=2), nothing, 2)["confirmed"]
        rec = hud_map.judge(_group(kind, (0, 0, 300, 200), seen=1), nothing, 2)
        assert not rec["confirmed"] and rec["why"] == "marked in one frame"
    rec = hud_map.judge(_group("clock", (0, 0, 300, 200)), nothing, 2)
    assert not rec["confirmed"] and rec["why"] == "ignored kind"


# ---------------------------------------------------------------- settle and apply


def _prop(name: str, kind: str, x: int, y: int, w: int = 380, h: int = 140) -> dict[str, Any]:
    return {"name": name, "kind": kind, "score": 1.0, "x": x, "y": y, "w": w, "h": h}


def _el(kind: str, rect: tuple[int, int, int, int]) -> dict[str, Any]:
    return {"kind": kind, "box": list(rect), "rect": list(rect), "seen": 2, "confirmed": True}


def test_settle_labels_a_proposal_and_makes_a_region_for_an_element_on_none() -> None:
    hud = {
        "model": "ollama@h:m",
        "elements": [
            _el("kill_feed", (1500, 100, 380, 140)),
            _el("kill_feed", (100, 800, 380, 140)),  # no proposal there
            _el("replay", (800, 60, 300, 60)),
            _el("chat", (0, 300, 400, 300)),
            {"kind": "score", "box": [0, 0, 9, 9], "seen": 1, "confirmed": False},
        ],
    }
    props = [_prop("feed", "feed", 1510, 105), _prop("chatfeed", "feed", 20, 320, 300, 200)]
    taken = {"feed", "chatfeed"}
    labels, regions, excl = hud_map.settle(hud, props, taken)
    assert labels["feed"] == {"label": "kill_feed", "model": "ollama@h:m", "from": "hud_map"}
    assert labels["chatfeed"]["label"] == "chat"
    assert [(r.name, r.kind) for r in regions] == [("vision_feed", "feed"), ("replay", "replay")]
    assert regions[0].gate == "det" and regions[1].every == 1
    assert excl == [(0, 300, 400, 300)]
    assert {"vision_feed", "replay"} <= taken


def test_settle_labels_a_score_proposal_inside_a_wide_score_element() -> None:
    hud = {"model": "m", "elements": [_el("score", (586, 148, 670, 72))]}
    props = [
        _prop("score2", "score", 600, 150, 100, 60),
        _prop("score", "score", 1100, 150, 100, 60),
    ]
    taken = {"score2", "score"}
    labels, regions, _ = hud_map.settle(hud, props, taken)
    assert regions == [] and set(labels) == {"score2", "score"}
    assert {v["label"] for v in labels.values()} == {hud_map.LABEL_OF["score"]}


def test_settle_leaves_out_an_oversized_banner_and_keeps_a_normal_one() -> None:
    hud = {
        "model": "m",
        "elements": [
            _el("objective_banner", (12, 44, 1908, 1036)),
            _el("objective_banner", (714, 185, 400, 80)),
        ],
    }
    _, regions, _ = hud_map.settle(hud, [], set())
    assert [r.kind for r in regions] == ["banner"] and regions[0].x == 714


@pytest.mark.parametrize(
    ("element", "kind"), [("chat", "feed"), ("webcam", "notice"), ("caption", "notice")]
)
def test_a_rule_made_feed_in_an_ignored_box_is_a_feed_only_in_a_chat_box(
    element: str, kind: str
) -> None:
    props = [_prop("feed", "feed", 32, 332, 276, 176)]
    hud = {
        "model": "m",
        "elements": [{"kind": element, "box": [0, 300, 400, 300], "confirmed": True}],
    }
    labels, new, excl = hud_map.settle(hud, props, {"feed"})
    regions = discover.proposals_to_regions(props)
    regions, extra = discover.apply_labels(props, labels, regions, [])
    got, _ = hud_map.apply(regions, extra, new, excl, labels, [])
    assert [(r.name, r.kind) for r in got] == [("feed", kind)]


def test_apply_does_not_turn_an_ignored_area_feed_into_a_bogus_notice_when_judged() -> None:
    regions = [Region("feed", 20, 320, 300, 200, "feed", gate="det")]
    labels = {"feed": {"label": "chat", "from": "hud_map", "model": "m"}}
    frames = _frames(4, 2, lambda _b, n: _feed_rows(n))
    vision = Region("vision_feed", 800, 500, 380, 140, "feed", gate="det")
    got, _ = hud_map.apply(regions, [], [vision], [(0, 300, 400, 300)], labels, frames)
    assert [(r.name, r.kind) for r in got] == [("vision_feed", "feed"), ("feed", "feed")]


def test_apply_keeps_a_feed_in_ocr_reads_feed_lines_and_demotes_one_that_it_does_not() -> None:
    frames = _frames(4, 2, lambda _b, n: _feed_rows(n))  # feed lines at FEED only
    loadout = Region("loadout", 100, 800, 380, 140, "feed", gate="det")
    real = Region("real", 1500, 100, 380, 140, "feed", gate="det")
    vision = Region("vision_feed", 800, 500, 380, 140, "feed", gate="det")
    got, _ = hud_map.apply([loadout, real], [], [vision], [], {}, frames)
    assert [(r.name, r.kind) for r in got] == [
        ("vision_feed", "feed"),  # new feeds first
        ("loadout", "notice"),
        ("real", "feed"),
    ]


def test_apply_leaves_a_feed_a_model_named_and_judges_nothing_without_a_vision_feed() -> None:
    frames = _frames(4, 2, lambda _b, n: _feed_rows(n))
    odd = Region("odd", 100, 800, 380, 140, "feed", gate="det")
    got, _ = hud_map.apply([odd], [], [], [], {}, frames)  # no kill feed from the map: as before
    assert got[0].kind == "feed"
    labels = {"odd": {"label": "kill_feed", "model": "m", "from": "hud_map"}}
    got, _ = hud_map.apply([odd], [], [], [], labels, frames)
    assert got[0].kind == "feed"


def test_apply_drops_an_outside_notice_on_a_new_feed() -> None:
    vision = Region("vision_feed", 1500, 100, 380, 140, "feed", gate="det")
    extra = [
        {"name": "outside", "x": 1510, "y": 110, "w": 360, "h": 120, "kind": "notice"},
        {"name": "far", "x": 100, "y": 900, "w": 300, "h": 80, "kind": "notice"},
    ]
    _, kept = hud_map.apply([], extra, [vision], [], {}, [])
    assert [e["name"] for e in kept] == ["far"]


# ---------------------------------------------------------------- top_up and wants_top_up


def _preset(*regions: Region, **more: Any) -> dict[str, Any]:
    return {"game": "g", "source": "discovered", "regions": [r.key() for r in regions], **more}


def _hud(*elements: dict[str, Any]) -> dict[str, Any]:
    return {"v": 1, "model": "m", "elements": list(elements)}


FEED_FRAMES = _frames(4, 2, lambda _b, n: _feed_rows(n))


def test_top_up_relabels_a_feed_region_the_map_confirms_and_changes_nothing_else() -> None:
    data = _preset(Region("feed", 1500, 100, 380, 140, "feed", gate="det"))
    new, changes = hud_map.top_up(data, _hud(_el("kill_feed", (1490, 95, 400, 150))), FEED_FRAMES)
    assert changes == {"added": [], "replaced": [], "relabelled": ["feed"]}
    assert new["regions"][0]["label"] == "kill_feed" and len(new["regions"]) == 1
    assert "feed" not in new  # a labelled feed needs no row_notices
    assert data["regions"][0].get("label") is None  # the input is not changed


def test_top_up_replaces_a_loadout_box_and_adds_the_feed_with_row_notices() -> None:
    data = _preset(Region("loadout", 100, 800, 380, 140, "feed", gate="det"))
    new, changes = hud_map.top_up(data, _hud(_el("kill_feed", FEED)), FEED_FRAMES)
    assert changes["replaced"] == ["loadout"] and changes["added"] == ["vision_feed"]
    first, loadout = new["regions"]
    assert first["name"] == "vision_feed" and first["origin"] == "vision"
    assert first["label"] == "kill_feed"
    # the loadout box is read as a notice from now on, not deleted
    assert (loadout["name"], loadout["kind"], loadout["was"]) == ("loadout", "notice", "feed")
    assert new["feed"] == {"row_notices": True}


def test_top_up_keeps_the_kill_words_of_a_preset_instead_of_row_notices() -> None:
    data = _preset(feed={"kill_words": {"zapped": "kill"}})
    new, _ = hud_map.top_up(data, _hud(_el("kill_feed", FEED)), FEED_FRAMES)
    assert new["feed"] == {"kill_words": {"zapped": "kill"}}


def test_top_up_lets_the_feed_take_over_a_notice_region_it_lies_on() -> None:
    data = _preset(Region("outside", 1500, 100, 380, 140, "notice", gate="det", every=3))
    new, changes = hud_map.top_up(data, _hud(_el("kill_feed", FEED)), FEED_FRAMES)
    assert changes["relabelled"] == ["outside"] and changes["added"] == []
    assert [(r["name"], r["kind"]) for r in new["regions"]] == [("outside", "feed")]


def test_top_up_adds_a_score_and_a_replay_once_and_only_where_the_preset_lacks_them() -> None:
    score = _el("score", (900, 40, 60, 40))
    score["evidence"] = {"values": [1, 2]}
    replay = _el("replay", (800, 60, 300, 60))
    other_replay = _el("replay", (100, 500, 300, 60))
    banner = _el("objective_banner", (400, 400, 600, 100))
    data = _preset(Region("score", 10, 10, 60, 40, "score", gate="read"))
    new, changes = hud_map.top_up(data, _hud(score, replay, other_replay, banner), [])
    # a score exists; one replay marker (the replay marker comes last)
    assert changes["added"] == ["vision_banner", "replay"]
    assert sorted(r["kind"] for r in new["regions"]) == ["banner", "replay", "score"]
    assert all(r.get("origin") == "vision" for r in new["regions"][1:])
    # nothing over a region that is there
    data = _preset(Region("banner", 400, 400, 600, 100, "banner", gate="det"))
    _, changes = hud_map.top_up(data, _hud(banner), [])
    assert changes["added"] == []


def test_top_up_adds_no_oversized_banner() -> None:
    data = _preset()
    big = _el("objective_banner", (12, 44, 1908, 1036))
    _, changes = hud_map.top_up(data, _hud(big), [])
    assert changes["added"] == []


def test_top_up_records_the_map_once() -> None:
    data = _preset(Region("loadout", 100, 800, 380, 140, "feed", gate="det"), discovery={"x": 1})
    new, changes = hud_map.top_up(data, _hud(_el("kill_feed", FEED)), FEED_FRAMES)
    rec = new["discovery"]["hud_map"]
    assert rec["top_up"] == changes | {"originals": data["regions"]}
    assert new["discovery"]["x"] == 1
    assert rec["elements"][0]["kind"] == "kill_feed"


def test_top_up_ignores_unconfirmed_and_unwanted_elements() -> None:
    chat = _el("chat", (0, 300, 400, 300))
    weak = _el("score", (900, 40, 60, 40)) | {"confirmed": False}
    new, changes = hud_map.top_up(_preset(), _hud(chat, weak), [])
    assert changes == {"added": [], "replaced": [], "relabelled": []} and new["regions"] == []


def test_wants_top_up_only_for_a_discovered_preset_in_the_presets_folder(tmp_path: Path) -> None:
    presets, project = tmp_path / "presets", tmp_path / "project"
    presets.mkdir()
    project.mkdir()

    def write(folder: Path, data: dict[str, Any]) -> Path:
        path = folder / "g.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def preset(path: Path, source: str = "discovered") -> SimpleNamespace:
        return SimpleNamespace(path=path, source=source)

    plain = write(presets, {"discovery": {"params": {}}})
    assert hud_map.wants_top_up(preset(plain), presets)
    assert not hud_map.wants_top_up(preset(plain, "bundled"), presets)
    assert not hud_map.wants_top_up(preset(write(project, {"discovery": {}})), presets)
    assert not hud_map.wants_top_up(preset(None), presets)
    handmade = write(presets, {"regions": []})  # no discovery block: made by hand
    assert not hud_map.wants_top_up(preset(handmade), presets)
    mapped = write(presets, {"discovery": {"hud_map": {"v": 1}}})
    assert not hud_map.wants_top_up(preset(mapped), presets)  # already mapped


# ---------------------------------------------------------------- map_hud and the cache


def _reply(*kinds_boxes: tuple[str, list[int]]) -> dict[str, Any]:
    return {"elements": [{"kind": k, "box": b, "text_seen": ""} for k, b in kinds_boxes]}


FEED_BOX = ("kill_feed", [781, 92, 979, 222])  # FEED on the 0-1000 scale


def _map(
    tmp_path: Path, replies: list[dict[str, Any] | None], model: str = "m", **kw: Any
) -> tuple[dict[str, Any] | None, list[str]]:
    frames = _frames(4, 2, lambda _b, n: _feed_rows(n))
    folder = tmp_path / "frames"
    _write_images(folder, frames)
    prompts: list[str] = []
    it = iter(replies)

    def ask(text: str, jpegs: list[bytes], schema: dict[str, Any], num_predict: int) -> Any:
        prompts.append(text)
        assert len(jpegs) == 1 and schema is hud_map.SCHEMA
        return next(it)

    return hud_map.map_hud([], frames, folder, ask, model, **kw), prompts


def test_map_hud_skips_the_third_frame_when_the_first_two_agree(tmp_path: Path) -> None:
    hud, prompts = _map(tmp_path, [_reply(FEED_BOX)] * 3)
    assert hud is not None and len(prompts) == 2 and hud["calls"] == 2
    assert len(hud["frames"]) == 2
    (el,) = hud["elements"]
    assert el["kind"] == "kill_feed" and el["confirmed"] and el["seen"] == 2


def test_map_hud_asks_the_third_frame_when_the_first_two_disagree(tmp_path: Path) -> None:
    hud, prompts = _map(tmp_path, [_reply(FEED_BOX), _reply(), _reply(FEED_BOX)])
    assert hud is not None and len(prompts) == 3 and len(hud["frames"]) == 3
    assert [e["seen"] for e in hud["elements"]] == [2]


def test_map_hud_caches_per_model_frame_context_and_category(tmp_path: Path) -> None:
    replies: list[dict[str, Any] | None] = [_reply(FEED_BOX)] * 2
    first, prompts = _map(tmp_path, replies)
    assert first is not None and len(prompts) == 2
    again, prompts = _map(tmp_path, [])  # an answered frame is not asked again
    assert again is not None and prompts == [] and again["calls"] == 0
    assert again["elements"] == first["elements"]
    _, prompts = _map(tmp_path, [_reply(FEED_BOX)] * 2, model="other")
    assert len(prompts) == 2
    _, prompts = _map(tmp_path, [_reply(FEED_BOX)] * 2, context="a tank game")
    assert len(prompts) == 2 and "a tank game" in prompts[0]


def test_map_hud_gives_up_after_failed_replies_and_keeps_no_map(tmp_path: Path) -> None:
    hud, prompts = _map(tmp_path, [None] * 3)
    assert hud is None and len(prompts) == feed_labels.STOP_AFTER


def test_map_hud_goes_on_with_the_frames_answered_when_the_calls_stop(tmp_path: Path) -> None:
    frames = _frames(4, 2, lambda _b, n: _feed_rows(n))
    _write_images(tmp_path / "frames", frames)
    n = [0]

    def ask(*_a: Any) -> dict[str, Any]:
        n[0] += 1
        if n[0] == 2:
            raise StopLabelling("the budget is spent")
        return _reply(FEED_BOX)

    hud = hud_map.map_hud([], frames, tmp_path / "frames", ask, "m")
    assert hud is not None and len(hud["frames"]) == 1
    assert hud["elements"][0]["seen"] == 1 and not hud["elements"][0]["confirmed"]  # weak


def test_the_cache_key_follows_model_prompt_frame_context_and_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = hud_map._key("m", "f1", None, "gaming")
    assert hud_map._key("m", "f1", None, "gaming") == base
    assert hud_map._key("m2", "f1", None, "gaming") != base
    assert hud_map._key("m", "f2", None, "gaming") != base
    assert hud_map._key("m", "f1", "a tank game", "gaming") != base
    assert hud_map._key("m", "f1", None, "action") != base
    monkeypatch.setattr(hud_map, "_PROMPT", hud_map._PROMPT + " Be brief.")
    assert hud_map._key("m", "f1", None, "gaming") != base


# ---------------------------------------------------------------- the shared vision calls


def _rater(**kw: Any) -> SimpleNamespace:
    warned: list[bool] = []
    base: dict[str, Any] = {
        "consent_needed": None,
        "available": lambda: True,
        "warn_consent": lambda: warned.append(True),
        "provider_name": "hosted",
        "host": "h",
        "model": "m",
        "unload": lambda: None,
        "describe": lambda *a, **k: {"elements": []},
        "timed_out": False,
        "warned": warned,
    }
    return SimpleNamespace(**(base | kw))


def test_the_map_and_the_labels_share_one_budget_when_it_is_carried_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(feed_labels.time, "monotonic", lambda: clock[0])
    calls = VisionCalls(_rater())
    calls.begin("game", 100)
    calls.ask("p", [], {})  # the budget starts with the first call
    clock[0] = 80.0
    calls.carry_over("game")
    calls.begin("game", 120)  # the labels go on with what is left (20 s)
    clock[0] = 90.0
    with pytest.raises(StopLabelling, match="100s for the vision model are spent"):
        calls.ask("p", [], {})
    calls.begin("game", 120)  # no carry-over: a fresh budget
    assert calls.ask("p", [], {}) == {"elements": []}


def test_a_carry_over_is_for_one_game_only(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(feed_labels.time, "monotonic", lambda: clock[0])
    calls = VisionCalls(_rater())
    calls.begin("a", 100)
    calls.ask("p", [], {})
    clock[0] = 95.0
    calls.carry_over("a")
    calls.begin("b", 100)  # another game: its own budget
    assert calls.ask("p", [], {}) == {"elements": []}


def test_a_timed_out_call_ends_the_calls_of_its_game() -> None:
    calls = VisionCalls(_rater(describe=lambda *a, **k: None, timed_out=True))
    calls.begin("g", 100)
    with pytest.raises(StopLabelling, match="ran out of its"):
        calls.ask("p", [], {})


def test_a_top_up_without_vision_consent_gives_one_info_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rater = _rater(consent_needed="vision-v2:host")
    mapper = hud_map.make_mapper(rater)
    with caplog.at_level(logging.INFO, logger="cutpoint.events.hud_map"):
        assert mapper.usable(top_up=True) is False
        assert mapper.usable(top_up=True) is False
    lines = [r for r in caplog.records if "Vision consent" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    assert rater.warned == []  # no vision_consent_needed warning for a top-up


def test_a_discovery_without_vision_consent_still_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rater = _rater(consent_needed="vision-v2:host")
    assert hud_map.make_mapper(rater).usable() is False
    assert rater.warned == [True]


def test_the_mapper_returns_nothing_without_frames_or_a_model(tmp_path: Path) -> None:
    sent: list[int] = []
    rater = _rater(describe=lambda *a, **k: sent.append(1))
    assert hud_map.make_mapper(rater)([], [], tmp_path / "frames") is None
    rater = _rater(available=lambda: False, describe=lambda *a, **k: sent.append(1))
    frames = _frames(2, 1, lambda _b, n: _feed_rows(n))
    assert hud_map.make_mapper(rater)([], frames, tmp_path / "frames") is None
    assert not sent


# ---------------------------------------------------------------- discover_regions


def _discover(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kw: Any) -> dict[str, Any]:
    from test_discover_feed_labels import FRAMES  # noqa: PLC0415 (synthetic two-style feed)

    monkeypatch.setattr(discover, "_ocr_frames", lambda *_a, **_k: FRAMES)
    monkeypatch.setattr(discover, "probe", lambda _p: None)
    return discover.discover_regions([], tmp_path / "disc", "zonewar", **kw)


class _Mapper:
    def __init__(self, make: Any) -> None:
        self.make, self.released, self.seen = make, 0, []

    def __call__(self, proposals: list[dict[str, Any]], frames: list[Any], folder: Path) -> Any:
        self.seen.append([p["name"] for p in proposals])
        return self.make(proposals)

    def release(self) -> None:
        self.released += 1


def test_discovery_without_a_map_is_what_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = _discover(tmp_path / "a", monkeypatch)
    for i, mapper in enumerate((None, _Mapper(lambda _p: None))):
        got = _discover(tmp_path / f"b{i}", monkeypatch, hud_mapper=mapper)
        assert got["regions"] == plain["regions"]
        assert got.get("feed") == plain.get("feed") and "hud_map" not in got["discovery"]
    assert mapper.released == 1  # asked once, freed once


def test_a_map_labels_proposals_adds_regions_and_spares_the_labeler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def make(proposals: list[dict[str, Any]]) -> dict[str, Any]:
        top = min((p for p in proposals if p["kind"] == "feed"), key=lambda p: p["y"])
        on = (top["x"], top["y"], top["w"], top["h"])
        return {"model": "m", "elements": [_el("kill_feed", on), _el("replay", (800, 60, 300, 60))]}

    asked: list[list[str]] = []

    def labeler(proposals: list[dict[str, Any]], frames: list[Any], folder: Path) -> dict[str, Any]:
        asked.append([p["name"] for p in proposals])
        return {}

    mapper = _Mapper(make)
    preset = _discover(tmp_path, monkeypatch, hud_mapper=mapper, labeler=labeler)
    settled = [r for r in preset["regions"] if r.get("label") == "kill_feed"]
    assert len(settled) == 1 and settled[0]["name"] not in asked[0]
    replay = [r for r in preset["regions"] if r["kind"] == "replay"]
    assert len(replay) == 1 and replay[0]["origin"] == "vision"
    assert preset["discovery"]["hud_map"]["model"] == "m"
    assert preset["feed"] == {"row_notices": True}  # a feed the map named has icon lines as kills
    assert mapper.released == 1


# ---------------------------------------------------------------- the top-up in _discover


class _TopMapper:
    """A mapper for ``pipeline._discover``'s top-up (the discovery itself is stubbed)."""

    model = "ollama@localhost:11434:m"

    def __init__(self, usable: bool = True, ready: str | None = None) -> None:
        self._usable, self._ready, self.released = usable, ready, 0

    def usable(self, *, top_up: bool = False) -> bool:
        return self._usable

    def ready(self) -> str | None:
        return self._ready

    def release(self) -> None:
        self.released += 1


LOADOUT = Region("loadout", 100, 800, 380, 140, "feed", gate="det", every=1)


def _installed(folder: Path) -> tuple[Path, dict[str, Any]]:
    data = {
        "game": "zonewar",
        "source": "discovered",
        "regions": [LOADOUT.key() | {"note": "calibrated by hand"}],
        "discovery": {"params": {}},
    }
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "zonewar.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, data


def _top_up(
    home: Path, monkeypatch: pytest.MonkeyPatch, mapper: Any, hud: dict[str, Any] | None
) -> tuple[dict[str, Any], list[dict[str, Any]], list[int]]:
    """Run ``pipeline._discover`` over one source of the game "zonewar"; returns its stats,
    the discoveries it ran (their keyword arguments) and the before-top-up hook's calls."""
    from cutpoint.events import pipeline  # noqa: PLC0415
    from cutpoint.events.killfeed import OcrParams  # noqa: PLC0415

    ran: list[dict[str, Any]] = []

    def fake(srcs: Any, out: Path, game: str, params: Any, on_frame: Any = None, **kw: Any):
        ran.append(kw)
        disc: dict[str, Any] = {"frames": [f["id"] for f in FEED_FRAMES]}
        return {"regions": [], "discovery": disc | ({"hud_map": hud} if hud else {})}

    monkeypatch.setattr(discover, "discover_regions", fake)
    monkeypatch.setattr(discover, "read_ocr_samples", lambda _p: FEED_FRAMES)
    monkeypatch.setattr(discover, "gameplay_frames", lambda recs, _params: recs)
    stopped: list[int] = []
    src = SimpleNamespace(path=Path("/synthetic/footage/session.mp4"))
    stats = pipeline._discover(
        home / "proj",
        [src],
        OcrParams(game="zonewar"),
        None,
        1,
        "user",
        hud_mapper=mapper,
        before_top_up=lambda: stopped.append(1),
    )
    return stats, ran, stopped


def test_a_discovered_preset_is_topped_up_once_and_keeps_what_it_changed(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cutpoint import paths  # noqa: PLC0415

    path, before = _installed(paths.presets_dir())
    mapper = _TopMapper()
    stats, ran, stopped = _top_up(home, monkeypatch, mapper, _hud(_el("kill_feed", FEED)))
    assert stopped == [1] and len(ran) == 1  # the background OCR stopped first
    assert "labeler" not in ran[0] and ran[0]["hud_mapper"] is mapper
    assert mapper.released == 1
    data = json.loads(path.read_text(encoding="utf-8"))
    regs = {r["name"]: r for r in data["regions"]}
    assert regs["vision_feed"]["kind"] == "feed" and regs["vision_feed"]["origin"] == "vision"
    # the loadout box calibrated by hand is read as a notice, not deleted
    assert regs["loadout"]["kind"] == "notice" and regs["loadout"]["was"] == "feed"
    assert regs["loadout"]["note"] == "calibrated by hand"
    top = data["discovery"]["hud_map"]["top_up"]
    assert top["originals"] == before["regions"] and top["replaced"] == ["loadout"]
    assert stats["zonewar"]["topped_up"]["added"] == ["vision_feed"]
    # once: the next analysis runs no discovery for it
    _, ran, stopped = _top_up(home, monkeypatch, _TopMapper(), _hud(_el("kill_feed", FEED)))
    assert ran == [] and stopped == []


def test_no_map_leaves_the_preset_and_is_not_tried_again_until_it_changes(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cutpoint import paths  # noqa: PLC0415

    path, _ = _installed(paths.presets_dir())
    text = path.read_text(encoding="utf-8")
    _, ran, _ = _top_up(home, monkeypatch, _TopMapper(), None)
    assert len(ran) == 1 and path.read_text(encoding="utf-8") == text
    assert (paths.cache_dir() / hud_map.TRIED_FILE).is_file()
    _, ran, _ = _top_up(home, monkeypatch, _TopMapper(), None)
    assert ran == []  # recorded: no multi-minute discovery again
    path.write_text(text + "\n", encoding="utf-8")  # the user edits the preset
    _, ran, _ = _top_up(home, monkeypatch, _TopMapper(), None)
    assert len(ran) == 1


def test_a_model_that_cannot_answer_runs_no_discovery(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cutpoint import paths  # noqa: PLC0415

    path, _ = _installed(paths.presets_dir())
    text = path.read_text(encoding="utf-8")
    mapper = _TopMapper(ready="m runs on the CPU only, too slow to label")
    _, ran, _ = _top_up(home, monkeypatch, mapper, _hud(_el("kill_feed", FEED)))
    assert ran == [] and mapper.released == 1 and path.read_text(encoding="utf-8") == text
    record = json.loads((paths.cache_dir() / hud_map.TRIED_FILE).read_text(encoding="utf-8"))
    assert "CPU only" in next(iter(record.values()))["why"]
    other = _TopMapper()  # another model is tried
    other.model = "ollama@localhost:11434:other"
    _, ran, _ = _top_up(home, monkeypatch, other, _hud(_el("kill_feed", FEED)))
    assert len(ran) == 1


def test_project_presets_and_runs_without_consent_are_never_topped_up(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cutpoint import paths  # noqa: PLC0415

    path, _ = _installed(home / "proj" / "hud_presets")
    text = path.read_text(encoding="utf-8")
    _, ran, stopped = _top_up(home, monkeypatch, _TopMapper(), _hud(_el("kill_feed", FEED)))
    assert ran == [] and stopped == [] and path.read_text(encoding="utf-8") == text
    path.unlink()
    path, _ = _installed(paths.presets_dir())
    text = path.read_text(encoding="utf-8")
    mapper = _TopMapper(usable=False)  # no Vision consent, or no model
    _, ran, _ = _top_up(home, monkeypatch, mapper, _hud(_el("kill_feed", FEED)))
    assert ran == [] and path.read_text(encoding="utf-8") == text


def test_parse_reads_a_whole_reply_on_one_scale() -> None:
    # one box in pixels makes every box of the reply pixels
    reply = {
        "elements": [
            {"kind": "score", "box": [500, 10, 600, 50]},
            {"kind": "kill_feed", "box": [1000, 60, 1270, 200]},
        ]
    }
    got = hud_map.parse(reply, (1280, 720))
    assert got is not None and got[0]["box"] == [750, 15, 150, 60]


def test_a_carry_over_ends_when_the_calls_are_released(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(feed_labels.time, "monotonic", lambda: clock[0])
    calls = VisionCalls(_rater())
    calls.begin("g", 100)
    calls.ask("p", [], {})
    clock[0] = 95.0
    calls.carry_over("g")
    calls.release()  # the labels had nothing to ask
    calls.begin("g", 100)  # a later discovery of the game: its own budget
    assert calls.ask("p", [], {}) == {"elements": []}


def test_ready_loads_the_model_and_refuses_one_on_the_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[float] = []
    rater = _rater(
        provider_name="ollama",
        url="http://localhost:11434",
        used=False,
        load=lambda timeout, num_ctx=None: loads.append(timeout) or True,
        on_gpu=lambda: False,
    )
    calls = VisionCalls(rater)
    assert calls.ready() is False and "CPU only" in calls.stop[0] and len(loads) == 1
    mapper = hud_map.make_mapper(rater, calls=calls)
    assert "CPU only" in (mapper.ready() or "")


def test_judge_learns_the_replay_marker_from_the_ocr_and_skips_other_text() -> None:
    zone = (800, 60, 300, 60)

    def reads(b: int, _n: int) -> list[dict[str, Any]]:
        text = {1: "REPLAY", 3: "DREPLAY", 5: "REPLAY", 6: "FAVELA", 7: "FAVELA"}.get(b, "")
        return [_box(text, 820, 70, 200, 30)] if text else []

    rec = hud_map.judge(_group("replay", zone, hints=("replay",)), _frames(10, 2, reads), 2)
    assert rec["confirmed"] and rec["marker"] == "replay"
    region = hud_map.new_region(rec, "replay")
    assert region.marker == "replay" and region.key()["marker"] == "replay"
    # a different name each time (a killcam card): no text read again and again
    names = _frames(10, 2, lambda b, _n: [_box(NAMES[b], 820, 70, 200, 30)] if b < 3 else [])
    rec = hud_map.judge(_group("replay", zone, hints=("replay",)), names, 2)
    assert not rec["confirmed"] and "marker" in rec["why"]
    # the model read no marker text: OCR alone does not pick one
    rec = hud_map.judge(_group("replay", zone), _frames(10, 2, reads), 2)
    assert not rec["confirmed"] and "model read no marker" in rec["why"]


def test_the_model_hint_picks_the_marker_among_texts_read_as_often() -> None:
    zone = (800, 60, 300, 60)

    def reads(b: int, _n: int) -> list[dict[str, Any]]:
        text = {1: "FAVELA", 2: "FAVELA", 5: "REPLAY", 6: "REPLAY"}.get(b, "")
        return [_box(text, 820, 70, 200, 30)] if text else []

    frames = _frames(10, 2, reads)
    assert hud_map.replay_marker(frames, zone) == ""  # no hint: OCR alone picks none
    assert hud_map.replay_marker(frames, zone, ["REPLAY"]) == "replay"
    reply = {"elements": [{"kind": "replay", "box": [400, 50, 600, 100], "text_seen": "Replay 2"}]}
    els = hud_map.parse(reply, (1280, 720))
    assert els is not None and els[0]["hint"] == "replay"
    g = hud_map.group([els, els])[0]
    assert g["hints"] == ["replay", "replay"]


def test_a_replay_box_over_the_feed_learns_no_marker_from_its_rows() -> None:
    """#1069 review: the streamer's own kill line, read in every burst, sits in the part of
    the replay box that overlaps the feed. It is no marker, with a hint or without one, and the
    replay rectangle leaves the feed out."""
    zone = (1400, 20, 480, 200)  # reaches into FEED (1500, 100, 380, 140)

    def reads(b: int, _n: int) -> list[dict[str, Any]]:
        own = [_box("Alder killed Playmaker", 1500, 100, 380, 28)]
        return own + ([_box("REPLAY", 1450, 30, 160, 40)] if b in (2, 3, 6) else [])

    frames = _frames(8, 2, reads)
    assert hud_map.replay_marker(frames, zone, feeds=[FEED]) == ""  # no hint: none
    assert hud_map.replay_marker(frames, zone, ["playmaker"], feeds=[FEED]) == ""
    rec = hud_map.judge(_group("replay", zone, hints=("replay",)), frames, 2, [FEED])
    assert rec["confirmed"] and rec["marker"] == "replay"
    x, y, w, h = rec["rect"]
    assert y + h <= FEED[1] or x + w <= FEED[0]  # beside the feed, not on it
    rec = hud_map.judge(_group("replay", zone), frames, 2, [FEED])
    assert not rec["confirmed"] and "model read no marker" in rec["why"]


def test_a_replay_marker_on_a_banner_is_left_out() -> None:
    banner = (700, 300, 520, 120)
    hud = {
        "model": "ollama@h:m",
        "elements": [_el("objective_banner", banner), _el("replay", (710, 305, 500, 110))],
    }
    labels, regions, _ = hud_map.settle(hud, [], set())
    assert [r.kind for r in regions] == ["banner"]
    hud = {"model": "ollama@h:m", "elements": [_el("replay", (710, 305, 500, 110))]}
    labels, regions, _ = hud_map.settle(hud, [_prop("notice", "notice", *banner)], {"notice"})
    assert not regions and not labels
    data = {"game": "g", "regions": [Region("notice", *banner, "notice").key()]}
    new, changes = hud_map.top_up(data, hud, [])
    assert changes["added"] == [] and len(new["regions"]) == 1
