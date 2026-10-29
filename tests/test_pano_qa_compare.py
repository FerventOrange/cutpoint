"""``cutpoint 360 qa <dir> --compare RENDERED PATHS`` (#11): edge-cut gap causes on
hand-built qa.json pairs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from conftest import run_json, warning_codes
from cutpoint.pano import qa_compare as QC
from cutpoint.pano.qa import QAThresholds

TH = QAThresholds()
LEFT = [0.0, 0.3, 0.1, 0.8]  # cut at the left edge
RIGHT = [0.9, 0.3, 1.0, 0.8]


def fr(t: float, flags=(), boxes=(), own=(), view_dt: float | None = 0.1, slivers=()) -> dict:
    """A qa.json frame: ``boxes`` (non-self) and ``own`` (self) as ``[x1, y1, x2, y2]``,
    ``slivers`` as ``[x1, y1, x2, y2, is_self]``."""
    rows = [[0.99, *b, False] for b in boxes] + [[0.99, *b, True] for b in own]
    return {
        "t": t,
        "target": "friend",
        "flags": list(flags),
        "subj": list(boxes[0]) if boxes else None,
        "self": [list(b) for b in own],
        "boxes": rows,
        "view_dt": view_dt,
        "slivers": [list(b) for b in slivers],
    }


def report(mode: str, source_dir: Path, clips: list[dict]) -> dict:
    return {
        "version": 1,
        "source_dir": str(source_dir),
        "mode": mode,
        "fps": 2.0,
        "thresholds": asdict(TH),
        "clips": clips,
        "edit": {},
    }


def clip(file: str, frames: list[dict], segment=(10.0, 15.0)) -> dict:
    return {"file": file, "offset": 10.0, "segment": list(segment), "frames": frames}


def pair(d: Path) -> tuple[dict, dict]:
    """One clip with a frame of every cause (offset 10 s, segment 10-15 s)."""
    pairs = [[0.40, 0.3, 0.60, 0.8], [0.45, 0.3, 0.65, 0.8]]  # rendered 0.05 left of projected
    rend = [
        fr(0.0, ["edge_cut"], [LEFT]),  # both flag it
        fr(0.5, ["edge_cut"], [LEFT]),  # the projected wearer is there
        fr(1.0, ["edge_cut"], [RIGHT]),  # nothing projected
        fr(1.5, ["edge_cut"], [RIGHT]),  # nothing projected, from a view frame 0.9 s away
        *(fr(2.0 + k / 2, [], [pairs[0]]) for k in range(5)),  # the median offset
        fr(4.5, ["edge_cut"], [[0.0, 0.3, 0.15, 0.8]]),  # 0.04 inside once shifted
        fr(5.0, ["edge_cut"], [[0.0, 0.3, 0.3, 0.8]]),  # projected well inside
        fr(5.5, ["edge_cut"], [LEFT]),  # source 15.5 s: past the segment, nothing projected
        fr(6.0, [], [pairs[0]]),
    ]
    paths = [
        fr(0.0, ["edge_cut"], [LEFT]),
        fr(0.5, [], [], [[0.0, 0.3, 0.12, 0.8]]),
        fr(1.0),
        fr(1.5, view_dt=0.9),
        *(fr(2.0 + k / 2, [], [pairs[1]]) for k in range(5)),
        fr(4.5, [], [[0.04, 0.3, 0.19, 0.8]]),
        fr(5.0, [], [[0.2, 0.3, 0.4, 0.8]]),
        fr(5.5),
        fr(6.0, ["edge_cut"], [RIGHT]),  # only the geometric QA flags it
    ]
    return (
        report("rendered", d, [clip("001_a.mp4", rend), clip("009_x.mp4", [])]),
        report("paths", d, [clip("001_a.json", paths)]),
    )


def write_pair(tmp_path: Path, d: Path) -> tuple[Path, Path]:
    rend, paths = pair(d)
    a = d.with_name(d.name + "_qa") / "qa.json"
    b = d.with_name(d.name + "_paths_qa") / "qa.json"
    for f, rep in ((a, rend), (b, paths)):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rep), encoding="utf-8")
    return a, b


def test_every_cause_is_attributed(tmp_path: Path) -> None:
    d = tmp_path / "mix"
    a, b = write_pair(tmp_path, d)
    rep = QC.run_compare(d, b, a)  # either order
    assert rep["out_file"] == str(a.parent / "edge_gap.json")
    saved = json.loads((a.parent / "edge_gap.json").read_text(encoding="utf-8"))
    (c,) = saved["clips"]
    assert c["file"] == "001_a.mp4"  # joined by stem: the paths QA names the .json
    assert c["median_dx"] == pytest.approx(-0.05)
    assert {f["t"]: f["cause"] for f in c["rendered_only_frames"]} == {
        0.5: "wearer_overlap",
        1.0: "no_projected_box",
        1.5: "stale_view",
        4.5: "yaw_offset",
        5.0: "other",
        5.5: "no_projected_box",
    }
    inside = {f["t"]: f["in_selected"] for f in c["rendered_only_frames"]}
    assert inside[5.5] is False and inside[5.0] is True and c["outside_selected"] == 1
    assert (c["rendered_edge"], c["paths_edge"], c["both"]) == (7, 2, 1)
    assert (c["rendered_only"], c["paths_only"], c["frames"]) == (6, 1, 13)
    e = saved["edit"]
    assert e["causes"] == {**dict.fromkeys(QC.CAUSES, 1), "unpaired": 0, "no_projected_box": 2}
    assert e["outside_selected"] == 1
    assert e["cause_share"]["stale_view"] == pytest.approx(0.167)
    assert saved["unmatched"] == {"rendered": ["009_x.mp4"], "paths": []}
    assert rep["source_mismatch"] == []


def test_stale_without_any_view_frame_and_no_median_with_few_pairs() -> None:
    p = fr(0.0, view_dt=None)  # no view frame within the hold
    assert QC.cause_of(fr(0.0, ["edge_cut"], [LEFT]), p, TH, None, None, None) == "stale_view"
    pairs = QC.join_frames([fr(0.0, [], [LEFT])], [fr(0.0, [], [LEFT])], 2.0)
    assert QC.median_dx(pairs, TH) is None
    # an old paths report without view_dt: never stale
    old = {k: v for k, v in fr(0.0).items() if k != "view_dt"}
    assert QC.cause_of(fr(0.0, ["edge_cut"], [LEFT]), old, TH, None, None, None) == (
        "no_projected_box"
    )


def test_join_pairs_nearest_times_within_half_a_frame() -> None:
    rend = [fr(0.0), fr(0.5), fr(1.0)]
    paths = [fr(0.01), fr(0.52), fr(2.0)]
    got = [(r and r["t"], p and p["t"]) for r, p in QC.join_frames(rend, paths, 2.0)]
    assert got == [(0.0, 0.01), (0.5, 0.52), (1.0, None), (None, 2.0)]


def test_two_reports_of_one_mode_are_rejected(tmp_path: Path) -> None:
    d = tmp_path / "mix"
    a, _ = write_pair(tmp_path, d)
    with pytest.raises(ValueError, match="one rendered and one --paths"):
        QC.run_compare(d, a, a)
    with pytest.raises(FileNotFoundError):
        QC.run_compare(d, a, tmp_path / "nope.json")
    assert not (a.parent / "edge_gap.json").exists()


def test_360_qa_compare_json(home, capsys) -> None:
    d = home / "projects" / "trip" / "clips_mix"
    d.mkdir(parents=True)
    a, b = write_pair(home, d)
    rel = [str(p.relative_to(home)).replace("\\", "/") for p in (a, b)]
    code, doc = run_json(capsys, ["360", "qa", "projects/trip/clips_mix", "--compare", *rel])
    assert code == 0 and doc["ok"] is True and doc["command"] == "360 qa"
    assert doc["project"] == str(home / "projects" / "trip")
    out = doc["outputs"]
    assert out["edge_gap"] == str(a.parent / "edge_gap.json")
    assert (out["mode"], out["rendered"], out["paths"]) == ("compare", str(a), str(b))
    assert (out["count"], out["rendered_only"]) == (1, 6)
    assert out["causes"] == {**dict.fromkeys(QC.CAUSES, 1), "unpaired": 0, "no_projected_box": 2}
    assert out["other_detail"] == {**dict.fromkeys(QC.OTHER_DETAILS, 0), "paths_inside": 1}
    assert out["clips"][0]["other_detail"]["paths_inside"] == 1
    assert out["clips"][0]["file"] == str(d / "001_a.mp4")
    # the reports record no render mode or self method: all unknown
    counts = {"clips": 1, "frames": 13, "rendered_edge": 7, "paths_edge": 2, "rendered_only": 6}
    assert out["by_mode"] == out["by_self_method"] == {"unknown": counts}
    assert warning_codes(doc) == ["qa_clips_unmatched"]
    # --paths with --compare: a bad option, nothing written
    (a.parent / "edge_gap.json").unlink()
    code, doc = run_json(
        capsys, ["360", "qa", "projects/trip/clips_mix", "--paths", "--compare", *rel]
    )
    assert code == 1 and doc["error"]["code"] == "bad_args"
    assert not (a.parent / "edge_gap.json").exists()


def test_reports_of_another_folder_warn(home, capsys) -> None:
    d = home / "projects" / "trip" / "clips_mix"
    d.mkdir(parents=True)
    a, b = write_pair(home, home / "projects" / "trip" / "other")
    code, doc = run_json(capsys, ["360", "qa", str(d), "--compare", str(a), str(b)])
    assert code == 0
    assert "qa_source_mismatch" in warning_codes(doc)


def test_a_wearer_seen_only_as_a_sliver_is_wearer_overlap() -> None:
    """The geometric QA dropped the wearer as mostly outside the view (F2): its sliver
    still explains the rendered cut; another person's sliver is a projected box."""
    r = fr(0.0, ["edge_cut"], [LEFT])
    wearer = fr(0.0, slivers=[[0.0, 0.3, 0.11, 0.8, True]])
    assert QC.cause_of(r, wearer, TH, None, None, None) == "wearer_overlap"
    other = fr(0.0, slivers=[[0.0, 0.3, 0.11, 0.8, False]])
    assert QC.cause_of(r, other, TH, None, None, None) == "other"  # matched, not no box
    # the sliver would be the paths subject: the paths QA counts it (a hand-built pair)
    assert QC.explain(r, other, TH, None, None, None)["other_detail"] == "unexplained"
    assert QC.cause_of(r, fr(0.0), TH, None, None, None) == "no_projected_box"


def test_frames_without_a_paths_partner_are_unpaired_and_fps_mismatch_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    d = tmp_path / "mix"
    rend = [fr(0.0, ["edge_cut"], [LEFT]), fr(0.5, ["edge_cut"], [LEFT])]
    paths = [fr(0.0)]  # checked at 1 fps: nothing at 0.5 s
    a, b = tmp_path / "r.json", tmp_path / "p.json"
    a.write_text(json.dumps(report("rendered", d, [clip("001_a.mp4", rend)])), encoding="utf-8")
    p_rep = {**report("paths", d, [clip("001_a.mp4", paths)]), "fps": 1.0}
    b.write_text(json.dumps(p_rep), encoding="utf-8")
    with caplog.at_level("WARNING"):
        rep = QC.run_compare(d, a, b)
    assert "checked at 2 and 1 fps" in caplog.text
    assert rep["fps_mismatch"] is True
    (c,) = rep["clips"]
    assert [f["cause"] for f in c["rendered_only_frames"]] == ["no_projected_box", "unpaired"]
    assert rep["edit"]["causes"]["unpaired"] == 1
    (tmp_path / "edge_gap.json").unlink()


def test_a_tiny_far_person_at_the_edge_does_not_decide_the_cause() -> None:
    """Only the boxes frame_flags counts for the cut are matched (F4): a far person at
    the right edge (not the subject, under edge_min_area) is no cut box."""
    tiny = [0.97, 0.4, 1.0, 0.48]
    r = fr(0.0, ["edge_cut"], [LEFT, tiny])
    p = fr(0.0, own=[[0.96, 0.4, 1.0, 0.5]])  # the projected wearer covers the tiny one
    assert QC.cut_boxes(r, TH).tolist() == [LEFT]
    assert QC.cause_of(r, p, TH, None, None, None) == "no_projected_box"


BIG_LEFT = [0.0, 0.3, 0.25, 0.8]  # counted by its area (0.125)
BIG_INSIDE = [0.4, 0.1, 0.8, 0.9]  # the paths subject, well inside
TOP = [0.4, 0.0, 0.6, 0.5]  # cut at the top
# sub-cause -> (rendered boxes, paths frame keyword arguments)
OTHER_CASES = {
    "paths_top_edge": ([TOP], {"boxes": [[0.4, 0.0, 0.6, 0.25]]}),  # too short for the top
    "paths_inside_margin": ([BIG_LEFT], {"boxes": [[0.02, 0.3, 0.27, 0.8]]}),
    # a bottom-cut sliver as wide as the cut box, shifted right by 0.155 (Windows' evidence)
    "position_mismatch": (
        [[0.0, 0.2, 0.2, 0.9]],
        {"slivers": [[0.155, 0.2, 0.355, 0.5, False]]},
    ),
    "rendered_box_larger": ([[0.0, 0.2, 0.3, 0.9]], {"boxes": [[0.1, 0.4, 0.2, 0.7]]}),
    "paths_inside": ([[0.0, 0.3, 0.3, 0.8]], {"boxes": [[0.2, 0.3, 0.4, 0.8]]}),
    "paths_below_size": ([BIG_LEFT], {"boxes": [[0.0, 0.5, 0.2, 0.55]]}),  # under subject_min_h
    # the projected box shares half its area with a paths self box, the cut box 0.4 of its own
    "paths_self": ([LEFT], {"boxes": [[0.0, 0.3, 0.12, 0.8]], "own": [[0.06, 0.3, 0.3, 0.8]]}),
    "paths_below_area": ([BIG_LEFT], {"boxes": [[0.0, 0.3, 0.15, 0.8], BIG_INSIDE]}),
    "paths_not_subject": ([LEFT], {"boxes": [LEFT, BIG_INSIDE]}),  # LEFT: the rendered subject
}


@pytest.mark.parametrize("detail", list(OTHER_CASES))
def test_each_other_detail(detail: str) -> None:
    rend_boxes, kw = OTHER_CASES[detail]
    r = fr(0.0, ["edge_cut"], rend_boxes)
    assert len(QC.cut_boxes(r, TH)) == 1
    got = QC.explain(r, fr(0.0, **kw), TH, None, None, None)
    assert (got["cause"], got["other_detail"]) == ("other", detail)


def test_the_paths_subject_under_size_min_h_and_a_top_sliver() -> None:
    # the rendered subject (0.5 tall) is the only projected box but 0.15 tall
    r = fr(0.0, ["edge_cut"], [LEFT])
    got = QC.explain(r, fr(0.0, [], [[0.0, 0.3, 0.1, 0.45]]), TH, None, None, None)
    assert got["other_detail"] == "paths_below_size"
    # a large sliver at the top, smaller than a kept box: the sliver rule counts side edges
    p = fr(0.0, [], [[0.65, 0.1, 0.95, 0.95]], slivers=[[0.4, 0.0, 0.7, 0.5, False]])
    got = QC.explain(fr(0.0, ["edge_cut"], [TOP]), p, TH, None, None, None)
    assert (got["other_detail"], got["evidence"]["paths_kind"]) == ("paths_top_edge", "sliver")


def test_evidence_names_the_boxes_and_the_gap() -> None:
    rend_boxes, kw = OTHER_CASES["paths_inside_margin"]
    got = QC.explain(fr(0.0, ["edge_cut"], rend_boxes), fr(0.0, **kw), TH, None, None, None)
    assert got["evidence"] == {
        "rendered_box": BIG_LEFT,
        "rendered_area": 0.125,
        "edge": "left",
        "paths_box": [0.02, 0.3, 0.27, 0.8],
        "paths_kind": "box",
        "paths_h": 0.5,
        "paths_area": 0.125,
        "edge_gap": 0.02,
    }
    # the wearer's box for wearer_overlap; nothing projected: no paths box
    r = fr(0.0, ["edge_cut"], [RIGHT])
    w = QC.explain(r, fr(0.0, own=[RIGHT]), TH, None, None, None)
    assert (w["cause"], w["other_detail"]) == ("wearer_overlap", None)
    assert (w["evidence"]["paths_kind"], w["evidence"]["edge_gap"]) == ("wearer", 0.0)
    n = QC.explain(r, None, TH, None, None, None)
    assert n["cause"] == "unpaired" and n["evidence"]["edge"] == "right"
    assert n["evidence"]["paths_box"] is None


def test_old_frames_without_slivers_or_view_dt_get_a_detail() -> None:
    rend_boxes, kw = OTHER_CASES["paths_not_subject"]
    old = {k: v for k, v in fr(0.0, **kw).items() if k not in ("slivers", "view_dt", "self")}
    got = QC.explain(fr(0.0, ["edge_cut"], rend_boxes), old, TH, None, None, None)
    assert (got["cause"], got["other_detail"]) == ("other", "paths_not_subject")


def test_other_detail_totals_match_other(tmp_path: Path) -> None:
    d = tmp_path / "mix"
    rend, paths = [], []
    for k, (rb, kw) in enumerate(OTHER_CASES.values()):
        rend.append(fr(k / 2, ["edge_cut"], rb))
        paths.append(fr(k / 2, **kw))
    n = len(rend)
    rend.append(fr(n / 2, ["edge_cut"], [RIGHT]))  # no_projected_box
    paths.append(fr(n / 2))
    a, b = tmp_path / "r.json", tmp_path / "p.json"
    seg = (10.0, 20.0)
    a.write_text(
        json.dumps(report("rendered", d, [clip("001_a.mp4", rend, seg)])), encoding="utf-8"
    )
    b.write_text(json.dumps(report("paths", d, [clip("001_a.mp4", paths, seg)])), encoding="utf-8")
    rep = QC.run_compare(d, a, b)
    saved = json.loads((tmp_path / "edge_gap.json").read_text(encoding="utf-8"))
    assert saved["version"] == QC.GAP_VERSION
    (c,) = saved["clips"]
    assert [f["other_detail"] for f in c["rendered_only_frames"]] == [*OTHER_CASES, None]
    assert sum(c["other_detail"].values()) == c["causes"]["other"] == n
    e = saved["edit"]
    assert e["causes"] == {**dict.fromkeys(QC.CAUSES, 0), "other": n, "no_projected_box": 1}
    assert sum(e["other_detail"].values()) == e["causes"]["other"]
    assert e["cause_share"]["other"] == pytest.approx(round(n / (n + 1), 3))
    assert sum(e["other_detail_share"].values()) == pytest.approx(
        e["cause_share"]["other"], abs=0.005
    )
    assert rep["edit"]["other_detail"] == e["other_detail"]
    (tmp_path / "edge_gap.json").unlink()


def test_a_side_sliver_over_edge_min_area_is_counted_even_over_a_self_box() -> None:
    """F1: sliver_edge_frames has no self-overlap exclusion, so this is no paths_self."""
    r = fr(0.0, ["edge_cut"], [BIG_LEFT])
    p = fr(0.0, own=[[0.3, 0.3, 0.5, 0.8]], slivers=[[0.0, 0.3, 0.5, 0.8, False]])
    got = QC.explain(r, p, TH, None, None, None)
    assert (got["cause"], got["other_detail"]) == ("other", "unexplained")


def test_the_paths_subject_under_size_min_h_beats_below_area() -> None:
    """F2: the only (largest) projected box, 0.15 tall, against a cut box of 0.125 area."""
    r = fr(0.0, ["edge_cut"], [BIG_LEFT])
    got = QC.explain(r, fr(0.0, [], [[0.0, 0.3, 0.2, 0.45]]), TH, None, None, None)
    assert (got["cause"], got["other_detail"]) == ("other", "paths_below_size")


@pytest.mark.parametrize(
    ("sliver", "detail"),
    [
        ([0.155, 0.2, 0.355, 0.5, False], "position_mismatch"),  # same width, 0.155 off
        ([0.03, 0.2, 0.23, 0.5, False], "rendered_box_larger"),  # same width, 0.03 off
        ([0.1, 0.2, 0.2, 0.5, False], "rendered_box_larger"),  # half the width
    ],
)
def test_a_sliver_of_the_same_width_off_to_the_side_is_a_position_mismatch(sliver, detail):
    r = fr(0.0, ["edge_cut"], [[0.0, 0.2, 0.2, 0.9]])
    got = QC.explain(r, fr(0.0, slivers=[sliver]), TH, None, None, None)
    assert (got["cause"], got["other_detail"], got["evidence"]["paths_kind"]) == (
        "other",
        detail,
        "sliver",
    )
    # a projected box (not a sliver) of the same width and offset stays rendered_box_larger
    box = fr(0.0, [], [sliver[:4]])
    assert QC.explain(r, box, TH, None, None, None)["other_detail"] == "rendered_box_larger"


def _write(tmp_path: Path, d: Path, rend: dict, paths: dict) -> tuple[Path, Path]:
    a, b = tmp_path / "r.json", tmp_path / "p.json"
    a.write_text(json.dumps(rend), encoding="utf-8")
    b.write_text(json.dumps(paths), encoding="utf-8")
    return a, b


def test_a_tail_frame_gets_its_cause_and_in_selected(tmp_path: Path) -> None:
    """#11: a clip's tail (source 16-20 s) is planned and checked like the moment, so its
    frames get the full cause chain; ``in_selected`` says they lie outside the pick."""
    d = tmp_path / "clips"
    pairs = [[0.40, 0.3, 0.60, 0.8], [0.45, 0.3, 0.65, 0.8]]  # rendered 0.05 left
    rend = [fr(2.0 + k / 2, [], [pairs[0]]) for k in range(5)]
    paths = [fr(2.0 + k / 2, [], [pairs[1]]) for k in range(5)]
    rend.append(fr(7.0, ["edge_cut"], [[0.0, 0.3, 0.15, 0.8]]))  # source 17 s: the tail
    paths.append(fr(7.0, [], [[0.04, 0.3, 0.19, 0.8]]))  # at the edge once shifted
    rc, pc = clip("001_a.mp4", rend, (10.0, 20.0)), clip("001_a.mp4", paths, (10.0, 20.0))
    for c in (rc, pc):
        c["selected"] = [10.0, 16.0]
    a, b = _write(tmp_path, d, report("rendered", d, [rc]), report("paths", d, [pc]))
    rep = QC.run_compare(d, a, b)
    (f,) = rep["clips"][0]["rendered_only_frames"]
    assert (f["cause"], f["in_selected"]) == ("yaw_offset", False)
    assert rep["clips"][0]["outside_selected"] == rep["edit"]["outside_selected"] == 1
    for c in (rc, pc):
        del c["selected"]  # an older report: the segment is the moment
    a, b = _write(tmp_path, d, report("rendered", d, [rc]), report("paths", d, [pc]))
    (f,) = QC.run_compare(d, a, b)["clips"][0]["rendered_only_frames"]
    assert (f["cause"], f["in_selected"]) == ("yaw_offset", True)
    (tmp_path / "edge_gap.json").unlink()


def test_edge_gap_counts_frames_by_mode_and_self_method(tmp_path: Path) -> None:
    d = tmp_path / "clips"
    hit, miss = fr(0.0, ["edge_cut"], [RIGHT]), fr(0.0)
    r1 = {**clip("001_a.mp4", [hit]), "mode": "sdk", "self_method": "clip"}
    r2 = {**clip("002_b.mp4", [hit]), "mode": "dfisheye", "self_method": "cone+clip"}
    r3 = clip("003_c.mp4", [miss])  # an older qa.json record: no mode, no self_method
    rend = report("rendered", d, [r1, r2, r3])
    paths = report("paths", d, [clip(f, [miss]) for f in ("001_a", "002_b", "003_c")])
    a, b = _write(tmp_path, d, rend, paths)
    QC.run_compare(d, a, b)
    saved = json.loads((tmp_path / "edge_gap.json").read_text(encoding="utf-8"))
    one = {"clips": 1, "frames": 1, "rendered_edge": 1, "paths_edge": 0, "rendered_only": 1}
    none = {**one, "rendered_edge": 0, "rendered_only": 0}
    e = saved["edit"]
    assert e["by_mode"] == {"dfisheye": one, "sdk": one, "unknown": none}
    assert e["by_self_method"] == {"clip": one, "cone+clip": one, "unknown": none}
    assert [c["mode"] for c in saved["clips"]] == ["sdk", "dfisheye", None]
    assert {"causes", "cause_share", "other_detail", "other_detail_share"} <= set(e)
    (tmp_path / "edge_gap.json").unlink()


def test_size_ratio_of_matched_boxes(tmp_path: Path) -> None:
    """#11 phase 2a: rendered boxes 1.2x the projected ones (after the median offset)."""
    d = tmp_path / "clips"
    proj = [0.40, 0.30, 0.50, 0.60]  # 0.1 x 0.3, centre (0.45, 0.45)
    big = [0.39, 0.27, 0.51, 0.63]  # 1.2x about the same centre ...
    shifted = [b + (0.03 if k % 2 == 0 else 0.0) for k, b in enumerate(big)]  # ... 0.03 right
    rend = [fr(k / 2, [], [shifted]) for k in range(6)]
    paths = [fr(k / 2, [], [proj]) for k in range(6)]
    few = [fr(0.0, [], [shifted])]
    a, b = _write(
        tmp_path,
        d,
        report("rendered", d, [clip("001_a.mp4", rend), clip("002_b.mp4", few)]),
        report("paths", d, [clip("001_a.json", paths), clip("002_b.json", paths[:1])]),
    )
    QC.run_compare(d, a, b)
    saved = json.loads((tmp_path / "edge_gap.json").read_text(encoding="utf-8"))
    one, two = saved["clips"]
    assert one["median_dx"] == pytest.approx(0.03)
    assert one["size_ratio"] == {"width": 1.2, "height": 1.2, "pairs": 6}
    assert two["size_ratio"] == {"width": None, "height": None, "pairs": 1}  # too few
    assert saved["edit"]["size_ratio"] == {"width": 1.2, "height": 1.2, "pairs": 7}
    assert saved["version"] == QC.GAP_VERSION == 4
    (tmp_path / "edge_gap.json").unlink()
