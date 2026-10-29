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
        fr(5.5, ["edge_cut"], [LEFT]),  # source 15.5 s: past the segment
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
        5.5: "outside_segment",
    }
    assert (c["rendered_edge"], c["paths_edge"], c["both"]) == (7, 2, 1)
    assert (c["rendered_only"], c["paths_only"], c["frames"]) == (6, 1, 13)
    e = saved["edit"]
    assert e["causes"] == {**dict.fromkeys(QC.CAUSES, 1), "unpaired": 0}
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
    assert out["causes"] == {**dict.fromkeys(QC.CAUSES, 1), "unpaired": 0}
    assert out["clips"][0]["file"] == str(d / "001_a.mp4")
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
