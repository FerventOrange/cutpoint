"""``scripts/moment_eval.py``, the ranking benchmark: the overlap rule, the budget cut, the
seeded control and its percentile, holdout handling, ``--compare`` and the command line. Labels
and highlights are synthetic."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from conftest import load_script

pytestmark = pytest.mark.script
me = load_script("moment_eval")

SRC = "D:\\Videos\\Clip-A.mp4"


def cand(i: int, start: float, end: float, score: float, source: str = SRC) -> dict:
    return {
        "id": f"h{i:03d}",
        "source": source,
        "source_index": 0,
        "start": start,
        "end": end,
        "duration": end - start,
        "score": score,
        "label": "x",
    }


def write_project(tmp_path: Path, cands: list[dict], name: str = "proj") -> Path:
    project = tmp_path / name
    project.mkdir()
    data = {
        "version": 1,
        "style": "gaming",
        "settings": {"target_total": 30.0},
        "sources": [{"path": SRC, "duration": 1000.0}],
        "suggested_top": 2,
        "candidates": cands,
    }
    (project / "highlights.json").write_text(json.dumps(data), encoding="utf-8")
    return project


def write_labels(tmp_path: Path, name: str = "labels.json", **fields) -> Path:
    data = {"video": "clip-a.mp4", **fields}
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def grid(n: int = 20) -> list[dict]:
    """``n`` candidates of 10 s: candidate i covers [i*50, i*50+10], its score falls with i."""
    return [cand(i, i * 50.0, i * 50.0 + 10, 100.0 - i) for i in range(n)]


def test_overlap_rule_needs_half_of_the_shorter():
    assert me.hit((0, 10), (8, 12), 0.5) is True  # 2 of 4 s is exactly half
    assert me.hit((0, 10), (6, 12), 0.5) is True  # 4 of 6 s
    assert me.hit((0, 10), (4, 12), 0.5) is True  # 6 of 8 s
    assert me.hit((0, 10), (9, 20), 0.5) is False  # grazes
    assert me.hit((0, 100), (40, 46), 0.5) is True  # a long candidate holds a short range
    assert me.hit((40, 46), (0, 100), 0.5) is True  # a short candidate inside a long range
    assert me.hit((0, 10), (9, 13), 0.5) is False  # 1 of 4 s
    assert me.hit((0, 10), (7, 13), 0.8) is False


def test_top_hits_counts_labels_not_candidates():
    cands = [(0, 10), (5, 15), (100, 110)]
    labels = [(6, 9), (101, 109), (500, 510)]
    assert me.top_hits(cands, labels, 1, 0.5) == 1
    assert me.top_hits(cands, labels, 3, 0.5) == 2


def test_union_adds_only_new_seconds():
    u = me.Union()
    u.add(0, 10)
    assert u.added(5, 15) == 5
    u.add(5, 15)
    u.add(30, 40)
    u.add(8, 32)  # bridges both
    assert u.starts == [0] and u.ends == [40] and u.total == 40
    assert u.covered(35, 50) == 5


def test_budget_union_skips_what_does_not_fit_and_tries_the_next():
    cands = [(0, 10), (100, 140), (200, 205), (3, 8)]
    u, taken = me.budget_union(cands, 20)
    # (100, 140) is too long and skipped; (3, 8) adds nothing
    assert u.starts == [0, 200] and u.total == 15
    assert taken == [(0, 10), (200, 205), (3, 8)]
    assert me.budget_union(cands, 0)[0].total == 0


def test_budget_cover_counts_ranges_and_seconds():
    u, taken = me.budget_union([(0, 10)], 100)
    hits, secs = me.budget_cover(u, taken, [(0, 8), (8, 20)], 0.5)
    assert hits == 1
    assert secs == pytest.approx(10 / 20)


def test_the_30_percent_set_extends_the_15_percent_set():
    cands = [(0, 15), (100, 105), (200, 205)]
    labels = {"keep": [(100, 105), (200, 205)], "skip": [], "reference": []}
    out = me.measure(cands, labels, 70, 0.5)  # budgets of 10.5 s and 21 s
    assert out["keep_15"] == 2  # (0, 15) does not fit; the two short ones do
    assert out["keep_30"] >= out["keep_15"]
    # budgets of 15 s and 30 s: (0, 15) fills the first, so the 30% set must still hold it
    out = me.measure(cands, labels, 100, 0.5)
    assert (out["keep_15"], out["keep_30"]) == (0, 2)


def test_budget_metrics_never_fall_with_the_budget_in_any_order():
    cands = [(i * 40.0, i * 40.0 + 3 + (i * 7) % 25) for i in range(25)]
    labels = {
        "keep": [(i * 80.0 + 1, i * 80.0 + 6) for i in range(12)],
        "skip": [],
        "reference": [(i * 120.0, i * 120.0 + 30) for i in range(8)],
    }
    rng = random.Random(5)
    order = list(cands)
    for _ in range(200):
        rng.shuffle(order)
        m = me.measure(order, labels, 1000, 0.5)
        assert m["keep_30"] >= m["keep_15"]
        assert m["ref_30"] >= m["ref_15"]
        assert m["refsec_30"] >= m["refsec_15"]


def test_short_candidate_inside_a_long_reference_range_counts():
    u, taken = me.budget_union([(40, 46)], 100)
    hits, secs = me.budget_cover(u, taken, [(0, 100)], 0.5)
    assert hits == 1  # the candidate hits by the shorter rule though the union covers 6%
    assert secs == pytest.approx(0.06)
    # a taken candidate that only grazes the range does not count
    u, taken = me.budget_union([(95, 140)], 100)
    assert me.budget_cover(u, taken, [(0, 100)], 0.5)[0] == 0


def test_percentile_nearest_rank():
    assert me.percentile(list(range(1, 101)), 0.95) == 95
    assert me.percentile([4], 0.95) == 4


def test_measure_names_follow_the_labels():
    labels = {"keep": [(0, 10)], "skip": [], "reference": []}
    out = me.measure([(0, 10)], labels, 1000, 0.5)
    assert set(out) == {"keep_top10", "keep_top20", "keep_15", "keep_30"}


def test_control_is_deterministic_with_a_fixed_seed():
    cands = [(c["start"], c["end"]) for c in grid()]
    labels = {"keep": [(0, 10), (50, 60)], "skip": [], "reference": []}
    a = me.control(cands, labels, 1000, 0.5, 200, seed=3)
    assert a == me.control(cands, labels, 1000, 0.5, 200, seed=3)
    # each keep is in the 10 of 20 candidates a random top 10 holds: the mean is 1 of 2 keeps
    assert a["keep_top10"]["mean"] == pytest.approx(1.0, abs=0.15)
    assert a["keep_top10"]["p95"] == 2
    assert a["keep_top20"] == {"mean": 2.0, "p95": 2.0}


def test_beats_noise_only_above_the_p95(tmp_path):
    project = write_project(tmp_path, grid())
    # both keeps are in the top 2; the control reaches 2 of 2 at its 95th percentile
    labels = me.load_labels(write_labels(tmp_path, keep=[[0, 10], [50, 60]], skip=[[0, 10]]))
    v = me.evaluate_video(project, labels, 300, 1, 0.5, None)
    k10 = v["metrics"]["keep_top10"]
    assert k10["value"] == 2 and k10["control_p95"] == 2
    assert k10["beats_noise"] is False  # equal to the p95 is not above it
    assert v["metrics"]["skip_top10"]["value"] == 1
    assert v["metrics"]["skip_top10"]["beats_noise"] is False  # a skip is never flagged
    # ten keeps, all in the top 10: far above the p95 of a random order
    keeps = [[i * 50, i * 50 + 10] for i in range(10)]
    labels = me.load_labels(write_labels(tmp_path, keep=keeps))
    v = me.evaluate_video(project, labels, 300, 1, 0.5, None)
    assert v["metrics"]["keep_top10"]["value"] == 10
    assert v["metrics"]["keep_top10"]["beats_noise"] is True
    assert v["metrics"]["keep_top10"]["lift"] > 1.5


def test_reference_budget_and_default_edit(tmp_path):
    project = write_project(tmp_path, grid())
    # 15% of 1000 s is 150 s: the 15 best candidates (0..14). Reference 700 is candidate 14,
    # 800 is candidate 16 and only fits at 30%.
    ref = [[0, 10], [700, 710], [800, 810]]
    labels = me.load_labels(write_labels(tmp_path, reference=ref, keep=[[0, 10], [100, 110]]))
    v = me.evaluate_video(project, labels, 100, 1, 0.5, None)
    assert v["metrics"]["ref_15"]["value"] == 2
    assert v["metrics"]["keep_15"]["value"] == 2  # candidates 0 and 2 are among the 15 best
    assert v["metrics"]["refsec_15"]["value"] == pytest.approx(20 / 30)
    assert v["metrics"]["ref_30"]["value"] == 3
    # no total: the suggested top (2) of the project, as plain `clips`
    edit = v["edit"]
    assert edit["max_total"] is None and edit["clips"] == 2 and edit["seconds"] == 20
    assert edit["keep"]["contained"] == [[0.0, 10.0]]
    assert edit["reference"]["total"] == 3 and len(edit["reference"]["contained"]) == 1
    # edit highlights: max_total 180 holds 18 of the 10 s moments
    assert v["edit_highlights"]["max_total"] == 180 and v["edit_highlights"]["clips"] == 18
    assert me.evaluate_video(project, labels, 10, 1, 0.5, 30)["edit"]["clips"] == 3


def test_edit_counts_only_this_video_and_the_duration_must_agree(tmp_path):
    other = cand(50, 800, 810, 99.5, source="D:/Videos/other.mp4")
    project = write_project(tmp_path, [*grid(3), other])
    labels = me.load_labels(write_labels(tmp_path, keep=[[0, 10]]))
    edit = me.evaluate_video(project, labels, 5, 1, 0.5, None)["edit"]
    assert (edit["clips"], edit["clips_all"]) == (1, 2)  # the other source's pick is not counted
    bad = me.load_labels(write_labels(tmp_path, keep=[[0, 10]], duration=1002.0))
    with pytest.raises(ValueError, match="another file"):
        me.evaluate_video(project, bad, 5, 1, 0.5, None)
    ok = me.load_labels(write_labels(tmp_path, keep=[[0, 10]], duration=1000.5))
    assert me.evaluate_video(project, ok, 5, 1, 0.5, None)["duration_s"] == 1000.0


def test_candidates_match_the_video_by_name(tmp_path):
    other = cand(50, 0, 10, 99, source="D:/Videos/other.mp4")
    project = write_project(tmp_path, [*grid(3), other])
    info = me.load_project(project, "CLIP-A.mp4")
    assert len(info["candidates"]) == 3 and info["duration"] == 1000.0
    with pytest.raises(ValueError, match="no candidates"):
        me.load_project(project, "missing.mp4")


def test_visual_pool_candidates_are_not_ranked(tmp_path):
    pool = {**cand(60, 0, 10, 100), "pool": "visual"}
    project = write_project(tmp_path, [*grid(3), pool])
    assert len(me.load_project(project, "clip-a.mp4")["candidates"]) == 3


def test_label_validation(tmp_path):
    for bad in (
        {"keep": [[5, 5]]},
        {"keep": [[1]]},
        {"keep": "x"},
        {"keep": [[1, 2]], "holdout": "yes"},
        {},
    ):
        with pytest.raises(ValueError):
            me.load_labels(write_labels(tmp_path, **bad))
    lab = me.load_labels(write_labels(tmp_path, keep=[[1, 2]], note="n"))
    assert lab["holdout"] is False and lab["skip"] == [] and lab["keep"] == [(1.0, 2.0)]


def test_holdout_is_listed_apart_and_out_of_the_totals(tmp_path, capsys):
    project = write_project(tmp_path, grid())
    a = write_labels(tmp_path, "a.json", keep=[[0, 10]])
    b = write_labels(tmp_path, "b.json", keep=[[50, 60]], holdout=True)
    c = write_labels(tmp_path, "c.json", keep=[[100, 110]])
    out = tmp_path / "run.json"
    argv = ["--runs", "20", "--json", str(out)]
    for labels in (a, b, c):
        argv += ["--project", str(project), "--labels", str(labels)]
    assert me.main(argv) == 0
    text = capsys.readouterr().out
    held = text.index("HOLDOUT")
    assert "do not tune" in text[held:]
    assert "totals" in text[:held] and "keep_top10  2/2" in text[:held]
    assert [v["holdout"] for v in json.loads(out.read_text())["videos"]] == [False, True, False]


def test_manifest_pairs_are_relative_to_it(tmp_path):
    write_project(tmp_path, grid())
    write_labels(tmp_path, keep=[[0, 10]])
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps({"pairs": [{"project": "proj", "labels": "labels.json"}]}))
    assert me.load_manifest(manifest) == [(tmp_path / "proj", tmp_path / "labels.json")]
    assert me.main(["--manifest", str(manifest), "--runs", "5"]) == 0


def test_compare_shows_changes(tmp_path, capsys):
    project = write_project(tmp_path, grid())
    labels = write_labels(tmp_path, keep=[[i * 50, i * 50 + 10] for i in range(10)])
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    me.main(["--project", str(project), "--labels", str(labels), "--runs", "50", "--json", str(a)])
    # the same candidates ranked the other way round: the keeps fall to the bottom
    worse = write_project(
        tmp_path, [{**c, "score": 81.0 + i} for i, c in enumerate(grid())], "worse"
    )
    me.main(["--project", str(worse), "--labels", str(labels), "--runs", "50", "--json", str(b)])
    capsys.readouterr()
    old, new = (json.loads(p.read_text()) for p in (a, b))
    row = {r["metric"]: r for r in me.compare(old, new)[0]["metrics"]}["keep_top10"]
    assert (row["old"], row["new"], row["delta"]) == (10, 0, -10)
    assert row["old_beats_noise"] is True and row["new_beats_noise"] is False
    assert me.main(["--compare", str(a), str(b)]) == 0
    text = capsys.readouterr().out
    assert "10 ->" in text and "(-10)" in text and "no longer beats noise" in text


def test_bad_input_is_an_error_exit(tmp_path):
    project = write_project(tmp_path, grid())
    labels = write_labels(tmp_path, video="nothing.mp4", keep=[[0, 10]])
    assert me.main(["--project", str(project), "--labels", str(labels)]) == 1
    with pytest.raises(SystemExit):
        me.main(["--project", str(project)])


def test_compare_warns_on_different_settings_and_has_edit_rows(tmp_path):
    project = write_project(tmp_path, grid())
    labels = me.load_labels(write_labels(tmp_path, keep=[[0, 10]]))
    video = me.evaluate_video(project, labels, 5, 1, 0.5, None)
    old = {"schema": 2, "seed": 7, "runs": 1000, "min_overlap": 0.5, "videos": [video]}
    new = {
        **old,
        "seed": 8,
        "runs": 10,
        "videos": [{**video, "edit": {**video["edit"], "clips": 5}}],
    }
    warnings = me.compare_warnings(old, new)
    assert len(warnings) == 2 and "seed differs" in warnings[0]
    assert me.compare_warnings(old, old) == []
    rows = {r["metric"]: r for r in me.compare(old, new)[0]["metrics"]}
    assert (rows["edit_clips"]["old"], rows["edit_clips"]["new"]) == (2, 5)
    assert "edit_keep" in rows and "edit_highlights_secs" in rows
