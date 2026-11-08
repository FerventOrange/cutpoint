"""default_fit: the default moment selection, and its agreement with analyze, scorer and planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cutpoint import default_fit, scorer
from cutpoint.planning import select_moments
from cutpoint.styles import get_style


def _c(cid: str, score: float, start: float, end: float, source: str = "a.mp4", **extra: Any):
    return {
        "id": cid, "source": source, "start": start, "end": end,
        "duration": end - start, "score": score, **extra,
    }  # fmt: skip


def _project(tmp_path: Path, cands: list[dict[str, Any]], target: float = 90.0) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    data = {"style": "gaming", "settings": {"target_total": target}, "candidates": cands}
    (project / "highlights.json").write_text(json.dumps(data), encoding="utf-8")
    return project


def _ids(moments: list[dict[str, Any]]) -> list[str]:
    return [c["id"] for c in moments]


def test_a_219_s_number_one_does_not_become_the_whole_edit(tmp_path):
    cands = [
        _c("h001", 90, 1000, 1219),
        _c("h002", 80, 0, 30),
        _c("h003", 70, 100, 130),
        _c("h004", 60, 200, 230),
    ]
    chosen, _ = select_moments(_project(tmp_path, cands))
    assert _ids(chosen) == ["h002", "h003", "h004"]


def test_a_long_number_one_is_still_pickable_explicitly(tmp_path):
    project = _project(tmp_path, [_c("h001", 90, 1000, 1219), _c("h002", 80, 0, 30)])
    assert _ids(select_moments(project, ids=["h001"])[0]) == ["h001"]
    assert _ids(select_moments(project, top=1)[0]) == ["h001"]
    assert _ids(select_moments(project, max_total=300)[0]) == ["h001", "h002"]


def test_candidates_that_do_not_fit_are_skipped_and_the_next_is_tried(tmp_path):
    cands = [_c("a", 9, 0, 60), _c("b", 8, 100, 160), _c("c", 7, 200, 230)]
    chosen, _ = select_moments(_project(tmp_path, cands))
    assert _ids(chosen) == ["a", "c"]  # b (60 s) does not fit in the 30 s left; c does


def test_must_include_is_kept_past_the_target(tmp_path):
    cands = [
        _c("x", 50, 0, 100, must_include=True),
        _c("y", 40, 200, 230, must_include=True),
        _c("z", 99, 400, 410),
    ]
    chosen, _ = select_moments(_project(tmp_path, cands))
    assert _ids(chosen) == ["x", "y"]  # 130 s kept; z no longer fits


def test_overlap_skip_still_applies_and_a_skipped_clash_uses_no_room(tmp_path):
    cands = [
        _c("long", 9, 100, 160),
        _c("inside", 8, 120, 130),
        _c("far", 7, 500, 530),
    ]
    chosen, _ = select_moments(_project(tmp_path, cands))
    assert _ids(chosen) == ["long", "far"]


def test_nothing_fits_the_best_one_is_chosen(tmp_path):
    cands = [_c("a", 5, 0, 200), _c("b", 9, 300, 500)]
    chosen, _ = select_moments(_project(tmp_path, cands, target=60))
    assert _ids(chosen) == ["b"]


def test_explicit_top_and_max_total_are_unchanged(tmp_path):
    cands = [_c("a", 9, 0, 60), _c("b", 8, 100, 160), _c("c", 7, 200, 230)]
    project = _project(tmp_path, cands)
    assert _ids(select_moments(project, top=2)[0]) == ["a", "b"]
    assert _ids(select_moments(project, max_total=100)[0]) == ["a", "c"]


def test_a_project_without_a_target_keeps_suggested_top(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    cands = [_c(f"c{i}", i, i * 100, i * 100 + 10) for i in range(8)]
    (project / "highlights.json").write_text(
        json.dumps({"suggested_top": 3, "candidates": cands}), encoding="utf-8"
    )
    assert len(select_moments(project)[0]) == 3


def test_a_stale_suggested_top_does_not_decide(tmp_path):
    cands = [_c("a", 9, 0, 30), _c("b", 8, 100, 130)]
    project = _project(tmp_path, cands)
    data = json.loads((project / "highlights.json").read_text(encoding="utf-8"))
    data["suggested_top"] = 1  # from before this rule
    (project / "highlights.json").write_text(json.dumps(data), encoding="utf-8")
    assert len(select_moments(project)[0]) == 2


def test_commentary_fills_its_long_target():
    style = get_style("commentary")
    cands = [_c(f"c{i}", 100 - i, i * 500, i * 500 + 120) for i in range(12)]
    picks = default_fit.fit_target(cands, style.target_total)
    assert len(picks) == 6 and sum(default_fit.length(c) for c in picks) == 720


def test_action_cam_crashes_stay_first_and_past_the_target():
    crashes = [_c(f"k{i}", 10, i * 100, i * 100 + 40, must_include=True) for i in range(3)]
    rest = [_c("r", 99, 900, 920), _c("s", 98, 1000, 1030)]
    picks = default_fit.fit_target(rest + crashes, get_style("action-cam").target_total)
    assert _ids(picks[:3]) == ["k0", "k1", "k2"]


def test_rescore_and_planning_agree(tmp_path):
    """The suggested_top the events re-rank writes is the count select_moments picks."""
    cands = [
        _c("h001", 90, 1000, 1219),
        _c("h002", 80, 0, 30),
        _c("h003", 70, 10, 20),  # inside h002
        _c("h004", 60, 200, 230),
        _c("h005", 50, 300, 330),
        _c("h006", 40, 400, 410),
    ]
    for c in cands:
        c["signals"] = {}
    hl = {"style": "action", "settings": {"target_total": 90.0}, "candidates": cands}
    project = tmp_path / "proj"
    project.mkdir()
    scorer.rescore_events(hl, {"events": [], "windows": []}, project)
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    chosen, _ = select_moments(project)
    assert hl["suggested_top"] == len(chosen) == 3
    assert _ids(chosen) == ["h002", "h004", "h005"]
