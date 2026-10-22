"""QA-driven repair: clean-part trim, drop and replace via exclusions, stable ids."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from cutpoint import cache, mix
from cutpoint.mix import Cand, apply_exclusions
from cutpoint.pano import cli as pano_cli
from cutpoint.pano.repair import _fit_budget, plan, repair


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A one-source project: lift 100-105, social 300-309 / 700-706, run 500-506."""

    def cands(_d: object) -> list[Cand]:
        return [
            Cand("lift", "a.mp4", 100, 105, 102, 1.0, "beauty", "lift"),
            Cand("social", "a.mp4", 300, 309, 304, 0.9, "laughter", "laugh"),
            Cand("social", "a.mp4", 700, 706, 703, 0.5, "laughter", "laugh"),
            Cand("run", "a.mp4", 500, 506, 503, 1.0, "carve", "run"),
        ]

    monkeypatch.setattr(mix.ski, "load_project_features", lambda _p: [{}])
    monkeypatch.setattr(mix, "part_candidates", cands)
    monkeypatch.setattr(mix, "load_crashes", lambda _p: None)
    monkeypatch.setattr(mix, "_runs_index", lambda _p: ([], {}))
    monkeypatch.setattr(mix, "_embed", lambda _c: None)
    mix.run_mix(tmp_path, target_s=25, sheet=False)
    return tmp_path


def _qa(project: Path, flags: list[list[str]], subj: bool = True) -> Path:
    """QA of preview 002_laugh (the 300-309 social clip), one frame per 0.5 s."""
    (project / "pano_paths" / "mix").mkdir(parents=True)
    cache.save_json(project / "pano_paths/mix/002_laugh.json", {"segment": {"start": 300.0}})
    (project / "pano_paths/mix/002_laugh.mp4").write_bytes(b"")
    frames = [
        {"t": i / 2, "flags": f, "subj": [0.4, 0.2, 0.6, 0.8] if subj else None}
        for i, f in enumerate(flags)
    ]
    run = sum(bool(f) for f in flags) / 2
    clip = {"file": "002_laugh.mp4", "duration": 9.0, "runs": {"occl_close": run}, "frames": frames}
    qa = project / "pano_paths" / "mix_qa" / "qa.json"
    qa.parent.mkdir()
    cache.save_json(qa, {"clips": [clip]})
    return qa


def test_repair_json_output_shape(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``360 repair --json`` prints the report dict; a clean QA changes nothing."""
    qa = _qa(project, [[]] * 18)
    before = (project / "mix_segments.json").read_bytes()
    args = argparse.Namespace(
        project=str(project), segments="mix_segments.json", qa=str(qa), out=None, json=True
    )
    pano_cli.cmd_repair(args)
    rep = json.loads(capsys.readouterr().out)
    assert set(rep) == {
        "segments",
        "written",
        "findings",
        "skipped",
        "trimmed",
        "dropped",
        "added",
        "capped",
        "exclusions",
        "rendered_s",
        "content_qa",
    }
    assert rep["written"] is False and rep["trimmed"] == [] and rep["dropped"] == []
    assert (project / "mix_segments.json").read_bytes() == before


def test_occluded_start_is_trimmed_to_the_clean_part(project: Path) -> None:
    segs = project / "mix_segments.json"
    before = cache.load_json(segs)
    assert [s["id"] for s in before] == ["m001", "m002", "m003"]
    qa = _qa(project, [["occluded"]] * 9 + [[]] * 9)  # 4.75 s clean: >= 3/4 of social 5 s
    rep = repair(project, segs, qa, sheet=False)
    after = cache.load_json(segs)
    assert [(s["start"], s["end"]) for s in after if s["id"] == "m002"] == [(304.25, 309.0)]
    assert [s for s in after if s["id"] != "m002"] == [s for s in before if s["id"] != "m002"]
    assert [s["id"] for s in rep["trimmed"]] == ["m002"] and not rep["dropped"]
    assert cache.load_json(project / "mix_segments.bak.json") == before
    assert cache.load_json(project / mix.EXCLUDE_FILE)[0]["end"] == 304.25


def test_social_without_people_is_replaced_and_stays_excluded(project: Path) -> None:
    out = project / "scratch" / "repaired.json"
    out.parent.mkdir()
    qa = _qa(project, [[]] * 18, subj=False)
    rep = repair(project, project / "mix_segments.json", qa, out=out, sheet=False)
    assert [s["id"] for s in rep["dropped"]] == ["m002"]
    assert [(s["id"], s["start"]) for s in rep["added"]] == [("m004", 700)]
    assert [s["id"] for s in cache.load_json(out)] == ["m001", "m003", "m004"]
    assert not (project / "mix_segments.bak.json").exists()
    excl = cache.load_json(out.with_name(mix.EXCLUDE_FILE))
    assert excl[0]["category"] == "social"
    mix.run_mix(project, target_s=25, sheet=False, exclude=excl)  # a fresh mix keeps it out
    assert 300 not in [s["start"] for s in cache.load_json(project / "mix_segments.json")]


def test_crash_keeps_the_part_with_its_impact() -> None:
    crash = Cand("crash", "a.mp4", 0, 12, 4, 1.0, "wipeout", "crash")
    late = apply_exclusions([crash], [{"source": "a.mp4", "start": 6, "end": 7}])
    assert [(c.start, c.end) for c in late] == [(0, 6)]
    assert apply_exclusions([crash], [{"source": "a.mp4", "start": 3, "end": 5}]) == []


def test_plan_ignores_a_crash_tumble_and_stale_previews(tmp_path: Path) -> None:
    cache.save_json(tmp_path / "001_crash.json", {"segment": {"start": 10.0}})
    (tmp_path / "001_crash.mp4").write_bytes(b"")
    seg = {"id": "m001", "source": "a.mp4", "start": 10, "end": 22, "category": "crash"}
    frames = [
        {"t": i / 2, "flags": ["occluded"] if i == 9 else [], "subj": None} for i in range(24)
    ]
    qa = {"clips": [{"file": "001_crash.mp4", "duration": 12.0, "frames": frames}]}
    assert plan([seg], qa, tmp_path, impacts=[("a.mp4", 14.0)])[1] == []
    assert len(plan([seg], qa, tmp_path)[1]) == 1
    assert plan([seg], qa, tmp_path, qa_time=0.0)[1:3] == ([], 1)


def test_plan_matches_impact_sources_case_insensitively_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache.save_json(tmp_path / "001_crash.json", {"segment": {"start": 10.0}})
    (tmp_path / "001_crash.mp4").write_bytes(b"")
    seg = {"id": "m001", "source": "a.mp4", "start": 10, "end": 22, "category": "crash"}
    frames = [
        {"t": i / 2, "flags": ["occluded"] if i == 9 else [], "subj": None} for i in range(24)
    ]
    qa = {"clips": [{"file": "001_crash.mp4", "duration": 12.0, "frames": frames}]}
    impacts = [("A.MP4", 14.0)]
    monkeypatch.setattr(sys, "platform", "win32")
    assert plan([seg], qa, tmp_path, impacts=impacts)[1] == []
    monkeypatch.setattr(sys, "platform", "linux")  # case-sensitive: a different file
    assert len(plan([seg], qa, tmp_path, impacts=impacts)[1]) == 1


def test_crash_keeps_its_approach_before_the_impact(tmp_path: Path) -> None:
    """Impact at 16 (clip 6 s): occluded at 11 (excluded), 12.5-14 (cut back to end 2 s
    before the impact) and 15 (the camera knocked in the tumble: ignored)."""
    cache.save_json(tmp_path / "001_crash.json", {"segment": {"start": 10.0}})
    (tmp_path / "001_crash.mp4").write_bytes(b"")
    seg = {"id": "m001", "source": "a.mp4", "start": 10, "end": 22, "category": "crash"}
    frames = [
        {"t": i / 2, "flags": ["occluded"] if i in (2, 5, 6, 7, 8, 10) else [], "subj": None}
        for i in range(24)
    ]
    qa = {"clips": [{"file": "001_crash.mp4", "duration": 12.0, "frames": frames}]}
    excl = plan([seg], qa, tmp_path, impacts=[("a.mp4", 16.0)])[0]
    assert [(e["start"], e["end"]) for e in excl] == [(10.75, 11.25), (12.25, 14.0)]


def test_carry_skips_a_neighbour_that_only_overlaps() -> None:
    """An exclusion shrank m043 (496.7-501.7) to 496.7-499.45; the next clip (501.3-507.3)
    overlaps the old window by 0.4 s but is not m043."""
    prev = [{"id": "m043", "source": "a.mp4", "start": 496.7, "end": 501.7, "category": "social"}]
    cut = Cand("social", "a.mp4", 496.7, 499.45, 498, 1.0, "laughter", "laugh")
    nxt = Cand("social", "a.mp4", 501.3, 507.3, 504, 1.0, "laughter", "laugh")
    assert mix._carry([nxt], prev) == []
    assert mix._carry([cut, nxt], prev) == [cut]


def test_rendered_tail_problem_caps_the_tail_and_keeps_the_segment(project: Path) -> None:
    """Rendered 002_laugh: m002 (300-309) with a 3 s lead-in and 6 s tail, occluded at 16 s
    and one low-information frame inside the segment (not a problem of its own)."""
    clips_dir = project / "clips_mix"
    clips_dir.mkdir()
    (clips_dir / "002_laugh.mp4").write_bytes(b"")
    entry = {"file": "002_laugh.mp4", "id": "m002", "source": "a.mp4", "start": 297.0}
    entry |= {"end": 315.0, "duration": 18.0, "original_end": 309.0, "pick": {"start": 300.0}}
    cache.save_json(clips_dir / "manifest.json", {"clips": [entry]})
    frames = [
        {"t": i / 2, "flags": ["occluded"] if 32 <= i < 34 else [], "subj": [0.4, 0.2, 0.6, 0.8]}
        for i in range(36)
    ]
    frames[10]["flags"] = ["low_info"]
    qa = project / "clips_mix_qa" / "qa.json"
    qa.parent.mkdir()
    clip = {"file": "002_laugh.mp4", "duration": 18.0, "frames": frames}
    cache.save_json(qa, {"source_dir": str(clips_dir), "clips": [clip]})
    segs = project / "mix_segments.json"
    before = cache.load_json(segs)
    rep = repair(project, segs, qa, sheet=False)
    after = cache.load_json(segs)
    assert after == [{**s, "max_end": 312.75} if s["id"] == "m002" else s for s in before]
    assert not rep["trimmed"] and not rep["dropped"] and rep["exclusions"] == 0
    assert [s["id"] for s in rep["capped"]] == ["m002"]
    # 18 s - 2.25 s of tail, plus m001 / m003 (not rendered) at their own 5 + 6 s
    assert rep["rendered_s"] == {"before": 18.0, "after": 26.8, "limit": 300.0}
    assert not (project / mix.EXCLUDE_FILE).exists()


def test_fit_budget_caps_quiet_social_tails_first_then_the_longest() -> None:
    def case(run_s: float) -> tuple[list[dict], dict[str, float]]:
        segs = [
            {"id": "a", "category": "social", "start": 0.0, "end": 100.0},  # tail 2 s, silent
            {"id": "b", "category": "run", "start": 10.0, "end": 110.0},  # lead-in 3 s
            {"id": "c", "category": "run", "start": 0.0, "end": run_s},  # tail 4 s
        ]
        clips = [
            {"id": "a", "start": 0.0, "end": 102.0, "pick": {"start": 0.0}, "original_end": 100.0},
            {"id": "b", "start": 7.0, "end": 110.0, "pick": {"start": 10.0}, "original_end": 110.0},
            {
                "id": "c",
                "start": 0.0,
                "end": run_s + 4,
                "pick": {"start": 0.0},
                "original_end": run_s,
            },
        ]
        return segs, _fit_budget(segs, {"clips": clips})

    segs, est = case(92.0)  # 301 s: 1 s from the silent social tail alone
    assert [{k: s.get(k) for k in ("min_start", "max_end")} for s in segs] == [
        {"min_start": None, "max_end": 101.0},
        {"min_start": None, "max_end": None},
        {"min_start": None, "max_end": None},
    ]
    assert sum(est.values()) == pytest.approx(300.0)
    segs, est = case(96.0)  # 305 s: the silent tail, then the 3 s / 4 s ones down to 2 s
    assert [(s.get("min_start"), s.get("max_end")) for s in segs] == [
        (None, 100.0),
        (8.0, None),
        (None, 98.0),
    ]
    assert sum(est.values()) == pytest.approx(300.0)
