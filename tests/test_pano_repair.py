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


def test_failed_reselection_leaves_no_backup_and_the_segments_untouched(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    segs = project / "mix_segments.json"
    before = segs.read_bytes()
    qa = _qa(project, [[]] * 18, subj=False)  # the social clip goes, so the mix is re-run
    only = [Cand("social", "a.mp4", 300, 309, 304, 0.9, "laughter", "laugh")]
    monkeypatch.setattr(mix, "part_candidates", lambda _d: only)  # ...and nothing is left to pick
    with pytest.raises(ValueError, match="Nothing selected"):
        repair(project, segs, qa, sheet=False)
    assert segs.read_bytes() == before
    assert not (project / "mix_segments.bak.json").exists()


def test_backup_never_overwrites_an_earlier_one(project: Path) -> None:
    segs = project / "mix_segments.json"
    before = segs.read_bytes()
    first = project / "mix_segments.bak.json"
    first.write_text('["hand-kept"]')
    qa = _qa(project, [["occluded"]] * 9 + [[]] * 9)
    repair(project, segs, qa, sheet=False)
    assert first.read_text() == '["hand-kept"]'
    assert (project / "mix_segments.1.bak.json").read_bytes() == before


def test_out_spelled_differently_from_segments_still_gets_a_backup(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    segs = project / "mix_segments.json"
    before = cache.load_json(segs)
    qa = _qa(project, [["occluded"]] * 9 + [[]] * 9)
    (project / "scratch").mkdir()
    monkeypatch.chdir(project)
    repair(project, segs, qa, out=Path("scratch/../mix_segments.json"), sheet=False)
    assert cache.load_json(segs) != before
    assert cache.load_json(project / "mix_segments.bak.json") == before


def test_backup_failing_midway_leaves_no_partial_file_and_the_segments_intact(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno  # noqa: PLC0415

    real = Path.write_bytes

    def disk_fills(self: Path, data: bytes) -> int:
        if "bak" in self.name or self.name.endswith(".tmp"):  # the restore's temp write too
            real(self, data[:5])
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(self, data)

    segs = project / "mix_segments.json"
    before = segs.read_bytes()
    qa = _qa(project, [["occluded"]] * 9 + [[]] * 9)
    monkeypatch.setattr(Path, "write_bytes", disk_fills)
    with pytest.raises(OSError, match="No space"):
        repair(project, segs, qa, sheet=False)
    assert segs.read_bytes() == before
    assert not [p for p in project.iterdir() if "bak" in p.name or p.name.endswith(".tmp")]


def _handwritten(project: Path, previews: bool, ids: bool = False, relative: bool = False) -> Path:
    """A category-less ``final_segments.json`` (user ids ``hNNN`` if ``ids``; intro clean, jump
    occluded for 3 s at the start then low-info for 2 s, short 3 s with its middle too close) plus the QA of its previews or rendered clips."""
    # Real absolute paths (a "/v/a.insv" is not absolute on Windows); with ``relative`` the file
    # spells them "raw/a.insv" (next to it) while the rendered manifest records the absolute one.
    a, b = str(project / "raw" / "a.insv"), str(project / "raw" / "b.insv")
    spell = {a: "raw/a.insv", b: "raw/b.insv"} if relative else {a: a, b: b}
    segs = [
        {"source": spell[a], "start": 10.0, "end": 20.0, "label": "intro"},
        {"source": spell[a], "start": 100.0, "end": 108.0, "label": "jump"},
        {"source": spell[b], "start": 50.0, "end": 53.0, "label": "short"},
    ]
    if ids:
        for i, s in enumerate(segs, 1):
            s["id"] = f"h{i:03d}"
    cache.save_json(project / "final_segments.json", segs)
    src = project / ("pano_paths/final" if previews else "clips_final")
    src.mkdir(parents=True)
    clips, entries = [], []
    lead = 0.0 if previews else 1.0  # rendered clips run a second either side of the pick
    for i, s in enumerate(segs, 1):
        stem = f"{i:03d}_{s['label']}"
        (src / f"{stem}.mp4").write_bytes(b"")
        dur = s["end"] - s["start"] + 2 * lead
        if previews:
            cache.save_json(src / f"{stem}.json", {"segment": {"start": s["start"]}})
        else:
            entries.append(
                {
                    "file": f"{stem}.mp4",
                    "id": segs[i - 1].get("id", f"s{i:03d}"),
                    "source": str(project / s["source"]) if relative else s["source"],
                    "duration": dur,
                }
                | {"start": s["start"] - lead, "end": s["end"] + lead}
                | {"pick": {"start": s["start"]}, "original_end": s["end"]}
            )

        def flags(k: int, i: int = i) -> list[str]:
            t = k / 2 - lead  # seconds into the segment
            if i == 2 and 0 <= t < 3:
                return ["occluded"]
            if i == 2 and 3 <= t < 5:
                return ["low_info"]
            return ["too_close"] if i == 3 and 1 <= t < 2.5 else []

        frames = [
            {"t": k / 2, "flags": flags(k), "subj": [0.4, 0.2, 0.6, 0.8]}
            for k in range(int(dur * 2))
        ]
        clips.append({"file": f"{stem}.mp4", "duration": dur, "frames": frames})
    if not previews:
        cache.save_json(src / "manifest.json", {"clips": entries})
    qa = project / f"{src.name}_qa" / "qa.json"
    qa.parent.mkdir()
    cache.save_json(qa, {"source_dir": str(src), "clips": clips})
    return qa


@pytest.mark.parametrize("ids", [False, True], ids=["no-ids", "user-ids"])
@pytest.mark.parametrize("previews", [True, False], ids=["director-previews", "rendered-clips"])
def test_handwritten_segments_are_trimmed_or_dropped(
    tmp_path: Path, previews: bool, ids: bool
) -> None:
    """Issue #110: ``repair`` on a ``[{source,start,end,label}]`` file used to die on KeyError
    'id'. Now the occluded clip is cut, the too-short one dropped, nothing refilled; user ids
    (not a ``cutpoint mix`` file without categories) are kept. Low-info frames of a clip
    without category are no problem."""
    qa = _handwritten(tmp_path, previews, ids)
    segs = tmp_path / "final_segments.json"
    rep = repair(tmp_path, segs, qa, sheet=False)
    assert rep["written"] is True
    out = cache.load_json(segs)
    assert [s["label"] for s in out] == ["intro", "jump"]
    assert out[0]["start"] == 10.0 and out[0]["end"] == 20.0
    assert 102.5 <= out[1]["start"] < out[1]["end"] == 108.0
    assert [s.get("id") for s in out] == (["h001", "h002"] if ids else [None, None])
    assert out[1]["start"] < 103.0 + 1.0
    excl = cache.load_json(tmp_path / mix.EXCLUDE_FILE)
    assert excl
    assert all(e["end"] <= 103.1 or e["source"] == str(tmp_path / "raw" / "b.insv") for e in excl)
    assert (tmp_path / "final_segments.bak.json").is_file()


@pytest.mark.parametrize(
    "segment",
    [
        pytest.param({"source": "raw/a.insv", "end": 3.0}, id="missing-start"),
        pytest.param({"source": "raw/a.insv", "start": "x", "end": 3.0}, id="non-numeric-start"),
    ],
)
def test_cli_repair_reports_a_malformed_segment_as_an_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, segment: dict[str, object]
) -> None:
    qa = _handwritten(tmp_path, True)
    bad = tmp_path / "bad.json"
    cache.save_json(bad, [segment])
    from cutpoint.cli import main  # noqa: PLC0415

    code = main(["360", "repair", str(tmp_path), "--segments", str(bad), "--qa", str(qa)])
    assert code == 1
    assert f"{bad}: segment 1 needs source, start and end" in caplog.text


def test_relative_source_is_matched_to_the_manifest_and_written_back_as_spelt(
    tmp_path: Path,
) -> None:
    """Issue #119: a hand-written "raw/a.insv" (next to the file) is the manifest's resolved
    absolute source: the occluded clip is trimmed (not skipped as stale), the exclusion has
    the absolute source and the file keeps its own spelling."""
    qa = _handwritten(tmp_path, False, relative=True)
    segs = tmp_path / "final_segments.json"
    rep = repair(tmp_path, segs, qa, sheet=False)
    assert rep["written"] is True
    out = cache.load_json(segs)
    assert [s["label"] for s in out] == ["intro", "jump"]
    assert [s["source"] for s in out] == ["raw/a.insv", "raw/a.insv"]
    assert 102.5 <= out[1]["start"] < out[1]["end"] == 108.0
    assert all("id" not in s for s in out)
    excl = cache.load_json(tmp_path / mix.EXCLUDE_FILE)
    assert excl
    assert {e["source"] for e in excl} <= {
        str(tmp_path / "raw" / "a.insv"),
        str(tmp_path / "raw" / "b.insv"),
    }
    assert any(e["source"] == str(tmp_path / "raw" / "a.insv") for e in excl)


@pytest.mark.parametrize(
    ("index", "bad_id", "match"),
    [
        (2, "h001", r"final_segments\.json: Segments 1 and 3 .*'h001'"),
        (1, ["x"], r"final_segments\.json: Segment 2: id must be"),
    ],
    ids=["duplicate", "non-scalar"],
)
def test_repair_names_the_file_and_segment_of_a_bad_id(
    tmp_path: Path, index: int, bad_id: object, match: str
) -> None:
    qa = _handwritten(tmp_path, True, ids=True)
    segs = tmp_path / "final_segments.json"
    data = cache.load_json(segs)
    data[index]["id"] = bad_id
    cache.save_json(segs, data)
    with pytest.raises(ValueError, match=match):
        repair(tmp_path, segs, qa, sheet=False)


def test_relative_source_written_to_another_folder_keeps_its_identity(tmp_path: Path) -> None:
    """Review F1: a "raw/a.insv" is relative to the segments file, so an output in a different
    folder must carry a source that still resolves to the same file."""
    qa = _handwritten(tmp_path, False, relative=True)
    segs = tmp_path / "final_segments.json"
    out_path = tmp_path / "sub" / "x.json"
    out_path.parent.mkdir()
    rep = repair(tmp_path, segs, qa, out=out_path, sheet=False)
    assert rep["written"] is True
    out = cache.load_json(out_path)
    assert [s["label"] for s in out] == ["intro", "jump"]
    assert {(out_path.parent / s["source"]).resolve() for s in out} == {
        (tmp_path / "raw" / "a.insv").resolve()
    }


@pytest.mark.parametrize(
    "spelt", ["D:\\Videos\\a.insv", "D:/Videos/a.insv"], ids=["backslash", "forward-slash"]
)
def test_windows_source_matches_a_manifest_written_on_windows(tmp_path: Path, spelt: str) -> None:
    """Review F3: a hand-written ``D:\\Videos\\a.insv`` with a manifest written on Windows
    (it records that string) is matched on any OS, not joined to the segments folder and
    skipped as stale: the occluded clip is trimmed, the file keeps its spelling."""
    qa = _handwritten(tmp_path, False)
    segs = tmp_path / "final_segments.json"
    a = str(tmp_path / "raw" / "a.insv")
    data = cache.load_json(segs)
    for s in data:
        s["source"] = spelt if s["source"] == a else s["source"]
    cache.save_json(segs, data)
    manifest = tmp_path / "clips_final" / "manifest.json"
    entries = cache.load_json(manifest)
    for e in entries["clips"]:
        e["source"] = "D:\\Videos\\a.insv" if e["source"] == a else e["source"]
    cache.save_json(manifest, entries)
    rep = repair(tmp_path, segs, qa, sheet=False)
    assert rep["skipped"] == 0
    assert rep["written"] is True
    out = cache.load_json(segs)
    assert [s["label"] for s in out] == ["intro", "jump"]
    assert [s["source"] for s in out] == [spelt, spelt]
    assert 102.5 <= out[1]["start"] < out[1]["end"] == 108.0
    excl = cache.load_json(tmp_path / mix.EXCLUDE_FILE)
    assert any(e["source"] == "D:\\Videos\\a.insv" for e in excl)


def test_repair_out_into_a_new_folder_creates_it(tmp_path: Path) -> None:
    """Issue #151: ``--out`` into a folder that does not exist creates it (exclusions too)."""
    qa = _handwritten(tmp_path, False)
    out = tmp_path / "new" / "deep" / "x.json"
    rep = repair(tmp_path, tmp_path / "final_segments.json", qa, out=out, sheet=False)
    assert rep["written"] is True
    assert out.is_file() and (out.parent / mix.EXCLUDE_FILE).is_file()


def test_repair_takes_a_relative_exclusion_against_its_own_folder(tmp_path: Path) -> None:
    """Issue #151: a relative source in ``mix_exclude.json`` (written before #132) is taken
    against that file's folder, so it matches and is written back absolute."""
    qa = _handwritten(tmp_path, False)
    cache.save_json(
        tmp_path / mix.EXCLUDE_FILE, [{"source": "raw/b.insv", "start": 49.0, "end": 54.0}]
    )
    repair(tmp_path, tmp_path / "final_segments.json", qa, sheet=False)
    sources = {e["source"] for e in cache.load_json(tmp_path / mix.EXCLUDE_FILE)}
    assert str(tmp_path / "raw" / "b.insv") in sources
    assert "raw/b.insv" not in sources


def test_repair_matches_a_source_reached_through_a_symlinked_folder(tmp_path: Path) -> None:
    """Issue #151: the manifest stores the resolved source; a ``cutpoint mix`` segment spelt
    through a symlink still matches it instead of being skipped as stale."""
    real = tmp_path / "real"
    real.mkdir()
    try:
        (tmp_path / "link").symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    src = tmp_path / "clips"
    src.mkdir()
    (src / "001_x.mp4").write_bytes(b"")
    seg = {"id": "m1", "category": "run", "start": 10.0, "end": 20.0}
    seg["source"] = str(tmp_path / "link" / "a.insv")
    entry = {"file": "001_x.mp4", "id": "m1", "source": str(real / "a.insv"), "start": 9.0}
    cache.save_json(
        src / "manifest.json",
        {"clips": [entry | {"end": 21.0, "pick": {"start": 10.0}, "original_end": 20.0}]},
    )
    qa = {"clips": [{"file": "001_x.mp4", "duration": 12.0, "frames": []}]}
    assert plan([seg], qa, src)[2] == 0  # skipped


@pytest.mark.parametrize("case", ["out-is-folder", "out-parent-is-file", "drive-relative"])
def test_cli_repair_reports_bad_out_and_drive_relative_source_as_one_error(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Issue #151: ``--out`` that is a folder, or inside a file, and (on Windows) a
    drive-relative source are one ERROR line, exit 1, before anything is written."""
    import cutpoint.clips as clips_mod  # noqa: PLC0415
    from cutpoint.cli import main  # noqa: PLC0415

    qa = _handwritten(tmp_path, True)
    segs = tmp_path / "final_segments.json"
    argv = ["360", "repair", str(tmp_path), "--qa", str(qa)]
    if case == "out-is-folder":
        (tmp_path / "d").mkdir()
        argv += ["--segments", str(segs), "--out", str(tmp_path / "d")]
        phrase = f"{tmp_path / 'd'} is a folder, not a segments file"
    elif case == "out-parent-is-file":
        (tmp_path / "f").write_text("x", encoding="utf-8")
        out = tmp_path / "f" / "o.json"
        argv += ["--segments", str(segs), "--out", str(out)]
        phrase = f"Cannot write {out}: "
    else:
        monkeypatch.setattr(clips_mod, "_WINDOWS", True)
        segs = tmp_path / "drive.json"
        cache.save_json(segs, [{"source": "D:foo.insv", "start": 1, "end": 3}])
        argv += ["--segments", str(segs)]
        phrase = "segment 1 source 'D:foo.insv'"
    assert main(argv) == 1
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and phrase in errors[0]
    if case == "out-parent-is-file":
        assert errors[0].endswith(" is not a folder")
    if case == "drive-relative":
        assert str(segs) in errors[0] and "drive D:" in errors[0]
    assert not (tmp_path / "final_segments.bak.json").exists()


def test_drive_relative_source_is_a_plain_file_name_off_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #151: ``D:foo.insv`` is only drive-relative on Windows; elsewhere it is a name."""
    import cutpoint.clips as clips_mod  # noqa: PLC0415

    monkeypatch.setattr(clips_mod, "_WINDOWS", False)
    qa = _handwritten(tmp_path, True)
    segs = tmp_path / "drive.json"
    cache.save_json(segs, [{"source": "D:foo.insv", "start": 1, "end": 3}])
    repair(tmp_path, segs, qa, sheet=False)  # no ValueError


@pytest.mark.parametrize("spelling", ["symlink", "dotdot"])
def test_repair_honours_an_exclusion_spelt_differently_from_the_segment_source(
    tmp_path: Path, spelling: str
) -> None:
    """Issue #175: a ``mix_exclude.json`` source that is a symlink to, or a ``..`` spelling
    of, the segment's source still bounds the lead-in / tail at it."""
    qa = _handwritten(tmp_path, False)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "a.insv").write_bytes(b"")
    if spelling == "symlink":
        alias = tmp_path / "alias.insv"
        try:
            alias.symlink_to(tmp_path / "raw" / "a.insv")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")
    else:
        alias = tmp_path / "raw" / ".." / "raw" / "a.insv"
    cache.save_json(
        tmp_path / mix.EXCLUDE_FILE, [{"source": str(alias), "start": 110.0, "end": 112.0}]
    )
    repair(tmp_path, tmp_path / "final_segments.json", qa, sheet=False)
    jump = next(
        s for s in cache.load_json(tmp_path / "final_segments.json") if s["label"] == "jump"
    )
    assert jump["max_end"] == 110.0
