"""Variety selection (quotas, variety rules, run coverage, crash rule), arc order, content QA."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import link_file, messages
from cutpoint import cache, mix
from cutpoint.clips import REACTION_LABEL, order_moments
from cutpoint.mix import (
    Cand,
    MixRules,
    content_qa,
    crash_candidates,
    effective_quotas,
    normalise,
    parse_quotas,
    part_candidates,
    select_mix,
)


def _c(cat: str, t: float, score: float, src: str = "a", dur: float = 6.0, **kw) -> Cand:
    return Cand(cat, src, t - dur / 2, t + dur / 2, t, score, cat, cat, **kw)


def test_parse_quotas_overrides_and_normalises() -> None:
    q = parse_quotas("run=.7")
    assert sum(q.values()) == pytest.approx(1.0)
    assert q["run"] > q["social"] > q["jump"]
    with pytest.raises(ValueError, match="CATEGORY=SHARE"):
        parse_quotas("bogus=1")


def test_unused_quota_goes_to_run_and_scenery() -> None:
    q = effective_quotas(parse_quotas(None), {"run", "scenery", "social"})
    assert set(q) == {"run", "scenery", "social", "flex"}
    assert sum(q.values()) == pytest.approx(1.0)
    assert q["run"] / q["scenery"] == pytest.approx(3.5)  # split in proportion
    assert q["social"] == pytest.approx(0.2)


def test_selection_respects_spacing_source_cap_and_budget() -> None:
    cands = [_c("run", t, t) for t in (100, 150, 300, 500, 700)]  # one source
    cands += [_c("run", t, 1.0, src=f"s{i}") for i, t in enumerate((10, 20, 30, 40, 50, 60))]
    normalise(cands)
    chosen = select_mix(cands, target_s=40, quotas={"run": 1.0})
    a = sorted(c.peak_t for c in chosen if c.source == "a")
    assert len(a) <= 2 and all(y - x >= 90 for x, y in itertools.pairwise(a))
    assert sum(c.seconds for c in chosen) <= 0.9 * 40


def test_redistributed_quota_never_exceeds_max_share() -> None:
    cands = [
        _c(cat, 100 + 120 * k, k + s, src=f"s{s}", dur=8.0)
        for s in range(20)
        for k in range(5)
        for cat in ("run", "scenery", "social")
    ]
    normalise(cands)
    rules = MixRules()
    chosen = select_mix(cands, target_s=240, rules=rules)
    for cat in ("run", "scenery", "social"):
        assert sum(c.seconds for c in chosen if c.category == cat) <= rules.max_share * 0.9 * 240


def test_near_duplicates_are_vetoed_in_a_category_or_close_in_time() -> None:
    e = np.ones(4)
    cands = [
        _c("lift", 10, 1.0, src="a", emb=e),
        _c("lift", 10, 0.9, src="b", emb=e * 2),  # same look, other source
        _c("scenery", 50, 0.8, src="b", emb=e),  # other category, far apart: kept
        _c("scenery", 30, 0.9, src="a", emb=e),  # other category, 20 s from the lift
    ]
    normalise(cands)
    chosen = select_mix(cands, target_s=100, quotas={"lift": 0.5, "scenery": 0.5})
    assert sorted((c.category, c.source) for c in chosen) == [("lift", "a"), ("scenery", "b")]


def test_run_coverage_swaps_in_an_uncovered_run() -> None:
    cands = [_c("run", t, 1.0, src="a", run=0) for t in (100, 300)]
    cands += [_c("run", 50, 0.1, src="b", run=1), _c("crash", 500, 0.1, src="a", run=0)]
    normalise(cands)
    rules = MixRules(run_cover=1.0, max_share=1.0)
    quotas = {"run": 0.67, "crash": 0.33}
    chosen = select_mix(cands, target_s=20, quotas=quotas, rules=rules, n_runs=2)
    assert {c.run for c in chosen} == {0, 1}
    assert any(c.category == "crash" for c in chosen)  # a lone crash never gives way


def test_crash_rule_keeps_weak_but_corroborated_crashes() -> None:
    def crash(conf: float, scores: dict[str, float], t: float) -> dict:
        clip = {"start": t - 5, "end": t + 20, "reaction": {"start": t + 60, "end": t + 80}}
        return {"source": "a", "t0": t, "confidence": conf, "scores": scores, "clip": clip}

    out = crash_candidates(
        [
            crash(0.29, {"words": 0.99, "telemetry": 0.1}, 100),  # weak telemetry, words agree
            crash(0.28, {"visual": 0.9}, 300),  # nothing corroborates
            crash(0.1, {"audio": 1.0}, 500),  # below max(0.2, 0.6 x best)
        ]
    )
    assert [c.peak_t for c in out] == [100]
    assert out[0].seconds == pytest.approx(12 + 8)  # crash and reaction, both capped


def test_part_candidates_in_source_time_and_one_lift_per_ride() -> None:
    n = 2000  # 200 s part starting at 100 s in the source
    f = {k: np.zeros(n) for k in ("carve", "people", "stopped", "airtime", "laughter", "voice")}
    f["beauty"] = np.full(n, 0.3)
    f["carve"][500:700] = 1.0
    mask = np.zeros(n)
    mask[1200:1450] = 1.0  # 25 s telemetry lift ride
    d = {
        "features": f,
        "mask": mask,
        "source": "a.mp4",
        "offset": 100.0,
        "duration": 200.0,
        "events": [],
    }
    out = part_candidates(d)
    runs = [c for c in out if c.category == "run"]
    lifts = [c for c in out if c.category == "lift"]
    assert runs and all(c.start >= 100 and 6 <= c.end - c.start <= 9 for c in runs)
    assert len(lifts) == 1 and 220 <= lifts[0].start < lifts[0].end <= 245


def _seg(i: int, cat: str, score: float, chapter: int, **kw) -> dict:
    return {
        "id": f"m{i:03d}",
        "source": f"s{chapter}",
        "start": 10.0 * i,
        "end": 10.0 * i + 6,
        "score": score,
        "category": cat,
        "label": cat,
        "chapter": chapter,
        **kw,
    }


@pytest.fixture
def day() -> list[dict]:
    return [
        _seg(1, "run", 0.9, 1),
        _seg(2, "scenery", 0.5, 1),
        _seg(3, "lift", 0.8, 1),  # best view of the first third: opener
        _seg(4, "social", 0.7, 2),
        _seg(5, "run", 0.6, 2),
        _seg(6, "crash", 0.4, 2),
        _seg(7, "run", 0.5, 2),
        _seg(8, "crash", 0.4, 2, label=REACTION_LABEL),
        _seg(9, "run", 0.7, 2),
        _seg(10, "run", 0.8, 2),
        _seg(11, "crash", 0.9, 3),  # best crash: finale
        _seg(12, "crash", 0.9, 3, label=REACTION_LABEL),
        _seg(13, "social", 0.3, 3),
        _seg(14, "scenery", 0.6, 3),  # late view: outro
    ]


def test_arc_order_keeps_every_moment(day) -> None:
    arc = order_moments(day, "arc")
    assert sorted(m["id"] for m in arc) == sorted(m["id"] for m in day)


def test_arc_order_opens_with_the_trimmed_best_view(day) -> None:
    first = order_moments(day, "arc")[0]
    assert first["id"] == "m003" and first["end"] - first["start"] == 5.0


def test_arc_order_ends_with_the_best_crash_its_reaction_and_the_outro(day) -> None:
    assert [m["id"] for m in order_moments(day, "arc")][-3:] == ["m011", "m012", "m014"]


def test_arc_order_puts_a_crash_reaction_right_after_its_crash(day) -> None:
    ids = [m["id"] for m in order_moments(day, "arc")]
    assert ids[ids.index("m006") + 1] == "m008"


def test_arc_order_does_not_repeat_a_category_three_times_in_a_row(day) -> None:
    cats = [m["category"] for m in order_moments(day, "arc")]
    assert all(len(set(cats[i : i + 3])) > 1 for i in range(len(cats) - 2))


def test_arc_order_is_chronological_by_stretch() -> None:
    """A stretch plays whole before the next one, whatever the category order within it."""
    moments = [
        _seg(1, "lift", 0.9, 1),  # opener
        _seg(2, "social", 0.5, 1),  # social normally follows a run, but stretch 1 comes first
        _seg(3, "run", 0.5, 2),
        _seg(4, "crash", 0.9, 2),  # finale
    ]
    assert [m["id"] for m in order_moments(moments, "arc")] == ["m001", "m002", "m003", "m004"]


def test_arc_order_without_categories_is_the_story_order(day) -> None:
    plain = [{k: v for k, v in m.items() if k != "category"} for m in day]
    assert order_moments(plain, "arc") == order_moments(plain, "story")


def test_arc_finale_fix_never_joins_two_streaks_of_a_category() -> None:
    runs = [*(_seg(i, "run", 0.5, 1) for i in range(1, 5)), _seg(5, "social", 0.5, 2)]
    cats = [m["category"] for m in order_moments([*runs, _seg(6, "run", 0.9, 2)], "arc")]
    assert "run,run,run,run" not in ",".join(cats)


def test_arc_finale_never_makes_a_third_clip_of_a_category_in_a_row() -> None:
    moments = [
        _seg(1, "lift", 0.9, 1),  # opener
        _seg(2, "run", 0.5, 1),
        _seg(3, "crash", 0.3, 1),
        _seg(4, "crash", 0.4, 1),
        _seg(5, "crash", 0.9, 1),  # finale: would be the third crash after the body's two
    ]
    arc = order_moments(moments, "arc")
    cats = [m["category"] for m in arc]
    assert arc[-1]["id"] == "m005"
    assert "crash,crash,crash" not in ",".join(cats)


def test_content_qa_flags_each_failed_check() -> None:
    good = [
        _seg(1, cat, 1, 1, start=a, end=b)
        for cat, a, b in (
            ("lift", 0, 40),
            ("run", 40, 100),
            ("social", 100, 160),
            ("crash", 160, 240),
        )
    ]
    assert content_qa(good, runs_covered=7, runs=10)["pass"]
    bad = content_qa(good[1:3], near_dups=1, runs_covered=6, runs=10)
    failed = {k for k, v in bad["checks"].items() if not v["ok"]}
    assert failed == {"categories", "max_share", "near_dups", "runs_covered", "opener", "length_s"}


@pytest.fixture
def one_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_mix on a project whose only candidate is a 6 s run in a.mp4 (features faked)."""
    monkeypatch.setattr(mix.ski, "load_project_features", lambda _p: [{}])
    monkeypatch.setattr(mix, "part_candidates", lambda _d: [_c("run", 50, 0.9)])
    monkeypatch.setattr(mix, "_runs_index", lambda _p, _k: ([], {}))
    monkeypatch.setattr(mix, "_embed", lambda _c: None)


@pytest.mark.parametrize(
    ("target", "exclude", "why"),
    [
        (1.0, [], "fits"),  # the 6 s run does not fit a 1 s target
        (60.0, [{"source": "a", "start": 0, "end": 100}], "excluded"),
    ],
)
def test_run_mix_with_nothing_selected_raises_and_writes_nothing(
    one_candidate, tmp_path, target, exclude, why
) -> None:
    with pytest.raises(ValueError, match=rf"Nothing selected in .*{why}"):
        mix.run_mix(tmp_path, target_s=target, exclude=exclude, out=tmp_path / "mix_segments.json")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("spelling", ["symlink", "dotdot"])
def test_apply_exclusions_matches_a_differently_spelt_source(tmp_path, spelling) -> None:
    """Issue #175: an exclusion whose source is a symlink to, or a ``..`` spelling of, the
    candidate's source still cuts it."""
    (tmp_path / "sub").mkdir()
    real = tmp_path / "sub" / "a.mp4"
    real.write_bytes(b"")
    if spelling == "symlink":
        alias = tmp_path / "alias.mp4"
        link_file(alias, real)
    else:
        alias = tmp_path / "sub" / ".." / "sub" / "a.mp4"
    cand = _c("run", 50, 0.9, src=str(real))
    assert mix.apply_exclusions([cand], [{"source": str(alias), "start": 0, "end": 100}]) == []


def test_load_exclusions_takes_a_relative_source_against_the_file_folder(
    tmp_path, monkeypatch
) -> None:
    """Issue #175: a relative source (as old ``360 repair`` wrote it) is the file's folder's,
    not the working directory's, so it cuts the candidate with the absolute path."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.insv").write_bytes(b"")
    path = project / mix.EXCLUDE_FILE
    path.write_text('[{"source": "a.insv", "start": 0, "end": 100}]', encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cand = _c("run", 50, 0.9, src=str(project / "a.insv"))
    assert mix.apply_exclusions([cand], mix.load_exclusions(path)) == []


def _alias(real: Path, spelling: str) -> Path:
    """Another spelling of ``real``: a symlink to it, or a ``..`` path."""
    if spelling == "dotdot":
        return real.parent / ".." / real.parent.name / real.name
    alias = real.with_name(f"alias_{real.name}")
    link_file(alias, real)
    return alias


@pytest.mark.parametrize("spelling", ["symlink", "dotdot"])
def test_run_mix_matches_previous_and_telemetry_sources_spelt_differently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    """Issue #181: a previous segment (keeps its id and lead-in / tail bound) and
    telemetry.json (orders the clips by start time) name the source by another spelling
    than the candidate's."""
    (tmp_path / "sub").mkdir()
    a, b = tmp_path / "sub" / "a.mp4", tmp_path / "sub" / "b.mp4"
    a.write_bytes(b"")
    b.write_bytes(b"")
    cands = [_c("run", 50, 0.9, src=str(a)), _c("run", 50, 0.8, src=str(b))]
    monkeypatch.setattr(mix.ski, "load_project_features", lambda _p: [{}])
    monkeypatch.setattr(mix, "part_candidates", lambda _d: cands)
    monkeypatch.setattr(mix, "_embed", lambda _c: None)
    alias_a, alias_b = str(_alias(a, spelling)), str(_alias(b, spelling))
    clips = [  # b was filmed first, though a sorts first
        {"source": alias_a, "start_epoch": 2000.0},
        {"source": alias_b, "start_epoch": 1000.0},
    ]
    cache.save_json(tmp_path / "telemetry.json", {"runs": [], "clips": clips})
    prev = [{"id": "m007", "source": alias_a, "start": 47.0, "end": 53.0}]
    prev[0] |= {"category": "run", "max_end": 60.0}
    mix.run_mix(tmp_path, target_s=60, sheet=False, exclude=[], previous=prev)
    segs = cache.load_json(tmp_path / "mix_segments.json")
    assert [(s["source"], s["id"]) for s in segs] == [(str(b), "m008"), (str(a), "m007")]
    assert segs[1]["max_end"] == 60.0


def test_run_mix_rerun_reads_relative_sources_of_the_segments_file_against_its_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #181: a plain re-run (no ``previous``) keeps the lead-in / tail bounds of a
    ``mix_segments.json`` entry whose source is relative to that file's folder, whatever
    the working directory."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "sub" / "a.mp4").write_bytes(b"")
    monkeypatch.setattr(mix.ski, "load_project_features", lambda _p: [{}])
    monkeypatch.setattr(
        mix, "part_candidates", lambda _d: [_c("run", 50, 0.9, src=str(tmp_path / "sub" / "a.mp4"))]
    )
    monkeypatch.setattr(mix, "_embed", lambda _c: None)
    prev = {"id": "m001", "source": "sub/a.mp4", "start": 47.0, "end": 53.0, "category": "run"}
    cache.save_json(tmp_path / "mix_segments.json", [prev | {"max_end": 60.0}])
    monkeypatch.chdir(tmp_path / "elsewhere")
    mix.run_mix(tmp_path, target_s=60, sheet=False, exclude=[])
    (seg,) = cache.load_json(tmp_path / "mix_segments.json")
    assert seg["max_end"] == 60.0


@pytest.mark.parametrize(
    "on_windows",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.skipif(
                sys.platform == "win32", reason="D:a.insv is drive-relative on Windows"
            ),
        ),
    ],
    ids=["windows", "elsewhere"],
)
def test_load_exclusions_drops_a_drive_relative_source_only_on_windows(
    tmp_path: Path, windows, caplog, on_windows: bool
) -> None:
    """Issue #181: ``D:a.insv`` is relative to drive D's current folder on Windows, so it
    never matches: dropped with a WARNING, the other entries kept. Elsewhere it is a file
    name next to the exclusions file."""
    windows(on_windows)
    path = tmp_path / mix.EXCLUDE_FILE
    other = {"source": "b.insv", "start": 0, "end": 5}
    cache.save_json(path, [other, {"source": "D:a.insv", "start": 1, "end": 2}, other])
    with caplog.at_level("WARNING", logger="cutpoint.mix"):
        got = mix.load_exclusions(path)
    if on_windows:
        assert [e["source"] for e in got] == [str(tmp_path / "b.insv")] * 2
        (msg,) = messages(caplog)
        assert str(path) in msg
    else:
        assert [e["source"] for e in got][1] == str(tmp_path / "D:a.insv")
        assert not caplog.records
