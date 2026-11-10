"""``scripts/vision_eval.py``, the vision benchmark (#909): the overlap rule, the shares, Kendall
tau-b, the changes recovered and the command line, on synthetic ``highlights.json`` files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import load_script

pytestmark = pytest.mark.script
ve = load_script("vision_eval")

SRC = r"D:\Videos\Clip-A.mp4"


def project(
    tmp_path: Path,
    name: str,
    scores: list[float],
    vision: dict | None = None,
    target: float = 30,
    rated: tuple[int, ...] = (),
) -> Path:
    """Candidate i covers [i*100, i*100+10] with ``scores[i]``; the default edit fits 30 s.
    The candidates in ``rated`` carry a vision rating."""
    cands = [
        {"id": f"h{i:03d}", "source": SRC, "start": i * 100.0, "end": i * 100.0 + 10,
         "score": sc, "label": "x", "signals": {"vision": 7.0 if i in rated else None}}
        for i, sc in enumerate(scores)
    ]  # fmt: skip
    settings: dict = {"target_total": target}
    if vision is not None:
        settings["vision"] = vision
    out = tmp_path / name
    out.mkdir()
    data = {"version": 1, "settings": settings, "suggested_top": 3, "candidates": cands}
    (out / "highlights.json").write_text(json.dumps(data), encoding="utf-8")
    return out


def test_kendall_tau_b_of_known_orders():
    assert ve.kendall_tau([(1, 1), (2, 2), (3, 3)]) == 1.0
    assert ve.kendall_tau([(1, 3), (2, 2), (3, 1)]) == -1.0
    # one swap of four items: 5 concordant, 1 discordant of 6 pairs
    assert ve.kendall_tau([(1, 1), (2, 2), (3, 4), (4, 3)]) == pytest.approx(4 / 6)
    # a tie in y only: tau-b stays defined (5 concordant, 1 tied in y)
    assert ve.kendall_tau([(1, 1), (2, 2), (3, 3), (4, 3)]) == pytest.approx(5 / (6 * 5) ** 0.5)
    assert ve.kendall_tau([(1, 1)]) is None
    assert ve.kendall_tau([(1, 5), (2, 5)]) is None  # no spread


def test_same_moment_needs_half_of_the_shorter_and_one_file_name():
    a = {"source": SRC, "start": 0, "end": 10}
    assert ve.same(
        a, {"source": "/mnt/videos/clip-a.mp4", "start": 5, "end": 20}
    )  # name, 5 of 10 s
    assert not ve.same(a, {"source": SRC, "start": 6, "end": 20})  # 4 of 10 s
    assert not ve.same(a, {"source": r"D:\Videos\other.mp4", "start": 0, "end": 10})


def test_overlaps_and_tau_against_a_reference(tmp_path):
    ref = project(tmp_path, "ref", [90, 80, 70, 60, 50, 40])
    # the same moments, but 0 and 1 swapped and 5 far up: top 3 is {1, 0, 5}
    var = project(tmp_path, "var", [80, 90, 10, 20, 30, 70])
    r, v = ve.load_project(ref), ve.load_project(var)
    out = ve.evaluate(r, v, None)
    assert [m["id"] for m in r["edit"]] == ["h000", "h001", "h002"]
    assert [m["id"] for m in v["edit"]] == ["h001", "h000", "h005"]
    assert out["edit"] == pytest.approx(2 / 3) and out["edit_clips"] == 3
    assert out["top5"] == pytest.approx(4 / 5)  # ref top 5 is 0-4; variant top 5 is 1,0,5,4,3
    assert out["top10"] == 1.0
    assert out["tau20_n"] == 6
    assert out["tau20"] == pytest.approx((8 - 7) / 15)  # 15 pairs: 8 concordant, 7 discordant
    assert "recovered" not in out


def test_identical_projects_keep_everything(tmp_path):
    ref = project(tmp_path, "ref", [90, 80, 70, 60])
    out = ve.evaluate(ve.load_project(ref), ve.load_project(ref), None)
    assert (out["edit"], out["top5"], out["top10"], out["tau20"]) == (1.0, 1.0, 1.0, 1.0)


def test_recovered_counts_added_and_dropped_picks(tmp_path):
    nv = project(tmp_path, "nv", [90, 80, 70, 60, 50])  # no vision: edit 0,1,2
    ref = project(tmp_path, "ref", [10, 80, 70, 95, 50])  # vision: edit 3,1,2 (0 dropped, 3 added)
    good = project(tmp_path, "good", [10, 80, 70, 95, 50])  # makes both changes
    bad = project(tmp_path, "bad", [90, 80, 70, 60, 50])  # makes neither
    r, n = ve.load_project(ref), ve.load_project(nv)
    assert ve.recovered(r["edit"], n["edit"], ve.load_project(good)["edit"]) == {
        "added": 1, "dropped": 1, "share": 1.0,
    }  # fmt: skip
    assert ve.recovered(r["edit"], n["edit"], ve.load_project(bad)["edit"])["share"] == 0.0


def test_main_prints_the_cost_and_writes_valid_json(tmp_path, capsys):
    cost = {"vision_s": 50.0, "pre_vision_s": 100.0}
    ref = project(tmp_path, "ref", [90, 80, 70, 60], vision=cost)
    var = project(
        tmp_path, "var", [90, 80, 70, 60], vision={"vision_s": 10.0, "pre_vision_s": 100.0}
    )
    nv = project(tmp_path, "nv", [60, 70, 80, 90])
    out = tmp_path / "res" / "out.json"
    code = ve.main([str(ref), str(var), "--novision", str(nv), "--json", str(out)])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == ve.SCHEMA and data["reference"]["cost"]["ratio"] == 0.5
    (v,) = data["variants"]
    assert v["cost"]["ratio"] == 0.1 and v["edit"] == 1.0 and v["recovered"]["share"] == 1.0
    assert "0.10" in capsys.readouterr().out


def test_a_project_without_the_timings_shows_a_dash_for_cost(tmp_path, capsys):
    ref = project(tmp_path, "ref", [90, 80, 70, 60])  # analyzed before #909: no settings.vision
    var = project(tmp_path, "var", [90, 80, 70, 60], vision={"requested": 3, "failed": 0})
    assert ve.main([str(ref), str(var)]) == 0
    out = capsys.readouterr().out
    assert "cost -x" in out
    row = next(x for x in out.splitlines() if x.rstrip().endswith(str(var)))
    assert row.split()[0] == "-"


def test_main_reports_a_project_without_scored_moments(tmp_path, caplog):
    ref = project(tmp_path, "ref", [90, 80])
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "highlights.json").write_text('{"candidates": []}', encoding="utf-8")
    assert ve.main([str(ref), str(empty)]) == 1
    assert "no scored candidates" in caplog.text
    assert ve.main([str(ref), str(tmp_path / "missing")]) == 1


REF_COST = {"vision_s": 50.0, "calls": 5, "pre_vision_s": 100.0}  # 10 s per call, 100 s before


@pytest.mark.parametrize(
    ("variant", "want"),
    [
        ({"rated": 4, "share": 0.3}, 0.30),  # 40 s of calls, the cap is 30 s
        ({"rated": 2, "share": 0.3}, 0.20),  # under the cap
        ({"rated": 4, "share": 0.0}, 0.10),  # one call at least
        ({"rated": 4}, 0.30),  # no share recorded: the 0.3 default
        ({"share": 0.3}, None),  # nothing rated recorded
    ],
)
def test_est_ratio_is_the_cost_on_the_references_run_with_the_cap(variant, want):
    got = ve.est_ratio(REF_COST, variant)
    assert got == pytest.approx(want) if want is not None else got is None


def test_est_ratio_needs_the_references_timings():
    assert ve.est_ratio({"vision_s": 50.0, "pre_vision_s": 100.0}, {"rated": 3}) is None
    assert ve.est_ratio({}, {"rated": 3}) is None


def test_flips_count_rated_and_unrated_pairs_the_variant_reorders(tmp_path):
    nv = ve.load_project(project(tmp_path, "nv", [90, 80, 70, 60]))
    # h002 rated up to 95: it now passes the unrated h000 and h001, not h003 (as before)
    up = ve.load_project(project(tmp_path, "up", [90, 80, 95, 60], rated=(2,)))
    assert ve.flips(nv["ranked"], up["ranked"]) == 2
    # rated, but kept at or under the unrated above it (a tie is not a flip): none
    held = ve.load_project(project(tmp_path, "held", [90, 80, 80, 60], rated=(2,)))
    assert ve.flips(nv["ranked"], held["ranked"]) == 0
    # two rated moments reordering between themselves is free
    free = ve.load_project(project(tmp_path, "free", [90, 80, 65, 75], rated=(2, 3)))
    assert ve.flips(nv["ranked"], free["ranked"]) == 0


def test_main_shows_est_and_flips_and_writes_them(tmp_path, capsys):
    ref = project(tmp_path, "ref", [90, 80, 70, 60], vision=REF_COST)
    var = project(
        tmp_path, "var", [90, 80, 95, 60], rated=(2,),
        vision={"vision_s": 10.0, "pre_vision_s": 100.0, "rated": 4, "share": 0.3},
    )  # fmt: skip
    nv = project(tmp_path, "nv", [90, 80, 70, 60])
    out = tmp_path / "out.json"
    assert ve.main([str(ref), str(var), "--novision", str(nv), "--json", str(out)]) == 0
    (v,) = json.loads(out.read_text(encoding="utf-8"))["variants"]
    assert v["est_ratio"] == pytest.approx(0.3) and v["flips"] == 2
    text = capsys.readouterr().out
    assert "est" in text and "flips" in text
    row = next(x for x in text.splitlines() if x.rstrip().endswith(str(var)))
    assert "0.30" in row.split()
