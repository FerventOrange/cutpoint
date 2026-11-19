"""Mix categories from the activity profile in use (audit #1085 A788): the profile's
optional ``mix`` section names, weighs or leaves out the mix categories and picks the
opener; without one the mix is unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cutpoint import activity, cache, mix
from cutpoint.clips import REACTION_LABEL, load_segments, order_moments
from cutpoint.mix import Cand, content_qa
from cutpoint.pano import repair

MIX = {
    "categories": {
        "run": {"name": "Wave", "share": 0.5},
        "lift": {"share": 0},
        "scenery": {"name": "line up"},
    },
    "opener": ["run", "scenery"],
}
SURF = {
    "name": "surf-test",
    "description": "test",
    "incident": {"pos": ["a surfer falling off the board"], "neg": ["a surfer riding a wave"]},
    "mix": MIX,
}


def _c(cat: str, t: float, score: float, src: str = "a", dur: float = 6.0, label: str = "") -> Cand:
    return Cand(cat, src, t - dur / 2, t + dur / 2, t, score, cat, label or cat)


def _day() -> list[Cand]:
    cands = [
        _c(cat, 100.0 * i + 10, 1.0 + i, src=f"{cat}{i}", label=word)
        for cat, word in (
            ("run", "run"),
            ("scenery", "view"),
            ("lift", "lift"),
            ("social", "laugh"),
            ("jump", "jump 1.2s"),
        )
        for i in range(8)
    ]
    crash = _c("crash", 50, 0.9, src="crash0", label="crash")
    crash.extra.append(Cand("crash", "crash0", 60, 64, 50, 0.9, "wipeout", REACTION_LABEL))
    return [*cands, crash]


@pytest.fixture
def fake_day(monkeypatch: pytest.MonkeyPatch):
    def install(cands_factory=_day) -> None:
        monkeypatch.setattr(mix.actioncam, "load_project_features", lambda _p: [{}])
        monkeypatch.setattr(mix, "part_candidates", lambda _d: cands_factory())
        monkeypatch.setattr(mix, "load_crashes", lambda _p: None)
        monkeypatch.setattr(mix, "_runs_index", lambda _p, _k: ([], {}))
        monkeypatch.setattr(mix, "_embed", lambda _c: None)

    return install


def _record(project: Path, profile: dict) -> None:
    project.mkdir(parents=True, exist_ok=True)
    rec = {
        "version": activity.PROFILE_VERSION,
        "name": profile["name"],
        "source": "flag",
        "profile": activity.full(activity.validate(profile)),
    }
    cache.save_json(project / activity.ACTIVITY_FILE, rec)


# ---------------------------------------------------------------- the profile schema
def test_validate_keeps_the_valid_part_of_mix() -> None:
    raw = {
        "categories": {
            "run": {"name": "  Big   Wave ", "share": 0.4, "junk": 1},
            "lift": {"share": 0},
            "jump": {"name": "x" * 30, "share": 2},  # both invalid: dropped
            "social": {"name": "!!!"},  # no letter or digit
            "paddle": {"name": "paddle"},  # not a category
            "crash": "wipeout",
        },
        "opener": ["scenery", "paddle", "run", "scenery"],
    }
    assert activity.validate({"name": "x", "mix": raw})["mix"] == {
        "categories": {"run": {"name": "big wave", "share": 0.4}, "lift": {"share": 0.0}},
        "opener": ["run", "scenery"],
    }


@pytest.mark.parametrize("raw", [None, {}, "run", {"categories": {"paddle": {}}, "opener": []}])
def test_a_profile_without_a_usable_mix_has_no_mix_key(raw) -> None:
    prof = {"name": "x"} if raw is None else {"name": "x", "mix": raw}
    assert "mix" not in activity.validate(prof)


def test_the_shipped_profiles_have_no_mix() -> None:
    for name in activity.BUILTIN_NAMES:
        assert "mix" not in activity.builtin(name)
    assert "mix" not in activity.full(activity.builtin("generic"))


def test_check_profile_names_what_mix_ignores_and_what_breaks_it() -> None:
    raw = {
        **SURF,
        "mix": {
            "categories": {
                "run": {"name": "", "share": -1, "speed": 1},
                "paddle": {},
                "lift": 3,
            },
            "opener": ["scenery", "paddle"],
            "order": [],
        },
    }
    problems, warnings = activity.check_profile(raw)
    assert problems == []
    assert any("mix: unknown keys ignored: order" in w for w in warnings)
    assert any("mix.categories: unknown categories ignored: paddle" in w for w in warnings)
    assert any("mix.categories.lift: not an object" in w for w in warnings)
    assert any("mix.categories.run: unknown keys ignored: speed" in w for w in warnings)
    assert any("mix.categories.run.name ignored" in w for w in warnings)
    assert any("mix.categories.run.share ignored" in w for w in warnings)
    assert any("mix.opener: unknown categories ignored: paddle" in w for w in warnings)
    assert activity.check_profile(SURF) == ([], [])
    for bad in ("run", {"categories": []}, {"opener": "run"}):
        problems, _ = activity.check_profile({**SURF, "mix": bad})
        assert problems and problems[0].startswith("mix")


def test_create_saves_mix_and_a_context_merge_keeps_it(home) -> None:
    path, warnings = activity.create_profile("surf-test", SURF)
    assert warnings == []
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["mix"]["categories"]["run"] == {"name": "wave", "share": 0.5}
    assert activity.validate_profile("surf-test")["valid"]
    full = activity.full(activity.library()["surf-test"])
    assert full["mix"] == saved["mix"]
    # an LLM addition (--context over a library profile) has no mix: the profile's stays
    llm = activity.validate({"name": "y", "incident": {"pos": ["a board flying"]}})
    assert activity.merge(full, llm)["mix"] == saved["mix"]


def test_project_mix_reads_activity_json(tmp_path: Path) -> None:
    assert activity.project_mix(tmp_path) == {}  # no file
    _record(tmp_path, SURF)
    assert activity.project_mix(tmp_path)["opener"] == ["run", "scenery"]
    rec = cache.load_json(tmp_path / activity.ACTIVITY_FILE)
    cache.save_json(tmp_path / activity.ACTIVITY_FILE, {**rec, "version": 1})
    assert activity.project_mix(tmp_path) == {}  # an older record
    cache.save_json(tmp_path / activity.ACTIVITY_FILE, {**rec, "profile": {"mix": "junk"}})
    assert activity.project_mix(tmp_path) == {}  # a hand-edited record does not crash


# ---------------------------------------------------------------- run_mix
def _files(project: Path) -> dict[str, bytes]:
    return {n: (project / n).read_bytes() for n in ("mix_segments.json", "mix_report.json")}


def test_without_a_profile_mix_the_output_is_the_same(tmp_path: Path, fake_day) -> None:
    """generic, base or a profile without mix: byte-identical to no activity.json."""
    fake_day()
    plain = tmp_path / "plain"
    plain.mkdir()
    mix.run_mix(plain, sheet=False)
    segs = cache.load_json(plain / "mix_segments.json")
    assert not any("opener" in s for s in segs)
    assert not {"names", "off", "opener"} & set(cache.load_json(plain / "mix_report.json"))
    for name, prof in (
        ("generic", activity.builtin("generic")),
        ("base", activity.builtin("base")),
        ("own", {k: v for k, v in SURF.items() if k != "mix"}),
    ):
        _record(tmp_path / name, prof)
        mix.run_mix(tmp_path / name, sheet=False)
        assert _files(tmp_path / name) == _files(plain), name


def test_a_profile_mix_names_weighs_drops_and_opens(tmp_path: Path, fake_day) -> None:
    fake_day()
    _record(tmp_path, SURF)
    rep = mix.run_mix(tmp_path, sheet=False)
    segs = cache.load_json(tmp_path / "mix_segments.json")
    assert {s["category"] for s in segs} >= {"run", "scenery"}
    assert "lift" not in {s["category"] for s in segs}  # share 0: left out
    assert rep["candidates"]["lift"] == 0 and rep["off"] == ["lift"]
    labels = {s["category"]: s["label"] for s in segs if s["label"] != REACTION_LABEL}
    assert labels["run"] == "wave" and labels["scenery"] == "line up"
    assert labels["social"] == "laugh" and labels["jump"] == "jump 1.2s"  # not named
    assert any(s["label"] == REACTION_LABEL for s in segs)  # a crash reaction keeps it
    assert rep["names"] == {"run": "wave", "scenery": "line up"}
    assert rep["opener"] == ["run", "scenery"]
    assert rep["quotas"]["run"] > mix.DEFAULT_QUOTAS["run"]  # the profile's 0.5
    assert {s["category"] for s in segs if s.get("opener")} == {"run", "scenery"}
    assert all(s.get("opener") for s in segs if s["category"] in ("run", "scenery"))
    first = cache.load_json(tmp_path / "mix_segments.json")
    first_id = rep["arc"][0]
    assert next(s for s in first if s["id"] == first_id)["category"] in ("run", "scenery")
    assert rep["content_qa"]["checks"]["opener"]["ok"]


def test_quotas_override_the_profile_share_and_bring_a_category_back(
    tmp_path: Path, fake_day
) -> None:
    fake_day()
    _record(tmp_path, SURF)
    rep = mix.run_mix(tmp_path, sheet=False, quota_spec="lift=.1,run=.2")
    assert rep["candidates"]["lift"] == 8 and "off" not in rep
    assert rep["quotas"]["run"] < 0.35  # --quotas wins over the profile's 0.5


def test_an_earlier_reports_quotas_keep_the_category_off(tmp_path: Path, fake_day) -> None:
    """360 repair runs mix again with the report's quotas, which lack the dropped category."""
    fake_day()
    _record(tmp_path, SURF)
    rep = mix.run_mix(tmp_path, sheet=False)
    again = mix.run_mix(tmp_path, sheet=False, quotas=rep["quotas"])
    assert again["candidates"]["lift"] == 0


def test_the_sheet_captions_use_the_names(tmp_path: Path, fake_day, monkeypatch) -> None:
    fake_day()
    _record(tmp_path, SURF)
    seen: list[str] = []
    monkeypatch.setattr(
        mix, "contact_sheet", lambda items, out, **_k: seen.extend(i["caption"] for i in items)
    )
    mix.run_mix(tmp_path)
    assert any(" wave " in c for c in seen) and not any(" run " in c for c in seen)


# ---------------------------------------------------------------- arc order and QA
def _seg(i: int, cat: str, score: float, **kw) -> dict:
    return {
        "id": f"m{i:03d}",
        "source": "s1",
        "start": 10.0 * i,
        "end": 10.0 * i + 6,
        "score": score,
        "category": cat,
        "label": cat,
        "chapter": 1,
        **kw,
    }


def test_arc_opens_with_the_best_marked_moment() -> None:
    day = [
        _seg(1, "scenery", 0.9),
        _seg(2, "run", 0.5, opener=True),
        _seg(3, "run", 0.8, opener=True),
        _seg(4, "social", 0.7),
        _seg(5, "run", 0.3, opener=True),
        _seg(6, "scenery", 0.6),
        _seg(7, "crash", 0.9),
        _seg(8, "social", 0.4),
        _seg(9, "scenery", 0.9),
    ]
    arc = order_moments(day, "arc")
    assert arc[0]["id"] == "m003"  # best marked of the first third, not the best view
    assert sorted(m["id"] for m in arc) == sorted(m["id"] for m in day)
    assert arc[-1]["id"] == "m009"  # the scenery outro stays


def test_an_opener_crash_is_not_also_the_finale() -> None:
    day = [
        _seg(1, "crash", 0.9, opener=True),
        _seg(2, "run", 0.4),
        _seg(3, "run", 0.8),
        _seg(4, "social", 0.5),
        _seg(5, "run", 0.6),
        _seg(6, "social", 0.3),
    ]
    arc = order_moments(day, "arc")
    assert [m["id"] for m in arc].count("m001") == 1 and arc[0]["id"] == "m001"
    assert sorted(m["id"] for m in arc) == sorted(m["id"] for m in day)
    assert arc[-1]["id"] == "m003"  # the finale falls back to the best run


def test_content_qa_opener_follows_the_profile() -> None:
    segs = [_seg(1, "run", 0.5, opener=True), _seg(2, "scenery", 0.5), _seg(3, "social", 0.5)]
    have = {"run", "scenery", "social"}
    assert content_qa(segs, available=have)["checks"]["opener"]["ok"]  # from the marks
    assert content_qa(segs, available=have, opener=("run",))["checks"]["opener"]["ok"]
    assert not content_qa(segs, available=have, opener=("scenery",))["checks"]["opener"]["ok"]
    plain = [{k: v for k, v in s.items() if k != "opener"} for s in segs]
    assert not content_qa(plain, available=have)["checks"]["opener"]["ok"]  # lift / scenery
    # the profile's opener categories had no candidates: no opener rule
    assert content_qa(plain, available={"run", "social"}, opener=("scenery",))["checks"]["opener"][
        "ok"
    ]


def test_load_segments_keeps_the_opener_mark(tmp_path: Path) -> None:
    src = tmp_path / "a.mp4"
    src.write_bytes(b"")
    segs = [
        {"source": str(src), "start": 0, "end": 5, "category": "run", "opener": True},
        {"source": str(src), "start": 6, "end": 9, "category": "run", "opener": "yes"},
    ]
    cache.save_json(tmp_path / "s.json", segs)
    loaded = load_segments(tmp_path / "s.json")
    assert loaded[0]["opener"] is True and "opener" not in loaded[1]


def test_a_mix_that_leaves_out_every_category_says_so(tmp_path: Path, fake_day) -> None:
    fake_day(lambda: [_c("lift", 10, 1.0, src="l0"), _c("run", 110, 2.0, src="r0")])
    off = {"categories": {"lift": {"share": 0}, "run": {"share": 0}}}
    _record(tmp_path, {**SURF, "mix": off})
    with pytest.raises(ValueError, match=r"Nothing selected.*leaves out lift, run.*--quotas"):
        mix.run_mix(tmp_path, sheet=False)
    fake_day(list)  # no candidates at all: the old message
    with pytest.raises(ValueError, match="No candidates"):
        mix.run_mix(tmp_path, sheet=False)
    _, warnings = activity.check_profile({**SURF, "mix": off})
    assert any("every share is 0" in w for w in warnings)
    _, warnings = activity.check_profile(SURF)
    assert not any("every share is 0" in w for w in warnings)


def test_the_reserved_reaction_name_is_not_a_category_name() -> None:
    assert activity.MIX_RESERVED_NAME == REACTION_LABEL
    mixed = {"categories": {"run": {"name": "Crash  Reaction", "share": 0.4}}}
    assert activity.validate({**SURF, "mix": mixed})["mix"]["categories"]["run"] == {"share": 0.4}
    _, warnings = activity.check_profile({**SURF, "mix": mixed})
    assert any("reserved name 'crash reaction'" in w for w in warnings)


def test_a_non_object_mix_is_one_problem_not_also_an_unknown_key() -> None:
    problems, warnings = activity.check_profile({**SURF, "mix": "run"})
    assert problems and not any("unknown keys ignored: mix" in w for w in warnings)


def test_repair_content_qa_uses_the_reports_opener(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(mix, "content_qa", lambda *_a, **kw: seen.update(kw) or {})
    monkeypatch.setattr(repair, "_fit_budget", lambda *_a: {})
    rep: dict = {}
    repair._rendered_qa(rep, [], {"clips": []}, {"opener": ["run"]})
    assert seen["opener"] == ["run"]
    repair._rendered_qa(rep, [], {"clips": []}, {})
    assert seen["opener"] is None
