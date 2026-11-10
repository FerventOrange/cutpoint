"""Activity prompt sets: the default is ``generic``, library profiles get their own lists on
the floor, lift detection is off without lift captions. The profiles are invented ones in
the user's library (Cutpoint ships only base and generic)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import HOP, SLED, add_profile
from cutpoint import activity, crash, insta360
from cutpoint.events import visual as V
from cutpoint.events.vlm_windows import action_cam_wording as wording
from cutpoint.pano import project as pano_project
from cutpoint.pano import qa, views, vlcheck
from cutpoint.styles import STYLES


def _p(raw: dict) -> activity.Prompts:
    return activity.prompts(activity.full(activity.validate(raw)))


def _generic() -> activity.Prompts:
    return activity.prompts(activity.full(activity.builtin("generic")))


def test_cutpoint_ships_only_the_floor_and_generic():
    shipped = {f.stem for f in activity.BUILTIN_DIR.glob("*.json")}
    assert shipped == set(activity.BUILTIN_NAMES) == {"base", "generic"}
    assert set(activity.library()) == {"generic"}  # an empty user library
    with pytest.raises(KeyError):
        activity.builtin("ski")


def test_the_default_prompt_set_is_generic_without_lifts():
    p = activity.default_prompts()
    assert p == _generic() and p.name == "generic" and not p.lift
    floor = activity.prompts(activity.base())
    for field in ("highlight_pos", "group_pos", "person_neg", "occluder", "low_info", "scene"):
        assert getattr(p, field) == getattr(floor, field), field


def test_crash_params_take_the_profile_unless_given():
    p = crash.CrashParams()
    assert (p.pos, p.neg, p.words) == (None, None, None) and not p.resolved
    sled = activity.full(activity.validate(SLED))
    got = p.with_profile(sled)
    assert got.pos == _p(SLED).incident_pos and "a sled tipped over on the ice" in got.pos
    assert got.key() != p.with_profile(activity.full(activity.builtin("generic"))).key()
    own = crash.CrashParams(pos=("x",)).with_profile(sled)
    assert own.pos == ("x",) and own.neg == tuple(sled["incident"]["neg"])


def test_vision_check_wording_and_cache_keys():
    sled, default = _p(SLED), activity.default_prompts()
    first = vlcheck.prompts_for(default)[0]
    assert first.startswith("One frame from a first-person action video")
    assert vlcheck.prompts_for(sled)[0].startswith("One frame from a first-person sledding video")
    # every wording is hashed, the default's too (its replies are not those of the old default)
    assert len(vlcheck.wording_id(default)) == len(vlcheck.wording_id(sled)) == 8
    assert vlcheck.wording_id(default) != vlcheck.wording_id(sled)
    path = {"keys": [{"t": 0.0, "yaw": 0, "pitch": 0, "fov": 90}]}
    assert vlcheck.frame_key(path, 1.0, "m") == vlcheck.frame_key(path, 1.0, "m", wording="")
    assert vlcheck.frame_key(path, 1.0, "m", wording="ab") != vlcheck.frame_key(path, 1.0, "m")


@pytest.mark.parametrize("raw", [HOP, {"name": "generic"}], ids=["hop", "generic"])
def test_profiles_get_their_own_lists_on_the_floor(raw):
    p = _p(raw) if raw is HOP else _generic()
    floor = activity.prompts(activity.base())
    for field in ("highlight_pos", "group_pos", "person_neg", "occluder", "low_info", "scene"):
        assert set(getattr(floor, field)) <= set(getattr(p, field)), field
    assert not p.lift and p.lift_pos == p.lift_neg == ()
    assert dict(p.vl)["video"] == "action"


def test_exclude_base_replaces_floor_entries_and_lift_captions_turn_lifts_on():
    p = _p(SLED)
    assert "a dark blurry image" not in p.low_info and "a flat white image" in p.low_info
    assert "a dark blurry image" in activity.prompts(activity.base()).low_info
    assert p.lift and p.lift_pos == ("riding a cable car above a glacier",)
    assert p.key() != _generic().key()


def test_profile_highlight_and_low_info_reach_the_consumers():
    hop = _p(HOP)
    assert "a person in mid-air over a puddle" in hop.highlight_pos
    assert "a close-up of wet pavement" in hop.low_info


def _record(tmp_path: Path, rec: dict) -> Path:
    (tmp_path / activity.ACTIVITY_FILE).write_text(json.dumps(rec), encoding="utf-8")
    return tmp_path


def test_project_prompts_reads_activity_json_and_upgrades_old_records(tmp_path):
    assert activity.project_prompts(tmp_path) == activity.default_prompts()  # no file
    hop = activity.full(add_profile(HOP))
    _record(tmp_path, {"version": 1, "name": "puddle-hop", "source": "library", "profile": hop})
    assert activity.project_prompts(tmp_path) == activity.prompts(hop)
    # a record written before profiles had the caption sections: today's library lists
    old = {"name": "puddle-hop", "incident": hop["incident"]}
    old |= {"highlight": {"pos": ["an old highlight"], "neg": []}}
    _record(tmp_path, {"version": 1, "name": "puddle-hop", "source": "library", "profile": old})
    assert activity.project_prompts(tmp_path) == activity.prompts(hop)


def test_a_record_of_a_removed_built_in_keeps_its_stored_profile(tmp_path):
    """Projects analysed with a built-in sport profile Cutpoint no longer ships."""
    stored = activity.full(activity.validate({**SLED, "name": "ski"}))
    _record(tmp_path, {"version": 1, "name": "ski", "source": "library", "profile": stored})
    p = activity.project_prompts(tmp_path)
    assert p == activity.prompts(stored) and p.name == "ski" and p.lift


def test_use_prompts_sets_the_active_set_for_the_block():
    hop = _p(HOP)
    assert activity.active_prompts() == activity.default_prompts()
    with activity.use_prompts(hop):
        assert activity.active_prompts() is hop
        assert qa.qa_negatives() == hop.qa_negatives()
    assert activity.active_prompts() == activity.default_prompts()


def test_synthesized_profiles_take_the_floor_for_the_sections_the_llm_does_not_write():
    raw = activity.validate({"name": "x", "incident": {"pos": ["a"]}})
    new = {k: raw[k] for k in activity.SCHEMA_KEYS} | {"exclude_base": []}
    p = activity.prompts(activity.full(new))
    assert p.occluder == activity.prompts(activity.base()).occluder and not p.lift
    raw = activity.validate({"name": "x", "lift": {"pos": ["a cable car"], "neg": ["the ice"]}})
    new = {k: raw[k] for k in activity.SCHEMA_KEYS} | {"exclude_base": []}
    assert activity.prompts(activity.full(new)).lift  # discovery may write lift captions


# ---------------------------------------------------------------- vlm wording (#7, #511)
def test_vlm_wording_of_a_profile_replaces_the_floor_value_by_value():
    floor = activity.prompts(activity.base())
    assert set(dict(floor.vlm)) == set(activity.VLM_KEYS)  # the floor words every value
    sled = _p(SLED)
    vlm, tags = dict(sled.vlm), dict(sled.vlm_tags)
    assert vlm["video"] == "first-person glacier sledding video"
    assert vlm["edit"] == "sledding highlight edit" and vlm["hype"] == dict(floor.vlm)["hype"]
    assert tags["transit"] == "on a cable car" and tags["trees"] == dict(floor.vlm_tags)["trees"]
    for p in (_p(HOP), _generic()):  # no wording of their own
        assert dict(p.vlm) == dict(floor.vlm) and p.vlm_tags == floor.vlm_tags


def test_vlm_wording_does_not_change_the_caption_key():
    p = _p(SLED)
    assert dataclasses.replace(p, vlm=(), vlm_tags=()).key() == p.key()


def test_invalid_or_too_long_vlm_values_fall_back_to_the_floor_one_by_one():
    floor = activity.prompts(activity.base())
    raw = {
        "name": "x",
        "vlm": {
            "video": "  first-person   kayak video ",  # kept, on one line
            "edit": "x" * (activity.MAX_VLM_CHARS + 1),  # over the cap: the floor's
            "hype": 7,  # not a string: the floor's
            "bogus": "an unknown key",
            "tags": {"trees": "paddling past trees", "rapids": "a new tag", "indoor_break": ""},
        },
    }
    prof = activity.validate(raw)
    assert prof["vlm"] == {
        "video": "first-person kayak video",
        "tags": {"trees": "paddling past trees"},
    }
    p = activity.prompts(activity.full(prof))
    vlm, tags = dict(p.vlm), dict(p.vlm_tags)
    assert vlm["video"] == "first-person kayak video"
    assert vlm["edit"] == dict(floor.vlm)["edit"] and vlm["hype"] == dict(floor.vlm)["hype"]
    assert (
        tags["trees"] == "paddling past trees"
        and tags["indoor_break"] == dict(floor.vlm_tags)["indoor_break"]
    )
    assert "rapids" not in tags and "bogus" not in vlm
    # a hand-edited activity.json record is read the same way (not validated on load)
    rec_prof = {**activity.full(activity.validate(HOP)), "vlm": {"edit": 3, "video": "a"}}
    p = activity.prompts(rec_prof)
    assert dict(p.vlm)["edit"] == dict(floor.vlm)["edit"] and dict(p.vlm)["video"] == "a"


def test_old_records_without_vlm_take_their_library_wording_else_the_floor(tmp_path):
    sled = {k: v for k, v in activity.full(add_profile(SLED)).items() if k != "vlm"}
    _record(tmp_path, {"version": 1, "name": "glacier-sled", "source": "library", "profile": sled})
    assert activity.project_prompts(tmp_path).vlm == _p(SLED).vlm
    # an LLM-written profile (since #588: every caption section) keeps its own captions
    own = {**sled, "name": "kayak", "highlight": {"pos": ["a kayak in white water"], "neg": []}}
    _record(tmp_path, {"version": 1, "name": "kayak", "source": "synthesized", "profile": own})
    p = activity.project_prompts(tmp_path)
    assert p.highlight_pos == ("a kayak in white water",)
    assert p.vlm == activity.prompts(activity.base()).vlm


def test_a_discovered_profile_keeps_the_base_vlm_wording(monkeypatch):
    """#916: synthesis asks for no ``vlm`` wording and ignores one in the reply, so the VLM
    prompts (and their window cache key) of a discovered profile are generic's."""
    asked: list[tuple[str, dict | None]] = []
    reply = {
        "name": "kayak",
        "description": "Whitewater kayaking.",
        "detect": ["a kayak on a river"],
        "incident": {"pos": ["a capsized kayak"], "neg": ["paddling"], "words": {"swim": 0.5}},
        "highlight": {"pos": ["a big rapid"], "neg": []},
        "low_info": [],
        "vlm": {"video": "first-person whitewater kayaking video", "tags": {"spray": "x"}},
    }

    def ask(prompt, imgs, schema):
        asked.append((prompt, schema))
        return reply if len(asked) > 1 else None  # the retry asks the same keys

    seeds = [activity.builtin(k) for k in activity.BUILTIN_NAMES]
    prof = activity.synthesize("whitewater kayaking", "my river trip", ask, seeds)
    assert prof is not None and len(asked) == 2
    for prompt, schema in asked:
        assert "vlm" not in schema["properties"] and '"vlm"' not in prompt
        assert json.dumps(activity.base()["vlm"]) not in prompt
    assert "vlm" not in prof
    generic = activity.prompts(activity.full(activity.builtin("generic")))
    p = activity.prompts(activity.full(prof))
    assert (p.vlm, p.vlm_tags) == (generic.vlm, generic.vlm_tags)
    assert wording(p).key() == wording(generic).key()


def _fake_text(monkeypatch):
    """Text embeddings: one axis per caption (so every softmax is defined)."""
    axes: dict[str, int] = {}

    def text_embeddings(prompts, models=None):
        e = np.zeros((len(prompts), 768), np.float32)
        for k, s in enumerate(prompts):
            e[k, axes.setdefault(s, len(axes))] = 1.0
        return e

    monkeypatch.setattr(V, "text_embeddings", text_embeddings)
    return axes


def test_lift_logic_is_off_without_lift_captions(monkeypatch):
    _fake_text(monkeypatch)
    pv = SimpleNamespace(t=np.arange(8.0), emb=np.ones((8, 9, 768), np.float16) / 768**0.5)
    assert not views.lift_scores(pv, _generic()).any()
    assert views.lift_scores(pv, _p(SLED)).shape == (8,)
    monkeypatch.setattr(views, "_grid_views", lambda src: pv)
    assert views.lift_grid(Path("x.lrv"), 10) is None  # the default (generic): no lifts
    with activity.use_prompts(_p(SLED)):
        assert views.lift_grid(Path("x.lrv"), 10) is not None  # lift captions: looked for


def test_view_and_occlusion_scores_use_the_given_or_active_prompts(monkeypatch):
    axes = _fake_text(monkeypatch)
    monkeypatch.setattr(V, "aesthetic", lambda e: np.zeros(e.shape[:-1]))
    monkeypatch.setattr(views, "wearer_view_mask", lambda pv: None)
    hop = _p(HOP)
    hit = np.zeros((3, 9, 768), np.float32)
    hit[..., axes.setdefault(hop.highlight_pos[-1], len(axes))] = 1.0  # a hop highlight
    pv = SimpleNamespace(t=np.arange(3.0), emb=hit, meta={"source": "a.lrv"}, det=np.zeros((0, 11)))
    with activity.use_prompts(hop):
        assert views.view_scores(pv)["beauty_zs"].min() > 0.99
        occl = views.occlusion_scores(pv)
    assert views.view_scores(pv, _generic())["beauty_zs"].max() < 0.5  # cached per prompt set
    assert occl["occl_zs"].shape == (3, 8) and occl["selfie_ok"].shape == (3,)


def test_default_360_style_is_the_action_cam_scorer(monkeypatch):
    monkeypatch.setattr(pano_project, "pano_config", dict)
    name = pano_project.default_style()
    assert name == "action-cam" and STYLES[name].kind != "generic"
    monkeypatch.setattr(pano_project, "pano_config", lambda: {"default_style": "funny"})
    assert pano_project.default_style() == "funny"


def test_core_reads_x3_names_without_the_pano_package():
    assert insta360.parse_name("LRV_20240101_120000_11_002.lrv")["num"] == "002"
