"""Activity prompt sets: the ski profile reproduces the old hard-coded lists exactly,
other activities get their own, lift detection is off without lift captions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint import activity, crash, insta360
from cutpoint.events import visual as V
from cutpoint.pano import project as pano_project
from cutpoint.pano import qa, views, vlcheck, x3
from cutpoint.styles import STYLES

# The lists as they were hard-coded before profiles held them (events/visual.py,
# pano/qa.py, pano/views.py, crash.py): ski output depends on them, keep them pinned.
OLD = {
    "highlight_pos": (
        "a breathtaking mountain vista",
        "sunlit snowy mountain peaks under a blue sky",
        "snow covered trees in a winter forest",
        "a view over a valley from a mountain ridge",
        "alpenglow on snowy mountains at sunset",
        "a scenic panorama of mountains and a ski resort",
        "clouds and fog below a mountain summit",
        "a beautiful winter landscape",
    ),
    "highlight_neg": (
        "a close-up of ski boots",
        "a close-up of skis on the snow",
        "blurry snow",
        "the safety bar of a chairlift",
        "a phone screen",
        "a dark blurry image",
        "a close-up of a glove",
        "the inside of a building",
        "a flat white foggy image",
        "a close-up of a jacket",
        "a groomed ski run seen while skiing",
    ),
    "group_pos": (
        "a group of friends standing together in the snow",
        "friends sitting next to me on a chairlift",
        "a selfie of friends wearing ski helmets and goggles",
        "a person in ski clothes talking to the camera",
        "people laughing together",
        "a friend skiing right in front of the camera",
    ),
    "group_neg": (
        "an empty ski slope",
        "a skier far away in the distance",
        "a close-up of ski boots",
        "snowy trees with nobody around",
        "an empty chairlift",
        "a mountain landscape",
    ),
    "person_neg": (  # SELF_POS
        "a first-person view looking down at my own legs and ski boots",
        "my own arm and glove holding a ski pole",
        "my own skis seen from above",
        "my own snowboard and boots seen from above",
        "the shadow of a person on the snow",
        "a blurry close-up of a jacket sleeve",
    ),
    "person_pos": (  # OTHER_POS
        "a person wearing a ski helmet and goggles",
        "a skier",
        "a snowboarder",
        "a person sitting on a chairlift",
        "a person's face",
        "a person standing in the snow",
        "a group of people",
    ),
    "occluder": (  # OCCL_POS
        "a close-up of a glove",
        "a hand covering the camera lens",
        "a blurry close-up of a jacket sleeve",
        "a ski helmet filling the frame",
        "a person's face extremely close to the camera",
        "a backpack strap close-up",
    ),
    "low_info": (  # LOWINFO_POS
        "a flat white foggy image",
        "an empty white snow surface",
        "a plain sky",
        "a blurry image of snow",
        "a dark blurry image",
    ),
    "scene": ("a skier on a slope", "people standing in the snow", "a chairlift"),
    "lift_pos": (
        "riding a chairlift with a view over the trees",
        "the safety bar of a chairlift",
        "people sitting on a chairlift",
        "chairlift cable and towers",
        "inside a gondola cabin",
    ),
    "lift_neg": (
        "skiing down a snowy slope",
        "skiers standing on a ski run",
        "a snowy mountain landscape",
        "trees covered in snow",
        "the inside of a building",
        "a trail map sign board",
        "a ski resort information sign",
    ),
    "selfie_pos": (  # SELFIE_OK_POS
        "a selfie of a skier wearing a helmet and goggles",
        "a person's face wearing ski goggles",
        "a skier seen from above",
    ),
    "selfie_neg": ("the top of a helmet",),
}
OLD_VL_PROMPT = (
    "One frame from a first-person ski video (a virtual camera aimed inside a 360 video). "
    "Decide: blocked = something very close to the lens covers part of the view (a hand, "
    "arm, glove or mitten, ski pole, a shoulder or jacket strip at the frame edge, goggles "
    "or a face filling the frame). The friend the camera is framing is the subject, NOT "
    "blocked, even close up or wearing goggles, unless the frame edge cuts through them or "
    "their hand or pole reaches at the lens. People at a normal distance, snow, sky and "
    "trees are NOT blocked. low_info = the frame shows nothing worth watching (a close-up of a "
    "sign, board or wall, only snow or sky, heavy blur). Answer immediately, no reasoning, "
    'JSON only: {"blocked": true|false, "where": "left|right|center|bottom|top|none", '
    '"what": "<2-5 words>", "low_info": true|false}'
)
OLD_VL_RETRY = (
    "Ski video frame. Answer immediately, no reasoning. blocked = a hand, arm, glove, pole, "
    "goggles or a jacket edge right at the lens. low_info = a sign close-up, only snow or "
    'sky, or blur. Output only JSON: {"blocked": true|false, "where": '
    '"left|right|center|bottom|top|none", "what": "<2-5 words>", "low_info": true|false}'
)


def _ski() -> activity.Prompts:
    return activity.prompts(activity.full(activity.builtin("ski")))


def test_ski_profile_reproduces_the_old_lists_exactly():
    p = _ski()
    for field, old in OLD.items():
        assert getattr(p, field) == old, field
    assert p.qa_negatives() == OLD["group_pos"] + OLD["highlight_pos"] + OLD["scene"]
    assert activity.default_prompts() == p  # callers without a project keep the ski lists


def test_ski_crash_params_and_keys_are_unchanged():
    p = crash.CrashParams()
    assert (p.pos, p.neg, p.words) == (None, None, None) and not p.resolved
    ski = p.with_profile(activity.full(activity.builtin("ski")))
    assert ski.pos == _ski().incident_pos and len(ski.pos) == 10
    assert ski.key() == "e7244b388a" and ski.dense_key() == "24551ce592"
    own = crash.CrashParams(pos=("x",)).with_profile(activity.full(activity.builtin("mtb")))
    assert own.pos == ("x",) and own.neg == tuple(
        activity.full(activity.builtin("mtb"))["incident"]["neg"]
    )


def test_ski_vision_check_wording_and_cache_keys_are_unchanged():
    assert vlcheck.prompts_for(_ski()) == (OLD_VL_PROMPT, OLD_VL_RETRY)
    assert vlcheck.wording_id(_ski()) == ""
    mtb = activity.prompts(activity.full(activity.builtin("mtb")))
    first, retry = vlcheck.prompts_for(mtb)
    assert "ski" not in first.lower() and "ski" not in retry.lower()
    assert first.startswith("One frame from a first-person action video")
    assert len(vlcheck.wording_id(mtb)) == 8
    path = {"keys": [{"t": 0.0, "yaw": 0, "pitch": 0, "fov": 90}]}
    assert vlcheck.frame_key(path, 1.0, "m") == vlcheck.frame_key(path, 1.0, "m", wording="")
    assert vlcheck.frame_key(path, 1.0, "m", wording="ab") != vlcheck.frame_key(path, 1.0, "m")


@pytest.mark.parametrize("name", ["mtb", "generic", "surf", "hike"])
def test_other_activities_get_their_own_lists_from_the_floor(name):
    p = activity.prompts(activity.full(activity.builtin(name)))
    floor = activity.prompts(activity.base())
    for field in ("highlight_pos", "group_pos", "person_neg", "occluder", "low_info", "scene"):
        lst = getattr(p, field)
        assert lst and not any("ski" in s or "chairlift" in s for s in lst), field
        assert set(getattr(floor, field)) <= set(lst), field
    assert not p.lift and p.lift_pos == p.lift_neg == ()
    assert dict(p.vl)["video"] == "action"
    assert p.key() != _ski().key()


def test_profile_highlight_and_low_info_reach_the_consumers():
    mtb = activity.prompts(activity.full(activity.builtin("mtb")))
    assert "a mountain biker in mid-air off a jump" in mtb.highlight_pos
    assert "a close-up of a dirt trail" in mtb.low_info


def _record(tmp_path: Path, rec: dict) -> Path:
    (tmp_path / activity.ACTIVITY_FILE).write_text(json.dumps(rec), encoding="utf-8")
    return tmp_path


def test_project_prompts_reads_activity_json_and_upgrades_old_records(tmp_path):
    assert activity.project_prompts(tmp_path) == activity.default_prompts()  # no file
    mtb = activity.full(activity.builtin("mtb"))
    _record(tmp_path, {"version": 1, "name": "mtb", "source": "library", "profile": mtb})
    assert activity.project_prompts(tmp_path) == activity.prompts(mtb)
    # a ski record written before profiles had the caption sections: today's ski lists
    old = {k: v for k, v in activity.full(activity.builtin("ski")).items() if k == "incident"}
    old |= {"name": "ski", "highlight": {"pos": ["a skier in mid-air"], "neg": []}}
    _record(tmp_path, {"version": 1, "name": "ski", "source": "library", "profile": old})
    assert activity.project_prompts(tmp_path) == _ski()


def test_use_prompts_sets_the_active_set_for_the_block():
    mtb = activity.prompts(activity.full(activity.builtin("mtb")))
    assert activity.active_prompts() == activity.default_prompts()
    with activity.use_prompts(mtb):
        assert activity.active_prompts() is mtb
        assert qa.qa_negatives() == mtb.qa_negatives()
    assert activity.active_prompts() == activity.default_prompts()


def test_synthesized_profiles_take_the_floor_for_the_sections_the_llm_does_not_write():
    raw = activity.validate({"name": "x", "incident": {"pos": ["a"]}})
    new = {k: raw[k] for k in activity.SCHEMA_KEYS} | {"exclude_base": []}
    p = activity.prompts(activity.full(new))
    assert p.occluder == activity.prompts(activity.base()).occluder and not p.lift


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
    generic = activity.prompts(activity.full(activity.builtin("generic")))
    assert not views.lift_scores(pv, generic).any()
    assert views.lift_scores(pv, _ski()).shape == (8,)
    with activity.use_prompts(generic):
        assert views.lift_grid(Path("x.lrv"), 10) is None
    monkeypatch.setattr(views, "_grid_views", lambda src: pv)
    assert views.lift_grid(Path("x.lrv"), 10) is not None  # ski (default): looked for


def test_view_and_occlusion_scores_use_the_given_or_active_prompts(monkeypatch):
    axes = _fake_text(monkeypatch)
    monkeypatch.setattr(V, "aesthetic", lambda e: np.zeros(e.shape[:-1]))
    monkeypatch.setattr(views, "wearer_view_mask", lambda pv: None)
    mtb = activity.prompts(activity.full(activity.builtin("mtb")))
    hit = np.zeros((3, 9, 768), np.float32)
    hit[..., axes.setdefault(mtb.highlight_pos[-1], len(axes))] = 1.0  # an mtb highlight
    pv = SimpleNamespace(t=np.arange(3.0), emb=hit, meta={"source": "a.lrv"}, det=np.zeros((0, 11)))
    with activity.use_prompts(mtb):
        assert views.view_scores(pv)["beauty_zs"].min() > 0.99
        occl = views.occlusion_scores(pv)
    assert views.view_scores(pv, _ski())["beauty_zs"].max() < 0.5  # cached per prompt set
    assert occl["occl_zs"].shape == (3, 8) and occl["selfie_ok"].shape == (3,)


def test_default_360_style_is_derived_not_named(monkeypatch):
    monkeypatch.setattr(pano_project, "pano_config", dict)
    name = pano_project.default_style()
    assert STYLES[name].kind != "generic"
    monkeypatch.setattr(pano_project, "pano_config", lambda: {"default_style": "funny"})
    assert pano_project.default_style() == "funny"


def test_core_reads_x3_names_without_the_pano_package():
    assert x3.parse_name is insta360.parse_name and x3.imu_source is insta360.imu_source
    assert insta360.parse_name("LRV_20240101_120000_11_002.lrv")["num"] == "002"
