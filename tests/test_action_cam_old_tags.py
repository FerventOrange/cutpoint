"""Tag keys from before the generic action-cam names (powder_spray, fast_groomer, chairlift,
lodge) have no alias: events, profile and cache data that still carry them load, and the old
keys are ignored."""

import json

import numpy as np

from cutpoint import actioncam, activity
from cutpoint.events import vlm_windows as vw
from test_actioncam import _features, _win

OLD = ("powder_spray", "fast_groomer", "chairlift", "lodge")


def test_old_keys_are_not_tags():
    assert not set(OLD) & set(vw.ACTION_CAM_TAGS)
    assert {"spray", "fast_open", "transit", "indoor_break"} <= set(vw.ACTION_CAM_TAGS)


def test_a_vlm_reply_with_old_tag_keys_scores_the_new_keys_zero():
    reply = {
        "summary": "s",
        "events": [],
        "highlight_score": 5,
        "funny_score": 0,
        "label": "run",
        "tags": {"powder_spray": 9, "chairlift": 8, "jump_air": 7, "spray": 4},
    }
    out = vw.parse_reply(json.dumps(reply))
    assert out is not None
    assert set(out["tags"]) == set(vw.ACTION_CAM_TAGS)  # no old key is carried over
    assert out["tags"]["jump_air"] == 7 and out["tags"]["spray"] == 4
    assert out["tags"]["transit"] == 0 and out["tags"]["indoor_break"] == 0


def test_a_profile_with_old_tag_descriptions_loads_and_drops_them():
    raw = {
        "vlm": {"tags": {"powder_spray": "snow flying", "lodge": "in the hut", "trees": "in trees"}}
    }
    tags = activity.validate({"name": "mine", **raw}).get("vlm", {}).get("tags", {})
    assert tags == {"trees": "in trees"}
    _problems, warnings = activity.check_profile({"name": "mine", **raw})
    assert any("powder_spray" in w and "lodge" in w for w in warnings)  # said, not an error


def test_cached_windows_with_old_tags_load_without_effect(monkeypatch):
    monkeypatch.setattr(actioncam, "layer_features", lambda *_a, **_k: {})
    old = _features(monkeypatch, [_win(0.0, 30.0, powder_spray=10, chairlift=10, lodge=10)])
    none = _features(monkeypatch, [_win(0.0, 30.0)])
    assert old[0].keys() == none[0].keys()
    for k in old[0]:
        assert np.array_equal(old[0][k], none[0][k]), k
    assert np.array_equal(old[1], none[1]) and np.array_equal(old[2], none[2])
