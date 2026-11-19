"""The feature registry: every style's weights name registered features, and scores are
renormalised over the features an install can compute (lite: no ``events`` extra)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cutpoint import actioncam, analyze, extras, features, reset_state, scorer, styles
from cutpoint.events import scoring
from cutpoint.features import Feature
from cutpoint.styles import STYLES, get_style, load_learned

GENERIC = sorted(n for n, s in STYLES.items() if s.kind == "generic")


def write_style(folder: Path, data: dict) -> Path:
    """A schema-1 style file ``<folder>/<name>.json`` of ``data``."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{data['name']}.json"
    path.write_text(json.dumps({"schema": 1, "source": "fitted", **data}), encoding="utf-8")
    return path


@pytest.mark.parametrize("name", sorted(STYLES))
def test_builtin_style_weights_are_registered(name):
    assert styles.unknown_weights(STYLES[name].weights) == []
    styles.validate(name, STYLES[name].weights)


def test_every_ski_feature_is_registered():
    """refmatch fits weights over actioncam.FEATURES, so a learned ski preset may name any of them."""
    assert set(actioncam.FEATURES) <= set(features.REGISTRY)


def test_learned_style_with_ski_features_loads(tmp_path):
    path = write_style(
        tmp_path,
        {"name": "mine", "base": "action-cam", "weights": dict.fromkeys(actioncam.FEATURES, 0.1)},
    )
    assert load_learned(path).weights["wipeout"] == 0.1


def test_learned_style_with_unknown_feature_is_unusable(home):
    """A hand-edited style naming features nothing measures fails, naming them."""
    path = write_style(
        home / "styles",
        {"name": "odd", "base": "gaming", "weights": {"lodness": 0.4, "vlm_window": 0.1}},
    )
    with pytest.raises(ValueError, match="'lodness', 'vlm_window'") as err:
        get_style("odd")
    assert str(path) in str(err.value)
    assert scoring.style_and_base("odd") == ("odd",)  # the library cannot answer for it


def test_validate_names_the_style():
    with pytest.raises(ValueError, match="x: unknown feature"):
        styles.validate("x", {"loudness": 1.0, "nope": 0.1})


def test_validate_rejects_a_bonus_feature_and_lists_grid_features_only():
    with pytest.raises(ValueError, match="'vlm_window'") as err:
        styles.validate("x", {"vlm_window": 0.1})
    known = str(err.value).split("known: ", 1)[1].split(", ")
    assert known == features.names("grid")
    assert "vlm_window" not in known


def test_divisor_counts_names_that_are_not_grid_features():
    weights = {"loudness": 0.5, "vlm": 0.25, "lodness": 0.125, "vlm_window": 0.125}
    assert features.divisor(weights, features.available(())) == 0.75


def test_get_and_names():
    assert features.get("loudness").layer == "audio"
    assert "vlm_window" in features.names("bonus")
    assert "vlm_window" not in features.names("grid")
    with pytest.raises(ValueError, match="Unknown feature 'nope'"):
        features.get("nope")


@pytest.mark.parametrize(
    "feature",
    [Feature("x", "nowhere", "d"), Feature("x", "audio", "d", kind="other")],
)
def test_register_rejects_bad_layer_or_kind(feature):
    with pytest.raises(ValueError):
        features.register(feature)
    assert "x" not in features.REGISTRY


def test_register_rejects_a_different_feature_of_the_same_name():
    with pytest.raises(ValueError, match="already registered"):
        features.register(Feature("loudness", "visual", "something else"))
    assert features.register(features.get("loudness")) is features.get("loudness")


def test_available_by_installed_extras():
    lite, full = features.available(()), features.available(("events",))
    assert full == set(features.REGISTRY) - set(features.names("part"))
    assert {"loudness", "speech", "laughter", "continuity", "scene", "airtime", "social"} <= lite
    assert not {"vlm", "beauty", "voice_laughter", "vlm_window"} & lite


@pytest.mark.parametrize(("edition", "expected"), [("full", {"events"}), ("lite", set())])
def test_installed_extras_follow_the_edition_once_per_run(edition, expected, monkeypatch):
    calls = []
    monkeypatch.setattr(extras, "edition", lambda: calls.append(1) or edition)
    reset_state()
    assert features.installed_extras() == expected
    assert features.installed_extras() == expected
    assert len(calls) == 1
    assert features.available() == features.available(expected)


def test_divisor_and_score_renormalise_over_available_features():
    weights = {"loudness": 0.4, "vlm": 0.4, "speech": 0.2, "scene": 0.0}
    assert features.divisor(weights, None) == 1.0
    assert features.divisor(weights, features.available(("events",))) == 1.0
    assert features.divisor(weights, features.available(())) == pytest.approx(0.6)
    assert features.to_score(0.3, weights, features.available(())) == 50.0
    assert features.to_score(0.3, weights, None) == 30.0
    assert features.divisor({"vlm": 0.5}, features.available(())) == 1.0  # nothing left


@pytest.mark.parametrize("name", GENERIC)
def test_generic_styles_score_the_same_on_lite_and_full(name):
    w = STYLES[name].weights
    for peak in (0.05, 0.123456, 0.5, 0.97):
        lite = features.to_score(peak, w, features.available(()))
        full = features.to_score(peak, w, features.available(("events",)))
        assert lite == full == round(100 * (peak / sum(w.values())), 1)


def test_combine_adds_in_the_given_order():
    a, b = np.array([0.1, 0.7]), np.array([0.2, 0.3])
    out = features.combine({"a": a, "b": b}, {"a": 0.5, "c": 1.0}, ("a", "b"))
    assert list(out) == list(0 + 0.5 * a + 0.0 * b)


def test_lite_renormalises_a_learned_style_with_an_events_feature(home, monkeypatch):
    write_style(home / "styles", {"name": "mix", "base": "gaming", "weights": {"beauty": 1.0}})
    style = get_style("mix")
    n = 100
    loud = np.zeros(n)
    loud[40:50] = 1.0
    sig = {k: np.zeros(n) for k in ("loudness", "speech", "laughter", "scene", "continuity")}
    sig["loudness"] = loud

    def top(edition: str) -> float:
        monkeypatch.setattr(extras, "edition", lambda: edition)
        reset_state()
        ps = analyze.PartSignals(n, dict(sig), [], [], [], [], [])
        res = analyze.SourceResult(Path("a.mp4"), None, 0.0, n / 10, [], [], None, ps)
        return analyze.pick_candidates(res, style, 5, {})[0].score

    assert top("full") == 22.5  # 0.45 / (1.0 + 1.0)
    assert top("lite") == 45.0  # beauty cannot run: 0.45 / 1.0


def test_ski_score_is_renormalised_like_the_other_styles():
    assert scorer.available_for() == features.available()


def test_ski_features_come_from_the_registry_in_the_grid_order():
    assert actioncam.FEATURES == features.ACTION_CAM_FEATURES
    assert all(features.get(f).kind == "grid" for f in actioncam.FEATURES)
    assert "laughter" not in actioncam.FEATURES and "laughter" not in STYLES["action-cam"].weights


def test_learned_action_cam_preset_with_the_old_laughter_name_fails_plainly(tmp_path):
    """Before #514 phase 1b voice_laughter was called "laughter": such a file names the fix."""
    path = write_style(
        tmp_path, {"name": "mine", "base": "action-cam", "weights": {"laughter": 0.6}}
    )
    with pytest.raises(ValueError, match=r'mine\.json.*"laughter".*write "voice_laughter"'):
        load_learned(path)


def test_learned_generic_preset_keeps_transcript_laughter(tmp_path):
    path = write_style(tmp_path, {"name": "mine", "base": "funny", "weights": {"laughter": 0.6}})
    assert load_learned(path).weights["laughter"] == 0.6
    assert "voice_laughter" not in load_learned(path).weights


def test_old_ski_features_file_with_laughter_is_refused_and_treated_as_missing(tmp_path, caplog):
    """ski_features saved before the rename name the fix (refmatch, mix) and are a cache miss
    for load_part."""
    grids = {k: np.full(4, i / 4) for i, k in enumerate(("airtime", "laughter", "voice"))}
    src = tmp_path / "a.mp4"
    path = actioncam.save_features(tmp_path, src, 0.0, 0.4, grids, np.zeros(4), np.zeros(4), [])
    with pytest.raises(ValueError, match=r"grid 'laughter', now 'voice_laughter'.*analyze") as e:
        actioncam.load_project_features(tmp_path)
    assert str(path) in str(e.value)
    assert actioncam.load_part(tmp_path, src) is None
    assert "treated as missing" in caplog.text and "'laughter'" in caplog.text


def test_social_is_core_and_records_its_two_parts():
    social = features.get("social")
    assert social.extra is None and social.kind == "grid"
    assert social.parts == ("social_said", "social_heard")
    said, heard = (features.get(p) for p in social.parts)
    assert (said.kind, said.layer, said.extra) == ("part", "transcript", None)
    assert (heard.kind, heard.layer, heard.extra) == ("part", "events", "events")
    assert features.names("part") == ["social_heard", "social_said"]


@pytest.mark.parametrize("installed", [(), ("events",)], ids=["lite", "full"])
def test_parts_are_not_available_weighted_or_ski_grids(installed):
    parts = set(features.names("part"))
    assert not parts & features.available(installed)
    assert not parts & set(features.names("grid")) and not parts & set(actioncam.FEATURES)
    with pytest.raises(ValueError, match="'social_said'"):
        styles.validate("x", {"social_said": 0.1})
    w = {"airtime": 0.5, "social_heard": 0.5}  # a hand-edited preset: an unknown weight
    assert features.divisor(w, features.available(installed)) == 1.0


def test_register_rejects_parts_that_are_not_part_features():
    with pytest.raises(ValueError, match="not registered part features"):
        features.register(Feature("x", "audio", "d", parts=("loudness",)))
    with pytest.raises(ValueError, match="not registered part features"):
        features.register(Feature("x", "audio", "d", parts=("nope",)))
    assert "x" not in features.REGISTRY


def test_lite_ski_peak_tops_out_at_exactly_100():
    """social counts in the lite divisor (its transcript part runs on lite), so every grid a
    lite install computes at its maximum, 1.0, scores exactly 100 before any cap (#556)."""
    w = STYLES["action-cam"].weights
    lite = features.available(())
    assert "social" in lite and "social" not in features.dropped(w, lite)
    assert features.divisor(w, lite) == pytest.approx(0.72)
    n = 10
    grids = {k: np.full(n, 1.0 if k in lite else 0.0) for k in actioncam.FEATURES}
    peak = float(scorer.action_cam_combined(grids, w, np.ones(n)).max())
    # to_score's quotient before its cap; the grid sum and the divisor differ by one ulp
    # (sequential float64 sum, Python's compensated sum()), which the rounding drops.
    assert peak / features.divisor(w, lite) == pytest.approx(1.0, rel=1e-12, abs=0)
    assert round(100 * (peak / features.divisor(w, lite)), 1) == 100.0
    assert features.to_score(peak, w, lite) == 100.0
    assert features.dropped(w, features.available(("events",))) == []


def test_lite_ski_score_is_still_capped_at_100():
    """The cap guards a left-out feature's grid a full install cached in a shared project."""
    w = STYLES["action-cam"].weights
    lite = features.available(())
    peak = sum(v for k, v in w.items() if k in lite) + w["vlm"]
    assert features.to_score(peak, w, lite) == 100.0
