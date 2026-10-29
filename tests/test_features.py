"""The feature registry: every style's weights name registered features, and scores are
renormalised over the features an install can compute (lite: no ``events`` extra)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import messages
from cutpoint import analyze, extras, features, reset_state, scorer, ski, styles
from cutpoint.events import scoring
from cutpoint.features import Feature
from cutpoint.styles import STYLES, get_style, load_learned

GENERIC = sorted(n for n, s in STYLES.items() if s.kind == "generic")


def write_style(folder: Path, data: dict) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{data['name']}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("name", sorted(STYLES))
def test_builtin_style_weights_are_registered(name):
    assert styles.unknown_weights(STYLES[name].weights) == []
    styles.validate(name, STYLES[name].weights)


def test_every_ski_feature_is_registered():
    """refmatch fits weights over ski.FEATURES, so a learned ski preset may name any of them."""
    assert features.unknown(ski.FEATURES) == []
    assert features.SKI_ALIASES["laughter"] in features.REGISTRY


def test_learned_style_with_ski_features_loads(tmp_path):
    path = write_style(
        tmp_path, {"name": "mine", "base": "ski", "weights": dict.fromkeys(ski.FEATURES, 0.1)}
    )
    assert load_learned(path).weights["wipeout"] == 0.1


def _sig(n: int = 100) -> dict[str, np.ndarray]:
    sig = {k: np.zeros(n) for k in ("loudness", "speech", "laughter", "scene", "continuity")}
    sig["loudness"][40:50] = 1.0
    return sig


def _top_score(style, sig: dict[str, np.ndarray]) -> float:
    n = len(sig["loudness"])
    ps = analyze.PartSignals(n, dict(sig), [], [], [], [], [])
    res = analyze.SourceResult(Path("a.mp4"), None, 0.0, n / 10, [], [], None, ps)
    return analyze.pick_candidates(res, style, 5, {})[0].score


@pytest.mark.parametrize("edition", ["full", "lite"])
def test_learned_style_with_unknown_feature_loads_with_a_warning(
    home, monkeypatch, caplog, edition
):
    """A hand-edited preset keeps its unknown names: they count in the divisor, as before the
    registry, and add nothing to the grid."""
    monkeypatch.setattr(extras, "edition", lambda: edition)
    reset_state()
    path = write_style(
        home / "styles",
        {"name": "odd", "base": "gaming", "weights": {"lodness": 0.4, "vlm_window": 0.1}},
    )
    style = get_style("odd")
    get_style("odd")  # one analyze run loads the preset many times: one warning
    assert style.weights["lodness"] == 0.4
    assert style.weights["vlm_window"] == 0.1
    [msg] = messages(caplog, logger="cutpoint.styles")
    assert str(path) in msg
    assert "features 'lodness', 'vlm_window'" in msg
    assert "their weights still count" in msg
    peak = STYLES["gaming"].weights["loudness"]
    assert _top_score(style, _sig()) == round(100 * (peak / sum(style.weights.values())), 1)


def test_style_and_base_keeps_the_base_of_a_style_with_an_unknown_feature(home, caplog):
    write_style(home / "styles", {"name": "odd", "base": "funny", "weights": {"lodness": 0.4}})
    assert scoring.style_and_base("odd") == ("odd", "funny")
    [msg] = messages(caplog, logger="cutpoint.styles")
    assert (
        "feature 'lodness' in its weights; nothing measures it, but its weight still counts" in msg
    )


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
    assert full == set(features.REGISTRY)
    assert {"loudness", "speech", "laughter", "continuity", "scene", "airtime"} <= lite
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


def test_ski_score_keeps_every_weight_in_phase_1a():
    assert scorer.available_for(STYLES["ski"]) is None
    assert scorer.available_for(STYLES["gaming"]) == features.available()
