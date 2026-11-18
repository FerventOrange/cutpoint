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


# --------------------------------------------------------------------------- social (#556)
SOCIAL_N = 1200  # 120 s on the 0.1 s grid


def _bump(t: float, height: float = 1.0, width: float = 1.0) -> np.ndarray:
    x = np.arange(SOCIAL_N) * 0.1
    return height * np.exp(-(((x - t) / width) ** 2))


def _social_ski_run(tmp_path: Path, monkeypatch) -> tuple[list[analyze.Candidate], dict]:
    """An action-cam part with every kind of grid: telemetry, VLM, PANNs and transcript
    laughter, a crash, voice and visual layers; returns its candidates and saved features."""
    from cutpoint.telemetry import DT, Telemetry, paint_spans  # noqa: PLC0415

    n, src = SOCIAL_N, Path("ride.mp4")
    z = np.zeros(n)
    grid = {k: z.copy() for k in ("airtime", "crash", "carve", "wind", "vibration", "indoor")}
    grid |= {"faces": z.copy(), "still_mask": z.copy(), "gps_ok": np.ones(n)}
    grid["speed_kmh"] = 20.0 + 15.0 * _bump(60.0, width=8.0)
    grid["lift_mask"] = paint_spans(n, [(100.0, 110.0, 1.0)])
    grid["airtime"], grid["crash"], grid["carve"] = _bump(15.0), _bump(40.0, 0.8), _bump(60.0)
    events = [
        {"type": "airtime", "t": 14.5, "end": 15.5, "air_s": 1.0},
        {"type": "crash", "t": 39.8, "end": 40.8, "impact_t": 40.0, "peak_g": 7.4},
    ]
    tel = Telemetry(
        source=str(src), duration=n * DT, grid=grid, events=events, meta={"gps_samples": n}
    )
    layers = {
        "voice_laughter": _bump(85.0, 0.7),
        "voice": _bump(86.0, 0.5),
        "beauty": _bump(25.0, 0.6),
        "people": _bump(72.0, 0.9),
        "group": _bump(72.0, 0.4),
    }
    monkeypatch.setattr(actioncam, "layer_features", lambda *_a, **_k: dict(layers))
    key = actioncam._key(src)
    heard = [
        {"source": str(src), "start": 80.0, "end": 81.5, "label": "Laughter", "confidence": 0.09},
        {"source": str(src), "start": 92.0, "end": 93.0, "label": "Cheering", "confidence": 0.2},
    ]
    tags = {"jump_air": 6, "scenic_vista": 8, "friends_faces": 7}
    window = {"source": str(src), "start": 64.0, "end": 76.0, "tags": tags}
    crash = {"source": str(src), "id": "c1", "t0": 50.0, "t1": 51.0, "confidence": 0.4}
    ctx = actioncam.ActionCamContext(
        project=tmp_path / "proj",
        telemetry={str(src): tel},
        speed_ref=np.linspace(0.0, 40.0, 100),
        audio_events={key: heard},
        windows={key: [window]},
        crashes={key: [crash]},
    )
    sig = {k: np.zeros(n) for k in ("loudness", "speech", "laughter", "scene", "continuity")}
    sig["speech"], sig["laughter"] = _bump(30.0, 0.6), _bump(95.0, 0.8)
    ps = analyze.PartSignals(n, sig, [], [], [], [], [])
    res = analyze.SourceResult(src, None, 0.0, n / 10, [], [], None, ps)
    cands = analyze.pick_candidates(res, STYLES["action-cam"], None, {}, ctx)
    saved = actioncam.load_part(ctx.project, src)
    assert saved is not None
    return cands, saved


def _digest(cands: list[analyze.Candidate], saved: dict) -> dict[str, object]:
    import hashlib  # noqa: PLC0415

    h = hashlib.sha256()
    for k, v in saved["features"].items():
        h.update(k.encode() + v.astype(np.float32).tobytes())
    h.update(saved["mask"].astype(np.float32).tobytes())
    h.update(saved["friends"].astype(np.float32).tobytes())
    return {
        "scores": [c.score for c in cands],
        "kinds": [c.kind for c in cands],
        "times": [(c.start, c.end, round(c.peak_t, 3)) for c in cands],
        "signals": hashlib.sha256(
            json.dumps([c.signals for c in cands], sort_keys=False).encode()
        ).hexdigest(),
        "npz_keys": list(saved["features"]),
        "npz": h.hexdigest(),
    }


# Pinned (#1085): a full install stays byte-identical across feature refactors (#556)
def test_full_install_action_cam_output_is_pinned(tmp_path, monkeypatch):
    """Pinned on base 13d26aa, before #556: splitting ``social`` into registry parts changes
    nothing on a full install (features_available, cache key, scores, signals, npz). #738
    changed the scenic/beauty geometry (5-7 s), so the beauty candidate's times and signals
    digest were re-pinned. #1085 A36 dropped the unscored ``friends_vlm`` grid (npz and
    signals digests re-pinned; scores, kinds and times unchanged)."""
    monkeypatch.setattr(extras, "edition", lambda: "full")
    reset_state()
    cands, saved = _social_ski_run(tmp_path, monkeypatch)
    assert _digest(cands, saved) == {
        "scores": [24.5, 22.3, 15.7, 3.9, 3.7, 2.9],
        "kinds": ["airtime", "vlm", "crash", "social", "social", "beauty"],
        "times": [
            (13.0, 17.0, 15.0),
            (61.0, 67.0, 64.0),
            (38.0, 44.0, 40.0),
            (91.5, 97.5, 95.0),
            (78.5, 84.5, 82.0),
            (22.5, 27.5, 25.0),
        ],
        "signals": "e0df7f3ce8f80ea0db26ae67cd71b4745a1373facf329b91e18247d654fbd580",
        "npz_keys": list(actioncam.FEATURES),
        "npz": "23f94bf205b8e9e8ab11842f3c3e035e7f4c02d6403906f1c96ff821795db50a",
    }
    assert actioncam.features_path(tmp_path, Path("ride.mp4")).name == "ride-919e6a6c.npz"
    assert sorted(features.available()) == [  # settings.features_available
        "airtime", "beauty", "carve", "continuity", "crash", "event_laughter",
        "event_reactions", "event_scream", "event_types", "faces", "group", "laughter",
        "loudness", "people", "scene", "scenic", "short_clip", "social", "speech", "speed",
        "stopped", "vlm", "vlm_window", "voice", "voice_laughter", "wipeout",
    ]  # fmt: skip
    assert features.dropped(STYLES["action-cam"].weights, features.available()) == []
