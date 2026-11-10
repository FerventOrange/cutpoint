"""Ski scorer: one pick per stretch (#282) and friends never lifting the lift/still mask (#283)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from cutpoint import features, scorer, ski
from cutpoint.styles import get_style
from cutpoint.telemetry import DT, Telemetry, paint_spans

N = 900  # 90 s on the 0.1 s grid


def _grid(spans: list[tuple[float, float, float]]) -> np.ndarray:
    return paint_spans(N, spans)


def _bump(t: float, height: float, width: float = 1.0) -> np.ndarray:
    x = np.arange(N) * DT
    return height * np.exp(-(((x - t) / width) ** 2))


def _pick(vlm: np.ndarray, mask: np.ndarray | None = None) -> list[ski.Pick]:
    zero = np.zeros(N)
    return ski.pick(
        {"vlm": vlm}, {"vlm": 1.0}, zero if mask is None else mask, zero, [], N * DT, 10
    )


def test_flat_stretch_gives_one_pick() -> None:
    picks = _pick(_grid([(10.0, 40.0, 1.0)]))  # one 30 s VLM window at one value
    assert len(picks) == 1
    lo, hi = ski.GEOMETRY["vlm"][:2]
    assert lo <= picks[0].end - picks[0].start <= hi
    assert picks[0].start >= 10.0 and picks[0].end <= 40.0


@pytest.mark.parametrize("kind", ["beauty", "scenic"])
def test_scenic_picks_are_five_to_seven_seconds(kind: str) -> None:
    assert ski.GEOMETRY[kind] == (5.0, 7.0, 2.5, 2.5)
    zero = np.zeros(N)
    picks = ski.pick({kind: _bump(40.0, 1.0)}, {kind: 1.0}, zero, zero, [], N * DT, 10)
    assert len(picks) == 1 and picks[0].kind == kind
    assert 5.0 <= picks[0].end - picks[0].start <= 7.0
    assert picks[0].start <= 40.0 <= picks[0].end


@pytest.mark.parametrize("kind", ["beauty", "scenic"])
@pytest.mark.parametrize("t", [1.0, 2.0, 119.5])
def test_scenic_pick_near_a_source_edge_keeps_its_length(kind: str, t: float) -> None:
    n = 1200  # 120 s
    peak = np.exp(-(((np.arange(n) * DT - t) / 1.0) ** 2))
    zero = np.zeros(n)
    picks = ski.pick({kind: peak}, {kind: 1.0}, zero, zero, [], n * DT, 10)
    assert len(picks) == 1
    assert 5.0 <= picks[0].end - picks[0].start <= 7.0
    assert picks[0].start >= 0.0 and picks[0].end <= n * DT


def test_scenic_pick_trimmed_by_a_stronger_pick_grows_back_to_five_seconds() -> None:
    zero = np.zeros(N)
    feats = {"scenic": _bump(40.0, 0.6), "speed": _bump(35.0, 1.0)}
    picks = ski.pick(feats, {"scenic": 1.0, "speed": 1.0}, zero, zero, [], N * DT, 10)
    scenic = next(p for p in picks if p.kind == "scenic")
    other = next(p for p in picks if p.kind != "scenic")
    assert scenic.end - scenic.start >= 5.0
    assert scenic.start >= other.end or scenic.end <= other.start


def test_scenic_pick_in_a_source_shorter_than_five_seconds_still_works() -> None:
    n = 40  # 4 s
    x = np.arange(n) * DT
    peak = np.exp(-(((x - 2.0) / 0.5) ** 2))
    picks = ski.pick({"beauty": peak}, {"beauty": 1.0}, np.zeros(n), np.zeros(n), [], n * DT, 10)
    assert [(p.start, p.end) for p in picks] == [(0.0, 4.0)]


def test_shoulder_of_a_peak_is_not_a_second_pick() -> None:
    # A short peak on a long lower plateau (a jump on a fast section): one moment.
    picks = _pick(np.maximum(_grid([(10.0, 50.0, 0.6)]), _bump(30.0, 1.0)))
    assert len(picks) == 1
    assert picks[0].start <= 30.0 <= picks[0].end


def test_distinct_peaks_are_all_picked() -> None:
    curve = _bump(15.0, 1.0) + _bump(30.0, 0.8) + _bump(45.0, 0.5) + _bump(75.0, 0.9)
    times = sorted(p.peak_t for p in _pick(curve))
    assert times == pytest.approx([15.0, 30.0, 45.0, 75.0], abs=0.15)


def test_peaks_separated_by_a_dip_on_a_plateau_are_both_picked() -> None:
    # Two jumps on one fast section, with the plateau well below both: two moments.
    curve = np.maximum(_grid([(10.0, 60.0, 0.3)]), _bump(20.0, 1.0) + _bump(40.0, 0.9))
    assert len(_pick(curve)) == 2


def _tel(lift: tuple[float, float], still: tuple[float, float]) -> Telemetry:
    z = np.zeros(N)
    grid = {
        k: z.copy()
        for k in ("airtime", "crash", "carve", "wind", "vibration", "indoor", "faces", "gps_ok")
    }
    grid["gps_ok"][:] = 1.0
    grid["speed_kmh"] = np.full(N, 40.0)  # above the day's median: some speed score
    grid["lift_mask"], grid["still_mask"] = _grid([(*lift, 1.0)]), _grid([(*still, 1.0)])
    return Telemetry(source="a.mp4", duration=N * DT, grid=grid)


def _features(monkeypatch: pytest.MonkeyPatch, windows: list[dict]) -> tuple:
    monkeypatch.setattr(ski, "layer_features", lambda *_a, **_k: {})
    src = Path("a.mp4")
    ctx = ski.SkiContext(
        project=Path("p"),
        telemetry={str(src): _tel(lift=(0.0, 30.0), still=(30.0, 45.0))},
        speed_ref=np.linspace(0.0, 60.0, 100),
        windows={ski._key(src): [{"source": str(src), **w} for w in windows]},
    )
    return ski.build_features(src, N, {}, ctx)


def _win(start: float, end: float, **tags: int) -> dict:
    return {"start": start, "end": end, "tags": tags}


def test_friends_never_lift_the_lift_or_still_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    f, mask, friends = _features(monkeypatch, [_win(0.0, 45.0, friends_faces=10)])
    assert friends[: int(45 / DT)].min() == pytest.approx(1.0)  # friends seen strongly
    assert f["vlm"][: int(45 / DT)].max() == 0.0  # but they add nothing on the lift / stop
    combined = ski.combine(f, {"vlm": 1.0}, mask, friends)
    assert combined[: int(45 / DT)].max() == 0.0
    assert ski.effective_mask(mask, friends)[: int(45 / DT)].min() == 1.0


def test_friends_do_not_count_in_standing_or_lift_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Telemetry says moving (no mask after 45 s), the VLM sees people standing and talking.
    f, *_ = _features(monkeypatch, [_win(50.0, 80.0, friends_faces=10, standing_talking=10)])
    assert f["vlm"].max() == 0.0


def test_friends_raise_a_moving_moment(monkeypatch: pytest.MonkeyPatch) -> None:
    f, mask, friends = _features(
        monkeypatch, [_win(0.0, 45.0, friends_faces=10), _win(60.0, 75.0, friends_faces=8)]
    )
    assert f["vlm"][int(65 / DT)] == pytest.approx(0.8)
    weights = {"vlm": 0.2, "speed": 0.15}
    picks = ski.pick(f, weights, mask, friends, [], N * DT, 10)
    assert 60.0 <= picks[0].peak_t <= 75.0  # the friends window outranks plain riding


def test_action_tags_still_count_without_friends(monkeypatch: pytest.MonkeyPatch) -> None:
    f, *_ = _features(monkeypatch, [_win(50.0, 60.0, jump_air=9, standing_talking=3)])
    assert f["vlm"][int(55 / DT)] == pytest.approx(0.9)


def test_paint_spans_keeps_the_largest_value_and_includes_the_end_bin() -> None:
    x = paint_spans(10, [(0.25, 0.45, 0.5), (0.35, 0.55, 0.8), (-1.0, 0.05, 0.1), (0.95, 5.0, 0.3)])
    assert x.tolist() == [0.1, 0.0, 0.5, 0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.3]
    assert paint_spans(5, [(2.0, 3.0, 1.0), (0.25, -0.55, 1.0)]).tolist() == [0.0] * 5  # empty


def test_old_learned_presets_still_load(home: Path) -> None:
    """A preset saved by ``refmatch``/``fit`` (now ``scripts/refmatch.py``): an old numeric
    ``mask_friends`` is a hard mask."""
    styles = home / "styles"
    styles.mkdir(parents=True)
    preset = {"name": "old", "base": "action-cam", "weights": {"speed": 1.0}, "mask_friends": 5}
    (styles / "old.json").write_text(json.dumps(preset), encoding="utf-8")
    style = get_style("old")
    assert style.mask_friends == 5.0
    mask, friends = np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 1.0])
    assert ski.effective_mask(mask, friends, style.mask_friends).tolist() == [1.0, 1.0, 0.0]
    assert ski.effective_mask(mask, friends, 0).tolist() == [0.0, 0.0, 0.0]  # soft


def test_a_removed_taste_weight_scores_as_without_it(
    home: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The taste head was removed: a preset that weighs ``taste`` (and names its file) loads
    without a warning, and ``taste`` is not counted in the divisor, so it scores as the same
    preset without it."""
    styles = home / "styles"
    styles.mkdir(parents=True)
    plain = {"name": "plain", "base": "action-cam", "weights": {"speed": 1.0}}
    tasted = {**plain, "name": "tasted", "weights": {"speed": 1.0, "taste": 2.0}}
    tasted["taste"] = "t.taste.npz"
    for p in (plain, tasted):
        (styles / f"{p['name']}.json").write_text(json.dumps(p), encoding="utf-8")
    with caplog.at_level(logging.INFO):
        got, want = get_style("tasted"), get_style("plain")
    assert not caplog.records
    assert "taste" not in got.weights and got.weights == want.weights
    for avail in (features.available(("events",)), features.available(())):
        assert scorer.score(0.8, got, avail) == scorer.score(0.8, want, avail)


def test_features_live_under_the_action_cam_folder(tmp_path: Path) -> None:
    assert ski.FEATURES_DIR == "action_cam_features"
    src = tmp_path / "ride.mp4"
    got = ski.features_path(tmp_path, src)
    assert got.parent == tmp_path / "action_cam_features" and got.name.startswith("ride-")
