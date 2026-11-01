"""Scorer decisions per style: which moment ranks above which, what is kept and what is not.

Each test names one decision ("in funny, laughter ranks above loudness") on small synthetic
signals: one bump per signal, all of the same height, far apart. They compare candidates with
each other instead of freezing scores, so retuning a style's weights fails only the tests whose
decision the retune changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cutpoint import analyze, features, scorer, ski
from cutpoint.events import scoring
from cutpoint.styles import STYLES, Style, load_learned
from cutpoint.telemetry import DT, Telemetry, paint_spans

# --------------------------------------------------------------------------- generic styles
N = 2400  # 240 s on the 0.1 s grid
SIGNALS = ("loudness", "speech", "laughter", "scene", "continuity")  # analyze_source's order
SLOTS = (30.0, 90.0, 150.0, 210.0)  # bump times: 60 s apart, wider than any style's clip
GENERIC = sorted(n for n, s in STYLES.items() if s.kind == "generic")


def bump(t: float, n: int = N, height: float = 1.0, width: float = 1.0) -> np.ndarray:
    x = np.arange(n) * DT
    return height * np.exp(-(((x - t) / width) ** 2))


def part(sig: dict[str, np.ndarray], n: int = N, src: str = "a.mp4") -> analyze.SourceResult:
    ps = analyze.PartSignals(n, sig, [], [], [], [], [])
    return analyze.SourceResult(Path(src), None, 0.0, n / 10, [], [], None, ps)


def generic_picks(
    style: Style, at: dict[str, float], max_candidates: int | None = None
) -> list[analyze.Candidate]:
    """Candidates of ``style`` with one equal bump per signal of ``at`` (signal -> time)."""
    sig = {k: np.zeros(N) for k in SIGNALS}
    for name, t in at.items():
        sig[name] = bump(t)
    return analyze.pick_candidates(part(sig), style, max_candidates, {})


def ranking(style: Style, names: tuple[str, ...], max_candidates: int | None = None) -> list[str]:
    """The signals of ``names`` whose bump became a candidate, best score first."""
    at = dict(zip(names, SLOTS, strict=False))
    cands = generic_picks(style, at, max_candidates)
    hit = {
        name: c.score for name, t in at.items() for c in cands if c.start <= t <= c.end
    }  # fmt: skip
    return sorted(hit, key=lambda k: -hit[k])


# (style, higher, lower): a bump in ``higher`` ranks above an equal bump in ``lower``.
RANKS = [
    ("gaming", "loudness", "speech"),
    ("gaming", "speech", "laughter"),
    ("gaming", "laughter", "scene"),
    ("action", "loudness", "speech"),
    ("action", "scene", "laughter"),
    ("sport", "loudness", "speech"),
    ("sport", "speech", "scene"),
    ("sport", "scene", "laughter"),
    ("funny", "laughter", "loudness"),
    ("funny", "loudness", "speech"),
    ("commentary", "continuity", "loudness"),
    ("commentary", "loudness", "speech"),
    ("commentary", "speech", "laughter"),
]


@pytest.mark.parametrize(
    ("name", "higher", "lower"), RANKS, ids=[f"{s}: {h} > {lo}" for s, h, lo in RANKS]
)
def test_a_moment_ranks_above_an_equal_one_of_a_lower_weighted_signal(name, higher, lower):
    # The lower one comes first in time, so the order is not the chronological one.
    order = ranking(STYLES[name], (lower, higher))
    assert order[0] == higher, f"{name}: {order}"


@pytest.mark.parametrize("name", ["funny", "commentary"])
def test_a_scene_cut_alone_is_never_a_candidate_where_scene_weighs_nothing(name):
    assert generic_picks(STYLES[name], {"scene": 30.0}) == []


@pytest.mark.parametrize("name", GENERIC)
def test_top_n_keeps_the_best_moments(name):
    """``max_candidates`` drops the weakest moments, never a better one."""
    names = tuple(sorted(STYLES[name].weights, key=lambda k: -STYLES[name].weights[k]))[:3]
    assert ranking(STYLES[name], names, max_candidates=2) == ranking(STYLES[name], names)[:2]


@pytest.mark.parametrize("name", GENERIC)
def test_clips_stay_within_the_style_length(name):
    """Gaming and action clips stay short; commentary keeps at least 20 s of context."""
    style = STYLES[name]
    cands = generic_picks(style, dict(zip(("loudness", "speech"), SLOTS, strict=False)))
    assert len(cands) == 2
    for c in cands:
        assert style.min_len <= c.end - c.start <= style.max_len, (name, c.start, c.end)


def test_a_learned_preset_overrides_its_base_weights(tmp_path):
    """A learned gaming preset that weighs talk ranks it first; built-in gaming ignores it."""
    path = tmp_path / "talky.json"
    path.write_text(
        json.dumps({"name": "talky", "base": "gaming", "weights": {"continuity": 0.9}}),
        encoding="utf-8",
    )
    names = ("loudness", "continuity")
    assert ranking(load_learned(path), names)[0] == "continuity"
    assert ranking(STYLES["gaming"], names) == ["loudness"]


# --------------------------------------------------------------------------- events rescore
def rescore(style: str, cands: list[dict[str, Any]], events: list[dict[str, Any]], windows=()):
    hl = {"style": style, "settings": {}, "candidates": cands}
    scorer.rescore_events(hl, {"events": events, "windows": list(windows)}, Path("proj"))
    return [c["id"] for c in hl["candidates"]]


def cand(cid: str, start: float, score: float) -> dict[str, Any]:
    return {"id": cid, "source": "a.mp4", "start": start, "end": start + 8, "score": score}


def death(t: float) -> dict[str, Any]:
    return {
        "source": "a.mp4", "start": t, "end": t + 3, "kind": "ocr", "type": "kill",
        "label": "Ash > Bob", "player": True, "player_kill": False, "player_death": True,
        "teamkill": False,
    }  # fmt: skip


def kill(t: float) -> dict[str, Any]:
    return {**death(t), "player_kill": True, "player_death": False}


def laugh(t: float, confidence: float = 0.05) -> dict[str, Any]:
    return {
        "source": "a.mp4", "start": t, "end": t + 1.5, "kind": "audio", "label": "Laughter",
        "confidence": confidence,
    }  # fmt: skip


@pytest.fixture
def no_transcript(monkeypatch):
    monkeypatch.setattr(scoring, "_segments", lambda src, language=None: [])


def test_funny_puts_a_death_with_a_laugh_above_a_slightly_better_plain_moment(no_transcript):
    cands = [cand("plain", 10.0, 50.0), cand("death", 100.0, 45.0)]
    events = [death(102.0), laugh(104.0)]
    assert rescore("funny", cands, events) == ["death", "plain"]


def test_gaming_gives_a_death_no_reaction_points(no_transcript):
    cands = [cand("plain", 10.0, 50.0), cand("death", 100.0, 45.0)]
    events = [death(102.0), laugh(104.0)]
    assert rescore("gaming", cands, events) == ["plain", "death"]


def test_gaming_puts_a_double_kill_above_a_better_moment_without_kills(no_transcript):
    cands = [cand("plain", 10.0, 55.0), cand("kills", 100.0, 45.0)]
    assert rescore("gaming", cands, [kill(101.0), kill(104.0)]) == ["kills", "plain"]


@pytest.mark.parametrize("style", ["funny", "gaming"])
def test_a_vlm_window_raises_a_moment_but_never_lowers_one(style, no_transcript):
    cands = [cand("liked", 10.0, 40.0), cand("disliked", 100.0, 60.0)]
    windows = [
        {"source": "a.mp4", "start": 8.0, "end": 20.0, "highlight_score": 10, "funny_score": 0},
        {"source": "a.mp4", "start": 98.0, "end": 110.0, "highlight_score": 0, "funny_score": 0},
    ]
    hl = {"style": style, "settings": {}, "candidates": cands}
    scorer.rescore_events(hl, {"events": [], "windows": windows}, Path("proj"))
    got = {c["id"]: c["score"] for c in hl["candidates"]}
    assert got["liked"] > 40.0
    assert got["disliked"] == 60.0


# --------------------------------------------------------------------------- ski
SKI_N = 1800  # 180 s
SRC = Path("ski.mp4")
LIFT = (150.0, 175.0)  # the chairlift, where the hard mask forbids clips


def ski_telemetry(**bumps: float) -> Telemetry:
    """Telemetry with one equal bump per grid of ``bumps`` (grid -> time) and a jump or impact
    event at each airtime / crash bump; GPS at the day's median speed, so speed adds nothing."""
    z = np.zeros(SKI_N)
    grid = {k: z.copy() for k in ("airtime", "crash", "carve", "wind", "vibration", "indoor")}
    grid |= {"faces": z.copy(), "still_mask": z.copy()}
    grid["gps_ok"] = np.ones(SKI_N)
    grid["speed_kmh"] = np.full(SKI_N, 20.0)
    grid["lift_mask"] = paint_spans(SKI_N, [(*LIFT, 1.0)])
    events = []
    for name, t in bumps.items():
        grid[name] = bump(t, SKI_N)
        if name == "airtime":
            events.append({"type": "airtime", "t": t - 0.5, "end": t + 0.5, "air_s": 1.0})
        if name == "crash":
            events.append({"type": "crash", "t": t - 0.2, "end": t + 0.8, "impact_t": t, "peak_g": 7.4})  # fmt: skip
    return Telemetry(source=str(SRC), duration=SKI_N * DT, grid=grid, events=events)


def crash_record(cid: str, t: float, confidence: float) -> dict[str, Any]:
    """A crashes.json crash at ``t`` with its planned clip and reaction."""
    return {
        "source": str(SRC), "id": cid, "t0": t, "t1": t + 1.5, "confidence": confidence,
        "scores": {"imu": 0.8},
        "clip": {"start": t - 5, "end": t + 8, "reaction": {"start": t + 8, "end": t + 13}},
    }  # fmt: skip


def ski_picks(
    tmp: Path,
    style: Style,
    tel: Telemetry,
    crashes: list[dict[str, Any]] = (),  # type: ignore[assignment]
    max_candidates: int | None = None,
) -> list[analyze.Candidate]:
    ctx = ski.SkiContext(
        project=tmp / "proj",
        telemetry={str(SRC): tel},
        speed_ref=np.linspace(0.0, 40.0, 100),
        crashes={ski._key(SRC): list(crashes)},
    )
    sig = {k: np.zeros(SKI_N) for k in SIGNALS}
    return analyze.pick_candidates(part(sig, SKI_N, str(SRC)), style, max_candidates, {}, ctx)


def learned_ski(tmp: Path, **data: Any) -> Style:
    path = tmp / "mine.json"
    path.write_text(json.dumps({"name": "mine", "base": "ski", **data}), encoding="utf-8")
    return load_learned(path)


def at(cands: list[analyze.Candidate], t: float) -> analyze.Candidate | None:
    return next((c for c in cands if c.start <= t <= c.end and not c.must_include), None)


@pytest.fixture
def no_layers(monkeypatch):
    """No cached voice / visual layers (the 360 and voice grids stay empty)."""
    monkeypatch.setattr(ski, "layer_features", lambda *_a, **_k: {})


def test_ski_keeps_a_confident_crash_as_must_include(tmp_path, no_layers):
    """Even when the weights leave room for one clip only, the crash, its approach, aftermath
    and reaction are kept, and they rank first."""
    tel = ski_telemetry(airtime=30.0)
    cands = ski_picks(tmp_path, STYLES["ski"], tel, [crash_record("c1", 100.0, 0.72)], 1)
    forced = [c for c in cands if c.must_include]
    assert [c.label_hint for c in forced] == ["crash", "crash reaction"]
    assert forced[0].start <= 95.0 and forced[0].end >= 108.0  # approach to aftermath
    assert all(c.event["id"] == "c1" for c in forced)
    assert all(c.score == 72.0 for c in forced)  # its confidence
    assert max(cands, key=lambda c: c.score).must_include
    assert at(cands, 30.0) is not None  # the jump is still picked


def test_ski_does_not_force_a_crash_below_the_threshold(tmp_path, no_layers):
    low = STYLES["ski"].must_include_crash - 0.05
    cands = ski_picks(tmp_path, STYLES["ski"], ski_telemetry(), [crash_record("c2", 100.0, low)])
    assert not any(c.must_include for c in cands)


def test_a_learned_ski_preset_can_turn_must_include_off(tmp_path, no_layers):
    style = learned_ski(tmp_path, must_include_crash=0)
    cands = ski_picks(tmp_path, style, ski_telemetry(), [crash_record("c1", 100.0, 0.9)])
    assert not any(c.must_include for c in cands)


SKI_RANKS = [("airtime", "crash"), ("crash", "carve")]


@pytest.mark.parametrize(("higher", "lower"), SKI_RANKS, ids=[f"{h} > {lo}" for h, lo in SKI_RANKS])
def test_ski_ranks_a_moment_above_an_equal_one_of_a_lower_weighted_grid(
    tmp_path, no_layers, higher, lower
):
    cands = ski_picks(tmp_path, STYLES["ski"], ski_telemetry(**{lower: 30.0, higher: 90.0}))
    hi, lo = at(cands, 90.0), at(cands, 30.0)
    assert hi is not None and lo is not None
    assert hi.kind == higher and lo.kind == lower
    assert hi.score > lo.score


def test_ski_labels_jumps_and_crashes_from_their_telemetry_events(tmp_path, no_layers):
    cands = ski_picks(tmp_path, STYLES["ski"], ski_telemetry(airtime=30.0, crash=90.0))
    assert at(cands, 30.0).label_hint == "jump 1.0s"
    assert at(cands, 90.0).label_hint == "crash 7g"


def test_ski_never_clips_the_lift_unless_the_mask_is_soft(tmp_path, no_layers):
    tel = ski_telemetry(airtime=160.0)  # a "jump" on the chairlift: camera handling
    assert at(ski_picks(tmp_path, STYLES["ski"], tel), 160.0) is None
    soft = learned_ski(tmp_path, mask_friends=0)
    assert at(ski_picks(tmp_path, soft, tel), 160.0) is not None


def test_ski_on_lite_picks_the_same_moments_and_keeps_the_crash_score(
    tmp_path, no_layers, monkeypatch
):
    """Lite cannot run the events features: the same picks, the scores renormalised over the
    core features (so never lower), and a must-include crash still scores its confidence."""
    tel = ski_telemetry(airtime=30.0, carve=60.0)
    crashes = [crash_record("c1", 100.0, 0.72)]

    def run(installed: frozenset[str]) -> list[analyze.Candidate]:
        monkeypatch.setattr(features, "installed_extras", lambda: installed)
        return ski_picks(tmp_path, STYLES["ski"], tel, crashes)

    full, lite = run(frozenset({"events"})), run(frozenset())
    assert [(c.start, c.end, c.peak) for c in lite] == [(c.start, c.end, c.peak) for c in full]
    for f, li in zip(full, lite, strict=True):
        if f.must_include:
            assert li.score == f.score == 72.0
        else:
            assert li.score > f.score
