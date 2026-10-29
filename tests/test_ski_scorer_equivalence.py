"""Ski scores and picks are exactly those of the code before ski moved onto the feature registry
(#514 phase 1b).

``ski_scorer_baseline.json`` was written by the code of cba4698 (ski's own feature list, its
``laughter`` feature, ``100 * peak / sum(weights)``) on the synthetic inputs below: the
candidates of ``analyze.pick_candidates`` with a ski context, the saved ``ski_features`` grids,
and ``refmatch.evaluate`` and ``fit_weights`` on them. Every value is compared with ``==``: a full
install must score byte-identically. The only expected difference is the rename of the ski
feature ``laughter`` to ``voice_laughter`` (:data:`RENAMED`), in grid names, ``signals`` keys,
``kind`` values and fitted weight names. The grids are built with integer arithmetic and one
division, so they are the same on every platform. Each case runs twice: with float64 grids and
with float32 ones, as ``telemetry.extract`` loads cached telemetry (``@f32`` in the baseline;
the voice and 360 view layers are float32 there too).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cutpoint import analyze, features, refmatch, scorer, ski
from cutpoint.events import visual, voice
from cutpoint.pano import views
from cutpoint.styles import STYLES, Style, load_learned

BASELINE = Path(__file__).with_name("ski_scorer_baseline.json")
N = 1800  # 180 s on the 0.1 s grid
SRC = Path("ski.mp4")
RENAMED = {"laughter": "voice_laughter"}  # the 1b rename, applied to the baseline
LEARNED = {
    # Old feature name: the alias must give the scores it gave before the rename.
    "laughs": {"base": "ski", "weights": {"laughter": 0.6, "airtime": 0.1}},
    "soft": {
        "base": "ski",
        "mask_friends": 0,
        "must_include_crash": 0.5,
        "weights": {"stopped": 0.3, "people": 0.2, "group": 0.1, "voice": 0.1, "faces": 0.1},
    },
    "unknown": {"base": "ski", "weights": {"mystery": 0.2, "speed": 0.4}},
}
NAMES = ["ski", *(f"learned:{n}" for n in LEARNED)]


def grid(seed: int, n: int = N) -> np.ndarray:
    """0-1 grid: low noise plus four triangular bumps (exact integer arithmetic)."""
    i = np.arange(n)
    out = ((i * (7919 + seed * 104729)) % 1009) / 1009.0 * 0.2
    for k in range(4):
        center = (seed * 131 + k * 977) % n
        width = 20 + (seed + k) % 15
        height = ((seed * 37 + k * 53) % 10 + 1) / 10
        out = np.maximum(out, np.maximum(0, 1 - np.abs(i - center) / width) * height)
    return out


def centers(seed: int) -> list[float]:
    """The bump centres of :func:`grid` ``seed``, in seconds."""
    return [((seed * 131 + k * 977) % N) / 10 for k in range(4)]


def telemetry(f32: bool = False) -> SimpleNamespace:
    """A stand-in for :class:`cutpoint.telemetry.Telemetry` (grid, duration, events); ``f32``:
    float32 grids, as a cached one loads."""
    i = np.arange(N)
    g = {
        "airtime": grid(11),
        "crash": grid(12),
        "carve": grid(13),
        "wind": grid(14),
        "vibration": np.where(i % 400 < 300, 0.2, 0.05),
        "indoor": np.where((i >= 1500) & (i < 1550), 1.0, 0.0),
        "gps_ok": np.where(i < 900, 1.0, 0.0),
        "speed_kmh": grid(15) * 70,
        "faces": grid(16) * 0.03,
        "still_mask": np.where((i >= 600) & (i < 700), 1.0, 0.0),
        "lift_mask": np.where((i >= 1200) & (i < 1350), 1.0, 0.0),
    }
    events = [
        {"type": "airtime", "t": t - 0.4, "end": t + 0.6, "air_s": 1.0, "confidence": "high"}
        for t in centers(11)
    ] + [
        {"type": "crash", "t": t - 0.2, "end": t + 0.8, "impact_t": t, "peak_g": 7.4}
        for t in centers(12)
    ]
    if f32:
        g = {k: v.astype(np.float32) for k, v in g.items()}
    return SimpleNamespace(grid=g, duration=N / 10, events=events)


def context(project: Path, f32: bool = False) -> ski.SkiContext:
    src = str(SRC)
    key = ski._key(SRC)
    audio = [
        {"kind": "audio", "source": src, "start": 20.0, "end": 21.5, "label": "Laughter", "confidence": 0.09},
        {"kind": "audio", "source": src, "start": 75.0, "end": 76.0, "label": "Scream", "confidence": 0.2},
        {"kind": "audio", "source": src, "start": 160.0, "end": 162.0, "label": "Cheering", "confidence": 0.5},
    ]  # fmt: skip
    windows = [
        {"source": src, "start": 0.0, "end": 12.0, "tags": {"jump_air": 7, "scenic_vista": 8}},
        {"source": src, "start": 50.0, "end": 62.0, "tags": {"friends_faces": 9, "chairlift": 3}},
        {"source": src, "start": 118.0, "end": 130.0, "tags": {"landing": 5, "lodge": 0}},
        {"source": src, "start": 140.0, "end": 150.0, "tags": {"powder_spray": 6, "friends_faces": 4}},
    ]  # fmt: skip
    crashes = [
        {
            "source": src, "id": "c1", "t0": 100.0, "t1": 101.5, "confidence": 0.72,
            "scores": {"imu": 0.8, "audio": 0.4},
            "clip": {"start": 95.0, "end": 108.0, "reaction": {"start": 108.0, "end": 113.0}},
        },
        {
            "source": src, "id": "c2", "t0": 30.0, "t1": 31.0, "confidence": 0.55,
            "scores": {"imu": 0.6}, "clip": {"start": 26.0, "end": 36.0},
        },
    ]  # fmt: skip
    return ski.SkiContext(
        project=project,
        telemetry={src: telemetry(f32)},  # type: ignore[dict-item]
        speed_ref=np.arange(1, 101) * 0.8,
        audio_events={key: audio},
        windows={key: windows},
        crashes={key: crashes},
    )


def layers(monkeypatch: pytest.MonkeyPatch, f32: bool = False) -> None:
    """Voice and 360 view layers from grids, not caches (the visual layer stays off); ``f32``:
    float32 grids."""

    def cast(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: v.astype(np.float32) for k, v in d.items()} if f32 else d

    def features_of(grids, moving=None, cal=None):
        lf = grid(21) if moving is None else grid(21) * (1 - 0.5 * moving)
        return cast({"laughter": lf, "voice": grid(22), "wind": grid(14), "boom": grid(23)})

    monkeypatch.setattr(voice, "load_voice", lambda src, *a, **k: {"vad": np.zeros(N)})
    monkeypatch.setattr(voice, "voice_features", features_of)
    monkeypatch.setattr(
        views,
        "layer_grids",
        lambda src, n, offset=0.0, dt=0.1: cast(
            {"beauty": grid(24), "people": grid(25), "group": grid(26)}
        ),
    )
    monkeypatch.setattr(visual, "load_visual", lambda src, *a, **k: None)


def full(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every extra installed: the full edition, whatever this test run has."""
    monkeypatch.setattr(features, "installed_extras", lambda: frozenset({"events"}))


def signals() -> dict[str, np.ndarray]:
    names = ("loudness", "speech", "laughter", "scene", "continuity")  # analyze_source's order
    return {k: grid(seed) for seed, k in enumerate(names, 1)}


def part() -> analyze.SourceResult:
    words = [
        {"start": t / 10, "end": t / 10 + 0.3, "word": f"w{t}", "probability": 0.9}
        for t in range(0, N, 7)
    ]
    kw = [(12.5, 1.5, "send it"), (61.0, 0.8, "wow"), (140.2, 1.2, "no way")]
    laugh = [(30.0, 1.0, "haha"), (99.9, 0.8, "lol")]
    ps = analyze.PartSignals(N, signals(), kw, laugh, words, [], [])
    return analyze.SourceResult(SRC, None, 0.0, N / 10, [], [], None, ps)


def styles(tmp: Path) -> dict[str, Style]:
    out = {"ski": STYLES["ski"]}
    for name, data in LEARNED.items():
        path = tmp / f"{name}.json"
        path.write_text(json.dumps({"name": name, **data}), encoding="utf-8")
        out[f"learned:{name}"] = load_learned(path)
    return out


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()


def run(tmp: Path, name: str, max_candidates: int | None, f32: bool = False) -> dict[str, Any]:
    """Candidates, saved grids and refmatch results of the style ``name``."""
    style = styles(tmp)[name]
    project = tmp / "proj"
    cands = analyze.pick_candidates(part(), style, max_candidates, {}, context(project, f32))
    rows = [
        [
            *(c.start, c.end, c.peak_t, c.peak, c.score, c.label_hint, c.kind, c.must_include),
            *(c.event, c.keywords, c.signals, c.transcript),
        ]
        for c in cands
    ]
    (saved,) = ski.load_project_features(project)
    grids = {k: digest(v) for k, v in saved["features"].items()}
    grids |= {"mask": digest(saved["mask"]), "friends": digest(saved["friends"])}
    result = {
        "positives": [
            {"source": str(SRC), "start": 10.0, "end": 16.0},
            {"source": str(SRC), "start": 98.0, "end": 106.0},
            {"source": str(SRC), "start": 140.0, "end": 147.0},
        ],
        "reference_duration": 30.0,
        "reference": "ref.mp4",
        "shots": [
            {"shot": 0, "matched": True, "source": str(SRC), "src_start": 10.0, "src_end": 16.0}
        ],
    }
    ev = refmatch.evaluate(project, result, style.weights, friends_strong=style.mask_friends)
    out: dict[str, Any] = {"candidates": rows, "grids": grids, "evaluate": ev}
    if name == "ski":
        data = [refmatch.FitData(project, result, ski.load_project_features(project))]
        out["fit"] = refmatch.fit_weights(data, base="ski", folds=2, save=False, cross=False)
    return json.loads(json.dumps(out))


def rename(obj: Any) -> Any:
    """``obj`` with every dict key and string equal to a key of :data:`RENAMED` renamed."""
    if isinstance(obj, dict):
        return {RENAMED.get(k, k): rename(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rename(v) for v in obj]
    return RENAMED.get(obj, obj) if isinstance(obj, str) else obj


def cases() -> list[tuple[str, int | None, bool]]:
    return [(n, m, f32) for f32 in (False, True) for n in NAMES for m in (None, 6)]


def key(name: str, max_candidates: int | None, f32: bool) -> str:
    return f"{name}@{max_candidates}" + ("@f32" if f32 else "")


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("name", "max_candidates", "f32"), cases())
def test_ski_is_byte_identical_on_a_full_install(
    name, max_candidates, f32, tmp_path, monkeypatch, baseline
):
    layers(monkeypatch, f32)
    full(monkeypatch)
    got = run(tmp_path, name, max_candidates, f32)
    assert got == rename(baseline[key(name, max_candidates, f32)])


def test_float32_grids_give_their_own_scores(baseline):
    """The float32 cases test something: cached telemetry scores (slightly) differently."""
    assert baseline["ski@None@f32"]["candidates"] != baseline["ski@None"]["candidates"]


def test_the_baseline_covers_the_renamed_feature(baseline):
    """The alias case picks a moment of voice laughter, so ``kind`` and the signals are checked."""
    rows = baseline["learned:laughs@None"]["candidates"]
    assert any(r[6] == "laughter" for r in rows)
    assert any(r[7] for r in rows)  # and a must-include crash
    assert "laughter" in baseline["ski@None"]["fit"]["features_fitted"]


def test_lite_drops_the_events_features_from_the_divisor(tmp_path, monkeypatch, baseline):
    """Lite: the same picks, scores renormalised over the core features and capped at 100."""
    layers(monkeypatch)
    monkeypatch.setattr(features, "installed_extras", lambda: frozenset())
    style = STYLES["ski"]
    cands = analyze.pick_candidates(part(), style, None, {}, context(tmp_path / "proj"))
    old = rename(baseline["ski@None"]["candidates"])
    assert [[c.start, c.end, c.peak] for c in cands] == [[*r[:2], r[3]] for r in old]
    lite = features.divisor(style.weights, features.available(()))
    assert lite == pytest.approx(0.67)
    for c, r in zip(cands, old, strict=True):
        if c.must_include:  # its crash confidence, as on a full install
            assert c.score == r[4]
        else:
            assert c.score == min(100.0, round(100 * (c.peak / lite), 1))
            assert c.score >= r[4]


def test_the_cap_only_applies_when_a_weight_is_dropped():
    """A full install never renormalises, so the cap never changes its scores."""
    w = {"airtime": 0.5, "vlm": 0.5}
    assert scorer.score(1.2, STYLES["ski"], None) == features.to_score(
        1.2, STYLES["ski"].weights, None
    )
    assert features.to_score(1.2, w, features.available(("events",))) == 120.0
    assert features.to_score(1.2, w, features.available(())) == 100.0
    assert features.to_score(0.3, w, features.available(())) == 60.0
