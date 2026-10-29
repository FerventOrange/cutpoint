"""The scorer gives exactly the scores of the code before the feature registry (#514 phase 1).

``scorer_baseline.json`` was written by the pre-registry ``pick_candidates`` (its
``combined = sum(weights.get(k, 0) * v for k, v in sig.items())`` and ``100 * peak / wsum``)
and ``events.scoring.rescore`` on the synthetic inputs below. Every value is compared with
``==``: a full install must score byte-identically. The grids are built with integer
arithmetic and one division, so they are the same on every platform.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cutpoint import analyze, scorer
from cutpoint.events import scoring
from cutpoint.styles import STYLES, Style, load_learned

BASELINE = Path(__file__).with_name("scorer_baseline.json")
N = 1800  # 180 s on the 0.1 s grid
SIGNALS = ("loudness", "speech", "laughter", "scene", "continuity")  # analyze_source's order
LEARNED = {"name": "mine", "base": "gaming", "weights": {"continuity": 0.2, "loudness": 0.3}}
GENERIC = sorted(n for n, s in STYLES.items() if s.kind == "generic")


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


def signals() -> dict[str, np.ndarray]:
    return {k: grid(seed) for seed, k in enumerate(SIGNALS, 1)}


def part(sig: dict[str, np.ndarray]) -> analyze.SourceResult:
    words = [
        {"start": t / 10, "end": t / 10 + 0.3, "word": f"w{t}", "probability": 0.9}
        for t in range(0, N, 7)
    ]
    kw = [(12.5, 1.5, "let's go"), (61.0, 0.8, "wow"), (140.2, 1.2, "no way")]
    laugh = [(30.0, 1.0, "haha"), (99.9, 0.8, "lol")]
    ps = analyze.PartSignals(N, sig, kw, laugh, words, [], [])
    return analyze.SourceResult(Path("a.mp4"), None, 0.0, N / 10, [], [], None, ps)


def learned(tmp: Path) -> Style:
    path = tmp / "mine.json"
    path.write_text(json.dumps(LEARNED), encoding="utf-8")
    return load_learned(path)


def styles(tmp: Path) -> dict[str, Style]:
    return {**{n: STYLES[n] for n in GENERIC}, "learned:mine": learned(tmp)}


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes()).hexdigest()


def candidates(style: Style, max_candidates: int | None) -> list[list[Any]]:
    cands = analyze.pick_candidates(part(signals()), style, max_candidates, {})
    return [[c.start, c.end, c.peak_t, c.peak, c.score, c.label_hint, c.keywords] for c in cands]


def rescore_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    """A highlights.json and events.json covering every bonus term."""
    cands = [
        {"source": "a.mp4", "start": float(s), "end": float(s + 8), "score": sc}
        for s, sc in [(5, 61.3), (40, 48.0), (80, 33.3), (120, 22.1), (150, 90.4), (170, 12.0)]
    ]

    def kill(t: float, **flags: bool) -> dict[str, Any]:
        base = {"player": True, "player_kill": False, "player_death": False, "teamkill": False}
        return {
            "source": "a.mp4",
            "start": t,
            "end": t + 3,
            "kind": "ocr",
            "type": "kill",
            "label": "Ash > Bob",
            **base,
            **flags,
        }

    def audio(label: str, t: float, conf: float) -> dict[str, Any]:
        return {
            "source": "a.mp4",
            "start": t,
            "end": t + 1.5,
            "kind": "audio",
            "label": label,
            "confidence": conf,
        }

    events = [
        audio("Laughter", 7.0, 0.17),
        audio("Scream", 9.0, 0.09),
        kill(42.0, player_kill=True),
        kill(44.0, player_kill=True),
        kill(46.0, teamkill=True),
        kill(84.0, player_death=True),
        audio("Laughter", 86.0, 0.31),
        audio("Cheering", 125.0, 0.6),
        kill(152.0, player_kill=True, player_death=True),
    ]
    windows = [
        {"source": "a.mp4", "start": 0.0, "end": 12.0, "highlight_score": 8, "funny_score": 3},
        {"source": "a.mp4", "start": 118.0, "end": 130.0, "highlight_score": 2, "funny_score": 9},
        {"source": "a.mp4", "start": 148.0, "end": 160.0, "highlight_score": 4, "funny_score": 0},
    ]
    hl = {"style": "funny", "settings": {"target_total": 30.0}, "candidates": cands}
    ev = {"created": "2026-10-29T00:00:00", "events": events, "windows": windows}
    return hl, ev


def rescored(project: Path, style: str) -> dict[str, Any]:
    hl, ev = rescore_inputs()
    hl["style"] = style
    scoring.rescore(hl, ev, project)
    return hl


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", [*GENERIC, "learned:mine"])
def test_combined_grid_is_byte_identical(name, tmp_path, baseline):
    style = styles(tmp_path)[name]
    assert digest(scorer.combined(signals(), style)) == baseline["combined"][name]


@pytest.mark.parametrize("name", [*GENERIC, "learned:mine"])
@pytest.mark.parametrize("max_candidates", [None, 8])
def test_candidate_scores_are_identical(name, max_candidates, tmp_path, baseline):
    style = styles(tmp_path)[name]
    expected = baseline["candidates"][f"{name}@{max_candidates}"]
    assert json.loads(json.dumps(candidates(style, max_candidates))) == expected


@pytest.mark.parametrize("style", ["funny", "gaming"])
def test_events_rescore_is_identical(style, tmp_path, monkeypatch, baseline):
    segs = [{"start": 87.5, "end": 89.0, "text": "oh no"}]
    monkeypatch.setattr(scoring, "_segments", lambda src, language=None: segs)
    assert json.loads(json.dumps(rescored(tmp_path, style))) == baseline["rescore"][style]
