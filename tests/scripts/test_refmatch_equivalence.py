"""``refmatch.evaluate`` and ``fit_weights`` (``scripts/refmatch.py``) give exactly the results
of the code before ski moved onto the feature registry (#514 phase 1b), on the synthetic ski
project of ``tests/test_ski_scorer_equivalence.py`` (its baseline, ``evaluate`` and ``fit``)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from conftest import load_script
from cutpoint import ski
from test_ski_scorer_equivalence import BASELINE, SRC, analyzed, cases, full, key, layers, rename

pytestmark = pytest.mark.script
refmatch = load_script("refmatch")

RESULT = {
    "positives": [
        {"source": str(SRC), "start": 10.0, "end": 16.0},
        {"source": str(SRC), "start": 98.0, "end": 106.0},
        {"source": str(SRC), "start": 140.0, "end": 147.0},
    ],
    "reference_duration": 30.0,
    "reference": "ref.mp4",
    "shots": [{"shot": 0, "matched": True, "source": str(SRC), "src_start": 10.0, "src_end": 16.0}],
}


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("name", "max_candidates", "f32"), cases())
def test_refmatch_is_byte_identical_on_a_full_install(
    name, max_candidates, f32, tmp_path, monkeypatch, baseline
):
    layers(monkeypatch, f32)
    full(monkeypatch)
    style, project, _ = analyzed(tmp_path, name, max_candidates, f32)
    ev = refmatch.evaluate(project, RESULT, style.weights, friends_strong=style.mask_friends)
    got: dict[str, Any] = {"evaluate": ev}
    if name == "ski":
        data = [refmatch.FitData(project, RESULT, ski.load_project_features(project))]
        got["fit"] = refmatch.fit_weights(data, base="action-cam", folds=2, save=False, cross=False)
    want = rename(baseline[key(name, max_candidates, f32)])
    assert json.loads(json.dumps(got)) == {k: want[k] for k in got}
