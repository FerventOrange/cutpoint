"""``scripts/color_eval.py``, the ``--color`` development harness (#290 stage 2): every mode
and levels variant, saved-frame replay, 360 sources skipped as ``clips`` skips them."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import load_script, needs_ffmpeg
from cutpoint import color
from cutpoint.color import AutoGrade
from test_color import _scene, _stats, _underwater, _underwater_source

pytestmark = pytest.mark.script
color_eval = load_script("color_eval")


@pytest.fixture
def lenient(monkeypatch):
    """A clipping veto of 5 points instead of 0.5 (see ``test_color.lenient``)."""
    monkeypatch.setattr(color, "VETO_CLIPPED", 0.05)


def test_color_eval_reports_the_red_highlights_the_veto_counts():
    """A red highlight the levels intend is counted once, as ``color._red_highlights`` does."""
    frames = np.full((1, 20, 20, 3), (0.4, 0.5, 0.5), np.float32)
    frames[0, 0, 0] = (1.0, 0.5, 0.5)  # clipped in the source
    frames[0, 0, 1] = (0.99, 0.5, 0.5)  # red above its white point: the levels' tail
    frames[0, 0, 2] = (0.8, 0.5, 0.5)  # red pushed out by the gain alone
    grade = AutoGrade(
        "underwater", black=(0.0, 0.0, 0.0), white=(0.98, 1.0, 1.0), gains=(1.3, 1.0, 1.0)
    )
    assert color_eval.evaluate(frames, grade)["red_highlights"] == 1 / 400


@needs_ffmpeg
def test_color_eval_script_reports_every_mode_and_levels_variant(tmp_path, capsys, lenient):
    src = _underwater_source(tmp_path / "reef.mkv")
    saved = tmp_path / "stats"
    assert color_eval.main([f"{src}@0.5-2.5", "--json", "--save-stats", str(saved)]) == 0

    (window,) = json.loads(capsys.readouterr().out)
    assert window["frames"] > 0
    assert {(v["mode"], v["levels"]) for v in window["variants"]} == {
        (m, lv) for m in color.MODES for lv in color.LEVELS
    }
    assert {(v["mode"], v["levels"]) for v in window["variants"] if v["production"]} == set(
        color.MODE_LEVELS.items()
    )
    # the saved frames replay to the same results without decoding
    assert [f.name for f in saved.iterdir()] == ["reef_0.5-2.5.npz"]
    assert color_eval.main([f"{src}@0.5-2.5", "--json", "--stats", str(saved)]) == 0
    (again,) = json.loads(capsys.readouterr().out)
    assert again["variants"] == window["variants"]


def test_color_eval_script_replays_saved_frames_by_their_file_name(tmp_path, capsys):
    stats = _stats(_underwater())
    color_eval.save_stats(tmp_path / "GX010819_3-17.npz", replace(stats, high_bit=True))
    assert color_eval.main(["--stats", str(tmp_path), "--json", "--modes", "underwater"]) == 0
    (window,) = json.loads(capsys.readouterr().out)
    assert (window["source"], window["start"], window["end"]) == ("GX010819", 3.0, 17.0)
    assert window["high_bit"] and window["frames"] == len(stats.frames)
    production = next(v for v in window["variants"] if v["production"])
    assert production["skip"] == color.estimate(stats, "underwater").skip
    assert isinstance(production["params"]["black"], list)


def test_color_eval_script_replays_a_stem_with_a_dot_by_its_file_name(tmp_path, capsys):
    """#494: ``Desktop ... 02.56.27.19.DVR.mp4`` is saved as ``...19.DVR_120-132.npz``; its
    window's source is ``...19.DVR``, which Path.stem would cut to ``...19``."""
    name = "Desktop 2025.10.14 - 02.56.27.19.DVR_120-132.npz"
    color_eval.save_stats(tmp_path / name, _stats(_scene(2)))
    (w,) = color_eval.stats_windows(tmp_path)
    assert (w["source"], w["start"], w["end"]) == ("Desktop 2025.10.14 - 02.56.27.19.DVR", 120, 132)
    assert color_eval.stats_name(w) == name
    assert color_eval.main(["--stats", str(tmp_path), "--json", "--modes", "auto"]) == 0
    (window,) = json.loads(capsys.readouterr().out)
    assert window["frames"] == 2 and window["gate"]["rg"] == pytest.approx(1.0, abs=0.2)


def test_color_eval_script_skips_360_sources_as_clips_does(tmp_path, capsys, monkeypatch):
    lrv = tmp_path / "LRV_20250101_120000_01_001.lrv"
    monkeypatch.setattr(color_eval, "probe", lambda *_a: pytest.fail("a 360 source was sampled"))
    assert color_eval.main([f"{lrv}@1-9", "--json"]) == 0
    (window,) = json.loads(capsys.readouterr().out)
    assert {v["skip"] for v in window["variants"]} == {color.SKIP_360}
    # a project's 360 clip by its pano.json, whatever its name
    clip = str(tmp_path / "clip.mp4")
    pano = {color_eval.pano_project.norm(clip): SimpleNamespace(source=clip)}
    assert color_eval.is_360(clip, pano)
    assert not color_eval.is_360(clip, None)
    assert not color_eval.is_360(str(tmp_path / "b.mp4"), pano)
