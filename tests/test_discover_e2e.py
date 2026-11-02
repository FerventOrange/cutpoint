"""Real HUD discovery end to end: a synthetic clip drawn with OpenCV (invented HUD-like text)
and encoded by ffmpeg, the real OCR engine, ``pipeline._discover`` writing proposal.json and a
preset.

Skips without ffmpeg or rapidocr (events extra).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import needs_ffmpeg
from cutpoint import paths
from cutpoint.events import common, pipeline
from cutpoint.events.killfeed import OcrParams
from cutpoint.events.regions import load_preset
from ocr_helpers import FEED, GAME, SECONDS, make_clip


@needs_ffmpeg
@pytest.mark.slow
def test_discovery_proposes_feed_region_and_installs_preset(home: Path, tmp_path: Path) -> None:
    pytest.importorskip("rapidocr_onnxruntime")
    clip = tmp_path / "footage" / "session.mp4"
    clip.parent.mkdir()
    make_clip(clip)
    project = tmp_path / "proj"
    project.mkdir()

    # The duration is the one run_events reads from highlights.json; without it load_source
    # falls back to the cached WAV length, and there is none here.
    res = pipeline._discover(
        project, [common.load_source(clip, float(SECONDS))], OcrParams(game=GAME), None, 10, "user"
    )

    assert set(res) == {GAME}  # the key we passed, not a silent fallback to another name
    rec = res[GAME]
    assert rec["installed"] is True
    proposal = json.loads(Path(rec["proposal"]).read_text(encoding="utf-8"))
    assert proposal["game"] == GAME
    assert proposal["regions"], "discovery proposed no region"
    assert (project / "discovery" / GAME / "overlay.jpg").exists()

    # a region overlaps the drawn feed (reference frame is 1920x1080: scale by 1.5)
    fx, fy, fw, fh = (v * 1.5 for v in FEED)
    near = [
        r for r in proposal["regions"]
        if r["x"] < fx + fw and r["x"] + r["w"] > fx and r["y"] < fy + fh and r["y"] + r["h"] > fy
    ]  # fmt: skip
    assert near, f"no region near the feed {FEED}: {proposal['regions']}"

    # the preset is installed under the discovered key and loads back
    preset_file = paths.presets_dir() / f"{GAME}.json"
    assert Path(rec["preset"]) == preset_file
    assert load_preset(GAME, project) is not None
