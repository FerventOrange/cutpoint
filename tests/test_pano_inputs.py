"""PanoExporter.inputs: the files a 360 render may read, so ``clips --force`` keeps them."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import write
from cutpoint import config, ffmpeg_utils
from cutpoint.pano import project
from cutpoint.pano.render import PanoExporter
from cutpoint.pano.x3 import X3Clip
from pano_helpers import INSV, LRV


def _exporter(clip: X3Clip, source_mode: str) -> PanoExporter:
    pano = PanoExporter.__new__(PanoExporter)  # a stub: no project folder or pano.json needed
    pano.clips = {config.norm_path(clip.source): clip}
    pano.source_mode = source_mode
    return pano


@pytest.mark.parametrize("mode", [None, "studio"])
def test_inputs_lists_camera_files_as_recorded_and_remapped_without_raising(moved, mode):
    clip, old, new = moved
    got = _exporter(clip, "sdk").inputs(clip.source, mode)  # no stitch: falls back to auto
    for name in (LRV, *INSV):
        assert old / name in got
        assert new / name in got


def test_inputs_is_empty_for_a_source_that_is_not_a_360_clip(moved, tmp_path):
    clip, _old, _new = moved
    assert _exporter(clip, "sdk").inputs(str(tmp_path / "plain.mp4")) == []
    assert all(isinstance(p, Path) for p in _exporter(clip, "sdk").inputs(clip.source))


def _equirect_probe(monkeypatch, duration: float):
    """Make every probed file a 2:1 equirect of ``duration`` seconds."""

    def fake(path):
        return SimpleNamespace(width=5760, height=2880, duration=duration)

    monkeypatch.setattr(ffmpeg_utils, "probe", fake)


@pytest.mark.parametrize(("source_mode", "mode"), [("auto", None), ("sdk", "studio")])
def test_inputs_protects_a_studio_export(moved, monkeypatch, source_mode, mode):
    clip, _old, new = moved
    clip.duration = 12.0
    export = write(new / "export" / "Export_001.mp4")
    _equirect_probe(monkeypatch, 12.0)
    got = _exporter(clip, source_mode).inputs(clip.source, mode)
    assert export in got


@pytest.mark.parametrize("mode", [None, "sdk"])
def test_inputs_protects_an_sdk_stitched_equirect(moved, tmp_path, monkeypatch, mode):
    clip, _old, _new = moved
    stitched = tmp_path / "stitched"
    monkeypatch.setattr(project, "stitched_dir", lambda: stitched)
    equirect = write(project.sdk_output(clip))
    got = _exporter(clip, "auto").inputs(clip.source, mode)
    assert equirect in got
