"""Stage: ``--exclude`` follows the original spelling when ``source_remap`` is active."""

from __future__ import annotations

import json

import pytest

from conftest import write
from cutpoint import config
from cutpoint.config import remap_source
from cutpoint.stage import stage_sources, staged_path


@pytest.mark.parametrize(
    ("pattern", "staged"),
    [("*/old/skip.mp4", ["keep.mp4"]), ("*/new/skip.mp4", ["keep.mp4", "skip.mp4"])],
    ids=["original-spelling", "remapped-spelling-excludes-nothing"],
)
def test_stage_sources_matches_exclude_against_the_original_spelling(
    home, tmp_path, monkeypatch, pattern, staged
):
    old, new = tmp_path / "old", tmp_path / "new"  # old is gone; its footage lives in new
    write(new / "keep.mp4", "keep")
    write(new / "skip.mp4", "skip")
    write(home / "config.json", json.dumps({"source_remap": {str(old): str(new)}}))
    config.get_config.cache_clear()
    monkeypatch.setattr(config, "_prefix_forms", lambda p: (config.norm_path(p),))

    result = stage_sources([old], exclude=[pattern])

    assert result["files"] == len(staged)
    for name in ("keep.mp4", "skip.mp4"):
        copy = staged_path(str(remap_source(old / name)))
        assert (copy != str(new / name)) == (name in staged)
