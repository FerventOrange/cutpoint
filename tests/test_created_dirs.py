"""Issue #194: a failed or interrupted run removes the folders it created, and only those."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import set_config, write
from cutpoint import cache, refmatch, stage
from cutpoint.events import discover, killfeed, visual
from cutpoint.events.common import Source
from cutpoint.ffmpeg_utils import output_dir
from cutpoint.pano import qa, qa_paths, stitch, views
from pano_helpers import x3clip


def interrupt(*_a: object, **_k: object) -> None:
    raise KeyboardInterrupt  # what Ctrl+C raises inside a step


def test_a_failure_removes_new_nested_folders_deepest_first(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "c"
    removed: list[str] = []
    real_rmdir = Path.rmdir

    def spy(self: Path) -> None:
        real_rmdir(self)
        removed.append(self.name)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "rmdir", spy)
        # the original exception comes through
        with pytest.raises(RuntimeError, match="boom"), cache.created_dirs() as made:
            cache.make_dirs(deep, made)
            assert made == [deep, deep.parent, deep.parent.parent]
            raise RuntimeError("boom")
    assert removed == ["c", "b", "a"]
    assert not (tmp_path / "a").exists()


def test_a_failure_keeps_an_existing_folder_named_through_dotdot(tmp_path: Path) -> None:
    """``a/new/../x`` names the existing ``a``, which must stay."""
    (tmp_path / "a").mkdir()
    with pytest.raises(KeyboardInterrupt), output_dir(tmp_path / "a" / "new" / ".." / "x"):
        raise KeyboardInterrupt
    assert (tmp_path / "a").is_dir()
    assert not (tmp_path / "a" / "new").exists()


def test_failed_stage_leaves_no_new_stage_folder(home: Path, tmp_path: Path) -> None:
    stage_dir = tmp_path / "fast" / "stage"
    set_config(home, stage_dir=str(stage_dir))
    with pytest.raises(FileNotFoundError):
        stage.stage_sources([tmp_path / "missing.mp4"])
    assert not stage_dir.exists()
    assert not (tmp_path / "fast").exists()


def test_interrupted_check_frames_leaves_no_refmatch_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shot = {
        "matched": True, "shot": 1, "ref_start": 0.0, "ref_end": 2.0, "src_start": 5.0,
        "source": str(tmp_path / "a.mp4"),
    }  # fmt: skip
    result = {"shots": [shot], "reference": str(tmp_path / "ref.mp4")}
    out = tmp_path / "proj" / "refmatch_check"
    monkeypatch.setattr(refmatch, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        refmatch.check_frames(result, out)
    assert not out.exists()
    assert not (tmp_path / "proj").exists()


def make_qa_clips(root: Path) -> Path:
    clips = root / "preview"
    write(clips / "001_a.mp4", b"")
    write(
        clips / "001_a.json",
        json.dumps(
            {"keys": [[0.0, 0.0, 0.0, 90.0]], "segment": {"start": 0, "end": 3}, "source": "g.lrv"}
        ),
    )
    return clips


def test_interrupted_run_qa_leaves_no_qa_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clips = make_qa_clips(tmp_path)
    monkeypatch.setattr(visual, "VisualModels", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(qa, "decode", interrupt)  # the ffmpeg frame decode
    with pytest.raises(KeyboardInterrupt):
        qa.run_qa(clips, json_only=True)
    assert not (tmp_path / "preview_qa").exists()


def test_interrupted_run_paths_qa_leaves_no_paths_qa_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clips = make_qa_clips(tmp_path)
    pv = SimpleNamespace(t=np.arange(4.0))
    monkeypatch.setattr(views, "load_views", lambda *_a, **_k: pv)
    monkeypatch.setattr(qa_paths, "qa_path_clip", interrupt)
    with pytest.raises(KeyboardInterrupt):
        qa_paths.run_paths_qa(clips)
    assert not (tmp_path / "preview_paths_qa").exists()


def test_interrupted_stitch_leaves_no_new_stitched_folder(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "fast" / "stitched"
    set_config(home, pano={"stitched_dir": str(out_dir)})
    monkeypatch.setattr(stitch, "require_sdk", lambda: tmp_path / "MediaSDKTest")
    monkeypatch.setattr(stitch, "MIN_FREE_GB", -1e9)  # no disk-space check on the CI disk
    monkeypatch.setattr(stitch, "_stitch_one", interrupt)  # the SDK run
    with pytest.raises(KeyboardInterrupt):
        stitch.stitch_clips([x3clip()])
    assert not out_dir.exists()
    assert not (tmp_path / "fast").exists()


def test_interrupted_discovery_ocr_leaves_no_discovery_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = Source(path=tmp_path / "a.mp4", duration=60.0, cdir=tmp_path / "cache", audio_parts=[])
    frame_dir = tmp_path / "disc" / "game" / "frames"  # as pipeline._discover lays it out
    monkeypatch.setattr(
        discover, "_burst_frames", lambda *_a, **_k: [np.zeros((4, 4, 3), np.uint8)]
    )
    monkeypatch.setattr(killfeed, "OcrEngine", lambda **_k: interrupt)  # the OCR call
    with pytest.raises(KeyboardInterrupt):
        discover._ocr_frames(
            [(src, 5.0)], discover.DiscoverParams(), frame_dir, frame_dir.parent / "ocr.jsonl"
        )
    assert not frame_dir.exists()
    assert not (tmp_path / "disc").exists()
