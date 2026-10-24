"""Issue #194: a failed or interrupted run removes the folders it created, and only those."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import set_config, write
from cutpoint import analyze, cache, refmatch, stage
from cutpoint.ffmpeg_utils import output_dir
from cutpoint.styles import get_style


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


def test_interrupted_analyze_leaves_no_new_project_folder(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = write(tmp_path / "videos" / "a.mp4")
    out = tmp_path / "projects" / "deeper" / "proj"
    monkeypatch.setattr(analyze, "Transcriber", lambda **_k: object())
    monkeypatch.setattr(analyze, "analyze_source", interrupt)
    with pytest.raises(KeyboardInterrupt):
        analyze.analyze(src, get_style("funny"), out, vision=False, use_scenes=False)
    assert not out.exists()
    assert not (tmp_path / "projects").exists()


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
