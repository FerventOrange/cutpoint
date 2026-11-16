"""Files someone else may have made, and durable writes (#1085: A649, A663, A667)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

import pytest

from conftest import media_info, write
from cutpoint import activity, cache, config, extend, folders, planning, rerank
from cutpoint.boundaries import SpeechSettings
from cutpoint.planning import Padding
from cutpoint.styles import get_style
from test_rerank import lite_layers as lite_layers  # autouse: no events extra, as in CI
from test_rerank import make_project

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- A667: durable writes


@pytest.fixture
def fsyncs(monkeypatch) -> list[tuple[Any, ...]]:
    """Every ``os.fsync`` and ``os.replace`` of :mod:`cutpoint.cache`, in order:
    ``("fsync", file id)`` and ``("replace", target name, file id of the temp file)``. A
    rename keeps the file id, so :func:`flushed_before_replace` tells which file was flushed."""
    calls: list[tuple[Any, ...]] = []
    fsync, replace = cache.os.fsync, cache.os.replace

    def spy_fsync(fd: int) -> None:
        calls.append(("fsync", cache.os.fstat(fd).st_ino))
        fsync(fd)

    def spy_replace(src: Any, dst: Any) -> None:
        calls.append(("replace", Path(dst).name, Path(src).stat().st_ino))
        replace(src, dst)

    monkeypatch.setattr(cache.os, "fsync", spy_fsync)
    monkeypatch.setattr(cache.os, "replace", spy_replace)
    return calls


def flushed_before_replace(calls: list[tuple[Any, ...]], name: str) -> bool:
    """Whether the temp file renamed onto ``name`` was flushed before that rename."""
    at = next(i for i, c in enumerate(calls) if c[0] == "replace" and c[1] == name)
    return ("fsync", calls[at][2]) in calls[:at]


def test_a_durable_write_flushes_the_temp_file_before_the_rename(tmp_path, fsyncs) -> None:
    target = tmp_path / "highlights.json"
    cache.save_json(target, {"a": 1}, durable=True)
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    kinds = [c[0] for c in fsyncs]
    assert kinds[:2] == ["fsync", "replace"]
    assert flushed_before_replace(fsyncs, "highlights.json")
    # on POSIX the folder is flushed too, after the rename; Windows cannot open a folder
    assert kinds[2:] == (["fsync"] if cache.os.name == "posix" else [])
    assert list(tmp_path.iterdir()) == [target]  # no temp file left


def test_a_plain_write_does_not_flush(tmp_path, fsyncs) -> None:
    cache.save_json(tmp_path / "cache.json", {"a": 1})
    cache.save_bytes(tmp_path / "cache.bin", b"x")
    assert [c[0] for c in fsyncs] == ["replace", "replace"]


def test_a_failed_flush_leaves_the_target_and_no_temp_file(tmp_path, monkeypatch) -> None:
    target = write(tmp_path / "config.json", '{"old": true}')

    def full(fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cache.os, "fsync", full)
    with pytest.raises(OSError, match="No space"):
        cache.save_json(target, {"new": True}, durable=True)
    assert target.read_text(encoding="utf-8") == '{"old": true}'
    assert list(tmp_path.iterdir()) == [target]


@pytest.fixture
def durable_saves(monkeypatch) -> list[tuple[str, bool]]:
    """``(file name, durable)`` of every :func:`cutpoint.cache.save_json`."""
    saves: list[tuple[str, bool]] = []
    real = cache.save_json

    def spy(path: Path, data: Any, *, indent: int | str = 2, durable: bool = False) -> None:
        saves.append((Path(path).name, durable))
        real(path, data, indent=indent, durable=durable)  # type: ignore[arg-type]

    monkeypatch.setattr(cache, "save_json", spy)
    monkeypatch.setattr(folders, "save_json", spy)
    return saves


def test_config_json_is_written_durably(tmp_path, durable_saves) -> None:
    config.set_config_values({"player_tags": ["Example"]}, path=tmp_path / "config.json")
    assert durable_saves == [("config.json", True)]


def test_highlights_json_is_written_durably(tmp_path, durable_saves) -> None:
    project = tmp_path / "proj"
    write(project / "highlights.json", json.dumps({"sources": [], "settings": {}}))
    assert activity.record_context(project, "downhill mtb, chest mount")
    assert durable_saves == [("highlights.json", True)]


def test_a_held_durable_write_is_flushed_on_commit(tmp_path, fsyncs) -> None:
    """A write a ``hold_outputs`` hold keeps in memory stays durable when committed (F1)."""
    project = tmp_path / "proj"
    with cache.hold_outputs(project, ("highlights.json", "transcript.txt")) as held:
        cache.save_bytes(project / "transcript.txt", b"words")
        cache.save_json(project / "highlights.json", {"a": 1}, durable=True)
        assert fsyncs == []  # held: nothing on disk yet
        held.commit()
    assert json.loads((project / "highlights.json").read_text(encoding="utf-8")) == {"a": 1}
    assert flushed_before_replace(fsyncs, "highlights.json")
    assert not flushed_before_replace(fsyncs, "transcript.txt")  # not asked to be durable
    # one folder flush on POSIX, after the renames; none on Windows
    tail = [c[0] for c in fsyncs[max(i for i, c in enumerate(fsyncs) if c[0] == "replace") :]]
    assert tail[1:] == (["fsync"] if cache.os.name == "posix" else [])
    assert sorted(p.name for p in project.iterdir()) == ["highlights.json", "transcript.txt"]


def test_a_rerank_save_flushes_highlights_json(home, tmp_path, fsyncs) -> None:
    """``rerank`` writes highlights.json under a hold: the commit really flushes it (F1)."""
    project = make_project(home, tmp_path)
    fsyncs.clear()  # the project's own setup writes
    assert rerank.rerank(project, get_style("funny")).saved
    assert flushed_before_replace(fsyncs, "highlights.json")
    assert not any(p.name.endswith(".tmp") for p in project.iterdir())


def test_the_swap_journal_is_written_durably(tmp_path, durable_saves) -> None:
    new, dst = tmp_path / "new", tmp_path / "clips"
    write(new / "001_a.mp4", "new")
    write(dst / "001_a.mp4", "old")
    folders.swap_in(new, dst)
    assert (folders._SWAP_JOURNAL, True) in durable_saves
    assert (dst / "001_a.mp4").read_text(encoding="utf-8") == "new"


# ------------------------------------------------- A663: the project a clips folder names

FMT = {"width": 1920, "height": 1080, "fps": 30, "vertical": False}


class FakePano:
    """Stands in for :class:`cutpoint.pano.render.PanoExporter`: its path files go to
    ``paths_dir`` as the real one's do."""

    made: ClassVar[list[FakePano]] = []

    def __init__(self, project: Path, out_dir: Path, *args: Any) -> None:
        from cutpoint.pano.render import PATHS_DIR  # noqa: PLC0415

        self.project = project
        self.paths_dir = project / PATHS_DIR / out_dir.name
        self.fmt: Any = None
        FakePano.made.append(self)

    def inputs(self, source: str, mode: str | None) -> list[Path]:
        return []

    def export(self, source: str, start: float, end: float, out: Path, **kw: Any) -> tuple:
        path_file = write(self.paths_dir / (out.stem + ".json"), "{}")
        write(out, "encoded")
        return None, {"mode": "auto", "path_file": str(path_file)}


@pytest.fixture
def pano(monkeypatch) -> list[FakePano]:
    from cutpoint.pano import render  # noqa: PLC0415

    FakePano.made = []
    monkeypatch.setattr(render, "PanoExporter", FakePano)
    monkeypatch.setattr(extend, "probe", media_info)
    monkeypatch.setattr(planning.sources, "readable_range", lambda src, t, d: (0.0, d))
    return FakePano.made


def _clips(tmp_path: Path, project: Path | None, name: str = "clips") -> Path:
    clips = tmp_path / "proj" / name
    write(clips / "001_a.mp4")
    e = {"file": "001_a.mp4", "source": str(write(tmp_path / "a.insv")), "start": 10.0}
    e |= {"end": 20.0, "pano": {"mode": "auto"}}
    doc = {"format": FMT, "clips": [e]} | ({"project": str(project)} if project else {})
    write(clips / "manifest.json", json.dumps(doc))
    return clips


def _extend(clips: Path) -> Path:
    return extend.extend_clips(clips, Padding(1.0, 1.0), speech=SpeechSettings(enabled=False))


@pytest.mark.parametrize("name", ["clips", "picks"])
def test_a_clips_folder_naming_another_project_writes_360_paths_only_under_its_parent(
    tmp_path, pano, caplog, name
) -> None:
    other = tmp_path / "someone-else"
    other.mkdir()
    write(other / "highlights.json", json.dumps({"sources": []}))
    clips = _clips(tmp_path, other, name)
    before = sorted(other.rglob("*"))

    with caplog.at_level(logging.INFO, logger="cutpoint.extend"):
        out = _extend(clips)

    (exporter,) = pano
    assert exporter.project == other  # still read there
    assert sorted(other.rglob("*")) == before  # never written
    parent = clips.resolve().parent
    assert (parent / "pano_paths" / out.name / "001_a.json").is_file()
    (info,) = [
        r.getMessage() for r in caplog.records if "not the folder's parent" in r.getMessage()
    ]
    assert str(other) in info and str(parent / "pano_paths") in info


@pytest.mark.parametrize("recorded", [True, False])
def test_a_clips_folder_in_its_own_project_logs_nothing_and_writes_there(
    tmp_path, pano, caplog, recorded
) -> None:
    project = tmp_path / "proj"
    clips = _clips(tmp_path, project.resolve() if recorded else None)

    with caplog.at_level(logging.INFO, logger="cutpoint.extend"):
        out = _extend(clips)

    (exporter,) = pano
    assert exporter.paths_dir == project.resolve() / "pano_paths" / out.name
    assert (exporter.paths_dir / "001_a.json").is_file()
    assert not any("not the folder's parent" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------- A649: invented example names


def test_no_real_looking_player_name_in_the_sources_tests_or_docs() -> None:
    names = ("Sinis" + "tro", "BRo" + "FS")  # split, so this file does not match itself
    hits = [
        f"{p.relative_to(ROOT)}: {n}"
        for folder in ("src", "tests", "docs")
        for p in (ROOT / folder).rglob("*")
        if p.suffix in {".py", ".md", ".json", ".txt"} and p.is_file()
        for n in names
        if n in p.read_text(encoding="utf-8", errors="replace")
    ]
    assert hits == []
