"""Stage: ``--exclude`` follows the original spelling when ``source_remap`` is active."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from conftest import write
from cutpoint import config
from cutpoint.config import remap_source
from cutpoint.stage import clear_stage, stage_sources, staged_path


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

    result = stage_sources([old], exclude=[pattern], min_free_gb=0)

    assert result["files"] == len(staged)
    for name in ("keep.mp4", "skip.mp4"):
        copy = staged_path(str(remap_source(old / name)))
        assert (copy != str(new / name)) == (name in staged)


@pytest.mark.parametrize(
    ("recursive", "exclude", "removed", "kept"),
    [
        (True, [], 2, []),
        (False, [], 1, ["sub/b.mp4"]),
        (True, ["*/old/a.mp4"], 1, ["a.mp4"]),
    ],
    ids=["recursive-removes-subfolder-copy", "flat-keeps-subfolder-copy", "exclude-keeps-copy"],
)
def test_clear_stage_selects_the_files_stage_copies(
    home, tmp_path, monkeypatch, recursive, exclude, removed, kept
):
    old, new = tmp_path / "old", tmp_path / "new"  # old is gone; its footage lives in new
    write(new / "a.mp4", "aa")
    write(new / "sub" / "b.mp4", "bbb")
    write(home / "config.json", json.dumps({"source_remap": {str(old): str(new)}}))
    config.get_config.cache_clear()
    monkeypatch.setattr(config, "_prefix_forms", lambda p: (config.norm_path(p),))
    assert stage_sources([old], recursive=True, min_free_gb=0)["files"] == 2

    assert clear_stage([old], recursive=recursive, exclude=exclude) == removed

    for name in ("a.mp4", "sub/b.mp4"):
        copy = staged_path(str(remap_source(old / name)))
        assert (copy != str(new / name)) == (name in kept)


def test_clear_stage_finds_a_file_only_the_original_folder_has(home, tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"  # both exist; c.mp4 is only in old
    write(old / "a.mp4", "aa")
    write(old / "c.mp4", "cc")
    write(new / "a.mp4", "aa")
    write(home / "config.json", json.dumps({"source_remap": {str(old): str(new)}}))
    config.get_config.cache_clear()
    monkeypatch.setattr(config, "_prefix_forms", lambda p: (config.norm_path(p),))
    assert stage_sources([old], min_free_gb=0)["files"] == 2

    assert clear_stage([old]) == 2


@pytest.mark.parametrize(
    ("recursive", "removed", "kept"),
    [(False, 2, ["old/sub/b.mp4", "older/x.mp4"]), (True, 3, ["older/x.mp4"])],
    ids=["flat-keeps-subfolder-copy", "recursive-removes-subfolder-copy"],
)
def test_clear_stage_falls_back_to_the_index_for_inputs_that_cannot_be_listed(
    home, tmp_path, monkeypatch, caplog, recursive, removed, kept
):
    old, new, other, older = (tmp_path / n for n in ("old", "new", "other", "older"))
    write(new / "a.mp4", "aa")
    write(new / "sub" / "b.mp4", "bbb")
    write(other / "c.mp4", "cc")
    write(older / "x.mp4", "xx")  # a sibling named like old's prefix; never selected
    write(home / "config.json", json.dumps({"source_remap": {str(old): str(new)}}))
    config.get_config.cache_clear()
    monkeypatch.setattr(config, "_prefix_forms", lambda p: (config.norm_path(p),))
    assert stage_sources([old, other, older], recursive=True, min_free_gb=0)["files"] == 4
    names = {"old/a.mp4": old / "a.mp4", "old/sub/b.mp4": old / "sub" / "b.mp4"}
    names |= {"other/c.mp4": other / "c.mp4", "older/x.mp4": older / "x.mp4"}
    copies = {n: staged_path(str(remap_source(p))) for n, p in names.items()}
    for gone in (new, other):  # old and other can no longer be listed
        shutil.rmtree(gone)

    with caplog.at_level("WARNING"):
        count = clear_stage([old, tmp_path / "missing", other], recursive=recursive)

    assert count == removed
    for name, copy in copies.items():
        assert Path(copy).exists() == (name in kept)
    assert "no staged copy recorded under it" in caplog.text
    assert "missing" in caplog.text


@pytest.mark.skipif(sys.platform == "win32", reason="a Windows-form key is foreign only on POSIX")
@pytest.mark.parametrize("target", ["copy-path", "windows-path"])
def test_clear_stage_finds_copies_recorded_under_a_windows_form_key(home, tmp_path, target):
    """#149: keys recorded as ``D:\\Videos\\T\\x.mp4`` use backslashes, not ``/``."""
    new = tmp_path / "copy"
    write(new / "T" / "x.mp4", "xx")
    write(new / "Tx" / "z.mp4", "zz")  # a sibling named like T's prefix; never selected
    write(home / "config.json", json.dumps({"source_remap": {r"D:\Videos": str(new)}}))
    config.get_config.cache_clear()
    assert stage_sources([new / "T", new / "Tx"], min_free_gb=0)["files"] == 2
    copies = {n: Path(staged_path(str(new / n))) for n in ("T/x.mp4", "Tx/z.mp4")}
    twin = new / "T" / "X.mp4"  # a different file on a case-sensitive file system
    write(twin, "XX")
    twin_path = str(twin)
    shutil.rmtree(new)  # the remap target is gone, so only the index can find them

    # a differently-spelled Windows-form source still reads the staged copy
    for spelled in ("d:/videos/T/x.mp4", r"D:\Videos\T\x.mp4"):
        assert staged_path(spelled) == str(copies["T/x.mp4"])
    assert staged_path("d:/videos/Tx/z.mp4") == str(copies["Tx/z.mp4"])
    # ...but below the remap prefix the case must match: T/X.mp4 is not T/x.mp4
    assert staged_path(twin_path) == twin_path

    assert clear_stage([new / "T" if target == "copy-path" else Path(r"D:\Videos\T")]) == 1
    assert [c.exists() for c in copies.values()] == [False, True]


def _windows_remap(home, new):
    write(home / "config.json", json.dumps({"source_remap": {r"D:\Videos": str(new)}}))
    config.get_config.cache_clear()


@pytest.mark.skipif(sys.platform == "win32", reason="a Windows-form key is foreign only on POSIX")
def test_case_twins_below_a_windows_form_remap_are_staged_separately(home, tmp_path, caplog):
    """#163: ``T/x.mp4`` and ``T/X.MP4`` share a stage key but get their own copy and entry."""
    new = tmp_path / "copy"
    low, up = write(new / "T" / "x.mp4", "low"), write(new / "T" / "X.MP4", "UPPER")
    if len(list((new / "T").iterdir())) < 2:
        pytest.skip("case-insensitive file system")
    _windows_remap(home, new)
    caplog.set_level("INFO", logger="cutpoint.stage")

    assert stage_sources([new / "T"], min_free_gb=0)["files"] == 2
    assert stage_sources([new / "T"], min_free_gb=0)["files"] == 0
    assert caplog.text.count("Already staged") == 2

    copies = {p: Path(staged_path(str(p))) for p in (low, up)}
    assert [c.read_text() for c in copies.values()] == ["low", "UPPER"]
    assert copies[low] != copies[up]

    assert clear_stage([low]) == 1
    assert staged_path(str(up)) == str(copies[up])
    assert not copies[low].exists()


@pytest.mark.skipif(sys.platform == "win32", reason="a Windows-form key is foreign only on POSIX")
def test_index_entries_without_rel_still_match_and_get_it(home, tmp_path, caplog):
    """#163: an index written before ``rel`` existed is used without a new copy."""
    new = tmp_path / "copy"
    src = write(new / "T" / "x.mp4", "xx")
    _windows_remap(home, new)
    stage_sources([src], min_free_gb=0)
    copy = staged_path(str(src))
    index_file = config.get_config().stage_dir / "index.json"
    index = json.loads(index_file.read_text())
    assert [e.pop("rel") for e in index.values()] == [["T", "x.mp4"]]
    index_file.write_text(json.dumps(index))
    caplog.set_level("INFO", logger="cutpoint.stage")

    assert staged_path(str(src)) == copy
    assert stage_sources([src], min_free_gb=0)["files"] == 0
    assert "Already staged" in caplog.text
    assert len(list(index_file.parent.glob("*-x.mp4"))) == 1
    assert [e["rel"] for e in json.loads(index_file.read_text()).values()] == [["T", "x.mp4"]]


def test_staged_path_follows_stage_dir_and_index_changes_without_reload(home, tmp_path):
    """#163: a running process sees another process's index, an edited ``stage_dir`` or
    ``source_remap``, and a copy removed by hand."""
    src = write(tmp_path / "src" / "x.mp4", "xx")
    remap = json.dumps({"source_remap": {str(tmp_path / "gone"): str(src.parent)}})
    write(home / "config.json", remap)  # staged under its original path, as recorded
    stage_sources([src], min_free_gb=0)
    copy = staged_path(str(src))
    assert copy != str(src)
    index_file = Path(copy).parent / "index.json"
    saved = index_file.read_text()

    index_file.write_text("{}")  # another process cleared the stage
    assert staged_path(str(src)) == str(src)
    index_file.write_text(saved)  # ...and staged it again
    assert staged_path(str(src)) == copy

    write(home / "config.json", "{}")  # the remap is gone: the key no longer matches
    assert staged_path(str(src)) == str(src)
    write(home / "config.json", remap)
    assert staged_path(str(src)) == copy

    # a copy that vanished or was cut short is not returned, though the index still lists it
    Path(copy).write_text("")
    assert staged_path(str(src)) == str(src)
    Path(copy).unlink()
    assert staged_path(str(src)) == str(src)

    other = tmp_path / "other-stage"
    write(home / "config.json", json.dumps({"stage_dir": str(other)}))
    assert staged_path(str(src)) == str(src)
