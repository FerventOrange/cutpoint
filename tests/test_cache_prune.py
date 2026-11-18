"""Tests for ``cutpoint cache --prune`` (cutpoint.cacheprune) and the cache size in ``paths``.

Every cache, project and source here is synthetic, under the test's isolated home.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import time
from pathlib import Path

import pytest

from conftest import envelope, write
from cutpoint import cache, cacheprune, cli, folders, paths, runstate
from cutpoint.config import projects_dir

DAY = 86400.0


def _age(path: Path, days: float) -> None:
    """Make ``path`` and everything in it look last changed ``days`` ago."""
    t = time.time() - days * DAY
    for p in sorted(path.rglob("*"), reverse=True):
        # Windows has no lutime; its only symlink test is skipped there.
        os.utime(p, (t, t), follow_symlinks=os.utime not in os.supports_follow_symlinks)
    os.utime(path, (t, t))


def _folder(name: str, days: float, size: int = 10) -> Path:
    """A synthetic source cache folder ``name`` with one file, last used ``days`` ago."""
    folder = paths.cache_dir() / name
    write(folder / "loudness.json", "x" * size)
    _age(folder, days)
    return folder


def _source(tmp_path: Path, name: str = "match.mp4") -> Path:
    return write(tmp_path / "footage" / name, b"\0" * 64)


def _project(name: str, sources: list[str], file: str = "highlights.json") -> Path:
    project = projects_dir() / name
    data = (
        {"sources": [{"path": s} for s in sources]}
        if file == "highlights.json"
        else {"sources": sources}
    )
    write(project / file, json.dumps(data))
    return project


@pytest.fixture
def prune(capsys):
    """Run ``cutpoint cache --prune ARGS --json``; its envelope."""

    def run(*args: str) -> dict:
        capsys.readouterr()
        assert cli.main(["cache", "--prune", *args, "--json"]) == 0
        return envelope(capsys.readouterr().out)

    return run


def test_removes_old_folders_least_recently_used_first_and_keeps_the_rest(prune):
    old = _folder("a-00000000000000aa", 40, size=100)
    older = _folder("b@120-00000000000000bb", 90, size=50)
    recent = _folder("c-00000000000000cc", 5)
    other = [
        write(paths.cache_dir() / "activity" / "k.json"),
        write(paths.cache_dir() / "pano_vl" / "d.json"),
        write(paths.cache_dir() / "timeline-index.json", "{}"),
        write(paths.cache_dir() / "hud_map_tried.json", "{}"),
        write(paths.cache_dir() / "notes" / "mine.txt"),  # not a source cache folder
    ]
    for p in other:
        _age(p.parent if p.parent != paths.cache_dir() else p, 400)

    doc = prune("30")
    out = doc["outputs"]
    assert doc["ok"] and doc["command"] == "cache" and doc["project"] is None
    assert [r["path"] for r in out["removed"]] == [str(older), str(old)]
    assert [r["bytes"] for r in out["removed"]] == [50, 100]
    assert out["count"] == 2 and out["freed_bytes"] == 150
    assert out["kept"] == {"recent": 1, "project": 0, "in_use": 0}
    assert out["days"] == 30 and out["all"] is False and out["dry_run"] is False
    assert out["cache"] == str(paths.cache_dir()) and out["cache_bytes"] >= 150
    assert not old.exists() and not older.exists() and recent.is_dir()
    assert all(p.exists() for p in other)
    assert not list(paths.cache_dir().glob(".*cutpoint-*"))  # no lock or leftover stays


def test_a_file_written_recently_counts_as_used(prune):
    folder = _folder("a-00000000000000aa", 40)
    write(folder / "color" / "x.json")  # a subfolder's write leaves the folder's time alone
    os.utime(folder, (time.time() - 40 * DAY,) * 2)
    assert prune("30")["outputs"]["kept"]["recent"] == 1
    assert folder.is_dir()


def test_dry_run_removes_nothing_and_takes_no_lock(prune):
    folder = _folder("a-00000000000000aa", 40)
    out = prune("30", "--dry-run")["outputs"]
    assert out["dry_run"] is True and [r["path"] for r in out["removed"]] == [str(folder)]
    assert folder.is_dir()
    assert sorted(p.name for p in paths.cache_dir().iterdir()) == [folder.name]


def test_missing_cache_folder_is_left_missing(prune):
    out = prune("1")["outputs"]
    assert out["removed"] == [] and out["cache_bytes"] is None  # as paths
    assert not paths.cache_dir().exists()


def test_project_folders_stay_unless_all(tmp_path, prune):
    src = _source(tmp_path)
    folder = cache.source_dir(src)
    write(folder / "loudness.json")
    _age(folder, 40)
    _project("p", [str(src)])

    out = prune("30")["outputs"]
    assert out["removed"] == [] and out["kept"]["project"] == 1
    assert folder.is_dir()

    out = prune("30", "--all")["outputs"]
    assert [r["path"] for r in out["removed"]] == [str(folder)]
    assert not folder.exists() and src.is_file()  # the source is never touched


@pytest.mark.parametrize("file", ["highlights.json", "progress.json", "pano.json", "ui/meta.json"])
def test_every_project_file_names_its_sources(tmp_path, file, prune):
    src = _source(tmp_path)
    folder = cache.source_dir(src)
    _age(folder, 40)
    project = projects_dir() / "p"
    data = {
        "highlights.json": {"sources": [{"path": str(src)}]},
        "progress.json": {"sources": [str(src)]},
        "pano.json": {"clips": [{"lrv": None, "insv": [str(src)]}]},
        "ui/meta.json": {"source": str(src)},
    }[file]
    write(project / file, json.dumps(data))
    assert prune("30")["outputs"]["kept"]["project"] == 1
    assert folder.is_dir()


def test_recovered_parts_of_a_project_source_stay(tmp_path, prune):
    src = _source(tmp_path)
    base = cache.source_dir(src)
    write(base / "read_errors.json", json.dumps({"resumed_at": 120.5}))
    part = cache.source_dir(src, 120.5)
    write(part / "audio16k.wav")
    _age(base, 40)
    _age(part, 40)
    _project("p", [str(src)])
    assert prune("30")["outputs"]["kept"]["project"] == 2
    assert base.is_dir() and part.is_dir()


def test_missing_source_keeps_every_folder_named_after_it(tmp_path, prune):
    kept = [_folder("match-00000000000000aa", 40), _folder("match@300-00000000000000bb", 40)]
    gone = _folder("matches-00000000000000cc", 40)  # another name
    _project("p", [str(tmp_path / "offline" / "match.mp4")])
    out = prune("30")["outputs"]
    assert [r["path"] for r in out["removed"]] == [str(gone)]
    assert all(k.is_dir() for k in kept)


def test_unreadable_project_files_do_not_stop_the_prune(prune):
    project = projects_dir() / "broken"
    write(project / "highlights.json", "{not json")
    write(project / "progress.json", json.dumps({"sources": [3, None, ""]}))
    write(project / "pano.json", json.dumps({"clips": "x"}))
    write(projects_dir() / "a-file.txt")
    folder = _folder("a-00000000000000aa", 40)
    assert prune("30")["outputs"]["count"] == 1
    assert not folder.exists()


def test_running_project_keeps_its_folders_even_with_all(tmp_path, prune):
    src = _source(tmp_path)
    folder = cache.source_dir(src)
    _age(folder, 40)
    project = _project("p", [str(src)], "progress.json")
    with folders.swap_lock(project / runstate.LOCK_NAME, project):
        out = prune("30", "--all")["outputs"]
    assert out["removed"] == [] and out["kept"]["in_use"] == 1
    assert folder.is_dir()


def test_a_held_lock_in_the_folder_keeps_it(prune):
    folder = _folder("a-00000000000000aa", 40)
    with folders.swap_lock(folder / "timeline"):  # tsrepair's stream copy lock
        _age(folder, 40)
        out = prune("30", "--all")["outputs"]
        assert out["removed"] == [] and out["kept"]["in_use"] == 1
    assert folder.is_dir()


def test_a_stale_lock_in_the_folder_does_not_keep_it(prune):
    folder = _folder("a-00000000000000aa", 40)
    lock = folder / ".timeline.cutpoint-lock"
    write(lock, json.dumps({"pid": os.getpid(), "host": socket.gethostname()}))
    _age(folder, 40)
    assert prune("30")["outputs"]["count"] == 1  # this process holds no such lock


def test_another_prune_running_is_folder_busy(capsys):
    _folder("a-00000000000000aa", 40)
    with folders.swap_lock(paths.cache_dir() / cacheprune.PRUNE_LOCK):
        assert cli.main(["cache", "--prune", "30", "--json"]) == 1
    doc = envelope(capsys.readouterr().out)
    assert doc["ok"] is False and doc["error"]["code"] == "folder_busy"


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_symlinks_are_never_followed(tmp_path, prune):
    outside = write(tmp_path / "outside" / "keep.txt")
    _age(outside.parent, 400)
    link = paths.cache_dir() / "link-00000000000000aa"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside.parent, target_is_directory=True)
    folder = _folder("a-00000000000000bb", 40)
    (folder / "out").symlink_to(outside.parent, target_is_directory=True)
    _age(folder, 40)

    out = prune("30", "--all")["outputs"]
    assert [r["path"] for r in out["removed"]] == [str(folder)]
    assert link.is_symlink() and outside.is_file()


def test_a_folder_that_cannot_be_renamed_is_a_warning(monkeypatch, prune):
    folder = _folder("a-00000000000000aa", 40)

    def locked(op, path, action, locked=None):
        raise PermissionError(13, "in use")

    monkeypatch.setattr(cacheprune.cache, "retry_locked", locked)
    doc = prune("30")
    assert doc["ok"] is True and doc["outputs"]["removed"] == []
    assert doc["outputs"]["failed"] == [str(folder)]
    assert [w["code"] for w in doc["warnings"]] == ["cache_remove_failed"]
    assert doc["warnings"][0]["files"] == [str(folder)]
    assert folder.is_dir()


def test_a_removal_that_fails_part_way_is_a_warning_and_the_next_prune_ends_it(monkeypatch, prune):
    folder = _folder("a-00000000000000aa", 40)
    real = cacheprune._rmtree
    monkeypatch.setattr(cacheprune, "_rmtree", lambda p: ["x: denied"])
    doc = prune("30")
    assert [w["code"] for w in doc["warnings"]] == ["cache_remove_failed"]
    assert not folder.exists()
    (left,) = paths.cache_dir().glob(".a-00000000000000aa.*.cutpoint-prune")

    monkeypatch.setattr(cacheprune, "_rmtree", real)
    out = prune("30")["outputs"]
    assert out["freed_bytes"] == 10 and out["removed"] == []
    assert not left.exists()


def test_folder_used_after_listing_stays(monkeypatch, prune):
    folder = _folder("a-00000000000000aa", 40)
    listed = cacheprune.tree_stats
    calls = []

    def used_meanwhile(path):
        calls.append(path)
        if path == folder and len([c for c in calls if c == folder]) == 2:
            os.utime(folder)  # a run marks it used between the listing and the removal
        return listed(path)

    monkeypatch.setattr(cacheprune, "tree_stats", used_meanwhile)
    out = prune("30")["outputs"]
    assert out["removed"] == [] and out["kept"]["in_use"] == 1
    assert folder.is_dir()


def _locked_on_call(monkeypatch, n: int, then) -> None:
    """Patch ``cacheprune._locked`` to run ``then(path)`` on its ``n``-th call for a source
    cache folder (the listing makes the first)."""
    real, calls = cacheprune._locked, []

    def locked(path):
        calls.append(path)
        if len(calls) == n:
            then(path)
        return real(path)

    monkeypatch.setattr(cacheprune, "_locked", locked)


def test_folder_marked_used_after_the_last_walk_stays(monkeypatch, prune):
    """A run that calls mark_used while _remove checks the folder keeps it (F1)."""
    folder = _folder("a-00000000000000aa", 40)
    _locked_on_call(monkeypatch, 2, lambda p: os.utime(p))  # _remove's lock check
    out = prune("30")["outputs"]
    assert out["removed"] == [] and out["kept"]["in_use"] == 1
    assert folder.is_dir() and (folder / "loudness.json").is_file()


def test_project_run_started_after_listing_keeps_its_folder(tmp_path, monkeypatch, prune):
    src = _source(tmp_path)
    folder = cache.source_dir(src)
    _age(folder, 40)
    project = _project("p", [str(src)])
    with contextlib.ExitStack() as held:
        lock = folders.swap_lock(project / runstate.LOCK_NAME, project)
        _locked_on_call(monkeypatch, 2, lambda p: held.enter_context(lock))
        out = prune("30", "--all")["outputs"]
    assert out["removed"] == [] and out["kept"]["in_use"] == 1
    assert folder.is_dir()


def test_folder_used_during_the_rename_is_renamed_back(monkeypatch, prune):
    folder = _folder("a-00000000000000aa", 40)
    real = cacheprune._locked
    monkeypatch.setattr(
        cacheprune, "_locked", lambda p: p.name.endswith(".cutpoint-prune") or real(p)
    )
    out = prune("30")["outputs"]
    assert out["removed"] == [] and out["kept"]["in_use"] == 1
    assert (folder / "loudness.json").is_file()
    assert not list(paths.cache_dir().glob(".*.cutpoint-prune"))


def test_folder_made_again_during_the_rename_is_a_warning(monkeypatch, prune):
    folder = _folder("a-00000000000000aa", 40)
    real = cacheprune._used_since

    def made_again(path, cutoff):
        if path.name.endswith(".cutpoint-prune"):
            write(folder / "audio16k.wav")  # a run made the folder again meanwhile
            return True
        return real(path, cutoff)

    monkeypatch.setattr(cacheprune, "_used_since", made_again)
    doc = prune("30")
    assert [w["code"] for w in doc["warnings"]] == ["cache_remove_failed"]
    assert doc["outputs"]["failed"] == [str(folder)] and doc["outputs"]["removed"] == []
    assert (folder / "audio16k.wav").is_file()
    assert len(list(paths.cache_dir().glob(".a-00000000000000aa.*.cutpoint-prune"))) == 1

    monkeypatch.setattr(cacheprune, "_used_since", real)
    prune("30")
    assert not list(paths.cache_dir().glob(".*.cutpoint-prune"))
    assert (folder / "audio16k.wav").is_file()


def test_clips_manifest_sources_belong_to_the_project(tmp_path, prune):
    src = _source(tmp_path)
    folder = cache.source_dir(src)
    _age(folder, 40)
    manifest = {"clips": [{"file": "001_a.mp4", "source": str(src), "start": 0, "end": 1}]}
    write(projects_dir() / "p" / "clips_vertical" / "manifest.json", json.dumps(manifest))
    assert prune("30")["outputs"]["kept"]["project"] == 1
    assert folder.is_dir()


def test_days_must_be_one_or_more(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["cache", "--prune", "0", "--json"])
    assert exc.value.code == 2
    assert envelope(capsys.readouterr().out)["error"]["code"] == "bad_args"
    with pytest.raises(ValueError, match="1 or more"):
        cacheprune.prune(0)


def test_text_output_names_the_folders(caplog):
    folder = _folder("a-00000000000000aa", 40)
    _folder("b-00000000000000bb", 1)
    with caplog.at_level("INFO"):
        assert cli.main(["cache", "--prune", "30", "--dry-run"]) == 0
    assert f"Would remove {folder}" in caplog.text
    assert "Would remove 1 cache folder not used for 30 days" in caplog.text
    assert "kept 1 used since" in caplog.text


def test_library_profiles_survive_a_prune_with_all(tmp_path, monkeypatch, capsys, prune):
    """``cutpoint profile create`` files are never pruned, also with the cache folder around
    the profile library and the discovery cache of their footage long unused."""
    from conftest import SLED  # noqa: PLC0415

    monkeypatch.setenv("CUTPOINT_CACHE_DIR", str(paths.home()))  # the library inside the cache
    doc_file = write(tmp_path / "sled.json", json.dumps(SLED))
    for name in ("my-sled", "trail-0123456789abcdef"):  # one named like a source cache folder
        assert cli.main(["profile", "create", name, "--from", str(doc_file), "--json"]) == 0
    library = sorted(paths.profiles_dir().iterdir())
    found = write(paths.cache_dir() / "activity" / "00000000000000aa.json", "{}")
    _age(paths.profiles_dir(), 400)
    _age(found.parent, 400)
    folder = _folder("a-00000000000000aa", 400)

    out = prune("1", "--all")["outputs"]
    assert [r["path"] for r in out["removed"]] == [str(folder)]
    assert sorted(paths.profiles_dir().iterdir()) == library and len(library) == 2
    assert found.is_file()


# --------------------------------------------------------------------------- timeline index
def _timeline(folder: Path) -> dict:
    """A timeline copy in the cache folder ``folder``; its index entry."""
    write(folder / "timeline.mkv", b"\0" * 8)
    return {"file": f"{folder.name}/timeline.mkv", "size": 64, "mtime_ns": 1, "copy_size": 8}


def test_timeline_index_drops_the_copies_of_removed_folders(prune):
    from cutpoint import stage  # noqa: PLC0415

    old, recent = paths.cache_dir() / "a-00000000000000aa", paths.cache_dir() / "b-00000000000000bb"
    index = {
        "/v/old.ts": _timeline(old),
        "/v/recent.ts": _timeline(recent),
        "/v/gone.ts": {"file": "c-00000000000000cc/timeline.mkv"},  # not this prune's: stays
        "/v/odd.ts": "not an entry",
    }
    (old / "timeline.mkv").unlink()  # gone already: only the dry-run guard keeps its entry
    _age(old, 40)
    path = write(paths.cache_dir() / stage.TIMELINE_INDEX, json.dumps(index))

    assert prune("30", "--dry-run")["outputs"]["count"] == 1
    assert json.loads(path.read_text()) == index  # a dry run writes nothing

    assert prune("30")["outputs"]["count"] == 1
    del index["/v/old.ts"]
    assert json.loads(path.read_text()) == index
    assert not list(paths.cache_dir().glob(".*cutpoint-*"))  # no lock stays


def test_timeline_index_entry_of_a_copy_made_again_stays(monkeypatch, prune):
    """A run that made the removed folder and its copy again meanwhile keeps its entry."""
    from cutpoint import stage  # noqa: PLC0415

    old = paths.cache_dir() / "a-00000000000000aa"
    index = {"/v/old.ts": _timeline(old)}
    _age(old, 40)
    path = write(paths.cache_dir() / stage.TIMELINE_INDEX, json.dumps(index))
    real = cacheprune._rmtree

    def removed_then_made_again(trash):
        errors = real(trash)
        _timeline(old)
        return errors

    monkeypatch.setattr(cacheprune, "_rmtree", removed_then_made_again)
    assert prune("30")["outputs"]["count"] == 1
    assert json.loads(path.read_text()) == index


def test_timeline_index_held_by_another_run_is_left_alone(monkeypatch, caplog, prune):
    from cutpoint import stage  # noqa: PLC0415

    old = paths.cache_dir() / "a-00000000000000aa"
    index = {"/v/old.ts": _timeline(old)}
    _age(old, 40)
    path = write(paths.cache_dir() / stage.TIMELINE_INDEX, json.dumps(index))
    monkeypatch.setattr(stage, "_LOCK_WAIT_S", 0.0)
    with caplog.at_level("DEBUG", "cutpoint.cacheprune"), folders.swap_lock(path.with_suffix("")):
        doc = prune("30")
    assert doc["ok"] and doc["outputs"]["count"] == 1 and doc["warnings"] == []
    assert json.loads(path.read_text()) == index
    assert "timeline index not cleaned" in caplog.text


# --------------------------------------------------------------------------- used marker
def test_source_dir_marks_its_folder_used(tmp_path, monkeypatch):
    src = _source(tmp_path)
    folder = cache.source_dir(src)
    _age(folder, 40)
    cache._USED.clear()
    assert cache.source_dir(src, create=False) == folder
    assert time.time() - folder.stat().st_mtime < 60

    calls = []
    monkeypatch.setattr(cache.os, "utime", lambda *a, **k: calls.append(a))
    cache.source_dir(src)
    assert calls == []  # once per process per USED_INTERVAL_S
    cache._USED[str(folder)] -= cache.USED_INTERVAL_S + 1  # an hour later
    cache.source_dir(src)
    assert calls == [(folder,)]


def test_source_folder_only_names(tmp_path):
    src = _source(tmp_path)
    folder = cache.source_folder(src)
    assert folder == cache.source_dir(src, create=False) and not folder.exists()
    with pytest.raises(OSError):
        cache.source_folder(tmp_path / "gone.mp4")


# --------------------------------------------------------------------------- paths
def test_paths_shows_the_cache_size(capsys):
    write(paths.cache_dir() / "a-00000000000000aa" / "audio16k.wav", b"\0" * 2_500_000)
    assert cli.main(["paths"]) == 0
    (line,) = [x for x in capsys.readouterr().out.splitlines() if x.startswith("cache")]
    assert line.endswith("(2 MB)")
    assert cli.main(["paths", "--json"]) == 0
    assert envelope(capsys.readouterr().out)["outputs"]["cache_bytes"] == 2_500_000


def test_paths_without_a_cache_folder(capsys):
    assert cli.main(["paths", "--json"]) == 0
    out = envelope(capsys.readouterr().out)["outputs"]
    assert out["cache_bytes"] is None and "cache" in out["missing"]
    assert not paths.cache_dir().exists()
