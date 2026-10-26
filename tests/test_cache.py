from __future__ import annotations

import json
import os
from pathlib import PurePosixPath

import pytest

from cutpoint import cache, crash, telemetry
from cutpoint.events import common


# Pinned digests: a change here renames every user's existing cache files (and so re-runs
# their analysis). Change them only on purpose, with a version bump where one exists.
@pytest.mark.parametrize(
    ("params", "digest10", "digest16"),
    [
        ({}, "bf21a9e8fb", "bf21a9e8fbc5a384"),
        (
            {"a": 1, "b": [1, 2, 3], "c": {"z": 1.5, "y": None}},
            "76284c8594",
            "76284c8594057042",
        ),
        ({"path": PurePosixPath("a/b")}, "93c752d7fd", "93c752d7fdfe60b2"),  # default=str branch
    ],
)
def test_params_hash_is_pinned(params, digest10, digest16):
    assert cache.params_hash(params) == digest10
    assert cache.params_hash(params, 16) == digest16


@pytest.mark.parametrize(
    ("key", "digest"),
    [
        (crash.CrashParams().key(), "e7244b388a"),
        (crash.CrashParams().dense_key(), "24551ce592"),
        (telemetry.TelemetryParams().key(), "f60a858f09"),
        (common.params_key({"fps": 3.0, "game": "boxfeed", "tags": ["x"]}), "4183f1e1b3"),
    ],
    ids=["crash", "crash-dense", "telemetry", "events"],
)
def test_cache_keys_of_the_default_params_are_pinned(key, digest):
    assert key == digest


def test_save_json_roundtrip_and_no_tmp(tmp_path):
    target = tmp_path / "d.json"
    cache.save_json(target, {"name": "Café", "n": [1, 2]})
    assert cache.load_json(target) == {"name": "Café", "n": [1, 2]}
    assert "Café" in target.read_text(encoding="utf-8")  # ensure_ascii=False
    cache.save_json(target, {"n": 2})
    assert cache.load_json(target) == {"n": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["d.json"]


def test_save_json_bytes_are_utf8_with_the_platform_line_endings(tmp_path):
    target = tmp_path / "d.json"
    cache.save_json(target, {"name": "Café", "n": [1]})
    text = json.dumps({"name": "Café", "n": [1]}, indent=2, ensure_ascii=False)
    assert target.read_bytes() == text.replace("\n", os.linesep).encode("utf-8")


def test_load_json_missing_or_corrupt(tmp_path):
    assert cache.load_json(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    assert cache.load_json(bad) is None


def test_read_json_raises_instead_of_returning_none(tmp_path):
    with pytest.raises(FileNotFoundError):
        cache.read_json(tmp_path / "nope.json")
    # a directory: IsADirectoryError on Linux, PermissionError on Windows (both OSError)
    (tmp_path / "h.json").mkdir()
    with pytest.raises(ValueError, match="cannot be read") as err:
        cache.read_json(tmp_path / "h.json", hint="; re-run it")
    assert str(tmp_path / "h.json") in str(err.value)
    assert str(err.value).endswith("; re-run it")


def test_read_json_rejects_bad_utf8(tmp_path):
    # every reader shares this decode path, so it is covered here once
    (tmp_path / "bad.json").write_bytes(b'{"a": "\xff\xfe"}')
    with pytest.raises(ValueError, match="is not valid JSON") as err:
        cache.read_json(tmp_path / "bad.json", hint="; re-run it")
    assert str(err.value).endswith("; re-run it")


def test_save_json_retries_permission_error_then_succeeds(tmp_path, monkeypatch, no_sleep):
    target = tmp_path / "d.json"
    real = os.replace
    calls = []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) <= 3:
            raise PermissionError("locked")
        return real(src, dst)

    monkeypatch.setattr(cache.os, "replace", flaky)
    cache.save_json(target, {"ok": True})
    assert len(calls) == 4
    assert len(no_sleep) == 3 and all(0 < s <= 0.5 for s in no_sleep)
    assert cache.load_json(target) == {"ok": True}
    assert [p.name for p in tmp_path.iterdir()] == ["d.json"]


@pytest.mark.usefixtures("no_sleep")
def test_save_json_always_locked_raises_and_cleans_up(tmp_path, monkeypatch):
    target = tmp_path / "d.json"
    target.write_text('{"old": 1}', encoding="utf-8")
    calls = []

    def locked(src, dst):
        calls.append(1)
        raise PermissionError("locked")

    monkeypatch.setattr(cache.os, "replace", locked)
    with pytest.raises(PermissionError):
        cache.save_json(target, {"new": 2})
    assert len(calls) == cache.REPLACE_ATTEMPTS
    assert cache.load_json(target) == {"old": 1}  # original untouched
    assert [p.name for p in tmp_path.iterdir()] == ["d.json"]


def test_save_json_other_error_not_retried_no_tmp(tmp_path, monkeypatch):
    calls = []

    def boom(src, dst):
        calls.append(1)
        raise OSError("disk gone")

    monkeypatch.setattr(cache.os, "replace", boom)
    with pytest.raises(OSError):
        cache.save_json(tmp_path / "d.json", {"a": 1})
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def test_save_json_unserialisable_leaves_nothing(tmp_path):
    with pytest.raises(TypeError):
        cache.save_json(tmp_path / "d.json", {"a": object()})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("encoding", "expected"),
    [("utf-8-sig", {"name": "Café"}), ("utf-16", None), ("cp1252", None)],
)
def test_load_json_handles_the_encoding_a_file_was_saved_with(tmp_path, encoding, expected):
    target = tmp_path / "saved.json"
    target.write_bytes(json.dumps({"name": "Café"}, ensure_ascii=False).encode(encoding))
    assert cache.load_json(target) == expected


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("winerror", [32, 5])
def test_save_json_locked_file_gets_the_open_in_another_program_hint(
    tmp_path, monkeypatch, winerror
):
    target = tmp_path / "d.json"

    def locked(src, dst):
        err = PermissionError(13, "The process cannot access the file")
        err.winerror = winerror
        raise err

    monkeypatch.setattr(cache.os, "replace", locked)
    with pytest.raises(PermissionError, match="open in another program") as exc:
        cache.save_json(target, {"new": 2})
    assert str(target) in str(exc.value)
    assert [p.name for p in tmp_path.iterdir()] == []  # temp file removed
