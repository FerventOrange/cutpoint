from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from cutpoint import cache, crash, ffmpeg_utils, telemetry
from cutpoint.events import common


def legacy(params: dict, length: int = 10) -> str:
    """The inline formula used by crash/telemetry/events before cache.params_hash."""
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]


SAMPLES = [
    {},
    {"a": 1, "b": [1, 2, 3], "c": {"z": 1.5, "y": None}},
    {"path": Path("a/b")},  # default=str branch
]


@pytest.mark.parametrize("params", SAMPLES)
def test_params_hash_matches_legacy_formula(params):
    assert cache.params_hash(params) == legacy(params)
    assert cache.params_hash(params, 16) == legacy(params, 16)
    # Pinned: changing this invalidates every user's existing cache entries.
    assert cache.params_hash({"v": 1}) == hashlib.sha1(b'{"v": 1}').hexdigest()[:10]


def test_crash_keys_match_legacy():
    p = crash.CrashParams()
    blob = json.dumps({**asdict(p), "v": crash.CRASH_VERSION}, sort_keys=True, default=str)
    assert p.key() == hashlib.sha1(blob.encode()).hexdigest()[:10]
    blob = json.dumps(
        {"fps": p.dense_fps, "pos": p.pos, "neg": p.neg, "v": crash.CRASH_VERSION}, sort_keys=True
    )
    assert p.dense_key() == hashlib.sha1(blob.encode()).hexdigest()[:10]


def test_telemetry_key_matches_legacy():
    p = telemetry.TelemetryParams()
    blob = json.dumps({**asdict(p), "v": telemetry.TELEMETRY_VERSION}, sort_keys=True)
    assert p.key() == hashlib.sha1(blob.encode()).hexdigest()[:10]


def test_events_params_key_matches_legacy():
    params = {"fps": 3.0, "game": "r6", "tags": ["x"]}
    blob = json.dumps({"v": common.EVENTS_CACHE_VERSION, **params}, sort_keys=True, default=str)
    assert common.params_key(params) == hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def test_save_json_roundtrip_and_no_tmp(tmp_path):
    target = tmp_path / "d.json"
    cache.save_json(target, {"name": "Café", "n": [1, 2]})
    assert cache.load_json(target) == {"name": "Café", "n": [1, 2]}
    assert "Café" in target.read_text(encoding="utf-8")  # ensure_ascii=False
    cache.save_json(target, {"n": 2})
    assert cache.load_json(target) == {"n": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["d.json"]


def test_load_json_missing_or_corrupt(tmp_path):
    assert cache.load_json(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    assert cache.load_json(bad) is None


def test_save_json_retries_permission_error_then_succeeds(tmp_path, monkeypatch):
    target = tmp_path / "d.json"
    real = os.replace
    calls = []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) <= 3:
            raise PermissionError("locked")
        return real(src, dst)

    sleeps = []
    monkeypatch.setattr(cache.os, "replace", flaky)
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", sleeps.append)
    cache.save_json(target, {"ok": True})
    assert len(calls) == 4
    assert len(sleeps) == 3 and all(0 < s <= 0.5 for s in sleeps)
    assert cache.load_json(target) == {"ok": True}
    assert [p.name for p in tmp_path.iterdir()] == ["d.json"]


def test_save_json_always_locked_raises_and_cleans_up(tmp_path, monkeypatch):
    target = tmp_path / "d.json"
    target.write_text('{"old": 1}', encoding="utf-8")
    calls = []

    def locked(src, dst):
        calls.append(1)
        raise PermissionError("locked")

    monkeypatch.setattr(cache.os, "replace", locked)
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", lambda s: None)
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
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", lambda s: None)
    with pytest.raises(PermissionError, match="open in another program") as exc:
        cache.save_json(target, {"new": 2})
    assert str(target) in str(exc.value)
    assert [p.name for p in tmp_path.iterdir()] == []  # temp file removed
