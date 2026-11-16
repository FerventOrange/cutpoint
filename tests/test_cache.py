from __future__ import annotations

import json
import os
from pathlib import PurePosixPath

import numpy as np
import pytest

from cutpoint import actioncam, activity, cache, crash, telemetry
from cutpoint.events import common
from cutpoint.events import visual as V
from cutpoint.events import voice as VO
from cutpoint.pano import views as VW

_GENERIC = activity.full(activity.builtin("generic"))


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
        (crash.CrashParams().with_profile(_GENERIC).key(), "94bd9ff1d6"),  # the default profile
        (crash.CrashParams().with_profile(_GENERIC).dense_key(), "ccbc2d50ef"),
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
    [("utf-8-sig", {"name": "Café"}), ("utf-16", None)],
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


def test_source_dir_changes_when_only_the_modification_time_changes(home, tmp_path):
    """A re-recorded or re-exported file of the same size must not reuse the old analysis."""
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"same size")
    first = cache.source_dir(src)
    st = src.stat()
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))

    assert src.stat().st_size == st.st_size
    assert cache.source_dir(src) != first
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns))  # back: the old folder again
    assert cache.source_dir(src) == first


def test_save_npz_then_load_npz_round_trips(tmp_path):
    path = tmp_path / "a.npz"
    cache.save_npz(path, x=np.arange(4, dtype=np.float32), y=np.zeros((2, 3)))
    got = cache.load_npz(path)
    assert got is not None
    assert got["x"].dtype == np.float32
    assert got["x"].tolist() == [0, 1, 2, 3]
    assert got["y"].shape == (2, 3)
    assert [p.name for p in tmp_path.iterdir()] == ["a.npz"]
    path.unlink()  # no handle left open: the file can be removed (and replaced) on Windows


def test_load_npz_is_none_for_missing_truncated_or_garbage(tmp_path):
    assert cache.load_npz(tmp_path / "none.npz") is None
    good = tmp_path / "good.npz"
    cache.save_npz(good, x=np.arange(1000))
    cut = tmp_path / "cut.npz"
    cut.write_bytes(good.read_bytes()[:40])
    junk = tmp_path / "junk.npz"
    junk.write_bytes(b"not a zip at all")
    empty = tmp_path / "empty.npz"
    empty.write_bytes(b"")
    assert cache.load_npz(cut) is None
    assert cache.load_npz(junk) is None
    assert cache.load_npz(empty) is None


def _truncated(path):
    path.write_bytes(b"PK half written")
    return path


def test_load_voice_and_visual_treat_a_truncated_cache_as_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(VO, "cache_path", lambda *_a, **_k: _truncated(tmp_path / "v.npz"))
    monkeypatch.setattr(V, "cache_path", lambda *_a, **_k: _truncated(tmp_path / "f.npz"))
    assert VO.load_voice(tmp_path / "c.mp4") is None
    assert V.load_visual(tmp_path / "c.mp4") is None


def test_load_views_treats_a_truncated_cache_as_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(VW, "cache_path", lambda *_a, **_k: _truncated(tmp_path / "w.npz"))
    assert VW.load_views(tmp_path / "c.mp4") is None


def test_cached_speed_treats_a_truncated_cache_as_a_miss(home, tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    p = telemetry.TelemetryParams()
    _truncated(cache.source_dir(src) / f"telemetry-{p.key()}.npz")
    assert telemetry.cached_speed(src) is None


def test_imu_grids_recomputes_a_truncated_cache(home, tmp_path, monkeypatch):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    path = _truncated(cache.source_dir(src) / f"crashimu-v{crash.IMU_VERSION}.npz")
    calls = []
    monkeypatch.setattr(telemetry, "imu_read_path", lambda s: (None, None, s, False))
    monkeypatch.setattr(crash, "_compute_imu", lambda *a: calls.append(a))
    assert crash.imu_grids(src, 10.0) is None
    assert len(calls) == 1
    assert "none" in cache.load_npz(path)  # rewritten whole


def test_load_npz_never_raises_on_corrupt_compressed_data(tmp_path):
    good = tmp_path / "good.npz"
    cache.save_npz(good, x=np.arange(5000, dtype=np.int64))
    raw = good.read_bytes()
    bad = tmp_path / "bad.npz"
    hits = 0
    for i in range(30, len(raw) - 100, 7):  # inside the compressed member data
        bad.write_bytes(raw[:i] + bytes([raw[i] ^ 0xFF]) + raw[i + 1 :])
        hits += cache.load_npz(bad) is None
    assert hits  # some of the corruptions are caught, none raise


def test_load_npz_of_an_empty_archive_is_a_hit(tmp_path):
    path = tmp_path / "empty.npz"
    cache.save_npz(path)
    assert cache.load_npz(path) == {}


def test_extract_recomputes_a_truncated_telemetry_cache(home, tmp_path, monkeypatch):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    p = telemetry.TelemetryParams()
    stem = cache.source_dir(src) / f"telemetry-{p.key()}"
    cache.save_json(stem.with_suffix(".json"), {"duration": 1.0, "events": [], "meta": {}})
    _truncated(stem.with_suffix(".npz"))
    calls = []
    monkeypatch.setattr(telemetry, "imu_read_path", lambda s: (None, None, s, False))
    monkeypatch.setattr(telemetry, "_compute", lambda *a: calls.append(a))
    assert telemetry.extract(src, 1.0, p) is None
    assert len(calls) == 1


def test_text_embeddings_recompute_a_truncated_cache(tmp_path, monkeypatch):
    path = tmp_path / "text-x.npy"
    path.write_bytes(b"\x93NUMPY half")
    monkeypatch.setattr(V, "_prompt_file", lambda _p: path)

    class Models:
        def encode_text(self, prompts):
            return np.ones((len(prompts), 4))

    got = V.text_embeddings(("a",), Models())
    assert got.dtype == np.float32
    assert np.load(path).shape == (1, 4)  # rewritten whole


def test_load_part_treats_a_corrupt_features_file_as_a_miss(tmp_path):
    src = tmp_path / "clip.mp4"
    path = actioncam.features_path(tmp_path, src, 0.0)
    path.parent.mkdir(parents=True)
    _truncated(path)
    assert actioncam.load_part(tmp_path, src) is None
