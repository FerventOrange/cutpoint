from __future__ import annotations

import errno
import logging
import os
import sys
from pathlib import Path

import pytest

from cutpoint.events import regions
from cutpoint.events.common import parse_clock
from cutpoint.utils import clock, find_videos, input_name, matches_glob, seconds_keys, slugify


@pytest.mark.parametrize(
    "t, expected",
    [
        (0, "00:00:00.0"),
        (65, "00:01:05.0"),
        (3725.5, "01:02:05.5"),
        (59.96, "00:01:00.0"),  # rounding carries into minutes
        (3599.96, "01:00:00.0"),  # ... and into hours
        (-5.5, "-00:00:05.5"),
        (-0.04, "00:00:00.0"),  # no "-0"
    ],
)
def test_clock(t, expected):
    assert clock(t) == expected


@pytest.mark.parametrize("t", [-5.5, 59.96, 3599.96])
def test_parse_clock_round_trips_clock(t):
    assert parse_clock(clock(t)) == pytest.approx(t, abs=0.05)


@pytest.mark.parametrize(
    "path, patterns, expected",
    [
        ("C:\\Videos\\Round4_DVR.mp4", ["*DVR*"], True),
        ("C:\\Videos\\Round4_dvr.mp4", ["*DVR*"], True),  # case-insensitive
        ("C:\\Videos\\Captures\\a.mp4", ["*/Captures/*"], True),  # backslashes == slashes
        ("/videos/a.mp4", ["*/Captures/*"], False),
        ("/videos/a.mp4", ["b.mp4", "a.mp4"], True),  # any pattern, file name
        ("/videos/a.mp4", ["a.mp"], False),  # whole-name match only
    ],
)
def test_matches_glob(path, patterns, expected):
    assert matches_glob(path, patterns) is expected


@pytest.mark.parametrize(
    ("make", "name"),
    [
        (lambda d: d / "a.b.mp4", "a.b"),  # a file: its stem
        (lambda d: d / "trip.2024", "trip.2024"),  # a folder: its whole name
        (lambda d: Path(d.anchor), "fallback"),  # a drive root has no name
    ],
)
def test_input_name(tmp_path, make, name):
    target = make(tmp_path)
    if target.suffix == ".mp4":
        target.write_bytes(b"x")
    else:
        target.mkdir(exist_ok=True)
    assert input_name(target, "fallback") == name


def test_input_name_resolves_the_current_folder(tmp_path, monkeypatch):
    folder = tmp_path / "trip.2024"
    folder.mkdir()
    monkeypatch.chdir(folder)
    assert input_name(Path(), "fallback") == "trip.2024"


@pytest.mark.parametrize(
    ("value", "allow_negative", "expected"),
    [
        (None, False, {}),
        (2, False, {"t": 2.0}),
        (1.5, False, {"t": 1.5}),
        ("x", False, "t must be a number of seconds"),
        (float("nan"), False, "t must be a number of seconds"),
        (-1, False, "t must not be negative"),
        (-1, True, {"t": -1.0}),
    ],
)
def test_seconds_keys(value, allow_negative, expected):
    item = {"t": value, "other": "ignored"}
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=f"^seg: {expected}$"):
            seconds_keys(item, ("t", "missing"), "seg", allow_negative=allow_negative)
    else:
        assert (
            seconds_keys(item, ("t", "missing"), "seg", allow_negative=allow_negative) == expected
        )


def test_slugify():
    assert slugify("Caf\u00e9 Racer") == "cafe-racer"  # folds accents
    assert slugify("  Hello,  World!! ") == "hello-world"
    assert slugify("!!!") == "moment"
    assert slugify("???", default="x") == "x"
    assert slugify("a" * 100) == "a" * 40
    assert slugify("a" * 100, max_len=None) == "a" * 100
    assert slugify("ab cd", max_len=3) == "ab"  # no trailing dash after cut


@pytest.mark.parametrize(
    "name, key",
    [
        ("Caf\u00e9 Racer", "caf-racer"),  # legacy: no Unicode folding
        ("Battlefield\u2122 2042", "battlefield-2042"),
        ("Counter-strike 2", "counter-strike-2"),
        ("\u2122\u2122\u2122", "game"),
    ],
)
def test_game_slug_keeps_legacy_keys(name, key):
    assert regions._game_slug(name) == key


# -- find_videos: unreadable inputs ------------------------------------------------------

needs_chmod = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod 000 does not block this user",
)


@pytest.fixture
def locked():
    """Make a folder mode 000 for the test; restore it so tmp_path cleanup works."""
    paths = []

    def lock(folder):
        folder.chmod(0)
        paths.append(folder)
        return folder

    yield lock
    for folder in paths:
        folder.chmod(0o755)


@needs_chmod
@pytest.mark.parametrize("case", ["folder", "file_under_locked_parent"])
def test_find_videos_unreadable_input_is_a_clear_error(tmp_path, locked, case):
    (tmp_path / "vault").mkdir()
    (tmp_path / "vault" / "a.mp4").write_bytes(b"x")
    locked(tmp_path / "vault")
    target = tmp_path / "vault" if case == "folder" else tmp_path / "vault" / "a.mp4"
    with pytest.raises(FileNotFoundError, match="not readable"):
        find_videos(target)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX name limits")
def test_find_videos_name_too_long_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not readable"):
        find_videos(tmp_path / ("x" * 300) / "a.mp4")


@needs_chmod
def test_find_videos_walk_skips_an_unreadable_subfolder_with_a_warning(tmp_path, locked, caplog):
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "good.mp4").write_bytes(b"x")
    (tmp_path / "vault").mkdir()
    (tmp_path / "vault" / "hidden.mp4").write_bytes(b"x")
    locked(tmp_path / "vault")
    with caplog.at_level(logging.WARNING):
        got = find_videos(tmp_path, recursive=True)
    assert got == [(tmp_path / "ok" / "good.mp4").resolve()]
    assert any(r.levelno == logging.WARNING and "vault" in r.getMessage() for r in caplog.records)


@needs_chmod
def test_find_videos_walk_does_not_warn_about_an_excluded_unreadable_folder(
    tmp_path, locked, caplog
):
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "good.mp4").write_bytes(b"x")
    (tmp_path / "locked").mkdir()
    (tmp_path / "locked" / "hidden.mp4").write_bytes(b"x")
    locked(tmp_path / "locked")
    with caplog.at_level(logging.WARNING):
        got = find_videos(tmp_path, recursive=True, exclude=["*/locked/*"])
    assert got == [(tmp_path / "ok" / "good.mp4").resolve()]
    assert not any("Skipping unreadable" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("kind", ["file", "folder"])
def test_find_videos_unreadable_input_is_not_reported_missing_when_checks_return_false(
    tmp_path, monkeypatch, kind
):
    """Python 3.13 makes is_file/is_dir return False on EACCES instead of raising."""
    target = tmp_path / "vault" / "a.mp4" if kind == "file" else tmp_path / "vault"
    (tmp_path / "vault").mkdir()
    if kind == "file":
        target.write_bytes(b"x")

    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if self == target:
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_stat(self, *args, **kwargs)

    def false_for_target(real):
        return lambda self, *a, **k: False if self == target else real(self, *a, **k)

    monkeypatch.setattr(Path, "stat", stat)
    for name in ("is_file", "is_dir"):
        monkeypatch.setattr(Path, name, false_for_target(getattr(Path, name)))
    with pytest.raises(FileNotFoundError, match="Input not readable"):
        find_videos(target)
