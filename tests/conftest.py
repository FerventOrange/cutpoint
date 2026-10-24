"""Isolate every test from the real user data (CUTPOINT_HOME, env, LOCALAPPDATA)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import pytest

from cutpoint import clips, config, paths, reset_state
from cutpoint.ffmpeg_utils import MediaInfo


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point CUTPOINT_HOME and every platform default at ``tmp_path``; yields the home."""
    for name in list(os.environ):
        if name.startswith("CUTPOINT_"):
            monkeypatch.delenv(name)
    home = tmp_path / "home"
    monkeypatch.setenv("CUTPOINT_HOME", str(home))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "profile"))
    monkeypatch.setenv("HOME", str(tmp_path / "profile"))
    monkeypatch.setattr(config, "_drive_remote", lambda d: None)  # no real mapped drives
    reset_state()
    assert paths.home() == home, "isolation failed: CUTPOINT_HOME not honoured"
    yield home
    reset_state()


@pytest.fixture
def home(isolated_env: Path) -> Path:
    return isolated_env


def write(path: Path, text: str | bytes = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def set_config(home: Path, **data) -> None:
    """Write ``config.json`` under ``home`` and drop the cached copy."""
    write(home / "config.json", json.dumps(data))
    config.get_config.cache_clear()


def remap(home: Path, monkeypatch: pytest.MonkeyPatch, old, new, exclude=()) -> None:
    """``source_remap`` from ``old`` to ``new``; paths compare as plain spellings, on any OS."""
    data = {"source_remap": {str(old): str(new)}}
    if exclude:
        data["source_remap_exclude"] = list(exclude)
    set_config(home, **data)
    monkeypatch.setattr(config, "_prefix_forms", lambda p: (config.norm_path(p),))


def media_info(path: Path = Path("v.mp4"), duration: float = 100.0, **over) -> MediaInfo:
    """A 1920x1080, 30 fps, H.264 + AAC 48 kHz stereo probe result; ``over`` replaces fields."""
    fields = {
        "path": path, "duration": duration, "vduration": duration, "width": 1920, "height": 1080,
        "fps": Fraction(30), "pix_fmt": "yuv420p", "vcodec": "h264", "audio_streams": 1,
        "acodec": "aac", "sample_rate": 48000, "channels": 2,
    }  # fmt: skip
    return MediaInfo(**(fields | over))


def link_file(link: Path, target: Path) -> None:
    """Make ``link`` a file symlink to ``target`` (skip where files cannot be linked)."""
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create a file symlink here")


def link_dir(link: Path, target: Path) -> None:
    """Make ``link`` a directory link to ``target``: a symlink, or a junction on Windows."""
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        if sys.platform != "win32":
            pytest.skip("cannot create a directory symlink here")
    done = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=False
    )
    if done.returncode != 0:
        pytest.skip("cannot create a symlink or junction here")


def hard_link(link: Path, target: Path) -> None:
    try:
        os.link(target, link)
    except (OSError, NotImplementedError, AttributeError) as err:
        pytest.skip(f"hard links are not supported here: {err}")


def messages(caplog: pytest.LogCaptureFixture, level: str = "WARNING", logger=None) -> list[str]:
    """Messages logged at exactly ``level`` (optionally only by logger name ``logger``)."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelname == level and (logger is None or r.name == logger)
    ]


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Opt-in: ``time.sleep`` records its argument instead of waiting; returns the list."""
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    return slept


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch):
    """Returns ``set(flag)``, which makes ``clips._WINDOWS`` report Windows (or not)."""
    return lambda flag=True: monkeypatch.setattr(clips, "_WINDOWS", flag)


@pytest.fixture
def moved(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A 360 clip recorded under ``old/`` whose footage now lives in ``new/`` (``source_remap``)."""
    from pano_helpers import INSV, LRV, x3clip  # noqa: PLC0415

    old, new = tmp_path / "old", tmp_path / "new"
    for name in (*INSV, LRV):
        write(new / name)  # the old folder is gone; no stitched file anywhere
    remap(home, monkeypatch, old, new)
    return x3clip(lrv=str(old / LRV), insv=[str(old / n) for n in INSV]), old, new
