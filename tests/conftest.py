"""Isolate every test from the real user data (CUTPOINT_HOME, env, LOCALAPPDATA)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cutpoint import config, paths, reset_state


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


def write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
