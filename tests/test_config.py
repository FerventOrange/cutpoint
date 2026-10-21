"""Tests for config.json validation (player_tags, pano) and the stitch drive warning."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import write
from cutpoint import config
from cutpoint.pano import project, stitch


def set_config(home: Path, **data) -> None:
    write(home / "config.json", json.dumps(data))
    config.get_config.cache_clear()


# -- player_tags ----------------------------------------------------------------------


def test_player_tags_single_string_becomes_list(home):
    set_config(home, player_tags="Me")
    assert config.player_tags() == ("Me",)


@pytest.mark.parametrize("bad", [5])
def test_player_tags_non_list_raises(home, bad):
    set_config(home, player_tags=bad)
    with pytest.raises(ValueError, match="player_tags must be a list"):
        config.get_config()


def test_player_tags_blank_names_dropped(home):
    set_config(home, player_tags=["Me", "", "   ", "\t", "Other"])
    assert config.player_tags() == ("Me", "Other")
    set_config(home, player_tags="   ")
    assert config.player_tags() == ()


# -- pano_config ----------------------------------------------------------------------


@pytest.mark.parametrize("bad", [["a"]])
def test_pano_config_non_object_raises(home, bad):
    set_config(home, pano=bad)
    with pytest.raises(ValueError, match="'pano' must be an object"):
        project.pano_config()


# -- stitch._warn_system_drive ----------------------------------------------------------
windows_only = pytest.mark.skipif(os.name != "nt", reason="drive letters are Windows-only")


def make_clips(n: int = 2) -> list:
    return [SimpleNamespace(duration=1800.0) for _ in range(n)]


def stitch_warnings(caplog) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records if r.name == stitch.logger.name and r.levelno == logging.WARNING
    ]


@windows_only
def test_warns_when_unset_and_on_system_drive(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(stitch, "pano_config", dict)
    monkeypatch.setenv("SystemDrive", tmp_path.resolve().drive)
    with caplog.at_level(logging.WARNING, logger=stitch.logger.name):
        stitch._warn_system_drive(tmp_path / "out", make_clips(2))
    (rec,) = stitch_warnings(caplog)
    msg = rec.getMessage()
    assert "system drive" in msg and "2 clip(s)" in msg and "stitched_dir" in msg
    assert f"{2 * 1800 / 3600 * 45 * 1.2:.0f} GB" in msg


@windows_only
def test_no_warning_when_stitched_dir_configured(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(stitch, "pano_config", lambda: {"stitched_dir": str(tmp_path)})
    monkeypatch.setenv("SystemDrive", tmp_path.resolve().drive)
    with caplog.at_level(logging.WARNING, logger=stitch.logger.name):
        stitch._warn_system_drive(tmp_path / "out", make_clips())
    assert stitch_warnings(caplog) == []


@windows_only
def test_no_warning_when_output_on_another_drive(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(stitch, "pano_config", dict)
    other = "Q:" if tmp_path.resolve().drive.upper() != "Q:" else "R:"
    monkeypatch.setenv("SystemDrive", other)
    with caplog.at_level(logging.WARNING, logger=stitch.logger.name):
        stitch._warn_system_drive(tmp_path / "out", make_clips())
    assert stitch_warnings(caplog) == []
