"""Tests for config.json validation (player_tags, pano) and the stitch drive warning."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import write
from cutpoint import cli, config
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


# -- set_config_value -----------------------------------------------------------------


def test_set_config_value_creates_the_file_and_refreshes_the_cache(home):
    target = home / "projects-here"
    assert config.projects_dir() != target  # primes the cache
    written = config.set_config_value("projects_dir", str(target))
    assert written == home / "config.json"
    assert json.loads(written.read_text(encoding="utf-8")) == {"projects_dir": str(target)}
    assert config.projects_dir() == target


@pytest.mark.parametrize(
    ("text", "indent"),
    [
        ('{\n    "b": 1,\n    "a": 2\n}', "    "),
        ('{\n\t"b": 1,\n\t"a": 2\n}', "\t"),
        ('{"b": 1, "a": 2}', "  "),
    ],
)
def test_set_config_value_keeps_keys_order_and_indent(home, text, indent):
    path = write(home / "config.json", text)
    config.set_config_value("a", 3)
    config.set_config_value("c", 4)
    out = path.read_text(encoding="utf-8")
    assert json.loads(out) == {"b": 1, "a": 3, "c": 4}
    assert list(json.loads(out)) == ["b", "a", "c"]
    assert f'\n{indent}"b": 1' in out
    assert not list(home.glob("*.tmp")) and [p.name for p in home.iterdir()] == ["config.json"]


def test_set_config_value_refuses_an_invalid_file_and_leaves_it(home):
    path = write(home / "config.json", "{oops")
    with pytest.raises(ValueError, match="Invalid config file"):
        config.set_config_value("projects_dir", "x")
    assert path.read_text(encoding="utf-8") == "{oops"
    write(path, "[1]")
    with pytest.raises(ValueError, match="expected a JSON object"):
        config.set_config_value("projects_dir", "x")
    assert path.read_text(encoding="utf-8") == "[1]"


def test_set_config_value_writes_an_explicit_path(tmp_path):
    target = tmp_path / "sub" / "c.json"
    assert config.set_config_value("k", "v", target) == target
    assert json.loads(target.read_text(encoding="utf-8")) == {"k": "v"}


def test_resolve_projects_dir_prefers_env_then_data_then_default(tmp_path, monkeypatch):
    assert config.resolve_projects_dir({}) == config.paths.default_projects_dir()
    assert config.resolve_projects_dir({"projects_dir": str(tmp_path / "cfg")}) == tmp_path / "cfg"
    monkeypatch.setenv("CUTPOINT_PROJECTS_DIR", str(tmp_path / "env"))
    got = config.resolve_projects_dir({"projects_dir": str(tmp_path / "cfg")})
    assert got == tmp_path / "env"


# -- wrongly typed values and file encodings, through the CLI ---------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("projects_dir", 5),
        ("stage_dir", ["a"]),
        ("ollama_url", 5),
        ("video_encoder", 5),
        ("source_remap", ["a"]),
        ("source_remap", {"D:/": 5}),
        ("source_remap_exclude", "D:/"),
        ("source_remap_exclude", [5]),
        ("player_tags", 5),
        ("player_tags", [5]),
    ],
)
def test_wrongly_typed_config_value_is_one_error_not_a_traceback(home, caplog, key, value):
    """#138/#140: a wrongly typed config.json value exits 1 with one ERROR naming file and key."""
    set_config(home, **{key: value})
    assert cli.main(["paths"]) == 1
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert str(home / "config.json") in errors[0].getMessage()
    assert key in errors[0].getMessage()
    assert all(r.exc_info is None for r in caplog.records)


def test_config_with_utf8_bom_loads(home):
    """#140 follow-up: PowerShell 5.1 writes config.json with a BOM; it must still load."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.json").write_bytes(b'\xef\xbb\xbf{"player_tags": ["Me"]}')
    config.get_config.cache_clear()
    assert config.player_tags() == ("Me",)
