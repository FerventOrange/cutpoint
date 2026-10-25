"""Tests for config.json validation (player_tags, pano) and the stitch drive warning."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import messages, set_config, write
from cutpoint import cli, config
from cutpoint.pano import project, stitch

# -- player_tags ----------------------------------------------------------------------


def test_player_tags_single_string_becomes_list(home):
    set_config(home, player_tags="Me")
    assert config.player_tags() == ("Me",)


def test_player_tags_blank_names_dropped(home):
    set_config(home, player_tags=["Me", "", "   ", "\t", "Other"])
    assert config.player_tags() == ("Me", "Other")
    set_config(home, player_tags="   ")
    assert config.player_tags() == ()


# -- vision ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "want"),
    [({}, False), ({"vision": None}, False), ({"vision": False}, False), ({"vision": True}, True)],
)
def test_vision_default_is_off_unless_true(home, data, want):
    set_config(home, **data)
    assert config.vision_default() is want


@pytest.mark.parametrize("value", ["yes", "true", 1, 0, [True], {"on": True}])
def test_vision_must_be_a_boolean(home, value):
    set_config(home, vision=value)
    with pytest.raises(ValueError, match=r"vision must be true or false"):
        config.get_config()


def test_vision_is_read_from_cutpoint_config(home, tmp_path, monkeypatch):
    other = write(tmp_path / "elsewhere" / "c.json", json.dumps({"vision": True}))
    write(home / "config.json", json.dumps({"vision": False}))
    monkeypatch.setenv("CUTPOINT_CONFIG", str(other))
    config.get_config.cache_clear()
    assert config.vision_default() is True


@pytest.mark.parametrize(
    ("cfg", "argv", "vision"),
    [
        (None, [], False),
        (None, ["--vision"], True),
        (None, ["--no-vision"], False),
        (False, [], False),
        (False, ["--vision"], True),
        (True, [], True),
        (True, ["--no-vision"], False),
    ],
)
def test_analyze_vision_flag_wins_else_config(home, tmp_path, monkeypatch, cfg, argv, vision):
    import cutpoint.analyze  # noqa: PLC0415

    if cfg is not None:
        set_config(home, vision=cfg)
    video = write(tmp_path / "a.mp4")
    seen: list = []
    monkeypatch.setattr(cutpoint.analyze, "analyze", lambda *a, **kw: seen.append(kw["vision"]))
    assert cli.main(["analyze", str(video), "--out", "p", *argv]) == 0
    assert seen == [vision]


# -- pano_config ----------------------------------------------------------------------


def test_pano_config_non_object_raises(home):
    set_config(home, pano=["a"])
    with pytest.raises(ValueError, match="'pano' must be an object"):
        project.pano_config()


# -- stitch._warn_system_drive ----------------------------------------------------------
windows_only = pytest.mark.skipif(os.name != "nt", reason="drive letters are Windows-only")


def make_clips(n: int = 2) -> list:
    return [SimpleNamespace(duration=1800.0) for _ in range(n)]


@windows_only
@pytest.mark.parametrize(
    ("pano", "drive_is_tmp", "warns"),
    [
        ({}, True, True),
        ({"stitched_dir": "configured"}, True, False),
        ({}, False, False),
    ],
    ids=["unset-on-system-drive", "stitched-dir-configured", "output-on-another-drive"],
)
def test_warn_system_drive(tmp_path, monkeypatch, caplog, pano, drive_is_tmp, warns):
    pano = {k: str(tmp_path) if v == "configured" else v for k, v in pano.items()}
    monkeypatch.setattr(stitch, "pano_config", lambda: pano)
    drive = tmp_path.resolve().drive
    other = "Q:" if drive.upper() != "Q:" else "R:"
    monkeypatch.setenv("SystemDrive", drive if drive_is_tmp else other)
    with caplog.at_level(logging.WARNING, logger=stitch.logger.name):
        stitch._warn_system_drive(tmp_path / "out", make_clips(2))
    got = messages(caplog, logger=stitch.logger.name)
    if not warns:
        assert got == []
        return
    (msg,) = got
    assert "system drive" in msg and "2 clip(s)" in msg and "stitched_dir" in msg
    assert f"{2 * 1800 / 3600 * 45 * 1.2:.0f} GB" in msg


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


def test_set_config_values_writes_several_keys_at_once(home):
    path = write(home / "config.json", '{"b": 1}')
    config.set_config_values({"vision": True, "projects_dir": "x"})
    want = {"b": 1, "vision": True, "projects_dir": "x"}
    assert json.loads(path.read_text(encoding="utf-8")) == want


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


# -- source_remap_exclude with mapped drives (#193) -------------------------------------


def _fake_unc(path: str) -> str:
    """``config.unc_path`` with ``Q:`` mapped to ``\\\\nas\\share`` (no real drive, any OS)."""
    if re.match(r"^[Qq]:(?:[\\/]|$)", path):
        return r"\\nas\share" + path[2:].replace("/", "\\")
    return path


@pytest.mark.parametrize(
    ("exclude", "typed", "excluded"),
    [
        (r"Q:\Footage\skip\*", r"\\nas\share\Footage\skip\x.mp4", True),
        (r"\\nas\share\Footage\skip\*", r"Q:\Footage\skip\x.mp4", True),
        ("Q:/Footage/skip/*", r"\\nas\share\Footage\skip\x.mp4", True),
        (r"q:\footage\SKIP\*", r"\\NAS\share\Footage\skip\x.mp4", True),
        (r"Q:\Footage\skip\*", r"Q:\Footage\skip\x.mp4", True),
        (r"\\nas\share\Footage\skip\*", r"\\nas\share\Footage\skip\x.mp4", True),
        ("*skip*", r"\\nas\share\Footage\skip.mp4", True),
        (r"Q:\Footage\skip\*", r"\\nas\share\Footage\keep\x.mp4", False),
        (r"Q:\Footage\skip\*", r"\\other\share\Footage\skip\x.mp4", False),
        (r"R:\Footage\skip\*", r"\\nas\share\Footage\skip\x.mp4", False),
        (r"R:\Footage\skip\*", r"R:\Footage\skip\x.mp4", True),
    ],
    ids=[
        "drive-glob-unc-source",
        "unc-glob-drive-source",
        "drive-glob-forward-slashes",
        "other-case",
        "drive-glob-drive-source",
        "unc-glob-unc-source",
        "name-glob",
        "other-folder",
        "other-share",
        "unmapped-drive-not-converted",
        "unmapped-drive-as-before",
    ],
)
def test_remap_exclude_matches_mapped_drive_and_unc_spellings(
    home, monkeypatch, exclude, typed, excluded
):
    """#193: a ``Q:`` exclude also excludes the ``\\\\server\\share`` spelling, and back."""
    monkeypatch.setattr(config, "unc_path", _fake_unc)
    set_config(home, source_remap_exclude=[exclude])
    assert config._remap_excluded(typed) is excluded


@pytest.mark.parametrize("exclude", [[], [r"Q:\Footage\skip\*"]], ids=["no-exclude", "drive-glob"])
def test_remap_source_keeps_a_unc_source_excluded_by_a_drive_glob(
    home, tmp_path, monkeypatch, exclude
):
    """#193: a UNC-recorded source excluded in drive form is read where it is, not remapped."""
    monkeypatch.setattr(config, "unc_path", _fake_unc)
    monkeypatch.setattr(config, "_prefix_forms", lambda p: (config.windows_norm(p),))
    copy = write(tmp_path / "copy" / "skip" / "x.mp4")
    set_config(
        home,
        source_remap={r"\\nas\share\Footage": str(tmp_path / "copy")},
        source_remap_exclude=exclude,
    )
    typed = r"\\nas\share\Footage\skip\x.mp4"
    assert config.remap_source(typed) == (Path(typed) if exclude else copy)


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
    if (key, value) == ("player_tags", 5):
        assert "player_tags must be a list" in errors[0].getMessage()
    assert all(r.exc_info is None for r in caplog.records)


def test_config_encoding_bom_loads_and_utf16_is_one_clear_error(home, caplog):
    """#140 follow-up: a BOM (PowerShell 5.1) loads; UTF-16 is one error saying to save as UTF-8."""
    home.mkdir(parents=True, exist_ok=True)
    path = home / "config.json"
    path.write_bytes(b'\xef\xbb\xbf{"player_tags": ["Me"]}')
    config.get_config.cache_clear()
    assert config.player_tags() == ("Me",)
    path.write_text('{"player_tags": ["Me"]}', encoding="utf-16")
    config.get_config.cache_clear()
    assert cli.main(["paths"]) == 1
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert str(path) in errors[0].getMessage()
    assert "not UTF-8 text" in errors[0].getMessage()
