from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cutpoint import paths
from cutpoint.events import regions


@pytest.mark.parametrize(
    "name, func, default",
    [
        ("CONFIG", paths.config_file, "config.json"),
        ("CACHE_DIR", paths.cache_dir, "cache"),
        ("MODELS_DIR", paths.models_dir, "models"),
        ("TOOLS_DIR", paths.tools_dir, "tools"),
        ("STYLES_DIR", paths.styles_dir, "styles"),
    ],
)
def test_env_overrides_and_empty_falls_through(name, func, default, home, tmp_path, monkeypatch):
    assert func() == home / default
    monkeypatch.setenv("CUTPOINT_" + name, str(tmp_path / "cp"))
    assert func() == tmp_path / "cp"
    monkeypatch.setenv("CUTPOINT_" + name, "")  # empty counts as unset
    assert func() == home / default


def test_values_read_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setenv("CUTPOINT_HOME", str(tmp_path / "a"))
    assert paths.cache_dir() == tmp_path / "a" / "cache"
    monkeypatch.setenv("CUTPOINT_HOME", str(tmp_path / "b"))
    assert paths.cache_dir() == tmp_path / "b" / "cache"
    assert paths.presets_dir() == tmp_path / "b" / "presets"
    assert paths.default_stage_dir() == tmp_path / "b" / "stage"
    assert paths.default_projects_dir() == tmp_path / "b" / "projects"


def test_relative_home_becomes_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUTPOINT_HOME", str(Path("rel") / "home"))
    assert paths.home().is_absolute()
    assert paths.home() == tmp_path / "rel" / "home"
    assert paths.cache_dir() == tmp_path / "rel" / "home" / "cache"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows default")
def test_default_home_under_localappdata(tmp_path, monkeypatch):
    monkeypatch.delenv("CUTPOINT_HOME")
    assert paths.home() == tmp_path / "localappdata" / "cutpoint"
    assert paths.default_home() == paths.home()


def test_default_home_non_windows_uses_xdg(tmp_path, monkeypatch):
    monkeypatch.delenv("CUTPOINT_HOME")
    monkeypatch.setattr(paths.sys, "platform", "linux")
    assert paths.home() == tmp_path / "xdg" / "cutpoint"


def test_nothing_is_created(home):
    paths.cache_dir()
    paths.models_dir()
    paths.presets_dir()
    assert not home.exists()


# -- preset lookup order: project -> user -> bundled -------------------------
@pytest.fixture
def preset_env(tmp_path, home, monkeypatch):
    bundled = tmp_path / "bundled"
    monkeypatch.setattr(paths, "PRESETS_DIR", bundled)
    project = tmp_path / "proj"

    def put(folder: Path, tag: str, game: str = "foo") -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        f = folder / f"{game}.json"
        f.write_text(
            json.dumps({"game": game, "regions": [], "player_tags": [tag]}), encoding="utf-8"
        )
        return f

    return put, project, project / "hud_presets", paths.presets_dir(), bundled


def test_preset_lookup_order(preset_env):
    put, project, proj_dir, user_dir, bundled_dir = preset_env
    pb = put(bundled_dir, "bundled")
    pu = put(user_dir, "user")
    pp = put(proj_dir, "project")
    assert regions.preset_files(project) == [pp, pu, pb]
    assert regions.preset_files() == [pu, pb]
    assert regions.load_preset("foo", project).player_tags == ("project",)
    assert regions.load_preset("foo").player_tags == ("user",)
    pp.unlink()
    assert regions.load_preset("foo", project).player_tags == ("user",)
    pu.unlink()
    assert regions.load_preset("foo", project).player_tags == ("bundled",)
    pb.unlink()
    assert regions.load_preset("foo", project) is None


def test_load_preset_matches_game_key_not_file_name(preset_env):
    _, _, _, user_dir, _ = preset_env
    user_dir.mkdir(parents=True)
    (user_dir / "whatever.json").write_text(
        json.dumps({"game": "bar", "regions": []}), encoding="utf-8"
    )
    assert regions.load_preset("bar") is not None
    assert regions.load_preset("whatever") is None
    assert regions.load_preset("") is None and regions.load_preset(None) is None


def test_invalid_preset_file_ignored(preset_env):
    _, _, _, user_dir, _ = preset_env
    user_dir.mkdir(parents=True)
    (user_dir / "foo.json").write_text("[1, 2]", encoding="utf-8")
    assert regions.load_preset("foo") is None
