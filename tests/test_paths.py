from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cutpoint import paths
from cutpoint.events import regions
from cutpoint.events.killfeed import Interpreter, OcrParams
from hud_fixtures import boxfeed, write_preset


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


# -- the bundled R6 preset (presets/r6.json) keeps the original built-in behaviour
R6_PHRASES = (
    "WON ROUND", "ROUND WON", "LOST ROUND", "ROUND LOST", "VICTORY", "DEFEAT",
    "REVERSE FRIENDLY FIRE", "ACE", "CLUTCH", "OVERTIME", "MATCH POINT",
    "LAST OP STANDING", "DEFUSER PLANTED", "DEFUSER DISABLED", "BOMB DEFUSED",
    "ELIMINATED", "TEAM KILL", "TEAMKILL", "DRAW",
)  # fmt: skip
R6_REGIONS = (
    ("killfeed", 1300, 150, 560, 185, "killfeed", 0.008, 0.002),
    ("counter", 760, 200, 400, 110, "counter", 0.015, 0.002),
    ("banner", 520, 380, 880, 220, "banner", 0.03, 0.012),
    ("notice", 600, 740, 720, 80, "notice", 0.03, 0.006),
)
R6_KEY = {  # OcrParams().key("r6", load_preset("r6")), computed before the move to JSON
    "v": 3,
    "game": "r6",
    "fps": 3.0,
    "fuzzy": 0.8,
    "min_text_score": 0.6,
    "persist_s": 8.0,
    "extra": {},
    "tags": [],
    "regions": [
        {"name": n, "x": x, "y": y, "w": w, "h": h, "kind": k, "min_change": mc, "min_fill": mf}
        for n, x, y, w, h, k, mc, mf in R6_REGIONS
    ],
}


def test_r6_preset_parity():
    p = regions.load_preset("r6")
    assert (p.builtin, p.bigtext, p.path, p.feed, p.player_tags, p.title) == (
        True, False, None, {}, (), "Rainbow Six Siege",
    )  # fmt: skip
    got = tuple(
        (r.name, r.x, r.y, r.w, r.h, r.kind, r.min_change, r.min_fill) for r in p.all_regions()
    )
    assert got == R6_REGIONS
    assert p.phrases == R6_PHRASES  # no GENERIC_BANNERS
    assert p.key() == {}
    assert OcrParams().key("r6", p) == R6_KEY


def test_r6_builtin_fallback_without_bundled_folder(tmp_path, home, monkeypatch):
    default = regions.load_preset("r6")
    monkeypatch.setattr(paths, "PRESETS_DIR", tmp_path / "empty")
    assert regions.load_preset("r6") == default


def test_builtin_flag_honoured_in_user_folder(preset_env, caplog, monkeypatch):
    _, project, _, user_dir, _ = preset_env
    monkeypatch.setattr(regions, "_warned_builtin", set(), raising=False)
    write_preset(user_dir, boxfeed())
    with caplog.at_level("WARNING"):
        p = regions.load_preset("boxfeed", project)
    assert p.builtin is True
    assert p.path is None
    assert p.phrases == ("STACK COMPLETE",)  # no GENERIC_BANNERS
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_builtin_flag_ignored_in_project_folder(preset_env, caplog, monkeypatch):
    _, project, proj_dir, _, _ = preset_env
    monkeypatch.setattr(regions, "_warned_builtin", set(), raising=False)
    f = proj_dir / "r6.json"
    proj_dir.mkdir(parents=True)
    f.write_text(
        json.dumps({"game": "r6", "builtin": True, "banner_phrases": ["ACE"], "regions": []}),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        p = regions.load_preset("r6", project)
        regions.load_preset("r6", project)
    assert p.builtin is False
    assert p.path == f
    assert set(regions.GENERIC_BANNERS) <= set(p.phrases)
    assert "phrases" in p.key()
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_user_builtin_preset_equals_bundled_one(preset_env, caplog, monkeypatch):
    """A builtin preset in the user folder is the same as in the bundled folder; a project
    copy is generic (own OCR key) and warns once."""
    _, project, proj_dir, user_dir, bundled_dir = preset_env
    monkeypatch.setattr(regions, "_warned_builtin", set(), raising=False)
    user_file = write_preset(user_dir, boxfeed())
    user = regions.load_preset("boxfeed")
    bundled_dir.mkdir()
    user_file.rename(bundled_dir / user_file.name)
    bundled = regions.load_preset("boxfeed")
    assert user == bundled
    assert OcrParams().key("boxfeed", user) == OcrParams().key("boxfeed", bundled)
    assert "preset" not in OcrParams().key("boxfeed", user)

    write_preset(proj_dir, boxfeed())
    with caplog.at_level("WARNING"):
        generic = regions.load_preset("boxfeed", project)
        regions.load_preset("boxfeed", project)
    assert generic.builtin is False and generic.path is not None
    assert "preset" in OcrParams().key("boxfeed", generic)
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_interpreter_generic_only_for_non_installed_preset(preset_env):
    _, project, proj_dir, user_dir, _ = preset_env
    write_preset(user_dir, boxfeed())
    write_preset(proj_dir, boxfeed())
    for proj, generic in ((None, False), (project, True)):
        preset = regions.load_preset("boxfeed", proj)
        assert Interpreter("boxfeed", OcrParams(), 1.0, preset).generic is generic


@pytest.mark.parametrize(
    "path", ["D:/Rainbow Six Siege 2025.mp4", "/x/r6/match.mp4", "/x/SIEGE.mp4"]
)
def test_preset_for_path_r6(path):
    assert regions.preset_for_path(path) == "r6"


def test_r6_wins_over_user_preset_matching_siege(preset_env):
    _, _, _, user_dir, _ = preset_env
    user_dir.mkdir(parents=True)
    (user_dir / "other.json").write_text(
        json.dumps({"game": "other", "match": ["siege"], "regions": []}), encoding="utf-8"
    )
    assert regions.preset_for_path("/x/siege.mp4") == "r6"
