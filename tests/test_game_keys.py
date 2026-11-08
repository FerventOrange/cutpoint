"""The game key of a source (#822): generic folder names give way to the file name,
non-ASCII names stay readable, and presets saved under an older key keep matching
(invented games and paths)."""

from __future__ import annotations

import pytest

from cutpoint import paths
from cutpoint.events.regions import game_key_for_path
from hud_fixtures import write_preset


def _preset(game: str, **over) -> dict:
    return {"game": game, "regions": []} | over


def test_a_game_folder_names_the_game() -> None:
    assert game_key_for_path("/rec/Neon Drift/session 1.mp4") == "neon-drift"
    assert game_key_for_path("/rec/vidéo ünï 日本/a1.mp4") == "vidéo-ünï-日本"


@pytest.mark.parametrize(
    "folder",
    [
        "Videos",
        "VODs",
        "captures",
        "Recordings",
        "clips",
        "Gameplay",
        "footage",
        "Desktop",
        "Downloads",
        "src",
    ],
)
def test_a_generic_folder_gives_the_file_name(folder: str) -> None:
    assert game_key_for_path(f"/rec/{folder}/Neon Drift.mp4") == "neon-drift"
    assert game_key_for_path(f"/rec/{folder}/Box Stack.mp4") == "box-stack"


@pytest.mark.parametrize(
    "folder",
    ["a", "B", "b2", "2024", "2024-11-14", "2024_11_14", "07", "\U0001f3ae", "___", "Game"],
)
def test_short_and_date_folders_give_the_file_name(folder: str) -> None:
    assert game_key_for_path(f"/rec/{folder}/Neon Drift.mp4") == "neon-drift"


def test_a_generic_file_name_in_a_generic_folder_is_still_the_key() -> None:
    assert game_key_for_path("/rec/VODs/videos.mp4") == "videos"
    assert game_key_for_path("clip.mp4") == "clip"  # no folder at all


def test_a_recorder_file_name_wins_over_the_folder() -> None:
    name = "Neon Drift 2024.11.14 - 19.01.53.09.DVR.mp4"
    assert game_key_for_path(f"/rec/Box Stack/{name}") == "neon-drift"
    assert game_key_for_path(f"/rec/VODs/{name}") == "neon-drift"


def test_a_generic_recorder_file_name_gives_way_to_the_folder() -> None:
    """A desktop capture ("Desktop <date>.DVR.mp4") names no game: the folder does, or else
    the whole file name."""
    name = "Desktop 2024.11.14 - 19.01.53.09.DVR.mp4"
    assert game_key_for_path(f"/rec/Neon Drift/{name}") == "neon-drift"
    assert game_key_for_path(f"/rec/VODs/{name}") == "desktop-2024-11-14-19-01-53-09-dvr"


def test_an_override_and_match_hints_still_win() -> None:
    assert game_key_for_path("/rec/VODs/Neon Drift.mp4", "box-stack") == "box-stack"
    assert game_key_for_path("/rec/VODs/Neon Drift.mp4", "auto") == "neon-drift"
    write_preset(paths.presets_dir(), _preset("box-stack", match=["neon drift"]))
    assert game_key_for_path("/rec/VODs/Neon Drift.mp4") == "box-stack"
    assert game_key_for_path("/rec/Neon Drift/a1.mp4") == "box-stack"


def test_a_preset_under_the_legacy_key_keeps_matching() -> None:
    path = "/rec/Café Racer/session 1.mp4"
    assert game_key_for_path(path) == "café-racer"  # no preset yet: the new key
    write_preset(paths.presets_dir(), _preset("caf-racer"))
    assert game_key_for_path(path) == "caf-racer"  # saved before #822
    write_preset(paths.presets_dir(), _preset("café-racer"))
    assert game_key_for_path(path) == "café-racer"  # the new key's own preset wins


def test_an_old_game_preset_is_no_legacy_key_for_non_ascii_names() -> None:
    """Every name without ASCII letters or digits had the key "game": such a preset is not
    taken for a new non-ASCII folder."""
    write_preset(paths.presets_dir(), _preset("game"))
    assert game_key_for_path("/rec/\u65e5\u672c\u8a9e/a1.mp4") == "\u65e5\u672c\u8a9e"


def test_a_project_preset_under_the_legacy_key_keeps_matching(tmp_path) -> None:
    project = tmp_path / "proj"
    path = "/rec/Café Racer 2024.11.14 - 19.01.53.09.DVR.mp4"
    write_preset(project / "hud_presets", _preset("caf-racer"))
    assert game_key_for_path(path, None, project) == "caf-racer"
    assert game_key_for_path(path) == "café-racer"  # another project has none
