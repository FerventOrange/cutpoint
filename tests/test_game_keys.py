"""The game key of a source (#822): generic folder names give way to the file name,
non-ASCII names stay readable, and presets saved under an older key keep matching
(invented games and paths)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cutpoint import paths
from cutpoint.events import regions
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


def test_a_generic_recorder_file_name_gives_way_to_the_folder(tmp_path) -> None:
    """A desktop capture ("Desktop <date>.DVR.mp4") names no game: the folder does; in a
    generic folder the session's first such video does (the whole file name when the folder
    cannot be listed)."""
    name = "Desktop 2024.11.14 - 19.01.53.09.DVR.mp4"
    assert game_key_for_path(f"/rec/Neon Drift/{name}") == "neon-drift"
    assert game_key_for_path(f"/rec/VODs/{name}") == "desktop-2024-11-14-19-01-53-09-dvr"
    folder = tmp_path / "VODs"
    first = _video(folder, name)
    later = _video(folder, "Desktop 2024.11.14 - 21.30.00.01.DVR.mp4")
    for path in (first, later):
        assert game_key_for_path(str(path)) == "desktop-2024-11-14-19-01-53-09-dvr"


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


# --------------------------------------------------------------------------- fit decisions


def _video(folder, name: str, data: bytes = b"x"):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(data)
    return path


def _decide(project, path, key: str, guess: str = "neon-drift") -> None:
    entry = {"tag": regions.source_tag(path), "guess": guess, "key": key, "fit": None}
    regions.save_game_keys(project, {regions.source_id(path): entry})


def test_a_recorded_decision_wins_over_the_first_guess(tmp_path) -> None:
    project = tmp_path / "proj"
    path = _video(tmp_path / "Neon Drift", "session 1.mp4")
    _decide(project, path, "box-stack")
    assert game_key_for_path(str(path), None, project) == "box-stack"
    assert game_key_for_path(str(path)) == "neon-drift"  # no project: no decisions


def test_a_changed_file_or_guess_makes_the_decision_stale(tmp_path) -> None:
    project = tmp_path / "proj"
    path = _video(tmp_path / "Neon Drift", "session 1.mp4")
    _decide(project, path, "box-stack", guess="other-guess")
    assert game_key_for_path(str(path), None, project) == "neon-drift"
    _decide(project, path, "box-stack")
    path.write_bytes(b"longer now")  # another size: another file
    assert game_key_for_path(str(path), None, project) == "neon-drift"


def test_an_override_and_match_hints_beat_a_decision(tmp_path) -> None:
    project = tmp_path / "proj"
    path = _video(tmp_path / "Neon Drift", "session 1.mp4")
    _decide(project, path, "box-stack")
    assert game_key_for_path(str(path), "zone-war", project) == "zone-war"
    write_preset(paths.presets_dir(), _preset("hinted", match=["session 1"]))
    assert game_key_for_path(str(path), None, project) == "hinted"


@pytest.mark.parametrize(
    "text", ["{not json", '{"version": 99, "sources": {}}', '{"version": 1, "sources": []}']
)
def test_an_unreadable_decisions_file_gives_the_first_guess(tmp_path, text: str) -> None:
    project = tmp_path / "proj"
    path = _video(tmp_path / "Neon Drift", "session 1.mp4")
    regions.game_keys_path(project).parent.mkdir(parents=True)
    regions.game_keys_path(project).write_text(text, encoding="utf-8")
    assert game_key_for_path(str(path), None, project) == "neon-drift"


def test_date_named_videos_in_a_generic_folder_share_the_first_ones_key(tmp_path) -> None:
    folder = tmp_path / "VODs"
    first = _video(folder, "2026-11-08 19-01-53.mkv")
    later = _video(folder, "2026-11-09 20-15-00.mp4")
    _video(folder, "2026-01-01.txt")  # no video: not the first
    other = _video(folder, "Box Stack.mp4")  # named: its own key
    for path in (first, later):
        assert game_key_for_path(str(path)) == "2026-11-08-19-01-53"
    assert game_key_for_path(str(other)) == "box-stack"
    # a game folder names them all, date or not
    named = _video(tmp_path / "Neon Drift", "2026-11-08 19-01-53.mkv")
    assert game_key_for_path(str(named)) == "neon-drift"
    # a file not on disk (nothing to list) keeps its own name
    assert game_key_for_path("/nowhere/VODs/2026-11-10 10-00-00.mkv") == "2026-11-10-10-00-00"


def test_a_recorded_date_group_survives_deleting_the_first_file(tmp_path) -> None:
    project = tmp_path / "proj"
    folder = tmp_path / "VODs"
    first = _video(folder, "2026-11-08 19-01-53.mkv")
    later = _video(folder, "2026-11-09 20-15-00.mkv")
    regions.save_game_keys(project, {}, {regions.source_id(folder): first.stem})
    first.unlink()
    assert game_key_for_path(str(later), None, project) == "2026-11-08-19-01-53"
    assert game_key_for_path(str(later)) == "2026-11-09-20-15-00"  # no project: the listing


def test_a_recorded_date_group_holds_for_a_folder_that_cannot_be_listed(tmp_path) -> None:
    project = tmp_path / "proj"
    folder = tmp_path / "offline" / "VODs"  # not on disk
    regions.save_game_keys(project, {}, {regions.source_id(folder): "2026-11-08 19-01-53"})
    path = str(folder / "2026-11-09 20-15-00.mkv")
    assert game_key_for_path(path, None, project) == "2026-11-08-19-01-53"


def test_an_offline_folder_is_listed_through_its_remap_copy(tmp_path, monkeypatch) -> None:
    from cutpoint import config  # noqa: PLC0415

    copy = tmp_path / "copy" / "VODs"
    _video(copy, "2026-11-08 19-01-53.mkv")
    _video(copy, "2026-11-09 20-15-00.mkv")
    offline = tmp_path / "offline" / "VODs"  # not on disk
    monkeypatch.setattr(config, "remap_source", lambda p: copy / Path(p).name)
    path = str(offline / "2026-11-09 20-15-00.mkv")
    assert game_key_for_path(path) == "2026-11-08-19-01-53"


def test_replay_buffer_names_name_no_game(tmp_path) -> None:
    """OBS replay-buffer files ("Replay <date>.mkv") are keyed like a desktop capture: by
    the folder, or in a generic one by the session's first date-named video."""
    name = "Replay 2026-11-08 19-01-53.mkv"
    assert game_key_for_path(f"/rec/Neon Drift/{name}") == "neon-drift"
    folder = tmp_path / "Replay"
    replays = [_video(folder, name), _video(folder, "Replay 2026-11-08 19-20-00.mkv")]
    _video(folder, "Box Stack.mkv")  # named: not in the group
    for path in replays:
        assert game_key_for_path(str(path)) == "replay-2026-11-08-19-01-53"
    plain = _video(folder, "2026-11-08 18-00-00.mkv")  # first by name now, same group
    st = folder.stat()  # listings are cached per folder mtime: a later one, as on a slow disk
    os.utime(folder, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert game_key_for_path(str(plain)) == "2026-11-08-18-00-00"
    assert game_key_for_path(str(replays[0])) == "2026-11-08-18-00-00"
