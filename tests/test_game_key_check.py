"""A HUD preset key that would be a path is refused before it names a file (A652)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cutpoint.events.cli as events_cli
from conftest import envelope
from cutpoint import cache, cli, jsonout
from cutpoint.events import discover, pipeline
from cutpoint.events.killfeed import OcrParams

BAD = ["../../x", "a/b", "a\\b", "C:x", "..", ".", "", " ", "a\0b"]


@pytest.mark.parametrize("key", ["counter-strike-2", "vidéo-ünï-日本", "r6", "game", "a.b_c"])
def test_a_plain_key_is_kept(key: str) -> None:
    assert pipeline.check_game_key(key) == key


@pytest.mark.parametrize("key", BAD)
def test_a_path_is_no_key(key: str) -> None:
    with pytest.raises(jsonout.ArgumentError, match="HUD preset key"):
        pipeline.check_game_key(key)


def test_events_refuses_a_path_as_game(home: Path, monkeypatch, capsys) -> None:
    project = home / "projects" / "p"
    project.mkdir(parents=True)
    cache.save_json(project / "highlights.json", {"sources": []})
    calls: list[object] = []
    monkeypatch.setattr(events_cli, "run_args", lambda *a, **k: calls.append(a))
    code = cli.main(["events", "p", "--game", "../../escaped", "--json"])
    doc = envelope(capsys.readouterr().out)
    assert code == 1 and calls == []
    assert doc["error"]["code"] == "bad_args" and "HUD preset key" in doc["error"]["message"]


def test_run_events_refuses_a_path_as_game(tmp_path: Path) -> None:
    with pytest.raises(jsonout.ArgumentError):
        pipeline.run_events(tmp_path, ocr=OcrParams(game="../../escaped"))


def _discover(project: Path, monkeypatch, game: str) -> list[object]:
    ran: list[object] = []
    monkeypatch.setattr(discover, "discover_regions", lambda *a, **k: ran.append(a) or {})
    src = SimpleNamespace(path=Path("/synthetic/footage/session.mp4"))
    pipeline._discover(project, [src], OcrParams(game=game), None, 1, "project")
    return ran


def test_discovery_refuses_a_path_as_game(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "proj"
    with pytest.raises(jsonout.ArgumentError):
        _discover(project, monkeypatch, "../../escaped")
    assert not (tmp_path / "escaped").exists() and not (project / "discovery").exists()


def test_discovery_refuses_a_preset_game_that_is_a_path(tmp_path: Path, monkeypatch) -> None:
    """A project preset whose ``game`` is a path and whose ``match`` hint names the source."""
    project = tmp_path / "proj"
    presets = project / "hud_presets"
    presets.mkdir(parents=True)
    (presets / "x.json").write_text(
        json.dumps({"game": "../../escaped", "match": ["session"], "regions": []}),
        encoding="utf-8",
    )
    with pytest.raises(jsonout.ArgumentError):
        _discover(project, monkeypatch, "auto")
    assert not (tmp_path / "escaped").exists()
