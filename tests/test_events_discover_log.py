"""The INFO line `_discover` logs before discovering a game that has no preset."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutpoint import paths
from cutpoint.events import discover, pipeline
from cutpoint.events.killfeed import OcrParams
from hud_fixtures import boxfeed, write_preset

NOTICE = "No HUD preset matches"


def _run(home: Path, monkeypatch: pytest.MonkeyPatch, game: str, force: bool | None) -> None:
    def fake_discover(srcs, out, game, params, on_frame=None):
        return {"regions": []}

    monkeypatch.setattr(discover, "discover_regions", fake_discover)
    src = SimpleNamespace(path=Path("/synthetic/footage/session.mp4"))
    pipeline._discover(home / "proj", [src], OcrParams(game=game), force, 1, "user")


def test_missing_preset_logs_game_and_presets_dir(
    home: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    _run(home, monkeypatch, "nosuchgame", None)
    msgs = [r.getMessage() for r in caplog.records if NOTICE in r.getMessage()]
    assert len(msgs) == 1
    assert "'nosuchgame'" in msgs[0]
    assert str(paths.presets_dir()) in msgs[0]
    assert "session.mp4" in msgs[0]
    assert msgs[0].endswith(
        f"Put a preset in {paths.presets_dir()} {pipeline.EVENTS_DISCOVER_HINT}"
    )
    assert "--no-discover" in msgs[0]


def test_analyze_hint_names_its_own_option(
    home: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """analyze has no --game or --no-discover: its hint names --skip keyframes instead."""
    from cutpoint import analyze  # noqa: PLC0415

    monkeypatch.setattr(discover, "discover_regions", lambda *a, **k: {"regions": []})
    src = SimpleNamespace(path=Path("/synthetic/footage/session.mp4"))
    caplog.set_level(logging.INFO)
    pipeline._discover(
        home / "proj", [src], OcrParams(game="nosuchgame"), None, 1, "user", analyze.DISCOVER_HINT
    )
    msgs = [r.getMessage() for r in caplog.records if NOTICE in r.getMessage()]
    assert len(msgs) == 1
    assert "--no-discover" not in msgs[0] and "--game" not in msgs[0]
    assert msgs[0].endswith(
        f"Put a preset in {paths.presets_dir()} or pass --skip keyframes to leave out the whole "
        "keyframes step"
    )


def test_existing_preset_does_not_log_missing_notice(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    write_preset(paths.presets_dir(), boxfeed())
    caplog.set_level(logging.INFO)
    _run(home, monkeypatch, "boxfeed", None)
    assert NOTICE not in caplog.text
