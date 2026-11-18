"""The text (non ``--json``) output of ``style show`` and ``profile validate`` (#1085 A492)."""

from __future__ import annotations

import json
import logging

import pytest

from cutpoint import cli, stylefiles

PROFILE = {
    "detect": ["a person riding a raft down a river"],
    "incident": {"pos": ["a flipped raft"], "neg": ["a calm river"]},
}


@pytest.fixture
def styles_dir(home, monkeypatch):
    folder = home / "my-styles"
    monkeypatch.setenv("CUTPOINT_STYLES_DIR", str(folder))
    return folder


def _lines(out: str) -> list[str]:
    return out.splitlines()


def test_style_show_prints_a_built_in_style_as_text(capsys):
    assert cli.main(["style", "show", "action"]) == 0
    out = capsys.readouterr().out
    lines = _lines(out)
    assert not out.startswith("{")  # the text report, not the --json envelope
    assert lines[0] == "name: action (built-in style)"
    assert lines[1] == "base: action"
    assert "applies to: -" in lines
    assert any(line.startswith("hash: ") for line in lines)
    assert not any(line.startswith(("file:", "mask_friends:")) for line in lines)
    head = lines.index("weights:")
    assert lines[head + 1].split() == ["feature", "weight", "from"]
    shown = {row.split()[0]: row.split() for row in lines[head + 2 :]}
    expected = stylefiles.show("action")["weights"]
    assert set(shown) == {w["feature"] for w in expected}
    first = expected[0]
    assert shown[first["feature"]] == [first["feature"], f"{first['weight']:g}", first["from"]]


def test_style_show_prints_a_style_file_with_its_origin_and_applies_to(styles_dir, capsys):
    styles_dir.mkdir(parents=True)
    doc = {
        "schema": 1,
        "name": "mine",
        "base": "action-cam",
        "source": "hand",
        "description": "my cut",
        "applies_to": {"categories": ["action-cam"], "profiles": ["generic"]},
        "weights": {"scene": 0.4},
    }
    path = styles_dir / "mine.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert cli.main(["style", "show", "mine"]) == 0
    lines = _lines(capsys.readouterr().out)
    assert lines[0] == "name: mine (your style, source hand)"
    assert "base: action-cam" in lines
    assert f"file: {path}" in lines
    assert "description: my cut" in lines
    assert "applies to: category:action-cam profile:generic" in lines
    assert "mask_friends: true" in lines or "mask_friends: false" in lines
    assert any(line.startswith("must_include_crash: ") for line in lines)
    scene = next(r.split() for r in lines if r.split()[:1] == ["scene"])
    assert scene[1:] == ["0.4", "style"]  # set by the file, not the base


def test_profile_validate_logs_a_valid_profile_with_its_notes(home, caplog, tmp_path):
    caplog.set_level(logging.INFO)
    path = tmp_path / "mine.json"
    path.write_text(json.dumps(PROFILE), encoding="utf-8")
    assert cli.main(["profile", "validate", str(path)]) == 0
    messages = [r.getMessage() for r in caplog.records]
    assert f"{path} is a valid profile" in messages
    # a file outside the library: the note names the command that saves it
    assert any("profile create NAME --from" in m for m in messages)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_profile_validate_warns_and_still_passes_for_ignored_keys(home, caplog, tmp_path):
    caplog.set_level(logging.INFO)
    doc = dict(PROFILE)
    doc["colour_of_the_sky"] = "blue"
    path = tmp_path / "mine.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert cli.main(["profile", "validate", str(path)]) == 0
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert (
        len(warned) == 1 and warned[0].startswith(f"{path}: ") and "colour_of_the_sky" in warned[0]
    )
    assert f"{path} is a valid profile" in [r.getMessage() for r in caplog.records]


def test_profile_validate_of_an_invalid_profile_fails_and_names_the_problem(home, caplog, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"detect": "not a list"}), encoding="utf-8")
    assert cli.main(["profile", "validate", str(path)]) == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors and f"{path} is not a valid profile: " in errors[0] and "detect" in errors[0]
    assert "is a valid profile" not in caplog.text
