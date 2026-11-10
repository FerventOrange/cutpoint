"""``action-cam`` is the style; ``ski`` is just an unknown name (no alias, no rename message)."""

import pytest

from cutpoint import cli
from cutpoint.styles import ACTION_CAM, STYLES, get_style, style_names


def test_action_cam_is_the_style_and_its_kind():
    style = STYLES[ACTION_CAM]
    assert (style.name, style.kind, style.base) == ("action-cam", "action-cam", "action-cam")
    assert "ski" not in STYLES and "ski" not in style_names()


@pytest.mark.parametrize("command", [["analyze", "a.mp4"], ["edit", "a.mp4"]])
def test_style_ski_is_an_ordinary_unknown_style(capsys, command):
    parser = cli.build_parser()
    assert parser.parse_args([*command, "--style", "action-cam"]).style == "action-cam"
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([*command, "--style", "ski"])
    assert exc.value.code == 2
    assert "renamed" not in capsys.readouterr().err


def test_get_style_ski_is_unknown():
    with pytest.raises(ValueError) as exc:
        get_style("ski")
    assert "renamed" not in str(exc.value)


def test_style_auto_is_offered_by_analyze_and_edit_only():
    parser = cli.build_parser()
    for command in (["analyze", "a.mp4"], ["edit", "a.mp4"]):
        assert parser.parse_args([*command, "--style", "auto"]).style == "auto"
    with pytest.raises(SystemExit):
        parser.parse_args(["360", "analyze", "p", "--style", "auto"])


def test_analyze_help_lists_action_cam_not_ski(capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["analyze", "--help"])
    out = capsys.readouterr().out
    choices = out[out.index("--style {") + len("--style {") :].split("}", 1)[0].split(",")
    assert "action-cam" in choices and "ski" not in choices


@pytest.mark.parametrize("command", [["analyze", "a.mp4"], ["edit", "a.mp4"]])
def test_a_custom_style_is_accepted_by_style_arg(home, monkeypatch, capsys, command):
    """With the rename helper gone, ``--style NAME`` still takes a style made by ``style create``."""
    monkeypatch.setenv("CUTPOINT_STYLES_DIR", str(home / "my-styles"))
    assert cli.main(["style", "create", "mine", "--base", "action", "--weight", "scene=0.4"]) == 0
    capsys.readouterr()
    assert "mine" in style_names()
    assert cli.build_parser().parse_args([*command, "--style", "mine"]).style == "mine"
