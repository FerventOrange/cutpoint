"""The ``ski`` style is now ``action-cam``, with no alias: ``ski`` is refused with an error
naming ``action-cam``."""

import json

import pytest

from cutpoint import cli
from cutpoint.styles import ACTION_CAM, STYLES, get_style, load_learned, project_kind, style_names


def test_action_cam_is_the_style_and_its_kind():
    style = STYLES[ACTION_CAM]
    assert (style.name, style.kind, style.base) == ("action-cam", "action-cam", "action-cam")
    assert "ski" not in STYLES and "ski" not in style_names()


@pytest.mark.parametrize("command", [["analyze", "a.mp4"], ["edit", "a.mp4"]])
def test_style_ski_is_refused_naming_action_cam(capsys, command):
    parser = cli.build_parser()
    assert parser.parse_args([*command, "--style", "action-cam"]).style == "action-cam"
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([*command, "--style", "ski"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "'ski' was renamed to 'action-cam'" in err and "--style action-cam" in err


def test_analyze_help_lists_action_cam_not_ski(capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["analyze", "--help"])
    out = capsys.readouterr().out
    choices = out[out.index("--style {") + len("--style {") :].split("}", 1)[0].split(",")
    assert "action-cam" in choices and "ski" not in choices


def test_get_style_ski_names_action_cam():
    with pytest.raises(ValueError, match="renamed to 'action-cam'"):
        get_style("ski")


def test_a_project_analysed_as_ski_is_refused_naming_action_cam():
    assert project_kind({"style": "action-cam", "style_kind": "action-cam"}) == "action-cam"
    assert project_kind({"style": "gaming", "style_kind": "generic"}) == "generic"
    for hl in ({"style": "ski", "style_kind": "ski"}, {"style": "mine", "style_kind": "ski"}):
        with pytest.raises(ValueError, match="renamed to 'action-cam'"):
            project_kind(hl)


def test_a_learned_preset_based_on_ski_is_refused_naming_action_cam(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"name": "old", "base": "ski", "weights": {"speed": 0.5}}))
    with pytest.raises(ValueError, match="renamed to 'action-cam'"):
        load_learned(path)
