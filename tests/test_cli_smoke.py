from __future__ import annotations

import pytest

import cutpoint
from cutpoint import cli


@pytest.mark.parametrize(
    ("argv", "word"), [(["--help"], "paths"), (["events", "--help"], "events")]
)
def test_help_exits_zero(capsys, argv, word):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 0
    assert word in capsys.readouterr().out


def test_paths_command_uses_isolated_home(home, capsys):
    assert cli.main(["paths"]) == 0
    out = capsys.readouterr().out
    assert str(home) in out
    for key in (
        "home",
        "config",
        "cache",
        "models",
        "stage",
        "styles",
        "presets",
        "projects",
        "bundled",
    ):
        assert any(line.startswith(key) for line in out.splitlines())
    assert not home.exists()  # printing paths creates nothing


def test_clips_bad_size_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["clips", "some-project", "--size", "abc"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--size" in err and "expected WxH" in err
    assert "Traceback" not in err


def test_guide_prints_agents_md(capsys):
    assert cli.main(["guide"]) == 0
    assert capsys.readouterr().out.startswith("# Cutpoint for AI agents")


def test_version_flag_prints_version_without_subcommand(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"cutpoint {cutpoint.__version__}"
