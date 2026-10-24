from __future__ import annotations

import logging

import pytest

import cutpoint
from cutpoint import cli


@pytest.mark.parametrize(
    ("argv", "word"),
    [(["--help"], "paths"), (["events", "--help"], "events"), (["edit", "--help"], "--kind")],
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


def test_edit_bad_kind_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["edit", "video.mp4", "--kind", "trailer"])
    assert exc.value.code == 2
    assert "--kind" in capsys.readouterr().err


def test_360_qa_bad_size_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["360", "qa", "x", "--paths", "--size", "abc"])
    assert exc.value.code == 2
    assert "--size" in capsys.readouterr().err


def test_guide_prints_agents_md(capsys):
    assert cli.main(["guide"]) == 0
    assert capsys.readouterr().out.startswith("# Cutpoint for AI agents")


def test_version_flag_prints_version_without_subcommand(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"cutpoint {cutpoint.__version__}"


@pytest.mark.parametrize("preexisting", [None, "empty"])
def test_analyze_failure_removes_only_folder_it_created(home, tmp_path, preexisting):
    video = tmp_path / "bad.mp4"
    video.write_bytes(b"not a video")
    project = home / "projects" / "proj"
    if preexisting:
        project.mkdir(parents=True)
    assert cli.main(["analyze", str(video), "--out", "proj"]) == 1
    assert project.exists() == bool(preexisting)


def test_huggingface_warning_is_printed_once():
    pytest.importorskip("huggingface_hub.utils.logging")
    assert cli.main(["paths"]) == 0
    assert logging.getLogger("huggingface_hub").handlers == []


@pytest.mark.parametrize("target", ["0", "-5"])
def test_mix_rejects_non_positive_target(target):
    with pytest.raises(SystemExit) as exc:
        cli.main(["mix", "some-project", "--target", target])
    assert exc.value.code == 2


def test_director_on_unscanned_project_creates_no_folder(home, caplog):
    assert cli.main(["360", "director", "nosuchproj"]) == 1
    assert not (home / "projects" / "nosuchproj").exists()
    assert "cutpoint 360 scan" in caplog.text
