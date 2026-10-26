from __future__ import annotations

import logging
from pathlib import Path

import pytest

import cutpoint
import cutpoint.analyze
from cutpoint import cli


@pytest.mark.parametrize(
    ("argv", "word"),
    [
        (["--help"], "paths"),
        (["events", "--help"], "events"),
        (["edit", "--help"], "--kind"),
        (["paths", "--help"], "--json"),
    ],
)
def test_help_exits_zero(capsys, argv, word):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 0
    assert word in capsys.readouterr().out


def test_paths_command_uses_isolated_home(home, capsys):
    assert cli.main(["paths"]) == 0
    out = capsys.readouterr().out
    assert not out.startswith("{")  # the text report, not the --json envelope
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
    assert any(line.startswith("home") and "(missing)" in line for line in out.splitlines())
    assert not home.exists()  # printing paths creates nothing


def test_clips_bad_size_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["clips", "some-project", "--size", "abc"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--size" in err and "expected WxH" in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    ("option", "value"),
    [("--kind", "trailer"), ("--top", "0"), ("--max-total", "-5"), ("--vision-top", "0")],
)
def test_edit_bad_option_is_a_usage_error(capsys, option, value):
    """--top / --max-total are checked before the long analysis, not after it."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["edit", "video.mp4", option, value])
    assert exc.value.code == 2
    assert option in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["clips", "some-project", "--top", "0"],
        ["clips", "some-project", "--top", "-1"],
        ["clips", "some-project", "--max-total", "0"],
        ["clips", "some-project", "--max-total", "-5"],
        ["360", "director", "some-project", "--top", "0"],
        ["360", "stitch", "some-project", "--top", "-1"],
        ["360", "stitch", "some-project", "--max-total", "0"],
        ["assemble", "some-clips", "--transition", "fade", "--fade-dur", "0"],
        ["assemble", "some-clips", "--transition", "fade", "--fade-dur", "-0.5"],
    ],
)
def test_non_positive_top_max_total_or_fade_dur_is_a_usage_error(capsys, argv):
    """`--top -1` used to select all but one moment; `--fade-dur 0` cut the video short."""
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert argv[-2] in err and "must be greater than 0" in err


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--vision-top", "0", "must be greater than 0"),
        ("--vision-top", "-3", "must be greater than 0"),
        ("--max-candidates", "0", "must be greater than 0"),
        ("--max-candidates", "-1", "must be greater than 0"),
        ("--must-include-crash", "-0.5", "must be from 0 to 1"),
        ("--must-include-crash", "1.5", "must be from 0 to 1"),
        ("--must-include-crash", "high", "expected a number"),
    ],
)
def test_analyze_bad_count_or_confidence_is_a_usage_error(capsys, option, value, message):
    """`--vision-top -3` used to rate all but the last 3 candidates."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["analyze", "video.mp4", option, value])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert option in err and message in err


@pytest.mark.parametrize("value", ["0", "1", "0.6"])
def test_analyze_must_include_crash_accepts_0_to_1(home, tmp_path, monkeypatch, value):
    """0 turns must-include crashes off."""
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    styles: list[float] = []
    monkeypatch.setattr(
        cutpoint.analyze,
        "analyze",
        lambda targets, style, out, **kw: styles.append(style.must_include_crash),
    )
    assert cli.main(["analyze", str(video), "--style", "ski", "--must-include-crash", value]) == 0
    assert styles == [float(value)]


@pytest.mark.parametrize(
    ("make", "name"),
    [
        (lambda d: (d / "trip.2024", "."), "trip.2024"),  # "." resolved; dots kept
        (lambda d: (d, "a.b.mp4"), "a.b"),  # a file: its stem
        (lambda d: (d, d.anchor), "project"),  # a drive root has no name
    ],
)
def test_analyze_default_project_name(home, tmp_path, monkeypatch, make, name):
    cwd, arg = make(tmp_path / "videos")
    cwd.mkdir(parents=True)
    if arg.endswith(".mp4"):
        (cwd / arg).write_bytes(b"x")
    monkeypatch.chdir(cwd)
    outs: list[Path] = []
    monkeypatch.setattr(
        cutpoint.analyze, "analyze", lambda targets, style, out, **kw: outs.append(out)
    )
    assert cli.main(["analyze", arg]) == 0
    assert outs == [home / "projects" / name]


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


@pytest.mark.ffmpeg
@pytest.mark.parametrize("preexisting", [None, "empty"])
def test_analyze_failure_removes_only_folder_it_created(home, tmp_path, preexisting):
    video = tmp_path / "bad.mp4"
    video.write_bytes(b"not a video")
    project = home / "projects" / "proj"
    if preexisting:
        project.mkdir(parents=True)
    assert cli.main(["analyze", str(video), "--out", "proj"]) == 1
    assert project.exists() == bool(preexisting)


@pytest.mark.parametrize(
    ("out", "error", "code"),
    [
        ("proj", RuntimeError("bad video"), 1),
        ("projects/new/nested", RuntimeError("bad video"), 1),
        ("projects/new/nested", KeyboardInterrupt(), 130),
    ],
)
def test_failed_analyze_leaves_no_new_project_folder(home, tmp_path, monkeypatch, out, error, code):
    """analyze removes the folders it created (#217): the project and, for a new nested
    --out, its new parents; on an error and on Ctrl+C."""
    video = tmp_path / "videos" / "a.mp4"
    video.parent.mkdir()
    video.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cutpoint.analyze, "Transcriber", lambda **kw: None)
    reached: list[Path] = []

    def analyze_source(src, *a, **kw):
        reached.append(src)  # inside analyze's project folder, which exists by now
        raise error

    monkeypatch.setattr(cutpoint.analyze, "analyze_source", analyze_source)
    assert cli.main(["analyze", str(video), "--no-vision", "--out", out]) == code
    assert reached == [video]
    projects = home / "projects"
    assert not projects.exists() or list(projects.iterdir()) == []


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
