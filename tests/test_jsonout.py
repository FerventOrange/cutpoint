"""The ``--json`` result envelope (cutpoint.jsonout) and its CLI plumbing (#4)."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import envelope
from cutpoint import cli, ffmpeg_utils, jsonout
from cutpoint.ffmpeg_utils import FFmpegError, FFmpegNotFoundError


def test_envelope_collects_outputs_project_and_warnings(tmp_path, capsys):
    with jsonout.run("analyze") as result:
        assert jsonout.active()
        jsonout.set_project(tmp_path / "p")
        jsonout.output("highlights", tmp_path / "p" / "highlights.json")
        jsonout.output("count", 3)
        jsonout.output("clips", [tmp_path / "a.mp4"])
        jsonout.warn("kept_clips", "kept 1 clip", [tmp_path / "x.mp4"])
    assert not jsonout.active()
    doc = envelope(capsys.readouterr().out)
    assert doc == result.envelope()
    assert doc["ok"] is True and doc["command"] == "analyze"
    assert doc["project"] == str(tmp_path / "p")
    assert doc["outputs"] == {
        "highlights": str(tmp_path / "p" / "highlights.json"),
        "count": 3,
        "clips": [str(tmp_path / "a.mp4")],
    }
    assert doc["warnings"] == [
        {"code": "kept_clips", "message": "kept 1 clip", "files": [str(tmp_path / "x.mp4")]}
    ]


def test_collectors_do_nothing_without_json(tmp_path, capsys):
    assert not jsonout.active()
    jsonout.output("x", 1)
    jsonout.warn("w", "m")
    jsonout.fail("error", "m")
    jsonout.set_project(tmp_path)
    assert capsys.readouterr().out == ""


def test_relative_paths_become_absolute(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with jsonout.run("x"):
        jsonout.output("file", Path("a.mp4"))
    assert envelope(capsys.readouterr().out)["outputs"]["file"] == str(tmp_path / "a.mp4")


@pytest.mark.slow
def test_stdout_holds_only_the_envelope(capfd):
    """Prints, logs and child processes that inherit stdout go to stderr."""
    with jsonout.run("x"):
        print("stray print")
        sys.stdout.write("stray write\n")
        logging.getLogger("cutpoint").warning("a log line")
        subprocess.run([sys.executable, "-c", "print('child output')"], check=True)
    out, err = capfd.readouterr()
    assert envelope(out)["ok"] is True
    for text in ("stray print", "stray write", "child output"):
        assert text in err


def test_first_error_wins_and_exception_is_reraised(capsys):
    with pytest.raises(KeyError), jsonout.run("x"):
        raise KeyError("boom")
    doc = envelope(capsys.readouterr().out)
    assert doc["ok"] is False and doc["error"] == {"code": "error", "message": "'boom'"}
    with jsonout.run("x"):
        jsonout.fail("not_found", "first")
        jsonout.fail("error", "second")
    assert envelope(capsys.readouterr().out)["error"] == {"code": "not_found", "message": "first"}


@pytest.mark.parametrize(
    ("err", "code"),
    [
        (KeyboardInterrupt(), "interrupted"),
        (FFmpegNotFoundError("ffprobe not found. Install it"), "ffmpeg_missing"),
        (FFmpegError("ffmpeg not found. Install it"), "error"),  # the class decides, not the text
        (FFmpegError("ffmpeg failed (exit 1)"), "error"),
        (FileExistsError("clips is not empty; use --force"), "output_not_empty"),
        (FileNotFoundError("no such file"), "not_found"),
        (ValueError("bad"), "error"),
    ],
)
def test_error_code(err, code):
    assert jsonout.error_code(err) == code


def test_find_tool_raises_ffmpeg_not_found(monkeypatch):
    monkeypatch.setattr(ffmpeg_utils.shutil, "which", lambda name: None)
    monkeypatch.setattr(ffmpeg_utils.sys, "platform", "linux")  # no WinGet folder search
    ffmpeg_utils.find_tool.cache_clear()
    try:
        with pytest.raises(FFmpegNotFoundError, match=r"^ffprobe not found") as exc:
            ffmpeg_utils.find_tool("ffprobe")
    finally:
        ffmpeg_utils.find_tool.cache_clear()
    assert isinstance(exc.value, FFmpegError)  # existing `except FFmpegError` still catches it
    assert jsonout.error_code(exc.value) == "ffmpeg_missing"


def test_output_values_are_always_serializable(tmp_path, capsys):
    """Sets, tuples and other objects never break the final write."""

    class Odd:
        def __str__(self) -> str:
            return "odd"

    with jsonout.run("x"):
        jsonout.output("set", {tmp_path / "a"})
        jsonout.output("tuple", (1, Path("b")))
        jsonout.output("odd", Odd())
        jsonout.output("nested", {"k": [frozenset({2})]})
        jsonout.warn("w", "m", (tmp_path / "c",))
    doc = envelope(capsys.readouterr().out)
    assert doc["outputs"] == {
        "set": [str(tmp_path / "a")],
        "tuple": [1, str(Path("b").absolute())],
        "odd": "odd",
        "nested": {"k": [[2]]},
    }
    assert doc["warnings"][0]["files"] == [str(tmp_path / "c")]


def test_guard_without_stderr(monkeypatch, capsys):
    """No sys.stderr (closed, pythonw): stray output is dropped, stdout gets the envelope."""
    stdout = sys.stdout
    monkeypatch.setattr(sys, "stderr", None)
    with jsonout.run("x"):
        print("dropped")
        jsonout.output("x", 1)
    assert sys.stdout is stdout
    out = capsys.readouterr().out
    assert envelope(out)["outputs"] == {"x": 1} and "dropped" not in out
    # argparse's usage does not fall back to stdout either
    with pytest.raises(SystemExit) as exc:
        cli.main(["paths", "--json", "--bogus"])
    assert exc.value.code == 2
    error = envelope(capsys.readouterr().out)["error"]
    assert error == {"code": "bad_args", "message": "unrecognized arguments: --bogus"}


def test_systemexit_message_is_bad_args(monkeypatch, capsys):
    """sys.exit("bad --window-at ...") (cutpoint events) keeps its exit and message."""
    _paths_raising(monkeypatch, SystemExit("bad --window-at 'x'; use GLOB@START-END"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["paths", "--json"])
    assert exc.value.code == "bad --window-at 'x'; use GLOB@START-END"  # exit 1, as before
    error = envelope(capsys.readouterr().out)["error"]
    assert error == {"code": "bad_args", "message": "bad --window-at 'x'; use GLOB@START-END"}


def test_abbreviated_json(home, capsys):
    assert cli.main(["paths", "--js"]) == 0
    assert envelope(capsys.readouterr().out)["command"] == "paths"
    with pytest.raises(SystemExit) as exc:
        cli.main(["paths", "--jso", "--bogus"])
    assert exc.value.code == 2
    assert envelope(capsys.readouterr().out)["error"]["code"] == "bad_args"


def test_paths_json(home, capsys):
    assert cli.main(["paths", "--json"]) == 0
    doc = envelope(capsys.readouterr().out)
    assert doc["ok"] is True and doc["command"] == "paths" and doc["project"] is None
    outputs = doc["outputs"]
    assert outputs["home"] == str(home)
    keys = ["home", "config", "cache", "models", "tools", "llama.cpp", "stage", "styles"]
    keys += ["presets", "profiles", "projects", "bundled", "missing"]
    assert list(outputs) == keys
    assert all(Path(outputs[k]).is_absolute() for k in keys[:-1])
    assert "home" in outputs["missing"] and "bundled" not in outputs["missing"]
    assert not home.exists()  # creates nothing, as without --json


def _paths_raising(monkeypatch, err: BaseException) -> None:
    def boom(a):
        print("noise")
        raise err

    monkeypatch.setattr(cli, "_cmd_paths", boom)


@pytest.mark.parametrize(
    ("err", "code", "exit_code", "message"),
    [
        (FileNotFoundError("Segments file not found: x"), "not_found", 1, None),
        (FFmpegNotFoundError("ffmpeg not found. Install it"), "ffmpeg_missing", 1, None),
        (FileExistsError("clips is not empty; use --force"), "output_not_empty", 1, None),
        (RuntimeError("encoder failed"), "error", 1, None),
        (ModuleNotFoundError("x", name="telemetry_parser"), "missing_extra", 1, "'ski' extra"),
        (KeyboardInterrupt(), "interrupted", 130, "interrupted"),
    ],
)
def test_cli_error_envelope(monkeypatch, capsys, err, code, exit_code, message):
    _paths_raising(monkeypatch, err)
    assert cli.main(["paths", "--json"]) == exit_code
    out, stderr = capsys.readouterr()
    doc = envelope(out)
    assert doc["ok"] is False and doc["command"] == "paths"
    assert doc["error"]["code"] == code
    assert (message or str(err)) in doc["error"]["message"]
    assert "noise" in stderr


def test_cli_error_without_json_prints_nothing_on_stdout(monkeypatch, capsys):
    _paths_raising(monkeypatch, FileNotFoundError("gone"))
    assert cli.main(["paths"]) == 1
    assert capsys.readouterr().out == "noise\n"


def test_cli_unexpected_exception_still_writes_envelope(monkeypatch, capsys):
    _paths_raising(monkeypatch, KeyError("bug"))
    with pytest.raises(KeyError):
        cli.main(["paths", "--json"])
    assert envelope(capsys.readouterr().out)["error"]["code"] == "error"


def test_cli_nonzero_return_is_not_ok(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_cmd_paths", lambda a: 3)
    assert cli.main(["paths", "--json"]) == 3
    error = envelope(capsys.readouterr().out)["error"]
    assert error == {"code": "error", "message": "exit code 3"}


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["paths", "--json", "--bogus"], "unrecognized arguments: --bogus"),
        (["paths", "extra", "--json"], "unrecognized arguments: extra"),
        (["clips", "p", "--size", "abc", "--json"], "argument --size: 'abc': expected WxH"),
        (["crashes", "ski", "--dense-top", "many", "--json"], "argument --dense-top: invalid int"),
        (["profile", "ski", "--json", "--activity"], "argument --activity: expected one argument"),
        (["setup", "--only", "bogus", "--json"], "argument --only: invalid choice: 'bogus'"),
        (["clips", "p", "--size", "abc", "--j"], "argument --size: 'abc': expected WxH"),
        (["paths", "--json=1"], "argument --json: ignored explicit argument '1'"),
    ],
    ids=[
        *["paths-unknown-option", "paths-extra", "clips-size", "crashes", "profile", "setup"],
        "abbreviated",
        "json-with-value",
    ],
)
def test_bad_args_envelope(capsys, argv, message):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    out, err = capsys.readouterr()
    doc = envelope(out)
    assert doc["error"]["code"] == "bad_args" and doc["error"]["message"].startswith(message)
    assert doc["command"] == argv[0]
    assert "usage: cutpoint" in err and message in err


def test_bad_args_envelope_from_the_real_argv(monkeypatch, capsys):
    """``cli.main()`` without argv reads ``sys.argv[1:]``."""
    monkeypatch.setattr(sys, "argv", ["cutpoint", "paths", "--json", "--bogus"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert envelope(capsys.readouterr().out)["error"]["code"] == "bad_args"


def test_bad_args_inside_subcommand_parser(monkeypatch, capsys):
    """A usage error raised by the subcommand's own parser also gives an envelope."""
    real = cli._add_json_argument

    def with_number(p):
        real(p)
        p.add_argument("--n", type=int)

    monkeypatch.setattr(cli, "_add_json_argument", with_number)
    with pytest.raises(SystemExit) as exc:
        cli.main(["paths", "--js", "--n", "x"])
    assert exc.value.code == 2
    out, err = capsys.readouterr()
    doc = envelope(out)
    assert doc["command"] == "paths" and doc["error"]["code"] == "bad_args"
    assert "invalid int value" in doc["error"]["message"] and "invalid int value" in err


@pytest.mark.parametrize(
    "argv",
    [
        ["paths", "--bogus"],  # no --json
        ["pathz", "--json"],  # unknown command: no command to report
        ["fit", "x", "--json"],  # a command without --json (yet)
        ["clips", "p", "--size", "abc"],  # an error of the command's own parser, no --json
        ["clips", "p", "--size", "abc", "--", "--json"],  # after "--" it is not an option
        ["clips", "p", "--size", "abc", "--jsonx"],  # not an abbreviation of --json
    ],
)
def test_bad_args_without_envelope(capsys, argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    out, err = capsys.readouterr()
    assert out == "" and "usage:" in err


@pytest.mark.slow
@pytest.mark.parametrize("stderr", ["closed", "devnull"])
def test_json_subprocess_without_stderr(home, tmp_path, stderr):
    """With stderr closed (fd 2 invalid, sys.stderr None) or the null device: one envelope.

    A child cannot be started with a closed fd 2 portably (Windows has no ``2>&-``), so the
    "closed" case closes fd 2 inside the process before running the CLI.
    """
    script = tmp_path / "run.py"
    script.write_text(
        "import os, subprocess, sys\n"
        "from cutpoint import cli\n"
        f"if {stderr == 'closed'}:\n"
        "    sys.stderr.flush()\n"
        "    os.close(2)\n"
        "    sys.stderr = None\n"
        "def noisy(a):\n"
        "    print('stray print')\n"
        "    subprocess.run([sys.executable, '-c', 'print(\"child output\")'], check=True)\n"
        "    cli.jsonout.output('x', 1)\n"
        "real = sys.stdout\n"
        "cli._cmd_paths = noisy\n"
        "code = cli.main(['paths', '--json'])\n"
        "assert sys.stdout is real\n"
        "sys.exit(code)\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    doc = envelope(proc.stdout)
    assert doc["ok"] is True and doc["outputs"] == {"x": 1}


@pytest.mark.slow
@pytest.mark.parametrize("json_flag", [[], ["--json"]])
def test_subprocess_systemexit_message_keeps_exit_1(home, tmp_path, json_flag):
    script = tmp_path / "run.py"
    script.write_text(
        "import sys\n"
        "from cutpoint import cli\n"
        "def bad(a):\n"
        "    cli.logger.warning('a log line')\n"
        "    raise SystemExit('bad override')\n"
        "cli._cmd_paths = bad\n"
        f"sys.exit(cli.main(['paths', *{json_flag!r}]))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 1
    assert "bad override" in proc.stderr
    assert "a log line" in proc.stderr  # a real process logs to stderr, never to stdout
    if json_flag:
        assert envelope(proc.stdout)["error"] == {"code": "bad_args", "message": "bad override"}
    else:
        assert proc.stdout == ""


@pytest.mark.parametrize("json_flag", [True, False], ids=["json", "plain"])
def test_assemble_refuses_an_audio_crossfade_at_cuts_with_exit_2(capsys, json_flag):
    argv = ["assemble", "clips", "--transition", "cut", "--audio-transition", "crossfade"]
    with pytest.raises(SystemExit) as exc:
        cli.main([*argv, "--json"] if json_flag else argv)
    assert exc.value.code == 2
    out, err = capsys.readouterr()
    assert "--transition fade" in err and "usage: cutpoint" in err
    if json_flag:
        doc = envelope(out)
        assert doc["command"] == "assemble" and doc["ok"] is False
        assert doc["error"]["code"] == "bad_args" and "--transition fade" in doc["error"]["message"]
    else:
        assert out == ""


@pytest.mark.parametrize(
    ("extra", "audio"),
    [
        ([], None),  # follows --transition, decided in assemble()
        (["--transition", "fade"], None),
        (["--transition", "fade", "--audio-transition", "cut"], "cut"),
        (["--transition", "fade", "--audio-transition", "crossfade"], "crossfade"),
        (["--audio-transition", "cut"], "cut"),
    ],
)
def test_assemble_audio_transition_reaches_assemble(monkeypatch, tmp_path, extra, audio):
    calls = []
    monkeypatch.setattr(cli, "assemble", lambda *a, **kw: calls.append(kw) or tmp_path / "f.mp4")
    args = cli.build_parser().parse_args(["assemble", str(tmp_path), *extra])
    cli._cmd_assemble(args)
    assert calls[0]["audio_transition"] == audio
