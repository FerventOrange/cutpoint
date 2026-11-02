from __future__ import annotations

import logging
from pathlib import Path

import pytest

import cutpoint
import cutpoint.analyze
from cutpoint import cli
from cutpoint.assistant import cliproxy


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


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["clips", "some-project", "--size", "abc"], "expected WxH"),
        (["clips", "some-project", "--top", "-1"], "must be greater than 0"),
        (["edit", "video.mp4", "--vision-top", "0"], "must be greater than 0"),
        (["analyze", "video.mp4", "--must-include-crash", "1.5"], "must be from 0 to 1"),
        (["assemble", "c", "--transition", "fade", "--fade-dur", "0"], "must be greater than 0"),
    ],
)
def test_bad_option_value_is_a_usage_error(capsys, argv, message):
    """Checked before the long analysis, without a traceback: `--top -1` used to select all
    but one moment, `--vision-top -3` to rate all but the last 3 candidates."""
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert argv[-2] in err and message in err
    assert "Traceback" not in err


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


def test_guide_prints_agents_md(capsys):
    assert cli.main(["guide"]) == 0
    assert capsys.readouterr().out.startswith("# Cutpoint for AI agents")


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


def test_huggingface_hub_logs_through_cutpoints_handler():
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


@pytest.mark.parametrize("argv", [["proxy", "--port", "9999"], ["login", "codex", "--no-browser"]])
def test_proxy_and_login_warn_at_the_users_own_risk(monkeypatch, caplog, argv):
    ran = []
    monkeypatch.setattr(cliproxy, "run", lambda port=None: ran.append(port))
    monkeypatch.setattr(
        cliproxy, "start_login", lambda p: {"url": "https://auth.example/test", "state": "s1"}
    )
    monkeypatch.setattr(cliproxy, "poll_login", lambda s: {"state": "done", "error": None})
    monkeypatch.setattr(cliproxy, "status", lambda p: {"logged_in": True, "email": "u@example.com"})
    with caplog.at_level(logging.INFO):
        assert cli.main(argv) == 0
    assert "Use at your own risk" in caplog.text
    if argv[0] == "proxy":
        assert ran == [9999]
    else:
        assert "open this URL to log in: https://auth.example/test" in caplog.text


def test_proxy_command_exits_1_on_proxy_error(monkeypatch, caplog):
    def fail(port=None):
        raise cliproxy.ProxyError("port in use")

    monkeypatch.setattr(cliproxy, "run", fail)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as exc:
            cli.main(["proxy"])
        assert exc.value.code == 1
    assert "proxy failed: port in use" in caplog.text


def test_login_cli_poll_error_cancels_session(monkeypatch, caplog):
    class OAuthFixture:
        def __init__(self) -> None:
            self.sessions: dict[str, str] = {}

        def start_login(self, provider: str) -> dict[str, str]:
            state = f"state-{provider}"
            self.sessions[state] = "pending"
            return {"url": f"https://auth.example/{state}", "state": state}

        def cancel_login(self, state: str) -> None:
            if state in self.sessions:
                self.sessions[state] = "cancelled"

    fixture = OAuthFixture()
    monkeypatch.setattr(cliproxy, "start_login", fixture.start_login)
    monkeypatch.setattr(
        cliproxy,
        "poll_login",
        lambda s: {"state": "error", "error": "transport error"},
    )
    monkeypatch.setattr(cliproxy, "cancel_login", fixture.cancel_login)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as exc:
            cli.main(["login", "codex", "--no-browser"])
        assert exc.value.code == 1
    assert "login failed: transport error" in caplog.text
    assert fixture.sessions["state-codex"] == "cancelled"


def test_login_cli_cancellation_failure_preserves_primary_exit(monkeypatch, caplog):
    def failing_cancel(state: str) -> None:
        raise cliproxy.ProxyError("daemon socket closed")

    monkeypatch.setattr(
        cliproxy, "start_login", lambda p: {"url": "https://auth.example/test", "state": "s1"}
    )
    monkeypatch.setattr(
        cliproxy, "poll_login", lambda s: {"state": "error", "error": "transport failed"}
    )
    monkeypatch.setattr(cliproxy, "cancel_login", failing_cancel)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as exc:
            cli.main(["login", "codex", "--no-browser"])
        assert exc.value.code == 1
    assert "login failed: transport failed" in caplog.text


def test_login_antigravity_confirmation(monkeypatch, caplog):
    monkeypatch.setattr(
        cliproxy, "start_login", lambda p: {"url": "https://auth.example/test", "state": "s1"}
    )
    monkeypatch.setattr(cliproxy, "poll_login", lambda s: {"state": "done", "error": None})
    monkeypatch.setattr(
        cliproxy, "status", lambda p: {"logged_in": True, "email": "user@example.com"}
    )

    # Aborted when user inputs 'n'
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with caplog.at_level(logging.INFO):
        with pytest.raises(SystemExit) as exc:
            cli.main(["login", "antigravity", "--no-browser"])
        assert exc.value.code == 1
    assert "login aborted" in caplog.text

    # Succeeds when user inputs 'y'
    caplog.clear()
    monkeypatch.setattr("builtins.input", lambda _: "y")
    with caplog.at_level(logging.INFO):
        assert cli.main(["login", "antigravity", "--no-browser"]) == 0
    assert "logged in to" in caplog.text

    # Ctrl+C at the confirmation uses the CLI's interrupt exit, not rejection.
    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    assert cli.main(["login", "antigravity", "--no-browser"]) == 130


def test_logout_cli_command(monkeypatch, caplog):
    logged_out = []
    monkeypatch.setattr(cliproxy, "logout", lambda p: logged_out.append(p))
    with caplog.at_level(logging.INFO):
        assert cli.main(["logout", "codex"]) == 0
    assert logged_out == ["codex"]
    assert "logged out of" in caplog.text


def test_logout_cli_command_error(monkeypatch, caplog):
    def fail(provider):
        raise cliproxy.ProxyError("daemon offline")

    monkeypatch.setattr(cliproxy, "logout", fail)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as exc:
            cli.main(["logout", "codex"])
        assert exc.value.code == 1
    assert "logout failed: daemon offline" in caplog.text
