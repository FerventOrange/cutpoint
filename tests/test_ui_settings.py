"""Settings and the native picker of ``cutpoint ui``; settings changes vs a job start (#143)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

pytest.importorskip("httpx")
import importlib.util
import tempfile

from fastapi import HTTPException
from fastapi.testclient import TestClient

from conftest import link_dir, link_file, media_info, patch_ui, set_config
from cutpoint import cache, config, procs, ui
from ui_helpers import (
    FakeDialog,
    FakeProc,
    cli_project,
    make_project,
    media_link,
    peaks_job,
    record_export,
    touch,
)

# the encoder probe and the keyring stay stubbed in every test (ui_helpers fixtures)
pytestmark = pytest.mark.usefixtures("no_nvenc", "no_keyring")


def test_pick_returns_the_chosen_path_or_null_on_cancel(
    client: TestClient, dialog: FakeDialog, tmp_path: Path
) -> None:
    assert client.get("/api/session").json()["can_pick"] is True
    video = tmp_path / "v.mp4"
    dialog.out = f"{video}\n"
    res = client.post("/api/pick", params={"kind": "file"})
    assert res.status_code == 200 and res.json() == {"path": os.path.normpath(video)}
    assert dialog.calls == [["dlg", "file"]]

    dialog.out = ""  # closed without choosing (exit 0, nothing printed)
    assert client.post("/api/pick", params={"kind": "folder"}).json() == {"path": None}
    dialog.returncode = 1  # the cancel exit code of zenity/kdialog
    assert client.post("/api/pick", params={"kind": "folder"}).json() == {"path": None}


def test_pick_errors(
    client: TestClient, dialog: FakeDialog, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.post("/api/pick", params={"kind": "other"}).status_code == 400
    assert dialog.calls == []

    dialog.returncode, dialog.err = 2, "boom"
    res = client.post("/api/pick", params={"kind": "file"})
    assert res.status_code == 500 and "failed" in res.json()["detail"]

    dialog.returncode, dialog.out = 0, "relative/path.mp4\n"
    assert client.post("/api/pick", params={"kind": "file"}).status_code == 500

    def broken(*a, **kw):
        raise OSError(2, "No such file")

    monkeypatch.setattr(subprocess, "Popen", broken)
    res = client.post("/api/pick", params={"kind": "file"})
    assert res.status_code == 500 and "No such file" in res.json()["detail"]
    # The failure released the lock, so the next dialog can open.
    monkeypatch.setattr(subprocess, "Popen", dialog)
    dialog.out = ""
    assert client.post("/api/pick", params={"kind": "file"}).status_code == 200

    patch_ui(monkeypatch, "path_picker", lambda: None)
    assert client.post("/api/pick", params={"kind": "file"}).status_code == 501
    assert client.get("/api/session").json()["can_pick"] is False


def test_pick_allows_one_dialog_at_a_time(
    client: TestClient, dialog: FakeDialog, tmp_path: Path
) -> None:
    dialog.block, dialog.out = True, str(tmp_path / "v.mp4")
    first: list[int] = []
    t = threading.Thread(
        target=lambda: first.append(client.post("/api/pick", params={"kind": "file"}).status_code)
    )
    t.start()
    try:
        assert dialog.opened.wait(30)
        assert client.post("/api/pick", params={"kind": "folder"}).status_code == 409
    finally:
        dialog.release.set()
        t.join(30)
    assert first == [200] and len(dialog.calls) == 1
    assert client.post("/api/pick", params={"kind": "file"}).status_code == 200


def test_pick_start_is_the_nearest_existing_absolute_folder(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "f.mp4").write_bytes(b"x")
    assert ui.pick_start(str(tmp_path / "a")) == tmp_path / "a"
    assert ui.pick_start(str(tmp_path / "a" / "gone" / "deeper")) == tmp_path / "a"
    assert ui.pick_start(str(tmp_path / "f.mp4")) == tmp_path  # a file: its folder
    for ignored in ("", "relative/dir", "a", "bad\0path"):
        assert ui.pick_start(ignored) is None, ignored


def test_pick_passes_start_to_the_dialog(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    starts: list[Path | None] = []

    def build(kind, start):
        starts.append(start)
        return ["dlg"], None

    patch_ui(monkeypatch, "path_picker", lambda: build)
    monkeypatch.setattr(subprocess, "Popen", FakeDialog())
    client.post("/api/pick", params={"kind": "file", "start": str(tmp_path / "missing" / "x")})
    client.post("/api/pick", params={"kind": "file", "start": "relative"})
    assert starts == [tmp_path, None]


def test_path_picker_builds_the_platform_dialog_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    present: set[str] = set()
    monkeypatch.setattr(shutil, "which", lambda n: f"/bin/{n}" if n in present else None)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    present.update({"zenity", "kdialog"})
    assert ui.path_picker() is None  # no display: nobody would see the window
    monkeypatch.setenv("DISPLAY", ":0")
    argv, cancel = ui.path_picker()("folder", tmp_path)
    assert argv[:2] == ["/bin/zenity", "--file-selection"]
    assert "--directory" in argv and f"--filename={tmp_path}{os.sep}" in argv
    assert any(a.startswith("--title=") for a in argv)
    assert cancel == 1
    argv, _ = ui.path_picker()("file", None)
    assert argv[:2] == ["/bin/zenity", "--file-selection"] and "--directory" not in argv
    assert any("*.mp4" in a for a in argv) and not any(a.startswith("--filename") for a in argv)
    assert any(a.startswith("--file-filter=Insta360") and "*.insv" in a for a in argv)

    present.discard("zenity")  # kdialog is the fallback
    argv, cancel = ui.path_picker()("folder", tmp_path)
    assert argv[0] == "/bin/kdialog" and "--title" in argv
    assert argv[argv.index("--getexistingdirectory") + 1] == str(tmp_path)
    argv, cancel = ui.path_picker()("file", tmp_path)
    assert argv[0] == "/bin/kdialog" and "--getexistingdirectory" not in argv
    assert argv[argv.index("--getopenfilename") + 1] == str(tmp_path)
    assert "*.mp4" in argv[-1] and cancel == 1

    present.clear()
    assert ui.path_picker() is None  # a display but no dialog tool

    monkeypatch.setattr(sys, "platform", "darwin")
    assert ui.path_picker() is None
    present.add("osascript")
    argv, cancel = ui.path_picker()("folder", tmp_path)
    assert argv[:2] == ["/bin/osascript", "-e"] and "choose" in argv[2]
    assert argv[3:] == ["folder", str(tmp_path)] and cancel is None
    assert ui.path_picker()("file", None)[0][3:] == ["file"]

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert ui.path_picker() is None  # Python without tkinter
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    argv, cancel = ui.path_picker()("folder", tmp_path)
    assert argv[:3] == [sys.executable, "-I", "-c"]
    assert argv[4:6] == ["folder", str(tmp_path)] and "*.mp4" in argv[6] and cancel is None
    assert ui.path_picker()("file", None)[0][5] == ""


@pytest.mark.parametrize(("platform", "var"), [("win32", "SSH_CONNECTION"), ("darwin", "SSH_TTY")])
def test_path_picker_is_off_in_an_ssh_session_on_windows_and_macos(
    monkeypatch: pytest.MonkeyPatch, platform: str, var: str
) -> None:
    monkeypatch.setattr(shutil, "which", lambda n: f"/bin/{n}")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    assert ui.path_picker() is not None
    monkeypatch.setenv(
        var, "10.0.0.1 22 10.0.0.2 5555" if var == "SSH_CONNECTION" else "/dev/pts/0"
    )
    assert ui.path_picker() is None


def test_picker_shutdown_kills_an_open_dialog_and_frees_the_lock(
    client: TestClient, dialog: FakeDialog, tmp_path: Path
) -> None:
    dialog.block = True
    killed: list[bool] = []
    real_kill = dialog.release.set
    dialog.release.set = lambda: (killed.append(True), real_kill())[1]  # type: ignore[method-assign]
    status: list[int] = []
    t = threading.Thread(
        target=lambda: status.append(client.post("/api/pick", params={"kind": "file"}).status_code)
    )
    t.start()
    try:
        assert dialog.opened.wait(30)
        picker = client.app.state.picker
        assert picker.proc is not None
        picker.shutdown()
    finally:
        dialog.release.set()
        t.join(30)
    assert not t.is_alive() and killed and picker.proc is None
    dialog.block, dialog.out = False, ""
    assert client.post("/api/pick", params={"kind": "file"}).status_code == 200


# --------------------------------------------------------------------------- settings
def test_settings_report_the_folder_the_lock_and_the_config_file(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    res = client.get("/api/settings")
    assert res.status_code == 200
    got = res.json()
    assert got["projects_dir"] == str(config.projects_dir())
    assert got["projects_dir_locked"] is False
    assert got["config_file"] == str(home / "config.json")
    assert got["vision"] is False and got["vision_model"]
    locked = tmp_path / "locked"
    monkeypatch.setenv("CUTPOINT_PROJECTS_DIR", str(locked))
    config.get_config.cache_clear()
    got = client.get("/api/settings").json()
    assert got["projects_dir_locked"] is True and got["projects_dir"] == str(locked)


def test_put_settings_saves_the_folder_and_lists_projects_from_it(
    client: TestClient, home: Path, tmp_path: Path
) -> None:
    new = tmp_path / "elsewhere"  # only the last part may be created
    res = client.put("/api/settings", json={"projects_dir": str(new)})
    assert res.status_code == 200, res.text
    assert res.json()["projects_dir"] == str(new) and new.is_dir()
    assert json.loads((home / "config.json").read_text("utf-8")) == {"projects_dir": str(new)}
    assert client.get("/api/settings").json()["projects_dir"] == str(new)

    (new / "trip").mkdir()
    cache.save_json(new / "trip" / "highlights.json", {})
    assert [p["id"] for p in client.get("/api/projects").json()["projects"]] == ["trip"]

    other = tmp_path / "other"  # an existing folder works too
    other.mkdir()
    assert client.put("/api/settings", json={"projects_dir": f" {other} "}).status_code == 200
    assert client.get("/api/projects").json() == {"projects": []}


def test_put_settings_rejects_unusable_folders(
    client: TestClient, home: Path, tmp_path: Path
) -> None:
    a_file = tmp_path / "file.txt"
    a_file.write_text("x")
    cases = [
        {},
        {"projects_dir": ""},
        {"projects_dir": 5},
        {"projects_dir": "relative/dir"},
        {"projects_dir": str(a_file)},
        {"projects_dir": str(tmp_path / "no" / "parent")},
    ]
    for body in cases:
        assert client.put("/api/settings", json=body).status_code == 400, body
    assert not (tmp_path / "no").exists()
    assert not (home / "config.json").exists()


def test_put_settings_conflicts(
    client: TestClient, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = {"projects_dir": str(tmp_path)}
    cfg = home / "config.json"

    assert client.get("/api/settings").status_code == 200  # config loaded (and cached) while valid
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{not json", encoding="utf-8")
    new = tmp_path / "fresh"
    res = client.put("/api/settings", json={"projects_dir": str(new)})
    assert res.status_code == 409 and cfg.read_text("utf-8") == "{not json"  # never overwritten
    assert not new.exists()  # nor is the new folder created
    cfg.unlink()
    config.get_config.cache_clear()

    client.app.state.jobs.jobs["j1"] = SimpleNamespace(status="running", proc=None)
    assert client.put("/api/settings", json=body).status_code == 409
    client.app.state.jobs.jobs["j1"].status = "done"
    assert client.put("/api/settings", json=body).status_code == 200

    monkeypatch.setenv("CUTPOINT_PROJECTS_DIR", str(tmp_path / "env"))
    res = client.put("/api/settings", json={"projects_dir": str(tmp_path / "x")})
    assert res.status_code == 409 and "CUTPOINT_PROJECTS_DIR" in res.json()["detail"]
    assert json.loads(cfg.read_text("utf-8")) == {"projects_dir": str(tmp_path)}


def test_put_settings_saves_the_vision_default(
    client: TestClient, home: Path, tmp_path: Path
) -> None:
    cfg = home / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('{\n  "player_tags": ["Me"]\n}', encoding="utf-8")
    res = client.put("/api/settings", json={"vision": True})
    assert res.status_code == 200, res.text
    assert res.json()["vision"] is True
    assert json.loads(cfg.read_text("utf-8")) == {"player_tags": ["Me"], "vision": True}
    assert client.get("/api/settings").json()["vision"] is True
    assert config.get_config().vision is True

    # Unticking removes the key (the default) instead of writing false (#919).
    assert client.put("/api/settings", json={"vision": False}).json()["vision"] is False
    assert config.get_config().vision is None
    assert json.loads(cfg.read_text("utf-8")) == {"player_tags": ["Me"]}
    assert client.put("/api/settings", json={"vision": True}).json()["vision"] is True

    both = {"projects_dir": str(tmp_path), "vision": True}
    got = client.put("/api/settings", json=both).json()
    assert got["vision"] is True and got["projects_dir"] == str(tmp_path)
    data = json.loads(cfg.read_text("utf-8"))
    assert data == {"player_tags": ["Me"], "vision": True, "projects_dir": str(tmp_path)}

    other = tmp_path / "other"  # a later folder replaces the old one in place; other keys stay
    other.mkdir()
    assert client.put("/api/settings", json={"projects_dir": str(other)}).status_code == 200
    data = json.loads(cfg.read_text("utf-8"))
    assert list(data) == ["player_tags", "vision", "projects_dir"]
    assert data["projects_dir"] == str(other)


def test_put_settings_vision_needs_a_boolean_and_no_folder_rules(
    client: TestClient, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for value in ("yes", 1, None):
        assert client.put("/api/settings", json={"vision": value}).status_code == 400, value
    assert not (home / "config.json").exists()

    # Only the projects folder is locked by CUTPOINT_PROJECTS_DIR or a running job.
    monkeypatch.setenv("CUTPOINT_PROJECTS_DIR", str(tmp_path / "env"))
    client.app.state.jobs.jobs["j1"] = SimpleNamespace(status="running", proc=None, cancelled=False)
    assert client.put("/api/settings", json={"vision": True}).status_code == 200
    both = {"vision": False, "projects_dir": str(tmp_path)}
    assert client.put("/api/settings", json=both).status_code == 409
    assert json.loads((home / "config.json").read_text("utf-8")) == {"vision": True}


def test_settings_show_a_wrongly_typed_vision_as_off_and_replace_it(
    client: TestClient, home: Path
) -> None:
    set_config(home, vision="yes")
    assert client.get("/api/projects").status_code == 409  # rejected by get_config
    assert client.get("/api/settings").json()["vision"] is False
    assert client.put("/api/settings", json={"vision": True}).status_code == 200
    assert client.get("/api/projects").status_code == 200


@pytest.mark.parametrize(
    ("cfg", "shown"),
    [(None, "off"), ("light", "light"), ("strong", "strong"), ("loud", "off"), (3, "off")],
)
def test_settings_show_the_compress_peaks_default(
    client: TestClient, home: Path, cfg: object, shown: str
) -> None:
    """The page's Compress peaks select starts on config.json's value; an invalid one shows off."""
    if cfg is not None:
        set_config(home, compress_peaks=cfg)
    assert client.get("/api/settings").json()["compress_peaks"] == shown


@pytest.mark.parametrize("action", ["clips", "assemble"])
@pytest.mark.parametrize(
    ("body", "cfg", "passed"),
    [
        ({"compress_peaks": "strong"}, None, "strong"),
        ({"compress_peaks": "light"}, "strong", "light"),
        ({"compress_peaks": "off"}, "strong", "off"),
        ({}, "light", "light"),  # not sent: config.json's value, explicitly
        ({}, None, "off"),
    ],
)
def test_clips_and_assemble_jobs_pass_compress_peaks(
    client: TestClient,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action: str,
    body: dict,
    cfg: str | None,
    passed: str,
) -> None:
    if cfg is not None:
        set_config(home, compress_peaks=cfg)
    res, captured = peaks_job(client, monkeypatch, tmp_path, {"action": action, **body})
    assert res.status_code == 200, res.text
    args = captured[0]
    assert args.count("--compress-peaks") == 1
    assert args[args.index("--compress-peaks") + 1] == passed


# one validator for clips and assemble (both pass it: the test above)
@pytest.mark.parametrize("value", ["loud", None, ["light"]])
def test_clips_and_assemble_jobs_reject_an_unknown_compress_peaks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: object
) -> None:
    body = {"action": "clips", "compress_peaks": value}
    res, captured = peaks_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 400, res.text
    assert "compress_peaks must be one of off, light, strong" in res.json()["detail"]
    assert not captured


def test_put_settings_removes_a_folder_it_created_when_saving_fails(
    client: TestClient, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    new = tmp_path / "fresh"

    def broken(*a, **kw):
        raise OSError(28, "No space left")

    patch_ui(monkeypatch, "update_config", broken)
    res = client.put("/api/settings", json={"projects_dir": str(new)})
    assert res.status_code == 500 and not new.exists()


def test_put_settings_never_removes_an_existing_folder_on_failure(
    client: TestClient, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "keep"
    existing.mkdir()
    (existing / "f.txt").write_text("x")

    def broken(*a, **kw):
        raise OSError(28, "No space left")

    patch_ui(monkeypatch, "update_config", broken)
    res = client.put("/api/settings", json={"projects_dir": str(existing)})
    assert res.status_code == 500 and (existing / "f.txt").exists()


def test_check_projects_dir_removes_its_own_folder_when_the_write_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(*a, **kw):
        raise PermissionError(13, "Permission denied")

    new = tmp_path / "probe"
    monkeypatch.setattr(tempfile, "TemporaryFile", denied)
    with pytest.raises(HTTPException) as e:
        ui.check_projects_dir(str(new))
    assert e.value.status_code == 400 and "Permission denied" in e.value.detail
    assert not new.exists()
    existing = tmp_path / "keep"  # a folder that was already there is never removed
    existing.mkdir()
    (existing / "f.txt").write_text("x")
    with pytest.raises(HTTPException):
        ui.check_projects_dir(str(existing))
    assert (existing / "f.txt").exists()
    monkeypatch.undo()
    assert ui.check_projects_dir(str(new)) == (new, True)
    assert ui.check_projects_dir(str(new)) == (new, False)


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("{oops", "Expecting property name"),
        ('{"video_encoder": "bogus"}', "video_encoder is 'bogus'"),
        ('{"source_remap_exclude": 5}', "int"),  # wrong type: ValueError -> 409, not a 500
        ('{"projects_dir": 5}', "int"),
    ],
)
def test_project_routes_name_a_rejected_config_file_and_settings_still_work(
    client: TestClient,
    home: Path,
    tmp_path: Path,
    text: str,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the env var would win over the bogus config value
    monkeypatch.delenv("CUTPOINT_VIDEO_ENCODER")
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    cfg = home / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(text, encoding="utf-8")
    config.get_config.cache_clear()
    token = client.app.state.token
    calls = [
        ("GET", "/api/projects", {}),
        ("POST", "/api/projects", {"json": {"name": "x", "source": str(tmp_path / "footage")}}),
        ("GET", f"/api/projects/{pid}", {}),
        ("PUT", f"/api/projects/{pid}/review", {"json": {}}),
        ("PUT", f"/api/projects/{pid}/target", {"json": {"seconds": 60}}),
        ("POST", f"/api/projects/{pid}/reveal", {}),
        ("POST", f"/api/projects/{pid}/jobs", {"json": {"action": "analyze"}}),
        ("POST", f"/api/projects/{pid}/sources/s0/prepare", {}),
        ("GET", f"/api/projects/{pid}/sources/s0", {"params": {"t": token}}),
        ("GET", f"/api/projects/{pid}/sources/s0/frame", {"params": {"t": token}}),
        ("GET", f"/api/projects/{pid}/outputs/final.mp4", {"params": {"t": token}}),
    ]
    for method, path, kw in calls:
        res = client.request(method, path, **kw)
        assert res.status_code == 409, (method, path, res.text)
        detail = res.json()["detail"]
        assert str(cfg) in detail and error in detail, (path, detail)
    assert client.get("/api/projects", headers={"X-Cutpoint-Token": "nope"}).status_code == 403
    assert client.get("/api/session").status_code == 200
    assert client.get("/api/jobs").status_code == 200
    settings = client.get("/api/settings")
    if text.startswith("{oops"):  # unreadable: Settings say so too, and never overwrite it
        assert settings.status_code == 409 and str(cfg) in settings.json()["detail"]
    else:  # a rejected unrelated value: Settings still show and save the projects folder
        assert settings.status_code == 200 and settings.json()["config_file"] == str(cfg)
        if "projects_dir" in text:  # shown as typed, so it can be replaced
            assert settings.json()["projects_dir"] == "5"
        assert client.put("/api/settings", json={"projects_dir": str(tmp_path)}).status_code == 200
        # the rejected unrelated value is kept; a bad projects_dir is the one thing replaced
        assert json.loads(cfg.read_text("utf-8")) == {
            **json.loads(text),
            "projects_dir": str(tmp_path),
        }

    cfg.unlink()  # fixed by hand: the projects come back without a restart
    assert [p["id"] for p in client.get("/api/projects").json()["projects"]] == [pid]


def test_a_rejected_encoder_variable_is_named_instead_of_config_json(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = home / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "bogus")
    config.get_config.cache_clear()
    res = client.get("/api/projects")
    assert res.status_code == 409
    detail = res.json()["detail"]
    assert detail.startswith("CUTPOINT_VIDEO_ENCODER is 'bogus'; expected one of auto,"), detail
    assert str(cfg) not in detail


@pytest.mark.parametrize(
    "content",
    [
        "{oops",
        '{"base": "funny", "weights": [1, 2]}',
        '{"base": "funny", "weights": {"x": null}}',
        '{"name": "broken", "base": "funny"}',
    ],
    ids=["invalid-json", "weights-not-object", "weight-null", "no-schema"],
)
def test_a_broken_learned_style_is_left_out_with_a_warning(
    client: TestClient, home: Path, caplog: pytest.LogCaptureFixture, content: str
) -> None:
    styles = home / "styles"
    styles.mkdir(parents=True)
    (styles / "broken.json").write_text(content, encoding="utf-8")
    mine = {"schema": 1, "name": "mine", "base": "funny", "source": "hand"}
    (styles / "mine.json").write_text(json.dumps(mine), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        res = client.get("/api/session")
    assert res.status_code == 200
    listed = res.json()["styles"]
    assert "mine" in listed and "funny" in listed and "broken" not in listed
    assert str(styles / "broken.json") in caplog.text


# ------------------------------------------------- settings change vs job start (#143)
SLEEP_JOB = [sys.executable, "-c", "import time; time.sleep(60)"]


@pytest.mark.slow
def test_put_settings_refuses_a_job_that_started_during_the_check_and_removes_the_new_folder(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    jobs = client.app.state.jobs
    real_check = ui.check_projects_dir
    started: list = []

    def check_then_start_job(value):
        result = real_check(value)  # creates the folder
        started.append(jobs.start("p", "analyze", SLEEP_JOB))  # a second tab wins the race
        return result

    patch_ui(monkeypatch, "check_projects_dir", check_then_start_job)
    new = tmp_path / "fresh"
    try:
        res = client.put("/api/settings", json={"projects_dir": str(new)})
        assert res.status_code == 409 and "running job" in res.json()["detail"]
        assert not new.exists()
        assert not (home / "config.json").exists()
    finally:
        for job in started:
            jobs.cancel(job.id)


@pytest.mark.slow
def test_a_job_cannot_start_while_the_projects_folder_is_being_written(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    jobs = client.app.state.jobs
    real_write = ui.update_config
    attempt: dict = {}

    def write_while_another_tab_starts_a_job(change):
        def start() -> None:
            attempt["job"] = jobs.start("p", "analyze", SLEEP_JOB)
            attempt["wrote"] = wrote.is_set()
            done.set()

        wrote, done = threading.Event(), threading.Event()
        attempt["wrote_event"] = wrote
        threading.Thread(target=start, daemon=True).start()
        attempt["overlapped"] = done.wait(0.5)  # True: the job started mid-write
        data = real_write(change)
        wrote.set()
        return data

    patch_ui(monkeypatch, "update_config", write_while_another_tab_starts_a_job)
    new = tmp_path / "elsewhere"
    try:
        res = client.put("/api/settings", json={"projects_dir": str(new)})
        assert res.status_code == 200, res.text
        assert attempt["overlapped"] is False
        assert attempt["wrote_event"].wait(5)
    finally:
        deadline = time.time() + 5
        while "job" not in attempt and time.time() < deadline:
            time.sleep(0.02)
        if "job" in attempt:
            jobs.cancel(attempt["job"].id)
    assert attempt["wrote"] is True  # the job started only after the new folder was saved


def test_a_job_is_refused_when_the_projects_folder_changed_after_the_project_was_opened(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    jobs = client.app.state.jobs
    p = ui.projects_dir() / "p"
    p.mkdir(parents=True)
    settings = {"inputs": [str(tmp_path / "clip.mp4")]}
    cache.save_json(p / "highlights.json", {"style": "gaming", "settings": settings})
    other = tmp_path / "other"
    other.mkdir()
    real_cmd = ui.cutpoint_cmd

    def cmd_while_the_folder_changes(*args: str) -> list[str]:
        config.update_config(lambda _d: ({"projects_dir": str(other)}, ()))  # a second tab saves
        return real_cmd(*args)

    patch_ui(monkeypatch, "cutpoint_cmd", cmd_while_the_folder_changes)
    try:
        res = client.post("/api/projects/p/jobs", json={"action": "analyze"})
        assert res.status_code == 409 and "projects folder changed" in res.json()["detail"]
        assert not jobs.jobs
    finally:
        for job in list(jobs.jobs.values()):
            jobs.cancel(job.id)


def project_with_link(
    client: TestClient, tmp_path: Path, what: str, data: object = None
) -> tuple[str, Path, Path]:
    """A project whose ``what`` (``ui`` or a metadata file) is a link into a folder outside it.

    Returns (project id, project folder, the link's target); the target holds ``data`` as JSON.
    """
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / Path(what).name
    if what == "ui":
        shutil.move(p / "ui", target)
        link_dir(p / "ui", target)
    else:
        cache.save_json(target, data if data is not None else {})
        (p / what).unlink(missing_ok=True)
        link_file(p / what, target)
    return pid, p, target


@pytest.mark.parametrize("what", ["ui", "highlights.json", "no meta", "review", "exports"])
def test_metadata_read_through_a_link_outside_the_project_is_ignored(
    client: TestClient, tmp_path: Path, what: str
) -> None:
    secret = {"segments": [{"id": "h001"}], "candidates": [{"id": "h001"}], "style": "funny"}
    if what in {"ui", "no meta"}:  # a link alone does not make a folder a project
        pid, p, _ = project_with_link(client, tmp_path, "ui" if what == "ui" else "highlights.json")
        if what == "no meta":
            shutil.rmtree(p / "ui")
        assert client.get("/api/projects").json() == {"projects": []}
        assert client.get(f"/api/projects/{pid}").status_code == 404
        return
    if what == "highlights.json":
        pid, _, _ = project_with_link(client, tmp_path, what, secret)
        got = client.get(f"/api/projects/{pid}").json()
        assert got["highlights"] is None
    elif what == "review":
        pid, _, _ = project_with_link(client, tmp_path, "ui/review.json", secret)
        got = client.get(f"/api/projects/{pid}").json()
        assert got["review"] is None
    else:
        pid, p, target = project_with_link(client, tmp_path, "ui/exports.json")
        (p / "clips_ui_1").mkdir()
        cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": [{"label": "x"}]})
        (p / "final.mp4").write_bytes(b"x")
        cache.save_json(target, {"exports": [{"dir": "clips_ui_1", "final": "final.mp4"}]})
        got = client.get(f"/api/projects/{pid}").json()
        assert got["clips"] is None
        assert [o["name"] for o in got["outputs"]] == ["final.mp4"]  # listed by its name alone
        assert got["final_current"] is True  # no usable export record: nothing is pending


@pytest.mark.parametrize("what", ["review", "clips", "analyze", "looping review"])
def test_metadata_written_through_a_link_outside_the_project_is_refused(
    client: TestClient, tmp_path: Path, what: str
) -> None:
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    src = str((folder / "a.mp4").resolve())
    seg = {"id": "h001", "source": src, "start": 1, "end": 4.5, "label": "good"}
    cache.save_json(p / "highlights.json", {"sources": [{"path": src, "duration": 10.0}]})
    cache.save_json(p / "ui" / "review.json", {"segments": [seg]})
    outside = tmp_path / "outside"
    outside.mkdir()
    link = {
        "review": "ui/review.json",
        "clips": "ui/exports.json",
        "analyze": "highlights.json",
        "looping review": "ui/review.json",
    }[what]
    target = outside / "target.json"
    cache.save_json(target, {"keep": True})
    (p / link).unlink(missing_ok=True)
    if what == "looping review":
        link_file(p / link, p / link)  # a link to itself: resolving it fails
    else:
        link_file(p / link, target)
    before = target.read_bytes()
    base = f"/api/projects/{pid}"

    if what in {"review", "looping review"}:
        res = client.put(f"{base}/review", json={"segments": [seg]})
    else:
        res = client.post(f"{base}/jobs", json={"action": what})
    assert res.status_code == 409, res.text
    assert client.get("/api/jobs").json() == {"jobs": []}
    assert target.read_bytes() == before
    assert sorted(f.name for f in outside.iterdir()) == ["target.json"]
    assert not list((p / "ui").glob("segments-*.json"))  # no half-made export either


def test_proxy_in_a_cache_folder_linked_outside_is_not_served_or_built(
    client: TestClient, hevc_project: tuple[str, Path, Path, Path]
) -> None:
    pid, sid, _, outside, key = media_link(client, hevc_project, "proxy")
    (outside / f"{key}.mp4").write_bytes(b"outside")
    src = client.get(f"/api/projects/{pid}").json()["sources"][0]
    assert client.post(f"/api/projects/{pid}/sources/{sid}/prepare").status_code == 409
    src = client.get(f"/api/projects/{pid}").json()["sources"][0]
    assert src["status"] == "needs-proxy"  # probed by the prepare request
    assert client.get(src["url"]).status_code == 409
    assert [f.name for f in outside.iterdir()] == [f"{key}.mp4"]


def test_frames_folder_linked_outside_is_not_used_and_nothing_there_is_deleted(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid, sid, _, outside, key = media_link(client, hevc_project, "frames")
    stale = [outside / f"{'0' * 16}-20-160.jpg", outside / f"{key}-30-160.123.part"]
    for f in stale:
        f.write_bytes(b"x")

    def never(*_a: object, **_k: object) -> list[bytes]:
        raise AssertionError("a still must not be built into a linked folder")

    patch_ui(monkeypatch, "grab_images", never)
    token = client.headers["X-Cutpoint-Token"]
    res = client.get(f"/api/projects/{pid}/sources/{sid}/frame?at=0.5&t={token}")
    assert res.status_code == 409
    assert sorted(f.name for f in outside.iterdir()) == sorted(f.name for f in stale)


@pytest.mark.parametrize("refused", [False, True], ids=["crash", "refused-write"])
def test_a_failing_post_processing_step_fails_the_job_with_a_clean_message(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, refused: bool
) -> None:
    def on_success() -> None:
        raise ui.bad("x", 409) if refused else RuntimeError("boom")

    FakeProc().fake_spawn(monkeypatch)
    jobs = ui.Jobs()
    with caplog.at_level(logging.ERROR):
        job = jobs.start("p", "clips", ["x"], on_success=on_success)
        assert job.done.wait(10)
    assert (job.status, job.error) == ("failed", "x" if refused else "boom")
    # a refusal is expected and plain; only a real crash logs its traceback
    assert any(r.exc_info for r in caplog.records if r.levelno >= logging.ERROR) is not refused


def test_a_command_line_project_without_a_ui_folder_can_be_reviewed_and_exported(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, seg = cli_project(tmp_path)
    assert not (p / "ui").exists()
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    assert client.get("/api/projects/cli").json()["review"]["segments"][0]["id"] == "h001"

    FakeProc().fake_spawn(monkeypatch)
    res = client.post("/api/projects/cli/jobs", json={"action": "clips"})
    assert res.status_code == 200, res.text
    assert client.app.state.jobs.get(res.json()["id"]).done.wait(10)
    assert len(list((p / "ui").glob("segments-*.json"))) == 1


def test_review_through_a_dangling_ui_link_is_refused_and_creates_nothing_outside(
    client: TestClient, tmp_path: Path
) -> None:
    p, seg = cli_project(tmp_path)
    outside = tmp_path / "outside" / "ui"
    outside.parent.mkdir()
    link_dir(p / "ui", outside)  # the target does not exist
    res = client.put("/api/projects/cli/review", json={"segments": [seg]})
    assert res.status_code == 409, res.text
    assert not outside.exists() and list(outside.parent.iterdir()) == []


def test_a_link_in_the_projects_folder_to_an_outside_analysis_is_not_a_project(
    client: TestClient, tmp_path: Path
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    cache.save_json(outside / "highlights.json", {"candidates": []})
    link_dir(ui.projects_dir() / "linked", outside)
    listed = [x["id"] for x in client.get("/api/projects").json()["projects"]]
    assert listed == [pid]
    assert client.get("/api/projects/linked").status_code == 404


def swap_ui(p: Path) -> None:
    """Replace ``p/ui`` with a link to a folder that does not exist (inside p)."""
    shutil.rmtree(p / "ui", ignore_errors=True)
    link_dir(p / "ui", p / "gone")


@pytest.mark.parametrize("via", ["review", "clips job"])
def test_a_dangling_ui_link_is_a_409_and_nothing_is_created(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, via: str
) -> None:
    p, seg = cli_project(tmp_path)
    if via == "review":
        swap_ui(p)
        res = client.put("/api/projects/cli/review", json={"segments": [seg]})
    else:  # the review was read, then ui was replaced: the job must not write through it
        assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200

        def cmd_after_ui_changes(*_a: str) -> list[str]:
            swap_ui(p)
            return [sys.executable, "-c", "pass"]

        patch_ui(monkeypatch, "cutpoint_cmd", cmd_after_ui_changes)
        res = client.post("/api/projects/cli/jobs", json={"action": "clips"})
    assert res.status_code == 409, res.text
    assert res.json()["detail"] == "cli/ui is not a folder; remove or rename it"
    assert not client.app.state.jobs.jobs
    assert sorted(f.name for f in p.rglob("*") if f.name != "ui") == ["highlights.json"]
    assert not (p / "gone").exists()


@pytest.mark.slow
@pytest.mark.parametrize("why", ["folder changed", "job running", "cannot start"])
def test_a_clips_job_that_does_not_start_leaves_no_segments_file(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, why: str
) -> None:
    p, seg = cli_project(tmp_path)
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    jobs = client.app.state.jobs
    busy = None
    if why == "folder changed":
        real_cmd = ui.cutpoint_cmd

        def cmd_while_the_folder_changes(*args: str) -> list[str]:
            config.update_config(lambda _d: ({"projects_dir": str(tmp_path / "other")}, ()))
            return real_cmd(*args)

        patch_ui(monkeypatch, "cutpoint_cmd", cmd_while_the_folder_changes)
    elif why == "job running":
        busy = jobs.start("x", "analyze", [sys.executable, "-c", "import time; time.sleep(60)"])
    else:

        def no_spawn(_args: list[str], **_kw: object) -> None:
            raise OSError("not found")

        patch_ui(monkeypatch, "spawn", no_spawn)
    try:
        res = client.post("/api/projects/cli/jobs", json={"action": "clips"})
        assert res.status_code == (500 if why == "cannot start" else 409), res.text
        assert not list(p.rglob("segments-*.json"))
    finally:
        if busy:
            jobs.cancel(busy.id)


def test_one_unreadable_entry_does_not_fail_the_project_list(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("bad", "good"):
        (ui.projects_dir() / name).mkdir(parents=True)
        cache.save_json(ui.projects_dir() / name / "highlights.json", {"candidates": []})
    real_is_dir = Path.is_dir

    def is_dir(self: Path) -> bool:
        if self.name == "bad":
            raise PermissionError("denied")
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    res = client.get("/api/projects")
    assert res.status_code == 200
    assert [x["id"] for x in res.json()["projects"]] == ["good"]


def test_a_proxy_linked_during_the_encode_fails_with_a_warning_and_leaves_nothing(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    _pid, folder, p, outside = hevc_project
    patch_ui(monkeypatch, "proxy_args", lambda _real, tmp, _n, **_kw: [str(tmp)])

    class Encoder:
        returncode = 0

        def __init__(self, args: list[str], **_kw: object) -> None:
            self.tmp = Path(next(a for a in args if a.endswith(".part.mp4")))

        stdout: ClassVar[list[str]] = []

        def wait(self) -> int:
            shutil.rmtree(p / "ui" / "proxy")  # ui/proxy becomes a link while the encode runs
            link_dir(p / "ui" / "proxy", outside)
            self.tmp.write_bytes(b"proxy")
            return self.returncode

    patch_ui(monkeypatch, "spawn", Encoder)
    with caplog.at_level(logging.INFO):
        media = client.app.state.media
        media.prepare(p, str(folder / "a.mp4"))
        media.pool.submit(int).result(30)  # the one worker has finished the build
    assert list(outside.iterdir()) == []
    assert not list(p.rglob("*.part.mp4"))
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert [r.levelno for r in warnings] == [logging.WARNING] and not warnings[0].exc_info
    assert "resolves outside the project folder" in warnings[0].getMessage()
    assert media.status(p, str(folder / "a.mp4"))[0] == "failed"


def test_assemble_picks_a_fresh_name_when_only_a_caption_file_is_left(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leftover ``final_ui_<stamp>.srt`` (no video) is a taken name: nothing may overwrite it."""
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    (p / "clips_ui_1").mkdir()
    cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": []})
    (p / "clips_ui_1" / "001_a.mp4").write_bytes(b"x")
    cache.save_json(p / "ui" / "exports.json", {"exports": [{"dir": "clips_ui_1"}]})
    (p / "final_ui_1.srt").write_text("keep me", encoding="utf-8")
    seen: list[list[str]] = []
    proc = FakeProc()
    proc.release.set()
    patch_ui(monkeypatch, "spawn", lambda args, **_kw: seen.append(args) or proc)
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "assemble"})
    assert res.status_code == 200, res.text
    assert client.app.state.jobs.get(res.json()["id"]).done.wait(10)
    assert seen[0][seen[0].index("--out") + 1] == str(p / "final_ui_1-2.mp4")
    assert (p / "final_ui_1.srt").read_text(encoding="utf-8") == "keep me"


def test_a_cancelled_job_reads_cancelling_until_the_process_has_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = FakeProc()
    killed = threading.Event()
    patch_ui(monkeypatch, "spawn", lambda args: proc)
    monkeypatch.setattr(procs, "kill_tree", lambda _p: killed.set())  # the exit comes later
    jobs = ui.Jobs()
    job = jobs.start("p", "analyze", ["x"])
    assert job.view()["status"] == "running"
    canceller = threading.Thread(target=jobs.cancel, args=(job.id,), daemon=True)
    canceller.start()
    try:
        assert killed.wait(10)
        assert job.view()["status"] == "cancelling"
        with pytest.raises(HTTPException) as busy:  # it still holds the server
            jobs.start("p", "clips", ["x"])
        assert busy.value.status_code == 409
    finally:
        proc.release.set()
        canceller.join(10)
    assert job.done.wait(10)
    deadline = time.time() + 10
    while job.view()["status"] != "cancelled" and time.time() < deadline:
        time.sleep(0.01)
    assert job.view()["status"] == "cancelled"


def test_cancel_returns_before_the_kill_finishes_and_a_second_cancel_adds_no_killer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = FakeProc()
    started = threading.Event()
    finish = threading.Event()
    calls: list[object] = []

    def slow_kill(p: object, *a: object) -> None:
        calls.append(p)
        started.set()
        finish.wait(10)
        proc.release.set()

    patch_ui(monkeypatch, "spawn", lambda args: proc)
    monkeypatch.setattr(procs, "kill_tree", slow_kill)
    jobs = ui.Jobs()
    job = jobs.start("p", "analyze", ["x"])
    try:
        t0 = time.monotonic()
        jobs.cancel(job.id)  # must not wait for the killer
        assert time.monotonic() - t0 < 5
        assert started.wait(10)
        assert not finish.is_set()
        assert job.view()["status"] == "cancelling"
        jobs.cancel(job.id)
        jobs.cancel(job.id)
        assert calls == [proc]
    finally:
        finish.set()
        proc.release.set()
    assert job.done.wait(10)
    assert len(calls) == 1


def listed_outputs(client: TestClient, pid: str) -> list[str]:
    return [o["name"] for o in client.get(f"/api/projects/{pid}").json()["outputs"]]


def test_every_final_in_the_project_folder_is_listed_newest_first(
    client: TestClient, tmp_path: Path
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    touch(p / "final.mp4", 300)  # from the command line: no export record
    touch(p / "clip-final.mp4", 100)  # from ``edit``, with timelines
    touch(p / "clip-final.fcpxml", 100)
    touch(p / "clip-final.otio", 100)
    touch(p / "clip-final.srt", 100)
    touch(p / "x-final_vertical.mp4", 200)
    touch(p / "x-final_vertical.otio", 200)
    for hidden in (".final.part.mp4", ".final.stage.mp4"):  # a run in progress or killed
        touch(p / hidden, 1)
    (p / "final_dir.mp4").mkdir()
    outside = tmp_path / "outside.mp4"
    touch(outside, 1)
    link_file(p / "final_out.mp4", outside)
    assert listed_outputs(client, pid) == [
        "clip-final.mp4", "clip-final.fcpxml", "clip-final.otio", "clip-final.srt",
        "x-final_vertical.mp4", "x-final_vertical.otio",
        "final.mp4",
    ]  # fmt: skip
    out = client.get(f"/api/projects/{pid}").json()["outputs"]
    srt = next(o for o in out if o["name"] == "clip-final.srt")
    res = client.get(f"{srt['url']}&download=1")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/plain")
    assert "attachment" in res.headers["content-disposition"]


def test_a_command_line_final_can_be_downloaded_and_revealed(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Path] = []
    patch_ui(monkeypatch, "folder_opener", lambda: opened.append)
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    touch(p / "final.mp4", 10)
    out = client.get(f"/api/projects/{pid}").json()["outputs"]
    assert client.get(out[0]["url"]).status_code == 200
    res = client.post(f"/api/projects/{pid}/reveal", params={"target": "outputs"})
    assert res.status_code == 200 and opened == [p]


@pytest.mark.parametrize(
    ("final", "listed", "current"),
    [
        ("none", ["final.mp4", "final_ui_1.mp4"], True),  # no export record
        ("final_ui_1.mp4", ["final_ui_1.mp4", "final.mp4"], True),  # its own, though older
        (None, ["final.mp4", "final_ui_1.mp4"], False),  # not assembled yet: the earlier leads
    ],
    ids=["no-record", "own-final-first", "pending-export"],
)
def test_final_current_says_whether_the_leading_final_is_the_exports_own(
    client: TestClient, tmp_path: Path, final: str | None, listed: list[str], current: bool
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    touch(p / "final_ui_1.mp4", 500)
    touch(p / "final.mp4", 5)
    if final != "none":
        record_export(p, final)
    got = client.get(f"/api/projects/{pid}").json()
    assert [o["name"] for o in got["outputs"]] == listed  # no duplicates either
    assert got["final_current"] is current


def test_new_project_that_cannot_save_its_meta_leaves_no_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #194: the failed request removes the project and ``ui/`` (and the projects folder
    only when this call made it)."""
    root = ui.projects_dir()

    def full_disk(*_a: object, **_k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(cache, "save_json", full_disk)
    for pre_existing in (False, True):
        if pre_existing:
            root.mkdir(parents=True)
        with pytest.raises(OSError, match="disk full"):
            ui.new_project("Trip", str(tmp_path), "funny")
        assert root.is_dir() is pre_existing
        if pre_existing:
            assert list(root.iterdir()) == []


def test_failed_proxy_build_on_a_command_line_project_removes_the_ui_folders_it_made(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #194: a CLI project has no ``ui/``; a failed build removes ``ui/proxy`` and ``ui/``
    it created."""
    patch_ui(monkeypatch, "probe", lambda _path: media_info(vcodec="hevc", duration=10.0))
    src = tmp_path / "footage" / "a.mp4"
    src.parent.mkdir()
    src.write_bytes(b"x")
    p = ui.projects_dir() / "cli-proj"
    p.mkdir(parents=True)
    cache.save_json(p / "highlights.json", {"settings": {"inputs": [str(src.parent)]}})

    class Failing:
        returncode = 1

        def __init__(self, _args: list[str], **_kw: object) -> None:
            pass

        stdout: ClassVar[list[str]] = ["encoder failed\n"]

        def wait(self) -> int:
            return self.returncode

    patch_ui(monkeypatch, "spawn", Failing)
    media = client.app.state.media
    media.prepare(p, str(src))
    media.pool.submit(int).result(30)  # the one worker has finished the build
    assert media.status(p, str(src))[0] == "failed"
    assert not (p / "ui").exists()
