"""Analyze jobs of ``cutpoint ui``: media Range, the chosen source's proxy, analyze options."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("httpx")

from fastapi import HTTPException
from fastapi.testclient import TestClient

from conftest import needs_ffmpeg, set_config, write
from cutpoint import cache, cli, config, procs, ui
from ui_helpers import (
    PROC_WAIT_S,
    JobSpy,
    _after,
    _cli_args,
    _pid_alive,
    cli_project,
    ffmpeg,
    make_project,
    preset_job,
    wait_job,
    wait_until,
)

# the encoder probe and the keyring stay stubbed in every test (ui_helpers fixtures)
pytestmark = pytest.mark.usefixtures("no_nvenc", "no_keyring")


def job_failure(job: dict) -> str:
    """What a job that did not complete said: its status, error and last log lines."""
    logs = "\n".join(job.get("logs", [])[-30:])
    return (
        f"job {job['status']}: {job.get('error')} ({job.get('error_code')})\n"
        f"steps: {job.get('progress')}\nlog:\n{logs}"
    )


@pytest.mark.slow
@needs_ffmpeg
def test_proxy_is_built_only_for_the_chosen_source_and_keeps_every_audio_track(
    client: TestClient, tmp_path: Path
) -> None:
    """Opening a project builds nothing; preparing one source mixes all its audio tracks."""
    multi, other = tmp_path / "a.mp4", tmp_path / "b.mkv"
    ffmpeg(
        *("-f", "lavfi", "-i", "testsrc=d=3:s=160x120:r=25"),
        *("-f", "lavfi", "-i", "sine=f=440:d=3", "-f", "lavfi", "-i", "sine=f=880:d=3"),
        *("-map", "0:v", "-map", "1:a", "-map", "2:a", "-c:v", "libx264", "-pix_fmt", "yuv420p"),
        *("-c:a", "aac", str(multi)),
    )  # fmt: skip
    ffmpeg("-f", "lavfi", "-i", "testsrc=d=1:s=160x120:r=25", "-c:v", "mpeg4", str(other))
    pid, _ = make_project(
        client, tmp_path, {"a.mp4": multi.read_bytes(), "b.mkv": other.read_bytes()}
    )

    def sources() -> dict[str, dict]:
        return {s["name"]: s for s in client.get(f"/api/projects/{pid}").json()["sources"]}

    opened = sources()
    assert (opened["a.mp4"]["status"], opened["b.mkv"]["status"]) == ("unknown",) * 2
    assert not (ui.project_dir(pid) / ui.UI_DIR / "proxy").exists()  # nothing was queued
    assert client.get(opened["a.mp4"]["url"]).status_code == 409

    client.post(f"/api/projects/{pid}/sources/{opened['a.mp4']['id']}/prepare")
    # The proxy encodes at low priority (below normal on Windows, nice 10 elsewhere): on a loaded
    # machine its 3 s encode can wait minutes for the CPU, so the deadline is wait_job's.
    deadline = time.monotonic() + 600
    while (status := sources()["a.mp4"]["status"]) == "preparing":
        assert time.monotonic() < deadline, "the proxy was still preparing after 600 s"
        time.sleep(0.3)
    after = sources()
    assert (after["a.mp4"]["status"], after["b.mkv"]["status"]) == ("ready", "unknown"), status

    url = after["a.mp4"]["url"]
    assert client.get(url, headers={"Range": "bytes=0-99"}).status_code == 206
    played = tmp_path / "played.mp4"
    played.write_bytes(client.get(url).content)
    # #324's whole-source scrubbing relies on the proxy covering the full source, not a moment.
    duration = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(played)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    assert float(duration) == pytest.approx(3.0, abs=0.15)
    pcm = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(played),
            "-vn",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "8000",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    spectrum = np.abs(np.fft.rfft(np.frombuffer(pcm, dtype=np.int16).astype(float)))
    freqs = np.fft.rfftfreq(len(pcm) // 2, 1 / 8000)
    for tone in (440, 880):
        assert spectrum[abs(freqs - tone) < 5].max() > 20 * np.median(spectrum), tone


@pytest.mark.slow
@needs_ffmpeg
def test_reanalyze_keeps_recorded_excludes_and_recursion(
    client: TestClient, tmp_path: Path
) -> None:
    """A command-line project re-analyzed from the UI keeps its subfolders and its exclusions."""
    clip = tmp_path / "clip.mp4"  # silent, with scene cuts: nothing to transcribe or download
    ffmpeg(
        *("-f", "lavfi", "-i", "color=c=white:s=160x120:r=25:d=2"),
        *("-f", "lavfi", "-i", "color=c=black:s=160x120:r=25:d=2"),
        *("-f", "lavfi", "-i", "color=c=white:s=160x120:r=25:d=2"),
        *("-filter_complex", "[0][1][2]concat=n=3:v=1:a=0", "-c:v", "libx264"),
        *("-pix_fmt", "yuv420p", str(clip)),
    )  # fmt: skip
    footage = tmp_path / "footage"
    (footage / "sub").mkdir(parents=True)
    for rel in ("keep.mp4", "DVR_skip.mp4", "sub/nested.mp4", "sub/DVR_nested.mp4"):
        shutil.copy(clip, footage / rel)
    p = ui.projects_dir() / "cli-made"
    p.mkdir(parents=True)
    settings = {"inputs": [str(footage.resolve())], "exclude": ["*DVR*"], "recursive": True}
    # Not a game or action style: those run HUD OCR with region discovery (about 80 s a source)
    # or CLIP keyframes, which this source-set check does not need.
    style = "commentary"
    cache.save_json(p / "highlights.json", {"style": style, "settings": settings, "sources": []})

    assert client.post("/api/projects/cli-made/jobs", json={"action": "analyze"}).status_code == 200
    job = wait_job(client)
    assert job["status"] == "completed", job_failure(job)
    names = {s["name"] for s in client.get("/api/projects/cli-made").json()["sources"]}
    assert names == {"keep.mp4", "nested.mp4"}


@pytest.mark.parametrize(("recover", "passed"), [(True, True), (False, False), ("yes", False)])
def test_reanalyze_keeps_a_recorded_recover(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    recover: object,
    passed: bool,
) -> None:
    """Analyze again on a project made with --recover keeps the parts past a read error."""
    p, _ = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl["style"] = "funny"
    hl["settings"]["recover"] = recover
    cache.save_json(p / "highlights.json", hl)
    captured = JobSpy(monkeypatch, check=False).args
    res = client.post("/api/projects/cli/jobs", json={"action": "analyze"})
    assert res.status_code == 200, res.text
    assert ("--recover" in captured[0]) is passed
    cli.build_parser().parse_args(captured[0][3:])


ANALYZE_DEFAULTS = {
    "model": None,
    "language": None,
    "max_candidates": None,
    "no_scenes": False,
    "skip": None,
    "must_include_crash": None,
    "recover": False,
}

RECORDED = {  # an analysis run with every Analyze option but --skip (not recorded)
    "whisper_model": "large-v3",
    "whisper_device": "cpu",
    "language": "de",
    "max_candidates": 30,
    "scenes": False,
    "recover": True,
}


@pytest.mark.parametrize(
    ("settings", "body", "parsed"),
    [
        (
            {},
            {
                "model": " turbo ",
                "language": "en",
                "max_candidates": 12,
                "no_scenes": True,
                "skip": ["crashes", "reactions"],
                "must_include_crash": 0.25,
                "recover": True,
            },
            {
                "model": "turbo",
                "language": "en",
                "max_candidates": 12,
                "no_scenes": True,
                "skip": ["reactions", "crashes"],
                "must_include_crash": 0.25,
                "recover": True,
            },
        ),
        # Analyze again from a request without them: what the analysis recorded
        (
            RECORDED,
            {},
            {
                "model": "large-v3",
                "language": "de",
                "max_candidates": 30,
                "no_scenes": True,
                "recover": True,
            },
        ),
        # the page cleared them: the CLI defaults (a recorded --recover is dropped too)
        (
            RECORDED,
            {
                "model": "",
                "language": None,
                "max_candidates": None,
                "no_scenes": False,
                "recover": False,
                "skip": [],
            },
            {},
        ),
        # each refused, nothing started
        ({}, [{"skip": ["vision"]}, {"model": "../x"}], None),
    ],
    ids=["set", "recorded", "cleared", "invalid"],
)
def test_analyze_options_reach_the_analyze_command_line_unchanged(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: dict,
    body: dict | list[dict],
    parsed: dict | None,
) -> None:
    p, _ = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl["style"] = "funny"
    hl["settings"].update(settings)
    cache.save_json(p / "highlights.json", hl)
    if settings:  # the page shows what the analysis used
        assert client.get("/api/projects/cli").json()["analyze_options"] == {
            **ANALYZE_DEFAULTS, "model": "large-v3", "language": "de", "max_candidates": 30,
            "no_scenes": True, "recover": True, "skip": [], "vision_detect_all": False,
        }  # fmt: skip
    captured = JobSpy(monkeypatch, check=False).args
    if isinstance(body, list):
        for bad_body, message in zip(
            body, ("skip must be a list of", "model must be a whisper"), strict=True
        ):
            res = client.post("/api/projects/cli/jobs", json={"action": "analyze", **bad_body})
            assert res.status_code == 400 and message in res.json()["detail"]
        assert not captured
        return
    res = client.post("/api/projects/cli/jobs", json={"action": "analyze", **body})
    assert res.status_code == 200, res.text
    a = cli.build_parser().parse_args(_cli_args(captured[0]))
    assert {k: getattr(a, k) for k in ANALYZE_DEFAULTS} == {**ANALYZE_DEFAULTS, **parsed}


def _auto_project(tmp_path: Path, **settings: object) -> Path:
    """A command-line project analyzed with ``--style auto`` (it chose gaming)."""
    det = {"style": "gaming", "category": "gameplay"}
    settings = {"style_requested": "auto", **settings}
    return cli_project(tmp_path, settings=settings, style="gaming", detection=det)[0]


@pytest.mark.parametrize(
    ("settings", "config", "want"),
    [
        ({"vision_detect_all": True}, {}, True),  # recorded by the analysis
        ({}, {"vision_detect_all": True}, False),  # an auto analysis without it: off
    ],
)
def test_analyze_options_report_vision_detect_all(
    client: TestClient, tmp_path: Path, settings: dict, config: dict, want: bool
) -> None:
    _auto_project(tmp_path, **settings)
    set_config(ui.paths.home(), **config)
    opts = client.get("/api/projects/cli").json()["analyze_options"]
    assert opts["vision_detect_all"] is want


def test_analyze_options_vision_detect_all_defaults_to_the_config_before_an_auto_analysis(
    client: TestClient, tmp_path: Path
) -> None:
    cli_project(tmp_path)  # no analysis style recorded
    assert client.get("/api/projects/cli").json()["analyze_options"]["vision_detect_all"] is False
    set_config(ui.paths.home(), vision_detect_all=True)
    assert client.get("/api/projects/cli").json()["analyze_options"]["vision_detect_all"] is True
    set_config(ui.paths.home(), vision_detect_all="yes")  # invalid: the helper does not raise
    assert ui._vision_detect_all(None) is False


@pytest.mark.parametrize(
    ("recorded", "body", "flag"),
    [
        (False, {"vision_detect_all": True}, "--vision-detect-all"),
        (True, {"vision_detect_all": False}, "--no-vision-detect-all"),
        (True, {}, "--vision-detect-all"),  # Analyze again repeats the analysis's
        (False, {}, "--no-vision-detect-all"),  # explicit, even off: config.json may differ
        (False, {"vision_detect_all": None}, None),  # the page sent none: the CLI's default
    ],
)
def test_an_auto_analyze_job_sends_vision_detect_all_explicitly(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    recorded: bool,
    body: dict,
    flag: str | None,
) -> None:
    _auto_project(tmp_path, **({"vision_detect_all": True} if recorded else {}))
    captured = JobSpy(monkeypatch, check=False).args
    res = client.post("/api/projects/cli/jobs", json={"action": "analyze", **body})
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    assert _after(args, "--style") == "auto"
    got = [a for a in args if a in ("--vision-detect-all", "--no-vision-detect-all")]
    assert got == ([flag] if flag else [])


@pytest.mark.parametrize("style", ["funny", "gaming"])
def test_other_styles_send_no_vision_detect_all(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, style: str
) -> None:
    p, _ = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl["style"] = style
    hl["settings"]["vision_detect_all"] = True
    cache.save_json(p / "highlights.json", hl)
    captured = JobSpy(monkeypatch, check=False).args
    body = {"action": "analyze", "vision_detect_all": True}
    assert client.post("/api/projects/cli/jobs", json=body).status_code == 200
    assert not [a for a in captured[0] if a.endswith("-vision-detect-all")]


def test_a_non_boolean_vision_detect_all_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _auto_project(tmp_path)
    captured = JobSpy(monkeypatch, check=False).args
    for bad_value in ("yes", 1, [True]):
        body = {"action": "analyze", "vision_detect_all": bad_value}
        res = client.post("/api/projects/cli/jobs", json=body)
        assert res.status_code == 400
        assert "vision_detect_all must be true or false" in res.json()["detail"]
    assert not captured


@pytest.mark.parametrize(("recorded", "flag"), [(True, "--vision-detect-all"), (False, None)])
def test_one_click_edit_repeats_an_auto_analyses_vision_detect_all(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    recorded: bool,
    flag: str | None,
) -> None:
    _auto_project(tmp_path, **({"vision_detect_all": True} if recorded else {}))
    captured = JobSpy(monkeypatch, check=False).args
    res = client.post("/api/projects/cli/jobs", json={"action": "edit", "kind": "clips"})
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    assert args[0] == "edit"
    got = [a for a in args if a.endswith("-vision-detect-all")]
    assert got == [flag or "--no-vision-detect-all"]


def test_the_analyze_skip_is_kept_for_analyze_again_and_one_click_edit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """highlights.json does not record --skip: the UI keeps it for that analysis only, so
    edit (which analyzes again) leaves the same steps out, and a command-line re-run drops it."""
    p, _ = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl.update(style="funny", created="2026-10-01T10:00:00")
    cache.save_json(p / "highlights.json", hl)
    spy = JobSpy(monkeypatch)
    captured, done = spy.args, spy.on_success
    body = {"action": "analyze", "skip": ["crashes", "keyframes"]}
    assert client.post("/api/projects/cli/jobs", json=body).status_code == 200
    assert _cli_args(captured[0])[-3:] == ["--skip", "keyframes", "crashes"]
    hl["created"] = "2026-10-01T11:00:00"  # what that analysis wrote
    cache.save_json(p / "highlights.json", hl)
    done[0]()
    view = client.get("/api/projects/cli").json()
    assert view["analyze_options"]["skip"] == ["keyframes", "crashes"]
    assert view["preset_problem"] is None  # edit takes --skip
    assert (
        client.post("/api/projects/cli/jobs", json={"action": "edit", "kind": "clips"}).status_code
        == 200
    )
    assert _cli_args(captured[1])[-3:] == ["--skip", "keyframes", "crashes"]

    hl["created"] = "2026-10-02T09:00:00"  # analyzed again on the command line
    cache.save_json(p / "highlights.json", hl)
    assert client.get("/api/projects/cli").json()["analyze_options"]["skip"] == []
    assert client.post("/api/projects/cli/jobs", json={"action": "analyze"}).status_code == 200
    assert "--skip" not in captured[2]


def test_a_ski_analysis_without_no_scenes_keeps_presets_and_scene_detection_on(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: ski never detects scenes, so its analysis records "scenes": false without
    --no-scenes; that turned the presets off with the --no-scenes reason."""
    meta = {"style": "action-cam", "source": str(tmp_path / "clip.mp4")}
    settings = {"scenes": False, "must_include_crash": 0.6}  # the action-cam style's own
    res, captured, _, _ = preset_job(
        client,
        monkeypatch,
        tmp_path,
        {"kind": "clips"},
        settings=settings,
        meta=meta,
        style="action-cam",
    )
    assert res.status_code == 200, res.text
    _cli_args(captured[0])
    view = client.get("/api/projects/cli").json()
    assert view["preset_problem"] is None and view["must_include_crash_default"] == 0.6
    opts = view["analyze_options"]
    assert (opts["no_scenes"], opts["must_include_crash"]) == (False, None)


def test_an_auto_project_detected_as_action_cam_keeps_presets_and_scene_detection_on(
    client: TestClient, tmp_path: Path
) -> None:
    """An Auto project's detected style (not "auto") decides whether "scenes": false means
    --no-scenes."""
    p, _ = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl["style"] = "action-cam"
    hl["settings"]["scenes"] = False
    cache.save_json(p / "highlights.json", hl)
    write(p / "ui" / "meta.json", json.dumps({"style": "auto", "source": str(tmp_path / "c.mp4")}))
    view = client.get("/api/projects/cli").json()
    assert view["preset_problem"] is None
    assert view["analyze_options"]["no_scenes"] is False


@pytest.mark.parametrize(
    ("saved", "flags"),
    [
        (
            {"vision_top": 5, "ollama_url": "http://h:1", "ollama_model": "m"},
            ["--vision-top=5", "--ollama-url=http://h:1", "--ollama-model=m"],
        ),
        ({"vision_top": True, "ollama_url": "", "ollama_model": ""}, []),
        ({"vision_top": 0}, []),
    ],
    ids=["kept", "invalid-true-and-empty", "zero"],
)
def test_reanalyze_passes_recorded_vision_and_ollama_options(
    client: TestClient,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    saved: dict,
    flags: list[str],
) -> None:
    p = ui.projects_dir() / "cli-made"
    p.mkdir(parents=True)
    footage = tmp_path / "clip.mp4"
    footage.write_bytes(b"x")
    settings = {"inputs": [str(footage)], **saved}
    cache.save_json(p / "highlights.json", {"style": "gaming", "settings": settings, "sources": []})
    captured = JobSpy(monkeypatch, check=False).args
    assert client.post("/api/projects/cli-made/jobs", json={"action": "analyze"}).status_code == 200
    args = captured[0]
    assert [a for a in args if a.startswith(("--vision-top", "--ollama"))] == flags


@pytest.mark.parametrize(
    ("body", "saved", "cfg", "vision"),
    [
        ({}, {}, None, None),  # no key in config.json: no flag, the CLI asks unsure sources
        ({}, {}, False, False),
        ({}, {}, True, True),  # Analyze again: the Settings value
        ({}, {"vision_requested": True}, None, None),  # not what the last run asked
        ({}, {"vision_requested": False}, True, True),
        ({"vision": True}, {}, False, False),  # no per-run choice: a body flag is ignored
        ({"vision": False}, {}, True, True),
    ],
)
def test_analyze_job_passes_the_settings_vision_explicitly(
    client: TestClient,
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    body: dict,
    saved: dict,
    cfg: bool | None,
    vision: bool | None,
) -> None:
    """--vision or --no-vision as Settings says when the job starts; neither when config.json
    has no ``vision`` key."""
    if cfg is not None:
        set_config(home, vision=cfg)
    p = ui.projects_dir() / "cli-made"
    p.mkdir(parents=True)
    footage = tmp_path / "clip.mp4"
    footage.write_bytes(b"x")
    settings = {"inputs": [str(footage)], **saved}
    cache.save_json(p / "highlights.json", {"style": "gaming", "settings": settings, "sources": []})
    captured = JobSpy(monkeypatch, check=False).args
    res = client.post("/api/projects/cli-made/jobs", json={"action": "analyze", **body})
    assert res.status_code == 200
    assert ("--vision" in captured[0]) is (vision is True)
    assert ("--no-vision" in captured[0]) is (vision is False)


def test_put_settings_saves_visual_index_and_jobs_pass_it_explicitly(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """visual_index (#798): saved by Settings, shown by GET, and --visual-index or
    --no-visual-index on every analysis and preset the page starts."""
    assert client.get("/api/settings").json()["visual_index"] is False
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    captured = JobSpy(monkeypatch, check=False).args
    for value in (True, False):
        res = client.put("/api/settings", json={"visual_index": value})
        assert res.status_code == 200, res.text
        assert res.json()["visual_index"] is value
        assert client.get("/api/settings").json()["visual_index"] is value
        assert config.visual_index_default() is value  # the CLI default too
        assert (
            client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"}).status_code == 200
        )
        assert ("--visual-index" in captured[-1]) is value
        assert ("--no-visual-index" in captured[-1]) is not value
    for bad_value in ("yes", 1, None):
        assert client.put("/api/settings", json={"visual_index": bad_value}).status_code == 400


def test_analyze_follows_settings_saved_just_before(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Saving Settings changes the next analysis at once, without a reload or restart."""
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    captured = JobSpy(monkeypatch, check=False).args
    for value in (True, False, True):
        assert client.put("/api/settings", json={"vision": value}).status_code == 200
        res = client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"})
        assert res.status_code == 200
    assert [("--vision" in a, "--no-vision" in a) for a in captured] == [
        (True, False),
        (False, False),  # unticked: the key is gone, so neither flag
        (True, False),
    ]


@pytest.mark.slow
def test_one_job_at_a_time_and_cancel_kills_the_process_group(tmp_path: Path) -> None:
    script = (
        "import subprocess, sys, time\n"
        # longer than PROC_WAIT_S: a child left alive cannot pass the death check by exiting
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        "print('child', c.pid, flush=True)\n"
        "time.sleep(600)\n"
    )
    jobs = ui.Jobs()
    job = jobs.start("p", "analyze", [sys.executable, "-c", script])
    child = 0
    try:
        # two interpreter starts can be slow under a loaded suite
        wait_until(lambda: job.logs or job.status != "running", PROC_WAIT_S, "the child's pid line")
        assert job.logs, (job.status, job.error)
        child = int(job.logs[0].split()[1])
        with pytest.raises(HTTPException) as busy:
            jobs.start("p", "clips", [sys.executable, "-c", "pass"])
        assert busy.value.status_code == 409
        cancelled = jobs.cancel(job.id)
        # cancel returns at once; the killer thread and the watcher finish in the background
        wait_until(lambda: cancelled.status == "cancelled", PROC_WAIT_S, "the job to be cancelled")
        assert job.proc is not None and job.proc.poll() is not None
        if sys.platform.startswith("linux"):  # a reaped-or-zombie child counts as dead
            wait_until(lambda: _linux_dead(child), PROC_WAIT_S, f"the child {child} to die")
    finally:  # what a failed cancel left running
        if child and sys.platform.startswith("linux") and not _linux_dead(child):
            os.kill(child, signal.SIGKILL)
        elif child and os.name == "nt" and _pid_alive(child):
            subprocess.run(["taskkill", "/PID", str(child), "/T", "/F"], capture_output=True)
        if job.proc is not None and job.proc.poll() is None:
            procs.kill_tree(job.proc, 0.0)
    failing = jobs.start(
        "p", "analyze", [sys.executable, "-c", "print('12:00:00 ERROR boom')\nraise SystemExit(1)"]
    )
    assert failing.done.wait(20)
    assert (failing.status, failing.error) == ("failed", "boom")


def _linux_dead(pid: int) -> bool:
    """``pid`` is gone, or a zombie (dead, not yet reaped)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return True
    return stat.rsplit(")", 1)[1].split()[0] == "Z"
