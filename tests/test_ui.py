"""Backend of ``cutpoint ui``: trust boundaries, review validation, media Range, job semantics."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi import HTTPException
from fastapi.testclient import TestClient

from conftest import link_dir, link_file, media_info, needs_ffmpeg, remap, set_config, write
from cutpoint import cache, capabilities, cli, config, edit, encode, ffmpeg_utils, ui
from cutpoint.assistant import session as ai_session
from cutpoint.pano import project as pano_project
from cutpoint.utils import find_videos
from pano_helpers import x3clip

PORT = 8765


def fake_start(captured: list[list[str]]):
    """A ``Jobs.start`` stand-in that records the CLI args of each job and reports "running"."""

    def start(self, project_id, action, args, *rest, **kw):
        captured.append(args)
        return SimpleNamespace(view=lambda: {"status": "running"})

    return start


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = ui.create_app(PORT)
    # ``with`` runs the app's shutdown (jobs.shutdown kills any job still running)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        c.headers["X-Cutpoint-Token"] = app.state.token
        yield c


@pytest.fixture(autouse=True)
def no_nvenc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proxies encode with libx264 unless a test says otherwise: no GPU probe in tests."""
    monkeypatch.setattr(ui, "video_encoder", lambda: ui.VideoEncoder("cpu"))


def make_project(client: TestClient, tmp_path: Path, files: dict[str, bytes]) -> tuple[str, Path]:
    folder = tmp_path / "footage"
    folder.mkdir()
    for name, data in files.items():
        (folder / name).write_bytes(data)
    res = client.post("/api/projects", json={"name": "My Trip", "source": str(folder)})
    assert res.status_code == 200, res.text
    return res.json()["id"], folder


X3_00 = "VID_20260101_100000_00_001.insv"
X3_10 = "VID_20260101_100000_10_001.insv"
X3_LRV = "LRV_20260101_100000_11_001.lrv"


@pytest.mark.parametrize(
    ("files", "folders", "pick", "status", "expect"),
    [
        (
            {"a.insv": b"", "notes.txt": b""},
            ["sub"],
            None,
            400,
            ["no supported videos found (found 1 .insv, 1 .txt)", "camera names"],
        ),
        ({X3_00: b"", X3_LRV: b"", "a.mp4": b""}, [], None, 400, ["mixes 360 footage"]),
        ({X3_10: b"", X3_LRV: b""}, [], None, 400, ["needs both files", "_00_.insv"]),
    ],
    ids=[
        "unnamed-insv",
        "mixed",
        "lrv-without-00",
    ],
)
def test_create_project_recognises_360_sources_and_refuses_them_clearly(
    client: TestClient,
    tmp_path: Path,
    files: dict[str, bytes],
    folders: list[str],
    pick: str | None,
    status: int,
    expect: list[str],
) -> None:
    """Regression for #321: a folder of .insv files said "no videos", not what was wrong."""
    folder = tmp_path / "footage"
    folder.mkdir()
    for name, data in files.items():
        (folder / name).write_bytes(data)
    for name in folders:
        (folder / name).mkdir()
    source = folder / pick if pick else folder
    res = client.post("/api/projects", json={"name": "Trip", "source": str(source)})
    assert res.status_code == status, res.text
    for part in expect:
        assert part in res.json()["detail"]
    if status != 200:
        assert client.get("/api/projects").json() == {"projects": []}


X3_CLIPS = {X3_00: b"", X3_10: b"", X3_LRV: b""}


@pytest.mark.parametrize(
    ("pick", "style", "meta"),
    [
        (None, None, {"kind": "360", "style": "ski"}),
        (X3_00, None, {"kind": "360", "style": "ski", "clips": ["001"]}),
        (None, "ski", {"kind": "360", "style": "ski"}),
    ],
    ids=["folder-flat-style-becomes-pano-default", "file-keeps-its-clip", "ski-style"],
)
def test_create_project_makes_a_360_project_from_x3_footage(
    client: TestClient, tmp_path: Path, pick: str | None, style: str | None, meta: dict
) -> None:
    """#320: X3 sources become 360 projects; ``clips`` only for one picked file."""
    folder = tmp_path / "footage"
    folder.mkdir()
    for name in X3_CLIPS:
        (folder / name).write_bytes(b"")
    body = {"name": "Trip", "source": str(folder / pick if pick else folder)}
    if style:
        body["style"] = style
    res = client.post("/api/projects", json=body)
    assert res.status_code == 200, res.text
    saved = ui.read_meta(ui.project_dir(res.json()["id"]))
    assert {k: saved.get(k) for k in meta} == meta
    assert ("clips" in saved) == ("clips" in meta)


def test_create_project_refuses_a_ski_style_on_flat_footage(
    client: TestClient, tmp_path: Path
) -> None:
    folder = tmp_path / "footage"
    folder.mkdir()
    (folder / "a.mp4").write_bytes(b"x")
    res = client.post("/api/projects", json={"name": "T", "source": str(folder), "style": "ski"})
    assert res.status_code == 400 and "unknown style" in res.json()["detail"]
    assert client.get("/api/projects").json() == {"projects": []}


def test_360_project_lists_one_lrv_per_usable_clip_before_analysis(
    client: TestClient, tmp_path: Path
) -> None:
    folder = tmp_path / "footage"
    folder.mkdir()
    for name in (*X3_CLIPS, "VID_20260101_110000_10_002.insv", "LRV_20260101_110000_11_002.lrv"):
        (folder / name).write_bytes(b"")  # clip 002 has no _00_ lens: not usable
    res = client.post("/api/projects", json={"name": "T", "source": str(folder)})
    p = ui.project_dir(res.json()["id"])
    assert [Path(e["path"]).name for e in ui.source_entries(p, None)] == [X3_LRV]


def test_list_projects_shows_ui_360_projects_but_hides_command_line_ones(
    client: TestClient, tmp_path: Path
) -> None:
    folder = tmp_path / "footage"
    folder.mkdir()
    for name in X3_CLIPS:
        (folder / name).write_bytes(b"")
    pid = client.post("/api/projects", json={"name": "T", "source": str(folder)}).json()["id"]
    cache.save_json(ui.project_dir(pid) / "pano.json", {"clips": {}})  # as after the scan
    cli_only = ui.projects_dir() / "cli-360"
    cli_only.mkdir()
    cache.save_json(cli_only / "pano.json", {"clips": {}})
    assert [p["id"] for p in client.get("/api/projects").json()["projects"]] == [pid]


@pytest.mark.parametrize("only", [None, "001"], ids=["folder", "file"])
def test_360_analyze_job_runs_the_360_command_with_a_plan(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, only: str | None
) -> None:
    folder = tmp_path / "footage"
    folder.mkdir()
    for name in X3_CLIPS:
        (folder / name).write_bytes(b"")
    source = folder / X3_00 if only else folder
    pid = client.post("/api/projects", json={"name": "T", "source": str(source)}).json()["id"]
    seen: list[list[str]] = []
    monkeypatch.setattr(ui, "cutpoint_cmd", lambda *a: ["cutpoint", *a])
    monkeypatch.setattr(ui.Jobs, "start", fake_start(seen))
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"})
    assert res.status_code == 200, res.text
    p = ui.project_dir(pid)
    want = ["cutpoint", "360", "analyze", str(p), "--scan", str(folder), "--style", "ski"]
    assert seen == [want + (["--only", only] if only else [])]
    assert "--vision" not in seen[0] and "--no-vision" not in seen[0]


def test_360_analyze_job_view_carries_the_step_plan(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "footage"
    folder.mkdir()
    for name in X3_CLIPS:
        (folder / name).write_bytes(b"")
    pid = client.post("/api/projects", json={"name": "T", "source": str(folder)}).json()["id"]
    # No real interpreter: spawn starts jobs at below-normal priority, and on a busy machine
    # one took up to ~15 s to exit, close to the old 20 s wait. The plan comes from the real Job.
    FakeProc().fake_spawn(monkeypatch)
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"})
    assert res.status_code == 200, res.text
    assert [s["step"] for s in res.json()["progress"]["plan"]] == [
        "scan", "views", "analyze", "events",
    ]  # fmt: skip
    assert client.app.state.jobs.get(res.json()["id"]).done.wait(10)


class StitchExport:
    """A 360 project with one recorded clip and one moment, and fake job processes.

    ``spawn`` records each command; the ``360 stitch`` one exits ``stitch_code`` (after
    ``hold`` is set, when given), the ``clips`` one writes its manifest. ``sdk_status`` and the
    free space are mocked: the real SDK never runs.
    """

    def __init__(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        sdk: dict | None = None,
        free: float | None = 500.0,
        nvidia: bool = True,
    ) -> None:
        folder = tmp_path / "footage"
        folder.mkdir()
        for name in X3_CLIPS:
            (folder / name).write_bytes(b"")
        res = client.post("/api/projects", json={"name": "T", "source": str(folder)})
        self.client, self.pid = client, res.json()["id"]
        self.p = ui.project_dir(self.pid)
        lrv = str(folder / X3_LRV)
        self.clip = x3clip(lrv=lrv, insv=[str(folder / X3_00), str(folder / X3_10)], duration=60)
        pano_project.save(self.p, [self.clip], [folder])
        cand = {"id": "h001", "source": lrv, "start": 1.0, "end": 5.0, "score": 80, "label": "a"}
        hl = {"sources": [{"path": lrv, "duration": 60.0}], "candidates": [cand], "style": "ski"}
        cache.save_json(self.p / "highlights.json", hl)
        self.segment = {k: cand[k] for k in ("id", "source", "start", "end", "label")}
        self.seen: list[list[str]] = []
        self.stitch_code = 0
        self.stitch_lines = ["10:00:00 INFO clip 001: ok"]
        self.clips_code = 0
        self.clips_lines: list[str] = []
        self.hold: threading.Event | None = None
        self.stitching = threading.Event()
        monkeypatch.setattr(ui, "cutpoint_cmd", lambda *a: ["cutpoint", *a])
        monkeypatch.setattr(ui, "spawn", self.spawn)
        monkeypatch.setattr(ui, "kill_tree", lambda proc, *a, **k: proc.release.set())
        status = {"found": True, "runs": True} if sdk is None else sdk
        monkeypatch.setattr(ui.pano_stitch, "sdk_status", lambda: status)
        monkeypatch.setattr(ui.pano_stitch, "free_gb", lambda _folder: free)
        monkeypatch.setattr(capabilities, "nvidia_present", lambda: nvidia)

    def spawn(self, args: list[str], **_kw: object) -> FakeProc:
        self.seen.append(args)
        if args[1:3] == ["360", "stitch"]:
            proc = FakeProc(self.stitch_lines, self.stitch_code)
            self.stitching.set()
            if self.hold is None:
                proc.release.set()
            else:  # until cancelled
                self.hold = proc.release
            return proc
        out = Path(args[args.index("--out") + 1])
        out.mkdir()
        cache.save_json(out / "manifest.json", {"clips": []})
        proc = FakeProc(self.clips_lines, self.clips_code)
        proc.release.set()
        return proc

    def export(self, body: dict | None = None) -> dict:
        """Start an export (the sequence's, unless ``body`` says otherwise); the job's view."""
        body = body or {}
        if body.get("picks") != "auto":
            res = self.client.put(
                f"/api/projects/{self.pid}/review", json={"segments": [self.segment]}
            )
            assert res.status_code == 200, res.text
        res = self.client.post(f"/api/projects/{self.pid}/jobs", json={"action": "clips", **body})
        assert res.status_code == 200, res.text
        job = self.client.app.state.jobs.get(res.json()["id"])
        if self.hold is None:
            assert job.done.wait(10)
        return job

    def commands(self) -> list[list[str]]:
        return [a[1:3] for a in self.seen]


def test_a_360_export_stitches_the_sequence_recordings_first(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = StitchExport(client, tmp_path, monkeypatch)
    view = run.export().view()
    assert view["status"] == "completed", view
    stitch, clips = run.seen
    (seg_file,) = run.p.glob("ui/segments-*.json")
    assert stitch == ["cutpoint", "360", "stitch", str(run.p), "--segments", str(seg_file)]
    assert clips[1:3] == ["clips", str(run.p)]
    assert any("stitching 1 recording(s)" in ln and "500 GB free" in ln for ln in view["logs"])


def test_a_360_export_of_automatic_picks_stitches_their_clips(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = StitchExport(client, tmp_path, monkeypatch)
    assert run.export({"picks": "auto", "ids": ["h001"]}).view()["status"] == "completed"
    assert run.seen[0] == ["cutpoint", "360", "stitch", str(run.p), "--clips", "001"]
    assert run.commands()[1] == ["clips", str(run.p)]


@pytest.mark.parametrize(
    ("pano", "sdk", "said"),
    [
        ({"auto_stitch": False}, None, "pano.auto_stitch is off"),
        ({}, {"found": False, "runs": False}, "SDK was not found"),
        ({}, {"found": True, "runs": False}, "SDK does not run"),
    ],
    ids=["auto-stitch-off", "no-sdk", "sdk-does-not-run"],
)
def test_a_360_export_without_auto_stitch_or_a_working_sdk_exports_the_preview(
    client: TestClient,
    home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pano: dict,
    sdk: dict | None,
    said: str,
) -> None:
    set_config(home, pano=pano)
    run = StitchExport(client, tmp_path, monkeypatch, sdk=sdk)
    view = run.export().view()
    assert view["status"] == "completed", view
    assert run.commands() == [["clips", str(run.p)]]
    assert any(said in ln and "360 preview (not stitched)" in ln for ln in view["logs"])


def test_a_360_export_without_an_nvidia_gpu_skips_the_stitch_with_a_warning(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = StitchExport(client, tmp_path, monkeypatch, nvidia=False)
    view = run.export().view()
    assert view["status"] == "completed", view
    assert run.commands() == [["clips", str(run.p)]]
    assert any("WARNING no NVIDIA GPU; exporting the 360 preview" in ln for ln in view["logs"])


def test_a_360_export_skips_the_stitch_with_a_warning_when_space_is_short(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = StitchExport(client, tmp_path, monkeypatch, free=ui.pano_stitch.MIN_FREE_GB + 0.1)
    view = run.export().view()
    assert view["status"] == "completed", view
    assert run.commands() == [["clips", str(run.p)]]
    assert any("WARNING not enough free space" in ln for ln in view["logs"])


def test_a_360_export_of_stitched_recordings_does_not_stitch_again(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = StitchExport(client, tmp_path, monkeypatch)
    write(pano_project.sdk_output(run.clip), b"stitched")
    assert run.export().view()["status"] == "completed"
    assert run.commands() == [["clips", str(run.p)]]


def test_a_failed_stitch_still_exports_with_a_warning(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = StitchExport(client, tmp_path, monkeypatch)
    run.stitch_code = 1
    view = run.export().view()
    assert view["status"] == "completed" and "error" not in view, view
    assert run.commands() == [["360", "stitch"], ["clips", str(run.p)]]
    assert any("WARNING the step before clips failed (exit code 1)" in ln for ln in view["logs"])


def test_a_stitch_that_exits_0_after_an_error_warns_and_still_exports(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``360 stitch`` exits 0 when the SDK failed on a recording; its ERROR line says so."""
    run = StitchExport(client, tmp_path, monkeypatch)
    run.stitch_lines = ["10:00:00 ERROR clip 001: stitch FAILED: x"]
    view = run.export().view()
    assert view["status"] == "completed", view
    assert run.commands() == [["360", "stitch"], ["clips", str(run.p)]]
    want = "WARNING the step before clips reported errors: those recordings export from the 360"
    assert any(want in ln for ln in view["logs"])


@pytest.mark.parametrize(
    ("clips_lines", "error"),
    [([], "exit code 1"), (["10:00:00 ERROR own"], "own")],
    ids=["no-error-line", "own-error-line"],
)
def test_a_failed_export_after_a_failed_stitch_reports_its_own_error(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clips_lines: list[str],
    error: str,
) -> None:
    """The job's error comes from the clips step, never from the stitch's ERROR lines."""
    run = StitchExport(client, tmp_path, monkeypatch)
    run.stitch_lines, run.stitch_code = ["10:00:00 ERROR clip 001: stitch FAILED: x"], 1
    run.clips_lines, run.clips_code = clips_lines, 1
    view = run.export().view()
    assert view["status"] == "failed", view
    assert view["error"] == error


def test_cancelling_a_360_export_while_it_stitches_stops_both_steps(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = StitchExport(client, tmp_path, monkeypatch)
    run.stitch_code = 130
    run.hold = threading.Event()
    job = run.export()
    assert run.stitching.wait(10)
    client.post(f"/api/jobs/{job.id}/cancel")
    assert job.done.wait(10)
    assert job.view()["status"] == "cancelled"
    assert run.commands() == [["360", "stitch"]]  # clips never started


def test_a_flat_export_never_stitches(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, seg = cli_project(tmp_path)
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    monkeypatch.setattr(
        ui.pano_stitch, "sdk_status", lambda: pytest.fail("a flat export asked for the SDK")
    )
    FakeProc().fake_spawn(monkeypatch)
    res = client.post("/api/projects/cli/jobs", json={"action": "clips"})
    assert res.status_code == 200, res.text
    assert client.app.state.jobs.get(res.json()["id"]).done.wait(10)


def test_create_project_skips_unreadable_folder_entries(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry whose is_file() raises PermissionError must not make a good folder fail."""
    folder = tmp_path / "footage"
    folder.mkdir()
    (folder / "a.mp4").write_bytes(b"x")
    (folder / "locked").mkdir()
    real = Path.is_file

    def is_file(self: Path) -> bool:
        if self.name == "locked":
            raise PermissionError(13, "Permission denied", str(self))
        return real(self)

    monkeypatch.setattr(Path, "is_file", is_file)
    res = client.post("/api/projects", json={"name": "Trip", "source": str(folder)})
    assert res.status_code == 200, res.text


def test_find_videos_never_returns_raw_360_files(tmp_path: Path) -> None:
    """analyze must not read dual-fisheye .insv or .lrv proxies as flat video."""
    for name in (X3_00, X3_10, X3_LRV, "a.mp4"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in find_videos([tmp_path])] == ["a.mp4"]


def wait_job(client: TestClient, timeout: float = 180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = client.get("/api/jobs").json()["jobs"]
        if jobs and jobs[-1]["status"] != "running":
            return jobs[-1]
        time.sleep(0.5)
    for job in client.get("/api/jobs").json()["jobs"]:
        if job["status"] == "running":
            client.post(f"/api/jobs/{job['id']}/cancel")  # do not leave it running past the test
    raise AssertionError("job did not finish")


def test_requests_must_be_loopback_and_carry_the_token(
    client: TestClient, dialog: FakeDialog
) -> None:
    """DNS rebinding / cross-site pages / tokenless callers get nothing."""
    assert client.get("/api/projects", headers={"Host": "evil.example:8765"}).status_code == 403
    assert client.get("/api/projects", headers={"Origin": "http://evil.example"}).status_code == 403
    assert client.get("/api/session", headers={"Origin": "http://localhost:9"}).status_code == 403
    assert (
        client.get("/api/session", headers={"X-Cutpoint-Token": ""}).status_code == 403
    )  # no bootstrap
    assert client.get("/api/projects", headers={"X-Cutpoint-Token": "nope"}).status_code == 403
    bare = {"X-Cutpoint-Token": ""}
    assert client.post("/api/pick", params={"kind": "file"}, headers=bare).status_code == 403
    assert client.get("/api/settings", headers=bare).status_code == 403
    body = {"projects_dir": "x"}
    assert client.put("/api/settings", json=body, headers=bare).status_code == 403
    assert client.get("/api/settings", headers={"X-Cutpoint-Token": "nope"}).status_code == 403
    assert dialog.calls == []
    framed = client.get("/api/session")
    assert framed.headers["x-frame-options"] == "DENY"
    assert client.get("/api/projects").json() == {"projects": []}


def test_a_refused_request_says_whether_the_token_or_the_host_was_wrong(
    client: TestClient, tmp_path: Path
) -> None:
    """Contract pin: web/api.js TOKEN_DETAIL must match the token detail below exactly; it shows
    the "server restarted" banner and stops polling on it, and must not for the host detail."""
    token_detail = {"detail": "missing or wrong session token"}
    api_js = (ui.WEB_DIR / "api.js").read_text(encoding="utf-8")
    assert f"const TOKEN_DETAIL = '{token_detail['detail']}';" in api_js
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    media = client.get(f"/api/projects/{pid}").json()["sources"][0]["url"]
    wrong_t = media.split("?")[0] + "?t=stale"
    for wrong in ("", "stale"):
        bad = {"X-Cutpoint-Token": wrong}
        for url in ("/api/session", f"/api/projects/{pid}", wrong_t):
            res = client.get(url, headers=bad)
            assert (res.status_code, res.json()) == (403, token_detail), url
    host_detail = {"detail": "forbidden host or origin"}
    res = client.get("/api/session", headers={"Host": "evil.example:8765"})
    assert (res.status_code, res.json()) == (403, host_detail)
    res = client.get("/api/session", headers={"Origin": "http://evil.example"})
    assert (res.status_code, res.json()) == (403, host_detail)


def test_the_scrolling_columns_clip_their_visually_hidden_labels() -> None:
    """Contract pin (#354): .vh labels are position: absolute. Unless .col is positioned, their
    containing block is the page, the column's overflow does not clip them, and a long moments
    list makes the page scroll past the Jobs bar on a wide window."""
    css = (ui.WEB_DIR / "app.css").read_text(encoding="utf-8")
    assert re.search(r"^\.vh \{[^}]*position: absolute;", css, re.M)
    col = re.search(r"^\.col \{([^}]*)\}", css, re.M)
    assert col and "position: relative;" in col[1] and "overflow: auto;" in col[1]


def test_stats_for_nerds_is_an_off_by_default_overlay_on_the_preview() -> None:
    """Contract pin (#353): the scoring line is an overlay in the video panel, hidden by default,
    toggled by a labelled Settings box and remembered per browser; the line under the transcript
    is gone.

    Check in a browser (no browser in the test suite): with a moment open, tick Settings > Stats
    for Nerds; the stats show at once in the preview's top-left, clicks on them still play/pause,
    the transport bar stays uncovered, and they stay inside the preview at a 360 px wide window.
    Reload: still on. Untick: #signals is hidden and empty."""
    html = (ui.WEB_DIR / "index.html").read_text(encoding="utf-8")
    css = (ui.WEB_DIR / "app.css").read_text(encoding="utf-8")
    js = (ui.WEB_DIR / "app.js").read_text(encoding="utf-8")
    screen = re.search(r'<div class="screen">(.*?)</div>', html, re.S)
    assert screen and '<p id="signals" class="nerd-stats" hidden></p>' in screen[1]
    assert html.count('id="signals"') == 1
    settings = re.search(r'<dialog id="settingsDialog".*?</dialog>', html, re.S)
    assert settings and re.search(
        r'<label class="check"><input type="checkbox" id="setNerdStats"[^>]*> Stats for Nerds</label>',
        settings[0],
    )
    overlay = re.search(r"^\.nerd-stats \{([^}]*)\}", css, re.M)
    assert overlay and "position: absolute;" in overlay[1] and "pointer-events: none;" in overlay[1]
    assert "max-width: calc(100% - 16px);" in overlay[1] and "monospace" in overlay[1]
    assert "const NERD_KEY = 'cutpoint.nerdStats';" in js
    assert "nerdStats: false," in js  # off unless the browser stored "1"


def test_moment_cards_show_a_read_only_strip_of_their_place_in_the_source() -> None:
    """Contract pin: the score bar on the moment cards became a strip like the preview's overview
    (the whole source, the moment in --gold on --deep), with no control of its own, hidden from
    screen readers (the time line states the times) and the score in its hover title.

    Check in a browser (no browser in the test suite): in Blue, Light and Dark, each card's strip
    marks its moment in gold where the overview does; clicking it opens the moment as before."""
    css = (ui.WEB_DIR / "app.css").read_text(encoding="utf-8")
    js = (ui.WEB_DIR / "app.js").read_text(encoding="utf-8")
    strip = re.search(r"^function sourceStrip\(c\) \{(.*?)^\}", js, re.M | re.S)
    assert strip and "sourceStrip(c)," in js
    assert "'aria-hidden': 'true'" in strip[1]
    assert "title: `Score ${c.score.toFixed(3)}`" in strip[1]
    assert "input" not in strip[1] and "tabindex" not in strip[1]
    assert "const dur = sourceDur(c.source);" in strip[1]  # no request per card
    assert re.search(r"^\.cand-strip \{[^}]*background: var\(--deep\);", css, re.M)
    assert re.search(r"^\.cand-strip-clip \{[^}]*background: var\(--gold\);", css, re.M)
    assert "class: 'score'" not in js and not re.search(r"^\.score\b", css, re.M)


def _theme_colours(css: str, selector: str) -> dict[str, str]:
    block = re.search(re.escape(selector) + r" \{(.*?)\n\}", css, re.S)
    assert block, selector
    return dict(re.findall(r"--([\w-]+): (#[0-9a-f]{6})", block[1]))


def _contrast(a: str, b: str) -> float:
    def lum(hex_colour: str) -> float:
        rgb = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_settings_offer_blue_light_and_dark_themes_applied_before_first_paint() -> None:
    """Contract pin (#445): Settings > Theme picks Blue (the default, the :root colours), Light or
    Dark; themes only override the custom properties, the choice is stored per browser like Stats
    for Nerds, and an inline script in <head> sets data-theme before the stylesheet paints.

    Check in a browser (no browser in the test suite): pick Light in Settings, the whole page
    recolours at once; Cancel or Escape goes back; Save and reload: still Light, with no blue
    flash. Clear the key ``cutpoint.theme``: Blue, as before."""
    html = (ui.WEB_DIR / "index.html").read_text(encoding="utf-8")
    css = (ui.WEB_DIR / "app.css").read_text(encoding="utf-8")
    js = (ui.WEB_DIR / "app.js").read_text(encoding="utf-8")
    settings = re.search(r'<dialog id="settingsDialog".*?</dialog>', html, re.S)
    assert settings
    select = re.search(r'<select id="setTheme"[^>]*>(.*?)</select>', settings[0], re.S)
    assert select and re.findall(r'<option value="(\w+)">(\w+)</option>', select[1]) == [
        ("blue", "Blue"),
        ("light", "Light"),
        ("dark", "Dark"),
    ]
    head = html.split("</head>")[0]
    early = re.search(r"<script>(.*?)</script>", head, re.S)
    assert early and "localStorage.getItem('cutpoint.theme')" in early[1] and "try {" in early[1]
    assert "document.documentElement.dataset.theme = t" in early[1]
    assert head.index("<script>") < head.index('href="app.css"')  # set before the styles apply
    assert "const THEME_KEY = 'cutpoint.theme';" in js
    assert "localStorage.removeItem(THEME_KEY)" in js  # Blue is stored as no key

    blue = _theme_colours(css, ":root")
    # no saved choice: today's colours, unchanged
    keep = ("slate", "panel", "raise", "deep", "line", "track", "text", "muted")
    assert {k: blue[k] for k in keep} == {
        "slate": "#233348",
        "panel": "#2d4058",
        "raise": "#384e6b",
        "deep": "#19273a",
        "line": "#455d7d",
        "track": "#455d7d",  # bar tracks: today's --line
        "text": "#edf2f8",
        "muted": "#b3c2d2",
    }
    assert (blue["amber"], blue["bad"], blue["ok"], blue["glow"]) == (
        "#f3bd60",
        "#ffa89b",
        "#94d6a8",
        "#ffae19",
    )
    # the colours rules used to hold themselves, now :root custom properties, are the same values
    root = re.search(r":root \{(.*?)\n\}", css, re.S)
    assert root
    moved = {
        "hover": "#435b7a",
        "amber-hover": "#f7cb80",
        "danger-ink": "#ffffff",  # was #fff
        "danger-btn-hover": "#d23a3a",
        "bad-bg": "#4a2c33",
        "bad-line": "#7a4650",
        "art": "#8195b0",
        "well": "#0d1622",
        "log-bg": "#101b2a",
        "tick": "rgba(179, 194, 210, 0.2)",
        "hatch": "rgba(179, 194, 210, 0.16)",
        "gap-shade": "rgba(10, 18, 28, 0.45)",
        "glow-shadow": "rgba(255, 174, 25, 0.35)",
        "danger-tint": "rgba(224, 62, 62, 0.16)",
        "danger-tint-line": "rgba(224, 62, 62, 0.5)",
        "shadow": "rgba(5, 10, 18, 0.6)",
        "backdrop": "rgba(10, 18, 28, 0.72)",
        "scrim": "rgb(8 12 18 / 0.78)",
        "scrim-ink": "#edf2f8",  # was var(--text)
        "thumb-scrim": "rgba(0, 0, 0, 0.55)",
        "thumb-scrim-ink": "#f3bd60",  # was var(--amber)
    }
    root_values = dict(re.findall(r"--([\w-]+): (#[0-9a-fA-F]+|rgba?\([^)]*\))", root[1]))
    assert {k: root_values.get(k) for k in moved} == moved
    # outside the theme blocks, rules use the custom properties, not colours of their own
    # (comments name issues as #352; the backdrop keeps a fallback for browsers where it does not inherit)
    rules = re.sub(
        r"/\*.*?\*/|var\(--backdrop, [^)]*\)\)", "", css.split("\n}\n", 3)[3], flags=re.S
    )
    assert not re.findall(r"#[0-9a-fA-F]{3,6}\b|rgba?\(\d", rules)

    for name in ("blue", "light", "dark"):
        t = (
            blue
            if name == "blue"
            else {**blue, **_theme_colours(css, f':root[data-theme="{name}"]')}
        )
        text = [
            ("text", b)
            for b in ("slate", "panel", "raise", "deep", "picked", "bar", "hover", "log-bg")
        ]
        text += [
            (f, b) for f in ("muted", "amber-text", "bad", "ok") for b in ("slate", "panel", "deep")
        ]
        text += [("muted", "raise"), ("glow", "panel"), ("bad", "bad-bg"), ("amber-ink", "amber")]
        text += [
            ("amber-ink", "amber-hover"),
            ("danger-ink", "danger-btn"),
            ("danger-ink", "danger-btn-hover"),
        ]
        text += [("gold-ink", "gold-btn"), ("gold-ink", "gold-btn-hover")]
        for fg, bg in text:
            assert _contrast(t[fg], t[bg]) >= 4.5, (name, fg, bg)
        # focus rings, the focused-card glow and other controls: 3:1 (WCAG 1.4.11)
        # --amber-text: focus rings and amber text; --amber is the fill, with --amber-ink on it
        ui_parts = [("amber-text", b) for b in ("slate", "panel", "raise", "deep", "bar")]
        ui_parts += [
            ("glow", "deep"),
            ("danger", "deep"),
            ("muted", "well"),
            ("gold", "deep"),
            ("gold-edge", "deep"),
        ]
        if name != "blue":  # Blue keeps today's borders (below 3:1)
            ui_parts += [("line", "panel"), ("line", "slate"), ("line", "deep")]
            # the --amber fill of progress bars on their track
            ui_parts += [("amber", "track")]
        for fg, bg in ui_parts:
            assert _contrast(t[fg], t[bg]) >= 3, (name, fg, bg)
    assert re.search(r':root\[data-theme="light"\] \{[^}]*color-scheme: light;', css)
    # the selected tab's underline and the range accents use that token, not the --amber fill
    assert re.search(r'\.side-tab\[aria-selected="true"\][^{]*\{[^}]*var\(--amber-text\)', css)
    for rule in re.findall(r"[^{}]*input\[type=range\][^{}]*\{[^}]*accent-color[^}]*\}", css):
        assert "accent-color: var(--amber-text)" in rule, rule


def test_the_trim_markers_are_keyboard_and_pointer_slider_handles() -> None:
    """Contract pin (#352): Clip start and Clip end on the rail are focusable sliders with a 24 px
    hit area, moved by keys and pointer drags through setRange (the path of the time fields and
    "Use playhead"), and Space is not taken from the global play/pause handler.

    Check in a browser (no browser in the test suite): Tab to a handle, arrows move it 0.1 s,
    Shift+arrows 1 s, PageUp/PageDown 5 s; the time field and aria-valuetext follow; start stops
    MIN_CLIP before end; Space plays and pauses; a drag on a zoomed rail pans at the edge."""
    html = (ui.WEB_DIR / "index.html").read_text(encoding="utf-8")
    css = (ui.WEB_DIR / "app.css").read_text(encoding="utf-8")
    js = (ui.WEB_DIR / "app.js").read_text(encoding="utf-8")
    rail = re.search(r'<div class="rail" id="rail">(.*?)\n        </div>', html, re.S)
    assert rail
    for which, name in (("start", "Clip start"), ("end", "Clip end")):
        tag = re.search(rf'<div id="{which}Handle"[^>]*>', rail[1])
        assert tag, which
        for attr in (
            'role="slider"',
            'tabindex="0"',
            f'aria-label="{name}"',
            "aria-valuemin=",
            "aria-valuemax=",
            "aria-valuenow=",
        ):
            assert attr in tag[0], (which, attr)
    for attr in ("aria-valuemin", "aria-valuemax", "aria-valuenow"):
        assert f"el.setAttribute('{attr}'" in js
    assert "el.setAttribute('aria-valuetext', `${HANDLE_NAMES[which]}, ${fmt(t)}`);" in js
    assert (
        "const HANDLE_STEPS = { ArrowLeft: -0.1, ArrowDown: -0.1, ArrowRight: 0.1, ArrowUp: 0.1, PageDown: -5, PageUp: 5 };"
        in js
    )
    assert "step * 10" in js  # Shift+arrows: 1 s
    assert "return t === c[which] || setRange(which, t, widen);" in js
    assert "el.setPointerCapture(e.pointerId);" in js
    # a drag whose clip changed is dropped when the stage renders, not on the next pointer event
    at = js.index("function renderStage(")
    stage = js[at : js.index("const c = current();", at)]
    assert "if (trimDrag && dragStale(trimDrag)) dropDrag();" in stage
    assert (
        "' '"
        not in js[js.index("const HANDLE_STEPS") : js.index("el.addEventListener('pointerdown'")]
    )
    handle = re.search(r"^\.rail-handle \{([^}]*)\}", css, re.M)
    assert handle and "width: 24px;" in handle[1] and "top: 0; bottom: 0;" in handle[1]
    assert "touch-action: none;" in handle[1]
    assert re.search(r"^\.rail-handle:focus-visible \{[^}]*outline: 2px solid", css, re.M)
    # the rail is 46 px tall: the handle's hit area is at least 24 x 24
    assert re.search(r"^\.rail \{[^}]*height: 46px;", css, re.M)


def test_the_page_files_are_revalidated_on_each_load(client: TestClient) -> None:
    """#415: with no Cache-Control the browser reused an app.css from before the #354 fix, judged
    fresh by its age, and the page scrolled past the Jobs bar again. Each load must revalidate.

    Layout check in a browser (no browser in the test suite): a project with 20+ moments and
    clips, the editor open, at 1915x950 and 1280x720 the document's scrollHeight equals
    innerHeight and #jobs ends at the bottom edge; at 960px wide the page scrolls as before."""
    for path in ("/", "/index.html", "/app.css", "/app.js", "/api.js"):
        res = client.get(path)
        assert res.status_code == 200, path
        assert res.headers["cache-control"] == "no-cache", path
        again = client.get(path, headers={"If-None-Match": res.headers["etag"]})
        assert again.status_code == 304, path
    assert client.get("/api/session").headers["cache-control"] == "no-store"


def test_project_ids_cannot_leave_the_projects_folder(client: TestClient, tmp_path: Path) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    for bad_id in ("..", "%2e%2e", "a%2Fb", pid + "%00"):
        assert client.get(f"/api/projects/{bad_id}").status_code in (404, 422)
    assert client.get(f"/api/projects/{pid}").status_code == 200


def test_review_is_validated_and_saved_atomically(client: TestClient, tmp_path: Path) -> None:
    """Only project sources, real ranges inside the source; bad input leaves the saved review."""
    private = tmp_path / "private.mp4"
    private.write_bytes(b"secret")
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    src = str((folder / "a.mp4").resolve())
    p = ui.project_dir(pid)
    # highlights.json names a file outside the picked folder: it must not become a source.
    cache.save_json(
        p / "highlights.json",
        {
            "settings": {"inputs": [str(private)]},
            "sources": [
                {"path": src, "duration": 10.0},
                {"path": str(private), "duration": 10.0},
                {"path": "/etc/hostname", "duration": 1.0},
            ],
            "candidates": [],
        },
    )
    assert [s["path"] for s in client.get(f"/api/projects/{pid}").json()["sources"]] == [src]

    def put(*segments: dict):
        return client.put(f"/api/projects/{pid}/review", json={"segments": list(segments)})

    ok = {"id": "h001", "source": src, "start": 1, "end": 4.5, "label": "good"}
    assert put(ok).status_code == 200
    saved = client.get(f"/api/projects/{pid}").json()["review"]["segments"]
    assert saved == [{**ok, "start": 1.0, "end": 4.5}]
    # #324: the editor plays the whole source, so a trim may reach far past the analyzed moment.
    assert put({**ok, "end": 9.9}).status_code == 200
    assert put(ok).status_code == 200
    for broken in (
        {**ok, "start": 5, "end": 5},  # empty
        {**ok, "end": 10.5},  # past the source
        {**ok, "source": str(private)},  # not the project's
        {**ok, "start": "1"},  # not a number
    ):
        assert put(broken).status_code == 400
    assert put(ok, ok).status_code == 400  # duplicate id
    assert client.get(f"/api/projects/{pid}").json()["review"]["segments"] == saved


def test_moment_ids_survive_reanalysis_without_relinking_saved_trims(
    client: TestClient, tmp_path: Path
) -> None:
    """Rank ids (h001) are reassigned by analysis; a saved trim must never follow one."""
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x", "b.mp4": b"x"})
    a, b = (str((folder / n).resolve()) for n in ("a.mp4", "b.mp4"))
    p = ui.project_dir(pid)

    def analyse(*cands: tuple[str, str, float, float]) -> dict[tuple[str, float, float], str]:
        cache.save_json(
            p / "highlights.json",
            {
                "sources": [{"path": a, "duration": 400.0}, {"path": b, "duration": 400.0}],
                "candidates": [
                    {"id": i, "source": s, "start": t0, "end": t1} for i, s, t0, t1 in cands
                ],
            },
        )
        got = client.get(f"/api/projects/{pid}").json()["highlights"]["candidates"]
        return {(c["source"], c["start"], c["end"]): c["id"] for c in got}

    first = analyse(("h001", a, 10, 20), ("h002", a, 30, 40))
    old = {"id": "h001", "source": a, "start": 11.0, "end": 19.0, "label": "old"}
    assert client.put(f"/api/projects/{pid}/review", json={"segments": [old]}).status_code == 200

    again = analyse(("h001", b, 300.0, 310.0), ("h002", a, 30.0, 40.0), ("h003", a, 10.0, 20.0))
    assert "h001" not in again.values()
    assert again[(a, 10, 20)] == first[(a, 10, 20)]  # reranked, same moment
    assert first[(a, 10, 20)] != first[(a, 30, 40)]
    assert again[(b, 300, 310)] not in first.values()
    assert client.get(f"/api/projects/{pid}").json()["review"]["segments"] == [old]


def test_moment_cards_get_the_cli_candidate_id_that_ids_takes(tmp_path: Path) -> None:
    """#487: the payload keeps the CLI id (h001) next to the internal id, and Candidate ids
    (``clips --ids``) takes exactly what the cards show."""
    src = str(tmp_path / "a.mp4")
    hl = {
        "candidates": [{"id": f"h00{i}", "source": src, "start": i, "end": i + 5.0} for i in (1, 2)]
    }
    view = ui.moments_view(hl)["candidates"]
    assert [c["cli_id"] for c in view] == ["h001", "h002"]
    assert all(c["id"] not in ("h001", "h002") for c in view)  # the internal id stays the hash
    shown = [c["cli_id"] for c in view][::-1]
    assert ui._auto_picks({"ids": shown}, hl) == ["--ids", "h002", "h001"]


def test_moment_cards_show_the_cli_id_as_text_and_filter_by_it() -> None:
    """#487: the id is real text with a screen-reader prefix, not a tooltip; data-id stays the
    internal id (markMoments, AI highlights and the sequence use it)."""
    render = _js_function(_web("app.js"), "renderMoments")
    assert "h('span', { class: 'vh', text: 'Candidate ' }), c.cli_id)" in render
    assert "'data-id': c.id" in render
    assert re.search(r"\.filter\(\(\{ c \}\) => [^\n]*c\.cli_id", render)
    assert (
        ".cand-id {" in _web("app.css")
        and "white-space: nowrap" in _web("app.css").split(".cand-id {")[1].split("}")[0]
    )
    assert "by the ids on the moment cards" in _web("index.html")


def test_media_supports_range_and_only_serves_project_sources(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = media_info(duration=10.0)  # h264 yuv420p + one AAC track: the browser plays it
    monkeypatch.setattr(ui, "probe", lambda _path: info)
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"0123456789"})
    source = client.get(f"/api/projects/{pid}").json()["sources"][0]
    assert source["status"] == "unknown"  # opening a project probes nothing
    client.post(f"/api/projects/{pid}/sources/{source['id']}/prepare")
    source = client.get(f"/api/projects/{pid}").json()["sources"][0]
    assert source["status"] == "ready"
    res = client.get(source["url"], headers={"Range": "bytes=2-5"})
    assert (res.status_code, res.content) == (206, b"2345")
    assert res.headers["content-range"] == "bytes 2-5/10"
    assert client.get(source["url"], headers={"Range": "bytes=50-"}).status_code == 416
    bare = source["url"].split("?")[0]  # no ?t= token, and the header token is blanked
    assert client.get(bare, headers={"X-Cutpoint-Token": ""}).status_code == 403
    unknown = source["url"].replace(source["url"].split("/")[-1].split("?")[0], "0" * 12)
    assert client.get(unknown).status_code == 404


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


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
    deadline = time.time() + 120
    while sources()["a.mp4"]["status"] == "preparing" and time.time() < deadline:
        time.sleep(0.3)
    after = sources()
    assert (after["a.mp4"]["status"], after["b.mkv"]["status"]) == ("ready", "unknown")

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
    assert wait_job(client)["status"] == "completed"
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
    captured: list[list[str]] = []
    monkeypatch.setattr(ui.Jobs, "start", fake_start(captured))
    res = client.post("/api/projects/cli/jobs", json={"action": "analyze"})
    assert res.status_code == 200, res.text
    assert ("--recover" in captured[0]) is passed
    cli.build_parser().parse_args(captured[0][3:])


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
    captured: list[list[str]] = []

    monkeypatch.setattr(ui.Jobs, "start", fake_start(captured))
    assert client.post("/api/projects/cli-made/jobs", json={"action": "analyze"}).status_code == 200
    args = captured[0]
    assert [a for a in args if a.startswith(("--vision-top", "--ollama"))] == flags


@pytest.mark.parametrize(
    ("body", "saved", "cfg", "vision"),
    [
        ({}, {}, None, False),  # opt-in: off by default
        ({}, {}, False, False),
        ({}, {}, True, True),  # Analyze again: the Settings value
        ({}, {"vision_requested": True}, None, False),  # not what the last run asked
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
    vision: bool,
) -> None:
    """--vision or --no-vision as Settings says when the job starts, never neither."""
    if cfg is not None:
        set_config(home, vision=cfg)
    p = ui.projects_dir() / "cli-made"
    p.mkdir(parents=True)
    footage = tmp_path / "clip.mp4"
    footage.write_bytes(b"x")
    settings = {"inputs": [str(footage)], **saved}
    cache.save_json(p / "highlights.json", {"style": "gaming", "settings": settings, "sources": []})
    captured: list[list[str]] = []

    monkeypatch.setattr(ui.Jobs, "start", fake_start(captured))
    res = client.post("/api/projects/cli-made/jobs", json={"action": "analyze", **body})
    assert res.status_code == 200
    assert ("--vision" in captured[0]) is vision
    assert ("--no-vision" in captured[0]) is not vision


def test_analyze_follows_settings_saved_just_before(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Saving Settings changes the next analysis at once, without a reload or restart."""
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    captured: list[list[str]] = []

    monkeypatch.setattr(ui.Jobs, "start", fake_start(captured))
    for value in (True, False, True):
        assert client.put("/api/settings", json={"vision": value}).status_code == 200
        res = client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"})
        assert res.status_code == 200
    assert [("--vision" in a, "--no-vision" in a) for a in captured] == [
        (True, False),
        (False, True),
        (True, False),
    ]


@pytest.mark.slow
def test_one_job_at_a_time_and_cancel_kills_the_process_group(tmp_path: Path) -> None:
    script = (
        "import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print('child', c.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    jobs = ui.Jobs()
    job = jobs.start("p", "analyze", [sys.executable, "-c", script])
    deadline = time.time() + 60  # two interpreter starts can be slow under a loaded suite
    while not job.logs and job.status == "running" and time.time() < deadline:
        time.sleep(0.05)
    assert job.logs, (job.status, job.error)
    child = int(job.logs[0].split()[1])
    with pytest.raises(HTTPException) as busy:
        jobs.start("p", "clips", [sys.executable, "-c", "pass"])
    assert busy.value.status_code == 409
    cancelled = jobs.cancel(job.id)
    deadline = (
        time.time() + 10
    )  # cancel returns at once; the killer thread and the watcher finish in the background
    while cancelled.status != "cancelled" and time.time() < deadline:
        time.sleep(0.05)
    assert cancelled.status == "cancelled"
    assert job.proc is not None and job.proc.poll() is not None
    if sys.platform.startswith("linux"):  # a reaped-or-zombie child counts as dead
        time.sleep(0.2)
        stat = Path(f"/proc/{child}/stat")
        assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    failing = jobs.start(
        "p", "analyze", [sys.executable, "-c", "print('12:00:00 ERROR boom')\nraise SystemExit(1)"]
    )
    assert failing.done.wait(20)
    assert (failing.status, failing.error) == ("failed", "boom")


@pytest.mark.parametrize(
    ("event_sent", "exits_in_time", "job", "expected"),
    [
        (1, False, 777, ["ctrl_break", "wait", "terminate", "taskkill"]),  # job object first
        (1, False, None, ["ctrl_break", "wait", "taskkill"]),  # ignores the event
        (1, True, None, ["ctrl_break", "wait", "taskkill"]),  # taskkill still follows
        (0, False, None, ["ctrl_break", "taskkill"]),  # no console: nothing to wait for
    ],
    ids=["job-object", "ignores-the-event", "soft-stop-works", "no-console"],
)
def test_windows_cancel_tries_ctrl_break_then_the_job_object_then_taskkill(
    monkeypatch: pytest.MonkeyPatch,
    event_sent: int,
    exits_in_time: bool,
    job: int | None,
    expected: list[str],
) -> None:
    calls: list[tuple[str, object]] = []

    def wait(timeout: float) -> int:
        calls.append(("wait", timeout))
        if exits_in_time:
            return 130
        raise subprocess.TimeoutExpired("job", timeout)

    kernel32 = SimpleNamespace(
        GenerateConsoleCtrlEvent=lambda event, pid: (
            calls.append(("ctrl_break", (event, pid))) or event_sent
        ),
        TerminateJobObject=lambda handle, code: calls.append(("terminate", (handle, code))) or 1,
    )
    fake_ctypes = SimpleNamespace(WinDLL=lambda name: kernel32, c_void_p=lambda value: value)
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    monkeypatch.setattr(ui, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(ui, "signal", SimpleNamespace(CTRL_BREAK_EVENT=1))
    monkeypatch.setattr(
        ui,
        "subprocess",
        SimpleNamespace(
            run=lambda args, **kw: calls.append(("taskkill", args)),
            TimeoutExpired=subprocess.TimeoutExpired,
        ),
    )
    ui.kill_tree(SimpleNamespace(pid=4242, wait=wait, tree_job=job), 0.01)  # type: ignore[arg-type]
    # the job object reaches descendants that outlive the leader, which taskkill /T cannot
    assert [name for name, _ in calls] == expected
    got = dict(calls)
    assert got["ctrl_break"] == (1, 4242)
    assert got["taskkill"] == ["taskkill", "/PID", "4242", "/T", "/F"]
    assert got.get("wait", 0.01) == 0.01
    assert got.get("terminate", (777, 1)) == (777, 1)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_posix_kill_tree_waits_job_stop_s_between_term_and_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []
    waits: list[float | None] = []
    monkeypatch.setattr(ui.os, "killpg", lambda pid, sig: signals.append(sig))

    def wait(timeout: float | None = None) -> int:
        waits.append(timeout)
        raise subprocess.TimeoutExpired("job", timeout or 0)

    ui.kill_tree(SimpleNamespace(pid=4242, wait=wait))  # type: ignore[arg-type]
    # a job's own soft stop (two ffmpeg waits plus cleanup) fits before the hard kill (#257)
    assert waits == [ui.JOB_STOP_S]
    assert ui.JOB_STOP_S > 2 * ui.SOFT_STOP_S
    assert signals == [signal.SIGTERM, signal.SIGKILL]


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_posix_kill_tree_without_term_sends_only_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []
    monkeypatch.setattr(ui.os, "killpg", lambda pid, sig: signals.append(sig))

    def wait(timeout: float | None = None) -> int:
        raise AssertionError("no soft stop to wait for")

    ui.kill_tree(SimpleNamespace(pid=4242, wait=wait), term=False)  # type: ignore[arg-type]
    assert signals == [signal.SIGKILL]

    def gone(pid: int, sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(ui.os, "killpg", gone)  # the group already exited: nothing to report
    ui.kill_tree(SimpleNamespace(pid=4242, wait=wait), term=False)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("done", "cancelled", "expected"),
    [
        (True, True, []),  # exited by itself while stopping: no kill at all
        (False, True, [False]),  # still stuck after JOB_STOP_S: KILL only, no second TERM
        (False, False, [True]),  # not yet stopped: the full TERM then KILL, as before
    ],
)
def test_jobs_shutdown_does_not_term_a_cancelling_job_twice(
    monkeypatch: pytest.MonkeyPatch, done: bool, cancelled: bool, expected: list[bool]
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(ui, "kill_tree", lambda proc, wait=0, *, term=True: calls.append(term))
    monkeypatch.setattr(ui, "JOB_STOP_S", 0.01)
    jobs = ui.Jobs()
    job = ui.Job("p", "analyze")
    job.proc = SimpleNamespace(pid=4242)  # type: ignore[assignment]
    job.cancelled = cancelled
    if done:
        job.done.set()
    jobs.jobs[job.id] = job
    jobs.shutdown()
    assert calls == expected
    assert job.cancelled


def _pid_alive(pid: int) -> bool:
    import ctypes  # noqa: PLC0415 - Windows only

    k32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
    k32.OpenProcess.restype = ctypes.c_void_p
    handle = k32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        return False
    try:
        return k32.WaitForSingleObject(ctypes.c_void_p(handle), 0) == 0x102  # WAIT_TIMEOUT
    finally:
        k32.CloseHandle(ctypes.c_void_p(handle))


@pytest.mark.slow
@pytest.mark.skipif(os.name != "nt", reason="Windows job objects and CTRL_BREAK")
def test_windows_cancel_ends_a_grandchild_that_ignores_ctrl_break() -> None:
    grandchild = (
        "import signal, time\n"
        "signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    leader = (
        "import subprocess, sys, time\n"
        f"c = subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
        "print('child', c.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    jobs = ui.Jobs()
    job = jobs.start("p", "analyze", [sys.executable, "-c", leader])
    child = 0
    try:
        deadline = time.time() + 60  # two interpreter starts can be slow under a loaded suite
        while (
            job.status == "running"
            and time.time() < deadline
            and not any(line == "ready" for line in list(job.logs))
        ):
            time.sleep(0.05)
        logs = list(job.logs)
        assert "ready" in logs, (job.status, job.error, logs)
        child = int(next(line for line in logs if line.startswith("child")).split()[1])
        cancelled = jobs.cancel(job.id)
        # the leader dies on the event at once (only the grandchild ignores it), so the job
        # object kill follows promptly; JOB_STOP_S bounds the wait, +20 s for the watcher to see EOF
        deadline = time.time() + ui.JOB_STOP_S + 20
        while cancelled.status != "cancelled" and time.time() < deadline:
            time.sleep(0.05)
        assert cancelled.status == "cancelled"
        deadline = time.time() + 5
        while _pid_alive(child) and time.time() < deadline:
            time.sleep(0.05)
        assert not _pid_alive(child)
    finally:
        if child and _pid_alive(child):
            subprocess.run(["taskkill", "/PID", str(child), "/T", "/F"], capture_output=True)
        if job.proc is not None and job.proc.poll() is None:
            ui.kill_tree(job.proc)


def test_assemble_ignores_a_hidden_part_file_left_by_a_killed_export(
    client: TestClient, tmp_path: Path
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    (p / "clips_ui_1").mkdir()
    cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": []})
    (p / "clips_ui_1" / ".001_a.part.mp4").write_bytes(b"x")
    cache.save_json(p / "ui" / "exports.json", {"exports": [{"dir": "clips_ui_1"}]})
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "assemble"})
    assert res.status_code == 400
    assert "export clips before assembling" in res.text


@pytest.mark.slow
@needs_ffmpeg
def test_export_keeps_saved_order_and_each_assembly_is_kept(
    client: TestClient, tmp_path: Path
) -> None:
    """Review order is the clip order; re-exporting or re-assembling never overwrites."""
    video = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=d=8:s=320x240:r=25",
         "-f", "lavfi", "-i", "sine=d=8", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(video)],
        check=True,
    )  # fmt: skip
    pid, folder = make_project(client, tmp_path, {"src.mp4": video.read_bytes()})
    src = str((folder / "src.mp4").resolve())
    segments = [
        {"id": "s-2", "source": src, "start": 5, "end": 7, "label": "late"},
        {"id": "s-1", "source": src, "start": 0.5, "end": 2.5, "label": "early"},
    ]
    assert client.put(f"/api/projects/{pid}/review", json={"segments": segments}).status_code == 200

    def run(body: dict) -> dict:
        res = client.post(f"/api/projects/{pid}/jobs", json=body)
        assert res.status_code == 200, res.text
        job = wait_job(client)
        assert job["status"] == "completed", job
        return client.get(f"/api/projects/{pid}").json()

    project = run({"action": "clips"})
    assert [(c["id"], c["start"], c["file"][:3]) for c in project["clips"]["clips"]] == [
        ("s-2", 5.0, "001"),
        ("s-1", 0.5, "002"),
    ]
    first_dir = project["clips"]["dir"]
    assert project["outputs"] == []

    taken = ui.project_dir(pid) / f"final{first_dir[5:]}.ass"  # a caption file already there
    taken.write_text("keep me", encoding="utf-8")
    first = run({"action": "assemble"})["outputs"]
    assert taken.read_text(encoding="utf-8") == "keep me"
    assert first[0]["name"] != f"{taken.stem}.mp4"
    second = run({"action": "assemble", "transition": "cut"})["outputs"]
    assert first[0]["name"].endswith(".mp4") and first[0]["name"] != second[0]["name"]
    assert [o["name"].rsplit(".", 1)[1] for o in second[:3]] == ["mp4", "fcpxml", "otio"]
    assert [o["name"] for o in second[3:]][:1] == [
        first[0]["name"]
    ]  # the earlier final stays listed
    assert (ui.project_dir(pid) / first[0]["name"]).is_file()  # the earlier final survives
    assert client.get(second[0]["url"]).status_code == 200

    again = run({"action": "clips"})  # fresh folder; the earlier export stays untouched
    assert again["clips"]["dir"] != first_dir
    # Every final stays listed (newest first), but none is the new export's own yet.
    assert again["outputs"] == second and again["final_current"] is False
    assert (ui.project_dir(pid) / first_dir / "manifest.json").is_file()


@pytest.fixture
def still_source(client: TestClient, tmp_path: Path) -> tuple[str, str, Path, Path]:
    """A project with one 1 s clip: (project id, source id, source file, project's frames dir)."""
    clip = tmp_path / "clip.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=d=1:s=320x240:r=25", "-c:v", "mpeg4", str(clip))
    pid, folder = make_project(client, tmp_path, {"a.mp4": clip.read_bytes()})
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    return pid, sid, folder / "a.mp4", ui.project_dir(pid) / ui.UI_DIR / "frames"


@needs_ffmpeg
@pytest.mark.parametrize(
    ("pid", "sid", "query", "token", "status", "cached"),
    [
        ("ok", "ok", "at=0.5", False, 403, None),
        ("nope", "ok", "at=0.5", True, 404, None),
        ("ok", "0" * 12, "at=0.5", True, 404, None),
        ("ok", "ok", "at=nan", True, 400, None),
        ("ok", "ok", "at=-1", True, 400, None),
        ("ok", "ok", "at=soon", True, 422, None),
        ("ok", "ok", "at=0.5&w=1", True, 200, "-50-32.jpg"),
        ("ok", "ok", "at=0.5&w=99999", True, 200, "-50-640.jpg"),
        ("ok", "ok", "at=0.5", True, 200, "-50-160.jpg"),
        ("ok", "ok", "at=99", True, 200, "-90-160.jpg"),  # clamped to duration - 0.1 s
        ("ok", "ok", "at=1e308", True, 200, "-90-160.jpg"),
    ],
)
def test_frame_endpoint_validates_input_and_clamps(
    client: TestClient,
    still_source: tuple[str, str, Path, Path],
    pid: str,
    sid: str,
    query: str,
    token: bool,
    status: int,
    cached: str | None,
) -> None:
    real_pid, real_sid, _, frames = still_source
    pid, sid = (real_pid if pid == "ok" else pid), (real_sid if sid == "ok" else sid)
    suffix = f"&t={client.headers['X-Cutpoint-Token']}" if token else ""
    res = client.get(
        f"/api/projects/{pid}/sources/{sid}/frame?{query}{suffix}",
        headers={"X-Cutpoint-Token": ""},
    )
    assert res.status_code == status, res.text
    if cached:
        assert [f.name[-len(cached) :] for f in frames.glob("*.jpg")] == [cached]
    else:
        assert not frames.exists()


@needs_ffmpeg
def test_frame_is_a_cached_jpeg_inside_the_project_and_the_source_is_untouched(
    client: TestClient, still_source: tuple[str, str, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, sid, source, frames = still_source
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    url = f"/api/projects/{pid}/sources/{sid}/frame?at=0.5&t={client.headers['X-Cutpoint-Token']}"
    calls: list[float] = []
    real_grab = ui.grab_images
    monkeypatch.setattr(
        ui, "grab_images", lambda *a, **k: (calls.append(a[1]), real_grab(*a, **k))[1]
    )

    first = client.get(url, headers={"X-Cutpoint-Token": ""})
    assert first.status_code == 200
    assert first.headers["content-type"] == "image/jpeg"
    assert first.content[:3] == b"\xff\xd8\xff"
    assert "max-age=86400" in first.headers["cache-control"]
    files = list(frames.iterdir())
    assert [f.suffix for f in files] == [".jpg"]  # no .part left behind
    assert files[0].read_bytes() == first.content
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before

    second = client.get(url, headers={"X-Cutpoint-Token": ""})
    assert second.content == first.content
    assert calls == [0.5]  # ffmpeg ran once; the second request was served from the cache


@needs_ffmpeg
def test_frame_failure_is_a_500_and_leaves_no_cache_file(
    client: TestClient, still_source: tuple[str, str, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, sid, _, frames = still_source

    def broken(*_a: object, **_k: object) -> list[bytes]:
        raise ui.FFmpegError("no image")

    monkeypatch.setattr(ui, "grab_images", broken)
    res = client.get(
        f"/api/projects/{pid}/sources/{sid}/frame?at=0.5&t={client.headers['X-Cutpoint-Token']}",
        headers={"X-Cutpoint-Token": ""},
    )
    assert res.status_code == 500
    assert not list(frames.glob("*")) if frames.exists() else True


@needs_ffmpeg
def test_frame_builds_one_still_at_a_time_even_after_a_failed_build(
    still_source: tuple[str, str, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, _, source, frames = still_source
    media = ui.Media()
    first_in, release_first, second_in, release_second = (threading.Event() for _ in range(4))
    calls: list[int] = []
    active = [0, 0]  # now, most at once

    def grab(*_a: object, **_k: object) -> list[bytes]:
        calls.append(1)
        active[0] += 1
        active[1] = max(active)
        try:
            if len(calls) == 1:
                first_in.set()
                release_first.wait(5)
                raise ui.FFmpegError("no image")
            if len(calls) == 2:
                second_in.set()
                release_second.wait(5)
            return [b"\xff\xd8\xffjpeg"]
        finally:
            active[0] -= 1

    monkeypatch.setattr(ui, "grab_images", grab)
    results: list[object] = []

    def request() -> None:
        try:
            results.append(media.frame(ui.project_dir(pid), str(source), 0.5, 160))
        except ui.FFmpegError as e:
            results.append(e)

    a = threading.Thread(target=request)
    a.start()
    assert first_in.wait(5)
    b = threading.Thread(target=request)  # waits for the first builder's lock
    b.start()
    time.sleep(0.2)
    release_first.set()  # the first build fails; the waiting request builds next
    assert second_in.wait(5)
    c = threading.Thread(target=request)  # arrives mid-build: must wait, not build alongside
    c.start()
    time.sleep(0.2)
    release_second.set()
    for t in (a, b, c):
        t.join(5)
    assert active[1] == 1 and len(calls) == 2
    assert sum(isinstance(r, Path) for r in results) == 2
    assert [f.suffix for f in frames.iterdir()] == [".jpg"]
    assert media.frame_locks == {}


@needs_ffmpeg
def test_frame_cache_drops_stale_stills_and_leftovers_once_per_run(
    client: TestClient, still_source: tuple[str, str, Path, Path]
) -> None:
    pid, sid, source, frames = still_source
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    key = ui.Media.key(source.resolve())
    frames.mkdir(parents=True)
    kept = frames / f"{key}-20-160.jpg"
    stale = frames / f"{'0' * 16}-20-160.jpg"  # a source that changed or left the project
    part = frames / f"{key}-30-160.123.part"
    other = frames.parent / "keep.txt"  # outside ui/frames: never touched
    for f in (kept, stale, part, other):
        f.write_bytes(b"x")
    url = f"/api/projects/{pid}/sources/{sid}/frame?at=0.5&t={client.headers['X-Cutpoint-Token']}"

    assert client.get(url, headers={"X-Cutpoint-Token": ""}).status_code == 200
    assert sorted(f.name for f in frames.iterdir()) == sorted([kept.name, f"{key}-50-160.jpg"])
    assert other.is_file()
    stale.write_bytes(b"x")  # pruned once per run: a later request leaves it
    assert client.get(url, headers={"X-Cutpoint-Token": ""}).status_code == 200
    assert stale.is_file()
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before


@needs_ffmpeg
def test_frame_cache_evicts_the_least_recently_used_beyond_the_cap(
    still_source: tuple[str, str, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, _, source, frames = still_source
    media = ui.Media()
    project = ui.project_dir(pid)
    monkeypatch.setattr(ui, "FRAME_CACHE_MAX", 2)
    monkeypatch.setattr(ui, "grab_images", lambda *_a, **_k: [b"\xff\xd8\xffjpeg"])
    first = media.frame(project, str(source), 0.1, 160)
    second = media.frame(project, str(source), 0.2, 160)
    old = time.time() - 100
    os.utime(first, (old, old))
    os.utime(second, (old + 1, old + 1))

    assert media.frame(project, str(source), 0.1, 160) == first  # a hit: now the newest
    third = media.frame(project, str(source), 0.3, 160)
    assert sorted(frames.iterdir()) == sorted([first, third])  # the least recently used went
    assert not second.exists()


def test_keyframe_grab_skips_to_the_keyframe_before_the_input() -> None:
    """#261: the filmstrip's stills decode only the keyframe at or before the time; the exact
    stills of the sequence thumbnails keep their accurate seek."""
    exact = ffmpeg_utils.grab_images_args("v.mp4", 5.0, width=160, quality=5)
    key = ffmpeg_utils.grab_images_args("v.mp4", 5.0, width=160, quality=5, keyframe=True)
    assert "-skip_frame" not in exact and "-noaccurate_seek" not in exact
    assert "-fps_mode" not in exact
    flags = ["-skip_frame", "nokey", "-noaccurate_seek"]
    i = key.index("-skip_frame")
    assert key[i : i + 3] == flags and i < key.index("-ss") < key.index("-i")
    # the keyframe lands before the seek point (a negative timestamp): passthrough keeps it
    j = key.index("-fps_mode")
    assert key[j : j + 2] == ["-fps_mode", "passthrough"] and key.index("-i") < j
    assert key[:i] + key[i + 3 : j] + key[j + 2 :] == exact


@needs_ffmpeg
def test_frame_key_mode_asks_for_the_keyframe_and_is_cached_apart_from_exact_stills(
    client: TestClient, still_source: tuple[str, str, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, sid, _, frames = still_source
    base = f"/api/projects/{pid}/sources/{sid}/frame?t={client.headers['X-Cutpoint-Token']}"
    calls: list[tuple[object, object]] = []

    def grab(*a: object, **k: object) -> list[bytes]:
        calls.append((a[1], k["keyframe"]))
        return [b"\xff\xd8\xff" + (b"key" if k["keyframe"] else b"exact")]

    def get(query: str):
        return client.get(f"{base}&{query}", headers={"X-Cutpoint-Token": ""})

    monkeypatch.setattr(ui, "grab_images", grab)
    key = get("at=0.5&key=1")
    assert key.status_code == 200 and key.content.endswith(b"key")
    exact = get("at=0.5")  # a keyframe still is never served for an exact one
    assert exact.status_code == 200 and exact.content.endswith(b"exact")
    assert get("at=0.5&key=1").content == key.content  # cached: ffmpeg is not run again
    assert calls == [(0.5, True), (0.5, False)]
    assert sorted(f.name.split("-", 1)[1] for f in frames.glob("*.jpg")) == [
        "50-160-k.jpg",
        "50-160.jpg",
    ]
    assert get("at=0.5&key=maybe").status_code == 422


@needs_ffmpeg
def test_a_real_keyframe_still_is_the_keyframe_at_or_before_the_time(
    client: TestClient, still_source: tuple[str, str, Path, Path]
) -> None:
    """The 1 s mpeg4 clip has a keyframe every 12 frames (0, 0.48 and 0.96 s at 25 fps): a
    keyframe still at 0.7 s is the frame at 0.48 s, not the one at 0.7 s nor the next keyframe
    (the default fps mode drops the keyframe's negative timestamp and writes the next one)."""
    pid, sid, source, _ = still_source
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    base = f"/api/projects/{pid}/sources/{sid}/frame?t={client.headers['X-Cutpoint-Token']}"

    def get(query: str):
        return client.get(f"{base}&{query}", headers={"X-Cutpoint-Token": ""})

    key = get("at=0.7&key=1")
    assert key.status_code == 200 and key.content[:3] == b"\xff\xd8\xff"
    assert key.content == get("at=0.48").content
    assert key.content != get("at=0.7").content
    assert key.content != get("at=0.96").content
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before


def test_a_huge_still_time_of_a_source_of_unknown_length_is_cut_not_an_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``at=1e308`` is finite, so the endpoint accepts it; with no known length to clamp to,
    its centiseconds overflowed (OverflowError, an HTTP 500). It is cut to FRAME_MAX_AT."""
    source = write(tmp_path / "a.mp4", b"x" * 10)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(ui, "probe", lambda _path: media_info(source, duration=0.0, **SD))
    starts: list[object] = []
    monkeypatch.setattr(
        ui, "grab_images", lambda *a, **_k: (starts.append(a[1]), [b"\xff\xd8\xffjpeg"])[1]
    )
    media = ui.Media()
    still = media.frame(project, str(source), 1e308, 160)
    cs = round(ui.FRAME_MAX_AT * 100)
    assert still is not None and still.name == f"{media.key(source)}-{cs}-160.jpg"
    assert starts == [cs / 100]


def test_rail_filmstrip_is_a_decorative_layer_under_the_markers() -> None:
    """Contract pin (#261): the trim rail's filmstrip is an aria-hidden layer first in the rail's
    markers, drawn by its own drawFilm (keyframe stills, key=1, on a 0.25 s x 2^k grid at most one
    per 78 px, exact stills below a 1 s step, asked for 150 ms after the view stops), loaded ahead of the sequence thumbnails
    under the same limit of 3, stale queued stills dropped by a generation counter, stopped after
    3 failures for a source and skipped by data savers. Each theme sets its --film-opacity.

    Check in a browser (no browser in the test suite): in Blue, Light and Dark, open a moment;
    stills fade in from the playhead outward; the gold selection, its handles, the dashed analyzed
    range, the playhead and the hatched gaps stay readable over them. Pan: the stills move with
    the view and only new ones load; zoom in and out: no still flashes. Resize the window: the
    strip refills. With a source that cannot be read, the ticks stay."""
    html = (ui.WEB_DIR / "index.html").read_text(encoding="utf-8")
    css = (ui.WEB_DIR / "app.css").read_text(encoding="utf-8")
    js = (ui.WEB_DIR / "app.js").read_text(encoding="utf-8")
    marks = re.search(
        r'<div class="rail" id="rail">\s*<div class="rail-marks">(.*?)</div>\s*<input', html, re.S
    )
    film_div = '<div class="rail-film" id="railFilm" aria-hidden="true"></div>'
    assert marks and marks[1].lstrip().startswith(film_div)
    assert html.count('id="railFilm"') == 1
    themes = {
        ":root": "0.55",
        ':root[data-theme="light"]': "0.35",
        ':root[data-theme="dark"]': "0.55",
    }
    for theme, opacity in themes.items():
        block = re.search(re.escape(theme) + r" \{(.*?)\n\}", css, re.S)
        assert block and f"--film-opacity: {opacity};" in block[1], theme
    assert re.search(r"^\.rail-film \{[^}]*opacity: var\(--film-opacity\);", css, re.M)
    assert "future filmstrip" not in css
    draw = re.search(r"^function drawRail\(c = current\(\)\) \{(.*?)^\}", js, re.M | re.S)
    assert draw and draw[1].count("drawFilm(c);") == 1 and "railFilm" not in draw[1]
    film = re.search(r"^function drawFilm\(c = current\(\)\) \{(.*?)^\}", js, re.M | re.S)
    assert film and "$('#railFilm')" in film[1] and "filmGen += 1;" in film[1]
    assert "setTimeout(loadFilm, FILM_WAIT)" in film[1]
    for const in (
        "const FILM_TILE = 78;",
        "const FILM_STEP = 0.25;",
        "const FILM_WAIT = 150;",
        "const FILM_FAILS = 3;",
        "const THUMB_LOADS = 3;",
    ):
        assert const in js, const
    assert "&w=${THUMB_W}${key ? '&key=1' : ''}`" in js
    assert "const FILM_EXACT = 1;" in js and "const key = step >= FILM_EXACT;" in film[1]
    gaps = re.search(
        r"^\.rail-marks:has\(> \.rail-film > img\.ok\) > \.rail-gaps > span \{", css, re.M
    )
    assert gaps  # Light shades the rail's gaps darker while stills show
    assert "prefers-reduced-data: reduce" in js and "navigator.connection?.saveData" in js
    assert "new ResizeObserver(() => drawFilm()).observe($('#rail'))" in js
    pump = re.search(r"^function pumpThumbs\(\) \{(.*?)^\}", js, re.M | re.S)
    assert pump and pump[1].index("filmQueue.shift()") < pump[1].index("thumbQueue.shift()")
    assert "next.gen !== filmGen" in pump[1]


STILL_BT2020 = {"color_primaries": "bt2020", "color_space": "bt2020nc", "pix_fmt": "yuv420p10le"}


TO_JPEG_709 = "colorspace=all=bt470bg:iall=bt709:range=pc:fast=1"
SD = {"width": 640, "height": 480}


@pytest.mark.parametrize(
    ("over", "kind"),
    [
        ({"vcodec": "hevc", "color_transfer": "smpte2084", **STILL_BT2020}, "hdr"),
        ({"vcodec": "av1", "color_transfer": "arib-std-b67", **STILL_BT2020}, "hdr"),
        ({"vcodec": "hevc"}, "bt709"),  # HD without a matrix tag is BT.709, as exports read it
        ({"color_transfer": "bt709", "color_primaries": "bt709", "color_space": "bt709"}, "bt709"),
        (SD, "bt601"),  # BT.601 (a JPEG already is): today's command and file name
        ({"color_space": "smpte170m", **SD}, "bt601"),
        ({"color_space": "bt470bg", "color_range": "pc", **SD}, "bt601"),
    ],
)
def test_still_is_converted_to_what_the_export_shows_and_cached_under_a_new_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, over: dict[str, object], kind: str
) -> None:
    """#412, #441: a JPEG is BT.601 full range, so stills of HLG/PQ sources (an AV1 one the
    browser plays as is too) go through the export's sdr_filter and then to BT.601, and BT.709
    ones to BT.601; the filter's hash ends their name, so a still cached before is never served.
    BT.601 stills keep their command and name."""
    source = write(tmp_path / "a.mp4", b"x" * 10)
    project = tmp_path / "project"
    project.mkdir()
    info = media_info(source, duration=10.0, **over)
    monkeypatch.setattr(ui, "probe", lambda _path: info)
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    calls: list[dict[str, object]] = []

    def grab(*_a: object, **k: object) -> list[bytes]:
        calls.append(k)
        return [b"\xff\xd8\xffjpeg"]

    monkeypatch.setattr(ui, "grab_images", grab)
    media = ui.Media()
    still = media.frame(project, str(source), 0.5, 160)

    key = media.key(source)
    color = {
        "hdr": f"{encode.sdr_filter(info)},{TO_JPEG_709}",
        "bt709": TO_JPEG_709,
        "bt601": "",
    }[kind]
    assert color.startswith("zscale=") if kind == "hdr" else True
    assert [k["color"] for k in calls] == [color]
    tag = f"-{hashlib.sha1(color.encode()).hexdigest()[:8]}" if color else ""
    assert still is not None and still.name == f"{key}-50-160{tag}.jpg"
    media.prune_frames(project, [str(source)])  # the tagged name still belongs to its source
    assert still.is_file()


def test_ui_on_a_taken_port_fails_cleanly_without_printing_a_url(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uvicorn = pytest.importorskip("uvicorn")
    monkeypatch.setattr(uvicorn.Server, "run", lambda *_a, **_k: pytest.fail("server started"))
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        port = held.getsockname()[1]
        with caplog.at_level(logging.ERROR):
            assert cli.main(["ui", "--port", str(port)]) == 1
    assert f"Port {port} is already in use on 127.0.0.1; pass --port <n>" in caplog.text
    assert "Traceback" not in caplog.text
    out = capsys.readouterr()
    assert "http://" not in out.out + out.err


def test_show_folder_opens_only_the_projects_own_source_folder(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The folder comes from the project, never the request; token, 404 and 501 cases."""
    opened: list[Path] = []
    monkeypatch.setattr(ui, "folder_opener", lambda: opened.append)
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    assert client.get("/api/session").json()["can_reveal"] is True

    assert (
        client.post(f"/api/projects/{pid}/reveal", headers={"X-Cutpoint-Token": ""}).status_code
        == 403
    )
    assert opened == []
    # A path in the query or body is ignored: the project's folder is what opens.
    res = client.post(
        f"/api/projects/{pid}/reveal", params={"path": str(tmp_path)}, json={"path": str(tmp_path)}
    )
    assert res.status_code == 200, res.text
    assert opened == [folder] and res.json() == {"path": str(folder)}
    assert client.post("/api/projects/nope/reveal").status_code == 404
    assert client.post("/api/projects/..%2F..%2Fx/reveal").status_code != 200
    assert opened == [folder]

    # A source that is one video file opens its folder.
    meta = cache.load_json(ui.project_dir(pid) / "ui" / "meta.json")
    cache.save_json(
        ui.project_dir(pid) / "ui" / "meta.json", {**meta, "source": str(folder / "a.mp4")}
    )
    opened.clear()
    assert client.post(f"/api/projects/{pid}/reveal").status_code == 200
    assert opened == [folder]

    # A picked video that was deleted while its folder remains still opens that folder.
    cache.save_json(
        ui.project_dir(pid) / "ui" / "meta.json", {**meta, "source": str(folder / "gone.mp4")}
    )
    opened.clear()
    assert client.post(f"/api/projects/{pid}/reveal").status_code == 200
    assert opened == [folder]

    shutil.rmtree(folder)
    opened.clear()
    res = client.post(f"/api/projects/{pid}/reveal")
    assert res.status_code == 404 and "no longer exists" in res.json()["detail"]
    assert opened == []

    folder.mkdir()
    monkeypatch.setattr(ui, "folder_opener", lambda: None)
    assert client.get("/api/session").json()["can_reveal"] is False
    res = client.post(f"/api/projects/{pid}/reveal")
    assert (
        res.status_code == 501
        and res.json()["detail"] == "Opening folders isn't available on this machine"
    )


def test_show_folder_for_a_command_line_project_uses_its_recorded_sources(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Path] = []
    monkeypatch.setattr(ui, "folder_opener", lambda: opened.append)
    footage = tmp_path / "footage"
    for name in ("day1/a.mp4", "day2/b.mp4"):
        (footage / name).parent.mkdir(parents=True, exist_ok=True)
        (footage / name).write_bytes(b"x")
    p = ui.projects_dir() / "cli-proj"
    p.mkdir(parents=True)
    cache.save_json(
        p / "highlights.json",
        {
            "settings": {"inputs": [str(footage)]},
            "sources": [{"path": str(footage / n)} for n in ("day1/a.mp4", "day2/b.mp4")],
            "candidates": [],
        },
    )
    assert client.post("/api/projects/cli-proj/reveal").status_code == 200
    assert opened == [footage]


def test_show_folder_is_404_when_the_picked_folder_is_gone_or_unreadable(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deleted source folder is never swapped for its parent; OS errors are a 404, not a 500."""
    opened: list[Path] = []
    monkeypatch.setattr(ui, "folder_opener", lambda: opened.append)
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    meta = cache.load_json(ui.project_dir(pid) / "ui" / "meta.json")
    assert Path(meta["source"]) == folder

    def unreadable(real: Callable[..., bool]) -> Callable[..., bool]:
        def check(self: Path, *args: object, **kwargs: object) -> bool:
            if self == folder or folder in self.parents:
                raise PermissionError(13, "Permission denied", str(self))
            return real(self, *args, **kwargs)

        return check

    with monkeypatch.context() as m:
        for name in ("is_file", "is_dir", "exists"):
            m.setattr(Path, name, unreadable(getattr(Path, name)))
        res = client.post(f"/api/projects/{pid}/reveal")
    assert res.status_code == 404 and "no longer exists" in res.json()["detail"]
    assert opened == []

    shutil.rmtree(folder)  # a picked folder that is gone is never swapped for its parent
    res = client.post(f"/api/projects/{pid}/reveal")
    assert res.status_code == 404 and "no longer exists" in res.json()["detail"]
    assert opened == []


@pytest.fixture
def reveal_env(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[Path], str, Path, Path, str]:
    """A project whose folder opener records what it opens.

    Returns (opened folders, project id, source folder, project folder, reveal URL).
    """
    opened: list[Path] = []
    monkeypatch.setattr(ui, "folder_opener", lambda: opened.append)
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    return opened, pid, folder, ui.project_dir(pid), f"/api/projects/{pid}/reveal"


def test_show_folder_targets_open_the_latest_clips_and_the_final_folder(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reveal_env: tuple[list[Path], str, Path, Path, str],
) -> None:
    """``target`` picks one of the project's folders; the path still never comes from the client."""
    opened, _pid, folder, p, reveal = reveal_env

    assert (
        client.post(
            reveal, params={"target": "clips"}, headers={"X-Cutpoint-Token": ""}
        ).status_code
        == 403
    )
    for target in ("clips", "outputs"):
        res = client.post(reveal, params={"target": target})
        assert res.status_code == 404, target
    for target in ("../x", str(tmp_path), "Clips", ""):
        assert client.post(reveal, params={"target": target}).status_code == 400, target
    assert opened == []

    # An export: clips opens its folder, outputs is still 404 until something is assembled.
    (p / "clips_vertical").mkdir()
    cache.save_json(p / "clips_vertical" / "manifest.json", {"clips": []})
    cache.save_json(p / "ui" / "exports.json", {"exports": [{"dir": "clips_vertical"}]})
    res = client.post(reveal, params={"target": "clips", "path": str(tmp_path)})
    assert res.status_code == 200 and res.json() == {"path": str(p / "clips_vertical")}
    assert opened == [p / "clips_vertical"]
    assert client.post(reveal, params={"target": "outputs"}).status_code == 404

    cache.save_json(
        p / "ui" / "exports.json", {"exports": [{"dir": "clips_vertical", "final": "final.mp4"}]}
    )
    (p / "final.mp4").write_bytes(b"x")
    opened.clear()
    res = client.post(reveal, params={"target": "outputs"}, json={"path": str(tmp_path)})
    assert res.status_code == 200 and res.json() == {"path": str(p)}
    assert opened == [p]

    # No target still means the sources, as before.
    opened.clear()
    assert client.post(reveal).status_code == 200
    assert client.post(reveal, params={"target": "sources"}).status_code == 200
    assert opened == [folder, folder]

    def broken(_: Path) -> None:
        raise OSError(5, "Access is denied")

    monkeypatch.setattr(ui, "folder_opener", lambda: broken)
    res = client.post(reveal, params={"target": "outputs"})
    assert res.status_code == 500 and "Access is denied" in res.json()["detail"]


@pytest.mark.parametrize("bad", ["..", ".", ".hidden"])
def test_exports_json_cannot_point_outside_the_project(
    client: TestClient, reveal_env: tuple[list[Path], str, Path, Path, str], bad: str
) -> None:
    opened, _pid, _folder, p, reveal = reveal_env
    # Manifests and videos where a "." / ".." / dot-leading name would resolve to.
    for folder in (p, p.parent, p / ".hidden"):
        folder.mkdir(exist_ok=True)
        cache.save_json(folder / "manifest.json", {"clips": []})
    (p / ".hidden.mp4").write_bytes(b"x")

    def write(*items: dict[str, str]) -> None:
        cache.save_json(p / "ui" / "exports.json", {"exports": list(items)})

    write({"dir": bad})
    assert client.post(reveal, params={"target": "clips"}).status_code == 404

    # A bad final is ignored; the good export stays usable.
    (p / "clips_ui_1").mkdir()
    cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": []})
    write({"dir": "clips_ui_1", "final": ".hidden.mp4" if bad == ".hidden" else bad})
    assert client.post(reveal, params={"target": "outputs"}).status_code == 404

    # A bad entry after a legitimate one is dropped, not used as the current export.
    (p / "final.mp4").write_bytes(b"x")
    write({"dir": "clips_ui_1", "final": "final.mp4"}, {"dir": bad})
    assert client.post(reveal, params={"target": "clips"}).json() == {"path": str(p / "clips_ui_1")}
    assert client.post(reveal, params={"target": "outputs"}).json() == {"path": str(p)}
    assert opened == [p / "clips_ui_1", p]


@pytest.mark.parametrize("kind", ["dir", "final"])
def test_exports_json_cannot_point_through_a_link_outside_the_project(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    opened: list[Path] = []
    monkeypatch.setattr(ui, "folder_opener", lambda: opened.append)
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    reveal = f"/api/projects/{pid}/reveal"
    outside = tmp_path / "outside"
    outside.mkdir()
    cache.save_json(outside / "manifest.json", {"clips": []})
    (outside / "final.mp4").write_bytes(b"x")
    (p / "clips_ui_1").mkdir()
    cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": []})
    (p / "final.mp4").write_bytes(b"x")

    def write(*items: dict[str, str]) -> None:
        cache.save_json(p / "ui" / "exports.json", {"exports": list(items)})

    good = {"dir": "clips_ui_1", "final": "final.mp4"}
    if kind == "dir":
        link_dir(p / "clips_link", outside)
        bad = {"dir": "clips_link", "final": "final.mp4"}
        write(bad)
        assert client.post(reveal, params={"target": "clips"}).status_code == 404
    else:
        try:
            (p / "final_link.mp4").symlink_to(outside / "final.mp4")
        except (OSError, NotImplementedError):
            pytest.skip("cannot create a file symlink here")
        bad = {"dir": "clips_ui_1", "final": "final_link.mp4"}
        write(bad)

    def listed() -> list[str]:
        return [o["name"] for o in client.get(f"/api/projects/{pid}").json()["outputs"]]

    # dir: an earlier legitimate export takes over; final: the latest export stays current,
    # its linked final is never listed or served (the project's own final.mp4 still is).
    write(good, bad)
    if kind == "dir":
        assert client.post(reveal, params={"target": "clips"}).json() == {
            "path": str(p / "clips_ui_1")
        }
        assert opened == [p / "clips_ui_1"]
    else:
        assert listed() == ["final.mp4"]
        assert client.get(f"/api/projects/{pid}/outputs/final_link.mp4").status_code == 404
        write(good)
        assert listed() == ["final.mp4"]
        assert client.post(reveal, params={"target": "outputs"}).json() == {"path": str(p)}
        # Timelines next to the final are served only when they stay inside the project.
        (outside / "final.fcpxml").write_text("x")
        (p / "final.fcpxml").symlink_to(outside / "final.fcpxml")
        assert client.get(f"/api/projects/{pid}/outputs/final.fcpxml").status_code == 404
        assert listed() == ["final.mp4"]  # the linked timeline stays excluded


def test_export_manifest_symlinked_outside_the_project_is_not_served(
    client: TestClient, tmp_path: Path
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    outside = tmp_path / "outside"
    outside.mkdir()
    cache.save_json(outside / "manifest.json", {"clips": [{"label": "SECRET-OUTSIDE"}]})
    (p / "clips_ui_1").mkdir()
    try:
        (p / "clips_ui_1" / "manifest.json").symlink_to(outside / "manifest.json")
    except (OSError, NotImplementedError):
        pytest.skip("cannot create a file symlink here")
    cache.save_json(p / "ui" / "exports.json", {"exports": [{"dir": "clips_ui_1"}]})
    res = client.get(f"/api/projects/{pid}")
    assert res.status_code == 200
    assert "SECRET-OUTSIDE" not in res.text


def test_fresh_name_treats_a_broken_symlink_as_taken(tmp_path: Path) -> None:
    """A dangling link is a taken name: nothing may be written through it."""
    try:
        (tmp_path / "final.mp4").symlink_to(tmp_path / "missing-target")
    except (OSError, NotImplementedError):
        pytest.skip("cannot create a symlink here")
    assert ui.fresh_name(tmp_path, "final", ("", ".mp4", ".fcpxml")) == "final-2"


def test_folder_opener_runs_the_platform_tool_without_a_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[object, dict]] = []

    class FakeProc:
        def wait(self) -> int:
            return 0

    def popen(args, **kw):
        calls.append((args, kw))
        return FakeProc()

    monkeypatch.setattr(ui.subprocess, "Popen", popen)
    monkeypatch.setattr(ui.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    monkeypatch.setattr(sys, "platform", "linux")
    assert ui.folder_opener() is None  # no display: nobody would see the window
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    ui.folder_opener()(tmp_path)
    monkeypatch.setattr(sys, "platform", "darwin")
    ui.folder_opener()(tmp_path)
    assert [c[0] for c in calls] == [
        ["/usr/bin/xdg-open", str(tmp_path)],
        ["/usr/bin/open", str(tmp_path)],
    ]
    assert all(not kw.get("shell") for _, kw in calls)
    assert all(kw["stdout"] == kw["stderr"] == subprocess.DEVNULL for _, kw in calls)

    monkeypatch.setattr(ui.shutil, "which", lambda name: None)
    assert ui.folder_opener() is None  # macOS without `open`, or Linux without xdg-open

    started: list[str] = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ui.os, "startfile", started.append, raising=False)
    ui.folder_opener()(tmp_path)
    assert started == [str(tmp_path)] and len(calls) == 2


# --------------------------------------------------------------------------- native picker
class FakeDialog:
    """A stand-in for the dialog process: fixed output, or blocks until released."""

    def __init__(self, out: str = "", code: int = 0, err: str = "") -> None:
        self.out, self.returncode, self.err = out, code, err
        self.calls: list[list[str]] = []
        self.opened = threading.Event()
        self.release = threading.Event()
        self.block = False

    def __call__(self, args, **kw):
        self.calls.append(args)
        outer = self

        class Proc:
            returncode = outer.returncode

            def communicate(self):
                outer.opened.set()
                if outer.block:
                    assert outer.release.wait(30)
                return outer.out, outer.err

            def kill(self):
                outer.release.set()

        return Proc()


@pytest.fixture
def dialog(monkeypatch: pytest.MonkeyPatch) -> FakeDialog:
    fake = FakeDialog()
    monkeypatch.setattr(ui.subprocess, "Popen", fake)
    monkeypatch.setattr(ui, "path_picker", lambda: lambda kind, start: (["dlg", kind], 1))
    return fake


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

    monkeypatch.setattr(ui.subprocess, "Popen", broken)
    res = client.post("/api/pick", params={"kind": "file"})
    assert res.status_code == 500 and "No such file" in res.json()["detail"]
    # The failure released the lock, so the next dialog can open.
    monkeypatch.setattr(ui.subprocess, "Popen", dialog)
    dialog.out = ""
    assert client.post("/api/pick", params={"kind": "file"}).status_code == 200

    monkeypatch.setattr(ui, "path_picker", lambda: None)
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

    monkeypatch.setattr(ui, "path_picker", lambda: build)
    monkeypatch.setattr(ui.subprocess, "Popen", FakeDialog())
    client.post("/api/pick", params={"kind": "file", "start": str(tmp_path / "missing" / "x")})
    client.post("/api/pick", params={"kind": "file", "start": "relative"})
    assert starts == [tmp_path, None]


def test_path_picker_builds_the_platform_dialog_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    present: set[str] = set()
    monkeypatch.setattr(ui.shutil, "which", lambda n: f"/bin/{n}" if n in present else None)
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
    monkeypatch.setattr(ui.importlib.util, "find_spec", lambda name: None)
    assert ui.path_picker() is None  # Python without tkinter
    monkeypatch.setattr(ui.importlib.util, "find_spec", lambda name: object())
    argv, cancel = ui.path_picker()("folder", tmp_path)
    assert argv[:3] == [sys.executable, "-I", "-c"]
    assert argv[4:6] == ["folder", str(tmp_path)] and "*.mp4" in argv[6] and cancel is None
    assert ui.path_picker()("file", None)[0][5] == ""


@pytest.mark.parametrize(("platform", "var"), [("win32", "SSH_CONNECTION"), ("darwin", "SSH_TTY")])
def test_path_picker_is_off_in_an_ssh_session_on_windows_and_macos(
    monkeypatch: pytest.MonkeyPatch, platform: str, var: str
) -> None:
    monkeypatch.setattr(ui.shutil, "which", lambda n: f"/bin/{n}")
    monkeypatch.setattr(ui.importlib.util, "find_spec", lambda name: object())
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
    assert config.vision_default() is True  # the CLI default too

    assert client.put("/api/settings", json={"vision": False}).json()["vision"] is False
    assert config.vision_default() is False

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


def peaks_job(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: dict
) -> tuple[object, list[list[str]]]:
    """Start ``body``'s job on a command-line project that has a review and an export."""
    p, seg = cli_project(tmp_path)
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    (p / "clips_ui_1").mkdir()
    cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": []})
    (p / "clips_ui_1" / "001_a.mp4").write_bytes(b"x")
    cache.save_json(p / "ui" / "exports.json", {"exports": [{"dir": "clips_ui_1"}]})
    captured: list[list[str]] = []
    monkeypatch.setattr(ui.Jobs, "start", fake_start(captured))
    return client.post("/api/projects/cli/jobs", json=body), captured


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


@pytest.mark.parametrize("action", ["clips", "assemble"])
@pytest.mark.parametrize("value", ["loud", "", None, True, ["light"]])
def test_clips_and_assemble_jobs_reject_an_unknown_compress_peaks(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, action: str, value: object
) -> None:
    body = {"action": action, "compress_peaks": value}
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

    monkeypatch.setattr(ui, "set_config_values", broken)
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

    monkeypatch.setattr(ui, "set_config_values", broken)
    res = client.put("/api/settings", json={"projects_dir": str(existing)})
    assert res.status_code == 500 and (existing / "f.txt").exists()


def test_check_projects_dir_removes_its_own_folder_when_the_write_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(*a, **kw):
        raise PermissionError(13, "Permission denied")

    new = tmp_path / "probe"
    monkeypatch.setattr(ui.tempfile, "TemporaryFile", denied)
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
    client: TestClient, home: Path, tmp_path: Path, text: str, error: str
) -> None:
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
    ["{oops", '{"base": "funny", "weights": [1, 2]}', '{"base": "funny", "weights": {"x": null}}'],
    ids=["invalid-json", "weights-not-object", "weight-null"],
)
def test_a_broken_learned_style_is_left_out_with_a_warning(
    client: TestClient, home: Path, caplog: pytest.LogCaptureFixture, content: str
) -> None:
    styles = home / "styles"
    styles.mkdir(parents=True)
    (styles / "broken.json").write_text(content, encoding="utf-8")
    (styles / "mine.json").write_text('{"base": "funny"}', encoding="utf-8")
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

    monkeypatch.setattr(ui, "check_projects_dir", check_then_start_job)
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
    real_write = ui.set_config_values
    attempt: dict = {}

    def write_while_another_tab_starts_a_job(values):
        def start() -> None:
            attempt["job"] = jobs.start("p", "analyze", SLEEP_JOB)
            attempt["wrote"] = wrote.is_set()
            done.set()

        wrote, done = threading.Event(), threading.Event()
        attempt["wrote_event"] = wrote
        threading.Thread(target=start, daemon=True).start()
        attempt["overlapped"] = done.wait(0.5)  # True: the job started mid-write
        real_write(values)
        wrote.set()

    monkeypatch.setattr(ui, "set_config_values", write_while_another_tab_starts_a_job)
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
        config.set_config_values({"projects_dir": str(other)})  # a second tab saves the settings
        return real_cmd(*args)

    monkeypatch.setattr(ui, "cutpoint_cmd", cmd_while_the_folder_changes)
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


@pytest.fixture
def hevc_project(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, Path, Path, Path]:
    """A project with an HEVC source (a browser cannot play it) and an empty ``outside`` folder.

    Returns (project id, source folder, project folder, the outside folder).
    """
    probed = media_info(vcodec="hevc", duration=10.0)
    monkeypatch.setattr(ui, "probe", lambda _path: probed)
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    outside = tmp_path / "outside"
    outside.mkdir()
    return pid, folder, ui.project_dir(pid), outside


def media_link(
    client: TestClient, hevc_project: tuple[str, Path, Path, Path], sub: str
) -> tuple[str, str, Path, Path, str]:
    """A project with a source the browser cannot play, ``ui/<sub>`` linked to a folder outside.

    Returns (project id, source id, project folder, the outside folder, the source's key).
    """
    pid, folder, p, outside = hevc_project
    (p / "ui").mkdir(exist_ok=True)
    link_dir(p / "ui" / sub, outside)
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    return pid, sid, p, outside, ui.Media.key((folder / "a.mp4").resolve()) or ""


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

    monkeypatch.setattr(ui, "grab_images", never)
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


def cli_project(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    """A command-line project: ``highlights.json`` only, no ``ui`` folder. Returns it and a segment."""
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"x")
    p = ui.projects_dir() / "cli"
    p.mkdir(parents=True)
    hl = {"sources": [{"path": str(src), "duration": 10.0}], "settings": {"inputs": [str(src)]}}
    cache.save_json(p / "highlights.json", hl)
    seg = {"id": "h001", "source": str(src), "start": 1, "end": 4.5, "label": "good"}
    return p, seg


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

        monkeypatch.setattr(ui, "cutpoint_cmd", cmd_after_ui_changes)
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
            config.set_config_values({"projects_dir": str(tmp_path / "other")})
            return real_cmd(*args)

        monkeypatch.setattr(ui, "cutpoint_cmd", cmd_while_the_folder_changes)
    elif why == "job running":
        busy = jobs.start("x", "analyze", [sys.executable, "-c", "import time; time.sleep(60)"])
    else:

        def no_spawn(_args: list[str]) -> None:
            raise OSError("not found")

        monkeypatch.setattr(ui, "spawn", no_spawn)
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
    monkeypatch.setattr(ui, "proxy_args", lambda _real, tmp, _n, **_kw: [str(tmp)])

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

    monkeypatch.setattr(ui, "spawn", Encoder)
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
    monkeypatch.setattr(ui, "spawn", lambda args, **_kw: seen.append(args) or proc)
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "assemble"})
    assert res.status_code == 200, res.text
    assert client.app.state.jobs.get(res.json()["id"]).done.wait(10)
    assert seen[0][seen[0].index("--out") + 1] == str(p / "final_ui_1-2.mp4")
    assert (p / "final_ui_1.srt").read_text(encoding="utf-8") == "keep me"


class FakeProc:
    """A job process that stays alive until ``release`` is set, then prints ``lines``, exits ``code``."""

    def __init__(self, lines: list[str] | None = None, code: int = 0) -> None:
        self.release = threading.Event()
        self.stdout = self
        self.lines = [f"{ln}\n" for ln in lines or []]
        self.code = code

    def __iter__(self):
        self.release.wait(10)
        return iter(self.lines)

    def wait(self, timeout: float | None = None) -> int:
        return self.code

    def fake_spawn(self, monkeypatch: pytest.MonkeyPatch) -> FakeProc:
        """Make jobs start this process (already exited) instead of a real interpreter.

        ``ui.spawn`` runs a real child at below-normal priority, which is slow on a busy machine.
        """
        self.release.set()
        monkeypatch.setattr(ui, "spawn", lambda args, **_kw: self)
        return self


def test_a_cancelled_job_reads_cancelling_until_the_process_has_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = FakeProc()
    killed = threading.Event()
    monkeypatch.setattr(ui, "spawn", lambda args: proc)
    monkeypatch.setattr(ui, "kill_tree", lambda _p: killed.set())  # the exit comes later
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

    monkeypatch.setattr(ui, "spawn", lambda args: proc)
    monkeypatch.setattr(ui, "kill_tree", slow_kill)
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


def record_export(p: Path, final: str | None) -> None:
    """An export record for ``p`` with a clips folder, whose final is ``final`` (None: not yet)."""
    (p / "clips_ui_1").mkdir(exist_ok=True)
    cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": []})
    cache.save_json(p / "ui" / "exports.json", {"exports": [{"dir": "clips_ui_1", "final": final}]})


def touch(path: Path, age: int) -> None:
    """Create ``path`` as a file last modified ``age`` seconds ago."""
    path.write_bytes(b"x")
    then = time.time() - age
    os.utime(path, (then, then))


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
    monkeypatch.setattr(ui, "folder_opener", lambda: opened.append)
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
    monkeypatch.setattr(ui, "probe", lambda _path: media_info(vcodec="hevc", duration=10.0))
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

    monkeypatch.setattr(ui, "spawn", Failing)
    media = client.app.state.media
    media.prepare(p, str(src))
    media.pool.submit(int).result(30)  # the one worker has finished the build
    assert media.status(p, str(src))[0] == "failed"
    assert not (p / "ui").exists()


# --- strain: priority, connection resets, proxy encoder ---


@pytest.mark.slow
def test_spawn_runs_children_below_normal_priority_and_asks_for_progress_lines() -> None:
    """Jobs and proxy encodes (and what they start) yield the CPU to the desktop."""
    if os.name == "nt":
        code = (
            "import ctypes; k = ctypes.windll.kernel32; "
            "k.GetCurrentProcess.restype = ctypes.c_void_p; "
            "k.GetPriorityClass.argtypes = [ctypes.c_void_p]; "
            "import os; print(k.GetPriorityClass(k.GetCurrentProcess()), "
            "os.environ.get('CUTPOINT_PROGRESS'))"
        )
        want = subprocess.BELOW_NORMAL_PRIORITY_CLASS
    else:
        code = "import os; print(os.nice(0), os.environ.get('CUTPOINT_PROGRESS'))"
        want = min(os.nice(0) + 10, 19)
    proc = ui.spawn([sys.executable, "-c", code])
    out = proc.communicate(timeout=30)[0]
    assert proc.returncode == 0, out
    priority, progress = out.split()[-2:]
    assert int(priority) == want
    assert progress == "1"  # the CLI is asked for PROGRESS lines


@pytest.mark.skipif(os.name != "nt", reason="Windows creation flags")
def test_spawn_keeps_its_own_process_group_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl+Break reaches a job only in its own process group: the priority flag is added to it."""
    seen: dict = {}

    def popen(args: list[str], **kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(pid=1)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(ui, "_join_tree_job", lambda _proc: None)
    ui.spawn(["x"])
    flags = seen["creationflags"]
    assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
    assert flags & subprocess.BELOW_NORMAL_PRIORITY_CLASS


@pytest.mark.parametrize("low", [True, False])
def test_spawn_lowers_the_priority_unless_asked_not_to(
    monkeypatch: pytest.MonkeyPatch, low: bool
) -> None:
    """Window proxies (the user waits on them) start at normal priority, still in their own group."""
    seen: dict = {}

    def popen(args: list[str], **kwargs: object) -> SimpleNamespace:
        seen.update(kwargs, args=args)
        return SimpleNamespace(pid=1)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(ui, "_join_tree_job", lambda _proc: None)
    monkeypatch.setattr(ui.shutil, "which", lambda name: f"/usr/bin/{name}")
    ui.spawn(["ffmpeg"], low_priority=low)
    if os.name == "nt":
        assert seen["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
        assert bool(seen["creationflags"] & subprocess.BELOW_NORMAL_PRIORITY_CLASS) is low
    else:
        assert seen["start_new_session"]
        assert seen["args"] == (["/usr/bin/nice", "-n", "10", "ffmpeg"] if low else ["ffmpeg"])


@pytest.mark.parametrize(
    ("exc", "passed_on"),
    [
        (ConnectionResetError(10054, "reset"), False),
        (ConnectionAbortedError(10053, "aborted"), False),
        (BrokenPipeError(32, "broken pipe"), True),
        (OSError(5, "other"), True),
        (RuntimeError("bug"), True),
        (None, True),
    ],
)
def test_connection_resets_are_not_logged_but_other_loop_errors_are(
    exc: BaseException | None, passed_on: bool
) -> None:
    """A browser cancelling a range request (seek, source switch) is not an error."""
    seen: list[dict] = []
    loop = SimpleNamespace(default_exception_handler=seen.append)
    context = {"message": "Exception in callback", "exception": exc}
    if exc is None:
        del context["exception"]
    ui.quiet_resets(loop, context)  # type: ignore[arg-type]
    assert seen == ([context] if passed_on else [])


def test_the_server_loop_uses_the_quiet_reset_handler() -> None:
    app = ui.create_app(PORT)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:

        async def handler() -> object:
            return asyncio.get_running_loop().get_exception_handler()

        assert c.portal.call(handler) is ui.quiet_resets


@pytest.mark.parametrize("kind", [None, "nvenc", "vaapi"])
def test_proxy_has_short_keyframe_interval_and_at_most_30_fps(kind: str | None) -> None:
    enc = ui.VideoEncoder(kind, "/dev/dri/renderD128") if kind else None
    args = ui.proxy_args(Path("in.mkv"), Path("out.mp4"), 1, encoder=enc)
    assert args[args.index("-g") + 1] == "30"
    assert args[args.index("-fpsmax") + 1] == "30"
    codec = args[args.index("-c:v") + 1]
    assert codec == {None: "libx264", "nvenc": "h264_nvenc", "vaapi": "h264_vaapi"}[kind]
    assert ("-maxrate" in args) is (kind == "nvenc")  # NVENC's size is capped near libx264's
    # VAAPI frames are uploaded after the scale and never get a -pix_fmt
    assert args[args.index("-vf") + 1].endswith("hwupload") is (kind == "vaapi")
    assert ("-pix_fmt" in args) is (kind != "vaapi")
    assert args[-1] == "out.mp4"


class ProxyEncoder:
    """A fake proxy ffmpeg: records its command; fails when it is NVENC and ``nvenc_fails``."""

    calls: list[list[str]]
    nvenc_fails = False

    def __init__(self, args: list[str], **_kw: object) -> None:
        self.calls.append(args)
        self.returncode = 1 if self.nvenc_fails and "h264_nvenc" in args else 0
        if self.returncode == 0:
            Path(args[-1]).write_bytes(b"proxy")

    @property
    def stdout(self) -> list[str]:
        return [] if self.returncode == 0 else ["nvenc: driver error\n"]

    def wait(self) -> int:
        return self.returncode


@pytest.fixture
def proxy_encoder(monkeypatch: pytest.MonkeyPatch) -> type[ProxyEncoder]:
    monkeypatch.setattr(ProxyEncoder, "calls", [], raising=False)
    monkeypatch.setattr(ui, "spawn", ProxyEncoder)
    return ProxyEncoder


def build_proxy(client: TestClient, p: Path, src: Path) -> str:
    media = client.app.state.media
    media.prepare(p, str(src))
    media.pool.submit(int).result(30)  # the one worker has finished the build
    return media.status(p, str(src))[0]


@pytest.mark.parametrize(
    ("kind", "codec"),
    [
        (None, "libx264"),  # no GPU encoder works here, or the user chose the CPU
        ("nvenc", "h264_nvenc"),
        ("vaapi", "h264_vaapi"),
        ("amf", "libx264"),  # its proxy size is unmeasured
    ],
)
def test_proxy_uses_the_gpu_encoder_exports_use_else_libx264(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    proxy_encoder: type[ProxyEncoder],
    kind: str | None,
    codec: str,
) -> None:
    enc = ui.VideoEncoder(kind or "cpu", "/dev/dri/renderD128" if kind == "vaapi" else None)
    monkeypatch.setattr(ui, "video_encoder", lambda: enc)
    _, folder, p, _ = hevc_project
    assert build_proxy(client, p, folder / "a.mp4") == "ready"
    [args] = proxy_encoder.calls
    assert args[args.index("-c:v") + 1] == codec


def test_a_failed_nvenc_proxy_is_built_again_with_libx264(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    proxy_encoder: type[ProxyEncoder],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The probe encodes 0.2 s of black; a driver can still fail on the real file."""
    monkeypatch.setattr(ui, "video_encoder", lambda: ui.VideoEncoder("nvenc"))
    monkeypatch.setattr(proxy_encoder, "nvenc_fails", True)
    _, folder, p, _ = hevc_project
    (folder / "b.mp4").write_bytes(b"y")
    with caplog.at_level(logging.WARNING):
        assert build_proxy(client, p, folder / "a.mp4") == "ready"
    codecs = [args[args.index("-c:v") + 1] for args in proxy_encoder.calls]
    assert codecs == ["h264_nvenc", "libx264"]
    assert "retrying with libx264" in caplog.text
    assert not list(p.rglob("*.part.mp4"))
    # libx264 worked where NVENC failed: the next proxy goes straight to libx264
    assert build_proxy(client, p, folder / "b.mp4") == "ready"
    assert [args[args.index("-c:v") + 1] for args in proxy_encoder.calls[2:]] == ["libx264"]


def test_a_proxy_failing_with_nvenc_and_libx264_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    proxy_encoder: type[ProxyEncoder],
) -> None:
    monkeypatch.setattr(ui, "video_encoder", lambda: ui.VideoEncoder("nvenc"))

    class Failing(ProxyEncoder):
        def __init__(self, args: list[str], **_kw: object) -> None:
            self.calls.append(args)
            self.returncode = 1

    monkeypatch.setattr(ui, "spawn", Failing)
    _, folder, p, _ = hevc_project
    assert build_proxy(client, p, folder / "a.mp4") == "failed"
    assert len(proxy_encoder.calls) == 2
    assert not client.app.state.media.gpu_failed  # libx264 did not work either
    assert not list(p.rglob("*.part.mp4"))


def test_no_proxy_ffmpeg_starts_after_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[list[str]] = []
    monkeypatch.setattr(ui, "spawn", lambda args, **_kw: spawned.append(args))
    media = ui.Media()
    media.shutdown()
    code, _ = media._encode(["ffmpeg"])  # e.g. the libx264 retry after a failed NVENC encode
    assert code != 0
    assert not spawned


def test_a_proxy_ffmpeg_is_registered_as_it_starts_so_shutdown_kills_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = ui.Media()
    killed: list[object] = []
    monkeypatch.setattr(ui, "kill_tree", lambda proc, wait=0, *, term=True: killed.append(proc))

    class Encode:
        def __init__(self, args: list[str], **_kw: object) -> None:
            assert media.lock.locked()  # started in the same step as the closing check
            self.returncode = 1

        def communicate(self) -> tuple[str, None]:
            assert media.procs == {self}
            media.shutdown()  # the server stops during the encode
            return "", None

    monkeypatch.setattr(ui, "spawn", Encode)
    media._encode(["ffmpeg"])
    assert len(killed) == 1
    assert isinstance(killed[0], Encode)
    assert not media.procs


# --- window proxies (#323) ---

HLG = {"color_transfer": "arib-std-b67", "color_primaries": "bt2020", "color_space": "bt2020nc"}
WINDOW_BYTES = b"0123456789"


def test_proxy_args_defaults_are_the_whole_source_command() -> None:
    assert ui.proxy_args(Path("in.mkv"), Path("out.mp4"), 1) == [
        *("ffmpeg", "-y", "-nostdin", "-v", "error", "-i", "in.mkv", "-map", "0:v:0"),
        *("-map", "0:a:0?", "-vf", "scale=-2:min(720\\,ih)"),
        *("-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p"),
        *("-g", "30", "-fpsmax", "30", "-c:a", "aac", "-b:a", "128k"),
        *("-movflags", "+faststart", "out.mp4"),
    ]  # fmt: skip


def test_proxy_args_window_seeks_before_the_input_and_limits_after_it() -> None:
    args = ui.proxy_args(Path("in.mkv"), Path("out.mp4"), 1, lo=3.0, hi=33.5)
    assert args[args.index("-ss") + 1] == "3.000"
    assert args.index("-ss") < args.index("-i") < args.index("-t")
    assert args[args.index("-t") + 1] == "30.500"  # the window's length, not hi


def test_proxy_args_color_filters_go_between_scale_and_the_pixel_format_with_bt709_tags() -> None:
    color = "zscale=t=bt709"
    args = ui.proxy_args(Path("in.mkv"), Path("out.mp4"), 1, color=color)
    assert args[args.index("-vf") + 1] == f"scale=-2:min(720\\,ih),{color},format=yuv420p"
    tags = ui.BT709_TAGS
    assert args[args.index(tags[0]) :][: len(tags)] == tags
    plain = ui.proxy_args(Path("in.mkv"), Path("out.mp4"), 1)
    assert "-colorspace" not in plain


def test_info_has_no_color_filter_for_playable_sources_and_the_sdr_filter_for_hlg_hevc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    media = ui.Media()
    probed = media_info(duration=10.0)
    monkeypatch.setattr(ui, "probe", lambda _path: probed)
    assert media.info(src, "k1") == (True, 10.0, 1, "")
    hlg = media_info(vcodec="hevc", pix_fmt="yuv420p10le", duration=10.0, **HLG)
    probed = hlg
    ok, _, _, color = media.info(src, "k2") or (True, 0, 0, "")
    assert not ok
    assert color == encode.sdr_filter(hlg)
    assert color.startswith("zscale=tin=arib-std-b67")


@pytest.mark.parametrize(
    ("over", "color"),
    [
        ({"width": 640, "height": 480, "color_space": "smpte170m", "color_range": "pc"}, "range"),
        ({"width": 720, "height": 576, "color_range": "pc"}, "range"),  # untagged SD is BT.601
        ({"width": 640, "height": 480, "color_space": "smpte170m", "color_range": "tv"}, ""),
        ({"width": 640, "height": 480}, ""),
        ({"color_space": "bt709", "color_range": "pc"}, ""),  # full-range BT.709: as before
        ({}, ""),  # HD without tags
        ({"pix_fmt": "yuv420p10le", **HLG}, "sdr"),
    ],
)
def test_info_narrows_only_full_range_bt601_proxies_and_keeps_the_others_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, over: dict[str, object], color: str
) -> None:
    """#441: a full-range BT.601 proxy becomes TV range with its BT.601 tags (not BT.709 ones);
    limited BT.601, BT.709 and HDR sources keep their filter, so their proxies keep their key."""
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    src = tmp_path / "a.mkv"
    src.write_bytes(b"x")
    info = media_info(vcodec="mpeg2video", duration=10.0, **over)
    monkeypatch.setattr(ui, "probe", lambda _path: info)
    media = ui.Media()
    got = (media.info(src, "k") or (True, 0, 0, ""))[3]
    want = {"range": encode.TV_RANGE, "sdr": encode.sdr_filter(info), "": ""}[color]
    assert got == want == encode.preview_filter(info)
    assert (media._proxy_key("k") == "k") == (color == "")
    args = ui.proxy_args(src, tmp_path / "out.mp4", 1, color=got)
    if color == "range":
        assert args[args.index("-vf") + 1] == (
            "scale=-2:min(720\\,ih),scale=out_range=tv,format=yuv420p"
        )
        assert "-colorspace" not in args  # the BT.601 tags of the frames stay


def test_whole_proxy_of_an_hlg_source_is_converted_to_sdr(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    proxy_encoder: type[ProxyEncoder],
) -> None:
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    hlg = media_info(vcodec="hevc", pix_fmt="yuv420p10le", duration=10.0, **HLG)
    monkeypatch.setattr(ui, "probe", lambda _path: hlg)
    _, folder, p, _ = hevc_project
    assert build_proxy(client, p, folder / "a.mp4") == "ready"
    [args] = proxy_encoder.calls
    vf = args[args.index("-vf") + 1]
    assert vf == f"scale=-2:min(720\\,ih),{encode.sdr_filter(hlg)},format=yuv420p"
    assert args[args.index("-colorspace") + 1] == "bt709"


def test_proxy_key_mixes_in_the_colour_filter_only_when_the_source_has_one() -> None:
    media = ui.Media()
    media.probed["sdr"] = (False, 10.0, 1, "")
    media.probed["hdr"] = (False, 10.0, 1, "zscale=a")
    assert media._proxy_key("sdr") == "sdr"  # existing SDR proxies keep their names
    assert media._proxy_key("unprobed") == "unprobed" and media._proxy_key(None) is None
    old = media._proxy_key("hdr")
    assert old and len(old) == 16 and old != "hdr"
    media.probed["hdr"] = (False, 10.0, 1, "zscale=b")  # a changed tone map
    assert media._proxy_key("hdr") not in (old, "hdr")


def test_hlg_proxy_built_with_another_tone_map_is_not_served_but_rebuilt(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    proxy_encoder: type[ProxyEncoder],
) -> None:
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    hlg = media_info(vcodec="hevc", pix_fmt="yuv420p10le", duration=10.0, **HLG)
    monkeypatch.setattr(ui, "probe", lambda _path: hlg)
    _, folder, p, _ = hevc_project
    key = ui.Media.key((folder / "a.mp4").resolve()) or ""
    (p / "ui" / "proxy").mkdir(parents=True)
    stale = p / "ui" / "proxy" / f"{key}.mp4"  # named by the content key only, as before
    stale.write_bytes(b"old tone map")
    assert client.app.state.media.status(p, str(folder / "a.mp4"))[0] == "needs-proxy"
    assert build_proxy(client, p, folder / "a.mp4") == "ready"
    assert len(proxy_encoder.calls) == 1
    served = client.app.state.media.status(p, str(folder / "a.mp4"))[1]
    assert served is not None and served != stale and served.name != f"{key}.mp4"


class FakeWindowProc:
    """A fake ffmpeg: a window encode stays alive until finished or killed; a whole one ends."""

    def __init__(self, args: list[str], held: bool, low_priority: bool = True) -> None:
        self.args = args
        self.low_priority = low_priority
        self.returncode: int | None = None
        self.done = threading.Event()
        if not held:
            self.finish(0)

    def finish(self, code: int) -> None:
        if self.returncode is None:
            self.returncode = code
            if code == 0:
                Path(self.args[-1]).write_bytes(WINDOW_BYTES)
            self.done.set()

    def communicate(self) -> tuple[str, None]:
        self.done.wait(10)
        return "", None

    @property
    def stdout(self) -> list[str]:  # a whole build streams its progress; it ends at once
        return []

    def wait(self) -> int | None:
        self.done.wait(10)
        return self.returncode

    @property
    def start(self) -> float:
        return float(self.args[self.args.index("-ss") + 1])


class FakeWindowFfmpeg:
    """Replaces ``spawn`` and ``kill_tree``; ``hold``: window encodes wait for ``finish``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.hold = True
        self.procs: list[FakeWindowProc] = []
        self.killed: list[FakeWindowProc] = []
        self.kill_seen = threading.Event()
        monkeypatch.setattr(ui, "spawn", self.spawn)
        monkeypatch.setattr(ui, "kill_tree", self.kill)

    def spawn(self, args: list[str], *, low_priority: bool = True) -> FakeWindowProc:
        proc = FakeWindowProc(args, self.hold and "-ss" in args, low_priority)
        self.procs.append(proc)
        return proc

    def kill(self, proc: FakeWindowProc, wait: float = 0, *, term: bool = True) -> None:
        self.killed.append(proc)
        proc.finish(-9)
        self.kill_seen.set()

    def next_proc(self, n: int) -> FakeWindowProc:
        """The ``n``-th ffmpeg started (0-based), once it has."""
        wait_until(lambda: len(self.procs) > n)
        return self.procs[n]


def wait_until(cond: Callable[[], object], timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def lane_idle(media: ui.Media) -> bool:
    with media.lock:
        return not media.windows_queue and media.window_running is None


@pytest.fixture
def window_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> FakeWindowFfmpeg:
    return FakeWindowFfmpeg(monkeypatch)


@pytest.fixture
def window_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window_ffmpeg: FakeWindowFfmpeg
) -> Iterator[tuple[ui.Media, Path, Path]]:
    """A Media, a project folder and a 1000 s HEVC source; its windows spawn the fake ffmpeg."""
    monkeypatch.setattr(ui, "probe", lambda _path: media_info(vcodec="hevc", duration=1000.0))
    media, p, src = ui.Media(), tmp_path / "proj", tmp_path / "a.mp4"
    p.mkdir()
    src.write_bytes(b"x")
    yield media, p, src
    media.shutdown()


def focus(
    media: ui.Media,
    p: Path,
    src: Path,
    lo: float,
    hi: float,
    dur: float | None = None,
    *,
    queue: bool = False,
):
    real = ui.remap_source(str(src))
    key, res = media._lookup(real)
    assert key is not None and res is not None
    if dur is not None:
        res = (res[0], dur, res[2], res[3])
    media._focus(p, real, key, res, lo, hi, queue=queue)
    return key


@pytest.mark.parametrize(
    ("lo", "hi", "dur", "span"),
    [
        (100.0, 110.0, 1000.0, (9000, 13000)),  # 10 s before, 20 s after
        (103.2, 107.9, 1000.0, (9000, 13000)),  # rounded outward to multiples of 5 s
        (2.0, 5.0, 1000.0, (0, 2500)),  # clamped to 0
        (95.0, 99.0, 100.0, (8500, 10000)),  # clamped to the duration
        (100.0, 100.0, 100.0, (9000, 10000)),  # a request at the duration
    ],
)
def test_focus_rounds_the_window_outward_and_clamps_it(
    window_media: tuple[ui.Media, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    lo: float,
    hi: float,
    dur: float,
    span: tuple[int, int],
) -> None:
    media, p, src = window_media
    window_ffmpeg.hold = False
    key = focus(media, p, src, lo, hi, dur)
    wait_until(lambda: lane_idle(media) and media.windows.get((p, key)))
    assert list(media.windows[(p, key)]) == [span]


def test_a_request_at_the_duration_is_covered_by_a_ready_window(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    window_ffmpeg.hold = False
    key = focus(media, p, src, 95.0, 99.0, 100.0)
    wait_until(lambda: lane_idle(media) and media.windows.get((p, key)))
    focus(media, p, src, 100.0, 100.0, 100.0)  # the end of the source: already inside
    focus(media, p, src, 96.0, 98.0, 100.0)
    assert lane_idle(media)
    assert len(window_ffmpeg.procs) == 1


def test_a_running_window_that_does_not_overlap_the_new_focus_is_killed(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    focus(media, p, src, 100, 110)
    first = window_ffmpeg.next_proc(0)
    focus(media, p, src, 500, 510)
    assert window_ffmpeg.kill_seen.wait(10)
    assert window_ffmpeg.killed == [first]
    second = window_ffmpeg.next_proc(1)  # the new focus encodes next
    assert second.start == 490.0
    second.finish(0)
    wait_until(lambda: lane_idle(media))
    assert [s for s, _ in media.windows[(p, ui.Media.key(src) or "")]] == [49000]
    assert media.failed_windows(p, str(src)) == []  # killed, not failed


def test_a_running_window_that_overlaps_the_new_focus_is_not_killed(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    focus(media, p, src, 100, 110)  # [90, 130]
    first = window_ffmpeg.next_proc(0)
    focus(media, p, src, 125, 135)  # [115, 155]: overlaps, and past the running one's end
    first.finish(0)
    second = window_ffmpeg.next_proc(1)
    assert second.start == 115.0
    second.finish(0)
    wait_until(lambda: lane_idle(media))
    assert window_ffmpeg.killed == []
    assert len(media.windows[(p, ui.Media.key(src) or "")]) == 2


def test_a_superseded_queued_window_never_spawns_and_neither_does_a_cancelled_one(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    focus(media, p, src, 100, 110)
    window_ffmpeg.next_proc(0)
    focus(media, p, src, 125, 135)  # queued behind the running one
    focus(media, p, src, 700, 710)  # supersedes the queued one and kills the running one
    window_ffmpeg.next_proc(1).finish(0)
    wait_until(lambda: lane_idle(media))
    assert [pr.start for pr in window_ffmpeg.procs] == [90.0, 690.0]

    win = ui.ProxyWindow(p, src, "k" * 16, 0, 500, 1, "")
    win.cancelled = True
    assert media._encode(["ffmpeg", "-ss", "0"], win)[0] != 0
    assert len(window_ffmpeg.procs) == 2


def test_a_whole_proxy_build_and_its_progress_belong_to_its_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each project builds its own proxy file: another project on the same footage does not show
    the first one's encode as its own, and its prepare queues its own build."""
    release = threading.Event()

    class Held:
        returncode = 0

        def __init__(self, args: list[str], **_kw: object) -> None:
            self.out = Path(args[-1])

        @property
        def stdout(self) -> list[str]:
            release.wait(10)
            self.out.write_bytes(b"proxy")
            return []

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(ui, "spawn", Held)
    monkeypatch.setattr(ui.Media, "_gpu_encoder", lambda _self: None)
    monkeypatch.setattr(ui, "probe", lambda _path: media_info(vcodec="hevc", duration=100.0))
    src, a, b = tmp_path / "a.mp4", tmp_path / "A", tmp_path / "B"
    src.write_bytes(b"x")
    a.mkdir()
    b.mkdir()
    key = ui.Media.key(ui.remap_source(str(src))) or ""
    media = ui.Media()
    try:
        assert media.prepare(a, str(src)) == "preparing"
        wait_until(lambda: (a, key) in media.progress)
        assert media.cached_view(a, str(src)) == ("preparing", True, 0.0)
        assert media.cached_view(b, str(src)) == ("needs-proxy", True, None)
        assert media.prepare(b, str(src)) == "preparing"  # queued behind A's build
        assert media.cached_view(b, str(src)) == ("preparing", True, None)
        release.set()
        wait_until(lambda: media.cached_view(b, str(src))[0] == "ready")
        assert media.cached_view(a, str(src))[0] == "ready"
        wait_until(lambda: media.proxies == {} and media.progress == {})
    finally:
        release.set()
        media.shutdown()


def test_window_encodes_run_at_normal_priority_and_the_whole_proxy_at_low(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    """The whole proxy's encode would starve the window the user waits on: only it yields."""
    media, p, src = window_media
    window_ffmpeg.hold = False  # both encodes end at once
    assert media._encode(["ffmpeg", str(p / "whole.mp4")])[0] == 0
    win = ui.ProxyWindow(p, src, "k" * 16, 0, 500, 1, "")
    assert media._encode(["ffmpeg", str(p / "window.mp4")], win)[0] == 0
    assert [pr.low_priority for pr in window_ffmpeg.procs] == [True, False]


def test_shutdown_stops_the_window_encode_and_its_lane_and_starts_no_more(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    focus(media, p, src, 100, 110)
    running = window_ffmpeg.next_proc(0)
    focus(media, p, src, 125, 135)  # queued
    media.shutdown()
    assert window_ffmpeg.killed == [running]
    assert media.window_lane is not None
    media.window_lane.join(10)
    assert not media.window_lane.is_alive()
    focus(media, p, src, 400, 410)
    win = ui.ProxyWindow(p, src, "k" * 16, 0, 500, 1, "")
    assert media._encode(["ffmpeg", "-ss", "0"], win)[0] != 0
    assert len(window_ffmpeg.procs) == 1
    assert not media.windows_queue
    assert not list(p.rglob("*.part.mp4"))


def test_at_most_eight_windows_are_kept_and_the_least_recently_used_goes_first(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    window_ffmpeg.hold = False
    key = ""
    for i in range(8):
        key = focus(media, p, src, i * 100 + 20, i * 100 + 25)
        wait_until(lambda: lane_idle(media))
    assert len(media.windows[(p, key)]) == 8
    oldest, second = sorted(media.windows[(p, key)])[:2]
    assert media.window(p, str(src), *oldest) is not None  # used again: now the newest
    focus(media, p, src, 820, 825)
    wait_until(lambda: lane_idle(media))
    kept = media.windows[(p, key)]
    assert len(kept) == 8
    assert oldest in kept and second not in kept
    assert not (p / ui.UI_DIR / "proxy" / f"{key}.w{second[0]}-{second[1]}.mp4").exists()


def age(*files: Path) -> None:
    """Last written or used before prune_proxies' grace period."""
    then = time.time() - ui.PROXY_PRUNE_GRACE_S - 60
    for f in files:
        os.utime(f, (then, then))


def test_windows_are_retired_not_deleted_when_the_whole_proxy_lands_and_pruned_next_run(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    window_ffmpeg.hold = False
    key = ""
    for lo in (20, 420):
        key = focus(media, p, src, lo, lo + 5)
        wait_until(lambda: lane_idle(media))
    folder = p / ui.UI_DIR / "proxy"
    windows = sorted(folder.glob(f"{key}.w*.mp4"))
    assert len(windows) == 2
    media._build(p, src, folder / f"{key}.mp4", key, 1)
    assert (p, key) not in media.windows
    assert media.windows_of(p, str(src)) == ([], [])  # no longer listed
    assert sorted(folder.glob(f"{key}.w*.mp4")) == windows  # but a page may still play one
    # w1000-4500: 10 s before, 20 s after, from 20 s
    assert media.window(p, str(src), 1000, 4500) == folder / f"{key}.w1000-4500.mp4"
    media.shutdown()

    age(*windows)
    fresh = ui.Media()  # the next run
    fresh.prune_proxies(p, [str(src)])
    assert [f.name for f in folder.iterdir()] == [f"{key}.mp4"]


@pytest.mark.parametrize(
    ("want", "dur_cs", "covered"),
    [
        ((1000, 2000), 10000, True),
        ((999, 2000), 10000, False),  # starts before the window
        ((1000, 2995), 10000, True),  # hi - 5 cs: inside
        ((1000, 2996), 10000, False),  # the window's last frame (4 cs) is not played
        ((1000, 3000), 10000, False),
        ((1000, 3000), 3000, True),  # the window ends with the source: all of it plays
        ((1000, 3001), 3000, False),
        ((1000, 3000), 0, False),  # duration unknown
    ],
)
def test_a_window_covers_a_request_except_its_last_frame_unless_it_ends_the_source(
    want: tuple[int, int], dur_cs: int, covered: bool
) -> None:
    assert ui._covers((1000, 3000), want, dur_cs) is covered


def test_a_request_in_the_last_frame_of_a_ready_window_queues_a_new_one(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    window_ffmpeg.hold = False
    key = focus(media, p, src, 100, 110)  # [90, 130]
    wait_until(lambda: lane_idle(media) and media.windows.get((p, key)))
    focus(media, p, src, 119.9, 129.95)  # inside: nothing new
    assert len(window_ffmpeg.procs) == 1
    focus(media, p, src, 120.0, 129.97)  # reaches the last frame: a new window
    wait_until(lambda: len(window_ffmpeg.procs) == 2 and lane_idle(media))
    assert window_ffmpeg.procs[1].start == 110.0


def test_a_cancelled_running_encode_does_not_count_as_covering_the_focus(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    focus(media, p, src, 100, 110)
    first = window_ffmpeg.next_proc(0)
    with media.lock:
        assert media.window_running is not None
        media.window_running.cancelled = True  # e.g. its source's whole proxy landed
    focus(media, p, src, 102, 108)  # inside the running window's range
    first.finish(0)
    second = window_ffmpeg.next_proc(1)
    assert second.start == 90.0
    second.finish(0)
    wait_until(lambda: lane_idle(media))
    assert len(media.windows[(p, ui.Media.key(src) or "")]) == 1


def test_a_window_encode_for_a_source_whose_whole_proxy_lands_is_stopped(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    key = focus(media, p, src, 100, 110)
    running = window_ffmpeg.next_proc(0)
    media._drop_windows(p, key)
    assert window_ffmpeg.kill_seen.wait(10)
    assert window_ffmpeg.killed == [running]
    wait_until(lambda: lane_idle(media))
    assert not list(p.rglob("*.mp4"))
    assert media.failed_windows(p, str(src)) == []  # stopped, not failed


def prepare(client: TestClient, pid: str, sid: str, body: object = None) -> object:
    url = f"/api/projects/{pid}/sources/{sid}/prepare"
    return client.post(url) if body is None else client.post(url, json=body)


@pytest.fixture
def whole_builds(client: TestClient) -> list[tuple]:
    """The whole-proxy builds queued; none runs."""
    queued: list[tuple] = []
    client.app.state.media.pool.submit = lambda fn, *a: queued.append(a)  # type: ignore[method-assign]
    return queued


@pytest.mark.parametrize("body", [None, {}, {"x": 1}], ids=["no-body", "empty", "unrelated"])
def test_prepare_without_a_window_keeps_the_whole_source_behaviour(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
    body: object,
) -> None:
    pid, _, _, _ = hevc_project
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    res = prepare(client, pid, sid, body)
    assert (res.status_code, res.json()) == (200, {"status": "preparing"})
    assert len(whole_builds) == 1
    assert not window_ffmpeg.procs
    assert client.app.state.media.window_lane is None


@pytest.mark.parametrize(
    "raw",
    [
        b'{"lo": true, "hi": 5}',
        b'{"lo": "1", "hi": 5}',
        b'{"lo": NaN, "hi": 5}',
        b'{"lo": 1, "hi": Infinity}',
        b'{"lo": 6, "hi": 5}',
        b'{"lo": -1, "hi": 5}',
        b'{"lo": 1}',
        b'{"hi": 5}',
    ],
)
def test_prepare_refuses_an_invalid_window_and_queues_nothing(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
    raw: bytes,
) -> None:
    pid, _, _, _ = hevc_project
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    url = f"/api/projects/{pid}/sources/{sid}/prepare"
    res = client.post(url, content=raw, headers={"Content-Type": "application/json"})
    assert res.status_code == 400, res.text
    assert not whole_builds
    assert not window_ffmpeg.procs
    assert client.app.state.media.proxies == {}


def test_prepare_with_a_window_queues_it_and_the_whole_proxy_and_the_listing_shows_it(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
) -> None:
    pid, _, _, _ = hevc_project
    base = f"/api/projects/{pid}"
    sid = client.get(base).json()["sources"][0]["id"]
    assert prepare(client, pid, sid, {"lo": 2, "hi": 4.5}).json() == {"status": "preparing"}
    assert len(whole_builds) == 1
    proc = window_ffmpeg.next_proc(0)
    assert (proc.start, proc.args[proc.args.index("-t") + 1]) == (0.0, "10.000")  # clamped to 10 s

    def probing(_path: object) -> None:
        raise AssertionError("listing a project must not run ffprobe")

    monkeypatch.setattr(ui, "probe", probing)
    src = client.get(base).json()["sources"][0]
    assert src["windows"] == []
    assert src["pending"] == [{"lo": 0.0, "hi": 10.0, "state": "encoding"}]
    assert src["duration"] == 10.0  # the page needs it to play the window

    proc.finish(0)
    wait_until(lambda: lane_idle(client.app.state.media))
    src = client.get(base).json()["sources"][0]
    assert src["pending"] == []
    [win] = src["windows"]
    assert (win["lo"], win["hi"]) == (0.0, 10.0)
    assert win["url"].split("?")[0].endswith(f"/sources/{sid}/window/0-1000")


def test_the_listing_shows_a_second_overlapping_window_as_queued(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
) -> None:
    pid, folder, p, _ = hevc_project
    media = client.app.state.media
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    real = ui.remap_source(str(folder / "a.mp4"))
    key, res = media._lookup(real)
    res = (res[0], 1000.0, res[2], res[3])
    media._focus(p, real, key, res, 100, 110)
    window_ffmpeg.next_proc(0)
    media._focus(p, real, key, res, 125, 135)
    src = client.get(f"/api/projects/{pid}").json()["sources"][0]
    assert [(w["lo"], w["hi"], w["state"]) for w in src["pending"]] == [
        (90.0, 130.0, "encoding"),
        (115.0, 155.0, "queued"),
    ]
    assert src["id"] == sid


# --- whole-build progress and queued windows (#323, part 2) ---


class LineProc:
    """A fake ffmpeg whose merged output is ``lines``; ``seen`` is what ``media.progress[key]``
    held after each line was read (and once more after the last)."""

    def __init__(
        self, media: ui.Media, key: tuple[Path, str], lines: list[str], returncode: int = 0
    ) -> None:
        self.returncode = returncode
        self.seen: list[float | None] = []
        self._media, self._key, self._lines = media, key, lines

    @property
    def stdout(self) -> Iterator[str]:
        for line in self._lines:
            yield line + "\n"
            self.seen.append(self._media.progress.get(self._key))
        self.seen.append(self._media.progress.get(self._key))

    def wait(self) -> int:
        return self.returncode


@pytest.mark.parametrize(
    ("lines", "duration", "expected"),
    [
        (["out_time_us=2500000", "out_time_us=5000000"], 10.0, [0.25, 0.5, 0.5]),
        (["out_time_us=15000000"], 10.0, [1.0, 1.0]),  # a rounding overshoot is capped
        (["out_time_us=N/A", "out_time_us=5000000"], 10.0, [0.0, 0.5, 0.5]),
        (["out_time_us=5000000"], 0.0, [0.0, 0.0]),  # an unknown duration shows nothing
    ],
    ids=["updates", "capped", "not-available", "no-duration"],
)
def test_read_progress_turns_out_time_into_a_share_of_the_duration(
    lines: list[str], duration: float, expected: list[float]
) -> None:
    media, pk = ui.Media(), (Path("p"), "k")
    media.progress[pk] = 0.0
    proc = LineProc(media, pk, lines)
    media._read_progress(proc, pk, duration)  # type: ignore[arg-type]
    assert proc.seen == expected


def test_read_progress_keeps_errors_in_the_tail_and_progress_lines_out_of_it() -> None:
    media = ui.Media()
    proc = LineProc(
        media,
        (Path("p"), "k"),
        [
            "frame=10",
            "bitrate=   4.1kbits/s",  # ffmpeg pads some values
            "out_time_us=1000000",
            "out_time=00:00:01.000000",
            "progress=continue",
            "",
            "[hevc @ 0x1] Invalid NAL unit size",
            "Error while decoding stream #0:0",
            "progress=end",
        ],
    )
    tail = media._read_progress(proc, (Path("p"), "k"), 10.0)  # type: ignore[arg-type]
    assert tail == "[hevc @ 0x1] Invalid NAL unit size\nError while decoding stream #0:0"


def test_a_whole_build_asks_ffmpeg_for_progress_on_stdout(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    proxy_encoder: type[ProxyEncoder],
) -> None:
    _, folder, p, _ = hevc_project
    assert build_proxy(client, p, folder / "a.mp4") == "ready"
    [args] = proxy_encoder.calls
    assert args[1:4] == ["-progress", "pipe:1", "-nostats"]
    assert args[0] == "ffmpeg"


class ScriptedEncoder:
    """Whole-build fake ffmpegs that report ``reports`` in turn: ``(out_time_us, exit code)``.
    ``started`` holds ``media.progress[key]`` as each one starts to run."""

    media: ui.Media
    reports: list[tuple[int, int]]
    started: list[float | None]
    calls: list[list[str]]

    def __init__(self, args: list[str], **_kw: object) -> None:
        self.calls.append(args)
        self.us, self.returncode = self.reports[len(self.calls) - 1]
        if self.returncode == 0:
            Path(args[-1]).write_bytes(b"proxy")

    @property
    def stdout(self) -> Iterator[str]:
        self.started.append(next(iter(self.media.progress.values()), None))
        yield f"out_time_us={self.us}\n"
        self.started.append(next(iter(self.media.progress.values()), None))  # after the report

    def wait(self) -> int:
        return self.returncode


def scripted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, reports: list[tuple[int, int]]
) -> type[ScriptedEncoder]:
    monkeypatch.setattr(ScriptedEncoder, "media", client.app.state.media, raising=False)
    monkeypatch.setattr(ScriptedEncoder, "reports", reports, raising=False)
    monkeypatch.setattr(ScriptedEncoder, "started", [], raising=False)
    monkeypatch.setattr(ScriptedEncoder, "calls", [], raising=False)
    monkeypatch.setattr(ui, "spawn", ScriptedEncoder)
    return ScriptedEncoder


@pytest.mark.parametrize(("code", "status"), [(0, "ready"), (1, "failed")])
def test_the_whole_build_progress_is_reported_then_cleared_when_it_ends(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    code: int,
    status: str,
) -> None:
    enc = scripted(client, monkeypatch, [(5_000_000, code)])  # of a 10 s source
    monkeypatch.setattr(ui, "video_encoder", lambda: ui.VideoEncoder("cpu"))
    _, folder, p, _ = hevc_project
    assert build_proxy(client, p, folder / "a.mp4") == status
    assert enc.started == [0.0, 0.5]
    assert client.app.state.media.progress == {}


def test_the_nvenc_retry_restarts_the_progress_at_zero(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
) -> None:
    enc = scripted(client, monkeypatch, [(8_000_000, 1), (2_000_000, 0)])
    monkeypatch.setattr(ui, "video_encoder", lambda: ui.VideoEncoder("nvenc"))
    _, folder, p, _ = hevc_project
    assert build_proxy(client, p, folder / "a.mp4") == "ready"
    assert len(enc.calls) == 2
    assert enc.started == [0.0, 0.8, 0.0, 0.2]  # the libx264 run did not start at 0.8
    assert client.app.state.media.progress == {}


def test_the_listing_shows_proxy_and_progress_only_for_a_source_that_needs_a_proxy(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
    whole_builds: list[tuple],
) -> None:
    pid, folder, _p, _ = hevc_project
    media = client.app.state.media
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]

    def entry() -> dict:
        return client.get(f"/api/projects/{pid}").json()["sources"][0]

    assert (entry()["proxy"], entry()["progress"]) == (False, None)  # not probed yet
    assert prepare(client, pid, sid).json() == {"status": "preparing"}
    assert (entry()["proxy"], entry()["progress"]) == (True, None)  # queued, not encoding yet
    key = ui.Media.key(ui.remap_source(str(folder / "a.mp4"))) or ""
    media.progress[_p, key] = 0.123456
    assert (entry()["proxy"], entry()["progress"]) == (True, 0.123)
    media.proxies[_p, key] = "failed"
    media.progress.pop((_p, key))
    assert (entry()["proxy"], entry()["progress"]) == (True, None)


def test_the_listing_of_a_source_the_browser_plays_has_no_proxy_and_no_progress(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    hevc_project: tuple[str, Path, Path, Path],
) -> None:
    pid, _, _, _ = hevc_project
    monkeypatch.setattr(ui, "probe", lambda _path: media_info(duration=10.0))  # H.264
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    assert prepare(client, pid, sid).json() == {"status": "ready"}
    src = client.get(f"/api/projects/{pid}").json()["sources"][0]
    assert (src["status"], src["proxy"], src["progress"]) == ("ready", False, None)


def queued_starts(media: ui.Media) -> list[float]:
    with media.lock:
        return [w.lo / 100 for w in media.windows_queue]


def test_queued_windows_wait_in_order_behind_the_running_one_and_kill_nothing(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    focus(media, p, src, 100, 110)  # [90, 130]
    first = window_ffmpeg.next_proc(0)
    for lo in (300, 500, 700):
        focus(media, p, src, lo, lo + 10, queue=True)
    assert queued_starts(media) == [290.0, 490.0, 690.0]
    assert window_ffmpeg.killed == []
    first.finish(0)
    for n, start in enumerate((290.0, 490.0, 690.0), 1):
        proc = window_ffmpeg.next_proc(n)
        assert proc.start == start
        proc.finish(0)
    wait_until(lambda: lane_idle(media))
    assert window_ffmpeg.killed == []
    assert len(media.windows[(p, ui.Media.key(src) or "")]) == 4


def test_a_queued_window_covered_by_a_queued_or_running_one_is_not_queued_again(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    focus(media, p, src, 100, 110)  # running: [90, 130]
    window_ffmpeg.next_proc(0)
    focus(media, p, src, 102, 108, queue=True)  # inside the running one
    assert queued_starts(media) == []
    focus(media, p, src, 300, 310, queue=True)
    focus(media, p, src, 300, 310, queue=True)  # the same again
    focus(media, p, src, 302, 308, queue=True)  # inside the queued one
    assert queued_starts(media) == [290.0]
    assert window_ffmpeg.killed == []


def test_at_most_four_windows_wait_and_the_running_one_is_not_one_of_them(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    assert ui.PROXY_WINDOWS_QUEUED == 4
    focus(media, p, src, 100, 110)
    window_ffmpeg.next_proc(0)
    for lo in (200, 300, 400, 500, 600, 700):
        focus(media, p, src, lo, lo + 10, queue=True)
    assert queued_starts(media) == [190.0, 290.0, 390.0, 490.0]  # the later ones were refused
    assert window_ffmpeg.killed == []


def test_a_default_focus_after_queued_windows_clears_them_and_stops_the_running_encode(
    window_media: tuple[ui.Media, Path, Path], window_ffmpeg: FakeWindowFfmpeg
) -> None:
    media, p, src = window_media
    focus(media, p, src, 100, 110)
    first = window_ffmpeg.next_proc(0)
    for lo in (300, 500):
        focus(media, p, src, lo, lo + 10, queue=True)
    focus(media, p, src, 800, 810)  # the user moved on
    assert window_ffmpeg.kill_seen.wait(10)
    assert window_ffmpeg.killed == [first]
    window_ffmpeg.next_proc(1).finish(0)  # the 300 and 500 windows were dropped, not run
    wait_until(lambda: lane_idle(media))
    assert [pr.start for pr in window_ffmpeg.procs] == [90.0, 790.0]


@pytest.mark.parametrize(
    "raw",
    [
        b'{"queue": true}',
        b'{"queue": true, "lo": 1}',
        b'{"queue": true, "hi": 5}',
        b'{"queue": true, "lo": true, "hi": 5}',
        b'{"queue": true, "lo": "1", "hi": 5}',
        b'{"queue": true, "lo": NaN, "hi": 5}',
        b'{"queue": true, "lo": 6, "hi": 5}',
        b'{"queue": true, "lo": -1, "hi": 5}',
        b'{"queue": "true", "lo": 1, "hi": 5}',
        b'{"queue": 1, "lo": 1, "hi": 5}',
        b'{"queue": null, "lo": 1, "hi": 5}',
        b'{"queue": "yes"}',
    ],
)
def test_prepare_refuses_a_bad_queue_request_and_queues_nothing(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
    raw: bytes,
) -> None:
    pid, _, _, _ = hevc_project
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    url = f"/api/projects/{pid}/sources/{sid}/prepare"
    res = client.post(url, content=raw, headers={"Content-Type": "application/json"})
    assert res.status_code == 400, res.text
    assert not whole_builds
    assert not window_ffmpeg.procs
    assert client.app.state.media.proxies == {}


def test_prepare_with_queue_waits_behind_the_running_window_instead_of_replacing_it(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
) -> None:
    pid, folder, p, _ = hevc_project
    media = client.app.state.media
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    real = ui.remap_source(str(folder / "a.mp4"))
    key, res = media._lookup(real)
    media._focus(p, real, key, (res[0], 1000.0, res[2], res[3]), 100, 110)
    first = window_ffmpeg.next_proc(0)
    body = {"lo": 500, "hi": 510, "queue": True}
    assert prepare(client, pid, sid, body).json() == {"status": "preparing"}
    assert queued_starts(media) == [0.0]  # clamped to the 10 s source: waits, kills nothing
    assert window_ffmpeg.killed == []
    first.finish(0)
    window_ffmpeg.next_proc(1).finish(0)
    wait_until(lambda: lane_idle(media))
    assert window_ffmpeg.killed == []


def test_a_failed_window_logs_ffmpeg_s_last_lines_and_is_listed_until_asked_for_again(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warning says why (ffmpeg's last lines); the listing tells the page to stop waiting."""
    pid, _, _, _ = hevc_project
    base = f"/api/projects/{pid}"
    sid = client.get(base).json()["sources"][0]["id"]
    output = [
        "",
        "ffmpeg version x",
        *[f"line {i}" for i in range(1, 6)],
        "  ",
        "Conversion failed!  ",
    ]

    class Failing:
        returncode = 1

        def __init__(self, args: list[str], **_kw: object) -> None:
            self.args = args

        def communicate(self) -> tuple[str, None]:
            return "\n".join(output) + "\n", None

    monkeypatch.setattr(ui, "spawn", Failing)
    monkeypatch.setattr(ui.Media, "_gpu_encoder", lambda _self: None)
    entry = lambda: client.get(base).json()["sources"][0]  # noqa: E731
    with caplog.at_level(logging.WARNING, logger="cutpoint.ui"):
        assert prepare(client, pid, sid, {"lo": 2, "hi": 4}).status_code == 200
        wait_until(lambda: entry()["failed"])
    assert entry()["failed"] == [{"lo": 0.0, "hi": 10.0, "reason": "Conversion failed!"}]
    assert entry()["pending"] == []
    [warning] = [r.getMessage() for r in caplog.records if "window proxy" in r.getMessage()]
    assert warning.endswith("failed: line 2\nline 3\nline 4\nline 5\nConversion failed!")

    monkeypatch.setattr(ui, "spawn", window_ffmpeg.spawn)
    window_ffmpeg.hold = False
    assert prepare(client, pid, sid, {"lo": 2, "hi": 4}).status_code == 200
    assert entry()["failed"] == []  # asked for again
    wait_until(lambda: entry()["windows"])


@pytest.mark.parametrize(
    ("code", "tail", "reason"),
    [
        (1, "Conversion failed!", "Conversion failed!"),
        (1, "", "ffmpeg exited with code 1"),
        (-9, "", "ffmpeg exited with code -9 (0xFFFFFFF7)"),
        (3221225477, "", "ffmpeg exited with code 3221225477 (0xC0000005)"),
    ],
)
def test_a_window_failure_without_output_names_the_exit_code(
    code: int, tail: str, reason: str
) -> None:
    assert ui._exit_reason(code, tail) == reason


def test_window_endpoint_serves_ranges_and_refuses_what_is_not_ready(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
) -> None:
    pid, _, proj, _ = hevc_project
    media = client.app.state.media
    base = f"/api/projects/{pid}"
    token = client.headers["X-Cutpoint-Token"]
    sid = client.get(base).json()["sources"][0]["id"]
    ready = f"{base}/sources/{sid}/window/0-1000?t={token}"
    assert client.get(ready).status_code == 409  # not probed or built yet

    window_ffmpeg.hold = False
    assert prepare(client, pid, sid, {"lo": 2, "hi": 4}).status_code == 200
    wait_until(lambda: client.get(base).json()["sources"][0]["windows"])
    res = client.get(ready, headers={"Range": "bytes=2-5"})
    assert (res.status_code, res.content) == (206, b"2345")
    assert res.headers["content-range"] == "bytes 2-5/10"
    assert client.head(ready).status_code == 200
    assert client.get(f"{base}/sources/{sid}/window/0-500?t={token}").status_code == 409
    for bad in ("abc", "0-", "-5", "1-2-3", "xyz-5"):
        assert client.get(f"{base}/sources/{sid}/window/{bad}?t={token}").status_code == 404, bad
    assert client.get(f"{base}/sources/{'0' * 12}/window/0-1000?t={token}").status_code == 404
    assert (
        client.get(
            f"{base}/sources/{sid}/window/0-1000", headers={"X-Cutpoint-Token": ""}
        ).status_code
        == 403
    )

    key = ui.Media.key(ui.remap_source(client.get(base).json()["sources"][0]["path"])) or ""
    media.proxies[proj, key] = "failed"
    assert client.get(f"{base}/sources/{sid}/window/0-500?t={token}").status_code == 404


def test_window_in_a_cache_folder_linked_outside_is_not_served_and_nothing_there_is_deleted(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
) -> None:
    pid, sid, p, outside, key = media_link(client, hevc_project, "proxy")
    media = client.app.state.media
    window = outside / f"{key}.w0-1000.mp4"
    window.write_bytes(b"outside")
    media.windows[(p, key)] = {(0, 1000): window}  # as if it had been recorded before the link
    token = client.headers["X-Cutpoint-Token"]
    assert (
        client.get(f"/api/projects/{pid}/sources/{sid}/window/0-1000?t={token}").status_code == 409
    )
    assert prepare(client, pid, sid, {"lo": 2, "hi": 4}).status_code == 409
    assert not window_ffmpeg.procs
    assert sorted(f.name for f in outside.iterdir()) == [window.name]


def prune_setup(tmp_path: Path) -> tuple[ui.Media, Path, list[str], str, str]:
    """A project with two sources; returns (media, ui/proxy folder, their paths, keys...)."""
    p = tmp_path / "proj"
    folder = p / ui.UI_DIR / "proxy"
    folder.mkdir(parents=True)
    paths = []
    for name, data in (("a.mp4", b"x"), ("b.mp4", b"xy")):
        (tmp_path / name).write_bytes(data)
        paths.append(str(tmp_path / name))
    keys = [ui.Media.key(ui.remap_source(s)) or "" for s in paths]
    return ui.Media(), folder, paths, keys[0], keys[1]


def test_prune_proxies_removes_leftovers_and_adopts_the_windows_that_are_still_wanted(
    tmp_path: Path,
) -> None:
    media, folder, paths, a, b = prune_setup(tmp_path)
    stale = "1" * 16
    keep = [f"{a}.w0-500.mp4", f"{a}.w1000-1500.mp4", "notes.txt", f"{a}.w1-2.txt", f"{b}.mp4"]
    gone = [f"{a}.part.mp4", f"{a}.w0-500.part.mp4", f"{stale}.w0-500.mp4", f"{b}.w0-500.mp4"]
    gone.append(f"{stale}.mp4")  # the whole proxy of a source that left
    for name in keep + gone:
        (folder / name).write_bytes(b"x")
    age(*folder.iterdir())
    p = folder.parents[1]
    media.prune_proxies(p, paths)
    assert sorted(f.name for f in folder.iterdir()) == sorted(keep)
    assert set(media.windows[(p, a)]) == {(0, 500), (1000, 1500)}
    assert (p, b) not in media.windows
    (folder / f"{a}.part.mp4").write_bytes(b"x")  # once per run and project
    media.prune_proxies(p, paths)
    assert (folder / f"{a}.part.mp4").exists()


def test_prune_proxies_keeps_hdr_windows_of_the_current_tone_map_before_any_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    hlg = media_info(vcodec="hevc", pix_fmt="yuv420p10le", duration=10.0, **HLG)
    probed: list[str] = []
    monkeypatch.setattr(ui, "probe", lambda path: probed.append(Path(path).name) or hlg)
    media, folder, paths, a, _ = prune_setup(tmp_path)
    hdr = hashlib.sha1(f"{a}|{encode.sdr_filter(hlg)}".encode()).hexdigest()[:16]
    (folder / f"{hdr}.w0-500.mp4").write_bytes(b"x")
    (folder / f"{a}.w0-500.mp4").write_bytes(b"x")  # built with an earlier tone map
    age(*folder.iterdir())
    p = folder.parents[1]
    media.prune_proxies(p, paths)  # a new run: nothing probed yet
    assert sorted(f.name for f in folder.iterdir()) == [f"{hdr}.w0-500.mp4"]
    assert set(media.windows[(p, hdr)]) == {(0, 500)}
    assert probed == ["a.mp4"]  # matched: b.mp4 is not probed


def test_prepare_deletes_hdr_proxies_of_an_earlier_tone_map_that_prune_could_not_tell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    hlg = media_info(vcodec="hevc", pix_fmt="yuv420p10le", duration=10.0, **HLG)
    probed: list[str] = []
    monkeypatch.setattr(ui, "probe", lambda path: probed.append(Path(path).name) or hlg)
    media, folder, paths, a, b = prune_setup(tmp_path)
    monkeypatch.setattr(media.pool, "submit", lambda *_args: None)  # no build
    # named by the content key: no current proxy beside them
    stale = [folder / f"{a}.w0-500.mp4", folder / f"{a}.mp4"]
    recent = folder / f"{a}.w1000-1500.mp4"  # another run may be serving it
    for f in [*stale, recent]:
        f.write_bytes(b"x")
    age(*stale)
    (folder / f"{b}.mp4").write_bytes(b"x")  # another source's
    p = folder.parents[1]
    media.prune_proxies(p, paths)
    assert probed == []  # nothing to tell them from without probing
    assert media.prepare(p, paths[0]) == "preparing"
    assert probed == ["a.mp4"]
    assert sorted(f.name for f in folder.iterdir()) == sorted([recent.name, f"{b}.mp4"])
    assert (p, a) not in media.windows  # adopted by prune_proxies, now gone


def test_prune_proxies_probes_no_source_whose_proxy_it_knows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probed: list[str] = []
    hevc = media_info(vcodec="hevc", duration=10.0)
    monkeypatch.setattr(ui, "probe", lambda path: probed.append(Path(path).name) or hevc)
    media, folder, paths, a, b = prune_setup(tmp_path)
    names = [f"{a}.mp4", f"{b}.w0-500.mp4"]
    for name in names:
        (folder / name).write_bytes(b"x")
    age(*folder.iterdir())
    media.prune_proxies(folder.parents[1], paths)
    assert probed == []
    assert sorted(f.name for f in folder.iterdir()) == sorted(names)


def test_prune_proxies_keeps_the_proxies_while_a_source_is_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probed: list[str] = []
    monkeypatch.setattr(ui, "probe", lambda path: probed.append(Path(path).name))
    nas = tmp_path / "nas"
    nas.mkdir()
    paths = []
    for name, data in (("a.mp4", b"x"), ("b.mp4", b"xy")):
        (nas / name).write_bytes(data)
        paths.append(str(nas / name))
    a, b = (ui.Media.key(ui.remap_source(s)) or "" for s in paths)
    folder = tmp_path / "proj" / ui.UI_DIR / "proxy"
    folder.mkdir(parents=True)
    # an HDR source's proxies would carry other keys: these may be any source's
    keep = [f"{a}.mp4", f"{b}.w0-500.mp4", f"{'1' * 16}.w0-500.mp4", f"{'2' * 16}.mp4"]
    for name in [*keep, f"{a}.part.mp4"]:
        (folder / name).write_bytes(b"x")
    age(*folder.iterdir())
    nas.rename(tmp_path / "nas-offline")  # the drive goes offline
    ui.Media().prune_proxies(folder.parents[1], paths)
    assert sorted(f.name for f in folder.iterdir()) == sorted(keep)  # the temporary goes
    assert probed == []


def test_prune_proxies_deletes_proxies_idle_a_month_while_a_source_is_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probed: list[str] = []
    sdr = media_info(duration=10.0)
    monkeypatch.setattr(ui, "probe", lambda path: probed.append(Path(path).name) or sdr)
    media, folder, paths, a, b = prune_setup(tmp_path)
    paths.append(str(tmp_path / "offline" / "c.mp4"))  # a drive that stays offline
    month = [f"{'1' * 16}.mp4", f"{'2' * 16}.w0-500.mp4"]  # also an HDR window
    recent = [f"{'3' * 16}.mp4", f"{'4' * 16}.w0-500.mp4"]  # may be the offline source's
    for name in [*month, *recent, f"{a}.mp4", f"{b}.w0-500.mp4"]:
        (folder / name).write_bytes(b"x")
    age(*folder.iterdir())
    for name in [*month, *recent]:
        then = time.time() - 29 * 86400 - 60
        os.utime(folder / name, (then, then))
    os.utime(folder / f"{a}.mp4", (time.time() - 40 * 86400,) * 2)  # a reachable source's
    everything = sorted(f.name for f in folder.iterdir())
    media.prune_proxies(folder.parents[1], paths)
    assert sorted(f.name for f in folder.iterdir()) == everything
    assert probed == []  # nothing idle a month: nothing to probe for
    for name in month:
        then = time.time() - 30 * 86400 - 60
        os.utime(folder / name, (then, then))
    ui.Media().prune_proxies(folder.parents[1], paths)  # a later run
    assert sorted(f.name for f in folder.iterdir()) == sorted(
        [*recent, f"{a}.mp4", f"{b}.w0-500.mp4"]
    )
    assert probed == ["a.mp4", "b.mp4"]  # neither is HDR: no source names them


@pytest.mark.parametrize("window", [False, True])
def test_prune_proxies_keeps_a_reachable_hdr_sources_proxy_idle_a_month_while_one_is_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: bool
) -> None:
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    hlg = media_info(vcodec="hevc", pix_fmt="yuv420p10le", duration=10.0, **HLG)
    probed: list[str] = []
    monkeypatch.setattr(ui, "probe", lambda path: probed.append(Path(path).name) or hlg)
    media, folder, paths, a, _ = prune_setup(tmp_path)
    paths.append(str(tmp_path / "offline" / "c.mp4"))  # a drive that stays offline
    hdr = hashlib.sha1(f"{a}|{encode.sdr_filter(hlg)}".encode()).hexdigest()[:16]
    name = f"{hdr}.w0-500.mp4" if window else f"{hdr}.mp4"  # a whole proxy retires windows
    (folder / name).write_bytes(b"x")
    then = time.time() - 31 * 86400
    os.utime(folder / name, (then, then))
    media.prune_proxies(folder.parents[1], paths)
    assert [f.name for f in folder.iterdir()] == [name]  # not rebuilt for an hour
    assert probed == ["a.mp4"]  # matched: b.mp4 is not probed


def test_prune_proxies_deletes_stale_whole_proxies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    hlg = media_info(vcodec="hevc", pix_fmt="yuv420p10le", duration=10.0, **HLG)
    hevc = media_info(vcodec="hevc", duration=10.0)
    monkeypatch.setattr(ui, "probe", lambda path: hlg if Path(path).name == "a.mp4" else hevc)
    media, folder, paths, a, b = prune_setup(tmp_path)
    hdr = hashlib.sha1(f"{a}|{encode.sdr_filter(hlg)}".encode()).hexdigest()[:16]
    (tmp_path / "b.mp4").write_bytes(b"changed")  # b's proxy is of its earlier content
    b_now = ui.Media.key(ui.remap_source(paths[1])) or ""
    keep = [f"{hdr}.mp4", f"{b_now}.mp4"]
    gone = [f"{a}.mp4", f"{'2' * 16}.mp4", f"{b}.mp4"]  # earlier tone maps, b before it changed
    for name in keep + gone:
        (folder / name).write_bytes(b"x")
    age(*folder.iterdir())
    media.prune_proxies(folder.parents[1], paths)
    assert sorted(f.name for f in folder.iterdir()) == sorted(keep)


def test_prune_proxies_keeps_what_another_run_may_be_encoding_or_serving(
    tmp_path: Path,
) -> None:
    media, folder, paths, a, _ = prune_setup(tmp_path)
    stale = "1" * 16
    p = folder.parents[1]
    names = [f"{a}.part.mp4", f"{a}.w0-500.part.mp4", f"{stale}.mp4", f"{stale}.w0-500.mp4"]
    for name in names:
        (folder / name).write_bytes(b"x")  # just written: another run is encoding them
    used = folder / f"{a}.w1000-1500.mp4"
    used.write_bytes(b"x")
    age(used)
    other = ui.Media()  # another run serves the window
    other.windows[(p, a)] = {(1000, 1500): used}
    assert other.window(p, paths[0], 1000, 1500) == used
    (folder / f"{a}.mp4").write_bytes(b"x")  # its whole proxy landed meanwhile
    media.prune_proxies(p, paths)
    assert sorted(f.name for f in folder.iterdir()) == sorted([*names, used.name, f"{a}.mp4"])
    fresh = ui.Media()
    age(*folder.iterdir())  # an hour later, nothing in use
    fresh.prune_proxies(p, paths)
    assert [f.name for f in folder.iterdir()] == [f"{a}.mp4"]


def test_prune_proxies_skips_a_file_it_cannot_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media, folder, paths, _, _ = prune_setup(tmp_path)
    for name in ("1" * 16, "2" * 16):
        (folder / f"{name}.mp4").write_bytes(b"x")
    age(*folder.iterdir())
    unlink = Path.unlink

    def held(self: Path, missing_ok: bool = False) -> None:
        if self.name.startswith("1"):
            raise PermissionError(13, "in use by another process", str(self))  # Windows
        unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", held)
    media.prune_proxies(folder.parents[1], paths)
    assert [f.name for f in folder.iterdir()] == [f"{'1' * 16}.mp4"]


def test_a_served_proxy_is_marked_used(
    client: TestClient, hevc_project: tuple[str, Path, Path, Path]
) -> None:
    pid, folder, p, _ = hevc_project
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    source = folder / "a.mp4"
    age(source)
    key = ui.Media.key(source.resolve()) or ""
    (p / ui.UI_DIR / "proxy").mkdir(parents=True)
    proxy = p / ui.UI_DIR / "proxy" / f"{key}.mp4"
    proxy.write_bytes(b"proxy")
    age(proxy)
    before = source.stat().st_mtime_ns
    url = f"/api/projects/{pid}/sources/{sid}?t={client.headers['X-Cutpoint-Token']}"
    first = client.get(url)
    assert first.status_code == 200
    assert first.headers["etag"] == f'"{key}-5"'  # from its name and size, not its mtime
    touched = proxy.stat().st_mtime_ns
    assert time.time() - touched / 1e9 < 60
    assert source.stat().st_mtime_ns == before  # never the source
    seek = client.get(url, headers={"Range": "bytes=1-2", "If-Range": first.headers["etag"]})
    assert (seek.status_code, seek.content) == (206, b"ro")
    assert proxy.stat().st_mtime_ns == touched  # touched recently: not again


def test_a_window_written_recently_is_not_touched_again_when_served(tmp_path: Path) -> None:
    media, folder, paths, a, _ = prune_setup(tmp_path)
    p = folder.parents[1]
    window = folder / f"{a}.w0-500.mp4"
    window.write_bytes(b"x")
    media.windows[(p, a)] = {(0, 500): window}
    touched = window.stat().st_mtime_ns
    assert media.window(p, paths[0], 0, 500) == window
    assert window.stat().st_mtime_ns == touched  # just written: not touched


def test_prune_proxies_adopts_at_most_eight_windows_keeping_the_newest(tmp_path: Path) -> None:
    media, folder, paths, a, _ = prune_setup(tmp_path)
    for i in range(10):
        f = folder / f"{a}.w{i * 1000}-{i * 1000 + 500}.mp4"
        f.write_bytes(b"x")
        os.utime(f, ns=(10**9 * (i + 1),) * 2)
    media.prune_proxies(folder.parents[1], paths)
    assert sorted(s[0] for s in media.windows[(folder.parents[1], a)]) == list(
        range(2000, 10000, 1000)
    )
    assert len(list(folder.iterdir())) == 8


def test_prune_proxies_deletes_nothing_through_a_link_to_a_folder_outside(
    client: TestClient, hevc_project: tuple[str, Path, Path, Path]
) -> None:
    _, _, p, outside, key = media_link(client, hevc_project, "proxy")
    names = [f"{key}.part.mp4", f"{key}.w0-500.mp4", f"{'1' * 16}.w0-500.mp4"]
    for name in names:
        (outside / name).write_bytes(b"x")
    media = ui.Media()
    media.prune_proxies(p, [str(hevc_project[1] / "a.mp4")])
    assert sorted(f.name for f in outside.iterdir()) == sorted(names)
    assert (p, key) not in media.windows


@pytest.mark.slow
@needs_ffmpeg
def test_a_real_window_proxy_keeps_the_flash_and_the_beep_in_sync_from_a_keyframe(
    tmp_path: Path,
) -> None:
    """A window from lo=3.0 of a source with a flash and a beep at 5.0 s: both land at 2.0 s."""
    src = tmp_path / "src.mp4"
    ffmpeg(
        *("-f", "lavfi", "-i", "color=c=black:s=160x120:r=25:d=12"),
        *("-f", "lavfi", "-i", "aevalsrc='if(between(t,5,5.1),0.8*sin(2*PI*1000*t),0)':s=48000:d=12"),
        *("-vf", "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='between(t,5,5.03)'"),
        *("-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "250", "-c:a", "aac", str(src)),
    )  # fmt: skip
    p = tmp_path / "proj"
    p.mkdir()
    key = ui.Media.key(src) or ""
    media = ui.Media()
    try:
        media._build_window(ui.ProxyWindow(p, src, key, 300, 1000, 1, ""))
    finally:
        media.shutdown()
    out = p / ui.UI_DIR / "proxy" / f"{key}.w300-1000.mp4"
    assert out.is_file()
    assert not list(p.rglob("*.part.mp4"))

    def probe_csv(*args: str) -> list[list[str]]:
        text = subprocess.run(
            ["ffprobe", "-v", "error", *args, "-of", "csv=p=0", str(out)],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        return [line.split(",") for line in text.split()]

    assert float(probe_csv("-show_entries", "format=duration")[0][0]) == pytest.approx(
        7.0, abs=0.15
    )
    frames = probe_csv("-select_streams", "v", "-show_entries", "frame=key_frame,pts_time")
    assert frames[0][0] == "1"  # the first frame is a keyframe
    pts = [float(f[1]) for f in frames]
    gray = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(out), "-an", "-vf", "scale=8:8,format=gray",
         "-fps_mode", "passthrough", "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    means = np.frombuffer(gray, dtype=np.uint8).reshape(-1, 64).mean(axis=1)
    bright = [i for i, m in enumerate(means) if m > 128]
    assert bright, "the flash is missing"
    assert pts[bright[0]] == pytest.approx(2.0, abs=0.04 + 1e-3)  # within one frame at 25 fps

    start = float(probe_csv("-select_streams", "a", "-show_entries", "stream=start_time")[0][0])
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(out), "-vn", "-f", "s16le", "-ac", "1", "-ar", "48000", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    samples = np.abs(np.frombuffer(pcm, dtype=np.int16).astype(float))
    onset = int(np.argmax(samples > 0.3 * 32768))
    assert samples[onset] > 0.3 * 32768, "the beep is missing"
    assert start + onset / 48000 == pytest.approx(2.0, abs=0.025)


@pytest.mark.slow
@needs_ffmpeg
def test_a_real_whole_build_reports_progress_up_to_the_end_of_the_source(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src.mp4"
    ffmpeg(
        *("-f", "lavfi", "-i", "testsrc=s=160x120:r=25:d=3"),
        *("-f", "lavfi", "-i", "sine=f=440:d=3"),
        *("-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)),
    )  # fmt: skip
    p = tmp_path / "proj"
    p.mkdir()
    key = ui.Media.key(src) or ""

    class Recorder(dict):
        def __init__(self) -> None:
            super().__init__()
            self.values: list[float] = []

        def __setitem__(self, k: str, v: float) -> None:
            self.values.append(v)
            super().__setitem__(k, v)

    media = ui.Media()
    media.progress = Recorder()  # type: ignore[assignment]
    try:
        proxy = p / ui.UI_DIR / "proxy" / f"{key}.mp4"
        proxy.parent.mkdir(parents=True)
        media._build(p, src, proxy, key, 1, "", 3.0)
    finally:
        media.shutdown()
    assert proxy.is_file()
    values = media.progress.values  # type: ignore[attr-defined]
    assert values[0] == 0.0
    assert values == sorted(values)
    assert max(values) == pytest.approx(1.0, abs=0.1)
    assert media.progress == {}


# --- source list cache (#129) ---


@pytest.fixture
def folder_listings(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """One entry per ``find_videos`` call made by the UI."""
    calls: list[int] = []
    real = ui.find_videos

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ui, "find_videos", counting)
    return calls


def source_names(client: TestClient, pid: str) -> list[str]:
    res = client.get(f"/api/projects/{pid}")
    assert res.status_code == 200, res.text
    return sorted(s["name"] for s in res.json()["sources"])


@pytest.mark.ffmpeg
def test_media_requests_list_an_unanalyzed_folder_once(
    client: TestClient, tmp_path: Path, folder_listings: list[int]
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    folder_listings.clear()
    base = f"/api/projects/{pid}/sources/{sid}"
    for _ in range(3):
        client.head(base)
        client.get(f"{base}/frame?at=0")
        client.post(f"/api/projects/{pid}/sources/{sid}/prepare")
    assert folder_listings == []


def test_a_video_added_or_removed_in_the_picked_folder_shows_on_the_next_request(
    client: TestClient, tmp_path: Path
) -> None:
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    assert source_names(client, pid) == ["a.mp4"]
    start = folder.stat().st_mtime_ns
    (folder / "b.mp4").write_bytes(b"y")
    os.utime(folder, ns=(start + 10**9, start + 10**9))
    assert source_names(client, pid) == ["a.mp4", "b.mp4"]
    (folder / "a.mp4").unlink()
    os.utime(folder, ns=(start + 2 * 10**9, start + 2 * 10**9))
    assert source_names(client, pid) == ["b.mp4"]


def test_writing_highlights_rebuilds_the_list_from_the_analysis(
    client: TestClient, tmp_path: Path
) -> None:
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x", "b.mp4": b"y"})
    assert source_names(client, pid) == ["a.mp4", "b.mp4"]
    hl = {"sources": [{"path": str(folder / "b.mp4"), "duration": 3.0}], "candidates": []}
    (ui.project_dir(pid) / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    assert source_names(client, pid) == ["b.mp4"]
    cache.save_json(
        ui.project_dir(pid) / "highlights.json",
        {"sources": [{"path": str(folder / "a.mp4"), "duration": 3.0}], "candidates": []},
    )
    assert source_names(client, pid) == ["a.mp4"]


def test_a_failed_folder_listing_is_not_cached_as_an_empty_list(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    real = ui.find_videos
    calls: list[int] = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise FileNotFoundError("folder briefly unavailable")
        return real(*args, **kwargs)

    monkeypatch.setattr(ui, "find_videos", flaky)
    client.get(f"/api/projects/{pid}")  # the failed listing; the response does not matter
    assert source_names(client, pid) == ["a.mp4"]


def test_the_cache_keeps_at_most_max_projects_and_expires_after_its_ttl(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, folder_listings: list[int]
) -> None:
    monkeypatch.setattr(ui, "SOURCES_CACHE_MAX", 1)
    pid1, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    other = tmp_path / "other"
    other.mkdir()
    (other / "c.mp4").write_bytes(b"z")
    pid2 = client.post("/api/projects", json={"name": "Other", "source": str(other)}).json()["id"]
    source_names(client, pid1)
    source_names(client, pid2)  # evicts pid1
    folder_listings.clear()
    source_names(client, pid2)
    assert len(folder_listings) == 0
    source_names(client, pid1)
    assert len(folder_listings) == 1
    monkeypatch.setattr(ui, "SOURCES_CACHE_S", 0)
    source_names(client, pid1)
    assert len(folder_listings) == 2


# --- lazy probe and unreachable picked folder ---


def test_opening_a_project_probes_nothing_until_a_source_is_prepared(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probes: list[object] = []

    def counting(path):
        probes.append(path)
        return media_info(duration=10.0)

    monkeypatch.setattr(ui, "probe", counting)
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    source = client.get(f"/api/projects/{pid}").json()["sources"][0]
    assert (probes, source["status"]) == ([], "unknown")
    client.post(f"/api/projects/{pid}/sources/{source['id']}/prepare")
    assert client.get(f"/api/projects/{pid}").json()["sources"][0]["status"] == "ready"
    assert len(probes) == 1


def test_a_picked_folder_that_cannot_be_listed_is_a_field_not_an_error(
    client: TestClient, tmp_path: Path
) -> None:
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    moved = tmp_path / "moved"
    folder.rename(moved)
    res = client.get(f"/api/projects/{pid}")
    assert res.status_code == 200, res.text
    assert str(folder) in res.json()["source_error"]
    assert res.json()["sources"] == []
    moved.rename(folder)
    res = client.get(f"/api/projects/{pid}")
    assert res.json()["source_error"] is None
    assert source_names(client, pid) == ["a.mp4"]


def test_an_unanalyzed_folder_missing_in_place_is_listed_at_its_source_remap_target(
    client: TestClient, tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"0123456789"})
    copy = tmp_path / "copy"
    write(copy / "a.mp4", b"0123456789")
    write(copy / "skip_unverified.mp4", b"x")  # source_remap_exclude: would be read at the original
    shutil.rmtree(folder)
    monkeypatch.setattr(ui, "probe", lambda _path: media_info(duration=10.0))
    remap(home, monkeypatch, folder, copy, exclude=["*unverified*"])
    data = client.get(f"/api/projects/{pid}").json()
    assert data["source_error"] is None
    assert [s["path"] for s in data["sources"]] == [str(folder / "a.mp4")]
    assert source_names(client, pid) == ["a.mp4"]
    src = data["sources"][0]
    client.post(f"/api/projects/{pid}/sources/{src['id']}/prepare")
    assert client.get(src["url"]).content == b"0123456789"


def test_a_source_remap_change_rebuilds_the_list_at_once_and_other_edits_keep_it(
    client: TestClient, tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builds: list[int] = []
    real = ui.source_entries

    def counting(*args, **kwargs):
        builds.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ui, "source_entries", counting)
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})

    def after(**data) -> int:
        """List sources once under this config.json; how many rebuilds that took."""
        set_config(home, **data)
        builds.clear()
        source_names(client, pid)
        return len(builds)

    assert after() == 1  # first listing
    assert after() == 0  # re-saved without a change
    assert after(source_remap={"/old": "/new"}) == 1  # remap added
    assert after(source_remap={"/old": "/new"}) == 0  # same values re-saved
    assert after(source_remap={"/old": "/new"}, unrelated=1) == 0  # other key changed
    assert after(source_remap={"/old": "/new"}, source_remap_exclude=["*skip*"]) == 1  # exclude
    assert after(source_remap={"/old": "/new"}, source_remap_exclude=["*skip*"]) == 0
    assert after(source_remap={"/old": "/other"}, source_remap_exclude=["*skip*"]) == 1  # retarget


SAMPLE_PROGRESS = [
    'PROGRESS {"step":"transcribe","index":1,"total":6,"label":"Transcription","state":"running","pct":0,"detail":"sample.mp4"}',
    "12:00:01 INFO transcribing sample.mp4",
    'PROGRESS {"step":"transcribe","index":1,"total":6,"label":"Transcription","state":"done","pct":100,"detail":null}',
    'PROGRESS {"step":"reactions","index":2,"total":6,"label":"Reactions","state":"skipped","pct":null,"detail":"events extra not installed"}',
    'PROGRESS {"step":"crashes","index":4,"total":6,"label":"Crashes","state":"skipped","pct":null,"detail":"not action-cam footage"}',
    'PROGRESS {"step":"vision","index":6,"total":6,"label":"Vision rating","state":"skipped","pct":null,"detail":"vision is off (Settings, config.json or --vision)"}',
]  # fmt: skip


def progress_job(lines: list[str], monkeypatch: pytest.MonkeyPatch, code: int = 0) -> ui.Job:
    """A finished job whose (faked) child printed ``lines`` and exited ``code``."""
    FakeProc(lines, code).fake_spawn(monkeypatch)
    job = ui.Jobs().start("p", "analyze", ["x"])
    assert job.done.wait(10)
    return job


def test_progress_lines_become_steps_and_stay_out_of_the_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = progress_job(SAMPLE_PROGRESS, monkeypatch)
    view = job.view()
    assert view["status"] == "completed"
    assert view["logs"] == ["12:00:01 INFO transcribing sample.mp4"]
    assert view["progress"]["total"] == 6
    steps = view["progress"]["steps"]
    assert [(s["step"], s["state"]) for s in steps] == [
        ("transcribe", "done"),  # its latest state only
        ("reactions", "skipped"),
        ("crashes", "skipped"),
        ("vision", "skipped"),
    ]
    assert steps[0] == {
        "step": "transcribe",
        "index": 1,
        "label": "Transcription",
        "state": "done",
        "pct": 100.0,
        "detail": None,
    }
    assert steps[1]["detail"] == "events extra not installed" and steps[1]["pct"] is None


def test_a_step_sent_again_later_keeps_its_index_place_and_a_log_prefix_is_accepted() -> None:
    job = ui.Job("p", "edit")
    for line in [
        'PROGRESS {"step":"keyframes","index":3,"total":8,"label":"Keyframes","state":"done","pct":100,"detail":null}',
        '12:00:02 INFO PROGRESS {"step":"vision","index":6,"total":8,"label":"Vision rating","state":"done","pct":100,"detail":null}',
        'PROGRESS {"step":"keyframes","index":3,"total":8,"label":"Keyframes","state":"running","pct":40.5,"detail":"events"}\r\n',
    ]:  # fmt: skip
        job.add_output(line)
    view = job.view()
    assert [(s["step"], s["state"], s["pct"]) for s in view["progress"]["steps"]] == [
        ("keyframes", "running", 40.5),
        ("vision", "done", 100.0),
    ]
    assert view["progress"]["total"] == 8
    assert view["logs"] == []


@pytest.mark.parametrize(
    "line",
    [
        "PROGRESS {not json",
        "PROGRESS [1, 2]",
        'PROGRESS {"step":"x","index":1,"state":"exploded"}',
        'PROGRESS {"index":1,"state":"done"}',
        'Saying PROGRESS {"step":"x","index":1,"state":"done"} mid-line',
    ],
)
def test_a_malformed_progress_line_stays_in_the_log_as_text(line: str) -> None:
    job = ui.Job("p", "analyze")
    job.add_output(line + "\n")
    assert job.view()["logs"] == [line]
    assert job.view()["progress"] == {"total": None, "steps": []}


def test_odd_progress_fields_are_tidied() -> None:
    step = ui.parse_progress(
        'PROGRESS {"step":"x","index":2,"total":true,"label":"","state":"running","pct":250,"detail":5}'
    )
    assert step == {
        "step": "x",
        "index": 2,
        "total": None,
        "label": "x",
        "state": "running",
        "pct": 100.0,
        "detail": None,
    }
    nan = ui.parse_progress('PROGRESS {"step":"x","index":2,"state":"running","pct":NaN}')
    assert nan is not None and nan["pct"] is None


@pytest.mark.parametrize("code", [0, 1])
def test_a_step_still_running_when_the_job_fails_reads_interrupted(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    job = progress_job([*SAMPLE_PROGRESS[:1], "12:00:03 ERROR boom"], monkeypatch, code)
    view = job.view()
    assert view["status"] == ("completed" if code == 0 else "failed")
    (step,) = view["progress"]["steps"]
    assert step["state"] == ("running" if code == 0 else "interrupted")
    assert step["pct"] == 0.0 and step["detail"] == "sample.mp4"


def test_a_cancelled_job_interrupts_its_running_step(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = FakeProc()
    monkeypatch.setattr(ui, "spawn", lambda args: proc)
    monkeypatch.setattr(ui, "kill_tree", lambda _p: proc.release.set())
    jobs = ui.Jobs()
    job = jobs.start("p", "analyze", ["x"])
    try:
        job.add_output(SAMPLE_PROGRESS[0])
        job.add_output(SAMPLE_PROGRESS[3])
        jobs.cancel(job.id)
    finally:
        proc.release.set()
    assert job.done.wait(10)
    assert job.view()["status"] == "cancelled"
    assert [s["state"] for s in job.view()["progress"]["steps"]] == ["interrupted", "skipped"]


# --- Export options: Smart edges, Audio transition (#462) ---


def _segments_written(p: Path) -> list[dict]:
    (seg_file,) = list(p.rglob("segments-*.json"))
    return json.loads(seg_file.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("body", "smart"),
    [({}, False), ({"smart_edges": False}, False), ({"smart_edges": True}, True)],
)
def test_clips_job_smart_edges_hands_the_trims_to_the_smart_flags(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: dict, smart: bool
) -> None:
    p, seg = cli_project(tmp_path)
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    captured: list[list[str]] = []

    def start(self, project_id, action, args, on_success, same_folder, *rest):
        captured.append(args)
        same_folder()  # the check under the jobs lock, which writes the segments file
        return SimpleNamespace(view=lambda: {"status": "running"})

    monkeypatch.setattr(ui.Jobs, "start", start)
    res = client.post("/api/projects/cli/jobs", json={"action": "clips", **body})
    assert res.status_code == 200, res.text
    args = captured[0]
    assert ("--smart-tail" in args) is smart and ("--speech-aware" in args) is smart
    (seg,) = _segments_written(p)
    if smart:  # the flags decide: no per-segment override
        assert "smart_tail" not in seg and "speech_aware" not in seg
    else:  # today's behaviour: exactly the user's trims
        assert seg["smart_tail"] is False and seg["speech_aware"] is False
    assert (seg["start"], seg["end"]) == (1, 4.5)


@pytest.mark.parametrize("value", ["true", 1, 0, None, ["x"]])
def test_clips_job_rejects_a_non_boolean_smart_edges(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: object
) -> None:
    res, captured = peaks_job(
        client, monkeypatch, tmp_path, {"action": "clips", "smart_edges": value}
    )
    assert res.status_code == 400, res.text
    assert "smart_edges must be true or false" in res.json()["detail"]
    assert not captured


@pytest.mark.parametrize(
    ("body", "audio"),
    [
        ({}, "cut"),  # old pages send none: follows the video
        ({"transition": "cut"}, "cut"),
        ({"transition": "fade"}, "crossfade"),
        ({"transition": "fade", "audio_transition": "cut"}, "cut"),
        ({"transition": "fade", "audio_transition": "crossfade"}, "crossfade"),
        ({"transition": "cut", "audio_transition": "cut"}, "cut"),
    ],
)
def test_assemble_job_passes_the_audio_transition(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: dict, audio: str
) -> None:
    res, captured = peaks_job(client, monkeypatch, tmp_path, {"action": "assemble", **body})
    assert res.status_code == 200, res.text
    args = captured[0]
    assert args.count("--audio-transition") == 1
    assert args[args.index("--audio-transition") + 1] == audio
    assert args[args.index("--transition") + 1] == body.get("transition", "cut")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"transition": "cut", "audio_transition": "crossfade"}, "needs the fade"),
        ({"audio_transition": "crossfade"}, "needs the fade"),  # transition defaults to cut
        ({"transition": "fade", "audio_transition": "echo"}, "audio_transition must be"),
        ({"transition": "fade", "audio_transition": None}, "audio_transition must be"),
    ],
)
def test_assemble_job_refuses_an_audio_transition_that_does_not_fit(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    body: dict,
    message: str,
) -> None:
    res, captured = peaks_job(client, monkeypatch, tmp_path, {"action": "assemble", **body})
    assert res.status_code == 400, res.text
    assert message in res.json()["detail"]
    assert not captured


@pytest.mark.parametrize(
    ("cfg", "is_set", "shown"),
    [
        (None, False, "off"),
        ("light", True, "light"),
        ("strong", True, "strong"),
        ("off", True, "off"),
        ("loud", False, "off"),  # invalid: not a default the page should start from
        (3, False, "off"),
    ],
)
def test_settings_say_whether_config_json_sets_compress_peaks(
    client: TestClient, home: Path, cfg: object, is_set: bool, shown: str
) -> None:
    """The page starts Compress Audio Peaks on Light unless config.json sets a value."""
    if cfg is not None:
        set_config(home, compress_peaks=cfg)
    st = client.get("/api/settings").json()
    assert (st["compress_peaks_set"], st["compress_peaks"]) == (is_set, shown)


# --- Web contract pins for the Export tab (#462) ---


def _web(name: str) -> str:
    return (ui.WEB_DIR / name).read_text(encoding="utf-8")


def test_the_360_empty_state_counts_recordings_apart_from_the_360() -> None:
    """Contract pin: a 360 project with no moments reads "360 footage: N recording(s) found"
    (or "No 360 recordings found"), never "N 360 clips found", which read as 5,360."""
    js = _web("app.js")
    body = re.search(r"function renderUnanalyzed\(.*?\n\}\n", js, re.S)[0]
    assert "`360 footage: ${n} recording${n === 1 ? '' : 's'} found`" in body
    assert "'No 360 recordings found'" in body
    assert not re.search(r"\$\{n\} 360", js)  # no count directly before "360"


def test_stage_and_final_video_never_sound_together() -> None:
    """Contract pin: the Export tab's final video plays on while its tab is hidden, so "Play
    sequence" (or a moment) starting on the stage pauses it, and playing it stops the sequence
    (or pauses the stage). Before, its audio ran on under every sequence clip."""
    js = _web("app.js")
    stage_play = re.search(r"onStage\('play', \(\) => \{(.*?)\n\}\);", js, re.S)[1]
    assert "finalVideo.pause();" in stage_play
    final_play = re.search(
        r"finalVideo\.addEventListener\('play', \(\) => \{(.*?)\n\}\);", js, re.S
    )
    assert final_play and "if (S.seqPlay) seqStop();" in final_play[1]
    # off the sequence it pauses the stage and cancels a resume at a window's end, which would
    # start the stage again (and so pause the final video) once the next window loads
    assert re.search(r"else \{\s*cancelResume\(\);[^\n]*\n\s*v\.pause\(\);", final_play[1])


def test_a_hidden_side_tab_pauses_its_videos() -> None:
    """Contract pin: leaving the Export tab pauses its final video, which would otherwise play
    on unseen (a hidden element keeps playing)."""
    js = _web("app.js")
    body = js[js.index("function initSideTabs(") : js.index("function initMomentOrder(")]
    assert "panel.hidden = !on;" in body
    assert "if (!on) for (const vid of panel.querySelectorAll('video')) vid.pause();" in body
    html = _web("index.html")
    export = html[html.index('id="panelExport"') :]
    assert '<video id="finalVideo"' in export  # the video this pauses lives in a tab panel


def test_play_sequence_keeps_a_window_proxy_only_when_it_covers_the_whole_clip() -> None:
    """Contract pin: during "Play sequence" the stage keeps its loaded window proxy for the next
    clip of the same source only when the window plays all of it, as ``fileFor`` decides. A
    window covering the clip's start alone was kept and sought, so the clip was cut short at
    the window's end or the sequence stalled waiting for a file that never loaded."""
    js = _web("app.js")
    # a file of the clip's own source: a window of another source whose times hold the clip
    # played that source's picture and sound as the clip
    holds = re.search(r"const holdsClip = \(c\) => \{(.*?)\n\};", js, re.S)[1]
    assert "ofSource(src)" in holds and "coversSeg(r, c, sourceDur(c.source))" in holds
    body = js[js.index("function stageVideo(") : js.index("/** Stop a video element")]
    keep = "const keep = mine && inLoaded(at, durOf(c)) && (!S.seqPlay || holdsClip(c));"
    assert keep in body
    assert "if (fits && (keep || p))" in body and "if (!keep) loadFile(src, p, at);" in body
    playing_on = re.search(r"const playingOn = ([^;]*);", body)[1]
    assert "ofSource(src)" in playing_on
    assert playing_on.endswith("&& (!S.seqPlay || holdsClip(c))")


def test_sequence_column_is_a_tablist_of_tabs_and_tabpanels() -> None:
    """Contract pin (#462, #346): the column has a Sequence, an AI Chat and an Export tab, each
    controlling a labelled tabpanel; only the selected tab is in the tab order (roving
    tabindex)."""
    html = _web("index.html")
    (tablist,) = re.findall(r'<div[^>]*role="tablist"[^>]*>(.*?)</div>', html, re.S)
    tabs = re.findall(r"<button\b[^>]*>", tablist)
    assert len(tabs) == 3 and all('role="tab"' in t for t in tabs)
    panels = dict(
        re.findall(r'<div id="(\w+)"[^>]*role="tabpanel"[^>]*aria-labelledby="(\w+)"', html)
    )
    assert panels == {
        "panelSequence": "tabSequence",
        "panelAssistant": "tabAssistant",
        "panelExport": "tabExport",
    }
    for tab in tabs:
        tab_id = re.search(r'id="(\w+)"', tab)[1]
        controls = re.search(r'aria-controls="(\w+)"', tab)[1]
        assert panels[controls] == tab_id  # tab and panel name each other
    assert 'aria-selected="true"' in tabs[0]
    assert all('aria-selected="false"' in t and 'tabindex="-1"' in t for t in tabs[1:])
    assert "tabindex" not in tabs[0]
    for later in ("panelAssistant", "panelExport"):
        assert " hidden" in re.search(rf'<div id="{later}"[^>]*>', html)[0]  # Sequence first
    js = _web("app.js")
    assert "[role=tab]" in js and "aria-selected" in js and "ArrowRight" in js


def test_side_tabs_share_the_column_width_evenly_at_any_width() -> None:
    """Contract pin: the tab band spans the whole column (bleeding over its padding) and its tabs
    are equal grid columns with no gap or side padding, so at any width the splitter sets they
    fill it evenly; a label too long for its tab ends in an ellipsis (its text, the tab's name for
    screen readers, stays whole). Before, the tabs were as wide as their labels, at the left.

    Check in a browser (no browser in the test suite): at the narrowest, default and widest
    column, the three tabs are equally wide within 1 px and together as wide as the band."""
    css = _web("app.css")
    band = re.search(r"^\.side-tabs \{([^}]*)\}", css, re.M)[1]
    assert "display: grid;" in band and "grid-auto-flow: column;" in band
    assert "grid-auto-columns: minmax(0, 1fr);" in band  # equal, for any number of tabs
    assert "margin: -12px -14px 0;" in band  # the .col padding: edge to edge
    assert "gap" not in band and "padding" not in band
    tab = re.search(r"^\.side-tab \{([^}]*)\}", css, re.M)[1]
    for decl in ("min-width: 0;", "overflow: hidden;", "white-space: nowrap;",
                 "text-overflow: ellipsis;"):  # fmt: skip
        assert decl in tab
    assert "flex" not in tab
    html = _web("index.html")
    (tablist,) = re.findall(r'<div[^>]*role="tablist"[^>]*>(.*?)</div>', html, re.S)
    names = re.findall(r'<button\b[^>]*role="tab"[^>]*>([^<]+)</button>', tablist)
    assert names == ["Sequence", "AI Chat", "Export"]  # the visible text is the accessible name
    assert "aria-label" not in tablist.replace('aria-label="Sequence column"', "")


def test_export_panel_has_the_options_and_actions_and_no_single_padding_field() -> None:
    """Contract pin (#462): the Export tab holds every export and assembly option and the
    actions; the old single padding field (``#optExtend``) stays gone: padding is a start and an
    end field, sent as ``extend_start`` / ``extend_end``, never as ``extend``."""
    html = _web("index.html")
    export = re.search(r'<div id="panelExport".*?\n      </div>\n    </section>', html, re.S)[0]
    ids = set(re.findall(r'\bid="(\w+)"', export))
    assert {
        "exportPanel", "optVertical", "optCaptions", "optSpeech", "optTail", "optLoudnorm",
        "optLink", "optTransition", "optAudioTransition", "optPeaks", "clipsBtn", "assembleBtn",
        "extendBtn", "revealExportBtn", "exportState", "outputs", "finalVideo", "optPicksSeq",
        "optPicksAuto", "optExtendStart", "optExtendEnd",
    } <= ids  # fmt: skip
    assert 'id="optExtend"' not in html and "'#optExtend'" not in _web("app.js")
    assert "Padding per side" not in html
    assert re.search(r'<select id="optAudioTransition"[^>]*>(.*?)</select>', html, re.S)
    values = re.findall(
        r'value="(\w+)"', re.search(r'<select id="optAudioTransition".*?</select>', html, re.S)[0]
    )
    assert values == ["cut", "crossfade"]
    # the page sends no single padding; the server still accepts one
    js = _web("app.js")
    for builder in ("exportOptions", "extendOptions"):
        body = re.search(rf"function {builder}\(\) \{{\n(.*?)\n\}}", js, re.S)[1]
        assert not re.search(r"\bextend\s*:", body), builder


def test_every_export_option_is_described_for_screen_readers() -> None:
    """Contract pin (#462): each option names a description element (aria-describedby) that
    exists in the page and is not empty, so hover titles are not the only help."""
    html = _web("index.html")
    panel = re.search(
        r'<fieldset class="panel opts" id="exportPanel">.*?</fieldset>\s*</fieldset>', html, re.S
    )[0]
    controls = re.findall(r"<(?:input|select)\b[^>]*>", panel)
    option_ids = {"optVertical", "optCaptions", "optSpeech", "optTail", "optLoudnorm", "optLink",
                  "optTransition", "optAudioTransition", "optPeaks", "optPicksSeq",
                  "optPicksAuto", "optTop", "optMaxTotal", "optIds", "optOrder",
                  "optSpeechExtend", "optSpeechLead", "optTailMin", "optTailMax",
                  "optEventTails", "optEventWindow", "optExtendStart",
                  "optExtendEnd", "optColor", "optColorStrength"}  # fmt: skip
    seen = set()
    for control in controls:
        cid = re.search(r'id="(\w+)"', control)[1]
        seen.add(cid)
        described = re.search(r'aria-describedby="(\w+)"', control)
        assert described, f"{cid} has no aria-describedby"
        text = re.search(rf'<span id="{described[1]}"[^>]*>(.+?)</span>', panel, re.S)
        assert text and len(text[1].strip()) > 20, f"{cid}: description missing or empty"
    assert seen == option_ids


def test_export_tab_says_sequence_and_max_video_length() -> None:
    """Contract pin (mock2): the first Clips to Export choice is "Sequence" (was "The sequence
    (hand-picked)") and --max-total is "Max video length (s)" (was "Max total (s)"), the cap on
    all clips together; its title and description say so."""
    html = _web("index.html")
    assert re.search(r'id="optPicksSeq"[^>]*> Sequence</label>', html)
    assert "hand-picked" not in html and "Max total" not in html
    label = re.search(r'<label for="optMaxTotal" title="([^"]*)">([^<]*)</label>', html)
    assert label and label[2] == "Max video length (s)"
    described = re.search(r'<span id="dMaxTotal" class="vh">([^<]*)</span>', html)[1]
    for text in (label[1], described):
        assert text.startswith("Caps the video's length: the clips' total in seconds")
        assert "(--max-total)" in text


ADVANCED_EXPORT = ("optTop", "optIds", "optSpeechExtend", "optSpeechLead", "optTailMin",
                   "optTailMax", "optEventWindow", "optExtendStart", "optExtendEnd")  # fmt: skip


def test_advanced_export_options_are_hidden_until_a_browser_setting_shows_them() -> None:
    """Contract pin (mock2): Top moments, Candidate ids, the Cut Timing fields but Event tails,
    the "Latest export" facts and Extend export carry ``adv`` and are hidden (display: none, so the
    grids keep no gaps) unless Settings > "Show advanced export options" is ticked. Like Stats for
    Nerds it is off by default and kept in this browser (``cutpoint.advancedExport`` = "1"); it
    previews on change, and Cancel or Escape restores it.

    Check in a browser (no browser in the test suite): advanced off, Cut Timing shows Event tails
    alone and Clips to Export Max video length and Order; ticking the box shows the rest at once."""
    html = _web("index.html")
    js = _web("app.js")
    css = _web("app.css")
    for cid in ADVANCED_EXPORT:
        assert re.search(rf'<label for="{cid}" class="adv"', html), cid
        assert re.search(rf'<input type="\w+" id="{cid}" class="adv"', html), cid
    for cid in ("optMaxTotal", "optOrder", "optEventTails"):  # stay visible
        assert not re.search(rf'<label for="{cid}" class="adv"', html), cid
    # only the "Latest export ..." facts are advanced: the 360 preview note and guidance such as
    # "Add moments to the sequence" always show, and an empty line takes no room
    assert '<p id="exportState" class="muted"></p>' in html
    render = re.search(r"function renderExport\(\) \{\n(.*?)\n\}", js, re.S)[1]
    assert "h('span', { class: 'adv', text: `Latest export (" in render
    assert render.count("latest(") == 3 and "st.textContent" not in render
    for shown in ("say(`${note} Automatic picks", "say(note)", "say('Add moments to the sequence",
                  "say(` It differs from this sequence.${note} Export again"):  # fmt: skip
        assert shown in render, shown
    # no 360 note makes say() null: every branch drops it, so "null" is never rendered
    assert "const show = (...parts) => st.replaceChildren(...parts.filter(Boolean));" in render
    assert render.count("st.replaceChildren(") == 1
    assert re.search(
        r"^#panelExport:not\(\.show-adv\) #exportState:not\(:has\(span:not\(\.adv\)\)\), "
        r"#exportState:empty \{ display: none; \}",
        css,
        re.M,
    )
    assert re.search(r'<button type="button" id="extendBtn" class="btn outline adv"', html)
    assert 'id="revealExportBtn" class="btn outline"' in html  # Show Exports Folder stays
    assert re.search(r"^#panelExport:not\(\.show-adv\) \.adv \{ display: none; \}", css, re.M)
    settings = re.search(r'<dialog id="settingsDialog".*?</dialog>', html, re.S)[0]
    assert re.search(
        r'<label class="check"><input type="checkbox" id="setAdvExport"[^>]*> Show advanced export options</label>',
        settings,
    )
    assert " checked" not in re.search(r'<input type="checkbox" id="setAdvExport"[^>]*>', html)[0]
    assert "const ADV_KEY = 'cutpoint.advancedExport';" in js
    assert "advExport: false," in js  # off unless the browser stored "1"
    init = re.search(r"function initAdvExport\(\) \{\n(.*?)\n\}", js, re.S)[1]
    assert "S.advExport = localStorage.getItem(ADV_KEY) === '1';" in init
    assert "showAdvExport(S.advExport);" in init
    show = re.search(r"function showAdvExport\(on\) \{\n(.*?)\n\}", js, re.S)[1]
    assert "$('#panelExport').classList.toggle('show-adv', on);" in show
    assert "S.advExportWas = S.advExport;" in js
    # Cancel/Escape restore only a changed box: a field revealed for an out-of-range value stays
    assert "if ($('#setAdvExport').checked !== S.advExportWas) setAdvExport(S.advExportWas);" in js
    assert "initAdvExport();" in js


def test_hidden_advanced_export_fields_still_send_their_values() -> None:
    """Contract pin (mock2): hiding never changes an export. A job's fields are skipped only when
    blank or disabled, never because they are hidden; a hidden field that a job sends is named
    under the options ("Advanced options in use: ..."), and a hidden field out of its range shows
    the advanced options so the browser can point at it."""
    js = _web("app.js")
    html = _web("index.html")
    fields = re.search(r"function fields\(\.\.\.keys\) \{\n(.*?)\n\}", js, re.S)[1]
    assert "if (el.disabled || !v) continue;" in fields
    assert "adv" not in fields and "hidden" not in fields and "checkVisibility" not in fields
    in_use = re.search(r"function renderAdvInUse\(\) \{\n(.*?)\n\}", js, re.S)[1]
    assert "el.classList.contains('adv') && !el.disabled && el.value.trim()" in in_use
    assert "Advanced options in use: ${used.join(', ')}" in in_use
    assert '<p id="advInUse" class="hint" hidden></p>' in html
    sync = re.search(r"function syncTiming\(\) \{\n(.*?)\n\}", js, re.S)[1]
    assert sync.rstrip().endswith("renderAdvInUse();")  # after every field and box change
    valid = re.search(r"function validOptions\(body\) \{\n(.*?)\n\}", js, re.S)[1]
    assert "if (bad?.classList.contains('adv')) showAdvExport(true);" in valid
    assert valid.index("showAdvExport(true)") < valid.index("bad?.reportValidity();")


def test_slider_tick_labels_sit_centred_under_their_stops() -> None:
    """Contract pin (mock2): Compress Audio Peaks and Color are centred with equal margins
    (--inset); their tick labels sit in a box inset by half the native thumb (--thumb) more, so
    its ends and middle are the thumb's stops, and each label is centred on its stop. Color's
    Strength slider spans the same width under its name and percentage.

    Check in a browser (no browser in the test suite): Light and Auto are centred under the
    middle stop within 1 px, None/Off and Max/Underwater under the end stops, in every theme."""
    css = _web("app.css")
    assert re.search(
        r"^\.opts \.peaks, \.opts \.color \{ --inset: 24px; --thumb: 16px; \}", css, re.M
    )
    rng = re.search(
        r"^\.peaks input\[type=range\], \.color input\[type=range\] \{([^}]*)\}", css, re.M
    )[1]
    assert "width: calc(100% - 2 * var(--inset));" in rng and "justify-self: center;" in rng
    assert "margin: 0;" in rng
    ticks = re.search(r"^\.peaks \.ticks, \.color \.ticks \{([^}]*)\}", css, re.M)[1]
    assert "position: relative;" in ticks
    assert "margin-inline: calc(var(--inset) + var(--thumb) / 2);" in ticks
    assert "space-between" not in ticks
    span = re.search(r"^\.ticks span \{([^}]*)\}", css, re.M)[1]
    assert "position: absolute;" in span and "transform: translateX(-50%);" in span
    for n, left in ((1, "0"), (2, "50%"), (3, "100%")):
        assert f".ticks span:nth-child({n}) {{ left: {left}; }}" in css
    strength = re.search(r"^\.color-strength \{([^}]*)\}", css, re.M)[1]
    assert "margin: 6px var(--inset) 0;" in strength
    assert re.search(
        r"^\.color \.color-strength input\[type=range\] \{ grid-row: 2; grid-column: 1 / -1; width: 100%; \}",
        css,
        re.M,
    )
    html = _web("index.html")
    for names in (["None", "Light", "Max"], ["Off", "Auto", "Underwater"]):
        assert (
            f'<div class="ticks" aria-hidden="true">{"".join(f"<span>{n}</span>" for n in names)}</div>'
            in html
        )


def test_export_option_defaults_are_smart_edges_loudnorm_link_on_and_light_peaks() -> None:
    """Contract pin (#462): a viewer who changed nothing gets Speech-aware cuts and Smart tails
    (Smart edges before), Normalize loudness and Link Transitions on, Light peaks, the sequence
    as the clips and Event tails on Auto; Vertical and captions off. The markup and the script's
    defaults agree, and the slider's stops are None, Light, Max."""
    html = _web("index.html")
    js = _web("app.js")

    def checked(cid: str) -> bool:
        return " checked" in re.search(rf'<input[^>]*id="{cid}"[^>]*>', html)[0]

    assert [checked(c) for c in ("optSpeech", "optTail", "optLoudnorm", "optLink")] == [True] * 4
    assert checked("optPicksSeq") and not checked("optPicksAuto")
    assert [checked(c) for c in ("optVertical", "optCaptions")] == [False] * 2
    defaults = re.search(r"const OPT_DEFAULTS = \{(.*?)\};", js, re.S)[1]
    for pair in ("speech_aware: true", "smart_tail: true", "picks: 'sequence'",
                 "event_tails: 'auto'", "order: ''", "loudnorm: true", "link: true", "vertical: false",
                 "captions: false", "transition: 'cut'", "audio_transition: 'cut'",
                 "compress_peaks: 'light'"):  # fmt: skip
        assert pair in defaults
    assert "const PEAK_STOPS = ['off', 'light', 'strong'];" in js
    slider = re.search(r'<input type="range" id="optPeaks"[^>]*>', html)[0]
    assert 'min="0"' in slider and 'max="2"' in slider and 'value="1"' in slider  # Light
    # config.json's compress_peaks, when it sets one, beats the Light default
    assert "compress_peaks_set" in js


def test_the_page_sends_the_new_options_with_its_jobs() -> None:
    js = _web("app.js")
    assert "smart_tail: $('#optTail').checked" in js
    assert "speech_aware: $('#optSpeech').checked" in js
    assert "saved.smart_edges" in js  # a stored Smart edges choice is both boxes' default
    assert "audio_transition: $('#optAudioTransition').value" in js


def test_focus_rings_use_the_amber_text_colour_so_light_theme_keeps_contrast() -> None:
    """Contract pin (#462): --amber is a fill (dark ink on it); focus rings and amber text use
    --amber-text, which Light darkens. test_settings_offer_blue_light_and_dark_themes... holds
    the contrast ratios."""
    css = _web("app.css")
    rings = re.findall(r"[^{}]*:focus-visible[^{}]*\{[^}]*outline[^}]*\}", css)
    assert rings
    for rule in rings:
        for colour in re.findall(r"outline(?:-color)?:[^;]*var\((--[\w-]+)\)", rule):
            assert colour == "--amber-text", rule
    assert re.search(r":focus-visible \{ outline: 2px solid var\(--amber-text\)", css)
    light = re.search(r':root\[data-theme="light"\][^{]*\{(.*?)\}', css, re.S)
    assert light and "--amber-text:" in light[1]


# --- Direct controls for clips and extend options (#462) ---


def options_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    body: dict,
    *,
    candidates: int = 3,
) -> tuple[object, list[list[str]], list[Callable[[], None] | None], Path]:
    """Start ``body``'s job on a command-line project with ``candidates`` analysed moments (ids
    h001...), a one-moment review and a vertical export ``clips_ui_1``; the jobs lock check runs,
    so the segments file is written. Returns the response, the argv, the on_success callbacks
    and the project folder."""
    p, seg = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl["candidates"] = [
        {"id": f"h{i:03d}", "source": seg["source"], "start": i, "end": i + 2, "score": 50}
        for i in range(1, candidates + 1)
    ]
    cache.save_json(p / "highlights.json", hl)
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    (p / "clips_ui_1").mkdir()
    cache.save_json(p / "clips_ui_1" / "manifest.json", {"clips": []})
    (p / "clips_ui_1" / "001_a.mp4").write_bytes(b"x")
    cache.save_json(
        p / "ui" / "exports.json", {"exports": [{"dir": "clips_ui_1", "vertical": True}]}
    )
    captured: list[list[str]] = []
    done: list[Callable[[], None] | None] = []

    def start(self, project_id, action, args, on_success, same_folder, *rest):
        captured.append(args)
        done.append(on_success)
        same_folder()  # the check under the jobs lock, which writes the segments file
        return SimpleNamespace(view=lambda: {"status": "running"})

    monkeypatch.setattr(ui.Jobs, "start", start)
    return client.post("/api/projects/cli/jobs", json=body), captured, done, p


def _cli_args(args: list[str]) -> list[str]:
    """``args`` without the interpreter prefix, after checking the CLI parses them as given."""
    assert args[:3] == ui.cutpoint_cmd()
    cli.build_parser().parse_args(args[3:])  # exits (SystemExit) on an option it does not know
    return args[3:]


def _after(args: list[str], flag: str) -> str:
    assert args.count(flag) == 1, args
    return args[args.index(flag) + 1]


def test_the_default_page_options_export_exactly_what_smart_edges_did(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The page's defaults (sequence, Speech-aware cuts and Smart tails on, Event tails Auto,
    blank numbers) build the argv and segments file Smart edges built before #462."""
    page = {"action": "clips", "picks": "sequence", "vertical": False, "speech_aware": True,
            "smart_tail": True, "event_tails": "auto", "compress_peaks": "light"}  # fmt: skip
    old = {"action": "clips", "vertical": False, "smart_edges": True, "compress_peaks": "light"}
    runs = []
    for body in (page, old):
        res, captured, _, p = options_job(client, monkeypatch, tmp_path, body)
        assert res.status_code == 200, res.text
        args = _cli_args(captured[0])
        # the output and segments names carry a per-second stamp: the runs may land in different seconds
        norm = [re.sub(r"\d{8}-\d{6}", "STAMP", a.replace(str(p), "P")) for a in args]
        runs.append((norm, _segments_written(p)))
        shutil.rmtree(p)
    assert runs[0] == runs[1]
    args, segments = runs[0]
    assert args[:3] == ["clips", "P", "--segments"] and args[4] == "--out"
    assert args[6:] == ["--smart-tail", "--speech-aware", "--compress-peaks", "light"]
    assert "smart_tail" not in segments[0] and "speech_aware" not in segments[0]


@pytest.mark.parametrize(("speech", "tail"), [(True, False), (False, True), (False, False)])
def test_clips_job_speech_aware_and_smart_tail_are_separate_choices(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    speech: bool,
    tail: bool,
) -> None:
    body = {"action": "clips", "speech_aware": speech, "smart_tail": tail}
    res, captured, _, p = options_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    assert ("--speech-aware" in args, "--smart-tail" in args) == (speech, tail)
    assert "--no-speech-aware" not in args and "--no-smart-tail" not in args
    (seg,) = _segments_written(p)
    # an exact trim is the segment's own setting; the flag decides otherwise
    assert seg.get("speech_aware", True) is speech and seg.get("smart_tail", True) is tail


def test_clips_job_passes_the_timing_and_padding_fields_it_was_sent(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = {"action": "clips", "speech_aware": True, "smart_tail": True, "event_tails": "on",
            "speech_max_extend": 3, "speech_max_lead": 1.25, "tail_min": 0, "tail_max": 4.5,
            "event_window": 6, "extend_start": 1.5, "extend_end": 0}  # fmt: skip
    res, captured, _, _ = options_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    expect = {"--speech-max-extend": "3", "--speech-max-lead": "1.25", "--tail-min": "0",
              "--tail-max": "4.5", "--event-window": "6", "--extend-start": "1.5",
              "--extend-end": "0"}  # fmt: skip
    assert {flag: _after(args, flag) for flag in expect} == expect
    assert "--event-tails" in args and "--extend" not in args


@pytest.mark.parametrize(
    ("value", "flag"), [("auto", None), ("on", "--event-tails"), ("off", "--no-event-tails")]
)
def test_clips_job_passes_event_tails_on_or_off_and_nothing_for_auto(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str, flag: str
) -> None:
    body = {"action": "clips", "smart_tail": True, "event_tails": value}
    res, captured, _, _ = options_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    assert [a for a in args if "event-tails" in a] == ([flag] if flag else [])


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"picks": "best"}, "picks must be 'sequence' or 'auto'"),
        ({"top": 3}, "need automatic picks"),
        ({"order": "score"}, "need automatic picks"),
        ({"speech_aware": "yes"}, "speech_aware must be true or false"),
        ({"smart_tail": 1}, "smart_tail must be true or false"),
        ({"event_tails": "maybe"}, "event_tails must be"),
        ({"event_tails": ["on"]}, "event_tails must be"),  # not a 500 (unhashable)
        ({"smart_tail": False, "event_tails": "on"}, "event tails need smart tails"),
        ({"speech_max_extend": -1}, "speech_max_extend must be 0-60 seconds"),
        ({"speech_max_lead": 61}, "speech_max_lead must be 0-60 seconds"),
        ({"tail_min": "1"}, "tail_min must be 0-60 seconds"),
        ({"tail_max": True}, "tail_max must be 0-60 seconds"),
        ({"event_window": 0}, "event_window must be above 0-60 seconds"),
        ({"extend_start": 31}, "extend_start must be 0-30 seconds"),
        ({"extend_end": [1]}, "extend_end must be 0-30 seconds"),
    ],
)
def test_clips_job_refuses_option_values_the_cli_would_not_get(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: dict, message: str
) -> None:
    res, captured, _, p = options_job(client, monkeypatch, tmp_path, {"action": "clips", **extra})
    assert res.status_code == 400, res.text
    assert message in res.json()["detail"]
    assert not captured and not list(p.rglob("segments-*.json"))


@pytest.mark.parametrize(
    ("extra", "selection"),
    [
        ({}, []),  # the style's count and order
        ({"top": 2, "max_total": 60, "order": "score"},
         ["--top", "2", "--max-total", "60", "--order", "score"]),
        ({"ids": ["h003", "h001"], "order": "chronological"},
         ["--ids", "h003", "h001", "--order", "chronological"]),
        ({"max_total": 12.5}, ["--max-total", "12.5"]),
    ],
)  # fmt: skip
def test_automatic_picks_hand_the_selection_to_the_cli(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra: dict,
    selection: list[str],
) -> None:
    body = {"action": "clips", "picks": "auto", "speech_aware": False, "smart_tail": True, **extra}
    res, captured, _, p = options_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    assert args[:2] == ["clips", str(p)] and "--segments" not in args
    assert args[2 : 2 + len(selection)] == selection
    assert _after(args, "--out").startswith(str(p / "clips_ui_"))
    # explicit both ways: the style's defaults must not override what the page showed
    assert "--smart-tail" in args and "--no-speech-aware" in args
    assert not list(p.rglob("segments-*.json"))  # no segments file for automatic picks


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"top": 2, "ids": ["h001"]}, "give top or ids, not both"),
        ({"top": 0}, "top must be a whole number"),
        ({"top": 1.5}, "top must be a whole number"),
        ({"top": True}, "top must be a whole number"),
        ({"top": 501}, "top must be a whole number"),
        ({"ids": ["h009"]}, "ids must be different candidate ids"),  # not in the analysis
        ({"ids": ["h001", "h001"]}, "ids must be different candidate ids"),
        ({"ids": ["--force"]}, "ids must be different candidate ids"),
        ({"ids": "h001"}, "ids must be different candidate ids"),
        ({"ids": []}, "ids must be different candidate ids"),
        ({"ids": [1]}, "ids must be different candidate ids"),
        ({"max_total": 0}, "max_total must be above 0"),
        ({"max_total": -5}, "max_total must be above 0"),
        ({"order": "arc"}, "order must be one of story, score, chronological"),
    ],
)
def test_automatic_picks_refuse_a_bad_selection(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: dict, message: str
) -> None:
    body = {"action": "clips", "picks": "auto", **extra}
    res, captured, _, _ = options_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 400, res.text
    assert message in res.json()["detail"]
    assert not captured


def test_automatic_picks_need_an_analysis(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = {"action": "clips", "picks": "auto"}
    res, captured, _, _ = options_job(client, monkeypatch, tmp_path, body, candidates=0)
    assert res.status_code == 400, res.text
    assert "analyze the footage before exporting automatic picks" in res.json()["detail"]
    assert not captured


@pytest.mark.parametrize(
    ("extra", "tail"),
    [
        ({}, ["--no-speech-aware"]),
        ({"extend_start": 1, "extend_end": 2.5, "speech_aware": True, "speech_max_extend": 4},
         ["--start", "1", "--end", "2.5", "--speech-aware", "--speech-max-extend", "4"]),
        ({"extend_end": 3}, ["--end", "3", "--no-speech-aware"]),
    ],
)  # fmt: skip
def test_extend_job_re_exports_the_latest_export_into_a_new_folder(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: dict, tail: list
) -> None:
    res, captured, done, p = options_job(
        client, monkeypatch, tmp_path, {"action": "extend", **extra}
    )
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    out = Path(args[3])
    assert args[:3] == ["extend", str(p / "clips_ui_1"), "--out"] and args[4:] == tail
    assert out.parent == p and out.name.startswith("clips_ui_") and not out.exists()
    # once the CLI wrote the folder, it is the current export (vertical as its source)
    out.mkdir()
    cache.save_json(out / "manifest.json", {"clips": []})
    (out / "001_a.mp4").write_bytes(b"x")
    done[0]()
    exports = cache.load_json(p / "ui" / "exports.json")["exports"]
    assert [(e["dir"], e["vertical"], e["final"]) for e in exports[1:]] == [(out.name, True, None)]
    assert client.get("/api/projects/cli").json()["clips"]["dir"] == out.name


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"extend_start": -1}, "extend_start must be 0-30 seconds"),
        ({"extend_end": "2"}, "extend_end must be 0-30 seconds"),
        ({"speech_aware": "on"}, "speech_aware must be true or false"),
        ({"speech_max_lead": 99}, "speech_max_lead must be 0-60 seconds"),
    ],
)
def test_extend_job_refuses_bad_values(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: dict, message: str
) -> None:
    res, captured, _, _ = options_job(client, monkeypatch, tmp_path, {"action": "extend", **extra})
    assert res.status_code == 400, res.text
    assert message in res.json()["detail"]
    assert not captured


def test_extend_job_needs_an_export(client: TestClient, tmp_path: Path) -> None:
    cli_project(tmp_path)
    res = client.post("/api/projects/cli/jobs", json={"action": "extend"})
    assert res.status_code == 400, res.text
    assert res.json()["detail"] == "export clips before extending them"


def test_the_export_tab_offers_the_direct_controls_with_labels() -> None:
    """Contract pin (#462): each new control is a native, labelled input whose limits match the
    server's, and the page sends each under the option name the server reads."""
    html = _web("index.html")
    js = _web("app.js")
    for cid in ("optTop", "optMaxTotal", "optIds", "optOrder", "optSpeechExtend", "optSpeechLead",
                "optTailMin", "optTailMax", "optEventTails", "optEventWindow", "optExtendStart",
                "optExtendEnd"):  # fmt: skip
        assert re.search(rf'<label for="{cid}"[^>]*>[^<]+</label>', html), cid
    radios = re.findall(r'<input type="radio" name="optPicks"[^>]*value="(\w+)"', html)
    assert radios == ["sequence", "auto"]
    assert re.search(r'id="optPicksSeq"[^>]*checked', html)

    def limits(cid: str) -> tuple[str, str]:
        tag = re.search(rf'<input[^>]*id="{cid}"[^>]*>', html)[0]
        return re.search(r'min="([\d.]+)"', tag)[1], re.search(r'max="([\d.]+)"', tag)[1]

    assert limits("optTop") == ("1", str(ui.MAX_SEGMENTS))
    assert limits("optMaxTotal")[1] == f"{ui.MAX_TOTAL_S:g}"
    assert limits("optExtendStart") == limits("optExtendEnd") == ("0", f"{ui.MAX_EXTEND_S:g}")
    for cid in ("optSpeechExtend", "optSpeechLead", "optTailMin", "optTailMax"):
        assert limits(cid) == ("0", f"{ui.MAX_TIMING_S:g}")
    assert limits("optEventWindow")[1] == f"{ui.MAX_TIMING_S:g}"

    def values(sid: str) -> list[str]:
        return re.findall(r'value="(\w*)"', re.search(rf'<select id="{sid}".*?</select>', html)[0])

    assert sorted(values("optOrder")) == sorted(["", *ui.CLIP_ORDERS])
    assert values("optEventTails") == list(ui.EVENT_TAILS)
    fields = re.search(r"const OPT_FIELDS = \{(.*?)\};", js, re.S)[1]
    for key in ("top", "max_total", "ids", "speech_max_extend", "speech_max_lead", "tail_min",
                "tail_max", "event_window", "extend_start", "extend_end"):  # fmt: skip
        assert re.search(rf"\b{key}: '#opt\w+'", fields), key
    assert "startJob('extend', body)" in js and "extendOptions()" in js
    assert "extend: 'Export extension'" in js  # the Jobs bar names the job


# --- One-click presets: cutpoint edit --kind (#462) ---


def preset_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    body: dict,
    *,
    settings: dict | None = None,
    meta: dict | None = None,
) -> tuple[object, list[list[str]], list[Callable[[], None] | None], Path]:
    """Start ``body``'s edit job on a command-line project analysed in the funny style (extra
    ``settings`` recorded, an optional ``ui/meta.json``). Returns as :func:`options_job`."""
    p, _ = cli_project(tmp_path)
    hl = cache.load_json(p / "highlights.json")
    hl["style"] = "funny"
    hl["settings"].update(settings or {})
    cache.save_json(p / "highlights.json", hl)
    if meta is not None:
        write(p / "ui" / "meta.json", json.dumps(meta))
    captured: list[list[str]] = []
    done: list[Callable[[], None] | None] = []

    def start(self, project_id, action, args, on_success, same_folder, *rest):
        captured.append(args)
        done.append(on_success)
        same_folder()
        return SimpleNamespace(view=lambda: {"status": "running"})

    monkeypatch.setattr(ui.Jobs, "start", start)
    return client.post("/api/projects/cli/jobs", json={"action": "edit", **body}), captured, done, p


@pytest.mark.parametrize("kind", list(edit.KINDS))
def test_a_preset_runs_edit_with_its_kind_in_the_projects_own_style(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    res, captured, _, p = preset_job(client, monkeypatch, tmp_path, {"kind": kind})
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    src = str(tmp_path / "clip.mp4")
    assert args == ["edit", src, "--kind", kind, "--style", "funny", "--out", str(p),
                    "--compress-peaks", "off", "--no-vision"]  # fmt: skip


def test_a_preset_passes_vertical_compress_peaks_and_the_settings_vision(
    client: TestClient, home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    set_config(home, vision=True)
    body = {"kind": "montage", "vertical": True, "compress_peaks": "strong"}
    settings = {"vision_top": 12, "ollama_model": "m:1"}  # what the analysis recorded
    res, captured, _, _ = preset_job(client, monkeypatch, tmp_path, body, settings=settings)
    assert res.status_code == 200, res.text
    args = _cli_args(captured[0])
    assert "--vertical" in args and _after(args, "--compress-peaks") == "strong"
    assert args[-3:] == ["--vision", "--vision-top=12", "--ollama-model=m:1"]
    assert "--force" not in args


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({}, "kind must be one of highlights, montage, clips, funny-moments"),
        ({"kind": "trailer"}, "kind must be one of"),
        ({"kind": ["clips"]}, "kind must be one of"),  # not a 500 (unhashable)
        ({"kind": {"k": 1}}, "kind must be one of"),
        ({"kind": "clips", "vertical": "yes"}, "vertical must be true or false"),
        ({"kind": "clips", "force": 1}, "force must be true or false"),
        ({"kind": "clips", "compress_peaks": "max"}, "compress_peaks must be one of"),
    ],
)
def test_a_preset_refuses_bad_values(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: dict, message: str
) -> None:
    res, captured, _, _ = preset_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 400, res.text
    assert message in res.json()["detail"]
    assert not captured


@pytest.mark.parametrize(
    ("settings", "meta", "reason"),
    [
        ({"exclude": ["*.lrv"]}, None, "--exclude or --recursive"),
        ({"recursive": True}, None, "--exclude or --recursive"),
        ({}, {"kind": "360", "style": "funny", "source": "x"}, "360 projects"),
        # edit has no --model or --no-scenes: it would transcribe or detect scenes again
        ({"whisper_model": "large-v3", "whisper_device": "cuda"}, None, "large-v3 model"),
        ({"whisper_model": "medium", "whisper_device": "cpu"}, None, "medium model"),
        ({"scenes": False}, None, "--no-scenes"),
        ({"recover": True}, None, "--recover"),
        ({"language": "en"}, None, "--language"),
        ({"max_candidates": 40}, None, "--max-candidates"),
    ],
)
def test_presets_are_off_with_the_reason_where_edit_cannot_remake_the_analysis(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: dict,
    meta: dict | None,
    reason: str,
) -> None:
    body = {"kind": "highlights"}
    res, captured, _, _ = preset_job(
        client, monkeypatch, tmp_path, body, settings=settings, meta=meta
    )
    assert res.status_code == 400, res.text
    assert reason in res.json()["detail"] and not captured
    assert reason in client.get("/api/projects/cli").json()["preset_problem"]


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {
            "whisper_model": "small",
            "whisper_device": "cpu",
            "scenes": True,
            "language": None,
            "max_candidates": None,
            "recover": False,
        },
        {"whisper_model": "medium", "whisper_device": "cuda", "exclude": [], "recursive": False},
    ],
)
def test_a_project_edit_can_remake_has_no_preset_problem(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, settings: dict
) -> None:
    res, *_ = preset_job(client, monkeypatch, tmp_path, {"kind": "clips"}, settings=settings)
    assert res.status_code == 200, res.text
    view = client.get("/api/projects/cli").json()
    assert (view["preset_problem"], view["preset_taken"]) == (None, [])


@pytest.mark.parametrize("vertical", [False, True])
def test_a_preset_replaces_an_earlier_runs_clips_only_when_confirmed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, vertical: bool
) -> None:
    name = "clips_vertical" if vertical else "clips"
    body = {"kind": "clips", "vertical": vertical}
    res, captured, _, p = preset_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 200 and "--force" not in captured[0]  # nothing there yet
    write(p / name / "001_a.mp4")  # what that run exported
    assert client.get("/api/projects/cli").json()["preset_taken"] == [name]
    res = client.post("/api/projects/cli/jobs", json={"action": "edit", **body})
    assert res.status_code == 409, res.text
    assert "confirm to replace" in res.json()["detail"] and len(captured) == 1
    res = client.post("/api/projects/cli/jobs", json={"action": "edit", **body, "force": True})
    assert res.status_code == 200, res.text
    assert "--force" in _cli_args(captured[1])


def test_a_finished_preset_is_the_current_export_with_its_final(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    res, _, done, p = preset_job(client, monkeypatch, tmp_path, {"kind": "montage"})
    assert res.status_code == 200, res.text
    # an earlier run's entry for the same folder is replaced, other exports stay
    cache.save_json(
        p / "ui" / "exports.json",
        {"exports": [{"dir": "clips", "final": "old.mp4"}, {"dir": "clips_ui_1", "final": None}]},
    )
    with pytest.raises(RuntimeError, match="without its clips and final"):
        done[0]()
    write(p / "clips" / "001_a.mp4")
    cache.save_json(p / "clips" / "manifest.json", {"clips": []})
    write(p / "final.mp4")
    done[0]()
    exports = cache.load_json(p / "ui" / "exports.json")["exports"]
    assert [(e["dir"], e["final"], e.get("vertical")) for e in exports] == [
        ("clips_ui_1", None, None),
        ("clips", "final.mp4", False),
    ]
    view = client.get("/api/projects/cli").json()
    assert view["clips"]["dir"] == "clips" and view["final_current"] is True


def test_the_export_tab_offers_the_presets_and_asks_before_replacing() -> None:
    """Contract pin (#462): the preset select lists edit's kinds; Make sends ``force`` only after
    the shared confirm dialog, and the page shows the server's reason when presets are off."""
    html = _web("index.html")
    js = _web("app.js")
    assert re.search(r'<label for="optKind"[^>]*>Preset</label>', html)
    kinds = re.findall(
        r'value="([\w-]+)"', re.search(r'<select id="optKind".*?</select>', html, re.S)[0]
    )
    assert kinds == list(edit.KINDS)
    make_btn = re.search(r'<button type="button" id="makeBtn"[^>]*>Make</button>', html)[0]
    assert 'aria-describedby="dMake presetWhy"' in make_btn  # the reason it is off, if it is
    # short option text, so the selects read fully in a 290 px column (details in titles)
    for sid in ("optKind", "optOrder"):
        select = re.search(rf'<select id="{sid}".*?</select>', html, re.S)[0]
        names = re.findall(r">([^<]+)</option>", select)
        assert names and max(map(len, names)) <= 14, names
    make = re.search(r"async function makePreset\(\) \{\n(.*?)\n\}", js, re.S)[1]
    # the folders as they are now: refreshed before deciding whether to ask
    assert make.index("await refreshProject()") < make.index("preset_taken")
    assert make.index("confirmAction('Replace the clips of the earlier preset run?'") < make.index(
        "body.force = true"
    )
    assert "startJob('edit', body)" in make and "preset_taken" in make
    assert "p?.preset_problem" in js and "#presetWhy" in js


# --- Web contract pins for the AI Chat tab (#346) ---


def _chat_js() -> str:
    """app.js's AI Chat section."""
    js = _web("app.js")
    return js[js.index("// ---------- AI Chat (#346)") : js.index("// ---------- jobs")]


def _js_function(js: str, name: str) -> str:
    """The body of top-level function ``name`` in ``js``."""
    found = re.search(rf"^(?:async )?function {name}\((.*?)\) \{{\n(.*?)^\}}", js, re.M | re.S)
    assert found, name
    return found[2]


def test_the_ai_chat_tab_has_one_labelled_input_and_one_send_button() -> None:
    """Contract pin (#346): an AI Chat tab beside Sequence; the panel names the assistant, has a
    Clear chat button, one labelled input limited as the server limits it, one send button and
    no mic or attach button; replies are announced from outside the panels, so a hidden tab's
    reply is still read out.

    Check in a browser (no browser in the test suite): in Blue, Light and Dark, at a 290 px
    column and in the narrow layout, the input stays at the bottom and nothing overflows."""
    html = _web("index.html")
    tab = re.search(r'<button\b[^>]*id="tabAssistant"[^>]*>([^<]*)</button>', html)
    assert tab and 'role="tab"' in tab[0] and tab[1] == "AI Chat"
    panel = re.search(r'<div id="panelAssistant".*?\n      </div>\n', html, re.S)[0]
    assert re.search(r'<h2 id="chatTitle"[^>]*>AI Assistant</h2>', panel)
    assert re.search(r'<button type="button" id="chatClear"[^>]*>Clear chat</button>', panel)
    buttons = re.findall(r"<button\b[^>]*>[^<]*</button>", panel)
    assert [b for b in buttons if 'type="submit"' in b] == [
        '<button type="submit" id="chatSend" class="btn primary" disabled>Send</button>'
    ]
    # Clear chat, the hosted-provider consent (#346) and Send; no mic, no attach
    assert [re.search(r'id="(\w+)"', b)[1] for b in buttons] == [
        "chatClear",
        "chatConsentOk",
        "chatSend",
    ]
    assert not re.search(r"\b(mic|microphone|attach|voice)\b", panel, re.I)
    fields = re.findall(r"<(?:input|textarea)\b[^>]*>", panel)
    assert len(fields) == 1 and 'id="chatInput"' in fields[0]
    assert f'maxlength="{ai_session.MAX_TEXT}"' in fields[0]
    assert '<label for="chatInput"' in panel
    side = re.search(r'<section class="col side".*?</section>\s*</main>', html, re.S)[0]
    live = '<p id="chatLive" class="vh" role="status"></p>'
    assert live in side and side.index(live) < side.index('<div id="panelSequence"')
    chat = _chat_js()
    assert "$('#chatLive').textContent = turnNews(t)" in chat
    # the header is the provider's label (name, model, host); the small-model note
    assert "prov ? prov.label" in chat and "prov?.local" in chat


def test_model_text_is_never_set_as_markup() -> None:
    """Contract pin (#346): replies, summaries, labels and notes come from the model or the
    project; the page sets them as text only (h()'s ``text`` or textContent)."""
    sinks = ("innerHTML", "outerHTML", "insertAdjacentHTML", "DOMParser", "document.write")
    for name in ("app.js", "util.js", "api.js", "picker.js"):
        js = _web(name)
        for sink in sinks:
            assert sink not in js, (name, sink)
    chat = _chat_js()
    assert "h('p', { text: ev.text })" in chat and "h('p', { text: ev.summary })" in chat
    assert "$('#propSummary').textContent = p.summary;" in chat
    assert "else if (k === 'text') el.textContent = v;" in _web("util.js")


def test_a_proposal_is_previewed_before_it_replaces_the_working_sequence() -> None:
    """Contract pin (#346): a proposal only sets S.proposal, which the Sequence tab previews with
    its changes in words, the moments list marks by class and the rail shows as ghost handles
    in their own element and function. Apply posts the working sequence and takes the server's
    result (rebased, or a 409 in words); suggestions are at most three and preview the same
    way; Save still uses S.segments only."""
    html = _web("index.html")
    js = _web("app.js")
    chat = _chat_js()
    assert "  proposal: null," in js
    assert re.search(r'<button type="button" id="propApply"[^>]*>Apply</button>', html)
    assert re.search(r'<button type="button" id="propDiscard"[^>]*>Discard</button>', html)
    rail = re.search(r'<div class="rail" id="rail">.*?<div class="rail-ends', html, re.S)[0]
    assert '<div class="rail-ghost" id="railGhost" aria-hidden="true" hidden></div>' in rail
    draw = _js_function(js, "drawRail")
    assert "drawGhost(c);" in draw and "railGhost" not in draw  # its own function
    assert "$('#railGhost')" in _js_function(js, "drawGhost")
    moments = _js_function(js, "renderMoments")
    assert "markMoments();" in moments and "ai-" not in moments  # marked apart, by class
    marks = _js_function(chat, "markMoments")
    assert "classList.toggle" in marks and "replaceChildren" not in marks
    on_event = _js_function(chat, "onChatEvent")
    assert "previewProposal(ev)" in on_event and "S.segments" not in on_event
    apply = _js_function(chat, "applyProposal")
    assert (
        "/applied`), { method: 'POST', body: { segments: JSON.parse(key()), base_key: p.base_key } }"
        in apply
    )
    assert apply.index("await api(") < apply.index("remember();")
    # "changed since" compares with the sequence the proposal was made on (its diff's "before"),
    # not with the sequence when the preview opened
    preview = _js_function(chat, "previewProposal")
    assert "seqKey: baseOf(p.diff)" in preview and "key()" not in preview
    assert "seqShape() === p.seqKey" in _js_function(chat, "renderProposal")
    # while Apply waits: Discard is off and a new proposal waits for it
    assert "S.pendingProposal = p;" in preview
    assert "$('#propDiscard').disabled = !S.proposal || S.applying;" in chat
    assert "previewProposal(next)" in apply
    assert "revertChange(" in apply  # a revert previewed on an older sequence is previewed again
    # ... and finally's pending proposal does not replace that new preview
    again = apply.index("await revertChange(")
    assert apply.index("if (!err) S.pendingProposal = null;", again) < apply.index("finally")
    assert apply.index("remember();") < apply.index("replaceSequence(entry.after);")
    assert "e.status === 409" in apply and "entry.rebased" in apply
    assert "const MAX_SUGGESTIONS = 3;" in chat and ".slice(0, MAX_SUGGESTIONS)" in chat
    assert "onclick: () => reviewProposal(s)" in chat  # a suggestion previews like a proposal
    assert "'/turns'), { method: 'POST', body: { text, segments: JSON.parse(key()) } }" in chat
    assert "?after=${run.next}" in chat and "/cancel`), { method: 'POST' }" in chat
    revert = _js_function(chat, "revertChange")
    assert "/revert`), { method: 'POST'" in revert and "reviewProposal(p)" in revert
    # Send waits for the saved chat, which would otherwise replace a turn sent meanwhile
    assert "!!S.chat?.loaded" in _js_function(chat, "renderChatForm")
    assert "!chat?.loaded" in _js_function(chat, "sendChat")
    clear = _js_function(chat, "clearChat")
    assert "remember" not in clear and "S.segments" not in clear  # Clear chat never reverts
    assert "proposal" not in _js_function(js, "saveReview")


def test_one_undo_stack_holds_manual_and_ai_edits_with_ctrl_z() -> None:
    """Contract pin (#346): every edit of the working sequence (add, remove, move, trims,
    rename, batch remove, Apply) first pushes a snapshot of the segments and drafts onto one
    stack of at most 50; Ctrl+Z undoes, Ctrl+Y or Shift+Ctrl+Z redoes; text fields keep their
    own undo."""
    html = _web("index.html")
    js = _web("app.js")
    assert "  undo: []," in js and "  redo: []," in js and "const UNDO_MAX = 50;" in js
    assert "if (S.undo.length > UNDO_MAX) S.undo.shift();" in js
    snap = re.search(r"const snapshot = \(\) => \(\{(.*?)\}\);", js)[1]
    assert "structuredClone(S.segments)" in snap and "structuredClone(S.drafts)" in snap
    for name in ("addSeg", "removeSeg", "moveSeg", "removeSelected", "finishRename"):
        assert "remember();" in _js_function(js, name), name
    assert "remember(`trim:${c.id}:${which}`)" in _js_function(js, "setRange")
    assert "remember();" in _js_function(js, "applyProposal")
    keys = js[js.index("// Ctrl+Z / Ctrl+Y") : js.index("if (e.key === '[' || e.key === ']')")]
    assert "(k === 'z' || k === 'y')" in keys and "textarea" in keys
    assert "if (k === 'y' || e.shiftKey) redo();" in keys and "else undo();" in keys
    for bid, shortcut in (("undoBtn", "Control+Z"), ("redoBtn", "Control+Y Control+Shift+Z")):
        assert re.search(
            rf'<button type="button" id="{bid}"[^>]*aria-keyshortcuts="{re.escape(shortcut)}"', html
        )


# --- Moments list order (#469) ---


def _sort_positions(tmp: Path, moments: list[dict]) -> list[str]:
    """Run util.js's positionCompare under node on ``moments``; return their ids in sorted order.

    util.js is copied to ``util.mjs`` so every node >= 18 loads it as an ES module: there is no
    package.json, and node before 22.7 would read a ``.js`` file as CommonJS. util.js has no
    imports of its own, so the copy runs on its own."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    mod = tmp / "util.mjs"
    shutil.copyfile(ui.WEB_DIR / "util.js", mod)
    url = mod.resolve().as_uri()
    script = (
        f"import {{ positionCompare }} from {json.dumps(url)};"
        "const m = JSON.parse(process.argv[1]);"
        "console.log(JSON.stringify(m.sort(positionCompare).map((c) => c.id)));"
    )
    out = subprocess.run(
        [node, "--input-type=module", "-e", script, json.dumps(moments)],
        capture_output=True, text=True, check=True, timeout=30,
    )  # fmt: skip
    return json.loads(out.stdout)


def _m(id_: str, source: str, start: float) -> dict:
    return {"id": id_, "source": source, "start": start}


def test_position_order_is_natural_by_file_then_start_time(tmp_path: Path) -> None:
    """#469: video order groups moments by source (clip2 before clip10, A-Z), then start time."""
    moments = [
        _m("c10b", "/v/clip10.mp4", 5),
        _m("c2late", "/v/clip2.mp4", 90),
        _m("a", "/v/Alpha.mp4", 1),
        _m("c10a", "/v/clip10.mp4", 2),
        _m("c2early", "/v/clip2.mp4", 3),
    ]
    assert _sort_positions(tmp_path, moments) == ["a", "c2early", "c2late", "c10a", "c10b"]


def test_position_order_is_stable_and_keeps_same_named_files_apart(tmp_path: Path) -> None:
    """#469: equal starts keep score order (stable), case variants and same-named files in two
    folders sort the same whatever the input order, and each folder's file stays together."""
    ties = [_m("first", "/v/a.mp4", 4), _m("second", "/v/a.mp4", 4), _m("third", "/v/a.mp4", 4)]
    assert _sort_positions(tmp_path, ties) == ["first", "second", "third"]
    mixed = [
        _m("x1", "/two/clip.mp4", 1), _m("y1", "/one/clip.mp4", 50),
        _m("x2", "/two/clip.mp4", 9), _m("y0", "/one/clip.mp4", 2),
    ]  # fmt: skip
    got = _sort_positions(tmp_path, mixed)
    assert _sort_positions(tmp_path, mixed[::-1]) == got
    assert got == ["y0", "y1", "x1", "x2"]
    cased = [_m("up", "/v/Clip.mp4", 1), _m("low", "/v/clip.mp4", 1), _m("z", "/v/zed.mp4", 0)]
    got = _sort_positions(tmp_path, cased)
    assert _sort_positions(tmp_path, cased[::-1]) == got and got[-1] == "z"


def test_moment_order_control_is_labelled_with_score_default() -> None:
    """#469: a labelled select beside the filter offers Score (default) and Position in video;
    a status region announces the change."""
    html = _web("index.html")
    label = re.search(r'<label class="order">(.*?)</label>', html, re.S)[1]
    assert '<span class="vh">Sort moments by</span>' in label
    sel = re.search(r'<select id="momentOrder">(.*?)</select>', label, re.S)[1]
    assert re.findall(r'<option value="(\w+)">([^<]+)</option>', sel) == [
        ("score", "Score"),
        ("position", "Position in video"),
    ]
    assert re.search(r'<p id="orderLive"[^>]*role="status"', html)
    assert html.index('id="filter"') < html.index('id="momentOrder"')


def test_moment_order_is_remembered_safely_and_sorts_after_filtering() -> None:
    """#469: storage access is guarded, the sort runs after the filter, and the rank badge
    still comes from the score rank."""
    js = _web("app.js")
    init = re.search(r"function initMomentOrder\(\) \{(.*?)\n\}\n", js, re.S)[1]
    assert re.search(r"try \{[^}]*localStorage\.getItem\(ORDER_KEY\)[^}]*\} catch", init, re.S)
    assert re.search(r"try \{[^}]*localStorage\.(set|remove)Item\(ORDER_KEY", init, re.S)
    assert "'cutpoint.momentOrder'" in js
    render = re.search(r"function renderMoments\(\) \{(.*?)\n\}\n", js, re.S)[1]
    assert render.index(".filter(") < render.index("positionCompare")
    assert re.search(r"\.map\(\(c, i\) => \(\{ c, rank: i \+ 1 \}\)\)", render)


# --- Color (#461): the export's --color and --color-strength ---


def color_job(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action: str,
    body: dict,
) -> tuple[object, list[list[str]]]:
    """Start a clips (the sequence) or edit (a preset) job with the Color fields ``body``."""
    if action == "edit":
        res, captured, _, _ = preset_job(client, monkeypatch, tmp_path, {"kind": "clips", **body})
        return res, captured
    return peaks_job(client, monkeypatch, tmp_path, {"action": action, **body})


def _color_flags(args: list[str]) -> list[str]:
    """The ``--color`` and ``--color-strength`` flags of ``args``, each with its value."""
    out: list[str] = []
    for i, a in enumerate(args):
        if a in ("--color", "--color-strength"):
            out += args[i : i + 2]
    return out


@pytest.mark.parametrize("action", ["clips", "edit"])
@pytest.mark.parametrize(
    ("body", "passed"),
    [
        ({"color": "auto"}, ["--color", "auto"]),
        ({"color": "underwater"}, ["--color", "underwater"]),
        ({"color": "off"}, ["--color", "off"]),  # explicit: config.json's color does not apply
        ({"color": "auto", "color_strength": 0.7}, ["--color", "auto", "--color-strength", "0.7"]),
        ({"color": "underwater", "color_strength": 0}, ["--color", "underwater", "--color-strength", "0"]),
        ({"color": "auto", "color_strength": 1}, ["--color", "auto", "--color-strength", "1"]),
        ({}, []),  # not sent: the CLI applies config.json's color itself (never to gaming)
    ],
)  # fmt: skip
def test_clips_and_edit_jobs_pass_color_and_its_strength(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action: str,
    body: dict,
    passed: list[str],
) -> None:
    res, captured = color_job(client, monkeypatch, tmp_path, action, body)
    assert res.status_code == 200, res.text
    assert _color_flags(_cli_args(captured[0])) == passed  # and the CLI parses them


@pytest.mark.parametrize("action", ["clips", "edit"])
@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"color": "max"}, "color must be one of off, auto, underwater"),
        ({"color": ""}, "color must be one of"),
        ({"color": True}, "color must be one of"),
        ({"color": ["auto"]}, "color must be one of"),  # not a 500
        ({"color": 1}, "color must be one of"),
        ({"color": "auto", "color_strength": 1.5}, "color_strength must be a number from 0 to 1"),
        ({"color": "auto", "color_strength": -0.1}, "color_strength must be a number from 0 to 1"),
        ({"color": "auto", "color_strength": "0.5"}, "color_strength must be a number from 0 to 1"),
        ({"color": "auto", "color_strength": True}, "color_strength must be a number from 0 to 1"),
        ({"color": "off", "color_strength": 0.5}, "color_strength needs color auto or underwater"),
        ({"color_strength": 0.5}, "color_strength needs color auto or underwater"),
        ({"color": "auto", "lut": "flat.cube"}, "color cannot be combined with lut"),
        ({"color": "underwater", "lut": "flat.cube"}, "color cannot be combined with lut"),
        ({"color": "auto", "color_strength": 0.5, "lut": "flat.cube"}, "color cannot be combined with lut"),
    ],
)  # fmt: skip
def test_clips_and_edit_jobs_reject_a_bad_color(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action: str,
    body: dict,
    message: str,
) -> None:
    res, captured = color_job(client, monkeypatch, tmp_path, action, body)
    assert res.status_code == 400, res.text
    assert message in res.json()["detail"]
    assert not captured


def test_extend_sends_no_color_it_regrades_from_the_manifest(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``extend`` has no --color: it re-applies the grade each clip's manifest entry records."""
    body = {"action": "extend", "speech_aware": True, "color": "auto"}
    res, captured = peaks_job(client, monkeypatch, tmp_path, body)
    assert res.status_code == 200, res.text
    assert _color_flags(_cli_args(captured[0])) == []


@pytest.mark.parametrize(
    ("cfg", "shown"),
    [(None, "off"), ("auto", "auto"), ("underwater", "underwater"), ("off", "off"),
     ("pink", "off"), (3, "off"), (["auto"], "off")],
)  # fmt: skip
def test_settings_show_the_color_default(
    client: TestClient, home: Path, cfg: object, shown: str
) -> None:
    """The page's Color starts on config.json's color; unset or invalid shows Off."""
    if cfg is not None:
        set_config(home, color=cfg)
    assert client.get("/api/settings").json()["color"] == shown


@pytest.mark.parametrize(("style", "gaming"), [("gaming", True), ("funny", False), (None, False)])
def test_the_project_view_says_whether_the_footage_is_gaming(
    client: TestClient, tmp_path: Path, style: str | None, gaming: bool
) -> None:
    """The page's Color starts at Off for gaming footage, as config.json's color does."""
    p, _ = cli_project(tmp_path)
    if style is not None:
        hl = cache.load_json(p / "highlights.json")
        hl["style"] = style
        cache.save_json(p / "highlights.json", hl)
    assert client.get("/api/projects/cli").json()["gaming"] is gaming


def test_color_is_a_three_stop_slider_like_compress_audio_peaks() -> None:
    """Contract pin (#461): Color is a range slider with the stops Off, Auto and Underwater, built
    like Compress Audio Peaks, with a 0-100% Strength shown only with a grade; both name their
    value for screen readers. The page has no LUT option, so nothing disables Color."""
    html = _web("index.html")
    js = _web("app.js")
    panel = re.search(r'<fieldset class="panel color">(.*?)</fieldset>', html, re.S)[1]
    slider = re.search(r'<input type="range" id="optColor"[^>]*>', panel)[0]
    for attr in ('min="0"', 'max="2"', 'step="1"', 'value="0"', 'aria-labelledby="colorLegend"',
                 'aria-describedby="dColor"', 'aria-valuetext="Off"'):  # fmt: skip
        assert attr in slider
    assert re.search(r'<legend id="colorLegend"[^>]*>Color</legend>', panel)
    ticks = re.search(r'<div class="ticks" aria-hidden="true">(.*?)</div>', panel)[1]
    assert re.findall(r"<span>(\w+)</span>", ticks) == ["Off", "Auto", "Underwater"]
    row = re.search(
        r'<div class="color-strength" id="colorStrengthRow" hidden>(.*?)</div>', panel, re.S
    )[1]
    strength = re.search(r'<input type="range" id="optColorStrength"[^>]*>', row)[0]
    for attr in ('min="0"', 'max="100"', 'value="100"', 'aria-valuetext="Strength 100%"'):
        assert attr in strength
    assert '<label for="optColorStrength"' in row
    assert "const COLOR_STOPS = ['off', 'auto', 'underwater'];" in js
    assert "underwater: 'Underwater'" in js
    assert "`Strength ${range.value}%`" in js
    assert "$('#colorStrengthRow').hidden = color() === 'off';" in js
    assert not re.search(r"\blut\b", html, re.I)  # no LUT option that would disable Color


def test_the_page_sends_color_with_clips_and_presets_and_strength_below_100() -> None:
    """Contract pin (#461): Export clips and One-Click Edit send ``color``; ``color_strength``
    only with a grade and below 100% (the CLI's default is 1). Extend export and Assemble send
    neither."""
    js = _web("app.js")
    options = re.search(r"function colorOptions\(\) \{\n(.*?)\n\}", js, re.S)[1]
    assert "{ color: color() }" in options
    assert "body.color !== 'off' && pct < 100" in options and "pct / 100" in options
    for builder, sends in (("exportOptions", True), ("makePreset", True),
                           ("extendOptions", False), ("assembleOptions", False)):  # fmt: skip
        body = re.search(rf"function {builder}\(\) \{{\n(.*?)\n\}}", js, re.S)[1]
        assert ("...colorOptions()" in body) is sends, builder


def test_color_starts_from_config_json_but_never_on_for_gaming() -> None:
    """Contract pin (#461): a project opens with the Color this browser saved for it (so a dive's
    Underwater does not carry over to land footage), else config.json's color (GET
    /api/settings), else Off; a gaming project opens at Off. Strength is one choice for all."""
    js = _web("app.js")
    start = re.search(r"function startColor\(\) \{\n(.*?)\n\}", js, re.S)[1]
    assert "S.savedOpts.color_by_project?.[S.id]" in start and "S.colorDefault" in start
    assert "setColor(S.project?.gaming ? 'off' : value);" in start
    init = re.search(r"function initExportOptions\(\) \{\n(.*?)\n\}\n", js, re.S)[1]
    assert "store('color_by_project', { ...mine, [S.id]: color() });" in init
    assert "store('color'," not in init  # no browser-wide Color
    assert "store('color_strength', Number(strength.value));" in init  # browser-wide Strength
    assert "colorDefault: 'off'" in js
    assert "if (COLOR_STOPS.includes(st.color)) S.colorDefault = st.color;" in js
    adopt = re.search(r"function adopt\(p, switched\) \{\n(.*?)\n\}", js, re.S)[1]
    assert "if (switched) startColor();" in adopt
