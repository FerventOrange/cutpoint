"""Backend of ``cutpoint ui``: trust boundaries, review validation, media Range, job semantics."""

from __future__ import annotations

import asyncio
import json
import logging
import os
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
from cutpoint import cache, cli, config, encode, ui
from cutpoint.utils import find_videos

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
    monkeypatch.setattr(ui, "find_encoder", lambda kind: (None, [f"{kind}: not probed"]))


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
    monkeypatch.setattr(ui, "cutpoint_cmd", lambda *_a: [sys.executable, "-c", "pass"])
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "analyze"})
    assert res.status_code == 200, res.text
    assert [s["step"] for s in res.json()["progress"]["plan"]] == [
        "scan", "views", "analyze", "events",
    ]  # fmt: skip
    assert client.app.state.jobs.get(res.json()["id"]).done.wait(20)


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


def test_media_supports_range_and_only_serves_project_sources(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = SimpleNamespace(
        vcodec="h264", pix_fmt="yuv420p", acodec="aac", duration=10.0, audio_streams=1
    )
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
    client.app.state.jobs.jobs["j1"] = SimpleNamespace(status="running", proc=None)
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


@pytest.mark.slow
@pytest.mark.parametrize("refused", [False, True], ids=["crash", "refused-write"])
def test_a_failing_post_processing_step_fails_the_job_with_a_clean_message(
    caplog: pytest.LogCaptureFixture, refused: bool
) -> None:
    def on_success() -> None:
        raise ui.bad("x", 409) if refused else RuntimeError("boom")

    jobs = ui.Jobs()
    with caplog.at_level(logging.ERROR):
        job = jobs.start("p", "clips", [sys.executable, "-c", "pass"], on_success=on_success)
        assert job.done.wait(20)
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


@pytest.mark.slow
def test_a_command_line_project_without_a_ui_folder_can_be_reviewed_and_exported(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, seg = cli_project(tmp_path)
    assert not (p / "ui").exists()
    assert client.put("/api/projects/cli/review", json={"segments": [seg]}).status_code == 200
    assert client.get("/api/projects/cli").json()["review"]["segments"][0]["id"] == "h001"

    monkeypatch.setattr(ui, "cutpoint_cmd", lambda *_a: [sys.executable, "-c", "pass"])
    res = client.post("/api/projects/cli/jobs", json={"action": "clips"})
    assert res.status_code == 200, res.text
    assert client.app.state.jobs.get(res.json()["id"]).done.wait(20)
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

        def __init__(self, args: list[str]) -> None:
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


@pytest.mark.slow
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
    seen: list[tuple[str, ...]] = []

    def cmd(*args: str) -> list[str]:
        seen.append(args)
        return [sys.executable, "-c", "pass"]

    monkeypatch.setattr(ui, "cutpoint_cmd", cmd)
    res = client.post(f"/api/projects/{pid}/jobs", json={"action": "assemble"})
    assert res.status_code == 200, res.text
    assert client.app.state.jobs.get(res.json()["id"]).done.wait(20)
    assert seen[0][seen[0].index("--out") + 1] == str(p / "final_ui_1-2.mp4")
    assert (p / "final_ui_1.srt").read_text(encoding="utf-8") == "keep me"


class FakeProc:
    """A job process that stays alive until ``release`` is set, then exits 0."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.stdout = self

    def __iter__(self):
        self.release.wait(10)
        return iter(())

    def wait(self, timeout: float | None = None) -> int:
        return 0


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

        def __init__(self, _args: list[str]) -> None:
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


@pytest.mark.parametrize("nvenc", [False, True])
def test_proxy_has_short_keyframe_interval_and_at_most_30_fps(nvenc: bool) -> None:
    args = ui.proxy_args(Path("in.mkv"), Path("out.mp4"), 1, nvenc=nvenc)
    assert args[args.index("-g") + 1] == "30"
    assert args[args.index("-fpsmax") + 1] == "30"
    codec = args[args.index("-c:v") + 1]
    assert codec == ("h264_nvenc" if nvenc else "libx264")
    assert ("-maxrate" in args) is nvenc  # NVENC's size is capped near libx264's
    assert args[-1] == "out.mp4"


class ProxyEncoder:
    """A fake proxy ffmpeg: records its command; fails when it is NVENC and ``nvenc_fails``."""

    calls: list[list[str]]
    nvenc_fails = False

    def __init__(self, args: list[str]) -> None:
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
    ("probed", "setting", "codec"),
    [
        (True, None, "h264_nvenc"),
        (True, "nvenc", "h264_nvenc"),
        (False, None, "libx264"),
        (True, "cpu", "libx264"),  # the user chose the CPU for every encode
        (True, "amf", "libx264"),
    ],
)
def test_proxy_uses_nvenc_when_it_works_here_else_libx264(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    home: Path,
    hevc_project: tuple[str, Path, Path, Path],
    proxy_encoder: type[ProxyEncoder],
    probed: bool,
    setting: str | None,
    codec: str,
) -> None:
    if setting:
        set_config(home, video_encoder=setting)
    enc = ui.VideoEncoder("nvenc") if probed else None
    monkeypatch.setattr(ui, "find_encoder", lambda kind: (enc if kind == "nvenc" else None, []))
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
    monkeypatch.setattr(ui, "find_encoder", lambda kind: (ui.VideoEncoder("nvenc"), []))
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
    monkeypatch.setattr(ui, "find_encoder", lambda kind: (ui.VideoEncoder("nvenc"), []))

    class Failing(ProxyEncoder):
        def __init__(self, args: list[str]) -> None:
            self.calls.append(args)
            self.returncode = 1

    monkeypatch.setattr(ui, "spawn", Failing)
    _, folder, p, _ = hevc_project
    assert build_proxy(client, p, folder / "a.mp4") == "failed"
    assert len(proxy_encoder.calls) == 2
    assert not client.app.state.media.nvenc_failed  # libx264 did not work either
    assert not list(p.rglob("*.part.mp4"))


def test_no_proxy_ffmpeg_starts_after_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[list[str]] = []
    monkeypatch.setattr(ui, "spawn", spawned.append)
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
        def __init__(self, args: list[str]) -> None:
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


class FakeWindowProc:
    """A fake ffmpeg: a window encode stays alive until finished or killed; a whole one ends."""

    def __init__(self, args: list[str], held: bool) -> None:
        self.args = args
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

    def spawn(self, args: list[str]) -> FakeWindowProc:
        proc = FakeWindowProc(args, self.hold and "-ss" in args)
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

    def __init__(self, media: ui.Media, key: str, lines: list[str], returncode: int = 0) -> None:
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
    media = ui.Media()
    media.progress["k"] = 0.0
    proc = LineProc(media, "k", lines)
    media._read_progress(proc, "k", duration)  # type: ignore[arg-type]
    assert proc.seen == expected


def test_read_progress_keeps_errors_in_the_tail_and_progress_lines_out_of_it() -> None:
    media = ui.Media()
    proc = LineProc(
        media,
        "k",
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
    tail = media._read_progress(proc, "k", 10.0)  # type: ignore[arg-type]
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

    def __init__(self, args: list[str]) -> None:
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
    monkeypatch.setattr(ui, "find_encoder", lambda kind: (None, []))
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
    monkeypatch.setattr(ui, "find_encoder", lambda kind: (ui.VideoEncoder("nvenc"), []))
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
    media.progress[key] = 0.123456
    assert (entry()["proxy"], entry()["progress"]) == (True, 0.123)
    media.proxies[key] = "failed"
    media.progress.pop(key)
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


def test_window_endpoint_serves_ranges_and_refuses_what_is_not_ready(
    client: TestClient,
    hevc_project: tuple[str, Path, Path, Path],
    window_ffmpeg: FakeWindowFfmpeg,
    whole_builds: list[tuple],
) -> None:
    pid, _, _, _ = hevc_project
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
    media.proxies[key] = "failed"
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
    for name in keep + gone:
        (folder / name).write_bytes(b"x")
    p = folder.parents[1]
    media.prune_proxies(p, paths)
    assert sorted(f.name for f in folder.iterdir()) == sorted(keep)
    assert set(media.windows[(p, a)]) == {(0, 500), (1000, 1500)}
    assert (p, b) not in media.windows
    (folder / f"{a}.part.mp4").write_bytes(b"x")  # once per run and project
    media.prune_proxies(p, paths)
    assert (folder / f"{a}.part.mp4").exists()


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


def progress_job(lines: list[str], code: int = 0) -> ui.Job:
    """A finished job whose child printed ``lines`` and exited ``code``."""
    script = f"import sys\nfor ln in {lines!r}:\n    print(ln)\nsys.exit({code})"
    job = ui.Jobs().start("p", "analyze", [sys.executable, "-c", script])
    assert job.done.wait(20)
    return job


@pytest.mark.slow
def test_progress_lines_become_steps_and_stay_out_of_the_log() -> None:
    job = progress_job(SAMPLE_PROGRESS)
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


@pytest.mark.slow
@pytest.mark.parametrize("code", [0, 1])
def test_a_step_still_running_when_the_job_fails_reads_interrupted(code: int) -> None:
    job = progress_job([*SAMPLE_PROGRESS[:1], "12:00:03 ERROR boom"], code)
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
