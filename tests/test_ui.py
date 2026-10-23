"""Backend of ``cutpoint ui``: trust boundaries, review validation, media Range, job semantics."""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi import HTTPException
from fastapi.testclient import TestClient

from cutpoint import cache, cli, config, ui

PORT = 8765
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg"
)


@pytest.fixture
def client() -> TestClient:
    app = ui.create_app(PORT)
    c = TestClient(app, base_url=f"http://127.0.0.1:{PORT}")
    c.headers["X-Cutpoint-Token"] = app.state.token
    return c


def make_project(client: TestClient, tmp_path: Path, files: dict[str, bytes]) -> tuple[str, Path]:
    folder = tmp_path / "footage"
    folder.mkdir()
    for name, data in files.items():
        (folder / name).write_bytes(data)
    res = client.post("/api/projects", json={"name": "My Trip", "source": str(folder)})
    assert res.status_code == 200, res.text
    return res.json()["id"], folder


def wait_job(client: TestClient, timeout: float = 180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = client.get("/api/jobs").json()["jobs"]
        if jobs and jobs[-1]["status"] != "running":
            return jobs[-1]
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_requests_must_be_loopback_and_carry_the_token(client: TestClient) -> None:
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
    framed = client.get("/api/session")
    assert framed.headers["x-frame-options"] == "DENY"
    assert client.get("/api/projects").json() == {"projects": []}


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


@needs_ffmpeg
def test_proxy_is_built_only_for_the_chosen_source_and_keeps_every_audio_track(
    client: TestClient, tmp_path: Path
) -> None:
    """Opening a project builds nothing; preparing one source mixes all its audio tracks."""
    multi, other = tmp_path / "a.mp4", tmp_path / "b.mkv"
    ffmpeg(
        *("-f", "lavfi", "-i", "testsrc=d=1:s=160x120:r=25"),
        *("-f", "lavfi", "-i", "sine=f=440:d=1", "-f", "lavfi", "-i", "sine=f=880:d=1"),
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
    assert (opened["a.mp4"]["status"], opened["b.mkv"]["status"]) == ("needs-proxy",) * 2
    assert not (ui.project_dir(pid) / ui.UI_DIR / "proxy").exists()  # nothing was queued
    assert client.get(opened["a.mp4"]["url"]).status_code == 409

    client.post(f"/api/projects/{pid}/sources/{opened['a.mp4']['id']}/prepare")
    deadline = time.time() + 120
    while sources()["a.mp4"]["status"] == "preparing" and time.time() < deadline:
        time.sleep(0.3)
    after = sources()
    assert (after["a.mp4"]["status"], after["b.mkv"]["status"]) == ("ready", "needs-proxy")

    url = after["a.mp4"]["url"]
    assert client.get(url, headers={"Range": "bytes=0-99"}).status_code == 206
    played = tmp_path / "played.mp4"
    played.write_bytes(client.get(url).content)
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
    cache.save_json(p / "highlights.json", {"style": "gaming", "settings": settings, "sources": []})

    assert client.post("/api/projects/cli-made/jobs", json={"action": "analyze"}).status_code == 200
    assert wait_job(client)["status"] == "completed"
    names = {s["name"] for s in client.get("/api/projects/cli-made").json()["sources"]}
    assert names == {"keep.mp4", "nested.mp4"}


def test_one_job_at_a_time_and_cancel_kills_the_process_group(tmp_path: Path) -> None:
    script = (
        "import subprocess, sys, time\n"
        "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print('child', c.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    jobs = ui.Jobs()
    job = jobs.start("p", "analyze", [sys.executable, "-c", script])
    deadline = time.time() + 20
    while not job.logs and time.time() < deadline:
        time.sleep(0.05)
    child = int(job.logs[0].split()[1])
    with pytest.raises(HTTPException) as busy:
        jobs.start("p", "clips", [sys.executable, "-c", "pass"])
    assert busy.value.status_code == 409
    assert jobs.cancel(job.id).status == "cancelled"
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
    assert {o["name"].rsplit(".", 1)[1] for o in second} == {"mp4", "fcpxml", "otio"}
    assert (ui.project_dir(pid) / first[0]["name"]).is_file()  # the earlier final survives
    assert client.get(second[0]["url"]).status_code == 200

    again = run({"action": "clips"})  # fresh folder; the earlier export stays untouched
    assert again["clips"]["dir"] != first_dir and again["outputs"] == []
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
        ("ok", "ok", "at=inf", True, 400, None),
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

    shutil.rmtree(folder)
    res = client.post(f"/api/projects/{pid}/reveal")
    assert res.status_code == 404 and "no longer exists" in res.json()["detail"]
    assert opened == []


def test_show_folder_targets_open_the_latest_clips_and_the_final_folder(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``target`` picks one of the project's folders; the path still never comes from the client."""
    opened: list[Path] = []
    monkeypatch.setattr(ui, "folder_opener", lambda: opened.append)
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    reveal = f"/api/projects/{pid}/reveal"

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

    monkeypatch.setattr(ui, "folder_opener", lambda: None)
    assert client.post(reveal, params={"target": "clips"}).status_code == 501

    def broken(_: Path) -> None:
        raise OSError(5, "Access is denied")

    monkeypatch.setattr(ui, "folder_opener", lambda: broken)
    res = client.post(reveal, params={"target": "outputs"})
    assert res.status_code == 500 and "Access is denied" in res.json()["detail"]


@pytest.mark.parametrize("bad", ["..", ".", ".hidden"])
def test_exports_json_cannot_point_outside_the_project(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    opened: list[Path] = []
    monkeypatch.setattr(ui, "folder_opener", lambda: opened.append)
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    reveal = f"/api/projects/{pid}/reveal"
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


def link_dir(link: Path, target: Path) -> None:
    """Make ``link`` a directory link to ``target``: a symlink, or a junction on Windows."""
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        if sys.platform != "win32":
            pytest.skip("cannot create a directory symlink here")
    done = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=False
    )
    if done.returncode != 0:
        pytest.skip("cannot create a symlink or junction here")


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
        assert client.post(reveal, params={"target": "outputs"}).status_code == 404

    # dir: an earlier legitimate export takes over; final: the latest export stays current
    # and its linked final is hidden until it is replaced.
    write(good, bad)
    if kind == "dir":
        assert client.post(reveal, params={"target": "clips"}).json() == {
            "path": str(p / "clips_ui_1")
        }
        assert opened == [p / "clips_ui_1"]
    else:
        assert client.post(reveal, params={"target": "outputs"}).status_code == 404
        assert opened == []
        write(good)
        assert client.post(reveal, params={"target": "outputs"}).json() == {"path": str(p)}
        # Timelines next to the final are served only when they stay inside the project.
        (outside / "final.fcpxml").write_text("x")
        (p / "final.fcpxml").symlink_to(outside / "final.fcpxml")
        assert client.get(f"/api/projects/{pid}/outputs/final.fcpxml").status_code == 404
        names = [o["name"] for o in client.get(f"/api/projects/{pid}").json()["outputs"]]
        assert names == ["final.mp4"]


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


@pytest.mark.parametrize("ext", ["", ".mp4", ".fcpxml"])
def test_fresh_name_treats_a_broken_symlink_as_taken(tmp_path: Path, ext: str) -> None:
    try:
        (tmp_path / f"final{ext}").symlink_to(tmp_path / "missing-target")
    except (OSError, NotImplementedError):
        pytest.skip("cannot create a symlink here")
    exts = ("", ".mp4", ".fcpxml") if ext else ("",)
    assert ui.fresh_name(tmp_path, "final", exts) == "final-2"


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


def test_session_reports_can_pick(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui, "path_picker", lambda: None)
    assert client.get("/api/session").json()["can_pick"] is False
    monkeypatch.setattr(ui, "path_picker", lambda: lambda kind, start: ([], None))
    assert client.get("/api/session").json()["can_pick"] is True


def test_pick_and_settings_need_the_token(client: TestClient, dialog: FakeDialog) -> None:
    bare = {"X-Cutpoint-Token": ""}
    assert client.post("/api/pick", params={"kind": "file"}, headers=bare).status_code == 403
    assert client.get("/api/settings", headers=bare).status_code == 403
    body = {"projects_dir": "x"}
    assert client.put("/api/settings", json=body, headers=bare).status_code == 403
    assert client.get("/api/settings", headers={"X-Cutpoint-Token": "nope"}).status_code == 403
    assert dialog.calls == []


def test_pick_returns_the_chosen_path_or_null_on_cancel(
    client: TestClient, dialog: FakeDialog, tmp_path: Path
) -> None:
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
    assert argv == [
        "/bin/zenity",
        "--file-selection",
        "--directory",
        "--title=Choose a folder",
        f"--filename={tmp_path}{os.sep}",
    ]
    assert cancel == 1
    argv, _ = ui.path_picker()("file", None)
    assert argv[:3] == ["/bin/zenity", "--file-selection", "--title=Choose a video"]
    assert "*.mp4" in argv[3] and not any(a.startswith("--filename") for a in argv)

    present.discard("zenity")  # kdialog is the fallback
    argv, cancel = ui.path_picker()("folder", tmp_path)
    assert argv == [
        "/bin/kdialog",
        "--title",
        "Choose a folder",
        "--getexistingdirectory",
        str(tmp_path),
    ]
    argv, cancel = ui.path_picker()("file", tmp_path)
    assert argv[:5] == [
        "/bin/kdialog",
        "--title",
        "Choose a video",
        "--getopenfilename",
        str(tmp_path),
    ]
    assert "*.mp4" in argv[5] and cancel == 1

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


@pytest.mark.parametrize("platform", ["win32", "darwin"])
@pytest.mark.parametrize("var", ["SSH_CONNECTION", "SSH_TTY"])
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
    assert res.json() == {
        "projects_dir": str(config.projects_dir()),
        "projects_dir_locked": False,
        "config_file": str(home / "config.json"),
    }
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


def test_put_settings_rejects_an_unwritable_folder(
    client: TestClient, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(*a, **kw):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(ui.tempfile, "TemporaryFile", denied)
    res = client.put("/api/settings", json={"projects_dir": str(tmp_path)})
    assert res.status_code == 400 and "Permission denied" in res.json()["detail"]
    assert not (home / "config.json").exists()


def test_put_settings_conflicts(
    client: TestClient, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = {"projects_dir": str(tmp_path)}
    cfg = home / "config.json"

    assert client.get("/api/settings").status_code == 200  # config loaded (and cached) while valid
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{not json", encoding="utf-8")
    res = client.put("/api/settings", json=body)
    assert res.status_code == 409 and cfg.read_text("utf-8") == "{not json"  # never overwritten
    cfg.unlink()
    config.get_config.cache_clear()

    client.app.state.jobs.jobs["j1"] = SimpleNamespace(status="running")
    assert client.put("/api/settings", json=body).status_code == 409
    client.app.state.jobs.jobs["j1"].status = "done"
    assert client.put("/api/settings", json=body).status_code == 200

    monkeypatch.setenv("CUTPOINT_PROJECTS_DIR", str(tmp_path / "env"))
    res = client.put("/api/settings", json={"projects_dir": str(tmp_path / "x")})
    assert res.status_code == 409 and "CUTPOINT_PROJECTS_DIR" in res.json()["detail"]
    assert json.loads(cfg.read_text("utf-8")) == {"projects_dir": str(tmp_path)}


def test_put_settings_keeps_other_config_keys(
    client: TestClient, home: Path, tmp_path: Path
) -> None:
    cfg = home / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('{\n    "player_tags": ["Me"],\n    "projects_dir": "/old"\n}', encoding="utf-8")
    assert client.put("/api/settings", json={"projects_dir": str(tmp_path)}).status_code == 200
    data = json.loads(cfg.read_text("utf-8"))
    assert list(data) == ["player_tags", "projects_dir"] and data["player_tags"] == ["Me"]
    assert data["projects_dir"] == str(tmp_path)


def test_settings_with_an_invalid_config_file_conflict_even_on_a_fresh_cache(
    client: TestClient, home: Path, tmp_path: Path
) -> None:
    cfg = home / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{oops", encoding="utf-8")
    config.get_config.cache_clear()
    assert client.get("/api/settings").status_code == 409
    config.get_config.cache_clear()
    res = client.put("/api/settings", json={"projects_dir": str(tmp_path)})
    assert res.status_code == 409 and cfg.read_text("utf-8") == "{oops"


def test_settings_ignore_an_invalid_unrelated_key(
    client: TestClient, home: Path, tmp_path: Path
) -> None:
    cfg = home / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('{"video_encoder": "bogus"}', encoding="utf-8")
    config.get_config.cache_clear()
    assert client.get("/api/settings").status_code == 200
    config.get_config.cache_clear()
    new = tmp_path / "proj"
    assert client.put("/api/settings", json={"projects_dir": str(new)}).status_code == 200
    data = json.loads(cfg.read_text("utf-8"))
    assert data == {"video_encoder": "bogus", "projects_dir": str(new)}


def test_put_settings_removes_a_folder_it_created_when_saving_fails(
    client: TestClient, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    new = tmp_path / "fresh"
    cfg = home / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{oops", encoding="utf-8")
    config.get_config.cache_clear()
    assert client.put("/api/settings", json={"projects_dir": str(new)}).status_code == 409
    assert not new.exists()

    cfg.unlink()
    config.get_config.cache_clear()

    def broken(*a, **kw):
        raise OSError(28, "No space left")

    monkeypatch.setattr(ui, "set_config_value", broken)
    res = client.put("/api/settings", json={"projects_dir": str(new)})
    assert res.status_code == 500 and not new.exists()


def test_put_settings_never_removes_an_existing_folder_on_failure(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "keep"
    existing.mkdir()
    (existing / "f.txt").write_text("x")

    def broken(*a, **kw):
        raise OSError(28, "No space left")

    monkeypatch.setattr(ui, "set_config_value", broken)
    res = client.put("/api/settings", json={"projects_dir": str(existing)})
    assert res.status_code == 500 and (existing / "f.txt").exists()

    def denied(*a, **kw):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(ui.tempfile, "TemporaryFile", denied)
    assert client.put("/api/settings", json={"projects_dir": str(existing)}).status_code == 400
    assert existing.is_dir()


def test_check_projects_dir_removes_its_own_folder_when_the_write_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(*a, **kw):
        raise PermissionError(13, "Permission denied")

    new = tmp_path / "probe"
    monkeypatch.setattr(ui.tempfile, "TemporaryFile", denied)
    with pytest.raises(HTTPException) as e:
        ui.check_projects_dir(str(new))
    assert e.value.status_code == 400 and not new.exists()
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
        assert client.put("/api/settings", json={"projects_dir": str(tmp_path)}).status_code == 200

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


def test_settings_show_a_wrongly_typed_projects_folder_so_it_can_be_replaced(
    client: TestClient, home: Path, tmp_path: Path
) -> None:
    cfg = home / "config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text('{"projects_dir": 5}', encoding="utf-8")
    config.get_config.cache_clear()
    res = client.get("/api/settings")
    assert res.status_code == 200 and res.json()["projects_dir"] == "5"
    assert client.put("/api/settings", json={"projects_dir": str(tmp_path)}).status_code == 200
    assert json.loads(cfg.read_text("utf-8")) == {"projects_dir": str(tmp_path)}


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
