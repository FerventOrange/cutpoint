"""Backend of ``cutpoint ui``: trust boundaries, review validation, media Range, job semantics."""

from __future__ import annotations

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

from cutpoint import cache, cli, ui

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
    assert client.get("/api/browse", headers={"X-Cutpoint-Token": ""}).status_code == 403
    assert client.get("/api/browse", params={"path": "/\0"}).status_code == 400
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
