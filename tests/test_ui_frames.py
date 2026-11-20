"""Frames and stills of ``cutpoint ui``, and the trim rail's audio envelope and events."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("httpx")

from fastapi import HTTPException
from fastapi.testclient import TestClient

from conftest import media_info, needs_ffmpeg, patch_ui, write
from cutpoint import cli, encode, errors, ffmpeg_utils, ui
from cutpoint.ui import media as media_module
from ui_helpers import PROC_WAIT_S, ffmpeg, make_project, wait_until

# the encoder probe and the keyring stay stubbed in every test (ui_helpers fixtures)
pytestmark = pytest.mark.usefixtures("no_nvenc", "no_keyring")


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
        ("ok", "ok", "at=99", True, 200, "-90-160.jpg"),  # clamped to duration - 0.1 s, w 160
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
    patch_ui(
        monkeypatch, "grab_images", lambda *a, **k: (calls.append(a[1]), real_grab(*a, **k))[1]
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
        raise errors.FFmpegError("no image")

    patch_ui(monkeypatch, "grab_images", broken)
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
    arrived = threading.Event()  # a later request waits for the build, or builds alongside

    class Contended:
        """The still's build lock; tells when a request has to wait for it."""

        def __init__(self) -> None:
            self.lock = threading.Lock()

        def __enter__(self) -> None:
            if not self.lock.acquire(blocking=False):
                arrived.set()
                self.lock.acquire()

        def __exit__(self, *_exc: object) -> None:
            self.lock.release()

    class Locks(dict):
        def get(self, name, default=None):
            return super().get(name, (spy, 0))

    spy = Contended()
    media.frame_locks = Locks()

    def grab(*_a: object, **_k: object) -> list[bytes]:
        calls.append(1)
        if len(calls) > 1:
            arrived.set()
        active[0] += 1
        active[1] = max(active)
        try:
            if len(calls) == 1:
                first_in.set()
                release_first.wait(5)
                raise errors.FFmpegError("no image")
            if len(calls) == 2:
                second_in.set()
                release_second.wait(5)
            return [b"\xff\xd8\xffjpeg"]
        finally:
            active[0] -= 1

    patch_ui(monkeypatch, "grab_images", grab)
    results: list[object] = []

    def request() -> None:
        try:
            results.append(media.frame(ui.project_dir(pid), str(source), 0.5, 160))
        except errors.FFmpegError as e:
            results.append(e)

    a = threading.Thread(target=request)
    a.start()
    assert first_in.wait(5)
    b = threading.Thread(target=request)  # waits for the first builder's lock
    b.start()
    assert arrived.wait(5)
    release_first.set()  # the first build fails; the waiting request builds next
    assert second_in.wait(5)
    arrived.clear()
    c = threading.Thread(target=request)  # arrives mid-build: must wait, not build alongside
    c.start()
    assert arrived.wait(5)
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
    patch_ui(monkeypatch, "FRAME_CACHE_MAX", 2)
    patch_ui(monkeypatch, "grab_images", lambda *_a, **_k: [b"\xff\xd8\xffjpeg"])
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

    patch_ui(monkeypatch, "grab_images", grab)
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
    patch_ui(monkeypatch, "probe", lambda _path: media_info(source, duration=0.0, **SD))
    starts: list[object] = []
    patch_ui(
        monkeypatch, "grab_images", lambda *a, **_k: (starts.append(a[1]), [b"\xff\xd8\xffjpeg"])[1]
    )
    media = ui.Media()
    still = media.frame(project, str(source), 1e308, 160)
    cs = round(ui.FRAME_MAX_AT * 100)
    assert still is not None and still.name == f"{media.key(source)}-{cs}-160.jpg"
    assert starts == [cs / 100]


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
    patch_ui(monkeypatch, "probe", lambda _path: info)
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    calls: list[dict[str, object]] = []

    def grab(*_a: object, **k: object) -> list[bytes]:
        calls.append(k)
        return [b"\xff\xd8\xffjpeg"]

    patch_ui(monkeypatch, "grab_images", grab)
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


@pytest.mark.network
def test_ui_on_a_taken_port_fails_cleanly_without_printing_a_url(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn  # noqa: PLC0415

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


# ---- the trim rail's audio envelope and events ----


@pytest.fixture
def audio_sources(client: TestClient, tmp_path: Path) -> tuple[str, dict[str, str], Path]:
    """A project with a 3 s clip (a 440 Hz tone for 2 s, then silence) and a 1 s clip without
    audio: (project id, source id by name, the envelope cache folder)."""
    loud, mute = tmp_path / "tone.mp4", tmp_path / "mute.mp4"
    ffmpeg(
        *("-f", "lavfi", "-i", "testsrc=d=3:s=160x120:r=25"),
        *("-f", "lavfi", "-i", "sine=f=440:d=2", "-af", "apad=whole_dur=3", "-t", "3"),
        *("-c:v", "mpeg4", "-c:a", "aac", str(loud)),
    )  # fmt: skip
    ffmpeg("-f", "lavfi", "-i", "testsrc=d=1:s=160x120:r=25", "-c:v", "mpeg4", str(mute))
    pid, _ = make_project(
        client, tmp_path, {"tone.mp4": loud.read_bytes(), "mute.mp4": mute.read_bytes()}
    )
    ids = {s["name"]: s["id"] for s in client.get(f"/api/projects/{pid}").json()["sources"]}
    return pid, ids, ui.project_dir(pid) / ui.UI_DIR / "envelope"


def get_envelope(client: TestClient, url: str, **kw: object):
    """GET ``url`` until the decode it queued is done (202 while it runs)."""
    deadline = time.monotonic() + 60
    while (res := client.get(url, **kw)).status_code == 202 and time.monotonic() < deadline:
        assert res.json() == {"pending": True}
        time.sleep(0.05)
    return res


@needs_ffmpeg
def test_envelope_is_the_peak_level_per_bucket_and_cached(
    client: TestClient,
    audio_sources: tuple[str, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid, ids, folder = audio_sources
    calls: list[list[str]] = []
    real_spawn = ui.spawn
    patch_ui(
        monkeypatch, "spawn", lambda args, **kw: (calls.append(args), real_spawn(args, **kw))[1]
    )
    url = f"/api/projects/{pid}/sources/{ids['tone.mp4']}/envelope"

    assert client.get(url).status_code == 202  # queued: the page asks again
    first = get_envelope(client, url)
    assert first.status_code == 200, first.text
    data = first.json()
    peaks = data["peaks"]
    assert data["step"] == pytest.approx(0.05)
    assert 58 <= len(peaks) <= 62  # 3 s at 0.05 s
    assert all(0.0 <= v <= 1.0 for v in peaks)
    assert min(peaks[5:35]) > 0.1  # the tone (AAC's priming makes the very first values soft)
    assert max(peaks[-8:]) < 0.02  # the silence after it
    assert [f.suffix for f in folder.iterdir()] == [".json"]  # nothing else left behind

    assert client.get(url).json() == data
    assert len(calls) == 1  # the second request came from the cache


@needs_ffmpeg
def test_envelope_of_a_source_without_audio_is_empty(
    client: TestClient, audio_sources: tuple[str, dict[str, str], Path]
) -> None:
    pid, ids, folder = audio_sources
    res = client.get(f"/api/projects/{pid}/sources/{ids['mute.mp4']}/envelope")
    assert res.status_code == 200
    assert res.json()["peaks"] == []
    assert not folder.exists()


@needs_ffmpeg
@pytest.mark.parametrize(
    ("pid", "sid", "token", "status"),
    [
        ("ok", "0" * 12, True, 404),  # not one of the project's sources
        ("ok", "../../x", True, 404),
        ("nope", "ok", True, 404),
        ("ok", "ok", False, 403),
    ],
)
def test_envelope_accepts_only_a_source_of_the_project(
    client: TestClient,
    audio_sources: tuple[str, dict[str, str], Path],
    pid: str,
    sid: str,
    token: bool,
    status: int,
) -> None:
    real_pid, ids, folder = audio_sources
    pid = real_pid if pid == "ok" else pid
    sid = ids["tone.mp4"] if sid == "ok" else sid
    res = client.get(
        f"/api/projects/{pid}/sources/{sid}/envelope",
        headers=None if token else {"X-Cutpoint-Token": ""},
    )
    assert res.status_code == status, res.text
    assert not folder.exists()


@needs_ffmpeg
@pytest.mark.slow
def test_envelope_failure_is_a_500_and_leaves_no_cache_file(
    client: TestClient,
    audio_sources: tuple[str, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid, ids, folder = audio_sources
    patch_ui(
        monkeypatch,
        "spawn",
        lambda args, **kw: subprocess.Popen(
            [sys.executable, "-c", "raise SystemExit(1)"], stdout=subprocess.PIPE
        ),
    )
    url = f"/api/projects/{pid}/sources/{ids['tone.mp4']}/envelope"
    res = get_envelope(client, url)
    assert res.status_code == 500
    assert not folder.exists() or not list(folder.iterdir())
    assert get_envelope(client, url).status_code == 500  # asked again: decoded again, fails again


@needs_ffmpeg
@pytest.mark.slow
def test_an_envelope_decode_over_the_time_limit_is_killed_and_reported(
    client: TestClient,
    audio_sources: tuple[str, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pid, ids, folder = audio_sources
    started: list[subprocess.Popen] = []
    real_spawn = ui.spawn

    def silent(args: list[str], **kw: object) -> subprocess.Popen:
        # through the real spawn: its own process group, as kill_tree expects
        proc = real_spawn([sys.executable, "-c", "import time; time.sleep(600)"], binary=True)
        started.append(proc)
        return proc

    patch_ui(monkeypatch, "ENVELOPE_TIMEOUT_S", 0.5)
    patch_ui(monkeypatch, "spawn", silent)
    media = client.app.state.media
    url = f"/api/projects/{pid}/sources/{ids['tone.mp4']}/envelope"
    try:
        res = get_envelope(client, url)  # 202 while it runs, then the stored error
        assert res.status_code == 500, res.text
        # the page gets the generic 500; the reason is in the server log
        assert any("took longer than" in r.getMessage() for r in caplog.records)
        assert len(started) == 1
        wait_until(
            lambda: not media.procs and started[0].poll() is not None,
            PROC_WAIT_S,
            "the timed-out decode's process to end",
        )  # no process left behind
        assert not folder.exists() or not list(folder.iterdir())
    finally:
        for proc in started:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


@needs_ffmpeg
@pytest.mark.slow
def test_shutdown_stops_an_envelope_decode_in_flight(
    client: TestClient,
    audio_sources: tuple[str, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid, ids, _folder = audio_sources
    started: list[subprocess.Popen] = []
    real_spawn = ui.spawn

    def silent(args: list[str], **kw: object) -> subprocess.Popen:
        # through the real spawn: its own process group, as kill_tree expects
        proc = real_spawn([sys.executable, "-c", "import time; time.sleep(600)"], binary=True)
        started.append(proc)
        return proc

    patch_ui(monkeypatch, "spawn", silent)
    media = client.app.state.media
    url = f"/api/projects/{pid}/sources/{ids['tone.mp4']}/envelope"
    try:
        assert client.get(url).status_code == 202
        wait_until(lambda: started and media.procs, PROC_WAIT_S, "the decode to start")
        media.shutdown()
        wait_until(
            lambda: not media.procs and started[0].poll() is not None,
            PROC_WAIT_S,
            "the decode's process to end with the server",
        )
    finally:
        for proc in started:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


@needs_ffmpeg
def test_an_envelope_decode_failing_in_any_way_is_stored_not_requeued(
    client: TestClient,
    audio_sources: tuple[str, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pid, ids, _folder = audio_sources
    spawned: list[list[str]] = []

    def refuse(project: Path, name: str) -> Path:
        raise HTTPException(409, "the folder is not ours")

    patch_ui(monkeypatch, "_ui_folder", refuse)
    patch_ui(monkeypatch, "spawn", lambda args, **kw: spawned.append(args))
    media = client.app.state.media
    url = f"/api/projects/{pid}/sources/{ids['tone.mp4']}/envelope"
    res = get_envelope(client, url)  # 202 while it runs, then the stored error
    assert res.status_code == 500, res.text
    assert any("envelope decode" in r.getMessage() for r in caplog.records)
    assert not spawned  # refused before any ffmpeg
    deadline = time.monotonic() + 30
    while media.envelopes_running and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not media.envelopes_running
    assert not media.envelope_errors  # reported once, not kept


@needs_ffmpeg
def test_an_envelope_nobody_asks_for_any_more_is_dropped_from_the_queue(
    client: TestClient,
    audio_sources: tuple[str, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pid, _ids, folder = audio_sources
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    shutil.copyfile(tmp_path / "tone.mp4", first)
    shutil.copyfile(tmp_path / "tone.mp4", second)
    started, release = threading.Event(), threading.Event()
    spawned: list[list[str]] = []
    real_spawn, real_folder = ui.spawn, ui._ui_folder

    def stuck(project: Path, name: str) -> Path:
        started.set()
        release.wait(30)
        return real_folder(project, name)

    class Clock:
        """The media module's ``time``: ``monotonic()`` jumps when the test says so."""

        skip = 0.0

        def monotonic(self) -> float:
            return time.monotonic() + self.skip

        def __getattr__(self, name: str) -> object:
            return getattr(time, name)

    clock = Clock()
    monkeypatch.setattr(media_module, "time", clock)
    patch_ui(monkeypatch, "_ui_folder", stuck)
    patch_ui(
        monkeypatch, "spawn", lambda args, **kw: (spawned.append(args), real_spawn(args, **kw))[1]
    )
    media = client.app.state.media
    project = ui.project_dir(pid)
    try:
        assert media.envelope(project, str(first)) == {"pending": True}
        assert started.wait(30)  # the first holds the worker
        assert media.envelope(project, str(second)) == {"pending": True}  # queued behind it
        clock.skip = ui.ENVELOPE_STALE_S + 1  # the page moved on: nobody asks for the second
    finally:
        release.set()
    wait_until(lambda: not media.envelopes_running, 60, "the envelope worker to finish")
    assert len(spawned) == 1  # only the first was decoded
    assert len(list(folder.glob("*.json"))) == 1
    assert not media.envelope_asked


@needs_ffmpeg
def test_a_cached_envelope_answers_while_another_source_decodes(
    client: TestClient,
    audio_sources: tuple[str, dict[str, str], Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pid, ids, folder = audio_sources
    url = f"/api/projects/{pid}/sources/{ids['tone.mp4']}/envelope"
    assert get_envelope(client, url).status_code == 200  # cached
    other = tmp_path / "other.mp4"
    shutil.copyfile(tmp_path / "tone.mp4", other)
    started, release = threading.Event(), threading.Event()
    real_folder = ui._ui_folder

    def stuck(project: Path, name: str) -> Path:
        # before the decode takes the lock (it starts ffmpeg under it): other calls still run
        started.set()
        release.wait(30)
        return real_folder(project, name)

    patch_ui(monkeypatch, "_ui_folder", stuck)
    media = client.app.state.media
    try:
        assert media.envelope(ui.project_dir(pid), str(other)) == {"pending": True}
        assert started.wait(30)  # the decode holds the slot
        assert media.envelope(ui.project_dir(pid), str(other)) == {"pending": True}  # not twice
        t0 = time.monotonic()
        assert client.get(url).status_code == 200  # the cached source answers at once
        assert time.monotonic() - t0 < 5
    finally:
        release.set()
    for _ in range(600):
        if len(list(folder.glob("*.json"))) == 2:
            break
        time.sleep(0.05)
    assert len(list(folder.glob("*.json"))) == 2


def test_envelope_ffmpeg_mixes_every_audio_track() -> None:
    from cutpoint import envelope as audio_envelope  # noqa: PLC0415

    one = audio_envelope.ffmpeg_args("a.mp4")
    assert "-filter_complex" not in one
    two = audio_envelope.ffmpeg_args("a.mp4", 2)
    assert two[two.index("-filter_complex") + 1].startswith("[0:a:0][0:a:1]amix=inputs=2")
    assert two[two.index("-map") + 1] == "[a]"


def test_prune_envelopes_keeps_only_the_current_version_of_current_sources(
    client: TestClient, tmp_path: Path
) -> None:
    pid, footage = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    media = ui.Media()
    key = media.key(footage / "a.mp4")
    assert key is not None
    folder = p / ui.UI_DIR / "envelope"
    folder.mkdir(parents=True)
    names = [
        f"{key}-{ui.ENVELOPE_VERSION}.json",
        f"{key}-0.json",
        f"{'f' * 16}-{ui.ENVELOPE_VERSION}.json",
    ]
    for n in names:
        (folder / n).write_text("{}", encoding="utf-8")
    media.prune_envelopes(p, [str(footage / "a.mp4")])
    assert [f.name for f in folder.iterdir()] == names[:1]


def test_prune_envelopes_drops_the_envelope_of_a_source_that_changed(
    client: TestClient, tmp_path: Path
) -> None:
    pid, footage = make_project(client, tmp_path, {"a.mp4": b"x"})
    p = ui.project_dir(pid)
    media = ui.Media()
    source = footage / "a.mp4"
    old = media.key(source)
    folder = p / ui.UI_DIR / "envelope"
    folder.mkdir(parents=True)
    stale = folder / f"{old}-{ui.ENVELOPE_VERSION}.json"
    stale.write_text("{}", encoding="utf-8")
    st = source.stat()
    os.utime(source, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))  # same size, new mtime
    new = media.key(source)
    assert new != old
    fresh = folder / f"{new}-{ui.ENVELOPE_VERSION}.json"
    fresh.write_text("{}", encoding="utf-8")
    media.envelopes_pruned.discard(p)
    media.prune_envelopes(p, [str(source)])
    assert not stale.exists()
    assert fresh.exists()


def test_project_events_sig_follows_events_json(client: TestClient, tmp_path: Path) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    assert client.get(f"/api/projects/{pid}").json()["events_sig"] is None
    path = ui.project_dir(pid) / "events.json"
    path.write_text('{"events": []}', encoding="utf-8")
    first = client.get(f"/api/projects/{pid}").json()["events_sig"]
    assert first
    path.write_text('{"events": [], "x": 1}', encoding="utf-8")
    assert client.get(f"/api/projects/{pid}").json()["events_sig"] != first
    second = client.get(f"/api/projects/{pid}").json()["events_sig"]
    (ui.project_dir(pid) / "crashes.json").write_text('{"crashes": []}', encoding="utf-8")
    third = client.get(f"/api/projects/{pid}").json()["events_sig"]
    assert third and third != second  # crashes.json counts too
    path.unlink()
    assert client.get(f"/api/projects/{pid}").json()["events_sig"]  # crashes.json alone


def test_events_are_the_located_events_without_model_descriptions(
    client: TestClient, tmp_path: Path
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    url = f"/api/projects/{pid}/events"
    assert client.get(url).json() == {"events": []}  # no events.json
    assert client.get("/api/projects/nope/events").status_code == 404

    def one(**over: object) -> dict:
        return {
            "source": "/v/a.mp4", "start": 1.0, "end": 2.0, "kind": "audio", "type": "audio",
            "label": "Laughter", "description": "x" * 500, "confidence": 0.91234, "details": {"a": 1},
        } | over  # fmt: skip

    doc = {
        "events": [
            one(start=9.0, end=9.5, kind="ocr", type="kill", label="A>B"),
            one(),
            one(
                kind="telemetry", type="crash", label="impact 9.1 g then a stop", start=5.0, end=5.0
            ),
            one(kind="telemetry", type="coverage", label="riding"),  # a span, not an event
            one(kind="vlm", type="vlm", label="a long model sentence"),
            one(start="soon"),
            one(source=3),
            "junk",
        ]
    }
    (ui.project_dir(pid) / "events.json").write_text(json.dumps(doc), encoding="utf-8")
    events = client.get(url).json()["events"]
    assert [(e["kind"], e["start"], e["label"]) for e in events] == [
        ("audio", 1.0, "Laughter"),
        ("telemetry", 5.0, "crash"),
        ("ocr", 9.0, "A>B"),
    ]
    assert events[0] == {
        "source": "/v/a.mp4", "start": 1.0, "end": 2.0, "kind": "audio", "type": "audio",
        "label": "Laughter", "confidence": 0.912,
    }  # fmt: skip

    (ui.project_dir(pid) / "events.json").write_text("{not json", encoding="utf-8")
    assert client.get(url).json() == {"events": []}


def test_events_serve_talk_stretches_with_their_title(client: TestClient, tmp_path: Path) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    talk = {
        "source": "/v/a.mp4", "start": 10.0, "end": 24.5, "kind": "text", "type": "dialogue",
        "label": "funny", "title": "Mia drops the cup", "confidence": 0.7,
        "description": "x" * 300, "details": {"a": 1},
    }  # fmt: skip
    plain = talk | {"start": 40.0, "end": 41.0, "title": ""}
    doc = {"events": [plain, talk]}
    (ui.project_dir(pid) / "events.json").write_text(json.dumps(doc), encoding="utf-8")
    events = client.get(f"/api/projects/{pid}/events").json()["events"]
    assert events == [
        {
            "source": "/v/a.mp4", "start": 10.0, "end": 24.5, "kind": "text", "type": "dialogue",
            "label": "Mia drops the cup", "confidence": 0.7,
        },
        {
            "source": "/v/a.mp4", "start": 40.0, "end": 41.0, "kind": "text", "type": "dialogue",
            "label": "funny", "confidence": 0.7,
        },
    ]  # fmt: skip


def test_events_add_crashes_json_entries_not_already_telemetry_crashes(
    client: TestClient, tmp_path: Path
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    url = f"/api/projects/{pid}/events"
    src = "/v/a.mp4"
    crashes = {
        "crashes": [
            {
                "id": "c001",
                "source": src,
                "t0": 5.0,
                "t1": 7.0,
                "confidence": 0.81234,
            },  # = telemetry
            {"id": "c002", "source": src, "t0": 50.0, "t1": 52.0, "confidence": 0.6},
            {"id": "c006", "source": src, "t0": 7.5, "t1": 8.0, "confidence": 0.55},  # 1.5 s on
            {"id": "c007", "source": src, "t0": 8.5, "t1": 9.0, "confidence": 0.5},  # 2.5 s on
            {"id": "c003", "source": "/v/b.mp4", "t0": 5.0, "t1": 7.0, "confidence": 0.5},
            {"id": "c004", "source": src, "t0": "soon", "t1": 1.0},
            {"id": "c005", "source": 3, "t0": 1.0, "t1": 2.0},
            "junk",
        ]
    }
    (ui.project_dir(pid) / "crashes.json").write_text(json.dumps(crashes), encoding="utf-8")
    only = client.get(url).json()["events"]  # no events.json: all the valid entries
    assert [(e["source"], e["start"], e["end"]) for e in only] == [
        (src, 5.0, 7.0), (src, 7.5, 8.0), (src, 8.5, 9.0), (src, 50.0, 52.0), ("/v/b.mp4", 5.0, 7.0)
    ]  # fmt: skip
    assert only[0] == {
        "source": src, "start": 5.0, "end": 7.0, "kind": "telemetry", "type": "crash",
        "label": "crash", "confidence": 0.812,
    }  # fmt: skip

    tel = {
        "source": src, "start": 6.0, "end": 6.0, "kind": "telemetry", "type": "crash",
        "label": "impact", "confidence": 0.9,
    }  # fmt: skip
    (ui.project_dir(pid) / "events.json").write_text(
        json.dumps({"events": [tel]}), encoding="utf-8"
    )
    events = client.get(url).json()["events"]
    assert [(e["source"], e["start"], e["confidence"]) for e in events] == [
        (src, 6.0, 0.9),
        (src, 8.5, 0.5),
        (src, 50.0, 0.6),
        ("/v/b.mp4", 5.0, 0.5),
    ]  # fmt: skip  (c001 and the 7.5 s one are within 2 s of the telemetry crash, 8.5 s is not)

    (ui.project_dir(pid) / "crashes.json").write_text("{not json", encoding="utf-8")
    assert [e["start"] for e in client.get(url).json()["events"]] == [6.0]


def test_events_serve_only_crashes_above_the_relative_confidence_floor(
    client: TestClient, tmp_path: Path
) -> None:
    pid, _ = make_project(client, tmp_path, {"a.mp4": b"x"})
    url = f"/api/projects/{pid}/events"
    path = ui.project_dir(pid) / "crashes.json"

    def served(*confs: float | None) -> list[float]:
        rows = [
            {"source": "/v/a.mp4", "t0": 10.0 * i, "t1": 10.0 * i + 1, "confidence": c}
            for i, c in enumerate(confs)
        ]
        path.write_text(json.dumps({"crashes": rows}), encoding="utf-8")
        return [e["confidence"] for e in client.get(url).json()["events"]]

    assert served(0.9, 0.55, 0.53, 0.3) == [0.9, 0.55]  # 0.6 x 0.9 = 0.54 and up
    assert served(0.5, 0.3, 0.29) == [0.5, 0.3]  # relative to the best, not a fixed number
    assert served(0.3, 0.19, 0.1) == [0.3]  # 0.6 x 0.3 = 0.18 is under the 0.2 minimum
    assert served(0.15, 0.1) == []  # even the best is under the minimum
    assert served(0.9, None) == [0.9]  # no confidence, no marker
