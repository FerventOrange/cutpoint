"""Proxies of ``cutpoint ui``: spawn priority, window and whole proxies, the source list cache."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from conftest import media_info, needs_ffmpeg, patch_ui, remap, set_config, write
from cutpoint import cache, encode, procs, ui
from ui_helpers import PORT, FakeProc, ffmpeg, make_project, media_link, wait_until

# the encoder probe and the keyring stay stubbed in every test (ui_helpers fixtures)
pytestmark = pytest.mark.usefixtures("no_nvenc", "no_keyring")


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
    monkeypatch.setattr(procs, "join_tree_job", lambda _proc: None)
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
    monkeypatch.setattr(procs, "join_tree_job", lambda _proc: None)
    monkeypatch.setattr(procs.shutil, "which", lambda name: f"/usr/bin/{name}")
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
    patch_ui(monkeypatch, "spawn", ProxyEncoder)
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
    patch_ui(monkeypatch, "video_encoder", lambda: enc)
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
    patch_ui(monkeypatch, "video_encoder", lambda: ui.VideoEncoder("nvenc"))
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
    patch_ui(monkeypatch, "video_encoder", lambda: ui.VideoEncoder("nvenc"))

    class Failing(ProxyEncoder):
        def __init__(self, args: list[str], **_kw: object) -> None:
            self.calls.append(args)
            self.returncode = 1

    patch_ui(monkeypatch, "spawn", Failing)
    _, folder, p, _ = hevc_project
    assert build_proxy(client, p, folder / "a.mp4") == "failed"
    assert len(proxy_encoder.calls) == 2
    assert not client.app.state.media.gpu_failed  # libx264 did not work either
    assert not list(p.rglob("*.part.mp4"))


def test_no_proxy_ffmpeg_starts_after_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    spawned: list[list[str]] = []
    patch_ui(monkeypatch, "spawn", lambda args, **_kw: spawned.append(args))
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
    monkeypatch.setattr(procs, "kill_tree", lambda proc, wait=0, *, term=True: killed.append(proc))

    class Encode:
        def __init__(self, args: list[str], **_kw: object) -> None:
            assert media.lock.locked()  # started in the same step as the closing check
            self.returncode = 1

        def communicate(self) -> tuple[str, None]:
            assert media.procs == {self}
            media.shutdown()  # the server stops during the encode
            return "", None

    patch_ui(monkeypatch, "spawn", Encode)
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
    patch_ui(monkeypatch, "probe", lambda _path: probed)
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
    patch_ui(monkeypatch, "probe", lambda _path: info)
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
    patch_ui(monkeypatch, "probe", lambda _path: hlg)
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
    patch_ui(monkeypatch, "probe", lambda _path: hlg)
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
        patch_ui(monkeypatch, "spawn", self.spawn)
        monkeypatch.setattr(procs, "kill_tree", self.kill)

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


def lane_idle(media: ui.Media) -> bool:
    with media.lock:
        return not media.window_lane.queue and media.window_lane.running is None


@pytest.fixture
def window_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> FakeWindowFfmpeg:
    return FakeWindowFfmpeg(monkeypatch)


@pytest.fixture
def window_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window_ffmpeg: FakeWindowFfmpeg
) -> Iterator[tuple[ui.Media, Path, Path]]:
    """A Media, a project folder and a 1000 s HEVC source; its windows spawn the fake ffmpeg."""
    patch_ui(monkeypatch, "probe", lambda _path: media_info(vcodec="hevc", duration=1000.0))
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

    patch_ui(monkeypatch, "spawn", Held)
    monkeypatch.setattr(ui.Media, "_gpu_encoder", lambda _self: None)
    patch_ui(monkeypatch, "probe", lambda _path: media_info(vcodec="hevc", duration=100.0))
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
    assert media.window_lane.thread is not None
    media.window_lane.thread.join(10)
    assert not media.window_lane.thread.is_alive()
    focus(media, p, src, 400, 410)
    win = ui.ProxyWindow(p, src, "k" * 16, 0, 500, 1, "")
    assert media._encode(["ffmpeg", "-ss", "0"], win)[0] != 0
    assert len(window_ffmpeg.procs) == 1
    assert not media.window_lane.queue
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
        assert media.window_lane.running is not None
        media.window_lane.running.cancelled = True  # e.g. its source's whole proxy landed
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
    assert client.app.state.media.window_lane.thread is None


@pytest.mark.parametrize(
    "raw",
    [
        b'{"lo": "1", "hi": 5}',
        b'{"lo": 1, "hi": 1' + b"0" * 400 + b"}",  # too large for a float: no 500
        b'{"lo": 6, "hi": 5}',
        b'{"lo": -1, "hi": 5}',
        b'{"lo": 1}',
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

    patch_ui(monkeypatch, "probe", probing)
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
        (
            ["out_time_us=5000000", "out_time_us=3000000", "out_time_us=7000000"],
            10.0,
            [0.5, 0.5, 0.7, 0.7],  # a report that steps back never lowers the bar
        ),
    ],
    ids=["updates", "capped", "not-available", "no-duration", "steps-back"],
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
    patch_ui(monkeypatch, "spawn", ScriptedEncoder)
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
    patch_ui(monkeypatch, "video_encoder", lambda: ui.VideoEncoder("cpu"))
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
    patch_ui(monkeypatch, "video_encoder", lambda: ui.VideoEncoder("nvenc"))
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
    patch_ui(monkeypatch, "probe", lambda _path: media_info(duration=10.0))  # H.264
    sid = client.get(f"/api/projects/{pid}").json()["sources"][0]["id"]
    assert prepare(client, pid, sid).json() == {"status": "ready"}
    src = client.get(f"/api/projects/{pid}").json()["sources"][0]
    assert (src["status"], src["proxy"], src["progress"]) == ("ready", False, None)


def queued_starts(media: ui.Media) -> list[float]:
    with media.lock:
        return [w.lo / 100 for w in media.window_lane.queue]


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
        b'{"queue": "true", "lo": 1, "hi": 5}',
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

    patch_ui(monkeypatch, "spawn", Failing)
    monkeypatch.setattr(ui.Media, "_gpu_encoder", lambda _self: None)
    entry = lambda: client.get(base).json()["sources"][0]  # noqa: E731
    with caplog.at_level(logging.WARNING, logger="cutpoint.ui"):
        assert prepare(client, pid, sid, {"lo": 2, "hi": 4}).status_code == 200
        wait_until(lambda: entry()["failed"])
    assert entry()["failed"] == [{"lo": 0.0, "hi": 10.0, "reason": "Conversion failed!"}]
    assert entry()["pending"] == []
    [warning] = [r.getMessage() for r in caplog.records if "window proxy" in r.getMessage()]
    assert warning.endswith("failed: line 2\nline 3\nline 4\nline 5\nConversion failed!")

    patch_ui(monkeypatch, "spawn", window_ffmpeg.spawn)
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


@pytest.mark.ffmpeg  # probes the sources with the real ffprobe
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
    patch_ui(monkeypatch, "probe", lambda path: probed.append(Path(path).name) or hlg)
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


def test_hdr_proxies_of_an_earlier_tone_map_go_once_the_current_proxy_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: frozenset({"zscale", "tonemap"}))
    hlg = media_info(vcodec="hevc", pix_fmt="yuv420p10le", duration=10.0, **HLG)
    probed: list[str] = []
    patch_ui(monkeypatch, "probe", lambda path: probed.append(Path(path).name) or hlg)
    media, folder, paths, a, b = prune_setup(tmp_path)
    monkeypatch.setattr(media.pool, "submit", lambda *_args: None)  # no build
    stale = folder / f"{a}.mp4"  # named by the content key: no current proxy beside it
    stale.write_bytes(b"x")
    age(stale)
    (folder / f"{b}.mp4").write_bytes(b"x")  # another source's
    p = folder.parents[1]
    media.prune_proxies(p, paths)
    assert probed == []  # nothing to tell it from without probing
    assert media.prepare(p, paths[0]) == "preparing"
    assert probed == ["a.mp4"]
    assert stale.is_file()  # left for now, never served: the proxy key holds the filter
    hdr = media._proxy_key(a)
    assert hdr and hdr != a
    (folder / f"{hdr}.mp4").write_bytes(b"x")  # the build landed
    age(folder / f"{hdr}.mp4")
    later = ui.Media()  # the next run: nothing probed yet
    later.prune_proxies(p, paths)
    assert sorted(f.name for f in folder.iterdir()) == sorted([f"{hdr}.mp4", f"{b}.mp4"])


def test_prune_proxies_probes_no_source_whose_proxy_it_knows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probed: list[str] = []
    hevc = media_info(vcodec="hevc", duration=10.0)
    patch_ui(monkeypatch, "probe", lambda path: probed.append(Path(path).name) or hevc)
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
    patch_ui(monkeypatch, "probe", lambda path: probed.append(Path(path).name))
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
    patch_ui(monkeypatch, "probe", lambda path: probed.append(Path(path).name) or sdr)
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
    patch_ui(monkeypatch, "probe", lambda path: probed.append(Path(path).name) or hlg)
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
    patch_ui(monkeypatch, "probe", lambda path: hlg if Path(path).name == "a.mp4" else hevc)
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


@pytest.mark.ffmpeg  # probes the sources with the real ffprobe
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


@pytest.mark.ffmpeg  # probes the sources with the real ffprobe
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

    patch_ui(monkeypatch, "find_videos", counting)
    return calls


def source_names(client: TestClient, pid: str) -> list[str]:
    res = client.get(f"/api/projects/{pid}")
    assert res.status_code == 200, res.text
    return sorted(s["name"] for s in res.json()["sources"])


@needs_ffmpeg
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


def test_project_reports_the_video_length_and_time_of_the_last_analysis(
    client: TestClient, tmp_path: Path
) -> None:
    """#459: from highlights.json's sources and timings; a recovered part counts what it read,
    and highlights.json without timings has none."""
    pid, folder = make_project(client, tmp_path, {"a.mp4": b"x", "b.mp4": b"y"})
    hl = {
        "sources": [
            {"path": str(folder / "a.mp4"), "duration": 600.0},
            {"path": str(folder / "b.mp4"), "duration": 900.0, "offset": 0, "analyzed_s": 300.5},
        ],
        "timings": {"transcribe_s": 40.0, "total_s": 75.25},
        "candidates": [],
    }
    cache.save_json(ui.project_dir(pid) / "highlights.json", hl)
    analysis = client.get(f"/api/projects/{pid}").json()["analysis"]
    assert analysis == {"video_s": 900.5, "took_s": 75.25}
    cache.save_json(ui.project_dir(pid) / "highlights.json", {**hl, "timings": {}})
    assert client.get(f"/api/projects/{pid}").json()["analysis"] is None
    # 360: total_s is the flat analysis alone, so the time is pano_timings.json's total_s
    p = ui.project_dir(pid)
    cache.save_json(p / "ui" / "meta.json", {**ui.read_meta(p), "kind": "360"})
    cache.save_json(p / "highlights.json", hl)
    assert client.get(f"/api/projects/{pid}").json()["analysis"] is None
    pano = {"scan_s": 12.5, "analyze_s": 90.0, "analyze_total_s": 300.0, "total_s": 320.5}
    cache.save_json(p / "pano_timings.json", pano)
    analysis = client.get(f"/api/projects/{pid}").json()["analysis"]
    assert analysis == {"video_s": 900.5, "took_s": 320.5}
    # --no-events: total_s without the removed analyze_total_s
    cache.save_json(p / "pano_timings.json", {"analyze_s": 90.0, "total_s": 95.0})
    assert client.get(f"/api/projects/{pid}").json()["analysis"]["took_s"] == 95.0
    # --views-only times the views alone and leaves highlights.json as it was: no time
    cache.save_json(p / "pano_timings.json", {"views": {"033": 4.0}, "total_s": 40.0})
    assert client.get(f"/api/projects/{pid}").json()["analysis"] is None
    # a pano_timings.json from before total_s: no time, no crash
    del pano["total_s"]
    cache.save_json(p / "pano_timings.json", pano)
    assert client.get(f"/api/projects/{pid}").json()["analysis"] is None


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

    patch_ui(monkeypatch, "find_videos", flaky)
    client.get(f"/api/projects/{pid}")  # the failed listing; the response does not matter
    assert source_names(client, pid) == ["a.mp4"]


def test_the_cache_keeps_at_most_max_projects_and_expires_after_its_ttl(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, folder_listings: list[int]
) -> None:
    patch_ui(monkeypatch, "SOURCES_CACHE_MAX", 1)
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
    patch_ui(monkeypatch, "SOURCES_CACHE_S", 0)
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

    patch_ui(monkeypatch, "probe", counting)
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
    patch_ui(monkeypatch, "probe", lambda _path: media_info(duration=10.0))
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

    patch_ui(monkeypatch, "source_entries", counting)
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
    'PROGRESS {"step":"transcribe","index":1,"total":7,"label":"Transcription","state":"running","pct":0,"detail":"sample.mp4"}',
    "12:00:01 INFO transcribing sample.mp4",
    'PROGRESS {"step":"transcribe","index":1,"total":7,"label":"Transcription","state":"done","pct":100,"detail":null}',
    'PROGRESS {"step":"reactions","index":2,"total":7,"label":"Reactions","state":"skipped","pct":null,"detail":"events extra not installed"}',
    'PROGRESS {"step":"crashes","index":4,"total":7,"label":"Crashes","state":"skipped","pct":null,"detail":"not action-cam footage"}',
    'PROGRESS {"step":"vision","index":7,"total":7,"label":"Vision rating","state":"skipped","pct":null,"detail":"vision is off (Settings, config.json or --vision)"}',
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
    assert view["progress"]["total"] == 7
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
    huge = ui.parse_progress(
        'PROGRESS {"step":"x","index":2,"state":"running","pct":1' + "0" * 400 + "}"
    )  # too large for a float: no OverflowError in the job watcher
    assert huge is not None and huge["pct"] is None


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
    patch_ui(monkeypatch, "spawn", lambda args: proc)
    monkeypatch.setattr(procs, "kill_tree", lambda _p: proc.release.set())
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
