"""The scene pass split between NVDEC and the CPU (#730 PR 2): the gate
(``CUTPOINT_SCENE_DECODE``, CUDA, length, CPU threads, nothing CPU-heavy meanwhile), the seam
merge, the fallback to one decode when a half fails, the temp proxies and the wiring in
``analyze_source``. ffmpeg and PySceneDetect are faked, apart from one real-ffmpeg seam test
that decodes both halves on the CPU (no GPU needed)."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import StubTranscriber, media_info, needs_ffmpeg, write
from cutpoint import analyze, cache, ffmpeg_utils, scenes
from cutpoint.errors import FFmpegError
from cutpoint.events.ocr_prefetch import OcrPrefetch
from cutpoint.ffmpeg_utils import RunCancelled
from cutpoint.styles import get_style
from scene_helpers import (
    CUDA,
    concat_clip,
    late_copy,
    near,
    nvdec_source,
    scene_world,
)

ENV = "CUTPOINT_SCENE_DECODE"


@pytest.fixture(autouse=True)
def plain(monkeypatch):
    """No env override, 8 logical CPUs, NVDEC present, H.264; tests change what they check."""
    monkeypatch.delenv(ENV, raising=False)
    nvdec_source(monkeypatch, 600.0)
    monkeypatch.setattr(analyze.os, "cpu_count", lambda: 8)


def plan(cpu_free=True, duration=600.0, offset=0.0, limit=None):
    return scenes._hybrid_plan(Path("a.mp4"), duration, offset, limit, cpu_free)


# ----------------------------------------------------------------- the switch and the gate
@pytest.mark.parametrize(
    ("value", "mode"),
    [(None, "auto"), ("HYBRID", "hybrid"), (" gpu ", "gpu"), ("split", "auto")],
)
def test_the_env_mode(monkeypatch, value, mode):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert scenes.scene_decode_mode() == mode


def test_auto_splits_a_long_enough_source_on_nvdec_with_the_cpu_free():
    hw, split, threads = plan()
    assert hw == CUDA and threads == 6  # 8 logical CPUs, 2 left as headroom
    assert split == scenes._hybrid_split(600.0, 6) == 380.7


@pytest.mark.parametrize(
    ("setup", "splits"),
    [
        ({"cpu_free": False}, False),  # the background OCR or Whisper on the CPU runs meanwhile
        ({"duration": scenes.HYBRID_MIN_S - 0.1}, False),  # too short to pay
        ({"duration": scenes.HYBRID_MIN_S}, True),
        ({"duration": scenes.LONG_FILE_S + 1}, False),  # keyframes only: already fast
        ({"offset": 300.0}, False),  # a recovered part
        ({"limit": 200.0}, False),  # a source with a read error
        ({"cpus": 5}, False),  # fewer than HYBRID_MIN_THREADS left after the headroom
        # the CPU half of a 4K HEVC source took twice as long as NVDEC alone; NVDEC here
        # decodes no 10-bit or 4:2:2 H.264; forced, any codec and format NVDEC decodes
        ({"probe": {"vcodec": "hevc"}}, False),
        ({"probe": {"pix_fmt": "yuv420p10le"}}, False),
        ({"probe": {"vcodec": "hevc"}, "env": "hybrid"}, True),
        # the CPU half seeks: MPEG-TS seeks inexactly (#976), so it decodes in one pass, also
        # when forced
        ({"name": "a.ts"}, False),
        ({"name": "a.ts", "env": "hybrid"}, False),
        ({"name": "a.MOV"}, True),
        ({"name": "a.webm"}, True),  # #1038: WebM seeks exactly, also without cues
    ],
    ids=[
        "cpu-busy", "short", "minimum-length", "long", "recovered-part", "read-error",
        "few-cpus", "hevc", "10-bit", "forced-hevc", "ts", "forced-ts", "mov", "webm",
    ],
)  # fmt: skip
def test_the_split_gate(monkeypatch, setup, splits):
    setup = dict(setup)
    if "env" in setup:
        monkeypatch.setenv(ENV, setup.pop("env"))
    if "cpus" in setup:
        cpus = setup.pop("cpus")
        monkeypatch.setattr(analyze.os, "cpu_count", lambda: cpus)
    if "probe" in setup:
        over = setup.pop("probe")
        monkeypatch.setattr(scenes, "probe", lambda p: media_info(p, 600.0, **over))
    name = setup.pop("name", "a.mp4")
    kw = {"cpu_free": True, "duration": 600.0, "offset": 0.0, "limit": None} | setup
    got = scenes._hybrid_plan(Path(name), kw["duration"], kw["offset"], kw["limit"], kw["cpu_free"])
    assert (got is not None) is splits


def test_a_ts_on_cuda_is_decoded_by_ffmpeg_alone(monkeypatch):
    """A non-exact container on CUDA: -hwaccel cuda alone and the plain chain (NVDEC
    reinitialises at a join, so no hwdownload in the chain)."""
    args, chain = scenes._scene_decode(CUDA, "fps=2,scale=256:-2", Path("clip.ts"))
    assert args == ["-hwaccel", "cuda"] and "hwdownload" not in chain


def test_auto_splits_with_just_enough_cpus(monkeypatch):
    """6 logical CPUs leave the minimum 4 decode threads after the headroom of 2."""
    monkeypatch.setattr(analyze.os, "cpu_count", lambda: 6)
    hw, split, threads = plan()
    assert hw == CUDA and threads == scenes.HYBRID_MIN_THREADS == 4
    assert split == scenes._hybrid_split(600.0, 4)


@pytest.mark.parametrize("hw", [None, ffmpeg_utils.HwDecoder("vaapi", "/dev/dri/renderD128")])
def test_no_split_without_nvdec(monkeypatch, hw):
    monkeypatch.setattr(scenes, "decoder_for", lambda src: hw)
    assert plan() is None
    monkeypatch.setenv(ENV, "hybrid")
    assert plan() is None  # forced: still needs NVDEC


@pytest.mark.parametrize("mode", ["gpu", "cpu"])
def test_gpu_and_cpu_never_split(monkeypatch, mode):
    monkeypatch.setenv(ENV, mode)
    assert plan() is None


def test_hybrid_splits_without_the_length_cpu_and_thread_gates(monkeypatch):
    monkeypatch.setenv(ENV, "hybrid")
    monkeypatch.setattr(analyze.os, "cpu_count", lambda: 2)
    hw, _split, threads = plan(cpu_free=False, duration=30.0)
    assert hw == CUDA and threads == 1
    assert plan(offset=10.0) is None and plan(duration=scenes.LONG_FILE_S + 1) is None


def test_the_split_balances_nvdec_against_the_cpu_threads():
    """On 8 threads (the measured machine) NVDEC takes about 57%; fewer threads, more; always
    on the 0.1 s proxy grid."""
    assert scenes._hybrid_split(1000.0, 8) == 565.6
    assert scenes._hybrid_split(1000.0, 6) == 634.5
    for d in (120.0, 218.3, 599.97):
        s = scenes._hybrid_split(d, 6)
        assert abs(s * 10 - round(s * 10)) < 1e-6


# ----------------------------------------------------------------- the seam
SPLIT, CSTART = 100.0, 97.0


@pytest.mark.parametrize(
    ("gpu", "cpu", "want"),
    [
        ([10.0, 50.0], [20.0, 40.0], [10.0, 50.0, 117.0, 137.0]),  # each side its own cuts
        ([10.0, 100.0], [3.0], [10.0, 100.0]),  # at the seam: from the CPU half only
        ([99.9, 101.0], [2.9, 4.0], [99.9, 101.0]),  # in the overlap: each half's own side
        ([99.95], [3.0], [99.95]),  # the same cut a frame off at the seam: once
        ([99.5], [3.0], [99.5, 100.0]),  # 0.5 s apart: two cuts, as one pass gives them
        ([], [0.4, 2.0, 3.5], [100.5]),  # the CPU lead-in's cuts belong to the NVDEC half
        ([102.0], [], []),  # the NVDEC half's cuts past the split are dropped
    ],
)
def test_the_seam_merge(gpu, cpu, want):
    assert scenes._merge_seam(gpu, cpu, SPLIT, CSTART) == want


# ----------------------------------------------------------------- the split decode, faked
@pytest.fixture
def ff(tmp_path, monkeypatch):
    """Fake ffmpeg (writes the proxy) and scan (cuts by half); ``f.fail`` names a failing half."""
    f = type("FF", (), {})()
    f.calls, f.fail, f.seen = [], None, []
    f.cuts = {"gpu": [10.0, 99.95, 101.0], "cpu": [3.0, 50.0]}
    monkeypatch.setattr(scenes, "probe", lambda p: media_info(p, 200.0))
    monkeypatch.setattr(scenes, "STOP_GRACE_S", 0.01)
    monkeypatch.setattr(analyze.importlib, "import_module", lambda name: None)

    def run_progress(args, total, label, cancel=None):
        half = "gpu" if "-hwaccel" in args else "cpu"
        f.calls.append((half, args, total))
        proxy = Path(args[-1])
        write(proxy, "partial")
        f.seen.append(proxy)
        if f.fail == half:
            raise FFmpegError("ffmpeg failed during scene proxy (exit 1):\nInvalid data found")
        return ffmpeg_utils.ProgressResult(stderr="", out_time=total)

    def scan(proxy, cancel):
        return list(f.cuts["gpu" if "proxy-gpu" in proxy.name else "cpu"])

    monkeypatch.setattr(scenes, "run_progress", run_progress)
    monkeypatch.setattr(scenes, "_scan", scan)
    f.cdir = tmp_path / "cdir"
    f.cdir.mkdir()
    f.src = tmp_path / "a.mp4"
    return f


def leftovers(cdir: Path) -> list[str]:
    return sorted(p.name for p in cdir.glob("proxy*"))


def test_the_halves_decode_their_ranges_and_their_cuts_merge(ff):
    cuts = scenes._hybrid_cuts(ff.src, ff.cdir, 200.0, CUDA, SPLIT, 6)
    assert cuts == [10.0, 99.95, 147.0]  # the CPU's 100.0 is the NVDEC 99.95 a frame off
    by = {half: (args, total) for half, args, total in ff.calls}
    gpu, gsecs = by["gpu"]
    assert gpu[gpu.index("-t") + 1] == "103.000" and gsecs == 103.0
    assert "-ss" not in gpu and gpu.index("-hwaccel") < gpu.index("-i")
    cpu, csecs = by["cpu"]
    assert cpu[cpu.index("-ss") + 1] == "97.000" and csecs == 103.0
    assert cpu[cpu.index("-threads") + 1] == "6" and cpu.index("-threads") < cpu.index("-i")
    # the format NVDEC downloads to, before the scale, as in the NVDEC half: the same pixels
    assert gpu[gpu.index("-vf") + 1] == "fps=10:start_time=0,hwdownload,format=nv12,scale=320:180"
    assert "-hwaccel" not in cpu
    assert cpu[cpu.index("-vf") + 1] == "fps=10:start_time=0,format=nv12,scale=320:180"
    assert "-t" not in cpu  # to the end
    for half in (gpu, cpu):  # one filter graph per half
        assert half[half.index("-reinit_filter") : half.index("-reinit_filter") + 2] == [
            "-reinit_filter",
            "0",
        ] and half.index("-reinit_filter") < half.index("-i")
    assert gpu.index("-reinit_filter") < gpu.index("-hwaccel")
    assert leftovers(ff.cdir) == [] and len(set(ff.seen)) == 2


def test_a_format_nvdec_does_not_download_keeps_the_plain_cpu_chain(ff, monkeypatch):
    monkeypatch.setattr(scenes, "probe", lambda p: media_info(p, 200.0, pix_fmt="yuv444p"))
    scenes._hybrid_cuts(ff.src, ff.cdir, 200.0, CUDA, SPLIT, 6)
    cpu = next(args for half, args, _t in ff.calls if half == "cpu")
    assert cpu[cpu.index("-vf") + 1] == "fps=10:start_time=0,scale=320:180"


def test_both_halves_of_a_late_video_count_from_the_container_start(ff, monkeypatch):
    """#976: in a container that seeks exactly ffmpeg counts both halves from the container
    start; ``fps`` start_time pads the NVDEC half's late video, the CPU half seeks the split
    point as clips are cut. No ``-copyts``, and downloads stay after ``fps``."""
    monkeypatch.setattr(scenes, "probe", lambda p: media_info(p, 200.0, vstart=0.25))
    assert scenes._hybrid_cuts(ff.src, ff.cdir, 200.0, CUDA, SPLIT, 6) == [10.0, 99.95, 147.0]
    by = {half: args for half, args, _t in ff.calls}
    gpu, cpu = by["gpu"], by["cpu"]
    assert gpu[gpu.index("-vf") + 1] == "fps=10:start_time=0,hwdownload,format=nv12,scale=320:180"
    assert "-copyts" not in gpu and "-copyts" not in cpu
    assert cpu[cpu.index("-ss") + 1] == "97.000"
    assert cpu[cpu.index("-vf") + 1] == "fps=10:start_time=0,format=nv12,scale=320:180"


def test_a_split_near_the_start_starts_the_cpu_half_at_0(ff):
    scenes._hybrid_cuts(ff.src, ff.cdir, 200.0, CUDA, 2.0, 6)
    cpu = next(args for half, args, _t in ff.calls if half == "cpu")
    assert cpu[cpu.index("-ss") + 1] == "0.000"


@pytest.mark.parametrize("half", ["gpu", "cpu"])
def test_a_failing_half_gives_none_with_a_warning_and_no_proxies(ff, caplog, half):
    ff.fail = half
    with caplog.at_level("WARNING", logger="cutpoint.scenes"):
        assert scenes._hybrid_cuts(ff.src, ff.cdir, 200.0, CUDA, SPLIT, 6) is None
    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert leftovers(ff.cdir) == []


@pytest.mark.parametrize("half", ["gpu", "cpu"])
def test_scene_cuts_falls_back_to_one_decode_when_a_half_fails(ff, monkeypatch, half):
    ff.fail = half
    single = []

    def make_proxy(src, proxy, args, seconds, cancel=None, gpu=True):
        single.append(gpu)
        write(proxy, "proxy")

    monkeypatch.setattr(scenes, "_make_proxy", make_proxy)
    monkeypatch.setattr(scenes, "_scan", lambda proxy, cancel: [42.0])
    cuts = scenes.scene_cuts(ff.src, ff.cdir, 200.0, cpu_free=True)
    assert cuts == [42.0] and single == [True]
    assert cache.load_json(ff.cdir / scenes.SCENES_FILE) == [42.0] and leftovers(ff.cdir) == []


def test_the_one_decode_proxy_name_is_not_shared_with_other_runs(ff, monkeypatch):
    """Two runs on one cache folder must not overwrite each other's proxy (#1085 A103)."""
    names = []

    def make_proxy(src, proxy, args, seconds, cancel=None, gpu=True):
        names.append(proxy.name)
        write(proxy, "proxy")

    monkeypatch.setattr(scenes, "_make_proxy", make_proxy)
    monkeypatch.setattr(scenes, "_scan", lambda proxy, cancel: [1.0])
    scenes.scene_cuts(ff.src, ff.cdir, 200.0)
    assert names[0] != "proxy.mp4" and str(os.getpid()) in names[0]
    assert not list(ff.cdir.glob("proxy*"))  # removed after the scan


def test_the_one_decode_sweeps_old_proxies_and_keeps_a_live_run_s(ff, monkeypatch):
    live = ff.cdir / f"proxy.{os.getpid()}.{ffmpeg_utils._host_tag()}.abcd1234.tmp.mp4"
    dead = ff.cdir / f"proxy.99999999.{ffmpeg_utils._host_tag()}.abcd1234.tmp.mp4"
    old = ff.cdir / "proxy.mp4"
    for f in (live, dead, old):
        write(f, "x")
    monkeypatch.setattr(ffmpeg_utils.procs, "pid_alive", lambda pid, mtime=None: pid == os.getpid())
    monkeypatch.setattr(scenes, "_make_proxy", lambda src, proxy, *a, **k: write(proxy, "p"))
    monkeypatch.setattr(scenes, "_scan", lambda proxy, cancel: [1.0])
    scenes.scene_cuts(ff.src, ff.cdir, 200.0)
    assert live.exists() and not dead.exists() and not old.exists()


def test_scene_cuts_saves_the_merged_cuts_as_one_pass_would(ff, monkeypatch):
    monkeypatch.setattr(scenes, "_make_proxy", lambda *a, **k: pytest.fail("one decode"))
    cuts = scenes.scene_cuts(ff.src, ff.cdir, 200.0, cpu_free=True)
    assert cache.load_json(ff.cdir / scenes.SCENES_FILE) == cuts
    assert all(isinstance(c, float) for c in cuts) and cuts == sorted(cuts)
    assert scenes.scene_cuts(ff.src, ff.cdir, 200.0, cpu_free=True) == cuts  # cached
    assert len(ff.calls) == 2


@pytest.mark.parametrize(("mode", "gpu"), [("gpu", True), ("cpu", False), ("auto", True)])
def test_one_decode_without_cpu_free_or_with_the_switch(ff, monkeypatch, mode, gpu):
    monkeypatch.setenv(ENV, mode)
    single = []

    def make_proxy(src, proxy, args, seconds, cancel=None, gpu=True):
        single.append(gpu)
        write(proxy, "proxy")

    monkeypatch.setattr(scenes, "_make_proxy", make_proxy)
    scenes.scene_cuts(ff.src, ff.cdir, 200.0, cpu_free=mode != "auto")
    assert single == [gpu] and ff.calls == []


@pytest.mark.parametrize("gpu", [True, False])
def test_a_failed_proxy_encode_removes_the_partial_proxy(monkeypatch, tmp_path, gpu):
    proxy = tmp_path / "p.mp4"
    seen = []

    def run_progress(args, total, label, **kw):
        seen.append(proxy.exists())  # the re-run finds no partial file of the first run
        proxy.write_bytes(b"partial")
        raise FFmpegError("ffmpeg failed during scene proxy (exit 1):\n\nboom")

    monkeypatch.setattr(scenes, "run_progress", run_progress)
    monkeypatch.setattr(scenes, "decoder_for", lambda src: CUDA if gpu else None)
    with pytest.raises(FFmpegError):
        scenes._make_proxy(tmp_path / "a.mp4", proxy, scenes._proxy_args(10.0), 10.0, gpu=gpu)
    assert not proxy.exists()
    assert seen == ([False, False] if gpu else [False])


def test_a_failed_gpu_proxy_encode_reruns_on_the_cpu(monkeypatch, tmp_path):
    runs = []

    def run_progress(args, total, label, **kw):
        runs.append(args)
        if len(runs) == 1:
            raise FFmpegError(
                "ffmpeg failed during scene proxy (exit 1):\n\nCannot load libnvcuvid\nmore"
            )

    monkeypatch.setattr(scenes, "run_progress", run_progress)
    src = tmp_path / "a.mp4"
    scenes._make_proxy(src, tmp_path / "p.mp4", scenes._proxy_args(10.0), 10.0)
    assert len(runs) == 2


def test_ctrl_c_stops_both_halves_and_removes_the_proxies(ff, monkeypatch):
    """KeyboardInterrupt in the waiting thread: both ffmpeg runs are told to stop (their
    ``cancel``), the proxies go and the interrupt is raised."""
    started, stopped = threading.Barrier(3), []

    def run_progress(args, total, label, cancel=None):
        write(Path(args[-1]), "partial")
        started.wait(5)
        assert cancel.wait(5)
        stopped.append(label)
        raise RunCancelled(label)

    monkeypatch.setattr(scenes, "run_progress", run_progress)
    real_join = threading.Thread.join

    def join(self, timeout=None):
        if self.name.startswith("scene-") and threading.current_thread() is threading.main_thread():
            started.wait(5)
            monkeypatch.setattr(threading.Thread, "join", real_join)
            raise KeyboardInterrupt
        return real_join(self, timeout)

    monkeypatch.setattr(threading.Thread, "join", join)
    with pytest.raises(KeyboardInterrupt):
        scenes._hybrid_cuts(ff.src, ff.cdir, 200.0, CUDA, SPLIT, 6)
    assert len(stopped) == 2 and leftovers(ff.cdir) == []


def test_a_set_cancel_stops_the_split_and_writes_nothing(ff, monkeypatch):
    """The overlap thread's ``cancel`` (Whisper failed or stopped) reaches both halves."""
    outer = threading.Event()

    def run_progress(args, total, label, cancel=None):
        write(Path(args[-1]), "partial")
        outer.set()
        assert cancel is not outer and cancel.wait(5)  # the split's own stop, set from outer
        raise RunCancelled(label)

    monkeypatch.setattr(scenes, "run_progress", run_progress)
    with pytest.raises(RunCancelled):
        scenes.scene_cuts(ff.src, ff.cdir, 200.0, cancel=outer, cpu_free=True)
    assert leftovers(ff.cdir) == [] and not (ff.cdir / scenes.SCENES_FILE).exists()


# ----------------------------------------------------------------- the OCR busy gate
def prefetch_with(current, jobs):
    pre = OcrPrefetch.__new__(OcrPrefetch)  # only the state busy reads
    pre.current, pre._jobs = current, jobs
    return pre


def job(done: bool):
    ev = threading.Event()
    if done:
        ev.set()
    return SimpleNamespace(done=ev)


def test_the_prefetch_is_busy_with_an_offered_source_the_worker_has_not_taken():
    """``offer`` queues a job before the worker sets ``current``: busy from the offer on."""
    assert prefetch_with(None, {"a": job(False)}).busy is True
    assert prefetch_with(object(), {"a": job(True)}).busy is True
    assert prefetch_with(None, {"a": job(True), "b": job(True)}).busy is False
    assert prefetch_with(None, {}).busy is False


# ----------------------------------------------------------------- analyze_source's gate
@pytest.fixture
def world(home, tmp_path, monkeypatch):
    """``analyze_source`` with ffprobe, audio, Whisper and loudness faked; ``w.kw`` records the
    keyword arguments of each ``scene_cuts`` call, ``w.jobs`` the ``SceneJob`` arguments."""
    w = scene_world(monkeypatch, tmp_path, 600.0, flat_lufs=-30.0)
    src, w.kw, w.jobs = w.src, [], []

    def scene(*a, **k):
        w.kw.append(k)
        return [16.4]

    class Job:
        def __init__(self, *args):
            w.jobs.append(args)
            self.cuts, self.took = [16.4], 0.5

        def result(self):
            return self.cuts

    monkeypatch.setattr(scenes, "scene_cuts", scene)
    monkeypatch.setattr(scenes, "SceneJob", Job)
    w.source = lambda tr=None, **kw: analyze.analyze_source(
        src, get_style("action"), tr or StubTranscriber(), True, None, {}, **kw
    )
    return w


@pytest.mark.parametrize(
    ("busy", "free"), [(None, True), (lambda: False, True), (lambda: True, False)],
    ids=["no-ocr", "ocr-idle", "ocr-reading"],
)  # fmt: skip
def test_the_serial_scene_pass_is_free_unless_the_ocr_reads(world, busy, free):
    world.source(cpu_busy=busy)
    (kw,) = world.kw
    assert kw.get("cpu_free", False) is free
    if not free:
        assert "cpu_free" not in kw  # the plain call, as before


@pytest.mark.parametrize(
    ("device", "busy", "free"),
    [("cpu", None, False), ("cuda", None, True), ("cuda", lambda: True, False)],
    ids=["cpu-whisper", "cuda-whisper", "cuda-whisper-ocr-reading"],
)
def test_the_overlapped_scene_pass_is_free_only_with_whisper_on_the_gpu(
    world, monkeypatch, device, busy, free
):
    monkeypatch.setenv("CUTPOINT_ANALYZE_SCENE_OVERLAP", "1")
    tr = type("Tr", (StubTranscriber,), {"device": device, "load": lambda self: None})()
    res = world.source(tr, cpu_busy=busy)
    (args,) = world.jobs
    assert args[-1] is free and res.stages["overlap"] is True and world.kw == []


def test_the_ocr_busy_gate_follows_the_prefetch_worker(world, monkeypatch):
    """``analyze`` asks the prefetch whether it reads or is about to read a source."""
    pre = SimpleNamespace(mode=None, waits={}, offer=lambda *a: None, busy=False)
    monkeypatch.setattr(analyze, "_ocr_prefetch", lambda *a, **k: contextlib.nullcontext(pre))
    monkeypatch.setattr(analyze, "Transcriber", StubTranscriber)
    for busy, free in ((False, True), (True, False)):
        pre.busy = busy
        world.kw.clear()
        analyze.analyze(world.src, get_style("action"), world.src.parent.parent / "p",
                        vision=False, use_scenes=True, layers=False)  # fmt: skip
        (kw,) = world.kw
        assert kw.get("cpu_free", False) is free


# ----------------------------------------------------------------- real ffmpeg: the seam
@pytest.fixture
def cuts_clip(tmp_path) -> Path:
    """A 24 s 160x120 10 fps clip with hard cuts at 4, 8, 11, 11.6, 15, 15.3 and 19 s; one pass
    drops 15.3 (min_scene_len: under 0.5 s after 15)."""
    parts = [
        ("testsrc2", 4.0), ("color=c=red", 4.0), ("testsrc", 3.0), ("color=c=blue", 0.6),
        ("smptebars", 3.4), ("color=c=green", 0.3), ("rgbtestsrc", 3.7), ("color=c=white", 5.0),
    ]  # fmt: skip
    codec = ("-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "20")
    return concat_clip(tmp_path / "cuts.mp4", parts, *codec)


@needs_ffmpeg
def test_a_split_anywhere_gives_the_cuts_of_one_pass(cuts_clip, tmp_path, monkeypatch):
    """Both halves on the CPU (no GPU needed): with the split on a cut, a frame either side of
    one and inside the overlap of two close cuts, the merged cuts equal one pass's."""
    monkeypatch.setattr(scenes, "decoder_for", lambda src: None)
    cdir = tmp_path / "cdir"
    cdir.mkdir()
    duration = 24.0
    proxy = cdir / "one.mp4"
    scenes._make_proxy(cuts_clip, proxy, scenes._proxy_args(duration), duration, gpu=False)
    one = scenes._scan(proxy, None)
    proxy.unlink()
    assert one == [4.0, 8.0, 11.0, 11.6, 15.0, 19.0]
    for split in (11.0, 10.9, 11.1, 11.6, 11.3, 15.0, 15.2, 15.3, 15.4, 6.0):
        got = scenes._hybrid_cuts(cuts_clip, cdir, duration, None, split, 2)
        assert got == one, split
    assert leftovers(cdir) == []


@pytest.fixture
def late_and_ts(cuts_clip, tmp_path) -> tuple[Path, Path]:
    """``cuts_clip`` with an audio track and its video 0.25 s late (MP4), and an MPEG-TS copy."""
    late = late_copy(cuts_clip, tmp_path / "late.mp4", "aac")
    ts = tmp_path / "cuts.ts"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-i", str(late), "-c", "copy", "-f", "mpegts",
         str(ts)],
        check=True,
    )  # fmt: skip
    return late, ts


@needs_ffmpeg
def test_auto_splits_a_late_mp4_but_not_mpegts(cuts_clip, late_and_ts, monkeypatch):
    """A late MP4 splits (#1038: its halves match one pass); MPEG-TS stays one decode."""
    monkeypatch.setattr(scenes, "probe", ffmpeg_utils.probe)
    late, ts = late_and_ts
    assert ffmpeg_utils.probe(late).vstart > 0.2
    assert scenes._hybrid_plan(cuts_clip, 600.0, 0.0, None, True) is not None  # the control
    assert scenes._hybrid_plan(late, 600.0, 0.0, None, True) is not None
    assert scenes._hybrid_plan(ts, 600.0, 0.0, None, True) is None


CLIP_CUTS = [4.0, 8.0, 11.0, 11.6, 15.0, 19.0]  # one pass of cuts_clip (15.3 dropped)


def one_pass(src: Path, cdir: Path) -> list[float]:
    """The full pass of ``src`` as ``scene_cuts`` makes it (one CPU decode), cached in ``cdir``."""
    cdir.mkdir()
    return scenes.scene_cuts(src, cdir, ffmpeg_utils.probe(src).duration)


@needs_ffmpeg
def test_a_forced_split_of_a_late_video_matches_one_pass(late_and_ts, tmp_path, monkeypatch):
    """Both halves count from the container start, as the full pass does (#976): the split's
    cuts land where one pass's do (within one 0.1 s proxy frame), on any ffmpeg."""
    monkeypatch.setattr(scenes, "probe", ffmpeg_utils.probe)
    monkeypatch.setattr(scenes, "decoder_for", lambda src: None)
    late, _ts = late_and_ts
    duration = ffmpeg_utils.probe(late).duration
    one = one_pass(late, tmp_path / "one")
    assert near(one, [c + 0.25 for c in CLIP_CUTS]), one  # the container clock
    cdir = tmp_path / "late-cdir"
    cdir.mkdir()
    for split in (11.0, 10.9, 11.6, 15.2, 6.0):
        got = scenes._hybrid_cuts(late, cdir, duration, None, split, 2)
        assert near(got, one), (split, got, one)
    assert leftovers(cdir) == []


@needs_ffmpeg
def test_every_scene_pass_counts_from_the_container_start(late_and_ts, tmp_path, monkeypatch):
    """#976 on a video 0.25 s late: the full pass, the edge windows (one at the part start, so
    without a seek) give the cuts 0.25 s after the clip's; the MPEG-TS copy gets its full pass
    instead of windows, on the same clock."""
    monkeypatch.setattr(scenes, "probe", ffmpeg_utils.probe)
    monkeypatch.setattr(scenes, "decoder_for", lambda src: None)
    late, ts = late_and_ts
    want = [c + 0.25 for c in CLIP_CUTS]
    full = one_pass(late, tmp_path / "full")
    assert near(full, want), full
    span = ffmpeg_utils.probe(late).duration
    wdir = tmp_path / "windows"
    wdir.mkdir()
    edges = [2.5, 8.0, 11.5, 15.0, 19.0]  # the first window starts at 0: no seek
    got = scenes.EdgeCuts(late, wdir, span).around(edges)
    assert (wdir / scenes.SCENE_WINDOWS_FILE).exists()
    assert not (wdir / scenes.SCENES_FILE).exists()
    assert near(got, full), (got, full)
    tdir = tmp_path / "ts"
    tdir.mkdir()
    got = scenes.EdgeCuts(ts, tdir, ffmpeg_utils.probe(ts).duration).around(edges)
    assert near(got, want), got
    assert (tdir / scenes.SCENES_FILE).exists()
    assert not (tdir / scenes.SCENE_WINDOWS_FILE).exists()


@pytest.fixture
def mid_gop_ts(cuts_clip, tmp_path) -> tuple[Path, float]:
    """``cuts_clip`` with audio as MPEG-TS (3 s GOPs, no B-frames) with its first 8% of 188-byte
    packets cut off, so the video starts in the middle of a GOP and decodes from the next
    keyframe; and where the clip's time 0 lies on that file's container clock."""
    ff = shutil.which("ffmpeg")
    whole, cut = tmp_path / "whole.ts", tmp_path / "mid.ts"
    subprocess.run(
        [ff, "-v", "error", "-i", str(cuts_clip), "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=48000:duration=24", "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-g", "30", "-bf", "0", "-c:a", "aac", "-f", "mpegts", str(whole)],
        check=True,
    )  # fmt: skip
    data = whole.read_bytes()
    cut.write_bytes(data[(len(data) // 188 * 8 // 100) * 188 :])
    info, part = ffmpeg_utils.probe(whole), ffmpeg_utils.probe(cut)
    return cut, info.fstart + info.vstart - part.fstart


@needs_ffmpeg
def test_an_mpegts_cut_mid_gop_keeps_its_cuts_on_the_container_clock(
    mid_gop_ts, tmp_path, monkeypatch
):
    """#976 review: the full pass counts from the container start, not from the first frame
    it could decode (the dropped lead-in would move every cut early by about 1.7 s)."""
    monkeypatch.setattr(scenes, "probe", ffmpeg_utils.probe)
    monkeypatch.setattr(scenes, "decoder_for", lambda src: None)
    ts, zero = mid_gop_ts
    assert zero < 0  # the clip's start was cut off
    want = [round(c + zero, 3) for c in CLIP_CUTS if c + zero > 0.5]
    got = one_pass(ts, tmp_path / "mid")
    assert near(got, want), (got, want)


@pytest.fixture
def joined_ts(cuts_clip, tmp_path) -> Path:
    """``cuts_clip`` as two 12 s MPEG-TS files with audio joined byte for byte, as camera clips
    or DVR captures are: the second one's timestamps start again where the first one's did, and
    its color parameters differ, so ffmpeg rebuilds the filter graph at the join."""
    ff = shutil.which("ffmpeg")
    halves = []
    for i, start in enumerate(("0", "12")):
        color = ["-colorspace", "bt470bg", "-color_primaries", "bt470bg", "-color_trc", "gamma28"]
        half = tmp_path / f"half{i}.ts"
        subprocess.run(
            [ff, "-v", "error", "-ss", start, "-t", "12", "-i", str(cuts_clip), "-f", "lavfi",
             "-i", "sine=frequency=440:sample_rate=48000:duration=12", "-map", "0:v", "-map",
             "1:a", "-c:v", "libx264", *(color if i else []), "-c:a", "aac", "-f", "mpegts",
             str(half)],
            check=True,
        )  # fmt: skip
        halves.append(half.read_bytes())
    out = tmp_path / "joined.ts"
    out.write_bytes(b"".join(halves))
    return out


@needs_ffmpeg
def test_an_mpegts_with_a_timestamp_reset_keeps_the_cuts_after_it(joined_ts, tmp_path, monkeypatch):
    """#976 on Windows: ffmpeg runs a joined TS on past its timestamp reset, and so does the
    full pass (its own timestamps, the first frame moved to its container time): every cut of
    the clip, on the container clock, also the two after the join at 12 s."""
    monkeypatch.setattr(scenes, "probe", ffmpeg_utils.probe)
    monkeypatch.setattr(scenes, "decoder_for", lambda src: None)
    vstart = ffmpeg_utils.probe(joined_ts).vstart
    got = one_pass(joined_ts, tmp_path / "joined")
    assert near(got, [c + vstart for c in CLIP_CUTS]), got


@needs_ffmpeg
def test_a_picture_that_widens_mid_file_reaches_the_encoder_in_one_size(tmp_path, monkeypatch):
    """A 320x240 and a 640x360 TS joined byte for byte: with ``-reinit_filter 0`` the proxy
    chain keeps one graph, so ``scale=320:-2`` alone would pass 320x180 frames on (ffmpeg 9
    then crashes in libx264 now and then). The probed size fixes the height: every raw frame
    of the pass has the one size."""
    monkeypatch.setattr(scenes, "probe", ffmpeg_utils.probe)
    halves = []
    for i, size in enumerate(("320x240", "640x360")):
        half = tmp_path / f"w{i}.ts"
        subprocess.run(
            [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
             f"testsrc2=size={size}:rate=10:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-f", "mpegts", str(half)],
            check=True,
        )  # fmt: skip
        halves.append(half.read_bytes())
    joined = tmp_path / "widen.ts"
    joined.write_bytes(b"".join(halves))
    pre, _post, vf, _enc = scenes._proxy_args(4.0)
    vf = scenes._fixed_scale(vf, joined)
    assert vf.endswith("scale=320:240")
    raw = subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", *pre, "-i", str(joined), "-an", "-vf", vf,
         "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"],
        check=True, capture_output=True,
    ).stdout  # fmt: skip
    frame = 320 * 240 * 3 // 2
    assert len(raw) % frame == 0 and len(raw) // frame >= 38, len(raw)  # 40 at one size


@needs_ffmpeg
def test_a_joined_mpegts_keeps_its_cuts_after_the_join_on_the_gpu(joined_ts, tmp_path, monkeypatch):
    """Windows, NVDEC: the decoder reinitialises at the join and a ``hwdownload`` chain lost
    every cut after it (the encoder dropped the second half, ffmpeg exited 0)."""
    dec = ffmpeg_utils.gpu_decoder()
    if dec is None:
        pytest.skip("no GPU decoder works here")
    monkeypatch.setattr(scenes, "probe", ffmpeg_utils.probe)
    monkeypatch.setattr(scenes, "decoder_for", lambda src: dec)
    vstart = ffmpeg_utils.probe(joined_ts).vstart
    got = one_pass(joined_ts, tmp_path / "joined")
    assert near(got, [c + vstart for c in CLIP_CUTS]), got


@needs_ffmpeg
@pytest.mark.parametrize("suffix", [".mkv", ".mp4"])
def test_a_picture_that_changes_size_mid_file_keeps_its_cuts_on_the_gpu(
    tmp_path, monkeypatch, suffix
):
    """#1031: an MKV (or MP4) whose second part has another size (1280x720, then 1920x1080): NVDEC starts
    again at the change, ffmpeg rebuilds the graph and the encoder dropped frames, so every
    later cut came late or not at all. The GPU result is the CPU's."""
    dec = ffmpeg_utils.gpu_decoder()
    if dec is None:
        pytest.skip("no GPU decoder works here")
    ff = shutil.which("ffmpeg")
    first, second = tmp_path / "a.ts", tmp_path / "b.ts"
    subprocess.run(
        [ff, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:duration=6", "-c:v",
         "libx264", "-pix_fmt", "yuv420p", "-f", "mpegts", str(first)],
        check=True,
    )  # fmt: skip
    subprocess.run(  # 1920x1080: 3 s of testsrc2, then a cut to 3 s of testsrc
        [ff, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:duration=3", "-f",
         "lavfi", "-i", "testsrc=size=1920x1080:duration=3", "-filter_complex",
         "[0][1]concat=n=2:v=1:a=0,format=yuv420p", "-c:v", "libx264", "-output_ts_offset", "6",
         "-f", "mpegts", str(second)],
        check=True,
    )  # fmt: skip
    parts = [first.read_bytes(), second.read_bytes()]
    joined = tmp_path / "ab.ts"
    joined.write_bytes(b"".join(parts))
    mkv = tmp_path / f"ab{suffix}"
    subprocess.run([ff, "-v", "error", "-i", str(joined), "-c", "copy", str(mkv)], check=True)
    monkeypatch.setattr(scenes, "probe", ffmpeg_utils.probe)
    monkeypatch.setattr(scenes, "decoder_for", lambda src: None)
    cpu = one_pass(mkv, tmp_path / "cpu")
    monkeypatch.setattr(scenes, "decoder_for", lambda src: dec)
    gpu = one_pass(mkv, tmp_path / "gpu")
    assert len(cpu) == 2, cpu  # the size change and the cut after it
    # half a proxy frame: without the re-run the second cut came a whole frame (0.1 s) late
    assert len(gpu) == len(cpu) and all(
        abs(g - c) < scenes.SCENE_GRID_S / 2 for g, c in zip(gpu, cpu, strict=True)
    ), (gpu, cpu)


# ----------------------------------------------------------------- the first frame's time
SHOWINFO_6 = """[Parsed_showinfo_0 @ 0x1] config in time_base: 1/90000, frame_rate: 10/1
[Parsed_showinfo_0 @ 0x1] config out time_base: 0/0, frame_rate: 0/0
[Parsed_showinfo_0 @ 0x1] n:   0 pts:4889019933 pts_time:54322.4 duration:   9000 \
duration_time:0.1     fmt:yuv444p cl:left sar:1/1 s:64x48 i:P iskey:1 type:I checksum:6152D0EA
[Parsed_showinfo_0 @ 0x1] n:   1 pts:4889028933 pts_time:54322.5 duration:   9000"""
SHOWINFO_7 = """[Parsed_showinfo_0 @ 0x1] config in time_base: 1/1000, frame_rate: 30000/1001
[Parsed_showinfo_0 @ 0x1] n:   0 pts:     33 pts_time:0.033    duration:     33 \
duration_time:0.033   fmt:yuv420p cl:left sar:1/1 s:160x120 i:P iskey:1 type:I"""


@pytest.mark.parametrize(
    ("log", "want"),
    [(SHOWINFO_6, 4889019933 / 90000), (SHOWINFO_7, 0.033), ("no frame", None),
     ("[x] config in time_base: 1/90000\n[x] n:   0 pts:NOPTS pts_time:NOPTS", None)],
    ids=["ffmpeg-6", "ffmpeg-7", "no-frame", "nopts"],
)  # fmt: skip
def test_the_first_frame_time_comes_from_the_integer_pts(log, want):
    """ffmpeg 6 prints pts_time with 6 significant digits (54322.4 for 54322.4437 s): the time
    is the integer pts times the time base."""
    got = ffmpeg_utils.showinfo_first_time(log)
    assert got == want if want is None else abs(got - want) < 1e-9


@needs_ffmpeg
def test_a_high_timestamp_ts_puts_its_first_frame_at_the_video_start(
    cuts_clip, tmp_path, monkeypatch
):
    """A DVR or broadcast capture with timestamps past 10000 s: the shift is the video start to
    the millisecond, not 0.05 s off as ffmpeg 6's pts_time would give."""
    monkeypatch.setattr(scenes, "probe", ffmpeg_utils.probe)
    ts = tmp_path / "high.ts"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-i", str(cuts_clip), "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=48000:duration=24", "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", "-output_ts_offset", "54321.0437", "-f", "mpegts",
         str(ts)],
        check=True,
    )  # fmt: skip
    info = ffmpeg_utils.probe(ts)
    assert info.fstart > 54000
    shift = scenes._first_shift(ts, 0.0, info.duration)
    assert shift is not None and abs(shift - info.vstart) < 1e-3, (shift, info.vstart)


@needs_ffmpeg
def test_the_first_frame_is_of_the_video_stream_the_pass_decodes(cuts_clip, tmp_path, monkeypatch):
    """Two video streams: the pass decodes the largest (no ``-map``), which starts 2 s after a
    small one, so its cuts and its first frame come from that stream."""
    monkeypatch.setattr(scenes, "probe", ffmpeg_utils.probe)
    monkeypatch.setattr(scenes, "decoder_for", lambda src: None)
    ts = tmp_path / "two.ts"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-i", str(cuts_clip), "-itsoffset", "2",
         "-i", str(cuts_clip), "-filter_complex", "[0:v]scale=64:48[s]", "-map", "[s]",
         "-map", "1:v", "-c:v", "libx264", "-f", "mpegts", str(ts)],
        check=True,
    )  # fmt: skip
    data = ffmpeg_utils.ffprobe_json(ts, show_format=True, show_streams=True)
    big = next(s for s in data["streams"] if s.get("width") == 160)
    lead = float(big["start_time"]) - float(data["format"]["start_time"])
    assert lead > 1.9
    got = one_pass(ts, tmp_path / "two")
    assert near(got, [c + lead for c in CLIP_CUTS]), (got, lead)


def test_the_scene_proxy_is_lossless():
    """A lossy proxy has a keyframe every 250 frames (25 s at 10 fps) that adds to the
    detector's content value: cuts appeared there and moved with a pass's start (#994)."""
    _pre, _post, _vf, enc = scenes._proxy_args(60.0)
    assert enc[enc.index("-qp") + 1] == "0"
    assert "-crf" not in enc


@needs_ffmpeg
def test_the_scene_proxy_holds_the_filtered_frames_exactly(tmp_path):
    """The proxy decodes to the very frames the filter chain gives (no coding loss)."""
    src = tmp_path / "s.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=320x180:rate=10:duration=4", "-pix_fmt", "yuv420p", str(src)],
        check=True,
    )  # fmt: skip
    args = scenes._proxy_args(4.0)
    proxy = tmp_path / "proxy.mp4"
    scenes._make_proxy(src, proxy, args, 4.0, gpu=False)

    def md5(*cmd: str) -> list[str]:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", *cmd, "-f", "framemd5", "-"],
            capture_output=True, text=True, check=True,
        ).stdout  # fmt: skip
        return [ln.rsplit(",", 1)[-1].strip() for ln in out.splitlines() if ln[:1] != "#"]

    direct = md5("-i", str(src), "-vf", args[2], "-pix_fmt", "yuv420p")
    assert direct and md5("-i", str(proxy)) == direct
