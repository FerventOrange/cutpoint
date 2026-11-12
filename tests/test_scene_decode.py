"""The scene pass split between NVDEC and the CPU (#730 PR 2): the gate
(``CUTPOINT_SCENE_DECODE``, CUDA, length, CPU threads, nothing CPU-heavy meanwhile), the seam
merge, the fallback to one decode when a half fails, the temp proxies and the wiring in
``analyze_source``. ffmpeg and PySceneDetect are faked, apart from one real-ffmpeg seam test
that decodes both halves on the CPU (no GPU needed)."""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import StubTranscriber, media_info, needs_ffmpeg, write
from cutpoint import analyze, cache, ffmpeg_utils
from cutpoint.errors import FFmpegError
from cutpoint.events.ocr_prefetch import OcrPrefetch
from cutpoint.ffmpeg_utils import RunCancelled
from cutpoint.styles import get_style

ENV = "CUTPOINT_SCENE_DECODE"
CUDA = ffmpeg_utils.HwDecoder("cuda")


@pytest.fixture(autouse=True)
def plain(monkeypatch):
    """No env override, 8 logical CPUs, NVDEC present, H.264; tests change what they check."""
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(analyze, "probe", lambda p: media_info(p, 600.0))
    monkeypatch.setattr(analyze.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(analyze, "decoder_for", lambda src: CUDA)


def plan(cpu_free=True, duration=600.0, offset=0.0, limit=None):
    return analyze._hybrid_plan(Path("a.mp4"), duration, offset, limit, cpu_free)


# ----------------------------------------------------------------- the switch and the gate
@pytest.mark.parametrize(
    ("value", "mode"),
    [(None, "auto"), ("auto", "auto"), ("HYBRID", "hybrid"), (" gpu ", "gpu"), ("cpu", "cpu"),
     ("split", "auto"), ("", "auto")],
)  # fmt: skip
def test_the_env_mode(monkeypatch, value, mode):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert analyze.scene_decode_mode() == mode


def test_auto_splits_a_long_enough_source_on_nvdec_with_the_cpu_free():
    hw, split, threads = plan()
    assert hw == CUDA and threads == 6  # 8 logical CPUs, 2 left as headroom
    assert split == analyze._hybrid_split(600.0, 6) == 380.7


@pytest.mark.parametrize(
    "kw",
    [
        {"cpu_free": False},  # the background OCR or Whisper on the CPU runs meanwhile
        {"duration": analyze.HYBRID_MIN_S - 0.1},  # too short to pay
        {"duration": analyze.LONG_FILE_S + 1},  # keyframes only: already fast
        {"offset": 300.0},  # a recovered part
        {"limit": 200.0},  # a source with a read error
    ],
    ids=["cpu-busy", "short", "long", "recovered-part", "read-error"],
)
def test_auto_keeps_one_decode(kw):
    assert plan(**kw) is None


def test_auto_splits_exactly_at_the_minimum_length():
    assert plan(duration=analyze.HYBRID_MIN_S) is not None


@pytest.mark.parametrize("vcodec", ["hevc", "av1", "vp9", None])
def test_auto_splits_only_h264(monkeypatch, vcodec):
    """The CPU half of a 4K HEVC source took twice as long as NVDEC alone."""
    monkeypatch.setattr(analyze, "probe", lambda p: media_info(p, 600.0, vcodec=vcodec))
    assert plan() is None
    monkeypatch.setenv(ENV, "hybrid")
    assert plan() is not None  # forced: any codec NVDEC decodes


@pytest.mark.parametrize(
    "over",
    [{"pix_fmt": "yuv420p10le"}, {"pix_fmt": "yuv422p"}, {"pix_fmt": "yuv444p"}, {"vstart": 0.25}],
    ids=["10-bit", "4:2:2", "4:4:4", "late-video"],
)
def test_auto_splits_only_8bit_420_starting_with_the_file(monkeypatch, over):
    """NVDEC here decodes no 10-bit or 4:2:2 H.264; a late video start stays one decode until the
    NVDEC half's padded start (#976) is measured on NVDEC."""
    monkeypatch.setattr(analyze, "probe", lambda p: media_info(p, 600.0, **over))
    assert plan() is None
    monkeypatch.setenv(ENV, "hybrid")
    assert plan() is not None


@pytest.mark.parametrize("name", ["a.ts", "a.m2ts", "a.avi", "a.webm"])
def test_auto_splits_only_containers_that_seek_exactly(name):
    assert analyze._hybrid_plan(Path(name), 600.0, 0.0, None, True) is None


@pytest.mark.parametrize("name", ["a.ts", "a.mts", "a.avi"])
def test_a_forced_split_also_needs_a_container_that_seeks_exactly(monkeypatch, name):
    """The CPU half seeks: MPEG-TS seeks inexactly (#976), so it decodes in one pass."""
    monkeypatch.setenv(ENV, "hybrid")
    assert analyze._hybrid_plan(Path(name), 600.0, 0.0, None, True) is None


@pytest.mark.parametrize("name", ["a.mp4", "a.MOV", "a.m4v", "a.mkv"])
def test_auto_splits_mp4_mov_m4v_and_mkv(name):
    assert analyze._hybrid_plan(Path(name), 600.0, 0.0, None, True) is not None


@pytest.mark.parametrize("cpus", [1, 4, 5])
def test_auto_keeps_one_decode_with_too_few_cpus(monkeypatch, cpus):
    monkeypatch.setattr(analyze.os, "cpu_count", lambda: cpus)
    assert plan() is None


@pytest.mark.parametrize("hw", [None, ffmpeg_utils.HwDecoder("vaapi", "/dev/dri/renderD128")])
def test_no_split_without_nvdec(monkeypatch, hw):
    monkeypatch.setattr(analyze, "decoder_for", lambda src: hw)
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
    assert plan(offset=10.0) is None and plan(duration=analyze.LONG_FILE_S + 1) is None


def test_the_split_balances_nvdec_against_the_cpu_threads():
    """On 8 threads (the measured machine) NVDEC takes about 57%; fewer threads, more; always
    on the 0.1 s proxy grid."""
    assert analyze._hybrid_split(1000.0, 8) == 565.6
    assert analyze._hybrid_split(1000.0, 6) == 634.5
    for d in (120.0, 218.3, 599.97):
        s = analyze._hybrid_split(d, 6)
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
    assert analyze._merge_seam(gpu, cpu, SPLIT, CSTART) == want


# ----------------------------------------------------------------- the split decode, faked
@pytest.fixture
def ff(tmp_path, monkeypatch):
    """Fake ffmpeg (writes the proxy) and scan (cuts by half); ``f.fail`` names a failing half."""
    f = type("FF", (), {})()
    f.calls, f.fail, f.seen = [], None, []
    f.cuts = {"gpu": [10.0, 99.95, 101.0], "cpu": [3.0, 50.0]}
    monkeypatch.setattr(analyze, "probe", lambda p: media_info(p, 200.0))
    monkeypatch.setattr(analyze, "STOP_GRACE_S", 0.01)
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

    monkeypatch.setattr(analyze, "run_progress", run_progress)
    monkeypatch.setattr(analyze, "_scan", scan)
    f.cdir = tmp_path / "cdir"
    f.cdir.mkdir()
    f.src = tmp_path / "a.mp4"
    return f


def leftovers(cdir: Path) -> list[str]:
    return sorted(p.name for p in cdir.glob("proxy*"))


def test_the_halves_decode_their_ranges_and_their_cuts_merge(ff):
    cuts = analyze._hybrid_cuts(ff.src, ff.cdir, 200.0, CUDA, SPLIT, 6)
    assert cuts == [10.0, 99.95, 147.0]  # the CPU's 100.0 is the NVDEC 99.95 a frame off
    by = {half: (args, total) for half, args, total in ff.calls}
    gpu, gsecs = by["gpu"]
    assert gpu[gpu.index("-t") + 1] == "103.000" and gsecs == 103.0
    assert "-ss" not in gpu and gpu.index("-hwaccel") < gpu.index("-i")
    cpu, csecs = by["cpu"]
    assert cpu[cpu.index("-ss") + 1] == "97.000" and csecs == 103.0
    assert cpu[cpu.index("-threads") + 1] == "6" and cpu.index("-threads") < cpu.index("-i")
    # the format NVDEC downloads to, before the scale, as in the NVDEC half: the same pixels
    assert gpu[gpu.index("-vf") + 1] == "fps=10:start_time=0,hwdownload,format=nv12,scale=320:-2"
    assert "-hwaccel" not in cpu
    assert cpu[cpu.index("-vf") + 1] == "fps=10:start_time=0,format=nv12,scale=320:-2"
    assert "-t" not in cpu  # to the end
    assert leftovers(ff.cdir) == [] and len(set(ff.seen)) == 2


def test_a_format_nvdec_does_not_download_keeps_the_plain_cpu_chain(ff, monkeypatch):
    monkeypatch.setattr(analyze, "probe", lambda p: media_info(p, 200.0, pix_fmt="yuv444p"))
    analyze._hybrid_cuts(ff.src, ff.cdir, 200.0, CUDA, SPLIT, 6)
    cpu = next(args for half, args, _t in ff.calls if half == "cpu")
    assert cpu[cpu.index("-vf") + 1] == "fps=10:start_time=0,scale=320:-2"


def test_both_halves_of_a_late_video_count_from_the_container_start(ff, monkeypatch):
    """#976: in a container that seeks exactly ffmpeg counts both halves from the container
    start; ``fps`` start_time pads the NVDEC half's late video, the CPU half seeks the split
    point as clips are cut. No ``-copyts``, and downloads stay after ``fps``."""
    monkeypatch.setattr(analyze, "probe", lambda p: media_info(p, 200.0, vstart=0.25))
    assert analyze._hybrid_cuts(ff.src, ff.cdir, 200.0, CUDA, SPLIT, 6) == [10.0, 99.95, 147.0]
    by = {half: args for half, args, _t in ff.calls}
    gpu, cpu = by["gpu"], by["cpu"]
    assert gpu[gpu.index("-vf") + 1] == "fps=10:start_time=0,hwdownload,format=nv12,scale=320:-2"
    assert "-copyts" not in gpu and "-copyts" not in cpu
    assert cpu[cpu.index("-ss") + 1] == "97.000"
    assert cpu[cpu.index("-vf") + 1] == "fps=10:start_time=0,format=nv12,scale=320:-2"


def test_a_split_near_the_start_starts_the_cpu_half_at_0(ff):
    analyze._hybrid_cuts(ff.src, ff.cdir, 200.0, CUDA, 2.0, 6)
    cpu = next(args for half, args, _t in ff.calls if half == "cpu")
    assert cpu[cpu.index("-ss") + 1] == "0.000"


@pytest.mark.parametrize("half", ["gpu", "cpu"])
def test_a_failing_half_gives_none_with_a_warning_and_no_proxies(ff, caplog, half):
    ff.fail = half
    with caplog.at_level("WARNING", logger="cutpoint.analyze"):
        assert analyze._hybrid_cuts(ff.src, ff.cdir, 200.0, CUDA, SPLIT, 6) is None
    assert "the split scene decode failed (Invalid data found); decoding it in one pass" in (
        caplog.text
    )
    assert leftovers(ff.cdir) == []


@pytest.mark.parametrize("half", ["gpu", "cpu"])
def test_scene_cuts_falls_back_to_one_decode_when_a_half_fails(ff, monkeypatch, half):
    ff.fail = half
    single = []

    def make_proxy(src, proxy, args, seconds, cancel=None, gpu=True):
        single.append(gpu)
        write(proxy, "proxy")

    monkeypatch.setattr(analyze, "_make_proxy", make_proxy)
    monkeypatch.setattr(analyze, "_scan", lambda proxy, cancel: [42.0])
    cuts = analyze.scene_cuts(ff.src, ff.cdir, 200.0, cpu_free=True)
    assert cuts == [42.0] and single == [True]
    assert cache.load_json(ff.cdir / analyze.SCENES_FILE) == [42.0] and leftovers(ff.cdir) == []


def test_scene_cuts_saves_the_merged_cuts_as_one_pass_would(ff, monkeypatch):
    monkeypatch.setattr(analyze, "_make_proxy", lambda *a, **k: pytest.fail("one decode"))
    cuts = analyze.scene_cuts(ff.src, ff.cdir, 200.0, cpu_free=True)
    assert cache.load_json(ff.cdir / analyze.SCENES_FILE) == cuts
    assert all(isinstance(c, float) for c in cuts) and cuts == sorted(cuts)
    assert analyze.scene_cuts(ff.src, ff.cdir, 200.0, cpu_free=True) == cuts  # cached
    assert len(ff.calls) == 2


@pytest.mark.parametrize(("mode", "gpu"), [("gpu", True), ("cpu", False), ("auto", True)])
def test_one_decode_without_cpu_free_or_with_the_switch(ff, monkeypatch, mode, gpu):
    monkeypatch.setenv(ENV, mode)
    single = []

    def make_proxy(src, proxy, args, seconds, cancel=None, gpu=True):
        single.append(gpu)
        write(proxy, "proxy")

    monkeypatch.setattr(analyze, "_make_proxy", make_proxy)
    analyze.scene_cuts(ff.src, ff.cdir, 200.0, cpu_free=mode != "auto")
    assert single == [gpu] and ff.calls == []


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

    monkeypatch.setattr(analyze, "run_progress", run_progress)
    real_join = threading.Thread.join

    def join(self, timeout=None):
        if self.name.startswith("scene-") and threading.current_thread() is threading.main_thread():
            started.wait(5)
            monkeypatch.setattr(threading.Thread, "join", real_join)
            raise KeyboardInterrupt
        return real_join(self, timeout)

    monkeypatch.setattr(threading.Thread, "join", join)
    with pytest.raises(KeyboardInterrupt):
        analyze._hybrid_cuts(ff.src, ff.cdir, 200.0, CUDA, SPLIT, 6)
    assert len(stopped) == 2 and leftovers(ff.cdir) == []


def test_a_set_cancel_stops_the_split_and_writes_nothing(ff, monkeypatch):
    """The overlap thread's ``cancel`` (Whisper failed or stopped) reaches both halves."""
    outer = threading.Event()

    def run_progress(args, total, label, cancel=None):
        write(Path(args[-1]), "partial")
        outer.set()
        assert cancel is not outer and cancel.wait(5)  # the split's own stop, set from outer
        raise RunCancelled(label)

    monkeypatch.setattr(analyze, "run_progress", run_progress)
    with pytest.raises(RunCancelled):
        analyze.scene_cuts(ff.src, ff.cdir, 200.0, cancel=outer, cpu_free=True)
    assert leftovers(ff.cdir) == [] and not (ff.cdir / analyze.SCENES_FILE).exists()


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
    keyword arguments of each ``scene_cuts`` call, ``w.jobs`` the ``_SceneJob`` arguments."""
    monkeypatch.delenv("CUTPOINT_ANALYZE_SCENE_OVERLAP", raising=False)
    src = write(tmp_path / "videos" / "a.mp4")
    w = type("World", (), {})()
    w.src, w.kw, w.jobs = src, [], []

    def scene(*a, **k):
        w.kw.append(k)
        return [16.4]

    class Job:
        def __init__(self, *args):
            w.jobs.append(args)
            self.cuts, self.took = [16.4], 0.5

        def result(self):
            return self.cuts

    monkeypatch.setattr(analyze, "scene_cuts", scene)
    monkeypatch.setattr(analyze, "_SceneJob", Job)
    monkeypatch.setattr(analyze, "probe", lambda p: media_info(p, 600.0, audio_streams=1))
    monkeypatch.setattr(analyze, "extract_audio", lambda io, wav, *a, **k: write(wav, "wav"))
    monkeypatch.setattr(analyze, "check_read", lambda *a, **k: None)
    times = np.arange(0.0, 600.0, 0.5)
    monkeypatch.setattr(analyze, "_loudness", lambda wav, cdir: (times, np.full(len(times), -30.0)))
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
    args: list[str] = []
    for name, dur in parts:
        sep = ":" if "=" in name else "="
        args += ["-f", "lavfi", "-i", f"{name}{sep}size=160x120:rate=10:duration={dur}"]
    graph = "".join(f"[{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1[v]"
    out = tmp_path / "cuts.mp4"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", *args, "-filter_complex", graph, "-map", "[v]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "20", str(out)],
        check=True,
    )  # fmt: skip
    return out


@needs_ffmpeg
def test_a_split_anywhere_gives_the_cuts_of_one_pass(cuts_clip, tmp_path, monkeypatch):
    """Both halves on the CPU (no GPU needed): with the split on a cut, a frame either side of
    one and inside the overlap of two close cuts, the merged cuts equal one pass's."""
    monkeypatch.setattr(analyze, "decoder_for", lambda src: None)
    cdir = tmp_path / "cdir"
    cdir.mkdir()
    duration = 24.0
    proxy = cdir / "one.mp4"
    analyze._make_proxy(cuts_clip, proxy, analyze._proxy_args(duration), duration, gpu=False)
    one = analyze._scan(proxy, None)
    proxy.unlink()
    assert one == [4.0, 8.0, 11.0, 11.6, 15.0, 19.0]
    for split in (11.0, 10.9, 11.1, 11.6, 11.3, 15.0, 15.2, 15.3, 15.4, 6.0):
        got = analyze._hybrid_cuts(cuts_clip, cdir, duration, None, split, 2)
        assert got == one, split
    assert leftovers(cdir) == []


@pytest.fixture
def late_and_ts(cuts_clip, tmp_path) -> tuple[Path, Path]:
    """``cuts_clip`` with an audio track and its video 0.25 s late (MP4), and an MPEG-TS copy."""
    ff = shutil.which("ffmpeg")
    late, ts = tmp_path / "late.mp4", tmp_path / "cuts.ts"
    subprocess.run(
        [ff, "-v", "error", "-itsoffset", "0.25", "-i", str(cuts_clip), "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=48000:duration=24", "-map", "0:v", "-map", "1:a",
         "-c:v", "copy", "-c:a", "aac", str(late)],
        check=True,
    )  # fmt: skip
    subprocess.run(
        [ff, "-v", "error", "-i", str(late), "-c", "copy", "-f", "mpegts", str(ts)], check=True
    )
    return late, ts


@needs_ffmpeg
def test_a_late_video_start_or_mpegts_is_not_split_by_auto(cuts_clip, late_and_ts, monkeypatch):
    monkeypatch.setattr(analyze, "probe", ffmpeg_utils.probe)
    late, ts = late_and_ts
    assert ffmpeg_utils.probe(late).vstart > 0.2
    assert analyze._hybrid_plan(cuts_clip, 600.0, 0.0, None, True) is not None  # the control
    assert analyze._hybrid_plan(late, 600.0, 0.0, None, True) is None
    assert analyze._hybrid_plan(ts, 600.0, 0.0, None, True) is None


CLIP_CUTS = [4.0, 8.0, 11.0, 11.6, 15.0, 19.0]  # one pass of cuts_clip (15.3 dropped)


def near(got: list[float], want: list[float]) -> bool:
    """``got`` and ``want`` hold the same cuts within one 0.1 s proxy frame."""
    return len(got) == len(want) and all(
        abs(a - b) <= analyze.SCENE_GRID_S + 1e-6 for a, b in zip(got, want, strict=True)
    )


def one_pass(src: Path, cdir: Path) -> list[float]:
    """The full pass of ``src`` as ``scene_cuts`` makes it (one CPU decode), cached in ``cdir``."""
    cdir.mkdir()
    return analyze.scene_cuts(src, cdir, ffmpeg_utils.probe(src).duration)


@needs_ffmpeg
def test_a_forced_split_of_a_late_video_matches_one_pass(late_and_ts, tmp_path, monkeypatch):
    """Both halves count from the container start, as the full pass does (#976): the split's
    cuts land where one pass's do (within one 0.1 s proxy frame), on any ffmpeg."""
    monkeypatch.setattr(analyze, "probe", ffmpeg_utils.probe)
    monkeypatch.setattr(analyze, "decoder_for", lambda src: None)
    late, _ts = late_and_ts
    duration = ffmpeg_utils.probe(late).duration
    one = one_pass(late, tmp_path / "one")
    assert near(one, [c + 0.25 for c in CLIP_CUTS]), one  # the container clock
    cdir = tmp_path / "late-cdir"
    cdir.mkdir()
    for split in (11.0, 10.9, 11.6, 15.2, 6.0):
        got = analyze._hybrid_cuts(late, cdir, duration, None, split, 2)
        assert near(got, one), (split, got, one)
    assert leftovers(cdir) == []


@needs_ffmpeg
def test_every_scene_pass_counts_from_the_container_start(late_and_ts, tmp_path, monkeypatch):
    """#976 on a video 0.25 s late: the full pass, the edge windows (one at the part start, so
    without a seek) give the cuts 0.25 s after the clip's; the MPEG-TS copy gets its full pass
    instead of windows, on the same clock."""
    monkeypatch.setattr(analyze, "probe", ffmpeg_utils.probe)
    monkeypatch.setattr(analyze, "decoder_for", lambda src: None)
    late, ts = late_and_ts
    want = [c + 0.25 for c in CLIP_CUTS]
    full = one_pass(late, tmp_path / "full")
    assert near(full, want), full
    span = ffmpeg_utils.probe(late).duration
    wdir = tmp_path / "windows"
    wdir.mkdir()
    edges = [2.5, 8.0, 11.5, 15.0, 19.0]  # the first window starts at 0: no seek
    got = analyze.EdgeCuts(late, wdir, span).around(edges)
    assert (wdir / analyze.SCENE_WINDOWS_FILE).exists()
    assert not (wdir / analyze.SCENES_FILE).exists()
    assert near(got, full), (got, full)
    tdir = tmp_path / "ts"
    tdir.mkdir()
    got = analyze.EdgeCuts(ts, tdir, ffmpeg_utils.probe(ts).duration).around(edges)
    assert near(got, want), got
    assert (tdir / analyze.SCENES_FILE).exists()
    assert not (tdir / analyze.SCENE_WINDOWS_FILE).exists()


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
    monkeypatch.setattr(analyze, "probe", ffmpeg_utils.probe)
    monkeypatch.setattr(analyze, "decoder_for", lambda src: None)
    ts, zero = mid_gop_ts
    assert zero < 0  # the clip's start was cut off
    want = [round(c + zero, 3) for c in CLIP_CUTS if c + zero > 0.5]
    got = one_pass(ts, tmp_path / "mid")
    assert near(got, want), (got, want)


@pytest.fixture
def joined_ts(cuts_clip, tmp_path) -> Path:
    """``cuts_clip`` as two 12 s MPEG-TS files with audio joined byte for byte, as camera clips
    or DVR captures are: the second one's timestamps start again where the first one's did."""
    ff = shutil.which("ffmpeg")
    halves = []
    for i, start in enumerate(("0", "12")):
        half = tmp_path / f"half{i}.ts"
        subprocess.run(
            [ff, "-v", "error", "-ss", start, "-t", "12", "-i", str(cuts_clip), "-f", "lavfi",
             "-i", "sine=frequency=440:sample_rate=48000:duration=12", "-map", "0:v", "-map",
             "1:a", "-c:v", "libx264", "-c:a", "aac", "-f", "mpegts", str(half)],
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
    monkeypatch.setattr(analyze, "probe", ffmpeg_utils.probe)
    monkeypatch.setattr(analyze, "decoder_for", lambda src: None)
    vstart = ffmpeg_utils.probe(joined_ts).vstart
    got = one_pass(joined_ts, tmp_path / "joined")
    assert near(got, [c + vstart for c in CLIP_CUTS]), got


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
    monkeypatch.setattr(analyze, "probe", ffmpeg_utils.probe)
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
    shift = analyze._first_shift(ts, 0.0, info.duration)
    assert shift is not None and abs(shift - info.vstart) < 1e-3, (shift, info.vstart)


@needs_ffmpeg
def test_the_first_frame_is_of_the_video_stream_the_pass_decodes(cuts_clip, tmp_path, monkeypatch):
    """Two video streams: the pass decodes the largest (no ``-map``), which starts 2 s after a
    small one, so its cuts and its first frame come from that stream."""
    monkeypatch.setattr(analyze, "probe", ffmpeg_utils.probe)
    monkeypatch.setattr(analyze, "decoder_for", lambda src: None)
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
