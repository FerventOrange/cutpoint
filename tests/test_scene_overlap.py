"""Scene cuts alongside Whisper (#725, ``CUTPOINT_ANALYZE_SCENE_OVERLAP=1``): the env switch, the
same results either way, when no scene thread starts, errors and cancels of either side, and
the ``cancel`` event of ``run_progress`` and ``scene_cuts``. Whisper, ffprobe and the signal
files are faked; the child process of the cancel tests is a real Python sleeper."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

from conftest import StubTranscriber, media_info, needs_ffmpeg, write
from cutpoint import analyze, cache, ffmpeg_utils
from cutpoint.errors import FFmpegError, LowMemoryError
from cutpoint.ffmpeg_utils import RunCancelled
from cutpoint.styles import get_style
from cutpoint.transcribe import transcript_name

ENV = "CUTPOINT_ANALYZE_SCENE_OVERLAP"
WAIT_S = 15  # a broken implementation fails after this long instead of hanging
CUTS = [3.0, 7.5]


def scene_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "scene-overlap" and t.is_alive()]


# ----------------------------------------------------------------- the env switch
@pytest.mark.parametrize(
    ("value", "on"),
    [
        (None, False),
        ("", False),
        ("1", True),
        (" 1 ", True),
        ("0", False),
        ("true", False),
        ("on", False),
        ("2", False),
    ],
)
def test_scene_overlap_is_on_only_for_1(monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert analyze.scene_overlap() is on


# ----------------------------------------------------------------- analyze_source with fakes
class FakeTr(StubTranscriber):
    """Whisper stand-in: ``hook`` runs inside ``transcribe`` (to wait, or to fail)."""

    model_name = "stub"
    hook = None
    calls = 0

    def load(self):
        pass

    def transcribe(self, wav, on_progress=None):
        type(self).calls += 1
        if type(self).hook is not None:
            type(self).hook()
        return super().transcribe(wav)


@pytest.fixture
def world(home, tmp_path, monkeypatch):
    """Everything around ``analyze_source`` faked but the cache folder and the code under test.
    ``w.scene`` is the fake ``scene_cuts``: it records its thread and saves ``scenes.json``."""
    monkeypatch.delenv(ENV, raising=False)
    src = write(tmp_path / "videos" / "a.mp4")
    FakeTr.hook, FakeTr.calls = None, 0
    w = type("World", (), {})()
    w.src, w.tr, w.proj = src, FakeTr(), tmp_path / "proj"
    w.cdir = cache.source_dir(src, 0.0)
    w.scene_calls, w.scene_threads, w.scene_error = [], [], None
    w.started = threading.Event()  # the fake scene work has begun

    def scene(src_, cdir, duration, offset=0.0, limit=None, shared=None, cancel=None):
        w.scene_calls.append(cancel)
        w.scene_threads.append(threading.current_thread())
        w.started.set()
        if w.scene_error is not None:
            raise w.scene_error
        cache.save_json(cdir / "scenes.json", CUTS)
        return list(CUTS)

    w.scene = scene
    w.real_scene_cuts = analyze.scene_cuts
    monkeypatch.setattr(analyze, "probe", lambda p: media_info(p, 60.0, audio_streams=1))
    monkeypatch.setattr(analyze, "Transcriber", FakeTr)
    monkeypatch.setattr(analyze, "extract_audio", lambda io, wav, *a, **k: write(wav, "wav"))
    monkeypatch.setattr(analyze, "check_read", lambda *a, **k: None)
    times = np.arange(0.0, 60.0, 0.5)
    lufs = -50.0 + 35.0 * np.exp(-(((times - 20.0) / 3.0) ** 2))  # one loud burst at 20 s
    monkeypatch.setattr(analyze, "_loudness", lambda wav, cdir: (times, lufs))
    monkeypatch.setattr(analyze, "scene_cuts", scene)
    w.source = lambda use_scenes=True, timings=None: analyze.analyze_source(
        src, get_style("funny"), w.tr, use_scenes, None, {} if timings is None else timings
    )
    w.source_with = lambda **kw: analyze.analyze_source(
        src, get_style("funny"), w.tr, True, None, {}, **kw
    )
    w.run = lambda: analyze.analyze(
        src, get_style("funny"), w.proj, vision=False, use_scenes=True, layers=False
    )
    yield w
    for t in scene_threads():  # never leave a thread behind for the next test
        t.join(WAIT_S)


def cache_files(cdir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(cdir.iterdir()) if p.is_file()}


def highlights(proj: Path) -> dict:
    return json.loads((proj / "highlights.json").read_text(encoding="utf-8"))


def test_the_results_are_the_same_with_the_overlap_on_and_off(world, monkeypatch):
    """highlights.json (apart from ``timings``) and the cache files do not depend on the switch."""
    runs = {}
    for flag in ("off", "on"):
        if flag == "on":
            monkeypatch.setenv(ENV, "1")
        shutil.rmtree(world.cdir, ignore_errors=True)
        world.cdir.mkdir(parents=True)
        world.run()
        hl = highlights(world.proj)
        runs[flag] = (hl.pop("timings"), hl, cache_files(world.cdir))
    t_off, hl_off, files_off = runs["off"]
    t_on, hl_on, files_on = runs["on"]
    assert hl_on["candidates"] and CUTS  # something was compared
    assert hl_on == hl_off
    assert files_on == files_off and "scenes.json" in files_on
    assert [s.get("overlap") for s in t_on["sources"]] == [True]
    assert all("overlap" not in s for s in t_off["sources"])
    assert all(isinstance(t_on["sources"][0][k], float) for k in analyze.SOURCE_STAGES)


def test_the_scene_cuts_run_while_whisper_transcribes(world, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    seen = {}

    def whisper():  # returns only once the scene work has begun, in another thread
        seen["scene_began"] = world.started.wait(WAIT_S)

    FakeTr.hook = whisper
    res = world.source()
    assert seen == {"scene_began": True}
    assert len(world.scene_threads) == 1
    assert world.scene_threads[0] is not threading.current_thread()
    assert res.stages["overlap"] is True and res.signals.cuts == CUTS
    assert len(world.scene_calls) == 1 and world.scene_calls[0] is not None  # got a cancel event
    assert not scene_threads()


def test_the_serial_path_starts_no_thread_and_passes_no_cancel(world):
    res = world.source()
    assert world.scene_threads == [threading.current_thread()] and world.scene_calls == [None]
    assert "overlap" not in res.stages and res.signals.cuts == CUTS


def test_the_overlap_time_is_the_scene_threads_own(world, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    timings: dict = {}
    res = world.source(timings=timings)
    assert res.stages["scenes_s"] >= 0 and timings["scenes_s"] == res.stages["scenes_s"]


@pytest.mark.parametrize("case", ["scenes cached", "transcript cached", "no scenes in the style"])
def test_no_scene_thread_when_it_is_not_needed(world, monkeypatch, case):
    monkeypatch.setenv(ENV, "1")
    started = []
    monkeypatch.setattr(analyze, "_SceneJob", lambda *a, **k: started.append(a))
    use_scenes = case != "no scenes in the style"
    if case == "scenes cached":
        cache.save_json(world.cdir / "scenes.json", CUTS)
    if case == "transcript cached":
        cache.save_json(world.cdir / transcript_name("stub", None), [])
    # scene_cuts runs in its turn, in the calling thread, exactly as with the switch off
    res = world.source(use_scenes=use_scenes)
    assert started == [] and "overlap" not in res.stages
    assert world.scene_threads == ([threading.current_thread()] if use_scenes else [])
    assert FakeTr.calls == (0 if case == "transcript cached" else 1)


def test_no_scene_thread_for_a_style_without_scenes_through_analyze(world, monkeypatch):
    """``analyze`` ANDs the style's ``use_scenes`` into the flag; here the caller's is False."""
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(analyze, "_SceneJob", lambda *a, **k: pytest.fail("thread started"))
    analyze.analyze(
        world.src, get_style("funny"), world.proj, vision=False, use_scenes=False, layers=False
    )
    (rec,) = highlights(world.proj)["timings"]["sources"]
    assert "overlap" not in rec and world.scene_calls == []


# ----------------------------------------------------------------- errors
@pytest.mark.parametrize("flag", [None, "1"], ids=["serial", "overlap"])
def test_a_scene_error_is_raised_after_the_transcription_with_its_type(world, monkeypatch, flag):
    if flag:
        monkeypatch.setenv(ENV, flag)
    world.scene_error = ValueError("scene boom")
    with pytest.raises(ValueError, match="scene boom"):
        world.source()
    assert FakeTr.calls == 1  # Whisper finished first (serial: it ran before the scenes)
    assert any(world.cdir.glob("transcript-*.json"))  # and its transcript is cached
    assert not (world.cdir / "scenes.json").exists()
    assert not scene_threads()


def test_a_scene_error_is_raised_in_the_main_thread(world, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    world.scene_error = RuntimeError("scene boom")
    FakeTr.hook = lambda: world.started.wait(WAIT_S)  # the error happens during Whisper
    with pytest.raises(RuntimeError, match="scene boom"):
        world.source()
    assert world.scene_threads[0] is not threading.current_thread()


# the real scene_cuts, with a fake ffmpeg
@pytest.fixture
def real_scenes(world, monkeypatch):
    monkeypatch.setattr(analyze, "scene_cuts", world.real_scene_cuts)
    monkeypatch.setattr(analyze, "decoder_for", lambda src: None)
    monkeypatch.setattr(analyze.importlib, "import_module", lambda name: None)
    return world


def fake_ffmpeg(monkeypatch, *, finish=False):
    """``analyze.run_progress`` that writes a partial proxy, then waits for ``cancel``."""
    seen = type("Seen", (), {})()
    seen.cancel, seen.began, seen.proxy = None, threading.Event(), None

    def run_progress(args, total, label, cancel=None):
        seen.cancel, seen.proxy = cancel, Path(args[-1])
        write(seen.proxy, "partial")
        seen.began.set()
        if finish:
            return ffmpeg_utils.ProgressResult(stderr="", out_time=total)
        if cancel is None or not cancel.wait(WAIT_S):
            raise AssertionError("never cancelled")
        raise RunCancelled(label)

    monkeypatch.setattr(analyze, "run_progress", run_progress)
    return seen


@pytest.mark.parametrize(
    "error",
    [RuntimeError("whisper blew up"), KeyboardInterrupt(), LowMemoryError("1 GB free, 4 needed")],
    ids=["error", "interrupt", "low_memory"],
)
def test_a_failed_or_stopped_transcription_stops_the_scene_job(real_scenes, monkeypatch, error):
    world = real_scenes
    monkeypatch.setenv(ENV, "1")
    ffmpeg = fake_ffmpeg(monkeypatch)

    def whisper():
        assert ffmpeg.began.wait(WAIT_S)  # the scene job is busy when Whisper fails
        raise error

    FakeTr.hook = whisper
    with pytest.raises(type(error)) as err:
        world.source()
    assert err.value is error  # unchanged, not wrapped nor replaced by RunCancelled
    assert ffmpeg.cancel is not None and ffmpeg.cancel.is_set()
    assert not scene_threads()  # stop() waited for it
    assert not (world.cdir / "scenes.json").exists()
    assert not ffmpeg.proxy.exists()  # the partial proxy is gone
    assert not any(world.cdir.glob("transcript-*.json"))


def test_an_error_after_the_transcription_also_stops_the_job(real_scenes, monkeypatch):
    """Whisper is done but ``on_transcript`` raises: the scene job is still stopped."""
    world = real_scenes
    monkeypatch.setenv(ENV, "1")
    ffmpeg = fake_ffmpeg(monkeypatch)

    def boom():
        assert ffmpeg.began.wait(WAIT_S)
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        analyze.analyze_source(
            world.src, get_style("funny"), world.tr, True, None, {}, on_transcript=boom
        )
    assert ffmpeg.cancel.is_set() and not scene_threads()
    assert not (world.cdir / "scenes.json").exists()


def test_scene_cuts_with_a_cancel_removes_the_proxy_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "decoder_for", lambda src: None)
    monkeypatch.setattr(analyze.importlib, "import_module", lambda name: None)
    ffmpeg = fake_ffmpeg(monkeypatch)
    cancel = threading.Event()
    threading.Timer(0.1, cancel.set).start()
    with pytest.raises(RunCancelled):
        analyze.scene_cuts(tmp_path / "s.mp4", tmp_path, 60.0, cancel=cancel)
    assert ffmpeg.cancel is cancel
    assert not (tmp_path / "proxy.mp4").exists() and not (tmp_path / "scenes.json").exists()


def test_scene_cuts_cancelled_just_as_ffmpeg_finishes_still_stops(tmp_path, monkeypatch):
    """ffmpeg ended normally but the event was set meanwhile: no detection on a stopped run."""
    monkeypatch.setattr(analyze, "decoder_for", lambda src: None)
    monkeypatch.setattr(analyze.importlib, "import_module", lambda name: None)
    fake_ffmpeg(monkeypatch, finish=True)
    detected = []
    monkeypatch.setattr(analyze, "_detect", lambda proxy, path: detected.append(path) or [])
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RunCancelled):
        analyze.scene_cuts(tmp_path / "s.mp4", tmp_path, 60.0, cancel=cancel)
    assert detected == [] and not (tmp_path / "proxy.mp4").exists()


def test_scene_cuts_without_a_cancel_passes_none_and_detects(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "decoder_for", lambda src: None)
    monkeypatch.setattr(analyze.importlib, "import_module", lambda name: None)
    given = {}

    def run_progress(args, total, label, **kw):
        given.update(kw)
        write(Path(args[-1]))

    monkeypatch.setattr(analyze, "run_progress", run_progress)
    monkeypatch.setattr(analyze, "_detect", lambda proxy, path: CUTS)
    assert analyze.scene_cuts(tmp_path / "s.mp4", tmp_path, 60.0) == CUTS
    assert given == {}  # the plain call, as before the overlap existed


# ----------------------------------------------------------------- review fixes
def test_whisper_is_loaded_before_the_scene_thread_starts(world, monkeypatch):
    """F1: ``tr.load()`` (the memory guard) runs first, so the scene job never takes VRAM first."""
    monkeypatch.setenv(ENV, "1")
    order = []
    monkeypatch.setattr(FakeTr, "load", lambda self: order.append("load"))
    real_job = analyze._SceneJob

    def job(*a, **k):
        order.append("job")
        return real_job(*a, **k)

    monkeypatch.setattr(analyze, "_SceneJob", job)
    world.source()
    assert order == ["load", "job"]


def test_a_low_memory_load_starts_no_scene_thread(world, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    err = LowMemoryError("1 GB free, 4 needed")

    def load(self):
        raise err

    monkeypatch.setattr(FakeTr, "load", load)
    monkeypatch.setattr(analyze, "_SceneJob", lambda *a, **k: pytest.fail("thread started"))
    with pytest.raises(LowMemoryError) as caught:
        world.source()
    assert caught.value is err
    assert world.scene_calls == [] and FakeTr.calls == 0 and not scene_threads()
    assert not (world.cdir / "scenes.json").exists()


def test_a_shared_decode_uses_the_serial_path(world, monkeypatch):
    """F2: a scene share (shared-decode mode) replaces the decode; no overlap thread then."""
    monkeypatch.setenv(ENV, "1")
    share = types.SimpleNamespace(mode="split", summary=lambda: "stub")
    monkeypatch.setattr(analyze, "_SceneJob", lambda *a, **k: pytest.fail("thread started"))
    seen = []
    scene = analyze.scene_cuts
    monkeypatch.setattr(
        analyze, "scene_cuts", lambda *a, **k: seen.append(k.get("shared")) or scene(*a, **k)
    )
    res = world.source_with(read_checked=lambda info, rec: share)
    assert seen == [share]
    assert world.scene_threads == [threading.current_thread()] and world.scene_calls == [None]
    assert "overlap" not in res.stages and res.signals.cuts == CUTS


def hw_ffmpeg(monkeypatch, *, cancels, calls):
    """For the real ``scene_cuts``: the GPU run fails with FFmpegError (and sets the event when
    ``cancels``); the CPU run succeeds. ``calls`` records which ran."""
    monkeypatch.setattr(analyze, "decoder_for", lambda src: ffmpeg_utils.HwDecoder("cuda"))
    monkeypatch.setattr(analyze.importlib, "import_module", lambda name: None)
    monkeypatch.setattr(analyze, "STOP_GRACE_S", 0.01)
    monkeypatch.setattr(analyze, "_detect", lambda proxy, path, **k: CUTS)

    def run_progress(args, total, label, cancel=None):
        gpu = "-hwaccel" in args
        calls.append("gpu" if gpu else "cpu")
        write(Path(args[-1]), "partial")
        if gpu:
            if cancels:
                cancel.set()
            raise FFmpegError("killed")
        return ffmpeg_utils.ProgressResult(stderr="", out_time=total)

    monkeypatch.setattr(analyze, "run_progress", run_progress)


def test_an_ffmpeg_failure_while_cancelled_does_not_retry_on_the_cpu(tmp_path, monkeypatch):
    """F3: the GPU run died because of the cancel (Ctrl+C reached ffmpeg): no CPU retry."""
    calls = []
    hw_ffmpeg(monkeypatch, cancels=True, calls=calls)
    with pytest.raises(RunCancelled):
        analyze.scene_cuts(tmp_path / "s.mp4", tmp_path, 60.0, cancel=threading.Event())
    assert calls == ["gpu"]
    assert not (tmp_path / "proxy.mp4").exists() and not (tmp_path / "scenes.json").exists()


@pytest.mark.parametrize("with_cancel", [False, True], ids=["no-event", "unset-event"])
def test_an_ffmpeg_failure_without_a_cancel_still_retries_on_the_cpu(
    tmp_path, monkeypatch, with_cancel
):
    calls = []
    hw_ffmpeg(monkeypatch, cancels=False, calls=calls)
    cancel = threading.Event() if with_cancel else None
    assert analyze.scene_cuts(tmp_path / "s.mp4", tmp_path, 60.0, cancel=cancel) == CUTS
    assert calls == ["gpu", "cpu"]


@pytest.fixture
def cut_clip(tmp_path) -> Path:
    """A 4 s 64x48 clip with one hard cut at 2 s (a test pattern, then flat white)."""
    out = tmp_path / "cut.mp4"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
         "testsrc2=size=64x48:rate=10:duration=2", "-f", "lavfi", "-i",
         "color=c=white:size=64x48:rate=10:duration=2", "-filter_complex",
         "[0][1]concat=n=2:v=1[v]", "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(out)],
        check=True,
    )  # fmt: skip
    return out


def proxy_of(clip: Path, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy(clip, folder / "proxy.mp4"))


@needs_ffmpeg
def test_detect_without_a_cancel_writes_scenes_json_and_removes_the_proxy(cut_clip, tmp_path):
    proxy = proxy_of(cut_clip, tmp_path / "cdir")
    path = proxy.parent / "scenes.json"
    cuts = analyze._detect(proxy, path)
    assert cuts and all(isinstance(c, float) for c in cuts)
    assert cache.load_json(path) == cuts and not proxy.exists()


@needs_ffmpeg
def test_detect_stops_through_the_real_scenedetect_api_once_cancelled(cut_clip, tmp_path):
    """F4: a set event stops the real SceneManager; nothing is written, the proxy is gone."""
    proxy = proxy_of(cut_clip, tmp_path / "cdir")
    path = proxy.parent / "scenes.json"
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RunCancelled):
        analyze._detect(proxy, path, cancel)
    assert not path.exists() and not proxy.exists()


@needs_ffmpeg
def test_an_unset_cancel_detects_the_same_cuts_as_none(cut_clip, tmp_path):
    plain = analyze._detect(proxy_of(cut_clip, tmp_path / "a"), tmp_path / "a" / "s.json")
    with_event = analyze._detect(
        proxy_of(cut_clip, tmp_path / "b"), tmp_path / "b" / "s.json", threading.Event()
    )
    assert plain == with_event


def scene_cancel_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "scene-cancel" and t.is_alive()]


def fake_scenedetect(monkeypatch, state, *, cancel_at=None, cancel=None, error=None):
    """A scenedetect whose detect_scenes loops over frames until stop(); it never calls a callback
    and, like scenedetect 0.7.2, clears its stop flag when it starts."""

    class Manager:
        auto_downscale = True

        def add_detector(self, detector):
            pass

        def stop(self):
            state["stops"] += 1
            state["stopped"] = True

        def detect_scenes(self, video, show_progress=False, **kwargs):
            assert "callback" not in kwargs
            state["stopped"] = False
            for i in range(300):  # bounded: 3 s at most
                state["frames"] += 1
                if i == cancel_at:
                    cancel.set()
                if error and i == 1:
                    raise error
                if state["stopped"]:
                    return
                time.sleep(0.01)

        def get_scene_list(self):
            return []

    fake = types.SimpleNamespace(
        AdaptiveDetector=lambda **k: None, SceneManager=Manager, open_video=lambda p: object()
    )
    monkeypatch.setitem(sys.modules, "scenedetect", fake)


def test_a_cancel_set_during_the_scan_stops_the_detection(tmp_path, monkeypatch):
    """F6: the watcher thread stops the manager (repeatedly, as detect_scenes clears the flag)."""
    state = {"stopped": False, "frames": 0, "stops": 0}
    cancel = threading.Event()
    fake_scenedetect(monkeypatch, state, cancel_at=3, cancel=cancel)
    proxy = write(tmp_path / "proxy.mp4")
    with pytest.raises(RunCancelled):
        analyze._detect(proxy, tmp_path / "scenes.json", cancel)
    assert state["stops"] >= 1 and state["stopped"]
    assert state["frames"] < 300  # stopped, not run to the end
    assert not proxy.exists() and not (tmp_path / "scenes.json").exists()


def test_a_stop_lost_before_the_scan_starts_is_repeated(tmp_path, monkeypatch):
    """A cancel set before detect_scenes clears its flag still stops it (the watcher repeats)."""
    state = {"stopped": False, "frames": 0, "stops": 0}
    cancel = threading.Event()
    cancel.set()
    fake_scenedetect(monkeypatch, state)
    with pytest.raises(RunCancelled):
        analyze._detect(write(tmp_path / "proxy.mp4"), tmp_path / "scenes.json", cancel)
    assert state["frames"] < 300 and not (tmp_path / "scenes.json").exists()


@pytest.mark.parametrize("outcome", ["returns", "cancelled", "raises"])
def test_the_cancel_watcher_thread_is_gone_after_the_scan(tmp_path, monkeypatch, outcome):
    state = {"stopped": False, "frames": 0, "stops": 0}
    cancel = threading.Event()
    error = RuntimeError("boom") if outcome == "raises" else None
    fake_scenedetect(
        monkeypatch, state, cancel_at=2 if outcome == "cancelled" else None, cancel=cancel,
        error=error,
    )  # fmt: skip
    state["frames"] = 0
    if outcome == "returns":
        # no cancel: detect_scenes runs its 300 frames and returns
        monkeypatch.setattr(time, "sleep", lambda s: None)
    proxy = write(tmp_path / "proxy.mp4")
    expected = {"returns": None, "cancelled": RunCancelled, "raises": RuntimeError}[outcome]
    if expected is None:
        analyze._scan(proxy, cancel)
    else:
        with pytest.raises(expected):
            analyze._scan(proxy, cancel)
    assert scene_cancel_threads() == []


@needs_ffmpeg
def test_a_cancel_stops_a_real_scan_long_before_it_would_finish(tmp_path):
    """The real scenedetect (no per-frame callback at a cut-free clip) stops within the poll time."""
    clip = tmp_path / "gray.mp4"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
         "color=c=gray:s=320x180:r=30:d=120", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(clip)],
        check=True,
    )  # fmt: skip
    full_dir = tmp_path / "full"
    start = time.monotonic()
    analyze._detect(proxy_of(clip, full_dir), full_dir / "scenes.json")
    full_s = time.monotonic() - start

    proxy = proxy_of(clip, tmp_path / "cdir")
    path = proxy.parent / "scenes.json"
    ev = threading.Event()
    timer = threading.Timer(0.2, ev.set)
    timer.start()
    start = time.monotonic()
    try:
        with pytest.raises(RunCancelled):
            analyze._detect(proxy, path, ev)
    finally:
        timer.cancel()
    cancelled_s = time.monotonic() - start
    assert cancelled_s < max(full_s * 0.6, 1.0), (cancelled_s, full_s)
    assert not proxy.exists() and not path.exists()
    assert scene_cancel_threads() == []


def test_an_interrupt_inside_the_detection_removes_the_proxy(tmp_path, monkeypatch):
    def scan(proxy, cancel):
        raise KeyboardInterrupt

    monkeypatch.setattr(analyze, "_scan", scan)
    proxy = write(tmp_path / "proxy.mp4")
    with pytest.raises(KeyboardInterrupt):
        analyze._detect(proxy, tmp_path / "scenes.json")
    assert not proxy.exists() and not (tmp_path / "scenes.json").exists()


# ----------------------------------------------------------------- run_progress cancel
SLEEPER = [sys.executable, "-c", "import time; time.sleep(120)"]


@pytest.fixture
def sleeper(monkeypatch):
    """``run_progress`` runs a Python sleeper instead of ffmpeg; the list holds its Popens."""
    started = []
    real = ffmpeg_utils.subprocess.Popen

    def popen(*a, **k):
        started.append(real(*a, **k))
        return started[-1]

    monkeypatch.setattr(ffmpeg_utils, "_command", lambda tool, args: SLEEPER)
    monkeypatch.setattr(ffmpeg_utils.subprocess, "Popen", popen)
    yield started
    for p in started:  # only the children this fixture started
        if p.poll() is None:
            p.kill()
            p.wait(WAIT_S)


@pytest.mark.slow
def test_run_progress_cancel_kills_ffmpeg_and_raises(sleeper):
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with pytest.raises(RunCancelled):
        ffmpeg_utils.run_progress(["ffmpeg", "out.mp4"], 60.0, "scene proxy", cancel=cancel)
    assert time.monotonic() - t0 < WAIT_S  # soon after the event, not after the 120 s sleep
    (proc,) = sleeper
    assert proc.poll() is not None  # dead already: stop_child waited for it


@pytest.mark.slow
def test_run_progress_cancel_works_from_a_background_thread(sleeper):
    """As the scene job calls it: not the main thread."""
    cancel, out = threading.Event(), {}

    def work():
        try:
            ffmpeg_utils.run_progress(["ffmpeg", "o.mp4"], 60.0, "t", cancel=cancel)
        except BaseException as exc:  # noqa: BLE001
            out["exc"] = exc

    t = threading.Thread(target=work, daemon=True)
    t.start()
    deadline = time.monotonic() + WAIT_S
    while not sleeper and time.monotonic() < deadline:
        time.sleep(0.02)
    cancel.set()
    t.join(WAIT_S)
    assert not t.is_alive() and isinstance(out["exc"], RunCancelled)
    assert sleeper[0].poll() is not None


@pytest.mark.slow
def test_run_progress_with_an_unset_cancel_returns_when_the_child_exits(monkeypatch):
    monkeypatch.setattr(ffmpeg_utils, "_command", lambda tool, args: [sys.executable, "-c", "pass"])
    cancel = threading.Event()
    res = ffmpeg_utils.run_progress(["ffmpeg", "o.mp4"], 1.0, "t", cancel=cancel)
    assert res.out_time == 0.0 and not cancel.is_set()


@pytest.mark.slow
def test_run_progress_with_a_cancel_still_reports_a_failing_child(monkeypatch):
    monkeypatch.setattr(
        ffmpeg_utils, "_command", lambda tool, args: [sys.executable, "-c", "raise SystemExit(3)"]
    )
    with pytest.raises(ffmpeg_utils.FFmpegError):
        ffmpeg_utils.run_progress(["ffmpeg", "o.mp4"], 1.0, "t", cancel=threading.Event())


@pytest.mark.slow
def test_a_stopped_transcription_kills_the_real_child_of_the_scene_job(
    real_scenes, sleeper, monkeypatch
):
    """The whole chain without a fake ffmpeg: Whisper fails, the sleeper child dies."""
    world = real_scenes
    monkeypatch.setenv(ENV, "1")
    boom = RuntimeError("whisper blew up")

    def whisper():
        deadline = time.monotonic() + WAIT_S
        while not sleeper and time.monotonic() < deadline:
            time.sleep(0.02)
        assert sleeper
        raise boom

    FakeTr.hook = whisper
    with pytest.raises(RuntimeError) as err:
        world.source()
    assert err.value is boom
    assert sleeper[0].poll() is not None and not scene_threads()
    assert not (world.cdir / "scenes.json").exists() and not (world.cdir / "proxy.mp4").exists()
