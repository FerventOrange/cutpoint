"""#464: parallel clip export (export_workers, _ExportPool, ChildScope).

The pool is tested with fake exports (threads and events), and with small real subprocesses
(``python -c``) standing in for ffmpeg where the point is that children are started, stopped
or killed. Nothing here needs ffmpeg.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import media_info, set_config, write
from cutpoint import clips, config, ffmpeg_utils, paths, sources, srt
from cutpoint.clips import ClipPlan
from cutpoint.encode import OutputFormat, atomic_output
from cutpoint.ffmpeg_utils import ChildScope, FFmpegError, VideoEncoder, child_scope
from cutpoint.tail import TailResult

FMT = OutputFormat(1920, 1080, 30)
WAIT_S = 30  # upper bound of every wait; nothing takes this long unless a test is failing

# A stand-in for ffmpeg: runs until the flag file appears, then exits with code 7
# (a stopped ffmpeg exits non-zero); a killed one never reaches the exit.
CHILD = """\
import os, sys, time
deadline = time.monotonic() + 30
while not os.path.exists(sys.argv[1]) and time.monotonic() < deadline:
    time.sleep(0.02)
sys.exit(7)
"""


def until(condition, what: str = "condition") -> None:
    deadline = time.monotonic() + WAIT_S
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


def plan(i: int, pano: bool = False) -> ClipPlan:
    src = Path("src.mp4")
    tail = TailResult(10.0 + i, 0.0, "none")
    entry = {"id": f"h{i:03d}", "source": str(src), "end": 10.0 + i, "score": 50}
    return ClipPlan(
        f"{i:03d}_c.mp4", entry, src, media_info(src), float(i), 10.0 + i, (float(i), 10.0 + i),
        tail, pano=pano,
    )  # fmt: skip


@pytest.fixture(autouse=True)
def no_transcripts(monkeypatch):
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)


@pytest.fixture
def kids(monkeypatch, tmp_path):
    """Every :func:`ffmpeg_utils.run` starts the CHILD script; ``kids.procs`` are the Popens.

    ``kids.release()`` lets the children exit by themselves; ``kids.killed`` lists the
    ``Popen.kill`` calls.
    """
    flag = tmp_path / "release"
    state = SimpleNamespace(procs=[], killed=[], flag=flag, release=flag.touch)
    real_popen = ffmpeg_utils.subprocess.Popen

    class Spy(real_popen):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            state.procs.append(self)

        def kill(self):
            state.killed.append(self)
            super().kill()

    monkeypatch.setattr(
        ffmpeg_utils, "_command", lambda tool, args: [sys.executable, "-c", CHILD, str(flag)]
    )
    monkeypatch.setattr(ffmpeg_utils.subprocess, "Popen", Spy)
    yield state
    flag.touch()  # no child outlives the test
    for p in state.procs:
        p.wait(WAIT_S)


def ffmpeg() -> None:
    ffmpeg_utils.run(["ffmpeg", "-i", "x"])


def render(plans, tmp_path, workers):
    """Render into ``tmp_path / "out"`` (the stand-in children keep their flag in tmp_path)."""
    (tmp_path / "out").mkdir(exist_ok=True)
    return clips.render_clips(
        plans, tmp_path / "out", fmt=FMT, vertical=False, audio_fade=0.0, workers=workers
    )


def fake_export(monkeypatch, behaviors):
    """``clips.export_clip_checked`` runs ``behaviors[clip name](out)`` (default: write it)."""
    started: list[str] = []

    def export(src, info, start, end, out, *_a, **_kw):
        started.append(out.name)
        fn = behaviors.get(out.name)
        return fn(out) if fn else out.write_bytes(b"new") and None

    monkeypatch.setattr(clips, "export_clip_checked", export)
    return started


# -- order, failure and cancel of the pool ---------------------------------------------------


def test_results_come_back_in_plan_order_when_jobs_finish_out_of_order(
    monkeypatch, tmp_path, caplog
):
    caplog.set_level("INFO")
    done = {n: threading.Event() for n in ("001_c.mp4", "002_c.mp4")}

    def first(out):  # finishes last
        assert done["002_c.mp4"].wait(WAIT_S)
        out.write_bytes(b"new")

    def last(out):  # finishes first
        out.write_bytes(b"new")
        done["002_c.mp4"].set()

    fake_export(monkeypatch, {"000_c.mp4": first, "002_c.mp4": last})
    results = render([plan(0), plan(1), plan(2)], tmp_path, workers=3)
    assert [r.out.name for r in results] == ["000_c.mp4", "001_c.mp4", "002_c.mp4"]
    logged = [m for m in (r.getMessage() for r in caplog.records) if "<- src.mp4" in m]
    assert [m.split()[0] for m in logged] == ["000_c.mp4", "001_c.mp4", "002_c.mp4"]


@pytest.mark.slow
def test_first_failure_is_raised_starts_no_more_and_kills_the_encodes_in_flight(
    monkeypatch, tmp_path, kids
):
    seen: dict[str, BaseException] = {}

    def in_flight(out):
        with atomic_output(out) as part:
            part.write_bytes(b"half")
            try:
                ffmpeg()
            except BaseException as err:
                seen["in_flight"] = err
                raise

    def fails(out):
        until(lambda: kids.procs, "the first encode to start ffmpeg")
        raise FFmpegError("ffmpeg failed (exit 1)")

    started = fake_export(monkeypatch, {"000_c.mp4": in_flight, "001_c.mp4": fails})
    with pytest.raises(FFmpegError, match="exit 1"):
        render([plan(i) for i in range(4)], tmp_path, workers=2)
    assert sorted(started) == ["000_c.mp4", "001_c.mp4"]  # the other two never started
    assert len(kids.procs) == 1
    assert kids.procs[0].poll() not in (None, 7)  # killed, not let go
    # what a stop causes in the other encode is no failure: a stop, so no retry on the CPU
    assert type(seen["in_flight"]) is KeyboardInterrupt
    assert list((tmp_path / "out").iterdir()) == []  # no .part, no clip


@pytest.fixture
def soft_stop(monkeypatch):
    monkeypatch.setattr(clips, "SOFT_STOP_S", 0.3)
    return 0.3


@pytest.fixture
def main_thread_interrupt(monkeypatch):
    """Ctrl+C in the main thread, once the first clip's result is taken (subtitles are
    written there); ``fired`` is set when it was raised."""
    fired = threading.Event()
    interrupt = KeyboardInterrupt()

    def write_srt(self, clip, source, start, end):
        fired.set()
        raise interrupt

    monkeypatch.setattr(srt.ClipSubtitles, "write", write_srt)
    return SimpleNamespace(fired=fired, exc=interrupt)


def interruptible_encodes(monkeypatch, kids):
    """Clip 0 writes at once; clips 1 and 2 run ffmpeg (the CHILD) inside atomic_output."""
    seen: dict[str, BaseException] = {}

    def encode(out):
        with atomic_output(out) as part:
            part.write_bytes(b"half")
            try:
                ffmpeg()
            except BaseException as err:
                seen[out.name] = err
                raise

    fake_export(monkeypatch, {"001_c.mp4": encode, "002_c.mp4": encode})
    return seen


@pytest.mark.slow
def test_soft_interrupt_gives_ffmpeg_the_soft_stop_time_then_kills_it(
    monkeypatch, tmp_path, kids, soft_stop, main_thread_interrupt
):
    seen = interruptible_encodes(monkeypatch, kids)

    def interrupt_when_running(self, clip, source, start, end):
        until(lambda: len(kids.procs) == 2, "both encodes to run ffmpeg")
        raise main_thread_interrupt.exc

    monkeypatch.setattr(srt.ClipSubtitles, "write", interrupt_when_running)
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt) as err:
        render([plan(0), plan(1), plan(2)], tmp_path, workers=3)
    assert err.value is main_thread_interrupt.exc
    assert time.monotonic() - t0 >= soft_stop  # they were given the time first
    assert len(kids.killed) == 2
    assert all(p.poll() not in (None, 7) for p in kids.procs)
    assert {type(e) for e in seen.values()} == {KeyboardInterrupt}
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["000_c.mp4"]  # no .part files


@pytest.mark.slow
def test_soft_interrupt_lets_ffmpeg_that_got_the_ctrl_c_exit_without_a_kill(
    monkeypatch, tmp_path, kids, main_thread_interrupt
):
    """Within SOFT_STOP_S the children exit by themselves (the same Ctrl+C reached them): the
    non-zero exit is the stop, not an ffmpeg failure, and nothing is killed."""
    monkeypatch.setattr(clips, "SOFT_STOP_S", float(WAIT_S))
    seen = interruptible_encodes(monkeypatch, kids)

    def interrupt_when_running(self, clip, source, start, end):
        until(lambda: len(kids.procs) == 2, "both encodes to run ffmpeg")
        threading.Timer(0.1, kids.release).start()  # ffmpeg exits after the interrupt
        raise main_thread_interrupt.exc

    monkeypatch.setattr(srt.ClipSubtitles, "write", interrupt_when_running)
    with pytest.raises(KeyboardInterrupt) as err:
        render([plan(0), plan(1), plan(2)], tmp_path, workers=3)
    assert err.value is main_thread_interrupt.exc
    assert kids.killed == []
    assert [p.returncode for p in kids.procs] == [7, 7]
    assert {type(e) for e in seen.values()} == {KeyboardInterrupt}  # not FFmpegError
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["000_c.mp4"]


@pytest.mark.slow
def test_second_interrupt_kills_ffmpeg_at_once(monkeypatch, tmp_path, kids, main_thread_interrupt):
    monkeypatch.setattr(clips, "SOFT_STOP_S", float(WAIT_S))
    seen = interruptible_encodes(monkeypatch, kids)

    def interrupt_when_running(self, clip, source, start, end):
        until(lambda: len(kids.procs) == 2, "both encodes to run ffmpeg")
        main_thread_interrupt.fired.set()
        raise main_thread_interrupt.exc

    monkeypatch.setattr(srt.ClipSubtitles, "write", interrupt_when_running)
    real_wait = clips.wait
    second = []

    def wait(futures, timeout):
        if main_thread_interrupt.fired.is_set() and not second:
            second.append(1)
            raise KeyboardInterrupt  # Ctrl+C again while waiting for the workers
        return real_wait(futures, timeout)

    monkeypatch.setattr(clips, "wait", wait)
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt) as err:
        render([plan(0), plan(1), plan(2)], tmp_path, workers=3)
    assert err.value is main_thread_interrupt.exc
    assert time.monotonic() - t0 < WAIT_S / 2  # not the soft stop time
    assert len(kids.killed) == 2
    assert {type(e) for e in seen.values()} == {KeyboardInterrupt}
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["000_c.mp4"]


# -- ChildScope and run() --------------------------------------------------------------------


@pytest.mark.slow
def test_run_refuses_to_start_a_child_after_cancel(kids):
    scope = ChildScope()
    scope.cancel()
    with child_scope(scope), pytest.raises(KeyboardInterrupt):
        ffmpeg()
    assert kids.procs == []


@pytest.mark.slow
def test_run_outside_a_scope_is_unchanged(kids):
    kids.release()
    with pytest.raises(FFmpegError, match=r"exit 7"):
        ffmpeg()
    assert kids.killed == []


@pytest.mark.slow
def test_child_that_ends_after_cancel_is_a_stop_so_the_gpu_is_not_blamed(monkeypatch, kids):
    """encode_with_fallback retries FFmpegError on libx264 and marks the GPU encoder failed; a
    cancelled child must raise KeyboardInterrupt instead, so neither happens."""
    nvenc = VideoEncoder("nvenc")
    monkeypatch.setattr(ffmpeg_utils, "video_encoder", lambda: nvenc)
    monkeypatch.setattr(ffmpeg_utils, "_GPU_ENCODE_FAILED", set())
    scope = ChildScope()
    attempts: list[str] = []
    outcome: list[BaseException] = []

    def encode(enc):
        attempts.append(enc.kind)
        ffmpeg()

    def worker():
        with child_scope(scope):
            try:
                ffmpeg_utils.encode_with_fallback(encode, "001_c.mp4")
            except BaseException as err:  # noqa: BLE001
                outcome.append(err)

    t = threading.Thread(target=worker)
    t.start()
    until(lambda: kids.procs, "ffmpeg to start")
    scope.cancel()
    kids.release()  # ffmpeg ends non-zero after the cancel
    t.join(WAIT_S)
    assert [type(e) for e in outcome] == [KeyboardInterrupt]
    assert attempts == ["nvenc"]  # no retry on libx264
    assert not ffmpeg_utils._GPU_ENCODE_FAILED


@pytest.mark.slow
def test_child_that_fails_just_before_the_cancel_is_a_stop_not_an_nvenc_failure(monkeypatch, kids):
    """ffmpeg gets the Ctrl+C with Python and can exit before the main thread cancels the scope
    (#464 F1): the failed exit waits STOP_GRACE_S for the cancel, so no retry on libx264."""
    nvenc = VideoEncoder("nvenc")
    monkeypatch.setattr(ffmpeg_utils, "video_encoder", lambda: nvenc)
    monkeypatch.setattr(ffmpeg_utils, "_GPU_ENCODE_FAILED", set())
    scope = ChildScope()
    attempts: list[str] = []
    outcome: list[BaseException] = []

    def encode(enc):
        attempts.append(enc.kind)
        ffmpeg()

    def worker():
        with child_scope(scope):
            try:
                ffmpeg_utils.encode_with_fallback(encode, "001_c.mp4")
            except BaseException as err:  # noqa: BLE001
                outcome.append(err)

    t = threading.Thread(target=worker)
    t.start()
    until(lambda: kids.procs, "ffmpeg to start")
    kids.release()  # ffmpeg exits non-zero first ...
    until(lambda: kids.procs[0].poll() is not None, "ffmpeg to exit")
    threading.Timer(0.1, scope.cancel).start()  # ... and the cancel follows
    t.join(WAIT_S)
    assert [type(e) for e in outcome] == [KeyboardInterrupt]
    assert attempts == ["nvenc"]
    assert not ffmpeg_utils._GPU_ENCODE_FAILED


@pytest.mark.parametrize("redo_fails", [False, True])
def test_clips_starting_during_a_cpu_redo_use_the_cpu(monkeypatch, redo_fails):
    """A failed GPU encode marks its codec suspect until the CPU redo ends (#464 F6): a clip
    starting meanwhile goes to the CPU. A redo that fails too removes the mark, so the clips
    after it use the GPU again; one that works leaves the codec failed."""
    nvenc = VideoEncoder("nvenc")
    monkeypatch.setattr(ffmpeg_utils, "video_encoder", lambda: nvenc)
    monkeypatch.setattr(ffmpeg_utils, "_GPU_ENCODE_FAILED", set())
    monkeypatch.setattr(ffmpeg_utils, "_GPU_ENCODE_SUSPECT", {})
    in_redo, release = threading.Event(), threading.Event()
    outcome: list[object] = []

    def first(enc):
        if enc.kind != "cpu":
            raise FFmpegError("nvenc failed (exit 1)")
        in_redo.set()
        assert release.wait(WAIT_S)
        if redo_fails:
            raise FFmpegError("bad read (exit 1)")
        return "cpu"

    def worker():
        try:
            outcome.append(ffmpeg_utils.encode_with_fallback(first, "000_c.mp4"))
        except FFmpegError as err:
            outcome.append(err)

    t = threading.Thread(target=worker)
    t.start()
    assert in_redo.wait(WAIT_S)
    assert {nvenc.codec: 1} == ffmpeg_utils._GPU_ENCODE_SUSPECT
    assert ffmpeg_utils.encode_with_fallback(lambda enc: enc.kind, "001_c.mp4") == "cpu"
    release.set()
    t.join(WAIT_S)
    assert not ffmpeg_utils._GPU_ENCODE_SUSPECT
    if redo_fails:
        assert isinstance(outcome[0], FFmpegError)
        assert not ffmpeg_utils._GPU_ENCODE_FAILED
        assert ffmpeg_utils.encode_with_fallback(lambda enc: enc.kind, "002_c.mp4") == "nvenc"
    else:
        assert outcome == ["cpu"]
        assert {nvenc.codec} == ffmpeg_utils._GPU_ENCODE_FAILED
        assert ffmpeg_utils.encode_with_fallback(lambda enc: enc.kind, "002_c.mp4") == "cpu"


@pytest.mark.slow
def test_main_thread_interrupt_is_not_replaced_by_an_earlier_worker_failure(monkeypatch, tmp_path):
    """A worker's FFmpegError is recorded first (ffmpeg exited from the same Ctrl+C), then the
    KeyboardInterrupt reaches the main thread in its result wait: KeyboardInterrupt (exit 130)
    leaves render_clips, not the FFmpegError (#464 F1)."""
    release = threading.Event()
    pools: list[clips._ExportPool] = []

    class Pool(clips._ExportPool):
        def __init__(self, *a, **kw):
            pools.append(self)
            super().__init__(*a, **kw)

    def slow(out):
        assert release.wait(WAIT_S)
        out.write_bytes(b"new")

    def fails(out):
        raise FFmpegError("ffmpeg failed (exit 255)")

    fake_export(monkeypatch, {"000_c.mp4": slow, "001_c.mp4": fails})
    monkeypatch.setattr(clips, "_ExportPool", Pool)
    real_wait = clips.wait
    interrupt = KeyboardInterrupt()
    fired: list[int] = []

    def wait(futures, timeout):
        if not fired:
            fired.append(1)
            until(lambda: pools and pools[0].failure is not None, "the worker failure")
            release.set()
            raise interrupt  # Ctrl+C in the main thread's result wait
        return real_wait(futures, timeout)

    monkeypatch.setattr(clips, "wait", wait)
    with pytest.raises(KeyboardInterrupt) as err:
        render([plan(0), plan(1), plan(2)], tmp_path, workers=2)
    assert err.value is interrupt
    assert isinstance(pools[0].failure, FFmpegError)  # it was recorded first


@pytest.mark.slow
def test_scope_kill_kills_running_children_and_new_ones_do_not_start(kids):
    scope = ChildScope()
    outcome: list[BaseException] = []

    def worker():
        with child_scope(scope):
            try:
                ffmpeg()
            except BaseException as err:  # noqa: BLE001
                outcome.append(err)

    t = threading.Thread(target=worker)
    t.start()
    until(lambda: kids.procs, "ffmpeg to start")
    scope.kill()
    t.join(WAIT_S)
    assert [type(e) for e in outcome] == [KeyboardInterrupt]
    assert kids.procs[0].poll() not in (None, 7)
    with child_scope(scope), pytest.raises(KeyboardInterrupt):
        ffmpeg()
    assert len(kids.procs) == 1


# -- export_workers --------------------------------------------------------------------------


def encoder(monkeypatch, kind: str | None):
    """:func:`clips.video_encoder` answers ``kind``, or fails to choose one for None."""

    def choose():
        if kind is None:
            raise ffmpeg_utils.EncoderUnavailableError("cannot encode")
        return VideoEncoder(kind)

    monkeypatch.setattr(clips, "video_encoder", choose)


PLANS = [SimpleNamespace(pano=False)] * 3


@pytest.mark.parametrize(
    ("kind", "configured", "expected"),
    [
        ("nvenc", "auto", 2),
        ("vaapi", "auto", 2),
        ("amf", "auto", 2),
        ("cpu", "auto", 1),
        ("cpu", 3, 3),  # a number is taken as is, libx264 included
        ("nvenc", 3, 3),
        ("nvenc", 1, 1),
        ("nvenc", 0, 1),  # clamped to 1-3
        ("nvenc", 8, 3),
        ("nvenc", "2", 2),
    ],
)
def test_export_workers_follows_config_and_encoder(home, monkeypatch, kind, configured, expected):
    set_config(home, export_workers=configured)
    encoder(monkeypatch, kind)
    assert clips.export_workers(PLANS) == expected


def test_export_workers_is_one_without_a_second_clip_or_with_a_360_clip(home, monkeypatch):
    set_config(home, export_workers=3)
    encoder(monkeypatch, "nvenc")
    assert clips.export_workers(PLANS[:1]) == 1
    assert clips.export_workers([]) == 1
    assert clips.export_workers([*PLANS, SimpleNamespace(pano=True)]) == 1


def test_export_workers_is_one_when_the_encoder_cannot_be_chosen(home, monkeypatch):
    """The first export raises the encoder's error, as a sequential run does."""
    set_config(home, export_workers=3)
    encoder(monkeypatch, None)
    assert clips.export_workers(PLANS) == 1


def test_export_workers_does_not_probe_the_encoder_when_one_is_certain(home, monkeypatch):
    set_config(home, export_workers=1)

    def probe():
        raise AssertionError("encoder probed")

    monkeypatch.setattr(clips, "video_encoder", probe)
    assert clips.export_workers(PLANS) == 1
    set_config(home, export_workers=3)
    assert clips.export_workers(PLANS[:1]) == 1


# -- config parsing --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("auto", None), ("AUTO", None), (" auto ", None), ("", None), (2, 2), ("3", 3), (0, 1),
     (-4, 1), (99, config.MAX_EXPORT_WORKERS)],
)  # fmt: skip
def test_config_export_workers_values(home, value, expected):
    set_config(home, export_workers=value)
    assert config.export_workers() == expected


def test_config_export_workers_defaults_to_auto(home):
    assert config.export_workers() is None
    assert config.MAX_EXPORT_WORKERS == 3


@pytest.mark.parametrize("value", [True, False, 2.0, 1.5, "1.5", "two", "2 clips", [2], {"a": 1}])
def test_config_export_workers_other_text_is_one_error_naming_the_key(home, value):
    set_config(home, export_workers=value)
    with pytest.raises(ValueError, match="export_workers is") as err:
        config.export_workers()
    assert str(home / "config.json") in str(err.value)
    assert repr(value) in str(err.value)


@pytest.mark.parametrize("value", ["1.5", "two", "True"])
def test_env_export_workers_other_text_is_one_error_naming_the_variable(home, monkeypatch, value):
    monkeypatch.setenv("CUTPOINT_EXPORT_WORKERS", value)
    with pytest.raises(ValueError, match="CUTPOINT_EXPORT_WORKERS is"):
        config.export_workers()


def test_env_export_workers_overrides_the_file_and_is_in_the_cache_key(home, monkeypatch):
    set_config(home, export_workers=1)
    assert config.export_workers() == 1
    monkeypatch.setenv("CUTPOINT_EXPORT_WORKERS", "3")  # no cache_clear: the key changed
    assert config.export_workers() == 3
    monkeypatch.setenv("CUTPOINT_EXPORT_WORKERS", "auto")
    assert config.export_workers() is None
    monkeypatch.delenv("CUTPOINT_EXPORT_WORKERS")
    assert config.export_workers() == 1
    assert paths.env("EXPORT_WORKERS") is None


def test_a_bad_env_value_is_not_cached(home, monkeypatch):
    monkeypatch.setenv("CUTPOINT_EXPORT_WORKERS", "x")
    for _ in range(2):
        with pytest.raises(ValueError, match="CUTPOINT_EXPORT_WORKERS"):
            config.export_workers()


# -- render_clips and export_clips -----------------------------------------------------------


def test_one_worker_takes_the_sequential_path_on_the_calling_thread(monkeypatch, tmp_path):
    def no_pool(*_a, **_kw):
        raise AssertionError("pool used")

    monkeypatch.setattr(clips, "_ExportPool", no_pool)
    threads: list[threading.Thread] = []

    def export(out):
        threads.append(threading.current_thread())
        out.write_bytes(b"x")

    fake_export(monkeypatch, {f"{i:03d}_c.mp4": export for i in range(3)})
    results = render([plan(i) for i in range(3)], tmp_path, workers=1)
    assert [r.out.name for r in results] == ["000_c.mp4", "001_c.mp4", "002_c.mp4"]
    assert set(threads) == {threading.current_thread()}


def test_once_per_source_checks_run_on_the_main_thread_before_any_export_starts(
    monkeypatch, tmp_path
):
    """The log/flat, color and audio track checks run once per source, up front (#464)."""
    main = threading.current_thread()
    events: list[tuple[str, str, threading.Thread]] = []

    def recorder(name):
        def record(src, *_a, **_kw):
            events.append((name, Path(src).name, threading.current_thread()))

        return record

    monkeypatch.setattr(clips, "warn_if_log", recorder("log"))
    monkeypatch.setattr(clips, "bt709_filter", lambda info: recorder("color")(info.path))
    monkeypatch.setattr(clips, "export_tracks", recorder("tracks"))

    def export(out):
        events.append(("export", out.name, threading.current_thread()))
        out.write_bytes(b"x")

    fake_export(monkeypatch, {f"{i:03d}_c.mp4": export for i in range(3)})
    a, b = Path("a.mp4"), Path("b.mp4")
    plans = []
    for i, (src, streams) in enumerate([(a, 1), (b, 2), (a, 1)]):
        p = plan(i)
        p.src = src
        p.info = media_info(src, audio_streams=streams)
        plans.append(p)
    render(plans, tmp_path, workers=2)

    checks = [e for e in events if e[0] != "export"]
    assert sorted((n, s) for n, s, _ in checks) == [
        ("color", "a.mp4"), ("color", "b.mp4"), ("log", "a.mp4"), ("log", "b.mp4"),
        ("tracks", "b.mp4"),
    ]  # fmt: skip
    assert {t for _, _, t in checks} == {main}
    first_export = next(i for i, e in enumerate(events) if e[0] == "export")
    assert all(e[0] != "export" for e in events[: len(checks)])
    assert first_export == len(checks)


def test_a_360_plan_is_never_pooled(monkeypatch, tmp_path):
    """workers > 1 with a 360 plan renders in order with the exporter, on this thread."""
    monkeypatch.setattr(clips, "_ExportPool", lambda *a, **kw: pytest.fail("pool used"))
    threads: list[threading.Thread] = []

    class Pano:
        def export(self, source, start, end, out, **_kw):
            threads.append(threading.current_thread())
            out.write_bytes(b"x")
            return None, {"pano": True}

    fake_export(monkeypatch, {})
    results = clips.render_clips(
        [plan(0), plan(1, pano=True)], tmp_path, fmt=FMT, vertical=False, audio_fade=0.0,
        pano=Pano(), workers=2,
    )  # fmt: skip
    assert [r.pano for r in results] == [None, {"pano": True}]
    assert threads == [threading.current_thread()]


def _segments_project(home: Path, n: int) -> tuple[Path, Path]:
    project = home / "projects" / "p"
    src = write(project / "src.mp4", b"x")
    segs = [
        {"source": str(src), "start": 10.0 * i, "end": 10.0 * i + 4, "label": f"m{i}",
         "speech_aware": False, "smart_tail": False}
        for i in range(n)
    ]  # fmt: skip
    return project, write(project / "seg.json", json.dumps(segs))


def test_manifest_and_names_are_the_same_for_one_and_two_workers(home, monkeypatch):
    n = 4
    project, seg = _segments_project(home, n)
    monkeypatch.setattr(clips, "probe", media_info)
    encoder(monkeypatch, "nvenc")
    live = {"now": 0, "max": 0}
    lock = threading.Lock()
    done = [threading.Event() for _ in range(n)]
    pairs = [False]  # two workers: clip 2k+1 is done before clip 2k

    def export(src, info, start, end, out, *_a, **_kw):
        i = int(out.name[:3]) - 1
        with lock:
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
        # Clips finish in reverse order within each pair: 2 before 1, 4 before 3.
        if i % 2 == 0 and pairs[0]:
            assert done[i + 1].wait(WAIT_S)
        out.write_bytes(f"clip {out.name}".encode())
        done[i].set()
        with lock:
            live["now"] -= 1

    monkeypatch.setattr(clips, "export_clip_checked", export)

    def run(workers: str) -> tuple[list[str], list[dict], dict]:
        monkeypatch.setenv("CUTPOINT_EXPORT_WORKERS", workers)
        pairs[0] = workers == "2"
        for e in done:
            e.clear()
        live.update(now=0, max=0)
        out = clips.export_clips(project, segments=seg, force=True)
        files = sorted(p.name for p in out.iterdir())
        data = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        return files, data["clips"], dict(live)

    files1, manifest1, live1 = run("1")
    files2, manifest2, live2 = run("2")
    assert live1["max"] == 1
    assert live2["max"] == 2
    assert files1 == files2
    assert [c["file"] for c in manifest1] == [c["file"] for c in manifest2]
    assert [c["file"] for c in manifest2] == sorted(c["file"] for c in manifest2)
    assert manifest1 == manifest2
    assert len(manifest2) == n
    out = home / "projects" / "p" / "clips"
    assert [(out / c["file"]).read_bytes() for c in manifest2] == [
        f"clip {c['file']}".encode() for c in manifest2
    ]


def test_failed_parallel_export_leaves_no_clips_folder_or_partials(home, monkeypatch):
    project, seg = _segments_project(home, 3)
    monkeypatch.setattr(clips, "probe", media_info)
    monkeypatch.setenv("CUTPOINT_EXPORT_WORKERS", "2")
    encoder(monkeypatch, "nvenc")

    def export(src, info, start, end, out, *_a, **_kw):
        if out.name.startswith("002"):
            raise FFmpegError("ffmpeg failed (exit 1)")
        out.write_bytes(b"new")

    monkeypatch.setattr(clips, "export_clip_checked", export)
    with pytest.raises(FFmpegError):
        clips.export_clips(project, segments=seg)
    assert not (project / "clips").exists()
    assert sorted(p.name for p in project.iterdir() if p.name.startswith(".")) == []
