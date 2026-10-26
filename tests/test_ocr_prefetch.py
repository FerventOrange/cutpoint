"""Background HUD OCR during ``analyze`` (#375): results taken in their turn, logs replayed in
order, progress forwarded, errors re-raised, the off switches, and ``decode_raw(isolate=True)``
(fake OCR reads, invented game ``zonewar``)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import needs_ffmpeg
from cutpoint import analyze, ffmpeg_utils, progress
from cutpoint.events import ocr_prefetch, pipeline
from cutpoint.events.common import Source
from cutpoint.events.killfeed import OcrCancelled, OcrParams
from cutpoint.styles import get_style


class _Stop(Exception):
    pass


class _Records(logging.Handler):
    """Every message the root logger's handlers get, in order."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@pytest.fixture
def records():
    root = logging.getLogger()
    handler, level = _Records(), root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    yield handler
    root.removeHandler(handler)
    root.setLevel(level)


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """OcrPrefetch with fake sources, presets and OCR reads. ``state.read(src, cancel)``
    is what a read does (default: two log lines, four progress shares, one event)."""
    state = SimpleNamespace(reads=[], preset=True, vram=True, read=None, gate=None)

    def load_source(path, duration):
        return Source(Path(path), duration, tmp_path / "cache" / Path(path).stem, [])

    def detect(src, params, engine, *, on_progress=None, make_engine=None, cancel=None, **kw):
        assert engine is None and not any(kw.values())  # the plain call run_events makes
        state.reads.append(src.name)
        if state.gate is not None:
            state.gate.wait(5)
        if state.read is not None:
            return state.read(src, cancel)
        log = logging.getLogger("cutpoint.events.killfeed")
        log.info("%s: read 1", src.name)
        for share in (0.25, 0.5, 0.75, 1.0):
            if on_progress is not None:
                on_progress(share)
        log.info("%s: read 2", src.name)
        return [{"t": 1.0, "type": "kill", "src": src.name}], {"frames": 4}

    monkeypatch.setattr(ocr_prefetch, "load_source", load_source)
    monkeypatch.setattr(ocr_prefetch, "detect_ocr_events", detect)
    monkeypatch.setattr(ocr_prefetch, "load_preset", lambda game, project: state.preset or None)
    monkeypatch.setattr(ocr_prefetch, "vram_ok", lambda: state.vram)
    paths = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    state.paths = paths
    state.project = tmp_path / "proj"
    state.source = lambda i: load_source(paths[i], 60.0)
    state.params = lambda: OcrParams(game="zonewar", project=state.project)

    def make() -> ocr_prefetch.OcrPrefetch:
        pre = ocr_prefetch.OcrPrefetch(state.project, paths, OcrParams(game="zonewar"))
        state.pre = pre
        return pre

    state.make = make
    yield state
    if getattr(state, "pre", None) is not None:
        state.pre.close()


# ----------------------------------------------------------------- taking results
def test_results_are_taken_in_turn_with_their_progress_and_logs(fake, records) -> None:
    pre = fake.make()
    log = logging.getLogger("cutpoint.analyze")
    log.info("before")
    pre.offer(fake.paths[0], 60.0, True)
    pre.offer(fake.paths[1], 60.0, True)
    deadline = time.monotonic() + 5
    while fake.reads != ["a.mp4", "b.mp4"] and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)
    log.info("between")  # the main thread goes on while the worker reads
    shares: list[float] = []
    evs, st = pre.take(fake.source(0), fake.params(), shares.append)
    assert evs == [{"t": 1.0, "type": "kill", "src": "a.mp4"}] and st == {"frames": 4}
    assert shares == [0.25, 0.5, 0.75, 1.0]
    log.info("after a")
    assert pre.take(fake.source(1), fake.params(), None)[0][0]["src"] == "b.mp4"
    # the worker's lines come where a serial run logs them: when its result is taken
    assert records.lines == [
        "before",
        "between",
        "a.mp4: read 1",
        "a.mp4: read 2",
        "after a",
        "b.mp4: read 1",
        "b.mp4: read 2",
    ]


def test_take_waits_for_the_read(fake) -> None:
    fake.gate = threading.Event()
    pre = fake.make()
    pre.offer(fake.paths[0], 60.0, True)
    threading.Timer(0.3, fake.gate.set).start()
    t0 = time.monotonic()
    assert pre.take(fake.source(0), fake.params(), None) is not None
    assert time.monotonic() - t0 >= 0.25


def test_a_read_error_is_raised_in_turn_and_stops_the_rest(fake) -> None:
    def read(src, cancel):
        raise RuntimeError(f"{src.name}: decoder broke")

    fake.read = read
    pre = fake.make()
    pre.offer(fake.paths[0], 60.0, True)
    pre.offer(fake.paths[1], 60.0, True)
    with pytest.raises(RuntimeError, match=r"a\.mp4: decoder broke"):
        pre.take(fake.source(0), fake.params(), None)
    pre.close()
    assert fake.reads == ["a.mp4"]  # a serial run stops at the failed source too


def test_a_source_not_offered_is_read_inline_after_the_worker_stops(fake) -> None:
    pre = fake.make()
    pre.offer(fake.paths[0], 60.0, False)  # a read error: neither it nor a later one
    pre.offer(fake.paths[1], 60.0, True)
    assert pre.take(fake.source(0), fake.params(), None) is None
    assert pre.take(fake.source(1), fake.params(), None) is None
    assert fake.reads == []


def test_other_inputs_are_read_inline(fake) -> None:
    pre = fake.make()
    pre.offer(fake.paths[0], 60.0, True)
    pre.offer(fake.paths[1], 60.0, True)
    other = OcrParams(game="zonewar", project=fake.project, player_tags=("Someone",))
    assert pre.take(fake.source(0), other, None) is None
    longer = Source(fake.paths[1], 61.0, fake.source(1).cdir, [])
    assert pre.take(longer, fake.params(), None) is None


@pytest.mark.parametrize("why", ["preset", "vram"])
def test_declined_without_a_preset_or_vram(fake, records, why) -> None:
    setattr(fake, why, False)
    pre = fake.make()
    pre.offer(fake.paths[0], 60.0, True)
    assert pre.take(fake.source(0), fake.params(), None) is None
    assert fake.reads == []
    (line,) = [x for x in records.lines if x.startswith("OCR prefetch off: ")]
    assert ("no HUD preset" in line) == (why == "preset")


def test_close_cancels_a_read_and_drops_its_logs(fake, records) -> None:
    started = threading.Event()

    def read(src, cancel):
        logging.getLogger("cutpoint.events.killfeed").info("partial read")
        started.set()
        assert cancel.wait(5)
        raise OcrCancelled

    fake.read = read
    pre = fake.make()
    pre.offer(fake.paths[0], 60.0, True)
    assert started.wait(5)
    t0 = time.monotonic()
    pre.close()
    assert time.monotonic() - t0 < 2
    assert pre.take(fake.source(0), fake.params(), None) is None
    assert "partial read" not in records.lines
    assert not any(isinstance(f, ocr_prefetch._Hold) for h in logging.getLogger().handlers
                   for f in h.filters)  # fmt: skip


def test_close_keeps_the_logs_of_a_finished_read(fake, records) -> None:
    pre = fake.make()
    pre.offer(fake.paths[0], 60.0, True)
    deadline = time.monotonic() + 5
    while fake.reads != ["a.mp4"] and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)
    pre.close()  # e.g. a later source is read inline
    assert pre.take(fake.source(0), fake.params(), None) is not None
    assert records.lines[-2:] == ["a.mp4: read 1", "a.mp4: read 2"]


def test_a_source_that_cannot_be_loaded_never_fails_the_transcription(fake, monkeypatch) -> None:
    def broken(path, duration):
        raise OSError("gone")

    monkeypatch.setattr(ocr_prefetch, "load_source", broken)
    pre = fake.make()
    pre.offer(fake.paths[0], 60.0, True)  # no error
    monkeypatch.undo()
    pre.offer(fake.paths[1], 60.0, True)  # nor anything later
    assert pre.take(fake.source(1), fake.params(), None) is None
    assert fake.reads == []


def test_a_worker_that_outlives_close_is_stuck_until_it_leaves_its_source(
    fake, monkeypatch
) -> None:
    monkeypatch.setattr(ocr_prefetch, "JOIN_S", 0.05)
    gate, started = threading.Event(), threading.Event()

    def read(src, cancel):  # one long OCR call: does not see the cancel
        started.set()
        gate.wait(5)
        raise OcrCancelled

    fake.read = read
    pre = fake.make()
    pre.offer(fake.paths[0], 60.0, True)
    assert started.wait(5)
    pre.close()
    assert pre.stuck
    pre.wait_for(fake.paths[1])  # another source: no wait
    threading.Timer(0.3, gate.set).start()
    t0 = time.monotonic()
    pre.wait_for(fake.paths[0])
    assert time.monotonic() - t0 >= 0.25 and not pre.stuck


# ----------------------------------------------------------------- run_events
@pytest.fixture
def ocr_run(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake):
    """``run_events`` with the OCR layer only on the fake's two sources; it stops at the
    VLM layer."""
    monkeypatch.setenv(progress.ENV, "1")
    monkeypatch.setattr(progress, "MIN_INTERVAL_S", 0.0)  # every update is a line
    by_path = {str(p): fake.source(i) for i, p in enumerate(fake.paths)}
    hl = {"sources": [{"path": str(p), "duration": 60.0} for p in fake.paths]}
    monkeypatch.setattr(pipeline, "load_highlights", lambda project: hl)
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "load_source", lambda path, dur: by_path[str(path)])
    monkeypatch.setattr(pipeline, "bad_ranges", lambda *a: {})
    monkeypatch.setattr(pipeline, "load_transcript_lines", lambda project: [])
    monkeypatch.setattr(pipeline, "load_preset", lambda *a: None)
    monkeypatch.setattr(pipeline, "_discover", lambda *a: {})
    monkeypatch.setattr(pipeline, "OcrEngine", lambda: None)
    monkeypatch.setattr(pipeline, "detect_ocr_events", ocr_prefetch.detect_ocr_events)

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(pipeline, "game_prompt", stop)  # the VLM layer's first call
    seen: list = []
    fake.seen = seen
    monkeypatch.setattr(pipeline, "link_teamkills", lambda evs, **k: seen.append(evs))

    def run(pre=None, stage=None) -> list[dict]:
        with progress.run(("keyframes", "ranking")) as steps:
            hook = stage(steps) if stage else None
            if hook is None:
                steps.start("keyframes")
            with pytest.raises(_Stop):
                pipeline.run_events(
                    fake.project,
                    layers=("ocr",),
                    ocr=OcrParams(game="zonewar"),
                    prefetch=pre,
                    stage=hook,
                )
        return seen

    fake.run = run
    return fake


def _lines(capsys) -> list[dict]:
    return [
        {k: v for k, v in json.loads(x.removeprefix("PROGRESS ")).items() if k != "elapsed_s"}
        for x in capsys.readouterr().err.splitlines()
        if x.startswith("PROGRESS ")
    ]


def test_run_events_gives_the_same_events_and_progress_with_the_prefetch(
    ocr_run, capsys, records
) -> None:
    serial = list(ocr_run.run())
    serial_lines, serial_log = _lines(capsys), list(records.lines)
    records.lines.clear()
    ocr_run.reads.clear()
    ocr_run.seen.clear()
    pre = ocr_run.make()
    for p in ocr_run.paths:
        pre.offer(p, 60.0, True)
    assert ocr_run.run(pre) == serial
    assert _lines(capsys) == serial_lines
    assert records.lines == serial_log
    assert ocr_run.reads == ["a.mp4", "b.mp4"]  # each read once, by the worker


def test_a_worker_error_fails_only_the_keyframes_step(ocr_run, capsys) -> None:
    def read(src, cancel):
        raise RuntimeError("decoder broke")

    ocr_run.read = read
    pre = ocr_run.make()
    pre.offer(ocr_run.paths[0], 60.0, True)
    ocr_run.run(pre, stage=analyze._StepHook)
    got = [(g["step"], g["state"]) for g in _lines(capsys)]
    assert ("keyframes", "failed") in got
    assert ocr_run.reads == ["a.mp4"]


def test_an_engine_is_shared_with_the_inline_reads(ocr_run, monkeypatch) -> None:
    pre = ocr_run.make()
    pre.offer(ocr_run.paths[0], 60.0, True)
    pre.offer(ocr_run.paths[1], 60.0, False)  # b.mp4 is read inline, after the worker
    engines = []
    inline = pipeline.detect_ocr_events

    def detect(src, params, engine, *, make_engine, **kw):
        engines.append(make_engine)
        return inline(src, params, engine, make_engine=make_engine, **kw)

    monkeypatch.setattr(pipeline, "detect_ocr_events", detect)
    ocr_run.run(pre)
    assert engines == [pre.make_engine]  # one engine factory: one RapidOCR at most
    assert ocr_run.reads == ["a.mp4", "b.mp4"]


def test_a_stuck_worker_gets_neither_its_source_nor_its_engine(ocr_run, monkeypatch) -> None:
    """A source the prefetch cannot give (here: a bad range) is read inline; when the worker
    outlives close(), that read waits until the worker left the source, with an engine of
    its own."""
    monkeypatch.setattr(ocr_prefetch, "JOIN_S", 0.05)
    gate, started = threading.Event(), threading.Event()

    def read(src, cancel):
        started.set()
        gate.wait(5)
        raise OcrCancelled

    ocr_run.read = read
    a = str(ocr_run.paths[0])
    monkeypatch.setattr(pipeline, "bad_ranges", lambda *a_: {a: [(1.0, 2.0)]})
    pre = ocr_run.make()
    pre.offer(ocr_run.paths[0], 60.0, True)
    assert started.wait(5)
    seen = []

    def detect(src, params, engine, *, make_engine, **kw):
        seen.append((src.name, make_engine is pre.make_engine, pre.stuck))
        return [], {}

    monkeypatch.setattr(pipeline, "detect_ocr_events", detect)
    threading.Timer(0.3, gate.set).start()
    ocr_run.run(pre)
    assert seen == [("a.mp4", False, False), ("b.mp4", False, False)]


# ----------------------------------------------------------------- the switches
def test_env_var_turns_it_off(monkeypatch) -> None:
    assert ocr_prefetch.enabled()
    monkeypatch.setenv("CUTPOINT_ANALYZE_OVERLAP", "0")
    assert not ocr_prefetch.enabled()
    monkeypatch.setenv("CUTPOINT_ANALYZE_OVERLAP", "1")
    assert ocr_prefetch.enabled()


@pytest.mark.parametrize(
    ("out", "want"),
    [
        ("4000, 10240\n", True),
        ("1500, 10240\n", False),  # too little free
        ("4000, 6144\n", False),  # too small a GPU
        ("", True),  # nothing parsable: unknown
        (None, True),  # no nvidia-smi
    ],
)
def test_vram_gate(monkeypatch, out, want) -> None:
    def run(*a, **k):
        if out is None:
            raise FileNotFoundError("nvidia-smi")
        return SimpleNamespace(stdout=out)

    monkeypatch.setattr(ocr_prefetch.subprocess, "run", run)
    assert ocr_prefetch.vram_ok() is want


@pytest.fixture
def eligible(monkeypatch):
    """Everything ``_ocr_prefetch`` checks is in place (extra, model files)."""
    monkeypatch.setattr(analyze.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(analyze.event_paths, "missing_files", lambda layers: [])


@pytest.mark.parametrize(
    ("style", "skip", "layers", "env", "want"),
    [
        ("gaming", (), True, None, True),
        ("funny", (), True, None, True),
        ("gaming", (), True, "0", False),
        ("gaming", ("keyframes",), True, None, False),
        ("gaming", (), False, None, False),  # reanalyze, 360 analyze
        ("action", (), True, None, False),  # no HUD OCR
    ],
)
def test_prefetch_only_when_ocr_runs(
    eligible, tmp_path, monkeypatch, style, skip, layers, env, want
) -> None:
    if env is not None:
        monkeypatch.setenv("CUTPOINT_ANALYZE_OVERLAP", env)
    ctx = analyze._ocr_prefetch(get_style(style), [tmp_path / "a.mp4"], skip, tmp_path, layers)
    with ctx as pre:
        assert isinstance(pre, ocr_prefetch.OcrPrefetch) is want
    if want:
        assert pre.params.project == tmp_path and pre.params.game == "auto"


def test_no_prefetch_without_the_model_files(eligible, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(analyze.event_paths, "missing_files", lambda layers: [("ocr", "x")])
    with analyze._ocr_prefetch(get_style("gaming"), [], (), tmp_path, True) as pre:
        assert pre is None


# ----------------------------------------------------------------- decode_raw(isolate=True)
def test_isolated_popen_options(monkeypatch) -> None:
    assert ffmpeg_utils._isolated() == (
        {"stdin": subprocess.DEVNULL, "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"stdin": subprocess.DEVNULL, "start_new_session": True}
    )
    monkeypatch.setattr(ffmpeg_utils.os, "name", "nt")
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    assert ffmpeg_utils._isolated() == {"stdin": subprocess.DEVNULL, "creationflags": 0x200}


@needs_ffmpeg
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX sessions")
def test_isolated_decode_runs_in_its_own_session_and_dies_on_close(tmp_path, monkeypatch) -> None:
    clip = tmp_path / "bars.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x36:rate=10:duration=5",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )  # fmt: skip
    procs: list[subprocess.Popen] = []
    real = subprocess.Popen

    def popen(*a, **k):
        procs.append(real(*a, **k))
        return procs[-1]

    monkeypatch.setattr(ffmpeg_utils.subprocess, "Popen", popen)
    frames = ffmpeg_utils.decode_raw(clip, size=(64, 36), hwaccel=None, isolate=True)
    next(frames)
    (proc,) = procs
    assert os.getsid(proc.pid) == proc.pid != os.getsid(0)  # a session of its own
    frames.close()  # the owner stops it
    assert proc.poll() is not None
    plain = ffmpeg_utils.decode_raw(clip, size=(64, 36), hwaccel=None)
    next(plain)
    assert os.getsid(procs[-1].pid) == os.getsid(0)  # without isolate: as before
    plain.close()


# ----------------------------------------------------------------- real engine, real ffmpeg
@needs_ffmpeg
@pytest.mark.slow
def test_real_background_read_equals_the_serial_read(home, tmp_path, monkeypatch) -> None:
    """The real OCR engine and an isolated ffmpeg in the worker thread give the events and
    raw read of a serial read (the synthetic clip of the discovery e2e test)."""
    pytest.importorskip("rapidocr_onnxruntime")
    import shutil  # noqa: PLC0415

    import test_discover_e2e as e2e  # noqa: PLC0415
    from cutpoint import paths  # noqa: PLC0415
    from cutpoint.events import common, killfeed  # noqa: PLC0415
    from hud_fixtures import write_preset  # noqa: PLC0415
    from test_ocr_raw import _files  # noqa: PLC0415

    monkeypatch.setattr(e2e, "SECONDS", 12)
    folder = tmp_path / "footage"
    folder.mkdir()
    serial_clip, background_clip = folder / "a.mp4", folder / "b.mp4"
    e2e.make_clip(serial_clip)
    shutil.copy2(serial_clip, background_clip)
    fx, fy, fw, fh = (round(v * 1.5) for v in e2e.FEED)
    region = {"name": "feed", "x": fx, "y": fy, "w": fw, "h": fh, "kind": "feed", "every": 1}
    feed = {"kill_words": {"downed": "kill"}}  # the clip's invented feed lines
    write_preset(paths.presets_dir(), {"game": e2e.GAME, "regions": [region], "feed": feed})
    project = tmp_path / "proj"
    params = OcrParams(game=e2e.GAME, fps=1.0, hwaccel=False, project=project)
    engine = killfeed.OcrEngine()
    serial = killfeed.detect_ocr_events(common.load_source(serial_clip, 12.0), params, engine)
    pre = ocr_prefetch.OcrPrefetch(project, [background_clip], OcrParams(game=e2e.GAME, fps=1.0))
    pre.params.hwaccel = False
    pre.make_engine = lambda: engine  # one engine in this test too
    with pre:
        pre.offer(background_clip, 12.0, True)
        src = common.load_source(background_clip, 12.0)
        events, stats = pre.take(src, params, None)
    assert events == serial[0] and serial[0], "the clip's feed gave no events"
    drop = ("elapsed_s",)
    assert {k: v for k, v in stats.items() if k not in drop} == {
        k: v for k, v in serial[1].items() if k not in drop
    }
    a, b = (_files(common.load_source(c, 12.0).cdir) for c in (serial_clip, background_clip))
    assert a == b


@needs_ffmpeg
def test_cancel_kills_the_decode_at_once_and_outside_exits_are_reported(tmp_path) -> None:
    clip = tmp_path / "bars.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x36:rate=10:duration=5",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )  # fmt: skip
    # cancel: ffmpeg dies while the consumer holds a frame; the kill is no outside exit
    cancel, codes = threading.Event(), []
    frames = ffmpeg_utils.decode_raw(
        clip, size=(64, 36), hwaccel=None, isolate=True, cancel=cancel, exit_codes=codes
    )
    next(frames)
    cancel.set()
    rest = list(frames)  # what was already in the pipe, then the end
    assert len(rest) < 49 and codes == []
    # a whole decode exits 0: nothing reported
    codes = []
    assert (
        len(list(ffmpeg_utils.decode_raw(clip, size=(64, 36), hwaccel=None, exit_codes=codes)))
        == 50
    )
    assert codes == []


@needs_ffmpeg
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_an_outside_stop_of_ffmpeg_is_reported(tmp_path, monkeypatch) -> None:
    """What a console Ctrl+Break does to an isolated ffmpeg (here SIGINT)."""
    import signal  # noqa: PLC0415

    clip = tmp_path / "bars.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x36:rate=10:duration=60",
         "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )  # fmt: skip
    procs: list[subprocess.Popen] = []
    real = subprocess.Popen

    def popen(*a, **k):
        procs.append(real(*a, **k))
        return procs[-1]

    monkeypatch.setattr(ffmpeg_utils.subprocess, "Popen", popen)
    codes: list[int] = []
    frames = ffmpeg_utils.decode_raw(
        clip, size=(64, 36), hwaccel=None, isolate=True, exit_codes=codes
    )
    next(frames)
    os.kill(procs[0].pid, signal.SIGINT)
    list(frames)  # how many frames still arrive depends on ffmpeg's internal queues
    assert codes and codes[0] != 0
