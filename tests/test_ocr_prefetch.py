"""Background HUD OCR during ``analyze`` (#375): results taken in their turn, logs replayed in
order, progress forwarded, errors re-raised, the off switches, and ``decode_raw(isolate=True)``
(fake OCR reads, invented game ``zonewar``)."""

from __future__ import annotations

import contextlib
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
from cutpoint import analyze, ffmpeg_utils, jsonout, memguard, progress
from cutpoint.errors import LowMemoryError, is_out_of_memory
from cutpoint.events import ocr_prefetch, pipeline, vlm_windows
from cutpoint.events.common import Source
from cutpoint.events.killfeed import OcrCancelled, OcrParams
from cutpoint.styles import get_style


class _Stop(Exception):
    pass


_REAL_DISCOVER = pipeline._discover  # the fixture replaces it


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
    assert pre.take(fake.source(0), fake.params(), None) is not None
    assert fake.gate.is_set()  # it returned only once the read could finish


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

    def run(pre=None, stage=None, batch_size=0, stops=_Stop) -> list[dict]:
        """``stops``: the error the run stops with (``fake.stopped`` gets it)."""
        with progress.run(("keyframes", "ranking")) as steps:
            hook = stage(steps) if stage else None
            if hook is None:
                steps.start("keyframes")
            with pytest.raises(stops) as fake.stopped:
                pipeline.run_events(
                    fake.project,
                    layers=("ocr",),
                    ocr=OcrParams(game="zonewar"),
                    prefetch=pre,
                    stage=hook,
                    batch_size=batch_size,
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


def _ocr_warnings(caplog) -> list[tuple[int, str]]:
    return [
        (r.levelno, r.getMessage())
        for r in caplog.records
        if r.levelno >= logging.WARNING and "player_tags" not in r.getMessage()
    ]


def _oom() -> RuntimeError:
    return RuntimeError("CUDA out of memory. Tried to allocate 64.00 MiB (GPU 0; 5.80 GiB total)")


def _stopped(info, what: str, name: str) -> None:
    """The run stopped with a resumable LowMemoryError chained to the OCR read's error."""
    assert info.type is LowMemoryError
    assert str(info.value).startswith(f"Out of memory in {what} ({name}); close other programs")
    assert is_out_of_memory(info.value.__cause__)


def _handed_back(caplog, name: str) -> list[str]:
    """The INFO lines of a background read handed back to its turn for out of memory."""
    want = f"{name}: the background OCR ran out of memory ("
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(want)]


@pytest.mark.parametrize("err", [_oom(), MemoryError()])
def test_out_of_memory_in_a_worker_read_is_read_again_in_its_turn(ocr_run, caplog, err) -> None:
    """Out of memory in the background read of the first source (the GPU steps it overlaps
    hold the GPU) hands it back: it is read inline in its turn, and so is the source after
    it, which the worker does not start. No WARNING, no step_failed; the worker is stopped."""

    def read(src, cancel):
        if cancel is not None:  # the background read
            raise err
        return [{"t": 1.0, "type": "kill", "src": src.name}], {"frames": 4}

    ocr_run.read = read
    pre = ocr_run.make()
    for p in ocr_run.paths:
        pre.offer(p, 60.0, True)
    thread = pre._thread
    with caplog.at_level(logging.INFO), jsonout.run("analyze") as res, pre:
        seen = ocr_run.run(pre, stage=analyze._StepHook)
    assert [evs[0]["src"] for evs in seen] == ["a.mp4", "b.mp4"]
    assert ocr_run.reads == ["a.mp4", "a.mp4", "b.mp4"]  # a: background, then inline
    reason = str(err) or "MemoryError"
    assert _handed_back(caplog, "a.mp4") == [
        f"a.mp4: the background OCR ran out of memory ({reason}); reading it again in its turn"
    ]
    assert _handed_back(caplog, "b.mp4") == []  # not started in the background: no OOM
    assert _ocr_warnings(caplog) == [] and res.warnings == []
    assert thread is not None and not thread.is_alive() and not pre.stuck


@pytest.mark.parametrize("err", [_oom(), MemoryError()])
def test_out_of_memory_in_the_read_again_stops_the_run(ocr_run, caplog, err) -> None:
    """The second source runs out of memory in the background and again inline, in its
    turn: that stops the run as low_memory (no WARNING, no step_failed); the first source
    keeps its background result."""

    def read(src, cancel):
        if src.name == "b.mp4":
            raise err
        return [{"t": 1.0, "type": "kill", "src": src.name}], {"frames": 4}

    ocr_run.read = read
    pre = ocr_run.make()
    for p in ocr_run.paths:
        pre.offer(p, 60.0, True)
    thread = pre._thread
    with caplog.at_level(logging.INFO), jsonout.run("analyze") as res, pre:
        seen = ocr_run.run(pre, stage=analyze._StepHook, stops=LowMemoryError)
    _stopped(ocr_run.stopped, "the HUD OCR", "b.mp4")
    assert ocr_run.stopped.value.__cause__ is err  # the inline read's error
    assert [evs[0]["src"] for evs in seen] == ["a.mp4"]
    assert ocr_run.reads == ["a.mp4", "b.mp4", "b.mp4"]  # b: background, then inline
    assert len(_handed_back(caplog, "b.mp4")) == 1
    assert _ocr_warnings(caplog) == [] and res.warnings == []
    assert thread is not None and not thread.is_alive() and not pre.stuck


def _guarded_read(mem, background_gb: float, inline_gb: float):
    """A read that makes its OCR engine as the real one does (``memguard.check("ocr")``),
    with ``background_gb`` of RAM free in the worker and ``inline_gb`` in its turn."""

    def read(src, cancel):
        mem.ram = background_gb if cancel is not None else inline_gb
        memguard.check("ocr")
        return [{"t": 1.0, "type": "kill", "src": src.name}], {"frames": 4}

    return read


def _freed_on(monkeypatch, mem) -> list[str]:
    """The names of the threads that freed cached models."""
    threads: list[str] = []
    counted = analyze.free_models

    def free_models() -> None:
        threads.append(threading.current_thread().name)
        counted()

    monkeypatch.setattr(analyze, "free_models", free_models)
    return threads


def test_a_failed_ram_check_in_a_worker_read_is_read_again_in_its_turn(
    ocr_run, caplog, mem, monkeypatch
) -> None:
    """The memory guard finds too little RAM free for the OCR engine in the background: the
    source is handed back and read in its turn, where the RAM is free, and so is the source
    after it. One INFO line, no LowMemoryError, no WARNING; the worker frees nothing."""
    freed = _freed_on(monkeypatch, mem)
    ocr_run.read = _guarded_read(mem, background_gb=1.0, inline_gb=64.0)
    pre = ocr_run.make()
    for p in ocr_run.paths:
        pre.offer(p, 60.0, True)
    with caplog.at_level(logging.INFO), jsonout.run("analyze") as res, pre:
        seen = ocr_run.run(pre, stage=analyze._StepHook)
    assert [evs[0]["src"] for evs in seen] == ["a.mp4", "b.mp4"]
    assert ocr_run.reads == ["a.mp4", "a.mp4", "b.mp4"]  # a: background, then inline
    want = "a.mp4: the background OCR had too little memory free ("
    lines = [m for m in caplog.messages if m.startswith(want)]
    assert lines == [f"{want}1.0 GB RAM free, 3.0 GB needed); reading it again in its turn"]
    assert not any("freeing cached models" in m for m in caplog.messages)
    assert freed == [] and mem.unloaded == 0
    assert _ocr_warnings(caplog) == [] and res.warnings == []


def test_a_ram_check_that_fails_again_in_its_turn_stops_the_run(
    ocr_run, caplog, mem, monkeypatch
) -> None:
    """Too little RAM in the background and again in the source's turn: the inline check
    frees cached models (on the main thread, the worker freed nothing), then stops the run
    with low_memory."""
    freed = _freed_on(monkeypatch, mem)
    ocr_run.read = _guarded_read(mem, background_gb=1.0, inline_gb=1.5)
    pre = ocr_run.make()
    for p in ocr_run.paths:
        pre.offer(p, 60.0, True)
    with caplog.at_level(logging.INFO), jsonout.run("analyze") as res, pre:
        seen = ocr_run.run(pre, stage=analyze._StepHook, stops=LowMemoryError)
    err = ocr_run.stopped.value
    assert err.args[1] == "1.5 GB RAM free, 3.0 GB needed"
    assert str(err).startswith("Not enough memory for the OCR engine: 1.5 GB RAM free")
    assert seen == [] and ocr_run.reads == ["a.mp4", "a.mp4"]
    assert freed == [threading.main_thread().name] and mem.unloaded == 1
    assert _ocr_warnings(caplog) == [] and res.warnings == []


def test_no_free_holds_only_in_its_own_thread(mem) -> None:
    """A check inside ``no_free`` frees nothing and stops at once; another thread's check
    meanwhile frees as before, and so does this thread's after it."""
    mem.ram = 1.0
    with memguard.no_free():
        with pytest.raises(LowMemoryError):
            memguard.check("ocr")
        assert mem.freed == mem.unloaded == 0
        other: list[BaseException] = []

        def check() -> None:
            try:
                memguard.check("ocr")
            except LowMemoryError as e:
                other.append(e)

        t = threading.Thread(target=check)
        t.start()
        t.join(5)
        assert len(other) == 1 and mem.freed == mem.unloaded == 1
    with pytest.raises(LowMemoryError):
        memguard.check("ocr")
    assert mem.freed == mem.unloaded == 2


def test_a_non_memory_worker_error_still_only_warns(ocr_run, caplog) -> None:
    def read(src, cancel):
        if src.name == "b.mp4":
            raise RuntimeError("decoder broke")
        return [], {"menu_spans": [[1.0, 2.0]]}

    ocr_run.read = read
    pre = ocr_run.make()
    for p in ocr_run.paths:
        pre.offer(p, 60.0, True)
    with jsonout.run("analyze") as res:
        ocr_run.run(pre, stage=analyze._StepHook)
    want = (
        "OCR failed (decoder broke): menu filtering and HUD events (kill feed, banners, "
        "notices) skipped for b.mp4"
    )
    assert _ocr_warnings(caplog) == [(logging.WARNING, want)]
    assert res.warnings == [
        {"code": "step_failed", "message": want, "files": [str(ocr_run.paths[1])]}
    ]


@pytest.mark.parametrize(
    ("err", "reason"),
    [(RuntimeError("decoder broke\nmore detail"), "decoder broke"), (ValueError(), "ValueError")],
)
def test_an_inline_ocr_failure_names_every_source_not_read(ocr_run, caplog, err, reason) -> None:
    def read(src, cancel):
        raise err

    ocr_run.read = read
    with jsonout.run("analyze") as res:
        ocr_run.run(stage=analyze._StepHook)
    want = (
        f"OCR failed ({reason}): menu filtering and HUD events (kill feed, banners, notices) "
        "skipped for a.mp4 and 1 later source"
    )
    assert _ocr_warnings(caplog) == [(logging.WARNING, want)]
    assert [(w["code"], w["message"], len(w["files"])) for w in res.warnings] == [
        ("step_failed", want, 2)
    ]
    assert ocr_run.reads == ["a.mp4"]  # the layer stops at its first failure


@pytest.mark.parametrize("err", [_oom(), MemoryError()])
def test_out_of_memory_in_an_inline_read_stops_the_run(ocr_run, caplog, err) -> None:
    def read(src, cancel):
        raise err

    ocr_run.read = read
    with jsonout.run("analyze") as res:
        ocr_run.run(stage=analyze._StepHook, stops=LowMemoryError)
    _stopped(ocr_run.stopped, "the HUD OCR", "a.mp4")
    assert _ocr_warnings(caplog) == [] and res.warnings == []
    assert ocr_run.reads == ["a.mp4"]


def test_the_sources_read_before_a_failure_keep_their_menu_spans(ocr_run, monkeypatch) -> None:
    """#555: the stats of a.mp4 (its menu spans) are written although b.mp4 failed."""

    def read(src, cancel):
        if src.name == "b.mp4":
            raise RuntimeError("decoder broke")
        return [], {"menu_spans": [[1.0, 2.0]]}

    written = []

    def write_outputs(project, sources, events, results, transcript, settings, timings, stats):
        written.append(stats["ocr"])
        raise _Stop

    ocr_run.read = read
    monkeypatch.setattr(pipeline, "game_prompt", lambda *a: "")
    monkeypatch.setattr(pipeline, "select_windows", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "describe_windows", lambda *a, **k: ([], {}))
    monkeypatch.setattr(pipeline, "write_outputs", write_outputs)
    with jsonout.run("analyze"):
        ocr_run.run(stage=analyze._StepHook)
    assert written == [{"a.mp4": {"menu_spans": [[1.0, 2.0]]}}]


def test_a_discovery_failure_names_the_missing_preset(ocr_run, caplog, monkeypatch) -> None:
    def discover(*a):
        raise ValueError("no frames to sample")

    monkeypatch.setattr(pipeline, "_discover", discover)
    with jsonout.run("analyze") as res:
        ocr_run.run(stage=analyze._StepHook)
    want = (
        "OCR failed in HUD region discovery (no frames to sample): no HUD preset installed, "
        "menu filtering and HUD events (kill feed, banners, notices) skipped for every source"
    )
    assert _ocr_warnings(caplog) == [(logging.WARNING, want)]
    assert [w["message"] for w in res.warnings] == [want]
    assert ocr_run.reads == []


def _discover_regions(monkeypatch, ocr_run, fail: str, err: Exception) -> None:
    """The real ``_discover`` with two games (no presets), ``fail``'s discovery raising
    ``err``."""
    from cutpoint.events import discover  # noqa: PLC0415

    def discover_regions(srcs, *a, **k):
        if srcs[0].name == fail:
            raise err
        return {"regions": []}

    monkeypatch.setattr(pipeline, "_discover", _REAL_DISCOVER)
    monkeypatch.setattr(pipeline, "game_key_for_path", lambda path, game, project: Path(path).stem)
    monkeypatch.setattr(discover, "discover_regions", discover_regions)
    monkeypatch.setattr(pipeline.cache, "save_json", lambda *a: None)
    monkeypatch.setattr(pipeline.cutpoint_paths, "presets_dir", lambda: ocr_run.project / "p")


@pytest.mark.parametrize("reader", [False, True])
def test_out_of_memory_in_discovery_stops_the_run(ocr_run, caplog, monkeypatch, reader) -> None:
    err = MemoryError("Unable to allocate 3.1 GiB for an array")
    _discover_regions(monkeypatch, ocr_run, "a.mp4", err)
    pre = ocr_run.make() if reader else None
    for p in ocr_run.paths if reader else ():
        pre.offer(p, 60.0, True)
    with jsonout.run("analyze") as res, pre or contextlib.nullcontext():
        ocr_run.run(pre, stage=analyze._StepHook, stops=LowMemoryError)
    _stopped(ocr_run.stopped, "HUD region discovery", "a")
    assert _ocr_warnings(caplog) == [] and res.warnings == []
    if not reader:
        assert ocr_run.reads == []


def test_out_of_memory_outside_the_reads_stops_the_run(ocr_run, caplog, monkeypatch) -> None:
    """An untagged out-of-memory error of the OCR layer (here: its discovery's setup) stops
    the run too."""

    def discover(*a):
        raise _oom()

    monkeypatch.setattr(pipeline, "_discover", discover)
    with jsonout.run("analyze") as res:
        ocr_run.run(stage=analyze._StepHook, stops=LowMemoryError)
    _stopped(ocr_run.stopped, "HUD region discovery", "every source")
    assert _ocr_warnings(caplog) == [] and res.warnings == []


def test_a_discovery_failure_for_the_second_game_says_some_games(
    ocr_run, caplog, monkeypatch
) -> None:
    """Two games to discover (no presets): the first may have installed its preset."""
    _discover_regions(monkeypatch, ocr_run, "b.mp4", RuntimeError("decoder broke"))
    with jsonout.run("analyze") as res:
        ocr_run.run(stage=analyze._StepHook)
    want = (
        "OCR failed in HUD region discovery (decoder broke): no HUD preset installed for some "
        "games, menu filtering and HUD events (kill feed, banners, notices) skipped for every "
        "source"
    )
    assert _ocr_warnings(caplog) == [(logging.WARNING, want)]
    assert res.warnings[0]["files"] == [str(p) for p in ocr_run.paths]
    assert ocr_run.reads == []


def test_a_failure_after_a_read_names_its_source(ocr_run, caplog, monkeypatch) -> None:
    """#555: link_teamkills (after the read) fails on b.mp4: the WARNING names b.mp4, and
    a.mp4 keeps its events."""
    seen = []

    def link_teamkills(evs, **k):
        if evs and evs[0]["src"] == "b.mp4":
            raise ValueError("bad feed")
        seen.append(evs)

    monkeypatch.setattr(pipeline, "link_teamkills", link_teamkills)
    with jsonout.run("analyze") as res:
        ocr_run.run(stage=analyze._StepHook)
    want = (
        "OCR failed (bad feed): menu filtering and HUD events (kill feed, banners, notices) "
        "skipped for b.mp4"
    )
    assert _ocr_warnings(caplog) == [(logging.WARNING, want)]
    assert res.warnings[0]["files"] == [str(ocr_run.paths[1])]
    assert [evs[0]["src"] for evs in seen] == ["a.mp4"]


def test_an_untagged_ocr_failure_does_not_claim_every_source(ocr_run, caplog, monkeypatch) -> None:
    """A failure in the OCR layer's setup, outside discovery and the reads."""

    def vram_monitor():
        raise RuntimeError("no monitor")

    monkeypatch.setattr(pipeline, "VramMonitor", vram_monitor)
    with jsonout.run("analyze") as res:
        ocr_run.run(stage=analyze._StepHook)
    want = (
        "OCR failed (no monitor): menu filtering and HUD events (kill feed, banners, notices) "
        "skipped for the sources not read yet"
    )
    assert _ocr_warnings(caplog) == [(logging.WARNING, want)]
    assert res.warnings == [{"code": "step_failed", "message": want, "files": []}]


def test_without_the_hook_the_ocr_error_is_raised_unchanged(ocr_run, monkeypatch) -> None:
    """``cutpoint events`` (no stage): the same exception, as before."""
    err = ValueError("bad feed")

    def read(src, cancel):
        raise err

    ocr_run.read = read
    with pytest.raises(ValueError) as got:
        pipeline.run_events(ocr_run.project, layers=("ocr",), ocr=OcrParams(game="zonewar"))
    assert got.value is err
    assert pipeline.ocr_failure(err) == ([str(p) for p in ocr_run.paths], False, 0)


def test_without_the_hook_out_of_memory_stops_as_low_memory(ocr_run) -> None:
    """``cutpoint events``: out of memory in the OCR read is ``low_memory`` there too."""
    err = _oom()

    def read(src, cancel):
        raise err

    ocr_run.read = read
    with pytest.raises(LowMemoryError) as got:
        pipeline.run_events(ocr_run.project, layers=("ocr",), ocr=OcrParams(game="zonewar"))
    _stopped(got, "the HUD OCR", "a.mp4")
    assert got.value.__cause__ is err


def test_without_the_hook_out_of_memory_in_discovery_stops_as_low_memory(
    ocr_run, monkeypatch
) -> None:
    """``cutpoint events``: out of memory in HUD region discovery is ``low_memory`` too."""
    err = _oom()

    def discover(*args, **kwargs):
        raise err

    monkeypatch.setattr(pipeline, "_discover", discover)
    with pytest.raises(LowMemoryError) as got:
        pipeline.run_events(
            ocr_run.project, layers=("ocr",), ocr=OcrParams(game="zonewar"), discover=True
        )
    _stopped(got, "HUD region discovery", "every source")
    assert got.value.__cause__ is err


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


@pytest.mark.parametrize("size", [0, 1])
def test_a_stuck_worker_gets_neither_its_source_nor_its_engine(ocr_run, monkeypatch, size) -> None:
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
    pre = ocr_run.pre = ocr_prefetch.OcrPrefetch(
        ocr_run.project, ocr_run.paths, OcrParams(game="zonewar"), batch_size=size
    )
    worker_factory = pre.make_engine
    pre.offer(ocr_run.paths[0], 60.0, True)
    assert started.wait(5)
    seen = []
    after_close = []
    real_close = pre.close

    def close():
        real_close()
        if pre.stuck:  # a close() that left the worker running swaps nothing
            after_close.append(pre.make_engine is worker_factory)

    monkeypatch.setattr(pre, "close", close)

    def detect(src, params, engine, *, make_engine, **kw):
        assert make_engine is not worker_factory
        seen.append((src.name, make_engine is pre.make_engine, pre.stuck))
        return [], {}

    monkeypatch.setattr(pipeline, "detect_ocr_events", detect)
    threading.Timer(0.3, gate.set).start()
    ocr_run.run(pre, batch_size=size)
    assert after_close == [True]
    assert seen == [("a.mp4", False, False), ("b.mp4", False, False)]


@pytest.mark.parametrize(("ocr_events", "want"), [([{"kind": "ocr", "type": "kill", "t": 1.0,
                          "text": "a > b"}], True), ([], False)])  # fmt: skip
def test_cached_ocr_kills_of_a_vlm_only_run_reach_the_game_prompt(
    ocr_run, monkeypatch, ocr_events, want
) -> None:
    """#596: the OCR events of layer 1, cached ones too when only the VLM layer runs, decide
    whether the prompt asks about kills."""
    seen: list[bool] = []

    def game_prompt(game, preset, combat=False):
        seen.append(combat)
        raise _Stop

    monkeypatch.setattr(pipeline, "game_prompt", game_prompt)
    monkeypatch.setattr(
        pipeline, "detect_ocr_events", lambda *a, **k: ([dict(e) for e in ocr_events], {})
    )
    with pytest.raises(_Stop):
        pipeline.run_events(ocr_run.project, layers=("vlm",), ocr=OcrParams(game="zonewar"))
    assert seen == [want]


# ----------------------------------------------------------------- the switches
@pytest.mark.parametrize(
    ("out", "want"),
    [
        ("0, GPU-a, 4000, 10240\n", True),
        ("0, GPU-a, 1500, 10240\n", False),  # too little free
        ("0, GPU-a, 4000, 6144\n", False),  # too small a GPU
        ("", True),  # nothing parsable: unknown
        (None, True),  # no nvidia-smi
    ],
)
def test_vram_gate(monkeypatch, out, want) -> None:
    def run(*a, **k):
        if out is None:
            raise FileNotFoundError("nvidia-smi")
        return SimpleNamespace(stdout=out)

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(memguard.subprocess, "run", run)
    assert ocr_prefetch.vram_ok() is want


@pytest.mark.parametrize(("visible", "want"), [("1", False), ("GPU-a", True), ("-1", True)])
def test_the_vram_gate_reads_the_gpu_cuda_visible_devices_names(monkeypatch, visible, want):
    """#721: the GPU the models use, not always nvidia-smi's first one; none visible: unknown."""
    rows = "0, GPU-a, 4000, 10240\n1, GPU-b, 1500, 10240\n"
    monkeypatch.setattr(memguard.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=rows))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
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
@pytest.mark.slow
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

    import ocr_helpers as e2e  # noqa: PLC0415
    from cutpoint import paths  # noqa: PLC0415
    from cutpoint.events import common, killfeed  # noqa: PLC0415
    from hud_fixtures import write_preset  # noqa: PLC0415

    folder = tmp_path / "footage"
    folder.mkdir()
    serial_clip, background_clip = folder / "a.mp4", folder / "b.mp4"
    e2e.make_clip(serial_clip, seconds=12)
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
    a, b = (
        e2e.ocr_cache_files(common.load_source(c, 12.0).cdir)
        for c in (serial_clip, background_clip)
    )
    assert a == b


@pytest.mark.slow
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


@pytest.mark.slow
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


# ----------------------------------------------------------------- before the VLM layer (#109)
@pytest.fixture
def released(ocr_run, monkeypatch: pytest.MonkeyPatch):
    """Fake OCR engines (made by every read through ``make_engine``) and the order of
    ``made``, ``free_torch`` and ``vlm`` (the VLM layer's first call)."""
    log: list[str] = []

    class Engine:
        def __init__(self) -> None:
            log.append("made")

    def detect(src, params, engine, *, make_engine=None, **kw):
        make_engine()
        return [], {}

    def vlm(*a, **k):
        log.append("vlm")
        raise _Stop

    monkeypatch.setattr(pipeline, "OcrEngine", Engine)
    monkeypatch.setattr(pipeline, "detect_ocr_events", detect)
    monkeypatch.setattr(pipeline, "free_torch", lambda: log.append("free_torch"))
    monkeypatch.setattr(pipeline, "game_prompt", vlm)
    monkeypatch.setattr(ocr_prefetch, "OcrEngine", Engine)
    ocr_run.log = log
    return ocr_run


def test_the_ocr_engine_and_the_cuda_cache_are_freed_before_the_vlm_layer(released) -> None:
    with pytest.raises(_Stop):
        pipeline.run_events(released.project, layers=("ocr", "vlm"), ocr=OcrParams(game="zonewar"))
    assert released.log == ["made", "free_torch", "vlm"]  # one engine for both reads


def test_the_prefetch_engine_is_freed_with_its_worker_before_the_vlm_layer(released) -> None:
    pre = released.make()
    pre.offer(released.paths[0], 60.0, False)  # nothing in the background: read with its engine
    with pytest.raises(_Stop):
        pipeline.run_events(
            released.project, layers=("ocr", "vlm"), ocr=OcrParams(game="zonewar"), prefetch=pre
        )
    assert released.log == ["made", "free_torch", "vlm"]
    assert pre.make_engine.cache_info().currsize == 0


def test_without_the_vlm_layer_torch_is_left_alone(released, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "select_windows", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "game_prompt", lambda *a: "")
    monkeypatch.setattr(pipeline, "write_outputs", lambda *a, **k: _Stop())
    pipeline.run_events(released.project, layers=("ocr",), ocr=OcrParams(game="zonewar"))
    assert released.log == ["made"]


@pytest.mark.parametrize("cached", [0, 2])
def test_a_partly_described_vlm_layer_says_how_far_it_got(
    ocr_run, monkeypatch, capsys, caplog, cached
) -> None:
    """#109: "vlm: N/M done, continuing" (INFO and the step's PROGRESS detail) only when
    windows are cached; a fresh run logs nothing new."""
    src = str(ocr_run.paths[0])
    windows = [vlm_windows.Window(src, t, t + 10.0, 1.0, []) for t in (0.0, 20.0, 40.0)]
    monkeypatch.setattr(pipeline, "game_prompt", lambda *a: "")
    monkeypatch.setattr(pipeline, "select_windows", lambda *a, **k: windows)
    monkeypatch.setattr(
        pipeline, "describe_windows", lambda server, *a, **k: ([], {"cached": cached})
    )

    def available(self):
        raise _Stop  # the VLM layer is about to load its model

    monkeypatch.setattr(pipeline.VisionRater, "available", available)
    with (
        caplog.at_level(logging.INFO),
        progress.run(("keyframes", "ranking")) as steps,
        pytest.raises(_Stop),
    ):
        steps.start("keyframes")
        pipeline.run_events(ocr_run.project, layers=("vlm",), ocr=OcrParams(game="zonewar"))
    details = [g["detail"] for g in _lines(capsys)]
    text = "vlm: 2/3 done, continuing"
    assert (text in caplog.messages) is bool(cached)
    assert (text in details) is bool(cached)
    assert not any("continuing" in m for m in caplog.messages if m != text)


# ----------------------------------------------------------------- --batch-size (#109, #713)
def test_analyze_gives_the_prefetch_its_batch_size(eligible, tmp_path) -> None:
    args = (get_style("gaming"), [tmp_path / "a.mp4"], (), tmp_path, True)
    with analyze._ocr_prefetch(*args) as pre:
        assert pre.batch_size == 0
    with analyze._ocr_prefetch(*args, batch_size=3) as pre:
        assert pre.batch_size == 3


@pytest.mark.parametrize(("size", "made"), [(0, 1), (1, 2), (2, 1)])
def test_the_worker_makes_its_engine_again_every_n_sources(fake, monkeypatch, size, made) -> None:
    """The background reader drops its engine at each batch boundary; the next read makes a
    new one. The results are those of one engine."""
    engines: list[object] = []

    class Engine:
        def __init__(self) -> None:
            engines.append(self)

    def detect(src, params, engine, *, make_engine=None, **kw):
        make_engine()
        return [{"t": 1.0, "type": "kill", "src": src.name}], {"frames": 4}

    monkeypatch.setattr(ocr_prefetch, "OcrEngine", Engine)
    monkeypatch.setattr(ocr_prefetch, "detect_ocr_events", detect)
    pre = fake.pre = ocr_prefetch.OcrPrefetch(
        fake.project, fake.paths, OcrParams(game="zonewar"), batch_size=size
    )
    for p in fake.paths:
        pre.offer(p, 60.0, True)
    got = [pre.take(fake.source(i), fake.params(), None) for i in range(2)]
    assert [evs[0]["src"] for evs, _ in got] == ["a.mp4", "b.mp4"]
    assert len(engines) == made


def test_batches_give_the_same_events_and_reads_with_the_prefetch(ocr_run, records) -> None:
    serial = list(ocr_run.run())
    serial_log = list(records.lines)
    records.lines.clear()
    ocr_run.reads.clear()
    ocr_run.seen.clear()
    pre = ocr_run.pre = ocr_prefetch.OcrPrefetch(
        ocr_run.project, ocr_run.paths, OcrParams(game="zonewar"), batch_size=1
    )
    for p in ocr_run.paths:
        pre.offer(p, 60.0, True)
    assert ocr_run.run(pre, batch_size=1) == serial
    assert ocr_run.reads == ["a.mp4", "b.mp4"]  # each read once, by the worker
    assert [x for x in records.lines if "continuing" not in x] == serial_log


@pytest.mark.parametrize("size", [0, 1])
def test_inline_reads_drop_the_prefetch_engine_at_each_batch(released, size) -> None:
    """Nothing read in the background: the inline reads use the prefetch's engine factory,
    which close() hands over with --batch-size, so run_events makes the engine again at its
    batch boundary; without it one engine serves both reads, as before."""
    pre = released.pre = ocr_prefetch.OcrPrefetch(
        released.project, released.paths, OcrParams(game="zonewar"), batch_size=size
    )
    pre.offer(released.paths[0], 60.0, False)
    with pytest.raises(_Stop):
        pipeline.run_events(
            released.project,
            layers=("ocr", "vlm"),
            ocr=OcrParams(game="zonewar"),
            prefetch=pre,
            batch_size=size,
        )
    if size:  # made, dropped at the boundary (free_torch), made again, dropped before the VLM
        assert released.log == ["made", "free_torch", "made", "free_torch", "vlm"]
    else:
        assert released.log == ["made", "free_torch", "vlm"]
    assert pre.make_engine.cache_info().currsize == 0


def test_take_records_the_seconds_the_main_thread_waited_per_source(fake) -> None:
    fake.gate = threading.Event()
    pre = fake.make()
    pre.offer(fake.paths[0], 60.0, True)
    threading.Timer(0.3, fake.gate.set).start()
    assert pre.take(fake.source(0), fake.params(), None) is not None
    assert pre.waits.keys() == {str(fake.paths[0])} and pre.waits[str(fake.paths[0])] >= 0.25
    assert pre.take(fake.source(1), fake.params(), None) is None  # not offered: read inline
    assert str(fake.paths[1]) not in pre.waits


def test_take_drops_a_result_whose_game_key_changed(fake, records, monkeypatch) -> None:
    """The HUD fit check gave the source another key after its background read (#822): the
    result and its log lines are dropped, the caller reads it again."""

    def read(src, cancel):
        logging.getLogger("cutpoint.events.killfeed").info("%s: read", src.name)
        return [{"t": 1.0, "type": "kill", "src": src.name}], {"game": "neon-drift"}

    fake.read = read
    keys = {"a.mp4": "neon-drift", "b.mp4": "beta-one"}  # b.mp4 was split off
    monkeypatch.setattr(
        ocr_prefetch, "game_key_for_path", lambda path, game, project: keys[Path(path).name]
    )
    pre = fake.make()
    pre.offer(fake.paths[0], 60.0, True)
    pre.offer(fake.paths[1], 60.0, True)
    assert pre.take(fake.source(0), fake.params(), None) is not None
    assert pre.take(fake.source(1), fake.params(), None) is None
    assert "a.mp4: read" in records.lines
    assert "b.mp4: read" not in records.lines  # the dropped read's lines are not replayed
