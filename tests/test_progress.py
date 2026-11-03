"""``cutpoint.progress``: the step plan and the ``PROGRESS {json}`` lines on stderr."""

from __future__ import annotations

import json

import pytest

from cutpoint import progress


def lines(capsys) -> list[dict]:
    """The PROGRESS lines written to stderr so far."""
    out = [ln for ln in capsys.readouterr().err.splitlines() if ln]
    assert all(ln.startswith("PROGRESS ") for ln in out), out
    return [json.loads(ln.removeprefix("PROGRESS ")) for ln in out]


class FakeClock:
    """A controllable ``time.monotonic``; Timer threads are replaced by a recorder."""

    def __init__(self, monkeypatch):
        self.now = 1000.0
        self.timers: list = []
        clock = self

        class FakeTimer:
            def __init__(self, wait, fn):
                self.wait, self.fn, self.cancelled, self.daemon = wait, fn, False, False
                clock.timers.append(self)

            def start(self):
                pass

            def cancel(self):
                self.cancelled = True

        monkeypatch.setattr(progress.time, "monotonic", lambda: self.now)
        monkeypatch.setattr(progress.threading, "Timer", FakeTimer)


# ----------------------------------------------------------------- pinned API
def test_step_names_are_the_stable_api():
    assert progress.ANALYZE_STEPS == (
        "transcribe", "reactions", "keyframes", "crashes", "ranking", "vision",
    )  # fmt: skip
    assert (*progress.ANALYZE_STEPS, "clips", "assemble") == progress.EDIT_STEPS
    assert progress.PANO_STEPS == ("scan", "views", "analyze", "events")
    assert progress.SKIPPABLE == ("reactions", "keyframes", "crashes")
    assert progress.LABELS == {
        "transcribe": "Transcription",
        "reactions": "Reactions",
        "keyframes": "Keyframes",
        "crashes": "Crashes",
        "ranking": "Ranking",
        "vision": "Vision rating",
        "clips": "Clips",
        "assemble": "Assemble",
        "scan": "Read 360 clips",
        "views": "360 views",
        "analyze": "Transcription and ranking",
        "events": "Events and re-rank",
    }


def test_line_keys_order_and_values(capsys):
    r = progress.Run(progress.ANALYZE_STEPS, active=True)
    r.start("transcribe", "first")
    r.done()
    r.start("reactions")
    r.fail("boom")
    r.skip("keyframes", "why not")
    got = lines(capsys)
    keys = ["step", "index", "total", "label", "state", "pct", "detail"]
    assert all(list(g) == keys for g in got)
    assert got[0] == {
        "step": "transcribe", "index": 1, "total": 6, "label": "Transcription",
        "state": "running", "pct": None, "detail": "first",
    }  # fmt: skip
    assert [(g["step"], g["state"]) for g in got] == [
        ("transcribe", "running"),
        ("transcribe", "done"),
        ("reactions", "running"),
        ("reactions", "failed"),
        ("keyframes", "skipped"),
    ]
    assert got[1]["pct"] == 100  # done is 100
    assert got[4]["detail"] == "why not"
    assert {g["state"] for g in got} <= {"running", "done", "skipped", "failed"}


# ----------------------------------------------------------------- gating
@pytest.mark.parametrize(("value", "on"), [(None, False), ("", False), ("0", False), ("1", True)])
def test_enabled_only_with_env_1(monkeypatch, capsys, value, on):
    if value is not None:
        monkeypatch.setenv(progress.ENV, value)
    assert progress.enabled() is on
    with progress.run(progress.ANALYZE_STEPS) as r:
        r.start("transcribe")
    assert bool(lines(capsys)) is on


def test_nothing_is_emitted_when_not_used(monkeypatch, capsys):
    monkeypatch.setenv(progress.ENV, "1")
    with progress.run(progress.ANALYZE_STEPS, use=False) as r:
        r.start("transcribe")
        progress.update(50)
    assert capsys.readouterr().err == ""
    assert progress.current() is None


# ----------------------------------------------------------------- throttling
def test_running_updates_are_throttled_and_the_last_is_flushed(monkeypatch, capsys):
    clock = FakeClock(monkeypatch)
    r = progress.Run(progress.ANALYZE_STEPS, active=True)
    r.start("transcribe")
    for pct in (10, 20, 30):  # all within the interval
        r.update(pct)
    assert [g["pct"] for g in lines(capsys)] == [None]  # only the start line
    assert len(clock.timers) == 1  # one flush timer however many updates
    clock.now += progress.MIN_INTERVAL_S + 0.01
    clock.timers[0].fn()
    flushed = lines(capsys)
    assert [(g["state"], g["pct"]) for g in flushed] == [("running", 30)]
    clock.now += 1
    r.update(40)  # interval over: written at once
    assert [g["pct"] for g in lines(capsys)] == [40]


def test_state_changes_are_never_throttled_and_drop_the_held_update(monkeypatch, capsys):
    clock = FakeClock(monkeypatch)
    r = progress.Run(progress.ANALYZE_STEPS, active=True)
    r.start("transcribe")
    r.update(10)  # held back
    r.done()
    r.start("reactions")
    r.done()  # four state changes in zero time
    got = lines(capsys)
    assert [(g["step"], g["state"]) for g in got] == [
        ("transcribe", "running"),
        ("transcribe", "done"),
        ("reactions", "running"),
        ("reactions", "done"),
    ]
    assert clock.timers[0].cancelled  # the held update never goes out after the done line


def test_continuing_is_logged_and_sent_at_once_then_updates_throttle_again(
    monkeypatch, capsys, caplog
):
    """#109: the "N/M done, continuing" detail is not held back (the per-source updates
    that follow would replace it); nothing at all for N = 0 or N = M."""
    clock = FakeClock(monkeypatch)
    monkeypatch.setenv(progress.ENV, "1")
    with caplog.at_level("INFO"), progress.run(progress.ANALYZE_STEPS) as r:
        r.start("transcribe")
        progress.continuing("transcribe", 0, 3)  # a fresh run: as before
        progress.continuing("transcribe", 3, 3)  # all cached, nothing to continue: as before
        assert lines(capsys)[1:] == [] and not caplog.messages
        r.update(10)  # held back
        progress.continuing("transcribe", 2, 3)
        got = lines(capsys)
        assert [(g["pct"], g["detail"]) for g in got] == [(10, "transcribe: 2/3 done, continuing")]
        assert {g["state"] for g in got} == {"running"}  # no new state
        assert clock.timers[0].cancelled
        r.update(20, "c.mp4")  # within the interval again: held
        assert lines(capsys) == []
        r.done()
    assert caplog.messages == ["transcribe: 2/3 done, continuing"]


def test_continuing_without_a_run_only_logs(caplog):
    with caplog.at_level("INFO"):
        progress.continuing("vlm", 4, 9)
    assert caplog.messages == ["vlm: 4/9 done, continuing"]


def test_update_without_a_running_step_and_pct_clamp(capsys):
    r = progress.Run(progress.ANALYZE_STEPS, active=True)
    r.update(50)  # no step yet
    assert lines(capsys) == []
    r.start("transcribe")
    r._last = float("-inf")
    r.update(250)
    assert lines(capsys)[-1]["pct"] == 100


# ----------------------------------------------------------------- failures of the output
class Broken:
    def __init__(self, exc):
        self.exc, self.writes = exc, 0

    def write(self, _):
        self.writes += 1
        raise self.exc

    def flush(self):
        pass


@pytest.mark.parametrize("exc", [BrokenPipeError(), OSError(22, "bad"), ValueError("closed")])
def test_a_broken_stderr_turns_progress_off_without_raising(monkeypatch, exc):
    bad = Broken(exc)
    monkeypatch.setattr(progress.sys, "stderr", bad)
    r = progress.Run(progress.ANALYZE_STEPS, active=True)
    r.start("transcribe")
    r.done()
    r.start("reactions")
    r.fail("x")
    r.finish()
    assert r.active is False
    assert bad.writes == 1  # tried once, then silent


def test_a_missing_stderr_does_not_raise(monkeypatch):
    monkeypatch.setattr(progress.sys, "stderr", None)
    r = progress.Run(progress.ANALYZE_STEPS, active=True)
    r.start("transcribe")
    assert r.active is False


# ----------------------------------------------------------------- run(): errors and cancel
@pytest.mark.parametrize(
    ("error", "detail"),
    [(KeyboardInterrupt(), "cancelled"), (RuntimeError("no audio in x.mp4"), "no audio in x.mp4")],
    ids=["ctrl-c", "run-stopping-error"],
)
def test_an_error_fails_the_current_step(monkeypatch, capsys, error, detail):
    """Ctrl+C fails the running step as cancelled; a run-stopping error with its text."""
    monkeypatch.setenv(progress.ENV, "1")
    with pytest.raises(type(error)), progress.run(progress.ANALYZE_STEPS) as r:
        r.start("transcribe")
        r.done()
        r.start("reactions")
        raise error
    got = lines(capsys)
    assert (got[-1]["step"], got[-1]["state"], got[-1]["detail"]) == ("reactions", "failed", detail)
    assert not any(g["step"] == "keyframes" for g in got)  # later steps are not announced
    assert progress.current() is None


def test_an_error_between_steps_fails_the_next_step_and_only_once(monkeypatch, capsys):
    monkeypatch.setenv(progress.ENV, "1")
    with pytest.raises(ValueError), progress.run(progress.ANALYZE_STEPS) as r:
        r.start("transcribe")
        r.done()
        raise ValueError
    got = lines(capsys)
    failed = [g for g in got if g["state"] == "failed"]
    assert [(g["step"], g["detail"]) for g in failed] == [("reactions", "ValueError")]


def test_step_context_fails_once_when_the_block_raises(monkeypatch, capsys):
    monkeypatch.setenv(progress.ENV, "1")
    with (
        pytest.raises(RuntimeError, match="bad"),
        progress.run(progress.ANALYZE_STEPS) as r,
        r.step("transcribe"),
    ):
        raise RuntimeError("bad")
    failed = [g for g in lines(capsys) if g["state"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["detail"] == "bad"


def test_a_failed_step_does_not_stop_later_steps(monkeypatch, capsys):
    monkeypatch.setenv(progress.ENV, "1")
    with progress.run(progress.ANALYZE_STEPS) as r:
        r.start("transcribe")
        r.start("reactions")
        r.fail("layer broke")
        r.start("keyframes")
    got = lines(capsys)
    assert [(g["step"], g["state"]) for g in got if g["step"] in ("reactions", "keyframes")] == [
        ("reactions", "running"),
        ("reactions", "failed"),
        ("keyframes", "running"),
        ("keyframes", "done"),
    ]


# ----------------------------------------------------------------- skipped steps
def test_every_step_of_a_finished_run_is_emitted_and_skip_reasons_kept(monkeypatch, capsys):
    monkeypatch.setenv(progress.ENV, "1")
    with progress.run(progress.EDIT_STEPS) as r:
        r.start("transcribe")
        r.skip("crashes", "not action-cam footage")
        r.start("ranking")  # reactions and keyframes were never started
    final: dict[str, dict] = {}
    for g in lines(capsys):
        final[g["step"]] = g
    assert list(final) == list(progress.EDIT_STEPS)
    assert final["crashes"]["state"] == "skipped"
    assert final["crashes"]["detail"] == "not action-cam footage"
    assert final["reactions"]["state"] == "skipped"
    assert final["keyframes"]["detail"] == "not run"
    assert final["ranking"]["state"] == "done"
    assert final["assemble"]["state"] == "skipped"
    assert {g["total"] for g in final.values()} == {8}


def test_a_planned_skip_is_emitted_in_step_order(capsys):
    r = progress.Run(progress.ANALYZE_STEPS, active=True)
    r.skip("keyframes", "later")  # transcribe not emitted yet: held back
    assert lines(capsys) == []
    r.start("transcribe")
    r.done()
    r.start("reactions")
    r.done()
    r.start("crashes")  # keyframes' turn passed: emitted before crashes
    assert [(g["step"], g["state"]) for g in lines(capsys)][-2:] == [
        ("keyframes", "skipped"),
        ("crashes", "running"),
    ]


def test_nested_run_reuses_the_active_one(monkeypatch, capsys):
    monkeypatch.setenv(progress.ENV, "1")
    with progress.run(progress.EDIT_STEPS) as outer:
        with progress.run(progress.ANALYZE_STEPS) as inner:
            assert inner is outer
            inner.start("transcribe")
        assert progress.current() is outer
    assert progress.current() is None
    assert {g["total"] for g in lines(capsys)} == {8}
