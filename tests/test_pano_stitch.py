"""Insta360 SDK wrapper: output parsing, argument validation, discovery, process handling.

Everything here uses minimal SYNTHETIC strings (one marker per case, built from the
parser's own constants), not SDK output; the real SDK is never needed.
``run_sdk`` / ``_stitch_one`` / ``probe_sdk`` run a tiny fake child
(``sys.executable`` + a script written below). ``stitch._verify`` (ffprobe based)
is monkeypatched so no ffmpeg is required.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import write
from cutpoint import config, paths
from cutpoint.pano import cli as pano_cli
from cutpoint.pano import project, stitch
from cutpoint.pano.x3 import X3Clip

# --------------------------------------------------------------------------- synthetic output
P = stitch.PROGRESS_KEY


def _pct(n: int) -> str:
    return f"{P} = {n}%"


def _feature_block(**features: str) -> str:
    body = "".join(f"  {k}: {v}\n" for k, v in features.items())
    return f"{stitch.FEATURES_MARKER}\n{body}{stitch.BLOCK_END}\n"


def _result(text: str) -> stitch.SdkResult:
    return stitch.parse_sdk_output(text)


def _done(**features: str) -> stitch.SdkResult:
    """A finished run (progress 100) with the given feature statuses."""
    return stitch.SdkResult(progress=100, features=dict(features))


# --------------------------------------------------------------------------- parsing


class TestParseProgress:
    def test_single_and_spacing_variants(self):
        assert stitch.parse_progress(_pct(7)) == [7]
        assert stitch.parse_progress(f"{P}=42 %") == [42]


class TestParseExportSummary:
    def test_pairs(self):
        got = stitch.parse_export_summary(f"{stitch.EXPORT_MARKER} A:1 | B:x:y | C:")
        assert got == {"A": "1", "B": "x:y", "C": ""}


class TestParseSdkOutput:
    def test_progress(self):
        res = _result(f"{_pct(0)}\r{_pct(50)}\r{_pct(100)}\n")
        assert res.progress == 100 and res.finished
        assert not res.ended and res.errors == []

    def test_end_marker(self):
        res = _result(f"{stitch.END_MARKER}\n")
        assert res.ended and not res.finished

    def test_cr_separated_progress_gives_separate_lines(self):
        res = _result(f"{_pct(10)}\r{_pct(20)}\r{_pct(30)}\r")
        assert res.lines == [_pct(10), _pct(20), _pct(30)]
        assert res.progress == 30

    def test_progress_is_the_maximum_not_the_last(self):
        assert _result(f"{_pct(80)}\n{_pct(20)}\n").progress == 80

    def test_progress_stops_before_100(self):
        res = _result(f"{_pct(10)}\r{_pct(63)}\n")
        assert res.progress == 63
        assert not res.finished

    def test_error_lines(self):
        res = _result("error: cannot open input\nERROR:  second one \n")
        assert res.errors == ["cannot open input", "second one"]

    def test_error_text_on_the_same_line_as_progress(self):
        res = _result(f"{_pct(5)} error: glued failure\n")
        assert res.progress == 5
        assert res.errors == ["glued failure"]

    def test_error_mid_line_is_not_an_error(self):
        assert _result("note: no error: here\n").errors == []

    def test_feature_status_values(self):
        text = _feature_block(a="ON", b="OFF", c="FAILED", x_y="3")
        assert _result(text).features == {"a": "ON", "b": "OFF", "c": "FAILED", "x_y": "3"}

    def test_feature_block_end_and_following_lines_parse_normally(self):
        res = _result(_feature_block(a="ON") + "error: late\n")
        assert res.features == {"a": "ON"}
        assert res.errors == ["late"]

    def test_feature_lookalike_outside_block_ignored(self):
        assert _result("a: ON\n").features == {}

    @pytest.mark.parametrize("text", [""])
    def test_empty_or_garbage(self, text):
        res = _result(text)
        assert res.progress == -1
        assert not res.finished and not res.ended
        assert res.errors == [] and res.features == {} and res.export == {}
        assert res.cost_s is None


# --------------------------------------------------------------------------- failure reasons
class TestRequestedFeatures:
    def test_normalised_and_deduplicated(self):
        args = [
            "-enable_flowstate",
            "-enable_Stitch-Fusion",
            "-enable_flowstate",
            "-bitrate",
            "1",
            "-stitch_type",
            "optflow",
            "-enable_h265_encoder",
        ]
        assert stitch.requested_features(args) == ["flowstate", "stitchfusion", "h265encoder"]


class TestFailureReasons:
    BIG = 5_000_000

    def _ok(self, **kw):
        res = _done(flowstate="ON", stitchfusion="ON", denoise="N/A", colorplus="OFF")
        args = {"returncode": 0, "requested": ["flowstate"], "output_bytes": self.BIG}
        args.update(kw)
        return stitch.failure_reasons(
            res, args["returncode"], args["requested"], args["output_bytes"]
        )

    def test_success_has_no_reasons(self):
        assert self._ok() == []

    def test_exit_zero_with_error_line_is_a_failure(self):
        res = _result(f"{_pct(100)}\nerror: encoder died\n")
        reasons = stitch.failure_reasons(res, 0, [], self.BIG)
        assert reasons == ["error: encoder died"]

    def test_incomplete_progress(self):
        res = _result(f"{_pct(40)}\n")
        reasons = stitch.failure_reasons(res, 0, [], self.BIG)
        assert reasons == ["stitch did not complete (progress 40%)"]
        reasons = stitch.failure_reasons(_result(""), 0, [], self.BIG)
        assert reasons == ["stitch did not complete (no progress)"]

    def test_killed(self):
        assert "process killed" in self._ok(returncode=None)

    def test_nonzero_exit_and_unsigned_windows_code(self):
        assert self._ok(returncode=3) == ["exit code 3 (bad arguments?)"]
        assert self._ok(returncode=2**32 - 2) == ["exit code -2 (bad arguments?)"]

    @pytest.mark.parametrize(
        "status, failed",
        [
            ("OFF", True),
            ("FAILED", True),
            ("failed (x)", True),
            ("0", True),
            ("3", True),
            ("1", False),
            ("N/A", False),
        ],
    )
    def test_feature_status(self, status, failed):
        reasons = stitch.failure_reasons(_done(a=status), 0, ["a"], self.BIG)
        assert bool(reasons) is failed

    def test_unrequested_off_feature_is_fine(self):
        assert stitch.failure_reasons(_done(a="OFF"), 0, [], self.BIG) == []

    def test_missing_and_tiny_output(self):
        assert self._ok(output_bytes=None) == ["no output file"]
        tiny = self._ok(output_bytes=10)
        assert len(tiny) == 1 and tiny[0].startswith("output only 10 bytes")


class TestFeaturesSummary:
    def test_hides_off_and_unknown_shows_the_rest(self):
        feats = {"a": "ON", "b": "OFF", "c": "0", "d": "-1", "e": "unknown", "f": " off "}
        assert stitch.features_summary(feats) == "a=ON"

    def test_numeric_codes_on_and_failed_are_shown(self):
        feats = {"a": "1", "b": "3", "c": "2", "d": "FAILED"}
        assert stitch.features_summary(feats) == "a=1, b=FAILED, c=2, d=FAILED"


class TestParseHelpVersion:
    def test_found(self):
        assert stitch.parse_help_version(f"x\n{stitch.VERSION_KEY} 1.2.3\ny") == "1.2.3"

    @pytest.mark.parametrize("text", ["no version here"])
    def test_missing(self, text):
        assert stitch.parse_help_version(text) is None


class TestProgressThrottle:
    def test_throttle(self):
        t = stitch.ProgressThrottle(step=5, every_s=10)
        assert t.should_log(0, 0.0)
        assert not t.should_log(4, 1.0)  # under both thresholds
        assert t.should_log(5, 2.0)  # step reached
        assert not t.should_log(6, 5.0)
        assert t.should_log(7, 12.0)  # time reached
        assert not t.should_log(8, 12.5)
        assert t.should_log(100, 12.6)  # 100% always logs
        assert not t.should_log(100, 99.0)  # never repeats


# --------------------------------------------------------------------------- fake child
FAKE_SDK = r"""
import sys, time
mode, part, ready = sys.argv[1], sys.argv[2], sys.argv[3]
out = sys.stdout.buffer
def emit(text):
    out.write(text.encode("utf-8"))
    out.flush()

def pct(n):
    return f"{MARK['progress']} = {n}%"

def write_output(n=1_200_000):
    with open(part, "wb") as f:
        f.write(b"\0" * n)

sys.stderr.write("synthetic stderr line\n")
sys.stderr.flush()
if mode == "ok":
    emit(pct(0) + "\r" + pct(50) + "\r" + pct(100) + "\r\n")
    write_output()
    emit("synthetic log café 日本\n")
elif mode == "error":
    emit(pct(100) + "\r\n")
    write_output()
    emit("error: synthetic failure é\n")
elif mode == "feature_failed":
    emit(pct(100) + "\r\n")
    write_output()
    emit(MARK["features"] + "\n  flowstate: FAILED\n" + MARK["block_end"] + "\n")
elif mode == "version":
    emit("synthetic help\n" + MARK["version"] + " 9.8.7\n")
elif mode == "noversion":
    emit("synthetic help without a version\n")
elif mode == "hang":
    emit(pct(12) + "\r\nSYNTHETIC-CAPTURED-LINE é\n")
    write_output(10)
    open(ready, "w").close()
    time.sleep(120)
"""

# Markers are injected from the parser's constants, not spelled out here.
FAKE_MARKERS = {
    "progress": stitch.PROGRESS_KEY,
    "features": stitch.FEATURES_MARKER,
    "block_end": stitch.BLOCK_END,
    "version": stitch.VERSION_KEY,
}


class _GatedPopen(subprocess.Popen):
    """Popen whose first ``poll()`` waits until the fake child says it is ready.

    Makes the timeout / Ctrl-C tests deterministic without sleeping: by the time
    ``run_sdk`` starts to judge the child, its first output is already flushed.
    ``interrupt=True`` makes the poll loop's ``wait(timeout=0.5)`` raise
    ``KeyboardInterrupt``, as Ctrl-C would.
    """

    ready: Path
    interrupt = False

    def poll(self):
        if not type(self).ready.exists() and self.returncode is None:
            deadline = threading.Event()
            for _ in range(600):
                if type(self).ready.exists():
                    break
                deadline.wait(0.05)
        return super().poll()

    def wait(self, timeout=None):
        if type(self).interrupt and timeout == 0.5:
            type(self).interrupt = False
            raise KeyboardInterrupt
        return super().wait(timeout=timeout)


@pytest.fixture
def fake(tmp_path: Path):
    script = tmp_path / "fake_sdk.py"
    script.write_text(f"MARK = {json.dumps(FAKE_MARKERS)}\n{FAKE_SDK}", encoding="utf-8")
    ready = tmp_path / "ready.flag"

    def cmd(mode: str, part: Path, *extra: str) -> list[str]:
        # "-output <part>" as in a real command (the fake ignores it; the sidecar needs it)
        return [
            sys.executable,
            str(script),
            mode,
            str(part),
            str(ready),
            "-output",
            str(part),
            *extra,
        ]

    cmd.ready = ready  # type: ignore[attr-defined]
    return cmd


@pytest.fixture
def gated(monkeypatch: pytest.MonkeyPatch, fake):
    _GatedPopen.ready = fake.ready
    _GatedPopen.interrupt = False
    monkeypatch.setattr(stitch.subprocess, "Popen", _GatedPopen)
    yield _GatedPopen
    _GatedPopen.interrupt = False


class TestRunSdk:
    def test_captures_stdout_stderr_and_exit_code(self, fake, tmp_path):
        run = stitch.run_sdk(fake("ok", tmp_path / "o.mp4"), label="t")
        assert run.returncode == 0 and not run.timed_out
        res = stitch.parse_sdk_output(run.stdout)
        assert res.finished and res.errors == []
        assert "café 日本" in run.stdout
        assert "synthetic stderr line" in run.stderr

    def test_timeout_kills_and_keeps_output(self, fake, tmp_path, gated):
        run = stitch.run_sdk(fake("hang", tmp_path / "o.mp4"), timeout_s=0.01)
        assert run.timed_out and run.returncode is None
        assert "SYNTHETIC-CAPTURED-LINE é" in run.stdout

    def test_interrupt_raises_with_captured_output(self, fake, tmp_path, gated):
        gated.interrupt = True
        with pytest.raises(stitch.SdkInterrupted) as ei:
            stitch.run_sdk(fake("hang", tmp_path / "o.mp4"))
        assert isinstance(ei.value, KeyboardInterrupt)
        assert ei.value.run.returncode is None
        assert "SYNTHETIC-CAPTURED-LINE" in ei.value.run.stdout


@pytest.fixture
def help_as(monkeypatch: pytest.MonkeyPatch, fake, tmp_path: Path):
    """Make ``probe_sdk``'s ``<exe> -help`` run the fake child in the given mode."""
    real_run = subprocess.run

    def use(mode: str) -> None:
        def run(cmd, **kw):
            assert list(cmd[1:]) == ["-help"]
            return real_run(fake(mode, tmp_path / "unused.mp4"), **kw)

        monkeypatch.setattr(stitch.subprocess, "run", run)

    return use


class TestProbeSdk:
    def test_version_line(self, help_as):
        help_as("version")
        got = stitch.probe_sdk(Path("sdk.exe"))
        assert got["runs"] is True and got["version"] == "9.8.7" and got["error"] is None
        assert isinstance(got["seconds"], float) and got["seconds"] >= 0

    def test_hang_times_out(self, help_as):
        help_as("hang")
        got = stitch.probe_sdk(Path("sdk.exe"), timeout_s=0.5)
        assert got["runs"] is False and got["version"] is None
        assert "timed out" in got["error"]
        assert got["seconds"] < 30

    def test_missing_exe(self, tmp_path):
        got = stitch.probe_sdk(tmp_path / "missing" / "MediaSDKTest.exe")
        assert got["runs"] is False and got["version"] is None
        assert got["error"]


# --------------------------------------------------------------------------- _stitch_one
@pytest.fixture
def clip() -> X3Clip:
    return X3Clip(
        clip_id="001",
        start_local="2026-01-01T10:00:00",
        mode="5.7k",
        lrv=None,
        insv=["VID_20260101_100000_00_001.insv", "VID_20260101_100000_10_001.insv"],
        frames=30,
        duration=1.0,
    )


@pytest.fixture
def paths_(tmp_path: Path):
    out_dir = tmp_path / "stitched"
    out_dir.mkdir()
    out = out_dir / "VID_20260101_100000_001.equirect.mp4"
    part = out.with_name(out.name + ".partial.mp4")
    return out_dir, out, part


@pytest.fixture(autouse=True)
def no_ffprobe(monkeypatch: pytest.MonkeyPatch):
    """Replace the ffprobe-based verification (no ffmpeg needed in tests)."""
    monkeypatch.setattr(stitch, "_verify", lambda clip, path: "ok")


def _log(out_dir: Path) -> str:
    logs = list((out_dir / "logs").glob("*.log"))
    assert len(logs) == 1, logs
    return logs[0].read_text(encoding="utf-8")


class TestStitchOne:
    def test_success(self, fake, clip, paths_):
        out_dir, out, part = paths_
        cmd = fake("ok", part, "-enable_flowstate", "-enable_stitchfusion")
        rec = stitch._stitch_one(clip, cmd, part, out, out_dir, None)
        assert rec["status"] == "ok"
        assert out.is_file() and not part.exists()
        assert out.stat().st_size == 1_200_000
        log_file = out_dir / "logs" / "VID_20260101_100000_001.log"
        assert Path(rec["log_file"]) == log_file and log_file.is_file()
        text = _log(out_dir)
        assert "result: ok; verify: ok" in text
        assert "===== stdout =====" in text and "===== stderr =====" in text
        assert "synthetic log" in text and "synthetic stderr line" in text
        assert "日本" in text  # UTF-8 preserved in the log
        assert rec["sdk"]["progress"] == 100 and rec["sdk"]["errors"] == []

    def test_success_writes_sidecar(self, fake, clip, paths_):
        out_dir, out, part = paths_
        cmd = fake("ok", part, "-stitch_type", "optflow", "-enable_stitchfusion")
        stitch._stitch_one(clip, cmd, part, out, out_dir, None)
        side = out_dir / "VID_20260101_100000_001.equirect.json"
        assert project.equirect_sidecar(out) == side
        meta = json.loads(side.read_text(encoding="utf-8"))
        assert meta == {
            "version": project.EQUIRECT_META_VERSION,
            "stitch_args": ["-stitch_type", "optflow", "-enable_stitchfusion"],
            "sdk_version": None,  # the fake's path has no MediaSDK-<version> folder
        }
        assert project.equirect_flowstate(out) is False

    def test_failure_writes_no_sidecar(self, fake, clip, paths_):
        out_dir, out, part = paths_
        stitch._stitch_one(clip, fake("error", part), part, out, out_dir, None)
        assert not project.equirect_sidecar(out).exists()

    def test_verification_notes_become_the_status(self, fake, clip, paths_, monkeypatch):
        out_dir, out, part = paths_
        monkeypatch.setattr(stitch, "_verify", lambda c, p: "not 2:1 (10x10)")
        rec = stitch._stitch_one(clip, fake("ok", part), part, out, out_dir, None)
        assert rec["status"] == "not 2:1 (10x10)"
        assert out.is_file()

    def test_stale_partial_is_replaced(self, fake, clip, paths_):
        out_dir, out, part = paths_
        part.write_bytes(b"stale")
        rec = stitch._stitch_one(clip, fake("ok", part), part, out, out_dir, None)
        assert rec["status"] == "ok" and out.stat().st_size == 1_200_000

    def test_exit_zero_with_error_line_fails_and_removes_partial(self, fake, clip, paths_):
        out_dir, out, part = paths_
        rec = stitch._stitch_one(clip, fake("error", part), part, out, out_dir, None)
        assert rec["status"] == "failed"
        assert "error: synthetic failure é" in rec["error"]
        assert not part.exists() and not out.exists()
        assert "synthetic failure" in rec["log"]
        text = _log(out_dir)
        assert "result: FAILED:" in text and "synthetic failure é" in text

    def test_timeout_fails_with_captured_output_in_log(self, fake, clip, paths_, gated):
        out_dir, out, part = paths_
        rec = stitch._stitch_one(clip, fake("hang", part), part, out, out_dir, 0.01)
        assert rec["status"] == "failed"
        assert rec["error"].startswith("timed out after 0s (pano.stitch_timeout_s)")
        assert "process killed" in rec["error"]
        assert not part.exists() and not out.exists()
        text = _log(out_dir)
        assert "TIMED OUT" in text
        assert "SYNTHETIC-CAPTURED-LINE é" in text

    def test_interrupt_removes_partial_logs_output_and_reraises(self, fake, clip, paths_, gated):
        out_dir, out, part = paths_
        gated.interrupt = True
        with pytest.raises(stitch.SdkInterrupted):
            stitch._stitch_one(clip, fake("hang", part), part, out, out_dir, None)
        assert not part.exists() and not out.exists()
        text = _log(out_dir)
        assert "interrupted (Ctrl-C); partial output removed" in text
        assert "SYNTHETIC-CAPTURED-LINE é" in text

    def test_plain_keyboard_interrupt_still_cleans_up(self, clip, paths_, monkeypatch):
        out_dir, out, part = paths_
        part.write_bytes(b"x")

        def boom(*a, **k):
            raise KeyboardInterrupt

        monkeypatch.setattr(stitch, "run_sdk", boom)
        with pytest.raises(KeyboardInterrupt):
            stitch._stitch_one(clip, ["sdk"], part, out, out_dir, None)
        assert not part.exists()
        assert "interrupted" in _log(out_dir)

    def test_cannot_start_returns_failed_record(self, clip, paths_, tmp_path):
        out_dir, out, part = paths_
        cmd = [str(tmp_path / "missing-sdk.exe"), "-inputs", "a"]
        rec = stitch._stitch_one(clip, cmd, part, out, out_dir, None)
        assert rec["status"] == "failed" and rec["error"].startswith("could not start:")
        assert "could not start" in _log(out_dir)

    def test_too_small_output_fails(self, fake, clip, paths_, monkeypatch):
        out_dir, out, part = paths_
        monkeypatch.setattr(stitch, "_min_bytes", lambda c, args=None: 5_000_000)
        rec = stitch._stitch_one(clip, fake("ok", part), part, out, out_dir, None)
        assert rec["status"] == "failed" and "output only" in rec["error"]
        assert not part.exists() and not out.exists()

    def test_low_bitrate_lowers_the_size_floor(self, fake, clip, paths_):
        out_dir, out, part = paths_
        clip.duration = 100.0  # default floor: 12.5 MB > the fake's 1.2 MB
        rec = stitch._stitch_one(clip, fake("ok", part), part, out, out_dir, None)
        assert rec["status"] == "failed" and "output only" in rec["error"]
        rec = stitch._stitch_one(
            clip, fake("ok", part, "-bitrate", "320000"), part, out, out_dir, None
        )
        assert rec["status"] == "ok"

    def test_locked_partial_is_left_with_a_warning(self, fake, clip, paths_, monkeypatch, caplog):
        out_dir, out, part = paths_
        real_unlink = Path.unlink
        calls = []

        def unlink(self, missing_ok=False):
            if self == part:
                calls.append(self)
                if len(calls) > 1:  # the first call is the stale-partial cleanup
                    raise PermissionError(32, "in use by another process")
            return real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", unlink)
        monkeypatch.setattr(stitch, "PARTIAL_RM_DELAY_S", 0.0)
        with caplog.at_level("WARNING", logger="cutpoint.pano.stitch"):
            rec = stitch._stitch_one(clip, fake("error", part), part, out, out_dir, None)
        assert rec["status"] == "failed"  # no exception despite the locked file
        assert len(calls) == 1 + stitch.PARTIAL_RM_ATTEMPTS
        assert any("could not remove partial output" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- misc helpers
class _LockedFile:
    """Stands in for a ``Path`` whose ``unlink`` fails ``fails`` times."""

    def __init__(self, fails: int) -> None:
        self.fails = fails
        self.calls = 0

    def unlink(self, missing_ok: bool = False) -> None:
        self.calls += 1
        if self.calls <= self.fails:
            raise PermissionError(32, "in use by another process")

    def __str__(self) -> str:
        return "locked.partial.mp4"


class TestRemovePartial:
    def test_retries_until_it_succeeds(self, caplog):
        f = _LockedFile(fails=2)
        with caplog.at_level("WARNING", logger="cutpoint.pano.stitch"):
            stitch._remove_partial(f, attempts=5, delay_s=0)  # type: ignore[arg-type]
        assert f.calls == 3
        assert not caplog.records


class TestHelpers:
    def test_min_bytes_floor_and_rate(self, clip):
        assert stitch._min_bytes(clip) == stitch.MIN_OUTPUT_BYTES
        clip.duration = 100.0
        assert stitch._min_bytes(clip) == 100 * stitch.MIN_OUTPUT_BYTES_PER_S

    @pytest.mark.parametrize(
        "args, rate",
        [
            (["-bitrate", "320000"], 10_000),  # 320 kb/s -> a quarter of 40 kB/s
            (["-bitrate", "100000000"], stitch.MIN_OUTPUT_BYTES_PER_S),  # never raised
            (None, stitch.MIN_OUTPUT_BYTES_PER_S),
        ],
    )
    def test_min_bytes_follows_a_low_bitrate(self, clip, args, rate):
        clip.duration = 1000.0
        assert stitch._min_bytes(clip, args) == max(stitch.MIN_OUTPUT_BYTES, 1000 * rate)

    def test_min_bytes_keeps_the_absolute_floor(self, clip):
        assert stitch._min_bytes(clip, ["-bitrate", "8"]) == stitch.MIN_OUTPUT_BYTES

    def test_require_sdk_missing_message(self):
        with pytest.raises(stitch.SdkNotConfigured, match="Media SDK not found"):
            stitch.require_sdk()

    def test_require_sdk_configured_but_absent(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CUTPOINT_INSTA360_SDK", str(tmp_path / "nope.exe"))
        with pytest.raises(stitch.SdkNotConfigured, match="does not exist"):
            stitch.require_sdk()

    def test_stitch_command_reads_moved_footage_through_source_remap(
        self, home, tmp_path, monkeypatch
    ):
        old, new = tmp_path / "old", tmp_path / "new"
        names = ["a_00_.insv", "a_10_.insv"]
        for n in names:
            write(new / n)
        write(home / "config.json", json.dumps({"source_remap": {str(old): str(new)}}))
        config.get_config.cache_clear()
        monkeypatch.setattr(config, "_prefix_forms", lambda p: (config.norm_path(p),))
        clip = X3Clip("001", "t", "5.7k", None, [str(old / n) for n in names])
        cmd = stitch.stitch_command(tmp_path / "sdk", clip, tmp_path / "o.mp4")
        assert cmd[1:4] == ["-inputs", *(str(new / n) for n in names)]

    def test_stitch_command_validates_extra_args(self, tmp_path):
        clip = X3Clip("001", "t", "5.7k", None, ["a_00_.insv", "a_10_.insv"])
        cmd = stitch.stitch_command(tmp_path / "sdk", clip, tmp_path / "o.mp4", ["-enable_denoise"])
        assert cmd[1:4] == ["-inputs", "a_00_.insv", "a_10_.insv"]
        assert cmd[-1] == "-enable_denoise"
        with pytest.raises(ValueError, match="stitch_type"):
            stitch.stitch_command(tmp_path / "sdk", clip, tmp_path / "o.mp4", ["-stitch_type", "x"])


# --------------------------------------------------------------------------- args (project.py)
def _config(**pano) -> None:
    cfg = paths.config_file()
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps({"pano": pano}), encoding="utf-8")


class TestValidateStitchArgs:
    def test_valid_args_pass_through(self):
        args = ["-stitch_type", "optflow", "-output_size", "5760x2880"]
        assert project.validate_stitch_args(args) == args

    @pytest.mark.parametrize("bad", ["template"])
    def test_rejected_types(self, bad):
        with pytest.raises(ValueError, match="Unsupported -stitch_type"):
            project.validate_stitch_args(["-stitch_type", bad])

    @pytest.mark.parametrize("size", ["5760x2000", "big"])
    def test_bad_output_sizes(self, size):
        with pytest.raises(ValueError, match="Invalid -output_size"):
            project.validate_stitch_args(["-output_size", size])


class TestStitchArgs:
    def test_default(self):
        assert project.stitch_args() == list(project.DEFAULT_STITCH_ARGS)

    def test_configured(self):
        _config(stitch_args=["-stitch_type", "aistitch", "-output_size", "2880x1440"])
        assert project.stitch_args() == ["-stitch_type", "aistitch", "-output_size", "2880x1440"]

    def test_template_rejected(self):
        _config(stitch_args=["-stitch_type", "template"])
        with pytest.raises(ValueError, match="template"):
            project.stitch_args()

    def test_not_a_list(self):
        _config(stitch_args="-stitch_type optflow")
        with pytest.raises(ValueError, match="must be a list"):
            project.stitch_args()

    @pytest.mark.parametrize("item", [None])
    def test_non_string_items_rejected(self, item):
        _config(stitch_args=["-stitch_type", "optflow", item])
        with pytest.raises(ValueError, match="items must be strings"):
            project.stitch_args()

    def test_number_items_stringified(self):
        _config(stitch_args=["-bitrate", 100000000])
        assert project.stitch_args() == ["-bitrate", "100000000"]


class TestStitchTimeout:
    def test_unset(self):
        assert project.stitch_timeout_s() is None

    @pytest.mark.parametrize("value, expected", [(7200, 7200.0)])
    def test_valid(self, value, expected):
        _config(stitch_timeout_s=value)
        got = project.stitch_timeout_s()
        assert got == expected and isinstance(got, float)

    @pytest.mark.parametrize("value", [0, "60"])
    def test_invalid(self, value):
        _config(stitch_timeout_s=value)
        with pytest.raises(ValueError, match="stitch_timeout_s"):
            project.stitch_timeout_s()

    def test_null_means_no_limit(self):
        _config(stitch_timeout_s=None)
        assert project.stitch_timeout_s() is None


# --------------------------------------------------------------------------- SDK discovery
def _fake_exe(root: Path, folder: str, name: str | None = None) -> Path:
    p = root / folder / "bin" / (name or project.SDK_EXE_NAME)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


class TestFindSdk:
    def test_newest_version_wins_numerically(self, tmp_path):
        _fake_exe(tmp_path, "MediaSDK-3.1.5-win")
        _fake_exe(tmp_path, "MediaSDK-3.2.0-win")
        newest = _fake_exe(tmp_path, "MediaSDK-3.10.1-win")
        _fake_exe(tmp_path, "MediaSDK-2.99.9-win")
        assert project.find_sdk([tmp_path]) == newest

    def test_versioned_beats_unversioned(self, tmp_path):
        _fake_exe(tmp_path, "unversioned")
        versioned = _fake_exe(tmp_path, "MediaSDK-0.0.1")
        assert project.find_sdk([tmp_path]) == versioned

    def test_unversioned_only_is_still_found(self, tmp_path):
        only = _fake_exe(tmp_path, "sdk")
        assert project.find_sdk([tmp_path]) == only

    def test_searches_every_root_and_any_depth(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        _fake_exe(a / "x" / "y", "MediaSDK-1.0.0")
        newest = _fake_exe(b, "MediaSDK-2.0.0")
        assert project.find_sdk([a, b]) == newest

    def test_missing_or_empty_roots(self, tmp_path):
        assert project.find_sdk([tmp_path / "nope"]) is None
        assert project.find_sdk([]) is None
        assert project.find_sdk([tmp_path]) is None

    def test_default_root_is_under_cutpoint_home(self, home):
        assert project.find_sdk() is None
        want = _fake_exe(home / "sdk" / "insta360", "MediaSDK-3.1.5-win")
        assert project.find_sdk() == want


class TestSdkVersion:
    @pytest.mark.parametrize(
        "path, version",
        [
            ("C:/x/MediaSDK-3.1.5-win/bin/MediaSDKTest.exe", "3.1.5"),
            ("/opt/sdk/bin/MediaSDKTest", None),
            ("/opt/MediaSDK-beta/bin/MediaSDKTest", None),
        ],
    )
    def test_from_folder_name(self, path, version):
        assert project.sdk_version(path) == version


class TestSdkLocation:
    def test_none(self):
        assert project.sdk_location() == (None, "none")
        assert project.sdk_exe() is None

    def test_auto(self, home):
        exe = _fake_exe(home / "sdk" / "insta360", "MediaSDK-3.1.5")
        assert project.sdk_location() == (exe, "auto")

    def test_config_beats_auto(self, home, tmp_path):
        _fake_exe(home / "sdk" / "insta360", "MediaSDK-3.1.5")
        cfg_exe = tmp_path / "cfg" / "sdk.exe"  # a configured path need not exist
        _config(sdk_exe=str(cfg_exe))
        assert project.sdk_location() == (cfg_exe, "config")

    def test_env_beats_config_and_auto(self, home, tmp_path, monkeypatch):
        _fake_exe(home / "sdk" / "insta360", "MediaSDK-3.1.5")
        _config(sdk_exe=str(tmp_path / "cfg.exe"))
        monkeypatch.setenv("CUTPOINT_INSTA360_SDK", str(tmp_path / "env.exe"))
        assert project.sdk_location() == (tmp_path / "env.exe", "env")
        assert project.sdk_exe() == tmp_path / "env.exe"


# --------------------------------------------------------------------------- sdk_status / cli
def _probe_result(**kw):
    def probe(exe, timeout_s=stitch.HELP_TIMEOUT_S):
        probe.calls.append(exe)
        return {"runs": True, "version": None, "seconds": 2.0, "error": None, **kw}

    probe.calls = []  # type: ignore[attr-defined]
    return probe


class TestSdkStatus:
    def test_origin_none(self, monkeypatch):
        monkeypatch.setattr(stitch, "probe_sdk", lambda *a, **k: pytest.fail("probed"))
        assert stitch.sdk_status() == {
            "exe": None,
            "origin": "none",
            "found": False,
            "runs": False,
            "version": None,
        }

    def test_configured_but_missing(self, tmp_path, monkeypatch):
        missing = tmp_path / "nope" / project.SDK_EXE_NAME
        _config(sdk_exe=str(missing))
        monkeypatch.setattr(stitch, "probe_sdk", lambda *a, **k: pytest.fail("probed"))
        assert stitch.sdk_status() == {
            "exe": str(missing),
            "origin": "config",
            "found": False,
            "runs": False,
            "version": None,
        }

    def test_auto_detected_folder_version_wins(self, home, monkeypatch):
        exe = _fake_exe(home / "sdk" / "insta360", "MediaSDK-3.1.5")
        probe = _probe_result(version="9.9.9", seconds=1.25)
        monkeypatch.setattr(stitch, "probe_sdk", probe)
        assert stitch.sdk_status() == {
            "exe": str(exe),
            "origin": "auto",
            "found": True,
            "runs": True,
            "version": "3.1.5",
            "help_seconds": 1.25,
            "error": None,
        }
        assert probe.calls == [exe]

    def test_auto_detected_version_from_probe(self, home, monkeypatch):
        _fake_exe(home / "sdk" / "insta360", "plain")
        monkeypatch.setattr(stitch, "probe_sdk", _probe_result(version="9.9.9"))
        assert stitch.sdk_status()["version"] == "9.9.9"

    def test_auto_detected_but_does_not_run(self, home, monkeypatch):
        _fake_exe(home / "sdk" / "insta360", "plain")
        monkeypatch.setattr(stitch, "probe_sdk", _probe_result(runs=False, error="boom"))
        s = stitch.sdk_status()
        assert s["found"] is True and s["runs"] is False
        assert s["version"] is None and s["error"] == "boom"


class TestCliStatus:
    def _sdk_lines(self, caplog) -> list[str]:
        with caplog.at_level(logging.INFO, logger="cutpoint"):
            pano_cli.cmd_status(argparse.Namespace(project=None))
        return [r.getMessage() for r in caplog.records if r.getMessage().startswith("SDK:")]

    def test_without_project_logs_the_sdk_line_only(self, home, monkeypatch, caplog):
        exe = _fake_exe(home / "sdk" / "insta360", "MediaSDK-3.1.5")
        monkeypatch.setattr(stitch, "probe_sdk", _probe_result())
        assert self._sdk_lines(caplog) == [
            f"SDK: {exe} (auto-detected), version 3.1.5, runs (-help in 2.0s)"
        ]
        assert len(caplog.records) == 1

    def test_not_found(self, caplog):
        lines = self._sdk_lines(caplog)
        assert len(lines) == 1 and lines[0].startswith("SDK: not found")


class TestCliStitchFromClips:
    @pytest.fixture
    def proj(self, monkeypatch):
        from cutpoint.config import projects_dir  # noqa: PLC0415

        proj = projects_dir() / "trip-360"
        man = proj / "clips_review" / "manifest.json"
        man.parent.mkdir(parents=True)
        man.write_text(json.dumps({"clips": [{"source": "D:/cam/VID_001.insv"}]}), "utf-8")
        clips = {
            n: SimpleNamespace(clip_id=n, source=f"D:/cam/VID_{n}.insv") for n in ("001", "002")
        }
        monkeypatch.setattr(project, "load", lambda _p: clips)
        self.stitched: list[list[str]] = []
        monkeypatch.setattr(
            stitch,
            "stitch_clips",
            lambda clips, **_kw: self.stitched.append([c.clip_id for c in clips]) or [],
        )
        return proj

    def _run(self, from_clips: str | None, segments: str | None = None) -> None:
        pano_cli.cmd_stitch(
            argparse.Namespace(
                project="trip-360",
                clips=None,
                from_clips=from_clips,
                segments=segments,
                top=None,
                max_total=None,
                force=False,
                dry_run=True,
            )
        )

    @pytest.mark.parametrize("arg", ["projects/trip-360/clips_review", "clips_review"])
    def test_relative_forms_resolve(self, proj, arg):
        self._run(arg)
        assert self.stitched == [["001"]]

    @pytest.mark.skipif(sys.platform != "win32", reason="backslash separators are Windows-only")
    def test_backslash_form_resolves(self, proj):
        self._run(r"projects\trip-360\clips_review")
        assert self.stitched == [["001"]]

    def test_absolute(self, proj):
        self._run(str(proj / "clips_review"))
        assert self.stitched == [["001"]]

    def test_segments_select_only_their_sources(self, proj):
        (proj / "VID_001.insv").write_bytes(b"x")
        seg = {"source": "VID_001.insv", "start": 1, "end": 5, "category": "run"}
        (proj / "mix_segments.json").write_text(json.dumps([seg, seg]), "utf-8")
        self._run(None, segments="mix_segments.json")  # relative = inside the project
        assert self.stitched == [["001"]]

    def test_missing_folder_is_a_clear_error(self, proj):
        with pytest.raises(FileNotFoundError, match=r"clips folder not found: .*nope"):
            self._run("projects/trip-360/nope")
        assert self.stitched == []


class TestCliCheck:
    @pytest.fixture
    def clips(self, monkeypatch):
        from cutpoint.config import projects_dir  # noqa: PLC0415
        from cutpoint.pano import check  # noqa: PLC0415

        clips = projects_dir() / "trip-360" / "clips_review"
        clips.mkdir(parents=True)
        self.seen: list[Path] = []
        monkeypatch.setattr(check, "clip_sheets", lambda d, **_kw: self.seen.append(d) or [])
        return clips

    def _run(self, clips_dir: str) -> None:
        pano_cli.cmd_check(argparse.Namespace(clips_dir=clips_dir, per_clip=6))

    def test_projects_relative_form_resolves(self, clips):
        self._run("projects/trip-360/clips_review")
        assert [p.resolve() for p in self.seen] == [clips.resolve()]

    def test_missing_folder_is_a_clear_error(self, clips):
        with pytest.raises(FileNotFoundError, match=r"clips folder not found: .*nope"):
            self._run("projects/trip-360/nope")
        assert self.seen == []


# --------------------------------------------------------------------------- sidecar
class TestSidecar:
    def test_default_args_have_no_flowstate(self):
        assert project.FLOWSTATE_ARG not in project.DEFAULT_STITCH_ARGS
        assert "-enable_stitchfusion" in project.DEFAULT_STITCH_ARGS

    def test_write_records_args_after_output_and_sdk_version(self, tmp_path):
        exe = tmp_path / "MediaSDK-3.1.5-20260101-win64" / "bin" / "MediaSDKTest.exe"
        out = tmp_path / "VID_x_001.equirect.mp4"
        cmd = [str(exe), "-inputs", "a.insv", "b.insv", "-output", str(out), "-enable_flowstate"]
        side = stitch.write_sidecar(out, cmd)
        assert side == tmp_path / "VID_x_001.equirect.json"
        meta = json.loads(side.read_text(encoding="utf-8"))
        assert meta == {
            "version": project.EQUIRECT_META_VERSION,
            "stitch_args": ["-enable_flowstate"],
            "sdk_version": "3.1.5",
        }
        assert project.equirect_flowstate(out) is True

    def test_write_error_removes_stale_sidecar(self, tmp_path, monkeypatch, caplog):
        out = tmp_path / "VID_x_001.equirect.mp4"
        side = project.equirect_sidecar(out)
        side.write_text(json.dumps({"stitch_args": []}), encoding="utf-8")

        def fail(*_a, **_k):
            raise PermissionError("locked")

        monkeypatch.setattr(stitch.cache, "save_json", fail)
        with caplog.at_level(logging.WARNING):
            stitch.write_sidecar(out, ["sdk", "-output", str(out), "-enable_stitchfusion"])
        assert not side.exists()
        assert "could not write" in caplog.text
        assert project.equirect_flowstate(out) is None

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            (None, None),
            (json.dumps({"version": 1, "stitch_args": ["-enable_stitchfusion"]}), False),
            (json.dumps({"version": 1, "stitch_args": [project.FLOWSTATE_ARG]}), True),
        ],
    )
    def test_read_flowstate(self, tmp_path, content, expected):
        out = tmp_path / "VID_x_001.equirect.mp4"
        if content is not None:
            project.equirect_sidecar(out).write_text(content, encoding="utf-8")
        assert project.equirect_flowstate(out) is expected


class TestSidecarOnRestitch:
    def test_force_removes_old_sidecar_before_replacing(self, fake, clip, paths_, monkeypatch):
        """An interruption between the rename and the new sidecar leaves no stale record."""
        out_dir, out, part = paths_
        out.write_bytes(b"old")
        old = {"version": 1, "stitch_args": ["-enable_stitchfusion"]}
        project.equirect_sidecar(out).write_text(json.dumps(old), encoding="utf-8")
        monkeypatch.setattr(stitch, "write_sidecar", lambda o, c: None)  # "interrupted"
        rec = stitch._stitch_one(clip, fake("ok", part), part, out, out_dir, None)
        assert rec["status"] == "ok" and out.stat().st_size == 1_200_000
        assert not project.equirect_sidecar(out).exists()
        assert project.equirect_flowstate(out) is None

    def test_existing_without_clean_sidecar_warns(self, tmp_path, caplog):
        out_dir = tmp_path / "stitched"
        out_dir.mkdir()
        _config(stitched_dir=str(out_dir))
        clips = [
            X3Clip(n, "t", "5.7k", None, [f"VID_20260101_100000_00_{n}.insv"])
            for n in ("001", "002", "003")
        ]
        sidecars = {"001": None, "002": ["-enable_flowstate"], "003": ["-enable_stitchfusion"]}
        for c in clips:
            out = project.sdk_output(c)
            out.write_bytes(b"x")
            args = sidecars[c.clip_id]
            if args is not None:
                meta = json.dumps({"version": 1, "stitch_args": args})
                project.equirect_sidecar(out).write_text(meta, encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            recs = stitch.stitch_clips(clips, dry_run=True)
        assert [r["status"] for r in recs] == ["exists"] * 3
        warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warned) == 2
        assert "clip 001" in warned[0] and "no sidecar" in warned[0]
        assert "--clips 001 --force" in warned[0]
        assert "clip 002" in warned[1] and "FlowState" in warned[1]
        assert "--clips 002 --force" in warned[1]
