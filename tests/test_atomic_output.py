"""Exports appear under their final name only when complete."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from cutpoint import assemble, cli, clips, encode, ffmpeg_utils
from cutpoint.ffmpeg_utils import FFmpegError, MediaInfo


def _info(path: Path) -> MediaInfo:
    return MediaInfo(
        path=path, duration=100.0, vduration=100.0, width=1920, height=1080,
        fps=Fraction(30), pix_fmt="yuv420p", vcodec="h264", audio_streams=1,
        acodec="aac", sample_rate=48000, channels=2,
    )  # fmt: skip


def _dies_midway(exc: BaseException):
    def run(args, **_kw):
        Path(args[-1]).write_bytes(b"truncated")  # ffmpeg had started writing
        raise exc

    return run


FAILURES = [FFmpegError("ffmpeg failed (exit 1)"), KeyboardInterrupt()]


@pytest.mark.parametrize("exc", FAILURES, ids=["ffmpeg-error", "ctrl-c"])
def test_failed_or_interrupted_clip_export_leaves_no_file(monkeypatch, tmp_path, exc):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(encode, "run", _dies_midway(exc))
    src = tmp_path / "src.mp4"
    with pytest.raises(type(exc)):
        encode.export_clip(
            src,
            _info(src),
            0.0,
            2.0,
            tmp_path / "001.mp4",
            encode.OutputFormat(1920, 1080, 30),
            False,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("exc", FAILURES, ids=["ffmpeg-error", "ctrl-c"])
def test_failed_or_interrupted_assemble_leaves_no_final_video(monkeypatch, tmp_path, exc):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "001.mp4").write_bytes(b"x")
    monkeypatch.setattr(assemble, "probe", _info)
    monkeypatch.setattr(assemble, "baked_fades", lambda d, files: [0.0] * len(files))

    def concat_copy(files, out, infos, fades):
        out.write_bytes(b"truncated")
        raise exc

    monkeypatch.setattr(assemble, "concat_copy", concat_copy)
    with pytest.raises(type(exc)):
        assemble.assemble(clips, tmp_path / "final.mp4")
    assert list(tmp_path.glob("*.mp4")) == [] and sorted(p.name for p in clips.iterdir()) == [
        "001.mp4"
    ]


@pytest.mark.skipif(os.name == "nt", reason="the UI kills Windows jobs with taskkill, not SIGTERM")
def test_sigterm_cancels_a_command_like_ctrl_c(monkeypatch):
    """A UI cancel (SIGTERM) unwinds as exit 130 and leaves the caller's handler alone."""

    def cancelled(_args):
        signal.raise_signal(signal.SIGTERM)
        return 0

    monkeypatch.setattr(cli, "_cmd_paths", cancelled)
    before = signal.getsignal(signal.SIGTERM)

    assert cli.main(["paths"]) == 130
    assert signal.getsignal(signal.SIGTERM) is before


# --- exit codes, failed-run cleanup, locked files ---


@pytest.mark.parametrize(
    ("rc", "shown"),
    [
        (3752568763, "-542398533 (0xDFABA7BB)"),  # Windows crash code, reported unsigned
        (-9, "-9"),  # POSIX: killed by SIGKILL
        (1, "1"),
    ],
)
def test_ffmpeg_failure_shows_crash_codes_readably(monkeypatch, rc, shown):
    monkeypatch.setattr(ffmpeg_utils, "find_tool", lambda name: name)
    monkeypatch.setattr(ffmpeg_utils, "stage_inputs", lambda cmd: cmd)
    monkeypatch.setattr(
        ffmpeg_utils.subprocess,
        "run",
        lambda cmd, **_kw: subprocess.CompletedProcess(cmd, rc, "", "boom"),
    )
    with pytest.raises(FFmpegError, match=rf"\(exit {re.escape(shown)}\)"):
        ffmpeg_utils.run(["ffmpeg", "-i", "x"])


def _project(home: Path) -> tuple[Path, Path]:
    project = home / "projects" / "p"
    project.mkdir(parents=True)
    src = project / "src.mp4"
    src.write_bytes(b"x")
    seg = project / "seg.json"
    seg.write_text(
        json.dumps([{"source": str(src), "start": 0.0, "end": 2.0, "label": "a"}]),
        encoding="utf-8",
    )
    return project, seg


@pytest.mark.parametrize("preexisting", [False, True], ids=["created", "preexisting"])
def test_failed_export_leaves_no_new_empty_clips_folder(monkeypatch, home, preexisting):
    project, seg = _project(home)
    out = project / "clips"
    if preexisting:
        out.mkdir()

    def broken_probe(_path):
        raise FFmpegError("ffprobe failed")

    monkeypatch.setattr(clips, "probe", broken_probe)
    with pytest.raises(FFmpegError):
        clips.export_clips(project, segments=seg)
    assert out.is_dir() is preexisting


def _locked(winerror: int) -> PermissionError:
    err = PermissionError(13, "The process cannot access the file")
    err.winerror = winerror
    return err


@pytest.mark.parametrize("winerror", [32, 5])
def test_clips_force_over_a_locked_clip_says_who_to_close(monkeypatch, home, caplog, winerror):
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", lambda _s: None)
    project, seg = _project(home)
    old = project / "clips" / "001_a.mp4"
    old.parent.mkdir()
    old.write_bytes(b"x")
    real_unlink = Path.unlink

    def unlink(self, missing_ok=False):
        if self == old:
            raise _locked(winerror)
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    assert cli.main(["clips", "p", "--segments", str(seg), "--force"]) == 1
    assert "open in another program" in caplog.text
    assert "001_a.mp4" in caplog.text


def test_clips_force_retries_a_lock_that_clears(monkeypatch, home):
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", lambda _s: None)
    project, seg = _project(home)
    old = project / "clips" / "001_a.mp4"
    old.parent.mkdir()
    old.write_bytes(b"x")
    real_unlink = Path.unlink
    attempts = []

    def unlink(self, missing_ok=False):
        if self == old and not attempts:
            attempts.append(1)
            raise _locked(32)
        return real_unlink(self, missing_ok=missing_ok)

    def stop(_path):  # past the delete; no real ffmpeg needed
        raise FFmpegError("stop here")

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(clips, "probe", stop)
    with pytest.raises(FFmpegError, match="stop here"):
        clips.export_clips(project, segments=seg, force=True)
    assert attempts and not old.exists()


def test_locked_final_video_names_the_file(monkeypatch, tmp_path):
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", lambda _s: None)
    out = tmp_path / "final.mp4"

    def replace(self, target):
        raise _locked(32)

    monkeypatch.setattr(Path, "replace", replace)
    with (
        pytest.raises(PermissionError, match="open in another program") as exc,
        ffmpeg_utils.atomic_output(out) as part,
    ):
        part.write_bytes(b"x")
    assert str(out) in str(exc.value)
    assert not ffmpeg_utils.part_path(out).exists()


@pytest.mark.parametrize("winerror", [None, 2])
def test_non_lock_permission_error_is_reraised_unchanged(monkeypatch, tmp_path, winerror):
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", lambda _s: None)
    out = tmp_path / "final.mp4"

    def replace(self, target):
        err = PermissionError(13, "denied")
        if winerror is not None:
            err.winerror = winerror
        raise err

    monkeypatch.setattr(Path, "replace", replace)
    with (
        pytest.raises(PermissionError) as exc,
        ffmpeg_utils.atomic_output(out) as part,
    ):
        part.write_bytes(b"x")
    assert "another program" not in str(exc.value)
    assert exc.value.errno == 13
    assert exc.value.strerror == "denied"
    assert not ffmpeg_utils.part_path(out).exists()
