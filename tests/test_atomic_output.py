"""Exports appear under their final name only when complete."""

from __future__ import annotations

import os
import signal
from fractions import Fraction
from pathlib import Path

import pytest

from cutpoint import assemble, cli, encode
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
