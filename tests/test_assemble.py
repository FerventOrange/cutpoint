"""assemble: cut joins keep the rebuilt audio on the stream-copied video (#235)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from cutpoint import assemble, sources
from cutpoint.ffmpeg_utils import MediaInfo

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not on PATH"
)


def _info(duration: float, vduration: float) -> MediaInfo:
    return MediaInfo(
        path=Path("x.mp4"), duration=duration, vduration=vduration, width=64, height=48,
        fps=Fraction(25), pix_fmt="yuv420p", vcodec="h264", audio_streams=1, acodec="aac",
        sample_rate=48000, channels=2,
    )  # fmt: skip


def test_cut_duration_is_the_container_duration_used_for_offsets():
    infos = [_info(2.3, 2.0), _info(2.3, 2.0), _info(2.0, 2.0)]
    assert [assemble.cut_duration(i) for i in infos] == [2.3, 2.3, 2.0]
    assert assemble.clip_offsets(infos) == pytest.approx([0.0, 2.3, 4.6])


def test_concat_copy_trims_rebuilt_audio_to_the_cut_duration(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(assemble, "run", lambda args, **_: calls.append(args) or "")
    files = [tmp_path / "001_a.mp4", tmp_path / "002_b.mp4"]
    infos = [_info(2.3, 2.0), _info(2.0, 2.0)]
    assemble.concat_copy(files, tmp_path / "final.mp4", infos, [0.25, 0.25])
    graph = calls[0][calls[0].index("-filter_complex") + 1]
    assert "atrim=end=2.300000[a0]" in graph
    assert "atrim=end=2.000000[a1]" in graph


def _clip(out: Path, video: float, audio: float) -> None:
    """A 64x48, 25 fps clip of ``video`` s whose audio runs ``audio`` s."""
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
         f"testsrc=size=64x48:rate=25:duration={video}", "-f", "lavfi", "-i",
         f"sine=frequency=440:sample_rate=48000:duration={audio}", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)],
        check=True,
    )  # fmt: skip


def _stream_durations(path: Path) -> dict[str, float]:
    out = subprocess.run(
        [shutil.which("ffprobe"), "-v", "error", "-show_entries",
         "stream=codec_type,duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return {line.split(",")[0]: float(line.split(",")[1]) for line in out.split()}


@needs_ffmpeg
@pytest.mark.parametrize("baked", [0.25, 0.0], ids=["stream-copy", "audio-faded"])
def test_assemble_cut_audio_stays_on_video_when_clip_audio_runs_on(tmp_path, monkeypatch, baked):
    """The first two clips' audio runs 0.3 s past their 2 s of video. concat starts each
    next clip at 2.3 s, and the audio-faded join used to trim each audio piece to 2 s:
    6.6 s of video against 6.0 s of audio."""
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    names = ["001_a.mp4", "002_b.mp4", "003_c.mp4"]
    for name, audio in zip(names, [2.3, 2.3, 2.0], strict=True):
        _clip(clips_dir / name, 2.0, audio)
    entries = [
        {"file": n, "source": f"{n[4]}.mp4", "start": 0.0, "end": 2.0, "audio_fade": baked}
        for n in names
    ]
    (clips_dir / "manifest.json").write_text(json.dumps({"clips": entries}), encoding="utf-8")
    monkeypatch.setattr(sources, "source_words", lambda _src: None)  # no final.srt
    final = assemble.assemble(clips_dir)
    durations = _stream_durations(final)
    assert durations["video"] == pytest.approx(6.6, abs=0.04)
    assert durations["audio"] == pytest.approx(durations["video"], abs=0.04)  # one frame


@pytest.mark.parametrize(
    ("fade", "fps", "expected"),
    [
        (0.25, Fraction(25), 0.25),
        (0.02, Fraction(25), 0.04),  # under one frame: raised to one
        (0.0, Fraction(25), 0.04),
        (0.01, Fraction(30000, 1001), 0.034),  # 33.37 ms rounded up to the millisecond
        (5.0, Fraction(25), 0.9),  # at most 45% of the shortest clip
    ],
)
def test_crossfade_duration_is_at_least_one_frame(fade, fps, expected):
    infos = [replace(_info(2.0, 2.0), fps=fps), replace(_info(3.0, 3.0), fps=fps)]
    assert assemble.crossfade_duration(infos, fade) == pytest.approx(expected)


def test_crossfade_duration_is_zero_without_room_for_one_frame():
    infos = [_info(2.0, 2.0), _info(0.08, 0.08)]  # 2 frames at 25 fps: 45% is 0.036 s
    assert assemble.crossfade_duration(infos, 0.25) == 0.0


def test_fade_join_without_room_falls_back_to_cuts(tmp_path, monkeypatch, caplog):
    joined = []
    monkeypatch.setattr(assemble, "concat_copy", lambda *a, **_: joined.append("cut"))
    monkeypatch.setattr(assemble, "crossfade", lambda *a, **_: joined.append("fade"))
    files = [tmp_path / "001_a.mp4", tmp_path / "002_b.mp4"]
    infos = [_info(2.0, 2.0), _info(0.08, 0.08)]
    used = assemble._join(files, infos, tmp_path, tmp_path / "final.mp4", "fade", 0.25, 0.25)
    assert (used, joined) == (0.0, ["cut"])
    assert "no room for a crossfade" in caplog.text
    assert "45% of" in caplog.text


@needs_ffmpeg
@pytest.mark.parametrize(("fade", "video"), [(0.02, 3.96), (0.25, 3.75)])
def test_assemble_short_crossfade_keeps_the_whole_video(tmp_path, monkeypatch, fade, video):
    """xfade shorter than a frame (0.02 s at 25 fps) cut the video to 2.04 s of 3.96 s."""
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    for name in ("001_a.mp4", "002_b.mp4"):
        _clip(clips_dir / name, 2.0, 2.0)
    monkeypatch.setattr(sources, "source_words", lambda _src: None)  # no final.srt
    final = assemble.assemble(clips_dir, transition="fade", fade_dur=fade)
    durations = _stream_durations(final)
    assert durations["video"] == pytest.approx(video, abs=0.04)
    assert durations["audio"] == pytest.approx(video, abs=0.04)
