"""assemble: cut joins keep the rebuilt audio on the stream-copied video (#235)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from conftest import media_info, needs_ffmpeg
from cutpoint import assemble, ffmpeg_utils, sources
from cutpoint.ffmpeg_utils import FFmpegError, MediaInfo


def _info(duration: float, vduration: float) -> MediaInfo:
    return media_info(
        Path("x.mp4"), duration, vduration=vduration, width=64, height=48, fps=Fraction(25)
    )


def test_cut_duration_is_the_container_duration_used_for_offsets():
    infos = [_info(2.3, 2.0), _info(2.3, 2.0), _info(2.0, 2.0)]
    assert [assemble.cut_duration(i) for i in infos] == [2.3, 2.3, 2.0]
    assert assemble.clip_offsets(infos) == pytest.approx([0.0, 2.3, 4.6])
    # xfade works on the video: crossfades use the video duration, not the container's
    assert assemble.clip_offsets(infos, 0.5) == pytest.approx([0.0, 1.5, 3.0])


def test_concat_copy_trims_rebuilt_audio_to_the_cut_duration(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(assemble, "run", lambda args, **_: calls.append(args) or "")
    files = [tmp_path / "001_a.mp4", tmp_path / "002_b.mp4"]
    infos = [_info(2.3, 2.0), _info(2.0, 2.0)]
    assemble.concat_copy(files, tmp_path / "final.mp4", infos, [0.25, 0.25])
    graph = calls[0][calls[0].index("-filter_complex") + 1]
    assert "atrim=end=2.300000[a0]" in graph
    assert "atrim=end=2.000000[a1]" in graph


def _clip(out: Path, video: float, audio: float, rate: str = "25") -> None:
    """A 64x48 clip (``rate`` fps, default 25) of ``video`` s whose audio runs ``audio`` s."""
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
         f"testsrc=size=64x48:rate={rate}:duration={video}", "-f", "lavfi", "-i",
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
    ("fade", "fps", "frames"),
    [
        (0.5, Fraction(30), 15),
        (0.25, Fraction(25), 6),  # 6.25 frames
        (0.26, Fraction(25), 7),  # 6.5 frames: rounded half up
        (0.25, Fraction(30000, 1001), 7),  # 7.49 frames
        (0.5, Fraction(30000, 1001), 15),  # 14.99 frames
        (0.3, Fraction(24), 7),  # 7.2 frames
        (0.33, Fraction(24), 8),  # 7.92 frames
        (0.02, Fraction(25), 1),  # under one frame: raised to one
        (0.0, Fraction(25), 1),
        (0.01, Fraction(30000, 1001), 1),  # one whole frame, not a rounded ms
        (5.0, Fraction(25), 22),  # at most 45% of the shortest clip (0.9 s)
    ],
)
def test_crossfade_duration_is_whole_frames_between_one_and_the_cap(fade, fps, frames):
    """xfade works in whole frames; acrossfade and final.srt must use the same fade (#243)."""
    infos = [replace(_info(2.0, 2.0), fps=fps), replace(_info(3.0, 3.0), fps=fps)]
    used = assemble.crossfade_duration(infos, fade)
    assert used == pytest.approx(frames / float(fps), abs=1e-12)


def test_crossfade_passes_one_fade_to_xfade_acrossfade_and_offsets(tmp_path, monkeypatch):
    graphs = []
    monkeypatch.setattr(assemble, "_encode", lambda _files, graph, _out: graphs.append(graph))
    fps = Fraction(24)
    infos = [replace(_info(2.0, 2.0), fps=fps) for _ in range(3)]
    files = [tmp_path / f"00{i}_x.mp4" for i in range(3)]
    fade = assemble.crossfade_duration(infos, 1 / 3)  # 8 frames: 0.333333..., not 0.333
    assert assemble.crossfade(files, infos, tmp_path / "final.mp4", fade) == fade
    lead = [o - 1 / 48 for o in assemble.clip_offsets(infos, fade)]  # half a frame early
    assert f"xfade=transition=fade:duration={fade:.6f}:offset={lead[1]:.6f}[vx1]" in graphs[0]
    assert f"duration={fade:.6f}:offset={lead[2]:.6f}[v]" in graphs[0]
    assert graphs[0].count(f"acrossfade=d={fade:.6f}[") == 2


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
def test_assemble_short_crossfade_keeps_the_whole_video(tmp_path, monkeypatch):
    """xfade shorter than a frame (0.02 s at 25 fps) cut the video to 2.04 s of 3.96 s."""
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    for name in ("001_a.mp4", "002_b.mp4"):
        _clip(clips_dir / name, 2.0, 2.0)
    monkeypatch.setattr(sources, "source_words", lambda _src: None)  # no final.srt
    final = assemble.assemble(clips_dir, transition="fade", fade_dur=0.02)
    durations = _stream_durations(final)
    assert durations["video"] == pytest.approx(3.96, abs=0.04)
    assert durations["audio"] == pytest.approx(3.96, abs=0.04)


@needs_ffmpeg
@pytest.mark.parametrize("rate", ["30000/1001", "24"])
def test_assemble_crossfade_length_matches_the_srt_offsets(tmp_path, monkeypatch, rate):
    """0.25 s is 7.49 frames at 29.97 fps and 6 at 24: xfade used whole frames while
    acrossfade and final.srt used 0.25 s, so the video ran longer than both (#243)."""
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    for name in ("001_a.mp4", "002_b.mp4", "003_c.mp4"):
        _clip(clips_dir / name, 2.0, 2.0, rate)
    monkeypatch.setattr(sources, "source_words", lambda _src: None)  # no final.srt
    final = assemble.assemble(clips_dir, transition="fade", fade_dur=0.25)
    infos = [assemble.probe(f) for f in assemble.list_clips(clips_dir, exclude={final})]
    fade = assemble.crossfade_duration(infos, 0.25)
    predicted = assemble.clip_offsets(infos, fade)[-1] + infos[-1].vduration
    frame = 1 / float(Fraction(rate))
    durations = _stream_durations(final)
    assert durations["video"] == pytest.approx(predicted, abs=frame / 2)
    assert durations["audio"] == pytest.approx(predicted, abs=frame)


def test_crossfade_trims_each_clip_audio_to_its_video(tmp_path, monkeypatch):
    graphs = []
    monkeypatch.setattr(assemble, "_encode", lambda _files, graph, _out: graphs.append(graph))
    infos = [_info(2.14, 2.0), _info(2.0, 2.0)]  # the first clip's audio runs 3.5 frames on
    files = [tmp_path / "001_a.mp4", tmp_path / "002_b.mp4"]
    fade = assemble.crossfade_duration(infos, 0.25)
    assemble.crossfade(files, infos, tmp_path / "final.mp4", fade)
    graph = graphs[0]
    assert "[a0]apad,atrim=end=2.000000,asetpts=PTS-STARTPTS[at0]" in graph
    assert "[a1]apad,atrim=end=2.000000,asetpts=PTS-STARTPTS[at1]" in graph
    assert f"[at0][at1]acrossfade=d={fade:.6f}[a]" in graph


@needs_ffmpeg
def test_assemble_crossfade_audio_stays_on_video_when_clip_audio_runs_on(tmp_path, monkeypatch):
    """The first clip's audio runs 0.3 s past its 2 s of video. xfade follows the video
    and acrossfade used the whole audio: 3.76 s of video against 4.06 s of audio."""
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    _clip(clips_dir / "001_a.mp4", 2.0, 2.3)
    _clip(clips_dir / "002_b.mp4", 2.0, 2.0)
    monkeypatch.setattr(sources, "source_words", lambda _src: None)  # no final.srt
    final = assemble.assemble(clips_dir, transition="fade", fade_dur=0.25)
    durations = _stream_durations(final)
    assert durations["video"] == pytest.approx(3.76, abs=0.02)
    assert durations["audio"] == pytest.approx(durations["video"], abs=0.04)  # one frame


_BUILDCONF = """configuration:
    --enable-gpl
    --enable-libsoxr
    --enable-libx264
"""


def test_buildconf_parser_finds_libsoxr():
    assert ffmpeg_utils.buildconf_has_soxr(_BUILDCONF)
    without = _BUILDCONF.replace("    --enable-libsoxr\n", "")
    assert not ffmpeg_utils.buildconf_has_soxr(without)
    assert not ffmpeg_utils.buildconf_has_soxr("    --disable-libsoxr\n")


def test_has_soxr_reads_buildconf_once(monkeypatch):
    calls = []
    monkeypatch.setattr(ffmpeg_utils, "run", lambda args, **_: calls.append(args) or _BUILDCONF)
    assert ffmpeg_utils.has_soxr()
    assert ffmpeg_utils.has_soxr()
    assert calls == [["ffmpeg", "-hide_banner", "-buildconf"]]


def test_has_soxr_is_false_when_ffmpeg_fails(monkeypatch):
    def fail(_args, **_):
        raise FFmpegError("ffmpeg failed (exit 1):")

    monkeypatch.setattr(ffmpeg_utils, "run", fail)
    assert not ffmpeg_utils.has_soxr()


@pytest.mark.parametrize("soxr", [True, False])
def test_limiter_filter_uses_soxr_only_when_ffmpeg_has_it(monkeypatch, soxr):
    monkeypatch.setattr(assemble, "has_soxr", lambda: soxr)
    graph = assemble.limiter_filter(-1.0)
    assert "alimiter=limit=0.8913" in graph  # -1 dB as a linear limit
    # oversample, limit, then back down to 48 kHz as the last stage (with soxr only if available)
    assert (
        graph.index("aresample=192000") < graph.index("alimiter") < graph.index("aresample=48000")
    )
    assert graph.endswith(",aresample=48000:resampler=soxr" if soxr else ",aresample=48000")
    assert ("soxr" in graph) is soxr


@needs_ffmpeg
def test_limiter_filter_runs_on_this_ffmpeg():
    """Builds without libsoxr (gyan.dev essentials) failed: resampling engine unavailable."""
    graph = f"{assemble.LOUDNORM},{assemble.limiter_filter(-1.0)}"
    ffmpeg_utils.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=d=0.5",
            "-af",
            graph,
            "-f",
            "null",
            "-",
        ]
    )
