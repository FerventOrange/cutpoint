"""assemble: cut joins keep the rebuilt audio on the stream-copied video (#235)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from conftest import media_info, needs_ffmpeg
from cutpoint import assemble, encode, ffmpeg_utils, sources
from cutpoint.ffmpeg_utils import FFmpegError, MediaInfo, VideoEncoder


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
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)  # no final.srt
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
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)  # no final.srt
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
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)  # no final.srt
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
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)  # no final.srt
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


# --------------------------------------------------------------------------- short audio (#250)
FADE_NORMAL = (
    ",asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.250:curve=qsin,"
    "afade=t=out:st=1.250:d=0.250:curve=qsin"
)


@pytest.mark.parametrize("audio", [None, 1.5, 2.0, 1.49], ids=["unknown", "exact", "long", "tol"])
def test_audio_fade_filter_is_unchanged_when_the_audio_covers_the_clip(audio):
    assert encode.audio_fade_filter(1.5, 0.25, audio) == FADE_NORMAL
    assert encode.audio_fade_filter(1.5, 0.25) == FADE_NORMAL


def test_audio_fade_filter_fades_out_at_the_end_of_shorter_audio():
    """A 1.0 s sound in a 1.5 s clip faded out from 1.25 s, after the sound had ended."""
    assert encode.audio_fade_filter(1.5, 0.25, 1.0) == (
        ",asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.250:curve=qsin,"
        "afade=t=out:st=0.750:d=0.250:curve=qsin"
    )
    # still at most a quarter, now of the audio; no audio left: no fade
    assert "afade=t=out:st=0.300:d=0.100" in encode.audio_fade_filter(1.5, 0.25, 0.4)
    assert encode.audio_fade_filter(1.5, 0.25, 0.0) == ""


def _export_graph(monkeypatch, info: MediaInfo, start: float, end: float) -> str:
    calls: list[list[str]] = []
    monkeypatch.setattr(encode, "run", lambda args, **_: calls.append(args) or "")
    fmt = encode.OutputFormat(64, 48, 25)
    encode._encode(VideoEncoder("cpu"), info.path, info, start, end, Path("o.mp4"), fmt,
                   False, None, 0.25)  # fmt: skip
    return calls[0][calls[0].index("-filter_complex") + 1]


@pytest.mark.parametrize(
    ("audio_end", "fade_out"),
    [(None, "st=1.250"), (20.0, "st=1.250"), (11.0, "st=0.750")],
    ids=["unknown", "covers", "ends-early"],
)
def test_clip_export_fades_out_where_the_source_audio_ends(monkeypatch, audio_end, fade_out):
    info = media_info(Path("v.mp4"), 12.0, audio_duration=audio_end)
    assert f"afade=t=out:{fade_out}:d=0.250" in _export_graph(monkeypatch, info, 10.0, 11.5)


def test_assemble_fades_out_where_the_clip_audio_ends():
    chain = assemble._audio_chain(0, _info(1.5, 1.5), 0.25)
    assert chain.endswith(f"{FADE_NORMAL}[a0]")
    short = replace(_info(1.5, 1.5), audio_duration=1.0)
    assert "afade=t=out:st=0.750:d=0.250" in assemble._audio_chain(0, short, 0.25)


def _tail_peak_db(path: Path, seconds: float = 0.05) -> float:
    pcm = ffmpeg_utils.decode_pcm(path, sr=48000).astype(np.float64) / 32768
    return 20 * np.log10(np.abs(pcm[-int(48000 * seconds) :]).max() + 1e-9)


@needs_ffmpeg
def test_clip_export_fades_out_audio_that_ends_before_the_clip(tmp_path):
    """#250: a 1.0 s sine (-18.1 dB peak) in a clip of 1.5 s with a 0.25 s fade: its last
    50 ms were not faded, as the fade-out started at 1.25 s, after the audio."""
    src = tmp_path / "src.mp4"
    _clip(src, 1.5, 1.0)
    info = ffmpeg_utils.probe(src)
    assert info.audio_duration == pytest.approx(1.0, abs=0.03)
    out = tmp_path / "001.mp4"
    fmt = encode.OutputFormat(64, 48, 25)
    encode.export_clip(src, info, 0.0, 1.5, out, fmt, False, audio_fade=0.25)
    assert _tail_peak_db(out) < -25.0


@needs_ffmpeg
def test_clip_export_keeps_late_starting_audio_audible_to_its_end(tmp_path):
    """F1: audio delayed 0.3 s (``-itsoffset``) ends 0.3 s after its stream duration; the last
    second's audio went silent at 0.72 s when the delay was ignored."""
    src = tmp_path / "late.mp4"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
         "testsrc=size=64x48:rate=25:duration=5", "-itsoffset", "0.3", "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=48000:duration=5", "-t", "5", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)],
        check=True,
    )  # fmt: skip
    info = ffmpeg_utils.probe(src)
    assert info.audio_duration == pytest.approx(5.0, abs=0.05)
    out = tmp_path / "001.mp4"
    encode.export_clip(src, info, 4.0, 5.0, out, encode.OutputFormat(64, 48, 25), False,
                       audio_fade=0.25)  # fmt: skip
    pcm = ffmpeg_utils.decode_pcm(out, sr=48000).astype(np.float64) / 32768
    window = pcm[int(48000 * 0.74) : int(48000 * 0.8)]
    assert 20 * np.log10(np.abs(window).max() + 1e-9) > -25.0  # still audible before the fade
    assert _tail_peak_db(out) < -25.0  # and faded out at the end


class _Transcribed(Exception):
    """Raised by the stub Transcriber once its arguments are recorded."""


@pytest.mark.parametrize("loudnorm", [False, True], ids=["captions", "loudnorm-captions"])
@pytest.mark.parametrize("language", [None, "de"], ids=["auto", "forced"])
def test_burned_in_captions_transcribe_in_the_analysis_language(
    tmp_path, monkeypatch, language, loudnorm
):
    """#371: ``assemble --captions`` transcribes in the language the project's analysis
    forced (highlights.json ``settings.language``); an auto-detect project passes None."""
    project = tmp_path / "proj"
    clips_dir = project / "clips"
    clips_dir.mkdir(parents=True)
    (clips_dir / "001_a.mp4").write_bytes(b"x")
    hl = {"settings": {"language": language}}
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    seen: list[dict] = []

    def transcriber(**kw):
        seen.append(kw)
        raise _Transcribed

    def join(files, infos, clips_dir, out, *a):
        out.write_bytes(b"joined")
        return 0.0

    monkeypatch.setattr(assemble, "probe", lambda p: media_info(Path(p), 3.0))
    monkeypatch.setattr(assemble, "peak_filters", lambda _d, files, *_a: [""] * len(files))
    monkeypatch.setattr(assemble, "_join", join)
    monkeypatch.setattr(assemble, "extract_audio", lambda *a, **k: None)
    monkeypatch.setattr(assemble, "Transcriber", transcriber)
    with pytest.raises(_Transcribed):
        assemble.assemble(clips_dir, captions=True, loudnorm=loudnorm)
    assert seen == [{"model": None, "language": language}]


# --- assemble --audio-transition {cut,crossfade} ---


@pytest.mark.parametrize(
    ("transition", "audio", "refused"),
    [
        ("cut", "crossfade", True),
        ("cut", "cut", False),
        ("cut", None, False),
        ("fade", "crossfade", False),
        ("fade", "cut", False),
        ("fade", None, False),
        ("fade", "echo", True),
    ],
)
def test_check_transitions_refuses_only_an_audio_crossfade_at_cuts(transition, audio, refused):
    why = assemble.check_transitions(transition, audio)
    assert (why is not None) is refused
    if refused and audio == "crossfade":
        assert "--transition fade" in why


def test_the_audio_transition_follows_the_video_transition_by_default():
    assert assemble.default_audio_transition("cut") == "cut"
    assert assemble.default_audio_transition("fade") == "crossfade"


@pytest.mark.parametrize(
    ("fade", "fps", "cut"),
    [
        (0.5, Fraction(30), 7 / 30),  # 15 frames: the middle rounded down to 7
        (8 / 24, Fraction(24), 4 / 24),  # 8 frames: 4
        (1 / 25, Fraction(25), 0.0),  # one frame: cuts at its start
        (0.4, Fraction(0), 0.2),  # no frame rate: half the fade
    ],
)
def test_audio_cut_point_is_a_whole_frame_in_the_middle_of_the_fade(fade, fps, cut):
    assert assemble.audio_cut_point(fade, fps) == pytest.approx(cut, abs=1e-9)


def test_assemble_refuses_an_audio_crossfade_at_cuts_before_touching_anything(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    with pytest.raises(ValueError, match="--transition fade"):
        assemble.assemble(clips_dir, transition="cut", audio_transition="crossfade")
    assert not (tmp_path / "final.mp4").exists()


def test_fade_with_audio_cut_joins_the_audio_at_the_cut_point(tmp_path, monkeypatch):
    graphs = []
    monkeypatch.setattr(assemble, "_encode", lambda _files, graph, _out: graphs.append(graph))
    fps = Fraction(24)
    infos = [replace(_info(2.1, 2.0), fps=fps) for _ in range(3)]
    files = [tmp_path / f"00{i}_x.mp4" for i in range(3)]
    fade = assemble.crossfade_duration(infos, 1 / 3)  # 8 frames
    cut = assemble.audio_cut_point(fade, fps)
    assert cut == pytest.approx(4 / 24)
    assemble.crossfade(files, infos, tmp_path / "final.mp4", fade, audio="cut")
    graph = graphs[0]
    edge = f"{assemble.CUT_FADE:.6f}"
    middle = 2.0 - fade + cut
    # first: from 0 to vduration - fade + cut, faded out at the cut only
    assert (
        f"[a0]apad,atrim=start=0.000000:end={middle:.6f},asetpts=PTS-STARTPTS,"
        f"afade=t=out:st={middle - assemble.CUT_FADE:.6f}:d={edge}[at0]" in graph
    )
    # middle: starts at the cut, ends at the next one, faded at both
    span = middle - cut
    assert (
        f"[a1]apad,atrim=start={cut:.6f}:end={middle:.6f},asetpts=PTS-STARTPTS,"
        f"afade=t=in:d={edge},afade=t=out:st={span - assemble.CUT_FADE:.6f}:d={edge}[at1]" in graph
    )
    # last: from the cut to its video's end, faded in only
    assert (
        f"[a2]apad,atrim=start={cut:.6f}:end=2.000000,asetpts=PTS-STARTPTS,"
        f"afade=t=in:d={edge}[at2]" in graph
    )
    assert "[at0][at1][at2]concat=n=3:v=0:a=1[a]" in graph
    assert "acrossfade" not in graph
    assert graph.count("xfade=transition=fade") == 2  # the video still crossfades


def test_fade_with_audio_cut_adds_the_edge_fades_of_clips_without_baked_ones(tmp_path, monkeypatch):
    graphs = []
    monkeypatch.setattr(assemble, "_encode", lambda _files, graph, _out: graphs.append(graph))
    monkeypatch.setattr(assemble, "baked_fades", lambda _d, files: [0.25, 0.0])
    infos = [_info(2.0, 2.0), _info(2.0, 2.0)]
    files = [tmp_path / "001_a.mp4", tmp_path / "002_b.mp4"]
    used = assemble._join(
        files, infos, tmp_path, tmp_path / "final.mp4", "fade", 0.25, 0.3, None, "cut"
    )
    assert used > 0
    assert "acrossfade" not in graphs[0]
    chains = {c.split("[a")[-1][0]: c for c in graphs[0].split(";") if c.endswith(("[a0]", "[a1]"))}
    assert "curve=qsin" not in chains["0"]  # baked: no edge fade added
    assert "afade=t=in:st=0:d=0.300:curve=qsin" in chains["1"]  # the 0.3 s edge fade
    assert "afade=t=out:st=1.700:d=0.300:curve=qsin" in chains["1"]


def test_audio_crossfade_is_still_the_fade_default(tmp_path, monkeypatch):
    graphs = []
    monkeypatch.setattr(assemble, "_encode", lambda _files, graph, _out: graphs.append(graph))
    infos = [_info(2.0, 2.0), _info(2.0, 2.0)]
    files = [tmp_path / "001_a.mp4", tmp_path / "002_b.mp4"]
    assemble._join(files, infos, tmp_path, tmp_path / "final.mp4", "fade", 0.25, 0.25)
    assert "acrossfade" in graphs[0]


@needs_ffmpeg
def test_assemble_fade_with_audio_cut_keeps_audio_as_long_as_the_video(tmp_path, monkeypatch):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    for name, audio in zip(("001_a.mp4", "002_b.mp4", "003_c.mp4"), (2.3, 2.0, 2.0), strict=True):
        _clip(clips_dir / name, 2.0, audio)
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)  # no final.srt
    final = assemble.assemble(clips_dir, transition="fade", fade_dur=0.25, audio_transition="cut")
    infos = [assemble.probe(f) for f in assemble.list_clips(clips_dir, exclude={final})]
    fade = assemble.crossfade_duration(infos, 0.25)
    predicted = assemble.clip_offsets(infos, fade)[-1] + infos[-1].vduration
    durations = _stream_durations(final)
    assert durations["video"] == pytest.approx(predicted, abs=0.02)
    assert durations["audio"] == pytest.approx(durations["video"], abs=0.04)  # one frame
    assert "Audio Crossfade" not in final.with_suffix(".fcpxml").read_text(encoding="utf-8")


def _mean_db(path: Path, start: float, dur: float) -> float:
    out = subprocess.run(
        [shutil.which("ffmpeg"), "-hide_banner", "-nostats", "-ss", str(start), "-t", str(dur),
         "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    ).stderr  # fmt: skip
    return float(out.split("mean_volume:")[1].split("dB")[0])


@needs_ffmpeg
def test_loudnorm_gain_is_linear_when_the_peaks_do_not_fit(tmp_path):
    """A quiet part, a burst near 0 dBFS, a part 6 dB louder: the gain to -14 LUFS pushes the
    burst far over the ceiling, where ffmpeg's loudnorm left linear mode for dynamic (AGC)
    and re-levelled the two parts (#523). The gain stays linear; the limiter takes the burst."""
    src, out = tmp_path / "src.wav", tmp_path / "out.wav"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
         "aevalsrc=if(lt(t\\,4)\\,0.03\\,if(lt(t\\,4.3)\\,0.8\\,0.06))*sin(2*PI*440*t)"
         ":s=48000:d=8", str(src)],
        check=True,
    )  # fmt: skip
    af = f"{assemble.loudnorm_filter(src)},{assemble.limiter_filter(assemble.LIMIT_START_DB)}"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-i", str(src), "-af", af, str(out)], check=True
    )
    gap_in = _mean_db(src, 5, 3) - _mean_db(src, 0.5, 3)
    gap_out = _mean_db(out, 5, 3) - _mean_db(out, 0.5, 3)
    assert gap_in == pytest.approx(6.0, abs=0.15)
    assert gap_out == pytest.approx(gap_in, abs=0.2)
