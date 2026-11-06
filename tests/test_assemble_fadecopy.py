"""assemble: the stream-copied crossfade (#463) gives the frames and audio of the old one."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from assemble_helpers import SHORT, export_clips
from conftest import media_info, needs_ffmpeg
from cutpoint import assemble, encode, jsonout, sources
from cutpoint.assemble import Body, Run, VideoPackets
from cutpoint.ffmpeg_utils import VideoEncoder, probe


def _packets(frames: int, keys: tuple[int, ...]) -> VideoPackets:
    return VideoPackets(frames, keys, tuple(k / 30 for k in keys), Fraction(1, 90000), "MD5:x")


def test_the_plan_copies_between_the_keyframes_next_to_each_fade():
    # keyframes 0.5, 1.0 and 1.5 s from both ends at 30 fps (a clip with extra keyframes)
    edges = (0, 15, 30, 45, 255, 270, 285)
    clips = [_packets(300, edges) for _ in range(3)]
    plan = assemble.fade_copy_plan(clips, 30)
    assert plan == [
        Body(0, 0, 270), Run(0, 270, 1, 30), Body(1, 30, 270), Run(1, 270, 2, 30),
        Body(2, 30, 300),
    ]  # fmt: skip
    pieces = sum(i.end - i.start if isinstance(i, Body) else 0 for i in plan)
    runs = sum(assemble.run_frames(i, [300] * 3, 30) for i in plan if isinstance(i, Run))
    assert pieces + runs == 3 * 300 - 2 * 30  # the old path's frames


def test_a_clip_without_a_keyframe_between_its_fades_stays_in_the_run():
    plan = assemble.fade_copy_plan([_packets(300, (0, 250)), _packets(100, (0,)),
                                    _packets(300, (0, 250))], 20)  # fmt: skip
    assert Run(0, 250, 2, 250) in plan
    assert not any(isinstance(i, Body) and i.clip == 1 for i in plan)


def test_too_much_to_re_encode_gives_no_plan():
    assert assemble.fade_copy_plan([_packets(100, (0,)), _packets(100, (0,))], 20) is None


def test_clips_not_tagged_bt709_fall_back_with_one_reason(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(assemble, "ffprobe_json", lambda *a, **k: pytest.fail("probed"))
    infos = [media_info(tmp_path / "001_a.mp4", 5.0), media_info(tmp_path / "002_b.mp4", 5.0)]
    files = [i.path for i in infos]
    assert not assemble.crossfade_copy(files, infos, tmp_path / "final.mp4", 0.5)
    assert "001_a.mp4 is not tagged BT.709" in caplog.text


def test_edge_keyframes_lie_inside_the_clip_and_are_idr_on_nvenc():
    assert encode.edge_keyframes(VideoEncoder("cpu"), 4.0) == ["-force_key_frames", "0.500,3.500"]
    assert encode.edge_keyframes(VideoEncoder("nvenc"), 4.0)[-2:] == ["-forced-idr", "1"]
    assert encode.edge_keyframes(VideoEncoder("cpu"), 0.4) == []
    # AMF makes them plain I frames: only the size cost, no keyframe in the MP4.
    assert encode.edge_keyframes(VideoEncoder("amf"), 4.0) == []


def _frames_and_audio(path: Path) -> tuple[str, str]:
    frames = subprocess.run(
        [shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    pcm = subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le",
         "-"], capture_output=True, check=True,
    ).stdout  # fmt: skip
    return frames, hashlib.md5(pcm).hexdigest()


@needs_ffmpeg
def test_exported_clips_get_keyframes_near_both_ends(tmp_path):
    clip = export_clips(tmp_path) / "001_m.mp4"
    keys = assemble._video_packets(clip, Fraction(30)).keys
    assert {15, 135} <= set(keys)  # 0.5 s in and 0.5 s before frame 150


@needs_ffmpeg
@pytest.mark.parametrize("audio", ["crossfade", "cut"])
def test_the_copied_crossfade_has_the_old_frames_and_audio(tmp_path, monkeypatch, caplog, audio):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)  # no final.srt
    clips = export_clips(tmp_path)
    kw = {"transition": "fade", "fade_dur": 0.25, "audio_transition": audio}
    monkeypatch.setenv("CUTPOINT_FADE_COPY", "0")
    old = assemble.assemble(clips, **kw)
    shutil.move(old, tmp_path / "old.mp4")
    monkeypatch.delenv("CUTPOINT_FADE_COPY")
    caplog.clear()
    caplog.set_level(logging.INFO)
    new = assemble.assemble(clips, **kw)
    assert "stream-copied the rest" in caplog.text
    assert _frames_and_audio(new) == _frames_and_audio(tmp_path / "old.mp4")
    assert _frames_and_audio(new)[0] == str(150 + 126 + 150 - 2 * 8)  # 0.25 s = 8 frames
    assert not list(new.parent.glob(".*fadecopy*"))


def _movie_duration(path: Path) -> Fraction:
    """The duration in the MP4's movie header (mvhd), in its own time scale."""
    data = path.read_bytes()
    at = data.index(b"mvhd") + 4
    if data[at] == 1:  # version 1: 64-bit times
        scale, length = int.from_bytes(data[at + 20 : at + 24]), data[at + 24 : at + 32]
    else:
        scale, length = int.from_bytes(data[at + 12 : at + 16]), data[at + 16 : at + 20]
    return Fraction(int.from_bytes(length), scale)


@needs_ffmpeg
def test_a_60_fps_window_of_no_whole_milliseconds_takes_the_fast_path(
    tmp_path, monkeypatch, caplog
):
    # 0.5 s keyframes and a 16-frame fade at 60 fps: 30 + 30 - 16 = 44-frame windows, 0.7333 s.
    # In a 1000 Hz movie header the concat demuxer started the next piece off the frame grid.
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)  # no final.srt
    clips = export_clips(tmp_path, "60", SHORT)
    kw = {"transition": "fade", "fade_dur": 16 / 60}
    monkeypatch.setenv("CUTPOINT_FADE_COPY", "0")
    old = shutil.move(assemble.assemble(clips, **kw), tmp_path / "old.mp4")
    monkeypatch.delenv("CUTPOINT_FADE_COPY")
    windows, real = [], assemble._encode_run

    def encode_run(enc, files, infos, packets, run_, fade, out):
        real(enc, files, infos, packets, run_, fade, out)
        frames = assemble.run_frames(run_, [p.frames for p in packets], round(fade * 60))
        windows.append((frames, _movie_duration(out), out.parent.name))

    monkeypatch.setattr(assemble, "_encode_run", encode_run)
    caplog.clear()
    caplog.set_level(logging.INFO)
    new = assemble.assemble(clips, **kw)
    assert "stream-copied the rest" in caplog.text
    assert "re-encoding the whole video" not in caplog.text
    assert [w[:2] for w in windows] == [(44, Fraction(44, 60))] * 2
    assert {w[2] for w in windows} == {".final.part.fadecopy"}  # not "..final.part"
    assert _frames_and_audio(new) == _frames_and_audio(old)
    assert _frames_and_audio(new)[0] == str(240 + 210 + 180 - 2 * 16)
    assert assemble._video_packets(new, Fraction(60)) is not None  # one frame apart throughout
    assert not list(new.parent.glob(".*fadecopy*"))


def test_a_piece_whose_duration_is_off_the_frame_grid_does_not_check_out(monkeypatch):
    # 44 frames at 60 fps stored as 733 ms: the next piece would start 30 ticks early.
    ref = VideoPackets(44, (0,), (0.0,), Fraction(1, 90000), "MD5:x", 44 / 60)
    for duration, ok in ((44 / 60, True), (0.733, False)):
        monkeypatch.setattr(
            assemble, "_video_packets", lambda *a, d=duration: replace(ref, duration=d)
        )
        assert assemble._piece_ok(Path("0001.mp4"), Fraction(60), 44, ref) is ok
        assert assemble._piece_ok(Path("final.mp4"), Fraction(60), 44, ref, timed=False)


def _listed(listing: Path) -> list[tuple[str, int]]:
    """The (file, duration in microseconds) pairs of a concat listing."""
    lines = listing.read_text(encoding="utf-8").splitlines()
    files = [ln.removeprefix("file ") for ln in lines[0::2]]
    durations = [ln.removeprefix("duration ") for ln in lines[1::2]]
    assert len(files) == len(durations) == len(lines) // 2 and len(lines) % 2 == 0
    return [(f, round(Fraction(d) * 10**6)) for f, d in zip(files, durations, strict=True)]


@pytest.mark.parametrize("rate", [Fraction(60), Fraction(30), Fraction(30000, 1001)])
def test_the_listed_durations_add_up_to_the_frame_grid_for_every_piece(tmp_path, rate):
    # Fade windows of no whole microseconds between bodies of any length, 60 joins: the demuxer
    # starts piece k at the sum of the durations before it, which must be T_k to half a us.
    frames = [237, 44, 181, 22, 300, 44, 7] * 10
    pieces = [(tmp_path / f"{n:04d}.mp4", f) for n, f in enumerate(frames)]
    listing = tmp_path / "pieces.txt"
    assemble._write_pieces(listing, pieces, rate)
    listed = _listed(listing)
    assert [f for f, _ in listed] == [f"'{p.resolve().as_posix()}'" for p, _ in pieces]
    start = 0
    for k, (_, us) in enumerate(listed):
        start += us
        exact = Fraction(sum(frames[: k + 1])) / rate * 10**6
        assert abs(start - exact) <= Fraction(1, 2)
    assert abs(start - Fraction(sum(frames)) / rate * 10**6) <= Fraction(1, 2)


def test_a_quote_in_a_piece_path_is_escaped_in_the_listing(tmp_path):
    listing = tmp_path / "pieces.txt"
    assemble._write_pieces(listing, [(tmp_path / "it's.mp4", 30)], Fraction(60))
    assert listing.read_text(encoding="utf-8").splitlines() == [
        "file '" + (tmp_path / "it").resolve().as_posix() + "'\\''s.mp4'",
        "duration 0.500000",
    ]


@needs_ffmpeg
@pytest.mark.parametrize(("rate", "frames"), [("60", 44), ("30000/1001", 22)])
def test_many_windows_of_no_whole_microseconds_join_on_the_frame_grid(tmp_path, rate, frames):
    # 24 fade windows written as _encode_run writes them (90 kHz movie and track time scale):
    # summed in whole microseconds, the 24th started a tick early (frame 748 at 60 fps).
    window = tmp_path / "window.mp4"
    subprocess.run(
        [shutil.which("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i",
         f"testsrc2=size=64x36:rate={rate}", "-frames:v", str(frames), "-an", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-video_track_timescale", "90000", "-movie_timescale", "90000",
         str(window)],
        check=True,
    )  # fmt: skip
    listing, joined = tmp_path / "pieces.txt", tmp_path / "joined.mp4"
    assemble._write_pieces(listing, [(window, frames)] * 24, Fraction(rate))
    subprocess.run(
        [shutil.which("ffmpeg"), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i",
         str(listing), "-c", "copy", str(joined)],
        check=True,
    )  # fmt: skip
    got = assemble._video_packets(joined, Fraction(rate))  # None unless every step is a frame
    assert got is not None and got.frames == 24 * frames


@needs_ffmpeg
def test_a_percent_sign_in_the_project_path_still_takes_the_fast_path(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)  # no final.srt
    folder = tmp_path / "100%d 50%"
    folder.mkdir()
    clips = export_clips(folder, spans=SHORT)
    caplog.set_level(logging.INFO)
    final = assemble.assemble(clips, transition="fade", fade_dur=0.25)
    assert "stream-copied the rest" in caplog.text
    assert _frames_and_audio(final)[0] == str(120 + 105 + 90 - 2 * 8)
    assert not list(final.parent.glob(".*fadecopy*"))


@needs_ffmpeg
def test_a_missing_copied_piece_falls_back_to_the_old_join(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)  # no final.srt
    clips = export_clips(tmp_path, spans=SHORT)
    real = assemble.run

    def no_segments(args, **kw):  # the segment muxer "succeeds" but writes no pieces
        return "" if "segment" in args else real(args, **kw)

    monkeypatch.setattr(assemble, "run", no_segments)
    caplog.set_level(logging.INFO)
    final = assemble.assemble(clips, transition="fade", fade_dur=0.25)
    assert "re-encoding the whole video (ffmpeg failed: the segment muxer wrote no piece" in (
        caplog.text
    )
    assert "stream-copied the rest" not in caplog.text
    assert _frames_and_audio(final)[0] == str(120 + 105 + 90 - 2 * 8)
    assert not list(final.parent.glob(".*fadecopy*"))


# ------------------------------------------------------------------ unit tests, no ffmpeg
def _tagged(path: Path, duration: float = 5.0, **over):
    """A 320x180 30 fps clip tagged BT.709 TV range, as ``cutpoint clips`` exports it."""
    fields = {"width": 320, "height": 180, "color_range": "tv", "color_space": "bt709",
              "color_primaries": "bt709", "color_transfer": "bt709"}  # fmt: skip
    return media_info(path, duration, **(fields | over))


def _pair(tmp_path: Path, **over):
    a, b = tmp_path / "001_a.mp4", tmp_path / "002_b.mp4"
    return [_tagged(a), _tagged(b, **over)]


def _output_frames(plan: list[Body | Run], frames: list[int], fade_frames: int) -> int:
    return sum(
        i.end - i.start if isinstance(i, Body) else assemble.run_frames(i, frames, fade_frames)
        for i in plan
    )


def test_the_first_clip_copies_only_from_its_start_and_the_last_to_its_end():
    keys = (0, 30, 270)
    clips = [_packets(300, (0,)), _packets(300, keys), _packets(250, keys)]
    plan = assemble.fade_copy_plan(clips, 30)
    # The first clip has no keyframe but 0 to cut at: no body, its fade run starts at 0.
    assert plan == [Run(0, 0, 1, 30), Body(1, 30, 270), Run(1, 270, 2, 30), Body(2, 30, 250)]
    assert _output_frames(plan, [300, 300, 250], 30) == 300 + 300 + 250 - 2 * 30


def test_the_total_frames_are_the_clips_less_one_fade_per_join_for_any_plan():
    for fade in (1, 10, 30):
        clips = [_packets(300, (0, 40, 100, 200, 260)) for _ in range(4)]
        plan = assemble.fade_copy_plan(clips, fade)
        assert plan is not None
        assert _output_frames(plan, [300] * 4, fade) == 4 * 300 - 3 * fade


def test_a_run_encodes_its_fades_at_the_old_graphs_offsets(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(assemble, "run", lambda args: calls.append(args))
    infos = [_tagged(tmp_path / f"00{i}.mp4", d) for i, d in enumerate([10.0, 40 / 30, 10.0])]
    files = [i.path for i in infos]
    packets = [
        VideoPackets(f, (0, 270) if f == 300 else (0,), (0.0, 9.0) if f == 300 else (0.0,),
                     Fraction(1, 90000), "MD5:x")
        for f in (300, 40, 300)
    ]  # fmt: skip
    fade = 10 / 30
    run_ = Run(0, 270, 2, 10)
    assemble._encode_run(VideoEncoder("cpu"), files, infos, packets, run_, fade, tmp_path / "o.mp4")
    args = calls[0]
    half, offsets = 0.5 / 30, assemble.clip_offsets(infos, fade)
    t0 = offsets[0] + 270 / 30
    graph = args[args.index("-filter_complex") + 1]
    for n in (1, 2):
        want = f"xfade=transition=fade:duration={fade:.6f}:offset={offsets[n] - t0 - half:.6f}"
        assert want in graph
    assert args[args.index("-ss") + 1] == f"{9.0 - half:.6f}"  # from the first clip's keyframe
    assert args[args.index("-frames:v") + 1] == str(30 + 40 + 10 - 2 * 10)


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"vcodec": "hevc"}, "not H.264"),
        ({"fps": Fraction(24)}, "size or frame rate"),
        ({"color_range": "pc"}, "not tagged BT.709"),
        ({"color_space": "smpte170m"}, "not tagged BT.709"),
    ],
    ids=["codec", "fps", "full-range", "matrix"],
)
def test_clips_that_differ_in_their_stream_fall_back(tmp_path, over, reason):
    assert assemble._why_no_copy(_pair(tmp_path)) is None  # the same two clips are fine
    assert reason in assemble._why_no_copy(_pair(tmp_path, **over))


@pytest.mark.parametrize(
    ("second", "reason"),
    [
        (None, "not constant frame rate"),
        (VideoPackets(149, (0,), (0.0,), Fraction(1, 90000), "MD5:x"), "149 frames"),
        (VideoPackets(150, (0,), (0.0,), Fraction(1, 15360), "MD5:x"), "another time base"),
        (VideoPackets(150, (0,), (0.0,), Fraction(1, 90000), "MD5:y"), "other settings"),
    ],
    ids=["non-cfr", "frame-count", "time-base", "extradata"],
)
def test_clips_whose_packets_differ_fall_back(tmp_path, second, reason):
    infos = _pair(tmp_path)
    first = VideoPackets(150, (0,), (0.0,), Fraction(1, 90000), "MD5:x")
    assert assemble._why_no_copy(infos, [first, first]) is None
    assert reason in assemble._why_no_copy(infos, [first, second])


def _graph_infos(n: int = 2):
    infos = [media_info(Path(f"00{i}.mp4"), 5.0, color_space="bt709") for i in range(n)]
    return infos


@pytest.mark.parametrize("audio", ["crossfade", "cut"])
def test_the_audio_only_chains_read_the_shifted_audio_inputs(audio):
    infos = _graph_infos(3)
    infos[2] = replace(infos[2], audio_streams=0, acodec=None)  # silence, no input index
    args = (infos, 1.0, [",acompressor", "", ""], audio, [0.25, 0.0, 0.25])
    got = ";".join(assemble._crossfade_chains(*args, video=False, base=1))
    assert ":v:" not in got and "xfade" not in got
    assert "[1:a:0]" in got and "[2:a:0]" in got and "[0:a:0]" not in got


def test_the_fade_copy_switch_forces_the_old_join(tmp_path, monkeypatch):
    monkeypatch.setenv("CUTPOINT_FADE_COPY", "0")
    monkeypatch.setattr(assemble, "_video_packets", lambda *a: pytest.fail("probed"))
    joined = []
    monkeypatch.setattr(assemble, "crossfade", lambda *a, **_: joined.append("old") or a[3])
    infos = _pair(tmp_path)
    files = [i.path for i in infos]
    used = assemble._join(files, infos, tmp_path, tmp_path / "final.mp4", "fade", 1.0, 0.25)
    assert (used, joined) == (1.0, ["old"])
    assert not list(tmp_path.glob(".*fadecopy*"))


# ------------------------------------------------------------------ ffmpeg tests
def _frame_hashes(path: Path) -> list[str]:
    out = subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "framemd5",
         "-"], capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return [ln.split(",")[-1].strip() for ln in out.splitlines() if ln and not ln.startswith("#")]


def _make_clip(path: Path, *, size="320x180", rate="30", tagged=True, preset="medium", dur=3.5):
    """A clip the fast path cannot join to the exported ones: other size, no tags or settings."""
    tags = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-color_range", "tv"] if tagged else []  # fmt: skip
    setp = (",setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
            if tagged else "")  # fmt: skip
    subprocess.run(
        [shutil.which("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i",
         f"testsrc2=size={size}:rate={rate}:duration={dur}{setp}", "-f", "lavfi", "-i",
         f"sine=frequency=440:sample_rate=48000:duration={dur}", "-c:v", "libx264", "-preset",
         preset, "-pix_fmt", "yuv420p", *tags, "-c:a", "aac", "-video_track_timescale", "90000", str(path)],
        check=True,
    )  # fmt: skip


def _words(_src, _lang=None):
    return [{"word": f"w{n}", "start": n * 0.5, "end": n * 0.5 + 0.4} for n in range(32)]


def _assemble_to(clips: Path, folder: Path, **kw) -> Path:
    return assemble.assemble(clips, folder / "final.mp4", **kw)


@needs_ffmpeg
@pytest.mark.slow
@pytest.mark.parametrize("rate", ["30", "24", "30000/1001"])
@pytest.mark.parametrize(
    ("transition", "audio"),
    [("cut", None), ("fade", "crossfade"), ("fade", "cut")],
    ids=["cut+cut", "fade+crossfade", "fade+audio-cut"],
)
def test_the_copied_join_equals_the_old_one_in_everything_but_its_encoder_bytes(
    tmp_path, monkeypatch, caplog, rate, transition, audio
):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(sources, "source_words", _words)
    clips = export_clips(tmp_path, rate, SHORT)
    # The default fade: the forced keyframes 0.5 s from the ends cover it.
    kw = {"transition": transition, "fade_dur": 0.25, "audio_transition": audio}
    monkeypatch.setenv("CUTPOINT_FADE_COPY", "0")
    old = _assemble_to(clips, tmp_path / "old", **kw)
    monkeypatch.delenv("CUTPOINT_FADE_COPY")
    caplog.clear()
    caplog.set_level(logging.INFO)
    new = _assemble_to(clips, tmp_path / "new", **kw)
    assert ("stream-copied the rest" in caplog.text) == (transition == "fade")
    assert _frames_and_audio(new) == _frames_and_audio(old)
    assert new.with_suffix(".srt").read_bytes() == old.with_suffix(".srt").read_bytes()
    assert new.with_suffix(".srt").stat().st_size > 0
    assert not list(new.parent.glob(".*fadecopy*"))
    if transition == "cut":
        return
    # The stream-copied bodies are the clips' own frames, bit for bit, at their places.
    files = assemble.list_clips(clips, exclude=set())
    infos = [probe(f) for f in files]
    fps = infos[0].fps
    fade_frames = round(assemble.crossfade_duration(infos, 0.25) * float(fps))
    packets = [assemble._video_packets(f, fps) for f in files]
    plan = assemble.fade_copy_plan(packets, fade_frames)
    assert plan is not None and any(isinstance(i, Body) for i in plan)
    joined, at = _frame_hashes(new), 0
    for item in plan:
        if isinstance(item, Body):
            clip = _frame_hashes(files[item.clip])
            assert joined[at : at + item.end - item.start] == clip[item.start : item.end]
            at += item.end - item.start
        else:
            at += assemble.run_frames(item, [p.frames for p in packets], fade_frames)
    assert at == len(joined)


@needs_ffmpeg
@pytest.mark.slow
@pytest.mark.parametrize(
    ("odd", "reason"),
    [
        ({"size": "160x90"}, "differs from 001_m.mp4 in size or frame rate"),
        ({"tagged": False}, "002_m.mp4 is not tagged BT.709"),
        ({"preset": "ultrafast"}, "002_m.mp4 was encoded with other settings"),
    ],
    ids=["mixed-resolution", "untagged", "extradata"],
)
def test_clips_the_fast_path_cannot_join_get_the_old_join_and_the_reason(
    tmp_path, monkeypatch, caplog, odd, reason
):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)  # no final.srt
    clips = export_clips(tmp_path, "30", SHORT)
    _make_clip(clips / "002_m.mp4", **odd)
    caplog.set_level(logging.INFO)
    final = assemble.assemble(clips, transition="fade", fade_dur=1.0)
    assert "re-encoding the whole video (" in caplog.text
    assert reason in caplog.text
    assert "stream-copied the rest" not in caplog.text
    assert _frames_and_audio(final)[0] == str(120 + 105 + 90 - 2 * 30)
    assert not list(final.parent.glob(".*fadecopy*"))


def test_packet_listing_has_no_ffprobe_timeout(monkeypatch, tmp_path):
    """Every packet of a multi-hour clip on slow storage can take longer than the default."""
    calls = []
    monkeypatch.setattr(assemble, "ffprobe_json", lambda *a, **k: calls.append(k) or {})
    assert assemble._video_packets(tmp_path / "a.mp4", Fraction(30)) is None
    assert calls[0]["timeout"] is None


# ---------------------------------------------------------------- per-join fades (#768)
def _joined_frames(plan: list[Body | Run], frames: list[int], joins: list[int]) -> int:
    return sum(
        i.end - i.start if isinstance(i, Body) else assemble.run_frames(i, frames, joins[i.i])
        for i in plan
    )


KEYS = (0, 15, 30, 270)


def test_the_plan_copies_up_to_a_fade_and_from_a_fade_but_to_the_edge_of_a_cut():
    clips = [_packets(300, KEYS) for _ in range(3)]
    fade_cut = assemble.fade_copy_plan(clips, [30, 0])
    assert fade_cut == [Body(0, 0, 270), Run(0, 270, 1, 30), Body(1, 30, 300), Body(2, 0, 300)]
    assert _joined_frames(fade_cut, [300] * 3, [30, 0]) == 900 - 30
    cut_fade = assemble.fade_copy_plan(clips, [0, 30])
    assert cut_fade == [Body(0, 0, 300), Body(1, 0, 270), Run(1, 270, 2, 30), Body(2, 30, 300)]
    assert _joined_frames(cut_fade, [300] * 3, [0, 30]) == 900 - 30


def test_a_clip_with_no_keyframe_inside_its_fade_stays_in_the_run_next_to_a_cut():
    # clip 1 has no keyframe at 30 or after: all of it is the fade's run, and it ends in a cut
    clips = [_packets(300, KEYS), _packets(300, (0,)), _packets(300, KEYS)]
    plan = assemble.fade_copy_plan(clips, [30, 0])
    assert plan == [Body(0, 0, 270), Run(0, 270, 1, 300), Body(2, 0, 300)]
    # no keyframe before the fade out of the first clip: its run starts at the clip's start
    plan = assemble.fade_copy_plan([_packets(300, (0,)), *clips[:1]], [30])
    assert plan is not None and plan[0] == Run(0, 0, 1, 30)


def test_a_run_goes_through_a_clip_between_two_fades_and_stops_at_a_cut():
    clips = [_packets(300, KEYS), _packets(60, (0,)), _packets(300, KEYS), _packets(300, KEYS)]
    plan = assemble.fade_copy_plan(clips, [30, 30, 0])
    assert plan == [Body(0, 0, 270), Run(0, 270, 2, 30), Body(2, 30, 300), Body(3, 0, 300)]
    assert _joined_frames(plan, [300, 60, 300, 300], [30, 30, 0]) == 960 - 60


def test_a_whole_clip_between_two_cuts_is_one_body():
    clips = [_packets(300, KEYS) for _ in range(4)]
    plan = assemble.fade_copy_plan(clips, [30, 0, 0])
    assert plan is not None and Body(2, 0, 300) in plan


# ------------------------------------------------------------------ ffmpeg tests: mixed joins
FOUR = [(0.5, 4.5), (5.0, 8.5), (9.0, 12.0), (12.5, 15.5)]  # 120, 105, 90 and 90 frames


@needs_ffmpeg
@pytest.mark.parametrize(
    ("transition", "joins"),
    [
        ("cut", [("001_m.mp4", "002_m.mp4", "fade")]),
        ("fade", [("002_m.mp4", "003_m.mp4", "cut"), ("003_m.mp4", "004_m.mp4", "cut")]),
    ],
    ids=["one-fade-among-cuts", "two-cuts-among-fades"],
)
def test_mixed_joins_copy_to_the_frames_and_audio_of_the_re_encode(
    tmp_path, monkeypatch, caplog, transition, joins
):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)  # no final.srt
    clips = export_clips(tmp_path, "30", FOUR)
    entries = [{"from": a, "to": b, "transition": t} for a, b, t in joins]
    kw = {"transition": transition, "fade_dur": 0.25, "joins": entries}
    monkeypatch.setenv("CUTPOINT_FADE_COPY", "0")
    with jsonout.run("assemble"):
        old = _assemble_to(clips, tmp_path / "old", **kw)
    monkeypatch.delenv("CUTPOINT_FADE_COPY")
    caplog.clear()
    caplog.set_level(logging.INFO)
    with jsonout.run("assemble") as res:
        new = _assemble_to(clips, tmp_path / "new", **kw)
    assert "stream-copied the rest" in caplog.text
    assert "on 1 of 3 joins" in caplog.text
    # one 0.25 s (8 frame) overlap, whatever the other joins do; the old and new joins agree
    assert _frames_and_audio(new) == _frames_and_audio(old)
    assert _frames_and_audio(new)[0] == str(120 + 105 + 90 + 90 - 8)
    # clip 3 sits between two cuts and clip 4 ends the video: both copied bit for bit
    joined = _frame_hashes(new)
    assert joined[-180:] == _frame_hashes(clips / "003_m.mp4") + _frame_hashes(clips / "004_m.mp4")
    assert not list(new.parent.glob(".*fadecopy*"))
    assert [(j["from"], j["to"]) for j in res.outputs["joins"]] == [(a, b) for a, b, _ in joins]
    assert {j["transition"] for j in res.outputs["joins"]} == {joins[0][2]}


@needs_ffmpeg
def test_mixed_joins_without_fast_path_keep_the_frame_count(tmp_path, monkeypatch):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setenv("CUTPOINT_FADE_COPY", "0")
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)
    clips = export_clips(tmp_path, "30", FOUR)
    joins = [{"from": "003_m.mp4", "to": "004_m.mp4", "transition": "fade"}]
    final = _assemble_to(clips, tmp_path / "out", transition="cut", fade_dur=0.25, joins=joins)
    assert _frames_and_audio(final)[0] == str(120 + 105 + 90 + 90 - 8)


@needs_ffmpeg
def test_mixed_joins_with_audio_cut_copy_to_the_frames_and_audio_of_the_re_encode(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)
    clips = export_clips(tmp_path, "30", FOUR)
    joins = [
        {"from": "002_m.mp4", "to": "003_m.mp4", "transition": "cut"},
        {"from": "003_m.mp4", "to": "004_m.mp4", "transition": "cut"},
    ]
    kw = {"transition": "fade", "fade_dur": 0.25, "audio_transition": "cut", "joins": joins}
    monkeypatch.setenv("CUTPOINT_FADE_COPY", "0")
    old = _assemble_to(clips, tmp_path / "old", **kw)
    monkeypatch.delenv("CUTPOINT_FADE_COPY")
    caplog.set_level(logging.INFO)
    new = _assemble_to(clips, tmp_path / "new", **kw)
    assert "stream-copied the rest" in caplog.text
    assert "audio cut" in caplog.text
    assert _frames_and_audio(new) == _frames_and_audio(old)
    assert _frames_and_audio(new)[0] == str(120 + 105 + 90 + 90 - 8)


@needs_ffmpeg
def test_assemble_joins_file_through_the_cli_on_real_clips(tmp_path, home, monkeypatch, capsys):
    from conftest import envelope  # noqa: PLC0415
    from cutpoint import cli  # noqa: PLC0415

    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)
    clips = export_clips(tmp_path, "30", FOUR)
    file = tmp_path / "joins.json"
    entries = [
        {"from": "001_m.mp4", "to": "002_m.mp4", "transition": "fade"},
        {"from": "004_m.mp4", "to": "001_m.mp4", "transition": "fade"},  # not adjacent
    ]
    file.write_text(json.dumps({"joins": entries}), encoding="utf-8")
    out = tmp_path / "cli" / "final.mp4"
    argv = ["assemble", str(clips), "--out", str(out), "--joins", str(file), "--fade-dur", "0.25"]
    assert cli.main([*argv, "--json"]) == 0
    doc = envelope(capsys.readouterr().out)
    assert doc["ok"] is True
    assert doc["outputs"]["joins"] == [{**entries[0], "transition": "fade"}]
    assert [w["code"] for w in doc["warnings"]] == ["join_unmatched"]
    assert _frames_and_audio(out)[0] == str(120 + 105 + 90 + 90 - 8)
