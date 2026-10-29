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

from conftest import media_info, needs_ffmpeg
from cutpoint import assemble, encode, sources
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
    assert "re-encoding the whole video (001_a.mp4 is not tagged BT.709" in caplog.text


def test_edge_keyframes_lie_inside_the_clip_and_are_idr_on_nvenc():
    assert encode.edge_keyframes(VideoEncoder("cpu"), 4.0) == ["-force_key_frames", "0.500,3.500"]
    assert encode.edge_keyframes(VideoEncoder("nvenc"), 4.0)[-2:] == ["-forced-idr", "1"]
    assert encode.edge_keyframes(VideoEncoder("cpu"), 0.4) == []


def test_amf_gets_no_forced_keyframes():
    # AMF makes them plain I frames: only the size cost, no keyframe in the MP4.
    assert encode.edge_keyframes(VideoEncoder("amf"), 4.0) == []


SPANS = [(0.5, 5.5), (6.0, 10.2), (10.5, 15.5)]  # source seconds of the three clips
SHORT = [(0.5, 4.5), (5.0, 8.5), (9.0, 12.0)]  # 4, 3.5 and 3 s


def _export_clips(tmp_path: Path, rate: str = "30", spans=SPANS) -> Path:
    src = tmp_path / "src.mp4"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
         f"testsrc2=size=320x180:rate={rate}:duration=16,"
         "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
         "-f", "lavfi", "-i", "sine=frequency=330:beep_factor=3:sample_rate=48000:duration=16",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc",
         "bt709", "-colorspace", "bt709", "-color_range", "tv", "-c:a", "aac", str(src)],
        check=True,
    )  # fmt: skip
    clips = tmp_path / "clips"
    clips.mkdir()
    fps = round(float(Fraction(rate)))
    info, fmt, entries = probe(src), encode.OutputFormat(320, 180, fps), []
    if rate != str(fps):  # export_clip writes the rate into the filter: keep 30000/1001
        fmt = encode.OutputFormat(320, 180, rate)  # type: ignore[arg-type]
    for n, (a, b) in enumerate(spans, 1):
        name = f"{n:03d}_m.mp4"
        encode.export_clip(src, info, a, b, clips / name, fmt, False, audio_fade=0.25)
        entries.append({"file": name, "source": str(src), "start": a, "end": b, "audio_fade": 0.25})
    (clips / "manifest.json").write_text(json.dumps({"clips": entries}), encoding="utf-8")
    return clips


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
    clip = _export_clips(tmp_path) / "001_m.mp4"
    keys = assemble._video_packets(clip, Fraction(30)).keys
    assert {15, 135} <= set(keys)  # 0.5 s in and 0.5 s before frame 150


@needs_ffmpeg
@pytest.mark.parametrize("audio", ["crossfade", "cut"])
def test_the_copied_crossfade_has_the_old_frames_and_audio(tmp_path, monkeypatch, caplog, audio):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)  # no final.srt
    clips = _export_clips(tmp_path)
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
    clips = _export_clips(tmp_path, "60", SHORT)
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


def test_a_piece_whose_duration_is_off_the_frame_grid_does_not_check_out(monkeypatch, caplog):
    # 44 frames at 60 fps stored as 733 ms: the next piece would start 30 ticks early.
    caplog.set_level(logging.DEBUG)
    ref = VideoPackets(44, (0,), (0.0,), Fraction(1, 90000), "MD5:x", 44 / 60)
    for duration, ok in ((44 / 60, True), (0.733, False)):
        monkeypatch.setattr(
            assemble, "_video_packets", lambda *a, d=duration: replace(ref, duration=d)
        )
        assert assemble._piece_ok(Path("0001.mp4"), Fraction(60), 44, ref) is ok
        assert assemble._piece_ok(Path("final.mp4"), Fraction(60), 44, ref, timed=False)
    assert "piece 0001.mp4 does not check out: lasts 0.733000s, not 0.733333s" in caplog.text


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
    clips = _export_clips(folder, spans=SHORT)
    caplog.set_level(logging.INFO)
    final = assemble.assemble(clips, transition="fade", fade_dur=0.25)
    assert "stream-copied the rest" in caplog.text
    assert _frames_and_audio(final)[0] == str(120 + 105 + 90 - 2 * 8)
    assert not list(final.parent.glob(".*fadecopy*"))


@needs_ffmpeg
def test_a_missing_copied_piece_falls_back_to_the_old_join(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)  # no final.srt
    clips = _export_clips(tmp_path, spans=SHORT)
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
    assert args.count("-ss") == 1 and args[args.index("-ss") + 1] == f"{9.0 - half:.6f}"
    assert args.index("-ss") < args.index("-i")  # an input option of the first clip
    assert "trim=end_frame=10," in graph.split("[v2]")[0].split(";")[-1]  # only the last clip
    assert graph.count("trim=end_frame") == 1
    assert args[args.index("-frames:v") + 1] == str(30 + 40 + 10 - 2 * 10)
    assert args[args.index("-video_track_timescale") + 1] == "90000"


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"vcodec": "hevc"}, "not H.264"),
        ({"pix_fmt": "yuv422p"}, "not H.264"),
        ({"width": 160}, "size or frame rate"),
        ({"fps": Fraction(24)}, "size or frame rate"),
        ({"color_range": None}, "not tagged BT.709"),
        ({"color_range": "pc"}, "not tagged BT.709"),
        ({"color_space": "smpte170m"}, "not tagged BT.709"),
        ({"color_transfer": None}, "not tagged BT.709"),
    ],
    ids=["codec", "pix_fmt", "size", "fps", "untagged", "full-range", "matrix", "transfer"],
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


_VIDEO = (
    "scale=1920:1080:force_original_aspect_ratio=decrease:out_range=tv,"
    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,"
    "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709,format=yuv420p,"
    "settb=AVTB,setpts=PTS-STARTPTS"
)
_A_IN = "aresample=48000,aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
# Captured from crossfade() before _crossfade_chains was split out of it (#463).
_OLD_GRAPHS = {
    "crossfade": (
        f"[0:v:0]{_VIDEO}[v0];[0:a:0]{_A_IN},asetpts=PTS-STARTPTS,acompressor[a0];"
        f"[1:v:0]{_VIDEO}[v1];[1:a:0]{_A_IN},asetpts=PTS-STARTPTS[a1];"
        "[a0]apad,atrim=end=5.000000,asetpts=PTS-STARTPTS[at0];"
        "[a1]apad,atrim=end=5.000000,asetpts=PTS-STARTPTS[at1];"
        "[v0][v1]xfade=transition=fade:duration=1.000000:offset=3.983333[v];"
        "[at0][at1]acrossfade=d=1.000000[a]"
    ),
    "cut": (
        f"[0:v:0]{_VIDEO}[v0];[0:a:0]{_A_IN},asetpts=PTS-STARTPTS,acompressor,"
        "asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.250:curve=qsin,"
        "afade=t=out:st=4.750:d=0.250:curve=qsin[a0];"
        f"[1:v:0]{_VIDEO}[v1];[1:a:0]{_A_IN},asetpts=PTS-STARTPTS[a1];"
        "[a0]apad,atrim=start=0.000000:end=4.500000,asetpts=PTS-STARTPTS,"
        "afade=t=out:st=4.480000:d=0.020000[at0];"
        "[a1]apad,atrim=start=0.500000:end=5.000000,asetpts=PTS-STARTPTS,"
        "afade=t=in:d=0.020000[at1];"
        "[at0][at1]concat=n=2:v=0:a=1[a];"
        "[v0][v1]xfade=transition=fade:duration=1.000000:offset=3.983333[v]"
    ),
}


@pytest.mark.parametrize("audio", ["crossfade", "cut"])
def test_the_crossfade_graph_is_the_one_from_before_the_split(audio):
    chains = assemble._crossfade_chains(
        _graph_infos(), 1.0, [",acompressor", ""], audio, [0.25, 0.0]
    )
    assert ";".join(chains) == _OLD_GRAPHS[audio]


@pytest.mark.parametrize("audio", ["crossfade", "cut"])
def test_the_audio_only_chains_are_the_old_audio_chains_on_shifted_inputs(audio):
    infos = _graph_infos(3)
    infos[2] = replace(infos[2], audio_streams=0, acodec=None)  # silence, no input index
    args = (infos, 1.0, [",acompressor", "", ""], audio, [0.25, 0.0, 0.25])
    expect = []
    for c in (c for c in assemble._crossfade_chains(*args) if ":v:0" not in c and "xfade" not in c):
        for i in (1, 0):  # from the top, so [0:a:0] -> [1:a:0] is not shifted again
            c = c.replace(f"[{i}:a:0]", f"[{i + 1}:a:0]")
        expect.append(c)
    got = assemble._crossfade_chains(*args, video=False, base=1)
    assert got == expect
    assert not any(":v:" in c or "xfade" in c for c in got)


def test_the_fade_copy_switch_forces_the_old_join(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("CUTPOINT_FADE_COPY", "0")
    monkeypatch.setattr(assemble, "_video_packets", lambda *a: pytest.fail("probed"))
    joined = []
    monkeypatch.setattr(assemble, "crossfade", lambda *a, **_: joined.append("old") or a[3])
    infos = _pair(tmp_path)
    files = [i.path for i in infos]
    used = assemble._join(files, infos, tmp_path, tmp_path / "final.mp4", "fade", 1.0, 0.25)
    assert (used, joined) == (1.0, ["old"])
    assert "CUTPOINT_FADE_COPY=0" in caplog.text
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
    clips = _export_clips(tmp_path, rate, SHORT)
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
    clips = _export_clips(tmp_path, "30", SHORT)
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
