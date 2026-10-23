"""Media I/O primitives in ``cutpoint.ffmpeg_utils``.

Argument building, JPEG splitting, error handling, the CUDA -> CPU fallback and
staging run on fakes (no ffmpeg needed). The tests at the bottom decode tiny
synthetic inputs and are skipped when ffmpeg / ffprobe are not on PATH.
"""

from __future__ import annotations

import functools
import io
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from cutpoint import assemble as asm
from cutpoint import cache, crash, refmatch, stage
from cutpoint import ffmpeg_utils as fu
from cutpoint.ffmpeg_utils import FFmpegError
from cutpoint.pano import x3


def _jpeg(payload: bytes) -> bytes:
    return b"\xff\xd8" + payload + b"\xff\xd9"


# --------------------------------------------------------------------------- fakes
class _Proc:
    """Stand-in for a ``subprocess.Popen`` decode process."""

    def __init__(self, stdout: bytes, stderr: bytes = b"", returncode: int = 0) -> None:
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.killed = False

    def poll(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.killed = True

    def wait(self) -> int:
        return self.returncode


@pytest.fixture
def fake_tools(monkeypatch):
    """No real executables: ``find_tool`` returns the bare name."""
    monkeypatch.setattr(fu, "find_tool", lambda name: name)


@pytest.fixture
def popen(monkeypatch, fake_tools):
    """Queue fake decode processes; records every argv in ``.cmds``."""
    state = SimpleNamespace(procs=[], cmds=[])

    def fake(cmd, **_kw):
        state.cmds.append(cmd)
        return state.procs[len(state.cmds) - 1]

    monkeypatch.setattr(fu.subprocess, "Popen", fake)
    return state


@pytest.fixture
def run(monkeypatch, fake_tools):
    """Queue fake ``subprocess.run`` results; records every argv in ``.cmds``.

    A result is ``(stdout, stderr, returncode)``, an exception to raise, or a
    callable taking the argv and returning a ``CompletedProcess``.
    """
    state = SimpleNamespace(results=[], cmds=[])

    def fake(cmd, **_kw):
        state.cmds.append(cmd)
        res = state.results[len(state.cmds) - 1]
        if isinstance(res, BaseException):
            raise res
        if callable(res):
            return res(cmd)
        stdout, stderr, code = res
        return subprocess.CompletedProcess(cmd, code, stdout, stderr)

    monkeypatch.setattr(fu.subprocess, "run", fake)
    return state


# --------------------------------------------------------------------------- _split_jpegs
def test_split_jpegs_splits_concatenated_images():
    a, b = _jpeg(b"one"), _jpeg(b"two")
    assert fu._split_jpegs(a + b) == [a, b]


def test_split_jpegs_skips_garbage_and_drops_truncated_tail():
    a = _jpeg(b"ok")
    assert fu._split_jpegs(b"noise" + a + b"junk\xff\xd8truncated") == [a]


# --------------------------------------------------------------------------- argv builders


def test_decode_raw_args_cuda_pass_keeps_frames_on_gpu_with_vf_cuda():
    args = fu.decode_raw_args("in.mp4", hw=True, vf="scale=8:8", vf_cuda="scale_cuda=8:8")
    assert args[args.index("-hwaccel") + 1] == "cuda"
    assert args[args.index("-hwaccel_output_format") + 1] == "cuda"
    assert args.index("-hwaccel") < args.index("-i")
    assert args[args.index("-vf") + 1] == "scale_cuda=8:8"


def test_decode_raw_args_cuda_pass_without_vf_cuda_downloads_frames():
    args = fu.decode_raw_args("in.mp4", hw=True, vf="fps=2,crop=4:4:0:0", pix_fmt="bgr24")
    assert "-hwaccel_output_format" not in args
    assert args[args.index("-vf") + 1] == "fps=2,crop=4:4:0:0"
    assert args[args.index("-pix_fmt") + 1] == "bgr24"


def test_decode_raw_args_pts_mode():
    args = fu.decode_raw_args(
        "in.mp4", hw=False, vf="fps=1,scale=8:8", keyframes_only=True, pts=True
    )
    assert args[args.index("-v") + 1] == "info"  # showinfo logs at info level
    assert args[args.index("-skip_frame") + 1] == "nokey"
    assert args.index("-skip_frame") < args.index("-i")
    assert args[args.index("-vf") + 1] == "fps=1,scale=8:8,showinfo"
    assert args[args.index("-fps_mode") + 1] == "passthrough"


def test_grab_images_args_single_png_by_height_on_cuda():
    args = fu.grab_images_args("in.mp4", 2.5, hw=True, height=360, fmt="png")
    assert args[args.index("-hwaccel") + 1] == "cuda"
    assert args[args.index("-vf") + 1] == "scale=-2:360"
    assert args[args.index("-c:v") + 1] == "png"
    assert "-t" not in args and "-q:v" not in args
    assert args[args.index("-frames:v") + 1] == "1"


def test_decode_pcm_args_variants():
    mixed = fu.decode_pcm_args("in.mp4", sr=16000, start=1.0, duration=2.0, audio_streams=3)
    assert mixed[mixed.index("-filter_complex") + 1] == (
        "[0:a:0][0:a:1][0:a:2]amix=inputs=3:normalize=0:duration=longest[a]"
    )
    assert mixed[mixed.index("-map") + 1] == "[a]"
    single = fu.decode_pcm_args("in.mp4", sr=16000, audio_streams=1)
    assert single[single.index("-map") + 1] == "0:a:0"
    default = fu.decode_pcm_args("in.mp4", sr=8000)
    assert "-map" not in default and "-ss" not in default and "-t" not in default
    assert default[-7:] == ["-ac", "1", "-ar", "8000", "-f", "s16le", "-"]
    stereo = fu.decode_pcm_args("in.mp4", sr=8000, mono=False)
    assert "-ac" not in stereo


def test_ffprobe_args_passes_input_with_dash_i():
    args = fu.ffprobe_args(
        "in.mp4", streams="v:0", entries="frame=pts_time", options=("-read_intervals", "%+12")
    )
    assert args[-2:] == ["-i", "in.mp4"]
    assert args[args.index("-of") + 1] == "json"
    assert args[args.index("-select_streams") + 1] == "v:0"
    assert args[args.index("-read_intervals") + 1] == "%+12"


# --------------------------------------------------------------------------- errors
def test_ffprobe_json_raises_with_stderr_tail(run):
    run.results = [(b"", b"line1\nline2\nmoov atom not found\n", 1)]
    with pytest.raises(FFmpegError, match="moov atom not found"):
        fu.ffprobe_json("bad.mp4", show_streams=True)


@pytest.mark.parametrize(
    "call",
    [
        lambda: fu.ffprobe_json("x.mp4", timeout=5),
        lambda: fu.run(["ffmpeg", "-i", "x.mp4", "out.mp4"], timeout=3),
    ],
    ids=["ffprobe_json", "run"],
)
def test_timeout_becomes_ffmpeg_error(run, call):
    run.results = [subprocess.TimeoutExpired("ffmpeg", 3)]
    with pytest.raises(FFmpegError, match="timed out"):
        call()


def test_grab_images_raises_when_no_image(run):
    run.results = [(b"", b"some error\n", 0)]
    with pytest.raises(FFmpegError, match="some error"):
        fu.grab_images("x.mp4", 1.0, width=64)


# --------------------------------------------------------------------------- CUDA -> CPU
FRAME = bytes(2 * 2 * 3)


def test_decode_raw_falls_back_to_cpu_when_cuda_yields_nothing(popen):
    popen.procs = [_Proc(b"", b"cuda init failed\n", 1), _Proc(FRAME * 3)]
    out = list(fu.decode_raw("x.mp4", size=(2, 2), stamp_fps=2.0, start=10.0))
    assert len(popen.cmds) == 2
    assert "cuda" in popen.cmds[0] and "cuda" not in popen.cmds[1]
    assert [t for t, _ in out] == [10.0, 10.5, 11.0]
    assert all(f.shape == (2, 2, 3) for _, f in out)


def test_decode_raw_keeps_cuda_pass_that_yields_frames(popen):
    popen.procs = [_Proc(FRAME * 2, b"", 1)]  # exit code is not checked once frames came
    out = list(fu.decode_raw("x.mp4", size=(2, 2)))
    assert len(popen.cmds) == 1 and len(out) == 2
    assert [t for t, _ in out] == [0.0, 1.0]  # no fps, no pts: frame index


def test_decode_raw_without_fallback_raises_with_stderr(popen):
    popen.procs = [_Proc(b"", b"No device available for decoder\n", 1)]
    errors: list[str] = []
    with pytest.raises(FFmpegError, match="No device available"):
        list(fu.decode_raw("x.mp4", size=(2, 2), fallback_cpu=False, stderr_lines=errors))
    assert len(popen.cmds) == 1
    assert errors == ["No device available for decoder"]


def test_decode_raw_pts_timestamps_from_showinfo(popen):
    err = (
        b"Stream mapping: ...\n"
        b"[Parsed_showinfo_2 @ 0] n:0 pts:0 pts_time:0.25 duration:1\n"
        b"[Parsed_showinfo_2 @ 0] n:1 pts:9 pts_time:3.75 duration:1\n"
    )
    popen.procs = [_Proc(FRAME * 2, err)]
    out = list(fu.decode_raw("x.mp4", size=(2, 2), pts=True, hwaccel=None))
    assert [t for t, _ in out] == [0.25, 3.75]


def test_decode_raw_kills_process_when_consumer_stops(popen):
    proc = _Proc(FRAME * 5)
    proc.poll = lambda: None  # still running
    popen.procs = [proc]
    gen = fu.decode_raw("x.mp4", size=(2, 2), hwaccel=None)
    next(gen)
    gen.close()
    assert proc.killed


def test_grab_images_falls_back_to_cpu(run):
    a = _jpeg(b"cpu")
    run.results = [(b"", b"cuda failed\n", 1), (a, b"", 0)]
    assert fu.grab_images("x.mp4", 1.0, height=360, hwaccel="cuda") == [a]
    assert "cuda" in run.cmds[0] and "cuda" not in run.cmds[1]


# --------------------------------------------------------------------------- staging
@pytest.fixture
def staged(monkeypatch):
    """Every ``*.mp4`` input has a staged copy ``/stage/<name>``."""

    @functools.cache  # conftest calls staged_path.cache_clear() on teardown
    def fake(path: str) -> str:
        return f"/stage/{Path(path).name}" if path.endswith(".mp4") else path

    monkeypatch.setattr(stage, "staged_path", fake)


def test_primitives_read_staged_copy(staged, run, popen):
    run.results = [(b"{}", b"", 0), (b"\x00\x00", b"", 0), (_jpeg(b"x"), b"", 0)]
    fu.ffprobe_json("/nas/a.mp4", show_streams=True)
    fu.decode_pcm("/nas/a.mp4", sr=8000)
    fu.grab_images("/nas/a.mp4", 1.0, width=64)
    popen.procs = [_Proc(FRAME)]
    list(fu.decode_raw("/nas/a.mp4", size=(2, 2), hwaccel=None))
    for cmd in [*run.cmds, *popen.cmds]:
        assert cmd[cmd.index("-i") + 1] == "/stage/a.mp4", cmd
        assert "/nas/a.mp4" not in cmd


def test_x3_probe_reads_staged_copy(staged, run):
    """Regression: the Insta360 ffprobe bypassed ``stage_inputs``."""
    run.results = [(b'{"streams": [{"width": 1024, "height": 512}]}', b"", 0)]
    assert x3.probe_video_stream(Path("/nas/clip.mp4")) == {"width": 1024, "height": 512}
    assert "/stage/clip.mp4" in run.cmds[0]


def test_gpmf_copy_reads_staged_copy(staged, run, tmp_path):
    """Regression: the GoPro telemetry remux bypassed ``stage_inputs``."""

    def remux(cmd):
        Path(cmd[-1]).write_bytes(b"gpmf")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    streams = b'{"streams": [{"index": 0, "codec_tag_string": "hvc1"}, '
    streams += b'{"index": 3, "codec_tag_string": "gpmd"}]}'
    run.results = [(streams, b"", 0), remux]
    out = crash._gpmf_copy(Path("/nas/GX010001.mp4"), tmp_path)
    assert out == tmp_path / "GX010001.gpmf.mp4"
    probe_cmd, remux_cmd = run.cmds
    assert probe_cmd[0] == "ffprobe" and "/stage/GX010001.mp4" in probe_cmd
    assert remux_cmd[0] == "ffmpeg" and "/stage/GX010001.mp4" in remux_cmd
    assert remux_cmd[remux_cmd.index("-map") + 1] == "0:3"


# --------------------------------------------------------------------------- timeouts
class _BlockingProc(_Proc):
    """A decode that writes ``frames`` frames, then hangs until killed."""

    def __init__(self, frames: int = 0) -> None:
        super().__init__(b"")
        self._frames = frames
        self._dead = threading.Event()
        self.stdout = SimpleNamespace(read=self._read, close=lambda: None)

    def _read(self, n: int) -> bytes:
        if self._frames:
            self._frames -= 1
            return bytes(n)
        self._dead.wait(10)  # blocks like a stalled ffmpeg pipe
        return b""

    def poll(self):
        return 1 if self._dead.is_set() else None

    def kill(self) -> None:
        self.killed = True
        self._dead.set()


def test_decode_raw_timeout_kills_and_raises(popen):
    proc = _BlockingProc()
    popen.procs = [proc]
    with pytest.raises(FFmpegError, match="timed out"):
        list(fu.decode_raw("x.mp4", size=(2, 2), hwaccel=None, timeout=0.05))
    assert proc.killed


def test_decode_raw_timeout_after_frames_raises_after_them(popen):
    popen.procs = [_BlockingProc(frames=2)]
    got = []
    with pytest.raises(FFmpegError, match="timed out"):
        for item in fu.decode_raw("x.mp4", size=(2, 2), hwaccel=None, timeout=0.05):
            got.append(item)
    assert len(got) == 2


def test_decode_raw_timer_after_process_exit_is_ignored(popen):
    """The timer firing after ffmpeg exited (EOF not read yet) is not a timeout."""

    class _SlowEof(_Proc):
        def __init__(self) -> None:
            super().__init__(b"")
            self._left = 2
            self.stdout = SimpleNamespace(read=self._read, close=lambda: None)

        def _read(self, n: int) -> bytes:
            if self._left:
                self._left -= 1
                return bytes(n)
            time.sleep(0.3)  # the timer (0.05 s) fires here; the process has exited
            return b""

    proc = _SlowEof()  # poll() returns 0: exited
    popen.procs = [proc]
    out = list(fu.decode_raw("x.mp4", size=(2, 2), hwaccel=None, timeout=0.05))
    assert len(out) == 2
    assert not proc.killed


def test_decode_window_returns_empty_arrays_on_ffmpeg_error(monkeypatch):
    def failing(*_a, **_kw):
        raise FFmpegError("ffmpeg decoded no frames")
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(
        fu, "probe", lambda _p: SimpleNamespace(width=1920, height=1080, duration=10.0)
    )
    monkeypatch.setattr(fu, "decode_raw", failing)
    ts, frames = crash.decode_window(Path("x.mp4"), 1.0, 3.0, 4.0)
    assert ts.shape == (0,)
    assert frames.shape == (0, 252, 448, 3) and frames.dtype == np.uint8


# --------------------------------------------------------------------------- caching
@pytest.fixture
def thumbs_env(monkeypatch, tmp_path):
    """refmatch.thumbnails with a 10 s video and a fake decoder (frames per pass)."""
    state = SimpleNamespace(per_pass={"cuda": 100, None: 100}, calls=[])
    monkeypatch.setattr(cache, "source_dir", lambda _p: tmp_path)
    monkeypatch.setattr(refmatch, "probe", lambda _p: SimpleNamespace(vduration=10.0))

    def fake_decode(_path, *, size, vf, pix_fmt, hwaccel, fallback_cpu=True):
        state.calls.append((hwaccel, fallback_cpu))
        n = state.per_pass[hwaccel]
        if isinstance(n, Exception):
            raise n
        w, h = size
        for k in range(n):
            yield k / refmatch.FPS, np.full((h, w), k % 256, np.uint8)

    monkeypatch.setattr(refmatch, "decode_raw", fake_decode)
    state.cpath = tmp_path / refmatch.THUMBS_FILE
    return state


def test_thumbnails_full_decode_is_cached(thumbs_env):
    frames = refmatch.thumbnails(Path("x.mp4"))
    assert frames.shape == (100, refmatch.W * refmatch.H)
    assert frames[3, 0] == 3
    assert thumbs_env.calls == [("cuda", False)]  # no hidden CPU pass in decode_raw
    assert thumbs_env.cpath.exists()


def test_thumbnails_short_cuda_decode_is_redone_on_cpu(thumbs_env):
    thumbs_env.per_pass["cuda"] = 40
    frames = refmatch.thumbnails(Path("x.mp4"))
    assert thumbs_env.calls == [("cuda", False), (None, True)]
    assert len(frames) == 100 and thumbs_env.cpath.exists()


def test_thumbnails_still_short_is_not_cached(thumbs_env):
    thumbs_env.per_pass.update({"cuda": 40, None: 60})
    frames = refmatch.thumbnails(Path("x.mp4"))
    assert len(frames) == 60
    assert not thumbs_env.cpath.exists()


def test_thumbnails_failed_cuda_decode_runs_cpu(thumbs_env):
    thumbs_env.per_pass["cuda"] = FFmpegError("No device")
    frames = refmatch.thumbnails(Path("x.mp4"))
    assert thumbs_env.calls == [("cuda", False), (None, True)]
    assert len(frames) == 100 and thumbs_env.cpath.exists()


def test_thumbnails_keeps_cuda_frames_when_cpu_retry_fails(thumbs_env):
    thumbs_env.per_pass.update({"cuda": 40, None: FFmpegError("read error")})
    frames = refmatch.thumbnails(Path("x.mp4"))
    assert len(frames) == 40
    assert not thumbs_env.cpath.exists()


def test_thumbnails_raise_when_neither_pass_decodes(thumbs_env):
    thumbs_env.per_pass.update({"cuda": FFmpegError("a"), None: FFmpegError("b")})
    with pytest.raises(FFmpegError, match="Could not decode thumbnails"):
        refmatch.thumbnails(Path("x.mp4"))
    assert not thumbs_env.cpath.exists()


@pytest.fixture
def audio_env(monkeypatch, tmp_path):
    """refmatch.audio_8k with a failing decoder; ``streams`` = probed audio streams."""
    state = SimpleNamespace(streams=1, decodes=0, cpath=tmp_path / "audio8k.npy")

    def failing(*_a, **_kw):
        state.decodes += 1
        raise FFmpegError("decode failed")

    monkeypatch.setattr(cache, "source_dir", lambda _p: tmp_path)
    monkeypatch.setattr(refmatch, "decode_pcm", failing)
    monkeypatch.setattr(refmatch, "probe", lambda _p: SimpleNamespace(audio_streams=state.streams))
    monkeypatch.setattr(refmatch, "_AUDIO_FAILED", set())
    return state


def test_audio_8k_no_audio_stream_is_cached(audio_env):
    audio_env.streams = 0
    x = refmatch.audio_8k(Path("x.mp4"))
    assert x.shape == (0,) and x.dtype == np.float32
    assert audio_env.cpath.exists()


def test_audio_8k_other_failure_not_cached_nor_retried_in_process(audio_env, caplog):
    for _ in range(3):
        x = refmatch.audio_8k(Path("x.mp4"))
        assert x.shape == (0,) and x.dtype == np.float32
    assert not audio_env.cpath.exists()
    assert audio_env.decodes == 1
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_x3_failed_probe_is_not_cached(monkeypatch, tmp_path):
    insv = tmp_path / "VID_00_000.insv"
    insv.write_bytes(b"x")
    monkeypatch.setattr(cache, "source_dir", lambda _p: tmp_path)
    monkeypatch.setattr(x3, "probe_video_stream", lambda _p: {})
    out = x3._cached_probe({}, [insv])
    assert out["width"] == 0
    assert not list(tmp_path.glob("x3-probe-*.json"))
    monkeypatch.setattr(
        x3, "probe_video_stream", lambda _p: {"width": 1024, "height": 512, "nb_frames": "9"}
    )
    assert x3._cached_probe({}, [insv])["frames"] == 9
    assert list(tmp_path.glob("x3-probe-*.json"))


# --------------------------------------------------------------------------- real ffmpeg
needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not on PATH"
)


@pytest.fixture
def clip(tmp_path) -> Path:
    """2 s, 64x48, 10 fps test pattern with a 440 Hz tone (skips without libx264)."""
    encoders = subprocess.run(
        [shutil.which("ffmpeg"), "-hide_banner", "-encoders"], capture_output=True, text=True
    ).stdout
    if "libx264" not in encoders:
        pytest.skip("ffmpeg has no libx264 encoder")
    out = tmp_path / "clip.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=2",
            "-c:v",
            "libx264",
            "-g",
            "5",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@needs_ffmpeg
def test_real_ffprobe_json(clip):
    data = fu.ffprobe_json(clip, streams="v:0", entries="stream=width,height")
    assert data["streams"][0] == {"width": 64, "height": 48}
    assert fu.probe(clip).audio_streams == 1


@needs_ffmpeg
def test_real_decode_raw(clip):
    out = list(
        fu.decode_raw(
            clip,
            size=(32, 24),
            vf="fps=5,scale=32:24:flags=area,format=rgb24",
            stamp_fps=5,
            start=0.4,
            duration=1.0,
            hwaccel=None,
        )
    )
    assert len(out) == 5
    assert out[1][0] == pytest.approx(0.6)
    assert out[0][1].shape == (24, 32, 3) and out[0][1].dtype == np.uint8
    gray = list(
        fu.decode_raw(
            clip, size=(16, 12), vf="scale=16:12,format=gray", pix_fmt="gray", hwaccel=None
        )
    )
    assert len(gray) == 20 and gray[0][1].shape == (12, 16)


@needs_ffmpeg
def test_real_decode_raw_pts_keyframes(clip):
    out = list(
        fu.decode_raw(
            clip, size=(16, 12), vf="scale=16:12", keyframes_only=True, pts=True, hwaccel=None
        )
    )
    assert [round(t, 2) for t, _ in out] == [0.0, 0.5, 1.0, 1.5]  # -g 5 at 10 fps


@needs_ffmpeg
def test_real_decode_raw_no_frames_raises(clip):
    with pytest.raises(FFmpegError, match="no frames"):
        list(fu.decode_raw(clip, size=(16, 12), vf="scale=16:12", start=30.0, hwaccel=None))


@needs_ffmpeg
def test_real_grab_images(clip):
    jpegs = fu.grab_images(clip, 0.0, count=3, duration=1.5, fps="1/0.5", width=32, quality=3)
    assert len(jpegs) == 3
    img = cv2.imdecode(np.frombuffer(jpegs[0], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (24, 32, 3)
    (png,) = fu.grab_images(clip, 1.0, height=24, fmt="png")
    assert png.startswith(b"\x89PNG")
    with pytest.raises(FFmpegError):
        fu.grab_images(clip, 30.0, width=32)


@needs_ffmpeg
def test_real_decode_pcm(clip):
    pcm = fu.decode_pcm(clip, sr=8000, start=0.5, duration=1.0)
    assert pcm.dtype == np.dtype("<i2")
    assert abs(len(pcm) - 8000) <= 80
    assert np.abs(pcm).max() > 1000  # the tone, not silence
    window = fu.read_pcm_window(clip, fu.probe(clip), 0.5, 1.5)
    assert abs(len(window) // 2 - 16000) <= 160


# --------------------------------------------------------------------------- assemble
def test_baked_fades_follow_the_manifest(tmp_path):
    cache.save_json(tmp_path / "manifest.json", {"clips": [{"file": "a.mp4", "audio_fade": 0.25}]})
    assert asm.baked_fades(tmp_path, [tmp_path / "a.mp4", tmp_path / "b.mp4"]) == [0.25, 0.0]


def test_caption_chunks_split_on_punctuation_and_gaps():
    def word(text, start, end):
        return {"word": text, "start": start, "end": end}

    seg = {
        "words": [
            word("hello", 0.0, 0.3),
            word("there.", 0.3, 0.7),  # sentence end: new chunk
            word("ok", 0.8, 1.0),
            word("then", 2.5, 2.8),  # long pause: new chunk
        ]
    }
    chunks = asm.caption_chunks([seg])
    assert [text for _, _, text in chunks] == ["hello there.", "ok", "then"]
    assert all(a < b for a, b, _ in chunks)
    assert all(chunks[i][1] <= chunks[i + 1][0] for i in range(len(chunks) - 1))  # no overlap


@needs_ffmpeg
def test_assemble_joins_two_clips(clip, tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    for name in ("clip_1.mp4", "clip_2.mp4"):
        shutil.copy(clip, clips / name)
    out = asm.assemble(clips)
    assert out == tmp_path.resolve() / "final.mp4"
    info = fu.probe(out)
    assert info.duration == pytest.approx(4.0, abs=0.3)  # two 2 s clips
    assert info.audio_streams == 1
    assert out.with_suffix(".otio").is_file() and out.with_suffix(".fcpxml").is_file()
