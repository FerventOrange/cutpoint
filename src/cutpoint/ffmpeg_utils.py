"""Thin wrappers around the ffmpeg / ffprobe executables.

Media reads go through a few typed primitives: :func:`ffprobe_json` (metadata),
:func:`decode_raw` (raw video frames), :func:`grab_images` (JPEG/PNG stills) and
:func:`decode_pcm` (audio samples). Each one reads the staged copy of a source
(:func:`stage_inputs`), accepts an optional timeout and raises :class:`FFmpegError`
with the tail of ffmpeg's stderr on failure. Their argv builders
(``*_args``) are pure functions. Note the hardware defaults differ, as their
callers expect: :func:`decode_raw` tries CUDA first (``hwaccel="cuda"``), while
:func:`grab_images` decodes on the CPU unless asked (``hwaccel=None``).

The executables are located (in order) via the ``CUTPOINT_FFMPEG`` /
``CUTPOINT_FFPROBE`` environment variables, ``PATH``, and the WinGet package folder (so a fresh
``winget install Gyan.FFmpeg`` works before the shell's PATH is refreshed).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from cutpoint import paths, state

logger = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    """Raised when ffmpeg/ffprobe is missing or a command fails."""


@state.register_cache
@cache
def find_tool(name: str) -> str:
    """Return the absolute path of ``ffmpeg`` or ``ffprobe``.

    Args:
        name: Either ``"ffmpeg"`` or ``"ffprobe"``.

    Raises:
        FFmpegError: If the executable cannot be found anywhere.
    """
    env = paths.env(name.upper())
    if env and Path(env).is_file():
        return env
    on_path = shutil.which(name)
    if on_path:
        return on_path
    roots = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
        Path("C:/ffmpeg/bin"),
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "ffmpeg" / "bin",  # noqa: SIM112 - Windows spelling
    ]
    for root in roots:
        if not root.is_dir():
            continue
        hits = sorted(root.glob(f"**/{name}.exe"), key=lambda p: len(str(p)))
        if hits:
            return str(hits[0])
    raise FFmpegError(
        f"{name} not found. Install it (winget install --id Gyan.FFmpeg -e) or set "
        f"the CUTPOINT_{name.upper()} environment variable to the full path of {name}.exe."
    )


def stage_inputs(cmd: list[str]) -> list[str]:
    """Redirect every ``-i <source>`` of an ffmpeg command to its staged copy.

    ``cutpoint stage`` copies sources on slow or network storage to a local folder once; see
    :func:`cutpoint.stage.staged_path`. Arguments that are not staged
    sources are returned unchanged.
    """
    from cutpoint.stage import staged_path  # noqa: PLC0415

    out = list(cmd)
    for i in range(len(out) - 1):
        if out[i] == "-i":
            out[i + 1] = staged_path(str(out[i + 1]))
    return out


def run(
    args: list[str],
    *,
    capture: bool = True,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> str:
    """Run ffmpeg/ffprobe (first arg is the tool name) and return stderr+stdout.

    Raises:
        FFmpegError: If the process exits non-zero or runs longer than ``timeout`` s.
    """
    cmd = stage_inputs([find_tool(args[0]), *args[1:]])
    logger.debug("exec: %s", subprocess.list2cmdline(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as err:
        raise FFmpegError(f"{args[0]} timed out after {timeout:g}s") from err
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-15:])
        raise FFmpegError(f"{args[0]} failed (exit {proc.returncode}):\n{tail}")
    return out


@dataclass(frozen=True)
class ProgressResult:
    """Diagnostics from a :func:`run_progress` call.

    Attributes:
        stderr: Everything ffmpeg wrote to stderr (warnings included).
        out_time: Last output timestamp ffmpeg reported (s), i.e. how far it got.
    """

    stderr: str
    out_time: float


# ffmpeg exits 0 after these, leaving a silently truncated output. On Windows a
# bad sector typically surfaces as "Permission denied" (EACCES) or "I/O error".
READ_ERROR_RE = re.compile(
    r"Error during demuxing[^\n]*|Input/output error|I/O error|"
    r"Error reading[^\n]*|partial file",
    re.IGNORECASE,
)


def run_progress(args: list[str], total: float, label: str) -> ProgressResult:
    """Run an ffmpeg command, logging progress roughly every 10% for long inputs.

    Args:
        args: Command starting with "ffmpeg" (output path last).
        total: Expected media duration in seconds (for percentages).
        label: Short description for log lines.

    Returns:
        stderr text and the last reported output time, so callers can detect
        read errors that ffmpeg reports but does not fail on.

    Raises:
        FFmpegError: If the process exits non-zero.
    """
    cmd = stage_inputs([find_tool(args[0]), "-progress", "pipe:1", "-nostats", *args[1:]])
    t0, next_pct, out_time = time.perf_counter(), 10, 0.0
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8", errors="replace"
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if not line.startswith("out_time_us="):
                continue
            try:
                out_time = max(out_time, int(line.split("=", 1)[1]) / 1e6)
            except ValueError:
                continue
            pct = out_time / total * 100 if total else 0
            if total >= 600 and pct >= next_pct:
                logger.info("  %s: %d%% (%.0fs elapsed)", label, int(pct), time.perf_counter() - t0)
                next_pct = (int(pct) // 10 + 1) * 10
        proc.wait()
        err.seek(0)
        stderr = err.read().decode("utf-8", "replace")
        if proc.returncode != 0:
            tail = "\n".join(stderr.strip().splitlines()[-15:])
            raise FFmpegError(f"ffmpeg failed during {label} (exit {proc.returncode}):\n{tail}")
    return ProgressResult(stderr=stderr, out_time=out_time)


@dataclass(frozen=True)
class MediaInfo:
    """Subset of ffprobe output the toolkit cares about."""

    path: Path
    duration: float
    vduration: float
    width: int
    height: int
    fps: Fraction
    pix_fmt: str
    vcodec: str
    audio_streams: int
    acodec: str | None
    sample_rate: int | None
    channels: int | None

    def concat_key(self) -> tuple:
        """Parameters that must match for lossless concat-demuxer joining."""
        return (
            self.vcodec,
            self.width,
            self.height,
            self.fps,
            self.pix_fmt,
            self.acodec,
            self.sample_rate,
            self.channels,
            self.audio_streams,
        )


_STANDARD_FPS = [
    Fraction(24000, 1001),
    Fraction(24),
    Fraction(25),
    Fraction(30000, 1001),
    Fraction(30),
    Fraction(50),
    Fraction(60000, 1001),
    Fraction(60),
    Fraction(120),
    Fraction(144),
    Fraction(240),
]


def snap_fps(fps: Fraction) -> Fraction:
    """Snap a measured (often VFR-averaged) rate to the nearest standard rate."""
    best = min(_STANDARD_FPS, key=lambda s: abs(float(s) - float(fps)))
    return (
        best if abs(float(best) - float(fps)) / float(best) < 0.01 else fps.limit_denominator(1001)
    )


def probe(path: Path) -> MediaInfo:
    """Probe a media file with ffprobe (a staged copy is probed if present)."""
    data = ffprobe_json(path, show_format=True, show_streams=True)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    if video is None:
        raise FFmpegError(f"No video stream in {path}")
    rate = video.get("avg_frame_rate") or "0/0"
    if rate == "0/0":
        rate = video.get("r_frame_rate") or "30/1"
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    a0 = audios[0] if audios else {}
    return MediaInfo(
        path=Path(path),
        duration=duration,
        vduration=float(video.get("duration") or duration),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=snap_fps(Fraction(rate)),
        pix_fmt=video.get("pix_fmt", ""),
        vcodec=video.get("codec_name", ""),
        audio_streams=len(audios),
        acodec=a0.get("codec_name"),
        sample_rate=int(a0["sample_rate"]) if a0.get("sample_rate") else None,
        channels=a0.get("channels"),
    )


def audio_mix_filter(n_streams: int, input_idx: int = 0) -> str | None:
    """Return a filter mixing all audio tracks (e.g. game + separate mic track).

    Returns ``None`` when there is at most one audio stream.
    """
    if n_streams < 2:
        return None
    ins = "".join(f"[{input_idx}:a:{i}]" for i in range(n_streams))
    return f"{ins}amix=inputs={n_streams}:normalize=0:duration=longest"


def extract_audio(src: Path, dst: Path, info: MediaInfo, start: float = 0.0) -> ProgressResult:
    """Extract all audio tracks of ``src`` mixed down to 16 kHz mono WAV.

    Args:
        src: Source video.
        dst: Output WAV path.
        info: Probe result for ``src``.
        start: Seek here first (input seek, so the WAV starts at ``start``).

    Returns:
        ffmpeg's diagnostics; check them (and the WAV length) for read errors.
    """
    if info.audio_streams == 0:
        raise FFmpegError(f"No audio stream in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    seek = ["-ss", f"{start:.3f}"] if start > 0 else []
    args = ["ffmpeg", "-y", "-v", "error", *seek, "-i", str(src), "-vn"]
    mix = audio_mix_filter(info.audio_streams)
    if mix:
        args += ["-filter_complex", f"{mix}[a]", "-map", "[a]"]
    else:
        args += ["-map", "0:a:0"]
    args += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tmp)]
    res = run_progress(args, info.duration - start, f"audio extract {src.name}")
    tmp.replace(dst)
    return res


def wav_duration(path: Path) -> float:
    """Length of a PCM WAV file in seconds (header only, no decode)."""
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def read_window_ok(src: Path, t: float, dur: float = 3.0) -> bool:
    """Decode ``dur`` s of audio at ``t``; False if ffmpeg reports a read error.

    Used to find where a damaged file becomes readable again. It reads only a
    few MB, so probing a bad region a handful of times stays gentle on the disk.
    """
    cmd = [
        find_tool("ffmpeg"),
        "-hide_banner",
        "-nostats",
        "-v",
        "error",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(src),
        "-t",
        f"{dur:.3f}",
        "-vn",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(
        stage_inputs(cmd), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return proc.returncode == 0 and not READ_ERROR_RE.search(proc.stderr or "")


def read_pcm_window(src: Path, info: MediaInfo, t0: float, t1: float) -> bytes:
    """16 kHz mono s16le PCM of ``[t0, t1]`` decoded straight from ``src``.

    Fallback for when no cached analysis WAV covers the window.
    """
    start = max(0.0, t0)
    try:
        pcm = decode_pcm(
            src, sr=16000, start=start, duration=t1 - start, audio_streams=info.audio_streams
        )
    except FFmpegError as err:
        raise FFmpegError(f"Audio window {t0:.1f}-{t1:.1f}s of {src.name} failed: {err}") from err
    return pcm.tobytes()


_CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")


def cropdetect_at(src: Path, t: float, frames: int = 30) -> tuple[int, int, int, int] | None:
    """Run ``cropdetect`` on ``frames`` frames at ``t``; (w, h, x, y) or None."""
    cmd = [
        find_tool("ffmpeg"),
        "-hide_banner",
        "-nostats",
        "-ss",
        f"{t:.3f}",
        "-i",
        str(src),
        "-frames:v",
        str(frames),
        "-an",
        "-vf",
        "cropdetect=limit=24:round=2:reset=0",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(
        stage_inputs(cmd), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    hits = _CROP_RE.findall(proc.stderr or "")
    if proc.returncode != 0 or not hits:
        return None
    w, h, x, y = map(int, hits[-1])
    return (w, h, x, y) if w > 0 and h > 0 else None


_INTEGRATED_RE = re.compile(r"Integrated loudness:\s*I:\s*(-?[\d.]+|-inf) LUFS")
_TRUE_PEAK_RE = re.compile(r"True peak:\s*Peak:\s*(-?[\d.]+|-inf) dBFS")


def measure_loudness(path: Path) -> tuple[float, float]:
    """Integrated loudness (LUFS) and true peak (dBTP) of ``path``'s audio."""
    out = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-vn",
            "-af",
            "ebur128=peak=true:framelog=quiet",
            "-f",
            "null",
            "-",
        ]
    )
    i_m, tp_m = _INTEGRATED_RE.findall(out), _TRUE_PEAK_RE.findall(out)
    if not i_m or not tp_m:
        raise FFmpegError(f"Could not parse loudness of {path}")

    def num(v: str) -> float:
        return -120.0 if v == "-inf" else float(v)

    return num(i_m[-1]), num(tp_m[-1])


_EBUR_RE = re.compile(r"t:\s*([\d.]+)\s+TARGET:.*?M:\s*(-?[\d.]+|-inf)")


def momentary_loudness(wav: Path) -> tuple[list[float], list[float]]:
    """Return (times, momentary LUFS) at ~10 Hz using the ``ebur128`` filter."""
    out = run(
        [
            "ffmpeg",
            "-nostats",
            "-hide_banner",
            "-i",
            str(wav),
            "-filter_complex",
            "ebur128=peak=none",
            "-f",
            "null",
            "-",
        ]
    )
    times: list[float] = []
    values: list[float] = []
    for m in _EBUR_RE.finditer(out):
        times.append(float(m.group(1)))
        val = m.group(2)
        values.append(-120.0 if val == "-inf" else float(val))
    if not times:
        raise FFmpegError(f"Could not parse ebur128 output for {wav}")
    return times, values


@state.register_cache
@cache
def nvenc_available() -> bool:
    """Check whether h264_nvenc can actually encode on this machine."""
    try:
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=black:s=256x256:d=0.2",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ]
        )
        return True
    except FFmpegError:
        logger.info("h264_nvenc unavailable; falling back to libx264")
        return False


def video_encoder_args(quality: int = 20) -> list[str]:
    """Encoder arguments: NVENC when available, else libx264."""
    if nvenc_available():
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-rc",
            "vbr",
            "-cq",
            str(quality),
            "-b:v",
            "0",
            "-profile:v",
            "high",
        ]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(quality)]


def grab_frame_jpeg(src: Path, t: float, width: int = 640) -> bytes:
    """Decode a single frame at time ``t`` and return it as JPEG bytes."""
    try:
        return grab_images(src, t, width=width, quality=4)[0]
    except FFmpegError as err:
        raise FFmpegError(f"Frame grab failed at {t:.2f}s of {src}: {err}") from err


# --------------------------------------------------------------------------- media I/O primitives
HwAccel = Literal["cuda"] | None
ImageFormat = Literal["jpeg", "png"]
PixFmt = Literal["rgb24", "bgr24", "gray"]

# Bytes per pixel of the raw formats :func:`decode_raw` can return.
PIX_CHANNELS = {"rgb24": 3, "bgr24": 3, "gray": 1}

_STDERR_TAIL = 15
_PTS_RE = re.compile(rb"pts_time:\s*([-\d.]+)")
_PTS_WAIT_S = 30.0  # showinfo logs a frame before it is written; a lag this long is a bug


def _command(tool: str, args: Sequence[str]) -> list[str]:
    """Full argv for ``tool`` with every ``-i`` input redirected to its staged copy."""
    return stage_inputs([find_tool(tool), *args])


def _tail(stderr: bytes | str, n: int = _STDERR_TAIL) -> str:
    """Last ``n`` lines of a process's stderr."""
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr
    return "\n".join(text.strip().splitlines()[-n:])


def _exec(
    tool: str, args: Sequence[str], timeout: float | None, path: Path | str
) -> subprocess.CompletedProcess:
    """Run ``tool`` with staged inputs, capturing stdout/stderr as bytes.

    Raises:
        FFmpegError: If the process runs longer than ``timeout`` seconds. The exit
            code is not checked; callers decide what counts as failure.
    """
    cmd = _command(tool, args)
    logger.debug("exec: %s", subprocess.list2cmdline(cmd))
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as err:
        raise FFmpegError(f"{tool} timed out after {timeout:g}s on {path}") from err


def _hw_passes(hwaccel: HwAccel, fallback_cpu: bool) -> tuple[bool, ...]:
    """Decode passes to try: ``(True, False)`` = CUDA first, then the CPU."""
    if hwaccel not in (None, "cuda"):
        raise ValueError(f"unsupported hwaccel {hwaccel!r} (expected 'cuda' or None)")
    if hwaccel is None:
        return (False,)
    return (True, False) if fallback_cpu else (True,)


# ----------------------------------------------------------------- ffprobe
def ffprobe_args(
    path: Path | str,
    *,
    entries: str | None = None,
    streams: str | None = None,
    show_format: bool = False,
    show_streams: bool = False,
    options: Sequence[str] = (),
) -> list[str]:
    """ffprobe arguments (without the executable) for :func:`ffprobe_json`.

    The input is passed as ``-i <path>`` (ffprobe accepts it) so that
    :func:`stage_inputs` redirects it like any ffmpeg input.
    """
    args = ["-v", "error"]
    if streams is not None:
        args += ["-select_streams", streams]
    args += list(options)
    if entries is not None:
        args += ["-show_entries", entries]
    if show_format:
        args.append("-show_format")
    if show_streams:
        args.append("-show_streams")
    return [*args, "-of", "json", "-i", str(path)]


def ffprobe_json(
    path: Path | str,
    *,
    entries: str | None = None,
    streams: str | None = None,
    show_format: bool = False,
    show_streams: bool = False,
    options: Sequence[str] = (),
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run ffprobe on ``path`` (its staged copy if any) and return the parsed JSON.

    Args:
        path: Media file.
        entries: ``-show_entries`` spec, e.g. ``"stream=width,height"`` or
            ``"frame=pts_time"``.
        streams: ``-select_streams`` spec, e.g. ``"v:0"``.
        show_format: Add ``-show_format``.
        show_streams: Add ``-show_streams``.
        options: Extra ffprobe options, e.g. ``("-read_intervals", "%+12")``.
        timeout: Seconds before ffprobe is killed.

    Raises:
        FFmpegError: If ffprobe fails, times out or prints invalid JSON.
    """
    args = ffprobe_args(
        path,
        entries=entries,
        streams=streams,
        show_format=show_format,
        show_streams=show_streams,
        options=options,
    )
    proc = _exec("ffprobe", args, timeout, path)
    if proc.returncode != 0:
        raise FFmpegError(
            f"ffprobe failed on {path} (exit {proc.returncode}):\n{_tail(proc.stderr)}"
        )
    try:  # decoded leniently, as run() always has (tags may hold invalid UTF-8)
        return json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    except ValueError as err:
        raise FFmpegError(f"ffprobe printed invalid JSON for {path}: {err}") from err


# ----------------------------------------------------------------- raw video
def decode_raw_args(
    path: Path | str,
    *,
    hw: bool,
    vf: str | None = None,
    vf_cuda: str | None = None,
    pix_fmt: PixFmt = "rgb24",
    start: float | None = None,
    duration: float | None = None,
    keyframes_only: bool = False,
    pts: bool = False,
) -> list[str]:
    """ffmpeg arguments (without the executable) for one :func:`decode_raw` pass.

    ``hw`` adds ``-hwaccel cuda``; with ``vf_cuda`` the frames also stay on the GPU
    (``-hwaccel_output_format cuda``) and ``vf_cuda`` replaces ``vf``.
    """
    args = ["-hide_banner", "-nostats", "-v", "info" if pts else "error"]
    if keyframes_only:
        args += ["-skip_frame", "nokey"]
    if hw:
        args += ["-hwaccel", "cuda"]
        if vf_cuda is not None:
            args += ["-hwaccel_output_format", "cuda"]
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(path)]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += ["-an", "-sn", "-dn"]
    chain = vf_cuda if hw and vf_cuda is not None else vf
    if pts:
        chain = f"{chain},showinfo" if chain else "showinfo"
    if chain:
        args += ["-vf", chain]
    if pts:
        args += ["-fps_mode", "passthrough"]  # exactly one showinfo line per written frame
    return [*args, "-pix_fmt", pix_fmt, "-f", "rawvideo", "-"]


class _StderrLog:
    """stderr of one decode pass, filled by that pass's reader thread.

    Each pass gets its own instance (passed to its thread as an argument), so a
    reader that outlives its pass (the join has a timeout) can never write into
    the next pass's state.
    """

    def __init__(self, pts: bool, lines: list[str] | None) -> None:
        self.pts = pts
        self.lines = lines
        self.times: list[float] = []
        self.tail: deque[str] = deque(maxlen=20)
        self.cond = threading.Condition()

    def read(self, stream: Any) -> None:
        """Consume ``stream`` (lines of bytes) until EOF, then post an end marker."""
        for raw in stream:
            m = _PTS_RE.search(raw) if self.pts else None
            with self.cond:
                if m and b"showinfo" in raw.lower():
                    self.times.append(float(m.group(1)))
                else:
                    line = raw.decode("utf-8", "replace").rstrip()
                    self.tail.append(line)
                    if self.lines is not None:
                        self.lines.append(line)
                self.cond.notify_all()
        with self.cond:
            self.times.append(float("nan"))  # end marker
            self.cond.notify_all()

    def pts_time(self, k: int) -> float:
        """Presentation time of output frame ``k`` (waits for its showinfo line)."""
        with self.cond:
            self.cond.wait_for(lambda: len(self.times) > k, timeout=_PTS_WAIT_S)
            t = self.times[k] if len(self.times) > k else float("nan")
        return t if np.isfinite(t) else float(k)  # should not happen; keep the frame

    def text(self) -> str:
        """The last stderr lines (``showinfo`` excluded)."""
        return "\n".join(self.tail)


class _Deadline:
    """Kill a decode pass's process ``timeout`` wall-clock seconds after it started.

    The kill (and :attr:`expired`) happens only if the pass is still running:
    not after :meth:`finish` (end of output or early stop) and not once the
    process has exited, so a timer firing just after a complete decode is a no-op.
    """

    def __init__(self, proc: Any, timeout: float) -> None:
        self.expired = False
        self._proc = proc
        self._done = False
        self._lock = threading.Lock()
        self._timer = threading.Timer(timeout, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        with self._lock:
            if self._done or self._proc.poll() is not None:
                return
            self.expired = True
        self._proc.kill()

    def finish(self) -> None:
        """The pass is over: cancel the timer (a late firing does nothing)."""
        with self._lock:
            self._done = True
        self._timer.cancel()


def decode_raw(
    path: Path | str,
    *,
    size: tuple[int, int],
    vf: str | None = None,
    vf_cuda: str | None = None,
    pix_fmt: PixFmt = "rgb24",
    start: float | None = None,
    duration: float | None = None,
    stamp_fps: float | None = None,
    keyframes_only: bool = False,
    pts: bool = False,
    hwaccel: HwAccel = "cuda",
    fallback_cpu: bool = True,
    stderr_lines: list[str] | None = None,
    timeout: float | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Stream decoded frames of ``path`` (its staged copy if any) as raw arrays.

    One sequential ffmpeg pass writes ``rawvideo`` to a pipe; frames are yielded
    as they arrive. With ``hwaccel="cuda"`` the first pass decodes on NVDEC; if it
    yields no frames at all and ``fallback_cpu`` is set, a CPU pass follows. Once
    a pass has yielded a frame it is final (its exit code is not checked).

    Args:
        path: Media file.
        size: ``(width, height)`` of the frames the filter chain outputs.
        vf: Filter chain; it must output ``size`` (scale, fps, crop etc. stay in
            the caller's order).
        vf_cuda: Filter chain for the CUDA pass with frames kept on the GPU
            (``-hwaccel_output_format cuda``), e.g.
            ``"scale_cuda=W:H,hwdownload,format=nv12,..."``. ``None`` uses ``vf``
            on the CUDA pass too (ffmpeg downloads the frames).
        pix_fmt: Output pixel format (``-pix_fmt``): ``rgb24``, ``bgr24`` or ``gray``.
        start: Input seek (``-ss`` before ``-i``) in seconds; ``None`` = no seek.
        duration: Decode this many seconds (``-t``); ``None`` = to the end.
        stamp_fps: Rate the chain outputs, used only for timestamps (it adds no
            filter): frame ``k`` is stamped ``start + k / stamp_fps``. Ignored
            with ``pts``; without either, frame ``k`` is stamped ``k``.
        keyframes_only: Decode keyframes only (``-skip_frame nokey``).
        pts: Stamp frames with their presentation time (``showinfo`` is appended
            to the chain and frames pass through without duplication).
        hwaccel: ``"cuda"`` or ``None`` (CPU only).
        fallback_cpu: Retry on the CPU when the CUDA pass yields nothing.
        stderr_lines: If given, ffmpeg's stderr lines (``showinfo`` excluded) are
            appended to it as they arrive.
        timeout: Wall-clock seconds per pass, including the time the consumer
            spends between frames, before ffmpeg is killed. Frames already
            yielded are followed by :class:`FFmpegError`. A pass whose output
            already ended (or whose process exited) never times out.

    Yields:
        ``(t, frame)``: ``frame`` is a read-only ``uint8`` array of shape
        ``(height, width, 3)`` (``(height, width)`` for ``gray``).

    Raises:
        FFmpegError: If the last pass yields no frames, or a pass times out.
        ValueError: For an unknown ``pix_fmt`` or ``hwaccel``.
    """
    if pix_fmt not in PIX_CHANNELS:
        raise ValueError(f"unsupported pix_fmt {pix_fmt!r} (expected one of {list(PIX_CHANNELS)})")
    passes = _hw_passes(hwaccel, fallback_cpu)
    width, height = size
    channels = PIX_CHANNELS[pix_fmt]
    shape = (height, width, channels) if channels > 1 else (height, width)
    frame_bytes = width * height * channels
    t_base = start or 0.0
    for i, hw in enumerate(passes):
        args = decode_raw_args(
            path,
            hw=hw,
            vf=vf,
            vf_cuda=vf_cuda,
            pix_fmt=pix_fmt,
            start=start,
            duration=duration,
            keyframes_only=keyframes_only,
            pts=pts,
        )
        cmd = _command("ffmpeg", args)
        logger.debug("exec: %s", subprocess.list2cmdline(cmd))
        log = _StderrLog(pts, stderr_lines)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 4
        )
        reader = threading.Thread(target=log.read, args=(proc.stderr,), daemon=True)
        reader.start()
        deadline = _Deadline(proc, timeout) if timeout is not None else None
        k = 0
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                t = log.pts_time(k) if pts else (t_base + k / stamp_fps if stamp_fps else float(k))
                yield t, np.frombuffer(buf, np.uint8).reshape(shape)
                k += 1
        finally:
            if deadline is not None:
                deadline.finish()  # output ended or the consumer stopped: no timeout now
            proc.stdout.close()
            if proc.poll() is None:
                proc.kill()  # the consumer stopped early
            proc.wait()
            reader.join(timeout=5)
        if deadline is not None and deadline.expired:
            raise FFmpegError(
                f"ffmpeg timed out after {timeout:g}s decoding {path} (hw={hw}):\n{log.text()}"
            )
        if k:
            return
        if i + 1 < len(passes):
            logger.debug("decode of %s (hw=%s) produced no frames: %s", path, hw, log.text())
            continue
        raise FFmpegError(f"ffmpeg decoded no frames from {path} (hw={hw}):\n{log.text()}")


# ----------------------------------------------------------------- stills
def _split_jpegs(data: bytes) -> list[bytes]:
    """Split concatenated JPEGs (an ``image2pipe`` stream) at their SOI/EOI markers.

    A trailing image without an end marker (truncated output) is dropped.
    """
    images, pos = [], 0
    while (a := data.find(b"\xff\xd8", pos)) != -1:
        b = data.find(b"\xff\xd9", a)
        if b == -1:
            break
        images.append(data[a : b + 2])
        pos = b + 2
    return images


def grab_images_args(
    path: Path | str,
    start: float,
    *,
    hw: bool = False,
    count: int = 1,
    duration: float | None = None,
    fps: float | str | None = None,
    width: int | None = None,
    height: int | None = None,
    fmt: ImageFormat = "jpeg",
    quality: int | None = None,
) -> list[str]:
    """ffmpeg arguments (without the executable) for one :func:`grab_images` pass."""
    if (width is None) == (height is None):
        raise ValueError("give exactly one of width / height")
    if fmt not in ("jpeg", "png"):
        raise ValueError(f"unsupported image format {fmt!r}")
    scale = f"scale={width}:-2" if width is not None else f"scale=-2:{height}"
    args = ["-hide_banner", "-v", "error"]
    if hw:
        args += ["-hwaccel", "cuda"]
    args += ["-ss", f"{start:.3f}", "-i", str(path)]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += ["-an", "-sn", "-vf", f"fps={fps},{scale}" if fps is not None else scale]
    args += ["-frames:v", str(count)]
    if fmt == "jpeg":
        args += ["-c:v", "mjpeg"]
        if quality is not None:
            args += ["-q:v", str(quality)]
    else:
        args += ["-c:v", "png"]
    return [*args, "-f", "image2pipe", "-"]


def grab_images(
    path: Path | str,
    start: float,
    *,
    count: int = 1,
    duration: float | None = None,
    fps: float | str | None = None,
    width: int | None = None,
    height: int | None = None,
    fmt: ImageFormat = "jpeg",
    quality: int | None = None,
    hwaccel: HwAccel = None,
    fallback_cpu: bool = True,
    timeout: float | None = None,
) -> list[bytes]:
    """Encoded stills from ``path`` (its staged copy if any), after one input seek.

    Args:
        path: Media file.
        start: Input seek (``-ss`` before ``-i``) in seconds (3 decimals).
        count: At most this many images (``-frames:v``).
        duration: Read only this many seconds after ``start`` (``-t``).
        fps: Rate of an ``fps`` filter placed before the scale, e.g. ``"1/0.5"``
            (one image every 0.5 s) or ``"6/12.000"``. ``None`` takes consecutive
            frames (use ``count=1`` for a single still).
        width: Scale to this width (the height keeps the aspect, rounded even).
        height: Scale to this height (the width keeps the aspect, rounded even).
            Give exactly one of ``width`` / ``height``.
        fmt: ``"jpeg"`` (mjpeg) or ``"png"`` (``count`` must be 1).
        quality: JPEG ``-q:v`` (2 = best .. 31); ``None`` = encoder default.
        hwaccel: ``"cuda"`` decodes on NVDEC first; ``None`` = CPU only.
        fallback_cpu: Retry on the CPU when the CUDA pass returns no image.
        timeout: Seconds per pass before ffmpeg is killed.

    Returns:
        The encoded images in time order (at least one; the exit code is not
        checked once ffmpeg produced an image).

    Raises:
        FFmpegError: If no image was produced or ffmpeg timed out.
        ValueError: For bad ``width`` / ``height`` / ``fmt`` / ``count`` combinations.
    """
    if fmt == "png" and count != 1:
        raise ValueError("png grabs support count=1 only")
    passes = _hw_passes(hwaccel, fallback_cpu)
    err = b""
    for hw in passes:
        args = grab_images_args(
            path,
            start,
            hw=hw,
            count=count,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            fmt=fmt,
            quality=quality,
        )
        proc = _exec("ffmpeg", args, timeout, path)
        if fmt == "jpeg":
            images = _split_jpegs(proc.stdout)
        else:
            images = [proc.stdout] if proc.stdout else []
        if images:
            return images
        err = proc.stderr
        logger.debug("image grab of %s (hw=%s) failed: %s", path, hw, _tail(err, 3))
    raise FFmpegError(f"No image decoded at {start:.2f}s of {path}:\n{_tail(err)}")


# ----------------------------------------------------------------- audio
def decode_pcm_args(
    path: Path | str,
    *,
    sr: int,
    start: float | None = None,
    duration: float | None = None,
    mono: bool = True,
    audio_streams: int | None = None,
) -> list[str]:
    """ffmpeg arguments (without the executable) for :func:`decode_pcm`."""
    args = ["-hide_banner", "-v", "error"]
    if start is not None:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(path)]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args.append("-vn")
    if audio_streams is not None:
        mix = audio_mix_filter(audio_streams)
        args += ["-filter_complex", f"{mix}[a]", "-map", "[a]"] if mix else ["-map", "0:a:0"]
    if mono:
        args += ["-ac", "1"]
    return [*args, "-ar", str(sr), "-f", "s16le", "-"]


def decode_pcm(
    path: Path | str,
    *,
    sr: int,
    start: float | None = None,
    duration: float | None = None,
    mono: bool = True,
    audio_streams: int | None = None,
    timeout: float | None = None,
) -> np.ndarray:
    """Decode the audio of ``path`` (its staged copy if any) to 16-bit PCM.

    Args:
        path: Media file.
        sr: Output sample rate (Hz).
        start: Input seek (``-ss`` before ``-i``) in seconds; ``None`` = no seek.
        duration: Decode this many seconds (``-t``); ``None`` = to the end.
        mono: Downmix to one channel (``-ac 1``). Otherwise the samples of all
            channels stay interleaved in the returned 1-D array.
        audio_streams: Number of audio streams (:attr:`MediaInfo.audio_streams`):
            2 or more are mixed (:func:`audio_mix_filter`), fewer map ``0:a:0``.
            ``None`` leaves the choice to ffmpeg's default stream selection.
        timeout: Seconds before ffmpeg is killed.

    Returns:
        Read-only ``int16`` samples (1-D); divide by 32768 for -1..1 floats.

    Raises:
        FFmpegError: If ffmpeg exits non-zero or times out.
    """
    args = decode_pcm_args(
        path, sr=sr, start=start, duration=duration, mono=mono, audio_streams=audio_streams
    )
    proc = _exec("ffmpeg", args, timeout, path)
    if proc.returncode != 0:
        raise FFmpegError(
            f"ffmpeg audio decode of {path} failed (exit {proc.returncode}):\n{_tail(proc.stderr)}"
        )
    data = proc.stdout
    return np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2")
