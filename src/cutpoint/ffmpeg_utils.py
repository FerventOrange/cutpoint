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
``CUTPOINT_FFPROBE`` environment variables, ``PATH``, and on Windows the WinGet package folder
(so a fresh ``winget install Gyan.FFmpeg`` works before the shell's PATH is refreshed).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from fractions import Fraction
from functools import cache, partial
from pathlib import Path
from typing import Any, Literal

import numpy as np

from cutpoint import paths, state
from cutpoint.cache import REPLACE_ATTEMPTS, load_json, save_json
from cutpoint.config import get_config, path_key

logger = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    """Raised when ffmpeg/ffprobe is missing or a command fails."""


def exit_code(rc: int) -> str:
    """``rc`` as error messages show it.

    Windows reports crash codes (NTSTATUS) unsigned, e.g. 3221225477; these are shown
    signed with their hex form, ``-1073741819 (0xC0000005)``. Ordinary codes and POSIX
    signals (``-9``) are shown as they are.
    """
    if -256 < rc < 2**31:
        return str(rc)
    u = rc & 0xFFFFFFFF
    return f"{u - 2**32 if u >= 2**31 else u} (0x{u:08X})"


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
    if sys.platform != "win32":
        raise FFmpegError(
            f"{name} not found. Install ffmpeg with your package manager (e.g. "
            f"sudo apt install ffmpeg) or set CUTPOINT_{name.upper()}=/path/to/{name}."
        )
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
        raise FFmpegError(f"{args[0]} failed (exit {exit_code(proc.returncode)}):\n{tail}")
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
            code = exit_code(proc.returncode)
            raise FFmpegError(f"ffmpeg failed during {label} (exit {code}):\n{tail}")
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


class EncoderUnavailableError(FFmpegError):
    """A video encoder the user asked for cannot encode on this machine."""


VIDEO_CODECS = {"nvenc": "h264_nvenc", "amf": "h264_amf", "vaapi": "h264_vaapi", "cpu": "libx264"}


@dataclass(frozen=True)
class VideoEncoder:
    """One way to encode H.264: ``kind`` is nvenc, amf, vaapi or cpu (libx264).

    ``device`` is the VAAPI render node. Callers add :attr:`upload` to their last video
    filter (:meth:`chain`) and put :meth:`args` where they used to put the encoder options;
    those include the pixel format, so callers must not add their own ``-pix_fmt``.
    """

    kind: str
    device: str | None = None

    @property
    def codec(self) -> str:
        return VIDEO_CODECS[self.kind]

    @property
    def upload(self) -> str:
        """Filters that move frames into GPU memory (VAAPI only; "" for the others)."""
        return "format=nv12,hwupload" if self.kind == "vaapi" else ""

    def chain(self, filters: str) -> str:
        """``filters`` followed by :attr:`upload`, so a ``-vf`` / filter_complex ends GPU-ready."""
        return f"{filters},{self.upload}" if self.upload else filters

    def args(self, quality: int = 20) -> list[str]:
        """Encoder options; ``quality`` is CRF (libx264), CQ (NVENC) or QP (AMF, VAAPI)."""
        q = str(quality)
        codec = ["-c:v", self.codec]
        yuv = ["-profile:v", "high", "-pix_fmt", "yuv420p"]
        if self.kind == "nvenc":
            return [*codec, "-preset", "p5", "-rc", "vbr", "-cq", q, "-b:v", "0", *yuv]
        if self.kind == "amf":
            qp = ["-qp_i", q, "-qp_p", q, "-qp_b", q]
            return [*codec, "-quality", "quality", "-rc", "cqp", *qp, *yuv]
        if self.kind == "vaapi":  # frames are VAAPI surfaces: a -pix_fmt would break the graph
            # A named device for hwupload, so a CUDA device opened for decoding is never used.
            bind = ["-init_hw_device", f"vaapi=cp:{self.device}", "-filter_hw_device", "cp"]
            return [*bind, *codec, "-profile:v", "high", "-qp", q]
        return [*codec, "-preset", "medium", "-crf", q, "-pix_fmt", "yuv420p"]


# ffmpeg's closing lines after the real error: they name no cause, so a probe skips them.
_TRAILER_RE = re.compile(
    r"Nothing was written into output file|Terminating thread|Task finished|Conversion failed!"
)


def _first_error_line(err: FFmpegError) -> str:
    """The first line of ``err``'s ffmpeg output that is not a trailer, else its last line."""
    lines = str(err).splitlines()
    for line in lines[1:]:  # lines[0] is run()'s "ffmpeg failed (exit N):" header
        if line.strip() and not _TRAILER_RE.search(line):
            return line.strip()
    return lines[-1]


def vaapi_devices() -> list[str]:
    """VAAPI render nodes (Linux), in a stable order; each is probed, none is assumed."""
    if not sys.platform.startswith("linux"):
        return []
    return sorted(str(p) for p in Path("/dev/dri").glob("renderD*"))


@state.register_cache
@cache
def encoder_error(enc: VideoEncoder) -> str | None:
    """None if ffmpeg really encodes with ``enc`` here, else the first real line of its error."""
    try:
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=black:s=640x360:d=0.2",
                "-vf",
                enc.chain("format=yuv420p"),
                *enc.args(),
                "-f",
                "null",
                "-",
            ],
            timeout=60,
        )
    except FFmpegError as err:
        return _first_error_line(err)
    return None


def find_encoder(kind: str) -> tuple[VideoEncoder | None, list[str]]:
    """The first working encoder of ``kind`` (VAAPI: first working render node) and the errors."""
    if kind == "vaapi" and not sys.platform.startswith("linux"):
        return None, ["VAAPI is Linux-only"]
    if kind == "vaapi":
        cands = [VideoEncoder("vaapi", d) for d in vaapi_devices()]
        errors = [] if cands else ["no /dev/dri/renderD* render device"]
    else:
        cands, errors = [VideoEncoder(kind)], []
    for enc in cands:
        error = encoder_error(enc)
        if error is None:
            return enc, []
        errors.append(f"{enc.device or enc.codec}: {error}")
    return None, errors


@state.register_cache
@cache
def video_encoder() -> VideoEncoder:
    """The encoder to use: the configured one (``video_encoder``), else the first that works.

    Auto tries NVENC, then AMF (Windows) or VAAPI and AMF (Linux), by actually encoding a short
    clip, and falls back to libx264. A named encoder that does not work is an error.

    Raises:
        EncoderUnavailableError: A GPU encoder was named but cannot encode here.
    """
    choice = get_config().video_encoder
    if choice == "cpu":
        return VideoEncoder("cpu")
    if choice != "auto":
        enc, errors = find_encoder(choice)
        if enc is None:
            raise EncoderUnavailableError(
                f"video_encoder {choice!r} cannot encode on this machine ({'; '.join(errors)}). "
                "Set CUTPOINT_VIDEO_ENCODER / video_encoder in config.json to 'auto' or 'cpu', "
                "or run `cutpoint capabilities`."
            )
        return enc
    for kind in ("nvenc", "amf") if sys.platform == "win32" else ("nvenc", "vaapi", "amf"):
        enc, _ = find_encoder(kind)
        if enc is not None:
            logger.info("Video encoder: %s%s", enc.codec, f" on {enc.device}" if enc.device else "")
            return enc
    logger.info("No GPU video encoder works; encoding with libx264")
    return VideoEncoder("cpu")


# Codecs of auto-chosen GPU encoders that failed on real footage while libx264 then worked:
# the rest of the run encodes on the CPU instead of failing and retrying every clip.
_GPU_ENCODE_FAILED: set[str] = state.register_container(__name__, "_GPU_ENCODE_FAILED", set())


def encode_with_fallback[T](encode: Callable[[VideoEncoder], T], what: str) -> T:
    """Run ``encode`` with :func:`video_encoder`; retry once on the CPU if a GPU encode fails.

    The probe in :func:`video_encoder` encodes 0.2 s of black, so a driver can still fail
    on real footage. With ``video_encoder`` "auto" a failed GPU encode is retried with
    libx264 (one WARNING); if that works, the GPU encoder is skipped for the rest of the
    process. ffmpeg's error does not reliably say whether the encoder or something else
    (a bad read, a filter) failed, so any failure is retried; a failure that is not the
    encoder's fails again on the CPU and that error is raised. A CPU encode is not retried.

    Raises:
        FFmpegError: If the CPU encode fails, or a GPU encoder named in ``video_encoder``
            fails (the message names the setting that switches to the CPU).
    """
    enc = VideoEncoder("cpu") if _GPU_ENCODE_FAILED else video_encoder()
    try:
        return encode(enc)
    except FFmpegError as err:
        if enc.kind == "cpu":
            raise
        choice = get_config().video_encoder
        if choice != "auto":
            raise FFmpegError(
                f"{err}\n{what}: {enc.codec} (video_encoder {choice!r}) failed. Set "
                'CUTPOINT_VIDEO_ENCODER=cpu or "video_encoder": "cpu" in config.json to '
                "encode on the CPU."
            ) from err
        logger.warning(
            "%s: %s failed (%s); retrying on the CPU (libx264)",
            what,
            enc.codec,
            _first_error_line(err),
        )
    result = encode(VideoEncoder("cpu"))
    if enc.codec not in _GPU_ENCODE_FAILED:
        logger.info("Encoding the rest of this run with libx264 instead of %s", enc.codec)
        _GPU_ENCODE_FAILED.add(enc.codec)
    return result


# Windows "Access is denied" and "being used by another process": the file is open elsewhere.
_LOCKED_WINERRORS = (5, 32)


def retry_locked[T](op: Callable[[], T], path: Path, action: str) -> T:
    """Run ``op`` (replacing or deleting ``path``), retrying briefly on PermissionError.

    On Windows another process (a player, an editor, a virus scanner) holding ``path``
    open makes a rename or delete fail for a moment.

    Raises:
        PermissionError: Naming ``path`` and the program holding it, if it is still locked
            (Windows error 5 or 32) after the retries; any other error as raised by ``op``.
    """
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            return op()
        except PermissionError as err:
            if attempt < REPLACE_ATTEMPTS - 1:
                time.sleep(min(0.05 * 2**attempt, 0.5))
                continue
            if getattr(err, "winerror", None) not in _LOCKED_WINERRORS:
                raise
            raise PermissionError(
                f"Cannot {action} {path}: it is open in another program (a player, an editor "
                f"or a virus scanner). Close it and re-run. ({err.strerror})"
            ) from err
    raise AssertionError("unreachable")


@contextmanager
def output_dir(path: Path) -> Iterator[Path]:
    """Create the folder ``path``; on failure (Ctrl+C included) remove it again if this call
    created it and it is still empty, so a failed run leaves no empty folder behind.

    Missing parents are created too and removed the same way. A folder that existed before,
    or that holds anything, is always kept.
    """
    created = [p for p in (path, *path.parents) if not p.exists()]
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    except BaseException:
        for p in created:  # innermost first; a non-empty folder stops the walk
            try:
                p.rmdir()
            except OSError:
                break
        raise


def part_path(out: Path) -> Path:
    """Temporary name :func:`atomic_output` writes ``out`` to: ``.<stem>.part<suffix>``.

    The suffix is kept so ffmpeg picks the muxer; the leading dot keeps it out of the
    clip listings (``assemble``, ``extend``).
    """
    return out.with_name(f".{out.stem}.part{out.suffix}")


@contextmanager
def atomic_output(out: Path) -> Iterator[Path]:
    """Yield :func:`part_path` of ``out`` to write to, renamed onto ``out`` on success.

    On any failure (Ctrl+C included) the partial file is removed and ``out`` is left as
    it was, so a truncated file never appears under the final name. On Windows the rename
    is retried briefly while another process (a virus scanner, a player) holds a file open.
    """
    part = part_path(out)
    try:
        yield part
        retry_locked(lambda: part.replace(out), out, "replace")
    except BaseException:
        # A still-locked part (Windows, right after Ctrl+C) must not replace the real error.
        with suppress(OSError):
            part.unlink(missing_ok=True)
        raise


# --- all-or-nothing folder replace (``clips --force``, ``extend --out <clips_dir> --force``)
#
# A run exports into the hidden staging folder ``.<name>.cutpoint-tmp`` next to the output,
# then :func:`swap_in` moves the originals it replaces into ``.<name>.cutpoint-bak`` and the
# new files in. The backup holds a journal (the old and new names) until every new file is
# in; on any failure the originals are put back. A hard kill leaves the journal, so the next
# run's :func:`recover_swap` finishes the job: journal present, put the originals back;
# backup without journal, the new set was complete, drop the backup.

_SWAP_JOURNAL = ".cutpoint-swap.json"


def staging_path(dst: Path) -> Path:
    """Hidden staging folder of ``dst``: ``.<name>.cutpoint-tmp`` next to it."""
    dst = dst.resolve()  # "." and ".." have no usable name; a symlink's target is the folder
    return dst.with_name(f".{dst.name}.cutpoint-tmp")


def backup_path(dst: Path) -> Path:
    """Hidden folder :func:`swap_in` moves the replaced originals of ``dst`` to."""
    dst = dst.resolve()
    return dst.with_name(f".{dst.name}.cutpoint-bak")


def _restore(dst: Path, bak: Path, journal: dict[str, Any]) -> None:
    """Put the originals in ``bak`` back into ``dst`` and drop the new-only files there.

    Both steps are best effort: one locked file does not stop the others. The journal is
    removed last and only if every file went back, so a restore that fails half-way is
    redone by :func:`recover_swap`. A ``dst`` the swap created is removed when empty.

    Raises:
        OSError: The first error, after every file was tried.
    """
    errors: list[OSError] = []
    old = set(journal["old"])
    for name in journal["old"]:
        src = bak / name
        if src.exists() or src.is_symlink():  # else it never left dst
            try:
                retry_locked(partial(src.replace, dst / name), dst / name, "restore")
            except OSError as err:
                errors.append(err)
    for name in journal["new"]:
        if name not in old:  # never a user file: swap_in records every existing name as old
            f = dst / name
            try:
                retry_locked(partial(f.unlink, missing_ok=True), f, "delete")
            except OSError as err:
                errors.append(err)
    if errors:
        raise errors[0]
    (bak / _SWAP_JOURNAL).unlink()
    bak.rmdir()
    if journal.get("created"):
        with suppress(OSError):  # not empty: something else was put there meanwhile
            dst.rmdir()


def recover_swap(dst: Path) -> None:
    """Clean up after a :func:`swap_in` or staged export killed hard (``taskkill /F``).

    With a journal the originals are put back; a backup without one is from a completed
    swap and is removed. A stale staging folder (only this tool's output) is removed.

    Raises:
        RuntimeError: If the journal is unreadable or malformed; the backup is left as is.
        PermissionError: If a leftover backup or staging folder cannot be removed (a file
            in it is still open); the next swap could not start with it in place.
    """
    bak = backup_path(dst)
    if bak.exists():
        path = bak / _SWAP_JOURNAL
        if not (path.exists() or path.is_symlink()):
            # Nothing in it is needed, but it must go: the next swap_in creates it anew.
            _remove_leftover(bak, f"left over from a finished replace of {dst}")
        else:
            journal = load_json(path)
            if not (
                isinstance(journal, dict)
                and all(
                    isinstance(journal.get(k), list) and all(isinstance(n, str) for n in journal[k])
                    for k in ("old", "new")
                )
            ):
                raise RuntimeError(
                    f"An earlier replace of {dst} was interrupted and its record {path} cannot "
                    f"be read. The original files are in {bak}: move them back into {dst} by "
                    f"hand, then delete {bak} and re-run."
                )
            _restore(dst, bak, journal)
            logger.warning("Put back the original files of %s from an interrupted replace", dst)
    tmp = staging_path(dst)
    if tmp.exists():
        _remove_leftover(tmp, f"left over from an interrupted export into {dst}")


def _remove_leftover(path: Path, why: str) -> None:
    """Remove the folder ``path`` (retried briefly while a file in it is locked).

    Raises:
        PermissionError: Saying ``why`` it is there and how to fix it, if it stays locked
            (on Windows: a file in it is open in a player or a virus scanner, or a scanner
            briefly keeps it non-empty, error 145).
        OSError: Saying ``why`` it is there, for any other failure.
    """
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            shutil.rmtree(path)
            return
        except OSError as err:
            locked = isinstance(err, PermissionError) or getattr(err, "winerror", None) == 145
            if not locked:
                raise OSError(
                    f"Cannot remove {path}, {why}: {err.strerror or err}. Delete the folder "
                    f"and re-run."
                ) from err
            if attempt < REPLACE_ATTEMPTS - 1:
                time.sleep(min(0.05 * 2**attempt, 0.5))
                continue
            raise PermissionError(
                f"Cannot remove {path}, {why}: it is open in another program (a player, an "
                f"editor or a virus scanner). Close it, or delete the folder, and re-run. "
                f"({err.strerror or err})"
            ) from err


@contextmanager
def staged_dir(dst: Path) -> Iterator[Path]:
    """Yield a fresh :func:`staging_path` of ``dst`` to export into, removed on failure.

    The caller moves the result in with :func:`swap_in`. Call :func:`recover_swap` first
    when ``dst`` is checked for existing files before this.
    """
    recover_swap(dst)
    tmp = staging_path(dst)
    tmp.mkdir()
    try:
        yield tmp
    except BaseException:
        # A still-locked part (Windows, right after Ctrl+C) must not replace the real error.
        try:
            shutil.rmtree(tmp)
        except OSError as err:
            logger.warning("Could not remove %s (%s); the next run removes it", tmp, err)
        raise


def file_keys(path: Path) -> set[Any]:
    """Identities of the file ``path`` for :func:`same_file`: its resolved path (case-folded
    on Windows) and, where the file system has them, its device and inode numbers, so a
    symlink, a hard link or another spelling of the same file matches. Empty if missing."""
    keys: set[Any] = set()
    try:
        st = path.stat()
    except OSError:
        return keys
    keys.add(path_key(str(path.resolve())))
    if st.st_ino:  # 0: no usable inode numbers (some network shares)
        keys.add((st.st_dev, st.st_ino))
    return keys


def same_file(path: Path, keys: set[Any]) -> bool:
    """True if ``path`` is one of the files whose :func:`file_keys` are in ``keys``."""
    return bool(file_keys(path) & keys)


def check_not_sources(dst: Path, names: Iterable[str], sources: set[Any]) -> None:
    """Fail before any export if writing ``names`` into ``dst`` would replace a source.

    ``sources`` holds the :func:`file_keys` of every file the run reads.

    Raises:
        ValueError: Naming the first such file.
    """
    for n in names:
        if same_file(dst / n, sources):
            raise ValueError(
                f"Cannot write {dst / n}: this run reads it as a source. Move the source out "
                f"of {dst}, or export into another folder (--out)."
            )


def swap_in(new: Path, dst: Path, old: Iterable[str] = ()) -> None:
    """Move every file of ``new`` into ``dst``, all or nothing, and remove ``new``.

    The files ``old`` (names in ``dst``) and those the new files replace are moved to
    :func:`backup_path` first and deleted only after every new file is in. On any failure
    (a locked file, Ctrl+C) the originals are put back and the error is re-raised. If even
    that fails, the backup is kept and the next run puts them back (:func:`recover_swap`).
    A missing ``dst`` is created, and removed again on failure.

    Raises:
        PermissionError: Naming the file, if one stays locked (see :func:`retry_locked`).
    """
    created = not dst.exists()
    # manifest.json last in, first out: it never lists a clip that is not there
    names = sorted((p.name for p in new.iterdir()), key=lambda n: (n == "manifest.json", n))
    olds = (
        []
        if created
        else sorted(
            {n for n in [*old, *names] if (dst / n).exists() or (dst / n).is_symlink()},
            key=lambda n: (n != "manifest.json", n),
        )
    )
    journal = {"old": olds, "new": names, "created": created}
    bak = backup_path(dst)
    bak.mkdir()
    try:
        save_json(bak / _SWAP_JOURNAL, journal)
        if created:
            dst.mkdir()
        for n in olds:
            retry_locked(partial((dst / n).replace, bak / n), dst / n, "replace")
        for n in names:
            retry_locked(partial((new / n).replace, dst / n), dst / n, "replace")
    except BaseException:
        try:
            if (bak / _SWAP_JOURNAL).exists():
                _restore(dst, bak, journal)
            else:
                bak.rmdir()
        except BaseException as err:  # a second Ctrl+C too: say where the originals are
            logger.error(
                "Could not put back every original file of %s (%s): the rest are in %s. "
                "The next cutpoint run on this folder puts them back.",
                dst,
                err or type(err).__name__,
                bak,
            )
            if not isinstance(err, Exception):
                raise
        raise
    # The new set is complete: the backup is no longer needed.
    journal_path = bak / _SWAP_JOURNAL
    retry_locked(journal_path.unlink, journal_path, "delete")
    try:
        shutil.rmtree(bak)
    except OSError as err:  # still open (a virus scanner): recover_swap removes it later
        logger.warning("Could not remove %s (%s); the next run removes it", bak, err)
    with suppress(OSError):  # still open (a virus scanner): recover_swap removes it later
        new.rmdir()


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
            f"ffprobe failed on {path} (exit {exit_code(proc.returncode)}):\n{_tail(proc.stderr)}"
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
        code = exit_code(proc.returncode)
        raise FFmpegError(
            f"ffmpeg audio decode of {path} failed (exit {code}):\n{_tail(proc.stderr)}"
        )
    data = proc.stdout
    return np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2")
