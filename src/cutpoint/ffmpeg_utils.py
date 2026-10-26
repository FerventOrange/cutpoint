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

import errno
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
from typing import IO, Any, Literal

import numpy as np

from cutpoint import paths, state
from cutpoint.cache import REPLACE_ATTEMPTS, created_dirs, load_json, make_dirs, save_json
from cutpoint.config import get_config, path_key

logger = logging.getLogger(__name__)

SOFT_STOP_S = 5.0  # a stopped child (ffmpeg; a UI job after CTRL_BREAK) gets this long
PROGRESS_DRAIN_S = 5.0  # run_progress reads the rest of the progress pipe this long after exit


class FFmpegError(RuntimeError):
    """Raised when ffmpeg/ffprobe is missing or a command fails."""


class FFmpegNotFoundError(FFmpegError):
    """Raised when the ffmpeg or ffprobe executable cannot be found."""


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
        FFmpegNotFoundError: If the executable cannot be found anywhere.
    """
    env = paths.env(name.upper())
    if env and Path(env).is_file():
        return env
    on_path = shutil.which(name)
    if on_path:
        return on_path
    if sys.platform != "win32":
        raise FFmpegNotFoundError(
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
    raise FFmpegNotFoundError(
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


def _stop_child(proc: subprocess.Popen, err: BaseException) -> None:
    """Let ``proc`` exit before the caller removes its partial output (#199).

    On Windows a running ffmpeg keeps its output open, so removing a ``.part`` or
    ``.tmp.wav`` file right after Ctrl+C fails. After a KeyboardInterrupt (Ctrl+C,
    Ctrl+Break, the UI's cancel) the child got the same event, so it is given
    ``SOFT_STOP_S`` to exit; after any other error, or when it is still running then, it is
    killed and waited for (again at most ``SOFT_STOP_S``). The caller re-raises ``err``.
    """
    try:
        if isinstance(err, KeyboardInterrupt):
            with suppress(subprocess.TimeoutExpired):
                proc.wait(SOFT_STOP_S)
    finally:  # a second Ctrl+C during the wait still kills it
        if proc.poll() is None:
            proc.kill()
            with suppress(subprocess.TimeoutExpired):
                proc.wait(SOFT_STOP_S)


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
    cmd = _command(args[0], args[1:])
    logger.debug("exec: %s", subprocess.list2cmdline(cmd))
    pipe = subprocess.PIPE if capture else None
    proc = subprocess.Popen(
        cmd, stdout=pipe, stderr=pipe, text=True, encoding="utf-8", errors="replace", cwd=cwd
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as err:
        proc.kill()
        proc.communicate()
        raise FFmpegError(f"{args[0]} timed out after {timeout:g}s") from err
    except BaseException as err:
        _stop_child(proc, err)
        raise
    out = (stdout or "") + (stderr or "")
    if proc.returncode != 0:
        raise FFmpegError(f"{args[0]} failed (exit {exit_code(proc.returncode)}):\n{_tail(out)}")
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
    cmd = _command(args[0], ["-progress", "pipe:1", "-nostats", *args[1:]])
    t0, next_pct, out_time = time.perf_counter(), 10, 0.0

    def read_progress(stream: IO[str]) -> None:
        nonlocal next_pct, out_time
        try:
            for line in stream:
                if not line.startswith("out_time_us="):
                    continue
                try:
                    out_time = max(out_time, int(line.split("=", 1)[1]) / 1e6)
                except ValueError:
                    continue
                pct = out_time / total * 100 if total else 0
                if total >= 600 and pct >= next_pct:
                    elapsed = time.perf_counter() - t0
                    logger.info("  %s: %d%% (%.0fs elapsed)", label, int(pct), elapsed)
                    next_pct = (int(pct) // 10 + 1) * 10
        finally:  # here, not in the caller: closing a pipe another thread reads blocks too
            with suppress(OSError):  # unread progress must not block ffmpeg's exit
                stream.close()

    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8", errors="replace"
        )
        assert proc.stdout is not None
        # The pipe reaches EOF only when every copy of its write end is closed, which a process
        # other than ffmpeg can delay forever (#338); so the exit of ffmpeg ends the run.
        reader = threading.Thread(
            target=read_progress, args=(proc.stdout,), name="ffmpeg-progress", daemon=True
        )
        reader.start()
        try:
            proc.wait()
            reader.join(PROGRESS_DRAIN_S)
        except BaseException as exc:
            _stop_child(proc, exc)
            raise
        if reader.is_alive():
            logger.debug("%s: progress pipe still open after ffmpeg exited; not waiting", label)
        err.seek(0)
        stderr = err.read().decode("utf-8", "replace")
        if proc.returncode != 0:
            code = exit_code(proc.returncode)
            raise FFmpegError(f"ffmpeg failed during {label} (exit {code}):\n{_tail(stderr)}")
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
    # End of the longest audio stream (s, from the container start); None without audio or
    # when a stream has no duration (MKV).
    audio_duration: float | None = None
    # ffprobe's color tags of the video stream (``color_transfer``, ``color_primaries``,
    # ``color_space``, ``color_range``); None when untagged or "unknown".
    color_transfer: str | None = None
    color_primaries: str | None = None
    color_space: str | None = None
    color_range: str | None = None

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
    # ``-ss`` seeks from the container start, so a stream starting late ends that much later
    f_start = float(data.get("format", {}).get("start_time") or 0)
    a_ends = [
        float(s.get("start_time") or f_start) - f_start + float(s["duration"])
        for s in audios
        if s.get("duration")
    ]
    audio_duration = max(a_ends) if audios and len(a_ends) == len(audios) else None
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
        audio_duration=audio_duration,
        color_transfer=_color_tag(video, "color_transfer"),
        color_primaries=_color_tag(video, "color_primaries"),
        color_space=_color_tag(video, "color_space"),
        color_range=_color_tag(video, "color_range"),
    )


def _color_tag(stream: dict[str, Any], key: str) -> str | None:
    """``stream[key]``, or None when it is missing, empty or "unknown"."""
    value = stream.get(key) or ""
    return None if value in ("", "unknown") else value


def audio_mix_filter(n_streams: int, input_idx: int = 0) -> str | None:
    """Return a filter mixing all audio tracks (e.g. game + separate mic track).

    Returns ``None`` when there is at most one audio stream.
    """
    if n_streams < 2:
        return None
    ins = "".join(f"[{input_idx}:a:{i}]" for i in range(n_streams))
    return f"{ins}amix=inputs={n_streams}:normalize=0:duration=longest"


def mix_head(tracks: Sequence[int], input_idx: int = 0) -> str:
    """Start of a filter chain reading ``tracks`` (audio track indices) of input ``input_idx``.

    One track gives its pad (``[0:a:0]``); several are summed (``amix``, no normalisation, so
    a game and a separate mic track both keep their level), with a trailing comma. Either way
    the next filter follows directly.
    """
    ins = "".join(f"[{input_idx}:a:{i}]" for i in tracks)
    return f"{ins}amix=inputs={len(tracks)}:normalize=0," if len(tracks) > 1 else ins


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
    try:
        res = run_progress(args, info.duration - start, f"audio extract {src.name}")
        tmp.replace(dst)
    except BaseException:
        with suppress(OSError):  # still locked (Windows, right after Ctrl+C): keep the real error
            tmp.unlink(missing_ok=True)
        raise
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
    args = [
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
    cmd = _command("ffmpeg", args)
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
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
    args = [
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
    cmd = _command("ffmpeg", args)
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
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


def clip_momentary(
    src: Path,
    streams: int,
    start: float | None = None,
    duration: float | None = None,
    chain: str = "",
) -> list[float]:
    """Momentary loudness (LUFS, 400 ms blocks every 100 ms) of a clip's audio.

    ``streams`` audio tracks are mixed as a clip export of the same span mixes them
    (duplicate tracks left out, see :func:`cutpoint.audio_tracks.export_tracks`); 1 reads
    track 0 only.
    ``start``/``duration`` (s) cut a source span (input seek, as the export). ``chain``
    (filters ending in a comma) is applied before the measurement. Returns ``[]`` for
    ``streams`` 0 or no blocks.
    """
    from cutpoint.audio_tracks import export_tracks  # noqa: PLC0415 - it imports this module

    if streams < 1:
        return []
    span = [*(["-ss", f"{start:.3f}"] if start is not None else [])]
    span += ["-t", f"{duration:.3f}"] if duration is not None else []
    clip = (start or 0.0, duration) if duration is not None else None
    head = mix_head(export_tracks(src, streams, span=clip))
    out = run(
        [
            "ffmpeg",
            "-nostats",
            "-hide_banner",
            *span,
            "-i",
            str(src),
            "-filter_complex",
            f"{head}{chain}ebur128=peak=none",
            "-vn",
            "-sn",
            "-dn",
            "-f",
            "null",
            "-",
        ]
    )
    return [-120.0 if m.group(2) == "-inf" else float(m.group(2)) for m in _EBUR_RE.finditer(out)]


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


def buildconf_has_soxr(buildconf: str) -> bool:
    """True if ``ffmpeg -buildconf`` output lists ``--enable-libsoxr``."""
    return re.search(r"(?m)^\s*--enable-libsoxr\s*$", buildconf) is not None


@state.register_cache
@cache
def has_soxr() -> bool:
    """True if this ffmpeg build has the soxr resampler (libsoxr); not every build does."""
    try:
        found = buildconf_has_soxr(run(["ffmpeg", "-hide_banner", "-buildconf"], timeout=30))
    except FFmpegError:
        found = False
    logger.debug("ffmpeg resampler: %s", "soxr" if found else "swr (no libsoxr in this build)")
    return found


def filter_names(listing: str) -> frozenset[str]:
    """Filter names in ``ffmpeg -filters`` output (lines like `` .S zscale  V->V  ...``)."""
    return frozenset(re.findall(r"(?m)^ [.A-Z|]{2,3} (\w+)\s+\S+->\S+", listing))


@state.register_cache
@cache
def ffmpeg_filters() -> frozenset[str]:
    """Names of this ffmpeg build's filters (empty if ``ffmpeg -filters`` fails)."""
    try:
        return filter_names(run(["ffmpeg", "-hide_banner", "-filters"], timeout=30))
    except FFmpegError:
        return frozenset()


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


# video_encoder() results by configured choice, so an edited ``video_encoder`` is followed (the
# probes stay cached in encoder_error).
_ENCODERS: dict[str, VideoEncoder] = state.register_container(__name__, "_ENCODERS", {})


def video_encoder() -> VideoEncoder:
    """The encoder to use: the configured one (``video_encoder``), else the first that works.

    Auto tries NVENC, then AMF (Windows) or VAAPI and AMF (Linux), by actually encoding a short
    clip, and falls back to libx264. A named encoder that does not work is an error. Cached per
    configured value; a failure is not cached.

    Raises:
        EncoderUnavailableError: A GPU encoder was named but cannot encode here.
    """
    choice = get_config().video_encoder
    enc = _ENCODERS.get(choice)
    if enc is None:
        enc = _ENCODERS[choice] = _pick_encoder(choice)
    return enc


def _pick_encoder(choice: str) -> VideoEncoder:
    """:func:`video_encoder` for ``choice`` without the cache."""
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
    process while ``video_encoder`` stays "auto". ffmpeg's error does not reliably say whether the encoder or something else
    (a bad read, a filter) failed, so any failure is retried; a failure that is not the
    encoder's fails again on the CPU and that error is raised. A CPU encode is not retried.

    Raises:
        FFmpegError: If the CPU encode fails, or a GPU encoder named in ``video_encoder``
            fails (the message names the setting that switches to the CPU).
    """
    choice = get_config().video_encoder
    enc = VideoEncoder("cpu") if _GPU_ENCODE_FAILED and choice == "auto" else video_encoder()
    try:
        return encode(enc)
    except FFmpegError as err:
        if enc.kind == "cpu":
            raise
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


# Windows "Access is denied" and "being used by another process": the file is open elsewhere
# (error 5 can also mean a read-only file or folder).
_LOCKED_WINERRORS = (5, 32)
_LOCK_HINT = "it is open in another program (a player, an editor or a virus scanner)"


def retry_locked[T](
    op: Callable[[], T], path: Path, action: str, locked: Callable[[OSError], bool] | None = None
) -> T:
    """Run ``op`` (replacing or deleting ``path``), retrying briefly on PermissionError.

    On Windows another process (a player, an editor, a virus scanner) holding ``path``
    open makes a rename or delete fail for a moment. ``locked`` picks the errors to retry
    instead; the last one is then re-raised as is, for the caller to word.

    Raises:
        PermissionError: Naming ``path`` and the program holding it, if it is still locked
            (Windows error 5 or 32) after the retries (for error 5 also a read-only file or
            folder); any other error as raised by ``op``.
    """
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            return op()
        except OSError as err:
            if not (locked(err) if locked else isinstance(err, PermissionError)):
                raise
            if attempt < REPLACE_ATTEMPTS - 1:
                time.sleep(min(0.05 * 2**attempt, 0.5))
                continue
            if locked or getattr(err, "winerror", None) not in _LOCKED_WINERRORS:
                raise
            read_only = (
                " If no program has it open, the file or its folder may be read-only: clear the"
                " read-only attribute and re-run."
                if getattr(err, "winerror", None) == 5
                else ""
            )
            raise PermissionError(
                f"Cannot {action} {path}: {_LOCK_HINT}. Close it and re-run.{read_only} "
                f"({err.strerror})"
            ) from err
    raise AssertionError("unreachable")


@contextmanager
def output_dir(path: Path) -> Iterator[Path]:
    """Create the folder ``path``; on failure (Ctrl+C included) remove it again if this call
    created it and it is still empty, so a failed run leaves no empty folder behind.

    Missing parents are created too and removed the same way. A folder that existed before,
    or that holds anything, is always kept (:func:`cutpoint.cache.created_dirs`).
    """
    with created_dirs() as made:
        make_dirs(path, made)
        yield path


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


# --- all-or-nothing folder replace (``clips --force``, ``extend --out <clips_dir> --force``,
# ``extend --out DIR --force`` into a non-empty DIR)
#
# A run exports into the hidden staging folder ``.<name>.cutpoint-tmp`` next to the output,
# then :func:`swap_in` moves the originals it replaces into ``.<name>.cutpoint-bak`` and the
# new files in. The backup holds a journal (the old and new names) until every new file is
# in; on any failure the originals are put back. A hard kill leaves the journal, so the next
# run's :func:`recover_swap` finishes the job: journal present, put the originals back;
# backup without journal, the new set was complete, drop the backup.
# Both folders sit on the parent's file system, so :func:`staged_dir` refuses a ``dst`` that
# is a drive root (no parent to hold them), a mount point or a bind mount before any export,
# unless an empty one is exported into directly (``direct_if_empty``).

_SWAP_JOURNAL = ".cutpoint-swap.json"
_SWAP_PROBE = ".cutpoint-swap-probe"
_SWAP_FIX = "use --out with a folder on that drive"


def _root_error(dst: Path) -> ValueError:
    return ValueError(f"{dst} is a drive root; {_SWAP_FIX}")


def staging_path(dst: Path) -> Path:
    """Hidden staging folder of ``dst``: ``.<name>.cutpoint-tmp`` next to it.

    Raises:
        ValueError: If ``dst`` is a drive root.
    """
    res = dst.resolve()  # "." and ".." have no usable name; a symlink's target is the folder
    if not res.name:
        raise _root_error(dst)
    return res.with_name(f".{res.name}.cutpoint-tmp")


def backup_path(dst: Path) -> Path:
    """Hidden folder :func:`swap_in` moves the replaced originals of ``dst`` to.

    Raises:
        ValueError: If ``dst`` is a drive root.
    """
    res = dst.resolve()
    if not res.name:
        raise _root_error(dst)
    return res.with_name(f".{res.name}.cutpoint-bak")


def check_swap_target(dst: Path) -> None:
    """Fail before any export if ``dst`` cannot be replaced through :func:`staged_dir`.

    Raises:
        ValueError: If ``dst`` is a drive root or a mount point, saying what to do.
    """
    res = dst.resolve()
    if not res.name:
        # A volume mounted into a folder (C:\mnt\vol) resolves to the volume's own root on
        # Windows, so only a folder that is a root before resolving is a drive root.
        own = Path(os.path.abspath(dst))  # noqa: PTH100 - normalises "..", no resolving
        if not own.name or not own.is_mount():
            raise _root_error(dst)
    if not res.name or res.is_mount():
        raise ValueError(
            f"{dst} is a mount point, so the clips cannot be swapped in safely; {_SWAP_FIX}"
        )


def _check_rename(tmp: Path, dst: Path) -> None:
    """Fail if a file cannot be renamed from ``tmp`` into ``dst``.

    Catches what :func:`check_swap_target` cannot see: a Linux bind mount of a folder on the
    same device, where ``is_mount()`` is False but a rename across it fails with EXDEV.
    Any other error is left to :func:`swap_in`, which retries and names the file.

    Raises:
        ValueError: If ``dst`` is on another file system than ``tmp``.
    """
    probe = tmp / _SWAP_PROBE
    try:
        probe.touch()
        probe.rename(dst / _SWAP_PROBE)  # a leftover there is removed by recover_swap
    except OSError as err:
        with suppress(OSError):  # else swap_in would move it into dst
            probe.unlink(missing_ok=True)
        if err.errno == errno.EXDEV:
            raise ValueError(
                f"{dst} is on another file system than its parent folder (a mount point or "
                f"bind mount), so the clips cannot be swapped in safely; {_SWAP_FIX}"
            ) from err
        return
    with suppress(OSError):
        (dst / _SWAP_PROBE).unlink()


def _restore(dst: Path, bak: Path, journal: dict[str, Any]) -> None:
    """Put the originals in ``bak`` back into ``dst`` and drop the new-only files there.

    Both steps are best effort: one locked file does not stop the others. The journal is
    removed last and only if every file went back, so a restore that fails half-way is
    redone by :func:`recover_swap`. A ``dst`` the swap created is removed when empty.

    Raises:
        OSError: The first error, after every file was tried.
    """
    errors: list[OSError] = []
    old = {path_key(n) for n in journal["old"]}
    for name in journal["old"]:
        src = bak / name
        if src.exists() or src.is_symlink():  # else it never left dst
            try:
                retry_locked(partial(src.replace, dst / name), dst / name, "restore")
            except OSError as err:
                errors.append(err)
    for name in journal["new"]:
        # never a user file: swap_in records every existing name as old (once per path_key,
        # so a case variant of a new name is the same file and is not deleted)
        if path_key(name) not in old:
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
    if not dst.resolve().name:  # a drive root: nothing is ever staged next to it
        return
    with suppress(OSError):  # a probe of _check_rename left by a hard kill
        (dst / _SWAP_PROBE).unlink(missing_ok=True)
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

    def locked(err: OSError) -> bool:
        return isinstance(err, PermissionError) or getattr(err, "winerror", None) == 145

    try:
        retry_locked(partial(shutil.rmtree, path), path, "remove", locked)
    except OSError as err:
        if locked(err):
            raise PermissionError(
                f"Cannot remove {path}, {why}: {_LOCK_HINT}. Close it, or delete the folder, "
                f"and re-run. ({err.strerror or err})"
            ) from err
        raise OSError(
            f"Cannot remove {path}, {why}: {err.strerror or err}. Delete the folder and re-run."
        ) from err


@contextmanager
def staged_dir(dst: Path, *, direct_if_empty: bool = False) -> Iterator[Path]:
    """Yield a fresh :func:`staging_path` of ``dst`` to export into, removed on failure.

    The caller moves the result in with :func:`swap_in`, unless ``dst`` itself was yielded:
    with ``direct_if_empty``, an existing empty ``dst`` that cannot be staged (a drive root,
    a mount point or a bind mount) is exported into directly and kept on failure. Call
    :func:`recover_swap` first when ``dst`` is checked for existing files before this.

    Raises:
        ValueError: If ``dst`` is a drive root, a mount point or a bind mount that is not
            exported into directly, before the caller exports anything (see
            :func:`check_swap_target`).
    """
    direct = direct_if_empty and dst.exists() and not any(dst.iterdir())
    try:
        check_swap_target(dst)
        stageable = True
    except ValueError:
        if not direct:
            raise
        stageable = False  # a root or mount point: no staging folder next to it
    if not stageable:
        yield dst
        return
    recover_swap(dst)
    tmp = staging_path(dst)
    tmp.mkdir()
    work = tmp
    try:
        if dst.exists():
            try:
                _check_rename(tmp, dst)
            except ValueError:
                if not direct:
                    raise
                work = dst
                with suppress(OSError):  # still open (a virus scanner): recover_swap removes it
                    shutil.rmtree(tmp)
        yield work
    except BaseException:
        # A still-locked part (Windows, right after Ctrl+C) must not replace the real error.
        try:
            if work is tmp:  # a direct export keeps its finished clips
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


def source_keys(reads: Iterable[tuple[str, Path, Iterable[Path]]]) -> set[Any]:
    """:func:`file_keys` of every file a run reads, given per source ``(source as recorded,
    path read (source_remap), files of a 360 source (PanoExporter.inputs))``."""
    keys: set[Any] = set()
    for s, p, pano_files in reads:
        for f in (Path(s), p, *pano_files):
            keys |= file_keys(f)
    return keys


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
    # one file per path_key (the first spelling): a case variant would move it twice
    present = {
        path_key(n): n
        for n in reversed([*old, *names])
        if (dst / n).exists() or (dst / n).is_symlink()
    }
    olds = [] if created else sorted(present.values(), key=lambda n: (n != "manifest.json", n))
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


class _CancelWatch:
    """Kill a decode pass's process as soon as ``cancel`` is set (checked every
    :data:`CANCEL_POLL_S`), so a stalled ffmpeg dies at once instead of at the consumer's
    next frame. :attr:`fired` tells the pass that the kill was its own."""

    def __init__(self, proc: Any, cancel: threading.Event) -> None:
        self.fired = False
        self._proc = proc
        self._cancel = cancel
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._done.is_set():
            if self._cancel.wait(CANCEL_POLL_S):
                if not self._done.is_set() and self._proc.poll() is None:
                    self.fired = True
                    self._proc.kill()
                return

    def finish(self) -> None:
        """The pass is over: stop watching."""
        self._done.set()


CANCEL_POLL_S = 0.1


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


def _isolated() -> dict[str, Any]:
    """``Popen`` options that start a child in its own process group (Windows) or session
    (POSIX), so the console's Ctrl+C (and the terminal's SIGINT) does not reach it: the
    caller stops it. No stdin, so it never reads the console."""
    if os.name == "nt":
        return {"stdin": subprocess.DEVNULL, "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"stdin": subprocess.DEVNULL, "start_new_session": True}


_JOB_HANDLE: Any = None


def _kill_with_parent(proc: subprocess.Popen) -> None:
    """Windows: put ``proc`` in a job object that is killed when Python exits.

    An isolated child leaves the console's group, so a ``taskkill /F`` of Python would leave
    it running until its next write to the closed pipe; the job (KILL_ON_JOB_CLOSE, as
    :mod:`cutpoint.events.llama_server` uses) ends it at once. POSIX needs nothing: the pipe
    closes with the parent and ffmpeg exits on EPIPE. No-op if the API is unavailable.
    """
    global _JOB_HANDLE
    if os.name != "nt":
        return
    try:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        if _JOB_HANDLE is None:

            class _Basic(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class _Ext(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", _Basic),
                    ("IoInfo", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            k32.CreateJobObjectW.restype = wintypes.HANDLE
            job = k32.CreateJobObjectW(None, None)
            if not job:
                logger.debug("job object not created: error %d", ctypes.get_last_error())
                return
            info = _Ext()
            info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
            if not k32.SetInformationJobObject(
                wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info)
            ):
                logger.debug("job object limits not set: error %d", ctypes.get_last_error())
                k32.CloseHandle(wintypes.HANDLE(job))
                return
            _JOB_HANDLE = job
        if not k32.AssignProcessToJobObject(
            wintypes.HANDLE(_JOB_HANDLE),
            wintypes.HANDLE(int(proc._handle)),  # type: ignore[attr-defined]
        ):
            logger.debug("ffmpeg not put in the job object: error %d", ctypes.get_last_error())
    except (OSError, AttributeError, ValueError) as err:
        logger.debug("job object not available: %s", err)


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
    isolate: bool = False,
    cancel: threading.Event | None = None,
    exit_codes: list[int] | None = None,
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
        isolate: Start ffmpeg in its own process group (Windows; also in a job object
            that ends it with Python) or session (POSIX), without stdin, so Ctrl+C does
            not stop it: for a decode in a background thread, whose owner stops it by
            closing the generator (ffmpeg is killed) instead of reading a cut-short pass.
        cancel: Kill ffmpeg as soon as this is set (a watcher thread), even while the
            consumer is busy or ffmpeg stalls; the pass then ends like a short one.
        exit_codes: If given, a pass's non-zero exit code is appended when the decode did
            not cause it (no early stop, timeout or ``cancel``): ffmpeg was stopped from
            outside, e.g. by a console Ctrl+Break, which reaches every process group.

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
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=frame_bytes * 4,
            **(_isolated() if isolate else {}),
        )
        if isolate:
            _kill_with_parent(proc)
        reader = threading.Thread(target=log.read, args=(proc.stderr,), daemon=True)
        reader.start()
        deadline = _Deadline(proc, timeout) if timeout is not None else None
        watch = _CancelWatch(proc, cancel) if cancel is not None else None
        stopped = False  # killed here because the consumer stopped early
        ended = False  # ffmpeg closed its output
        k = 0
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    ended = True
                    break
                t = log.pts_time(k) if pts else (t_base + k / stamp_fps if stamp_fps else float(k))
                yield t, np.frombuffer(buf, np.uint8).reshape(shape)
                k += 1
        finally:
            if deadline is not None:
                deadline.finish()  # output ended or the consumer stopped: no timeout now
            if watch is not None:
                watch.finish()
            if ended and exit_codes is not None:  # let it exit by itself: its code counts
                with suppress(subprocess.TimeoutExpired):
                    proc.wait(SOFT_STOP_S)
            proc.stdout.close()
            if proc.poll() is None:
                stopped = True
                proc.kill()  # the consumer stopped early
            proc.wait()
            reader.join(timeout=5)
        ours = stopped or (watch is not None and watch.fired) or bool(deadline and deadline.expired)
        if exit_codes is not None and proc.returncode != 0 and not ours:
            exit_codes.append(proc.returncode)
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


# ----------------------------------------------------------------- shared decode (#375)
SPOOL_DIR = "ocr-spool"  # in a source's cache folder: the raw frames of a split decode
SPOOL_SEGMENT_FRAMES = 600  # frames per spool file (each is deleted once read)
SPOOL_READ = 4 << 20  # bytes per read of ffmpeg's raw output
SPOOL_MARGIN = 1 << 30  # free disk a split decode leaves beyond its worst-case spool


def single_video_stream(path: Path | str) -> bool:
    """Whether ``path`` has exactly one video stream and it is not cover art, so ``[0:v:0]``
    is the stream a plain ``-vf`` decode picks."""
    try:
        data = ffprobe_json(
            path, entries="stream=index:stream_disposition=attached_pic", streams="v"
        )
    except FFmpegError:
        return False
    streams = data.get("streams", [])
    return len(streams) == 1 and not streams[0].get("disposition", {}).get("attached_pic")


def spool_fits(folder: Path, need: int) -> bool:
    """Whether the drive of ``folder`` has ``need`` bytes free plus :data:`SPOOL_MARGIN`."""
    try:
        return shutil.disk_usage(folder).free >= need + SPOOL_MARGIN
    except OSError:
        return False


def split_decode_args(
    path: Path | str,
    *,
    scene_vf: str,
    scene_enc: Sequence[str],
    proxy: Path,
    raw_vf: str,
    pix_fmt: PixFmt = "bgr24",
    hw: bool = True,
) -> list[str]:
    """ffmpeg arguments (without the executable) for one CUDA pass that writes both the
    scene proxy (``scene_vf`` and ``scene_enc`` as the scene cuts use them) to ``proxy`` and
    the raw frames of ``raw_vf`` (as :func:`decode_raw` gets them) to stdout. ``hw`` False
    decodes on the CPU (tests)."""
    graph = f"[0:v:0]split=2[s][o];[s]{scene_vf}[sv];[o]{raw_vf}[ov]"
    return [
        *("-hide_banner", "-nostats", "-v", "error", "-y"),
        *(("-hwaccel", "cuda") if hw else ()),
        *("-i", str(path), "-filter_complex", graph),
        *("-map", "[sv]", "-an", *scene_enc, str(proxy)),
        *("-map", "[ov]", "-an", "-sn", "-dn", "-pix_fmt", pix_fmt, "-f", "rawvideo", "-"),
    ]


def run_cancellable(args: Sequence[str], cancel: threading.Event) -> bool:
    """Run ffmpeg isolated from Ctrl+C (and, on Windows, ended with Python), killed as soon
    as ``cancel`` is set; True when it exited 0 by itself. Its progress goes to a pipe drained
    here, so it never blocks. On POSIX an orphan left by a killed Python finishes its pass
    (ffmpeg ignores a failed progress write); ``offer()`` sweeps its ``proxy.tmp.mp4``."""
    cmd = _command("ffmpeg", ["-progress", "pipe:1", *args])
    logger.debug("exec: %s", subprocess.list2cmdline(cmd))
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, **_isolated())
        _kill_with_parent(proc)
        watch = _CancelWatch(proc, cancel)
        assert proc.stdout is not None
        try:
            for _ in proc.stdout:  # drained, so ffmpeg never blocks on it
                pass
            proc.wait()
        finally:
            proc.stdout.close()
            watch.finish()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        if proc.returncode != 0 and not watch.fired:
            err.seek(0)
            logger.debug("ffmpeg exit %s: %s", exit_code(proc.returncode), _tail(err.read()))
    return proc.returncode == 0 and not watch.fired


class SplitDecode:
    """One CUDA pass (:func:`split_decode_args`) that writes the scene proxy and spools the
    raw frames to ``spool`` (files of :data:`SPOOL_SEGMENT_FRAMES` frames, each deleted once
    :meth:`frames` has read it), so ffmpeg runs at decode speed however slowly the frames are
    consumed. ffmpeg starts here, isolated from Ctrl+C and killed on ``cancel``.

    The proxy is written to ``proxy.tmp.mp4`` beside ``proxy`` and renamed to ``proxy`` only
    when ffmpeg exited 0 by itself and the spool took all of its output; otherwise it is
    deleted. ``on_exit`` then gets that outcome and the spool peak in bytes (from the spool
    thread).
    """

    def __init__(
        self,
        path: Path | str,
        *,
        size: tuple[int, int],
        raw_vf: str,
        stamp_fps: float,
        scene_vf: str,
        scene_enc: Sequence[str],
        proxy: Path,
        spool: Path,
        cancel: threading.Event,
        on_exit: Callable[[bool, int], None],
        stderr_lines: list[str] | None = None,
        hw: bool = True,
    ) -> None:
        width, height = size
        self.frame_bytes = width * height * PIX_CHANNELS["bgr24"]
        self.shape = (height, width, 3)
        self.seg_bytes = self.frame_bytes * SPOOL_SEGMENT_FRAMES
        self.stamp_fps = stamp_fps
        self.proxy = proxy
        self.tmp = proxy.with_name(proxy.stem + ".tmp" + proxy.suffix)
        self.spool = spool
        self.on_exit = on_exit
        self.peak = 0  # most spool bytes on disk at once
        self.returncode: int | None = None
        self._cond = threading.Condition()
        self._written = 0  # bytes of raw output spooled
        self._deleted = 0  # bytes of spool files read and deleted
        self._ended = False  # the spool thread is done (ffmpeg exited)
        self._error: OSError | None = None  # a spool write failed
        self._stopped = False  # ffmpeg killed here (spool error, or the consumer stopped)
        self._closed = False
        self._seg: IO[bytes] | None = None  # the spool file frames() reads
        args = split_decode_args(
            path, scene_vf=scene_vf, scene_enc=scene_enc, proxy=self.tmp, raw_vf=raw_vf, hw=hw
        )
        cmd = _command("ffmpeg", args)
        logger.debug("exec: %s", subprocess.list2cmdline(cmd))
        self._log = _StderrLog(False, stderr_lines)
        spool.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **_isolated()
        )
        _kill_with_parent(self._proc)
        self._reader = threading.Thread(
            target=self._log.read, args=(self._proc.stderr,), daemon=True
        )
        self._reader.start()
        self._watch = _CancelWatch(self._proc, cancel)
        self._cancel = cancel
        self._thread = threading.Thread(target=self._run, name="ffmpeg-spool", daemon=True)
        self._thread.start()

    # ---- spool thread
    def _kill(self) -> None:
        with self._cond:
            if self._proc.poll() is None:
                self._stopped = True
                self._proc.kill()

    def _run(self) -> None:
        out: IO[bytes] | None = None
        filled, idx = 0, 0  # bytes in the current spool file, its number
        stdout = self._proc.stdout
        assert stdout is not None
        try:
            while data := stdout.read1(SPOOL_READ):
                view = memoryview(data)
                while view:
                    if out is None:
                        out = (self.spool / f"seg-{idx:05d}.raw").open("wb")
                    n = min(len(view), self.seg_bytes - filled)
                    out.write(view[:n])
                    filled, view = filled + n, view[n:]
                    if filled == self.seg_bytes:  # closed before the consumer may delete it
                        out.close()
                        out, filled, idx = None, 0, idx + 1
                    else:
                        out.flush()
                    with self._cond:
                        self._written += n
                        self.peak = max(self.peak, self._written - self._deleted)
                        self._cond.notify_all()
        except OSError as err:
            with self._cond:
                self._error = err
            self._kill()
        finally:
            if out is not None:
                with suppress(OSError):
                    out.close()
            stdout.close()
            if self._error is None and self._proc.poll() is None:
                with suppress(subprocess.TimeoutExpired):
                    self._proc.wait(SOFT_STOP_S)
                self._kill()  # stdout closed and still running: never happens
            self._proc.wait()
            self._watch.finish()
            with self._cond:
                ours = self._stopped or self._watch.fired or self._error is not None
                ok = self._proc.returncode == 0 and not ours
            if ok:
                try:
                    self.tmp.replace(self.proxy)
                except OSError as err:
                    logger.debug("scene proxy not kept: %s", err)
                    ok = False
            if not ok:
                with suppress(OSError):
                    self.tmp.unlink(missing_ok=True)
            with self._cond:
                self.returncode = self._proc.returncode
                self._ended = True
                self._cond.notify_all()
            self.on_exit(ok, self.peak)

    # ---- consumer
    def frames(self, exit_codes: list[int] | None = None) -> Iterator[tuple[float, np.ndarray]]:
        """The raw frames, as :func:`decode_raw` yields them (``t = k / stamp_fps``). Stops
        early when ``cancel`` is set; ``exit_codes`` gets a non-zero exit of ffmpeg that was
        not caused here.

        Raises:
            FFmpegError: If a spool write failed (after the frames before it) or ffmpeg
                gave no frames.
        """
        fb, k, pos = self.frame_bytes, 0, 0
        try:
            while True:
                with self._cond:
                    while (
                        self._written < pos + fb
                        and not self._ended
                        and self._error is None
                        and not self._cancel.is_set()
                    ):
                        self._cond.wait(CANCEL_POLL_S)  # the cancel is checked in between
                    if self._error is not None or self._written < pos + fb:
                        break
                if self._seg is None:
                    name = f"seg-{pos // self.seg_bytes:05d}.raw"
                    try:
                        self._seg = (self.spool / name).open("rb")
                    except OSError as err:  # held by a scanner: both sides decode as before
                        raise FFmpegError(f"spool read failed: {err}") from err
                buf = self._seg.read(fb)
                pos += fb
                if pos % self.seg_bytes == 0:
                    self._seg.close()
                    self._seg = None
                    with suppress(OSError):  # held by a scanner: close() removes it later
                        (self.spool / f"seg-{pos // self.seg_bytes - 1:05d}.raw").unlink()
                    with self._cond:
                        self._deleted += self.seg_bytes
                yield k / self.stamp_fps, np.frombuffer(buf, np.uint8).reshape(self.shape)
                k += 1
        finally:
            self.close()
        if self._error is not None:
            raise FFmpegError(f"spool write failed: {self._error}")
        ours = self._stopped or self._watch.fired
        if exit_codes is not None and self.returncode not in (0, None) and not ours:
            exit_codes.append(self.returncode)
        if not k:
            raise FFmpegError(f"ffmpeg decoded no frames (shared decode):\n{self._log.text()}")

    def close(self) -> None:
        """Kill ffmpeg if it still runs (the frames are no longer read), wait for the spool
        thread and remove the spool. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._seg is not None:  # a paused frames(): its file must close before rmtree
            with suppress(OSError):
                self._seg.close()
            self._seg = None
        self._kill()
        self._thread.join()
        self._reader.join(timeout=5)
        shutil.rmtree(self.spool, ignore_errors=True)


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
        Read-only ``int16`` samples (1-D); ``pcm_float`` turns them into -1..1 floats.

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


def decode_tracks(
    path: Path | str,
    streams: int,
    *,
    sr: int,
    start: float,
    duration: float,
    timeout: float | None = None,
) -> np.ndarray:
    """Decode the first ``streams`` audio tracks of ``path`` side by side, each downmixed to mono.

    Returns:
        ``int16`` samples of shape ``(streams, samples)``, one row per track.

    Raises:
        FFmpegError: If ffmpeg exits non-zero or times out.
    """
    rows = "".join(
        f"[0:a:{i}]aformat=sample_fmts=s16:sample_rates={sr}:channel_layouts=mono[t{i}];"
        for i in range(streams)
    )
    merge = "".join(f"[t{i}]" for i in range(streams)) + f"amerge=inputs={streams}[a]"
    args = ["-hide_banner", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path)]
    args += ["-t", f"{duration:.3f}", "-vn", "-filter_complex", rows + merge, "-map", "[a]"]
    proc = _exec("ffmpeg", [*args, "-f", "s16le", "-"], timeout, path)
    if proc.returncode != 0:
        code = exit_code(proc.returncode)
        raise FFmpegError(
            f"ffmpeg track decode of {path} failed (exit {code}):\n{_tail(proc.stderr)}"
        )
    data = proc.stdout
    frames = len(data) // (2 * streams)
    pcm = np.frombuffer(data[: frames * 2 * streams], dtype="<i2")
    return pcm.reshape(frames, streams).T


def pcm_float(pcm: bytes | np.ndarray) -> np.ndarray:
    """16-bit little-endian PCM (bytes or :func:`decode_pcm` samples) as float32 in -1..1."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
