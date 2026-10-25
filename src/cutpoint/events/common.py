"""Shared helpers for the events layers: sources, caching, transcript, GPU hygiene."""

from __future__ import annotations

import contextlib
import logging
import re
import subprocess
import threading
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

from cutpoint import cache
from cutpoint.backends import ollama_url

logger = logging.getLogger(__name__)

WAV_SR = 16000
EVENTS_CACHE_VERSION = 1


def params_key(params: dict[str, Any]) -> str:
    """Short stable hash of a parameter dict (for cache file names)."""
    return cache.params_hash({"v": EVENTS_CACHE_VERSION, **params})


def layer_cache_path(cdir: Path, layer: str, params: dict[str, Any]) -> Path:
    """Cache file for one layer's results on one source, keyed by parameters."""
    return cdir / f"events-{layer}-{params_key(params)}.json"


@dataclass(frozen=True)
class AudioPart:
    """A contiguous cached 16 kHz WAV covering ``[offset, offset + duration]``."""

    offset: float
    wav: Path
    duration: float


@dataclass
class Source:
    """One source video plus its analysis cache."""

    path: Path
    duration: float
    cdir: Path
    audio_parts: list[AudioPart]

    @property
    def name(self) -> str:
        """File name used in log lines (matches transcript.txt)."""
        return self.path.name

    @property
    def io_path(self) -> Path:
        """File to decode: ``path`` or its ``source_remap`` copy (``path`` stays the id)."""
        from cutpoint.config import remap_source  # noqa: PLC0415

        return remap_source(self.path)


def _audio_parts(src: Path) -> list[AudioPart]:
    """Cached WAV parts for ``src``; uses ``sources.parts`` when available.

    ``sources.parts`` stitches files recovered by ``analyze --recover``; older
    trees only have the single base cache folder.
    """
    from cutpoint.sources import parts as cached_parts  # noqa: PLC0415

    found = [(p.offset, p.cdir) for p in cached_parts(src)]
    out: list[AudioPart] = []
    for offset, cdir in found:
        wav = cdir / "audio16k.wav"
        if wav.exists():
            with wave.open(str(wav), "rb") as w:
                out.append(AudioPart(offset, wav, w.getnframes() / w.getframerate()))
    return out


def load_source(path: Path, duration: float | None = None) -> Source:
    """Build a :class:`Source` without decoding anything (metadata only).

    Args:
        path: Source video (only ``stat``-ed here).
        duration: Known duration (e.g. from highlights.json); else the WAV length.
    """
    parts = _audio_parts(path)
    if duration is None:
        duration = max((p.offset + p.duration for p in parts), default=0.0)
    return Source(path=path, duration=duration, cdir=cache.source_dir(path), audio_parts=parts)


def iter_audio_blocks(
    src: Source, block_s: float, overlap_s: float = 0.0
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(abs_start_time, float32 samples)`` blocks of the cached 16 kHz audio.

    Blocks overlap by ``overlap_s`` so models get context at the edges. Only
    one block is in memory at a time.
    """
    for part in src.audio_parts:
        with wave.open(str(part.wav), "rb") as w:
            n_total = w.getnframes()
            step = int((block_s - overlap_s) * WAV_SR)
            size = int(block_s * WAV_SR)
            pos = 0
            while pos < n_total:
                w.setpos(pos)
                raw = w.readframes(min(size, n_total - pos))
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                yield part.offset + pos / WAV_SR, samples
                if pos + size >= n_total:
                    break
                pos += step


def audio_slice(src: Source, t0: float, t1: float) -> np.ndarray:
    """16 kHz mono float32 samples for ``[t0, t1]`` from the cached WAV (may be short)."""
    for part in reversed(src.audio_parts):
        if t0 >= part.offset:
            with wave.open(str(part.wav), "rb") as w:
                a = max(0, int((t0 - part.offset) * WAV_SR))
                b = min(w.getnframes(), int((t1 - part.offset) * WAV_SR))
                if b <= a:
                    break
                w.setpos(a)
                raw = w.readframes(b - a)
            return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return np.zeros(0, dtype=np.float32)


def wav_bytes(samples: np.ndarray, sr: int = WAV_SR) -> bytes:
    """Encode float32 mono samples as a 16-bit PCM WAV file in memory."""
    import io  # noqa: PLC0415

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


# --------------------------------------------------------------------------- project

_TRANSCRIPT_RE = re.compile(
    r"^\[(?P<name>.+?) (?P<a>\d\d:\d\d:\d\d\.\d)-(?P<b>\d\d:\d\d:\d\d\.\d)\] (?P<text>.*)$"
)


def parse_clock(s: str) -> float:
    """Inverse of :func:`cutpoint.utils.clock` (including a leading ``-``)."""
    neg = s.startswith("-")
    h, m, sec = s.removeprefix("-").split(":")
    t = int(h) * 3600 + int(m) * 60 + float(sec)
    return -t if neg else t


def load_transcript_lines(project: Path) -> list[dict[str, Any]]:
    """Parse ``transcript.txt`` into ``{source_name, start, end, text, line}`` dicts."""
    path = project / "transcript.txt"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _TRANSCRIPT_RE.match(line)
        if m:
            out.append(
                {
                    "name": m["name"],
                    "start": parse_clock(m["a"]),
                    "end": parse_clock(m["b"]),
                    "text": m["text"],
                    "line": line,
                }
            )
    return out


def transcript_excerpt(
    lines: list[dict[str, Any]], name: str, t0: float, t1: float, max_chars: int = 900
) -> str:
    """Transcript text overlapping ``[t0, t1]``, with times relative to ``t0``."""
    parts = [
        f"[{max(0.0, ln['start'] - t0):.1f}s] {ln['text']}"
        for ln in lines
        if ln["name"] == name and ln["end"] >= t0 and ln["start"] <= t1
    ]
    return "\n".join(parts)[:max_chars]


# Analyses in progress: project folder -> the highlights.json they will write.
_PENDING: dict[Path, dict[str, Any]] = {}


@contextlib.contextmanager
def pending_highlights(project: Path, data: dict[str, Any]) -> Iterator[None]:
    """Let :func:`load_highlights` return ``data`` for ``project`` in the block.

    ``cutpoint analyze`` runs the event layers and crash detection before it writes
    highlights.json (once, at the end); they read the analysis from here meanwhile.
    """
    key = project.resolve()
    _PENDING[key] = data
    try:
        yield
    finally:
        _PENDING.pop(key, None)


def load_highlights(project: Path) -> dict[str, Any]:
    """Read ``highlights.json`` of a project (or the analysis in progress, see
    :func:`pending_highlights`).

    Raises:
        FileNotFoundError: If it is missing.
        ValueError: If it is unreadable, not valid JSON or has no ``sources`` list.
    """
    if _PENDING and (pending := _PENDING.get(project.resolve())) is not None:
        return pending
    path = project / "highlights.json"
    try:
        data = cache.read_json(path, hint="; re-run `cutpoint analyze` to rewrite it")
    except FileNotFoundError:
        raise FileNotFoundError(f"{path} not found; run `cutpoint analyze` first") from None
    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):  # e.g. null
        raise ValueError(f"{path} has no sources list; re-run `cutpoint analyze` to rewrite it")
    return data


# --------------------------------------------------------------------------- GPU


def unload_ollama(url: str | None = None) -> list[str]:
    """Ask Ollama to unload every loaded model (``keep_alive: 0``).

    Args:
        url: Ollama base URL (default: :func:`cutpoint.backends.ollama_url`).

    Returns:
        Names of the models that were unloaded (empty if Ollama is not running).
    """
    try:
        url = url or ollama_url()
        loaded = requests.get(f"{url}/api/ps", timeout=3).json().get("models", [])
    except (requests.RequestException, ValueError):
        return []
    names = []
    for m in loaded:
        name = m.get("name") or m.get("model")
        try:
            requests.post(f"{url}/api/generate", json={"model": name, "keep_alive": 0}, timeout=30)
            logger.info("Unloaded Ollama model %s", name)
            names.append(name)
        except requests.RequestException as err:
            logger.warning("Could not unload Ollama model %s: %s", name, err)
    if not names:  # nothing was loaded (or nothing could be unloaded): no wait
        return names
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            if not requests.get(f"{url}/api/ps", timeout=3).json().get("models"):
                break
        except (requests.RequestException, ValueError):
            break
        time.sleep(0.5)
    return names


def gpu_memory_used_mib() -> int | None:
    """Current total GPU memory use in MiB via nvidia-smi (None if unavailable)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class VramMonitor:
    """Background thread that records peak GPU memory use while active.

    Usage::

        with VramMonitor() as vm:
            ...
        vm.peak_mib, vm.baseline_mib
    """

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.baseline_mib: int | None = None
        self.peak_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            used = gpu_memory_used_mib()
            if used is not None:
                self.peak_mib = max(self.peak_mib or 0, used)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> VramMonitor:
        self.baseline_mib = gpu_memory_used_mib()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self) -> dict[str, int | None]:
        """``{"baseline_mib", "peak_mib"}`` for timing reports."""
        return {"baseline_mib": self.baseline_mib, "peak_mib": self.peak_mib}


def free_torch() -> None:
    """Release cached CUDA memory held by torch (no-op without torch/CUDA)."""
    import gc  # noqa: PLC0415

    gc.collect()
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
