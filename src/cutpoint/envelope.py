"""Audio envelope of a source for the UI's trim rail: the peak level per short bucket of time.

ffmpeg decodes the audio to mono 16-bit PCM at ``RATE`` Hz and writes it to a pipe;
:func:`read_peaks` folds it into one peak (0-1) per ``BASE_S`` seconds as it arrives, so a long
source costs a few bytes per bucket, never its decoded audio. :func:`shrink` then merges
neighbouring buckets (keeping the peak) until at most ``MAX_POINTS`` remain.
"""

from __future__ import annotations

from typing import BinaryIO

import numpy as np

from cutpoint.ffmpeg_utils import audio_mix_filter

RATE = 8000  # Hz: plenty for a level meter
BASE_S = 0.05  # seconds per bucket as decoded
MAX_POINTS = 6000  # buckets sent: 5 min at BASE_S, longer sources get coarser buckets
PER_BUCKET = round(RATE * BASE_S)  # samples
READ_BUCKETS = 4096  # buckets per read: about 3.2 MB of PCM at a time


def ffmpeg_args(src: str, tracks: int = 1) -> list[str]:
    """ffmpeg arguments (first item the tool name) that write ``src``'s audio as raw mono
    16-bit little-endian PCM at ``RATE`` Hz to stdout. ``tracks``: the source's audio streams;
    2 or more are mixed as the preview proxy mixes them (a game track and a separate mic)."""
    mix = audio_mix_filter(tracks)
    pick = ["-filter_complex", f"{mix}[a]", "-map", "[a]"] if mix else []
    return [
        "ffmpeg", "-v", "error", "-nostdin", "-i", src, "-vn", *pick,
        *("-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1"),
    ]  # fmt: skip


def read_peaks(stream: BinaryIO) -> np.ndarray:
    """The peak level (0-1, float32) of each ``BASE_S`` bucket of the PCM read from ``stream``.
    The last bucket may be shorter than the others."""
    out: list[np.ndarray] = []
    rest = b""
    size = PER_BUCKET * 2 * READ_BUCKETS
    while chunk := stream.read(size):
        rest += chunk
        whole = len(rest) // 2 // PER_BUCKET * PER_BUCKET * 2
        if whole:
            out.append(_fold(np.frombuffer(rest[:whole], dtype="<i2")))
            rest = rest[whole:]
    tail = rest[: len(rest) // 2 * 2]
    if tail:
        out.append(_fold(np.frombuffer(tail, dtype="<i2")))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def _fold(pcm: np.ndarray) -> np.ndarray:
    """One peak per ``PER_BUCKET`` samples of ``pcm`` (a shorter last bucket included)."""
    mag = np.abs(pcm.astype(np.int32)) / 32768.0
    pad = -len(mag) % PER_BUCKET
    if pad:
        mag = np.concatenate([mag, np.zeros(pad)])
    return mag.reshape(-1, PER_BUCKET).max(axis=1).astype(np.float32)


def shrink(peaks: np.ndarray) -> tuple[float, list[float]]:
    """``(seconds per value, values)``: ``peaks`` with neighbours merged by their maximum until
    at most ``MAX_POINTS`` remain, values rounded to 3 decimals."""
    factor = max(1, -(-len(peaks) // MAX_POINTS))
    pad = -len(peaks) % factor
    if factor > 1:
        peaks = np.concatenate([peaks, np.zeros(pad, dtype=np.float32)]).reshape(-1, factor).max(1)
    return round(BASE_S * factor, 4), [round(float(x), 3) for x in peaks]
