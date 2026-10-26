"""Per-source access to cached analysis data, including recovered parts.

A damaged source analysed with ``cutpoint analyze --recover`` is split into parts:
the file itself (offset 0) plus one virtual source per resume point, each with
its own cache folder (``cache.source_dir(src, offset)``). Every part folder
may hold a ``read_errors.json`` record naming where reading failed and where
the next part resumes. The helpers here stitch the parts back together so
callers can work in absolute source time.
"""

from __future__ import annotations

import logging
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.ffmpeg_utils import FFmpegError, MediaInfo, pcm_float, read_pcm_window, wav_duration
from cutpoint.transcribe import filter_hallucinations, flat_words, transcript_name

logger = logging.getLogger(__name__)

READ_ERRORS_FILE = "read_errors.json"
SR = 16000  # analysis WAV sample rate
MAX_PARTS = 8  # safety bound when following resume links
TRANSCRIPT_MODELS = ("large-v3", "medium", "small")  # most accurate first
_LANGUAGE_SUFFIX = re.compile(r"-[a-z]{2,3}$")  # transcript-<model>-<language> (#371)


@dataclass(frozen=True)
class Part:
    """One contiguous, readable stretch of a source with its own cache folder.

    Attributes:
        offset: Absolute source time where the part's WAV starts.
        cdir: Cache folder of the part.
        end: Absolute time where readable audio ends (None if unknown).
    """

    offset: float
    cdir: Path
    end: float | None


def load_read_errors(cdir: Path) -> dict[str, Any] | None:
    """The ``read_errors.json`` record of a part folder, if any."""
    return cache.load_json(cdir / READ_ERRORS_FILE)


def parts(src: Path) -> list[Part]:
    """Cached parts of ``src`` in time order (at least the base part)."""
    out: list[Part] = []
    offset = 0.0
    for _ in range(MAX_PARTS):
        cdir = cache.source_dir(src, offset)
        wav = cdir / "audio16k.wav"
        end = offset + wav_duration(wav) if wav.exists() else None
        out.append(Part(offset=offset, cdir=cdir, end=end))
        rec = load_read_errors(cdir) or {}
        nxt = rec.get("resumed_at")
        if not nxt or nxt <= offset:
            break
        offset = float(nxt)
    return out


def readable_end(src: Path, t: float, duration: float) -> float:
    """Where readable audio ends for the part containing time ``t``.

    Falls back to ``duration`` when the part's audio extent is unknown.
    """
    for p in reversed(parts(src)):
        if t >= p.offset:
            return min(duration, p.end) if p.end is not None else duration
    return duration


def readable_range(src: Path, t: float, duration: float) -> tuple[float, float]:
    """``(first, last)`` readable source time of the part containing ``t``.

    For an undamaged source that is ``(0, duration)``. For a source with a
    recorded read error the range stops at the failure, and a recovered part
    starts at its resume point, so padding never reaches the bad region.
    """
    lo = 0.0
    for p in parts(src):
        if t >= p.offset:
            lo = p.offset
    return lo, readable_end(src, t, duration)


def _wav_slice(wav: Path, t0: float, t1: float) -> np.ndarray:
    with wave.open(str(wav), "rb") as w:
        n = w.getnframes()
        a, b = max(0, int(t0 * SR)), min(n, int(t1 * SR))
        if b <= a:
            return np.zeros(0, dtype=np.float32)
        w.setpos(a)
        raw = w.readframes(b - a)
    return pcm_float(raw)


def audio_window(src: Path, info: MediaInfo, t0: float, t1: float) -> tuple[np.ndarray, float]:
    """16 kHz mono samples covering ``[t0, t1]`` (absolute source time).

    Reads the cached analysis WAV of the matching part; if no cached part
    covers the window, decodes just that window from the source.

    Returns:
        (samples as float32 in -1..1, absolute time of the first sample).
    """
    t0 = max(0.0, t0)
    for p in reversed(parts(src)):
        wav = p.cdir / "audio16k.wav"
        if t0 >= p.offset and p.end is not None and wav.exists():
            if t1 <= p.end + 0.05:
                return _wav_slice(wav, t0 - p.offset, t1 - p.offset), t0
            break
    try:
        raw = read_pcm_window(src, info, t0, t1)
    except FFmpegError as err:
        logger.warning("%s", err)
        return np.zeros(0, dtype=np.float32), t0
    return pcm_float(raw), t0


def project_language(project: Path) -> str | None:
    """Transcription language the analysis of ``project`` forced (highlights.json
    ``settings.language``), or None: detected by Whisper, or no analysis."""
    hl = cache.load_json(project / "highlights.json")
    settings = hl.get("settings") if isinstance(hl, dict) else None
    return settings.get("language") if isinstance(settings, dict) else None


def _transcript_file(cdir: Path, language: str | None = None) -> Path | None:
    """The cached transcript in ``cdir`` to read, most accurate model first.

    With ``language``, its ``transcript-<model>-<language>.json``; without (or when none
    exists), a transcript of the detected language (``transcript-<model>.json``), and a
    file of a forced language only when there is no other.
    """
    files = sorted(cdir.glob("transcript-*.json"))
    if not files:
        return None
    names = {f.name: f for f in files}
    language = language.lower() if language else None
    if language:
        for model in TRANSCRIPT_MODELS:
            if (f := names.get(transcript_name(model, language))) is not None:
                return f
        if keyed := [f for f in files if f.stem.endswith(f"-{language}")]:
            return keyed[0]
    for model in TRANSCRIPT_MODELS:
        if (f := names.get(transcript_name(model))) is not None:
            return f
    detected = [f for f in files if not _LANGUAGE_SUFFIX.search(f.stem)]
    return (detected or files)[0]


def source_segments(src: Path, language: str | None = None) -> list[dict[str, Any]] | None:
    """Cached transcript segments of ``src`` in absolute time (their words too),
    hallucinations dropped, or None if untranscribed.

    ``language``: the one the analysis forced (``analyze --language``); None reads the
    transcript of the detected language (see :func:`_transcript_file`).
    """
    out: list[dict[str, Any]] = []
    found = False
    for p in parts(src):
        f = _transcript_file(p.cdir, language)
        if f is None:
            continue
        segments = cache.load_json(f)
        if segments is None:
            continue
        found = True
        segments, _ = filter_hallucinations(segments)
        out.extend(
            {
                **seg,
                "start": seg["start"] + p.offset,
                "end": seg["end"] + p.offset,
                "words": [
                    {**w, "start": w["start"] + p.offset, "end": w["end"] + p.offset}
                    for w in seg.get("words", [])
                ],
            }
            for seg in segments
        )
    return out if found else None


def source_words(src: Path, language: str | None = None) -> list[dict[str, Any]] | None:
    """All cached transcript words of ``src`` in absolute time, or None if untranscribed.

    Each word also carries its segment's ``avg_logprob`` (when the transcript
    stored it), a confidence for gating words heard in wind. ``language`` as in
    :func:`source_segments`.
    """
    segments = source_segments(src, language)
    if segments is None:
        return None
    words: list[dict[str, Any]] = []
    for seg in segments:
        conf = {"avg_logprob": seg["avg_logprob"]} if seg.get("avg_logprob") is not None else {}
        words.extend({**w, **conf} for w in flat_words([seg]))
    words.sort(key=lambda w: w["start"])
    return words
