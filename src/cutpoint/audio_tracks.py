"""Which audio tracks of a source a clip export mixes.

Exports sum every audio track (``amix``, no normalisation), so a game track and a separate
mic track both keep their full level. Recorders such as OBS can also write the same mix to
several tracks; summing N copies comes out 20*log10(N) dB too loud and clips. The tracks of a
source are compared once on a few short windows spread across it (cached per source in its
cache folder, ``audio-tracks.json``), and a track that copies an earlier one is left out of
the mix; each clip then checks the left-out tracks again on its own span, in 1 s chunks, and
keeps a track that differs anywhere in it (a mic callout between the windows). A track that
is silent in every window is kept: it adds nothing to the sum there, and it may carry sound
elsewhere.

Copies are told apart by the energy of their difference, not by correlation alone: AAC
coding noise of a re-encoded copy stays about 17 dB below the signal in every 1 s chunk
(measured: -16.6 dB at worst). With AAC, two encodes of slightly different input (a mic even
30 dB down) differ by about -11 dB, so the extra track stays; with lossless audio the
difference equals the extra content's level, so content 15 dB or more below the other track
in every 1 s chunk counts as a copy. Bit-identical copies (OBS) differ by nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from cutpoint import cache, state
from cutpoint.ffmpeg_utils import FFmpegError, decode_tracks, probe

logger = logging.getLogger(__name__)

TRACKS_CACHE = "audio-tracks.json"
TRACKS_CACHE_VERSION = 2  # 2: residual-energy test
WINDOWS = 6  # windows compared, spread evenly across the source
WINDOW_S = 3.0
SR = 8000  # band-limited: AAC coding noise sits mostly above 4 kHz
CHUNK_S = 1.0  # a clip's span is compared in chunks this long ...
CHUNK_HOP_S = 0.5  # ... overlapping by half, so a short callout is not split between two
MIN_WINDOW_S = 0.5  # a window decoding shorter than this (read error, end of file) is skipped
SILENT_DB = -60.0  # RMS (dBFS) below this is silence: no evidence either way
MAX_RESIDUAL_DB = -15.0  # copies: energy of their difference this far below the louder ...
MAX_LEVEL_DB = 0.5  # ... and their levels differ by at most this much, in every window

# (path, size, mtime_ns, streams[, start, duration]) -> kept tracks of the source (of a clip
# span); the INFO lines are logged once per source and span
_KEPT: dict[tuple, tuple[int, ...]] = state.register_container(__name__, "_KEPT", {})


def same_window(a: np.ndarray, b: np.ndarray) -> bool | None:
    """Whether two tracks' samples of one window are copies of each other.

    Returns:
        True when both have sound, their levels differ by at most :data:`MAX_LEVEL_DB` and
        the energy of ``a - b`` is at least :data:`MAX_RESIDUAL_DB` below the louder one's;
        None when both are silent; False otherwise (only one of them silent included).
    """
    energy = [float(np.mean(x * x)) for x in (a, b)]
    levels = [10 * np.log10(e + 1e-24) for e in energy]
    loud = [db >= SILENT_DB for db in levels]
    if not any(loud):
        return None
    if not all(loud) or abs(levels[0] - levels[1]) > MAX_LEVEL_DB:
        return False
    residual = 10 * np.log10(float(np.mean((a - b) ** 2)) / max(energy) + 1e-24)
    return bool(residual <= MAX_RESIDUAL_DB)  # a bool, not numpy's: callers test ``is False``


def kept_tracks(windows: Iterable[np.ndarray]) -> tuple[int, ...]:
    """Tracks to mix, given decoded windows (each ``(tracks, samples)`` floats, read once).

    A track is left out when it copies a track kept before it (:func:`same_window`) in every
    window where either has sound, with at least one such window. Track 0 is always kept.
    """
    n = 0
    same: set[tuple[int, int]] = set()
    differ: set[tuple[int, int]] = set()
    for w in windows:
        n = w.shape[0]
        for k in range(n):
            for j in range(k + 1, n):
                if (k, j) in differ:
                    continue
                verdict = same_window(w[k], w[j])
                if verdict is False:
                    differ.add((k, j))
                elif verdict:
                    same.add((k, j))
    copies = same - differ
    kept = [0]
    for j in range(1, n):
        if not any((k, j) in copies for k in kept):
            kept.append(j)
    return tuple(kept)


def chunks(pcm: np.ndarray) -> Iterator[np.ndarray]:
    """:data:`CHUNK_S` chunks (every :data:`CHUNK_HOP_S`) of ``int16`` ``(tracks, samples)``
    as -1..1 floats; the last ones, at the end of the span, are shorter. Chunks shorter than
    :data:`MIN_WINDOW_S` are left out."""
    size, hop = int(CHUNK_S * SR), int(CHUNK_HOP_S * SR)
    for c in range(0, pcm.shape[1], hop):
        part = pcm[:, c : c + size]
        if part.shape[1] >= MIN_WINDOW_S * SR:
            yield part.astype(np.float64) / 32768.0


def window_starts(duration: float) -> list[float]:
    """Start times (s) of the compared windows: :data:`WINDOWS` spread across ``duration``
    (fewer, overlapping windows merged, for a source shorter than that)."""
    w = min(WINDOW_S, duration)
    starts = {
        round(max(0.0, min(duration - w, (i + 0.5) * duration / WINDOWS - w / 2)), 3)
        for i in range(WINDOWS)
    }
    return sorted(starts)


def _detect(src: Path, streams: int, duration: float) -> tuple[int, ...] | None:
    """:func:`kept_tracks` of ``src``; None when no window could be decoded."""
    w = min(WINDOW_S, duration)
    windows = []
    for t in window_starts(duration):
        try:
            pcm = decode_tracks(src, streams, sr=SR, start=t, duration=w, timeout=60)
        except FFmpegError as err:  # one attempt each: never retry a bad spot
            logger.debug("track compare window %.1fs of %s failed: %s", t, src.name, err)
            continue
        if pcm.shape[1] >= MIN_WINDOW_S * SR:
            windows.append(pcm.astype(np.float64) / 32768.0)
    return kept_tracks(windows) if windows else None


def export_tracks(
    src: Path,
    streams: int,
    duration: float | None = None,
    span: tuple[float, float] | None = None,
) -> tuple[int, ...]:
    """Audio tracks of ``src`` a clip export mixes: all ``streams`` but duplicates.

    The comparison of the whole source is cached per source (path, size, modification
    time). Without a source file to compare (missing, or no window decodes) every track is
    mixed, as before. A track it leaves out is checked again on the clip's whole ``span``,
    in :func:`chunks`, and kept when it differs in any of them: a mic track that copies the
    game track except where the player talks keeps that talk.

    Args:
        src: Source video (the file actually read).
        streams: Its number of audio tracks (:attr:`~cutpoint.ffmpeg_utils.MediaInfo.audio_streams`).
        duration: Its length (s); probed when needed and not given.
        span: ``(start, duration)`` (s) of the clip exported from it, if any.
    """
    every = tuple(range(streams))
    if streams < 2:
        return every
    try:
        st = Path(src).stat()
    except OSError:
        return every
    key = (str(src), st.st_size, st.st_mtime_ns, streams)
    kept = _KEPT.get(key)
    if kept is None:
        kept = _KEPT[key] = _source_tracks(src, streams, duration)
    if span is None or len(kept) == streams:
        return kept
    start, dur = round(span[0], 3), round(span[1], 3)
    skey = (*key, start, dur)
    if skey not in _KEPT:
        _KEPT[skey] = _confirm_on_span(src, streams, kept, start, dur)
    return _KEPT[skey]


def _confirm_on_span(
    src: Path, streams: int, kept: tuple[int, ...], start: float, dur: float
) -> tuple[int, ...]:
    """``kept`` plus the tracks that do not copy another one in ``[start, start + dur]``
    (compared in :func:`chunks`)."""
    try:
        pcm = decode_tracks(src, streams, sr=SR, start=start, duration=dur, timeout=max(60, dur))
    except FFmpegError as err:
        logger.debug("track compare of %s at %.1fs failed: %s", src.name, start, err)
        return kept
    if pcm.shape[1] < MIN_WINDOW_S * SR:
        return kept
    here = kept_tracks(chunks(pcm))
    union = tuple(sorted({*kept, *here}))
    if union != kept:
        logger.info(
            "Audio of %s at %.1fs: track%s %s differ%s here; mixing tracks %s",
            src.name,
            start,
            "s" if len(union) - len(kept) > 1 else "",
            ", ".join(str(i + 1) for i in union if i not in kept),
            "" if len(union) - len(kept) > 1 else "s",
            ", ".join(str(i + 1) for i in union),
        )
    return union


def _source_tracks(src: Path, streams: int, duration: float | None) -> tuple[int, ...]:
    """:func:`kept_tracks` of the whole source (cached in its cache folder); every track when
    nothing could be compared (memoised by the caller, not cached on disk)."""
    every = tuple(range(streams))
    cfile = cache.source_dir(src) / TRACKS_CACHE
    cached = cache.load_json(cfile)
    if (
        cached
        and cached.get("version") == TRACKS_CACHE_VERSION
        and cached.get("streams") == streams
        and isinstance(cached.get("tracks"), list)
    ):
        kept = tuple(cached["tracks"])
    else:
        try:
            length = duration if duration is not None else probe(src).duration
        except FFmpegError as err:
            logger.debug("track compare of %s: probe failed: %s", src.name, err)
            return every
        found = _detect(src, streams, length)
        if found is None:
            logger.debug("track compare of %s: no window decoded; mixing every track", src.name)
            return every
        kept = found
        cache.save_json(
            cfile, {"version": TRACKS_CACHE_VERSION, "streams": streams, "tracks": list(kept)}
        )
    if len(kept) < streams:
        logger.info(
            "Audio of %s: %d of %d tracks duplicate another and are left out; mixing track%s %s",
            src.name,
            streams - len(kept),
            streams,
            "s" if len(kept) > 1 else "",
            ", ".join(str(i + 1) for i in kept),
        )
    return kept
