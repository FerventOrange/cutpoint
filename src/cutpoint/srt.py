"""SRT subtitles for exported clips and assembled videos.

Cues come from the cached source transcripts that ``cutpoint analyze`` wrote
(:func:`cutpoint.sources.source_words`, hallucinations dropped), so no
transcription runs for them. Words are grouped into cues by the burned-in
captions' :func:`~cutpoint.assemble.caption_chunks` and placed with the joined
video's :func:`~cutpoint.assemble.clip_offsets`. A clip's source range is its
manifest ``start`` / ``end``: the in/out points after tails, padding
(``extend``) and speech moves.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cutpoint import cache, sources
from cutpoint.assemble import caption_chunks, clip_offsets
from cutpoint.config import remap_source, source_path
from cutpoint.ffmpeg_utils import MediaInfo

logger = logging.getLogger(__name__)

Cue = tuple[float, float, str]  # (start s, end s, text)

SRT_MAX_WORDS = 8  # words per cue; subtitles read longer lines than the 4-word burned-in captions
MIN_CUE_S = 0.1  # a cue cut shorter than this at a clip edge is dropped


def srt_time(t: float) -> str:
    """``HH:MM:SS,mmm`` of ``t`` seconds (negative counts as 0)."""
    ms = round(max(t, 0.0) * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_srt(cues: list[Cue]) -> str:
    """SubRip text: 1-based index, ``start --> end``, the text, a blank line between cues."""
    return "".join(
        f"{i}\n{srt_time(s)} --> {srt_time(e)}\n{text}\n\n"
        for i, (s, e, text) in enumerate(cues, 1)
    )


def write_srt(path: Path, cues: list[Cue]) -> None:
    """Write ``cues`` to ``path`` atomically, UTF-8 (see :func:`cutpoint.cache.save_bytes`)."""
    cache.save_bytes(path, format_srt(cues).encode("utf-8"))


def clip_cues(words: list[dict[str, Any]], start: float, end: float) -> list[Cue]:
    """Cues of the source range ``start``-``end`` in clip time (0 = ``start``).

    Words overlapping the range are chunked like the burned-in captions, then
    clipped to ``0``-``end - start``.
    """
    dur = end - start
    inside = [
        {**w, "start": w["start"] - start, "end": w["end"] - start}
        for w in words
        if w["end"] > start and w["start"] < end
    ]
    cues = []
    for s, e, text in caption_chunks([{"words": inside}], max_words=SRT_MAX_WORDS):
        s, e = max(s, 0.0), min(e, dur)
        if e - s >= MIN_CUE_S:
            cues.append((s, e, text))
    return cues


def words_lookup() -> Callable[[str], list[dict[str, Any]] | None]:
    """``words(source)``: the cached transcript words of a recorded source (read through
    ``source_remap``), or None when it has none or cannot be read. Loaded once per source."""
    loaded: dict[str, list[dict[str, Any]] | None] = {}

    def words(source: str) -> list[dict[str, Any]] | None:
        if source not in loaded:
            try:
                loaded[source] = sources.source_words(remap_source(source))
            except OSError:  # the source is gone (moved, drive disconnected)
                loaded[source] = None
        return loaded[source]

    return words


def skipped_info(what: str, missing_sources: list[str]) -> None:
    """The one INFO line for subtitles left out: no cached transcript of
    ``missing_sources``, else (none given) no manifest.json entry."""
    names = ", ".join(dict.fromkeys(source_path(s).name for s in missing_sources))
    logger.info(
        "No subtitles for %s: %s",
        what,
        f"no cached transcript of {names} (run `cutpoint analyze` on it)"
        if names
        else "not in the folder's manifest.json (not exported by `cutpoint clips`)",
    )


def final_cues(
    files: list[Path],
    infos: list[MediaInfo],
    entries: dict[str, dict[str, Any]],
    fade: float,
    words: Callable[[str], list[dict[str, Any]] | None],
) -> tuple[list[Cue], list[str], list[str]]:
    """Cues of the joined video of ``files`` (crossfade ``fade``, 0 for cuts).

    ``entries`` maps file names to their manifest entries. Each clip's cues are shifted
    to its :func:`~cutpoint.assemble.clip_offsets` start; where a crossfade overlaps two
    clips, a cue ends when the next one starts.

    Returns:
        ``(cues, clips without subtitles, their recorded sources)``; a clip without a
        manifest entry adds no source.
    """
    cues: list[Cue] = []
    missing: list[str] = []
    missing_sources: list[str] = []
    for f, info, offset in zip(files, infos, clip_offsets(infos, fade), strict=True):
        e = entries.get(f.name)
        w = words(e["source"]) if e else None
        if w is None:
            missing.append(f.name)
            if e:
                missing_sources.append(e["source"])
            continue
        start = float(e["start"])
        end = min(float(e["end"]), start + info.vduration)
        cues += [(s + offset, t + offset, text) for s, t, text in clip_cues(w, start, end)]
    cues.sort(key=lambda c: c[0])
    out: list[Cue] = []
    for i, (s, e, text) in enumerate(cues):
        if i + 1 < len(cues):
            e = min(e, cues[i + 1][0])
        if e - s >= MIN_CUE_S:
            out.append((s, e, text))
    return out, missing, missing_sources


def write_final_srt(
    clips_dir: Path, files: list[Path], infos: list[MediaInfo], fade: float, video: Path
) -> Path | None:
    """Write ``<video>.srt`` for the video :func:`cutpoint.assemble.assemble` joined from
    ``files`` (crossfade ``fade``, 0 for cuts), using ``clips_dir``'s manifest.json.

    Clips without a manifest entry or a cached transcript get no cues (one INFO line);
    when no clip has any, no file is written.

    Returns:
        The .srt path, or None if none was written.
    """
    manifest = cache.load_json(clips_dir / "manifest.json") or {}
    entries = {e.get("file"): e for e in manifest.get("clips", []) if isinstance(e, dict)}
    cues, missing, missing_sources = final_cues(files, infos, entries, fade, words_lookup())
    srt = video.with_suffix(".srt")
    if len(missing) == len(files):
        skipped_info(video.name, missing_sources)
        if srt.exists():
            logger.warning("%s is from an earlier run and does not match %s", srt, video.name)
        return None
    if missing:
        skipped_info(f"{', '.join(missing)} in {video.name}", missing_sources)
    write_srt(srt, cues)
    logger.info("Wrote %s (%d subtitles)", srt, len(cues))
    return srt
