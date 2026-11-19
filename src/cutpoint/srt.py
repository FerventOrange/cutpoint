"""SRT subtitles for exported clips and assembled videos.

Cues come from the cached source transcripts that ``cutpoint analyze`` wrote
(:func:`cutpoint.sources.source_words`, hallucinations dropped), so no
transcription runs for them. Words are grouped into cues by the burned-in
captions' :func:`caption_chunks` and placed with the joined
video's :func:`~cutpoint.timeline.clip_offsets`. A clip's source range is its
manifest ``start`` / ``end``: the in/out points after tails, padding
(``extend``) and speech moves. A clip renamed during review has no manifest
entry; its own ``.srt``, renamed with it, is used instead.

A clips folder's ``.srt`` (or clip) that its manifest.json does not list was not made by
cutpoint and is never replaced (:func:`cutpoint.planning.replace_plan`).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cutpoint import cache, jsonout, sources
from cutpoint.config import path_key, remap_source, source_path
from cutpoint.ffmpeg_utils import MediaInfo
from cutpoint.planning import manifest_parts, manifest_project
from cutpoint.timeline import clip_offsets
from cutpoint.transcribe import Segment, flat_words

logger = logging.getLogger(__name__)

Cue = tuple[float, float, str]  # (start s, end s, text)

SRT_MAX_WORDS = 8  # words per cue; subtitles read longer lines than the 4-word burned-in captions
MIN_CUE_S = 0.1  # a cue cut shorter than this at a clip edge is dropped
_TIMES = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{3})\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d{3})")


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


def read_srt(path: Path) -> list[Cue] | None:
    """The cues of the SubRip file ``path``, or None if it is missing or not UTF-8 text."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None
    cues = []
    for block in re.split(r"\n[ \t]*\n", text.strip()):  # text mode: CRLF read as \n
        lines = block.split("\n")
        for i, line in enumerate(lines):
            if m := _TIMES.search(line):
                h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(g) for g in m.groups())
                start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
                end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
                cues.append((start, end, "\n".join(lines[i + 1 :]).strip()))
                break
    return cues


def write_srt(path: Path, cues: list[Cue]) -> None:
    """Write ``cues`` to ``path`` atomically, UTF-8 (see :func:`cutpoint.cache.save_bytes`)."""
    cache.save_bytes(path, format_srt(cues).encode("utf-8"))


def caption_chunks(
    segments: list[Segment], max_words: int = 4, max_gap: float = 0.6
) -> list[tuple[float, float, str]]:
    """Group words into short (2-4 word) caption chunks: the burned-in captions of
    ``assemble --captions`` and, with ``max_words`` :data:`SRT_MAX_WORDS`, the cues."""
    chunks: list[tuple[float, float, str]] = []
    cur: list[dict] = []

    def flush() -> None:
        if cur:
            chunks.append((cur[0]["start"], cur[-1]["end"], " ".join(w["word"] for w in cur)))
            cur.clear()

    for w in flat_words(segments):
        if cur and (
            len(cur) >= max_words
            or w["start"] - cur[-1]["end"] > max_gap
            or cur[-1]["word"][-1:] in ".?!,"
        ):
            flush()
        cur.append(w)
    flush()
    # Hold each chunk a little longer, without overlapping the next one.
    held = []
    for i, (s, e, text) in enumerate(chunks):
        nxt = chunks[i + 1][0] if i + 1 < len(chunks) else e + 1.0
        held.append((s, min(max(e + 0.15, s + 0.5), nxt), text))
    return held


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


def words_lookup(language: str | None = None) -> Callable[[str], list[dict[str, Any]] | None]:
    """``words(source)``: the cached transcript words of a recorded source (read through
    ``source_remap``), or None when it has none or cannot be read. Loaded once per source.

    ``language``: the one the project's analysis forced (see
    :func:`cutpoint.sources.project_language`); None reads the detected language's."""
    loaded: dict[str, list[dict[str, Any]] | None] = {}

    def words(source: str) -> list[dict[str, Any]] | None:
        if source not in loaded:
            try:
                loaded[source] = sources.source_words(remap_source(source), language)
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


class ClipSubtitles:
    """Writes each exported clip's ``.srt`` next to it (``cutpoint clips`` and ``extend``)
    from the cached source transcript; :meth:`report` logs what was left out, once.

    ``protected``: :func:`~cutpoint.config.path_key` of ``.srt`` names in the clips folder
    that an earlier run did not write; a clip with such a name gets no ``.srt``.
    ``language``: the transcript language, as in :func:`words_lookup`.
    ``folder``: the clips folder the keys of ``protected`` were made for (default: the
    folder each clip is written to).
    """

    def __init__(
        self,
        protected: set[str] | None = None,
        language: str | None = None,
        folder: Path | None = None,
    ) -> None:
        self.words = words_lookup(language)
        self.protected = protected or set()
        self.folder = folder
        self.no_words: list[str] = []
        self.not_written: list[Path] = []

    def write(self, clip: Path, source: str, start: float, end: float) -> str | None:
        """Write ``<clip>.srt`` for the source range ``start``-``end`` of ``source``.

        Returns:
            Its name (manifest ``srt``), or None if none was written.
        """
        if (w := self.words(source)) is None:
            self.no_words.append(source)
            return None
        name = clip.with_suffix(".srt").name
        if path_key(name, self.folder or clip.parent) in self.protected:
            self.not_written.append((self.folder or clip.parent) / name)
            return None
        write_srt(clip.with_suffix(".srt"), clip_cues(w, start, end))
        return name

    def report(self, total: int) -> None:
        """Log the clips without subtitles (INFO) and the ``.srt`` files kept (WARNING)."""
        if self.no_words:
            skipped_info(f"{len(self.no_words)} of {total} clips", self.no_words)
        if self.not_written:
            message = (
                f"Kept {', '.join(p.name for p in self.not_written)}: an earlier run's manifest.json does not "
                "list them (made by hand?), so they are not replaced and these clips get no new "
                "subtitles"
            )
            logger.warning("%s", message)
            jsonout.warn("kept_srt", message, self.not_written)


def final_cues(
    files: list[Path],
    infos: list[MediaInfo],
    entries: dict[str, dict[str, Any]],
    fade: float | list[float],
    words: Callable[[str], list[dict[str, Any]] | None],
) -> tuple[list[Cue], list[str], list[str]]:
    """Cues of the joined video of ``files`` (crossfade ``fade``, 0 for cuts, or a list with
    one such value per join).

    ``entries`` maps file names to their manifest entries. A clip without one (renamed
    during review) uses its sibling ``<clip>.srt``, if any. Each clip's cues are clipped
    to the clip and shifted to its :func:`~cutpoint.timeline.clip_offsets` start; where a
    crossfade overlaps two clips, a cue ends when the next one starts.

    Returns:
        ``(cues, clips without subtitles, their recorded sources)``; a clip without a
        manifest entry adds no source.
    """
    cues: list[Cue] = []
    missing: list[str] = []
    missing_sources: list[str] = []
    for f, info, offset in zip(files, infos, clip_offsets(infos, fade), strict=True):
        e = entries.get(f.name)
        if e is not None and not all(k in e for k in ("source", "start", "end")):
            e = None  # an entry without its range is no entry
        if e is None and (own := read_srt(f.with_suffix(".srt"))) is not None:
            for s, t, text in own:
                s, t = max(s, 0.0), min(t, info.vduration)
                if t - s >= MIN_CUE_S:
                    cues.append((s + offset, t + offset, text))
            continue
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


def clips_language(clips_dir: Path) -> str | None:
    """Transcription language the analysis of ``clips_dir``'s project forced (the project
    of :func:`~cutpoint.planning.manifest_project`), or None: detected, or no analysis."""
    manifest = cache.load_json(clips_dir / "manifest.json")
    if not isinstance(manifest, dict):
        manifest = {}
    return sources.project_language(manifest_project(manifest, clips_dir))


def write_final_srt(
    clips_dir: Path,
    files: list[Path],
    infos: list[MediaInfo],
    fade: float | list[float],
    video: Path,
    language: str | None = None,
) -> Path | None:
    """Write ``<video>.srt`` for the video :func:`cutpoint.assemble.assemble` joined from
    ``files`` (crossfade ``fade``, as in :func:`final_cues`), using ``clips_dir``'s manifest.json and
    the transcripts of ``language`` (as in :func:`words_lookup`; None: the language the
    analysis of the clips' project used, see :func:`clips_language`).

    Clips without a manifest entry (and no ``.srt`` of their own) or a cached transcript
    get no cues (one INFO line); when no clip has any, no file is written.

    Returns:
        The .srt path, or None if none was written.
    """
    try:
        entries = {
            e["file"]: e for e in manifest_parts(cache.load_json(clips_dir / "manifest.json"))[1]
        }
    except ValueError:
        entries = {}  # none, or not a clips manifest: no entries
    if language is None:
        language = clips_language(clips_dir)
    cues, missing, missing_sources = final_cues(files, infos, entries, fade, words_lookup(language))
    srt = video.with_suffix(".srt")
    if len(missing) == len(files):
        skipped_info(video.name, missing_sources)
        if srt.exists():
            message = f"{srt} is from an earlier run and does not match {video.name}"
            logger.warning("%s", message)
            jsonout.warn("stale_srt", message, [srt])
        return None
    if missing:
        skipped_info(f"{', '.join(missing)} in {video.name}", missing_sources)
    write_srt(srt, cues)
    logger.info("Wrote %s (%d subtitles)", srt, len(cues))
    return srt
