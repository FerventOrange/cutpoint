"""SRT subtitles for exported clips and assembled videos.

Cues come from the cached source transcripts that ``cutpoint analyze`` wrote
(:func:`cutpoint.sources.source_words`, hallucinations dropped), so no
transcription runs for them. Words are grouped into cues by the burned-in
captions' :func:`~cutpoint.assemble.caption_chunks` and placed with the joined
video's :func:`~cutpoint.assemble.clip_offsets`. A clip's source range is its
manifest ``start`` / ``end``: the in/out points after tails, padding
(``extend``) and speech moves. A clip renamed during review has no manifest
entry; its own ``.srt``, renamed with it, is used instead.

A clips folder's ``.srt`` (or clip) that its manifest.json does not list was not made by
cutpoint and is never replaced (:func:`listed_files` and the helpers after it).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cutpoint import cache, sources
from cutpoint.assemble import caption_chunks, clip_offsets
from cutpoint.config import path_key, remap_source, source_path
from cutpoint.ffmpeg_utils import MediaInfo

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


class ClipSubtitles:
    """Writes each exported clip's ``.srt`` next to it (``cutpoint clips`` and ``extend``)
    from the cached source transcript; :meth:`report` logs what was left out, once.

    ``protected``: :func:`~cutpoint.config.path_key` of ``.srt`` names in the clips folder
    that an earlier run did not write; a clip with such a name gets no ``.srt``.
    """

    def __init__(self, protected: set[str] | None = None) -> None:
        self.words = words_lookup()
        self.protected = protected or set()
        self.no_words: list[str] = []
        self.not_written: list[str] = []

    def write(self, clip: Path, source: str, start: float, end: float) -> str | None:
        """Write ``<clip>.srt`` for the source range ``start``-``end`` of ``source``.

        Returns:
            Its name (manifest ``srt``), or None if none was written.
        """
        if (w := self.words(source)) is None:
            self.no_words.append(source)
            return None
        name = clip.with_suffix(".srt").name
        if path_key(name) in self.protected:
            self.not_written.append(name)
            return None
        write_srt(clip.with_suffix(".srt"), clip_cues(w, start, end))
        return name

    def report(self, total: int) -> None:
        """Log the clips without subtitles (INFO) and the ``.srt`` files kept (WARNING)."""
        if self.no_words:
            skipped_info(f"{len(self.no_words)} of {total} clips", self.no_words)
        if self.not_written:
            logger.warning(
                "Kept %s: an earlier run's manifest.json does not list them (made by hand?), "
                "so they are not replaced and these clips get no new subtitles",
                ", ".join(self.not_written),
            )


def final_cues(
    files: list[Path],
    infos: list[MediaInfo],
    entries: dict[str, dict[str, Any]],
    fade: float,
    words: Callable[[str], list[dict[str, Any]] | None],
) -> tuple[list[Cue], list[str], list[str]]:
    """Cues of the joined video of ``files`` (crossfade ``fade``, 0 for cuts).

    ``entries`` maps file names to their manifest entries. A clip without one (renamed
    during review) uses its sibling ``<clip>.srt``, if any. Each clip's cues are clipped
    to the clip and shifted to its :func:`~cutpoint.assemble.clip_offsets` start; where a
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


def write_final_srt(
    clips_dir: Path, files: list[Path], infos: list[MediaInfo], fade: float, video: Path
) -> Path | None:
    """Write ``<video>.srt`` for the video :func:`cutpoint.assemble.assemble` joined from
    ``files`` (crossfade ``fade``, 0 for cuts), using ``clips_dir``'s manifest.json.

    Clips without a manifest entry (and no ``.srt`` of their own) or a cached transcript
    get no cues (one INFO line); when no clip has any, no file is written.

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


# ------------------------------------------------- what clips --force / extend may replace
def listed_files(out_dir: Path) -> list[str] | None:
    """The clip and ``.srt`` names the manifest.json in ``out_dir`` lists (files an earlier
    run made), or None if there is no readable manifest.json."""
    manifest = cache.load_json(out_dir / "manifest.json")
    entries = manifest.get("clips") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        return None
    listed: list[str] = []
    for e in entries:
        for key, ext in (("file", ".mp4"), ("srt", ".srt")):
            f = e.get(key) if isinstance(e, dict) else None
            # a bare clip name only: never a path out of the folder or another kind of file
            if isinstance(f, str) and f == Path(f).name and f.lower().endswith(ext):
                listed.append(f)
    return listed


def unlisted_srts(out_dir: Path, listed: list[str] | None) -> set[str]:
    """:func:`~cutpoint.config.path_key` of the ``.srt`` files in ``out_dir`` that
    ``listed`` (:func:`listed_files`) does not name: no subtitles are written under them."""
    ours = {path_key(n) for n in listed or []}
    return {k for f in out_dir.glob("*.srt") if f.is_file() and (k := path_key(f.name)) not in ours}


def check_unlisted_clips(out_dir: Path, new: list[str], listed: list[str] | None) -> None:
    """Fail before any export if writing the clip names ``new`` into ``out_dir`` would
    replace a ``.mp4`` that ``listed`` (:func:`listed_files`) does not name: renamed during
    review or added by hand (every one, without a readable manifest.json).

    Raises:
        FileExistsError: Naming them.
    """
    if not out_dir.is_dir():
        return
    ours = {path_key(n) for n in listed or []}
    names = {path_key(n) for n in new}
    hits = [
        p.name
        for p in sorted(out_dir.iterdir())
        if (k := path_key(p.name)) in names and k not in ours and (p.is_file() or p.is_symlink())
    ]
    if hits:
        why = (
            "the old manifest.json does not list them (renamed during review or added by hand)"
            if listed is not None
            else "there is no readable manifest.json, so they may not be clips cutpoint exported"
        )
        raise FileExistsError(
            f"Cannot replace {', '.join(hits)} in {out_dir}: {why}. Move or rename them, "
            "then run again."
        )
