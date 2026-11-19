"""Clip planning steps shared by ``cutpoint clips`` and ``cutpoint extend``.

* :func:`select_moments`: pick moments from a project's ``highlights.json``;
* :class:`Padding` / :func:`pad`: editor handles around a clip, clamped to the
  readable part of its source;
* :func:`speech_pass`: the speech-aware boundary pass
  (:mod:`cutpoint.boundaries`) over a list of planned clips;
* :func:`report_clips`: the ``--json`` outputs of the clips folder written.
* :func:`manifest_entries` / :func:`check_own_manifest`: which clips and which
  manifest.json a clips folder's run made (renamed clips by size; ``generator``).
* :func:`replace_plan` / :func:`warn_kept`: what ``clips --force`` and ``extend`` may
  replace in a clips folder (only what an earlier run made, never a source) and the
  warnings naming what they keep.

Nothing here encodes or writes files (see :mod:`cutpoint.encode`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from cutpoint import boundaries, cache, default_fit, folders, jsonout, sources, visual_pool
from cutpoint.config import path_key
from cutpoint.events.tails import load_events
from cutpoint.ffmpeg_utils import MediaInfo
from cutpoint.styles import style_name
from cutpoint.utils import clip_files

logger = logging.getLogger(__name__)

MANIFEST_GENERATOR = "cutpoint"  # manifest ``generator``: written by clips / extend
# the audio fade of an exact clip (``"exact": true``, a word splice of the AI Chat): only enough
# against clicks, so a short word keeps its sound
EXACT_AUDIO_FADE_S = 0.01


def select_moments(
    project: Path,
    *,
    top: int | None = None,
    ids: list[str] | None = None,
    max_total: float | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Pick moments from ``<project>/highlights.json``.

    Args:
        project: Project folder (analysed by ``cutpoint analyze``).
        top: Keep the ``top`` best-scoring candidates. Default: all of them
            when ``max_total`` is given, else the best moments that fit the
            style's ``target_total`` (:func:`~cutpoint.default_fit.fit_target`; their count is
            the analysis' ``suggested_top``), or without a target ``suggested_top`` (or 5).
        ids: Exact candidate ids, in this order (overrides ``top``).
        max_total: Keep the best-scoring chosen moments whose summed duration
            fits in this many seconds.

    Candidates marked ``must_include`` (confident crashes) are added to an
    automatic selection and kept even past ``max_total``.

    An automatic selection (not ``ids``) skips a candidate that mostly overlaps one already picked from
    the same source (not a ``must_include`` one) (a clip inside a long dialogue stretch); the next candidate takes its place.

    Visual pool moments (``pool: "visual"``, :mod:`cutpoint.visual_pool`) are left out of
    the automatic ranking. In a project of a pool style, an automatic selection reserves
    :func:`~cutpoint.visual_pool.reserved_slots` of its picks for them (by peak, only those
    with a peak of at least :data:`~cutpoint.visual_pool.SLOT_MIN_PEAK` that overlap no
    picked moment): each takes the place of the lowest-ranked pick that is not
    must-include, a crash or a jump (an action-cam ``kind`` of
    :data:`~cutpoint.visual_pool.PROTECTED_KINDS`, or overlapping a ``crashes.json`` crash of
    at least :data:`~cutpoint.visual_pool.CRASH_PROTECT_MIN` confidence, any pool style), and
    with ``max_total`` only when the total still fits.
    A reserved slot nobody qualifies for stays with its scored pick. ``ids`` may name pool
    moments.

    Returns:
        ``(moments, style name)``: the chosen candidate dicts (ids order, else
        by score) and the analysis style (:func:`cutpoint.styles.style_name`: for a file
        that records none, auto's style for unclear footage).

    Raises:
        FileNotFoundError: If the project has no highlights.json.
        ValueError: If highlights.json is unreadable, not valid JSON or has no
            candidates list, an id is unknown or nothing is selected.
    """
    try:
        data = cache.read_json(
            project / "highlights.json", hint="; re-run `cutpoint analyze` to rewrite it"
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No highlights.json in {project}; run 'cutpoint analyze' first"
        ) from None
    if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):  # e.g. null
        raise ValueError(
            f"{project / 'highlights.json'} has no candidates list; "
            "re-run `cutpoint analyze` to rewrite it"
        )
    cands: list[dict[str, Any]] = data["candidates"]
    pool = (
        [c for c in cands if visual_pool.is_pool(c)] if visual_pool.project_has_pool(data) else []
    )
    if ids:
        by_id = {c["id"]: c for c in cands}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise jsonout.ArgumentError(f"Unknown ids: {', '.join(missing)}")
        chosen = [by_id[i] for i in ids]
    else:
        ranked = sorted((c for c in cands if not visual_pool.is_pool(c)), key=lambda c: -c["score"])
        target = (data.get("settings") or {}).get("target_total")
        n = (
            top
            if top is not None
            else (len(ranked) if max_total else data.get("suggested_top") or 5)
        )
        if max_total and top is None:
            chosen = ranked  # the fit below skips the overlapping ones
        elif top is None and target:
            chosen = default_fit.fit_target(ranked, target)  # as analyze counts suggested_top
        else:
            chosen = default_fit.pick_best(ranked, count=n)  # skips the overlapping ones
        picks = len(chosen)
        # Confident crashes (action-cam analyze, crashes.json) are never left out.
        forced = [c for c in ranked if c.get("must_include") and c not in chosen]
        if forced:
            logger.info(
                "Adding %d must-include crash clip(s): %s",
                len(forced),
                ", ".join(c["id"] for c in forced),
            )
            chosen += forced
    if max_total:
        # Must-include crashes go first and stay even past the cap.
        ordered = sorted(chosen, key=lambda c: (not c.get("must_include"), -c["score"]))
        fitted = default_fit.pick_best(ordered, total=max_total, clash=not ids)
        chosen = [c for c in chosen if c in fitted]
    if not ids and pool:
        chosen = _reserve(project, chosen, pool, len(chosen) if max_total else picks, max_total)
    if not chosen:
        raise ValueError("No moments selected")
    return chosen, style_name(data)


def _reserve(
    project: Path,
    chosen: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    picks: int,
    max_total: float | None,
) -> list[dict[str, Any]]:
    """``chosen`` (an automatic selection of ``picks`` picks) with its reserved slots
    given to the qualifying ``pool`` moments (see :func:`select_moments`); the picks of
    :func:`~cutpoint.visual_pool.is_protected` (with ``project``'s crashes.json) keep theirs."""
    from cutpoint.crash import load_crashes  # noqa: PLC0415

    slots = visual_pool.reserved_slots(picks)
    if not slots:
        return chosen
    spans = visual_pool.protected_spans(load_crashes(project))
    victims = sorted(
        (c for c in chosen if not visual_pool.is_protected(c, spans)),
        key=lambda c: c["score"],
    )  # lowest-ranked first
    total = sum(default_fit.length(c) for c in chosen)
    taken: list[dict[str, Any]] = []
    for p in sorted(pool, key=lambda c: -round(10 * visual_pool.peak_of(c), 1)):
        if len(taken) == slots or not victims:
            break
        if visual_pool.peak_of(p) < visual_pool.SLOT_MIN_PEAK:
            continue
        if any(default_fit.span_overlaps(p, c) for c in chosen):
            continue
        v = victims[0]
        after = total - default_fit.length(v) + default_fit.length(p)
        if max_total and after > max_total and default_fit.length(p) > default_fit.length(v):
            continue
        victims.pop(0)
        chosen = [c for c in chosen if c is not v]
        total = after
        taken.append(p)
    if taken:
        logger.info(
            "Reserved %d slot(s) for visual moments: %s",
            len(taken),
            ", ".join(c["id"] for c in taken),
        )
    return chosen + taken


def exact_fade(entry: dict[str, Any], audio_fade: float) -> float:
    """The audio fade baked into the clip of ``entry`` (a segment or a manifest entry):
    ``audio_fade``, or at most :data:`EXACT_AUDIO_FADE_S` for an exact one."""
    return min(audio_fade, EXACT_AUDIO_FADE_S) if entry.get("exact") is True else audio_fade


@dataclass(frozen=True)
class Padding:
    """Requested padding in seconds (both >= 0)."""

    start: float = 0.0
    end: float = 0.0

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < 0:
            raise ValueError("Padding must not be negative")

    @property
    def active(self) -> bool:
        """Whether any padding is requested."""
        return self.start > 0 or self.end > 0

    @classmethod
    def from_args(cls, both: float | None, start: float | None, end: float | None) -> Padding:
        """``--extend`` for both sides, overridden per side by ``--extend-start``
        / ``--extend-end`` (all default to 0)."""
        base = both or 0.0
        return cls(start=base if start is None else start, end=base if end is None else end)

    def override(self, seg: dict[str, Any]) -> Padding:
        """Per-segment ``extend`` / ``extend_start`` / ``extend_end`` overrides."""
        if not any(k in seg for k in ("extend", "extend_start", "extend_end")):
            return self
        base_s = base_e = None
        if "extend" in seg:
            base_s = base_e = float(seg["extend"])
        s = (
            float(seg["extend_start"])
            if "extend_start" in seg
            else (base_s if base_s is not None else self.start)
        )
        e = (
            float(seg["extend_end"])
            if "extend_end" in seg
            else (base_e if base_e is not None else self.end)
        )
        return Padding(s, e)


def _readable_bounds(src: Path, info: MediaInfo, t: float) -> tuple[float, float]:
    """Readable ``(lo, hi)`` around ``t`` (:func:`sources.readable_range`)."""
    lo, hi = sources.readable_range(src, t, info.duration)
    if t >= hi:
        # ``t`` lies past the last known readable point (a recovered part
        # whose cache is not linked): the bad region is somewhere before it, so
        # never pad backwards, and only the duration bounds the end.
        lo, hi = t, info.duration
    return lo, hi


def pad(
    src: Path,
    info: MediaInfo,
    start: float,
    end: float,
    padding: Padding,
    entry: dict[str, Any] | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """Apply ``padding`` to ``[start, end]``, within the ``entry``'s (segment or
    manifest entry) ``min_start`` / ``max_end``.

    Returns:
        ``(new_start, new_end, record)``. ``record`` (for manifest.json) has
        the requested and applied padding and ``clamped`` when a side got
        less than requested.
    """
    lo, hi = _readable_bounds(src, info, start)
    b = entry or {}
    new_start = max(lo, 0.0, start - padding.start, min(start, b.get("min_start", 0.0)))
    # never shorter; never past the entry's bounds
    new_end = max(end, min(hi, info.duration, b.get("max_end", hi), end + padding.end))
    applied_s, applied_e = round(start - new_start, 3), round(new_end - end, 3)
    clamped = [
        side
        for side, req, got in (("start", padding.start, applied_s), ("end", padding.end, applied_e))
        if got < req - 0.001
    ]
    record = {
        "requested": {"start": padding.start, "end": padding.end},
        "applied": {"start": applied_s, "end": max(0.0, applied_e)},
    }
    if clamped:
        record["clamped"] = clamped
    return round(new_start, 3), round(new_end, 3), record


class SpeechPlan(Protocol):
    """A planned clip as seen by :func:`speech_pass` (its in/out are updated)."""

    name: str  # clip file name (records, logs)
    entry: dict[str, Any]  # moment / manifest entry: source, speech_aware, speech caps
    src: Path  # file actually read
    info: MediaInfo
    start: float
    end: float
    keep: tuple[float, float]  # the picked moment, before padding


def speech_pass[P: SpeechPlan](
    plans: Sequence[P],
    speech: boundaries.SpeechSettings,
    project: Path,
    *,
    fit_end: Callable[[dict[str, Any]], bool],
    hi_clamp: Callable[[P, float], float] | None = None,
    words_cache: dict[Path, list[dict[str, Any]] | None] | None = None,
    language: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Run the speech-aware boundary pass on planned clips (updates their in/out).

    Clips whose entry has ``"speech_aware": false`` are left alone; an entry's
    ``speech_max_extend`` / ``speech_max_lead`` replace the caps for that clip and its
    ``min_start`` / ``max_end`` bound it.
    Every clip stays inside the readable part of its source that holds its pick.

    Args:
        plans: Planned clips, in output order (a clip inside another's range is removed).
        speech: Pass settings (nothing happens when disabled).
        project: Project folder (its ``events.json`` feeds the speech maps).
        fit_end: Whether the end of a clip with this entry may move.
        hi_clamp: Adjusts a clip's last readable time before it is capped at
            the source duration, e.g. to stay before a recorded read error.
        words_cache: ``{source path: transcript words}`` shared with earlier
            steps (filled as needed).
        language: Transcript language the analysis forced; None: the detected one
            (:func:`cutpoint.sources.source_words`).

    Returns:
        ``{clip name: manifest record}`` for the clips the pass looked at.

    Raises:
        ValueError: If two plans share a name (records are keyed by it).
    """
    names = [plan.name for plan in plans]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"Duplicate clip names: {', '.join(dupes)}")
    if not speech.enabled:
        return {}
    words = {} if words_cache is None else words_cache
    events = load_events(project) or {}
    maps: dict[str, boundaries.SpeechMap | None] = {}
    bounds = []
    for plan in plans:
        e, src, info = plan.entry, plan.src, plan.info
        if e.get("speech_aware") is False:
            continue
        key = e["source"]
        if key not in maps:
            if src not in words:
                words[src] = sources.source_words(src, language)
            maps[key] = boundaries.build_speech_map(
                src, words[src], events.get(str(Path(key))), speech
            )
        keep_start, keep_end = plan.keep
        lo, hi = _readable_bounds(src, info, keep_start)
        if hi_clamp is not None:
            hi = hi_clamp(plan, hi)
        lo, hi = max(lo, e.get("min_start", lo)), min(hi, e.get("max_end", hi))
        bounds.append(
            boundaries.ClipBounds(
                name=plan.name,
                key=key,
                start=plan.start,
                end=plan.end,
                keep_start=keep_start,
                keep_end=keep_end,
                lo=max(0.0, lo),
                hi=min(hi, info.duration),
                smap=maps[key],
                fit_end_on=fit_end(e),
                max_extend=e.get("speech_max_extend"),
                max_lead=e.get("speech_max_lead"),
            )
        )
    dropped = boundaries.fit_clips(bounds, speech)
    if dropped:
        gone = {b.name for b in dropped}
        plans[:] = [p for p in plans if p.name not in gone]  # in place: the caller's list
        jsonout.warn(
            "nested_clip_dropped",
            f"{len(gone)} clip(s) lie inside a longer clip of the same source and were left "
            f"out: {', '.join(sorted(gone))}",
        )
    boundaries.log_summary(bounds)
    by_name = {b.name: b for b in bounds}
    for plan in plans:
        b = by_name.get(plan.name)
        if b is not None:
            plan.start, plan.end = b.start, b.end
    return {b.name: b.record for b in bounds}


def report_clips(out_dir: Path) -> None:
    """``--json`` outputs of a clips folder just written (``clips``, ``extend``): the folder,
    its manifest.json and, from it, each clip with its subtitles, the count and total length."""
    if not jsonout.active():
        return
    entries = (cache.load_json(out_dir / "manifest.json") or {}).get("clips", [])
    clips = [
        {
            "file": out_dir / e["file"],
            "srt": out_dir / e["srt"] if e.get("srt") else None,
            "label": e.get("label", ""),
            "source": e["source"],
            "start": e["start"],
            "end": e["end"],
            "score": e.get("score"),
        }
        for e in entries
    ]
    jsonout.output("clips_dir", out_dir)
    jsonout.output("manifest", out_dir / "manifest.json")
    jsonout.output("clips", clips)
    jsonout.output("count", len(clips))
    jsonout.output("total_seconds", round(sum(c["end"] - c["start"] for c in clips), 3))


# ------------------------------------------- manifest.json identity (clips, extend, assemble)
def file_size(path: Path) -> int | None:
    """Size of the file ``path`` in bytes (manifest ``size``), or None if it cannot be read."""
    try:
        return path.stat().st_size
    except OSError:
        return None


def foreign_manifest(folder: Path) -> str | None:
    """Why ``folder``'s manifest.json may not be one cutpoint wrote, or None: it is, or there
    is none. One with ``generator`` ``"cutpoint"`` is; one without (written before that key
    existed) is when it lists clips as cutpoint does, a ``clips`` list of entries with a
    ``file`` name."""
    path = folder / "manifest.json"
    if not (path.exists() or path.is_symlink()):
        return None
    data = cache.load_json(path)
    if data is None:
        return "it cannot be read as JSON"
    if not isinstance(data, dict):
        return "it is not a clips manifest"
    if "generator" in data:
        gen = data["generator"]
        return None if gen == MANIFEST_GENERATOR else f"its generator is {gen!r}"
    entries = data.get("clips")
    if isinstance(entries, list) and all(
        isinstance(e, dict) and isinstance(e.get("file"), str) for e in entries
    ):
        return None
    return "it does not list clips as cutpoint does"


def _has_range(e: dict[str, Any]) -> bool:
    return isinstance(e.get("source"), str) and all(
        isinstance(e.get(k), (int, float)) and not isinstance(e.get(k), bool)
        for k in ("start", "end")
    )


def manifest_parts(
    data: Any, *, need_format: bool = False, need_range: bool = False, path: Path | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``(format, entries)`` of a parsed manifest.json: its ``format`` record and its
    ``clips`` entries, each a dict with a ``file`` name.

    Args:
        need_format: Also require the format's ``width``, ``height`` and ``fps``.
        need_range: Also require every entry's str ``source`` and numeric ``start``, ``end``.
        path: The manifest's path, named in the error.

    Raises:
        ValueError: ``not a clips manifest`` if it has another shape.
    """
    err = ValueError(f"{path} is not a clips manifest" if path else "not a clips manifest")
    if not isinstance(data, dict):
        raise err
    fmt = data.get("format") or {}
    entries = data.get("clips", [])
    if (
        not isinstance(fmt, dict)
        or not isinstance(entries, list)
        or not all(isinstance(e, dict) and isinstance(e.get("file"), str) for e in entries)
        or (need_format and not all(k in fmt for k in ("width", "height", "fps")))
        or (need_range and not all(_has_range(e) for e in entries))
    ):
        raise err
    return fmt, entries


def check_own_manifest(folder: Path) -> None:
    """Fail before any export if ``--force`` would replace a manifest.json in ``folder``
    that cutpoint did not write (:func:`foreign_manifest`).

    Raises:
        FileExistsError: Naming it and saying what to do.
    """
    if (why := foreign_manifest(folder)) is not None:
        raise FileExistsError(
            f"{folder / 'manifest.json'} may not be one cutpoint wrote ({why}), so --force does "
            "not replace it. Move or rename it, or export into another folder (--out), then "
            "run again."
        )


def warn_pano_sources(clips: list[tuple[str, dict[str, Any] | None]], out_dir: Path | None) -> None:
    """One WARNING (and a ``pano_mixed_sources`` --json warning naming the 360 clips in
    ``out_dir``; none for a temporary folder, None) when the run's 360 clips are not
    rendered like-for-like (:func:`~cutpoint.pano.render.source_mix_warning`).

    ``clips``: each clip's name and 360 render record (manifest ``pano``, None for a flat
    clip), for ``clips`` and ``extend``."""
    from cutpoint.pano.render import MIXED_SOURCES, source_mix_warning  # noqa: PLC0415

    recs = [(name, rec) for name, rec in clips if rec]
    message = source_mix_warning(recs)
    if message is not None:
        logger.warning("%s", message)
        files = [out_dir / name for name, _ in recs] if out_dir is not None else []
        jsonout.warn(MIXED_SOURCES, message, files)


def manifest_project(manifest: dict[str, Any], clips_dir: Path) -> Path:
    """Project folder of a clips folder (manifest ``project``, else the parent)."""
    recorded = manifest.get("project")
    if recorded:
        try:
            if Path(recorded).is_dir():
                return Path(recorded)
        except OSError:
            pass
    return clips_dir.resolve().parent


def manifest_entries(folder: Path, files: list[Path]) -> dict[str, dict[str, Any]]:
    """The manifest.json entry of each of ``files`` (clips in ``folder``), by file name.

    A clip without an entry of its own name (renamed during review) gets the one entry whose
    clip is no longer in the folder and whose recorded ``size`` is the clip's, if exactly one
    such entry and one such clip match. Entries without ``size`` (written before it was
    recorded) match by name only. Clips with neither are left out.
    """
    manifest = cache.load_json(folder / "manifest.json")
    listed = manifest.get("clips") if isinstance(manifest, dict) else None
    if not isinstance(listed, list):
        return {}
    entries = [e for e in listed if isinstance(e, dict) and isinstance(e.get("file"), str)]
    by_name = {e["file"]: e for e in entries}
    found = {f.name: by_name[f.name] for f in files if f.name in by_name}
    present = {path_key(p.name, folder) for p in folder.iterdir()}
    gone: dict[int, list[dict[str, Any]]] = {}
    for e in entries:
        size = e.get("size")
        if isinstance(size, int) and path_key(e["file"], folder) not in present:
            gone.setdefault(size, []).append(e)
    if not gone:
        return found
    renamed: dict[int, list[str]] = {}
    for f in files:
        if f.name not in found and (size := file_size(f)) is not None:
            renamed.setdefault(size, []).append(f.name)
    for size, names in renamed.items():
        if len(names) == 1 and len(gone.get(size, [])) == 1:
            found[names[0]] = gone[size][0]
    return found


def kept_why(kept: list[str], renamed: dict[str, str], had_manifest: bool) -> str:
    """Why the clips ``kept`` stayed in a folder ``--force`` wrote into: renamed during review
    (``renamed`` maps them to their exported names, see :func:`manifest_entries` on the old
    manifest.json), else not listed in the old manifest.json, else no readable manifest.json."""
    if not had_manifest:
        return "not replaced because there is no readable manifest.json"
    ours = [f"{n} as {renamed[n]}" for n in kept if n in renamed]
    others = [n for n in kept if n not in renamed]
    if not ours:
        return "the old manifest.json does not list them (renamed during review or added by hand)"
    why = f"cutpoint exported {', '.join(ours)} (renamed during review)"
    if others:
        why += f"; the old manifest.json does not list {', '.join(others)} (added by hand?)"
    return why


# ------------------------------------------------- what clips --force / extend may replace
def _listed_pairs(out_dir: Path) -> list[tuple[str | None, str | None]] | None:
    """``(clip, .srt)`` names of each entry of the manifest.json in ``out_dir`` (None for a
    missing or invalid one), or None if there is no readable manifest.json."""
    manifest = cache.load_json(out_dir / "manifest.json")
    entries = manifest.get("clips") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        return None
    pairs: list[tuple[str | None, str | None]] = []
    for e in entries:
        names: list[str | None] = []
        for key, ext in (("file", ".mp4"), ("srt", ".srt")):
            f = e.get(key) if isinstance(e, dict) else None
            # a bare clip name only: never a path out of the folder or another kind of file
            ok = isinstance(f, str) and f == Path(f).name and f.lower().endswith(ext)
            names.append(f if ok else None)
        pairs.append((names[0], names[1]))
    return pairs


def listed_files(out_dir: Path) -> list[str] | None:
    """The clip and ``.srt`` names the manifest.json in ``out_dir`` lists (files an earlier
    run made), or None if there is no readable manifest.json."""
    pairs = _listed_pairs(out_dir)
    if pairs is None:
        return None
    return [f for pair in pairs for f in pair if f is not None]


def unlisted_srts(out_dir: Path, listed: list[str] | None) -> set[str]:
    """:func:`~cutpoint.config.path_key` of the ``.srt`` files in ``out_dir`` that
    ``listed`` (:func:`listed_files`) does not name: no subtitles are written under them."""
    ours = {path_key(n, out_dir) for n in listed or []}
    return {
        k
        for f in out_dir.glob("*.srt")
        if f.is_file() and (k := path_key(f.name, out_dir)) not in ours
    }


def check_unlisted_clips(out_dir: Path, new: list[str], listed: list[str] | None) -> None:
    """Fail before any export if writing the clip names ``new`` into ``out_dir`` would
    replace a ``.mp4`` that ``listed`` (:func:`listed_files`) does not name: renamed during
    review or added by hand (every one, without a readable manifest.json).

    Raises:
        FileExistsError: Naming them.
    """
    if not out_dir.is_dir():
        return
    ours = {path_key(n, out_dir) for n in listed or []}
    names = {path_key(n, out_dir) for n in new}
    hits = [
        p.name
        for p in sorted(out_dir.iterdir())
        if (k := path_key(p.name, out_dir)) in names
        and k not in ours
        and (p.is_file() or p.is_symlink())
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


@dataclass(frozen=True)
class ReplacePlan:
    """What ``clips --force`` or ``extend`` replaces in a clips folder and what it keeps
    (:func:`replace_plan`)."""

    replace: list[str]  # names replaced or removed besides the clips the run writes
    kept_sources: list[str]  # files kept because the run reads them as sources
    kept: list[str]  # other .mp4 kept: renamed during review or added by hand
    had_manifest: bool  # whether the old manifest.json was readable
    protected_srts: set[str]  # path_key of the .srt names no subtitles are written under
    renamed: dict[str, str]  # the exported name of each kept clip renamed during review


def replace_plan(
    out_dir: Path,
    new: list[str],
    sources: set[Any],
    *,
    clip_sources: set[Any] | None = None,
) -> ReplacePlan:
    """What a run that writes the clip names ``new`` into ``out_dir`` replaces there.

    Only files an earlier run made are replaced: the clips and their ``.srt`` subtitles
    listed in the old manifest.json (those the run does not write too, as the new
    manifest.json no longer lists them) and leftover ``.<name>.part.mp4`` files of a
    hard-killed run. A file the run reads (``sources``, see
    :func:`~cutpoint.folders.source_keys`) is never replaced; neither is any other ``.mp4``,
    so a name in ``new`` that is such a file stops the run (:func:`check_unlisted_clips`).
    Every other ``.srt`` (all of them without a readable manifest.json) is protected: no
    subtitles are written under its name. A manifest.json cutpoint did not write stops the
    run (:func:`check_own_manifest`).

    Only a name spelled on disk as listed is replaced: a clip renamed in case only still
    matches its listed name on a case-insensitive file system and must stay (kept, with a
    warning). A listed name spelled otherwise on disk goes, under that spelling, only if the
    run writes or replaces its clip: an ``.srt`` beside it would keep the old cues, unlisted.

    Args:
        clip_sources: More :func:`~cutpoint.folders.file_keys` (``extend``: every source of
            the manifest's clips, read this run or not). A kept ``.mp4`` that is one of them
            is named as a kept source, not a kept clip; it is not protected beyond
            ``sources``.

    Raises:
        ValueError: If a name in ``new`` is a source of this run (before any export).
        FileExistsError: If a name in ``new`` is another ``.mp4`` the old manifest.json
            does not list, or that manifest.json is not one cutpoint wrote (before any
            export).
    """
    folders.check_not_sources(out_dir, new, sources)
    if not out_dir.is_dir():
        return ReplacePlan([], [], [], True, set(), {})
    check_own_manifest(out_dir)
    listed = listed_files(out_dir)
    check_unlisted_clips(out_dir, new, listed)
    if listed is None and any(out_dir.iterdir()):
        logger.warning(
            "%s has no readable manifest.json: --force adds the clips this run writes and "
            "keeps every other file",
            out_dir,
        )
    parts = [p.name for p in out_dir.glob(".*.part.mp4")]
    present = {p.name for p in out_dir.iterdir()}
    on_disk = {path_key(n, out_dir): n for n in present}
    names = list(dict.fromkeys([*(listed or []), *parts]))
    replace: list[str] = []
    kept_sources: list[str] = []

    def add(n: str) -> None:
        f = out_dir / n
        if f.is_file() or f.is_symlink():
            (kept_sources if folders.same_file(f, sources) else replace).append(n)

    for n in names:
        if n in present:
            add(n)
    gone = {path_key(n, out_dir) for n in (*new, *replace)}
    for n in names:
        if (
            n not in present
            and (d := on_disk.get(path_key(n, out_dir))) is not None
            and path_key(Path(n).with_suffix(".mp4").name, out_dir) in gone
        ):
            add(d)
    done = {path_key(n, out_dir) for n in (*new, *replace, *kept_sources)}
    named_sources = sources | (clip_sources or set())
    kept: list[str] = []
    for p in clip_files(out_dir):
        if path_key(p.name, out_dir) not in done:
            (kept_sources if folders.same_file(p, named_sources) else kept).append(p.name)
    entries = manifest_entries(out_dir, [out_dir / n for n in kept])
    return ReplacePlan(
        replace,
        kept_sources,
        kept,
        listed is not None,
        unlisted_srts(out_dir, listed),
        {n: e["file"] for n, e in entries.items() if e["file"] != n},
    )


def warn_kept(out_dir: Path, plan: ReplacePlan, sources_why: str) -> None:
    """The WARNING and ``--json`` warning (``kept_sources``, ``kept_clips``) naming the files
    ``plan`` kept in ``out_dir``. ``sources_why`` says why a kept source stays; the hint that
    ``assemble`` includes them follows it after a space."""
    hint = "assemble includes them, so move them out if they are not part of the edit"
    shown = out_dir.absolute()
    if plan.kept_sources:
        message = f"Kept {', '.join(plan.kept_sources)} in {shown}: {sources_why} {hint}"
        logger.warning("%s", message)
        jsonout.warn("kept_sources", message, [out_dir / n for n in plan.kept_sources])
    if plan.kept:
        why = kept_why(plan.kept, plan.renamed, plan.had_manifest)
        message = f"Kept {', '.join(plan.kept)} in {shown}: {why}; {hint}"
        logger.warning("%s", message)
        jsonout.warn("kept_clips", message, [out_dir / n for n in plan.kept])
