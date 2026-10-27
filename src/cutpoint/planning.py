"""Clip planning steps shared by ``cutpoint clips`` and ``cutpoint extend``.

* :func:`select_moments`: pick moments from a project's ``highlights.json``;
* :class:`Padding` / :func:`pad`: editor handles around a clip, clamped to the
  readable part of its source;
* :func:`speech_pass`: the speech-aware boundary pass
  (:mod:`cutpoint.boundaries`) over a list of planned clips;
* :func:`report_clips`: the ``--json`` outputs of the clips folder written.
* :func:`manifest_entries` / :func:`check_own_manifest`: which clips and which
  manifest.json a clips folder's run made (renamed clips by size; ``generator``).

Nothing here encodes or writes files (see :mod:`cutpoint.encode`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from cutpoint import boundaries, cache, jsonout, sources
from cutpoint.config import path_key
from cutpoint.events.tails import load_events
from cutpoint.ffmpeg_utils import MediaInfo

logger = logging.getLogger(__name__)

MANIFEST_GENERATOR = "cutpoint"  # manifest ``generator``: written by clips / extend


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
            when ``max_total`` is given, else the analysis' ``suggested_top``
            (or 5).
        ids: Exact candidate ids, in this order (overrides ``top``).
        max_total: Keep the best-scoring chosen moments whose summed duration
            fits in this many seconds.

    Candidates marked ``must_include`` (confident crashes) are added to an
    automatic selection and kept even past ``max_total``.

    Returns:
        ``(moments, style name)``: the chosen candidate dicts (ids order, else
        by score) and the analysis style (default ``gaming``).

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
    if ids:
        by_id = {c["id"]: c for c in cands}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise jsonout.ArgumentError(f"Unknown ids: {', '.join(missing)}")
        chosen = [by_id[i] for i in ids]
    else:
        ranked = sorted(cands, key=lambda c: -c["score"])
        n = (
            top
            if top is not None
            else (len(ranked) if max_total else data.get("suggested_top") or 5)
        )
        chosen = ranked[:n]
        # Confident crashes (ski analyze, crashes.json) are never left out.
        forced = [c for c in ranked if c.get("must_include") and c not in chosen]
        if forced:
            logger.info(
                "Adding %d must-include crash clip(s): %s",
                len(forced),
                ", ".join(c["id"] for c in forced),
            )
            chosen += forced
    if max_total:
        fitted, total = [], 0.0
        for c in sorted(chosen, key=lambda c: (not c.get("must_include"), -c["score"])):
            # Must-include crashes go first and stay even past the cap.
            if c.get("must_include") or total + c["duration"] <= max_total:
                fitted.append(c)
                total += c["duration"]
        chosen = [c for c in chosen if c in fitted]
    if not chosen:
        raise ValueError("No moments selected")
    return chosen, data.get("style", "gaming")


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
        the requested and applied padding, the pre-pad in/out points and
        ``clamped`` when a side got less than requested.
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
        "pre_pad": {"start": round(start, 3), "end": round(end, 3)},
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
        plans: Planned clips, in output order.
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
    boundaries.fit_clips(bounds, speech)
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
    present = {path_key(p.name) for p in folder.iterdir()}
    gone: dict[int, list[dict[str, Any]]] = {}
    for e in entries:
        size = e.get("size")
        if isinstance(size, int) and path_key(e["file"]) not in present:
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
