"""Clip planning steps shared by ``cutpoint clips`` and ``cutpoint extend``.

* :func:`select_moments`: pick moments from a project's ``highlights.json``;
* :class:`Padding` / :func:`pad`: editor handles around a clip, clamped to the
  readable part of its source;
* :func:`speech_pass`: the speech-aware boundary pass
  (:mod:`cutpoint.boundaries`) over a list of planned clips.

Nothing here encodes or writes files (see :mod:`cutpoint.encode`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from cutpoint import boundaries, cache, sources
from cutpoint.events.tails import load_events
from cutpoint.ffmpeg_utils import MediaInfo

logger = logging.getLogger(__name__)


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
        ValueError: If highlights.json is unreadable, not valid JSON or not an
            object with candidates, an id is unknown or nothing is selected.
    """
    try:
        data = cache.read_json(
            project / "highlights.json", hint="; re-run `cutpoint analyze` to rewrite it"
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No highlights.json in {project}; run 'cutpoint analyze' first"
        ) from None
    if not isinstance(data, dict) or "candidates" not in data:  # e.g. null or a list
        raise ValueError(
            f"{project / 'highlights.json'} is not a JSON object with candidates; "
            "re-run `cutpoint analyze` to rewrite it"
        )
    cands: list[dict[str, Any]] = data["candidates"]
    if ids:
        by_id = {c["id"]: c for c in cands}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(f"Unknown ids: {', '.join(missing)}")
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
    lo, hi = sources.readable_range(src, start, info.duration)
    if start >= hi:
        # The clip lies past the last known readable point (a recovered part
        # whose cache is not linked): the bad region is somewhere before it, so
        # never pad backwards, and only the duration bounds the end.
        lo, hi = start, info.duration
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
                words[src] = sources.source_words(src)
            maps[key] = boundaries.build_speech_map(
                src, words[src], events.get(str(Path(key))), speech
            )
        keep_start, keep_end = plan.keep
        lo, hi = sources.readable_range(src, keep_start, info.duration)
        if keep_start >= hi:  # recovered part past the last known readable point
            lo, hi = keep_start, info.duration
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
