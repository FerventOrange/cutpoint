"""QA-driven repair of a ``cutpoint mix`` selection (``cutpoint 360 repair``).

Loop: ``360 director --segments mix_segments.json --preview --out-name mix``, then
``360 qa pano_paths/mix``, then ``360 repair``. A preview clip has a blocking
problem when

- it holds an occluded / too-close / dark run of :data:`BLOCK_RUN_S` or more (the
  wearer's arm, a friend's goggles at the lens, a hand over the lens), or
- it is a ``social`` clip with a person in fewer than :data:`MIN_PEOPLE` of its
  frames (a laugh heard, but only snow in view).

Its frames that are not clean become exclusions in source time, merged into
``mix_exclude.json``: occluded / too close / dark for every category, low information
(and, for the people problem, no person) for the clip's category only. A crash
clip's flags from :data:`PRE_IMPACT_S` before the impact to :data:`IMPACT_S` after
it are ignored (the tumble is the content) and blocked spans ending within
:data:`CRASH_LEAD_S` before it are cut back to end there (the approach stays). :func:`cutpoint.mix.run_mix` then
selects again keeping the previous clips: an affected clip is cut to its longest
clean part when that still meets :func:`cutpoint.mix.min_length` of its category,
else it is dropped and the freed budget is refilled under the usual quotas and
variety rules. Untouched clips keep their ids and bounds; the exclusions also
hold for later ``cutpoint mix`` runs. Edge cuts are not repaired.

Rendered clips (``cutpoint clips --mix``, a folder with ``manifest.json``) add a
lead-in and tails around each segment: QA time ``t`` is source time
``entry.start + t`` and the entry's ``id`` names the segment. Problems inside the
segment are repaired as above; a blocking span in the lead-in or tail only caps it
(``min_start`` / ``max_end`` on the segment, honoured by ``cutpoint clips``). When
the rendered edit runs past ``mix.QA_LENGTH_S`` the longest extensions are capped
the same way, social tails without speech first.
"""

from __future__ import annotations

import logging
import math
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from cutpoint import cache, mix, ski
from cutpoint.clips import BOUND_KEYS, index_sources, order_moments
from cutpoint.config import path_key, source_path
from cutpoint.crash import load_crashes
from cutpoint.pano.qa import runs_of

logger = logging.getLogger(__name__)

BLOCK_RUN_S = 0.5  # occluded / too-close / dark run that blocks a clip
MIN_PEOPLE = 0.5  # share of a social clip's frames that must show a person
IMPACT_S = 3.0  # a crash's flagged frames from the impact this long on are the content
PRE_IMPACT_S = 1.5  # ... and from this long before it (the camera knocked in the tumble)
CRASH_LEAD_S = 2.0  # a crash keeps this much approach: blocked spans are cut back from it
BLOCKED = ("occluded", "too_close", "dark")
EPS = 0.05  # s: rendered this far past the segment is lead-in / tail
REACH_S = 15.0  # s: the most a lead-in / tail adds (event tails reach 14 s)


def _dt(frames: list[dict[str, Any]], dur: float) -> float:
    return frames[1]["t"] - frames[0]["t"] if len(frames) > 1 else dur


def problem(
    frames: list[dict[str, Any]],
    dt: float,
    category: str | None,
    core: tuple[float, float] = (-math.inf, math.inf),
) -> str | None:
    """Why a clip (``qa.json`` frames, ``dt`` s apart) needs repair, or None. The
    people share only counts frames in ``core`` (the segment, in clip time)."""
    blocked = [any(f in fr["flags"] for f in BLOCKED) for fr in frames]
    run = max((b - a for a, b in runs_of(blocked)), default=0) * dt
    if run >= BLOCK_RUN_S - 1e-6:
        return f"occluded/close/dark run {run:.1f}s"
    inside = [fr for fr in frames if core[0] - 1e-6 <= fr["t"] < core[1]]
    people = sum(fr["subj"] is not None for fr in inside) / max(1, len(inside))
    if category == "social" and people < MIN_PEOPLE:
        return f"person in {people:.0%} of frames"
    return None


def _spans(
    frames: list[dict[str, Any]], dt: float, dur: float, bad: Callable[[dict[str, Any]], bool]
) -> list[tuple[float, float]]:
    """Clip-time ``(t0, t1)`` runs of frames where ``bad(frame)``, each frame standing
    for ``[t - dt / 2, t + dt / 2)``."""
    mask = [bad(fr) for fr in frames]
    return [
        (max(0.0, frames[a]["t"] - dt / 2), min(dur, frames[b - 1]["t"] + dt / 2))
        for a, b in runs_of(mask)
    ]


def _lead(t0: float, t1: float, hits: list[float]) -> tuple[float, float] | None:
    """Blocked span ``t0``-``t1`` ending before it reaches the last :data:`CRASH_LEAD_S`
    before an impact in ``hits`` (the crash's approach), or None if nothing is left."""
    for h in hits:
        if h - CRASH_LEAD_S < t1 <= h:
            t1 = h - CRASH_LEAD_S
    return (t0, t1) if t1 - t0 > EPS else None


def impact_time(crash: dict[str, Any]) -> float:
    """A ``crashes.json`` entry's impact (source s): a telemetry shock span starts at it,
    a spoken-cue window (``origin`` ``words``) is the impact +- 1 s."""
    return crash["t1"] - 1.0 if crash.get("origin") == "words" else crash["t0"]


def _match(
    segments: list[dict[str, Any]], qa: dict[str, Any], src_dir: Path, qa_time: float
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], float]], int]:
    """``(QA clip, segment, source time of its first frame)`` per usable clip, and the
    number skipped.

    Rendered clips (``manifest.json`` in ``src_dir``): the entry's ``id``, its pick
    inside the segment. Director previews ``NNN_<label>``: segment ``NNN`` of the list,
    checked against the path's ``segment``. Stale clips (no match, missing or newer
    than the QA at ``qa_time``) are skipped.
    """
    manifest = cache.load_json(src_dir / "manifest.json")
    entries = {e["file"]: e for e in (manifest or {}).get("clips", [])}
    by_id = {s["id"]: s for s in segments}
    out, skipped = [], 0
    for clip in qa.get("clips", []):
        if manifest is not None:
            e = entries.get(clip["file"]) or {}
            seg = by_id.get(e.get("id"))
            ok = (
                seg is not None
                and ski._key(seg["source"]) == ski._key(e["source"])
                and seg["start"] - 0.01 <= e["pick"]["start"]
                and e["original_end"] <= seg["end"] + 0.01
            )
            first = float(e.get("start", 0.0))
        else:
            stem = Path(clip["file"]).stem
            i = int(stem[:3]) - 1 if stem[:3].isdigit() else -1
            seg = segments[i] if 0 <= i < len(segments) else None
            span = (cache.load_json(src_dir / f"{stem}.json") or {}).get("segment") or {}
            ok = seg is not None and abs(span.get("start", -1) - seg["start"]) <= 0.01
            first = seg["start"] if seg else 0.0
        if not ok:
            logger.warning("%s: no matching segment (stale clip?); skipped", clip["file"])
            skipped += 1
            continue
        video = src_dir / clip["file"]
        if not video.is_file() or video.stat().st_mtime > qa_time:
            logger.warning("%s: clip missing or newer than QA; re-run 360 qa", clip["file"])
            skipped += 1
            continue
        out.append((clip, seg, first))
    return out, skipped


def plan(
    segments: list[dict[str, Any]],
    qa: dict[str, Any],
    src_dir: Path,
    qa_time: float = float("inf"),
    impacts: Sequence[tuple[str, float]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, dict[str, dict[str, float]]]:
    """``(exclusions, findings, skipped, bounds)`` for the blocking problems in ``qa``.

    QA clips are matched by :func:`_match`. Occluded / too-close / dark spans inside a
    segment exclude every category, low-information and no-person spans only the
    clip's own; blocking spans in a rendered lead-in / tail become ``bounds``
    (``{id: {"min_start" | "max_end": source s}}``). Crash clips ignore flags from
    :data:`PRE_IMPACT_S` before an ``impacts`` ``(source, t)`` to :data:`IMPACT_S` after
    it; blocked spans ending within :data:`CRASH_LEAD_S` before it are cut back to end
    there (the approach stays).
    """
    matched, skipped = _match(segments, qa, src_dir, qa_time)
    excl, findings, bounds = [], [], {}
    for clip, seg, a in matched:
        cat, dur, lo, hi = seg.get("category"), clip["duration"], seg["start"], seg["end"]
        frames = clip["frames"]
        hits = [t - a for src, t in impacts if path_key(src) == path_key(seg["source"])]
        hits = hits if cat == "crash" else []  # impacts in clip time
        frames = [
            {**fr, "flags": []}
            if any(-PRE_IMPACT_S <= fr["t"] - h < IMPACT_S for h in hits)
            else fr
            for fr in frames
        ]
        dt = _dt(frames, dur)
        why = problem(frames, dt, cat, (lo - a, hi - a))
        if why is None:
            continue
        findings.append({"id": seg["id"], "file": clip["file"], "category": cat, "problem": why})
        inside = [fr for fr in frames if lo - a - 1e-6 <= fr["t"] < hi - a]
        core_why = problem(inside, dt, cat)  # weak frames only count for a core problem
        people = bool(core_why) and core_why.startswith("person")

        def blocked(fr: dict[str, Any]) -> bool:
            return any(f in fr["flags"] for f in BLOCKED)

        def weak(fr: dict[str, Any], people: bool = people) -> bool:  # bad for this category
            return not blocked(fr) and (
                "low_info" in fr["flags"] or (people and fr["subj"] is None)
            )

        kept = (_lead(t0, t1, hits) for t0, t1 in _spans(frames, dt, dur, blocked))
        spans = [((a + t0, a + t1), None) for t0, t1 in filter(None, kept)]
        for (s0, s1), _ in spans:
            if s0 < lo - EPS:  # in the lead-in: start after it
                b = bounds.setdefault(seg["id"], {})
                b["min_start"] = round(max(b.get("min_start", -math.inf), s1), 2)
            if s1 > hi + EPS:  # in the tail: end before it
                b = bounds.setdefault(seg["id"], {})
                b["max_end"] = round(min(b.get("max_end", math.inf), s0), 2)
        if core_why:
            spans += [((a + t0, a + t1), cat) for t0, t1 in _spans(frames, dt, dur, weak)]
        for (s0, s1), only in spans:
            s0, s1 = max(s0, lo), min(s1, hi)  # inside the segment: excluded
            if s1 - s0 < EPS:
                continue
            excl.append(
                {
                    "source": seg["source"],
                    "start": round(s0, 2),
                    "end": round(s1, 2),
                    **({"category": only} if only else {}),
                    "reason": f"{seg['id']}: {why}",
                }
            )
    return excl, findings, skipped, bounds


def _merge(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(e: dict[str, Any]) -> tuple:
        return path_key(str(e["source"])), e["start"], e["end"], e.get("category")

    seen = {key(e) for e in old}
    return old + [e for e in new if key(e) not in seen]


def _bound(segs: list[dict[str, Any]], bounds: dict[str, dict[str, float]]) -> None:
    """Tighten each segment's ``min_start`` / ``max_end`` by ``bounds`` (in place),
    never into the segment."""
    for s in segs:
        b = bounds.get(s["id"], {})
        if "min_start" in b:
            s["min_start"] = round(
                min(max(s.get("min_start", -math.inf), b["min_start"]), s["start"]), 2
            )
        if "max_end" in b:
            s["max_end"] = round(max(min(s.get("max_end", math.inf), b["max_end"]), s["end"]), 2)


def _near(segs: list[dict[str, Any]], excl: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Bounds at the nearest exclusion (global or of the segment's category) within
    :data:`REACH_S` before / after each segment, so lead-in and tails stay out of it."""
    out: dict[str, dict[str, float]] = {}
    for s in segs:
        for e in excl:
            if ski._key(e["source"]) != ski._key(s["source"]) or e.get("category") not in (
                None,
                s.get("category"),
            ):
                continue
            if s["end"] - 0.01 <= e["start"] < s["end"] + REACH_S:
                b = out.setdefault(s["id"], {})
                b["max_end"] = min(b.get("max_end", math.inf), e["start"])
            if s["start"] - REACH_S < e["end"] <= s["start"] + 0.01:
                b = out.setdefault(s["id"], {})
                b["min_start"] = max(b.get("min_start", -math.inf), e["end"])
    return out


def _cap_level(xs: list[float], excess: float) -> float:
    """The level that capping ``xs`` at removes ``excess`` seconds (0 if it cannot)."""
    xs = sorted(xs, reverse=True)
    for k in range(1, len(xs) + 1):
        nxt = xs[k] if k < len(xs) else 0.0
        if sum(xs[:k]) - k * nxt >= excess:
            return max(nxt, (sum(xs[:k]) - excess) / k)
    return 0.0


def _rendered(
    segs: list[dict[str, Any]], manifest: dict[str, Any]
) -> tuple[dict[str, float], dict[tuple[str, str], float], set[tuple[str, str]]]:
    """``(core s by id, lead-in / tail s by (id, side), social tails without speech)``.

    Lead-in and tail are the manifest entry's (the clip as rendered; negative when the
    speech pass moved a boundary into the pick), within the segment's bounds; a segment
    without an entry (new) counts its own length.
    """
    by_id = {e["id"]: e for e in manifest.get("clips", [])}
    core, ext, quiet = {}, {}, set()
    for s in segs:
        e = by_id.get(s["id"])
        core[s["id"]] = s["end"] - s["start"]
        if e is None:
            continue
        core[s["id"]] = min(core[s["id"]], e["original_end"] - e["pick"]["start"])
        lead = e["pick"]["start"] - e["start"]
        tail = e["end"] - e["original_end"]
        ext[s["id"], "lead"] = min(lead, s["start"] - s.get("min_start", -math.inf))
        ext[s["id"], "tail"] = min(tail, s.get("max_end", math.inf) - s["end"])
        words = (e.get("tail") or {}).get("included_words") or (
            (e.get("speech") or {}).get("end") or {}
        ).get("words")
        if s.get("category") == "social" and not words:
            quiet.add((s["id"], "tail"))
    return core, ext, quiet


def _fit_budget(segs: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, float]:
    """Cap lead-ins / tails (bounds, in place) until the rendered edit fits
    ``mix.QA_LENGTH_S``: social tails without speech first, then the longest of all.

    Returns:
        The estimated rendered length per segment id.
    """
    core, ext, quiet = _rendered(segs, manifest)
    over = sum(core.values()) + sum(ext.values()) - mix.QA_LENGTH_S[1]
    cut: dict[tuple[str, str], float] = {}
    for pool in (quiet, set(ext)):
        if over <= 0.05:
            break
        xs = {k: ext[k] for k in pool if ext[k] > 0}
        level = _cap_level(list(xs.values()), over)
        for k, x in xs.items():
            if x > level:
                over -= x - level
                ext[k] = cut[k] = level
    by_id = {s["id"]: s for s in segs}
    caps: dict[str, dict[str, float]] = {}
    for (i, side), x in cut.items():
        s = by_id[i]
        key, at = ("min_start", s["start"] - x) if side == "lead" else ("max_end", s["end"] + x)
        caps.setdefault(i, {})[key] = at
    _bound(segs, caps)
    return {
        i: round(c + ext.get((i, "lead"), 0) + ext.get((i, "tail"), 0), 2) for i, c in core.items()
    }


def repair(
    project: Path, segments_file: Path, qa_file: Path, out: Path | None = None, sheet: bool = True
) -> dict[str, Any]:
    """Repair ``segments_file`` from ``qa_file`` (``<dir>_qa/qa.json`` of the director
    previews or of the rendered clips).

    Writes the repaired list to ``out`` (default: over ``segments_file``, the old one
    kept as ``<name>.bak.json``); when clips are cut or dropped also ``mix_report.json``,
    ``mix_sheet.jpg`` and the merged ``mix_exclude.json`` next to it. Nothing is written
    when nothing needs repair.

    Returns:
        ``{"segments", "written", "findings", "skipped", "trimmed", "dropped", "added",
        "capped", "exclusions", "rendered_s", "content_qa"}``; ``rendered_s`` (rendered
        clips only) is ``{"before", "after", "limit"}`` and the content QA then uses the
        rendered lengths.
    """
    old = cache.load_json(segments_file)
    qa = cache.load_json(qa_file)
    if not isinstance(old, list) or not old:
        raise ValueError(f"{segments_file}: not a segments list")
    if qa is None:
        raise FileNotFoundError(f"QA report not found: {qa_file}")
    src_dir = Path(qa.get("source_dir") or "")
    if not qa.get("source_dir") or not src_dir.is_dir():
        src_dir = qa_file.parent.with_name(qa_file.parent.name.removesuffix("_qa"))
    crashes = (load_crashes(project) or {}).get("crashes", [])
    impacts = [(c["source"], impact_time(c)) for c in crashes]
    new_excl, findings, skipped, bounds = plan(old, qa, src_dir, qa_file.stat().st_mtime, impacts)
    manifest = cache.load_json(src_dir / "manifest.json")
    out = out or segments_file
    report = cache.load_json(segments_file.with_name("mix_report.json")) or {}
    rep: dict[str, Any] = {
        "segments": str(out),
        "written": False,
        "findings": findings,
        "skipped": skipped,
        "trimmed": [],
        "dropped": [],
        "added": [],
        "capped": [],
        "exclusions": 0,
        "rendered_s": None,
        "content_qa": None,
    }
    new = [dict(s) for s in old]
    if new_excl:
        written = out.with_name(mix.EXCLUDE_FILE)
        have = written if written.is_file() else project / mix.EXCLUDE_FILE
        excl = _merge(mix.load_exclusions(have), new_excl)
        # Back up only once run_mix has written over the original: a failed selection
        # ("Nothing selected") leaves no .bak.json and an earlier one untouched.
        original = segments_file.read_bytes() if out == segments_file else None
        try:
            report = mix.run_mix(
                project,
                target_s=report.get("target_s", 240.0),
                quotas=report.get("quotas"),
                out=out,
                sheet=sheet,
                exclude=excl,
                previous=old,
            )
        except BaseException:
            try:  # failed after writing (the sheet): keep the old
                if original is not None and segments_file.read_bytes() != original:
                    _backup(segments_file, original)
            except OSError:
                logger.warning("%s: could not write the backup", segments_file)
            raise
        if original is not None:
            _backup(segments_file, original)
        cache.save_json(written, excl)
        new = cache.load_json(out) or []
        rep["exclusions"], rep["content_qa"] = len(excl), report["content_qa"]
        _bound(new, _near(new, excl))
    _bound(new, bounds)
    if manifest is not None:
        est = _fit_budget(new, manifest)
        before = sum(e["duration"] for e in manifest.get("clips", []))
        rep["rendered_s"] = {
            "before": round(before, 1),
            "after": round(sum(est.values()), 1),
            "limit": mix.QA_LENGTH_S[1],
        }
        rep["content_qa"] = mix.content_qa(
            order_moments(index_sources([dict(s) for s in new]), "arc"),
            near_dups=report.get("near_dup_pairs", 0),
            runs_covered=report.get("runs_covered", 0),
            runs=report.get("runs", 0),
            durations=est,
        )
    _changes(rep, old, new)
    if not new_excl and new == old:
        return rep
    if not new_excl and out == segments_file:
        shutil.copyfile(segments_file, segments_file.with_suffix(".bak.json"))
    cache.save_json(out, new)
    rep["written"] = True
    return rep


def _backup(segments_file: Path, original: bytes) -> None:
    """Write ``original`` to ``<name>.bak.json``; if that fails, put it back into
    ``segments_file`` (overwritten by then) and re-raise."""
    try:
        segments_file.with_suffix(".bak.json").write_bytes(original)
    except OSError:
        segments_file.write_bytes(original)
        raise


def _changes(rep: dict[str, Any], old: list[dict[str, Any]], new: list[dict[str, Any]]) -> None:
    """Fill ``rep``'s trimmed / dropped / added / capped lists."""
    by_id = {s["id"]: s for s in new}
    for s in old:
        n = by_id.get(s["id"])
        if n is None:
            rep["dropped"].append(_brief(s))
            continue
        if (n["start"], n["end"]) != (s["start"], s["end"]):
            rep["trimmed"].append({**_brief(n), "was": [s["start"], s["end"]]})
        if any(n.get(k) != s.get(k) for k in BOUND_KEYS):
            rep["capped"].append({**_brief(n), **{k: n[k] for k in BOUND_KEYS if k in n}})
    ids = {s["id"] for s in old}
    rep["added"] = [_brief(s) for s in new if s["id"] not in ids]


def _brief(s: dict[str, Any]) -> dict[str, Any]:
    return {k: s[k] for k in ("id", "category", "source", "start", "end")}


def log_repair(rep: dict[str, Any]) -> None:
    """Findings, then what was trimmed, dropped, added and capped, and the content QA."""
    for f in rep["findings"]:
        logger.info("%s %-16s %-8s %s", f["id"], f["file"], f["category"], f["problem"])
    if rep["skipped"]:
        logger.info("%d clip(s) skipped as stale; re-run 360 director/clips and qa", rep["skipped"])
    for what in ("trimmed", "dropped", "added", "capped"):
        for s in rep[what]:
            was = f" (was {s['was'][0]:.1f}-{s['was'][1]:.1f})" if "was" in s else ""
            lim = "".join(f" {k} {s[k]:.2f}" for k in BOUND_KEYS if k in s)
            logger.info(
                "%-7s %s %-8s %s %.1f-%.1f%s%s",
                what,
                s["id"],
                s["category"],
                source_path(s["source"]).name,
                s["start"],
                s["end"],
                was,
                lim,
            )
    r = rep["rendered_s"]
    if r:
        logger.info(
            "rendered %.1fs -> about %.1fs (limit %.0fs)%s",
            r["before"],
            r["after"],
            r["limit"],
            " OVER" if r["after"] > r["limit"] else "",
        )
    if not rep["written"]:
        if not rep["skipped"]:
            logger.info("nothing to repair")
        return
    qa = rep["content_qa"]
    failed = [k for k, v in qa["checks"].items() if not v["ok"]] if qa else []
    logger.info("content QA: %s", f"FAIL ({', '.join(failed)})" if failed else "pass")
    logger.info("wrote %s (%d exclusions)", rep["segments"], rep["exclusions"])
