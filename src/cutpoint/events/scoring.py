"""Feed events.json back into highlights.json candidate scores.

Each candidate keeps its analysis score as ``score_pre_events``; the new
``score`` adds event boosts for ``[start, end + lookahead]`` (so a payoff
right after the banter counts for the banter clip) and blends in the best
overlapping VLM window (boost-only: it can raise but never lower a score).
Re-running is idempotent because the boost is always
computed from ``score_pre_events``. Candidate ids are kept; the list is
re-sorted by the new score and ``suggested_top`` is recomputed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cutpoint import cache
from cutpoint.events.tails import BANNER_PAYOFFS

logger = logging.getLogger(__name__)

LOOKAHEAD_S = 10.0  # events this long after a candidate still count for it
VLM_WEIGHT = 0.35  # share of the final score taken from the VLM window score
MAX_BOOST = 30.0


def event_boost(cand: dict[str, Any], events: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    """Additive boost (points on the 0-100 scale) and the evidence behind it."""
    a, b = cand["start"], cand["end"] + LOOKAHEAD_S
    laugh = scream = 0.0
    kills = teamkill = banners = 0
    for e in events:
        if e["end"] < a or e["start"] > b:
            continue
        if e["kind"] == "audio":
            if e["label"] == "Laughter":
                laugh = max(laugh, e["confidence"] or 0)
            elif e["label"] in ("Scream", "Cheering"):
                scream = max(scream, e["confidence"] or 0)
        elif e["kind"] == "ocr":
            d = e.get("details") or {}
            label = (e.get("label") or "").upper()
            if e["type"] == "kill" and d.get("teamkill"):
                teamkill += 1
            elif e["type"] == "kill" and d.get("player"):
                kills += 1
            elif e["type"] == "banner" and "REVERSE FRIENDLY FIRE" in label:
                teamkill += 1
            elif (e["type"] == "banner" and any(label.startswith(x) for x in BANNER_PAYOFFS)) or e[
                "type"
            ] == "multikill":
                banners += 1
    boost = (
        min(12.0, 60 * laugh)
        + min(6.0, 40 * scream)
        + 6.0 * min(kills, 3)
        + 12.0 * min(teamkill, 1)
        + 6.0 * min(banners, 2)
    )
    evidence = {
        "laughter": round(laugh, 3),
        "scream": round(scream, 3),
        "player_kills": kills,
        "teamkills": teamkill,
        "banners": banners,
    }
    return min(MAX_BOOST, boost), evidence


def best_window(cand: dict[str, Any], windows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The VLM window overlapping the candidate the most (None if none)."""
    best, best_ov = None, 0.0
    for w in windows:
        if w["source"] != cand["source"]:
            continue
        ov = min(w["end"], cand["end"] + LOOKAHEAD_S / 2) - max(w["start"], cand["start"])
        if ov > best_ov:
            best, best_ov = w, ov
    return best if best_ov >= 2.0 else None


def rescore_highlights(project: Path) -> int:
    """Rescore ``<project>/highlights.json`` from ``events.json``.

    Returns:
        Number of candidates whose score changed (0 if either file is missing).
    """
    hl = cache.load_json(project / "highlights.json")
    ev = cache.load_json(project / "events.json")
    if not hl or not ev:
        return 0
    backup = project / "highlights.pre-events.json"
    if not backup.exists() and not hl.get("settings", {}).get("events_rescored"):
        cache.save_json(backup, hl)  # one-time copy of the untouched analysis
    by_src: dict[str, list[dict[str, Any]]] = {}
    for e in ev.get("events", []):
        by_src.setdefault(e["source"], []).append(e)
    windows = ev.get("windows", [])
    changed = 0
    for c in hl.get("candidates", []):
        base = c.get("score_pre_events", c["score"])
        boost, evidence = event_boost(c, by_src.get(c["source"], []))
        score = base + boost
        win = best_window(c, windows)
        if win:
            vlm = max(win.get("highlight_score", 0), win.get("funny_score", 0)) * 10
            # Boost-only: Omni-7B often scores real action 0, so it may raise a
            # candidate toward its window score but never pull it down.
            score += VLM_WEIGHT * max(0.0, vlm - score)
            evidence["vlm"] = {
                "highlight": win.get("highlight_score"),
                "funny": win.get("funny_score"),
                "label": win.get("label"),
                "window": [win["start"], win["end"]],
            }
        score = round(min(100.0, score), 1)
        c["score_pre_events"] = base
        c.setdefault("signals", {})["events"] = {"boost": round(boost, 1), **evidence}
        if score != c["score"]:
            changed += 1
        c["score"] = score
    hl["candidates"].sort(key=lambda c: -c["score"])
    target = (hl.get("settings") or {}).get("target_total")
    if target:
        total, n = 0.0, 0
        for c in hl["candidates"]:
            if total >= target:
                break
            total += c["end"] - c["start"]
            n += 1
        hl["suggested_top"] = n
    hl.setdefault("settings", {})["events_rescored"] = ev.get("created")
    cache.save_json(project / "highlights.json", hl)
    logger.info(
        "Rescored %d of %d candidates from events.json", changed, len(hl.get("candidates", []))
    )
    return changed
