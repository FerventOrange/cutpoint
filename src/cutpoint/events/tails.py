"""Event-aware clip tails: keep the payoff that lands after the dialogue.

Banter clips are cut where the speech ends, but the payoff (a kill, a
round-win banner, a burst of laughter) often comes several seconds later,
after a stretch with no speech that the audio-level smart tail
(:mod:`cutpoint.tail`) cannot distinguish from "nothing happens".

:func:`event_tail` uses ``events.json`` instead: if a *payoff event* starts
within ``window`` seconds after the clip's speech ends (or is still running
at the clip end), the end moves to the event's end plus ``pad`` seconds of
reaction, never further than ``max_ext`` past the speech and never past
``limit``.

Payoff events:
    * audio: Laughter, Scream, Cheering;
    * OCR: kill-feed entries involving the player or a teamkill, and round
      banners (WON/LOST ROUND, VICTORY, DEFEAT, ACE, CLUTCH, REVERSE FRIENDLY
      FIRE); "MATCH POINT"/"OVERTIME"/"x VS y" are context, not payoffs;
    * VLM: moments with funny or hype >= ``vlm_min``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cutpoint import cache

AUDIO_PAYOFFS = {"Laughter", "Scream", "Cheering"}
OCR_HOLD_S = 1.0  # an OCR payoff counts as lasting this long from its first sighting
BANNER_PAYOFFS = (
    "WON ROUND",
    "ROUND WON",
    "LOST ROUND",
    "ROUND LOST",
    "VICTORY",
    "DEFEAT",
    "ACE",
    "CLUTCH",
    "REVERSE FRIENDLY FIRE",
    "TARGET DESTROYED",
    "SHOT DOWN",
    "MISSION ACCOMPLISHED",
    # generic / War Thunder (never produced by the R6 preset)
    "YOU HAVE BEEN DESTROYED",
    "CREW KNOCKED OUT",
    "MISSION FAILED",
    "MISSION COMPLETE",
    "YOU DIED",
    "DOUBLE KILL",
    "TRIPLE KILL",
    "MULTI KILL",
)


@dataclass(frozen=True)
class EventTailSettings:
    """Parameters of :func:`event_tail`.

    Attributes:
        enabled: Apply event tails at all.
        window: Payoff must start at most this long after the speech ends (s).
        pad: Reaction kept after the payoff event ends (s).
        max_ext: Maximum end position after the speech end (s).
        vlm_min: Minimum funny/hype score of a VLM moment to count.
        audio_min: Minimum PANNs confidence of an audio event to count.
    """

    enabled: bool = True
    window: float = 10.0
    pad: float = 1.5
    max_ext: float = 14.0
    vlm_min: int = 6
    audio_min: float = 0.05

    def to_json(self) -> dict[str, Any]:
        """Dict for manifest.json."""
        return asdict(self)


def load_events(project: Path) -> dict[str, list[dict[str, Any]]] | None:
    """Events of ``<project>/events.json`` grouped by source path (None if absent)."""
    data = cache.load_json(project / "events.json")
    if not data:
        return None
    out: dict[str, list[dict[str, Any]]] = {}
    for e in data.get("events", []):
        out.setdefault(str(Path(e["source"])), []).append(e)
    for evs in out.values():
        evs.sort(key=lambda e: e["start"])
    return out


def is_payoff(e: dict[str, Any], settings: EventTailSettings) -> bool:
    """Whether an events.json entry is a payoff worth extending a clip for."""
    kind = e.get("kind")
    if kind == "audio":
        return e.get("label") in AUDIO_PAYOFFS and (e.get("confidence") or 0) >= settings.audio_min
    if kind == "ocr":
        if e.get("type") == "kill":
            d = e.get("details") or {}
            return bool(d.get("player") or d.get("teamkill"))
        if e.get("type") == "banner":
            label = (e.get("label") or "").upper()
            return any(label.startswith(b) for b in BANNER_PAYOFFS)
        if e.get("type") == "log":
            return bool((e.get("details") or {}).get("player"))
        return e.get("type") == "multikill"
    if kind == "vlm":
        return max(e.get("funny") or 0, e.get("hype") or 0) >= settings.vlm_min
    if kind == "telemetry":  # ski: a jump or crash right after the talk
        return e.get("type") in ("airtime", "crash")
    return False


def describe(e: dict[str, Any]) -> str:
    """Short text for logs and the manifest, e.g. ``OCR kill: A > B @391.0``."""
    kind = (e.get("kind") or "?").upper()
    typ = f" {e['type']}" if e.get("kind") == "ocr" else ""
    return f"{kind}{typ}: {e.get('label', '')[:60]} @{e['start']:.1f}"


def speech_end(start: float, end: float, words: list[dict[str, Any]] | None) -> float:
    """End of the last word inside ``[start, end]`` (``end`` if there is none)."""
    last = None
    for w in words or []:
        if w["start"] < end - 0.05 and w["end"] > start:
            last = w["end"] if last is None else max(last, w["end"])
    return min(end, last) if last is not None else end


def event_tail(
    start: float,
    end: float,
    words: list[dict[str, Any]] | None,
    events: list[dict[str, Any]] | None,
    settings: EventTailSettings,
    limit: float,
) -> tuple[float, str | None, list[str]]:
    """New clip end that covers a payoff event after the speech.

    Args:
        start: Clip start (absolute source time).
        end: Current clip end (after the smart tail).
        words: Transcript words of the source (absolute time), or None.
        events: events.json entries of this source.
        settings: Event-tail parameters.
        limit: Hard upper bound (source end, unreadable region, next clip).

    Returns:
        (new end >= ``end``, reason or None if unchanged, payoff descriptions).
    """
    if not settings.enabled or not events:
        return end, None, []
    ref = speech_end(start, end, words)
    cap = min(limit, ref + settings.max_ext)
    target, used = end, []
    for e in events:
        if e["start"] > ref + settings.window:
            break
        if not is_payoff(e, settings):
            continue
        e_end = max(e["start"], e.get("end") or e["start"])
        if e.get("kind") == "ocr":
            # An OCR event's end is how long the banner/kill row stays on screen,
            # not the reaction; the reaction is covered by ``pad``.
            e_end = min(e_end, e["start"] + OCR_HOLD_S)
        starts_after = ref - 0.5 <= e["start"]
        runs_past = e["start"] <= end < e_end  # still going at the cut
        if not (starts_after or runs_past) or e_end + settings.pad <= end:
            continue
        if e["start"] >= cap:
            continue
        target = max(target, min(cap, e_end + settings.pad))
        used.append(describe(e))
    if target <= end + 0.05:
        return end, None, []
    return round(target, 3), f"event payoff ({used[-1]})", used
