"""Write ``events.json`` and the human-readable ``events.txt`` of a project."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from cutpoint import cache
from cutpoint.config import source_path
from cutpoint.events.common import load_transcript_lines
from cutpoint.events.scoring import DIALOGUE_TYPE
from cutpoint.utils import clock


def _kill_flags(ev: dict[str, Any]) -> list[str]:
    """The set flags of a kill event, in display order."""
    return [
        f
        for f, on in (
            ("TEAMKILL", ev.get("teamkill")),
            ("player kill", ev.get("player_kill")),
            ("player death", ev.get("player_death")),
        )
        if on
    ]


def normalize(source: str, ev: dict[str, Any]) -> dict[str, Any]:
    """Convert a layer event into the events.json schema.

    Schema: ``{source, start, end, kind, type, label, description, confidence,
    funny, hype}`` plus layer-specific extras under ``details``.
    """
    kind = ev.get("kind", "audio")
    out: dict[str, Any] = {
        "source": source,
        "start": ev["t"],
        "end": ev.get("end", ev["t"]),
        "kind": kind,
        "type": ev.get("type", kind),
        "confidence": ev.get("confidence"),
        "funny": ev.get("funny"),
        "hype": ev.get("hype"),
    }
    if kind == "audio":
        out["label"] = ev["label"]
        out["description"] = (
            f"{ev['top_class']} peak {ev['confidence']:.2f} at "
            f"{clock(ev['peak_t'])}"
            + (" (relative to local baseline)" if ev.get("relative") else "")
        )
        out["details"] = {
            k: ev[k]
            for k in (
                "peak_t",
                "mean",
                "baseline",
                "top_class",
                "relative",
                "loud_jump",
                "backing",
                "confirmed",
            )
            if k in ev
        }
    elif kind == "ocr":
        out["label"] = ev.get("text", "")
        if ev.get("type") == "kill":
            flags = _kill_flags(ev)
            teams = f"{ev.get('killer_team') or '?'}>{ev.get('victim_team') or '?'}"
            out["description"] = (
                f"kill feed: {ev.get('killer')} killed {ev.get('victim')}"
                f" ({teams}{', ' + ', '.join(flags) if flags else ''})"
            )
        else:
            out["description"] = (
                f"{ev.get('type')} in {ev.get('region')}: {ev.get('raw', ev.get('text'))}"
            )
        out["details"] = {
            k: v
            for k, v in ev.items()
            if k not in ("t", "end", "kind", "type", "text", "confidence")
        }
    elif kind == "telemetry":
        out["label"] = ev.get("label", ev.get("type", ""))
        out["description"] = f"camera telemetry: {out['label']}"
        out["details"] = {
            k: v
            for k, v in ev.items()
            if k not in ("t", "end", "kind", "type", "label", "confidence")
        }
    else:
        out["label"] = ev.get("label", "")
        out["description"] = ev.get("label", "")
    return out


def _event_line(name: str, e: dict[str, Any]) -> str:
    t = f"[{name} {clock(e['start'])}]"
    if e["kind"] == "audio":
        conf = (e.get("details") or {}).get("confirmed")
        mark = "" if conf is None else (" confirmed" if conf else " unconfirmed")
        return (
            f"{t} AUDIO: {e['label']} ({e['confidence']:.2f}, {e['end'] - e['start']:.1f}s){mark}"
        )
    if e["kind"] == "ocr":
        if e["type"] == "kill":
            d = e.get("details", {})
            tags = _kill_flags(d)
            return f"{t} OCR kill: {e['label']}" + (f" ({', '.join(tags)})" if tags else "")
        detail = (e.get("details") or {}).get("detail")
        return f"{t} OCR: {e['label']}" + (f" ({detail})" if detail else "")
    if e["kind"] == "telemetry":
        return f"{t} TELEMETRY: {e['label']}"
    if e["kind"] == "text":
        length = e["end"] - e["start"]
        named = f'{e["label"]} "{e["title"]}"' if e.get("title") else e["label"]
        return f"{t} DIALOGUE {named} ({e['confidence']:.2f}, {length:.0f}s): {e['description']}"
    return f"{t} VLM: {e['label']} (funny {e.get('funny')}, hype {e.get('hype')})"


def write_outputs(
    project: Path,
    sources: list[str],
    events: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    transcript: list[dict[str, Any]],
    settings: dict[str, Any],
    timings: dict[str, Any],
    stats: dict[str, Any],
) -> tuple[Path, Path]:
    """Write ``events.json`` and ``events.txt``; returns their paths."""
    order = {s: i for i, s in enumerate(sources)}
    events = sorted(events, key=lambda e: (order.get(e["source"], 99), e["start"]))
    windows = sorted(windows, key=lambda w: (order.get(w["source"], 99), w["start"]))
    counts: dict[str, int] = {}
    for e in events:
        k = f"{e['kind']}:{e['label'] if e['kind'] == 'audio' else e['type']}"
        counts[k] = counts.get(k, 0) + 1
    data = {
        "version": 1,
        "created": datetime.now().isoformat(timespec="seconds"),
        "project": str(project),
        "settings": settings,
        "timings": timings,
        "stats": {**stats, "event_counts": dict(sorted(counts.items()))},
        "sources": sources,
        "events": events,
        "windows": [
            {
                k: w[k]
                for k in (
                    "source",
                    "start",
                    "end",
                    "label",
                    "summary",
                    "highlight_score",
                    "funny_score",
                    "tags",
                    "events",
                    "reasons",
                )
                if k in w
            }
            for w in windows
        ],
    }
    jpath = project / "events.json"
    cache.save_json(jpath, data)

    model = (settings.get("vlm") or {}).get("model") or "Vision model"
    lines = [
        "# events.txt: transcript lines interleaved with detected events.",
        "# AUDIO = PANNs sound events (confidence, duration); OCR = on-screen HUD text;",
        f"# VLM WINDOW = {model} summary of a window (hl = highlight 0-10);",
        "# VLM = moment inside a window. Times are source time.",
        "# DIALOGUE = talk a text model marked (kind, strength, duration).",
        "",
    ]
    for src in sources:
        name = source_path(src).name
        rows: list[tuple[float, int, str]] = []
        for ln in transcript:
            if ln["name"] == name:
                rows.append((ln["start"], 1, ln["line"]))
        for e in events:
            if e["source"] == src:
                rows.append((e["start"], 2, _event_line(name, e)))
        for w in windows:
            if w["source"] == src:
                tags = w.get("tags") or {}
                top = ", ".join(
                    f"{k} {v}" for k, v in sorted(tags.items(), key=lambda x: -x[1]) if v >= 5
                )
                rows.append(
                    (
                        w["start"] - 0.001,
                        0,
                        f"[{name} {clock(w['start'])}-{clock(w['end'])}] VLM WINDOW "
                        f"hl {w['highlight_score']} funny {w['funny_score']} "
                        f"'{w['label']}'" + (f" [{top}]" if top else "") + f": {w['summary']}",
                    )
                )
        if not rows:
            continue
        lines.append(f"# {src}")
        lines.extend(r[2] for r in sorted(rows, key=lambda r: (r[0], r[1])))
        lines.append("")
    tpath = project / "events.txt"
    # Text mode's line endings, as a plain write_text gave them; atomic, and held back with
    # events.json while `cutpoint analyze` runs (cache.hold_outputs)
    cache.save_bytes(tpath, "\n".join(lines).replace("\n", os.linesep).encode("utf-8"))
    return jpath, tpath


def is_dialogue(e: dict[str, Any]) -> bool:
    """Whether an events.json entry is a stretch of talk the dialogue reader marked."""
    return e.get("kind") == "text" and e.get("type") == DIALOGUE_TYPE


def kept_dialogue(project: Path, sources: list[str]) -> list[dict[str, Any]]:
    """The dialogue events of the events.json already in ``project`` for ``sources``: a run
    that does not read dialogue (``cutpoint events``, ``analyze --skip dialogue``) rewrites
    events.json and keeps them."""
    doc = cache.load_json(project / "events.json")
    found = doc.get("events") if isinstance(doc, dict) else None
    wanted = set(sources)
    return [
        e
        for e in found or []
        if isinstance(e, dict) and is_dialogue(e) and e.get("source") in wanted
    ]


def replace_dialogue(project: Path, new: list[dict[str, Any]], sources: list[str]) -> bool:
    """Replace the dialogue events of ``project``'s events.json by ``new`` (events.json
    entries) and rewrite events.json and events.txt; with no events.json yet (the event
    layers did not run) one is written with ``sources`` and these events only. Nothing is
    written when there were none and ``new`` has none. Returns whether it wrote.

    The transcript lines of events.txt are those of the analysis in progress, if any
    (:func:`cutpoint.events.common.pending_highlights`).
    """
    doc = cache.load_json(project / "events.json")
    doc = doc if isinstance(doc, dict) else None
    old = [e for e in (doc or {}).get("events", []) if is_dialogue(e)]
    if not old and not new:
        return False
    if doc is None:
        doc = {"settings": {"layers": []}, "sources": sources}
    events = [e for e in doc.get("events", []) if not is_dialogue(e)] + new
    write_outputs(
        project,
        list(doc.get("sources") or sources),
        events,
        doc.get("windows") or [],
        load_transcript_lines(project),
        doc.get("settings") or {},
        doc.get("timings") or {},
        {k: v for k, v in (doc.get("stats") or {}).items() if k != "event_counts"},
    )
    return True
