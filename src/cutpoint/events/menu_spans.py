"""Menu time in highlight candidates.

The OCR layer flags menu and scoreboard frames (:func:`cutpoint.events.killfeed.menu_times`)
and records them per source as ``menu_spans`` in its stats (``events.json``:
``stats.ocr.<source name>.menu_spans``, ``[start, end]`` in source seconds). A candidate
that lies mostly in menu time is dropped; one that starts or ends in a menu span is
trimmed to the gameplay part, when what is left is long enough for the style
(:func:`drop_menu_time`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cutpoint import sources
from cutpoint.styles import LAUGHTER, get_style
from cutpoint.utils import slugify

logger = logging.getLogger(__name__)

MENU_DROP_SHARE = 0.5  # a candidate with at least this share of menu time is dropped


def merge_menu_frames(times: Iterable[float], fps: float) -> list[list[float]]:
    """The menu frames at ``times`` (seconds) of a read at ``fps`` as ``[start, end]`` spans:
    frames one read interval apart join, and each frame lasts one interval."""
    step = 1.0 / fps
    spans: list[list[float]] = []
    last = None
    for t in sorted(times):
        if last is not None and t - last <= 1.5 * step:
            spans[-1][1] = t + step
        else:
            spans.append([t, t + step])
        last = t
    return [[round(a, 2), round(b, 2)] for a, b in spans]


def source_menu_spans(ev: dict[str, Any]) -> dict[str, list[list[float]]]:
    """The menu spans of each source path of the events.json object ``ev`` that has some.
    The OCR stats are keyed by file name, so two sources with one name get none."""
    names: dict[str, list[str]] = {}
    for p in ev.get("sources") or []:
        names.setdefault(Path(p).name, []).append(p)
    out: dict[str, list[list[float]]] = {}
    for name, st in (((ev.get("stats") or {}).get("ocr")) or {}).items():
        paths = names.get(name, [])
        if len(paths) == 1 and isinstance(st, dict) and st.get("menu_spans"):
            out[paths[0]] = st["menu_spans"]
    return out


def _overlap(a: float, b: float, spans: list[list[float]]) -> float:
    """Seconds of ``[a, b]`` inside ``spans``."""
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in spans)


def _min_len(style: str | None) -> float:
    """The style's minimum candidate length; unknown styles are never trimmed."""
    try:
        return get_style(style or "").min_len
    except ValueError:
        return float("inf")


def _retranscribe(c: dict[str, Any], segments: list[dict[str, Any]] | None, style: str) -> None:
    """The transcript snippet of trimmed ``c`` from the cached ``segments`` (kept without).

    Its ``signals.keywords`` keep only the phrases still said in it (matched as analyze
    matches them), and a label named after a dropped one becomes what analyze gives the
    trimmed span: its strongest remaining keyword, else the label without one.
    """
    if segments is None:
        return
    from cutpoint.analyze import PROFANE, _transcript_snippet, keyword_hits  # noqa: PLC0415

    said = [w for s in segments for w in s.get("words", []) if c["start"] <= w["start"] <= c["end"]]
    c["transcript"], unsure = _transcript_snippet(said)
    c.pop("transcript_unsure", None)
    if unsure:
        c["transcript_unsure"] = unsure
    signals = c.get("signals") or {}
    if not signals.get("keywords"):
        return
    st = get_style(style)
    hits = sorted(
        keyword_hits(said, st.keywords) + keyword_hits(said, LAUGHTER), key=lambda h: -h[1]
    )
    still = {p for *_, p in hits}
    dropped = [k for k in signals["keywords"] if k not in still]
    if not dropped:
        return
    signals["keywords"] = [k for k in signals["keywords"] if k in still]
    if c.get("label") not in {slugify(k, 30) for k in dropped}:
        return
    strong = [p for _, w, p in hits if w >= 0.5 and p not in PROFANE and p in signals["keywords"]]
    if strong:
        c["label"] = slugify(strong[0], 30)
    elif signals.get("laughter", 0) > 0.5:
        c["label"] = "laugh"
    elif signals.get("loudness", 0) > 0.6:
        c["label"] = "loud-moment"
    else:
        c["label"] = f"{st.name}-moment"


def drop_menu_time(hl: dict[str, Any], ev: dict[str, Any]) -> None:
    """Drop and trim the candidates of the analysis ``hl`` in the menu spans of ``ev``.

    A candidate with at least :data:`MENU_DROP_SHARE` of its time in menu spans is
    dropped. Otherwise a start in a menu span moves to the span's end, then an end in
    one to its start, each only when at least the style's ``min_len`` is left; its
    ``duration``, transcript snippet (from the analysis's ``whisper_model``), keywords
    and a label named after a cut-off keyword follow (:func:`_retranscribe`). Menu time
    only in the middle stays.
    One INFO line per source with menu time in its candidates (counts and seconds).
    """
    by_src = source_menu_spans(ev)
    if not by_src:
        return
    min_len = _min_len(hl.get("style"))
    settings = hl.get("settings") or {}
    language, model = settings.get("language"), settings.get("whisper_model")
    segments: dict[str, list[dict[str, Any]] | None] = {}  # loaded once per source
    counts: dict[str, list[float]] = {}  # dropped, s, trimmed, s cut, kept, menu s kept
    kept = []
    for c in hl.get("candidates", []):
        spans = by_src.get(c["source"])
        menu = _overlap(c["start"], c["end"], spans) if spans else 0.0
        if not spans or menu <= 0:
            kept.append(c)
            continue
        n = counts.setdefault(c["source"], [0, 0.0, 0, 0.0, 0, 0.0])
        length = c["end"] - c["start"]
        if menu >= MENU_DROP_SHARE * length:
            n[0], n[1] = n[0] + 1, n[1] + length
            continue
        start, end = c["start"], c["end"]
        for x, y in spans:
            if x <= start < y and end - y >= min_len:
                start = y
        for x, y in spans:
            if x < end <= y and x - start >= min_len:
                end = x
        if (start, end) != (c["start"], c["end"]):
            n[2], n[3] = n[2] + 1, n[3] + length - (end - start)
            c["start"], c["end"] = start, end
            c["duration"] = round(end - start, 3)
            if c["source"] not in segments:
                try:
                    segments[c["source"]] = sources.source_segments(
                        Path(c["source"]), language, model
                    )
                except OSError as err:  # the source is gone (moved, drive disconnected)
                    logger.debug("No transcript for %s (%s)", c["source"], err)
                    segments[c["source"]] = None
            _retranscribe(c, segments[c["source"]], hl["style"])
        left = _overlap(start, end, spans)
        if left > 0:
            n[4], n[5] = n[4] + 1, n[5] + left
        kept.append(c)
    hl["candidates"] = kept
    for src, (dropped, drop_s, trimmed, cut_s, inside, inside_s) in counts.items():
        logger.info(
            "%s: menu time in candidates: %d dropped (%.1fs), %d trimmed (%.1fs cut), "
            "%d kept with %.1fs of menu time inside",
            Path(src).name,
            dropped,
            drop_s,
            trimmed,
            cut_s,
            inside,
            inside_s,
        )
