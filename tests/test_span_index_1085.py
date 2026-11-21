"""Audit #1085 A687: the indexed rescore gives the same results as the full scan (A706 in
test_contact_sheet.py's neighbours is a pure hoist)."""

from __future__ import annotations

import random
from typing import Any

from cutpoint.events import scoring
from cutpoint.events.scoring import LOOKAHEAD_S, SpanIndex, span_index_by_source

SOURCES = ("a.mp4", "b.mp4", "c.mp4")


def _brute_window(cand: dict[str, Any], windows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best, best_ov = None, 0.0
    for w in windows:
        if w["source"] != cand["source"]:
            continue
        ov = min(w["end"], cand["end"] + LOOKAHEAD_S / 2) - max(w["start"], cand["start"])
        if ov > best_ov:
            best, best_ov = w, ov
    return best if best_ov >= 2.0 else None


def _spans(rng: random.Random, n: int, kind: str) -> list[dict[str, Any]]:
    out = []
    for _ in range(n):
        start = float(rng.randint(0, 120)) if rng.random() < 0.6 else rng.uniform(0, 120)
        length = rng.choice([0.0, 1.0, 4.0, rng.uniform(0, 8), rng.uniform(30, 200)])
        e: dict[str, Any] = {
            "source": rng.choice(SOURCES),
            "start": start,
            "end": start + length,
            "kind": kind,
        }
        if kind == "ocr":
            e |= {"type": "bigtext", "text": "SIGNAL", "announce": True, "label": "x"}
        elif kind == "audio":
            e |= {
                "label": rng.choice(["Laughter", "Scream", "Speech"]),
                "confidence": rng.choice([0.0, 0.3, 0.8]),
            }
        else:
            e |= {"highlight_score": rng.randint(0, 9), "funny_score": rng.randint(0, 9)}
        out.append(e)
    return out


def _cands(rng: random.Random, n: int) -> list[dict[str, Any]]:
    out = []
    for _ in range(n):
        start = float(rng.randint(0, 120)) if rng.random() < 0.6 else rng.uniform(0, 120)
        out.append(
            {
                "source": rng.choice(SOURCES),
                "start": start,
                "end": start + rng.choice([0.0, 5.0, 20.0, rng.uniform(1, 60), 100.0]),
            }
        )
    return out


def test_best_window_with_index_equals_the_full_scan() -> None:
    rng = random.Random(1)
    for _ in range(40):
        windows = _spans(rng, rng.randint(0, 40), "window")
        index = span_index_by_source(windows)
        for cand in _cands(rng, 40):
            assert scoring.best_window(cand, windows, index) is _brute_window(cand, windows)
            assert scoring.best_window(cand, windows) is _brute_window(cand, windows)


def test_best_window_touching_edges_and_equal_overlaps_keep_the_first() -> None:
    cand = {"source": "a.mp4", "start": 10.0, "end": 20.0}
    windows = [
        {"source": "a.mp4", "start": 0.0, "end": 10.0},  # touches the start: no overlap
        {"source": "a.mp4", "start": 12.0, "end": 16.0},
        {"source": "b.mp4", "start": 10.0, "end": 30.0},
        {"source": "a.mp4", "start": 14.0, "end": 18.0},  # same overlap: the first stays
        {"source": "a.mp4", "start": 25.0, "end": 40.0},  # touches the lookahead edge
    ]
    index = span_index_by_source(windows)
    assert scoring.best_window(cand, windows, index) is windows[1]
    assert scoring.best_window({**cand, "source": "zzz"}, windows, index) is None


def test_event_boost_with_index_equals_the_full_scan() -> None:
    rng = random.Random(2)
    for _ in range(30):
        events = [
            *_spans(rng, rng.randint(0, 30), "ocr"),
            *_spans(rng, rng.randint(0, 30), "audio"),
        ]
        rng.shuffle(events)  # unsorted input
        by_src: dict[str, list[dict[str, Any]]] = {}
        for e in events:
            by_src.setdefault(e["source"], []).append(e)
        indexes = {src: SpanIndex(evs) for src, evs in by_src.items()}
        for cand in _cands(rng, 30):
            evs = by_src.get(cand["source"], [])
            want = scoring.event_boost(cand, evs, style="gaming")
            got = scoring.event_boost(cand, evs, style="gaming", index=indexes.get(cand["source"]))
            assert got == want


def test_event_boost_counts_events_in_the_lookahead_and_touching_the_start() -> None:
    cand = {"start": 10.0, "end": 20.0}
    events = [
        {"kind": "audio", "label": "Laughter", "confidence": 0.5, "start": 0.0, "end": 10.0},
        {"kind": "audio", "label": "Laughter", "confidence": 0.9, "start": 30.0, "end": 31.0},
        {"kind": "audio", "label": "Laughter", "confidence": 0.99, "start": 30.5, "end": 31.0},
    ]
    want = scoring.event_boost(cand, events, style="gaming")
    got = scoring.event_boost(cand, events, style="gaming", index=SpanIndex(events))
    assert got == want and want[1]["laughter"] == 0.9


def _brute_reaction(t: float, events: list[dict[str, Any]]) -> bool:
    end = t + scoring.REACTION_WINDOW_S
    return any(
        e.get("kind") == "audio"
        and e.get("label") in scoring.AFTER_EVENT_SOUNDS
        and t < e["start"] <= end
        for e in events
    )


def test_has_reaction_with_index_equals_the_full_scan() -> None:
    rng = random.Random(3)
    for _ in range(30):
        events = _spans(rng, rng.randint(0, 40), "audio") + _spans(rng, 10, "ocr")
        rng.shuffle(events)
        index = SpanIndex(events)
        times = [float(x) for x in range(0, 130, 3)] + [e["start"] for e in events]
        times += [e["start"] - scoring.REACTION_WINDOW_S for e in events]  # window edge
        for t in times:
            want = _brute_reaction(t, events)
            assert scoring.has_reaction(t, events) == want
            assert scoring.has_reaction(t, events, index=index) == want


def test_event_boost_reactions_with_index_equal_the_full_scan() -> None:
    rng = random.Random(4)
    for _ in range(20):
        events = _spans(rng, rng.randint(0, 40), "audio") + _spans(rng, 15, "ocr")
        for e in events:
            if e["kind"] == "ocr":
                e |= {
                    "type": "kill",
                    "label": "A > B",
                    "confidence": 0.9,
                    "details": {"player_death": True},
                }
        rng.shuffle(events)
        index = SpanIndex(events)
        for cand in _cands(rng, 20):
            for style in ("funny", "gaming"):
                want = scoring.event_boost(cand, events, style=style)
                got = scoring.event_boost(cand, events, style=style, index=index)
                assert got == want


def test_the_reaction_test_events_do_reach_a_reaction() -> None:
    events = [
        {
            "kind": "ocr",
            "type": "kill",
            "label": "A > B",
            "confidence": 0.9,
            "details": {"player_death": True},
            "start": 5.0,
            "end": 6.0,
        },
        {"kind": "audio", "label": "Laughter", "confidence": 0.5, "start": 7.0, "end": 8.0},
    ]
    cand = {"start": 0.0, "end": 20.0}
    got = scoring.event_boost(cand, events, style="funny", index=SpanIndex(events))
    assert got[1]["player_deaths_reaction"] == 1
    assert got == scoring.event_boost(cand, events, style="funny")
