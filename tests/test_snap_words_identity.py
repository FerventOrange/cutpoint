"""``analyze._snap`` and ``_snap_sentences`` bisect one index of the part's words and sentence
edges instead of scanning every word per candidate (#1085, A685): they must snap to the same
floats as the old scans, frozen here (synthetic transcripts, seeded)."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

import pytest

from cutpoint import analyze
from cutpoint.analyze import (
    SNAP_REACH_S,
    Candidate,
    _longest_pause,
    sentence_bounds,
)
from cutpoint.styles import get_style
from cutpoint.transcribe import flat_words

# ------------------------------------------------------------------ the old code, frozen


def _in_speech(t: float, segments: list[dict[str, Any]]) -> bool:
    return any(s["start"] - 0.3 <= t <= s["end"] + 0.3 for s in segments if s.get("text"))


def _gap_before(t: float, words: list[dict[str, Any]]) -> float:
    prev = [w["end"] for w in words if w["end"] <= t]
    return t - max(prev) if prev else 1.0


def _gap_after(t: float, words: list[dict[str, Any]]) -> float:
    nxt = [w["start"] for w in words if w["start"] >= t]
    return min(nxt) - t if nxt else 1.0


def _old_snap_sentences(c: Candidate, segments: list[dict[str, Any]], duration: float) -> None:
    words = flat_words(segments)
    starts, ends = sentence_bounds(segments)
    if _in_speech(c.start, segments):
        before = [t for t in starts if c.start - 12 <= t <= c.start]
        after = [t for t in starts if c.start < t <= c.start + 6]
        if before or after:
            t = max(before) if before else min(after)
        else:
            pause = _longest_pause(words, c.start - 12, c.start + 6)
            t = pause[1] if pause else c.start
        c.start = max(0.0, t - min(0.2, _gap_before(t, words) / 2))
    if _in_speech(c.end, segments):
        after = [t for t in ends if c.end <= t <= c.end + 15]
        before = [t for t in ends if c.end - 6 <= t < c.end and t > c.start]
        if after or before:
            t = min(after) if after else max(before)
        else:
            pause = _longest_pause(words, c.end - 6, c.end + 15)
            t = pause[0] if pause else c.end
        c.end = min(duration, t + min(0.4, _gap_after(t, words) / 2))


def _old_snap(c, cuts, segments, duration, style) -> None:
    for attr in ("start", "end"):
        t = getattr(c, attr)
        near = [x for x in cuts if abs(x - t) <= SNAP_REACH_S]
        if near:
            setattr(c, attr, min(near, key=lambda x: abs(x - t)))
    if style.snap_sentences and not c.whole:
        _old_snap_sentences(c, segments, duration)
    words = flat_words(segments)
    for w in words:
        if w["start"] < c.start < w["end"]:
            c.start = max(0.0, w["start"] - min(0.05, _gap_before(w["start"], words) / 2))
    for w in reversed(words):
        if w["start"] < c.end < w["end"]:
            c.end = min(duration, w["end"] + min(0.15, _gap_after(w["end"], words) / 2))
    c.start, c.end = round(max(0.0, c.start), 3), round(min(duration, c.end), 3)


# ------------------------------------------------------------------ synthetic input


def _segments(rng: random.Random, n_segs: int, mode: str) -> list[dict[str, Any]]:
    """``mode``: ``clean`` (chronological), ``rough`` (zero-length and overlapping words,
    segments with no words, marker holders) or ``shuffled`` (segments out of order)."""
    segs: list[dict[str, Any]] = []
    t = rng.uniform(0.0, 3.0)
    for _ in range(n_segs):
        words = []
        start = t
        for k in range(rng.randint(1, 9)):
            dur = (
                rng.choice([0.0, 0.1, 0.2, 0.35, 0.5]) if mode == "rough" else rng.uniform(0.1, 0.5)
            )
            if mode == "rough" and rng.random() < 0.1:
                t -= rng.uniform(0.0, 0.4)  # overlaps the word before
            words.append({"word": f"w{k}", "start": round(t, 2), "end": round(t + dur, 2)})
            t += dur + rng.choice([0.0, 0.05, 0.1, 0.3])
        text = "word " * len(words) + rng.choice([".", "", "?", ","])
        seg = {"start": start, "end": words[-1]["end"], "text": text, "words": words}
        if mode == "rough" and rng.random() < 0.1:
            del seg["words"]
        segs.append(seg)
        if mode == "rough" and rng.random() < 0.05:
            segs.append({"start": t, "end": t, "text": "", "markers": []})
        t += rng.choice([0.1, 0.5, 0.8, 1.5, 4.0, 9.0])
    if mode == "shuffled":
        rng.shuffle(segs)
    return segs


def _cands(rng: random.Random, n: int, span: float) -> list[tuple[float, float, bool]]:
    out = []
    for _ in range(n):
        a = rng.uniform(-3.0, span + 3.0)
        out.append((max(0.0, a), a + rng.uniform(0.5, 40.0), rng.random() < 0.1))
    return out


STYLES = [get_style("commentary"), replace(get_style("commentary"), snap_sentences=False)]


def _check(segments, cands, cuts, duration, style):
    index = analyze._SnapIndex(segments)
    for a, b, whole in cands:
        old = Candidate(source="x", start=a, end=b, peak_t=a, peak=1.0, signals={}, whole=whole)
        new = Candidate(source="x", start=a, end=b, peak_t=a, peak=1.0, signals={}, whole=whole)
        _old_snap(old, cuts, segments, duration, style)
        analyze._snap(new, cuts, segments, duration, style, index)
        assert (new.start, new.end) == (old.start, old.end)
        solo = Candidate(source="x", start=a, end=b, peak_t=a, peak=1.0, signals={}, whole=whole)
        analyze._snap(solo, cuts, segments, duration, style)  # no index passed
        assert (solo.start, solo.end) == (old.start, old.end)


@pytest.mark.parametrize("mode", ["clean", "rough", "shuffled"])
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("style", STYLES, ids=["sentences", "words"])
def test_snap_matches_the_old_scans(seed, mode, style):
    rng = random.Random(seed * 31 + len(mode))
    segments = _segments(rng, rng.randint(1, 120), mode)
    duration = max((s["end"] for s in segments), default=0.0) + rng.uniform(-5.0, 20.0)
    duration = max(duration, 1.0)
    cuts = sorted(rng.uniform(0, duration) for _ in range(rng.randint(0, 60)))
    _check(segments, _cands(rng, 80, duration), cuts, duration, style)


@pytest.mark.parametrize("style", STYLES, ids=["sentences", "words"])
def test_snap_matches_with_no_transcript_and_edge_candidates(style):
    _check([], [(0.0, 5.0, False), (10.0, 20.0, True)], [], 30.0, style)
    segs = _segments(random.Random(1), 10, "clean")
    last = segs[-1]["end"]
    edge = [(0.0, 0.2, False), (last + 1.0, last + 9.0, False), (last - 0.1, last + 50.0, False)]
    _check(segs, edge, [], last + 2.0, style)
    only_markers = [{"start": 1.0, "end": 1.0, "text": "", "markers": []}]
    _check(only_markers, edge, [], 30.0, style)


@pytest.mark.parametrize("seed", range(4))
def test_snap_sentences_matches_the_old_scans_alone(seed):
    rng = random.Random(seed)
    segments = _segments(rng, 80, "rough" if seed % 2 else "clean")
    duration = segments[-1]["end"] + 5
    index = analyze._SnapIndex(segments)
    for a, b, _ in _cands(rng, 100, duration):
        old = Candidate(source="x", start=a, end=b, peak_t=a, peak=1.0, signals={})
        new = Candidate(source="x", start=a, end=b, peak_t=a, peak=1.0, signals={})
        _old_snap_sentences(old, segments, duration)
        analyze._snap_sentences(new, segments, duration, index)
        assert (new.start, new.end) == (old.start, old.end)
