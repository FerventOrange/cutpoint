"""Speech-aware clip boundaries: never start or end a clip mid-utterance.

Smart tails (:mod:`cutpoint.tail`) handle clips that *end on speech* at
cut time; padding (``--extend`` / ``cutpoint extend``) and hand-picked segments can
still put an in or out point in the middle of a sentence, which plays as an
abrupt sound-to-silence cut. This module runs last, on the final in/out points
of every clip in a folder, and moves them to utterance boundaries.

Speech map (per source, absolute time):

* whisper words. Words longer than ``MAX_WORD_S`` (whisper stretches a word
  over a preceding pause) are trimmed to the part the voice grid hears, else
  to their last ``MAX_WORD_S``. Above ``wind_kmh`` (GoPro telemetry speed)
  only words whose segment ``avg_logprob`` is at least ``wind_min_logprob``
  count;
* voice-activity runs from the ``cutpoint events`` voice layer: mean of Silero VAD
  and high-passed PANNs speech >= ``VOICE_THR`` for >= ``MIN_VOICE_RUN_S``.
  Masked above ``wind_kmh``, where wind defeats every detector;
* laughter: the voice layer's wind-gated laughter (>= 0.5), events.json
  ``Laughter`` events and laughter tokens in the transcript.

Spans closer than ``phrase_gap`` form *phrases* (a pause there is a natural
sentence break); phrases closer than ``chain_gap`` form *utterances*, so quick
back-and-forth banter counts as one.

End: an end inside an utterance (or within ``breath`` after it) moves to the
utterance end + ``breath``, at most ``max_extend`` later. If that is not
possible (cap, source end, unreadable region, or the next clip of the same
source), the end is pulled back to the last phrase end inside the clip (never
more than ``PULLBACK_INTO_PICK_S`` into the picked moment); failing that it
moves forward to the latest phrase break within the cap, else stays put and
the audio fade (``--audio-fade``) softens the cut.

Start: a start inside an utterance moves back to its start (``lead_breath``
earlier, at most ``max_lead``; else to the latest phrase start within reach),
or forward to the next phrase start if that is closer and stays within the
padding (at most ``FORWARD_INTO_PICK_S`` into the pick).

Neighbours: clips of the same source in one folder never share footage.
Padded ranges that overlap are split in the longest speech-free stretch
between the two picks, and every move is bounded by the neighbouring clips.

Per-clip limits: a segment / manifest entry may carry ``"speech_max_extend"``
and ``"speech_max_lead"`` (seconds, see :data:`LIMIT_KEYS`); they replace
``max_extend`` / ``max_lead`` for that clip only (e.g. a chairlift
conversation that runs 30 s past the cut). The neighbour bounds still apply.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint.styles import Style
from cutpoint.tail import is_laugh

logger = logging.getLogger(__name__)

DT = 0.1  # voice-layer grid step (s)
EPS = 0.05  # a boundary this close to a span edge counts as outside it
MAX_WORD_S = 1.2  # longer whisper words are stretched over a pause
VOICE_THR = 0.6  # mean(VAD, PANNs speech) above which a bin is voiced
MIN_VOICE_RUN_S = 0.3  # shorter voiced runs are ignored
VOICE_CLOSE_S = 0.2  # voiced runs closer than this are joined
LAUGH_THR = 0.5  # voice-layer laughter feature above which a bin is laughter
PULLBACK_INTO_PICK_S = 1.0  # a pulled-back end may cut this far into the pick
FORWARD_INTO_PICK_S = 0.5  # a start may move this far forward into the pick
MAX_FORWARD_S = 2.0  # a start never moves forward more than this
MIN_CLIP_S = 1.0  # boundaries never leave less than this
TEXT_CHARS = 80  # words quoted in manifest reasons
# Per-clip overrides of SpeechSettings (segment / manifest key -> attribute).
LIMIT_KEYS = {"speech_max_extend": "max_extend", "speech_max_lead": "max_lead"}


@dataclass(frozen=True)
class SpeechSettings:
    """Parameters of the speech-aware boundary pass.

    Attributes:
        enabled: Run the pass at all.
        max_extend: Maximum seconds an end moves later.
        max_lead: Maximum seconds a start moves earlier.
        breath: Seconds kept after the utterance end.
        lead_breath: Seconds kept before the utterance start.
        chain_gap: Speech closer than this is one utterance (banter).
        phrase_gap: A pause at least this long is a sentence break.
        wind_kmh: Above this speed VAD is ignored and words need confidence.
        wind_min_logprob: Minimum segment ``avg_logprob`` of a word in wind.
    """

    enabled: bool = True
    max_extend: float = 8.0
    max_lead: float = 5.0
    breath: float = 0.4
    lead_breath: float = 0.15
    chain_gap: float = 0.6
    phrase_gap: float = 0.25
    wind_kmh: float = 20.0
    wind_min_logprob: float = -0.8

    def to_json(self) -> dict[str, Any]:
        """Dict for manifest.json."""
        return asdict(self)

    @classmethod
    def from_style(
        cls,
        style: Style | None,
        *,
        enabled: bool | None = None,
        max_extend: float | None = None,
        max_lead: float | None = None,
    ) -> SpeechSettings:
        """Style defaults (or the built-in ones), overridden by non-None CLI values."""
        base = (
            cls()
            if style is None
            else cls(
                enabled=style.speech_aware,
                max_extend=style.speech_max_extend,
                max_lead=style.speech_max_lead,
            )
        )
        return cls(
            enabled=base.enabled if enabled is None else enabled,
            max_extend=base.max_extend if max_extend is None else max(0.0, max_extend),
            max_lead=base.max_lead if max_lead is None else max(0.0, max_lead),
        )

    def with_limits(
        self, max_extend: float | None = None, max_lead: float | None = None
    ) -> SpeechSettings:
        """These settings with a clip's own caps (None keeps the global value)."""
        if max_extend is None and max_lead is None:
            return self
        return replace(
            self,
            max_extend=self.max_extend if max_extend is None else max_extend,
            max_lead=self.max_lead if max_lead is None else max_lead,
        )


def clip_limits(item: dict[str, Any], where: str = "segment") -> dict[str, float]:
    """Validated per-clip ``speech_max_extend`` / ``speech_max_lead`` of a segment
    or manifest entry (only the keys present).

    Raises:
        ValueError: If a value is not a non-negative number.
    """
    out: dict[str, float] = {}
    for key in LIMIT_KEYS:
        if key not in item or item[key] is None:
            continue
        try:
            val = float(item[key])
        except (TypeError, ValueError):
            raise ValueError(f"{where}: {key} must be a number of seconds") from None
        if not val >= 0:  # also rejects NaN
            raise ValueError(f"{where}: {key} must not be negative")
        out[key] = val
    return out


@dataclass(frozen=True)
class Span:
    """A stretch of speech/laughter (``kinds``: word, voice, laugh)."""

    start: float
    end: float
    kinds: frozenset[str]
    text: str = ""


def _merge(spans: list[Span], gap: float) -> list[Span]:
    """Join time-sorted spans whose gap is below ``gap``."""
    out: list[Span] = []
    for s in spans:
        if out and s.start - out[-1].end < gap:
            last = out[-1]
            text = " ".join(t for t in (last.text, s.text) if t)
            out[-1] = Span(last.start, max(last.end, s.end), last.kinds | s.kinds, text)
        else:
            out.append(s)
    return out


def _runs(mask: np.ndarray, close_bins: int, min_bins: int) -> list[tuple[int, int]]:
    """``[i, j)`` runs of True, joining gaps up to ``close_bins``, keeping >= ``min_bins``."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    runs, a, b = [], int(idx[0]), int(idx[0]) + 1
    for i in idx[1:]:
        if i - b <= close_bins:
            b = int(i) + 1
        else:
            runs.append((a, b))
            a, b = int(i), int(i) + 1
    runs.append((a, b))
    return [(a, b) for a, b in runs if b - a >= min_bins]


class SpeechMap:
    """Phrases and utterances of one source (absolute time)."""

    def __init__(self, spans: list[Span], settings: SpeechSettings) -> None:
        self.spans = sorted(spans, key=lambda s: s.start)
        self.phrases = _merge(self.spans, settings.phrase_gap)
        self.chains = _merge(self.phrases, settings.chain_gap)
        self._phrase_starts = np.array([p.start for p in self.phrases])

    def chain_at(self, t: float, after: float = 0.0) -> Span | None:
        """The utterance with ``start < t < end + after`` (edges within ``EPS`` excluded)."""
        for c in self.chains:
            if c.start >= t - EPS:
                return None
            if t < c.end + after - EPS:
                return c
        return None

    def phrases_in(self, t0: float, t1: float) -> list[Span]:
        """Phrases overlapping ``[t0, t1]``."""
        return [p for p in self.phrases if p.end > t0 and p.start < t1]

    def next_phrase(self, t: float) -> Span | None:
        """First phrase starting at or after ``t``."""
        i = int(np.searchsorted(self._phrase_starts, t))
        return self.phrases[i] if i < len(self.phrases) else None

    def prev_phrase_end(self, t: float) -> float | None:
        """End of the last phrase that ends at or before ``t``."""
        best = None
        for p in self.phrases:
            if p.start > t:
                break
            if p.end <= t:
                best = p.end
        return best

    def words_between(self, t0: float, t1: float) -> str:
        """Transcript text of the spans in ``[t0, t1]`` (for manifest reasons)."""
        text = " ".join(s.text for s in self.spans if s.text and s.end > t0 and s.start < t1)
        return text if len(text) <= TEXT_CHARS else text[: TEXT_CHARS - 3] + "..."

    def quietest_cut(self, t0: float, t1: float) -> float:
        """Middle of the longest speech-free stretch in ``[t0, t1]`` (midpoint if none)."""
        if t1 <= t0:
            return (t0 + t1) / 2
        best, best_len, cur = (t0 + t1) / 2, 0.0, t0
        for c in self.chains:
            if c.end <= t0:
                continue
            if c.start >= t1:
                break
            if c.start - cur > best_len:
                best, best_len = (cur + c.start) / 2, c.start - cur
            cur = max(cur, c.end)
        if t1 - cur > best_len:
            best = (cur + t1) / 2
        return best


def build_speech_map(
    src: Path,
    words: list[dict[str, Any]] | None,
    events: list[dict[str, Any]] | None,
    settings: SpeechSettings,
) -> SpeechMap | None:
    """Speech map of one source from words, the voice layer, laughter and telemetry.

    Args:
        src: Source path (cache identity; only cached data is read).
        words: :func:`cutpoint.sources.source_words` output, or None.
        events: events.json entries of this source, or None.
        settings: Pass parameters.

    Returns:
        The map, or None when there is no transcript, voice grid or event.
    """
    from cutpoint import telemetry  # noqa: PLC0415
    from cutpoint.events import voice as VO  # noqa: PLC0415

    try:
        grids = VO.load_voice(src)
    except (OSError, ValueError) as err:
        logger.warning("%s: unreadable voice grid (%s); words only", src.name, err)
        grids = None
    try:
        speed = telemetry.cached_speed(src)
    except (OSError, ValueError):
        speed = None
    if not words and grids is None and not events:
        return None

    def windy(t0: float, t1: float) -> bool:
        if speed is None or speed.size == 0:
            return False
        i, j = max(0, int(t0 / DT)), min(speed.size, int(t1 / DT) + 1)
        seg = speed[i:j]
        return bool(seg.size) and bool(np.nanmax(np.nan_to_num(seg, nan=0.0)) > settings.wind_kmh)

    voiced = None
    spans: list[Span] = []
    if grids is not None:
        n = len(grids["vad"])
        moving = None
        if speed is not None:
            moving = np.zeros(n)
            k = min(n, speed.size)
            moving[:k] = np.clip(np.nan_to_num(speed[:k]) / 40.0, 0, 1)
        feats = VO.voice_features(grids, moving, VO.VoiceCalibration(smooth_s=DT))
        voiced = feats["voice"] >= VOICE_THR
        if speed is not None:  # wind: the detectors collapse, trust nothing
            fast = np.zeros(n, bool)
            k = min(n, speed.size)
            fast[:k] = np.nan_to_num(speed[:k]) > settings.wind_kmh
            voiced &= ~fast
        close = round(VOICE_CLOSE_S / DT)
        for a, b in _runs(voiced, close, round(MIN_VOICE_RUN_S / DT)):
            spans.append(Span(a * DT, b * DT, frozenset({"voice"})))
        for a, b in _runs(feats["laughter"] >= LAUGH_THR, close, 2):
            spans.append(Span(a * DT, b * DT, frozenset({"laugh"})))
    for e in events or []:
        if (
            e.get("kind") == "audio"
            and e.get("label") == "Laughter"
            and (e.get("confidence") or 0) >= 0.05
        ):
            spans.append(
                Span(
                    float(e["start"]), float(e.get("end") or e["start"] + 0.5), frozenset({"laugh"})
                )
            )
    for w in words or []:
        s, e = float(w["start"]), float(w["end"])
        if e - s > MAX_WORD_S:
            s, e = _trim_long_word(s, e, voiced)
        if windy(s, e) and (
            w.get("avg_logprob") is None or w["avg_logprob"] < settings.wind_min_logprob
        ):
            continue
        kind = "laugh" if is_laugh(w["word"]) else "word"
        spans.append(Span(s, max(e, s + 0.05), frozenset({kind}), w["word"].strip()))
    return SpeechMap(spans, settings) if spans else None


def _trim_long_word(s: float, e: float, voiced: np.ndarray | None) -> tuple[float, float]:
    """Part of a stretched whisper word that is actually voiced."""
    if voiced is not None:
        i, j = max(0, int(s / DT)), min(len(voiced), int(e / DT) + 1)
        hit = np.flatnonzero(voiced[i:j])
        if hit.size:
            return (i + hit[0]) * DT, min(e, (i + hit[-1] + 1) * DT)
    return e - MAX_WORD_S, e


# --------------------------------------------------------------------------- moves
@dataclass
class Move:
    """One boundary decision (recorded in manifest.json)."""

    old: float
    new: float
    reason: str
    words: str = ""

    @property
    def delta(self) -> float:
        """Signed change in seconds (+ = later)."""
        return round(self.new - self.old, 3)

    def to_json(self) -> dict[str, Any]:
        """Rounded dict for manifest.json."""
        out = {
            "from": round(self.old, 3),
            "to": round(self.new, 3),
            "moved": self.delta,
            "reason": self.reason,
        }
        if self.words:
            out["words"] = self.words
        return out


def fit_end(
    end: float, smap: SpeechMap | None, settings: SpeechSettings, *, lo: float, hi: float
) -> Move:
    """Move a clip end out of an utterance.

    Args:
        end: Current end.
        smap: Speech map of the source (None = nothing known).
        settings: Pass parameters.
        lo: Earliest acceptable end for a pull-back.
        hi: Latest possible end (source/readable end, next clip of the source).
    """
    if smap is None:
        return Move(end, end, "no speech data")
    chain = smap.chain_at(end, after=settings.breath)
    if chain is None:
        return Move(end, end, "not in speech")
    cap = min(hi, end + settings.max_extend)
    target = chain.end + settings.breath
    nxt = smap.next_phrase(chain.end + EPS)
    if nxt is not None:  # keep the breath clear of the next utterance's onset
        target = min(target, max(chain.end, nxt.start - EPS))
    if chain.end <= cap + 1e-6:  # the speech itself fits; the breath may be cut short
        target = min(target, cap)
    if target <= cap + 1e-6:
        if target <= end + 0.02:
            return Move(end, end, "speech ends at the cut")
        return Move(
            end, target, "extended to the end of the utterance", smap.words_between(end, chain.end)
        )
    why = "cap" if target > end + settings.max_extend else "limit"
    blocked = f"utterance runs {chain.end - end:.1f}s past the cut" + (
        f" (> {settings.max_extend:g}s cap)"
        if why == "cap"
        else " (source end / unreadable region / next clip)"
    )
    # Pull back to the last phrase end inside the clip.
    for p in reversed(smap.phrases_in(lo, end)):
        if p.end > end - EPS or p.end < lo:
            continue
        nxt_p = smap.next_phrase(p.end + EPS)
        room = (nxt_p.start - p.end) if nxt_p is not None else settings.breath
        new = min(end, p.end + min(settings.breath, max(0.0, room - EPS)))
        return Move(end, new, f"pulled back to a pause; {blocked}", smap.words_between(new, end))
    # Forward to the latest sentence break inside the cap.
    best = None
    for p in smap.phrases_in(end, cap):
        if p.end <= end or p.end > cap:
            continue
        nxt_p = smap.next_phrase(p.end + EPS)
        room = (nxt_p.start - p.end) if nxt_p is not None else settings.breath
        new = p.end + min(settings.breath, max(0.0, room - EPS))
        if new <= cap:
            best = new
    if best is not None:
        return Move(
            end,
            best,
            f"extended to a pause inside the banter; {blocked}",
            smap.words_between(end, best),
        )
    return Move(
        end,
        end,
        f"continuous speech, left to the audio fade; {blocked}",
        smap.words_between(end - 1.0, end + 1.0),
    )


def fit_start(
    start: float, smap: SpeechMap | None, settings: SpeechSettings, *, lo: float, hi: float
) -> Move:
    """Move a clip start out of an utterance.

    Args:
        start: Current start.
        smap: Speech map of the source.
        settings: Pass parameters.
        lo: Earliest possible start (source/readable start, previous clip).
        hi: Latest acceptable start for a forward move.
    """
    if smap is None:
        return Move(start, start, "no speech data")
    chain = smap.chain_at(start)
    if chain is None:
        return Move(start, start, "not in speech")
    floor = max(lo, start - settings.max_lead)
    back = None
    prev_end = smap.prev_phrase_end(chain.start - EPS)
    lead = settings.lead_breath
    if prev_end is not None:
        lead = min(lead, max(0.0, (chain.start - prev_end) / 2))
    if chain.start - lead >= floor:
        back = (chain.start - lead, "moved back to the start of the utterance")
    else:  # latest sentence start inside the utterance that is within reach
        for p in reversed(smap.phrases_in(floor, start)):
            if p.start >= start - EPS:
                continue
            if p.start < floor or p.start < chain.start:
                break
            pe = smap.prev_phrase_end(p.start - EPS) or (p.start - lead)
            new = p.start - min(lead, max(0.0, (p.start - pe) / 2))
            if new >= floor:
                back = (new, "moved back to a sentence start")
                break
    fwd = None
    nxt = smap.next_phrase(start + EPS)
    if nxt is not None and nxt.start < chain.end:  # a sentence break inside the utterance
        pe = smap.prev_phrase_end(nxt.start - EPS) or start
        new = nxt.start - min(settings.lead_breath, max(0.0, (nxt.start - pe) / 2))
        fwd_reason = "moved forward to the next sentence start"
    else:  # into the pause after the utterance
        gap = (nxt.start - chain.end) if nxt is not None else 1.0
        new = chain.end + min(0.1, gap / 2)
        fwd_reason = "moved forward past the end of the utterance"
    if start < new <= min(hi, start + MAX_FORWARD_S):
        fwd = (new, fwd_reason)
    choices = [c for c in (back, fwd) if c is not None]
    if not choices:
        return Move(
            start,
            start,
            "mid-utterance start, no boundary within reach "
            f"(utterance began {start - chain.start:.1f}s earlier)",
            smap.words_between(start - 1.0, start + 1.0),
        )
    new, reason = min(choices, key=lambda c: (abs(c[0] - start), c is fwd))
    lo_t, hi_t = sorted((start, new))
    return Move(start, new, reason, smap.words_between(lo_t, hi_t))


# --------------------------------------------------------------------------- folders
@dataclass
class ClipBounds:
    """In/out points of one clip of a folder, plus its constraints.

    Attributes:
        name: Clip file name (for logs/records).
        key: Source identity (clips of the same key never overlap).
        start: Current start (after padding).
        end: Current end.
        keep_start: Start of the picked moment (before padding).
        keep_end: End of the picked moment (after tails, before padding).
        lo: First readable time of the source part.
        hi: Last readable time (source end / read-error boundary).
        smap: Speech map of the source.
        fit_start_on: Adjust the start.
        fit_end_on: Adjust the end.
        max_extend: This clip's end cap (None = the pass's ``max_extend``).
        max_lead: This clip's start cap (None = the pass's ``max_lead``).
        record: Filled with the decisions (manifest ``speech`` key).
    """

    name: str
    key: str
    start: float
    end: float
    keep_start: float
    keep_end: float
    lo: float
    hi: float
    smap: SpeechMap | None
    fit_start_on: bool = True
    fit_end_on: bool = True
    max_extend: float | None = None
    max_lead: float | None = None
    record: dict[str, Any] = field(default_factory=dict)


def separate(clips: list[ClipBounds]) -> None:
    """Split overlapping ranges of same-source clips (no footage shown twice).

    The cut goes in the longest speech-free stretch between the two picks; if
    the picks themselves overlap, in the middle of the overlap.
    """
    groups: dict[str, list[ClipBounds]] = {}
    for c in clips:
        groups.setdefault(c.key, []).append(c)
    for group in groups.values():
        group.sort(key=lambda c: (c.keep_start, c.keep_end))
        for a, b in pairwise(group):
            if a.end <= b.start + 1e-6:
                continue
            if a.keep_end <= b.keep_start:
                lo_t, hi_t = max(a.keep_end, b.start), min(b.keep_start, a.end)
                cut = a.smap.quietest_cut(lo_t, hi_t) if a.smap is not None else (lo_t + hi_t) / 2
                note = "padding overlapped"
            else:
                cut = (a.keep_end + b.keep_start) / 2
                note = "picked moments overlap"
            cut = round(min(max(cut, a.start + MIN_CLIP_S), b.end - MIN_CLIP_S), 3)
            a.record["neighbour_end"] = {
                "from": round(a.end, 3),
                "to": cut,
                "with": b.name,
                "reason": note,
            }
            b.record["neighbour_start"] = {
                "from": round(b.start, 3),
                "to": cut,
                "with": a.name,
                "reason": note,
            }
            a.end, b.start = min(a.end, cut), max(b.start, cut)


def fit_clips(clips: list[ClipBounds], settings: SpeechSettings) -> None:
    """Separate same-source neighbours, then move ends/starts out of speech.

    Clips of one source are handled in source-time order: an end may extend up
    to the next clip's start, and the next clip's start may move back only to
    this clip's (final) end. A clip's own ``max_extend`` / ``max_lead``
    replace the global caps for it (recorded as ``limits``). Results go to
    each clip's ``record``.
    """
    separate(clips)
    groups: dict[str, list[ClipBounds]] = {}
    for c in clips:
        groups.setdefault(c.key, []).append(c)
    for group in groups.values():
        group.sort(key=lambda c: (c.keep_start, c.keep_end))
        prev_end = None
        for i, c in enumerate(group):
            nxt = group[i + 1] if i + 1 < len(group) else None
            own = settings.with_limits(c.max_extend, c.max_lead)
            if own is not settings:
                c.record["limits"] = {"max_extend": own.max_extend, "max_lead": own.max_lead}
            if c.fit_start_on:
                lo = max(c.lo, prev_end if prev_end is not None else c.lo)
                hi = min(max(c.start, c.keep_start + FORWARD_INTO_PICK_S), c.end - MIN_CLIP_S)
                mv = fit_start(c.start, c.smap, own, lo=lo, hi=hi)
                c.start = round(mv.new, 3)
                c.record["start"] = mv.to_json()
            if c.fit_end_on:
                hi = min(c.hi, nxt.start if nxt is not None else c.hi)
                lo = max(c.start + MIN_CLIP_S, min(c.end, c.keep_end - PULLBACK_INTO_PICK_S))
                mv = fit_end(c.end, c.smap, own, lo=lo, hi=max(hi, c.end))
                c.end = round(mv.new, 3)
                c.record["end"] = mv.to_json()
            prev_end = c.end


def log_summary(clips: list[ClipBounds]) -> None:
    """One info line per moved boundary and a total."""
    moved_s = moved_e = 0
    total = 0.0
    for c in clips:
        if "neighbour_end" in c.record:
            r = c.record["neighbour_end"]
            logger.info(
                "  %s end %.2f -> %.2f: %s with %s (split in a pause)",
                c.name,
                r["from"],
                r["to"],
                r["reason"],
                r["with"],
            )
        for side in ("start", "end"):
            r = c.record.get(side)
            if r and r["moved"]:
                moved_s += side == "start"
                moved_e += side == "end"
                total += -r["moved"] if side == "start" else r["moved"]
                logger.info(
                    "  %s %s %+.2fs: %s%s",
                    c.name,
                    side,
                    r["moved"],
                    r["reason"],
                    f' ("{r["words"]}")' if r.get("words") else "",
                )
    logger.info(
        "Speech-aware boundaries: moved %d end(s) and %d start(s), %+.1fs in total",
        moved_e,
        moved_s,
        total,
    )
