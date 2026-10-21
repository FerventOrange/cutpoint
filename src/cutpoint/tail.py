"""Smart clip tails: keep the punchline that follows the last transcribed word.

In banter clips the funny part often comes right after the speech: laughter, a
reaction, or a game event (kill, death, teamkill banner). A clip end snapped
to the last word cuts it off. :func:`smart_tail` extends a speech-ended clip:

1. Find the last word inside the clip. A clip counts as *speech-ended* if it
   ends at most ``SPEECH_END_S`` after that word.
2. Speech that resumes within ``join_gap`` (or laughter tokens) continues the
   same exchange and is included up to a sentence end, if it fits under the
   cap. Any other next speech is a hard stop, so the tail never runs into the
   start of an unrelated sentence.
3. From the last kept word on, short-term RMS (100 ms windows, 50 ms hop, on
   the cached 16 kHz analysis audio) is compared with the local baseline (20th
   percentile of the preceding ~20 s). The reaction is over once the smoothed
   level stays within ``BASELINE_MARGIN_DB`` of the baseline for ``settle`` s.
4. The new end is at least ``tail_min`` and at most ``tail_max`` after the last
   word. It never comes before the original end, never passes a hard stop, and
   never passes ``limit`` (source end, unreadable region or the next selected
   clip from the same source).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from cutpoint.styles import LAUGHTER, Style

logger = logging.getLogger(__name__)

SPEECH_END_S = 1.0  # clip ends this close after the last word -> speech-ended
JOIN_GAP_S = 0.8  # resumed speech closer than this continues the exchange
NEXT_SPEECH_MARGIN_S = 0.12  # stop this far before an unrelated next word
BASELINE_CONTEXT_S = 20.0  # audio before the last word used for the baseline
BASELINE_PCT = 20  # percentile of RMS dB used as the local baseline
BASELINE_MARGIN_DB = 6.0  # "settled" = within this many dB of the baseline
SETTLE_PAD_S = 0.3  # keep a beat of the quiet after the reaction ends
HOP_S, WIN_S, SMOOTH_S = 0.05, 0.10, 0.25
_LAUGH_STEMS = tuple(p.rstrip("*") for p in LAUGHTER)


@dataclass(frozen=True)
class TailSettings:
    """Tail parameters for one clip.

    Attributes:
        enabled: Run smart-tail detection at all.
        tail_min: Guaranteed seconds after the last word (speech-ended clips).
        tail_max: Maximum seconds after the last word.
        settle: Seconds the level must stay near the baseline.
        fixed: If set, extend the end by exactly this many seconds instead
            (per-segment ``"tail"`` override); no detection is run.
    """

    enabled: bool = True
    tail_min: float = 1.5
    tail_max: float = 5.0
    settle: float = 0.8
    fixed: float | None = None

    @classmethod
    def from_style(
        cls,
        style: Style,
        *,
        enabled: bool | None = None,
        tail_min: float | None = None,
        tail_max: float | None = None,
    ) -> TailSettings:
        """Style defaults, overridden by any CLI values that are not None."""
        lo = style.tail_min if tail_min is None else tail_min
        hi = style.tail_max if tail_max is None else tail_max
        return cls(
            enabled=style.smart_tail if enabled is None else enabled,
            tail_min=lo,
            tail_max=max(lo, hi),
            settle=style.tail_settle,
        )


@dataclass
class TailResult:
    """Outcome of :func:`smart_tail`, recorded in manifest.json.

    Attributes:
        end: New clip end (absolute source time).
        extension: Seconds added to the original end (>= 0).
        reason: Why the end was placed here.
        last_word_end: End of the last kept word, if any.
        settled_at: Where the audio fell back to the baseline, if it did.
        baseline_db: Local baseline level (dBFS RMS), if measured.
        included_words: Words of a continuing exchange added to the clip.
    """

    end: float
    extension: float
    reason: str
    last_word_end: float | None = None
    settled_at: float | None = None
    baseline_db: float | None = None
    included_words: str = ""

    def to_json(self) -> dict[str, Any]:
        """Rounded dict for manifest.json."""
        return {
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in asdict(self).items()
            if v not in (None, "")
        }


def is_laugh(word: str) -> bool:
    """Whether a transcript word starts with a laughter stem (:data:`cutpoint.styles.LAUGHTER`)."""
    tok = "".join(ch for ch in word.lower() if ch.isalnum())
    return bool(tok) and tok.startswith(_LAUGH_STEMS)


def rms_db(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Short-term RMS level in dBFS (``WIN_S`` windows every ``HOP_S``)."""
    hop, win = int(HOP_S * sr), int(WIN_S * sr)
    if samples.size < win:
        return np.zeros(0)
    n = 1 + (samples.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    power = np.mean(samples[idx].astype(np.float64) ** 2, axis=1)
    return 10 * np.log10(power + 1e-10)


def _continuation(
    words: list[dict[str, Any]], i_next: int, anchor: float, cap: float
) -> tuple[float, int, str]:
    """Words after the last kept one that continue the same exchange.

    Returns (new anchor, index of the first word NOT included, included text).
    Speech joins if it starts within ``JOIN_GAP_S`` of the previous word (or is
    laughter); it is cut back to the last sentence end (or laughter) that
    fits under ``cap``, so a long new sentence is never taken half.
    """
    chain_end, i, best = anchor, i_next, (anchor, i_next)
    while i < len(words):
        w = words[i]
        if w["start"] - chain_end >= JOIN_GAP_S and not is_laugh(w["word"]):
            break
        if w["end"] > cap:
            break
        chain_end = w["end"]
        i += 1
        if w["word"].rstrip()[-1:] in ".?!" or is_laugh(w["word"]):
            best = (chain_end, i)
            continue
        nxt = words[i] if i < len(words) else None
        if nxt is None or nxt["start"] - chain_end >= JOIN_GAP_S:
            best = (chain_end, i)  # a pause also ends the turn
    new_anchor, first_out = best
    text = " ".join(w["word"].strip() for w in words[i_next:first_out])
    return new_anchor, first_out, text


def smart_tail(
    start: float,
    end: float,
    words: list[dict[str, Any]] | None,
    audio: Callable[[float, float], tuple[np.ndarray, float]],
    settings: TailSettings,
    limit: float,
) -> TailResult:
    """Place the end of a clip so it keeps the reaction after the last word.

    Args:
        start: Clip start (absolute source time).
        end: Original clip end.
        words: Transcript words of the source in absolute time (None = no
            transcript; then only a ``fixed`` tail can apply).
        audio: ``audio(t0, t1)`` -> (16 kHz mono samples, time of first sample).
        settings: Tail parameters.
        limit: Hard upper bound for the end.

    Returns:
        The new end and how it was chosen. The end is never earlier than
        ``end`` (unless ``end`` itself exceeds ``limit``).
    """
    end = min(end, limit)
    if settings.fixed is not None:
        new_end = min(limit, end + max(0.0, settings.fixed))
        return TailResult(new_end, new_end - end, f"fixed tail {settings.fixed:g}s")
    if not settings.enabled:
        return TailResult(end, 0.0, "smart tail off")
    if not words:
        return TailResult(end, 0.0, "no transcript")

    inside = [i for i, w in enumerate(words) if w["start"] < end - 0.05 and w["end"] > start]
    if not inside:
        return TailResult(end, 0.0, "no speech in clip")
    i_last = max(inside, key=lambda i: words[i]["end"])
    anchor0 = words[i_last]["end"]
    if end - anchor0 > SPEECH_END_S:
        return TailResult(
            end,
            0.0,
            f"not speech-ended ({end - anchor0:.1f}s after last word)",
            last_word_end=anchor0,
        )

    cap = min(limit, anchor0 + settings.tail_max)
    # Continuing speech may use the tail budget minus the min tail, so there is
    # always room left for the reaction after it.
    join_cap = min(cap, anchor0 + max(0.5, settings.tail_max - settings.tail_min))
    anchor, i_next, joined = _continuation(words, i_last + 1, anchor0, join_cap)
    anchor = max(anchor, anchor0)
    hard_stop = limit
    if i_next < len(words):
        hard_stop = min(hard_stop, words[i_next]["start"] - NEXT_SPEECH_MARGIN_S)

    # Level analysis from the last kept word up to the cap.
    ctx0 = max(0.0, anchor - BASELINE_CONTEXT_S)
    samples, t_first = audio(ctx0, cap + settings.settle + 0.5)
    db = rms_db(samples)
    settled_at = baseline = None
    if db.size:
        times = t_first + np.arange(db.size) * HOP_S + WIN_S / 2
        k = max(1, int(SMOOTH_S / HOP_S))
        smooth = np.convolve(db, np.ones(k) / k, mode="same")
        pre = db[times < anchor]
        baseline = float(np.percentile(pre if pre.size >= 20 else db, BASELINE_PCT))
        quiet = smooth <= baseline + BASELINE_MARGIN_DB
        need = max(1, int(settings.settle / HOP_S))
        run = 0
        for j in np.nonzero(times >= anchor)[0]:
            run = run + 1 if quiet[j] else 0
            if run >= need:
                settled_at = float(times[j - need + 1])
                break

    if settled_at is not None:
        target, reason = settled_at + SETTLE_PAD_S, "audio settled"
    else:
        target, reason = cap, "hit max tail (still active)"
    if target < anchor + settings.tail_min:
        target, reason = anchor + settings.tail_min, "min tail"
    target = min(target, cap)
    if target > hard_stop:
        target, reason = hard_stop, "stopped before next speech"
    if target > limit:
        target, reason = limit, "limited by source end / unreadable region / next clip"
    new_end = max(end, target)
    return TailResult(
        end=round(new_end, 3),
        extension=round(new_end - end, 3),
        reason=reason,
        last_word_end=round(anchor0, 3),
        settled_at=round(settled_at, 3) if settled_at is not None else None,
        baseline_db=round(baseline, 1) if baseline is not None else None,
        included_words=joined,
    )
