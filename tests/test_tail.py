"""``smart_tail`` decides where a speech-ended clip ends, on synthetic words and audio (#1085)."""

from __future__ import annotations

import numpy as np
import pytest

from cutpoint.tail import TailSettings, smart_tail

SR = 16000
START, END, LAST_WORD_END = 100.0, 105.0, 104.8
SETTINGS = TailSettings(tail_min=1.5, tail_max=5.0, settle=0.8)
CAP = LAST_WORD_END + SETTINGS.tail_max


def _words(*later: tuple[float, float], word: str = "next") -> list[dict]:
    spoken = [{"start": 104.0, "end": LAST_WORD_END, "word": "done."}]
    return spoken + [{"start": s, "end": e, "word": word} for s, e in later]


def _audio(loud_until: float):
    """Quiet noise with a loud reaction from the last word until ``loud_until`` (absolute s)."""
    rng = np.random.default_rng(0)

    def audio(t0: float, t1: float) -> tuple[np.ndarray, float]:
        t = t0 + np.arange(int((t1 - t0) * SR)) / SR
        amp = np.where((t >= LAST_WORD_END) & (t < loud_until), 0.3, 0.001)
        return (rng.standard_normal(t.size) * amp).astype(np.float32), t0

    return audio


@pytest.mark.parametrize(
    ("words", "loud_until", "end", "reason", "new_end", "included"),
    [
        # the reaction dies down at 107 s: the end follows it, inside the tail budget
        (_words(), 107.0, END, "audio settled", (107.0, 108.3), ""),
        # still loud at the cap: the end is the cap, the tail never runs on
        (_words(), 1e9, END, "hit max tail", (CAP, CAP), ""),
        # still loud, but a new sentence starts at 107.0: stop before it
        (_words((107.0, 107.5)), 1e9, END, "stopped before next speech", (106.88, 106.88), ""),
        # the clip ends well after its last word: not speech-ended, left as it is
        (_words(), 1e9, 110.0, "not speech-ended", (110.0, 110.0), ""),
        # speech resumes within the join gap: its sentence is kept and named
        (
            _words((105.2, 105.9), word="right."),
            107.0,
            END,
            "audio settled",
            (105.9, CAP),
            "right.",
        ),
        # quiet at once: the guaranteed minimum after the last word
        (_words(), LAST_WORD_END, END, "min tail", (106.3, 106.3), ""),
    ],
    ids=[
        "settles",
        "never-settles",
        "next-speech",
        "not-speech-ended",
        "joins-continuation",
        "min-tail",
    ],
)
def test_smart_tail_places_the_end(words, loud_until, end, reason, new_end, included):
    got = smart_tail(START, end, words, _audio(loud_until), SETTINGS, limit=200.0)
    assert got.reason.startswith(reason)
    lo, hi = new_end
    assert lo - 0.01 <= got.end <= hi + 0.01
    assert got.end >= end and got.extension == pytest.approx(got.end - end, abs=0.002)
    assert got.included_words == included
