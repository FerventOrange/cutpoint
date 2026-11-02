"""Units that decide what the transcript keeps and where clip boundaries go.

``transcribe.filter_hallucinations`` drops segments, ``analyze.transcript_text`` writes the
``transcript.txt`` agents read, ``analyze.sentence_bounds`` / ``_snap_sentences`` and
``boundaries.fit_end`` / ``fit_start`` / ``separate`` move clip edges out of speech, and
``extend.apply_overrides`` applies ``--clip-override``. Synthetic segments and hand-built
speech maps; no model, no ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import media_info
from cutpoint import analyze, planning
from cutpoint.boundaries import (
    ClipBounds,
    Span,
    SpeechMap,
    SpeechSettings,
    fit_end,
    fit_start,
    separate,
)
from cutpoint.extend import ExtendPlan, apply_overrides
from cutpoint.transcribe import filter_hallucinations

SETTINGS = SpeechSettings()


def _seg(start: float, end: float, text: str, **extra) -> dict:
    return {"start": start, "end": end, "text": text, **extra}


def _words(*items: tuple[float, float, str]) -> list[dict]:
    return [{"start": s, "end": e, "word": w} for s, e, w in items]


# ------------------------------------------------------------------- filter_hallucinations
@pytest.mark.parametrize(
    ("seg", "kept"),
    [
        (_seg(0, 2, "We should go left here."), True),  # an older transcript stored no scores
        (_seg(0, 2, "Yeah. Yeah. Yeah. Yeah."), False),  # one short phrase looping
        (_seg(0, 2, "no no no"), True),  # three repeats are still speech
        (_seg(0, 2, "Real words.", avg_logprob=-1.5), False),
        (_seg(0, 2, "Real words.", avg_logprob=-0.9, no_speech_prob=0.8), False),
        (_seg(0, 2, "Real words.", avg_logprob=-0.3, no_speech_prob=0.8), True),  # confident
        (_seg(0, 2, "Real words.", avg_logprob=-0.3, compression_ratio=3.0), False),
    ],
    ids=["speech", "repeat-loop", "three-repeats", "low-logprob",
         "no-speech-and-unsure", "no-speech-but-confident", "compressed"],
)  # fmt: skip
def test_filter_hallucinations_drops_only_whisper_artefacts(seg, kept):
    out, dropped = filter_hallucinations([seg])
    assert (out == [seg]) is kept
    assert dropped == (0 if kept else 1)


def test_filter_hallucinations_keeps_the_first_two_of_a_run_of_repeated_short_segments():
    segs = [_seg(i, i + 1, " Thank you.") for i in range(5)]
    segs.append(_seg(5, 6, "Then we left."))
    segs.append(_seg(6, 7, "Thank you."))  # the run was broken: a new run starts

    out, dropped = filter_hallucinations(segs)

    assert out == [segs[0], segs[1], segs[5], segs[6]]
    assert dropped == 3


# ------------------------------------------------------------------- transcript_text
def test_transcript_text_breaks_lines_at_sentence_ends_and_pauses():
    segments = [
        _seg(10.0, 12.0, "First sentence.", words=_words((10.0, 11.0, "First"), (11.0, 12.0, "sentence."))),
        # no punctuation, but a 1 s pause inside the segment
        _seg(12.5, 15.0, "so then we went", words=_words(
            (12.5, 13.0, "so"), (13.0, 13.4, "then"), (14.4, 14.7, "we"), (14.7, 15.0, "went"))),
        _seg(15.1, 16.0, "home"),  # no word timings: the segment is one "word"
    ]  # fmt: skip

    lines = analyze.transcript_text("a.mp4", segments)

    assert lines == [
        "[a.mp4 00:00:10.0-00:00:12.0] First sentence.",
        "[a.mp4 00:00:12.5-00:00:13.4] so then",
        "[a.mp4 00:00:14.4-00:00:16.0] we went home",
    ]


def test_transcript_text_of_no_speech_is_empty():
    assert analyze.transcript_text("a.mp4", []) == []


# ------------------------------------------------------------------- sentence_bounds
def test_sentence_bounds_are_edges_at_terminal_punctuation_or_pauses():
    segments = [
        _seg(0.0, 1.5, "Hello there."),
        _seg(1.6, 2.4, "and then"),  # after a full stop: starts a sentence; ends at a pause
        _seg(2.5, 3.0, "we"),  # continues "and then": no edge before it
        _seg(4.0, 5.0, "left?"),  # after a 1 s pause
    ]
    starts, ends = analyze.sentence_bounds(segments)
    assert starts == [0.0, 1.6, 4.0]
    assert ends == [1.5, 3.0, 5.0]


# ------------------------------------------------------------------- _snap_sentences
SENTENCES = [
    _seg(10.0, 12.0, "First sentence.", words=_words((10.0, 11.0, "First"), (11.0, 12.0, "sentence."))),
    _seg(13.0, 16.0, "Second one here.", words=_words(
        (13.0, 14.0, "Second"), (14.0, 15.0, "one"), (15.0, 16.0, "here."))),
    _seg(18.0, 19.0, "Third.", words=_words((18.0, 19.0, "Third."))),
]  # fmt: skip


def _cand(start: float, end: float) -> analyze.Candidate:
    return analyze.Candidate(Path("a.mp4"), start, end, start, 1.0, {})


def test_snap_sentences_moves_edges_inside_speech_to_the_sentence_edges():
    c = _cand(14.5, 14.8)  # both edges inside "Second one here."
    analyze._snap_sentences(c, SENTENCES, duration=60.0)
    # start: back to the sentence start, padded by at most half the silence before it
    assert c.start == pytest.approx(13.0 - 0.2)
    # end: on to the sentence end, padded by at most half the silence after it
    assert c.end == pytest.approx(16.0 + 0.4)


def test_snap_sentences_padding_never_reaches_into_the_neighbouring_word():
    tight = [
        _seg(10.0, 12.0, "Done.", words=_words((10.0, 12.0, "Done."))),
        _seg(12.1, 14.0, "Next part.", words=_words((12.1, 13.0, "Next"), (13.0, 14.0, "part."))),
    ]
    c = _cand(13.5, 13.8)
    analyze._snap_sentences(c, tight, duration=60.0)
    assert c.start == pytest.approx(12.1 - 0.05)  # half of the 0.1 s pause
    assert c.end == pytest.approx(14.0 + 0.4)  # nothing after: the full 0.4 s


def test_snap_sentences_leaves_edges_in_silence_alone():
    c = _cand(16.5, 17.2)  # between "here." (16.0) and "Third." (18.0)
    analyze._snap_sentences(c, SENTENCES, duration=60.0)
    assert (c.start, c.end) == (16.5, 17.2)


# ------------------------------------------------------------------- fit_end / fit_start
def _map(*spans: tuple[float, float, str]) -> SpeechMap:
    return SpeechMap([Span(s, e, frozenset({"word"}), t) for s, e, t in spans], SETTINGS)


def test_fit_end_without_speech_data_or_outside_speech_keeps_the_end():
    assert fit_end(12.0, None, SETTINGS, lo=0, hi=100).new == 12.0
    move = fit_end(20.0, _map((10, 14, "hello world")), SETTINGS, lo=0, hi=100)
    assert (move.new, move.reason) == (20.0, "not in speech")


def test_fit_end_extends_to_the_end_of_the_utterance_plus_a_breath():
    move = fit_end(12.0, _map((10, 14, "hello world")), SETTINGS, lo=0, hi=100)
    assert move.new == pytest.approx(14.0 + SETTINGS.breath)
    assert move.reason == "extended to the end of the utterance"
    assert move.words == "hello world"


def test_fit_end_pulls_back_to_a_pause_when_the_utterance_runs_past_the_cap():
    # two phrases (0.3 s apart) of one utterance; the second runs 18 s past the cut
    smap = _map((5, 8, "first bit"), (8.3, 30, "long rant"))
    move = fit_end(12.0, smap, SETTINGS, lo=4, hi=100)
    assert move.new == pytest.approx(8.0 + 0.3 - 0.05)  # the breath stops short of the onset
    assert move.reason.startswith("pulled back to a pause; utterance runs 18.0s past the cut")
    assert "(> 8s cap)" in move.reason


def test_fit_end_blocked_by_the_next_clip_leaves_continuous_speech_to_the_fade():
    move = fit_end(12.0, _map((10, 14, "hello world")), SETTINGS, lo=11.5, hi=13.0)
    assert move.new == 12.0
    assert move.reason.startswith("continuous speech, left to the audio fade")
    assert "(source end / unreadable region / next clip)" in move.reason


def test_fit_start_moves_back_to_the_start_of_the_utterance():
    move = fit_start(11.0, _map((10, 14, "hello world")), SETTINGS, lo=0, hi=100)
    assert move.new == pytest.approx(10.0 - SETTINGS.lead_breath)
    assert move.reason == "moved back to the start of the utterance"
    assert fit_start(20.0, _map((10, 14, "x")), SETTINGS, lo=0, hi=100).reason == "not in speech"


def test_fit_start_never_moves_before_the_previous_clip():
    move = fit_start(11.0, _map((10, 14, "hello world")), SETTINGS, lo=10.5, hi=100)
    assert move.new == 11.0
    assert move.reason == (
        "mid-utterance start, no boundary within reach (utterance began 1.0s earlier)"
    )


def test_fit_start_moves_forward_to_a_nearby_sentence_start():
    # the utterance began 6 s earlier (past max_lead); the next sentence starts 0.5 s later
    smap = _map((5, 11.2, "earlier talk"), (11.5, 14, "new sentence"))
    move = fit_start(11.0, smap, SETTINGS, lo=0, hi=100)
    assert move.new == pytest.approx(11.5 - 0.15)
    assert move.reason == "moved forward to the next sentence start"


# ------------------------------------------------------------------- separate
def _clip(name: str, start: float, end: float, keep: tuple[float, float], key: str = "a.mp4",
          smap: SpeechMap | None = None) -> ClipBounds:  # fmt: skip
    return ClipBounds(name, key, start, end, keep[0], keep[1], 0.0, 100.0, smap)


def test_separate_cuts_overlapping_padding_in_the_longest_silence():
    smap = _map((20.2, 21.0, "hey"))
    a = _clip("001_a.mp4", 8, 24, (10, 20), smap=smap)
    b = _clip("002_b.mp4", 20, 32, (22, 30), smap=smap)
    other = _clip("003_c.mp4", 8, 24, (10, 20), key="b.mp4")  # another source: untouched

    separate([b, a, other])

    assert a.end == b.start == pytest.approx(21.5)  # middle of 21.0-22.0, the longest gap
    assert a.record["neighbour_end"]["reason"] == "padding overlapped"
    assert b.record["neighbour_start"]["with"] == "001_a.mp4"
    assert (other.start, other.end, other.record) == (8, 24, {})


def test_separate_splits_overlapping_picks_in_the_middle():
    a = _clip("001_a.mp4", 9, 21, (10, 20))
    b = _clip("002_b.mp4", 17, 26, (18, 25))
    separate([a, b])
    assert a.end == b.start == pytest.approx(19.0)
    assert a.record["neighbour_end"]["reason"] == "picked moments overlap"


# ------------------------------------------------------------------- apply_overrides
@pytest.fixture
def plans(monkeypatch) -> list[ExtendPlan]:
    monkeypatch.setattr(planning.sources, "readable_range", lambda src, t, d: (0.0, d))
    return [
        ExtendPlan(name, {"start": 10.0, "end": 20.0}, Path("v.mp4"), media_info(),
                   10.0, 20.0, {}, (10.0, 20.0))
        for name in ("001_Goal.mp4", "002_crowd.mp4")
    ]  # fmt: skip


def test_clip_override_speech_caps_go_to_the_matching_clip_only(plans):
    apply_overrides(plans, {"001_GOAL*": {"speech_max_extend": 30}})
    assert plans[0].entry["speech_max_extend"] == 30
    assert "speech_max_extend" not in plans[1].entry
    assert (plans[0].start, plans[0].end) == (10.0, 20.0)  # no padding key: not re-padded


def test_clip_override_padding_replaces_the_runs_padding_per_side(plans):
    run_padding = planning.Padding(1.0, 1.0)
    apply_overrides(plans, {"001*": {"extend": 2}, "002*": {"extend_end": 3}}, run_padding)
    assert (plans[0].start, plans[0].end) == (8.0, 22.0)
    assert plans[0].rec["override"] == {"extend": 2.0}
    assert (plans[1].start, plans[1].end) == (9.0, 23.0)  # the start keeps the run's 1 s
    assert plans[1].rec["override"] == {"extend_end": 3.0}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"001*": {"speech_max_extnd": 30}}, "unknown key"),
        ({"009*": {"extend": 1}}, "matches no clip"),
    ],
)
def test_clip_override_mistakes_are_refused(plans, overrides, message):
    with pytest.raises(ValueError, match=message):
        apply_overrides(plans, overrides)
