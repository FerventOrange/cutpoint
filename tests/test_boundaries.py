"""``boundaries.build_speech_map``: the speech map of one source from its words, voice grids,
laughter events and telemetry speed (#1085 A493). Only the cache readers are replaced."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cutpoint import telemetry
from cutpoint.boundaries import SpeechSettings, build_speech_map
from cutpoint.events import voice as VO

SRC = Path("clip.mp4")
SETTINGS = SpeechSettings()
N = 300  # 30 s on the 0.1 s grid


def word(start, end, text="hello", logprob=None):
    w = {"start": start, "end": end, "word": f" {text}"}
    if logprob is not None:
        w["avg_logprob"] = logprob
    return w


def grids(voiced=(), *, n=N):
    """Voice grids: speech (VAD and speech detectors agree) in the ``voiced`` (a, b) seconds,
    nothing else, no wind."""
    g = {
        k: np.zeros(n, np.float32) for k in ("vad", "hp_speech", "hp_laugh", "hp_cheer", "hp_boom")
    }
    g["wind_db"] = np.full(n, -120.0, np.float32)
    for a, b in voiced:
        g["vad"][round(a * 10) : round(b * 10)] = 1.0
        g["hp_speech"][round(a * 10) : round(b * 10)] = 1.0
    return g


@pytest.fixture
def cached(monkeypatch):
    """Set the cached voice grids and speed the map reads."""
    state = {"grids": None, "speed": None}
    monkeypatch.setattr(VO, "load_voice", lambda src: state["grids"])
    monkeypatch.setattr(telemetry, "cached_speed", lambda src: state["speed"])
    return state


def spans(m):
    return [(round(s.start, 2), round(s.end, 2), sorted(s.kinds), s.text) for s in m.spans]


def test_no_words_voice_or_events_is_no_map(cached):
    assert build_speech_map(SRC, None, None, SETTINGS) is None
    assert build_speech_map(SRC, [], [], SETTINGS) is None


def test_words_make_word_spans_and_laughter_tokens_make_laugh_spans(cached):
    m = build_speech_map(SRC, [word(1.0, 1.5, "so"), word(2.0, 2.4, "hahaha")], None, SETTINGS)
    assert spans(m) == [(1.0, 1.5, ["word"], "so"), (2.0, 2.4, ["laugh"], "hahaha")]
    # a zero-length word still gets a span
    tiny = build_speech_map(SRC, [word(5.0, 5.0)], None, SETTINGS)
    assert spans(tiny) == [(5.0, 5.05, ["word"], "hello")]


def test_a_stretched_word_is_trimmed_to_the_voiced_part_or_its_last_1_2_s(cached):
    cached["grids"] = grids([(6.0, 6.8)])
    m = build_speech_map(SRC, [word(2.0, 7.0, "well")], None, SETTINGS)
    word_span = next(s for s in m.spans if "word" in s.kinds)
    assert (word_span.start, word_span.end) == (pytest.approx(6.0), pytest.approx(6.8))
    cached["grids"] = None
    m = build_speech_map(SRC, [word(2.0, 7.0, "well")], None, SETTINGS)
    assert spans(m) == [(5.8, 7.0, ["word"], "well")]


def test_voice_runs_become_voice_spans_and_short_blips_are_ignored(cached):
    cached["grids"] = grids([(3.0, 4.0), (9.0, 9.1)])  # a second of speech, a 0.1 s blip
    m = build_speech_map(SRC, None, None, SETTINGS)
    assert [(a, b, k) for a, b, k, _ in spans(m)] == [(3.0, 4.0, ["voice"])]


def test_voice_layer_laughter_and_laughter_events_make_laugh_spans(cached):
    g = grids()
    g["hp_laugh"][100:130] = 1.0  # loud laughter at 10 to 13 s
    cached["grids"] = g
    events = [
        {"kind": "audio", "label": "Laughter", "confidence": 0.9, "start": 20.0, "end": 21.0},
        {"kind": "audio", "label": "Laughter", "confidence": 0.9, "start": 25.0},  # no end
        {"kind": "audio", "label": "Laughter", "confidence": 0.01, "start": 26.0},  # too weak
        {"kind": "audio", "label": "Speech", "confidence": 0.9, "start": 27.0},
    ]
    m = build_speech_map(SRC, None, events, SETTINGS)
    laugh = [(a, b) for a, b, k, _ in spans(m) if k == ["laugh"]]
    assert len(laugh) == 3
    assert 9.5 <= laugh[0][0] <= 10.5 and 12.5 <= laugh[0][1] <= 13.5
    assert laugh[1:] == [(20.0, 21.0), (25.0, 25.5)]


def test_a_unreadable_voice_grid_falls_back_to_the_words(cached, monkeypatch, caplog):
    def broken(src):
        raise OSError("disk gone")

    monkeypatch.setattr(VO, "load_voice", broken)
    m = build_speech_map(SRC, [word(1.0, 1.5)], None, SETTINGS)
    assert spans(m) == [(1.0, 1.5, ["word"], "hello")]
    assert "unreadable voice grid" in caplog.text and "clip.mp4" in caplog.text


def test_in_wind_only_confident_words_count(cached):
    speed = np.full(N, 5.0)
    speed[100:150] = 50.0  # 10 to 15 s: faster than wind_kmh (20)
    cached["speed"] = speed
    words = [
        word(2.0, 2.4, "calm", logprob=-2.0),  # not windy: any confidence counts
        word(11.0, 11.4, "sure", logprob=-0.2),  # windy but confident
        word(12.0, 12.4, "guess", logprob=-1.5),  # windy and unsure: dropped
        word(13.0, 13.4, "unknown"),  # windy and no confidence recorded: dropped
        word(17.0, 17.4, "after", logprob=-2.0),  # the gust is over
    ]
    m = build_speech_map(SRC, words, None, SETTINGS)
    assert [s.text for s in m.spans] == ["calm", "sure", "after"]
    lax = SpeechSettings(wind_kmh=60.0)  # the same speed is no wind for a looser setting
    assert len(build_speech_map(SRC, words, None, lax).spans) == 5


def test_a_gps_gap_in_the_speed_is_not_wind_and_an_empty_speed_is_ignored(cached):
    speed = np.full(N, 5.0)
    speed[100:150] = np.nan
    cached["speed"] = speed
    words = [word(11.0, 11.4, "gap", logprob=-3.0)]
    assert [s.text for s in build_speech_map(SRC, words, None, SETTINGS).spans] == ["gap"]
    cached["speed"] = np.zeros(0)
    assert len(build_speech_map(SRC, words, None, SETTINGS).spans) == 1


def test_in_wind_the_voice_runs_are_masked(cached):
    cached["grids"] = grids([(3.0, 4.0), (11.0, 12.0)])
    speed = np.full(N, 5.0)
    speed[100:150] = 50.0
    cached["speed"] = speed
    m = build_speech_map(SRC, None, None, SETTINGS)
    assert [(a, b, k) for a, b, k, _ in spans(m)] == [(3.0, 4.0, ["voice"])]


def test_the_map_joins_words_into_phrases_and_utterances(cached):
    m = build_speech_map(
        SRC, [word(1.0, 1.4), word(1.5, 1.9), word(2.4, 2.8), word(10.0, 10.4)], None, SETTINGS
    )
    assert [(p.start, p.end) for p in m.phrases] == [(1.0, 1.9), (2.4, 2.8), (10.0, 10.4)]
    assert [(c.start, c.end) for c in m.chains] == [(1.0, 2.8), (10.0, 10.4)]
