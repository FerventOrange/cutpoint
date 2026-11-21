"""The transcript's laughter: :func:`cutpoint.transcribe.laughter_hits` and
:func:`cutpoint.transcribe.is_laugh` (synthetic words only)."""

from __future__ import annotations

import pytest

from cutpoint.styles import LAUGHTER
from cutpoint.transcribe import is_laugh, keyword_hits, laughter_hits


def _said(*pieces: str, step: float = 1.0) -> list[dict]:
    """Transcript words, one per second, as Whisper writes them (a leading space each)."""
    return [
        {"word": f" {p}", "start": i * step, "end": i * step + 0.5} for i, p in enumerate(pieces)
    ]


@pytest.mark.parametrize(
    "marker", ["[laughter]", "(laughs)", "*laughing*", "[LAUGHTER]", "(Laughing)", "[giggles]"]
)
def test_whisper_laughter_markers_are_laughter_hits(marker):
    (hit,) = laughter_hits(_said("so", marker, "anyway"))
    assert hit[0] == 1.0 and hit[1] == 1.0
    assert hit[2] == marker[1:-1].lower()


def test_a_marker_of_several_words_is_one_hit_at_its_first_word():
    (hit,) = laughter_hits(_said("ok", "[light", "laughter]", "right"))
    assert hit == (1.0, 1.0, "light laughter")


@pytest.mark.parametrize(
    "words",
    [
        ("he", "said", "laughter", "is", "funny"),
        ("stop", "laughing", "at", "me"),
        ("i", "laugh", "every", "time"),
        ("Laughter,", "he", "says."),
        ("the", "(laughing", "stock", "of", "town"),  # no closing mark
        ("what", "the", "f***", "are", "you", "laughing", "at", "s***"),  # censored profanity
        ("so", "(laughs]", "anyway"),  # mismatched delimiters
        ("so", "[laughs)", "anyway"),
    ],
)
def test_the_spoken_word_laugh_or_laughter_is_not_a_laughter_hit(words):
    assert laughter_hits(_said(*words)) == []


def test_each_laughter_word_counts_once():
    hits = laughter_hits(_said("haha", "hah", "hahaha", "hehe", "lmao", "lol"))
    assert [(t, p) for t, _, p in hits] == [
        (0.0, "haha"),
        (1.0, "hah"),
        (2.0, "haha"),
        (3.0, "hehe"),
        (4.0, "lmao"),
        (5.0, "lol"),
    ]
    assert [w for _, w, _ in hits] == [1.0, 0.8, 1.0, 0.8, 1.0, 0.8]  # the strongest match


def test_keyword_hits_count_every_match_unless_asked_for_one():
    assert len(keyword_hits(_said("haha"), LAUGHTER)) == 2
    assert len(keyword_hits(_said("haha"), LAUGHTER, once=True)) == 1


def test_laughing_interjections_still_count_beside_the_markers():
    hits = laughter_hits(_said("haha", "no", "[laughs]", "lol"))
    assert {(t, p) for t, _, p in hits} >= {(0.0, "haha"), (2.0, "laughs"), (3.0, "lol")}


@pytest.mark.parametrize("word", ["laugh", "laughing", " Laughs,"])
def test_the_plain_word_laugh_is_a_laugh_only_with_spoken_laugh(word):
    assert is_laugh(word)
    assert is_laugh(word, spoken_laugh=True)
    assert not is_laugh(word, spoken_laugh=False)
    assert laughter_hits(_said("so", word.strip(), "anyway")) == []


@pytest.mark.parametrize("marker", ["[laughter]", "(laughs)", "*laughing*", "[light laughter]"])
def test_the_laughter_markers_count_in_both_modes(marker):
    # is_laugh reads one word: its stem after the brackets are dropped ("[light" is none)
    assert all(is_laugh(w) == ("laugh" in w) for w in marker.split())
    (hit,) = laughter_hits(_said("so", *marker.split(), "anyway"))
    assert hit == (1.0, 1.0, marker[1:-1])


@pytest.mark.parametrize(
    ("word", "weight"),
    [("haha", 1.0), ("hahaha", 1.0), ("lol", 0.8), ("lmao", 1.0), ("dying", 0.6)],
)
def test_the_interjections_count_in_both_modes(word, weight):
    assert is_laugh(word) and is_laugh(word, spoken_laugh=False)
    assert laughter_hits(_said("ok", word)) == [
        (1.0, weight, word[:4] if word.startswith("haha") else word)
    ]


@pytest.mark.parametrize("word", ["blorp", "", "...", "[music]"])
def test_a_word_without_a_laughter_stem_counts_in_neither_mode(word):
    assert not is_laugh(word) and not is_laugh(word, spoken_laugh=False)
    assert laughter_hits(_said("ok", word)) == []


def test_marker_stems_count_only_inside_brackets_for_is_laugh_too():
    """``giggl``/``chuckl`` are marker stems of laughter_hits, no is_laugh stems."""
    assert laughter_hits(_said("[giggles]")) == [(0.0, 1.0, "giggles")]
    assert not is_laugh("giggles") and not is_laugh("[chuckles]")
