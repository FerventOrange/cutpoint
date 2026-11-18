"""AI Chat's English-only intent readers (:mod:`cutpoint.assistant.intent`), one table per
function: the length a message asks for, the lengths a proposal's summary claims or names, a
reply that promises a lookup, and a message that asks for a mix of kinds or for dialogue.

Tests that read them through a turn stay with that behaviour (``test_assistant_length``,
``test_assistant_promise``, ``test_assistant_pacing``, ``test_assistant_routes``).
"""

from __future__ import annotations

import time

import pytest

from assistant_helpers import CLAIMS, TURN
from cutpoint.assistant import intent

# -- the length the user asks for ---------------------------------------------------------------


# One row per branch of ``intent.target_seconds``: each cue before or after a length, a clock
# time, minutes with seconds, a count word, and each way a length is a clip's, a change or a place.
TARGETS = [
    # a cue before the length
    ("get it to 4 minutes", 240),
    ("cut it to 90 s", 90),
    ("aim for 2 minutes", 120),
    ("target of 150 s", 150),
    ("total of 240 s", 240),
    ("~4 min of beauty shots", 240),
    ("about two minutes of skiing", 120),
    ("make it a minute and 4 s", 64),
    ("about 2 minutes and 15 seconds", 135),
    ("make it 4 min 30 s", 270),
    ("the video should be 2.5 minutes", 150),
    ("extend m3 to 60 s, make the video 4 minutes", 240),
    ("clips around 2 minutes", 120),
    ("make it about 4 minutes more or less", 240),
    # a cue after the length
    ("a 90 second reel", 90),
    ("a 4 minute long montage", 240),
    ("one minute overall", 60),
    ("remove m3, about 2 min total", 120),
    # a clock time
    ("make it 1:30", 90),
    ("keep it to 1:30", 90),
    ("make the video 3:30", 210),
    ("1:30 total", 90),
    # a length per clip first, then the sequence's
    ("tighten every clip and aim for a 2 minute montage", 120),
    ("shorten every clip so the video is about 2 min", 120),
    ("trim each clip, the whole thing under 2 minutes", 120),
    ("tighten every clip, total around 3 min", 180),
    ("cut each clip down and keep it under 90 seconds", 90),
    ("every moment about 6 s, cut to 90 s", 90),
    ("every clip around 5 s overall 2 min", 120),
    ("shorten each clip. under 2 minutes", 120),
    ("tighten each clip! around 3 min", 180),
    ("tighten each clip; around 3 min", 180),
    # a length per clip is none
    ("each of the highlights about 5 s", None),
    ("each edit about 4 s", None),
    ("make each highlight about 5 seconds", None),
    ("clip about 5 s", None),
    ("per clip: about 1 min", None),
    ("about 5 s each", None),
    ("about 1 min per clip", None),
    ("around 1 minute a clip", None),
    ("I want 10s clips", None),
    ("I want clips of about 5 seconds", None),
    ("make the clips about 5 s long", None),
    # nor is a change, a place, a moment's length or "as"
    ("need 30 s more", None),
    ("make it 5 min longer", None),
    ("extend the jump by about 10 s", None),
    ("extend m3 by 2 minutes", None),
    ("m3, make it 8 seconds", None),
    ("remove the clip at 3 minutes", None),
    ("move the 2 minute clip", None),
    ("what happens at 1:30?", None),
    ("around 4:20", None),
    ("make it as short as possible", None),
]


@pytest.mark.parametrize(("text", "seconds"), TARGETS, ids=[text for text, _ in TARGETS])
def test_the_length_the_user_asks_for_is_read_from_the_message(
    text: str, seconds: float | None
) -> None:
    assert intent.target_seconds(text) == seconds


def test_a_long_message_with_many_lengths_is_parsed_fast() -> None:
    text = "each clip about 5 s " * 1000 + "make it 2 min"
    # this thread's CPU time: other processes on a loaded machine do not count, a scan per
    # match over the whole message does
    start = time.thread_time()
    assert intent.target_seconds(text) == 120
    assert intent.target_seconds("about 5 s, " * 2000) == 5
    assert time.thread_time() - start < 2.0


# -- the length a proposal's summary claims -----------------------------------------------------


@pytest.mark.parametrize(
    ("summary", "claimed"),
    [(summary, claimed) for summary, claimed, _ in CLAIMS],
    ids=[summary for summary, *_ in CLAIMS],
)
def test_a_length_claim_in_a_summary_is_read(summary: str, claimed: float | None) -> None:
    assert intent.claimed_seconds(summary) == claimed


# One row per kind of length ``intent.named_seconds`` reads: lengths tied to a cue or not, a clock
# time, the length before a change (left out) and a summary with none.
NAMED = [
    ("Added the laugh; now 90 s, after 24 s.", [90.0, 24.0]),
    ("It now runs 1:30.", [90.0]),
    ("Took it from 34 s to 60 s.", [60.0]),
    ("Moved the laugh to the start.", []),
]


@pytest.mark.parametrize(("summary", "seconds"), NAMED, ids=[summary for summary, _ in NAMED])
def test_every_length_a_summary_names_is_read(summary: str, seconds: list[float]) -> None:
    assert intent.named_seconds(summary) == seconds


# -- a reply that promises a lookup -------------------------------------------------------------


# One row per branch of ``intent.promises``: each lookup verb form, a gerund, a moment asked
# for, a trailing colon; and each way a reply that looks like one answers, offers or asks instead.
PROMISES = [
    ("Searching now.", True),
    ("Checking the transcript...", True),
    ("Looking for the rest…", True),
    ("Let's search for each one.", True),
    ("I need to search for each.", True),
    ("Let me do that.", True),
    ("Okay, I'm going to check the transcript", True),
    ("I'm gonna look at the jump again", True),
    ("I'll look for the other words next.", True),
    ("Done with the first. I\N{RIGHT SINGLE QUOTATION MARK}ll search for the rest.", True),
    ("Let me check the moment at 12:30.", True),
    ("Let me check if she says it.", True),
    ("Let me check for the other terms: grifter, loser, and curse words.", True),
    ("Here are the hits:", True),
    ("Give me a moment.", True),
    ("Just a sec.", True),
    ("One moment.", True),
    # an answer, an offer, a question or another verb is none
    ("Looking at the transcript, she never says 'grifter'.", False),
    ("A moment later, she laughs.", False),
    ("Let's look at it another way: the jump is the best part.", False),
    ("Let me check: nothing there.", False),
    ("Let me check: no hits.", False),
    ("I'll try to keep it under 60 s next time.", False),
    ("I'll find a better transition if the fade feels off.", False),
    ("I'll check later.", False),
    ("Let me know if you want more.", False),
    ("Should I search for more?", False),
    ("Let's call it a draw.", False),
    ("I'll get back to you", False),
]


@pytest.mark.parametrize(("text", "promise"), PROMISES, ids=[text for text, _ in PROMISES])
def test_a_reply_is_a_promise_only_when_it_ends_on_a_lookup_still_to_make(
    text: str, promise: bool
) -> None:
    assert intent.promises(text) is promise


# -- a mix of kinds and dialogue ----------------------------------------------------------------


# One row per branch of ``intent.asks_for_mix``: each mix wording, "between the <any word> runs"
# and "between the action clips" (no activity is named, #1085 A303), a negation or transition before it, and each wording that only looks like a mix.
MIXES = [
    (TURN, True),
    ("a good mix of skiing and views", True),
    ("not too long, mix in some views", True),
    ("No music. Break up the skiing with laughs", True),
    ("don't cut it short; break up the skiing", True),
    ("sprinkle in a few funny clips", True),
    ("add a funny moment now and then", True),
    ("show a funny moment from time to time", True),
    ("alternate skiing and scenery", True),
    ("vary the pace please", True),
    ("an occasional joke", True),
    ("between the mtb runs add a funny moment", True),
    ("add dialogue between the snowboard runs", True),
    ("put a funny moment between the action clips", True),
    ("some laughs in between the action", True),
    ("trim the dead time between the action", False),  # a bare "between the action"
    # an audio mix, a negated request, a transition or another word is none
    ("mix the audio", False),
    ("mix in some music", False),
    ("mix in the background audio", False),
    ("don't break up the skiing", False),
    ("keep the skiing together, no alternating", False),
    ("without breaking up the runs", False),
    ("never intersperse dialogue", False),
    ("add a fade in between the clips", False),
    ("add a crossfade between the runs", False),
    ("put a transition between each run", False),
    ("use an alternate take of clip 3", False),
    ("an alternative cut", False),
    ("the occasional stutter in clip 4", False),
    ("break up the long clip into two", False),
    ("between the trailer and the intro", False),
    ("between the surface shots", False),
]


@pytest.mark.parametrize(("text", "wanted"), MIXES, ids=[text for text, _ in MIXES])
def test_a_mix_request_is_found_and_look_alike_wording_is_not(text: str, wanted: bool) -> None:
    assert intent.asks_for_mix(text) is wanted


# One row per branch of ``intent.names_dialogue``: each dialogue word, a negation before it, and
# "talk" or "chat" that is not a kind of moment.
DIALOGUE = [
    (TURN, True),
    ("some banter", True),
    ("add talking moments", True),
    ("some talk between the runs", True),
    ("not too long, with dialogue to break up the runs", True),
    ("the commentary", True),
    ("a conversation", True),
    ("chat clips", True),
    ("no music. add some dialogue", True),
    ("a funny moment", False),
    ("no dialogue, break up the skiing", False),
    ("never add banter", False),
    ("without any commentary", False),
    ("the talker is wrong", False),
    ("let's talk about the order", False),
]


@pytest.mark.parametrize(("text", "named"), DIALOGUE, ids=[text for text, _ in DIALOGUE])
def test_dialogue_is_named_by_its_words(text: str, named: bool) -> None:
    assert intent.names_dialogue(text) is named
