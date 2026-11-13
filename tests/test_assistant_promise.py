"""AI Chat: a reply that promises a lookup and makes no call ("I'll search each one now.") gets
one nudge (#1013), natively and in JSON mode; the turn never ends on a promise it did not keep.

The model is the scripted ``FakeProvider`` of ``test_assistant``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("httpx")

from cutpoint.assistant import session
from cutpoint.assistant.providers import Capabilities
from test_assistant import FakeProvider, Reply, call, run
from test_assistant_words import chat_project

ISSUE = (
    "Let me check for the other terms: grifter, loser, and curse words. I'll search each one now."
)
ASK = "find the other terms"
NATIVE = Capabilities(True, True)
NO_TOOLS = Capabilities(False, True)


def say(text: str) -> Reply:
    return Reply(text, [], {"role": "assistant", "content": text})


def as_json(tool: str, **args: Any) -> Reply:
    return Reply(json.dumps({"tool": tool, "arguments": args}))


def said_json(text: str) -> Reply:
    return Reply(json.dumps({"tool": "reply", "text": text}))


def replies(turn: session.Turn) -> list[str]:
    return [e["text"] for e in turn.view()["events"] if e["kind"] == "reply"]


def sent(fake: FakeProvider, text: str = session.PROMISE_NUDGE) -> int:
    return sum(1 for r in fake.requests if r["messages"][-1] == {"role": "user", "content": text})


# -- the detector ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "promise"),
    [
        ("Searching now.", True),
        ("Checking the transcript...", True),
        ("Looking for the rest…", True),
        ("Let's search for each one.", True),
        ("I need to search for each.", True),
        ("Let me do that.", True),
        ("Give me a moment.", True),
        ("Looking at the transcript, she never says 'grifter'.", False),
        ("Searching all three words, only 'loser' appears, at 5 s in a.mp4.", False),
        ("Looking back, the best part is the jump.", False),
        ("She says it at 5 s. Looking at the rest, nothing else.", False),
        ("Looking for more if you want…", False),
        ("One moment stands out: the crash at 1:20.", False),
        ("The best is the crash. One moment has all three words.", False),
        ("One moment in a.mp4 has it, at 5 s.", False),
        ("A moment later, she laughs.", False),
        ("Just a sec.", True),
        ("Let me check: she says it at 12:30.", False),
        ("Let's look at it another way: the jump is the best part.", False),
        ("Let me check: m3 has it.", False),
        ("Let's call it a draw.", False),
        ("I searched for loser: no hits.", False),
        ("Let me check for the other terms: grifter, loser, and curse words.", True),
        ("Let me check the moment at 12:30.", True),
        ("Let's do it!", False),
        ("Let's try it.", False),
        ("Let's get started!", False),
        ("Let's run with that.", False),
        ("Let's view it in the player.", False),
        ("Let's try a shorter cut.", False),
        ("I'll find a better transition if the fade feels off.", False),
        ("I'll try to keep it under 60 s next time.", False),
        ("I'll check later.", False),
        ("Let me check: nothing there.", False),
        ("Let me check: no hits.", False),
        ("Let's look at the results: no hits for grifter or loser.", False),
        ("Looking through the transcript, I don't see 'grifter' anywhere...", False),
        ("Let me check if she says it.", True),
        ("Let me do that.", True),
        ("Searching for grifter, loser, and curse words...", True),
        ("Searching now...", True),
        ("One moment.", True),
        ("I'll look for the other words next.", True),
        ("Here are the hits:", True),
        ("Done with the first. I\N{RIGHT SINGLE QUOTATION MARK}ll search for the rest.", True),
        ("Now let me search for more beauty shots.", True),
        ("Okay, I'm going to check the transcript", True),
        ("Let me know if you want more.", False),
        ("I'll leave it there.", False),
        ("I'll search more if you want.", False),
        ("Should I search for more?", False),
        ("Let me check: she says it at 12:30, m3. That is the only one.", False),
        ("I'll get back to you", False),
        ("She says it twice, at 5 s and 9 s.", False),
    ],
)
def test_a_reply_is_a_promise_only_when_it_ends_on_a_lookup_still_to_make(
    text: str, promise: bool
) -> None:
    assert session._promises(text) is promise


# -- the nudge, natively and in JSON mode ----------------------------------------------------


def test_the_issue_case_native_searches_after_the_nudge_and_shows_the_answer() -> None:
    final = "She says it at 5 s in a.mp4."
    fake = FakeProvider([say(ISSUE), call("find_words", query="a"), say(final)], NATIVE)
    turn = run(fake, ASK, proj=chat_project())
    assert turn.status == "done", turn.error
    assert fake.requests[1]["messages"][-1] == {"role": "user", "content": session.PROMISE_NUDGE}
    assert sent(fake) == 1
    assert replies(turn) == [final]
    assert session.PROMISE_NUDGE not in json.dumps(turn.view()) + json.dumps(turn.record())
    assert turn.record()["reply"] == final


def test_the_issue_case_in_json_mode() -> None:
    final = "She says it at 5 s in a.mp4."
    fake = FakeProvider(
        [said_json(ISSUE), as_json("find_words", query="a"), said_json(final)], NO_TOOLS
    )
    turn = run(fake, ASK, proj=chat_project())
    assert turn.status == "done", turn.error
    assert fake.requests[1]["messages"][-1] == {"role": "user", "content": session.PROMISE_NUDGE}
    assert sent(fake) == 1
    assert replies(turn) == [final]
    assert session.PROMISE_NUDGE not in json.dumps(turn.view()) + json.dumps(turn.record())


def test_a_reply_that_announces_a_search_to_a_question_gets_the_promise_nudge() -> None:
    fake = FakeProvider(
        [say("Now let me search for more beauty shots."), say("It shows a jump and a crash.")]
    )
    turn = run(fake, "what is in this video?")
    assert fake.requests[1]["messages"][-1]["content"] == session.PROMISE_NUDGE
    assert replies(turn) == ["It shows a jump and a crash."]


@pytest.mark.parametrize("json_mode", [False, True])
def test_a_reply_that_is_no_promise_is_not_nudged(json_mode: bool) -> None:
    text = "Let me know if you want more."
    fake = FakeProvider(
        [said_json(text)] if json_mode else [say(text)], NO_TOOLS if json_mode else NATIVE
    )
    turn = run(fake, ASK)
    assert len(fake.requests) == 1 and replies(turn) == [text]


# -- the bound: one nudge a turn, never ending on a promise ----------------------------------


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("again", ["promise", "empty"])
def test_after_the_nudge_the_models_own_reply_stands_whatever_it_is(
    json_mode: bool, again: str
) -> None:
    mk = said_json if json_mode else say
    second = "Searching again now..." if again == "promise" else ""
    fake = FakeProvider([mk(ISSUE), mk(second)], NO_TOOLS if json_mode else NATIVE)
    turn = run(fake, ASK, proj=chat_project())
    assert turn.status == "done", turn.error
    assert sent(fake) == 1  # an empty JSON reply is asked again as malformed: no second nudge
    assert json_mode or len(fake.requests) == 2
    # a promise again is shown as it is; an empty reply leaves the nudged one
    assert replies(turn) == [second or ISSUE]


def test_searches_after_the_nudge_that_end_on_a_promise_get_no_second_nudge() -> None:
    last = "I'll search each one now."
    fake = FakeProvider(
        [say(ISSUE), call("find_words", query="a"), call("find_words", query="am"), say(last)],
        NATIVE,
    )
    turn = run(fake, ASK, proj=chat_project())
    assert turn.status == "done", turn.error
    assert sent(fake) == 1 and len(fake.requests) == 4
    assert replies(turn) == [last]


def test_no_promise_nudge_in_the_closing_round() -> None:
    script = [call("get_moment", id="m1") for _ in range(session.MAX_ROUNDS - 1)]
    fake = FakeProvider([*script, say(ISSUE)], NATIVE)
    turn = run(fake, ASK)
    assert turn.status == "done", turn.error
    assert sent(fake) == 0
    assert replies(turn) == [ISSUE]  # shown as the model wrote it


def test_a_closing_round_answer_with_a_lets_run_sentence_is_shown_unchanged() -> None:
    final = "The crash at 1:20 is the best opener. Let's run with that."
    script = [call("get_moment", id="m1") for _ in range(session.MAX_ROUNDS - 1)]
    turn = run(FakeProvider([*script, say(final)], NATIVE), ASK)
    assert replies(turn) == [final]


def test_an_answer_in_the_closing_round_that_only_starts_like_a_promise_is_shown() -> None:
    final = "One moment stands out: the crash at 1:20."
    script = [call("get_moment", id="m1") for _ in range(session.MAX_ROUNDS - 1)]
    fake = FakeProvider([*script, say(final)], NATIVE)
    turn = run(fake, ASK)
    assert turn.status == "done", turn.error
    assert replies(turn) == [final]


def test_an_answer_starting_with_looking_after_the_nudge_is_shown() -> None:
    final = "Looking at the transcript, she never says 'grifter'."
    fake = FakeProvider([say(ISSUE), say(final)], NATIVE)
    turn = run(fake, ASK, proj=chat_project())
    assert turn.status == "done", turn.error
    assert sent(fake) == 1 and replies(turn) == [final]


# -- the older triggers keep their priority ---------------------------------------------------


def test_an_edit_request_answered_with_an_announce_gets_the_edit_nudge() -> None:
    said = say("Now let me search for more beauty shots.")
    ok = call("propose_edit", summary="Add the laugh", ops=[{"op": "add", "id": "m1"}])
    fake = FakeProvider([said, ok])
    turn = run(fake, "add some beauty shots")
    assert fake.requests[1]["messages"][-1]["content"] == session.NUDGE
    assert sent(fake) == 0 and [e["kind"] for e in turn.view()["events"]] == ["proposal"]
    # the second announce is then the answer: no promise nudge on top
    fake = FakeProvider([said, said])
    turn = run(fake, "add some beauty shots")
    assert sent(fake) == 0 and replies(turn) == [said.text]
