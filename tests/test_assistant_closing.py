"""The AI Chat's closing round (#988): once the rounds or the views are spent, the model is
asked for its answer with only ``propose_edit`` offered; a model that keeps looking things up
gets that round once more, and the turn then ends with an answer, never "no answer".

The model is a :class:`FakeProvider`; the hosted adapters run over the ``Wire`` stub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cutpoint.assistant import providers, session, watch
from cutpoint.assistant.providers import AnthropicProvider, OpenAIProvider, Reply
from test_assistant import FakeProvider, _Resp, call, project, run
from test_assistant_byo import KEY, Wire, an_reply, rs_call, rs_reply, rs_text, tool_use
from test_assistant_byo import no_keys as no_keys  # the autouse fixture

LAST = session.MAX_ROUNDS - 1  # the index of the closing round's request
ASK = "Create a montage of the large manta rays."
JSON_ONLY = providers.Capabilities(False, True)
FOUND_NONE = "I couldn't find manta rays in these moments; they show a jump and a crash."


def answer(text: str) -> Reply:
    return Reply(text, [], {"role": "assistant", "content": text})


def as_json(tool: str, **args: Any) -> Reply:
    return Reply(json.dumps({"tool": tool, "arguments": args}))


def lookups(n: int, name: str = "find_moments") -> list[Reply]:
    if name == "find_moments":
        return [call(name, query=f"manta {i}") for i in range(n)]
    return [call(name, id=f"m{1 + i % 3}") for i in range(n)]


def tool_names(request: dict[str, Any]) -> list[str]:
    return [t["function"]["name"] for t in request["tools"] or []]


def closing_noted(request: dict[str, Any]) -> bool:
    return any(m.get("content") == session.CLOSING for m in request["messages"])


def results(request: dict[str, Any]) -> list[str]:
    return [m["content"] for m in request["messages"] if m.get("role") == "tool"]


def replies(turn: session.Turn) -> list[str]:
    return [e["text"] for e in turn.events if e["kind"] == "reply"]


# -- the closing round --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["find_moments", "get_moment"])
def test_a_model_that_keeps_looking_ends_with_a_plain_answer_not_no_answer(name: str) -> None:
    fake = FakeProvider(lookups(session.MAX_ROUNDS + 1, name))
    turn = run(fake, ASK)
    assert turn.status == "done" and turn.error is None
    assert len(fake.requests) == session.MAX_ROUNDS + 1  # the closing round once more, no more
    for r in fake.requests[:LAST]:
        assert "view_moment" not in tool_names(r) and "find_moments" in tool_names(r)
        assert not closing_noted(r)
    for r in fake.requests[LAST:]:  # the closing round and its one retry
        assert tool_names(r) == [session.FINAL_TOOL] and closing_noted(r)
    # the closing round's lookup was not run: its result says to answer
    assert results(fake.requests[-1])[-1] == session.NOT_RUN
    ran = [e for e in turn.events if e["kind"] == "tool"]
    assert len(ran) == LAST
    (text,) = replies(turn)
    assert text.startswith("I couldn't find an answer to this in the moments")
    assert "montage of the best moments" in text
    if name == "find_moments":
        assert "searched for 'manta 0', 'manta 1', 'manta 2', 'manta 3'" in text
        assert "manta 4" not in text  # at most MAX_LOOKED searches named
    assert turn.record()["reply"] == text


def test_the_closing_round_answer_in_text_is_the_reply_and_gets_no_nudge() -> None:
    fake = FakeProvider([*lookups(LAST), answer(FOUND_NONE)])
    turn = run(fake, ASK)  # an edit asked for: a text answer would be nudged before the closing
    assert turn.status == "done", turn.error
    assert len(fake.requests) == session.MAX_ROUNDS
    assert replies(turn) == [FOUND_NONE]
    assert not any(m.get("content") == session.NUDGE for m in fake.requests[-1]["messages"])


def test_a_lookup_in_the_closing_round_is_not_run_and_the_round_is_asked_once_more() -> None:
    fake = FakeProvider([*lookups(session.MAX_ROUNDS), answer(FOUND_NONE)])
    turn = run(fake, ASK)
    assert turn.status == "done", turn.error
    assert len(fake.requests) == session.MAX_ROUNDS + 1
    assert results(fake.requests[-1])[-1] == session.NOT_RUN
    assert replies(turn) == [FOUND_NONE]


def test_an_empty_closing_answer_is_asked_once_more_not_an_error() -> None:
    fake = FakeProvider([*lookups(LAST), answer(""), answer("")])
    turn = run(fake, ASK)
    assert turn.status == "done" and turn.error is None
    assert len(fake.requests) == session.MAX_ROUNDS + 1
    assert replies(turn)[0].startswith("I couldn't find an answer")


def test_the_closing_round_may_still_propose() -> None:
    edit = {"summary": "Open on the laugh", "ops": [{"op": "add", "id": "m1", "at": 1}]}
    fake = FakeProvider([*lookups(LAST), call("propose_edit", **edit)])
    turn = run(fake, "build a montage")
    assert turn.status == "done", turn.error
    assert [e["kind"] for e in turn.events if e["kind"] != "tool"] == ["proposal"]


def test_rejected_proposals_in_the_closing_round_end_with_the_rejection_not_a_search() -> None:
    bad = {"summary": "Add it", "ops": [{"op": "add", "id": "m99"}]}
    fake = FakeProvider([*lookups(LAST), call("propose_edit", **bad), call("propose_edit", **bad)])
    turn = run(fake, ASK)
    assert turn.status == "done" and turn.error is None
    assert len(fake.requests) == session.MAX_ROUNDS + 1
    rejected = [e["text"] for e in turn.events if e["kind"] == "rejected"]
    assert len(rejected) == 2
    (text,) = replies(turn)
    assert text.startswith("I couldn't make an edit that passes the checks")
    assert rejected[-1] in text and "searched for" not in text


def test_a_nudged_answer_still_stands_when_the_rounds_run_out() -> None:
    first = "I would take m1 and m2 first."
    fake = FakeProvider([answer(first), *lookups(session.MAX_ROUNDS + 1)])
    turn = run(fake, "build a montage")
    assert turn.status == "done" and replies(turn) == [first]


def test_json_mode_closing_round_allows_only_propose_edit_and_reply() -> None:
    script = [as_json("find_moments", query=f"manta {i}") for i in range(session.MAX_ROUNDS + 1)]
    fake = FakeProvider(script, JSON_ONLY)
    turn = run(fake, ASK)
    assert turn.status == "done" and turn.error is None
    assert len(fake.requests) == session.MAX_ROUNDS + 1
    assert fake.requests[0]["schema"] is session.JSON_SCHEMA
    for r in fake.requests[LAST:]:
        assert r["schema"]["properties"]["tool"]["enum"] == [session.FINAL_TOOL, "reply"]
        assert any(m.get("content") == session.CLOSING_JSON for m in r["messages"])
    sent = fake.requests[-1]["messages"][-1]["content"]
    assert sent == f"find_moments result:\n{session.NOT_RUN}"
    assert replies(turn)[0].startswith("I couldn't find an answer")


def test_json_mode_closing_round_reply_ends_the_turn() -> None:
    script = [as_json("find_moments", query="manta") for _ in range(LAST)]
    reply = Reply(json.dumps({"tool": "reply", "text": FOUND_NONE}))
    fake = FakeProvider([*script, reply], JSON_ONLY)
    turn = run(fake, ASK)
    assert turn.status == "done" and replies(turn) == [FOUND_NONE]
    assert len(fake.requests) == session.MAX_ROUNDS


def test_a_cut_off_closing_round_still_names_the_cut_off() -> None:
    def cut(_messages: Any) -> Reply:
        raise providers.TruncatedError("cut off at the end of the context")

    turn = run(FakeProvider([*lookups(LAST), cut]), ASK)
    assert turn.status == "failed" and "cut off" in (turn.error or "")


# -- the view limit closes the turn ---------------------------------------------------------------


@pytest.fixture
def sheets(tmp_path: Path) -> Any:
    """``make(provider)``: a viewer of synthetic frames (no ffmpeg) for ``provider``."""
    cv2 = pytest.importorskip("cv2")
    ok, buf = cv2.imencode(".jpg", np.full((360, 640, 3), 128, np.uint8))
    assert ok
    src = tmp_path / "a.mp4"
    src.write_bytes(b"video")
    proj = tmp_path / "proj"
    (proj / "ui" / "assistant").mkdir(parents=True)

    def make(provider: Any) -> watch.Viewer:
        grab = lambda *_: buf.tobytes()  # noqa: E731
        return watch.Viewer(proj, provider, resolve=lambda _s: src, consent=tuple, grab=grab)

    return make


def run_viewing(fake: FakeProvider, sheets: Any) -> session.Turn:
    turn = session.Turn(fake, project(), ASK, viewer=sheets(fake))
    turn.start()
    assert turn.done.wait(10)
    return turn


@pytest.mark.parametrize("then", ["answers", "keeps viewing"])
def test_a_view_refused_for_the_limit_makes_the_next_round_the_closing_one(
    sheets: Any, then: str
) -> None:
    views = [call("view_moment", id="m1"), call("view_moment", id="m2")]  # the cap is 1 here
    more = [answer(FOUND_NONE)] if then == "answers" else lookups(5, "view_moment")
    fake = FakeProvider([*views, *more])
    turn = run_viewing(fake, sheets)
    assert watch.view_cap(fake) == 1
    assert turn.status == "done" and turn.error is None
    refused = results(fake.requests[2])[-1]
    assert "limit is reached; answer the user now" in refused
    assert tool_names(fake.requests[2]) == [session.FINAL_TOOL] and closing_noted(fake.requests[2])
    if then == "answers":
        assert len(fake.requests) == 3 and replies(turn) == [FOUND_NONE]
    else:  # well before the rounds run out
        assert len(fake.requests) == 4
        (text,) = replies(turn)
        assert "viewed m1" in text


# -- hosted adapters ------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
def test_hosted_closing_round_offers_propose_edit_and_answers_every_call(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """The closing request keeps a tool list (Anthropic wants tools next to earlier tool_use
    blocks) and answers each call by id, the one not run included."""
    n = session.MAX_ROUNDS + 1
    if kind == "openai":
        check = _Resp({"data": [{"id": "gpt-x"}]})
        steps = [rs_reply(rs_call(f"c{i}", "get_moment", '{"id": "m1"}')) for i in range(n - 1)]
        steps.append(rs_reply(rs_text(FOUND_NONE, "final_answer")))
        p: providers.Provider = OpenAIProvider("", "gpt-x", 0, KEY)
    else:
        check = _Resp({"id": "claude-sonnet-5-5"})
        steps = [an_reply(tool_use(f"tu{i}", "get_moment", id="m1")) for i in range(n - 1)]
        steps.append(an_reply({"type": "text", "text": FOUND_NONE}))
        p = AnthropicProvider("", "", 0, KEY)
    wire = Wire(monkeypatch, check, *steps)
    turn = session.Turn(p, project(), ASK)
    turn.start()
    assert turn.done.wait(10)
    assert turn.status == "done", turn.error
    assert replies(turn) == [FOUND_NONE]
    chats = [s["json"] for s in wire.sent if s["method"] == "POST"]
    assert len(chats) == n
    for body in chats[:LAST]:
        assert len(body["tools"]) > 1
    for body in chats[LAST:]:
        assert [t["name"] for t in body["tools"]] == [session.FINAL_TOOL]
    last = chats[-1]
    if kind == "openai":
        outs = [i for i in last["input"] if i.get("type") == "function_call_output"]
        assert [o["call_id"] for o in outs] == [f"c{i}" for i in range(n - 1)]
        assert outs[-1]["output"] == session.NOT_RUN
    else:
        blocks = [b for m in last["messages"] for b in m["content"] if b["type"] == "tool_result"]
        assert [b["tool_use_id"] for b in blocks] == [f"tu{i}" for i in range(n - 1)]
        assert blocks[-1]["content"] == session.NOT_RUN
        assert last["messages"][-1]["role"] == "user"
