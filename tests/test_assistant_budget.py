"""The AI Chat fixes for small contexts: a reply cut off at the context end, the budget and
what a turn leaves out of the conversation, the one nudge to propose, and the rules text.

The model is a :class:`FakeProvider`; every provider's HTTP is stubbed (``Wire``, ``_Resp``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from cutpoint.assistant import context, providers, session, watch
from cutpoint.assistant.providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
    Reply,
    ToolCall,
    TruncatedError,
)
from test_assistant import FakeProvider, _Resp, call, durations, project, run
from test_assistant_byo import KEY, Wire, an_reply, oa_call, oa_reply, rs_reply, rs_text
from test_assistant_byo import no_keys as no_keys  # the autouse fixture

HI = [{"role": "user", "content": "hi"}]
PROPOSE = {"summary": "Add the laugh", "ops": [{"op": "add", "id": "m1", "at": 1}]}
NAMES = "I would take m1 and m2 first."
OPENAI_COMPAT = (
    "https://openrouter.ai/api/v1"  # chat/completions (a bare OpenAI URL uses responses)
)
JSON_ONLY = providers.Capabilities(False, True)


def answer(text: str) -> Reply:
    return Reply(text, [], {"role": "assistant", "content": text})


def cut(_messages: Any) -> Reply:
    raise TruncatedError("cut off at the end of the context")


def as_json(tool: str, **args: Any) -> Reply:
    return Reply(json.dumps({"tool": tool, "arguments": args}))


def as_reply(text: str) -> Reply:
    return Reply(json.dumps({"tool": "reply", "text": text}))


# -- a reply cut off: each provider says so ------------------------------------------------------


def ollama(monkeypatch, chunks: list[dict]) -> OllamaProvider:
    monkeypatch.setattr(providers.requests, "post", lambda url, **kw: _Resp(lines=chunks))
    p = OllamaProvider("http://h:1", "qwen3.5:9b", 8192)
    p._model = "qwen3.5:9b"
    return p


def final(reason: str) -> list[dict]:
    return [
        {"message": {"role": "assistant", "content": "Take m1 and"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": reason},
    ]


def test_ollama_done_reason_length_is_a_truncated_reply(monkeypatch) -> None:
    """Ollama sends what it had as if whole: only the final chunk's done_reason says otherwise."""
    with pytest.raises(TruncatedError, match="cut off"):
        ollama(monkeypatch, final("length")).chat(HI)


def test_ollama_done_reason_stop_is_a_whole_reply(monkeypatch) -> None:
    assert ollama(monkeypatch, final("stop")).chat(HI).text == "Take m1 and"


def test_openai_chat_finish_reason_length_is_a_truncated_reply(monkeypatch) -> None:
    msg = {"role": "assistant", "content": "Take m1"}
    Wire(monkeypatch, _Resp({"choices": [{"message": msg, "finish_reason": "length"}]}))
    with pytest.raises(TruncatedError, match="cut off"):
        OpenAIProvider(OPENAI_COMPAT, "m", 0, KEY).chat(HI)


def test_openai_tool_call_cut_off_is_truncated_not_run(monkeypatch) -> None:
    calls = [oa_call("c1", "propose_edit", '{"summary": "x", "ops": [{"op"')]
    msg = {"role": "assistant", "content": "", "tool_calls": calls}
    Wire(monkeypatch, _Resp({"choices": [{"message": msg, "finish_reason": "length"}]}))
    with pytest.raises(TruncatedError):
        OpenAIProvider(OPENAI_COMPAT, "m", 0, KEY).chat(HI, tools=session.TOOLS)


def test_openai_finish_reason_stop_is_a_whole_reply(monkeypatch) -> None:
    Wire(monkeypatch, oa_reply("All good."))
    assert OpenAIProvider(OPENAI_COMPAT, "m", 0, KEY).chat(HI).text == "All good."


def test_openai_responses_incomplete_is_a_truncated_reply(monkeypatch) -> None:
    Wire(monkeypatch, rs_reply(rs_text("Take"), status="incomplete"))
    with pytest.raises(TruncatedError, match="cut off"):
        OpenAIProvider("", "gpt-6.1-sol", 0, KEY).chat(HI)


def test_anthropic_max_tokens_still_raises(monkeypatch) -> None:
    Wire(monkeypatch, _Resp({"content": [], "stop_reason": "max_tokens"}))
    with pytest.raises(TruncatedError, match="cut off at 4096 tokens"):
        AnthropicProvider("", "m", 0, KEY).chat(HI)
    Wire(monkeypatch, an_reply({"type": "text", "text": "ok"}))
    assert AnthropicProvider("", "m", 0, KEY).chat(HI).text == "ok"


def test_a_truncated_reply_is_a_provider_error() -> None:
    assert issubclass(TruncatedError, providers.ProviderError)


# -- the turn asks again, once ------------------------------------------------------------------


def many(n: int = 120) -> session.Project:
    """A project with more moments than a halved prompt can list."""
    cands = [
        {
            "id": f"c{i}",
            "source": "/v/a.mp4",
            "start": float(i * 10),
            "end": float(i * 10 + 5),
            "score": 90 - i % 50,
            "label": f"moment number {i} with a longish label",
        }
        for i in range(n)
    ]
    return session.Project(cands, [], durations(), "", "key0")


def sent(request: dict[str, Any]) -> int:
    return context.size(request["messages"])


def test_a_cut_off_reply_is_asked_again_with_a_smaller_conversation() -> None:
    fake = FakeProvider([cut, answer("Nothing to change.")])
    turn = run(fake, "what is here?", proj=many())
    assert turn.status == "done", turn.error
    assert len(fake.requests) == 2
    first, second = fake.requests
    assert sent(second) < sent(first)
    assert second["messages"][-1] == {"role": "user", "content": "what is here?"}
    assert turn.events[-1]["kind"] == "reply" and turn.events[-1]["text"] == "Nothing to change."


def test_a_cut_off_after_tool_results_leaves_out_the_older_ones() -> None:
    fake = FakeProvider(
        [
            call("get_moment", id="m1"),
            call("get_moment", id="m2"),
            cut,
            call("propose_edit", **PROPOSE),
        ]
    )
    turn = run(fake, "add the laugh", proj=many())
    assert turn.status == "done", turn.error
    retry = fake.requests[3]["messages"]
    results = [m for m in retry if m.get("role") == "tool"]
    assert results[0]["content"] == context.LEFT_OUT
    assert results[-1]["content"] != context.LEFT_OUT
    assert turn.events[-1]["kind"] == "proposal"


def test_a_cut_off_after_one_tool_result_asks_again_with_less() -> None:
    """With a single result there is nothing older to leave out; the retry must still be smaller
    than the request that was cut off, or asking again changes nothing."""
    fake = FakeProvider([call("get_moment", id="m1"), cut, answer("Done.")])
    turn = run(fake, "what is m1?", proj=many())
    assert turn.status == "done", turn.error
    assert sent(fake.requests[2]) < sent(fake.requests[1])


def test_cut_off_twice_fails_the_turn_naming_the_context_size() -> None:
    fake = FakeProvider([cut, cut, answer("never asked")])
    fake.context_tokens = 4096
    turn = run(fake, "make a montage", proj=many())
    assert turn.status == "failed"
    assert "cut off at the end" in (turn.error or "") and "cut off again" in (turn.error or "")
    assert len(fake.requests) == 2 and len(fake.script) == 1
    assert [e["kind"] for e in turn.events] == ["error"]
    assert turn.events[0]["text"] == turn.error


def cut_with(message: str):
    def step(_messages: Any) -> Reply:
        raise TruncatedError(message)

    return step


def test_the_second_cut_off_error_keeps_the_providers_message_and_advises_only_for_ollama() -> None:
    err = TruncatedError("Anthropic's reply was cut off at 4096 tokens; try a simpler request")
    anthropic = session.cut_again_error(AnthropicProvider("", "m", 0, KEY), err)
    assert "cut off at 4096 tokens" in str(anthropic)
    assert "context_tokens" not in str(anthropic)
    lmstudio = session.cut_again_error(OpenAIProvider("http://127.0.0.1:1234", "m", 0, None), err)
    assert "context_tokens" not in str(lmstudio)
    ollama = session.cut_again_error(OllamaProvider("http://h:1", "m", 4096), err)
    assert "cut off at 4096 tokens" in str(ollama)
    assert "assistant.context_tokens" in str(ollama) and "4096 tokens" in str(ollama)


def test_cut_again_error_says_again_only_when_asked_to() -> None:
    err = TruncatedError("cut off at the end")
    assert "cut off again" in str(session.cut_again_error(AnthropicProvider("", "m", 0, KEY), err))
    once = session.cut_again_error(AnthropicProvider("", "m", 0, KEY), err, again=False)
    assert str(once) == "cut off at the end"


def test_a_reply_cut_off_in_the_final_round_names_the_cut_off_not_the_round_count() -> None:
    script = [call("get_moment", id="m1") for _ in range(session.MAX_ROUNDS - 1)] + [cut]
    fake = FakeProvider(script)
    turn = run(fake, "x", proj=many())
    assert turn.status == "failed"
    assert "cut off" in (turn.error or "") and "no answer after" not in (turn.error or "")
    assert "cut off again" not in (turn.error or "")


def test_a_second_cut_off_in_a_later_round_still_fails() -> None:
    fake = FakeProvider([call("get_moment", id="m1"), cut, call("get_moment", id="m2"), cut])
    turn = run(fake, "x", proj=many())
    assert turn.status == "failed" and "cut off again" in (turn.error or "")


def test_json_mode_cut_off_once_is_asked_again() -> None:
    fake = FakeProvider([cut, as_reply("Nothing to change.")], JSON_ONLY)
    turn = run(fake, "what is here?", proj=many())
    assert turn.status == "done", turn.error
    assert sent(fake.requests[1]) < sent(fake.requests[0])


# -- _fit -----------------------------------------------------------------------------------------


def results_for(provider: Any, n: int, size: int = 4000) -> list[dict[str, Any]]:
    return [
        provider.tool_result(ToolCall("get_moment", {}, f"id{i}"), "x" * size) for i in range(n)
    ]


def text_of(msg: dict[str, Any]) -> str:
    """The tool result text in a message of any provider's shape."""
    if "output" in msg:
        return msg["output"]
    if isinstance(msg["content"], list):
        return msg["content"][0]["content"]
    return msg["content"]


PROVIDERS = {
    "ollama": lambda: OllamaProvider("http://h:1", "m", 2048),
    "openai chat": lambda: OpenAIProvider(OPENAI_COMPAT, "m", 2048, KEY),
    "openai responses": lambda: OpenAIProvider("", "gpt-6.1-sol", 2048, KEY),
    "anthropic": lambda: AnthropicProvider("", "m", 2048, KEY),
}


@pytest.mark.parametrize("which", list(PROVIDERS))
def test_fit_leaves_out_the_oldest_tool_results_of_every_provider_shape(which: str) -> None:
    provider = PROVIDERS[which]()
    turn = session.Turn(provider, project(), "tidy up")
    results = results_for(provider, 4)
    system = {"role": "system", "content": "rules " * 10}
    user = {"role": "user", "content": "tidy up"}
    messages = [system, user]
    for r in results:
        messages += [{"role": "assistant", "content": "calling"}, r]
    turn._results = list(results)

    assert turn._fit(messages, False) is True

    assert text_of(results[0]) == context.LEFT_OUT  # the oldest goes first
    assert text_of(results[-1]) == "x" * 4000  # the latest stays whole
    assert messages[0] is system and system["content"] == "rules " * 10
    assert messages[1] is user and user["content"] == "tidy up"
    assert len(messages) == 2 + 2 * 4  # nothing dropped, only the text replaced


def test_fit_only_leaves_out_as_much_as_needed() -> None:
    provider = OllamaProvider("http://h:1", "m", 4096)  # about 6500 chars of room
    turn = session.Turn(provider, project(), "x")
    results = results_for(provider, 4, size=1500)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "x"}, *results]
    turn._results = list(results)
    assert turn._fit(messages, False) is True
    left = [text_of(r) == context.LEFT_OUT for r in results]
    assert left[0] and not left[-1] and not all(left)


def test_fit_changes_nothing_when_the_conversation_fits() -> None:
    provider = OllamaProvider("http://h:1", "m", 50000)
    turn = session.Turn(provider, project(), "x")
    results = results_for(provider, 3, size=500)
    turn._results = list(results)
    assert turn._fit([{"role": "user", "content": "x"}, *results], False) is False
    assert all(text_of(r) == "x" * 500 for r in results)


def test_fit_never_drops_the_only_result_even_when_it_is_too_big() -> None:
    provider = OllamaProvider("http://h:1", "m", 2048)
    turn = session.Turn(provider, project(), "x")
    only = results_for(provider, 1, size=9000)
    turn._results = list(only)
    assert turn._fit([{"role": "user", "content": "x"}, *only], False) is False
    assert text_of(only[0]) == "x" * 9000


def test_fit_tight_leaves_out_all_but_the_latest_even_when_it_fits() -> None:
    provider = OllamaProvider("http://h:1", "m", 50000)
    turn = session.Turn(provider, project(), "x")
    results = results_for(provider, 3, size=200)
    turn._results = list(results)
    assert turn._fit([{"role": "user", "content": "x"}, *results], False, tight=True) is True
    assert [text_of(r) == context.LEFT_OUT for r in results] == [True, True, False]


def test_fit_in_json_mode_replaces_the_result_text_of_user_messages() -> None:
    provider = OllamaProvider("http://h:1", "m", 2048)
    turn = session.Turn(provider, project(), "x")
    msgs = [{"role": "user", "content": f"get_moment result:\n{'x' * 4000}"} for _ in range(3)]
    turn._results = list(msgs)
    assert turn._fit([{"role": "system", "content": "rules"}, *msgs], True) is True
    assert msgs[0]["content"] == context.LEFT_OUT and msgs[2]["content"].startswith("get_moment")


@pytest.mark.parametrize("tokens", [2048, 4096, 8192, 50000])
@pytest.mark.parametrize("json_mode", [False, True])
def test_fit_accepts_only_what_stays_under_two_characters_per_token(
    tokens: int, json_mode: bool
) -> None:
    """Whatever _fit leaves is, with the tool definitions, under ctx * 2.0 characters (less the
    room for a reply), unless only the latest result is left."""
    provider = OllamaProvider("http://h:1", "m", tokens)
    turn = session.Turn(provider, project(), "x")
    results = results_for(provider, 12, size=1000)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "x"}, *results]
    turn._results = list(results)
    turn._fit(messages, json_mode)
    tools = 0 if json_mode else len(json.dumps(turn._tools()))
    total = context.size(messages) + tools
    left = sum(text_of(r) == context.LEFT_OUT for r in results)
    assert total <= (tokens - context.REPLY_TOKENS) * 2.0 or left == len(results) - 1
    assert total < tokens * 2.0 or left == len(results) - 1


def test_a_long_turn_sends_old_results_out_of_the_request_itself() -> None:
    """End to end: with a small context the last request no longer carries the first results."""
    fake = FakeProvider([call("get_moment", id=f"m{i}") for i in range(1, 5)] + [answer("ok")])
    fake.context_tokens = 2048
    turn = run(fake, "look at them", proj=many(30))
    assert turn.status == "done", turn.error
    last = [m for m in fake.requests[-1]["messages"] if m.get("role") == "tool"]
    assert len(last) == 4 and last[0]["content"] == context.LEFT_OUT
    assert last[-1]["content"] != context.LEFT_OUT


# -- budget() and size() --------------------------------------------------------------------------


def test_budget_counts_the_tool_definitions_and_keeps_room_for_a_reply() -> None:
    plain = context.budget(8192)
    with_tools = context.budget(8192, session.TOOLS)
    assert plain.tools == 0
    assert with_tools.tools == int(len(json.dumps(session.TOOLS)) / context.CHARS_PER_TOKEN)
    assert with_tools.tools > 500 and with_tools.chars < plain.chars
    reserve = max(context.REPLY_TOKENS, int(8192 * context.REPLY_SHARE))
    assert reserve == 2457 and plain.chars == 7384  # the fit limit less one result, not 14337
    assert (with_tools.moments, with_tools.turns) == (40, 4)


@pytest.mark.parametrize(
    ("tokens", "watching"), [(12288, False), (16384, False), (24576, True), (32768, True)]
)
def test_a_prompt_leaves_room_for_a_result_inside_the_fit_limit(
    tokens: int, watching: bool
) -> None:
    """The table, the tool definitions and the sheets of a turn leave one whole tool result
    under the limit _fit works to, or every round would leave out every earlier result. (At
    8192 the rules, tools and sheets alone are over it: the budget is at its floor there.)"""
    provider = OllamaProvider("http://h:1", "m", tokens)
    turn = session.Turn(provider, many(), "x")
    turn._watch = watching
    sheets = watch.view_cap(provider) if watching else 0
    image = context.size({"type": "image"})
    total = context.size(turn._messages(False)) + len(json.dumps(turn._tools())) + sheets * image
    limit = (tokens - context.REPLY_TOKENS) * session.FIT_CHARS_PER_TOKEN
    assert total + session.MAX_RESULT_CHARS <= limit


def test_the_8192_budget_is_what_the_fit_allows_less_the_tools_and_one_result() -> None:
    plain = context.budget(8192)
    assert plain.chars == int((8192 - context.REPLY_TOKENS) * session.FIT_CHARS_PER_TOKEN) - 6000
    with_sheets = context.budget(8192, session.TOOLS, images=2)
    assert with_sheets.chars == int(8192 // 4 * context.CHARS_PER_TOKEN)  # the floor
    assert with_sheets.chars <= plain.chars


def test_the_budget_for_a_large_or_hosted_context_is_not_cut_by_the_fit_limit() -> None:
    for tokens in (32768, 50000, 200000):
        b = context.budget(tokens, session.TOOLS, images=4)
        reserve = max(context.REPLY_TOKENS, int(tokens * context.REPLY_SHARE))
        free = tokens - reserve - b.tools - 4 * context.IMAGE_TOKENS
        assert b.chars == int(free * context.CHARS_PER_TOKEN)


def test_budget_tight_halves_the_prompt_and_the_turns() -> None:
    normal = context.budget(32768, session.TOOLS)
    tight = context.budget(32768, session.TOOLS, tight=True)
    assert tight.tight and not normal.tight
    assert tight.chars == normal.chars // 2
    assert tight.turns == normal.turns // 2 and tight.moments == normal.moments


def test_budget_keeps_at_least_a_quarter_of_the_context_for_the_prompt() -> None:
    b = context.budget(4096, [{"x": "y" * 40000}])
    assert b.chars == int(4096 // 4 * context.CHARS_PER_TOKEN)


def test_size_counts_text_wherever_it_sits_and_an_image_as_its_tokens() -> None:
    assert context.size("abcd") == 4
    assert context.size([{"a": "ab"}, {"a": "cde"}]) == 5
    nested = {
        "content": [{"type": "text", "text": "hello"}, {"function": {"arguments": {"id": "m1"}}}]
    }
    assert context.size(nested) == len("text") + len("hello") + len("m1")
    image = {"type": "image", "media_type": "image/jpeg", "data": "A" * 100000}
    assert context.size(image) == int(context.IMAGE_TOKENS * context.CHARS_PER_TOKEN)
    assert context.size(None) == 0 and context.size(7) == 0


def test_the_default_ollama_context_is_50000_and_other_local_servers_keep_8192() -> None:
    assert providers.OLLAMA_CONTEXT_TOKENS == 50000
    assert providers.ON_MACHINE_CONTEXT_TOKENS == 8192
    assert OpenAIProvider("http://127.0.0.1:1234", "m", 0, None).context_tokens == 8192
    assert OpenAIProvider("http://127.0.0.1:1234", "m", 4096, None).context_tokens == 4096


# -- the nudge ------------------------------------------------------------------------------------


def nudges(fake: FakeProvider) -> list[dict[str, Any]]:
    return [
        r for r in fake.requests if r["messages"][-1] == {"role": "user", "content": session.NUDGE}
    ]


def test_native_text_naming_a_moment_gets_one_nudge_and_may_then_propose() -> None:
    fake = FakeProvider([answer(NAMES), call("propose_edit", **PROPOSE)])
    turn = run(fake, "build a montage")
    assert turn.status == "done", turn.error
    assert len(fake.requests) == 2 and len(nudges(fake)) == 1
    assert [e["kind"] for e in turn.events] == ["proposal"]
    assert turn.events[0]["summary"] == "Add the laugh"
    assert fake.requests[1]["tools"] == session.TOOLS  # the nudged request can still propose


def test_native_text_naming_a_moment_in_the_last_round_is_not_nudged_and_stands() -> None:
    script = [call("get_moment", id="m1") for _ in range(session.MAX_ROUNDS - 1)]
    fake = FakeProvider([*script, answer("Take m1.")])
    turn = run(fake, "build a montage")
    assert turn.status == "done", turn.error
    assert len(fake.requests) == session.MAX_ROUNDS and not nudges(fake)
    assert [e["text"] for e in turn.events if e["kind"] == "reply"] == ["Take m1."]


def test_native_a_nudge_then_tool_calls_until_the_rounds_run_out_keeps_the_first_answer() -> None:
    script = [answer(NAMES)] + [call("get_moment", id="m1") for _ in range(session.MAX_ROUNDS)]
    fake = FakeProvider(script)
    turn = run(fake, "build a montage")
    assert turn.status == "done", turn.error
    assert [e["text"] for e in turn.events if e["kind"] == "reply"] == [NAMES]


def test_native_a_second_text_only_reply_is_final() -> None:
    fake = FakeProvider([answer(NAMES), answer("Still m1 and m2, as you asked.")])
    turn = run(fake, "build a montage")
    assert turn.status == "done"
    assert len(fake.requests) == 2 and len(nudges(fake)) == 1
    replies = [e["text"] for e in turn.events if e["kind"] == "reply"]
    assert replies == ["Still m1 and m2, as you asked."]


def test_native_an_empty_second_answer_falls_back_to_the_first() -> None:
    fake = FakeProvider([answer(NAMES), answer("")])
    turn = run(fake, "which are best?")
    assert turn.status == "done", turn.error
    assert [e["text"] for e in turn.events if e["kind"] == "reply"] == [NAMES]


def test_an_empty_first_answer_is_still_an_error() -> None:
    turn = run(FakeProvider([answer("")]), "hi")
    assert turn.status == "failed" and "empty answer" in (turn.error or "")


@pytest.mark.parametrize("text", ["Nothing to change.", "Take m99 maybe.", "Try item3 or hm1x."])
def test_native_a_reply_naming_no_known_moment_is_not_nudged(text: str) -> None:
    fake = FakeProvider([answer(text)])
    turn = run(fake, "hi")
    assert turn.status == "done" and len(fake.requests) == 1
    assert turn.events[-1]["text"] == text


def test_native_a_turn_that_proposed_ends_there_and_is_not_nudged() -> None:
    fake = FakeProvider([call("propose_edit", **PROPOSE), answer(NAMES)])
    turn = run(fake, "build a montage")
    assert turn.status == "done" and len(fake.requests) == 1 and not nudges(fake)
    assert [e["kind"] for e in turn.events] == ["proposal"]


def test_native_a_rejected_proposal_is_not_one_made_so_the_text_after_it_is_nudged() -> None:
    bad = {"summary": "x", "ops": [{"op": "trim", "id": "m2", "end": 999}]}
    fake = FakeProvider(
        [call("propose_edit", **bad), answer(NAMES), call("propose_edit", **PROPOSE)]
    )
    turn = run(fake, "build a montage")
    assert turn.status == "done", turn.error
    assert len(nudges(fake)) == 1 and [e["kind"] for e in turn.events] == ["proposal"]


def test_the_nudge_is_not_an_event_nor_in_the_saved_turn() -> None:
    fake = FakeProvider([answer(NAMES), call("propose_edit", **PROPOSE)])
    turn = run(fake, "build a montage")
    assert session.NUDGE not in json.dumps(turn.view())
    rec = turn.record()
    assert session.NUDGE not in json.dumps(rec)
    assert rec["user"] == "build a montage" and rec["reply"] == "Proposed: Add the laugh"
    # a later turn repeats this one as user and reply only
    table = context.MomentTable(project().candidates, [])
    later = context.build_messages(
        table, [{"user": rec["user"], "reply": rec["reply"]}], "next", context.budget(8192)
    )
    assert session.NUDGE not in json.dumps(later)


def test_the_nudged_replys_text_is_what_a_failed_nudge_answer_leaves() -> None:
    """The turn that ends on the first text keeps it as its reply in chat.json."""
    fake = FakeProvider([answer(NAMES), answer("")])
    turn = run(fake, "which?")
    assert turn.record()["reply"] == NAMES


def test_json_text_naming_a_moment_gets_one_nudge_and_may_then_propose() -> None:
    fake = FakeProvider([as_reply(NAMES), as_json("propose_edit", **PROPOSE)], JSON_ONLY)
    turn = run(fake, "build a montage")
    assert turn.status == "done", turn.error
    assert len(fake.requests) == 2 and len(nudges(fake)) == 1
    assert [e["kind"] for e in turn.events] == ["proposal"]


def test_json_a_second_text_only_reply_is_final() -> None:
    fake = FakeProvider([as_reply(NAMES), as_reply("Those two, m1 and m2.")], JSON_ONLY)
    turn = run(fake, "which?")
    assert turn.status == "done"
    assert len(fake.requests) == 2 and len(nudges(fake)) == 1
    assert turn.events[-1]["text"] == "Those two, m1 and m2."


def test_json_no_known_id_gets_no_nudge() -> None:
    fake = FakeProvider([as_reply("Nothing to change.")], JSON_ONLY)
    assert run(fake, "hi").status == "done" and len(fake.requests) == 1


def test_json_a_turn_that_proposed_ends_there_and_is_not_nudged() -> None:
    fake = FakeProvider([as_json("propose_edit", **PROPOSE), as_reply(NAMES)], JSON_ONLY)
    turn = run(fake, "build")
    assert turn.status == "done" and len(fake.requests) == 1 and not nudges(fake)
    assert [e["kind"] for e in turn.events] == ["proposal"]


# -- the rules ------------------------------------------------------------------------------------


def test_the_rules_say_a_change_ends_in_propose_edit_and_an_empty_sequence_uses_add() -> None:
    table = context.MomentTable(project().candidates, [])
    msgs = context.build_messages(table, [], "build it", context.budget(8192))
    system = msgs[0]["content"]
    assert "end with propose_edit" in system
    assert "empty sequence is built with add ops" in system
