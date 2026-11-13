"""``preview_ranking`` of the AI Chat (#902 PR 3b): when it is offered, what the model gets back,
the card event, the "(moment ids changed since)" mark, the page's ``RankingHook`` and a real turn
through the route that stores the ranking in ``chat.json``.

The model is the scripted ``FakeProvider`` of ``test_assistant``.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("httpx")
from fastapi import HTTPException
from fastapi.testclient import TestClient

from conftest import write
from cutpoint import paths, ui
from cutpoint.assistant import context, providers, ranking, session
from cutpoint.assistant import store as ai_store
from cutpoint.assistant.providers import Capabilities, OllamaProvider
from cutpoint.pano import project as pano_project
from test_assistant import FakeProvider, call, project, run
from test_assistant_budget import as_json, many
from test_assistant_byo import no_keys as no_keys  # the autouse fixture
from test_rerank import lite_layers as lite_layers
from test_rerank import make_project

PORT = 8765
JSON_ONLY = Capabilities(False, True)
STYLES = [("gaming", ""), ("funny", ""), ("mine", "My own style")]
READY = {
    "ready": True,
    "style": "funny",
    "from": "gaming",
    "base": "k",
    "top": [
        {"rank": 1, "label": "big laugh", "was": 1},
        {"rank": 2, "label": "x" * 100, "was": 9},
        {"rank": 3, "label": "jump", "was": 2},
    ],
}
NOT_READY = {
    "ready": False,
    "style": "funny",
    "from": "gaming",
    "base": "k",
    "can_analyze": True,
    "message": "needs the laughter layer " + "z" * 2000,
}


class Hook:
    """A ranking hook with scripted styles and previews."""

    def __init__(self, styles=STYLES, current="gaming", payload=None, error=None) -> None:
        self._styles, self._current, self.payload, self.error = styles, current, payload, error
        self.previewed: list[str] = []

    def styles(self) -> list[tuple[str, str]]:
        return list(self._styles)

    def current(self) -> str:
        return self._current

    def preview(self, name: str) -> dict[str, Any]:
        self.previewed.append(name)
        if self.error:
            raise self.error
        return dict(self.payload or READY)


def with_hook(hook: Hook | None, key: str | None = None) -> session.Project:
    proj = project()
    proj.ranking, proj.ranking_key = hook, key
    return proj


def names(request: dict[str, Any]) -> list[str]:
    return [t["function"]["name"] for t in request["tools"]]


# -- when the tool is offered --------------------------------------------------------------------


def test_the_tool_is_offered_with_a_hook_and_its_enum_lists_the_hooks_styles() -> None:
    fake = FakeProvider([call("preview_ranking", style="funny")])
    run(fake, proj=with_hook(Hook()))
    tools = fake.requests[0]["tools"]
    assert "preview_ranking" in names(fake.requests[0])
    tool = next(t for t in tools if t["function"]["name"] == "preview_ranking")
    assert tool["function"]["parameters"]["properties"]["style"]["enum"] == [
        "gaming",
        "funny",
        "mine",
    ]
    system = fake.requests[0]["messages"][0]["content"]
    assert "ranked as gaming now" in system and "mine (My own style)" in system


def test_a_project_without_a_hook_gets_exactly_the_plain_tools_and_schema() -> None:
    fake = FakeProvider([as_json("find_moments", query="x")], JSON_ONLY)
    run(fake, proj=with_hook(None))
    assert fake.requests[0]["schema"] == session.JSON_SCHEMA
    assert "preview_ranking" not in json.dumps(fake.requests[0]["messages"])
    tooled = FakeProvider([call("find_moments", query="x")])
    run(tooled, proj=with_hook(None))
    assert tooled.requests[0]["tools"] == session.TOOLS


@pytest.mark.parametrize("styles", [[("gaming", "")], []])
def test_no_other_style_means_no_tool(styles: list[tuple[str, str]]) -> None:
    fake = FakeProvider([call("find_moments", query="x")])
    run(fake, proj=with_hook(Hook(styles=styles)))
    assert fake.requests[0]["tools"] == session.TOOLS


def test_json_mode_lists_the_tool_line_and_the_schema_enum() -> None:
    fake = FakeProvider([as_json("preview_ranking", style="funny")], JSON_ONLY)
    run(fake, proj=with_hook(Hook()))
    req = fake.requests[0]
    assert ranking.JSON_TOOL in req["messages"][0]["content"]
    assert "preview_ranking" in req["schema"]["properties"]["tool"]["enum"]


def test_a_failing_hook_offers_no_tool_and_the_turn_goes_on() -> None:
    class Broken(Hook):
        def styles(self):
            raise RuntimeError("boom")

    fake = FakeProvider([call("find_moments", query="x")])
    run(fake, proj=with_hook(Broken()))
    assert fake.requests[0]["tools"] == session.TOOLS


# -- the tool call -------------------------------------------------------------------------------


def result_of(fake: FakeProvider, round_: int = 1) -> str:
    return fake.requests[round_]["messages"][-1]["content"]


@pytest.mark.parametrize("style", ["nope", "gaming", None, 3])
def test_an_unknown_or_current_style_goes_back_to_the_model_with_the_list(style: Any) -> None:
    hook = Hook()
    fake = FakeProvider([call("preview_ranking", style=style), call("propose_edit", summary="s")])
    run(fake, proj=with_hook(hook))
    text = result_of(fake)
    assert text.startswith("error: style must be one of funny, mine")
    assert "ranked as gaming now" in text and hook.previewed == []


@pytest.mark.parametrize(
    ("error", "want"),
    [
        (ranking.RankingError("a run holds it", busy=True), "the project is busy"),
        (ranking.RankingError("no analysis"), "could not be previewed: no analysis"),
    ],
)
def test_a_refused_preview_is_a_non_final_error(error: Exception, want: str) -> None:
    fake = FakeProvider([call("preview_ranking", style="funny"), call("find_moments", query="x")])
    hook = Hook(error=error)
    turn = run(fake, proj=with_hook(hook))
    assert want in result_of(fake) and hook.previewed[0] == "funny"
    assert len(fake.requests) >= 2  # the turn went on
    assert not [e for e in turn.events if e["kind"] == "ranking"]


def tool_events(turn: session.Turn) -> list[dict[str, Any]]:
    return [e for e in turn.events if e["kind"] == "tool" and e["name"] == "preview_ranking"]


@pytest.mark.parametrize(
    ("error", "want"),
    [
        (ranking.RankingError("a run holds it", busy=True), "the project is busy"),
        (ranking.RankingError("no analysis"), "no analysis"),
        (PermissionError(13, "denied", "C:\\p\\highlights.json"), "an error on this computer"),
    ],
)
def test_every_refused_preview_emits_a_tool_event_for_the_page(error: Exception, want: str) -> None:
    fake = FakeProvider([call("preview_ranking", style="funny"), call("find_moments", query="x")])
    turn = run(fake, proj=with_hook(Hook(error=error)))
    (event,) = tool_events(turn)
    assert event["style"] == "funny" and event["error"] == want


def test_a_preview_that_raises_an_os_error_keeps_the_path_from_the_model_and_the_reply() -> None:
    err = PermissionError(13, "denied", "C:\\p\\highlights.json")
    fake = FakeProvider([call("preview_ranking", style="funny"), call("find_moments", query="x")])
    turn = run(fake, proj=with_hook(Hook(error=err)))
    assert turn.status != "failed" and turn.error is None  # the turn went on
    assert len(fake.requests) >= 2
    assert "could not be previewed (an error on this computer)" in result_of(fake)
    for text in (json.dumps(fake.requests), json.dumps(turn.record()), json.dumps(turn.view())):
        assert "C:\\\\" not in text and "highlights.json" not in text


def test_success_emits_the_card_event_with_the_reason_and_ends_the_turn() -> None:
    fake = FakeProvider([call("preview_ranking", style="funny", reason="  you  asked\nfor fun ")])
    turn = run(fake, proj=with_hook(Hook(), key="k1"))
    (event,) = [e for e in turn.events if e["kind"] == "ranking"]
    assert {k: event[k] for k in READY} == READY
    assert event["reason"] == "you asked for fun"
    assert len(fake.requests) == 1  # no second round
    rec = turn.record()
    assert rec["reply"] == "Previewed the funny ranking (1 new in the top 3); the user decides"
    assert rec["ranking_key"] == "k1"


def test_the_reply_of_a_not_ready_ranking_says_a_new_analysis_is_needed() -> None:
    fake = FakeProvider([call("preview_ranking", style="funny")])
    turn = run(fake, proj=with_hook(Hook(payload=NOT_READY)))
    assert turn.record()["reply"] == (
        "Previewed the funny ranking: it needs a new analysis; the user decides"
    )


def test_the_turn_view_carries_the_projects_ranking_key() -> None:
    turn = session.Turn(FakeProvider([]), with_hook(Hook(), key="k9"), "x")
    assert turn.view()["ranking_key"] == "k9"
    assert session.Turn(FakeProvider([]), with_hook(Hook()), "x").view()["ranking_key"] is None


def test_no_ranking_key_without_one_in_the_project() -> None:
    turn = run(FakeProvider([call("preview_ranking", style="funny")]), proj=with_hook(Hook()))
    assert "ranking_key" not in turn.record()


def test_the_result_for_the_model_is_short_and_names_no_path() -> None:
    ready = {**READY, "base": "C:\\Users\\me\\projects\\p", "file": "a.mp4"}
    for payload in (ready, NOT_READY):
        text = ranking.result(payload)
        assert len(text) <= ranking.MAX_RESULT
        assert not re.search(r"[A-Za-z]:[\\/]|/home/|/Users/", text)
    text = ranking.result(READY)
    assert "funny (was gaming)" in text and "1 new in the top 3" in text
    assert "1. big laugh" in text and "x" * 41 not in text
    assert "Analyze again" in ranking.result(NOT_READY)


# -- Q7: earlier replies made with another ranking -----------------------------------------------


def earlier_reply(stored: Any, current: Any, record_has_key: bool = True) -> str:
    rec = {"user": "first", "reply": "Take m1"}
    if record_has_key:
        rec["ranking_key"] = stored
    proj = project()
    proj.ranking_key = current
    turn = session.Turn(FakeProvider([]), proj, "next", [rec])
    msgs = turn._messages(False)
    return next(m["content"] for m in msgs if m["role"] == "assistant")


@pytest.mark.parametrize(
    ("stored", "current", "marked"),
    [
        ("a", "b", True),
        ("a", "a", False),
        (None, "b", False),
        ("a", None, False),
        (3, "b", False),
    ],
)
def test_an_earlier_reply_is_marked_only_when_both_keys_are_strings_and_differ(
    stored: Any, current: Any, marked: bool
) -> None:
    text = earlier_reply(stored, current)
    assert text.startswith(context.IDS_CHANGED) is marked


def test_an_old_record_without_a_key_is_never_marked() -> None:
    assert earlier_reply(None, "b", record_has_key=False) == "Take m1"


# -- the page's RankingHook ----------------------------------------------------------------------


@pytest.fixture
def client(home: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(ui, "cuda_probe", lambda: False)
    app = ui.create_app(PORT)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        c.headers["X-Cutpoint-Token"] = app.state.token
        yield c


def test_the_hook_lists_built_in_and_saved_styles_and_the_note_is_cut(home) -> None:
    write(
        paths.styles_dir() / "mine.json",
        json.dumps({"base": "gaming", "weights": {}, "description": "d" * 200}),
    )
    hook = ui.RankingHook(home / "projects" / "p", "gaming")
    got = dict(hook.styles())
    assert {"gaming", "funny", "mine"} <= set(got) and hook.current() == "gaming"
    listed = dict(ranking.listed(hook.styles()))
    assert len(listed["mine"]) == ranking.NOTE_CHARS
    assert listed["gaming"] == ranking.STYLE_NOTES["gaming"]


def test_the_hook_turns_refusals_and_http_errors_into_ranking_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hook = ui.RankingHook(tmp_path, "gaming")

    def refuse(code: str):
        def raiser(_p, _name):
            raise ui.RankingRefused("some text", code)

        return raiser

    monkeypatch.setattr(ui, "ranking_preview", refuse("folder_busy"))
    with pytest.raises(ranking.RankingError, match="some text") as err:
        hook.preview("funny")
    assert err.value.busy

    monkeypatch.setattr(ui, "ranking_preview", refuse("other"))
    with pytest.raises(ranking.RankingError) as err:
        hook.preview("funny")
    assert not err.value.busy

    def http(_p, _name):
        raise HTTPException(409, "no analysis")

    monkeypatch.setattr(ui, "ranking_preview", http)
    with pytest.raises(ranking.RankingError, match="no analysis") as err:
        hook.preview("funny")
    assert not err.value.busy


def test_refused_previews_name_no_absolute_path(home: Path, tmp_path: Path) -> None:
    messages = []
    bare = home / "projects" / "bare"  # no analysis
    bare.mkdir(parents=True)
    pano = make_project(home, tmp_path)
    (pano / pano_project.PANO_FILE).write_text("{}", encoding="utf-8")  # a 360 project
    for folder, style in ((bare, "funny"), (pano, "funny"), (bare, "auto"), (bare, "nope")):
        with pytest.raises(ranking.RankingError) as err:
            ui.RankingHook(folder, "gaming").preview(style)
        messages.append(str(err.value))
    assert all(messages)
    for text in messages:
        assert not re.search(r"[A-Za-z]:[\\/]|/home/|/Users/|highlights\.json", text), text
        assert str(home) not in text and str(tmp_path) not in text


# -- a real turn through the route ---------------------------------------------------------------


def use(monkeypatch: pytest.MonkeyPatch, fake: FakeProvider) -> FakeProvider:
    monkeypatch.setitem(providers.PROVIDERS, "ollama", lambda _settings: fake)
    return fake


def turn_on(client: TestClient, text: str = "show the funny bits first") -> dict[str, Any]:
    res = client.post("/api/projects/p/assistant/turns", json={"text": text, "segments": []})
    assert res.status_code == 200, res.text
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        view = client.get(f"/api/projects/p/assistant/turns/{res.json()['id']}").json()
        if view["status"] != "running":
            return view
        time.sleep(0.02)
    raise AssertionError("turn did not finish")


def test_a_turn_previews_a_ranking_and_chat_json_keeps_the_event_and_the_key(
    home, tmp_path, client, monkeypatch
) -> None:
    project_dir = make_project(home, tmp_path)
    fake = use(monkeypatch, FakeProvider([call("preview_ranking", style="funny", reason="fun")]))
    before = (project_dir / "highlights.json").read_bytes()
    view = turn_on(client)
    assert view["status"] == "done"
    (event,) = [e for e in view["events"] if e["kind"] == "ranking"]
    assert event["style"] == "funny" and event["from"] == "gaming" and event["reason"] == "fun"
    assert event["ready"] is True and event["top"]
    assert "preview_ranking" in names(fake.requests[0])
    assert (project_dir / "highlights.json").read_bytes() == before  # a preview writes nothing
    saved = ai_store.turns(project_dir)[-1]
    assert saved["ranking_key"] == ui.ranking_key(project_dir)
    assert any(e["kind"] == "ranking" for e in saved["events"])
    assert saved["reply"].startswith("Previewed the funny ranking")
    assert str(project_dir) not in json.dumps(fake.requests)


def test_a_360_project_is_not_offered_the_tool(home, tmp_path, client, monkeypatch) -> None:
    project_dir = make_project(home, tmp_path)
    (project_dir / pano_project.PANO_FILE).write_text("{}", encoding="utf-8")
    fake = use(monkeypatch, FakeProvider([call("find_moments", query="x")]))
    turn_on(client)
    assert fake.requests and "preview_ranking" not in names(fake.requests[0])


def ranked_turn(tokens: int) -> session.Turn:
    """A turn over many moments with the ranking tool offered (every built-in style, one saved)."""
    styles = [(n, "") for n in ranking.STYLE_NOTES] + [("mine", "d" * 200)]
    proj = many()
    proj.ranking = Hook(styles=styles, current="gaming")
    return session.Turn(OllamaProvider("http://h:1", "m", tokens), proj, "x")


def test_the_ranking_tool_keeps_the_8k_budget_at_its_floor_and_the_prompt_builds() -> None:
    turn = ranked_turn(8192)
    tools = turn._tools()
    assert "preview_ranking" in [t["function"]["name"] for t in tools]
    plan = context.budget(8192, tools)
    assert plan.chars >= int(8192 // 4 * context.CHARS_PER_TOKEN)  # never under the floor
    assert plan.moments > 0 and plan.turns > 0
    messages = turn._messages(False)
    assert "preview_ranking" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "x"}


def test_the_ranking_rules_do_not_empty_the_moment_table_at_8k() -> None:
    """At 8192 tokens a project without the tool lists a few moments in the prompt; offering
    preview_ranking adds ~800 characters of rules and must not leave the table at "(none)"."""
    plain = ranked_turn(8192)
    plain.project.ranking, plain._styles = None, []
    assert "(none)" not in plain._messages(False)[1]["content"]  # the baseline lists moments
    assert "(none)" not in ranked_turn(8192)._messages(False)[1]["content"]


@pytest.mark.parametrize("tokens", [12288, 16384, 32768])
def test_the_ranking_tool_leaves_room_for_a_result_inside_the_fit_limit(tokens: int) -> None:
    turn = ranked_turn(tokens)
    messages = turn._messages(False)
    total = context.size(messages) + len(json.dumps(turn._tools()))
    limit = (tokens - context.REPLY_TOKENS) * session.FIT_CHARS_PER_TOKEN
    assert total + session.MAX_RESULT_CHARS <= limit
