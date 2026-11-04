"""AI Chat building to a length target with clean clip ends (#738): the lengths and the target the
model sees, a short or mislabelled proposal sent back once, lengthened/shortened diffs, the
exported edges (planner and route) and the snapping of a lengthened end to the end of a phrase.

The model is the scripted ``FakeProvider`` of ``test_assistant``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from cutpoint import cache, ui
from cutpoint.assistant import context, edges, lookfor, ops, providers, session, topup
from test_assistant import (
    CANDS,
    SEQ,
    A,
    B,
    FakeProvider,
    Reply,
    _Resp,
    call,
    durations,
    project,
    run,
)
from test_assistant_routes import PORT
from test_assistant_routes import proj as proj


class Planner:
    """Stands in for :class:`edges.EdgePlanner`: every end is ``extra`` seconds later, and a
    phrase ends 3 s after where it is asked."""

    def __init__(self, extra: float = 0.0) -> None:
        self.extra = extra
        self.asked: list[list[str]] = []

    def plan(self, rows, **_kw):
        self.asked.append([r["id"] for r in rows])
        return {r["id"]: (r["start"], r["end"] + self.extra) for r in rows}

    def phrase_end(self, _source: str, t: float) -> float:
        return t + 3.0


def planned_project(extra: float = 0.0, **kw: Any) -> session.Project:
    p = project(**kw)
    p.edges = Planner(extra)  # type: ignore[assignment]
    return p


def propose(summary: str, *ops_: dict[str, Any]) -> Any:
    return call("propose_edit", summary=summary, ops=list(ops_))


def kinds(turn: session.Turn) -> list[str]:
    return [e["kind"] for e in turn.view()["events"]]


# -- the target the user asked for ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("~4 min of beauty shots", 240),
        ("make it 4 minutes", 240),
        ("a 4-minute montage", 240),
        ("make it 240 s", 240),
        ("about 90 seconds", 90),
        ("1:30 total", 90),
        ("make it 2.5 min", 150),
        ("about two minutes of skiing", 120),
        ("trim 2 s off the end", None),
        ("extend m3 by 5 seconds", None),
        ("extend m3 by 2 minutes", None),
        ("extend m3 to 60 seconds", None),
        ("what happens at 1:30?", None),
        ("add the laugh", None),
    ],
)
def test_the_length_the_user_asks_for_is_read_from_the_message(
    text: str, seconds: float | None
) -> None:
    assert context.target_seconds(text) == seconds


# -- what the model sees: lengths as exported, room, the total, the target, the score scale -----


def test_the_sequence_shows_each_clip_as_exported_with_its_room_and_a_total() -> None:
    table = context.MomentTable(list(CANDS), list(SEQ))
    table.durations = durations()
    table.planned = {"c2": (40.0, 53.5)}  # the export finishes the sentence 3.5 s later
    rows = [json.loads(x) for x in context.sequence_block(table).splitlines()[1:-1]]
    assert rows[0] | {"label": ""} == {
        "pos": 1,
        "id": "m2",
        "file": "a.mp4",
        "start": 40.0,
        "end": 50.0,
        "len": 13.5,
        "room_before": 10.0,  # capped (a trim moves an edge out by 10 s at most)
        "room_after": 10.0,
        "label": "",
    }
    assert rows[1]["len"] == 4.0 and rows[1]["room_after"] == 10.0  # raw where nothing is planned
    assert rows[-1] == {"total": {"clips": 3, "len": 27.5}}
    assert table.row("c2")["len"] == 13.5 and table.row("c1")["len"] == 10.0  # a moment not in it


def test_the_rules_say_how_to_lengthen_and_the_target_is_in_the_context() -> None:
    rules = context.rules(context.MomentTable(list(CANDS), list(SEQ)))
    assert "end_delta > 0" in rules and "ends abruptly" in rules and "must reach it" in rules
    table = context.MomentTable(list(CANDS), list(SEQ))
    msgs = context.build_messages(table, [], "make it 4 min", context.budget(8192), target=240)
    assert "asked for about 240 s. The sequence is 24 s now (216 s short)" in msgs[1]["content"]


def test_scores_are_0_100_except_the_visual_pool_which_is_0_10() -> None:
    plain = context.MomentTable(list(CANDS), [])
    both = context.MomentTable([*CANDS, {**CANDS[0], "id": "v1", "pool": "visual"}], [])
    first = context.build_messages(plain, [], "hi", context.budget(8192))[1]["content"]
    second = context.build_messages(both, [], "hi", context.budget(8192))[1]["content"]
    assert "(score 0-100;" in first
    assert "score 0-100, but 0-10 for the visual pool" in second


# -- the planner's lengths reach the model --------------------------------------------------------


def test_a_turn_gives_the_model_the_exported_lengths() -> None:
    fake = FakeProvider(
        [call("find_moments", query="jump"), propose("Add", {"op": "add", "id": "m1"})]
    )
    turn = run(fake, "add the laugh", proj=planned_project(extra=2.0))
    assert turn.status == "done"
    context_message = fake.requests[0]["messages"][1]["content"]
    assert '"len":12.0' in context_message.replace(" ", "")  # c2: 10 s raw, 12 s exported
    assert '"total":{"clips":3,"len":30.0}' in context_message.replace(" ", "")


# -- a proposal that falls short of the asked length is sent back once -----------------------------


def test_a_proposal_short_of_the_asked_length_is_sent_back_three_times_then_taken() -> None:
    add = {"op": "add", "id": "m1"}
    fake = FakeProvider([propose("Add the laugh", add)] * 4)
    turn = run(fake, "build about 4 minutes")
    assert turn.status == "done", turn.error
    sent_back = fake.requests[1]["messages"][-1]["content"]
    assert "makes 34 s of the 240 s asked; add about 206 s more" in sent_back
    assert "extend clips within their room or add moments" in sent_back
    assert kinds(turn) == ["rejected"] * 3 + ["proposal"]  # the fourth is accepted as it is
    assert "34 s of the 240 s" in turn.view()["events"][0]["text"]
    assert turn.view()["events"][3]["length"] > 34.0  # topped up by code
    assert turn.view()["events"][3]["auto"]["lengthened"]


def test_a_proposal_that_reaches_the_length_or_a_message_without_one_is_not_sent_back() -> None:
    long_add = {"op": "add", "id": "m1", "start": 0, "end": 30}
    fake = FakeProvider([propose("Add the laugh", long_add)])
    turn = run(fake, "build about 50 s")  # 24 + 30 = 54 s
    assert kinds(turn) == ["proposal"]
    fake = FakeProvider([propose("Add the laugh", {"op": "add", "id": "m1"})])
    assert kinds(run(fake, "add the laugh")) == ["proposal"]


def test_the_length_checked_is_the_exported_one() -> None:
    add = {"op": "add", "id": "m1", "start": 0, "end": 30}  # 54 s raw
    fake = FakeProvider([propose("Add the laugh", add)])
    # four clips end 20 s later when exported (4 x 20 = 80 s more): 134 s
    turn = run(fake, "make it 130 s", proj=planned_project(extra=20.0))
    assert kinds(turn) == ["proposal"]
    assert turn.view()["events"][0]["length"] == 134.0
    turn = run(
        FakeProvider([propose("Add the laugh", add)] * 4),
        "make it 200 s",
        proj=planned_project(20.0),
    )
    assert kinds(turn) == ["rejected"] * 3 + ["proposal"]


# -- a summary that claims what the ops do not do is sent back once --------------------------------


def test_a_summary_that_claims_a_change_the_ops_do_not_make_is_replaced_by_a_true_one() -> None:
    add = {"op": "add", "id": "m1"}
    fake = FakeProvider([propose("Extended the jump and added the laugh", add)])
    turn = run(fake, "add the laugh")
    assert turn.status == "done", turn.error
    assert kinds(turn) == ["proposal"]  # a summary alone does not lose the proposal
    assert turn.view()["events"][0]["summary"] == "Changes: 1 added; 34 s in all."
    # inside the asked length the same (24 + 10 = 34 s of 30)
    turn = run(FakeProvider([propose("Extended the jump", add)]), "make it 30 s")
    assert kinds(turn) == ["proposal"]
    assert turn.view()["events"][0]["summary"].startswith("Changes:")
    # outside it, the length is what sends it back, and the summary is told with it once
    fake = FakeProvider([propose("Extended the jump", add)] * 2)
    run(fake, "make it 90 s")
    sent_back = fake.requests[1]["messages"][-1]["content"]
    assert "summary says it extended" in sent_back and "really make: 1 added" in sent_back


def test_a_summary_that_matches_its_ops_goes_through() -> None:
    lengthen = {"op": "trim", "id": "m2", "end_delta": 5}
    fake = FakeProvider([propose("Lengthened the jump by 5 s", lengthen)])
    assert kinds(run(fake, "longer jump")) == ["proposal"]
    shift = [{"op": "trim", "id": "m2", "end_delta": 5}, {"op": "move", "id": "m3", "to": 1}]
    fake = FakeProvider([propose("Trimmed the jump and moved the crash first", *shift)])
    assert kinds(run(fake, "tidy")) == ["proposal"]


@pytest.mark.parametrize(
    ("summary", "op", "claimed"),
    [
        ("Shortened the jump", {"op": "trim", "id": "m2", "end_delta": 5}, "shortened"),
        ("Removed the crash", {"op": "add", "id": "m1"}, "removed"),
        ("Moved the laugh first", {"op": "add", "id": "m1"}, "moved"),
        ("Graded the colors", {"op": "add", "id": "m1"}, "color"),
    ],
)
def test_each_claim_is_checked_against_the_diff(summary: str, op: dict, claimed: str) -> None:
    rows = ops.apply_ops(
        SEQ, [{**op, "id": "c1" if op["id"] == "m1" else "c2"}], CANDS, durations()
    )
    gap = ops.summary_gap(summary, rows.diff, rows.color)
    assert gap is not None and claimed in gap


def test_a_later_turn_sees_what_the_proposal_really_changed() -> None:
    fake = FakeProvider(
        [propose("Lengthened the jump", {"op": "trim", "id": "m2", "end_delta": 5})]
    )
    rec = run(fake, "longer").record()
    assert rec["reply"] == "Proposed: Lengthened the jump (changes: 1 lengthened)"


# -- the diff tells lengthened from shortened -------------------------------------------------------


def test_the_diff_tells_lengthened_shortened_and_shifted_clips_apart() -> None:
    cases = {
        "lengthened": {"op": "trim", "id": "c2", "end_delta": 2},
        "shortened": {"op": "trim", "id": "c2", "start_delta": 2},
        "trimmed": {"op": "trim", "id": "c2", "start_delta": 2, "end_delta": 2},
    }
    for want, op in cases.items():
        out = ops.apply_ops(SEQ, [op], CANDS, durations())
        assert [r["changes"] for r in out.diff if r["changes"]] == [[want]], want
    assert ops.change_counts([{"changes": ["added"]}, {"changes": ["lengthened"]}], None) == (
        "1 added, 1 lengthened"
    )


# -- guardrails ---------------------------------------------------------------------------------


def test_ops_sent_as_a_json_string_are_taken() -> None:
    text = json.dumps([{"op": "add", "id": "m1"}])
    turn = run(FakeProvider([call("propose_edit", summary="Add the laugh", ops=text)]), "add it")
    assert kinds(turn) == ["proposal"]
    assert turn.view()["events"][0]["ops"] == [{"op": "add", "id": "c1"}]


def test_a_proposal_that_removes_every_clip_needs_the_user_to_ask_for_it() -> None:
    wipe = [
        {"op": "remove", "id": "m2"},
        {"op": "remove", "id": "m3"},
        {"op": "remove", "id": "m4"},
    ]
    fake = FakeProvider([propose("Remove them all", *wipe), propose("Remove them all", *wipe)])
    turn = run(fake, "make the montage better")
    assert turn.status == "failed" or kinds(turn)[:1] == ["rejected"]
    assert "remove every clip" in fake.requests[1]["messages"][-1]["content"]
    assert "proposal" not in kinds(turn)
    fake = FakeProvider([propose("Remove them all", *wipe)])
    assert kinds(run(fake, "start over")) == ["proposal"]
    fake = FakeProvider([propose("Remove them all", *wipe)])
    assert kinds(run(fake, "clear the sequence")) == ["proposal"]


def test_a_rejected_attempt_is_an_event_saved_with_the_turn(caplog) -> None:
    bad = propose("x", {"op": "trim", "id": "m2", "end": 999})
    fake = FakeProvider([bad, propose("Add the laugh", {"op": "add", "id": "m1"})])
    with caplog.at_level("INFO", logger="cutpoint.assistant.session"):
        turn = run(fake, "add the laugh")
    assert kinds(turn) == ["rejected", "proposal"]
    event = turn.record()["events"][0]
    assert event["kind"] == "rejected" and "is past the source's" in event["text"]
    assert any("proposal not accepted" in r.message for r in caplog.records)


def test_look_for_makes_up_to_ten_new_moments_per_call() -> None:
    assert lookfor.MAX_NEW == 10 and lookfor.MAX_NEW >= lookfor.MAX_LIMIT


# -- Ollama: AI Chat's sampling, not the Vision role's ---------------------------------------------


def test_the_vision_role_keeps_the_models_own_sampling() -> None:
    chat = providers.OllamaProvider("http://h:1", "m", 8192)
    assert chat.options() == {
        "num_ctx": 8192,
        "temperature": providers.CHAT_TEMPERATURE,
        "presence_penalty": 1.5,
        "top_k": 20,
        "top_p": 0.95,
        "num_predict": 16384,
    }
    seen = providers.make_vision(providers.AssistantSettings("ollama", "m", "http://h:1", 8192))
    assert isinstance(seen, providers.OllamaProvider) and seen.options() == {"num_ctx": 8192}


# -- snapping a lengthened end to the end of the phrase ---------------------------------------------


def test_a_trim_that_lengthens_an_end_ends_at_the_phrase_end_never_past_the_source() -> None:
    def snap(_source: str, t: float) -> float:
        return t + 3.0

    def end_of(op: dict[str, Any], **kw: Any) -> float:
        out = ops.apply_ops(SEQ, [{"op": "trim", "id": "c2", **op}], CANDS, durations(), **kw)
        assert out.error is None, out.error
        return out.after[0]["end"]

    assert end_of({"end_delta": 2}, snap=snap) == 55.0  # 52 -> 55
    assert end_of({"end": 56}, snap=snap) == 59.0
    assert end_of({"end": 60}, snap=snap) == 60.0  # the snap stays within +10 s of 50
    assert end_of({"end_delta": 2}) == 52.0  # no hook: where asked
    assert end_of({"end_delta": -2}, snap=snap) == 48.0  # shorter ends stay
    assert end_of({"start_delta": -2}, snap=snap) == 50.0  # a start is not snapped
    assert end_of({"end_delta": 2}, snap=lambda _s, t: t + 50) == 60.0  # at most +10 s from 50
    late = [{"id": "c2", "source": A, "start": 40.0, "end": 95.0, "label": "jump"}]
    out = ops.apply_ops(
        late,
        [{"op": "trim", "id": "c2", "end_delta": 2}],
        CANDS,
        durations(),
        snap=lambda _s, t: t + 50,
    )
    assert out.after[0]["end"] == 100.0  # the source ends at 100 s


def test_a_turns_lengthening_trim_is_snapped_to_the_phrase() -> None:
    lengthen = propose("Lengthened the jump", {"op": "trim", "id": "m2", "end_delta": 2})
    turn = run(FakeProvider([lengthen]), "longer jump", proj=planned_project())
    assert turn.view()["events"][0]["after"][0]["end"] == 55.0


def words(*spans: tuple[float, float]) -> list[dict[str, Any]]:
    return [{"word": "w", "start": a, "end": b} for a, b in spans]


@pytest.fixture
def planner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> edges.EdgePlanner:
    # speech 10-12.5 (words 0.3 s apart run on), a pause, speech 20-21
    spoken = words((10, 10.5), (10.8, 11.5), (11.8, 12.5), (20, 21))
    monkeypatch.setattr(edges.sources, "source_words", lambda _src, _lang=None: spoken)
    return edges.EdgePlanner(tmp_path, lambda s: tmp_path / Path(s).name)


def test_the_phrase_end_is_the_last_word_of_the_speech_that_runs_on_and_a_breath(planner) -> None:
    assert planner.phrase_end(A, 11.0) == pytest.approx(12.9)  # 12.5 + breath 0.4
    assert planner.phrase_end(A, 20.5) == pytest.approx(21.4)
    assert planner.phrase_end(A, 15.0) == 15.0  # silence: where it was asked
    assert planner.phrase_end(A, 5.0) == 5.0
    assert planner.phrase_end(A, 12.6) == 12.6  # after the phrase


def test_no_transcript_leaves_the_end_where_it_was_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(edges.sources, "source_words", lambda _src, _lang=None: None)
    assert edges.EdgePlanner(tmp_path).phrase_end(A, 11.0) == 11.0


# -- planning the exported edges ----------------------------------------------------------------------


class Plan:
    def __init__(self, entry: dict[str, Any], start: float, end: float) -> None:
        self.entry, self.start, self.end = entry, start, end


@pytest.fixture
def planning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """``plan_edges`` and ``probe`` replaced: the calls are recorded, the ends move 1.5 s."""
    seen: dict[str, Any] = {"calls": []}

    def fake_plan(moments, project, **kw):
        seen["calls"].append({"moments": moments, "project": project, **kw})
        return [Plan(m, m["start"] - 0.5, m["end"] + 1.5) for m in moments]

    monkeypatch.setattr(edges, "plan_edges", fake_plan)
    monkeypatch.setattr(edges, "probe", lambda p: p)
    cache.save_json(
        tmp_path / "highlights.json", {"style": "funny", "settings": {"language": "en"}}
    )
    return seen


def test_the_planner_plans_with_smart_tails_and_speech_aware_cuts_on(
    tmp_path: Path, planning
) -> None:
    got = edges.EdgePlanner(tmp_path, lambda s: tmp_path / Path(s).name).plan(SEQ)
    assert got["c2"] == (39.5, 51.5) and set(got) == {"c2", "c3", "x9"}
    call_ = planning["calls"][0]
    assert call_["tail"].enabled and call_["speech"].enabled and call_["language"] == "en"
    assert all("smart_tail" not in m and "speech_aware" not in m for m in call_["moments"])


def test_a_flag_off_or_a_rows_own_off_reaches_the_plan(tmp_path: Path, planning) -> None:
    planner = edges.EdgePlanner(tmp_path, lambda s: tmp_path / Path(s).name)
    planner.plan(SEQ, smart_tail=False)
    assert all(m["smart_tail"] is False for m in planning["calls"][-1]["moments"])
    rows = [{**SEQ[0], "speech_aware": False}, *SEQ[1:]]
    planner.plan(rows)
    flags = [m.get("speech_aware") for m in planning["calls"][-1]["moments"]]
    assert flags == [False, None, None]


def test_the_same_sequence_is_planned_once(tmp_path: Path, planning) -> None:
    planner = edges.EdgePlanner(tmp_path, lambda s: tmp_path / Path(s).name)
    first = planner.plan(SEQ)
    assert planner.plan([dict(r) for r in SEQ]) == first
    assert len(planning["calls"]) == 1
    planner.plan([{**SEQ[0], "end": 51.0}, *SEQ[1:]])  # one trim: planned again
    assert len(planning["calls"]) == 2
    assert edges.length(SEQ, first) == pytest.approx(24 + 3 * 2.0)


def test_a_sequence_that_cannot_be_planned_keeps_its_raw_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planning
) -> None:
    def broken(*_a, **_k):
        raise ValueError("c2: 40.00-50.00 is outside a.mp4")

    monkeypatch.setattr(edges, "plan_edges", broken)
    got = edges.EdgePlanner(tmp_path, lambda s: tmp_path / Path(s).name).plan(SEQ)
    assert got == {r["id"]: (r["start"], r["end"]) for r in SEQ}


# -- the route the page asks for the planned edges ----------------------------------------------------


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = ui.create_app(PORT)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        c.headers["X-Cutpoint-Token"] = app.state.token
        yield c


def test_the_edges_route_returns_each_rows_exported_range(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, proj, planning
) -> None:
    p, seq, _ids = proj
    before = {f: f.stat().st_mtime_ns for f in p.rglob("*") if f.is_file()}
    res = client.post("/api/projects/cli/edges", json={"segments": seq})
    assert res.status_code == 200, res.text
    edges_ = res.json()["edges"]
    assert edges_[seq[0]["id"]] == {"start": seq[0]["start"] - 0.5, "end": seq[0]["end"] + 1.5}
    assert planning["calls"][0]["tail"].enabled and planning["calls"][0]["speech"].enabled
    off = client.post(
        "/api/projects/cli/edges",
        json={"segments": seq, "smart_tail": False, "speech_aware": False},
    )
    assert off.status_code == 200
    assert all(m["smart_tail"] is False for m in planning["calls"][-1]["moments"])
    assert {
        f: f.stat().st_mtime_ns for f in p.rglob("*") if f.is_file()
    } == before  # writes nothing
    assert client.post("/api/projects/cli/edges", json={"segments": "x"}).status_code == 400
    bad = client.post("/api/projects/cli/edges", json={"segments": seq, "smart_tail": "yes"})
    assert bad.status_code == 400
    assert client.post(
        "/api/projects/cli/edges", json={}, headers={"X-Cutpoint-Token": "no"}
    ).status_code in (
        401,
        403,
    )


# -- round 2 ----------------------------------------------------------------------------------------


def done_chunk(reason: str, evaluated: int, prompt: int) -> list[dict]:
    return [
        {"message": {"role": "assistant", "content": "a long list"}, "done": False},
        {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": reason,
            "eval_count": evaluated,
            "prompt_eval_count": prompt,
        },
    ]


def test_the_output_cap_is_not_a_context_cut_off(monkeypatch: pytest.MonkeyPatch) -> None:

    def chat(lines: list[dict]) -> None:
        monkeypatch.setattr(providers.requests, "post", lambda *a, **k: _Resp(lines=lines))
        p = providers.OllamaProvider("http://h:1", "m", 50000)
        p.chat([])

    with pytest.raises(providers.OutputCapError, match="limit of 16384 tokens"):
        chat(done_chunk("length", 16384, 20000))  # a 20k prompt in 50k context
    with pytest.raises(providers.TruncatedError) as err:
        chat(done_chunk("length", 3000, 47500))  # the context ran out
    assert not isinstance(err.value, providers.OutputCapError)


def test_a_reply_at_the_output_cap_is_asked_again_shorter_without_shrinking_the_context() -> None:
    def capped(_messages: Any) -> Any:
        raise providers.OutputCapError("at its limit", 16384)

    ok = propose("Add the laugh", {"op": "add", "id": "m1"})
    fake = FakeProvider([capped, ok])
    turn = run(fake, "add the laugh")
    assert turn.status == "done", turn.error
    assert fake.requests[1]["messages"][-1]["content"] == session.SHORT_REPLY
    assert len(fake.requests[1]["messages"]) == len(fake.requests[0]["messages"]) + 1  # no shrink
    twice = run(FakeProvider([capped, capped]), "add the laugh")
    assert twice.status == "failed" and "reply limit (16384 tokens) twice" in twice.error
    assert "context_tokens" not in twice.error


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("make it ~4 min", 240),
        ("get it to 4 minutes", 240),
        ("a 4 minute long montage", 240),
        ("a 90 second reel", 90),
        ("total of 240 s", 240),
        ("under 2 minutes", 120),
        ("remove the clip at 3 minutes", None),
        ("move the 2 minute clip", None),
        ("the first 5 minutes", None),
        ("around 4:20", None),
        ("around 12:05", None),
        ("the last 30 seconds are boring", None),
    ],
)
def test_only_a_length_given_to_the_sequence_is_a_target(text: str, seconds: float | None) -> None:
    assert context.target_seconds(text) == seconds


@pytest.mark.parametrize(
    ("summary", "claims"),
    [
        ("Added the laugh at the start", True),
        ("A longer, colorful, top-grade montage with a steep drop-in and smooth moves", False),
        ("Cut out the crash and regraded the footage", True),
    ],
)
def test_a_claim_is_an_edit_verb_not_an_adjective_or_noun(summary: str, claims: bool) -> None:
    rows = [{"changes": []}]  # nothing happened
    assert (ops.summary_gap(summary, rows) is not None) is claims


def test_too_long_is_sent_back_too_and_a_retry_further_off_falls_back_on_the_first() -> None:
    long_add = {"op": "add", "id": "m1", "start": 10, "end": 20}
    fake = FakeProvider(
        [
            propose("Add the laugh", long_add),
            propose("Remove the jump", {"op": "remove", "id": "m2"}),
        ]
    )
    turn = run(fake, "make it 60 s")
    sent = fake.requests[1]["messages"][-1]["content"]
    assert "makes 34 s of the 60 s asked; add about 26 s more" in sent
    final = turn.view()["events"][-1]
    # the retry made 14 s: the first (34 s) is kept and topped up by code to the target
    assert final["kind"] == "proposal" and final["length"] >= 51 and final["auto"]
    over = run(FakeProvider([propose("Add the laugh", long_add)] * 2), "make it 20 s")
    assert "cut about 14 s" in over.view()["events"][0]["text"]


def test_a_model_that_answers_in_text_or_runs_out_of_rounds_gets_the_earlier_proposal() -> None:
    add = propose("Add the laugh", {"op": "add", "id": "m1"})
    text = Reply("I cannot reach it", [], {"role": "assistant", "content": "I cannot reach it"})
    turn = run(FakeProvider([add, text]), "make it 60 s")
    assert turn.status == "done" and kinds(turn) == ["rejected", "proposal"]
    lookups = [call("find_moments", query="laugh")] * session.MAX_ROUNDS
    turn = run(FakeProvider([add, *lookups]), "make it 60 s")
    assert turn.status == "done" and kinds(turn)[-1] == "proposal"


def test_a_trim_lengthens_an_edge_by_at_most_ten_seconds_per_proposal() -> None:
    def trim(*edits: dict) -> Any:
        return ops.apply_ops(
            SEQ, [{"op": "trim", "id": "c2", **e} for e in edits], CANDS, durations()
        )

    assert trim({"end_delta": 10}).error is None
    assert "at most +10 s per edge" in trim({"end_delta": 10.5}).error
    assert "at most +10 s per edge" in trim({"start": 29}).error
    assert (
        "at most +10 s per edge" in trim({"end_delta": 6}, {"end_delta": 6}).error
    )  # per proposal
    assert trim({"end_delta": -5}).error is None  # shorter is free


def test_an_add_stays_within_ten_seconds_of_the_moments_window() -> None:
    def add(**e: Any) -> Any:
        return ops.apply_ops(SEQ, [{"op": "add", "id": "c1", **e}], CANDS, durations())

    assert add(start=0, end=30).error is None  # 10-20, +-10
    assert "within 10 s of the moment's own 10-20" in add(end=31).error


def test_one_past_the_end_is_the_last_position() -> None:
    out = ops.apply_ops(SEQ, [{"op": "move", "id": "c2", "to": 4}], CANDS, durations())
    assert out.error is None and [s["id"] for s in out.after][-1] == "c2"
    out = ops.apply_ops(SEQ, [{"op": "add", "id": "c1", "at": 4}], CANDS, durations())
    assert out.error is None and out.after[-1]["id"] == "c1"


def test_the_room_follows_the_exported_edges() -> None:
    table = context.MomentTable(list(CANDS), list(SEQ))
    table.durations = durations()
    table.planned = {"c2": (47.0, 98.5)}  # exported edges, not the raw 40-50
    rows = [json.loads(x) for x in context.sequence_block(table).splitlines()[1:-1]]
    assert rows[0]["room_before"] == 10.0 and rows[0]["room_after"] == 1.5


@pytest.mark.parametrize(
    "text",
    [
        "remove all the laughing clips",
        "drop every clip under 3 s",
        "reset the order",
        "clear it up",
    ],
)
def test_a_narrow_request_does_not_disable_the_remove_all_guard(text: str) -> None:
    wipe = [{"op": "remove", "id": i} for i in ("m2", "m3", "m4")]
    fake = FakeProvider([propose("Remove them all", *wipe)] * 2)
    assert "proposal" not in kinds(run(fake, text))
    for ok in ("remove all clips", "remove everything", "empty the sequence", "start over"):
        assert kinds(run(FakeProvider([propose("Remove them all", *wipe)]), ok)) == ["proposal"]


def test_malformed_json_ops_are_rejected_with_the_parse_error() -> None:
    bad = call("propose_edit", summary="Add the laugh", ops='[{"op": "add", "id": "m1"')
    fake = FakeProvider([bad, propose("Add the laugh", {"op": "add", "id": "m1"})])
    run(fake, "add the laugh")
    sent = fake.requests[1]["messages"][-1]["content"]
    assert "not valid JSON" in sent and "at position" in sent
    assert "non-empty list" not in sent


# -- round 3 ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("make the video 3:30", 210),
        ("the video should be 2.5 minutes", 150),
        ("keep it to 1 min", 60),
        ("keep it to 1:30", 90),
        ("the clip at 3:30", None),
        ("around 4:20", None),
    ],
)
def test_sequence_length_phrasing_is_a_target_and_positions_are_not(
    text: str, seconds: float | None
) -> None:
    assert context.target_seconds(text) == seconds


def test_a_length_is_sent_back_up_to_three_times_then_the_closest_is_offered() -> None:
    def add(end: float) -> Any:
        return propose("Add the laugh", {"op": "add", "id": "m1", "start": 10, "end": end})

    # target 240 s, the sequence is 24 s: 34, 28 and 44 s are sent back; the fourth (29 s) is taken
    fake = FakeProvider([add(20), add(14), add(30), add(15)])
    turn = run(fake, "make it 240 s")
    texts = [e["text"] for e in turn.view()["events"] if e["kind"] == "rejected"]
    assert len(texts) == 3 and "makes 34 s of the 240 s asked; add about 206 s more" in texts[0]
    assert "makes 38 s" not in texts[1] and "still" not in texts[0]
    final = turn.view()["events"][-1]
    assert final["kind"] == "proposal"
    # the closest of the four to 240 s is the 44 s one, offered instead of the fourth
    assert final["length"] > 44.0 and final["auto"]  # topped up from the 44 s one


def test_the_fallback_is_never_over_the_tolerance_and_never_one_with_a_false_summary() -> None:
    long_add = {"op": "add", "id": "m1", "start": 10, "end": 20}  # 34 s of 30 s: over the +15%? no
    fake = FakeProvider(
        [propose("Add the laugh", long_add)] * 3 + [call("find_moments", query="x")] * 8
    )
    # 34 s against 20 s is over +15%: no fallback to offer, the turn ends without a proposal
    turn = run(fake, "make it 20 s")
    assert "proposal" not in kinds(turn)
    # a proposal sent back for a false summary is no candidate
    lie = propose("Extended the jump", {"op": "add", "id": "m1"})  # short of 240 s too
    fake = FakeProvider([lie, *[call("find_moments", query="x")] * 8])
    turn = run(fake, "make it 240 s")
    offered = turn.view()["events"][-1]
    assert offered["kind"] == "proposal" and offered["summary"].startswith("Changes: 1 added")
    assert "Extended" not in offered["summary"]


def test_ops_followed_by_punctuation_are_taken_and_other_text_is_not() -> None:
    good = call("propose_edit", summary="Add the laugh", ops='[{"op": "add", "id": "m1"}].')
    assert kinds(run(FakeProvider([good]), "add the laugh")) == ["proposal"]
    junk = call("propose_edit", summary="Add the laugh", ops='[{"op": "add", "id": "m1"}] and more')
    fake = FakeProvider([junk, propose("Add the laugh", {"op": "add", "id": "m1"})])
    run(fake, "add the laugh")
    assert "text after the JSON list" in fake.requests[1]["messages"][-1]["content"]


def test_a_reply_that_announces_more_work_gets_the_one_nudge() -> None:
    said = Reply(
        "Now let me search for more beauty shots.", [], {"role": "assistant", "content": "x"}
    )
    ok = propose("Add the laugh", {"op": "add", "id": "m1"})
    fake = FakeProvider([said, ok])
    turn = run(fake, "add some beauty shots")
    assert kinds(turn) == ["proposal"]
    assert fake.requests[1]["messages"][-1]["content"] == session.NUDGE
    # a question needs no edit: the reply stands
    fake = FakeProvider([said])
    assert kinds(run(fake, "what is in this video?")) == ["reply"]
    # only once: the second announcement is the answer
    fake = FakeProvider([said, said])
    assert kinds(run(fake, "add some beauty shots")) == ["reply"]


# -- round 4 ----------------------------------------------------------------------------------------


def evaluator(snap=None):
    """``evaluate`` of ``top_up`` over the real apply path of the ops."""

    def evaluate(op_list):
        out = ops.apply_ops(SEQ, op_list, CANDS, durations(), snap=snap)
        if out.error:
            return None
        total = sum(r["end"] - r["start"] for r in out.after)
        return {"ops": op_list, "after": out.after, "diff": out.diff, "length": total}

    return evaluate


def fill(target: float, searched=(), snap=None, extra_cands=()):
    start = {"ops": [], "after": [dict(r) for r in SEQ], "diff": [], "length": 24.0}
    return topup.top_up(
        start,
        target=target,
        tolerance=0.15,
        evaluate=evaluator(snap),
        before=SEQ,
        candidates={c["id"]: c for c in [*CANDS, *extra_cands]},
        searched=list(searched),
        durations=durations(),
        phrase_end=lambda _s, t: t + 3 if t == 9.0 else t,  # only c3 ends in speech
    )


def test_topup_lengthens_the_clips_first_the_ones_cut_mid_speech_then_adds_searched_moments() -> (
    None
):
    got = fill(60, searched=[["c1"]])
    assert got is not None and got.lengthened[0] == "c3"  # ends in speech: first
    assert set(got.lengthened) == {"c3", "c2", "x9"} and got.added == ["c1"]
    ends = {r["id"]: r["end"] for r in got.checked["after"]}
    assert ends["c2"] <= 60.0 and ends["x9"] <= 80.0  # at most +10 s per edge
    assert got.reached and 51 <= got.checked["length"] <= 69  # within 15%, none over
    extra = got.checked["ops"]
    assert all(op["auto"] is True for op in extra) and len(extra) == 4  # labelled auto ops


def test_topup_snaps_a_lengthened_end_and_never_goes_past_the_cap_or_the_source() -> None:
    got = fill(40, snap=lambda _s, t: t + 3.0)
    assert got is not None
    ends = {r["id"]: r["end"] for r in got.checked["after"]}
    assert ends["c3"] == 19.0  # 9 + 10 asked, a snap 3 s on stays within the +10 s cap
    far = fill(500)
    assert far is not None and not far.reached  # +10 s per edge is all there is: say so
    assert {r["id"]: r["end"] for r in far.checked["after"]} == {"c2": 60.0, "c3": 19.0, "x9": 80.0}


def test_topup_does_not_overshoot_and_uses_only_this_turns_searches() -> None:
    assert fill(30, snap=lambda _s, t: t + 10.0).checked["length"] <= 34.5  # the cap holds
    assert fill(30).checked["length"] <= 34.5
    nothing = fill(200)  # lengthening alone; c1 was never searched for
    assert nothing is not None and nothing.added == []
    overlap = {"id": "c9", "source": A, "start": 41.0, "end": 49.0, "label": "same place"}
    got = fill(80, searched=[["c2", "c9", "c1"]], extra_cands=[overlap])
    assert got.added == ["c1"]  # c2 is in the sequence and c9 overlaps it


def test_topup_does_nothing_for_a_sequence_that_is_long_enough() -> None:
    assert fill(25) is None


def test_a_short_final_proposal_is_filled_in_the_proposal_the_user_reviews() -> None:
    add = propose("Add the laugh", {"op": "add", "id": "m1"})
    fake = FakeProvider([call("find_moments", query="crash laugh"), add, add, add, add])
    turn = run(fake, "make it 70 s")
    event = next(e for e in turn.view()["events"] if e["kind"] == "proposal")
    assert event["auto"]["reached"] and 59.5 <= event["length"] <= 80.5
    assert "Auto-filled to reach ~1:10" in event["summary"]
    auto_rows = [r for r in event["diff"] if r.get("auto")]
    assert auto_rows and all(op.get("auto") for op in event["ops"][1:])
    again = ops.apply_ops(SEQ, event["ops"], CANDS, durations())  # the normal apply path
    assert again.error is None and again.after == event["after"]
    assert (
        f"{len(event['auto']['lengthened']) + len(event['auto']['added'])} auto-filled"
        in (turn.record()["reply"])
    )


def test_without_searches_the_fill_lengthens_only_and_says_how_far_it_got() -> None:
    add = propose("Add the laugh", {"op": "add", "id": "m1"})
    turn = run(FakeProvider([add] * 4), "make it 240 s")
    event = turn.view()["events"][-1]
    assert event["auto"]["added"] == [] and event["auto"]["reached"] is False
    assert "Auto-filled towards ~4:00" in event["summary"]


# summary claims about order (T3)
def test_sorted_by_time_over_added_clips_is_not_a_move_claim() -> None:
    added = [{"changes": ["added"]}, {"changes": ["added"]}]
    assert ops.summary_gap("Added two clips, sorted by time", added) is None
    assert ops.summary_gap("Sorted by score", [{"changes": []}]) is not None
    assert ops.summary_gap("Sorted by score", [{"changes": ["moved"]}]) is None
    assert ops.summary_gap("Moved the crash first", added) is not None


# a position out of range appends (T4)
@pytest.mark.parametrize("at", [0, 4, 9])
def test_an_add_at_a_position_out_of_range_is_appended(at: int) -> None:
    out = ops.apply_ops(SEQ, [{"op": "add", "id": "c1", "at": at}], CANDS, durations())
    assert out.error is None and out.after[-1]["id"] == "c1"
    empty = ops.apply_ops([], [{"op": "add", "id": "c1", "at": at}], CANDS, durations())
    assert empty.error is None and [r["id"] for r in empty.after] == ["c1"]


@pytest.mark.parametrize("at", [-1, 1.5, "2", True])
def test_an_add_at_a_negative_or_odd_position_is_still_rejected(at: Any) -> None:
    out = ops.apply_ops(SEQ, [{"op": "add", "id": "c1", "at": at}], CANDS, durations())
    assert out.error and "at must be a position" in out.error


# -- round 5 ----------------------------------------------------------------------------------------


def test_a_proposal_that_removes_half_the_clips_unasked_is_rejected_and_never_a_base() -> None:
    drop = [{"op": "remove", "id": "m2"}, {"op": "remove", "id": "m3"}]  # 2 of 3 rows
    fake = FakeProvider([propose("Fresh start", *drop), propose("Fresh start", *drop)])
    turn = run(fake, "Create a new sequence of beauty shots")
    sent = fake.requests[1]["messages"][-1]["content"]
    assert "you removed 2 of 3 clips; the user didn't ask to remove any" in sent
    assert "proposal" not in kinds(turn)  # nothing to keep as a fallback or to fill up
    ok = run(FakeProvider([propose("Take them out", *drop)]), "remove the jump and the crash")
    assert kinds(ok) == ["proposal"]
    one = run(FakeProvider([propose("Drop one", {"op": "remove", "id": "m2"})]), "tidy up")
    assert kinds(one) == ["proposal"]  # under half


def test_adds_take_one_hit_from_each_search_in_turn() -> None:
    cands = [
        {"id": f"n{i}", "source": A, "start": 82.0 + 2 * i, "end": 83.5 + 2 * i, "label": "x"}
        for i in range(1, 7)
    ]
    start = {"ops": [], "after": [dict(r) for r in SEQ], "diff": [], "length": 24.0}

    def evaluate(op_list):
        out = ops.apply_ops(SEQ, op_list, [*CANDS, *cands], durations())
        total = sum(r["end"] - r["start"] for r in out.after) if not out.error else 0
        return (
            None if out.error else {"ops": op_list, "after": out.after, "diff": [], "length": total}
        )

    got = topup.top_up(
        start,
        target=60,
        tolerance=0.15,
        evaluate=evaluate,
        before=SEQ,
        candidates={c["id"]: c for c in [*CANDS, *cands]},
        searched=[["n1", "n2", "n3"], ["n4", "n5"], ["n1", "n6"]],  # beauty, skiing, dialogue
        durations=durations(),
    )
    assert got is not None and got.added == ["n1", "n4"]  # the first of each search in turn
    assert list(topup._round_robin([["a", "b", "c"], ["d"], ["a", "e"]])) == [
        "a",
        "d",
        "b",
        "e",
        "c",
    ]


def test_the_gap_is_split_between_lengthening_and_adding() -> None:
    got = fill(60, searched=[["c1"]], extra_cands=[])
    assert got is not None
    order = [("trim" if op["op"] == "trim" else "add", op["id"]) for op in got.checked["ops"]]
    assert order[0] == ("trim", "c3")  # the clip cut mid-speech, 4 s long: first
    assert order[1] == ("add", "c1")  # then a moment before the next lengthening
    assert order[2][0] == "trim"
    assert got.reached and got.checked["length"] <= 69
    # not every clip got +10 s before the add: the target was reached with one clip untouched
    near = fill(50, searched=[["c1"]])
    assert near is not None and len(near.lengthened) < 3 and near.added == ["c1"]


# -- round 6 ----------------------------------------------------------------------------------------


def kinded(i: str, kind: str, score: int, start: float) -> dict[str, Any]:
    return {
        "id": i,
        "source": A,
        "start": start,
        "end": start + 4,
        "score": score,
        "label": i,
        "kind": kind,
    }


KINDS = [
    kinded("c2", "view", 70, 40.0),
    kinded("c3", "speed", 60, 5.0),
    kinded("v1", "view", 90, 80.0),
    kinded("v2", "view", 50, 86.0),
    kinded("s1", "speed", 70, 92.0),
    kinded("l1", "laugh", 99, 20.0),  # a kind the sequence does not have
]
KSEQ = [
    {"id": "c2", "source": A, "start": 40.0, "end": 44.0, "label": "c2"},
    {"id": "c3", "source": B, "start": 5.0, "end": 9.0, "label": "c3"},
]


def test_kind_groups_are_the_best_unused_moments_of_each_kind_in_the_sequence() -> None:
    cands = {c["id"]: c for c in KINDS}
    groups = topup.kind_groups(KSEQ, cands, lambda c: c["kind"])
    assert groups == [["v1", "v2"], ["s1"]]  # view, speed; laugh is not in the sequence
    assert topup.kind_groups([], cands, lambda c: c["kind"]) == [
        ["l1", "v1", "c2", "s1", "c3", "v2"]
    ]
    assert topup.kind_groups(KSEQ, {}, lambda c: c.get("kind", "")) == []


def test_without_searches_the_fill_adds_best_scoring_moments_by_the_kinds_present() -> None:
    p = session.Project(list(KINDS), list(KSEQ), durations(), "", "key0")
    short = propose("Lengthen the jump", {"op": "trim", "id": "m1", "end_delta": 1})
    turn = run(FakeProvider([short] * 4), "make it 40 s", proj=p)
    event = next(e for e in turn.view()["events"] if e["kind"] == "proposal")
    assert event["auto"]["added"][:2] == ["v1", "s1"]  # one of each kind in turn, best first
    assert "l1" not in event["auto"]["added"] and event["auto"]["reached"]
    assert 34 <= event["length"] <= 46


def test_with_searches_the_fill_still_goes_by_the_searches() -> None:
    laugh = [*KINDS[:-1], {**KINDS[-1], "label": "laugh"}]
    p = session.Project(laugh, list(KSEQ), durations(), "", "key0")
    short = propose("Lengthen the jump", {"op": "trim", "id": "m1", "end_delta": 1})
    turn = run(
        FakeProvider([call("find_moments", query="laugh"), *[short] * 4]), "make it 40 s", proj=p
    )
    event = next(e for e in turn.view()["events"] if e["kind"] == "proposal")
    assert "l1" in event["auto"]["added"] and "v1" not in event["auto"]["added"]


def text_reply(text: str) -> Reply:
    return Reply(text, [], {"role": "assistant", "content": text})


def test_a_turn_that_ends_in_text_gets_a_labelled_fill_to_review() -> None:
    p = session.Project(list(KINDS), list(KSEQ), durations(), "", "key0")
    turn = run(FakeProvider([text_reply("I would add a few more views.")]), "make it 40 s", proj=p)
    assert kinds(turn) == ["reply", "proposal"]  # the text stays
    event = turn.view()["events"][1]
    assert "(the assistant didn't propose an edit)" in event["summary"]
    assert event["summary"].startswith("Auto-filled to reach ~0:40")
    assert event["ops"] and all(op["auto"] is True for op in event["ops"])
    assert all(row.get("auto") for row in event["diff"] if row["changes"])
    again = ops.apply_ops(KSEQ, event["ops"], KINDS, durations())  # the normal apply path
    assert again.error is None and again.after == event["after"]
    assert turn.status == "done"


def test_the_rounds_running_out_gets_the_fill_too() -> None:
    p = session.Project(list(KINDS), list(KSEQ), durations(), "", "key0")
    lookups = [call("find_moments", query="v1")] * session.MAX_ROUNDS
    turn = run(FakeProvider(lookups), "make it 40 s", proj=p)
    assert turn.status == "done" and kinds(turn)[-1] == "proposal"
    assert "didn't propose an edit" in turn.view()["events"][-1]["summary"]


def test_no_fill_for_a_question_a_long_enough_sequence_or_no_target() -> None:
    p = session.Project(list(KINDS), list(KSEQ), durations(), "", "key0")
    said = text_reply("It is 8 s now.")
    assert kinds(run(FakeProvider([said]), "can it be 40 s long?", proj=p)) == ["reply"]
    assert kinds(run(FakeProvider([said]), "make it 8 s", proj=p)) == ["reply"]
    assert kinds(run(FakeProvider([said]), "what is in it?", proj=p)) == ["reply"]


# -- review fixes (PR #737) ---------------------------------------------------------------------------


def test_a_phrase_end_far_away_leaves_the_end_and_a_trim_never_passes_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = words(*[(10 + i * 0.5, 10.4 + i * 0.5) for i in range(240)])  # 120 s of speech
    monkeypatch.setattr(edges.sources, "source_words", lambda _src, _lang=None: chain)
    planner = edges.EdgePlanner(tmp_path, lambda s: tmp_path / Path(s).name)
    assert planner.phrase_end(A, 20.0) == 20.0  # the phrase ends over max_extend later
    seq = [{"id": "c1", "source": A, "start": 10.0, "end": 20.0, "label": "x"}]
    out = ops.apply_ops(
        seq,
        [{"op": "trim", "id": "c1", "end_delta": 2}],
        CANDS,
        durations(),
        snap=planner.phrase_end,
    )
    assert out.error is None and out.after[0]["end"] == 22.0
    far = lambda _s, t: t + 50  # noqa: E731
    out = ops.apply_ops(
        seq, [{"op": "trim", "id": "c1", "end_delta": 2}], CANDS, durations(), snap=far
    )
    assert out.after[0]["end"] == 30.0  # first end 20 + MAX_GROW_S


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("make each clip about 5 s", None),
        ("I want 10s clips", None),
        ("need clips around 8 seconds", None),
        ("need 30 s more", None),
        ("make it 5 min longer", None),
        ("every clip around 6 s", None),
        ("make each clip, about 5 s", None),
        ("every clip (about 5 s)", None),
        ("per clip: 5 s", None),
        ("make each clip about 1 minute", None),
        ("make a montage of clips around 2 minutes", 120),
        ("make it 4 min and 30 s", 270),
        ("make it 4 min 30 s", 270),
        ("about 2 minutes and 15 seconds", 135),
        ("make it 90 seconds", 90),
        ("pick 8 clips, ~3 min", 180),
        ("use the best clips, around 2 minutes", 120),
        ("remove the boring clips, under 2 minutes", 120),
        ("use fewer clips, about 1 min", 60),
        ("add 2 more clips, about 4 min", 240),
        ("8 clips, 3 minutes long", 180),
        ("make it about 4 minutes more or less", 240),
        ("each clip - about 5 s", None),
        ("each clip -- around 6 seconds", None),
        ("for every clip, ~4 s", None),
        ("every clip, around 1 min", None),
        ("per clip: about 1 min", None),
        ("each clip should be about 5 s", None),
        ("each clip should be about 1 minute", None),
        ("make every clip about 2 minutes", None),
        ("per clip about 1.5 min", None),
        ("about 5 s each", None),
        ("about 1 min each", None),
        ("about 1 min per clip", None),
        ("make them 5 s each", None),
        ("around 1 minute a clip", None),
        ("use clips about 3 min", 180),
        ("best clips, about 2 min", 120),
        ("funny clips around 1.5 minutes", 90),
        ("clips around 2 minutes", 120),
        ("clip about 2 min", 120),
        ("use clips about 10 s", None),
        ("clip about 5 s", None),
        ("pick 8 clips, ~30 s", 30),
        ("use 3 clips, about 45 seconds", 45),
        ("Trim every clip. About 3 minutes total", 180),
        ("shorten each clip. under 2 minutes", 120),
        ("tighten each clip! around 3 min", 180),
        ("tighten each clip; around 3 min", 180),
        ("shorten each clip.\nAbout 3 minutes total", 180),
        ("tighten each clip. about 90 s", 90),
        ("make the video about 3 minutes", 180),
        ("the montage should be around 2 min", 120),
        ("keep it under 90 seconds", 90),
        ("aim for 2 minutes", 120),
        ("roughly 3 minutes long", 180),
        ("a 2 minute reel", 120),
        ("target of 150 s", 150),
        ("i need about a minute", 60),
        ("I want the whole thing about two minutes", 120),
        ("can you make it 3 mins", 180),
        ("I want clips of about 5 seconds", None),
        ("make the clips about 5 s long", None),
        ("10-second clips please", None),
        ("keep the clips around 6 s", None),
        ("each one about 5 s", None),
        ("every moment about 6 s", None),
        ("make each clip about 5 s long", None),
        ("each clip should be 5 s long", None),
        ("each clip should be under 1 min", None),
        ("for each of the clips, about 5 s", None),
        ("make every clip about 5 s, and the video about 2 min", 120),
        ("each clip 5 s, the whole thing about 2 min", 120),
        ("the clips should be about 4 min in total", 240),
        ("make it 4 minutes, 30 s per clip", 240),
        ("make it 2 min with 10 s clips", 120),
        ("make it 3 minutes; 5 s each", 180),
        ("keep it to 4 min, more or less", 240),
        ("each of the clips about 1 minute", None),
        ("each clip about 5 s, about 2 min total", 120),
        ("each clip about 5 s, make it about 2 min", 120),
        ("every clip about 4 s; around 90 s overall", 90),
        ("tighten each clip and make it about 3 minutes", 180),
        ("shorten every clip so the video is about 2 min", 120),
        ("trim each clip, the whole thing under 2 minutes", 120),
        ("tighten every clip, total around 3 min", 180),
        ("cut each clip down and keep it under 90 seconds", 90),
        ("make each clip snappier and get it to 2 min", 120),
        ("tighten each clip and make the video about 3 min", 180),
        ("tighten every clip and aim for a 2 minute montage", 120),
    ],
)
def test_a_length_for_each_clip_or_a_change_is_no_sequence_target(
    text: str, seconds: float | None
) -> None:
    assert context.target_seconds(text) == seconds


def test_a_length_below_the_sequence_may_remove_half_the_clips() -> None:
    drop = [{"op": "remove", "id": "m3"}, {"op": "remove", "id": "m4"}]  # leaves the 10 s jump
    fake = FakeProvider([propose("Take them out", *drop)])
    turn = run(fake, "cut it to 10 s")
    assert kinds(turn) == ["proposal"]
    # no length asked: the guard stays
    fake = FakeProvider([propose("Take them out", *drop)] * 2)
    assert "proposal" not in kinds(run(fake, "make the montage better"))
    wipe = [{"op": "remove", "id": i} for i in ("m2", "m3", "m4")]
    fake = FakeProvider([propose("Take them out", *wipe)] * 2)
    assert "proposal" not in kinds(run(fake, "cut it to 5 s"))  # every clip: still refused


def test_a_proposal_far_over_the_length_is_never_shown_after_the_send_backs() -> None:
    big = propose("Add the laugh", {"op": "add", "id": "m1", "start": 0, "end": 30})  # 54 s
    turn = run(FakeProvider([big] * 6), "make it 20 s")
    assert "proposal" not in kinds(turn)
    assert [e["kind"] for e in turn.view()["events"]].count("rejected") >= 4


def test_a_trim_after_an_add_in_the_same_proposal_keeps_the_cap() -> None:
    both = [
        {"op": "add", "id": "c1"},
        {"op": "trim", "id": "c1", "end_delta": 25},
    ]
    out = ops.apply_ops(SEQ, both, CANDS, durations())
    assert out.error is not None and "at most +10 s per edge" in out.error


def test_a_failed_plan_is_not_cached_and_one_bad_source_keeps_only_its_rows_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planning
) -> None:
    def probe(path: Path) -> Path:
        if path.name == Path(B).name:
            raise OSError("unreadable")
        return path

    monkeypatch.setattr(edges, "probe", probe)
    planner = edges.EdgePlanner(tmp_path, lambda s: tmp_path / Path(s).name)
    got = planner.plan(SEQ)
    assert got["c2"] == (39.5, 51.5) and got["x9"] == (59.5, 71.5)  # planned
    assert got["c3"] == (5.0, 9.0)  # the unreadable source: raw
    assert [m["id"] for m in planning["calls"][0]["moments"]] == ["c2", "x9"]
    planner.plan(SEQ)
    assert len(planning["calls"]) == 2  # not cached
    monkeypatch.setattr(edges, "probe", lambda p: p)
    assert planner.plan(SEQ)["c3"] == (4.5, 10.5)  # the source is back
    planner.plan(SEQ)
    assert len(planning["calls"]) == 3  # a whole plan is


def test_a_source_that_changed_is_planned_again(tmp_path: Path, planning) -> None:
    src = tmp_path / Path(A).name
    src.write_bytes(b"x")
    planner = edges.EdgePlanner(tmp_path, lambda s: tmp_path / Path(s).name)
    planner.plan(SEQ)
    src.write_bytes(b"xy")
    planner.plan(SEQ)
    assert len(planning["calls"]) == 2
