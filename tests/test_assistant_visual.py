"""What analysis saw, for the assistant (#634): find_moments searches the vision labels and the
events VLM windows, a moment's row carries its cues, and a generic label gives way to what the
vision model saw, in AI Chat and in the ``cutpoint ui`` moments list."""

from __future__ import annotations

import json

from cutpoint import ui
from cutpoint.assistant import context, session
from test_assistant import FakeProvider, call, durations

A, B = "/v/a.mp4", "/v/b.mp4"
SKI = {  # an action-cam candidate's signals: the grids' maximum over the clip, 0-1
    "airtime": 0.0,
    "crash": 0.0,
    "vlm": 0.4,
    "speed": 0.62,
    "social": 0.0,
    "carve": 0.71,
    "scenic": 0.1,
    "speech": 0.0,
    "faces": 0.0,
    "stopped": 0.0,
    "short_clip": 0.0,
    "beauty": 0.33,
    "people": 0.05,
    "group": 0.0,
    "voice_laughter": 0.0,
    "voice": 0.2,
    "wipeout": 0.0,
    "keywords": [],
    "vision": None,
}
CANDS = [
    {"id": "c1", "source": A, "start": 10.0, "end": 20.0, "score": 90, "label": "big laugh",
     "transcript": "haha no way", "signals": {"events": {"laughter": 0.3, "scream": 0.0}}},
    {"id": "c2", "source": A, "start": 40.0, "end": 50.0, "score": 70, "label": "action-moment",
     "vision_label": "smooth skiing run", "signals": {"vision": 6.4}},
    {"id": "c3", "source": B, "start": 5.0, "end": 12.0, "score": 60, "label": "carve",
     "kind": "carve", "signals": SKI},
    {"id": "c4", "source": B, "start": 30.0, "end": 38.0, "score": 50, "label": "action-moment",
     "signals": {"events": {"vlm": {"label": "chairlift-ride", "highlight": 2}}}},
    {"id": "v001", "source": B, "start": 60.0, "end": 66.0, "score": 40, "label": "view",
     "pool": "visual", "kind": "view", "signals": {**SKI, "beauty": 0.9, "stopped": 1.0}},
]  # fmt: skip
EVENTS = {
    "windows": [
        {"source": B, "start": 4.0, "end": 14.0, "label": "fast-groomer",
         "summary": "Rider carves down a wide groomed run past snowy pines.",
         "highlight_score": 6, "funny_score": 1,
         "tags": {"fast_groomer": 8, "trees": 5, "jump_air": 0},
         "events": [{"t": 8.0, "what": "spray off a hard turn", "funny": 0, "hype": 6}]},
        {"source": B, "start": 58.0, "end": 70.0, "label": "lift-view",
         "summary": "Wide view of the valley from a stop at the top.",
         "highlight_score": 7, "funny_score": 0, "tags": {"scenic_vista": 9}},
        {"source": A, "start": 4.0, "end": 14.0, "summary": 3},  # not text: left out
        {"source": B, "start": 9.0, "end": 9.0},  # empty: dropped
        "junk",
    ]
}  # fmt: skip


def table(segments=()) -> context.MomentTable:
    return context.MomentTable(list(CANDS), list(segments), context.vlm_windows(EVENTS))


def test_vlm_windows_keeps_checked_fields_only() -> None:
    wins = context.vlm_windows(EVENTS)
    assert [(w["source"], w["start"]) for w in wins] == [(B, 4.0), (B, 58.0), (A, 4.0)]
    assert wins[0]["label"] == "fast groomer" and wins[0]["what"] == ["spray off a hard turn"]
    assert wins[0]["tags"]["fast_groomer"] == 8 and wins[0]["tags"]["chairlift"] == 0
    assert wins[2]["summary"] == "" and wins[2]["label"] == ""
    assert context.vlm_windows(None) == [] and context.vlm_windows({"windows": "x"}) == []


def test_find_moments_matches_vision_labels_vlm_text_and_tags() -> None:
    t = table()
    assert t.find("smooth skiing") == ["c2"]  # vision_label
    assert t.find("pines") == ["c3"]  # window summary
    assert t.find("spray") == ["c3"]  # a moment the window named
    assert t.find("chairlift") == ["c4"]  # the VLM window label in signals.events
    assert t.find("valley") == ["v001"]
    assert t.find("laughter") == ["c1"]  # heard


def test_find_moments_matches_strong_cues_by_name_strongest_first() -> None:
    t = table()
    # beauty 9 (v001) is found; beauty 3 (c3) is too weak to count
    assert t.find("beauty shots") == ["v001"]
    assert t.find("scenic") == ["v001"]  # scenic_vista 9 (c3's scenic is 1)
    assert t.find("carve") == ["c3", "v001"]  # carve 7 both; c3 also has it as label and kind
    # equal word counts: the stronger cue the word names goes first, whatever the rank
    hi = {**CANDS[2], "id": "w2", "signals": {**SKI, "speed": 0.95}}
    lo = {**CANDS[2], "id": "w1", "signals": {**SKI, "speed": 0.55}}
    assert context.MomentTable([lo, hi], []).find("speed") == ["w2", "w1"]


def test_a_cue_word_matches_parts_of_cue_names_by_prefix() -> None:
    """#634 F1: "air" names airtime and jump_air, not chairlift, so a lift ride never outranks
    a jump for "big air"."""
    lift = {"id": "lift", "source": A, "start": 0.0, "end": 8.0, "score": 90, "label": "ride"}
    jump = {**lift, "id": "jump", "start": 20.0, "end": 28.0, "score": 50, "label": "send"}
    wins = context.vlm_windows(
        {
            "windows": [
                {"source": A, "start": 0.0, "end": 8.0, "tags": {"chairlift": 9}},
                {"source": A, "start": 20.0, "end": 28.0, "tags": {"jump_air": 8}},
            ]
        }
    )
    t = context.MomentTable([lift, jump], [], wins)
    assert t.find("big air") == ["jump"]
    assert t.find("chair") == ["lift"]
    assert t.find("lift") == ["lift"]
    assert not context._names("air", "chairlift")
    assert context._names("crash", "crash_fall") and not context._names("air", "chairlift")


def test_a_row_carries_compact_cues_seen_heard_and_kind() -> None:
    t = table()
    row = t.row("c3")
    assert row["kind"] == "carve"
    cues = row["cues"]
    assert list(cues)[:4] == ["fast_groomer", "carve", "speed", "trees"]
    assert len(cues) <= context.TABLE_CUES and all(isinstance(v, int) for v in cues.values())
    assert all(0 < v <= 10 for v in cues.values()) and "vlm" not in cues and "voice" not in cues
    assert row["seen"] == "Rider carves down a wide groomed run past snowy pines."
    full = t.row("c3", full=True)
    assert len(full["cues"]) > len(cues) and full["cues"]["funny"] == 1
    assert t.row("c1")["heard"] == ["laughter"]
    assert "cues" not in t.row("c1")
    assert t.row("c2")["cues"] == {"vision": 6} and "seen" not in t.row("c2")  # label says it
    assert t.row("v001")["kind"] == "view"  # #635 visual pool kinds show in the compact row


def test_seen_is_cut_in_the_table_and_whole_in_get_moment() -> None:
    long = {**EVENTS["windows"][0], "summary": "x" * 500}
    t = context.MomentTable(list(CANDS), [], context.vlm_windows({"windows": [long]}))
    assert len(t.row("c3")["seen"]) == context.SEEN_CHARS + 1  # with the ellipsis
    assert t.row("c3", full=True)["seen"] == "x" * 500


def test_a_row_stays_small() -> None:
    """The table rows grow by a bounded amount: every field present, at most ~300 characters
    over the label, file, times and transcript snippet."""
    t = table()
    for real in t.rank:
        line = json.dumps(t.row(real), ensure_ascii=False, separators=(",", ":"))
        assert len(line) <= 330, line


def test_generic_labels_give_way_to_what_the_vision_model_saw() -> None:
    t = table([{"id": "c2", "source": A, "start": 40.0, "end": 50.0, "label": "action-moment"}])
    assert t.row("c2")["label"] == "smooth skiing run"  # vision_label
    assert t.row("c4")["label"] == "chairlift ride"  # signals.events.vlm.label
    assert t.row("c1")["label"] == "big laugh"  # a real label stays
    assert t.row("c3")["label"] == "carve"
    assert '"label": "smooth skiing run"' in context.sequence_block(t).replace('":"', '": "')
    # the events.json window label when the candidate has nothing of its own
    bare = {"id": "z", "source": B, "start": 6.0, "end": 12.0, "label": "loud-moment"}
    assert context.MomentTable([bare], [], context.vlm_windows(EVENTS)).row("z")["label"] == (
        "fast groomer"
    )
    assert context.shown_label("vlm", {}) == "vlm"  # nothing better
    assert context.shown_label("", {"vision_label": " static  group shot "}) == "static group shot"
    renamed = table([{"id": "c2", "source": A, "start": 40.0, "end": 50.0, "label": "my pick"}])
    assert renamed.row("c2")["label"] == "my pick"  # the user's own name stays


def test_the_moments_list_shows_the_same_label_and_keeps_the_analysis_one() -> None:
    hl = {"candidates": [dict(c) for c in CANDS]}
    view = ui.moments_view(hl)["candidates"]
    assert [c["display_label"] for c in view] == [
        "big laugh", "smooth skiing run", "carve", "chairlift ride", "view",
    ]  # fmt: skip
    assert [c["label"] for c in view] == [c["label"] for c in CANDS]
    assert hl["candidates"][1]["label"] == "action-moment"  # highlights.json is not changed


def test_the_rules_and_tool_say_to_use_the_visual_fields_before_asking() -> None:
    rules = context.rules(table())
    for word in (
        "seen",
        "cues",
        "heard",
        "kind",
        "beauty (a steady",
        "scenic_vista",
        "find_moments",
    ):
        assert word in rules, word
    assert "beauty shots = high beauty or scenic_vista with stopped" in rules
    assert "casual riding = high speed or carve or fast_groomer." in rules  # no airtime or crash
    assert "before asking the user for moment ids" in rules
    msgs = context.build_messages(table(), [], "hi", context.budget(8192))
    assert msgs[0]["content"] == rules
    find = next(t for t in session.TOOLS if t["function"]["name"] == "find_moments")
    assert "cue names" in find["function"]["description"]


def test_the_rules_name_only_the_cues_the_project_has() -> None:
    """#634 F6: a gaming or funny project gets no action-cam cue names or searches."""
    funny = [{**CANDS[0], "signals": {"laughter": 0.8, "events": {"laughter": 0.3}}}]
    rules = context.rules(context.MomentTable(funny, []))
    assert "Cues: laughter (said)." in rules and "heard (" in rules
    for word in ("beauty", "chairlift", "airtime", "bloopers", "casual", "seen (", "kind ("):
        assert word not in rules, word
    bare = context.rules(context.MomentTable([{**CANDS[0], "signals": {}}], []))
    assert "What analysis saw" not in bare
    assert "Look moments up with find_moments before asking the user for moment ids." in bare
    crash = [{**CANDS[0], "signals": {"crash": 0.9, "events": {"laughter": 0.2}}}]
    assert "bloopers = high crash with laughter heard" in context.rules(
        context.MomentTable(crash, [])
    )


def test_a_turn_searches_the_windows_of_its_project() -> None:
    proj = session.Project(list(CANDS), [], durations(), "", "k", context.vlm_windows(EVENTS))
    provider = FakeProvider([call("find_moments", query="valley view")])
    turn = session.Turn(provider, proj, "find the valley view")
    turn.start()
    assert turn.done.wait(10)
    result = provider.requests[1]["messages"][-1]["content"]
    assert '"label": "view"' in result and "valley" in result
    assert "scenic_vista" in result
