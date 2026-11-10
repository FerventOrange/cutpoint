"""AI Chat pacing: when the message asks to break up or mix kinds, code spreads the funny,
dialogue and scenic clips between the riding runs (labelled auto ops), and adds dialogue moments
when the message names dialogue and the proposal has none.

The model is the scripted ``FakeProvider`` of ``test_assistant``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from cutpoint.assistant import ops, pacing, session
from test_assistant import A, FakeProvider, call

FAST, LAUGH, VIEW = "fast", "laugh", "stunning-mountain-views"
TURN = (
    "Create a ~4 minute sequence of beautiful shots, good skiing of me and my friends and an "
    "occasional funny moment and dialogue to break up some of the skiing shots."
)


def durations() -> ops.Durations:
    return ops.Durations({A: 10000.0}, lambda _p: None)


def moment(
    i: int, label: str, *, length: float = 10.0, said: str = "", score: float = 50, **kw: Any
):
    start = 10.0 * i
    return {
        "id": f"k{i}",
        "source": A,
        "start": start,
        "end": start + length,
        "score": score,
        "label": label,
        **({"transcript": said} if said else {}),
        **kw,
    }


# the real turn: scenic, 2 fast, scenic, scenic, scenic, then fast clips with 4 laughs among the tail
REAL = [
    VIEW,
    FAST,
    FAST,
    VIEW,
    VIEW,
    VIEW,
    *[FAST] * 10,
    LAUGH,
    LAUGH,
    FAST,
    FAST,
    LAUGH,
    FAST,
    LAUGH,
]


def real_project(extra=(), segments=()) -> session.Project:
    cands = [moment(i, label) for i, label in enumerate(REAL)] + list(extra)
    return session.Project(cands, list(segments), durations(), "", "key0")


def short(cid: str) -> str:
    """The id the model sees: the moments are k0, k1, ... in rank order, m1, m2, ... for it."""
    return f"m{int(cid[1:]) + 1}"


def add_all(cands) -> Any:
    return call(
        "propose_edit",
        summary="Build the sequence",
        ops=[{"op": "add", "id": short(c["id"])} for c in cands],
    )


def proposal(turn: session.Turn) -> dict[str, Any]:
    return next(e for e in turn.view()["events"] if e["kind"] == "proposal")


def play(text: str, proj: session.Project, *script: Any) -> session.Turn:
    turn = session.Turn(FakeProvider(list(script) or [add_all(proj.candidates)] * 4), proj, text)
    turn.start()
    assert turn.done.wait(10)
    return turn


def kinds_of(turn: session.Turn, proj: session.Project) -> list[str]:
    by_id = {c["id"]: c for c in proj.candidates}
    return [
        pacing.moment_kind(by_id[r["id"]]["label"], by_id[r["id"]], {}, lambda: False)
        for r in proposal(turn)["after"]
    ]


def longest_run(kinds: list[str], kind: str = pacing.ACTION) -> int:
    best = run = 0
    for k in kinds:
        run = run + 1 if k == kind else 0
        best = max(best, run)
    return best


# -- the wording --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        TURN,
        "break up the skiing with some funny moments",
        "mix in some scenic shots",
        "a good mix of skiing and views",
        "intersperse the laughs between the runs",
        "sprinkle in a few funny clips",
        "add a funny moment now and then",
        "put a view in between the skiing clips",
        "alternate skiing and scenery",
        "vary the pace please",
        "an occasional joke",
        "no music, just break up the skiing",
        "No music. Break up the skiing with laughs",
        "not too long, mix in some views",
        "no fades, but alternate skiing and scenery",
        "don't cut it short; break up the skiing",
        "not too long, with dialogue to break up the runs",
        "no music. add some dialogue and break up the runs",
    ],
)
def test_wording_that_asks_for_a_mix_is_found(text: str) -> None:
    assert pacing.asks_for_mix(text)


@pytest.mark.parametrize(
    "text",
    [
        "put a view in between the biking clips",
        "break up the surfing with some dialogue",
        "alternate surf and scenery",
        "between the mtb runs add a funny moment",
        "add dialogue between the snowboard runs",
        "put a view in between the skiing clips",
    ],
)
def test_mix_wording_names_any_ride_sport_not_only_skiing(text: str) -> None:
    assert pacing.asks_for_mix(text)


@pytest.mark.parametrize(
    "label",
    ["biking", "mtb-trail", "surfing", "skateboard-skating", "snowboarding", "skiing"],
)
def test_ride_sport_labels_are_action(label: str) -> None:
    assert pacing.moment_kind(label, None, {}, lambda: False) == "action"


@pytest.mark.parametrize("label", ["drop-the-mic", "wave-hello"])
def test_everyday_verb_labels_are_not_action_words(label: str) -> None:
    """A moment falls back to action, so check these labels no longer name an action word."""
    assert not set(label.split("-")) & pacing._ACTION_WORDS


def test_python_and_web_action_words_stay_in_sync() -> None:
    """``pacing._ACTION_WORDS`` and ``ACTION_WORDS`` in web/util.js name the same words."""
    js = (Path(pacing.__file__).parents[1] / "web" / "util.js").read_text(encoding="utf-8")
    found = re.search(r"const ACTION_WORDS = \[(.*?)\];", js, re.S)
    assert found, "ACTION_WORDS not found in web/util.js"
    assert set(re.findall(r"'([^']+)'", found.group(1))) == set(pacing._ACTION_WORDS)


@pytest.mark.parametrize(
    "text",
    [
        "mix the audio",
        "no music mix",
        "mix in some music",
        "mix in the background audio",
        "a mix of music and sound",
        "run cutpoint mix on this",
        "add a fade transition between clips",
        "make it 4 minutes of skiing",
        "remove the last clip",
        "an alternative cut",
        "don't break up the skiing",
        "do not mix in any funny clips",
        "keep the skiing together, no alternating",
        "without breaking up the runs",
        "never intersperse dialogue",
        "add a fade in between the clips",
        "add a crossfade between the runs",
        "put a transition between each run",
        "use an alternate take of clip 3",
        "the occasional stutter in clip 4",
        "break up the long clip into two",
        "between the trailer and the intro",
        "between the surface shots",
        "Make it ~4 minutes; don't break up the skiing with dialogue, keep the runs together",
    ],
)
def test_audio_mix_and_plain_edits_are_not_a_mix_request(text: str) -> None:
    assert not pacing.asks_for_mix(text)


@pytest.mark.parametrize(
    ("text", "named"),
    [
        (TURN, True),
        ("some banter", True),
        ("add talking moments", True),
        ("some talk between the runs", True),
        ("not too long, with dialogue to break up the runs", True),
        ("no music. add some dialogue", True),
        ("no dialogue, break up the skiing", False),
        ("the commentary", True),
        ("a conversation", True),
        ("chat clips", True),
        ("a funny moment", False),
        ("the talker is wrong", False),
        ("break up the skiing with views, no dialogue", False),
        ("never add banter", False),
        ("without any commentary", False),
        ("can we chat about the plan", False),
        ("let's talk about the order", False),
    ],
)
def test_dialogue_is_named_by_its_words(text: str, named: bool) -> None:
    assert pacing.names_dialogue(text) is named


# -- the kind of a row --------------------------------------------------------------------------


def kind(label: str, cand: dict | None = None, cues: dict | None = None, spoken: bool = False):
    return pacing.moment_kind(label, cand, cues or {}, lambda: spoken)


@pytest.mark.parametrize(
    ("label", "cand", "cues", "spoken", "want"),
    [
        ("fast", None, {}, False, "action"),
        ("jump", None, {}, True, "action"),  # speech over a skiing clip stays action
        ("crash", None, {}, False, "action"),
        ("carve", None, {}, False, "action"),
        ("casual cruising run", {"kind": "cruise"}, {}, False, "action"),
        ("group / friends shot", {"kind": "crew"}, {}, True, "action"),
        ("view", None, {}, False, "scenic"),
        ("stunning-mountain-views", None, {}, False, "scenic"),
        ("scenic view / beauty shot", {"kind": "view", "pool": "visual"}, {}, False, "scenic"),
        ("beauty", None, {"speed": 9}, False, "scenic"),
        ("laugh", None, {}, True, "funny"),
        ("big laugh", None, {}, False, "funny"),
        ("talk", None, {}, False, "dialogue"),
        ("ski-moment", None, {}, True, "dialogue"),  # generic label, spoken
        ("loud-moment", None, {}, False, "action"),  # generic label, nothing else
        ("ski-moment", None, {"scenic_vista": 8}, True, "scenic"),  # the strongest cue
        ("ski-moment", None, {"funny": 7}, False, "funny"),
        ("ski-moment", None, {"speed": 6}, True, "action"),
        ("ski-moment", None, {"standing_talking": 6}, False, "dialogue"),
        ("ski-moment", None, {"scenic_vista": 3}, False, "action"),  # too weak to name it
        ("", {"vision_label": "laughing friends"}, {}, False, "funny"),
    ],
)
def test_the_kind_of_a_row_comes_from_its_label_kind_cues_and_speech(
    label, cand, cues, spoken, want
) -> None:
    assert kind(label, cand, cues, spoken) == want


@pytest.mark.parametrize(
    ("label", "cand", "want"),
    [
        ("crash", {"kind": "crash", "must_include": True}, "crash"),  # a crash moment (#848)
        ("crash reaction", {"kind": "crash"}, "crash"),
        ("wipeout", {"kind": "wipeout"}, "crash"),  # an action-cam crash
        ("ski-moment", {"crash": {"type": "crash"}, "crash_forced": True}, "crash"),
        ("ski-moment", {"crash": {"type": "crash"}, "must_include": True}, "crash"),
        ("crash", None, "action"),  # the word alone: unchanged
        ("fast", {"crash": {"type": "crash"}, "crash_forced": False}, "action"),  # only marked
        ("laugh", {"crash": {"type": "crash"}, "crash_forced": False}, "funny"),
        ("fast", {"kind": "cruise"}, "action"),
    ],
)
def test_a_crash_moment_is_the_crash_kind(label, cand, want) -> None:
    assert kind(label, cand) == want


def test_crash_moments_are_footage_of_the_activity_for_the_spread() -> None:
    assert pacing.arrange(["crash", "action", "crash", "funny"]) == [0, 1, 3, 2]
    assert pacing.longest_run(["crash", "action", "crash", "funny", "action"]) == 3


def test_a_turn_names_crash_moments_and_spreads_the_breakers_between_them() -> None:
    cands = [moment(i, FAST) for i in range(6)] + [
        moment(6, "crash", kind="crash", must_include=True),
        moment(7, "crash reaction", kind="crash", must_include=True),
        moment(8, LAUGH),
        moment(9, LAUGH),
    ]
    proj = session.Project(cands, [], durations(), "", "key0")
    turn = play("Break up the skiing and the crashes with the funny moments", proj)
    kinds = kinds_of(turn, proj)
    assert kinds.count("crash") == 2 and kinds.count("funny") == 2
    assert longest_run([k if k != "crash" else "action" for k in kinds]) <= 3
    assert "funny" in proposal(turn)["summary"]


def test_find_moments_finds_crash_moments_by_their_kind() -> None:
    proj = session.Project(
        [moment(0, FAST), moment(1, "vlm", kind="crash", must_include=True)],
        [],
        durations(),
        "",
        "key0",
    )
    fake = FakeProvider([call("find_moments", query="crash")] * 4)
    turn = session.Turn(fake, proj, "find the crash")
    turn.start()
    assert turn.done.wait(10)
    found = next(m for m in fake.requests[1]["messages"] if m.get("role") == "tool")["content"]
    assert '"kind":"crash"' in found.replace(" ", "")


# -- the order ----------------------------------------------------------------------------------


def test_arrange_spreads_breakers_between_action_runs_and_keeps_each_kinds_order() -> None:
    kinds = ["a"] * 8 + ["f"] * 2 + ["d"] * 2
    names = {"a": "action", "f": "funny", "d": "dialogue"}
    order = pacing.arrange([names[k] for k in kinds])
    assert sorted(order) == list(range(12))
    assert [i for i in order if kinds[i] == "f"] == [8, 9]
    assert [i for i in order if kinds[i] == "a"] == list(range(8))
    shown = "".join(kinds[i] for i in order)
    assert "aaaaa" not in shown and shown[0] == "a"


def test_arrange_keeps_a_scenic_opener_and_changes_nothing_without_breakers_or_action() -> None:
    assert pacing.arrange(["scenic", "action", "action", "funny"])[0] == 0
    assert pacing.arrange(["action"] * 3) == [0, 1, 2]
    assert pacing.arrange(["funny", "scenic"]) == [0, 1]
    assert pacing.arrange([]) == []


def test_move_ops_turn_one_order_into_another_with_the_fewest_moves() -> None:
    cur = list("abcdef")
    for target in (list("abcdef"), list("fabcde"), list("badcfe"), list("fedcba")):
        moves = pacing.move_ops(cur, target)
        rows = list(cur)
        for m in moves:
            assert m["op"] == "move" and m["auto"] is True
            rows.insert(m["to"] - 1, rows.pop(rows.index(m["id"])))
        assert rows == target
    assert pacing.move_ops(cur, list("fabcde")) == [
        {"op": "move", "id": "f", "to": 1, "auto": True}
    ]


# -- the turn -----------------------------------------------------------------------------------


def test_the_real_turn_comes_out_with_no_long_action_streak() -> None:
    proj = real_project()
    turn = play(TURN, proj)
    event = proposal(turn)
    kinds = kinds_of(turn, proj)
    assert kinds[0] == "scenic"  # the opener stays
    assert longest_run(kinds) <= 4  # was 10+ in a row
    assert kinds.count("funny") == 4 and kinds.count("scenic") == 4
    assert "Spread 4 funny and 3 scenic clips between the riding runs." in event["summary"]
    assert event["pace"]["moved"] and all(
        op.get("auto") for op in event["ops"] if op["op"] == "move"
    )
    again = ops.apply_ops([], event["ops"], proj.candidates, durations())  # the normal apply path
    assert again.error is None and again.after == event["after"]
    assert "auto-spread" in turn.record()["reply"]


def test_each_kind_keeps_its_relative_order() -> None:
    proj = real_project()
    turn = play(TURN, proj)
    ids = [r["id"] for r in proposal(turn)["after"]]
    for label in (LAUGH, FAST, VIEW):
        mine = [f"k{i}" for i, name in enumerate(REAL) if name == label]
        assert [i for i in ids if i in mine] == mine


def test_without_mix_wording_the_order_is_the_models() -> None:
    proj = real_project()
    turn = play("Create a ~4 minute sequence of skiing and funny moments", proj)
    event = proposal(turn)
    assert [r["id"] for r in event["after"]] == [f"k{i}" for i in range(len(REAL))]
    assert "pace" not in event and "Spread" not in event["summary"]
    assert not any(op["op"] == "move" for op in event["ops"])


def test_a_proposal_that_is_spread_already_is_left_alone() -> None:
    cands = [moment(i, label) for i, label in enumerate([FAST, FAST, LAUGH, FAST, FAST, LAUGH])]
    proj = session.Project(cands, [], durations(), "", "key0")
    turn = play("break up the skiing with funny moments", proj, add_all(cands))
    event = proposal(turn)
    assert "pace" not in event and [r["id"] for r in event["after"]] == [c["id"] for c in cands]


def test_a_sequence_that_was_there_is_spread_with_move_ops_that_mark_the_rows() -> None:
    cands = [moment(i, label) for i, label in enumerate([FAST] * 6 + [LAUGH, LAUGH])]
    seg = [{k: c[k] for k in ("id", "source", "start", "end", "label")} for c in cands]
    proj = session.Project(cands, seg, durations(), "", "key0")
    tidy = call(
        "propose_edit",
        summary="Trim the last clip",
        ops=[{"op": "trim", "id": short("k7"), "end_delta": -1}],
    )
    turn = play("mix in the laughs between the riding runs", proj, tidy)
    event = proposal(turn)
    kinds = [("funny" if r["label"] == LAUGH else "action") for r in event["after"]]
    assert longest_run(kinds) <= 3
    moved = {r["id"] for r in event["diff"] if "moved" in r["changes"]}
    assert moved and {r["id"] for r in event["diff"] if r.get("auto") == "spread"} == moved
    assert event["base_key"] == "key0"
    again = ops.apply_ops(seg, event["ops"], cands, durations())
    assert again.error is None and again.after == event["after"]


# -- dialogue ------------------------------------------------------------------------------------

SAID = "so we go left here and then down the long traverse to the lift"
TALK = ("k23", "k24", "k25", "k26")
MIXED = "occasional funny moments and dialogue to break up the skiing"


def talk_project() -> session.Project:
    talkers = [moment(23 + i, "ski-moment", length=6.0, said=SAID, score=40 + i) for i in range(4)]
    talkers[0]["transcript"] += " avalanche"
    return real_project(extra=talkers)


def without_talk(proj: session.Project) -> list[dict[str, Any]]:
    return [c for c in proj.candidates if c["id"] not in TALK]


def test_dialogue_is_added_when_it_is_named_and_missing() -> None:
    proj = talk_project()
    turn = play(MIXED, proj, add_all(without_talk(proj)))
    event = proposal(turn)
    got = [r["id"] for r in event["after"] if r["id"] in TALK]
    assert got == ["k26", "k25", "k24"]  # the best-scoring first, at most 3
    assert event["pace"]["added"] == got
    assert "Added 3 dialogue moments you asked for" in event["summary"]
    assert "3 dialogue" in event["summary"]  # and spread with the rest
    assert f"Now {event['length']:.0f} s." in event["summary"]  # the length with the dialogue
    adds = [op for op in event["ops"] if op["op"] == "add" and op["id"] in got]
    assert len(adds) == 3 and all(op.get("auto") for op in adds)
    marks = {r["id"]: r.get("auto") for r in event["diff"]}
    assert all(marks[i] == "spread" for i in got)


def test_this_turns_searches_come_before_the_best_unused_dialogue() -> None:
    proj = talk_project()
    find = call("find_moments", query="avalanche")
    turn = play(MIXED, proj, find, add_all(without_talk(proj)))
    assert proposal(turn)["pace"]["added"] == ["k23", "k26", "k25"]


def test_dialogue_already_there_is_not_added() -> None:
    proj = talk_project()
    base = [c for c in proj.candidates if c["id"] in {"k0", "k1", "k2", "k6", "k7", "k23"}]
    event = proposal(play(MIXED, proj, add_all(base)))
    assert [r["id"] for r in event["after"]].count("k23") == 1
    assert not event["pace"]["added"] and not event["pace"]["replaced"]


def test_dialogue_is_not_added_when_it_is_not_named() -> None:
    proj = talk_project()
    event = proposal(play("the occasional funny moment", proj, add_all(without_talk(proj))))
    assert not any(r["id"] in TALK for r in event["after"])
    assert event["pace"]["moved"] and not event["pace"]["added"]


def test_dialogue_that_does_not_fit_is_not_added_and_no_pick_is_removed() -> None:
    action = [moment(i, FAST, length=8.0, score=10 + i) for i in range(12)]
    laugh = [moment(12 + i, LAUGH, length=8.0) for i in range(2)]
    talk = [moment(14, "ski-moment", length=8.0, said=SAID, score=60)]
    cands = action + laugh + talk
    proj = session.Project(cands, [], durations(), "", "key0")
    full = add_all(action + laugh)  # 14 x 8 s = 112 s: 100 s asked, 115 s is the cap
    turn = play("make it 100 s: break up the skiing with laughs and dialogue", proj, full)
    event = proposal(turn)
    assert event["length"] <= 100 * 1.15
    assert not event["pace"]["added"] and not event["pace"]["replaced"]
    assert {r["id"] for r in event["after"]} == {c["id"] for c in action + laugh}
    assert not any(op["op"] == "remove" for op in event["ops"])


def swap_setup():
    """12 skiing clips of 8 s (k0 the lowest score) and one dialogue moment, as ``pace`` sees
    them: ``evaluate`` is the real apply path."""
    action = [moment(i, FAST, length=8.0, score=10 + i) for i in range(12)]
    talk = moment(14, "ski-moment", length=8.0, said=SAID, score=60)
    cands = [*action, talk]
    seed = [{"op": "add", "id": c["id"]} for c in action]

    def evaluate(op_list):
        out = ops.apply_ops([], op_list, cands, durations())
        if out.error:
            return None
        length = sum(r["end"] - r["start"] for r in out.after)
        return {"ops": op_list, "after": out.after, "diff": out.diff, "length": length}

    def kind_of(item):
        row = next(c for c in cands if c["id"] == item["id"])
        return pacing.moment_kind(row["label"], row, {}, lambda: bool(row.get("transcript")))

    return cands, evaluate(seed), evaluate, kind_of


def run_pace(removable, upper=100.0):
    cands, start, evaluate, kind_of = swap_setup()
    return pacing.pace(
        start,
        kind_of=kind_of,
        evaluate=evaluate,
        candidates={c["id"]: c for c in cands},
        searched=[],
        dialogue=True,
        upper=upper,
        removable=removable,
    )


def test_dialogue_swaps_only_a_clip_the_auto_fill_added_never_a_pick() -> None:
    got = run_pace({"k5", "k7"})  # the fill added k5 and k7; k0 is the lowest score overall
    assert got is not None and got.added == ["k14"] and got.replaced == ["k5"]
    ids = [r["id"] for r in got.checked["after"]]
    assert "k0" in ids and "k5" not in ids and "k7" in ids and got.checked["length"] <= 100.0
    assert all(op.get("auto") for op in got.checked["ops"][12:])


def test_dialogue_with_no_auto_filled_clip_to_swap_is_not_added() -> None:
    assert run_pace(()) is None  # nothing fits and nothing may go
    got = run_pace((), upper=120.0)  # with room it is simply added
    assert got is not None and got.added == ["k14"] and got.replaced == []


def test_the_fifteen_percent_cap_is_never_exceeded() -> None:
    action = [moment(i, FAST, length=8.0, score=10 + i) for i in range(12)]
    talk = [moment(12 + i, "ski-moment", length=40.0, said=SAID) for i in range(3)]
    cands = action + talk  # nothing of 40 s is similar to an 8 s action clip
    proj = session.Project(cands, [], durations(), "", "key0")
    turn = play("make it 90 s with dialogue to break up the skiing", proj, add_all(action))
    event = proposal(turn)
    assert event["length"] <= 90 * 1.15
    assert not event.get("pace", {}).get("added")


# -- positions ------------------------------------------------------------------------------------


@pytest.mark.parametrize("given", [3.0, "3", 3])
def test_a_whole_number_position_is_an_int_whatever_it_is_written_as(given: Any) -> None:
    seq = [moment(i, FAST) for i in range(4)]
    cands = [*seq, moment(9, LAUGH)]
    move = ops.apply_ops(seq, [{"op": "move", "id": "k0", "to": given}], cands, durations())
    assert move.error is None and [r["id"] for r in move.after] == ["k1", "k2", "k0", "k3"]
    add = ops.apply_ops(seq, [{"op": "add", "id": "k9", "at": given}], cands, durations())
    assert add.error is None and [r["id"] for r in add.after][2] == "k9"


@pytest.mark.parametrize("given", [True, 2.5, "two", "2.0", "-1", "", [2], float("nan")])
def test_other_positions_are_still_rejected(given: Any) -> None:
    seq = [moment(i, FAST) for i in range(4)]
    cands = [*seq, moment(9, LAUGH)]
    assert ops.apply_ops(seq, [{"op": "move", "id": "k0", "to": given}], cands, durations()).error
    bad_add = ops.apply_ops(seq, [{"op": "add", "id": "k9", "at": given}], cands, durations())
    assert bad_add.error


def test_a_negated_mix_request_changes_nothing_and_adds_no_dialogue() -> None:
    proj = talk_project()
    text = "Make it ~4 minutes; don't break up the skiing with dialogue, keep the runs together"
    event = proposal(play(text, proj, add_all(without_talk(proj))))
    assert "pace" not in event and not any(r["id"] in TALK for r in event["after"])
    assert not any(op["op"] == "move" for op in event["ops"])


# -- too many ops for moves: one order op --------------------------------------------------------


def test_past_the_op_cap_the_spread_is_one_order_op() -> None:
    kinds = [FAST] * 40 + [LAUGH] * 40  # 80 adds + 40 moves would pass the 100-op cap
    cands = [moment(i, label, length=3.0) for i, label in enumerate(kinds)]
    proj = session.Project(cands, [], durations(), "", "key0")
    turn = play("mix in the laughs between the riding runs", proj, add_all(cands))
    event = proposal(turn)
    last = event["ops"][-1]
    assert last["op"] == "order" and last["auto"] is True and len(event["ops"]) == 81
    assert not any(op["op"] == "move" for op in event["ops"])
    after = [FAST if r["label"] == FAST else LAUGH for r in event["after"]]
    assert longest_run([("action" if k == FAST else "funny") for k in after]) <= 2 < 40
    again = ops.apply_ops([], event["ops"], cands, durations())
    assert again.error is None and again.after == event["after"]
    moved = event["pace"]["moved"]
    before = [c["id"] for c in cands]
    only = {m["id"] for m in pacing.move_ops(before, [r["id"] for r in event["after"]])}
    assert set(moved) == only and len(moved) == len(set(moved)) < 80  # not rows that only shifted


def test_the_pace_note_survives_a_long_summary_and_fill_note() -> None:
    proj = real_project()
    turn = session.Turn(FakeProvider([]), proj, TURN)
    main = turn._checked_real([{"op": "add", "id": c["id"]} for c in proj.candidates])[0]
    main["auto"] = {"target": 240.0, "lengthened": [], "added": [], "reached": True}
    summary, _ = turn._paced("Long summary. " * 40, main)
    assert len(summary) <= session.MAX_SUMMARY
    assert summary.endswith("clips between the riding runs.")
    assert "Spread 4 funny and 3 scenic" in summary
