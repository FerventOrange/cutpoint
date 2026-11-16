"""Word splices in the AI Chat: ``find_words`` (where words are said, from the cached transcript
words), the ``add_range`` op (an exact source range) and the exact rows it makes, from the chat
through validate_segments to the clips manifest."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from conftest import media_info, needs_ffmpeg, write
from cutpoint import clips, planning, sources, ui
from cutpoint.assistant import context, ops, session, words
from cutpoint.assistant.providers import Capabilities, Reply
from cutpoint.ffmpeg_utils import probe
from test_assistant import CANDS, SEQ, A, B, FakeProvider, call, durations, run


def said(*items: tuple[str, float, float]) -> list[dict[str, Any]]:
    return [{"word": w, "start": a, "end": b} for w, a, b in items]


# What Whisper cached: words with their own spaces and punctuation, as analysis keeps them.
WORDS = {
    A: said(
        (" Honestly,", 1.0, 1.4),
        (" I", 1.5, 1.6),
        (" am", 1.65, 1.8),
        (" done.", 1.85, 2.2),
        (" A", 5.0, 5.12),
        (" shot", 5.15, 5.5),
        (" -", 5.5, 5.5),  # punctuation alone: skipped when matching
        (" that", 9.0, 9.2),
        (" was", 9.25, 9.4),
        (" a", 9.45, 9.5),
        (" disaster!", 9.55, 10.2),
    ),
    B: said((" This", 2.0, 2.2), (" is", 2.25, 2.4), (" SHIT.", 2.45, 2.8), (" I'm", 4.0, 4.2)),
}


def found(query: str, limit: int = words.DEFAULT_LIMIT) -> words.Found:
    return words.find([A, B], WORDS.get, query, limit)


def spans(got: words.Found) -> list[tuple[str, float, float]]:
    return [(h["source"], h["start"], h["end"]) for h in got.hits]


# -- find_words ------------------------------------------------------------------------------


def test_a_single_a_is_found_and_never_inside_another_word() -> None:
    got = found("a")
    assert spans(got) == [(A, 5.0, 5.12), (A, 9.45, 9.5)]  # not "am", "I'm" or "was"
    assert got.hits[0]["said"] == "Honestly, I am done. [A] shot - that was"


def test_a_phrase_matches_its_words_in_order_any_case_without_punctuation() -> None:
    assert spans(found("I am")) == [(A, 1.5, 1.8)]
    assert spans(found("am I")) == []  # out of order
    assert spans(found("shit")) == [(B, 2.45, 2.8)]  # "SHIT." in the transcript
    assert spans(found("Shit!")) == [(B, 2.45, 2.8)]
    assert spans(found("I'm")) == [(B, 4.0, 4.2)]
    assert spans(found("shot that")) == [(A, 5.15, 9.2)]  # the dash between is no word
    assert spans(found("  ")) == [] and found("?").total == 0  # nothing to look for


def test_the_limit_caps_the_hits_and_the_total_counts_all() -> None:
    got = found("a", limit=1)
    assert len(got.hits) == 1 and got.total == 2 and (got.sources, got.with_words) == (2, 2)


def test_sources_without_words_or_with_broken_words_are_skipped() -> None:
    def words_of(src: str) -> list[dict[str, Any]] | None:
        if src == B:
            raise OSError("cache unreadable")
        return [{"word": "a", "start": "x", "end": 1}, *said(("a", 3.0, 3.1)), "junk"]  # type: ignore[list-item]

    got = words.find([A, B], words_of, "a")
    assert spans(got) == [(A, 3.0, 3.1)] and (got.sources, got.with_words) == (2, 1)


def test_unspaced_scripts_match_the_letters_run_together() -> None:
    zh = said(("我", 1.0, 1.2), ("们", 1.2, 1.4), ("走", 1.5, 1.7), ("吧", 1.7, 1.9))
    got = words.find([A], lambda _s: zh, "我们走")
    assert spans(got) == [(A, 1.0, 1.7)] and got.hits[0]["said"] == "[我 们 走] 吧"


# -- add_range -------------------------------------------------------------------------------


def test_add_range_appends_an_exact_row_with_a_stable_id() -> None:
    op = {"op": "add_range", "source": B, "start": 2.45, "end": 2.8, "label": "shit"}
    out = ops.apply_ops([dict(s) for s in SEQ], [op], CANDS, durations())
    assert out.error is None, out.error
    row = out.after[-1]
    assert {k: row[k] for k in ("source", "start", "end", "label", "exact")} == {
        "source": B,
        "start": 2.45,
        "end": 2.8,
        "label": "shit",
        "exact": True,
    }
    again = ops.apply_ops([dict(s) for s in SEQ], [op], CANDS, durations())
    assert again.after[-1]["id"] == row["id"] and row["id"].startswith("r")  # a rebase agrees
    assert {r["id"]: r["changes"] for r in out.diff if r["changes"]} == {row["id"]: ["added"]}


def test_add_range_may_repeat_a_range_and_put_it_anywhere() -> None:
    word = {"op": "add_range", "source": A, "start": 5.0, "end": 5.12}
    out = ops.apply_ops([], [word, {**word, "at": 1}, word], CANDS, durations())
    assert out.error is None, out.error
    ids = [r["id"] for r in out.after]
    assert len(set(ids)) == 3 and ids[1] == ids[0].split("-")[0]  # the first one, moved down
    assert all(r["label"] == "words" and r["exact"] for r in out.after)


@pytest.mark.parametrize(
    ("op", "want"),
    [
        ({"source": "/v/other.mp4", "start": 1, "end": 2}, "source is not a file"),
        ({"source": A, "start": 1}, "needs start and end"),
        ({"source": A, "start": -1, "end": 2}, "needs 0 <= start"),
        ({"source": A, "start": 2.0, "end": 1.5}, "needs 0 <= start <= end"),
        ({"source": B, "start": 29.5, "end": 30.5}, "past the source's 30s"),
        ({"source": A, "start": 1, "end": 2, "at": -1}, "at must be a position"),
        ({"source": A, "start": 1, "end": 2, "label": 5}, "label must be text"),
    ],
)
def test_add_range_is_checked(op: dict[str, Any], want: str) -> None:
    out = ops.apply_ops([], [{"op": "add_range", **op}], CANDS, durations())
    assert out.after is None and want in (out.error or ""), out.error


@pytest.mark.parametrize(
    ("source", "start", "end", "want"),
    [
        (A, 5.0, 5.05, (4.975, 5.075)),  # Whisper's 0.05 s "a": widened around its middle
        (A, 7.0, 7.0, (6.95, 7.05)),  # a word timed with no length
        (A, 0.0, 0.04, (0.0, 0.1)),  # not before the source's start
        (B, 29.98, 30.0, (29.9, 30.0)),  # nor past its end
        (A, 93.79, 93.857, (93.773, 93.873)),  # 3 decimals: still 0.1 s once rounded
    ],
)
def test_add_range_widens_a_word_shorter_than_a_clip_can_be(
    source: str, start: float, end: float, want: tuple[float, float]
) -> None:
    op = {"op": "add_range", "source": source, "start": start, "end": end}
    out = ops.apply_ops([], [op], CANDS, durations())
    assert out.error is None, out.error
    row = out.after[0]
    assert (row["start"], row["end"]) == pytest.approx(want) and row["exact"] is True
    entries = [{"path": A, "duration": 100.0}, {"path": B, "duration": 30.0}]
    assert ui.validate_segments(out.after, entries, None) == out.after  # type: ignore[arg-type]


def test_a_trim_of_an_exact_row_is_never_snapped_to_a_phrase_end() -> None:
    word = {"op": "add_range", "source": A, "start": 5.0, "end": 5.12}
    first = ops.apply_ops([], [word], CANDS, durations()).after
    trim = {"op": "trim", "id": first[0]["id"], "end": 5.3}
    out = ops.apply_ops(first, [trim], CANDS, durations(), snap=lambda _s, _t: 9.0)
    assert out.error is None and out.after[0]["end"] == 5.3 and out.after[0]["exact"] is True


def test_exact_rows_round_trip_through_validate_segments() -> None:
    word = {"op": "add_range", "source": A, "start": 5.0, "end": 5.12, "label": "a"}
    after = ops.apply_ops([dict(SEQ[0])], [word], CANDS, durations()).after
    entries = [{"path": A, "duration": 100.0}, {"path": B, "duration": 30.0}]
    assert ui.validate_segments(after, entries, None) == after  # type: ignore[arg-type]
    assert "exact" not in after[0] and after[1]["exact"] is True
    off = ui.validate_segments([{**after[1], "exact": False}], entries, None)  # type: ignore[arg-type]
    assert "exact" not in off[0]
    with pytest.raises(Exception, match="exact must be true or false"):
        ui.validate_segments([{**after[1], "exact": 1}], entries, None)  # type: ignore[arg-type]


def test_a_trim_with_exact_makes_the_row_exact_and_unsnapped() -> None:
    trim = {"op": "trim", "id": "c3", "start": 4.0, "end": 9.5, "exact": True}
    out = ops.apply_ops([dict(SEQ[1])], [trim], CANDS, durations(), snap=lambda _s, _t: 12.0)
    assert out.error is None, out.error
    assert (out.after[0]["start"], out.after[0]["end"], out.after[0]["exact"]) == (4.0, 9.5, True)
    plain = ops.apply_ops([dict(SEQ[1])], [{**trim, "exact": False}], CANDS, durations()).after
    assert "exact" not in plain[0]
    bad = ops.apply_ops([dict(SEQ[1])], [{**trim, "exact": "yes"}], CANDS, durations())
    assert bad.error is not None and "exact must be true or false" in bad.error


@pytest.mark.parametrize(
    ("op", "want"),
    [
        (
            {"op": "trim", "id": "c1", "end": 15.0},
            "c1 is not in the sequence; add it first (add, then trim, in one proposal), "
            "or add_range a find_words hit",
        ),
        ({"op": "move", "id": "c1", "to": 1}, "c1 is not in the sequence; add it with add (at"),
        ({"op": "remove", "id": "c1"}, "c1 is not in the sequence; there is nothing to remove"),
        (
            {"op": "add", "id": "c1", "at": "end"},
            "at must be a position from 1 to 2, or left out to add it last",
        ),
        (
            {"op": "add_range", "source": A, "start": 1, "end": 2, "at": -1},
            "at must be a position from 1 to 2, or left out to add it last",
        ),
    ],
)
def test_a_wrong_op_is_told_which_op_fits(op: dict[str, Any], want: str) -> None:
    out = ops.apply_ops([dict(SEQ[1])], [op], CANDS, durations())
    assert out.after is None and want in (out.error or ""), out.error


def test_the_manifest_records_an_exact_clip_and_its_short_fade(home, monkeypatch) -> None:
    """Through ``clips --segments``: the word is cut at its own times with a 10 ms fade, the
    other segment keeps its padding and the export's fade."""
    project = home / "projects" / "p"
    src = write(project / "src.mp4")
    seg = write(
        project / "seg.json",
        json.dumps(
            [
                {"source": str(src), "start": 2.0, "end": 4.0, "label": "a"},
                {"source": str(src), "start": 5.0, "end": 5.12, "label": "word", "exact": True},
            ]
        ),
    )
    monkeypatch.setattr(clips, "probe", media_info)
    fades: list[float] = []

    def export(src, info, start, end, out, *args, **_kw):
        fades.append(args[3])  # fmt, vertical, crop, audio_fade
        out.write_bytes(b"n")

    monkeypatch.setattr(clips, "export_clip_checked", export)
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)
    folder = clips.export_clips(project, segments=seg, extend=clips.Padding(1.0, 1.0))
    entries = json.loads((Path(folder) / "manifest.json").read_text(encoding="utf-8"))["clips"]
    plain, word = entries
    assert (word["start"], word["end"], word["exact"], word["audio_fade"]) == (
        5.0,
        5.12,
        True,
        planning.EXACT_AUDIO_FADE_S,
    )
    assert "exact" not in plain and plain["start"] == 1.0 and plain["audio_fade"] == 0.25
    assert sorted(fades) == [planning.EXACT_AUDIO_FADE_S, 0.25]


def data_rows(result: str) -> list[dict[str, Any]]:
    """The JSON rows between a tool result's data markers."""
    body = result.split("<<<DATA words>>>\n", 1)[1].split("\n<<<END DATA>>>", 1)[0]
    return [json.loads(line) for line in body.splitlines()]


# -- the chat --------------------------------------------------------------------------------


def chat_project() -> session.Project:
    return session.Project(
        list(CANDS), [dict(s) for s in SEQ], durations(), "", "key0", words=WORDS.get
    )


def test_find_words_is_offered_and_answers_with_files_times_and_context() -> None:
    fake = FakeProvider([call("find_words", query="a"), Reply("It is said twice in a.mp4.")])
    turn = run(fake, "where does she say a", proj=chat_project())
    assert turn.status == "done", turn.error
    assert "find_words" in [t["function"]["name"] for t in fake.requests[0]["tools"]]
    assert words.RULES.strip() in fake.requests[0]["messages"][0]["content"]
    result = fake.requests[1]["messages"][-1]["content"]
    assert result.startswith("<<<DATA words>>>")
    hits = data_rows(result)
    assert "add_range op (its file, start and end); no moment id needed" in result
    assert [(h["file"], h["start"], h["end"]) for h in hits] == [
        ("a.mp4", 5.0, 5.12),
        ("a.mp4", 9.45, 9.5),
    ]
    event = turn.view()["events"][0]
    assert event == {**event, "kind": "tool", "name": "find_words", "query": "a", "found": 2}


@pytest.mark.parametrize(
    ("args", "want"),
    [
        ({"query": "banana"}, "nobody says 'banana' in the transcripts"),
        (
            {"query": "banana a"},
            "nobody says 'banana a' in one place. Its words alone: 'banana' 0 times, 'a' 2 times.",
        ),
        ({"query": ""}, "error: query must be"),
        ({"query": "a", "limit": 1}, "(1 of 2 shown; limit up to 30)"),
    ],
)
def test_find_words_says_what_it_did_not_find(args: dict[str, Any], want: str) -> None:
    fake = FakeProvider([call("find_words", **args), Reply("ok")])
    run(fake, "x", proj=chat_project())
    assert want in fake.requests[1]["messages"][-1]["content"]


def test_find_words_without_transcript_words_says_so() -> None:
    proj = chat_project()
    proj.words = None
    fake = FakeProvider([call("find_words", query="a"), Reply("ok")])
    run(fake, "x", proj=proj)
    result = fake.requests[1]["messages"][-1]["content"]
    assert "no source of this project has transcript words" in result


def test_add_range_names_a_file_the_model_knows() -> None:
    bad = [{"op": "add_range", "file": "c.mp4", "start": 1, "end": 2}]
    good = [{"op": "add_range", "file": "b.mp4", "start": 2.45, "end": 2.8}]
    fake = FakeProvider(
        [call("propose_edit", summary="x", ops=bad), call("propose_edit", summary="y", ops=good)]
    )
    turn = run(fake, "add it", proj=chat_project())
    rejected = fake.requests[1]["messages"][-1]["content"]
    assert "add_range needs file, one of the project's files (a.mp4, b.mp4)" in rejected
    assert turn.view()["events"][-1]["after"][-1]["source"] == B


def test_owner_scenario_make_her_say_a_sentence() -> None:
    """The owner's "Stitch clips together that make her say 'I am shit'": the words are found where they
    are said and joined in order as exact rows; nothing is refused."""
    stitched = [
        {"op": "add_range", "file": "a.mp4", "start": 1.5, "end": 1.8, "label": "I am"},
        {"op": "add_range", "file": "b.mp4", "start": 2.45, "end": 2.8, "label": "shit"},
    ]
    fake = FakeProvider(
        [
            call("find_words", query="I am shit"),  # never said in one place
            call("find_words", query="I am"),
            call("find_words", query="shit"),
            call("propose_edit", summary="She says 'I am shit'", ops=stitched),
        ]
    )
    text = "Stitch clips together that make her say 'I am shit'"
    turn = run(fake, text, proj=chat_project())
    assert turn.status == "done", turn.error
    apart = fake.requests[1]["messages"][-1]["content"]
    assert "nobody says 'I am shit' in one place" in apart
    assert "Its words are said apart: 'I' 1 times, 'am' 1 times, 'shit' 1 times" in apart
    # #974: the splice comes ready, one add_range per word in order
    assert (
        'propose_edit with these ops, in order: [{"op": "add_range", "file": "a.mp4", '
        '"start": 1.5, "end": 1.6}, {"op": "add_range", "file": "a.mp4", "start": 1.65, '
        '"end": 1.8}, {"op": "add_range", "file": "b.mp4", "start": 2.45, "end": 2.8}]'
    ) in apart
    assert "no moment id needed" in fake.requests[2]["messages"][-1]["content"]
    assert '"start": 1.5, "end": 1.8' in fake.requests[2]["messages"][-1]["content"]
    prop = turn.view()["events"][-1]
    assert prop["kind"] == "proposal"
    tail = [(r["source"], r["start"], r["end"], r.get("exact")) for r in prop["after"][-2:]]
    assert tail == [(A, 1.5, 1.8, True), (B, 2.45, 2.8, True)]
    assert [e["found"] for e in turn.view()["events"] if e.get("name") == "find_words"] == [0, 1, 1]


@pytest.mark.parametrize(
    ("trim", "exact"),
    [
        ({"start": 4.0}, True),  # the start of "I'm"
        ({"end": 2.81}, True),  # the end of "SHIT." (2.8), within 0.02 s
        ({"start": 4.2}, True),  # right after "I'm": the hit's end
        ({"end": 4.0}, True),  # right before "I'm": the hit's start
        ({"start": 3.8}, False),  # near no word (0.2 s from "I'm")
    ],
)
def test_a_trim_at_a_found_words_edge_makes_the_row_exact(
    tmp_path: Path, trim: dict[str, Any], exact: bool
) -> None:
    """The owner's "cut it at the end of 'shit'": a trim to a find_words hit's edge marks the
    row exact, so the export cuts there; validate_segments and the segments file keep it."""
    proj = chat_project()
    proj.segments = [{"id": "c3", "source": B, "start": 2.0, "end": 9.0, "label": "crash"}]
    fake = FakeProvider(
        [
            call("find_words", query="shit"),
            call("find_words", query="I'm"),
            call(
                "propose_edit", summary="Cut at the word", ops=[{"op": "trim", "id": "m3", **trim}]
            ),
        ]
    )
    turn = run(fake, "cut it at the word", proj=proj)
    prop = turn.view()["events"][-1]
    assert prop["kind"] == "proposal", prop
    row = prop["after"][0]
    assert row.get("exact", False) is exact
    assert (prop["ops"][0].get("exact") is True) is exact  # a rebase on save marks it again
    if not exact:
        return
    entries = [{"path": A, "duration": 100.0}, {"path": B, "duration": 30.0}]
    kept = ui.validate_segments(prop["after"], entries, None)  # type: ignore[arg-type]
    assert kept[0]["exact"] is True
    seg = tmp_path / "segments.json"
    video = write(tmp_path / "b.mp4")  # load_segments wants the source on disk
    seg.write_text(json.dumps([{**kept[0], "source": str(video)}]), encoding="utf-8")
    (planned,) = clips.load_segments(seg)
    assert {k: planned[k] for k in clips.EXACT_KEYS} == clips.EXACT_KEYS


@pytest.mark.parametrize(
    ("segments", "op", "exact"),
    [
        ([], {"op": "add", "id": "m3", "start": 4.0}, True),  # at the start of "I'm"
        ([], {"op": "add", "id": "m3", "start": 4.5}, False),  # near no word
        ([(2.0, 9.0)], {"op": "trim", "id": "m3", "end_delta": -6.2}, True),  # to "SHIT."'s end
    ],
)
def test_an_add_or_a_delta_trim_at_a_found_words_edge_makes_the_row_exact(
    segments: list[tuple[float, float]], op: dict[str, Any], exact: bool
) -> None:
    """#974: an add at a find_words hit edge, and a trim to one given as a delta, are cut at
    the word as a trim in seconds is."""
    proj = chat_project()
    proj.segments = [
        {"id": "c3", "source": B, "start": a, "end": b, "label": "x"} for a, b in segments
    ]
    fake = FakeProvider(
        [
            call("find_words", query="shit"),
            call("find_words", query="I'm"),
            call("propose_edit", summary="Cut at the word", ops=[op]),
        ]
    )
    turn = run(fake, "cut it at the word", proj=proj)
    prop = turn.view()["events"][-1]
    assert prop["kind"] == "proposal", prop
    assert prop["after"][-1].get("exact", False) is exact
    assert (prop["ops"][0].get("exact") is True) is exact  # a rebase on save marks it again


@pytest.mark.parametrize(
    ("start", "want", "exact"),
    [
        (3627.0, 3627.04, True),  # a small model's rounding of "bathtub"'s start
        (3627.5, 3627.5, False),  # 0.3 s from the nearest hit edge: as given
        (10.1, 10.12, True),  # "is" ends at 10.05, "it" starts at 10.12: the nearer
    ],
)
def test_an_edge_within_a_tenth_of_a_hit_edge_moves_onto_it(
    start: float, want: float, exact: bool
) -> None:
    """#974: an add edge within 0.1 s of a find_words hit edge is put on the hit and cut
    exactly there; one further away stays where the model put it."""
    src = "/v/c.mp4"
    spoken = {src: said((" is", 10.0, 10.05), (" it", 10.12, 10.3), (" bathtub", 3627.04, 3627.2))}
    cands = [
        {"id": "b", "source": src, "start": 3620.0, "end": 3632.0, "score": 90, "label": "tub"},
        {"id": "i", "source": src, "start": 5.0, "end": 15.0, "score": 80, "label": "it"},
    ]
    proj = session.Project(
        cands, [], ops.Durations({src: 4000.0}, lambda _p: None), "", "k", words=spoken.get
    )
    moment = "m1" if start > 100 else "m2"
    fake = FakeProvider(
        [
            call("find_words", query="bathtub"),
            call("find_words", query="is"),
            call("find_words", query="it"),
            call(
                "propose_edit", summary="Add it", ops=[{"op": "add", "id": moment, "start": start}]
            ),
        ]
    )
    turn = run(fake, "add it from the word", proj=proj)
    row = turn.view()["events"][-1]["after"][0]
    assert (row["start"], row.get("exact", False)) == (want, exact)


@pytest.mark.parametrize(
    ("search", "ops_", "want"),
    [
        # F1: remove and add again put the row at its own 5.0-9.0; the delta counts from there
        (
            "shit",
            [
                {"op": "remove", "id": "m3"},
                {"op": "add", "id": "m3"},
                {"op": "trim", "id": "m3", "start_delta": 0.5},
            ],
            (5.5, 9.0, False),
        ),
        # a delta after an add counts from the add's own edges, as apply_ops applies it
        (
            "shit",
            [
                {"op": "remove", "id": "m3"},
                {"op": "add", "id": "m3", "start": 3.0},
                {"op": "trim", "id": "m3", "start_delta": -0.52},
            ],
            (2.45, 9.0, True),
        ),
        # ... and after an earlier trim that moved an end earlier
        (
            "shit",
            [
                {"op": "trim", "id": "m3", "end": 3.0},
                {"op": "trim", "id": "m3", "end_delta": -0.21},
            ],
            (2.0, 2.8, True),
        ),
        # an end moved later: the delta counts from where apply_ops puts it (no phrase snap here)
        (
            "shit",
            [
                {"op": "trim", "id": "m3", "end": 9.5},
                {"op": "trim", "id": "m3", "end_delta": -6.72},
            ],
            (2.0, 2.8, True),
        ),
        ("shit", [{"op": "trim", "id": "m3", "end": 2.75, "exact": False}], (2.0, 2.75, False)),
        ("I am", [{"op": "trim", "id": "m3", "start": 1.52}], (1.52, 9.0, False)),  # a.mp4's hit
    ],
    ids=[
        "remove-add-delta",
        "add-then-delta",
        "trim-then-delta",
        "later-end-then-delta",
        "exact-false",
        "other-source",
    ],
)
def test_chained_ops_exact_false_and_other_sources_at_word_edges(
    search: str, ops_: list[dict[str, Any]], want: tuple[float, float, bool]
) -> None:
    """#974 review: a delta snaps from the row's edges as the ops before it leave them;
    exact false and another source's hit are left as given."""
    proj = chat_project()
    proj.segments = [{"id": "c3", "source": B, "start": 2.0, "end": 9.0, "label": "crash"}]
    fake = FakeProvider(
        [call("find_words", query=search), call("propose_edit", summary="Trim it", ops=ops_)]
    )
    turn = run(fake, "trim it", proj=proj)
    prop = turn.view()["events"][-1]
    assert prop["kind"] == "proposal", prop
    row = next(r for r in prop["after"] if r["id"] == "c3")
    assert (row["start"], row["end"], row.get("exact", False)) == want


def test_hits_older_than_the_turns_the_model_sees_are_forgotten() -> None:
    """#974 review: a hit from a turn the model is no longer shown neither snaps nor nudges."""
    first = run(
        FakeProvider([call("find_words", query="shit"), Reply("At 2.45 s in b.mp4.")]),
        "where does he say shit",
        proj=chat_project(),
    )
    filler = [{"user": "hi", "reply": "hello"}] * session.context.budget(8192).turns
    proj = chat_project()
    proj.segments = [{"id": "c3", "source": B, "start": 2.0, "end": 9.0, "label": "crash"}]
    trim = [{"op": "trim", "id": "m3", "end": 2.8}]
    fake = FakeProvider([call("propose_edit", summary="Cut", ops=trim)])
    turn = session.Turn(fake, proj, "cut the crash", [first.record(), *filler])
    turn.start()
    assert turn.done.wait(10)
    assert not turn._found_words
    assert "find_words hits" not in json.dumps(fake.requests[0]["messages"])
    assert "exact" not in turn.view()["events"][-1]["after"][0]


def test_hits_of_an_earlier_message_are_remembered() -> None:
    """#974: "find where he says 'shit'", then "cut the crash at that word": the next message
    sees the hit and its trim there is exact, with no new search."""
    first = run(
        FakeProvider([call("find_words", query="shit"), Reply("At 2.45 s in b.mp4.")]),
        "where does he say shit",
        proj=chat_project(),
    )
    proj = chat_project()
    proj.segments = [{"id": "c3", "source": B, "start": 2.0, "end": 9.0, "label": "crash"}]
    trim = [{"op": "trim", "id": "m3", "end": 2.8}]
    fake = FakeProvider([call("propose_edit", summary="Cut at shit", ops=trim)])
    turn = session.Turn(fake, proj, "cut the crash after that word", [first.record()])
    turn.start()
    assert turn.done.wait(10)
    history = fake.requests[0]["messages"][-2]["content"]
    assert history.endswith("(find_words hits: 'shit': b.mp4 2.45-2.8)")
    assert turn.view()["events"][-1]["after"][0]["exact"] is True


@pytest.mark.parametrize(
    ("query", "nudge"),
    [
        ("shit", "SPLICE_NUDGE"),
        ("I am shit", "SPLICE_NUDGE"),  # no hit, but each word is said somewhere
        ("banana", "NUDGE"),
        ("banana shit", "NUDGE"),  # one word is never said
    ],
)
def test_a_refused_stitch_after_found_words_is_nudged_to_splice(query: str, nudge: str) -> None:
    """A model that found the words and still answers "they are never said together" is told
    to splice them; a turn whose search found nothing gets the plain nudge."""
    stitched = [{"op": "add_range", "file": "b.mp4", "start": 2.45, "end": 2.8}]
    fake = FakeProvider(
        [
            call("find_words", query=query),
            Reply("The words are never said together, so I cannot make her say it."),
            call("propose_edit", summary="She says it", ops=stitched),
        ]
    )
    turn = run(fake, "Stitch clips together that make her say 'I am shit'", proj=chat_project())
    assert turn.status == "done", turn.error
    sent = fake.requests[2]["messages"][-1]
    assert sent["role"] == "user" and sent["content"].startswith(getattr(session, nudge))
    # #974: a phrase said apart gets its splice ops in the nudge too
    assert ("The ops: [" in sent["content"]) is (query == "I am shit")
    assert "add_range one find_words hit of each word" in session.SPLICE_NUDGE
    assert turn.view()["events"][-1]["kind"] == "proposal"


def test_owner_scenario_find_where_she_says_a_in_json_mode() -> None:
    """The owner's "Find me a clip where she says 'A'" with a model without tool calls: find_words is in
    the JSON tool list and its hits come back as data."""

    def as_json(tool: str, **args: Any) -> Reply:
        return Reply(json.dumps({"tool": tool, "arguments": args}))

    clip = [{"op": "add_range", "file": "a.mp4", "start": 5.0, "end": 5.12, "at": 1}]
    fake = FakeProvider(
        [as_json("find_words", query="A"), as_json("propose_edit", summary="A", ops=clip)],
        Capabilities(False, True),
    )
    turn = run(fake, "Find me a clip where she says 'A'", proj=chat_project())
    assert turn.status == "done", turn.error
    first = fake.requests[0]
    assert first["tools"] is None and first["schema"] == session.JSON_SCHEMA
    assert words.JSON_TOOL in first["messages"][0]["content"]
    result = fake.requests[1]["messages"][-1]["content"]
    assert result.startswith("find_words result:\n<<<DATA words>>>") and "[A]" in result
    row = turn.view()["events"][-1]["after"][0]
    assert (row["source"], row["start"], row["end"], row["exact"]) == (A, 5.0, 5.12, True)


DAY1, DAY2 = "/v/day1/a.mp4", "/v/day2/a.mp4"


def two_days() -> session.Project:
    """A project of two sources named a.mp4 (folders day1 and day2), a moment in the second."""
    spoken = {DAY1: said((" a", 1.0, 1.1)), DAY2: said((" a", 2.0, 2.1))}
    moment = {"id": "d2", "source": DAY2, "start": 1.0, "end": 5.0, "score": 50, "label": "x"}
    return session.Project(
        [moment],
        [],
        ops.Durations({DAY1: 60.0, DAY2: 60.0}, lambda _p: None),
        "",
        "k",
        words=spoken.get,
    )


def test_two_sources_with_one_file_name_are_told_apart_by_their_folder() -> None:
    proj = two_days()
    same = [{"op": "add_range", "file": "a.mp4", "start": 2.0, "end": 2.1}]
    told = [{"op": "add_range", "file": "day2/a.mp4", "start": 2.0, "end": 2.1}]
    fake = FakeProvider(
        [
            call("find_words", query="a"),
            call("propose_edit", summary="x", ops=same),
            call("propose_edit", summary="y", ops=told),
        ]
    )
    turn = run(fake, "add the a", proj=proj)
    assert turn.status == "done", turn.error
    hits = data_rows(fake.requests[1]["messages"][-1]["content"])
    assert [h["file"] for h in hits] == ["day1/a.mp4", "day2/a.mp4"]  # on this machine
    assert '"file":"day2/a.mp4"' in fake.requests[0]["messages"][1]["content"]  # #974: moments too
    rejected = fake.requests[2]["messages"][-1]["content"]
    assert "several sources are named 'a.mp4': use one of day1/a.mp4, day2/a.mp4" in rejected
    assert turn.view()["events"][-1]["after"][0]["source"] == DAY2


class Hosted(FakeProvider):
    local = False
    host = "api.example.com"


class FarOllama(FakeProvider):
    """Ollama on another machine: local in kind, but what it is sent leaves this machine."""

    url = "http://192.168.1.50:11434"


@pytest.mark.parametrize("kind", [Hosted, FarOllama])
def test_a_model_not_on_this_machine_tells_them_apart_by_a_number_not_the_folder(
    kind: type[FakeProvider],
) -> None:
    """#1060: no folder name for a model whose prompts leave this machine; the numbered name
    it gives back still means its source."""
    told = [{"op": "add_range", "file": "a.mp4 (2)", "start": 2.0, "end": 2.1}]
    fake = kind([call("find_words", query="a"), call("propose_edit", summary="y", ops=told)])
    turn = run(fake, "add the a", proj=two_days())
    assert turn.status == "done", turn.error
    hits = data_rows(fake.requests[1]["messages"][-1]["content"])
    assert [h["file"] for h in hits] == ["a.mp4", "a.mp4 (2)"]
    assert '"file":"a.mp4 (2)"' in fake.requests[0]["messages"][1]["content"]
    sent = json.dumps(fake.requests, default=str)
    assert "day1" not in sent and "day2" not in sent
    assert turn.view()["events"][-1]["after"][0]["source"] == DAY2


def test_a_numbered_name_is_never_the_name_of_another_source() -> None:
    real = "/w/a.mp4 (2)"  # a source that is named so
    paths = [DAY1, real, DAY2, "/v/day3/a.mp4"]
    assert context.file_names(paths, local=False) == {
        DAY1: "a.mp4",
        real: "a.mp4 (2)",
        DAY2: "a.mp4 (3)",
        "/v/day3/a.mp4": "a.mp4 (4)",
    }
    assert context.file_names(paths, local=True)[DAY2] == "day2/a.mp4"
    table = context.MomentTable([], [])
    table.durations = dict.fromkeys(paths, 60.0)
    for name, want in (("a.mp4 (2)", real), ("a.mp4 (3)", DAY2), ("a.mp4", DAY1)):
        assert session._source_of(name, table, 1) == (want, "")
    assert session._source_of("day2/a.mp4", table, 1)[0] is None  # never shown to it


def test_the_names_follow_the_table_s_locality_and_rename_an_earlier_reply() -> None:
    table = context.MomentTable([], [])
    table.durations = {DAY1: 60.0, DAY2: 60.0}
    table.local = True
    assert table.file(DAY2) == "day2/a.mp4"
    # a hosted model's earlier reply: its numbered name is one the local model can give back
    assert table.other_names() == {"a.mp4 (2)": "day2/a.mp4"}
    assert session._without_names("a.mp4 (2)", table.other_names()) == "day2/a.mp4"
    table.local = False  # the user switched to a hosted model: no cached folder name
    assert table.file(DAY2) == "a.mp4 (2)"
    renamed = table.other_names()
    assert renamed == {
        "day1/a.mp4": "a.mp4",
        "day1\\a.mp4": "a.mp4",
        "day2/a.mp4": "a.mp4 (2)",
        "day2\\a.mp4": "a.mp4 (2)",
    }
    said = "Cut day2/a.mp4 at 2 s, then day1\\a.mp4, not xday2/a.mp4."
    want = "Cut a.mp4 (2) at 2 s, then a.mp4, not xday2/a.mp4."
    assert session._without_names(said, renamed) == want


def test_a_source_at_a_drive_root_renames_no_path_that_ends_in_its_name() -> None:
    table = context.MomentTable([], [])
    table.durations = {"/a.mp4": 60.0, DAY2: 60.0}
    assert table.file(DAY2) == "a.mp4 (2)"
    renamed = table.other_names()
    assert "/a.mp4" not in renamed and "\\a.mp4" not in renamed
    said = "It is clips/a.mp4, from day2/a.mp4."
    assert session._without_names(said, renamed) == "It is clips/a.mp4, from a.mp4 (2)."


def test_a_reply_names_match_whatever_the_case_and_show_the_name_as_is() -> None:
    """Windows paths differ in case: the scrub finds "b/CLIP.MP4" and writes the shown name."""
    table = context.MomentTable([], [])
    table.durations = {"/v/B/Clip.mp4": 60.0, "/v/c/Clip.mp4": 60.0}
    table.local = False
    renamed = table.other_names()
    said = "Use B\\Clip.mp4 and b/CLIP.MP4, then C/clip.mp4; not xb/clip.mp4."
    want = "Use Clip.mp4 and Clip.mp4, then Clip.mp4 (2); not xb/clip.mp4."
    assert session._without_names(said, renamed) == want
    # names that differ only in case keep their own shown name
    table.durations = {"/v/B/clip.mp4": 60.0, "/v/b/clip.mp4": 60.0}
    said = "B/clip.mp4 and b/clip.mp4"
    assert session._without_names(said, table.other_names()) == "clip.mp4 and clip.mp4 (2)"


def test_a_case_pair_unicode_maps_differently_never_raises_nor_leaks_the_folder() -> None:
    table = context.MomentTable([], [])
    table.durations = {"/v/\u0130zmir/clip.mp4": 60.0, "/v/b/clip.mp4": 60.0}
    table.local = False
    # re.IGNORECASE matches "I" to the dotted capital I; str.lower() does not map them alike
    assert session._without_names("see Izmir/clip.mp4", table.other_names()) == "see clip.mp4"


ROOT, UNDER = "C:\\clip.mp4", "C:\\v\\a\\clip.mp4"  # Windows paths: not keys of "/clip.mp4"


def _root_twins() -> session.Project:
    spoken = {ROOT: said((" a", 1.0, 1.1)), UNDER: said((" a", 2.0, 2.1))}
    return session.Project(
        [],
        [],
        ops.Durations({ROOT: 60.0, UNDER: 60.0}, lambda _p: None),
        "",
        "k",
        words=spoken.get,
    )


def test_a_local_model_picks_the_source_at_a_drive_root_by_the_name_it_is_shown() -> None:
    table = context.MomentTable([], [])
    table.durations = {ROOT: 60.0, UNDER: 60.0}
    table.local = True
    assert [table.file(p) for p in table.durations] == ["C:/clip.mp4", "a/clip.mp4"]
    for p in table.durations:
        assert session._source_of(table.file(p), table, 1) == (p, "")
    assert session._source_of("/clip.mp4", table, 1)[0] is None  # not what it was shown
    table.durations = {ROOT: 60.0, "/clip.mp4": 60.0}  # a POSIX root twin keeps "/clip.mp4"
    assert [table.file(p) for p in table.durations] == ["C:/clip.mp4", "/clip.mp4"]
    for p in table.durations:
        assert session._source_of(table.file(p), table, 1) == (p, "")
    told = [{"op": "add_range", "file": "C:/clip.mp4", "start": 1.0, "end": 1.1}]
    fake = FakeProvider([call("propose_edit", summary="x", ops=told)])
    turn = run(fake, "add it", proj=_root_twins())
    assert turn.status == "done", turn.error
    assert turn.view()["events"][-1]["after"][0]["source"] == ROOT


def test_a_root_without_a_drive_is_shown_with_a_slash_and_resolves() -> None:
    table = context.MomentTable([], [])
    table.durations = {"\\clip.mp4": 60.0, UNDER: 60.0}
    table.local = True
    assert table.file("\\clip.mp4") == "/clip.mp4"
    for name in ("/clip.mp4", "\\clip.mp4"):
        assert session._source_of(name, table, 1) == ("\\clip.mp4", "")
    assert session._source_of("a/clip.mp4", table, 1) == (UNDER, "")


def test_drive_roots_of_two_drives_are_told_apart_and_the_drive_stays_from_a_hosted_model() -> None:
    other = "D:\\clip.mp4"
    table = context.MomentTable([], [])
    table.durations = {ROOT: 60.0, other: 60.0}
    table.local = True
    assert [table.file(p) for p in table.durations] == ["C:/clip.mp4", "D:/clip.mp4"]
    for p in table.durations:
        assert session._source_of(table.file(p), table, 1) == (p, "")
    assert table.other_names() == {"clip.mp4 (2)": "D:/clip.mp4"}
    table.local = False
    assert [table.file(p) for p in table.durations] == ["clip.mp4", "clip.mp4 (2)"]
    renamed = table.other_names()
    said = "Cut C:/clip.mp4 and D:\\clip.mp4."
    assert session._without_names(said, renamed) == "Cut clip.mp4 and clip.mp4 (2)."


def test_a_hosted_model_picks_either_twin_of_a_drive_root_by_its_shown_name() -> None:
    table = context.MomentTable([], [])
    table.durations = {ROOT: 60.0, UNDER: 60.0}
    table.local = False
    for name, want in (("clip.mp4", ROOT), ("clip.mp4 (2)", UNDER)):
        assert session._source_of(name, table, 1) == (want, "")
    told = [{"op": "add_range", "file": "clip.mp4 (2)", "start": 2.0, "end": 2.1}]
    turn = run(Hosted([call("propose_edit", summary="x", ops=told)]), "add", proj=_root_twins())
    assert turn.status == "done", turn.error
    assert turn.view()["events"][-1]["after"][0]["source"] == UNDER


TWO_DAYS_TRANSCRIPT = """# /v/day1/a.mp4
[a.mp4 00:00:01.0-00:00:03.0] morning on day one

# /v/day2/a.mp4
[a.mp4 00:00:01.0-00:00:03.0] morning on day two

# /v/day2/a.mp4 (from 00:00:30)
[a.mp4@30 00:00:31.0-00:00:33.0] after the gap
"""


def test_transcript_lines_are_of_their_own_source_and_named_as_the_model_sees_it() -> None:
    lines = context.transcript_lines(TWO_DAYS_TRANSCRIPT, DAY2, 0.0, 40.0, shown="a.mp4 (2)")
    assert lines == [
        "[a.mp4 (2) 00:00:01.0-00:00:03.0] morning on day two",
        "[a.mp4 (2)@30 00:00:31.0-00:00:33.0] after the gap",
    ]
    lines = context.transcript_lines(TWO_DAYS_TRANSCRIPT, DAY1, 0.0, 40.0, shown="day1/a.mp4")
    assert lines == ["[day1/a.mp4 00:00:01.0-00:00:03.0] morning on day one"]
    # no heading names the source (an older or moved transcript): every line of its file name
    bare = TWO_DAYS_TRANSCRIPT.replace("# /v/", "# /old/")
    assert len(context.transcript_lines(bare, DAY2, 0.0, 40.0, shown="a.mp4")) == 3


def test_get_transcript_for_a_hosted_model_names_the_numbered_file() -> None:
    proj = two_days()
    proj.transcript = TWO_DAYS_TRANSCRIPT
    fake = Hosted([call("get_transcript", id="m1", pad=0), Reply("ok")])
    turn = run(fake, "what is said?", proj=proj)
    assert turn.status == "done", turn.error
    result = fake.requests[1]["messages"][-1]["content"]
    assert "[a.mp4 (2) 00:00:01.0-00:00:03.0] morning on day two" in result
    assert "day one" not in result and "day2" not in json.dumps(fake.requests, default=str)


@needs_ffmpeg
def test_a_real_export_of_a_short_word_keeps_it(home, monkeypatch) -> None:
    """A 0.12 s word exports as a clip of about that length (no tail, padding or speech move)
    with its 10 ms fade recorded."""
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    monkeypatch.setattr(sources, "source_words", lambda _src, _lang=None: None)
    project = home / "projects" / "p"
    project.mkdir(parents=True)
    src = project / "src.mp4"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
         "testsrc2=size=320x180:rate=30:duration=3", "-f", "lavfi", "-i",
         "sine=frequency=330:sample_rate=48000:duration=3", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)],
        check=True,
    )  # fmt: skip
    seg = write(
        project / "seg.json",
        json.dumps([{"source": str(src), "start": 1.0, "end": 1.12, "label": "a", "exact": True}]),
    )
    folder = Path(clips.export_clips(project, segments=seg, extend=clips.Padding(1.0, 1.0)))
    (entry,) = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["clips"]
    assert (entry["start"], entry["end"], entry["exact"]) == (1.0, 1.12, True)
    assert entry["audio_fade"] == planning.EXACT_AUDIO_FADE_S
    length = probe(folder / entry["file"]).duration
    assert 0.09 <= length <= 0.2, length


def test_rows_after_follows_apply_ops_and_gives_none_for_wrong_ops() -> None:
    """#974 review: the rows the word snap reads come from the ops code itself."""
    seq = [{"id": "c1", "source": "/v/a.mp4", "start": 2.0, "end": 9.0, "label": "x"}]
    cands = [{"id": "m1", "source": "/v/a.mp4", "start": 20.0, "end": 26.0, "label": "y"}]
    durations = {"/v/a.mp4": 100.0}
    copy = ops.rows_after(seq, [], cands, durations)
    assert copy == seq and copy[0] is not seq[0]
    later = ops.rows_after(
        seq, [{"op": "trim", "id": "c1", "end": 9.5}], cands, durations, snap=lambda _s, t: t + 0.3
    )
    assert later is not None and later[0]["end"] == pytest.approx(9.8)
    assert ops.rows_after(seq, [{"op": "trim", "id": "nope", "end": 3.0}], cands, durations) is None
