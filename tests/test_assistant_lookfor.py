"""AI Chat's ``look_for`` (CLIP search of the whole timeline, #529) and the ``color`` op.

No test loads CLIP or reads a real cache: the cached visual layer and the text embedding are
stubbed with small vectors (cosine 0 for most frames, 0.3 for a run of them).
"""

from __future__ import annotations

import importlib.util as util
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cutpoint import ui
from cutpoint.assistant import context, lookfor, ops, session
from cutpoint.assistant import store as ai_store
from cutpoint.assistant.providers import Capabilities, Reply
from cutpoint.events import visual as V
from test_assistant import CANDS, SEQ, A, B, FakeProvider, call, durations, project, run

TEXT = np.eye(768, dtype=np.float32)[0]


def frames(n: int, hot: range | list[int], cos: float = 0.3) -> SimpleNamespace:
    """``n`` frames at 1 fps whose cosine with :data:`TEXT` is 0, and ``cos`` for ``hot``."""
    rng = np.random.default_rng(n)
    emb = rng.normal(size=(n, 768)).astype(np.float32)
    emb[:, 0] = 0
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    for i in hot:
        emb[i] = cos * TEXT + np.sqrt(1 - cos**2) * emb[i]
    return SimpleNamespace(t=np.arange(n, dtype=np.float32), mean_emb=emb)


@pytest.fixture
def clip(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """CLIP present. A: a clear run (cosine 0.3) at 12-16 s that a moment covers, a strong but
    not clear one (0.23) at 50-54 s, a weak one (0.21) at 70-74 s and one under the floor
    (0.19) at 85-89 s. B: a clear run at 10-14 s that nothing covers. The dict records the
    text queries."""
    layers = {A: frames(100, range(12, 17)), B: frames(30, range(10, 15))}
    for lo, cos in ((50, 0.23), (70, 0.21), (85, 0.19)):
        layers[A].mean_emb[lo : lo + 5] = (
            cos * TEXT + np.sqrt(1 - cos**2) * layers[A].mean_emb[lo : lo + 5]
        )
    asked: dict[str, Any] = {"queries": [], "loads": []}

    def load_visual(src: Path, params: Any = None) -> Any:
        asked["loads"].append(Path(src).as_posix())
        return layers.get(Path(src).as_posix())

    def text_embeddings(prompts: tuple[str, ...], models: Any = None, *, save: bool = True):
        asked["queries"].append((prompts, save))
        return TEXT[None]

    monkeypatch.setattr(lookfor, "unavailable", lambda: None)
    monkeypatch.setattr(V, "load_visual", load_visual)
    monkeypatch.setattr(V, "text_embeddings", text_embeddings)
    return asked


COVER = [{"id": "c9", "source": A, "start": 11.0, "end": 18.0}]


def search(query: str = "powder spray", limit: int = 5, cover=COVER) -> lookfor.Found:
    return lookfor.Finder([A, B, "/v/none.mp4"]).search(
        query, limit, cover, {A: 100.0, B: 30.0, "/v/none.mp4": 50.0}
    )


def test_the_whole_timeline_is_scored_not_only_the_candidates(clip) -> None:
    found = search()
    assert (found.searched, found.total) == (2, 3)
    # The run a moment covers is that moment's match; the clear uncovered one is new; the
    # uncovered ones below the clear bar are listed too (strong at 0.23, weak at 0.21), the
    # run under the floor (0.19) is not.
    got = {(h.source, round(h.start), cover) for h, cover in found.hits}
    assert got == {(A, 12, "c9"), (B, 10, None), (A, 50, None), (A, 70, None)}
    by = {(h.source, round(h.start)): h for h, _ in found.hits}
    assert (by[B, 10].start, by[B, 10].end) == (10.0, 15.0)
    assert [by[k].confidence for k in ((A, 12), (B, 10), (A, 50), (A, 70))] == [
        "strong",
        "strong",
        "strong",
        "weak",
    ]
    assert [by[k].clear for k in ((A, 12), (B, 10), (A, 50), (A, 70))] == [True, True, False, False]
    assert by[B, 10].match == by[A, 12].match == 10  # (0.3 - 0.19) / 0.11 x 10
    assert found.best == 10
    # best first
    assert [h.cos for h, _ in found.hits] == sorted((h.cos for h, _ in found.hits), reverse=True)
    # a free-text query never writes a text-<hash>.npy under models/clip
    assert clip["queries"] == [(("powder spray",), False)]


def test_a_weak_match_is_listed_with_its_label_and_a_covered_one_keeps_its_moment(clip) -> None:
    weak = [{"id": "w1", "source": A, "start": 69.0, "end": 76.0}]
    got = {(h.confidence, cover) for h, cover in search(cover=[*COVER, *weak]).hits}
    assert ("weak", "w1") in got and ("strong", "c9") in got


def test_a_query_the_footage_cannot_answer_lists_nothing_or_only_a_few_weak(
    clip, monkeypatch
) -> None:
    monkeypatch.setattr(V, "load_visual", lambda *_a, **_k: frames(30, []))  # cosine 0
    found = search()
    assert found.hits == [] and found.best == 0
    # sources at the floor's edge: at most MAX_UNSURE weak rows, never a new moment
    noisy = frames(120, [])
    for lo in range(0, 120, 25):
        noisy.mean_emb[lo : lo + 3] = (
            0.205 * TEXT + np.sqrt(1 - 0.205**2) * noisy.mean_emb[lo : lo + 3]
        )
    monkeypatch.setattr(V, "load_visual", lambda *_a, **_k: noisy)
    found = search(cover=[])
    assert 0 < len(found.hits) <= lookfor.MAX_UNSURE
    assert all(h.confidence == "weak" and not h.clear for h, _ in found.hits)


def test_hits_are_capped_by_limit_and_new_moments_by_max_new(clip, monkeypatch) -> None:
    assert len(search(limit=1).hits) == 1
    monkeypatch.setattr(lookfor, "MAX_NEW", 1)
    clear = [h for h, c in search(cover=[]).hits if c is None and h.clear]
    assert len(clear) == 1


def test_unsure_rows_are_capped(clip, monkeypatch) -> None:
    monkeypatch.setattr(lookfor, "MAX_UNSURE", 1)
    got = [h for h, _ in search().hits if h.confidence == "weak" or h.cos < lookfor.NEW_MIN]
    assert len(got) == 1 and got[0].cos == pytest.approx(0.23, abs=1e-3)


def test_the_embeddings_are_read_once_per_finder(clip) -> None:
    finder = lookfor.Finder([A, B])
    for query in ("one", "two"):
        finder.search(query, 5, [], {})
    assert clip["loads"] == [A, B]
    assert [q[0] for q in clip["queries"]] == [("one",), ("two",)]


def test_lite_names_the_edition_and_the_install_command(monkeypatch) -> None:
    monkeypatch.undo()  # the fixture is not used: the real check, with torch hidden
    real = util.find_spec
    monkeypatch.setattr(util, "find_spec", lambda m, *a: None if m == "torch" else real(m, *a))
    why = lookfor.unavailable()
    assert why is not None and "not available in Cutpoint lite" in why
    with pytest.raises(lookfor.LookForError, match="not available in Cutpoint lite"):
        lookfor.Finder([A]).search("x", 5, [], {})


def test_missing_clip_weights_say_how_to_get_them(clip, monkeypatch) -> None:
    def missing(*_a: Any, **_k: Any):
        raise V.ClipWeightsMissing("CLIP weights not found in /models")

    monkeypatch.setattr(V, "text_embeddings", missing)
    with pytest.raises(lookfor.LookForError, match=r"cutpoint setup --only visual"):
        search()


def test_no_cached_embeddings_say_so(clip, monkeypatch) -> None:
    monkeypatch.setattr(V, "load_visual", lambda *_a, **_k: None)
    with pytest.raises(lookfor.LookForError, match="no source of this project has cached frame"):
        search()


def test_an_unreadable_cache_is_skipped_not_fatal(clip, monkeypatch) -> None:
    good = V.load_visual

    def load(src: Path, params: Any = None):
        if Path(src).as_posix() == A:
            raise ValueError("bad npz")
        return good(src)

    monkeypatch.setattr(V, "load_visual", load)
    found = search()
    assert found.searched == 1 and {h.source for h, _ in found.hits} == {B}


def test_a_new_moment_has_the_id_the_page_gives_an_analysis_moment() -> None:
    hit = lookfor.Hit(B, 10.0, 15.0, 0.3)
    cand = lookfor.new_moment(hit, "  powder   spray ")
    assert cand == {
        "id": ui.moments_view({"candidates": [{"source": B, "start": 10.0, "end": 15.0}]})[
            "candidates"
        ][0]["id"],
        "source": B,
        "start": 10.0,
        "end": 15.0,
        "label": "powder spray",
        "score": 100,
        "pool": "look_for",
    }


# -- the tool in a turn ------------------------------------------------------------------------


def look_then_add() -> list[Any]:
    return [
        call("look_for", query="powder spray"),
        call("propose_edit", summary="Add the powder shot", ops=[{"op": "add", "id": "m5"}]),
    ]


def test_a_strong_hit_nobody_covers_becomes_a_moment_the_proposal_can_add(clip) -> None:
    fake = FakeProvider(look_then_add())
    turn = run(fake)
    assert turn.status == "done", turn.error
    step, prop = turn.events
    assert step["name"] == "look_for" and (step["found"], step["new"], step["weak"]) == (4, 1, 1)
    rows = [
        json.loads(line)
        for line in fake.requests[1]["messages"][-1]["content"].splitlines()
        if line.startswith("{")
    ]
    new = next(r for r in rows if r.get("new"))
    assert new["id"] == "m5" and new["file"] == "b.mp4" and (new["start"], new["end"]) == (10, 15)
    assert new["label"] == "powder spray" and new["match"] == 10
    assert new["confidence"] == "strong"
    covered = next(r for r in rows if r["id"] == "m1")
    assert covered["seq"] is None and "new" not in covered and "unconfirmed" not in covered  # c1
    # the others are listed for the model to confirm, never as new moments
    by_id = {r["id"]: r for r in rows}
    assert (by_id["m6"]["confidence"], by_id["m6"]["unconfirmed"]) == ("strong", True)
    assert (by_id["m7"]["confidence"], by_id["m7"]["unconfirmed"]) == ("weak", True)
    assert not any("new" in by_id[k] for k in ("m6", "m7"))
    assert "guesses" in fake.requests[1]["messages"][-1]["content"]
    real = lookfor.moment_id(B, 10.0, 15.0)
    assert prop["ops"] == [{"op": "add", "id": real}]
    assert prop["after"][-1] == {
        "id": real,
        "source": B,
        "start": 10.0,
        "end": 15.0,
        "label": "powder spray",
    }
    # the proposal carries the moment, so the apply route can add it again on a changed sequence
    assert [m["id"] for m in prop["moments"]] == [real]  # not the unconfirmed ones
    assert {c["id"] for c in turn.found} >= {real}  # ... which can still be viewed and added
    assert turn.project.candidates == CANDS  # the project's own list is untouched


def test_the_second_look_reuses_the_moment_and_its_short_id(clip) -> None:
    fake = FakeProvider([call("look_for", query="powder spray"), call("look_for", query="spray")])
    turn = run(fake)
    # the second look finds the first one's moment as a moment (no longer new)
    assert [e["new"] for e in turn.events if e.get("name") == "look_for"] == [1, 0]
    assert len(turn.found) == 3 and turn.table.name(turn.found[0]["id"]) == "m5"


def test_a_failed_look_is_reported_to_the_model_and_the_page(clip, monkeypatch) -> None:
    monkeypatch.setattr(lookfor, "unavailable", lambda: "torch is not installed: lite")
    fake = FakeProvider([call("look_for", query="sunset")])
    turn = run(fake)
    result = fake.requests[1]["messages"][-1]["content"]
    assert result.startswith("error: look_for is not available: torch is not installed")
    assert turn.events[0]["error"].startswith("torch is not installed")
    idle = session.Turn(FakeProvider([]), project(), "hi")
    for bad in ({}, {"query": "  "}, {"query": 3}):
        assert idle._tool_look_for(bad) == ("error: query must be text", False)


def test_nothing_close_says_so_with_the_best_match(clip, monkeypatch) -> None:
    monkeypatch.setattr(V, "load_visual", lambda *_a, **_k: frames(30, []))  # cosine 0 everywhere
    turn = session.Turn(FakeProvider([]), project(), "x")
    text, final = turn._tool_look_for({"query": "rainbow"})
    assert not final and text.startswith("no window matches closely (the best frame is 0/10)")
    assert turn.events[-1]["found"] == 0 and not turn.found


@pytest.mark.parametrize("caps", [Capabilities(True, True), Capabilities(False, True)])
def test_look_for_works_with_tool_calls_and_in_json_mode(clip, caps) -> None:
    if caps.tool_calls:
        fake = FakeProvider(look_then_add(), caps)
    else:
        script = [
            json.dumps({"tool": "look_for", "arguments": {"query": "powder spray"}}),
            json.dumps(
                {
                    "tool": "propose_edit",
                    "arguments": {"summary": "Add it", "ops": [{"op": "add", "id": "m5"}]},
                }
            ),
        ]
        fake = FakeProvider([Reply(t) for t in script], caps)
    turn = run(fake)
    assert turn.status == "done", turn.error
    assert turn.events[-1]["kind"] == "proposal" and turn.events[-1]["moments"]
    system = fake.requests[0]["messages"][0]["content"]
    assert "Similarity is not certainty" in system
    if not caps.tool_calls:
        assert "look_for {query, limit?}" in system
    else:
        assert "look_for" in [t["function"]["name"] for t in fake.requests[0]["tools"]]


def test_the_rules_tell_the_model_what_to_do_about_weak_windows() -> None:
    images, text = lookfor.rules(True), lookfor.rules(False)
    assert lookfor.RULES in images and lookfor.RULES in text
    assert "check it with view_moment" in images and "never add it" in images
    assert "view_moment" not in text and "you cannot look at it" in text
    assert "tell the user it is uncertain" in text
    assert "confirm" in lookfor.TOOL["function"]["description"]
    assert "weak" in lookfor.RULES and "confidence" in lookfor.RULES


def test_the_rules_say_when_to_look_and_that_similarity_is_not_certainty() -> None:
    rules = lookfor.RULES
    assert "do not match" in rules and "Similarity is not certainty" in rules
    assert "new: true" in rules
    assert "only grades underwater footage" in context.RULES
    assert '"op": "color"' in context.RULES


# -- the color op ------------------------------------------------------------------------------


def apply(*op: dict[str, Any]) -> ops.Outcome:
    return ops.apply_ops([dict(s) for s in SEQ], list(op), CANDS, durations())


@pytest.mark.parametrize(
    ("op", "want"),
    [
        ({"op": "color", "mode": "auto", "strength": 0.8}, {"mode": "auto", "strength": 0.8}),
        ({"op": "color", "mode": "underwater"}, {"mode": "underwater", "strength": None}),
        ({"op": "color", "mode": "auto", "strength": 0}, {"mode": "auto", "strength": 0.0}),
        ({"op": "color", "mode": "off", "strength": 0.5}, {"mode": "off", "strength": None}),
    ],
)
def test_a_color_op_alone_changes_no_row(op: dict, want: dict) -> None:
    out = apply(op)
    assert out.error is None, out.error
    assert out.color == want
    assert out.after == SEQ and all(not r["changes"] for r in out.diff)


@pytest.mark.parametrize(
    ("op", "why"),
    [
        ({"op": "color", "mode": "sepia"}, "mode must be one of off, auto, underwater"),
        ({"op": "color"}, "mode must be one of"),
        ({"op": "color", "mode": "auto", "strength": 1.5}, "from 0 to 1"),
        ({"op": "color", "mode": "auto", "strength": -0.1}, "from 0 to 1"),
        ({"op": "color", "mode": "auto", "strength": "high"}, "strength must be a number"),
        ({"op": "color", "mode": "auto", "strength": True}, "strength must be a number"),
    ],
)
def test_a_bad_color_op_is_refused_with_a_reason(op: dict, why: str) -> None:
    out = apply(op)
    assert out.after is None and out.color is None and why in (out.error or ""), out.error


def test_color_goes_with_other_ops_once_and_a_row_op_error_still_wins() -> None:
    out = apply({"op": "remove", "id": "c3"}, {"op": "color", "mode": "auto"})
    assert out.error is None and out.color == {"mode": "auto", "strength": None}
    assert [r["changes"] for r in out.diff if r["id"] == "c3"] == [["removed"]]
    twice = apply({"op": "color", "mode": "auto"}, {"op": "color", "mode": "off"})
    assert "op 2 (color): only one color op" in (twice.error or "")
    assert apply({"op": "remove", "id": "c1"}, {"op": "color", "mode": "auto"}).color is None


def test_extra_moments_keeps_only_usable_entries() -> None:
    good = {"id": "x", "source": A, "start": 1.0, "end": 2.0}
    junk = [good, {"id": "y"}, "z", {**good, "start": float("nan")}, {**good, "id": 3}]
    assert ops.extra_moments(junk) == [good]
    assert ops.extra_moments(None) == [] and ops.extra_moments({"id": "x"}) == []


def test_a_color_only_proposal_is_a_proposal() -> None:
    fake = FakeProvider(
        [
            call("propose_edit", summary="x", ops=[{"op": "move", "id": "m2", "to": 1}]),
            call(
                "propose_edit",
                summary="Fix the dull colors",
                ops=[{"op": "color", "mode": "auto", "strength": 0.7}],
                suggestions=[
                    {"label": "Underwater", "ops": [{"op": "color", "mode": "underwater"}]},
                    {"label": "Nothing", "ops": [{"op": "move", "id": "m1", "to": 1}]},
                ],
            ),
        ]
    )
    turn = run(fake)
    assert "leave the sequence as it is" in fake.requests[1]["messages"][-1]["content"]
    prop = turn.events[-1]
    assert prop["color"] == {"mode": "auto", "strength": 0.7} and "moments" not in prop
    assert [s["color"] for s in prop["suggestions"]] == [{"mode": "underwater", "strength": None}]
    assert prop["after"] == SEQ


# -- routes ------------------------------------------------------------------------------------

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from test_assistant_routes import PORT, ask, files, use  # noqa: E402
from test_assistant_routes import proj as proj  # noqa: E402

BASE = "/api/projects/cli/assistant"


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = ui.create_app(PORT)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        c.headers["X-Cutpoint-Token"] = app.state.token
        yield c


def test_a_color_change_is_applied_recorded_and_reverted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, proj
) -> None:
    p, seq, _ = proj
    use(
        monkeypatch,
        FakeProvider(
            [
                call(
                    "propose_edit",
                    summary="Grade it",
                    ops=[{"op": "color", "mode": "auto", "strength": 0.8}],
                ),
                call(
                    "propose_edit",
                    summary="Underwater",
                    ops=[{"op": "color", "mode": "underwater"}],
                ),
            ]
        ),
    )
    prop = ask(client, "colors are dull", seq)["events"][0]
    off = {"mode": "off", "strength": 0.6}
    before = files(p)
    entry = client.post(
        f"{BASE}/proposals/{prop['id']}/applied", json={"segments": seq, "color": off}
    ).json()
    assert entry["color"] == {"before": off, "after": {"mode": "auto", "strength": 0.8}}
    assert entry["after"] == seq and all(not r["changes"] for r in entry["diff"])
    assert files(p) != before  # the history was written; review.json never is
    assert not (p / "ui" / "review.json").exists()

    # a strength the op left out stays what the page has; an old page (no color) records none
    second = ask(client, "underwater", seq)["events"][0]
    kept = client.post(
        f"{BASE}/proposals/{second['id']}/applied",
        json={"segments": seq, "color": {"mode": "auto", "strength": 0.8}},
    ).json()
    assert kept["color"]["after"] == {"mode": "underwater", "strength": 0.8}
    old = client.post(f"{BASE}/proposals/{second['id']}/applied", json={"segments": seq}).json()
    assert old["color"] == {"before": None, "after": {"mode": "underwater", "strength": 1.0}}

    # revert: previewed with the Color it puts back, unless the page is already there
    now = {"mode": "auto", "strength": 0.8}
    rev = client.post(f"{BASE}/history/{entry['id']}/revert", json={"segments": seq, "color": now})
    assert rev.status_code == 200, rev.text
    assert rev.json()["color"] == off and rev.json()["after"] == seq
    same = client.post(f"{BASE}/history/{entry['id']}/revert", json={"segments": seq, "color": off})
    assert same.status_code == 409 and "already as it was" in same.json()["detail"]
    done = client.post(
        f"{BASE}/proposals/{rev.json()['id']}/applied",
        json={"segments": seq, "base_key": rev.json()["base_key"], "color": now},
    ).json()
    assert done["color"] == {"before": now, "after": off}


def test_the_page_color_is_checked_before_it_is_recorded(
    client: TestClient, monkeypatch, proj
) -> None:
    _, seq, _ = proj
    use(
        monkeypatch,
        FakeProvider(
            [call("propose_edit", summary="Grade", ops=[{"op": "color", "mode": "auto"}])]
        ),
    )
    prop = ask(client, "grade", seq)["events"][0]
    for junk in ("auto", {"mode": "sepia", "strength": 1}, {"mode": "auto", "strength": 7}, 5):
        got = client.post(
            f"{BASE}/proposals/{prop['id']}/applied", json={"segments": seq, "color": junk}
        ).json()
        assert got["color"]["before"] is None


def test_a_look_for_moment_is_added_persisted_and_rebased(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, proj
) -> None:
    p, seq, ids = proj
    src = seq[0]["source"]
    path = Path(src).as_posix()
    monkeypatch.setattr(lookfor, "unavailable", lambda: None)
    monkeypatch.setattr(V, "load_visual", lambda s, params=None: frames(100, range(70, 75)))
    monkeypatch.setattr(V, "text_embeddings", lambda *_a, **_k: TEXT[None])
    use(
        monkeypatch,
        FakeProvider(
            [
                call("look_for", query="sunset over the peaks"),
                call("propose_edit", summary="Add the sunset", ops=[{"op": "add", "id": "m4"}]),
            ]
        ),
    )
    turn = ask(client, "find a sunset", seq)
    step, prop = turn["events"]
    assert step["name"] == "look_for" and step["new"] == 1, path
    real = lookfor.moment_id(src, 70.0, 75.0)
    assert [m["id"] for m in prop["moments"]] == [real]
    assert prop["after"][-1]["id"] == real and prop["after"][-1]["label"] == "sunset over the peaks"
    # the saved chat has the moment: the proposal still applies after a restart ...
    saved = ai_store.find_proposal(p, prop["id"])
    assert saved is not None and saved[1]["moments"] == prop["moments"]
    # ... and on a changed sequence, where its ops run again (the analysis has no such moment)
    url = f"{BASE}/proposals/{prop['id']}/applied"
    moved = client.post(url, json={"segments": seq[1:]}).json()
    assert moved["rebased"] is True and [s["id"] for s in moved["after"]] == [ids[1], real]
    plain = client.post(url, json={"segments": seq}).json()
    assert plain["rebased"] is False and plain["after"] == prop["after"]
    # the saved sequence of a review accepts it
    entries = client.app.state.sources.entries(p)
    assert ui.validate_segments(plain["after"], entries, client.app.state.media) == plain["after"]
