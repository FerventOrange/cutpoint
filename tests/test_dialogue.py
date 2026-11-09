"""The dialogue reader of ``analyze`` (``cutpoint.events.dialogue``): pieces, parsing, merging,
strength, the cache, the choice of backend, the events it writes, the candidates it nominates and
how a long one exports.

Transcripts are invented neutral lines; every model and HTTP call is faked (no network).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import pytest
import requests

from conftest import set_config, warning_codes, write
from cutpoint import analyze, cache, jsonout, planning, progress, vision
from cutpoint.assistant import providers
from cutpoint.events import dialogue, report, scoring
from cutpoint.events.eventtypes import DEFAULT_EVENT_TYPES
from cutpoint.styles import get_style
from cutpoint.ui import moments_view
from cutpoint.utils import slugify, title_slug
from test_analyze_steps import progress_lines, run  # noqa: F401
from test_assistant_byo import KEY, no_keys  # noqa: F401  (autouse: no real keyring or key)
from test_event_candidates import (  # noqa: F401  (no_transcript is autouse there)
    SRC,
    STYLE,
    events_of,
    feed,
    grid,
    highlights,
    no_transcript,
    nominees,
    own,
    sound,
)
from test_planning import plan_env, readable, speech_env  # noqa: F401


# --------------------------------------------------------------------------- helpers
def talk_lines(n: int, step: float = 10.0, length: float = 8.0) -> list[dialogue.Line]:
    """``n`` transcript lines, one every ``step`` seconds."""
    return [(i * step, i * step + length, f"line {i}") for i in range(n)]


def stretch(
    start: float, end: float, rating: int = 5, what: str = "x", first: int = 0, last: int = 0
) -> dict[str, Any]:
    return {
        "what_happens": what,
        "first_line": first,
        "last_line": last,
        "kind": "funny",
        "rating": rating,
        "start": start,
        "end": end,
    }


def moment(
    first: Any, last: Any, rating: Any = 6, kind: Any = "funny", what: str = "a bit"
) -> dict:
    return {
        "what_happens": what,
        "first_line": first,
        "last_line": last,
        "kind": kind,
        "rating": rating,
    }


# --------------------------------------------------------------------------- pieces
def test_pieces_are_480_seconds_with_60_seconds_shared_with_the_next():
    lines = talk_lines(100)  # lines at 0..998 s
    pieces = dialogue.split_pieces(lines)
    assert [(p[0], p[-1]) for p in pieces] == [(0, 47), (42, 89), (84, 99)]
    assert set(pieces[0]) & set(pieces[1]) == set(range(42, 48))  # the shared minute
    assert sorted({i for p in pieces for i in p}) == list(range(100))  # no line is lost


def test_pieces_leave_out_stretches_without_speech_and_an_empty_transcript_has_none():
    lines = [(0.0, 5.0, "first"), (2000.0, 2005.0, "last")]
    assert dialogue.split_pieces(lines) == [[0], [1]]
    assert dialogue.split_pieces([]) == []


def test_lines_from_segments_join_the_words_and_drop_empty_ones():
    segments = [
        {"start": 1.0, "end": 2.0, "text": "  two   words "},
        {"start": 3.0, "end": 4.0, "text": "   "},
        {"start": 5.0, "end": 6.0},
    ]
    assert dialogue.lines_from_segments(segments) == [(1.0, 2.0, "two words")]


# --------------------------------------------------------------------------- parsing
def test_a_reply_keeps_valid_stretches_and_counts_the_dropped_ones():
    lines = talk_lines(10)
    idx = list(range(2, 8))  # the piece shows lines 2-7
    reply = {
        "moments": [
            moment(2, 4, rating=7, what="  a  joke\nlands "),
            moment("5", 6.0, rating=3.4),  # a whole number as a string or a float is fine
            moment(1, 4),  # first line is not in the piece
            moment(3, 99),  # last line is not in the piece
            moment(6, 3),  # ends before it starts
            moment(None, 4),
            moment(True, 4),
            moment(2, 4, kind="rant"),  # not one of the kinds
            moment(2, 4, rating="high"),
            moment(2, 4, rating=float("nan")),
            "not a moment",
        ]
    }
    got, dropped = dialogue.parse_reply(reply, lines, idx)
    assert dropped == 9
    assert [(s["first_line"], s["last_line"], s["rating"]) for s in got] == [(2, 4, 7), (5, 6, 3)]
    first = got[0]
    assert (first["start"], first["end"]) == (lines[2][0], lines[4][1])  # source time of the lines
    assert first["what_happens"] == "a joke lands"
    assert first["kind"] == "funny"


@pytest.mark.parametrize(("given", "kept"), [(0, 1), (-4, 1), (11, 10), (99, 10), (6.6, 7)])
def test_a_rating_is_clamped_to_the_scale(given, kept):
    got, dropped = dialogue.parse_reply(
        {"moments": [moment(0, 1, rating=given)]}, talk_lines(3), [0, 1]
    )
    assert dropped == 0 and got[0]["rating"] == kept


@pytest.mark.parametrize("reply", [None, {}, {"moments": "none"}, {"moments": None}, {"other": []}])
def test_a_reply_that_is_not_a_list_of_moments_has_none_and_counts_as_one_dropped(reply):
    assert dialogue.parse_reply(reply, talk_lines(3), [0, 1, 2]) == ([], 1)


def test_an_empty_list_of_moments_is_a_valid_reply():
    assert dialogue.parse_reply({"moments": []}, talk_lines(3), [0, 1, 2]) == ([], 0)


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ('{"moments": []}', {"moments": []}),
        ('Sure, here it is:\n```json\n{"moments": [{"first_line": 1}]}\n```\nHope it helps.',
         {"moments": [{"first_line": 1}]}),
        ('{not json} and then {"moments": []}', {"moments": []}),
        ("I can't help with that.", None),
        ("[1, 2, 3]", None),
        ("", None),
    ],
)  # fmt: skip
def test_json_is_taken_out_of_the_text_around_it(text, want):
    assert dialogue.extract_json(text) == want


# --------------------------------------------------------------------------- merging
def test_stretches_overlapping_by_more_than_half_of_the_shorter_become_their_union():
    a = stretch(100, 200, rating=5, what="early", first=10, last=20)
    b = stretch(140, 240, rating=8, what="late", first=14, last=24)  # 60 s of 100 s shared
    (got,) = dialogue.merge_stretches([b, a])
    assert (got["start"], got["end"]) == (100, 240)
    assert (got["first_line"], got["last_line"]) == (10, 24)
    assert (got["rating"], got["what_happens"]) == (8, "late")  # the higher rating's text


def test_a_tie_keeps_the_earlier_stretch():
    a = stretch(100, 200, rating=6, what="early")
    b = stretch(130, 230, rating=6, what="late")
    (got,) = dialogue.merge_stretches([a, b])
    assert got["what_happens"] == "early"


def test_a_stretch_inside_a_longer_one_is_a_duplicate_of_it():
    long = stretch(100, 400, rating=3, what="long")
    short = stretch(150, 200, rating=9, what="short")
    (got,) = dialogue.merge_stretches([long, short])
    assert (got["start"], got["end"], got["rating"]) == (100, 400, 9)


@pytest.mark.parametrize("b_start", [150.0, 190.0, 200.0, 260.0])
def test_stretches_overlapping_by_half_or_less_stay_apart(b_start):
    a = stretch(100, 200)
    b = stretch(b_start, b_start + 100)
    assert len(dialogue.merge_stretches([b, a])) == 2


def test_merged_stretches_come_by_start():
    got = dialogue.merge_stretches([stretch(500, 560), stretch(10, 50), stretch(200, 260)])
    assert [s["start"] for s in got] == [10, 200, 500]


# --------------------------------------------------------------------------- strength
@pytest.mark.parametrize(
    ("rating", "strength"), [(10, 1.0), (8, 0.8), (1, 0.1), (0, 0.0), (15, 1.0)]
)
def test_a_hosted_stretch_is_as_strong_as_the_model_rated_it(rating, strength):
    (ev,) = dialogue.to_events(SRC, [stretch(10, 70, rating=rating)], None, "hosted", "m")
    assert ev["confidence"] == pytest.approx(strength)


@pytest.mark.parametrize(
    ("sounds", "span", "strength"),
    [
        ([], 60.0, 0.2),  # no reaction: the baseline
        ([10.0], 60.0, 0.2 + 0.8 / 3),  # one a minute
        ([10.0, 30.0, 50.0], 60.0, 1.0),  # 3 a minute: full
        ([float(t) for t in range(10, 60, 4)], 60.0, 1.0),  # more does not go past 1
        ([100.0, 400.0], 60.0, 0.2),  # outside the stretch
        (
            [20.0],
            10.0,
            0.2 + 0.8 * 2 / 3,
        ),  # a short stretch counts as 30 s: one laugh is not a burst
    ],
)
def test_reaction_strength_rises_from_the_baseline_to_three_sounds_a_minute(sounds, span, strength):
    assert dialogue.reaction_strength(10.0, 10.0 + span, sounds) == pytest.approx(strength)


def test_a_local_stretch_is_ranked_by_its_reactions_and_ignores_the_models_rating():
    quiet = stretch(10, 70, rating=10)  # the model loved it, nobody reacted
    loud = stretch(100, 160, rating=1)  # the model did not, and three sounds fall inside
    evs = dialogue.to_events(SRC, [quiet, loud], [105.0, 120.0, 150.0], "local", "m")
    assert [e["confidence"] for e in evs] == [pytest.approx(0.2), pytest.approx(1.0)]


def test_a_local_stretch_without_audio_events_is_at_the_baseline():
    (ev,) = dialogue.to_events(SRC, [stretch(10, 70, rating=10)], None, "local", "m")
    assert ev["confidence"] == pytest.approx(dialogue.BASELINE_STRENGTH)


def test_reaction_times_are_the_confident_reaction_sounds_of_the_source():
    def audio(label, t, conf=0.5, src=SRC):
        return {"kind": "audio", "label": label, "start": t, "confidence": conf, "source": src}

    events = [
        audio("Laughter", 30.0),
        audio("Scream", 10.0),
        audio("Cheering", 20.0),
        audio("Speech", 5.0),  # not a reaction
        audio("Laughter", 6.0, conf=0.0),  # no confidence
        audio("Laughter", 7.0, src="other.mp4"),
        {"kind": "ocr", "label": "Laughter", "start": 8.0, "confidence": 0.9, "source": SRC},
    ]
    assert dialogue.reaction_times(events, SRC) == [10.0, 20.0, 30.0]
    assert dialogue.reaction_times(events) == [7.0, 10.0, 20.0, 30.0]


@pytest.mark.parametrize(
    ("given", "want"),
    [
        ("Grandpa's Wobbly Treehouse", "Grandpa's Wobbly Treehouse"),
        ('  "Chat Argues About Pizza".  ', "Chat Argues About Pizza"),
        ("One\ntwo\tthree  four", "One two three four"),
        ("one two three four five six seven eight", "one two three four five six"),
        ("Grandpa\u2019s Terrible Cooking", "Grandpa's Terrible Cooking"),
        ("", ""),
        ("  ...  ", ""),
        (None, ""),
        (5, ""),
        (["a title"], ""),
    ],
)
def test_a_title_is_one_line_of_at_most_six_words(given, want):
    assert dialogue.clean_title(given) == want


def test_a_title_is_cut_at_a_word_within_the_character_cap():
    long_words = " ".join(["extraordinarily"] * 6)
    got = dialogue.clean_title(long_words)
    assert len(got) <= dialogue.MAX_TITLE and long_words.startswith(got)
    assert not got.endswith("extraordinaril")  # not in the middle of a word
    assert len(dialogue.clean_title("x" * 200)) == dialogue.MAX_TITLE


def test_a_reply_title_is_kept_cleaned_and_a_missing_one_is_empty():
    lines = talk_lines(10)
    reply = {
        "moments": [
            {**moment(0, 2), "title": "  Grandpa's Treehouse, Again!  "},
            moment(3, 4),  # no title
            {**moment(5, 6), "title": 7},  # not text
            {**moment(7, 8), "title": " ... "},
        ]
    }
    got, dropped = dialogue.parse_reply(reply, lines, list(range(10)))
    assert dropped == 0  # a missing title never drops the stretch
    assert [s["title"] for s in got] == ["Grandpa's Treehouse, Again", "", "", ""]


def test_the_schema_and_the_prompt_ask_for_a_title():
    item = dialogue.SCHEMA["properties"]["moments"]["items"]
    assert "title" in item["properties"] and "title" in item["required"]
    assert "title" in dialogue.PROMPT and '"title"' in dialogue.JSON_ONLY


# --------------------------------------------------------------------------- cache key
def test_the_cache_key_follows_the_transcript_the_model_and_the_prompt(monkeypatch):
    lines = talk_lines(5)
    key = dialogue.cache_key(lines, "ollama@h:m")
    assert dialogue.cache_key(talk_lines(5), "ollama@h:m") == key  # same input, same key
    assert dialogue.cache_key(lines, "ollama@h:other") != key
    assert (
        dialogue.cache_key([*lines[:-1], (40.0, 48.0, "line four, reworded")], "ollama@h:m") != key
    )
    assert dialogue.cache_key([*lines[:-1], (40.0, 49.0, "line 4")], "ollama@h:m") != key
    monkeypatch.setattr(dialogue, "PROMPT_VERSION", dialogue.PROMPT_VERSION + 1)
    assert dialogue.cache_key(lines, "ollama@h:m") != key


# --------------------------------------------------------------------------- reading a source
def fake_backend(answer, *, key="fake@h:m", workers=1) -> tuple[dialogue.Backend, list[str]]:
    """A backend whose ``ask`` is ``answer(prompt)``; the prompts it got."""
    prompts: list[str] = []
    lock = threading.Lock()

    def ask(prompt: str) -> dict[str, Any] | None:
        with lock:
            prompts.append(prompt)
        return answer(prompt)

    return dialogue.Backend("hosted", "m", key, ask, workers), prompts


def test_a_stretch_found_in_two_overlapping_pieces_is_one(tmp_path):
    src = write(tmp_path / "a.mp4")
    lines = talk_lines(100)

    def answer(prompt: str):
        # lines 44-46 are in pieces 0 and 1 only
        return {"moments": [moment(44, 46)]} if "\n46: " in prompt else {"moments": []}

    backend, prompts = fake_backend(answer, workers=2)
    read = dialogue.read_source(src, lines, backend)
    assert len(prompts) == read.pieces == 3
    assert [(s["first_line"], s["last_line"]) for s in read.stretches] == [(44, 46)]
    assert read.failed == 0 and not read.cached
    assert prompts[0].startswith(dialogue.PROMPT) and "0: line 0" in prompts[0]


def test_a_reading_is_taken_from_the_cache_until_the_transcript_or_model_changes(tmp_path):
    src = write(tmp_path / "a.mp4")
    lines = talk_lines(10)
    backend, prompts = fake_backend(lambda _p: {"moments": [moment(1, 3, rating=9)]})
    first = dialogue.read_source(src, lines, backend)
    again = dialogue.read_source(src, lines, backend)
    assert len(prompts) == 1  # the second reading asked nothing
    assert again.cached and again.stretches == first.stretches
    other, more = fake_backend(lambda _p: {"moments": []}, key="fake@h:other")
    assert not dialogue.read_source(src, lines, other).cached and len(more) == 1
    edited = [*lines[:-1], (90.0, 98.0, "something else was said")]
    assert not dialogue.read_source(src, edited, backend).cached and len(prompts) == 2


def test_a_reading_cached_before_titles_is_read_again(tmp_path, monkeypatch):
    src = write(tmp_path / "a.mp4")
    lines = talk_lines(10)
    backend, prompts = fake_backend(
        lambda _p: {"moments": [{**moment(1, 3), "title": "Mike Burns Dinner"}]}
    )
    monkeypatch.setattr(dialogue, "PROMPT_VERSION", 1)  # the version before titles
    old = dialogue._cache_path(src, lines, backend)
    cache.save_json(old, {"version": 1, "stretches": [stretch(10.0, 40.0)]})
    assert dialogue._cached_stretches(old) is not None
    monkeypatch.undo()
    assert dialogue.PROMPT_VERSION >= 2
    first = dialogue.read_source(src, lines, backend)
    assert len(prompts) == 1 and not first.cached  # the stretch without a title is not reused
    assert [s["title"] for s in first.stretches] == ["Mike Burns Dinner"]
    again = dialogue.read_source(src, lines, backend)
    assert again.cached and again.stretches[0]["title"] == "Mike Burns Dinner"  # titles are cached


def test_a_failed_piece_is_skipped_and_the_reading_is_not_cached(tmp_path, caplog):
    src = write(tmp_path / "a.mp4")
    lines = talk_lines(100)

    def answer(prompt: str):
        if "\n80: " in prompt and "\n40: " not in prompt:  # the last piece: no usable reply
            return None
        return {"moments": [moment(2, 4)]}

    backend, prompts = fake_backend(answer)
    with caplog.at_level(logging.WARNING):
        read = dialogue.read_source(src, lines, backend)
    assert read.failed == 1 and read.pieces == 3
    assert [(s["first_line"], s["last_line"]) for s in read.stretches] == [(2, 4)]
    assert "1 of 3 dialogue pieces got no usable reply" in caplog.text
    assert not dialogue.read_source(src, lines, backend).cached  # asked again
    assert len(prompts) == 6


def test_invalid_entries_are_counted_and_do_not_stop_the_reading(tmp_path):
    src = write(tmp_path / "a.mp4")
    backend, _ = fake_backend(lambda _p: {"moments": [moment(1, 2), moment(500, 600)]})
    read = dialogue.read_source(src, talk_lines(10), backend)
    assert len(read.stretches) == 1 and read.dropped == 1


def test_read_all_gives_events_per_source_and_frees_the_model(tmp_path):
    a, b = write(tmp_path / "a.mp4"), write(tmp_path / "b.mp4", "yy")
    freed: list[bool] = []
    backend, _ = fake_backend(lambda _p: {"moments": [moment(0, 1)]})
    backend.close = lambda: freed.append(True)
    segs = [{"start": 0.0, "end": 5.0, "text": "one"}, {"start": 6.0, "end": 9.0, "text": "two"}]
    got = dialogue.read_all({str(a): segs, str(b): segs, "none.mp4": []}, None, backend)
    assert [e["source"] for e in got] == [str(a), str(b)]
    assert all(e["kind"] == "text" and e["type"] == "dialogue" for e in got)
    assert freed == [True]


# --------------------------------------------------------------------------- the backend
class Net:
    """Every ``requests.get`` and ``post``: a pulled Ollama model (``ollama``), the AI Chat
    provider's Anthropic API (``hosted_text`` is the model's reply); anything else fails the
    test. ``calls`` are the ``(method, url)`` that went out."""

    def __init__(self, monkeypatch, *, ollama: bool = False, hosted_text: str = '{"moments": []}'):
        self.ollama, self.hosted_text = ollama, hosted_text
        self.pulled = ["qwen3.5:9b"]  # the models /api/tags lists
        self.calls: list[tuple[str, str]] = []
        self.bodies: list[dict[str, Any]] = []
        monkeypatch.setattr(requests, "get", lambda url, **kw: self._go("GET", url, kw))
        monkeypatch.setattr(requests, "post", lambda url, **kw: self._go("POST", url, kw))

    def _go(self, method: str, url: str, kw: dict[str, Any]):
        self.calls.append((method, url))
        self.bodies.append(kw.get("json") or {})
        if "11434" in url:
            if not self.ollama:
                raise requests.ConnectionError("connection refused")
            if method == "GET" and url.endswith("/api/tags"):
                return Resp(
                    {"models": [{"name": n, "digest": "0123456789abcdef"} for n in self.pulled]}
                )
            if method == "POST" and url.endswith("/api/generate"):
                return Resp({"response": '{"moments": [{"first_line": 0}]}'})
        if "api.anthropic.com" in url:
            if method == "GET" and "/v1/models/" in url:
                return Resp({"id": "m"})
            if method == "POST" and url.endswith("/v1/messages"):
                body = {
                    "content": [{"type": "text", "text": self.hosted_text}],
                    "stop_reason": "end_turn",
                }
                return Resp(body)
        raise AssertionError(f"unexpected request {method} {url}")

    def to(self, host: str) -> list[tuple[str, str]]:
        return [c for c in self.calls if host in c[1]]


class Resp:
    def __init__(self, body: Any, status: int = 200) -> None:
        self.body, self.status_code = body, status
        self.is_redirect, self.headers = False, {}
        self.text = json.dumps(body)

    def json(self) -> Any:
        return self.body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


ANTHROPIC = "anthropic@api.anthropic.com"


@pytest.fixture
def hosted_chat(home, monkeypatch):
    """The AI Chat model on Anthropic's API, with an API key; consent is the test's to set."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)

    def configure(consent: list[str]) -> None:
        set_config(home, assistant={"provider": "anthropic", "consent": consent})

    return configure


def test_the_consent_prefix_of_the_assistant_is_v3():
    assert providers.ASSISTANT_CONSENT_PREFIX == "assistant-v3:"


def test_a_hosted_model_with_v3_consent_reads_the_transcript(hosted_chat, monkeypatch):
    hosted_chat([f"assistant-v3:{ANTHROPIC}"])
    net = Net(monkeypatch, hosted_text='Here you go: {"moments": [{"first_line": 1}]} done')
    with jsonout.run("analyze") as res:
        choice = dialogue.choose_backend()
    assert choice.backend is not None and choice.backend.name == "hosted"
    assert choice.backend.workers == dialogue.HOSTED_WORKERS
    assert ANTHROPIC in choice.backend.key and choice.backend.model in choice.backend.key
    assert choice.backend.ask("read this") == {"moments": [{"first_line": 1}]}  # JSON out of text
    assert res.warnings == [] and net.to("11434") == []  # no local model was looked for
    sent = net.bodies[-1]["messages"][0]["content"][0]["text"]  # Anthropic takes no schema
    assert sent.startswith("read this") and "Reply with JSON only" in sent


@pytest.mark.parametrize(
    "consent",
    [
        [],
        ["assistant-v2:" + ANTHROPIC],  # consent from before whole transcripts were sent
        [ANTHROPIC],
        ["vision-v2:" + ANTHROPIC],
    ],
    ids=["none", "v2", "unprefixed", "vision"],
)
def test_a_hosted_model_without_v3_consent_and_no_local_model_skips_the_step(
    hosted_chat, monkeypatch, caplog, consent
):
    hosted_chat(consent)
    net = Net(monkeypatch, ollama=False)
    with caplog.at_level(logging.INFO), jsonout.run("analyze") as res:
        choice = dialogue.choose_backend()
        dialogue.choose_backend()  # a second look in the same run
    assert choice.backend is None and "consent" in choice.reason
    assert warning_codes(res.warnings) == ["chat_consent_needed"]  # once per run
    msg = res.warnings[0]["message"]
    assert f"assistant-v3:{ANTHROPIC}" in msg and "whole transcripts" in msg
    assert net.to("api.anthropic.com") == []  # nothing went to the hosted model
    assert caplog.text.count("needs your consent") == 1


def test_a_hosted_model_without_consent_gives_way_to_a_local_ollama(hosted_chat, monkeypatch):
    hosted_chat(["assistant-v2:" + ANTHROPIC])
    net = Net(monkeypatch, ollama=True)
    with jsonout.run("analyze") as res:
        choice = dialogue.choose_backend()
    backend = choice.backend
    assert backend is not None and backend.name == "local" and backend.model == "qwen3.5:9b"
    assert "0123456789ab" in backend.key  # the model's digest keys the cache
    assert res.warnings == []  # the user is not asked for anything
    assert backend.ask("read this") == {"moments": [{"first_line": 0}]}
    assert net.to("api.anthropic.com") == []


def test_the_default_local_ollama_is_the_backend_when_ai_chat_is_the_default(home, monkeypatch):
    net = Net(monkeypatch, ollama=True)
    choice = dialogue.choose_backend()
    assert choice.backend is not None and choice.backend.name == "local"
    assert net.to("api.anthropic.com") == []


def test_no_model_at_all_skips_the_step_with_a_note_and_no_warning(home, monkeypatch):
    Net(monkeypatch, ollama=False)
    with jsonout.run("analyze") as res:
        choice = dialogue.choose_backend()
    assert choice.backend is None and choice.reason
    assert "consent" not in choice.reason and res.warnings == []


def test_a_bad_ollama_port_means_no_backend_not_a_crash(home, monkeypatch, caplog):
    """#825: OLLAMA_HOST with an invalid port: no local model, no exception."""
    Net(monkeypatch, ollama=False)
    monkeypatch.setenv("OLLAMA_HOST", "myhost:99999")
    with caplog.at_level(logging.WARNING), jsonout.run("analyze"):
        choice = dialogue.choose_backend()
    assert choice.backend is None and choice.reason
    assert "invalid port" in caplog.text


def test_a_hosted_model_that_cannot_be_asked_leaves_the_step_to_a_local_model(
    home, monkeypatch, hosted_chat
):
    hosted_chat([f"assistant-v3:{ANTHROPIC}"])
    monkeypatch.delenv("ANTHROPIC_API_KEY")  # consent but no key
    Net(monkeypatch, ollama=True)
    choice = dialogue.choose_backend()
    assert choice.backend is not None and choice.backend.name == "local"


def test_a_refusal_or_a_failed_request_is_no_reply(hosted_chat, monkeypatch):
    hosted_chat([f"assistant-v3:{ANTHROPIC}"])
    net = Net(monkeypatch, hosted_text="I'm sorry, I can't help with that.")
    backend = dialogue.choose_backend().backend
    assert backend is not None and backend.ask("read this") is None  # not JSON

    def down(url, **kw):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(requests, "post", down)
    assert backend.ask("read this") is None  # an unreachable model is logged, not raised
    assert net.to("api.anthropic.com")  # (it was asked first)


def _skips(style_name: str, transcripts, skip=()) -> tuple[Any, progress.Run]:
    steps = progress.Run(progress.ANALYZE_STEPS, active=False)
    got = analyze._dialogue_backend(get_style(style_name), skip, steps, transcripts, None, None)
    return got, steps


@pytest.mark.parametrize("style", ["funny", "action", "sport", "commentary", "action-cam"])
def test_a_style_without_an_open_list_skips_the_step_and_asks_no_model(style, monkeypatch):
    net = Net(monkeypatch)
    backend, steps = _skips(style, {"a.mp4": [{"start": 0, "end": 1, "text": "hi"}]})
    assert backend is None and "open-list" in steps.reasons["dialogue"]
    assert net.calls == []


def test_a_source_without_speech_skips_the_step(monkeypatch):
    net = Net(monkeypatch)
    backend, steps = _skips("gaming", {"a.mp4": []})
    assert backend is None and steps.reasons["dialogue"] == "no speech to read"
    assert net.calls == []


def test_a_source_with_only_marker_holders_skips_the_step(monkeypatch):
    net = Net(monkeypatch)
    holder = {"start": 0, "end": 1, "text": "", "words": [], "markers": [{"text": "[Music]"}]}
    backend, steps = _skips("gaming", {"a.mp4": [holder]})
    assert backend is None and steps.reasons["dialogue"] == "no speech to read"
    assert net.calls == []


def test_a_gaming_run_picks_its_backend_once(monkeypatch):
    Net(monkeypatch, ollama=True)
    backend, steps = _skips("gaming", {"a.mp4": [{"start": 0, "end": 1, "text": "hi"}]})
    assert backend is not None and backend.name == "local" and "dialogue" not in steps.reasons


def test_dialogue_is_a_step_analyze_may_skip():
    assert "dialogue" in progress.SKIPPABLE and "dialogue" in progress.ANALYZE_STEPS
    backend, _ = _skips("gaming", {"a.mp4": [{"start": 0, "end": 1, "text": "hi"}]}, ("dialogue",))
    assert backend is None


# --------------------------------------------------------------------------- the events
def dialogue_event(start, end, conf=0.8, kind="funny", what="a bit", src=SRC) -> dict[str, Any]:
    (ev,) = dialogue.to_events(
        src, [stretch(start, end, rating=round(conf * 10), what=what)], None, "hosted", "m"
    )
    return ev | {"kind": "text"}


def test_a_stretch_is_a_text_event_of_type_dialogue():
    s = stretch(12.34567, 80.0, rating=8, what="a bit", first=3, last=9)
    (ev,) = dialogue.to_events(SRC, [s], None, "hosted", "model-x")
    assert (ev["kind"], ev["type"]) == ("text", "dialogue")
    assert (ev["source"], ev["start"], ev["end"]) == (SRC, 12.346, 80.0)
    assert (ev["label"], ev["description"]) == ("funny", "a bit")
    assert ev["details"] == {
        "rating": 8, "backend": "hosted", "model": "model-x", "first_line": 3, "last_line": 9,
    }  # fmt: skip
    assert report.is_dialogue(ev) and not report.is_dialogue({"kind": "text", "type": "other"})


def test_a_stretch_event_carries_its_title_and_keeps_its_kind_as_the_label():
    titled = {**stretch(0.0, 80.0), "title": "Roommate Burnt Dinner", "kind": "story"}
    plain = {**stretch(100.0, 180.0), "title": ""}
    old = stretch(200.0, 280.0)  # a stretch with no title at all
    a, b, c = dialogue.to_events(SRC, [titled, plain, old], None, "local", "m")
    assert (a["label"], a["title"]) == ("story", "Roommate Burnt Dinner")
    assert (b["label"], b["title"]) == ("funny", "")
    assert (c["label"], c["title"]) == ("funny", "")
    assert 'DIALOGUE story "Roommate Burnt Dinner" (' in report._event_line("a.mp4", a)
    assert "DIALOGUE funny (" in report._event_line("a.mp4", b)


def test_dialogue_events_are_written_with_the_others_and_a_rerun_replaces_them(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    audio = {
        "source": SRC, "start": 5.0, "end": 6.0, "kind": "audio", "label": "Laughter",
        "type": "sound", "confidence": 0.7,
    }  # fmt: skip
    from cutpoint.events.report import write_outputs  # noqa: PLC0415

    write_outputs(proj, [SRC], [audio], [], [], {"layers": ["audio"]}, {}, {})
    first = [dialogue_event(10, 50, what="first bit"), dialogue_event(100, 130, what="second bit")]
    assert report.replace_dialogue(proj, first, [SRC]) is True
    doc = json.loads((proj / "events.json").read_text(encoding="utf-8"))
    assert [(e["kind"], e["type"]) for e in doc["events"] if e["kind"] == "text"] == [
        ("text", "dialogue")
    ] * 2
    assert any(e["kind"] == "audio" for e in doc["events"])  # the others stay
    assert doc["stats"]["event_counts"]["text:dialogue"] == 2

    assert report.replace_dialogue(proj, [dialogue_event(300, 340, what="third bit")], [SRC])
    doc = json.loads((proj / "events.json").read_text(encoding="utf-8"))
    talk = [e for e in doc["events"] if report.is_dialogue(e)]
    assert [e["description"] for e in talk] == ["third bit"]  # replaced, not added to
    assert sum(e["kind"] == "audio" for e in doc["events"]) == 1
    text = (proj / "events.txt").read_text(encoding="utf-8")
    assert text.count("DIALOGUE") == 2  # the legend line and the one stretch
    assert "DIALOGUE funny (0.80, 40s): third bit" in text
    assert "first bit" not in text


def test_replacing_with_nothing_over_nothing_writes_nothing(tmp_path):
    assert report.replace_dialogue(tmp_path, [], [SRC]) is False
    assert not (tmp_path / "events.json").exists()


def test_replacing_with_nothing_clears_the_earlier_dialogue(tmp_path):
    report.replace_dialogue(tmp_path, [dialogue_event(10, 50)], [SRC])
    assert report.replace_dialogue(tmp_path, [], [SRC]) is True
    doc = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    assert not [e for e in doc["events"] if report.is_dialogue(e)]


def test_a_run_that_reads_no_dialogue_keeps_the_earlier_stretches_of_its_sources(tmp_path):
    report.replace_dialogue(
        tmp_path, [dialogue_event(10, 50), dialogue_event(10, 50, src="b.mp4")], [SRC, "b.mp4"]
    )
    kept = report.kept_dialogue(tmp_path, [SRC])
    assert [e["source"] for e in kept] == [SRC]
    assert report.kept_dialogue(tmp_path / "nowhere", [SRC]) == []


# --------------------------------------------------------------------------- the analyze step
SPEECH = [
    {"start": 1.0, "end": 4.0, "text": "something was said"},
    {"start": 5.0, "end": 9.0, "text": "and then something else"},
]


def with_model(monkeypatch, reply: dict[str, Any]) -> list[str]:
    """``choose_backend`` answers with a text model that always replies ``reply``."""
    backend, prompts = fake_backend(lambda _p: reply)
    monkeypatch.setattr(dialogue, "choose_backend", lambda *a, **k: dialogue.Choice(backend))
    return prompts


def test_analyze_without_event_layers_reads_the_dialogue_and_writes_the_events(
    run,  # noqa: F811
    capsys,
    monkeypatch,
):
    run.segments = SPEECH
    prompts = with_model(monkeypatch, {"moments": [moment(0, 1, rating=9, what="a bit")]})
    run.go("gaming")
    assert len(prompts) == 1 and "0: something was said" in prompts[0]
    doc = json.loads((run.proj / "events.json").read_text(encoding="utf-8"))
    (ev,) = [e for e in doc["events"] if report.is_dialogue(e)]
    assert (ev["start"], ev["end"], ev["confidence"]) == (1.0, 9.0, 0.9)
    assert "DIALOGUE funny (0.90, 8s): a bit" in (run.proj / "events.txt").read_text(
        encoding="utf-8"
    )
    states = {g["step"]: g["state"] for g in progress_lines(capsys)}
    assert states["dialogue"] == "done"


def test_analyze_hands_the_reader_to_the_event_layers_with_their_audio_events(
    run,  # noqa: F811
    monkeypatch,
):
    run.segments = SPEECH
    run.plan = analyze.LayerPlan(layers={"reactions": ("audio",)})
    with_model(monkeypatch, {"moments": [moment(0, 1)]})
    run.go("gaming")
    ((_, _, kw),) = run.events_calls
    assert callable(kw["dialogue"])
    layer1 = {str(run.video): [{"kind": "audio", "t": 2.0, "label": "Laughter", "confidence": 0.5}]}
    (ev,) = kw["dialogue"](layer1)
    assert report.is_dialogue(ev) and ev["source"] == str(run.video)


def test_analyze_of_a_style_without_an_open_list_reads_no_dialogue(run, capsys, monkeypatch):  # noqa: F811
    run.segments = SPEECH
    prompts = with_model(monkeypatch, {"moments": [moment(0, 1)]})
    run.go("funny")
    got = {g["step"]: g for g in progress_lines(capsys)}
    assert got["dialogue"]["state"] == "skipped" and "open-list" in got["dialogue"]["detail"]
    assert prompts == [] and not (run.proj / "events.json").exists()


# --------------------------------------------------------------------------- nominating
def talk_event(start, end, conf=0.8, kind="funny") -> dict[str, Any]:
    return {
        "source": SRC, "start": start, "end": end, "kind": "text", "type": "dialogue",
        "confidence": conf, "funny": None, "hype": None, "label": kind, "description": "a bit",
    }  # fmt: skip


def test_a_dialogue_stretch_is_a_seed_that_keeps_its_whole_span():
    (seed,) = scoring.seeds([talk_event(100, 400)], DEFAULT_EVENT_TYPES, STYLE)
    assert seed.name == "dialogue-funny" and seed.whole is True
    assert (seed.start, seed.end, seed.hold, seed.tail_s) == (100, 400, 300, 0.0)
    (kill,) = scoring.seeds([feed(10)], DEFAULT_EVENT_TYPES, STYLE)
    assert kill.whole is False


def test_a_dialogue_stretch_nominates_a_candidate_longer_than_the_style_s_max_len(tmp_path):
    style = get_style("gaming")
    hl = highlights([own("h001", 20.0, 60.0)])
    assert scoring.nominate(hl, events_of([talk_event(100.0, 400.0)]), tmp_path, [grid()]) == 1
    (nom,) = nominees(hl)
    assert nom["signals"]["events"]["nominated"] == ["dialogue-funny"]
    assert nom["end"] - nom["start"] > 10 * style.max_len
    assert nom["start"] == pytest.approx(100.0 - analyze.DIALOGUE_PRE_ROLL_S)
    assert nom["end"] == pytest.approx(400.0)  # not cut, not extended to a sentence
    assert nom["duration"] == pytest.approx(nom["end"] - nom["start"], abs=0.01)


def titled_event(start, end, title="Grandpa's Wobbly Treehouse", conf=0.8, **kw):
    return {**talk_event(start, end, conf=conf), "title": title, **kw}


def test_a_titled_stretch_nominates_a_candidate_with_the_title_and_its_slug_as_label(tmp_path):
    hl = highlights([own("h001", 20.0, 60.0)])
    assert scoring.nominate(hl, events_of([titled_event(100.0, 400.0)]), tmp_path, [grid()]) == 1
    (nom,) = nominees(hl)
    assert nom["dialogue_title"] == "Grandpa's Wobbly Treehouse"
    assert nom["label"] == "grandpas-wobbly-treehouse"  # the file name part
    assert nom["signals"]["events"]["nominated"] == ["dialogue-funny"]  # the kind, as before
    assert scoring.is_long_dialogue(nom)  # still a whole dialogue nominee


def test_an_untitled_stretch_keeps_the_dialogue_kind_label(tmp_path):
    hl = highlights([own("h001", 20.0, 60.0)])
    ev = {**talk_event(100.0, 400.0, kind="out_of_pocket"), "title": ""}
    scoring.nominate(hl, events_of([ev]), tmp_path, [grid()])
    (nom,) = nominees(hl)
    assert nom["label"] == "dialogue-out-of-pocket" and "dialogue_title" not in nom
    old = talk_event(100.0, 400.0)  # events.json of an earlier version: no title field
    hl = highlights([own("h001", 20.0, 60.0)])
    scoring.nominate(hl, events_of([old]), tmp_path, [grid()])
    assert nominees(hl)[0]["label"] == "dialogue-funny"


def test_a_widened_candidate_takes_the_title_of_the_stretch_and_gives_it_back(tmp_path):
    hl = highlights([own("h001", 150.0, 60.0, end=160.0)])
    before = hl["candidates"][0]["label"]
    events = events_of([titled_event(100.0, 180.0)])
    assert scoring.nominate(hl, events, tmp_path, [grid()]) == 0
    (cand,) = hl["candidates"]
    assert (cand["start"], cand["end"]) == (100.0, 180.0)
    assert (cand["dialogue_title"], cand["label"]) == (
        "Grandpa's Wobbly Treehouse",
        "grandpas-wobbly-treehouse",
    )
    scoring.nominate(hl, events, tmp_path, [grid()])  # again: the same
    assert (hl["candidates"][0]["label"], hl["candidates"][0]["widened_from"]["label"]) == (
        "grandpas-wobbly-treehouse",
        before,
    )
    scoring.nominate(hl, events_of([]), tmp_path, [grid()])  # the stretch is gone
    (cand,) = hl["candidates"]
    assert cand["label"] == before and "dialogue_title" not in cand
    assert (cand["start"], cand["end"]) == (150.0, 160.0)


def test_several_stretches_widening_one_candidate_give_the_most_confident_title(tmp_path):
    hl = highlights([own("h001", 150.0, 60.0, end=170.0)])
    events = events_of(
        [
            titled_event(120.0, 160.0, "Low Confidence Bit", conf=0.3),
            titled_event(150.0, 200.0, "High Confidence Bit", conf=0.9),
            titled_event(165.0, 190.0, "Middle Bit", conf=0.5),
        ]
    )
    scoring.nominate(hl, events, tmp_path, [grid()])
    (cand,) = hl["candidates"]
    assert (cand["dialogue_title"], cand["label"]) == ("High Confidence Bit", "high-confidence-bit")


def merge_hl() -> dict[str, Any]:
    a, b = own("h001", 10.0, 60.0, end=20.0), own("h002", 40.0, 50.0, end=50.0)
    return highlights([{**a, "label": "kill-a", "transcript": "text a"}, {**b, "label": "kill-b"}])


def test_a_widened_candidate_merged_into_another_gets_its_label_and_text_back(tmp_path):
    hl = merge_hl()
    first = titled_event(5.0, 30.0, "First Bit Title", conf=0.9, description="first")
    second = titled_event(25.0, 45.0, "Second Bit Title", conf=0.5, description="second")
    snippet = lambda _s, a, b: (f"said {a:g}-{b:g}", None)  # noqa: E731
    by_src = {SRC: [first, second]}
    # one pass over the first stretch widens A; then B is merged into A by the second
    scoring.widen_for_dialogue(hl["candidates"], {SRC: [first]}, None, snippet)
    a = hl["candidates"][0]
    assert (a["dialogue_title"], a["label"]) == ("First Bit Title", "first-bit-title")
    hl2 = merge_hl()
    scoring.widen_for_dialogue(hl2["candidates"], by_src, None, snippet)
    scoring.restore_widened(hl2["candidates"])
    by_id = {c["id"]: c for c in hl2["candidates"]}
    assert sorted(by_id) == ["h001", "h002"]
    assert by_id["h001"]["label"] == "kill-a" and "dialogue_title" not in by_id["h001"]
    assert by_id["h001"]["transcript"] == "text a" and (
        by_id["h001"]["start"],
        by_id["h001"]["end"],
    ) == (10.0, 20.0)
    assert by_id["h002"]["label"] == "kill-b" and "dialogue_title" not in by_id["h002"]


def test_a_candidate_widened_in_one_pass_and_merged_in_the_next_is_whole_again(tmp_path):
    hl = merge_hl()
    first = titled_event(5.0, 30.0, "First Bit Title", conf=0.9)
    second = titled_event(25.0, 45.0, "Second Bit Title", conf=0.5)
    for c in hl["candidates"]:
        c["score"] = 70.0 if c["id"] == "h002" else 60.0  # B outscores A: A is merged into B
    scoring.widen_for_dialogue(hl["candidates"], {SRC: [first]}, None, None)  # A widened (5-30)
    a = next(c for c in hl["candidates"] if c["id"] == "h001")
    assert a["dialogue_title"] == "First Bit Title"
    scoring.widen_for_dialogue(hl["candidates"], {SRC: [second]}, None, None)
    (b,) = hl["candidates"]
    assert b["id"] == "h002"
    (absorbed,) = b["widened_from"]["absorbed"]
    assert absorbed["id"] == "h001" and absorbed["label"] == "kill-a"
    assert "dialogue_title" not in absorbed and (absorbed["start"], absorbed["end"]) == (10.0, 20.0)
    assert absorbed["transcript"] == "text a"
    # its title was the more confident one, so B carries it
    assert (b["dialogue_title"], b["label"]) == ("First Bit Title", "first-bit-title")
    scoring.restore_widened(hl["candidates"])
    assert sorted(c["id"] for c in hl["candidates"]) == ["h001", "h002"]
    assert all("dialogue_title" not in c for c in hl["candidates"])
    assert {c["id"]: c["label"] for c in hl["candidates"]} == {"h001": "kill-a", "h002": "kill-b"}


def test_a_stretch_without_a_title_does_not_take_the_confidence_of_the_title(tmp_path):
    hl = highlights([own("h001", 150.0, 60.0, end=170.0)])
    events = events_of(
        [titled_event(120.0, 160.0, "Kept Title", conf=0.3), talk_event(165.0, 190.0, conf=0.9)]
    )
    scoring.nominate(hl, events, tmp_path, [grid()])
    (cand,) = hl["candidates"]
    assert (cand["dialogue_title"], cand["widened_from"]["title_conf"]) == ("Kept Title", 0.3)


def test_a_widening_stretch_without_a_title_leaves_the_label(tmp_path):
    hl = highlights([own("h001", 150.0, 60.0, end=160.0)])
    before = hl["candidates"][0]["label"]
    scoring.nominate(hl, events_of([talk_event(100.0, 180.0)]), tmp_path, [grid()])
    (cand,) = hl["candidates"]
    assert cand["label"] == before and "dialogue_title" not in cand


def test_a_renamed_widened_candidate_keeps_its_new_label_when_widening_is_undone(tmp_path):
    hl = highlights([own("h001", 150.0, 60.0, end=160.0)])
    scoring.nominate(hl, events_of([titled_event(100.0, 180.0)]), tmp_path, [grid()])
    hl["candidates"][0]["label"] = "rated-by-vision"  # the vision rating renamed it
    scoring.nominate(hl, events_of([]), tmp_path, [grid()])
    assert hl["candidates"][0]["label"] == "rated-by-vision"


@pytest.mark.parametrize(
    ("title", "want"),
    [
        ("Mike's Expensive Car Fail", "mikes-expensive-car-fail"),
        ("Let's Go", "lets-go"),
        ("Don't Stop Believing", "dont-stop-believing"),
        ("Mike\u2019s Car", "mikes-car"),
        ("Grandpa Builds A Wobbly Treehouse Again", "grandpa-builds-a-wobbly"),  # no half word
        ("Short", "short"),
        ("x" * 50, "x" * 30),  # one long word is cut
        ("", "moment"),
        ("\u30c9\u30e9\u30de", "moment"),  # no Latin letters
    ],
)
def test_a_title_slug_drops_apostrophes_and_is_cut_at_a_word(title, want):
    assert title_slug(title) == want


def test_a_title_slug_takes_the_default_when_nothing_is_left():
    assert title_slug("\u30c9\u30e9\u30de", default="") == ""
    assert title_slug("\u30c9\u30e9\u30de", default="kept") == "kept"


def test_a_title_with_no_latin_letters_keeps_the_label_but_shows_as_the_title(tmp_path):
    kana = "\u30c9\u30e9\u30de"
    hl = highlights([own("h001", 20.0, 60.0)])
    scoring.nominate(hl, events_of([titled_event(100.0, 400.0, kana)]), tmp_path, [grid()])
    (nom,) = nominees(hl)
    assert nom["dialogue_title"] == kana and nom["label"] == "dialogue-funny"  # not "moment"
    hl = highlights([own("h001", 150.0, 60.0, end=160.0)])
    before = hl["candidates"][0]["label"]
    scoring.nominate(hl, events_of([titled_event(100.0, 180.0, kana)]), tmp_path, [grid()])
    (cand,) = hl["candidates"]
    assert cand["dialogue_title"] == kana and cand["label"] == before
    scoring.nominate(hl, events_of([]), tmp_path, [grid()])
    assert hl["candidates"][0]["label"] == before and "dialogue_title" not in hl["candidates"][0]


def test_slugify_itself_is_unchanged():
    assert slugify("Mike's Sister-in-Law", 8) == "mike-s-s"


def test_a_hostile_title_gives_a_safe_clip_file_name():
    label = title_slug(r'../C:\Mike: "Is" <Cooking>?')
    assert label and not set(label) & set(r'/\:*?"<>|. ') and label == label.lower()


def test_a_kill_in_the_same_run_is_still_cut_to_max_len(tmp_path):
    style = get_style("gaming")
    hl = highlights([own("h001", 20.0, 60.0)])
    scoring.nominate(hl, events_of([feed(100.0)]), tmp_path, [grid()])
    (nom,) = nominees(hl)
    assert nom["end"] - nom["start"] <= style.max_len


def test_dialogue_stretches_close_together_are_not_merged(tmp_path):
    hl = highlights([own("h001", 20.0, 60.0)])
    events = events_of([talk_event(100.0, 200.0), talk_event(203.0, 300.0)])
    assert scoring.nominate(hl, events, tmp_path, [grid()]) == 2
    first, second = sorted(nominees(hl), key=lambda c: c["start"])
    assert first["end"] <= second["start"]
    assert second["end"] == pytest.approx(300.0)


def test_a_stretch_with_a_candidate_inside_it_widens_that_candidate_and_nominates_nothing(tmp_path):
    inside = own("h001", 150.0, 60.0, end=160.0)
    hl = highlights([inside])
    events = events_of([talk_event(100.0, 180.0, conf=1.0)])
    assert scoring.nominate(hl, events, tmp_path, [grid()]) == 0
    assert nominees(hl) == []
    (cand,) = hl["candidates"]
    assert cand["id"] == "h001" and (cand["start"], cand["end"]) == (100.0, 180.0)
    assert cand["duration"] == 80.0 and cand["score"] == 60.0  # its own signals stay
    boost, evidence = scoring.event_boost(cand, events["events"], style=STYLE)
    assert boost == pytest.approx(scoring.DIALOGUE_POINTS) and evidence["dialogue"] == 1.0


@pytest.mark.parametrize("conf", [0.2, 0.5, 0.8, 1.0])
def test_the_dialogue_boost_scales_with_confidence_and_stays_below_a_kill(conf):
    cand = {"start": 100.0, "end": 108.0}
    boost, ev = scoring.event_boost(cand, [talk_event(90.0, 200.0, conf=conf)], style=STYLE)
    assert boost == pytest.approx(scoring.DIALOGUE_POINTS * conf)
    assert ev["dialogue"] == pytest.approx(conf)
    kill, _ = scoring.event_boost(cand, [feed(101.0)], style=STYLE)
    assert 0 < boost < kill


def test_a_candidate_a_stretch_does_not_touch_has_no_dialogue_evidence():
    boost, ev = scoring.event_boost(
        {"start": 10.0, "end": 18.0}, [talk_event(500.0, 600.0)], style=STYLE
    )
    assert boost == 0.0 and "dialogue" not in ev


def test_the_dialogue_boost_is_added_to_the_laughter_in_the_stretch():
    cand = {"start": 100.0, "end": 108.0}
    laugh, _ = scoring.event_boost(cand, [sound(102.0, confidence=0.1)], style=STYLE)
    both, _ = scoring.event_boost(
        cand, [sound(102.0, confidence=0.1), talk_event(90.0, 200.0, conf=1.0)], style=STYLE
    )
    assert both == pytest.approx(laugh + scoring.DIALOGUE_POINTS)


# --------------------------------------------------------------------------- exporting
def long_dialogue_project(tmp_path: Path, seconds: float = 240.0) -> Path:
    """A gaming project whose only candidate is a dialogue stretch of ``seconds``."""
    cand = own("h001", 100.0, 60.0, end=100.0 + seconds)
    cand["duration"] = seconds
    cand["signals"]["events"] = {"nominated": ["dialogue-funny"], "boost": 3.0}
    proj = tmp_path / "proj"
    write(proj / "highlights.json", json.dumps(highlights([cand])))
    return proj


def test_select_moments_does_not_trim_a_long_dialogue_candidate(tmp_path):
    proj = long_dialogue_project(tmp_path, 240.0)
    (chosen,), _ = planning.select_moments(proj, top=1)
    assert chosen["end"] - chosen["start"] == 240.0
    (chosen,), _ = planning.select_moments(proj, ids=["h001"])
    assert chosen["end"] - chosen["start"] == 240.0


def test_max_total_leaves_out_a_candidate_longer_than_the_total_rather_than_trimming_it(tmp_path):
    """Current behaviour: ``--max-total`` fits whole candidates, so a 4-minute dialogue stretch is
    dropped (and with nothing else, the selection is empty) under a 60 s cap."""
    proj = long_dialogue_project(tmp_path, 240.0)
    with pytest.raises(ValueError, match="No moments selected"):
        planning.select_moments(proj, max_total=60.0)
    (chosen,), _ = planning.select_moments(proj, max_total=300.0)
    assert chosen["id"] == "h001"


def test_plan_clips_exports_a_long_dialogue_candidate_whole(plan_env, speech_env):  # noqa: F811
    from conftest import media_info  # noqa: PLC0415
    from cutpoint import boundaries, clips  # noqa: PLC0415
    from cutpoint.events.tails import EventTailSettings  # noqa: PLC0415
    from cutpoint.tail import TailSettings  # noqa: PLC0415
    from test_planning import SRC_A, _moment  # noqa: PLC0415

    project, _ = speech_env  # plan_env faked the source reads: every source is readable
    moment_ = _moment("h001", SRC_A, 20.0, 260.0, label="dialogue-funny")
    (plan,) = clips.plan_clips(
        [moment_],
        project,
        io_paths={SRC_A: Path("a.mp4")},
        infos={SRC_A: media_info(duration=600.0)},
        tail=TailSettings(enabled=False),
        events=None,
        event_settings=EventTailSettings(enabled=False),
        speech=boundaries.SpeechSettings(enabled=False),
    )
    assert (plan.start, plan.end) == (20.0, 260.0)  # 240 s: nothing in the export caps max_len


# --------------------------------------------------------------------------- widening (F6)
def widen_hl(*cands):
    return highlights([*cands])


def test_a_widened_candidate_is_not_cut_to_max_len_and_keeps_its_id_and_signals(tmp_path):
    style = get_style("gaming")
    hl = widen_hl(own("h001", 150.0, 60.0, end=158.0))
    scoring.nominate(hl, events_of([talk_event(100.0, 180.0)]), tmp_path, [grid()])
    (cand,) = hl["candidates"]
    assert cand["end"] - cand["start"] == 80.0 > style.max_len
    assert (cand["widened_from"]["start"], cand["widened_from"]["end"]) == (150.0, 158.0)


def test_a_candidate_longer_than_the_stretch_is_the_union_of_both(tmp_path):
    hl = widen_hl(own("h001", 90.0, 60.0, end=130.0))
    scoring.nominate(hl, events_of([talk_event(100.0, 170.0)]), tmp_path, [grid()])
    (cand,) = hl["candidates"]
    assert (cand["start"], cand["end"]) == (90.0, 170.0)


def test_candidates_a_stretch_overlaps_become_the_best_scoring_one(tmp_path):
    hl = widen_hl(
        own("h001", 110.0, 30.0, end=118.0),
        own("h002", 130.0, 70.0, end=138.0),
        own("h003", 160.0, 50.0, end=180.0),
        own("h004", 700.0, 90.0, end=708.0),  # nothing to do with it
    )
    scoring.nominate(hl, events_of([talk_event(100.0, 170.0)]), tmp_path, [grid()])
    ids = {c["id"]: c for c in hl["candidates"]}
    assert sorted(ids) == ["h002", "h004"]
    assert (ids["h002"]["start"], ids["h002"]["end"]) == (100.0, 180.0)
    assert (ids["h004"]["start"], ids["h004"]["end"]) == (700.0, 708.0)
    assert [c["id"] for c in ids["h002"]["widened_from"]["absorbed"]] == ["h001", "h003"]


def test_a_stretch_that_no_candidate_overlaps_still_nominates(tmp_path):
    hl = widen_hl(own("h001", 20.0, 60.0))
    assert scoring.nominate(hl, events_of([talk_event(100.0, 400.0)]), tmp_path, [grid()]) == 1
    assert [c["id"] for c in hl["candidates"] if "widened_from" in c] == []


def test_a_union_of_exactly_the_cap_still_widens(tmp_path):
    cap = scoring.DIALOGUE_WIDEN_MAX_S
    hl = widen_hl(own("h001", 100.0, 60.0, end=108.0))
    assert (
        scoring.nominate(hl, events_of([talk_event(100.0, 100.0 + cap)]), tmp_path, [grid()]) == 0
    )
    (cand,) = hl["candidates"]
    assert (cand["start"], cand["end"]) == (100.0, 100.0 + cap)


def test_a_union_over_the_cap_leaves_the_candidate_and_nominates_the_stretch_whole(tmp_path):
    hl = widen_hl(own("h001", 150.0, 60.0, end=158.0))
    events = events_of([talk_event(100.0, 400.0)])  # 300 s
    assert scoring.nominate(hl, events, tmp_path, [grid()]) == 1
    (own_,) = [c for c in hl["candidates"] if not scoring.is_nominated(c)]
    assert (own_["start"], own_["end"]) == (150.0, 158.0) and "widened_from" not in own_
    (nom,) = nominees(hl)
    assert nom["end"] - nom["start"] >= 300.0  # as marked, over h001 (plus the pre-roll)
    assert nom["start"] <= 100.0 and nom["end"] == pytest.approx(400.0)
    assert nom["signals"]["events"]["nominated"] == ["dialogue-funny"]


def test_partial_widening_is_not_done_when_one_of_two_candidates_would_pass_the_cap(tmp_path):
    hl = widen_hl(own("h001", 110.0, 30.0, end=118.0), own("h002", 130.0, 70.0, end=138.0))
    # with h001 alone the union would be 100-170 s; with h002 it is 100-260
    events = events_of([talk_event(100.0, 260.0)])
    scoring.nominate(hl, events, tmp_path, [grid()])
    spans = sorted((c["id"], c["start"], c["end"]) for c in hl["candidates"] if c.get("id"))
    assert spans == [("h001", 110.0, 118.0), ("h002", 130.0, 138.0)]
    assert len(nominees(hl)) == 1


def test_two_candidates_merge_only_when_the_union_is_within_the_cap(tmp_path):
    hl = widen_hl(own("h001", 110.0, 30.0, end=118.0), own("h002", 150.0, 70.0, end=158.0))
    scoring.nominate(hl, events_of([talk_event(100.0, 170.0)]), tmp_path, [grid()])  # 70 s
    (cand,) = hl["candidates"]
    assert cand["id"] == "h002" and (cand["start"], cand["end"]) == (100.0, 170.0)
    assert [c["id"] for c in cand["widened_from"]["absorbed"]] == ["h001"]
    hl = widen_hl(own("h001", 110.0, 30.0, end=118.0), own("h002", 150.0, 70.0, end=158.0))
    scoring.nominate(hl, events_of([talk_event(100.0, 250.0)]), tmp_path, [grid()])  # 150 s
    assert sorted(c["id"] for c in hl["candidates"] if c.get("id")) == ["h001", "h002"]
    assert all("widened_from" not in c for c in hl["candidates"])


def test_a_long_stretch_inside_a_longer_candidate_is_not_nominated_again(tmp_path):
    hl = widen_hl(own("h001", 50.0, 60.0, end=400.0))
    assert scoring.nominate(hl, events_of([talk_event(100.0, 300.0)]), tmp_path, [grid()]) == 0
    (cand,) = hl["candidates"]
    assert (cand["start"], cand["end"]) == (50.0, 400.0)


def test_a_long_stretch_over_a_candidate_gives_the_same_result_when_run_again(tmp_path):
    hl = widen_hl(own("h001", 150.0, 60.0, end=158.0))
    events = events_of([talk_event(100.0, 400.0)])
    scoring.nominate(hl, events, tmp_path, [grid()])
    once = json.loads(json.dumps(hl))
    scoring.nominate(hl, events, tmp_path, [grid()])
    assert json.loads(json.dumps(hl)) == once
    scoring.nominate(hl, events_of([]), tmp_path, [grid()])  # the stretch is gone
    assert nominees(hl) == []
    assert [(c["start"], c["end"]) for c in hl["candidates"]] == [(150.0, 158.0)]


def test_a_stretch_that_stops_fitting_the_cap_is_taken_back_and_nominated_on_a_rerun(tmp_path):
    hl = widen_hl(own("h001", 150.0, 60.0, end=158.0))
    scoring.nominate(hl, events_of([talk_event(100.0, 180.0)]), tmp_path, [grid()])
    assert nominees(hl) == [] and hl["candidates"][0]["end"] == 180.0
    scoring.nominate(hl, events_of([talk_event(100.0, 400.0)]), tmp_path, [grid()])
    assert [(c["start"], c["end"]) for c in hl["candidates"] if not scoring.is_nominated(c)] == [
        (150.0, 158.0)
    ]
    assert len(nominees(hl)) == 1


def test_widening_twice_gives_the_same_candidates(tmp_path):
    hl = widen_hl(own("h001", 120.0, 30.0, end=128.0), own("h002", 150.0, 70.0, end=158.0))
    events = events_of([talk_event(100.0, 180.0)])
    scoring.nominate(hl, events, tmp_path, [grid()])
    once = json.loads(json.dumps(hl))
    scoring.nominate(hl, events, tmp_path, [grid()])
    assert json.loads(json.dumps(hl)) == once


def test_without_the_stretch_the_candidates_return_to_their_own_spans(tmp_path):
    hl = widen_hl(own("h001", 120.0, 30.0, end=128.0), own("h002", 150.0, 70.0, end=158.0))
    scoring.nominate(hl, events_of([talk_event(100.0, 180.0)]), tmp_path, [grid()])
    scoring.nominate(hl, events_of([]), tmp_path, [grid()])
    got = sorted((c["id"], c["start"], c["end"], c["duration"]) for c in hl["candidates"])
    assert got == [("h001", 120.0, 128.0, 8.0), ("h002", 150.0, 158.0, 8.0)]
    assert all("widened_from" not in c for c in hl["candidates"])


def test_a_shorter_stretch_re_widens_from_the_own_span_not_the_widened_one(tmp_path):
    hl = widen_hl(own("h001", 150.0, 60.0, end=158.0))
    scoring.nominate(hl, events_of([talk_event(100.0, 180.0)]), tmp_path, [grid()])
    scoring.nominate(hl, events_of([talk_event(140.0, 200.0)]), tmp_path, [grid()])
    (cand,) = hl["candidates"]
    assert (cand["start"], cand["end"]) == (140.0, 200.0)


def test_a_pool_candidate_is_not_widened_and_still_covers_the_stretch(tmp_path):
    pool = own("v001", 150.0, 0.0, end=158.0) | {"pool": "visual"}
    hl = widen_hl(pool)
    assert scoring.nominate(hl, events_of([talk_event(100.0, 180.0)]), tmp_path, [grid()]) == 0
    assert (pool["start"], pool["end"]) == (150.0, 158.0)


def test_rescoring_a_widened_candidate_adds_the_dialogue_boost(tmp_path):
    from cutpoint import scorer  # noqa: PLC0415

    hl = widen_hl(own("h001", 150.0, 40.0, end=158.0))
    events = events_of([talk_event(100.0, 180.0, conf=1.0)], created="x")
    scorer.rescore_events(hl, events, tmp_path, [grid()])
    (cand,) = hl["candidates"]
    assert cand["score"] == pytest.approx(40.0 + scoring.DIALOGUE_POINTS)
    assert (cand["start"], cand["end"]) == (100.0, 180.0)


# --------------------------------------------------------------------------- style change (F2)
def test_a_dialogue_event_left_by_an_open_list_style_scores_nothing_in_another():
    cand = {"start": 100.0, "end": 108.0}
    event = talk_event(90.0, 200.0, conf=1.0)
    other, ev = scoring.event_boost(cand, [event], style=("funny",))
    assert other == 0.0 and "dialogue" not in ev
    mine, _ = scoring.event_boost(cand, [event], style=("gaming",))
    assert mine == scoring.DIALOGUE_POINTS
    assert scoring.event_boost(cand, [event])[0] == 0.0  # no style: none


# --------------------------------------------------------------------------- an Ollama's consent (F1, F7)
REMOTE = "10.10.10.235"
REMOTE_URL = f"http://{REMOTE}:11434"
REMOTE_KEY = f"assistant-v3:ollama@{REMOTE}:11434"
REMOTE_VISION = f"vision-v2:ollama@{REMOTE}:11434"


def test_a_remote_ollama_as_ai_chat_with_only_vision_consent_gets_no_transcripts(home, monkeypatch):
    set_config(
        home,
        assistant={"provider": "ollama", "url": REMOTE_URL},
        vision_assistant={"consent": [REMOTE_VISION]},
    )
    net = Net(monkeypatch, ollama=False)
    with jsonout.run("analyze") as res:
        choice = dialogue.choose_backend()
    assert choice.backend is None
    assert net.to(REMOTE) == []  # not even a look at its models
    assert warning_codes(res.warnings) == ["chat_consent_needed"]
    assert REMOTE_KEY in res.warnings[0]["message"]


def test_a_remote_ollama_as_ai_chat_gives_way_to_the_local_one_without_asking_it(
    home, monkeypatch, caplog
):
    set_config(home, assistant={"provider": "ollama", "url": REMOTE_URL})
    net = Net(monkeypatch, ollama=True)
    with caplog.at_level(logging.INFO):
        choice = dialogue.choose_backend()
    assert choice.backend is not None and choice.backend.name == "local"
    assert "127.0.0.1" in choice.backend.key and net.to(REMOTE) == []
    assert f"Dialogue reader: the Ollama at {REMOTE}:11434 has no consent; using" in caplog.text
    assert "hosted AI Chat model" not in caplog.text


def test_hosted_chat_without_consent_and_a_remote_ollama_url_with_only_vision_consent(
    hosted_chat, home, monkeypatch
):
    set_config(
        home,
        assistant={"provider": "anthropic", "consent": []},
        vision_assistant={"consent": [REMOTE_VISION]},
    )
    net = Net(monkeypatch, ollama=True)
    with jsonout.run("analyze") as res:
        choice = dialogue.choose_backend(REMOTE_URL, None)
    assert choice.backend is None
    assert net.to(REMOTE) == [] and net.to("api.anthropic.com") == []
    assert warning_codes(res.warnings) == ["chat_consent_needed"]


def test_a_remote_ollama_url_with_only_vision_consent_is_not_used_either(
    hosted_chat, home, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY")  # the agreed hosted model cannot be asked
    set_config(
        home,
        assistant={"provider": "anthropic", "consent": [f"assistant-v3:{ANTHROPIC}"]},
        vision_assistant={"consent": [REMOTE_VISION]},
    )
    net = Net(monkeypatch, ollama=True)
    with jsonout.run("analyze") as res:
        choice = dialogue.choose_backend(REMOTE_URL, None)
    assert choice.backend is None and net.to(REMOTE) == []
    assert warning_codes(res.warnings) == ["chat_consent_needed"]
    assert REMOTE_KEY in res.warnings[0]["message"]


def test_a_remote_ollama_with_the_assistant_consent_is_read_like_a_local_one(home, monkeypatch):
    set_config(home, assistant={"provider": "ollama", "url": REMOTE_URL, "consent": [REMOTE_KEY]})
    net = Net(monkeypatch, ollama=True)
    backend = dialogue.choose_backend().backend
    assert backend is not None and backend.name == "remote" and backend.workers == 1
    assert REMOTE in backend.key and net.to(REMOTE)
    assert backend.ask("read this") == {"moments": [{"first_line": 0}]}
    # ranked by the reactions in it, not by the model's rating
    (ev,) = dialogue.to_events("a.mp4", [stretch(0.0, 60.0, rating=10)], None, "remote", "m")
    assert ev["confidence"] == dialogue.BASELINE_STRENGTH
    backend.close()  # frees the model


# --------------------------------------------------------------------------- the flags win (#805)
@pytest.fixture(autouse=True)
def _fresh_override_log(monkeypatch):
    monkeypatch.setattr(vision, "_override_logged", set())


def test_ollama_flags_win_over_a_consented_hosted_ai_chat_model(hosted_chat, monkeypatch):
    hosted_chat([f"assistant-v3:{ANTHROPIC}"])  # key and consent: it would be the backend
    net = Net(monkeypatch, ollama=True)
    net.pulled = ["tiny:1b"]
    made: list[bool] = []
    real = providers.make
    monkeypatch.setattr(providers, "make", lambda *a, **k: made.append(True) or real(*a, **k))
    with jsonout.run("analyze") as res:
        backend = dialogue.choose_backend(None, "tiny:1b").backend
    assert backend is not None and backend.name == "local" and backend.model == "tiny:1b"
    assert "tiny:1b" in backend.key
    assert net.to("api.anthropic.com") == [] and not made and res.warnings == []


def test_ollama_flags_over_a_hosted_model_log_one_info_line(hosted_chat, monkeypatch, caplog):
    hosted_chat([f"assistant-v3:{ANTHROPIC}"])
    Net(monkeypatch, ollama=True)
    with caplog.at_level(logging.INFO):
        dialogue.choose_backend("http://127.0.0.1:11434", None)
    want = "--ollama-url/--ollama-model given: using the Ollama at 127.0.0.1:11434, not anthropic"
    assert caplog.text.count(want) == 1


def test_ollama_flags_win_over_an_ai_chat_ollama(home, monkeypatch, caplog):
    set_config(home, assistant={"provider": "ollama", "url": REMOTE_URL, "consent": [REMOTE_KEY]})
    net = Net(monkeypatch, ollama=True)
    with caplog.at_level(logging.INFO):
        backend = dialogue.choose_backend("http://127.0.0.1:11434", None).backend
    assert backend is not None and backend.name == "local" and "127.0.0.1" in backend.key
    assert net.to(REMOTE) == []
    want = (
        "--ollama-url/--ollama-model given: using the Ollama at 127.0.0.1:11434, "
        "not the Ollama at 10.10.10.235:11434"
    )
    assert caplog.text.count(want) == 1


def test_ollama_flags_name_a_bare_ai_chat_url_with_its_port(home, monkeypatch, caplog):
    set_config(home, assistant={"provider": "ollama", "url": "myhost", "consent": []})
    Net(monkeypatch, ollama=True)
    with caplog.at_level(logging.INFO):
        dialogue.choose_backend("http://127.0.0.1:11434", None)
    assert "not the Ollama at myhost:11434" in caplog.text


def test_ollama_flags_at_the_ai_chat_ollama_host_log_no_override_line(home, monkeypatch, caplog):
    set_config(home, assistant={"provider": "ollama", "url": REMOTE_URL, "consent": [REMOTE_KEY]})
    Net(monkeypatch, ollama=True)
    with caplog.at_level(logging.INFO):
        dialogue.choose_backend(REMOTE_URL, None)
    assert "--ollama-url/--ollama-model given" not in caplog.text


def test_ollama_flags_with_a_bad_port_in_ollama_host_do_not_raise(home, monkeypatch):
    set_config(home, assistant={"provider": "ollama"})
    monkeypatch.setenv("OLLAMA_HOST", "myhost:99999")
    Net(monkeypatch, ollama=True)
    dialogue.choose_backend("http://127.0.0.1:11434")


def test_a_remote_ollama_url_without_consent_is_skipped_not_handed_to_the_hosted_model(
    hosted_chat, monkeypatch, caplog
):
    hosted_chat([f"assistant-v3:{ANTHROPIC}"])  # the hosted model is agreed to and usable
    net = Net(monkeypatch, ollama=True)
    with caplog.at_level(logging.INFO), jsonout.run("analyze") as res:
        choice = dialogue.choose_backend(REMOTE_URL, None)
    assert choice.backend is None
    assert warning_codes(res.warnings) == ["chat_consent_needed"]
    assert REMOTE_KEY in res.warnings[0]["message"]
    assert net.to(REMOTE) == [] and net.to("api.anthropic.com") == []


def test_a_remote_ollama_url_with_the_assistant_consent_is_used(hosted_chat, home, monkeypatch):
    hosted_chat([f"assistant-v3:{ANTHROPIC}", REMOTE_KEY])
    net = Net(monkeypatch, ollama=True)
    with jsonout.run("analyze") as res:
        backend = dialogue.choose_backend(REMOTE_URL, None).backend
    assert backend is not None and backend.name == "remote" and REMOTE in backend.key
    assert backend.ask("read this") == {"moments": [{"first_line": 0}]}
    assert net.to(REMOTE) and net.to("api.anthropic.com") == [] and res.warnings == []


def test_the_skip_reason_names_the_remote_ollama_url_that_needs_consent(hosted_chat, monkeypatch):
    hosted_chat([f"assistant-v3:{ANTHROPIC}"])
    Net(monkeypatch, ollama=True)
    with jsonout.run("analyze"):
        choice = dialogue.choose_backend(REMOTE_URL, None)
    assert choice.backend is None
    assert choice.reason == f"the Ollama at {REMOTE}:11434 needs your consent (Models tab)"


def test_the_skip_reason_names_a_remote_ai_chat_ollama_that_needs_consent(home, monkeypatch):
    set_config(home, assistant={"provider": "ollama", "url": REMOTE_URL})
    Net(monkeypatch, ollama=False)  # the default Ollama does not answer either
    with jsonout.run("analyze"):
        choice = dialogue.choose_backend()
    assert choice.backend is None
    assert choice.reason == f"the Ollama at {REMOTE}:11434 needs your consent (Models tab)"


def test_the_skip_reason_of_a_hosted_chat_api_without_consent_is_unchanged(
    hosted_chat, monkeypatch
):
    hosted_chat([])
    Net(monkeypatch, ollama=False)
    with jsonout.run("analyze"):
        choice = dialogue.choose_backend()
    assert choice.backend is None
    assert choice.reason == "the hosted AI Chat model needs consent and no local text model answers"


def test_an_ollama_model_flag_alone_uses_the_default_url(home, monkeypatch):
    net = Net(monkeypatch, ollama=True)
    net.pulled = ["tiny:1b"]
    backend = dialogue.choose_backend(None, "tiny:1b").backend
    assert backend is not None and backend.model == "tiny:1b" and "127.0.0.1" in backend.key
    assert [c for c in net.calls if "11434" not in c[1]] == []


# --------------------------------------------------------------------------- the hosted model (F5)
class FakeHosted:
    """An OpenAI-compatible provider whose model is the server's only one, until resolved."""

    name, host, local, context_tokens = "openai", "llm.example.test", False, 0
    url = "https://llm.example.test/v1"

    def __init__(self, fail: bool = False) -> None:
        self._model, self.fail = "", fail

    @property
    def model(self) -> str:
        return self._model or providers.AUTO_SERVER_MODEL

    def resolve_model(self) -> str:
        if self.fail:
            raise providers.ProviderError("no models listed")
        self._model = "served-model"
        return self._model

    def capabilities(self):
        return providers.Capabilities(True, True)


HOSTED_KEY = "assistant-v3:openai@llm.example.test"


def test_the_hosted_cache_is_keyed_on_the_resolved_model(home, monkeypatch):
    set_config(home, assistant={"provider": "openai", "consent": [HOSTED_KEY]})
    monkeypatch.setattr(providers, "make", lambda settings=None: FakeHosted())
    backend = dialogue.choose_backend().backend
    assert backend is not None and backend.model == "served-model"
    assert "served-model" in backend.key and providers.AUTO_SERVER_MODEL not in backend.key


def test_a_hosted_model_that_cannot_be_resolved_is_no_backend(home, monkeypatch):
    set_config(home, assistant={"provider": "openai", "consent": [HOSTED_KEY]})
    monkeypatch.setattr(providers, "make", lambda settings=None: FakeHosted(fail=True))
    Net(monkeypatch, ollama=False)
    assert dialogue.choose_backend().backend is None


# --------------------------------------------------------------------------- cancel (F4)
def test_pieces_are_asked_on_daemon_threads_and_come_back_in_order():
    seen: list[bool] = []

    def fn(x: int) -> int:
        seen.append(threading.current_thread().daemon)
        return x * 2

    assert list(dialogue.ask_many([1, 2, 3, 4, 5], fn, 3)) == [2, 4, 6, 8, 10]
    assert seen and all(seen)


def test_a_piece_that_raises_is_no_reply_and_the_others_go_on():
    def fn(x: int) -> int:
        if x == 2:
            raise RuntimeError("boom")
        return x

    assert list(dialogue.ask_many([1, 2, 3], fn, 2)) == [1, None, 3]


def test_abandoning_the_pieces_does_not_wait_for_a_request_in_flight():
    gate = threading.Event()

    def fn(x: int) -> int:
        if x == 1:
            gate.wait(30)  # a hosted model that takes its time
        return x

    gen = dialogue.ask_many([0, 1], fn, 2)
    try:
        assert next(gen) == 0
        gen.close()  # returns at once; the thread stays a daemon
    finally:
        gate.set()


# --------------------------------------------------------------------------- memory (F8)
def test_a_dialogue_model_that_does_not_fit_in_memory_skips_the_step_not_the_run():
    from cutpoint.errors import LowMemoryError  # noqa: PLC0415

    def ask(prompt: str):
        raise LowMemoryError("Not enough memory for the model", "1 GB free")

    backend = dialogue.Backend("local", "m1", "k", ask)
    steps = progress.Run(progress.ANALYZE_STEPS, active=False)
    timings: dict[str, float] = {}
    spoken = {SRC: [{"start": 0.0, "end": 5.0, "text": "hi"}]}
    reader = analyze._dialogue_reader(backend, spoken, steps, timings, [])
    with jsonout.run("analyze") as res:
        assert reader([]) == []
    assert warning_codes(res.warnings) == ["step_failed"]
    assert res.warnings[0]["message"] == "Dialogue skipped: not enough free memory for m1"
    assert "dialogue_s" in timings


# --------------------------------------------------------------------------- re-runs and text (F9, F10)
def widened_project(tmp_path: Path, menu: list[list[float]] | None = None) -> Path:
    proj = tmp_path / "proj"
    hl = highlights(
        [own("h001", 120.0, 30.0, end=128.0), own("h002", 150.0, 70.0, end=158.0)],
        events_rescored=None,
    )
    hl["sources"] = [{"path": SRC, "duration": 600.0}]
    ev = events_of([talk_event(100.0, 180.0)], created="x")
    if menu:
        ev["stats"] = {"ocr": {SRC: {"menu_spans": menu}}}
    proj.mkdir()
    cache.save_json(proj / "highlights.json", hl)
    cache.save_json(proj / "events.json", ev)
    return proj


def test_a_rerun_checks_the_menu_against_the_candidates_not_the_widened_span(tmp_path):
    proj = widened_project(tmp_path)
    scoring.rescore_highlights(proj)  # h002 widened over h001
    ev = cache.load_json(proj / "events.json")
    ev["stats"] = {"ocr": {SRC: {"menu_spans": [[110.0, 126.0]]}}}  # over most of h001
    cache.save_json(proj / "events.json", ev)
    scoring.rescore_highlights(proj)  # again, from the saved widened candidates
    (cand,) = cache.load_json(proj / "highlights.json")["candidates"]
    assert cand["id"] == "h002" and (cand["start"], cand["end"]) == (100.0, 180.0)
    # h001 was in menu time: dropped, not merged back in unchecked
    assert (cand["widened_from"]["start"], cand["widened_from"]["end"]) == (150.0, 158.0)
    assert not cand["widened_from"].get("absorbed")


def test_a_rerun_without_menu_time_keeps_what_was_merged_in(tmp_path):
    proj = widened_project(tmp_path)
    scoring.rescore_highlights(proj)
    first = cache.load_json(proj / "highlights.json")["candidates"]
    scoring.rescore_highlights(proj)
    again = cache.load_json(proj / "highlights.json")["candidates"]
    assert [c["id"] for c in again] == ["h002"] == [c["id"] for c in first]
    assert [c["id"] for c in again[0]["widened_from"]["absorbed"]] == ["h001"]


def test_a_widened_candidate_gets_the_transcript_of_its_new_span_and_gives_it_back(
    tmp_path, monkeypatch
):
    words = [
        {"start": float(t), "end": t + 0.5, "word": f"w{t}", "probability": 0.99}
        for t in range(100, 180, 5)
    ]
    segs = [{"start": 100.0, "end": 180.0, "text": "x", "words": words}]
    monkeypatch.setattr(scoring, "_segments", lambda src, language=None: segs)
    cand = own("h001", 150.0, 60.0, end=158.0) | {"transcript": "w160"}
    hl = highlights([cand])
    scoring.nominate(hl, events_of([talk_event(100.0, 180.0)]), tmp_path, [grid()])
    assert cand["transcript"].startswith("w100 w105") and cand["transcript"] != "w160"
    assert "dialogue_description" not in cand
    scoring.nominate(hl, events_of([]), tmp_path, [grid()])
    assert cand["transcript"] == "w160" and (cand["start"], cand["end"]) == (150.0, 158.0)


def test_without_a_transcript_a_widened_candidate_carries_the_stretch_description(tmp_path):
    cand = own("h001", 150.0, 60.0, end=158.0) | {"transcript": "w160"}
    hl = highlights([cand])
    scoring.nominate(hl, events_of([talk_event(100.0, 180.0)]), tmp_path, [grid()])
    assert cand["transcript"] == "w160" and cand["dialogue_description"] == "a bit"
    scoring.nominate(hl, events_of([]), tmp_path, [grid()])
    assert "dialogue_description" not in cand


# --------------------------------------------------------------------------- long stretches
def whole(start: float, end: float) -> dict[str, Any]:
    """A candidate as a dialogue stretch nominates it."""
    return {
        "start": start, "end": end, "signals": {"events": {"nominated": ["dialogue-funny"]}},
    }  # fmt: skip


LONG_MIN = (400.0 - 100.0 + scoring.LOOKAHEAD_S) / 60  # minutes of the stretch 100-400 s


def long_nominee(extra: list[dict[str, Any]], tmp_path: Path, bumps=((250.0, 1.0),)):
    """A gaming project with a picked 60-point moment, and the stretch 100-400 s as a nominee."""
    from test_event_candidates import quiet_grid  # noqa: PLC0415

    hl = highlights([own("h001", 20.0, 60.0)])
    events = events_of([talk_event(100.0, 400.0, conf=0.0), *extra])
    scoring.nominate(hl, events, tmp_path, [quiet_grid(600.0, bumps=bumps)])
    (nom,) = [c for c in nominees(hl) if scoring.is_long_dialogue(c)]
    return hl, events, nom


def laughs(n: int, start: float = 120.0, confidence: float = 0.1) -> list[dict[str, Any]]:
    return [sound(start + 10 * i, confidence=confidence) for i in range(n)]


def test_a_long_dialogue_nominee_has_the_floor_not_the_peak_and_keeps_its_grid_score(tmp_path):
    _, _, nom = long_nominee([], tmp_path)
    assert nom["grid_score"] > scoring.NOMINEE_FLOOR * 60  # the loud moment inside it
    assert nom["score"] == pytest.approx(scoring.NOMINEE_FLOOR * 60)


def test_a_short_dialogue_nominee_and_a_kill_nominee_keep_the_peak_over_the_floor(tmp_path):
    from test_event_candidates import quiet_grid  # noqa: PLC0415

    hl = highlights([own("h001", 20.0, 10.0)])
    events = events_of([talk_event(100.0, 160.0, conf=0.0), feed(300.0)])
    scoring.nominate(hl, events, tmp_path, [quiet_grid(600.0, bumps=[(130.0, 1.0), (300.0, 1.0)])])
    short_nominees = nominees(hl)
    assert len(short_nominees) == 2
    for nom in short_nominees:
        assert not scoring.is_long_dialogue(nom)
        assert nom["grid_score"] > scoring.NOMINEE_FLOOR * 10
        assert nom["score"] == nom["grid_score"]  # max(peak, floor), as before


def test_a_long_dialogue_nominee_with_no_other_candidate_scores_zero(tmp_path):
    from test_event_candidates import quiet_grid  # noqa: PLC0415

    hl = highlights([])
    events = events_of([talk_event(100.0, 400.0, conf=0.0)])
    scoring.nominate(hl, events, tmp_path, [quiet_grid(600.0, bumps=[(250.0, 1.0)])])
    (nom,) = nominees(hl)
    assert nom["score"] == 0.0 and nom["grid_score"] > 0.0


@pytest.mark.parametrize("count", [0, 1, 6, 11, 40])
def test_the_laughter_points_of_a_long_candidate_scale_with_laughs_per_minute(count):
    cand = whole(100.0, 400.0)
    per_minute = count / LONG_MIN
    rate = min(1.0, per_minute / dialogue.DENSITY_FULL)
    boost, ev = scoring.event_boost(cand, laughs(count, confidence=0.1), style=STYLE)
    assert boost == pytest.approx(min(12.0, 60 * ev["laughter"]) * rate)


def test_laughter_density_below_at_and_above_three_a_minute():
    cand = whole(100.0, 400.0)  # 310 s with the lookahead
    full = round(dialogue.DENSITY_FULL * LONG_MIN)  # 16 laughs: just under 3 a minute
    below, _ = scoring.event_boost(cand, laughs(full - 6), style=STYLE)
    at, _ = scoring.event_boost(cand, laughs(full + 1), style=STYLE)
    above, _ = scoring.event_boost(cand, laughs(full + 10), style=STYLE)
    assert 0 < below < at == above == pytest.approx(6.0)  # 60 * 0.1, at full strength
    # exactly 3 a minute over a window of 5 minutes
    five = whole(0.0, 290.0)  # 300 s with the lookahead
    exact, _ = scoring.event_boost(five, laughs(15, start=1.0), style=STYLE)
    assert exact == pytest.approx(6.0)
    one_less, _ = scoring.event_boost(five, laughs(14, start=1.0), style=STYLE)
    assert one_less == pytest.approx(6.0 * 14 / 15)


def test_a_laugh_without_confidence_is_not_counted_for_the_density():
    five = whole(0.0, 290.0)
    boost, _ = scoring.event_boost(five, [*laughs(14, start=1.0), sound(200.0, confidence=0.0)])
    assert boost == pytest.approx(6.0 * 14 / 15)


def test_on_screen_events_of_a_long_candidate_count_per_minute():
    five = whole(0.0, 290.0)  # 5 minutes with the lookahead
    kills = [feed(10.0 * i + 1) for i in range(10)]  # 2 a minute
    boost, ev = scoring.event_boost(five, kills, style=STYLE)
    assert ev["player_kills"] == 10  # the evidence is still the count
    assert boost == pytest.approx(6.0 * 2.0)  # a kill is 6 points: 2 a minute, not min(10, max)
    boost, _ = scoring.event_boost(five, kills[:5], style=STYLE)
    assert boost == pytest.approx(6.0 * 1.0)
    # a short window: the count, as before
    short_, _ = scoring.event_boost({"start": 0.0, "end": 80.0}, kills[:2], style=STYLE)
    assert short_ == pytest.approx(6.0 * 2)


def test_the_per_minute_count_is_capped_at_the_types_max():
    five = whole(0.0, 290.0)
    kills = [feed(2.0 * i + 1) for i in range(100)]  # 20 a minute
    boost, _ = scoring.event_boost(five, kills, style=STYLE)
    (kill_type,) = [t for t in DEFAULT_EVENT_TYPES if t.name == "player_kills"]
    assert boost == pytest.approx(kill_type.score * kill_type.max)


def test_a_long_candidate_that_is_not_a_dialogue_nominee_keeps_the_old_boost():
    cand = {"start": 100.0, "end": 200.0}  # 100 s, e.g. a commentary moment (max_len 120)
    kills = [feed(110.0 + 5 * i) for i in range(10)]
    boost, _ = scoring.event_boost(cand, [sound(110.0, confidence=0.1), *kills], style=STYLE)
    (kill_type,) = [t for t in DEFAULT_EVENT_TYPES if t.name == "player_kills"]
    assert boost == pytest.approx(
        6.0 + kill_type.score * kill_type.max
    )  # laugh whole, kills by count
    nominated = {"signals": {"events": {"nominated": ["player_kills"]}}}
    other, _ = scoring.event_boost(
        {**cand, **nominated}, [sound(110.0, confidence=0.1), *kills], style=STYLE
    )
    assert other == boost


def test_candidates_of_90_seconds_or_less_are_scored_as_before():
    cand = {"start": 100.0, "end": 100.0 + scoring.DIALOGUE_WIDEN_MAX_S}
    assert not scoring.is_long(cand) and scoring.is_long({**cand, "end": cand["end"] + 0.1})
    boost, ev = scoring.event_boost(cand, [sound(110.0, confidence=0.1), feed(120.0)], style=STYLE)
    assert boost == pytest.approx(6.0 + 6.0)  # one laugh at full strength, one kill
    assert ev["player_kills"] == 1


def test_rescoring_a_project_with_a_long_dialogue_nominee_is_stable(tmp_path):
    from test_event_candidates import quiet_grid  # noqa: PLC0415

    hl = highlights([own("h001", 20.0, 60.0)])
    ev = events_of([talk_event(100.0, 400.0, conf=0.5), *laughs(8), feed(200.0)])
    part = quiet_grid(600.0, bumps=[(250.0, 1.0)])
    scoring.rescore(hl, ev, tmp_path, [part])
    scoring.number_candidates(hl)
    (nom,) = [c for c in hl["candidates"] if scoring.is_long_dialogue(c)]
    assert nom["score_pre_events"] == pytest.approx(scoring.NOMINEE_FLOOR * 60)
    assert nom["signals"]["events"]["boost"] > 0
    once = json.loads(json.dumps(hl))
    scoring.rescore(hl, ev, tmp_path, [part])
    scoring.number_candidates(hl)
    assert json.loads(json.dumps(hl)) == once
    scoring.rescore(hl, ev, tmp_path, None)  # and without grids: the nominee is kept as it was
    assert json.loads(json.dumps(hl)) == once


def test_the_cap_ranks_nominees_with_the_per_minute_boost(tmp_path):
    from test_event_candidates import quiet_grid  # noqa: PLC0415

    # room for one nominee. Both are floored to 36; the long stretch has 4 kills in 5 minutes
    # (4.6 points counted per minute, 18 by count), the short one a kill (6): the short one wins
    hl = highlights([own("h001", 20.0, 60.0)], max_candidates=2)
    kills = [feed(110.0 + 60 * i) for i in range(4)]
    events = events_of(
        [
            talk_event(100.0, 400.0, conf=0.0),
            *kills,
            talk_event(500.0, 540.0, conf=0.0),
            feed(505.0),
        ]
    )
    scoring.nominate(hl, events, tmp_path, [quiet_grid(600.0)])
    (nom,) = nominees(hl)
    assert nom["start"] > 400.0 and nom["end"] < 560.0


def test_the_moments_view_marks_a_long_dialogue_bit_with_its_description(tmp_path):
    hl = highlights([own("h001", 20.0, 60.0)])
    ev = titled_event(100.0, 400.0, description="Grandpa builds")
    short = {**talk_event(500.0, 560.0), "description": "short one"}
    scoring.nominate(hl, events_of([ev, short]), tmp_path, [grid()])
    view = moments_view(hl, events_of([ev, short]))["candidates"]
    bits = [c for c in view if c["long_bit"]]
    assert [(c["label"], c["bit_description"]) for c in bits] == [
        ("grandpas-wobbly-treehouse", "Grandpa builds")
    ]
    own_moment = next(c for c in view if c["id"] != c["cli_id"] and c["cli_id"] == "h001")
    assert own_moment["bit_description"] == ""  # not a dialogue moment
    assert len(view) == 3  # the own moment, the short stretch's nominee and the bit
    descriptions = {c["label"]: c["bit_description"] for c in view}
    assert sorted(descriptions.values()) == ["", "Grandpa builds", "short one"]  # not only bits
    widened = highlights([own("h001", 150.0, 60.0, end=160.0)])
    short_ev = titled_event(120.0, 180.0, description="Grandpa builds")
    scoring.nominate(widened, events_of([short_ev]), tmp_path, [grid()])
    (cand,) = moments_view(widened, events_of([short_ev]))["candidates"]
    assert cand["bit_description"] == "Grandpa builds" and cand["long_bit"] is False
    assert (
        bits[0]["display_label"] == "Grandpa's Wobbly Treehouse"
    )  # the readable title, not the slug
    assert all(
        c["display_label"] != "Grandpa's Wobbly Treehouse" for c in view if not c["long_bit"]
    )
    assert moments_view(hl)["candidates"][0]["bit_description"] == ""  # no events.json


# --------------------------------------------------------------------------- widening: silences
def words_at(*spans: tuple[float, float]) -> list[dict[str, Any]]:
    """Transcript segments with a word every 2 s over each ``(start, end)``."""
    words = [
        {"word": f"w{int(t)}", "start": float(t), "end": t + 0.5, "probability": 0.99}
        for a, b in spans
        for t in range(int(a), int(b), 2)
    ]
    return [{"start": words[0]["start"], "end": words[-1]["end"], "text": "x", "words": words}]


def spoken_of(segs: list[dict[str, Any]]) -> list[tuple[float, float]]:
    return [(w["start"], w["end"]) for s in segs for w in s["words"]]


@pytest.mark.parametrize(
    ("spans", "want"),
    [
        ([(100, 178)], (100, 178)),  # talk all along: nothing to cut
        ([(100, 116), (122, 178)], (100, 178)),  # a gap of 7.5 s is not long
        ([(100, 116), (140, 178)], (140, 178)),  # 23.5 s of silence before the moment
        (
            [(100, 116), (130, 160), (170, 178)],
            (130, 178),
        ),  # only the silence before the moment moves the start
    ],
)
def test_trim_silence_starts_after_a_long_silence_before_the_moment(spans, want):
    got = scoring.trim_silence(spoken_of(words_at(*spans)), 100.0, 178.0, (150.0, 158.0))
    assert got[0] == want[0]


def test_trim_silence_ends_before_a_long_silence_after_the_moment():
    spoken = spoken_of(words_at((100, 166)))  # the last word ends at 164.5
    assert scoring.trim_silence(spoken, 100.0, 178.0, (150.0, 158.0)) == (100.0, 164.5)


def test_trim_silence_never_cuts_into_the_moment():
    # a silence under the moment: it starts where the moment starts and ends where it ends
    spoken = spoken_of(words_at((100, 120), (170, 178)))
    assert scoring.trim_silence(spoken, 100.0, 178.0, (140.0, 160.0)) == (140.0, 160.0)
    assert scoring.trim_silence([], 100.0, 178.0, (150.0, 158.0)) == (150.0, 158.0)


def test_trim_silence_cuts_a_silent_head_up_to_the_first_word():
    spoken = spoken_of(words_at((112, 178)))
    assert scoring.trim_silence(spoken, 100.0, 178.0, (150.0, 158.0)) == (112.0, 178.0)


def widened(tmp_path, monkeypatch, segs, stretch=(100.0, 178.0), **extra):
    monkeypatch.setattr(scoring, "_segments", lambda src, language=None: segs)
    hl = widen_hl(own("h001", 150.0, 60.0, end=158.0) | {"transcript": "w150"})
    scoring.nominate(hl, events_of([talk_event(*stretch)]), tmp_path, [grid()])
    return hl


def test_a_widened_candidate_starts_after_the_long_silence_before_its_moment(tmp_path, monkeypatch):
    hl = widened(tmp_path, monkeypatch, words_at((100, 116), (140, 178)))
    (cand,) = hl["candidates"]  # no nominee for the stretch the candidate was trimmed from
    assert (cand["start"], cand["end"]) == (140.0, 178.0)
    assert cand["duration"] == 38.0
    assert (cand["widened_from"]["start"], cand["widened_from"]["end"]) == (150.0, 158.0)
    assert cand["transcript"].startswith("w140") and "w100" not in cand["transcript"]


def test_a_widened_candidate_ends_before_a_long_silence_after_its_moment(tmp_path, monkeypatch):
    hl = widened(tmp_path, monkeypatch, words_at((100, 166)))
    (cand,) = hl["candidates"]
    assert (cand["start"], cand["end"]) == (100.0, 164.5)


def test_a_trimmed_widening_is_undone_like_any_other(tmp_path, monkeypatch):
    hl = widened(tmp_path, monkeypatch, words_at((100, 116), (140, 178)))
    scoring.nominate(hl, events_of([]), tmp_path, [grid()])
    (cand,) = hl["candidates"]
    assert (cand["start"], cand["end"]) == (150.0, 158.0) and cand["transcript"] == "w150"


def test_a_trim_back_to_the_moment_itself_widens_nothing_and_nominates_nothing(
    tmp_path, monkeypatch
):
    # nothing is said around the moment: the stretch has no part that leads into it
    hl = widened(tmp_path, monkeypatch, words_at((150, 158)))
    (cand,) = hl["candidates"]
    assert (cand["start"], cand["end"]) == (150.0, 158.0) and "widened_from" not in cand


def test_a_trimmed_widened_candidate_keeps_the_stretch_title_and_its_type(tmp_path, monkeypatch):
    monkeypatch.setattr(
        scoring, "_segments", lambda src, language=None: words_at((100, 116), (140, 178))
    )
    stretch = talk_event(100.0, 178.0) | {"title": "Grandpa And The Ladder"}
    hl = widen_hl(own("h001", 150.0, 60.0, end=158.0))
    scoring.nominate(hl, events_of([stretch]), tmp_path, [grid()])
    (cand,) = hl["candidates"]
    assert cand["dialogue_title"] == "Grandpa And The Ladder"
    assert (
        cand["label"] == "grandpa-and-the-ladder"
        and cand["widened_from"]["label"] == "gaming-moment"
    )
    assert (cand["start"], cand["end"]) == (140.0, 178.0)


def test_a_stretch_over_the_cap_is_not_trimmed_and_stays_a_moment_of_its_own(tmp_path, monkeypatch):
    monkeypatch.setattr(
        scoring, "_segments", lambda src, language=None: words_at((100, 116), (300, 400))
    )
    hl = widen_hl(own("h001", 320.0, 60.0, end=328.0))
    assert scoring.nominate(hl, events_of([talk_event(100.0, 400.0)]), tmp_path, [grid()]) == 1
    (own_,) = [c for c in hl["candidates"] if not scoring.is_nominated(c)]
    assert (own_["start"], own_["end"]) == (320.0, 328.0)
    (nom,) = nominees(hl)
    assert nom["start"] <= 100.0 and nom["end"] == pytest.approx(400.0)


def test_without_a_transcript_a_widened_candidate_keeps_the_whole_stretch(tmp_path):
    hl = widen_hl(own("h001", 150.0, 60.0, end=158.0))
    scoring.nominate(hl, events_of([talk_event(100.0, 178.0)]), tmp_path, [grid()])
    (cand,) = hl["candidates"]
    assert (cand["start"], cand["end"]) == (100.0, 178.0)


def test_a_fresh_reading_removes_the_sources_stale_caches_only(tmp_path):
    """#803: caches of an earlier prompt version or of the same model's other transcripts are
    never read again; another model's current cache and another source's stay."""
    a = write(tmp_path / "a.mp4")
    b = write(tmp_path / "b.mp4")
    lines = talk_lines(10)
    other_lines = talk_lines(11)
    same, _ = fake_backend(lambda _p: {"moments": []}, key="fake@h:same")
    other, _ = fake_backend(lambda _p: {"moments": []}, key="fake@h:other")
    dialogue.read_source(a, other_lines, same)
    dialogue.read_source(a, lines, other)
    dialogue.read_source(b, lines, same)
    same_old = dialogue._cache_path(a, other_lines, same)
    other_a = dialogue._cache_path(a, lines, other)
    same_b = dialogue._cache_path(b, lines, same)
    stale = dialogue._cache_path(a, lines, same).with_name("dialogue-oldversion.json")
    stale.write_text(
        json.dumps({"version": dialogue.PROMPT_VERSION - 1, "model": "fake@h:other"}),
        encoding="utf-8",
    )
    assert same_old.is_file() and other_a.is_file() and same_b.is_file() and stale.is_file()

    dialogue.read_source(a, lines, same)
    assert dialogue._cache_path(a, lines, same).is_file()
    assert not same_old.exists()  # same model, other transcript
    assert not stale.exists()  # earlier prompt version
    assert other_a.is_file()  # another model, current version
    assert same_b.is_file()  # another source
    assert dialogue.read_source(a, lines, same).cached
