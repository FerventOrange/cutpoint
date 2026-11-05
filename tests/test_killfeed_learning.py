"""Feeds that learn their row forms from the footage, clock-stamped logs, player-tag matching
and label rows (synthetic rows, invented names and words; no OCR)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cutpoint.events import killfeed
from cutpoint.events.killfeed import (
    EventTracker,
    FeedParser,
    Interpreter,
    OcrParams,
    _icon_name,
    _join_log_rows,
    _row_comings,
    is_player,
    norm,
)

NAMES = ["Alder", "Birch", "Cedar", "Dunmore", "Elmwood", "Fenwick", "Galewind"]
VICTIMS = ["Tuvo", "Mara", "Nilo", "Orla", "Pike", "Quin"]


def _rows(*lines: str) -> list[tuple[str, float, str | None]]:
    return [(line, 0.9, None) for line in lines]


def _learner(*lines: str) -> tuple[FeedParser, bool]:
    """Learn from rows that each come on screen once, 30 s apart (a feed row scrolls out)."""
    parser = FeedParser({}, (), 1.0)
    return parser, parser.learn([(30.0 * i, "feed", _rows(line)) for i, line in enumerate(lines)])


def _fragged(n: int = 5) -> list[str]:
    """Rows of one form where players kill and die: a name on both sides of the word."""
    return [f"{NAMES[i]} fragged {NAMES[(i + 2) % len(NAMES)]}" for i in range(n)]


# ------------------------------------------------------------------------ learning a verb


def test_a_verb_between_many_names_is_learned_and_read() -> None:
    parser, learned = _learner(*_fragged())
    assert learned and parser.learned == ["fragged"] and not parser.group_rows
    assert parser._learned_split("Hazelnut fragged Alder") == ("Hazelnut ", " Alder")
    (ev,) = parser.learned_events(_rows("Hazelnut fragged Alder"))
    assert (ev["type"], ev["killer"], ev["victim"], ev["entry"]) == (
        "kill",
        "Hazelnut",
        "Alder",
        "text",
    )


def test_a_row_cut_after_its_verb_goes_on_in_the_next_row() -> None:
    parser, _ = _learner(*_fragged())
    (ev,) = parser.learned_events(_rows("Hazelnut fragged", "Alder"))
    assert (ev["killer"], ev["victim"]) == ("Hazelnut", "Alder")
    assert parser.learned_events(_rows("Hazelnut fragged")) == []


def test_a_preset_word_list_is_final() -> None:
    parser = FeedParser({"kill_words": {"destroyed": "kill"}}, (), 1.0)
    assert not parser.learn([(0.0, "feed", _rows(*_fragged()))]) and parser.learned == []


@pytest.mark.parametrize(
    "lines",
    [
        # a notice: the few texts after its words
        [f"{n} has left" for n in NAMES],
        # chat and loot lines
        [f"{n}: nice shot {NAMES[i - 1]}" for i, n in enumerate(NAMES)],
        [f"{n} receives loot: [Cloak]" for n in NAMES],
        # an NPC title: the same name after the word every time
        [f"{n} slew Tuvo" for n in NAMES],
        # a word between names that never swap sides (killers and victims differ)
        [f"{n} slew {m}" for n, m in zip(NAMES, VICTIMS, strict=False)],
        # too few different names: a name read with one other name
        ["Alder fragged Birch", "Alder fragged Birch", "Birch fragged Alder"],
    ],
    ids=["notice", "chat", "loot", "npc-title", "no-shared-name", "few-names"],
)
def test_rows_that_are_no_kills_are_not_learned(lines: list[str]) -> None:
    parser, learned = _learner(*lines)
    assert not learned and parser.learned == []


def test_the_shared_name_gate_opens_when_a_name_kills_and_dies() -> None:
    disjoint = [f"{n} slew {m}" for n, m in zip(NAMES, VICTIMS, strict=False)]
    assert not _learner(*disjoint)[1]
    # one victim also killed: players, so the form is learned
    assert _learner(*disjoint, "Mara slew Alder")[1]


def test_rows_of_two_names_with_a_group_each_are_learned_with_wrapped_rows() -> None:
    parser, learned = _learner(
        "Alder (Tank) Birch (Plane)",
        "Cedar (Tank) Dunmore (Jeep)",
        "Elmwood (Tank) Fenwick (Plane)",
        "Birch (Jeep) Galewind (Tank)",
        "Dunmore (Plane) Alder (Tank)",
    )
    assert learned and parser.group_rows and parser.learned == []
    (ev,) = parser.learned_events(_rows("Hazelnut (Tank) Alder (Plane)"))
    assert (ev["killer"], ev["victim"]) == ("Hazelnut", "Alder")
    assert (ev["killer_vehicle"], ev["victim_vehicle"]) == ("Tank", "Plane")
    # a word of the weapon ("Ember Lance") ends the first row; the target is on the next one
    (ev,) = parser.learned_events(_rows("Hazelnut (Tank) Ember Lance", "Alder (Plane)"))
    assert (ev["killer"], ev["victim"], ev["weapon"]) == ("Hazelnut", "Alder", "Ember Lance")
    # a target without a group is no kill (_parse_paren without AI targets)
    assert parser.learned_events(_rows("Hazelnut (Tank) Ember Lance", "Alder")) == []


def test_learn_feed_turns_the_notices_of_a_learned_form_into_kills() -> None:
    interp = Interpreter(OcrParams(), 1.0)
    lines = _fragged()
    for i, line in enumerate(lines):
        interp.tracker.see(float(i * 30), "notice", norm(line), text=line, unknown=True)
        interp.feed_rows.append((float(i * 30), "feed", _rows(line)))
    interp.tracker.see(500.0, "notice", norm("Kip has left"), text="Kip has left")
    interp.learn_feed()
    events = interp.tracker.finish()
    assert [(e["killer"], e["victim"]) for e in events if e["type"] == "kill"] == [
        (line.split()[0], line.split()[2]) for line in lines
    ]
    assert [e["text"] for e in events if e["type"] == "notice"] == ["Kip has left"]


# ------------------------------------------------------------------------ default words


def _crop() -> tuple[np.ndarray, Any]:
    img = np.zeros((300, 300, 3), np.uint8)

    def rows(*texts: str) -> list:
        return [[[[0, 40 * i], [300, 40 * i], [300, 40 * i + 30], [0, 40 * i + 30]], t, 0.9]
                for i, t in enumerate(texts)]  # fmt: skip

    return img, rows


def test_only_generic_words_are_default_kill_words() -> None:
    img, rows = _crop()
    parser = FeedParser({}, (), 1.0)
    assert parser.parse(img, rows("Kip zorched Alder")) == []  # a game's own word
    assert [e["type"] for e in parser.parse(img, rows("Kip destroyed Alder"))] == ["kill"]


# ------------------------------------------------------------------------ logs


def _line(text: str, conf: float = 0.9) -> tuple:
    return (text, conf, None, [])


def test_a_logs_wrapped_rows_are_one_entry_without_the_clock() -> None:
    lines, log = _join_log_rows(
        [_line("2:20 Kip destroyed", 0.9), _line("Alder (Zeta)", 0.7), _line("2:25 Birch left")]
    )
    assert [line[0] for line in lines] == ["Kip destroyed Alder (Zeta)", "Birch left"]
    assert lines[0][1] == 0.7 and log == {0, 1}  # the weakest read counts


def test_one_clock_row_or_none_is_no_log() -> None:
    rows = [_line("2:20 Kip destroyed"), _line("Alder (Zeta)")]
    assert _join_log_rows(rows) == (rows, set())


def test_a_log_row_cut_by_the_scroll_stays_in_the_log() -> None:
    lines, log = _join_log_rows(
        [_line("Alder (Zeta)"), _line("2:20 Kip destroyed"), _line("2:25 Birch left")]
    )
    assert [line[0] for line in lines] == ["Alder (Zeta)", "Kip destroyed", "Birch left"]
    assert log == {0, 1, 2}


def test_log_entries_read_by_kill_words_only_and_make_no_notices() -> None:
    img, rows = _crop()
    parser = FeedParser({"row_notices": True}, (), 1.0)
    evs = parser.parse(
        img,
        rows(
            "2:20 Kip destroyed", "Alder (Zeta)", "2:25 Birch has left the game", "Cedar sat down"
        ),
    )
    assert [(e["type"], e["text"], e["entry"]) for e in evs] == [
        ("kill", "Kip > Alder (Zeta)", "text")
    ]
    # the same rows without a clock are a notice, as ever
    evs = parser.parse(img, rows("Birch has left the game"))
    assert [e["type"] for e in evs] == ["notice"]


# ------------------------------------------------------------------------ names and tags


@pytest.mark.parametrize(
    ("name", "tags", "expected"),
    [
        ("Mari Gold", ("Marigold",), True),  # a space inside the name
        ("Mari_gold", ("Marigold",), True),
        ("M00nrock", ("Moonrock",), True),  # O read as 0
        ("[AB] Marigold", ("Marigold",), True),  # a clan tag read with the name
        ("Alder", ("Marigold",), False),
        ("Kip", ("Kip",), True),
        ("[AB] Kip", ("Kip",), True),
        ("[AB] Ax_7", ("Ax_7",), True),  # short tags with marks, read after a clan tag
        ("[AB]KipX", ("KipX",), True),
        ("[AB] K.ip", ("K.ip",), True),
        ("Marco", ("Marcus",), False),  # near names are other players
        ("Viper", ("Sniper",), False),
        ("Skipper", ("Kip",), False),  # a short tag is a whole word
        ("Kipling", ("Kip",), False),
    ],
)
def test_is_player_allows_ocr_variants(name: str, tags: tuple[str, ...], expected: bool) -> None:
    assert is_player(name, tags) is expected


@pytest.mark.parametrize(
    ("text", "ok"),
    [("Kip", True), ("[AB] Kip", True), ("Kip [Cloak", False), ("Kip Cloak]", False)],
)
def test_icon_names_need_matched_brackets(text: str, ok: bool) -> None:
    assert _icon_name(text, 3) is ok


# ------------------------------------------------------------------------ what teaches


def _frames(lines: list[str], shown_s: float, apart_s: float = 30.0) -> list:
    """Each row on screen for ``shown_s`` (a frame every 2 s), the rows ``apart_s`` apart."""
    return [
        (apart_s * i + 2.0 * k, "feed", _rows(line))
        for i, line in enumerate(lines)
        for k in range(int(shown_s / 2) + 1)
    ]


def test_rows_that_come_once_teach_nothing() -> None:
    parser = FeedParser({}, (), 1.0)
    assert not parser.learn([(0.0, "feed", _rows(*_fragged()))])  # all at one time
    assert parser.learn(_frames(_fragged(), 10.0)) and parser.learned == ["fragged"]


def test_rows_that_stay_on_screen_teach_nothing() -> None:
    """A row that stays 40 s is lobby text, not a feed row that scrolls out."""
    assert _row_comings(_frames(["Alder fragged Birch"], 40.0)) == {}
    assert not FeedParser({}, (), 1.0).learn(_frames(_fragged(), 40.0, apart_s=100.0))


def test_a_row_gone_and_back_comes_twice() -> None:
    frames = [(t, "feed", _rows("Alder fragged Birch")) for t in (0.0, 2.0, 40.0, 42.0)]
    assert _row_comings(frames) == {"Alder fragged Birch": [0.0, 40.0]}


@pytest.mark.parametrize(("times", "comings"), [((0, 2, 11, 13), [0, 11]), ((0, 2, 9, 11), [0])])
def test_a_row_is_gone_when_unseen_for_longer_than_persist(times: tuple, comings: list) -> None:
    frames = [(float(t), "feed", _rows("Alder fragged Birch")) for t in times]
    assert _row_comings(frames) == {"Alder fragged Birch": [float(t) for t in comings]}


@pytest.mark.parametrize("word", ["and", "versus", "the", "with"])
def test_sentence_words_between_names_are_not_learned(word: str) -> None:
    pairs = zip(NAMES, [*NAMES[1:], NAMES[0]], strict=True)
    assert not _learner(*[f"{a} {word} {b}" for a, b in pairs])[1]


def test_objective_lines_are_not_learned() -> None:
    lines = ["Capture the Flag", "Defend the Flag", "Hold the Flag", "Flag the Capture"]
    assert not _learner(*lines, "Flag the Defend")[1]


def test_a_row_cut_after_its_verb_does_not_take_a_row_of_its_own() -> None:
    parser, _ = _learner(*_fragged())
    events = parser.learned_events(_rows("Hazel fragged", "Cedar fragged Dunmore"))
    assert [(e["killer"], e["victim"]) for e in events] == [("Cedar", "Dunmore")]
    events = parser.learned_events(_rows("Hazel fragged", "Dunmore"))
    assert [(e["killer"], e["victim"]) for e in events] == [("Hazel", "Dunmore")]


def test_a_learned_word_is_a_word_of_its_own() -> None:
    parser, _ = _learner(*_fragged())
    assert parser._learned_split("Bigfragged Alder") is None
    assert parser._learned_split("Ax7fragged Alder") == ("Ax7", " Alder")
    assert parser._learned_split("Kip fraggedly Alder") is None
    assert parser.learned_events(_rows("Bigfragged Alder")) == []


# ------------------------------------------------------------------------ second pass


def _screen_of(shows: list[tuple[str, float, float, float, float]], menu: bool = False) -> Any:
    """``(text, x, y, from_s, to_s)`` rows of the feed, over gameplay (or the menu at 20-30 s)."""
    from test_ocr_menus import _menu  # noqa: PLC0415

    def screen(t: float) -> list[tuple[str, float, float]]:
        out = _menu() if menu and 20 <= t < 30 else [("Hold", 900, 820)]
        return out + [(text, x, y) for text, x, y, a, b in shows if a <= t < b]

    return screen


def _run_feed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shows: list, **kw: Any) -> tuple:
    from test_ocr_menus import FEED, NOTICE, _Engine, _feed_preset, _run  # noqa: PLC0415

    preset = _feed_preset() | {"feed": {"row_notices": True}}
    engine = _Engine(_screen_of(shows, kw.pop("menu", False)), (NOTICE, FEED))
    return _run(tmp_path, monkeypatch, engine, preset, **kw)


# five rows of one form, 9+ s apart, each on screen 3 s; Alder and Birch also kill
ROWS = ["Alder fragged Birch", "Cedar fragged Dunmore", "Birch fragged Cedar"]
ROWS += ["Dunmore fragged Elmwood", "Elmwood fragged Alder"]
SHOWS = [(row, 1410, 100, 9.0 * i + 1.0, 9.0 * i + 4.0) for i, row in enumerate(ROWS)]
# the det gate reads the feed again when its boxes change: a one-word row (no notice, no kill)
# turns up 1.5 s into each row, so every row is read twice or more
SHOWS += [("Ready", 1410, 140, 9.0 * i + 2.5, 9.0 * i + 4.0) for i in range(len(ROWS))]
PAIRS = [tuple(row.split(" fragged ")) for row in ROWS]


def _kills(events: list[dict]) -> list[tuple]:
    return [(e["killer"], e["victim"], e["t"]) for e in events if e["type"] == "kill"]


def test_learned_rows_are_kills_and_their_notices_are_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shows = [*SHOWS, ("Kip has left the game", 1410, 200, 50.0, 54.0)]
    shows += [("Ready", 1410, 240, 51.5, 54.0)]
    events, stats = _run_feed(tmp_path, monkeypatch, shows)
    assert _kills(events) == [(a, b, 9.0 * i + 1.0) for i, (a, b) in enumerate(PAIRS)]
    assert stats["feed_words"] == ["fragged"]
    # each notice was read in several frames: only the plain row stays one
    notices = [e for e in events if e["type"] == "notice" and e.get("region") == "feed"]
    assert [e["text"] for e in notices] == ["Kip has left the game"]


def test_a_label_in_the_feed_is_dropped_after_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two words that come back three times beside nobody else's names are no kill."""
    label = [
        (w, x, 200, t, t + 3.0)
        for t in (10.0, 22.0, 34.0)
        for w, x in (("Zorb", 1410), ("Hexa", 1640))
    ]
    events, _ = _run_feed(tmp_path, monkeypatch, [*SHOWS, *label])
    assert ("Zorb", "Hexa") not in {k[:2] for k in _kills(events)}
    assert len(_kills(events)) == len(PAIRS)


def test_a_learned_kill_is_not_folded_into_a_distant_icon_kill_of_the_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second pass starts over in time: the icon line of the same pair, 40 s later, is
    another kill, not the learned one at the start."""
    shows = [*SHOWS, ("Alder", 1410, 300, 50.0, 53.0), ("Birch", 1640, 300, 50.0, 53.0)]
    events, _ = _run_feed(tmp_path, monkeypatch, shows)
    assert [k for k in _kills(events) if k[:2] == ("Alder", "Birch")] == [
        ("Alder", "Birch", 1.0),
        ("Alder", "Birch", 50.0),
    ]


def test_a_learned_kill_of_a_line_already_read_as_an_icon_kill_is_one_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shows = [*SHOWS, ("Alder", 1410, 300, 1.0, 4.0), ("Birch", 1640, 300, 1.0, 4.0)]
    events, _ = _run_feed(tmp_path, monkeypatch, shows)
    assert [k for k in _kills(events) if k[:2] == ("Alder", "Birch")] == [("Alder", "Birch", 1.0)]
    assert len(_kills(events)) == len(PAIRS)


def _stable(row_s: float, apart_s: float) -> list:
    """Rows of one form, each ``row_s`` on screen, a new one every ``apart_s``, in two slots:
    the feed's boxes change only when a row comes or goes, so in between it is "kept"."""
    return [
        (row, 1410, 100 + 40 * (i % 2), apart_s * i + 1.0, apart_s * i + 1.0 + row_s)
        for i, row in enumerate(ROWS)
    ]


def test_rows_kept_on_screen_are_one_kill_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row on screen for 12 s is read when it comes and when its neighbour goes, and kept
    alive in between (no reading): still one kill, at the time it came."""
    events, _ = _run_feed(tmp_path, monkeypatch, _stable(12.0, 10.0))
    assert _kills(events) == [(a, b, 10.0 * i + 1.0) for i, (a, b) in enumerate(PAIRS)]


@pytest.mark.parametrize(("row_s", "learned"), [(8.0, False), (3.0, True)])
def test_rows_that_stay_long_with_stable_boxes_teach_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row_s: float, learned: bool
) -> None:
    """Read once, a row that stays is told by the frames it was kept alive in."""
    monkeypatch.setattr(killfeed, "ROW_NOTICE_MAX_S", 5.0)
    events, stats = _run_feed(tmp_path, monkeypatch, _stable(row_s, 12.0))
    assert bool(_kills(events)) is learned
    assert ("feed_words" in stats) is learned


NOTES = [(f"Option {w} here", 500, 300 + 40 * i) for i, w in enumerate(("one", "two", "three"))]


def _frozen_screen(shows: list, notes: bool = True) -> Any:
    """Still-menu text in the notice region at 20-45 s, over the feed ``shows``."""

    def screen(t: float) -> list[tuple[str, float, float]]:
        out = list(NOTES) if notes and 20 <= t < 45 else [("Hold", 900, 820)]
        return out + [(text, x, y) for text, x, y, a, b in shows if a <= t < b]

    return screen


def _run_still(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shows: list, **kw: Any) -> tuple:
    from test_ocr_menus import FEED, NOTICE, _Engine, _feed_preset, _run  # noqa: PLC0415

    preset = _feed_preset() | {"feed": {"row_notices": True}}
    engine = _Engine(_frozen_screen(shows, **kw), (NOTICE, FEED))
    return _run(tmp_path, monkeypatch, engine, preset, still=lambda t: 20 <= t < 45)


def _teach(at: tuple, frozen_from: float | None) -> list:
    """The five rows of ROWS at ``at`` (3 s each); the row at ``frozen_from`` stays to 45 s."""
    shows = [
        (row, 1410, 100 if a == frozen_from else 180, a, 45.0 if a == frozen_from else a + 3.0)
        for row, a in zip(ROWS, at, strict=True)
    ]
    return shows + [("Ready", 1410, 140, a + 1.0, a + 3.0) for a in at if a != frozen_from]


def test_a_learned_row_frozen_on_a_still_menu_leaves_it_a_menu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its kept frames are not feed activity: the static rule still finds the menu, and the
    frozen row (read before it) stays a kill."""
    shows = _teach((1.0, 10.0, 18.0, 46.0, 55.0), frozen_from=18.0)
    events, stats = _run_still(tmp_path, monkeypatch, shows)
    assert any(a <= 30.0 <= b for a, b in stats["menu_spans"])  # nothing reads at 30 s
    assert [(a, b) for a, b, _ in _kills(events)] == PAIRS


def test_a_real_read_inside_a_still_stretch_still_guards_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row read anew at 30 s is feed activity: the frames around it are no menu."""
    shows = _teach((1.0, 10.0, 18.0, 30.0, 55.0), frozen_from=18.0)
    events, stats = _run_still(tmp_path, monkeypatch, shows)
    assert not any(a <= 30.0 <= b for a, b in stats["menu_spans"])  # the read at 30 s guards it
    assert [(a, b) for a, b, _ in _kills(events)] == PAIRS


@pytest.mark.parametrize(
    ("still", "gone_s"),
    [((20.0, 45.0), 51.0), ((20.0, 45.0), 47.0), ((20.0, 30.0), 45.0)],
    ids=["still-20-45-line-18-51", "still-20-45-line-18-47", "still-20-30-line-18-45"],
)
def test_an_icon_kill_read_once_and_kept_under_a_still_screen_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, still: tuple, gone_s: float
) -> None:
    """Read at 18 s, then the feed does not change (kept, not read again) for up to 33 s,
    partly under a still screen: no label, whatever the time on moving frames."""
    a, b = still
    shows = [("Zander", 1410, 300, 18.0, gone_s), ("Boxer7", 1640, 300, 18.0, gone_s)]
    from test_ocr_menus import FEED, NOTICE, _Engine, _feed_preset, _run  # noqa: PLC0415

    preset = _feed_preset() | {"feed": {"row_notices": True}}
    notes = [(f"Option {w} here", 500, 300 + 40 * i) for i, w in enumerate(("one", "two", "three"))]

    def screen(t: float) -> list[tuple[str, float, float]]:
        out = list(notes) if a <= t < b else [("Hold", 900, 820)]
        return out + [(text, x, y) for text, x, y, f, g in shows if f <= t < g]

    events, _ = _run(
        tmp_path, monkeypatch, _Engine(screen, (NOTICE, FEED)), preset, still=lambda t: a <= t < b
    )
    assert [(k[0], k[1]) for k in _kills(events)] == [("Zander", "Boxer7")]


def test_an_icon_line_read_again_for_35_s_is_a_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-read for 35 s (a blinking row beside it changes the boxes every 1.5 s): the
    longest unbroken run of readings is over ROW_NOTICE_MAX_S."""
    shows = [("Zander", 1410, 100, 5.0, 40.0), ("Boxer7", 1640, 100, 5.0, 40.0)]
    shows += [("Ready", 1410, 140, 5.0 + 3 * k, 6.5 + 3 * k) for k in range(12)]
    events, _ = _run_feed(tmp_path, monkeypatch, shows)
    assert _kills(events) == []


def test_a_kill_read_in_one_feed_is_kept_while_another_feed_is_only_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Feed 1 shows a plain row all along (read once, then kept on every frame); an icon kill
    is read in feed 2 at 32 s. Kept frames are per feed: feed 1's kept frame at 32 s must not
    hide feed 2's real read, so the kill stays and the still stretch is judged on its reads."""
    from test_ocr_menus import FEED, NOTICE, _Engine, _feed_preset, _run  # noqa: PLC0415

    feed2 = (1400, 500, 480, 300)
    x, y, w, h = feed2
    preset = _feed_preset() | {"feed": {"row_notices": True}}
    preset["regions"] = [
        *preset["regions"],
        {"name": "feed2", "x": x, "y": y, "w": w, "h": h, "kind": "feed", "gate": "det"},
    ]
    notes = [(f"Option {w} here", 500, 300 + 40 * i) for i, w in enumerate(("one", "two", "three"))]

    def screen(t: float) -> list[tuple[str, float, float]]:
        out = list(notes) if 20 <= t < 45 else [("Hold", 900, 820)]
        out.append(("Press F for help", 1410, 100))
        if 32 <= t < 35:
            out += [("Zander", 1410, 600), ("Boxer7", 1640, 600)]
        return out

    engine = _Engine(screen, (NOTICE, FEED, feed2))
    events, stats = _run(tmp_path, monkeypatch, engine, preset, still=lambda t: 20 <= t < 45)
    assert [(k[0], k[1]) for k in _kills(events)] == [("Zander", "Boxer7")]
    assert stats["menu_frames"] == 30


def test_a_learned_kill_during_a_menu_is_dropped_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shows = [*SHOWS[:2], ("Elmwood fragged Alder", 1410, 100, 40.0, 43.0)]
    shows += [("Cedar fragged Birch", 1410, 140, 22.0, 26.0), *SHOWS[3:4]]
    events, _ = _run_feed(tmp_path, monkeypatch, shows, menu=True)
    kills = [k[:2] for k in _kills(events)]
    assert ("Cedar", "Birch") not in kills and ("Elmwood", "Alder") in kills


def _handle(interp: Interpreter, t: float, *reads: tuple[str, float]) -> None:
    img = np.zeros((300, 400, 3), np.uint8)
    items = [
        [[[0, 40 * i], [300, 40 * i], [300, 40 * i + 30], [0, 40 * i + 30]], text, conf]
        for i, (text, conf) in enumerate(reads)
    ]
    interp.handle(t, SimpleNamespace(name="feed", kind="feed"), img, items)


def test_a_line_that_turns_into_a_log_entry_keeps_its_reads() -> None:
    """One clock row (no log yet), then two (a log) at a higher confidence, then again: the
    same kill, read three times, without a KeyError."""
    interp = Interpreter(OcrParams(), 1.0)
    kip, birch = ("2:20 Kip destroyed Alder", 0.9), ("2:25 Birch has left", 0.9)
    _handle(interp, 1.0, ("2:20 Kip destroyed Alder", 0.8))
    _handle(interp, 2.0, kip, birch)
    _handle(interp, 3.0, kip, birch)
    (event,) = interp.tracker.events
    assert (event["type"], event["killer"], event["victim"]) == ("kill", "Kip", "Alder")
    assert event["sightings"] == 3


def test_kept_rows_replay_only_the_last_read_with_spare_rows() -> None:
    interp = Interpreter(OcrParams(), 1.0)
    _handle(interp, 1.0, ("Kip has left the game", 0.9))
    interp.keep_rows(2.0, "feed")
    assert [t for t, _, _ in interp.feed_rows] == [1.0, 2.0]
    _handle(interp, 3.0, ("Kip destroyed Alder", 0.9))  # a read with no spare row
    interp.keep_rows(4.0, "feed")
    _handle(interp, 5.0)  # nothing read
    interp.keep_rows(6.0, "feed")
    assert [t for t, _, _ in interp.feed_rows] == [1.0, 2.0]


# ------------------------------------------------------------------------ the tracker


def _line_kill(tr: EventTracker, t: float, killer: str, victim: str, **kw: Any) -> Any:
    fields = {"killer": killer, "victim": victim, "text": f"{killer} > {victim}"} | kw
    return tr.see(t, "kill", killer.upper(), exclusive=True, **fields)


@pytest.mark.parametrize(("first", "second"), [("icon", "text"), ("text", "icon")])
def test_icon_and_text_lines_of_one_pair_are_one_kill(first: str, second: str) -> None:
    tr = EventTracker(persist_s=8.0, fuzzy=0.8)
    _line_kill(tr, 1.0, "Kip", "Boxer7", entry=first)
    _line_kill(tr, 2.0, "Klp", "Boxer7", entry=second)  # another key: by the names
    assert len(tr.events) == 1 and tr.events[0]["sightings"] == 2


def test_a_misread_killer_of_a_vehicle_victim_is_one_line() -> None:
    tr = EventTracker(persist_s=8.0, fuzzy=0.8)
    vehicle = {"entry": "text", "victim_vehicle": "Plane"}
    _line_kill(tr, 1.0, "Alder", "Boxer7", **vehicle)
    _line_kill(tr, 2.0, "Zqxwvu", "Boxer7", **vehicle)  # the killer unreadable
    assert len(tr.events) == 1
    _line_kill(tr, 3.0, "Zqxwvu", "Kipling", **vehicle)  # another victim: another kill
    assert len(tr.events) == 2


def _second_pass(first_at: float, again_at: float) -> tuple[bool, dict]:
    tr = EventTracker(persist_s=8.0, fuzzy=0.8)
    for t in (first_at, first_at + 1.0):
        _line_kill(tr, t, "Alder", "Boxer7", entry="icon")
    tr.rewind()
    (first,) = tr.events
    again = _line_kill(tr, again_at, "A1der", "Boxer7", entry="text")
    return tr.fold(again, [first]), first


def test_fold_merges_a_misread_line_of_the_second_pass() -> None:
    done, first = _second_pass(10.0, 11.5)
    assert done and first["_times"] == [10.0, 11.0, 11.5] and first["sightings"] == 3


def test_fold_leaves_a_line_40_s_apart() -> None:
    done, first = _second_pass(10.0, 50.0)
    assert not done and first["sightings"] == 2


# ------------------------------------------------------------------------ labels


def _icon_kill(tracker: EventTracker, t: float, killer: str, victim: str, **kw: Any) -> None:
    tracker.see(
        t, "kill", f"{killer} KILL".upper(), exclusive=True,
        killer=killer, victim=victim, entry="icon", text=f"{killer} > {victim}", **kw,
    )  # fmt: skip


BLUE_ORANGE = {"killer_hue": "blue", "victim_hue": "orange"}


def _kept(interp: Interpreter) -> list[str]:
    return [e["text"] for e in interp.tracker.events]


def test_drop_labels_drops_two_words_that_only_come_back_together() -> None:
    interp = Interpreter(OcrParams(), 1.0)
    for t in (100.0, 120.0, 140.0):
        _icon_kill(interp.tracker, t, "Zorb", "Hexa")
    for t in (400.0, 420.0):  # twice: a replay shows a kill twice
        _icon_kill(interp.tracker, t, "Pirate", "Quill")
    interp.drop_labels()
    assert _kept(interp) == ["Pirate > Quill"] * 2


def test_drop_labels_keeps_a_repeated_line_whose_names_are_seen_elsewhere() -> None:
    interp = Interpreter(OcrParams(), 1.0)
    for t in (200.0, 220.0, 240.0):
        _icon_kill(interp.tracker, t, "Alder", "Boxer")
    _icon_kill(interp.tracker, 300.0, "Alder", "Kipling")
    interp.drop_labels()
    assert len(_kept(interp)) == 4


def test_drop_labels_drops_a_line_that_stays_too_long() -> None:
    interp = Interpreter(OcrParams(), 1.0)
    for t in range(0, 26, 2):  # one line on screen for 24 s, lobby text
        _icon_kill(interp.tracker, float(t), "Zed", "Yan")
    _icon_kill(interp.tracker, 100.0, "Alder", "Boxer")
    interp.drop_labels()
    assert _kept(interp) == ["Alder > Boxer"]


def test_drop_labels_measures_the_longest_unbroken_run() -> None:
    """A line seen for 14 s, gone, and seen for 14 s again is not 28 s on screen."""
    interp = Interpreter(OcrParams(), 1.0)
    for t in (*range(0, 15, 2), *range(18, 33, 2)):
        _icon_kill(interp.tracker, float(t), "Zed", "Yan")
    assert len(interp.tracker.events) == 1  # one event, 32 s from first to last sighting
    interp.drop_labels()
    assert _kept(interp) == ["Zed > Yan"]


def test_drop_labels_does_not_count_white_as_a_team_colour() -> None:
    interp = Interpreter(OcrParams(), 1.0)
    for t in (100.0, 120.0, 140.0):
        _icon_kill(interp.tracker, t, "Zorb", "Hexa", killer_hue="blue", victim_hue="white")
    interp.drop_labels()
    assert _kept(interp) == []


def test_drop_labels_keeps_lines_with_two_team_colours() -> None:
    """Two kills of one pair can follow each other, and a coloured line is no lobby text."""
    interp = Interpreter(OcrParams(), 1.0)
    for t in range(0, 26, 2):
        _icon_kill(interp.tracker, float(t), "Zed", "Yan", **BLUE_ORANGE)
    for t in (100.0, 120.0, 140.0):
        _icon_kill(interp.tracker, t, "Zorb", "Hexa", **BLUE_ORANGE)
    interp.drop_labels()
    assert len(_kept(interp)) == 4


# ------------------------------------------------------------------------ menu colour rule


def _menu_with_kill(start: float) -> Any:
    from test_ocr_menus import _menu  # noqa: PLC0415

    def screen(t: float) -> list[tuple[str, float, float]]:
        out = _menu() if 20 <= t < 30 else [("Hold", 900, 820)]
        if start <= t < start + 3:
            out += [("Alder", 1410, 100), ("Boxer7", 1640, 100)]
        return out

    return screen


@pytest.mark.parametrize(
    ("start", "coloured", "kills"),
    [
        (24.0, True, [("Alder", "Boxer7")]),  # appears partway through the menu
        (24.0, False, []),
        (20.0, True, []),  # shown from the menu's opening: a coloured menu row
    ],
)
def test_a_two_colour_kill_line_survives_a_menu_but_plain_rows_do_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, start: float, coloured: bool, kills: list
) -> None:
    from test_ocr_menus import FEED, NOTICE, _Engine, _feed_preset, _run  # noqa: PLC0415

    def paint(t: float, frame: np.ndarray) -> None:
        if (
            coloured and start <= t < start + 3
        ):  # crops start at the regions' union corner (300, 40)
            frame[60:84, 1110:1170] = (255, 0, 0)  # "Alder", blue
            frame[60:84, 1340:1412] = (0, 128, 255)  # "Boxer7", orange

    preset = _feed_preset() | {"feed": {"row_notices": True}}
    events, stats = _run(
        tmp_path, monkeypatch, _Engine(_menu_with_kill(start), (NOTICE, FEED)), preset, paint=paint
    )
    assert stats["menu_frames"] >= 27
    assert [(e["killer"], e["victim"]) for e in events if e["type"] == "kill"] == kills
