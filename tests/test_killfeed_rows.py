"""Feed rows read by kill words and icons: default words, clock-stamped logs, viewer chat,
player-tag matching, label rows and the menu rules (synthetic rows, invented names; no OCR)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cutpoint.events.killfeed import (
    EventTracker,
    FeedParser,
    Interpreter,
    OcrParams,
    _icon_name,
    _join_log_rows,
    is_player,
)

NAMES = ["Alder", "Birch", "Cedar", "Dunmore", "Elmwood", "Fenwick", "Galewind"]
VICTIMS = ["Tuvo", "Mara", "Nilo", "Orla", "Pike", "Quin"]


def _screen_of(shows: list[tuple[str, float, float, float, float]]) -> Any:
    """``(text, x, y, from_s, to_s)`` rows of the feed over gameplay."""

    def screen(t: float) -> list[tuple[str, float, float]]:
        return [("Hold", 900, 820)] + [(text, x, y) for text, x, y, a, b in shows if a <= t < b]

    return screen


def _run_feed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shows: list, **kw: Any) -> tuple:
    from test_ocr_menus import FEED, NOTICE, _Engine, _feed_preset, _run  # noqa: PLC0415

    preset = _feed_preset() | {"feed": {"row_notices": True}}
    return _run(tmp_path, monkeypatch, _Engine(_screen_of(shows), (NOTICE, FEED)), preset, **kw)


def _kills(events: list[dict]) -> list[tuple]:
    return [(e["killer"], e["victim"], e["t"]) for e in events if e["type"] == "kill"]


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


# ------------------------------------------------------------------------ viewer chat


MIDDLE_WORDS = ["well", "sure", "okay", "maybe", "really", "today", "still"]


def _chat_stream() -> list[tuple[str, int, int, float, float]]:
    """Viewer chat in a feed region: many users, a common word between two tokens, wrapped
    lines and a user's line that ends on its word, the next row holding another user."""
    users = [*NAMES, *VICTIMS]
    shows = []
    for i in range(len(users)):
        a, b, c = users[i], users[(i + 1) % len(users)], users[(i + 5) % len(users)]
        word = MIDDLE_WORDS[i % len(MIDDLE_WORDS)]
        rows = [f"{a} well {b}", f"{b} {word} {c}", f"{a} okay", c, f"{a} {word} {c} {word}"]
        rows += ["Copper well TO Lantern", "Hey sure Maple Gate", "Harbor okay", "Quillfeather"]
        shows += [(r, 1410, 100 + 40 * k, 9.0 * i + 1.0, 9.0 * i + 4.0) for k, r in enumerate(rows)]
    return shows


def test_viewer_chat_in_the_feed_gives_no_kills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: words that many chat users share between two names were learned as a kill
    verb, and every chat row became a kill."""
    events, _ = _run_feed(tmp_path, monkeypatch, _chat_stream())
    assert _kills(events) == []


# ------------------------------------------------------------------------ log actors


@pytest.mark.parametrize(
    "chat",
    [
        ["12:30 Kip: he got killed", "by the tank lol", "12:31 Alder: gg"],
        ["12:30 Kip: nice assist", "by the tank lol", "12:31 Alder: gg"],
        ["12:30 Kip (mod): he got killed", "by the tank lol", "12:31 Alder (mod): gg"],
    ],
    ids=["plain", "assist", "mod"],
)
def test_a_clock_stamped_chat_with_a_colon_after_the_name_gives_no_kill(chat: list[str]) -> None:
    img, rows = _crop()
    parser = FeedParser({"row_notices": True}, (), 1.0)
    evs = parser.parse(img, rows(*chat))
    assert [e for e in evs if e["type"] in ("kill", "assist")] == []


def _types(parser: FeedParser, *texts: str) -> list[str]:
    img, rows = _crop()
    return [e["type"] for e in parser.parse(img, rows(*texts))]


@pytest.mark.parametrize(
    "row",
    [
        "Marlin: killed that corner",
        "Marlin : killed that corner",
        "Marlin:killed that corner",
        "Marlin: killed2 that corner",
        "MARLIN: KILLED THAT CORNER",
        "Marlin: Destroyed the ramp",
        "Marlin\uff1a killed that corner",
        "Marlin (mod): killed that corner",
        "Marlin [mod]: killed that corner",
        "<Marlin>: killed that corner",
        "Marlin-: killed that corner",
        "Marlin.: killed that corner",
        "[mod]: they killed that corner",
        "-Kip-: they killed that corner",
        "Kip 2:20 destroyed Alder",  # documented: a clock after the name is chat outside logs
        "Marlin: assist Alder",
        "Marlin: has crashed",
        "Marlin: crashed",
    ],
)
def test_a_colon_after_the_name_makes_a_chat_row_with_no_event_at_all(row: str) -> None:
    """Outside a log too, for kill, assist and crash. A row with a notice feed gives no notice
    either: the rejected row is used, so no other form reads it."""
    assert _types(FeedParser({"row_notices": True}, (), 1.0), row) == []
    assert _types(FeedParser({}, (), 1.0), row) == []


def test_a_handle_joined_to_a_clock_shaped_number_is_chat_but_a_lone_log_row_is_not() -> None:
    parser = FeedParser({"row_notices": True}, (), 1.0)
    assert _types(parser, "Marlin39:25 killed that corner") == []
    assert _types(parser, "Marlin39:25 destroyed Alder") == []
    img, rows = _crop()
    evs = parser.parse(img, rows("2:20 Kip destroyed Alder"))
    assert [(e["type"], e["text"].removeprefix("2:20 ")) for e in evs] == [("kill", "Kip > Alder")]


@pytest.mark.parametrize(
    "row",
    [
        "[2:20] Kip destroyed Alder",
        "(2:20) Kip destroyed Alder",
        "#2:20 Kip destroyed Alder",
        "2\uff1a20 Kip destroyed Alder",
        "2;20 Kip destroyed Alder",
        "2: 20 Kip destroyed Alder",
        "120:30 Kip destroyed Alder",
        "[A:B] Kip destroyed Alder",
        "2:20 [A:B] Kip destroyed Alder",
    ],
)
def test_a_real_row_with_a_clock_or_a_clan_tag_keeps_its_kill(row: str) -> None:
    img, rows = _crop()
    evs = FeedParser({"row_notices": True}, (), 1.0).parse(img, rows(row))
    assert [e["type"] for e in evs] == ["kill"]
    assert evs[0]["victim"] == "Alder"


def test_a_preset_with_its_own_kill_words_has_the_same_chat_rule() -> None:
    """Documented limit: a chat row with a colon is lost on any feed, a normal row is not."""
    own = {"kill_words": {"wrecked": "kill", "helped": "assist", "bailed": "crash"}}
    parser = FeedParser(own | {"row_notices": True}, (), 1.0)
    for row in ("Marlin: wrecked that corner", "Marlin: helped Alder", "Marlin: bailed"):
        assert _types(parser, row) == []
    assert _types(parser, "Kip wrecked Alder") == ["kill"]


def test_normal_kill_and_crash_rows_still_give_events() -> None:
    parser = FeedParser({"row_notices": True}, (), 1.0)
    img, rows = _crop()
    evs = parser.parse(img, rows("Kip destroyed Alder", "Marlin has crashed"))
    assert [(e["type"], e["text"]) for e in evs] == [
        ("kill", "Kip > Alder"),
        ("crash", "Marlin has crashed"),
    ]


def test_a_log_kill_still_reads_with_its_group_on_the_next_row() -> None:
    img, rows = _crop()
    parser = FeedParser({"row_notices": True}, (), 1.0)
    evs = parser.parse(img, rows("2:20 Kip destroyed", "Alder (Zeta)", "2:25 Birch has left"))
    assert [e["text"] for e in evs if e["type"] == "kill"] == ["Kip > Alder (Zeta)"]


def test_a_word_of_no_default_list_gives_no_kill_without_a_preset() -> None:
    """Documented loss: a game's own word ("wrecked") needs the preset's kill_words."""
    img, rows = _crop()
    parser = FeedParser({}, (), 1.0)
    assert [e for e in parser.parse(img, rows("Kip wrecked Alder")) if e["type"] == "kill"] == []
    own = FeedParser({"kill_words": {"wrecked": "kill"}}, (), 1.0)
    assert [e["type"] for e in own.parse(img, rows("Kip wrecked Alder"))] == ["kill"]


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


def test_an_icon_kill_in_a_second_feed_survives_a_still_stretch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Feed 1 shows a plain row all along (read once, then kept on every frame); an icon kill
    is read in feed 2 at 32 s, partway through a still stretch: the kill stays and the still
    stretch is judged on its reads."""
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


@pytest.mark.parametrize(
    ("shown", "kills"),
    [
        ((24.0, 27.0), []),  # read only inside the menu: dropped with it (the documented limit)
        ((24.0, 27.0, 40.0, 43.0), [("Kip", "Alder", "text")]),  # also seen outside it: kept
    ],
    ids=["menu-only", "also-outside"],
)
def test_a_log_kill_read_only_inside_a_menu_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shown: tuple, kills: list
) -> None:
    from test_ocr_menus import FEED, NOTICE, _Engine, _feed_preset, _menu, _run  # noqa: PLC0415

    def screen(t: float) -> list[tuple[str, float, float]]:
        out = _menu() if 20 <= t < 30 else [("Hold", 900, 820)]
        if any(a <= t < b for a, b in zip(shown[::2], shown[1::2], strict=True)):
            out += [("2:20 Kip destroyed Alder", 1410, 100), ("2:25 Birch has left", 1410, 140)]
        return out

    preset = _feed_preset() | {"feed": {"row_notices": True}}
    events, stats = _run(tmp_path, monkeypatch, _Engine(screen, (NOTICE, FEED)), preset)
    assert stats["menu_frames"] >= 27
    got = [(e["killer"], e["victim"], e["entry"]) for e in events if e["type"] == "kill"]
    assert got == kills
