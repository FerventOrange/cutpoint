"""#858: a busy gameplay moment is no text screen (fake decoder and OCR engine, invented HUD
texts). Name tags, location labels and prompts that pile up on screen can reach the text rule's
staying lines; the kill feed going on beside them tells gameplay from a lobby."""

from __future__ import annotations

from pathlib import Path

from cutpoint.events.killfeed import feed_stacks, menu_rules
from test_ocr_menus import FEED, NOTICE, _Engine, _feed_preset, _lobby, _moving_records, _run

KILL = [("Orrin", 1410, 60), ("destroyed", 1500, 60), ("Pell", 1640, 60)]


def _busy(t: float, kill: bool = True) -> list[tuple[str, float, float]]:
    """Gameplay; at 12 s three labels show at once, at 19 s two prompts join them, all
    until 26 s (5 lines that stay, 3 first read together, for 7 s); a kill in the feed at
    21-24 s."""
    out = [("Hold", 900, 820)]
    if 12 <= t < 26:
        out += [("Ridge Gate", 600, 230), ("Vela Ostrom", 600, 305), ("Mara Quill", 600, 380)]
    if 19 <= t < 26:
        out += [("Cargo Lift", 600, 455), ("Brace Door", 600, 530)]
    if kill and 21 <= t < 24:
        out += KILL
    return out


def _lobby_with_kills(t: float) -> list[tuple[str, float, float]]:
    """:func:`test_ocr_menus._lobby` (a moving lobby at 20-32 s) with kills in the
    feed at 8-11 s and 40-43 s, before and after it."""
    out = _lobby(t)
    if 8 <= t < 11 or 40 <= t < 43:
        out += KILL
    return out


def _text(records, spans, feed: set[float]) -> list[float]:
    rules = menu_rules(records, feed_times=feed, text_spans=spans)[0]
    return [f["t"] for f in records if rules[f["id"]] == "text"]


def test_busy_moment_with_feed_activity_is_no_text_screen() -> None:
    times = [n / 3 for n in range(120)]  # 40 s of moving frames
    records = _moving_records(times)
    busy = [(12.0, 26.0)] * 3 + [(19.0, 26.0)] * 2
    # without feed activity the lines read as a text screen at 19-26 s
    assert _text(records, busy, set()) == [t for t in times if 19.0 <= t <= 26.0]
    # a kill read at 21-24 s: every frame of it is within STATIC_FEED_GUARD_S of the feed
    kill = {21.0, 22.0, 23.0, 24.0}
    assert _text(records, busy, kill) == []
    # a kill far from it (gameplay elsewhere) changes nothing
    assert _text(records, busy, {2.0, 35.0}) == [t for t in times if 19.0 <= t <= 26.0]


def test_lobby_next_to_feed_activity_is_still_a_text_screen() -> None:
    times = [n / 3 for n in range(120)]
    records = _moving_records(times)
    lobby = [(4.0, 18.0)] * 5  # 5 lines that open at once, on screen 4-18 s
    assert _text(records, lobby, {1.0, 25.0}) == [t for t in times if 4.0 <= t <= 18.0]
    # a feed sighting at 3 s: the frames within 2 s of it go, the rest stays a text screen
    assert _text(records, lobby, {3.0}) == [t for t in times if 5.0 < t <= 18.0]
    # one at 12 s splits it: 4-10 s stays, 14-18 s too (each at least TEXT_MIN_S)
    assert _text(records, lobby, {12.0}) == [t for t in times if 4.0 <= t < 10 or 14 < t <= 18]
    # one at 9 s leaves 4-6.7 s (under TEXT_MIN_S, so no text screen) and 11.3-18 s
    assert _text(records, lobby, {9.0}) == [t for t in times if 11 < t <= 18]


def test_busy_moment_keeps_its_events(tmp_path: Path, monkeypatch) -> None:
    events, stats = _run(tmp_path, monkeypatch, _Engine(_busy, (NOTICE, FEED)), _feed_preset())
    assert stats["menu_spans"] == [] and stats["text_frames"] == 0
    kills = [(e["t"], e["killer"], e["victim"]) for e in events if e["type"] == "kill"]
    assert kills == [(21.0, "Orrin", "Pell")]
    # the labels read at 12 s stay notices (a text screen would drop them)
    notices = [e["t"] for e in events if e["type"] == "notice" and 12 <= e["t"] < 26]
    assert notices == [12.0, 12.0, 12.0]
    # the same lines without the kill: no feed activity, so the text rule takes them
    (quiet := tmp_path / "quiet").mkdir()
    events, stats = _run(
        quiet, monkeypatch, _Engine(lambda t: _busy(t, kill=False), (NOTICE, FEED)), _feed_preset()
    )
    assert stats["text_frames"] > 0 and stats["menu_spans"] != []


def test_lobby_between_kills_is_still_a_menu(tmp_path: Path, monkeypatch) -> None:
    engine = _Engine(_lobby_with_kills, (NOTICE, FEED))
    events, stats = _run(tmp_path, monkeypatch, engine, _feed_preset())
    assert stats["text_frames"] > 0
    assert len(stats["menu_spans"]) == 1
    start, end = stats["menu_spans"][0]
    assert 20.0 <= start < 23.0 and 29.0 < end <= 32.5
    assert not [e for e in events if 20 <= e["t"] < 32 and e["type"] != "kill"]
    kills = [e["t"] for e in events if e["type"] == "kill"]
    assert kills == [8.0, 40.0]


# #891: a busy feed whose rows read as no kill. In a discovered icon feed (``row_notices``) a
# row that is no kill line is a notice; it is still a feed line, not a menu's text.
FEED_ROWS = (
    "Orrin Vale [Kestrel Mk2] Pell Abernathy",
    "Sable Quintero-Hart (Heron) Tamsin",
    "Ivo 77 Lark (Wren) Odessa Marchbanks",
    "Brannock_the_Second (Gull IV) Mire",
    "Quill (Osprey) Harrow Vandersloot",
)


def _feed_rows(t: float) -> list[tuple[str, float, float]]:
    """Gameplay; from 20 s the feed fills: three rows at once, one more at 22 s and 23 s,
    all on screen until 34 s (5 lines that stay, 3 first read together, for 11 s)."""
    out = [("Hold", 900, 820)]
    for i, (row, start) in enumerate(zip(FEED_ROWS, (20, 20, 20, 22, 23), strict=True)):
        if start <= t < 34:
            out.append((row, 1410, 60 + 40 * i))
    return out


def _row_preset() -> dict:
    return _feed_preset() | {"feed": {"row_notices": True}}


def test_feed_rows_that_are_no_kills_are_no_text_screen(tmp_path: Path, monkeypatch) -> None:
    events, stats = _run(tmp_path, monkeypatch, _Engine(_feed_rows, (NOTICE, FEED)), _row_preset())
    rows = [e for e in events if e.get("region") == "feed"]
    assert {e["type"] for e in rows} == {"notice"}  # no row reads as a kill
    # the last row is read once (nothing changes after it): an unknown line read once is dropped
    assert sorted(e["text"] for e in rows) == sorted(FEED_ROWS[:4])
    assert stats["text_frames"] == 0 and stats["menu_spans"] == []


def test_the_same_lines_outside_the_feed_are_a_text_screen(tmp_path: Path, monkeypatch) -> None:
    """The rows of :func:`_feed_rows` in the notice region (a lobby list) stay a menu."""

    def lobby(t: float) -> list[tuple[str, float, float]]:
        return [(text, 600 if x == 1410 else x, y + 200) for text, x, y in _feed_rows(t)]

    _, stats = _run(tmp_path, monkeypatch, _Engine(lobby, (NOTICE, FEED)), _row_preset())
    assert stats["text_frames"] > 0
    assert len(stats["menu_spans"]) == 1
    start, end = stats["menu_spans"][0]
    assert 22.0 <= start <= 23.0 and 33.0 <= end <= 34.5


def test_a_lobby_beside_feed_rows_is_still_a_text_screen(tmp_path: Path, monkeypatch) -> None:
    """A lobby's lines (:func:`test_ocr_menus._lobby`, 20-32 s) while feed rows that are no
    kills stay in the feed's area: the rows neither add to nor guard the menu."""

    def screen(t: float) -> list[tuple[str, float, float]]:
        return [*_lobby(t), *(r for r in _feed_rows(t) if r[1] == 1410)]

    _, stats = _run(tmp_path, monkeypatch, _Engine(screen, (NOTICE, FEED)), _row_preset())
    assert stats["text_frames"] > 0 and len(stats["menu_spans"]) == 1
    start, end = stats["menu_spans"][0]
    assert 20.0 <= start < 23.0 and 29.0 < end <= 32.5


def test_overlapping_feed_rows_are_one_stack() -> None:
    rows = [(23.0, 34.0), (22.0, 31.0), (40.0, 44.0), (34.0, 36.0), (41.0, 43.0)]
    assert feed_stacks(rows) == [(22.0, 36.0), (40.0, 44.0)]
    assert feed_stacks([]) == []


def test_feed_rows_first_read_together_are_lines_of_their_own() -> None:
    """Every group of TEXT_TOGETHER or more rows first read in one frame adds its rows but
    the one that ends last (it is the stack's own line); rows that arrive one by one, or two
    at once (a double kill), add none."""
    rows = [(20.0, 30.0), (20.0, 34.0), (22.0, 31.0), (23.0, 34.0), (24.0, 33.0), (24.0, 32.0)]
    assert feed_stacks(rows) == [(20.0, 34.0)]  # pairs at 20 s and 24 s
    rows = [(20.0, 30.0), (20.0, 34.0), (20.0, 31.0), (23.0, 34.0)]
    assert feed_stacks(rows) == [(20.0, 34.0), (20.0, 31.0), (20.0, 30.0)]
    screen = [(10.0, 18.0)] * 5 + [(12.0, 16.0)]  # five rows that open at once
    assert feed_stacks(screen) == [(10.0, 18.0)] * 5


def test_a_double_kill_and_rows_one_by_one_are_no_text_screen() -> None:
    """Two feed rows at 20 s, one more at 22, 23 and 24 s, and three labels from 21 s, all
    until 34 s, with no feed sighting: the feed is one line, so four lines at most."""
    times = [n / 3 for n in range(150)]
    records = _moving_records(times)
    feed = feed_stacks([(20.0, 34.0)] * 2 + [(22.0, 34.0), (23.0, 34.0), (24.0, 34.0)])
    assert feed == [(20.0, 34.0)]
    assert _text(records, [*feed, *[(21.0, 34.0)] * 3], set()) == []


def test_two_screens_in_one_stack_each_keep_their_lines() -> None:
    """A screen's five rows at 20-32 s and the next one's at 32-44 s overlap into one stack;
    each opens at once, so each keeps its lines and the span covers both."""
    times = [n / 3 for n in range(150)]
    records = _moving_records(times)
    lines = feed_stacks([(20.0, 32.0)] * 5 + [(32.0, 44.0)] * 5)
    assert lines == [(20.0, 44.0), *[(20.0, 32.0)] * 4, *[(32.0, 44.0)] * 4]
    assert _text(records, lines, set()) == [t for t in times if 20.0 <= t <= 44.0]


def test_short_feed_rows_back_to_back_add_no_line(tmp_path: Path, monkeypatch) -> None:
    """Four lines of a screen (one short of a text screen) while three feed rows that are no
    kills follow each other, 5 s each: no row stays long enough to be a line, so their stack
    adds none and the screen stays gameplay (merged first, they made a 15 s line)."""

    def screen(t: float) -> list[tuple[str, float, float]]:
        out = [line for line in _lobby(t) if line[1] != 600 or line[2] < 230 + 75 * 4]
        for i, start in enumerate((20, 25, 30)):
            if start <= t < start + 5:
                out.append((FEED_ROWS[i], 1410, 60 + 40 * i))
        return out

    _, stats = _run(tmp_path, monkeypatch, _Engine(screen, (NOTICE, FEED)), _row_preset())
    assert stats["feed_regions"]["feed"]["lines"] == 3  # each row was read
    assert stats["text_frames"] == 0 and stats["menu_spans"] == []


def _screen_in_feed(t: float) -> list[tuple[str, float, float]]:
    """Gameplay, and at 20-32 s a screen (a lobby with a waiting dialog) that opens at once:
    five of its lines and a countdown lie in the feed's area, one line in the notice region."""
    out = [("Hold", 900, 820)]
    if 20 <= t < 32:
        words = ("Armory Hall", "Barracks Yard", "Courtyard Gate", "Dockyard Crane", "Embassy Row")
        out = [(w, 1650, 60 + 40 * i) for i, w in enumerate(words)]
        out += [(f"0:{32 - int(t):02d}", 1650, 260), ("Waiting for squad", 600, 400)]
    return out


def test_a_screen_whose_lines_sit_in_the_feed_stays_a_text_screen(
    tmp_path: Path, monkeypatch
) -> None:
    """Lines of a screen in the feed's area are first read in one frame, unlike a feed's rows
    (:func:`_feed_rows`, which come one by one or two at once): each stays a line, so the
    screen stays a text screen and gives no events."""
    events, stats = _run(
        tmp_path, monkeypatch, _Engine(_screen_in_feed, (NOTICE, FEED)), _row_preset()
    )
    assert stats["text_frames"] > 0 and len(stats["menu_spans"]) == 1
    start, end = stats["menu_spans"][0]
    assert 20.0 <= start < 23.0 and 29.0 < end <= 32.5
    assert not [e for e in events if 20 <= e["t"] < 32]
