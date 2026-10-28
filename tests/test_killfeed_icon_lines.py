"""Icon kill lines ("name [weapon icon] name", no kill word) in the generic feed parser
(#433; synthetic crops, invented names)."""

from __future__ import annotations

import numpy as np
import pytest

from cutpoint.events.killfeed import EventTracker, FeedParser, _icon_cut, icon_split

BLUE, ORANGE, BLACK, WHITE = (230, 140, 30), (30, 140, 240), (10, 10, 10), (255, 255, 255)
ICON = {"row_notices": True}  # a discovered icon feed's preset ``feed``


def _line(widths: tuple[int, int, int] = (110, 70, 120), h: int = 30) -> np.ndarray:
    """A kill line crop: a blue name box, a black icon box with a white gun, an orange
    name box; white letter strokes on the name boxes."""
    wa, wg, wb = widths
    img = np.zeros((h, wa + wg + wb, 3), np.uint8)
    img[:, :wa] = BLUE
    img[:, wa : wa + wg] = BLACK
    img[10:20, wa + 10 : wa + wg - 10] = WHITE
    img[:, wa + wg :] = ORANGE
    for x in [*range(6, wa - 6, 9), *range(wa + wg + 6, wa + wg + wb - 6, 9)]:
        img[8:22, x : x + 2] = WHITE
    return img


def _item(x0: int, x1: int, text: str, h: int = 30) -> tuple[list, str, float]:
    return [[x0, 0], [x1, 0], [x1, h], [x0, h]], text, 0.95


def test_icon_split_of_one_box() -> None:
    assert icon_split(_line(), "AlderBoxer7") == ("Alder", "Boxer7", "blue", "orange")


def test_one_box_line_is_a_kill() -> None:
    img = _line()
    evs = FeedParser(ICON, ("Boxer7",), 1.0).parse(img, [_item(0, 300, "AlderBoxer7")])
    assert len(evs) == 1
    ev = evs[0]
    assert (ev["type"], ev["killer"], ev["victim"]) == ("kill", "Alder", "Boxer7")
    assert (ev["killer_hue"], ev["victim_hue"]) == ("blue", "orange")
    assert ev["player_death"] and not ev["player_kill"]


def test_two_boxes_split_by_an_icon_gap_are_a_kill() -> None:
    img = _line()
    items = [_item(0, 110, "Alder"), _item(180, 300, "Boxer7")]
    evs = FeedParser(ICON, (), 1.0).parse(img, items)
    assert [(e["killer"], e["victim"]) for e in evs] == [("Alder", "Boxer7")]


def test_plain_rows_are_no_kills() -> None:
    img = np.full((30, 300, 3), BLACK, np.uint8)
    img[8:22, 10:290:7] = WHITE
    parser = FeedParser(ICON, (), 1.0)

    def kills(items: list) -> list:
        return [e for e in parser.parse(img, items) if e["type"] == "kill"]

    assert kills([_item(0, 300, "Alder has found the bombs")]) == []
    # two boxes without an icon gap, and readouts across one
    assert kills([_item(0, 110, "Alder"), _item(116, 300, "Boxer7")]) == []
    assert kills([_item(0, 90, "FPS:106"), _item(160, 300, "GPU:8.5")]) == []


def test_icon_lines_need_the_flag() -> None:
    """Without ``row_notices`` (a preset that is no discovered icon feed) icon lines are
    not read: two words split by a gap are often a menu row ("Brightness | Medium")."""
    img = _line()
    parser = FeedParser({}, ("Boxer7",), 1.0)
    assert parser.parse(img, [_item(0, 300, "AlderBoxer7")]) == []
    assert parser.parse(img, [_item(0, 110, "Alder"), _item(180, 300, "Boxer7")]) == []


def test_three_boxes_are_no_kill() -> None:
    """A row of three words split by gaps (a menu bar) pairs no names."""
    img = np.full((30, 420, 3), BLACK, np.uint8)
    img[8:22, 10:410:7] = WHITE
    items = [_item(0, 80, "PLAY"), _item(160, 250, "STORE"), _item(330, 420, "OPERATORS")]
    assert [e["type"] for e in FeedParser(ICON, (), 1.0).parse(img, items)] == ["notice"]


def test_colour_map_keeps_the_mapped_rule() -> None:
    """A preset with team colours keeps its one-line rule: the icon split is for presets
    without a colour map."""
    img = _line()
    parser = FeedParser(ICON | {"colors": {"blue": "ally", "orange": "enemy"}}, (), 1.0)
    assert parser.parse(img, [_item(0, 300, "AlderBoxer7")]) == []


def test_row_notices_read_other_feed_rows() -> None:
    """With ``row_notices`` a feed row that is no kill and has no icon gap is a notice; an
    icon kill line stays a kill, and a row split by an icon gap into no names is neither."""
    img = np.zeros((90, 300, 3), np.uint8)
    img[:30] = _line()
    img[30:60, :] = BLACK
    img[38:52, 10:290:7] = WHITE
    img[60:] = _line()
    items = [
        _item(0, 300, "AlderBoxer7"),
        [[[0, 30], [300, 30], [300, 60], [0, 60]], "Kip has left the game", 0.9],
        [[[0, 60], [300, 60], [300, 90], [0, 90]], "FPS:106GPU:8.5", 0.9],
    ]
    evs = FeedParser(ICON, (), 1.0).parse(img, items)
    assert [(e["type"], e["text"]) for e in evs] == [
        ("kill", "Alder > Boxer7"),
        ("notice", "Kip has left the game"),
    ]
    assert evs[1]["unknown"]
    assert FeedParser({}, (), 1.0).parse(img, items) == []  # off without the flag


def _track(parser: FeedParser, reads: list[tuple[float, list[tuple[str, str]], float]]) -> list:
    """Track one-box kill line reads ``(t, (killer, victim) of the rows top first,
    confidence)``; each row's crop is sized to its names (10 px a character)."""
    tracker = EventTracker(persist_s=3.0, fuzzy=0.8)
    for t, rows, conf in reads:
        for a, b in rows:
            img = _line((10 * len(a) + 10, 70, 10 * len(b) + 10))
            w = img.shape[1]
            item = ([[0, 0], [w, 0], [w, 30], [0, 30]], a + b, conf)
            for ev in parser.parse(img, [item]):
                tracker.see(t, ev.pop("type"), ev.pop("key"), exclusive=True, **ev)
    return tracker.finish()


def test_icon_kill_keeps_its_most_complete_names() -> None:
    """A kill line read cut while it slides in or fades ("Boxe") takes the longer names of
    later reads, even when a cut read has the higher confidence, and keeps the player
    flag of the read with the player's whole name."""
    reads = [(1.0, [("Alder", "Boxe")], 0.99), (1.3, [("Alder", "Boxer7")], 0.9)]
    reads.append((1.6, [("A1der", "Boxe")], 0.99))  # a misread and cut read: not taken
    (ev,) = _track(FeedParser(ICON, ("Boxer7",), 1.0), reads)
    assert (ev["killer"], ev["victim"], ev["text"]) == ("Alder", "Boxer7", "Alder > Boxer7")
    assert ev["confidence"] == 0.99
    assert ev["player_death"] and not ev["player_kill"]


def test_two_kills_of_one_killer_stay_two() -> None:
    """A long killer name makes two lines' keys similar; the victims are not one name cut,
    so the new line (on top) is a second kill, not a read of the first."""
    killer = "AlderwoodRanger2024"
    first, second = (killer, "Boxer7"), (killer, "Kip")
    reads = [(1.0, [first], 0.9), (1.3, [second, first], 0.9)]
    evs = _track(FeedParser(ICON, (), 1.0), reads)
    assert sorted(e["text"] for e in evs) == [f"{killer} > Boxer7", f"{killer} > Kip"]


@pytest.mark.parametrize(
    "reads",
    [
        [("Alder", "Nova"), ("Alder", "Nava"), ("Alder", "Nova")],  # one letter misread
        [("Kip", "Boxer7"), ("Klp", "Boxer7")],
        [("FootSniffa12", "dramaticrail39"), ("FootSniffa12", "drarnati")],  # cut + "rn"
    ],
)
def test_misread_short_names_stay_one_kill(reads: list[tuple[str, str]]) -> None:
    evs = _track(
        FeedParser(ICON, (), 1.0), [(1.0 + 0.3 * i, [r], 0.9) for i, r in enumerate(reads)]
    )
    assert [e["text"] for e in evs] == [" > ".join(reads[0])]


def test_a_read_with_junk_once_does_not_win() -> None:
    """The longest read seen twice wins over a longer read seen once."""
    reads = [("Alder", "Boxer7"), ("Alder", "Boxer7"), ("Alder", "Boxer7 Zx")]
    evs = _track(
        FeedParser(ICON, (), 1.0), [(1.0 + 0.3 * i, [r], 0.9) for i, r in enumerate(reads)]
    )
    assert [e["text"] for e in evs] == ["Alder > Boxer7"]


@pytest.mark.parametrize(
    ("text", "k", "cut"), [("Kip V2Boxer7", 6, 6), ("Alder Boxer7", 6, 6), ("AlderBoxer7", 5.4, 5)]
)
def test_icon_cut_takes_the_nearest_boundary(text: str, k: float, cut: int) -> None:
    """The nearest space or word boundary to the gap's share; a space wins only a tie."""
    assert _icon_cut(text, k) == cut
