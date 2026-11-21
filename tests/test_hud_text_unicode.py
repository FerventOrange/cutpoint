"""HUD text in any script (#1085, A137): the normalisers keep non-Latin letters and digits,
and ASCII text gives exactly what it gave before (invented names and lines only)."""

from __future__ import annotations

import ast
import random
import re
import string
from pathlib import Path

import numpy as np
import pytest

from cutpoint.events import discover, text
from cutpoint.events.discover import DiscoverParams, cluster_regions
from cutpoint.events.feed_parser import (
    FeedParser,
    _word_spans,
    fuzzy_contains,
    is_player,
    norm,
    similar,
)
from cutpoint.events.killfeed import OcrParams
from cutpoint.events.ocr_read import Interpreter
from cutpoint.events.regions import Region
from cutpoint.events.tracker import EventTracker

TESTS = Path(__file__).parent


def _old_norm(t: str) -> str:
    """killfeed.norm before #1085 A137."""
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", t.upper())).strip()


def _old_norm_text(t: str) -> str:
    """discover._norm_text before #1085 A137."""
    return re.sub(r"[^a-z0-9]", "", t.lower())


def _ascii_corpus() -> list[str]:
    """Every ASCII string literal of the event and discovery tests, and 5000 random ASCII
    strings (printable characters, control characters and runs of spaces)."""
    out: set[str] = set()
    for path in TESTS.glob("test_*.py"):
        if not re.search(r"discover|hud|killfeed|feed|ocr|event|announce|region|score", path.name):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.isascii()
            ):
                out.add(node.value)
    rng = random.Random(1085)
    chars = string.printable + "\x00\x07\x1b\x7f" + "  " * 10
    for _ in range(5000):
        out.add("".join(rng.choice(chars) for _ in range(rng.randint(0, 40))))
    return sorted(out)


ASCII = _ascii_corpus()


def test_ascii_text_normalises_as_before() -> None:
    assert len(ASCII) > 5000
    for t in ASCII:
        assert norm(t) == _old_norm(t), t
        assert text._alnum_words_any(t) == _old_norm(t), t  # the general rule agrees too
        assert discover._norm_text(t) == _old_norm_text(t), t
        assert _word_spans(norm(t), 6, 1) == _word_spans(_old_norm(t), 6, 1)


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("CAFÉ", "CAFÉ"),
        ("café", "CAFÉ"),  # one form for both cases
        ("Straße", "STRASSE"),
        ("".join(chr(ord(c) + 0xFEE0) for c in "Abc12"), "ABC12"),  # fullwidth
        ("x² ①", "X2 1"),  # superscript and circled digits
        ("Kip™", "KIP"),  # a symbol is no letter (NFKC would make it "TM")
        ("हिन्दी", "हिन्दी"),  # vowel signs stay
        ("ПОБЕДА!", "ПОБЕДА"),
        ("김철수 [x]", "김철수 X"),
        ("!!", ""),
        ("! ́ ?", ""),  # a mark with nothing before it is dropped
    ],
)
def test_text_in_any_script_keeps_its_letters(raw: str, want: str) -> None:
    assert norm(raw) == want


def test_the_audit_reproductions() -> None:
    assert fuzzy_contains("ПОБЕДА", "ПОБЕДА")
    assert fuzzy_contains("Игрок победа раунд", "ПОБЕДА")  # case aside
    assert fuzzy_contains("КОМАНДАПОБЕДИЛА", "КОМАНДА ПОБЕДИЛА")  # OCR dropped the space
    assert is_player("김철수", ("김철수",))
    assert not is_player("박영희", ("김철수",))


def test_two_empty_texts_are_no_match() -> None:
    assert similar("", "") == 0.0
    assert similar("--", "!!") == 0.0
    assert similar("Kip", "Kip") == 1.0
    assert similar("!́", "?́") == 0.0


NOTICE = Region("notice", 600, 700, 700, 100, "notice")
BOX = [[0, 0], [300, 0], [300, 60], [0, 60]]


def test_two_cyrillic_notices_are_two_events() -> None:
    interp = Interpreter(OcrParams(game="tilefall", fps=1.0), 1.0)
    interp.handle(5.0, NOTICE, None, [(BOX, "Игрок покинул игру", 0.9)])
    interp.handle(6.0, NOTICE, None, [(BOX, "Соединение потеряно", 0.9)])
    interp.handle(7.0, NOTICE, None, [(BOX, "Соединение потеряно", 0.9)])
    texts = [e["text"] for e in interp.tracker.events]
    assert texts == ["Игрок покинул игру", "Соединение потеряно"]


BLUE, ORANGE, BLACK, WHITE = (230, 140, 30), (30, 140, 240), (10, 10, 10), (255, 255, 255)


def _icon_line() -> np.ndarray:
    """A kill line crop (as tests/test_killfeed_icon_lines.py): blue name, icon, orange name."""
    img = np.zeros((30, 300, 3), np.uint8)
    img[:, :110] = BLUE
    img[:, 110:180] = BLACK
    img[10:20, 120:170] = WHITE
    img[:, 180:] = ORANGE
    for x in [*range(6, 104, 9), *range(186, 294, 9)]:
        img[8:22, x : x + 2] = WHITE
    return img


def test_korean_icon_kills_stay_two() -> None:
    parser = FeedParser({"row_notices": True}, (), 1.0)
    tracker = EventTracker(persist_s=3.0, fuzzy=0.8)
    img = _icon_line()
    for t, (a, b) in [(1.0, ("김철수", "박영희")), (1.3, ("이민호", "최지우"))]:
        items = [
            ([[0, 0], [110, 0], [110, 30], [0, 30]], a, 0.95),
            ([[180, 0], [300, 0], [300, 30], [180, 30]], b, 0.95),
        ]
        for ev in parser.parse(img, items):
            tracker.see(t, ev.pop("type"), ev.pop("key"), exclusive=True, **ev)
    evs = tracker.finish()
    assert sorted(e["text"] for e in evs) == ["김철수 > 박영희", "이민호 > 최지우"]


def test_a_non_latin_icon_feed_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """The icon feed of tests/test_discover_icon_feed.py with Cyrillic names: before, every
    name was "", so the region's text never changed and no name was seen."""
    import test_discover_icon_feed as icon_feed  # noqa: PLC0415

    names = ["Ольга", "Борис7", "Кирилл", "Нина", "Пётр", "Олег"]  # noqa: RUF001
    monkeypatch.setattr(icon_feed, "NAMES", names)
    proposals, _ = cluster_regions(icon_feed._frames(gap=70), DiscoverParams())
    feed = [p for p in proposals if icon_feed._at_feed(p)]
    assert [p["kind"] for p in feed] == ["feed"]
    assert feed[0]["evidence"]["icon_lines"] >= 0.9
