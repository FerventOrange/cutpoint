"""#858: a busy gameplay moment is no text screen (fake decoder and OCR engine, invented HUD
texts). Name tags, location labels and prompts that pile up on screen can reach the text rule's
staying lines; the kill feed going on beside them tells gameplay from a lobby."""

from __future__ import annotations

from pathlib import Path

from cutpoint.events.killfeed import menu_rules
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
