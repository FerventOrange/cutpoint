"""Over-flagging in OCR events: one kill-feed row read twice, a banner text that stays on a
screen, and a system line read as a kill (synthetic events, invented names)."""

from __future__ import annotations

from typing import Any

from cutpoint.events.killfeed import (
    EventTracker,
    FrameRec,
    Interpreter,
    OcrInterpretation,
    OcrParams,
    collapse_banner_bursts,
    merge_kill_reads,
)
from cutpoint.events.regions import Region, preset_from_dict

FEED = Region("feed", 1500, 0, 400, 400, "feed")
BANNER = Region("banner", 600, 400, 600, 200, "banner")


def _kill(
    t: float,
    end: float,
    killer: str,
    victim: str,
    conf: float = 0.9,
    sightings: int = 1,
    **flags: Any,
) -> dict[str, Any]:
    return {
        "t": t,
        "end": end,
        "type": "kill",
        "text": f"{killer} > {victim}",
        "killer": killer,
        "victim": victim,
        "raw": f"{killer}{victim}",
        "confidence": conf,
        "sightings": sightings,
        "entry": "icon",
        "player": False,
        "player_kill": False,
        "player_death": False,
        "teamkill": False,
    } | flags


def _banner(t: float, text: str = "ROUND WON", raw: str | None = None) -> dict[str, Any]:
    """A banner event; ``raw`` is the whole line it was read in (default: the phrase alone)."""
    return {
        "t": t,
        "end": t + 2,
        "type": "banner",
        "text": text,
        "raw": text if raw is None else raw,
        "detail": "",
        "confidence": 0.9,
        "sightings": 2,
    }


def _box(row: int) -> list[list[int]]:
    return [[0, 40 * row], [300, 40 * row], [300, 40 * row + 20], [0, 40 * row + 20]]


def _interpret(
    regions: tuple[Region, ...],
    frames: dict[float, dict[str, list[str]]],
    tags: tuple[str, ...] = (),
    phrases: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Events of invented OCR reads: ``frames`` maps a time (s, at 1 frame a second) to the
    lines each region reads then, one row per line."""
    data = {"game": "tilefall", "regions": [], "banner_phrases": list(phrases)}
    interp = OcrInterpretation(
        "rec.mp4",
        OcrParams(game="tilefall", fps=1.0, player_tags=tags),
        preset_from_dict(data, None),
        regions,
        (1920, 1080),
        {},
    )
    for n in range(int(max(frames)) + 1):
        recs = {}
        for r in regions:
            lines = frames.get(float(n), {}).get(r.name, [])
            items = [(_box(i), line, 0.95) for i, line in enumerate(lines)]
            recs[r.name] = {"g": "read", "items": items}
        interp.add(FrameRec(n=n, t=float(n), regions=recs))
    return interp.finish()


# --------------------------------------------------------------------------- duplicate kills


def test_ocr_variants_of_one_row_are_one_kill() -> None:
    evs = [
        _kill(10.0, 12.0, "Alderwick", "Boxer77", 0.9),
        _kill(10.7, 14.7, "Alderwlck", "Boxer7", 0.95),
    ]
    out = merge_kill_reads(evs)
    assert len(out) == 1
    assert (out[0]["t"], out[0]["end"]) == (10.0, 14.7)
    assert out[0]["sightings"] == 2
    # the longest read wins over the more confident, shorter one
    assert (out[0]["killer"], out[0]["victim"]) == ("Alderwick", "Boxer77")


def test_a_wrong_split_of_the_same_row_merges() -> None:
    # the split moved into the victim's name: neither side matches, the joined names do
    evs = [
        _kill(20.0, 24.0, "Stoneharrow", "vexlin42"),
        _kill(20.7, 21.0, "Stoneharrowvexlin", "42Quillmere"),
    ]
    out = merge_kill_reads(evs)
    assert len(out) == 1
    assert out[0]["victim"] == "42Quillmere"  # the longer read
    evs = [_kill(20.0, 24.0, "Alderw", "ickBoxer77"), _kill(21.3, 22.0, "Alderwick", "Boxer77")]
    assert len(merge_kill_reads(evs)) == 1


def test_same_killer_different_victims_stay_apart() -> None:
    evs = [
        _kill(30.0, 33.0, "LongKillerName78", "Kip"),
        _kill(30.7, 34.0, "LongKillerName78", "Zed"),
        _kill(31.3, 35.0, "LongKillerName78", "Boxer7"),
    ]
    assert len(merge_kill_reads(evs)) == 3


def test_same_victim_by_another_killer_stays_apart() -> None:
    evs = [_kill(30.0, 33.0, "Alderwick", "Boxer77"), _kill(30.7, 34.0, "Kippers", "Boxer77")]
    assert len(merge_kill_reads(evs)) == 2


def test_similar_victims_of_one_killer_stay_apart() -> None:
    # a wrong cut changes both sides: one side the same name and the other not is two kills
    evs = [_kill(30.0, 33.0, "Alderwick", "Boxer77"), _kill(31.5, 35.0, "Alderwick", "Boxer12")]
    assert len(merge_kill_reads(evs)) == 2


def test_kills_seen_in_the_same_frame_are_two_rows() -> None:
    a = _kill(10.0, 12.0, "Alderwick", "Boxer77") | {"_times": [10.0, 11.0, 12.0]}
    b = _kill(10.0, 12.0, "Alderwlck", "Boxer77") | {"_times": [10.0, 11.0, 12.0]}
    assert len(merge_kill_reads([a, b])) == 2
    # reads of a row that never share a frame are one kill
    c = _kill(10.7, 14.0, "Alderwlck", "Boxer77") | {"_times": [12.7, 13.0, 14.0]}
    assert len(merge_kill_reads([a, c])) == 1
    # a read that shares frames with one read of a group does not join it through another
    assert len(merge_kill_reads([a, c, b])) == 2


def test_two_rows_on_screen_together_stay_two_kills_through_the_tracker() -> None:
    tracker = EventTracker(8.0, 0.8)
    for n in range(7):
        for _ in range(2):  # two identical rows in every frame
            tracker.see(
                float(n),
                "kill",
                "ALDERWICK:KILL",
                exclusive=True,
                entry="icon",
                killer="Alderwick",
                victim="Boxer77",
            )
    assert len(merge_kill_reads(tracker.events)) == 2


def test_rows_on_screen_together_stay_two_kills_in_the_interpretation() -> None:
    same = ["Alderwick destroyed Boxer77", "Alderwick destroyed Boxer77"]
    other = ["Alderwick destroyed Boxer77", "Alderwick destroyed Kippers"]
    for rows in (same, other):
        events = _interpret((FEED,), {float(n): {"feed": rows} for n in range(7)})
        assert [e["type"] for e in events] == ["kill", "kill"]


def test_the_same_pair_again_after_the_row_is_gone_is_a_new_kill() -> None:
    evs = [_kill(40.0, 44.0, "Alderwick", "Boxer77"), _kill(52.0, 56.0, "Alderwick", "Boxer77")]
    assert len(merge_kill_reads(evs)) == 2


def test_merged_kill_keeps_the_flags_of_every_read() -> None:
    evs = [
        _kill(60.0, 63.0, "Alderwick", "Boxer77", player=False),
        _kill(60.7, 64.0, "Alderwlck", "Boxer77", 0.8, player=True, player_kill=True),
    ]
    (out,) = merge_kill_reads(evs)
    assert out["player"] and out["player_kill"] and not out["player_death"]


def test_other_events_pass_through_in_order() -> None:
    notice = {"t": 5.0, "end": 6.0, "type": "notice", "text": "Alderwick"}
    evs = [
        notice,
        _kill(10.0, 12.0, "Alderwick", "Boxer77"),
        _kill(10.7, 14.7, "Alderwlck", "Boxer7"),
    ]
    out = merge_kill_reads(evs)
    assert [e["type"] for e in out] == ["notice", "kill"]


# --------------------------------------------------------------------------- banner bursts

# a scoreboard line that holds a phrase in a player name: the banner's whole line stays the same
SCORE = "ACE Alderwick 12 4 8"


def _scoreboard(times: tuple[float, ...]) -> list[dict[str, Any]]:
    return [_banner(t, "ACE", SCORE) for t in times]


def test_text_that_stays_on_a_screen_is_one_event() -> None:
    out = collapse_banner_bursts(_scoreboard((100, 130, 160, 205, 250)))
    assert [e["t"] for e in out] == [100]
    assert out[0]["recurrences"] == 4


def test_ocr_variants_of_the_screen_text_count_as_one_text() -> None:
    evs = [
        _banner(100, "ACE", SCORE),
        _banner(140, "ACE", "ACE Alderw1ck 12 4 8"),
        _banner(180, "ACE", SCORE),
    ]
    assert len(collapse_banner_bursts(evs)) == 1


def test_announcing_banners_are_never_collapsed() -> None:
    # the phrase alone on its line, however often it comes
    evs = [_banner(t) for t in (100, 130, 160, 205, 250)]
    assert collapse_banner_bursts(evs) == evs
    # a phrase with another name after it each time
    names = ("Alderwick", "Boxer77", "Kippers", "Marigold")
    evs = [
        _banner(100 + 20 * i, "ELIMINATED", f"ELIMINATED {name}") | {"detail": name}
        for i, name in enumerate(names)
    ]
    assert collapse_banner_bursts(evs) == evs


def test_short_different_names_after_a_phrase_are_not_collapsed() -> None:
    evs = [
        _banner(100 + 10 * i, "ELIMINATED", f"ELIMINATED {name}")
        for i, name in enumerate(("Kip", "Kit", "Kim"))
    ]
    assert collapse_banner_bursts(evs) == evs


def test_short_names_after_a_consistently_misread_phrase_are_not_collapsed() -> None:
    # the phrase is read with the same wrong letter every time, so it is not on the line as is
    evs = [
        _banner(100 + 10 * i, "ELIMINATED", f"ELIMINATEO {name}")
        for i, name in enumerate(("Kip", "Kit", "Kim"))
    ]
    assert collapse_banner_bursts(evs) == evs


def test_scoreboard_with_a_misread_phrase_still_collapses() -> None:
    evs = [_banner(t, "ACE", "ACF Alderwick 12 4 8") for t in (100, 130, 160, 205)]
    out = collapse_banner_bursts(evs)
    assert [e["t"] for e in out] == [100]
    assert out[0]["recurrences"] == 3


def test_two_sightings_and_spaced_banners_are_kept() -> None:
    assert len(collapse_banner_bursts(_scoreboard((100, 150)))) == 2
    spaced = _scoreboard((100, 400, 700, 1000))
    assert len(collapse_banner_bursts(spaced)) == 4


def test_a_burst_ends_at_a_long_gap() -> None:
    evs = _scoreboard((100, 130, 160, 600, 630, 660))
    assert [e["t"] for e in collapse_banner_bursts(evs)] == [100, 600]


def test_different_banner_texts_are_not_one_burst() -> None:
    evs = [
        _banner(100, "ACE", SCORE),
        _banner(130, "DEFEAT", "DEFEAT Alderwick 12 4 8"),
        _banner(160, "ACE", SCORE),
    ]
    assert len(collapse_banner_bursts(evs)) == 3


def test_only_banners_are_collapsed() -> None:
    evs = [{"t": float(t), "end": t + 1.0, "type": "bigtext", "text": "ACE"} for t in (1, 20, 40)]
    assert collapse_banner_bursts(evs) == evs


def test_screen_text_and_announcements_through_the_interpretation() -> None:
    frames = {float(t): {"banner": [SCORE]} for t in (10, 40, 70, 100)}
    frames |= {float(t): {"banner": ["ELIMINATED"]} for t in (130, 150, 170, 190)}
    events = _interpret((BANNER,), frames, phrases=("ACE", "ELIMINATED"))
    banners = [e for e in events if e["type"] == "banner"]
    assert [(e["text"], e["t"]) for e in banners] == [
        ("ACE", 10.0),
        ("ELIMINATED", 130.0),
        ("ELIMINATED", 150.0),
        ("ELIMINATED", 170.0),
        ("ELIMINATED", 190.0),
    ]
    assert banners[0]["recurrences"] == 3


# --------------------------------------------------------------------------- system lines


def _system_line_events(notice_times: tuple[float, ...]) -> Interpreter:
    interp = Interpreter(OcrParams(game="tilefall", fps=1.0), 1.0)
    for t in notice_times:
        interp.tracker.see(
            t,
            "notice",
            "FRIENDLY FIRE IS NOW ACTIVE",
            text="Friendly Fire is now active",
            region="feed",
        )
    cuts = [("Friendly Fi", "reisnowactive"), ("Friendlyfire", "isnowactiv")]
    for t, (killer, victim) in zip((200.0, 300.0), cuts, strict=True):
        interp.tracker.see(
            t, "kill", f"{killer}:kill", entry="icon", killer=killer, victim=victim, region="feed"
        )
    interp.tracker.see(
        400.0,
        "kill",
        "ALDERWICK:KILL",
        entry="icon",
        killer="Alderwick",
        victim="Boxer77",
        region="feed",
    )
    return interp


def test_a_recurring_system_line_read_as_a_kill_is_dropped() -> None:
    interp = _system_line_events((10.0, 40.0, 70.0, 100.0, 130.0))
    interp.drop_notice_kills()
    kills = [e["killer"] for e in interp.tracker.events if e["type"] == "kill"]
    assert kills == ["Alderwick"]
    assert sum(e["type"] == "notice" for e in interp.tracker.events) == 5  # notices stay


def test_a_notice_that_does_not_recur_drops_no_kill() -> None:
    interp = _system_line_events((10.0, 40.0, 70.0))  # three reads: not a system line yet
    interp.drop_notice_kills()
    assert sum(e["type"] == "kill" for e in interp.tracker.events) == 3


def test_a_kill_read_also_as_a_notice_is_not_a_system_line() -> None:
    # one pair kills five times, each row also read as a notice (the icon gap was missed)
    interp = Interpreter(OcrParams(game="tilefall", fps=1.0), 1.0)
    for t in (10.0, 40.0, 70.0, 100.0, 130.0):
        interp.tracker.see(
            t, "notice", "ALDERWICK BOXER77", text="Alderwick Boxer77", region="feed"
        )
        interp.tracker.see(
            t + 1,
            "kill",
            f"ALDERWICK:KILL{t}",
            entry="icon",
            killer="Alderwick",
            victim="Boxer77",
            region="feed",
        )
    interp.drop_notice_kills()
    assert sum(e["type"] == "kill" for e in interp.tracker.events) == 5


def test_single_unknown_notices_are_no_system_line() -> None:
    interp = _system_line_events((10.0, 40.0, 70.0, 100.0, 130.0))
    for e in interp.tracker.events:
        if e["type"] == "notice":
            e["unknown"] = True
    interp.drop_notice_kills()
    assert sum(e["type"] == "kill" for e in interp.tracker.events) == 3


def test_a_phrase_with_a_score_is_not_a_screen() -> None:
    evs = [_banner(t, "HEADSHOT", "HEADSHOT +100") for t in (100, 130, 160)]
    assert collapse_banner_bursts(evs) == evs
