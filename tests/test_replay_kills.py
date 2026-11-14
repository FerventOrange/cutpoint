"""Kills shown again in a replay (#1069): the replay marker's spans, the kills inside them
lose their player and teamkill flags, and the report says so. Invented events only."""

from __future__ import annotations

from typing import Any

from cutpoint.events.killfeed import (
    FrameRec,
    OcrInterpretation,
    OcrParams,
    link_teamkills,
    mark_replay_kills,
    replay_spans,
)
from cutpoint.events.regions import Region, marker_seen, preset_from_dict
from cutpoint.events.report import _event_line, normalize


def _replay(t: float, end: float, sightings: int | None = None) -> dict[str, Any]:
    ev: dict[str, Any] = {"type": "replay", "t": t, "end": end}
    if sightings is not None:
        ev["sightings"] = sightings
    return ev


def _kill(t: float, **flags: Any) -> dict[str, Any]:
    return {
        "type": "kill",
        "t": t,
        "killer": "Alder",
        "victim": "Boxer",
        "player": True,
        "player_kill": True,
        "player_death": True,
        "teamkill": True,
    } | flags


def test_a_replay_marker_read_once_and_gone_is_no_replay_but_two_reads_are() -> None:
    events = [_replay(10, 11, 1), _replay(20, 20), _replay(30, 34, 2), _kill(31)]
    assert replay_spans(events) == [(30.0, 34.0)]


def test_kills_in_a_replay_lose_their_flags_and_keep_their_event() -> None:
    events = [
        _kill(28.0),  # outside, even with the pad
        _kill(29.5),  # before the marker: live, even within the pad
        _kill(32.0),
        _kill(34.9),  # within the pad after it
        _kill(36.0, killer="Corvo", victim="Dune"),  # within the chain window after it
        _kill(50.0, killer="Corvo", victim="Eagle"),  # long after: live
        {"type": "banner", "t": 32.0, "end": 33.0},
    ]
    assert mark_replay_kills(events, [(30.0, 34.0)], pad=1.0) == 3
    marked = [e for e in events if e.get("replay")]
    assert [e["t"] for e in marked] == [32.0, 34.9, 36.0]
    assert events[5]["player_kill"] and not events[5].get("replay")
    for e in marked:
        assert not any(e[k] for k in ("player", "player_kill", "player_death", "teamkill"))
        assert e["killer"] and e["victim"]  # still a kill that happened
    for e in (events[0], events[1], events[5]):  # live kills are untouched
        assert e["player_kill"] and "replay" not in e
    assert "replay" not in events[5]


def test_marking_adds_no_flags_a_kill_never_had() -> None:
    kill = {"type": "kill", "t": 5.0}
    assert mark_replay_kills([kill], [(4.0, 6.0)], pad=0.5) == 1
    assert kill == {"type": "kill", "t": 5.0, "replay": True}


def test_no_spans_mark_nothing() -> None:
    events = [_kill(5.0)]
    assert mark_replay_kills(events, [], pad=1.0) == 0 and "replay" not in events[0]


def test_a_replayed_kill_between_allies_is_no_teamkill() -> None:
    ally = {"killer_team": "ally", "victim_team": "ally"}
    live = _kill(5.0, **ally)
    shown_again = _kill(40.0, replay=True, **ally)
    link_teamkills([live, shown_again], generic_teamkills=True)
    assert live["teamkill"] is True and shown_again["teamkill"] is False
    link_teamkills([live], generic_teamkills=False)
    assert live["teamkill"] is False


def test_the_report_shows_the_replay_flag() -> None:
    ev = {"kind": "ocr", "type": "kill", "t": 5.0, "killer": "Alder", "victim": "Boxer"}
    assert "(?>?)" in normalize("rec.mp4", ev)["description"]
    ev["replay"] = True
    norm = normalize("rec.mp4", ev)
    assert norm["description"].endswith("(?>?, replay)")
    assert norm["details"]["replay"] is True
    line = _event_line("rec.mp4", norm | {"label": "Alder > Boxer"})
    assert line.endswith("(replay)")


# ---------------------------------------------------------------- the marker and the reads


def test_a_still_marker_read_once_and_kept_on_screen_is_a_replay() -> None:
    # the det gate reads a still marker once and keeps it while its boxes stay
    events = [_replay(10, 10.5, 1), _replay(20, 24, 1), _kill(21)]
    assert replay_spans(events) == [(20.0, 24.0)]


def test_marker_text_matches_misreads_and_rejects_other_text() -> None:
    assert marker_seen("REPLAY", "replay")
    assert marker_seen("DREPLAY", "replay") and marker_seen("DREPL", "replay")
    assert not marker_seen("Attacker safeguards active", "replay")
    assert not marker_seen("Aptor", "replay") and not marker_seen("", "replay")


FEED_R = Region("feed", 1500, 100, 380, 140, "feed", gate="read")
REPLAY_R = Region("replay", 800, 60, 300, 60, "replay", gate="det", marker="replay")


def _box(row: int, x: int = 0, w: int = 300) -> list[list[int]]:
    return [[x, 40 * row], [x + w, 40 * row], [x + w, 40 * row + 20], [x, 40 * row + 20]]


def _interpret(
    frames: dict[int, dict[str, Any]], regions: tuple[Region, ...] = (FEED_R, REPLAY_R)
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Events and stats of invented reads at 1 frame a second: ``frames`` maps a second to
    each region's lines, or to "keep" (the det gate: the same text still on screen); a line
    given as a tuple is OCR items side by side on one row."""
    stats: dict[str, Any] = {}
    interp = OcrInterpretation(
        "rec.mp4",
        OcrParams(game="tilefall", fps=1.0, player_tags=("Alder",)),
        preset_from_dict({"game": "tilefall", "regions": []}, None),
        regions,
        (1920, 1080),
        stats,
    )
    for n in range(max(frames) + 1):
        recs = {}
        for r in regions:
            got = frames.get(n, {}).get(r.name, [])
            if got == "keep":
                recs[r.name] = {"g": "keep", "items": []}
            else:
                items = []
                for i, line in enumerate(got):
                    parts = line if isinstance(line, tuple) else (line,)
                    items += [(_box(i, 160 * j, 140), part, 0.95) for j, part in enumerate(parts)]
                recs[r.name] = {"g": "read", "items": items}
        interp.add(FrameRec(n=n, t=float(n), regions=recs))
    return interp.finish(), stats


def _kills(events: list[dict[str, Any]]) -> list[tuple[float, bool, bool]]:
    return [
        (e["t"], bool(e.get("player_kill")), bool(e.get("replay")))
        for e in events
        if e.get("type") == "kill"
    ]


def test_a_kill_read_while_the_marker_shows_is_no_kill_of_the_player() -> None:
    frames: dict[int, dict[str, Any]] = {
        5: {"feed": ["Alder killed Boxer"]},  # live
        # a still marker: read once, then kept by the det gate
        20: {"replay": ["REPLAY"]},
        **{n: {"replay": "keep"} for n in range(21, 26)},
        22: {"replay": "keep", "feed": ["Alder killed Kippers"]},  # shown again
    }
    events, stats = _interpret(frames)
    assert _kills(events) == [(5.0, True, False), (22.0, False, True)]
    assert stats["replays"] == 1 and stats["replay_kills"] == 1


def test_a_round_notice_where_the_marker_shows_is_no_replay() -> None:
    frames: dict[int, dict[str, Any]] = {
        20: {"replay": ["Attacker safeguards active"]},
        **{n: {"replay": "keep"} for n in range(21, 26)},
        22: {"replay": "keep", "feed": ["Alder killed Kippers"]},
    }
    events, stats = _interpret(frames)
    assert _kills(events) == [(22.0, True, False)]
    assert "replays" not in stats
    assert not [e for e in events if e.get("type") == "replay"]


def test_a_replay_region_without_a_marker_takes_any_text() -> None:
    plain = Region("replay", 800, 60, 300, 60, "replay", gate="det")
    frames: dict[int, dict[str, Any]] = {
        20: {"replay": ["Killcam"]},
        21: {"replay": ["Killcam"], "feed": ["Alder killed Kippers"]},
    }
    events, _ = _interpret(frames, (FEED_R, plain))
    assert _kills(events) == [(21.0, False, True)]


def test_names_read_again_and_again_where_the_marker_shows_are_no_replay() -> None:
    """The acceptance clip (#1069): a killcam-like card of names stood where the marker shows,
    read 11 times, around a live kill of the player; the marker itself came later, misread
    ("DREPLAY", "DREPL") in places. Only the marker's stretch is a replay."""
    frames: dict[int, dict[str, Any]] = {}
    for n in range(2, 13):  # names, read again and again
        frames[n] = {"replay": ["Aptorr leean518" if n % 2 else "AptOr leean518"]}
    frames[7]["feed"] = ["Alder killed Brightbritzka"]  # live
    for n in range(30, 40):
        frames[n] = {"replay": ["DREPLAY" if n % 3 else "DREPL"]}
    frames[33]["feed"] = ["Alder killed Kippers"]  # shown again
    events, stats = _interpret(frames)
    assert _kills(events) == [(7.0, True, False), (33.0, False, True)]
    assert stats["replays"] == 1 and stats["replay_kills"] == 1


def test_a_live_kill_right_before_the_replay_of_it_keeps_its_credit() -> None:
    """The game-winning kill (#1069 acceptance): read live, the round-end replay of it starts a
    second later and shows the same kill again. One player kill, the copy marked."""
    frames: dict[int, dict[str, Any]] = {
        10: {"feed": ["Alder killed Brightbritzka"]},  # live
        11: {"replay": ["REPLAY"]},
        **{n: {"replay": "keep"} for n in range(12, 26)},
    }
    frames[20]["feed"] = ["Alder killed Brightbritzka"]  # the replay shows it again, re-read
    events, _ = _interpret(frames)
    kills = [e for e in events if e.get("type") == "kill"]
    assert len(kills) == 2  # past PERSIST_S: not the same row seen again
    assert kills[0]["t"] == 10.0 and kills[0]["player_kill"] and not kills[0].get("replay")
    assert kills[1]["t"] == 20.0 and kills[1]["replay"] and kills[1]["replay_of"] == 10.0
    assert not kills[1]["player_kill"]


def test_a_replay_copy_points_at_its_live_kill() -> None:
    live, copy = _kill(5.0), _kill(31.0)
    assert mark_replay_kills([live, copy], [(30.0, 34.0)], 1.0) == 1
    assert copy["replay"] and copy["replay_of"] == 5.0 and not copy["player_kill"]
    assert live["player_kill"] and "replay_of" not in live and not live.get("replay")


def test_a_replay_goes_on_after_its_marker_with_another_point_of_view() -> None:
    """#1069 acceptance: the marker ended, the replay went on from a drone and showed the
    player's own death again, then other kills of the round. Those are no live kills."""
    death = _kill(100.0, killer="KingCulprit", victim="Alder", player_kill=False)
    other = _kill(101.0, killer="Flapjack", victim="Corvo", player=False, player_kill=False)
    death_copy = _kill(138.0, killer="KingCulprt", victim="Alder", player_kill=False)
    after = [
        _kill(134.3, killer="Leean518", victim="Brkout", player=False, player_kill=False),
        _kill(140.7, killer="Flapjack", victim="Bright", player=False, player_kill=False),
        _kill(146.7, killer="KingCulprit", victim="NickGud", player=False, player_kill=False),
    ]
    later = _kill(170.0, killer="Alder", victim="Dune", player_death=False)  # next round
    events = [death, other, death_copy, *after, later]
    assert mark_replay_kills(events, [(126.7, 132.7)], 1.0) == 4
    assert death["player_death"] and not death.get("replay")  # the live death keeps its flags
    assert death_copy["replay"] and death_copy["replay_of"] == 100.0
    assert not death_copy["player_death"] and not death_copy["player"]
    assert all(e["replay"] for e in after)
    assert later["player_kill"] and not later.get("replay")


def test_the_same_pair_twice_without_a_replay_is_two_kills() -> None:
    # a respawn game: no replay span between them, both stay
    first, again = _kill(10.0), _kill(40.0)
    assert mark_replay_kills([first, again], [], 1.0) == 0
    assert mark_replay_kills([first, again], [(60.0, 64.0)], 1.0) == 0
    assert first["player_kill"] and again["player_kill"]


def test_a_feed_row_in_the_replay_box_is_no_marker() -> None:
    """The replay box reached into the feed (#1069 review): a victim named "Playmaker" read
    there is no replay marker, so the live kill keeps its credit."""
    overlapping = Region("replay", 1500, 60, 380, 140, "replay", gate="det", marker="replay")
    frames: dict[int, dict[str, Any]] = {
        n: {"replay": ["Alder killed Playmaker"], "feed": ["Alder killed Playmaker"]}
        for n in range(10, 14)
    }
    events, stats = _interpret(frames, (FEED_R, overlapping))
    assert _kills(events) == [(10.0, True, False)]
    assert "replays" not in stats
    assert not marker_seen("Playmaker", "replay")


def test_a_replay_chain_stops_at_its_cap_on_a_busy_feed() -> None:
    """A respawn game (#1069 review): a killcam, then a kill every 5 s. The kills right after
    it chain on; past REPLAY_CHAIN_MAX_S after the marker they are live, the player's too."""
    events = [
        _kill(float(t), killer=f"K{t}", victim=f"V{t}", player=False, player_kill=False)
        for t in range(110, 255, 5)
    ]
    mine = [e for e in events if e["t"] in (160.0, 220.0)]
    for e in mine:
        e.update(killer="Alder", player=True, player_kill=True)
    assert mark_replay_kills(events, [(100.0, 105.0)], 1.0) == 4
    assert [e["t"] for e in events if e.get("replay")] == [110.0, 115.0, 120.0, 125.0]
    assert all(e["player_kill"] and not e.get("replay") for e in mine)


def test_a_copy_read_after_the_replay_marker_and_its_chain_points_at_its_live_kill() -> None:
    live, copy = _kill(10.0), _kill(31.0)  # span 20-22; 31 s is past the chain of 8 s
    assert mark_replay_kills([live, copy], [(20.0, 22.0)], 1.0) == 1
    assert copy["replay"] and copy["replay_of"] == 10.0 and not copy["player_kill"]
    assert live["player_kill"] and not live.get("replay")


def test_the_same_kill_again_long_after_a_killcam_is_a_new_kill() -> None:
    # Boxer killed at 10 s, a killcam at 20-25 s, Boxer killed again live at 50 s
    first, again = _kill(10.0), _kill(50.0)
    assert mark_replay_kills([first, again], [(20.0, 25.0)], 1.0) == 0
    assert first["player_kill"] and again["player_kill"] and "replay_of" not in again


def test_a_marker_read_in_two_parts_marks_a_replay() -> None:
    killcam = Region("replay", 800, 60, 300, 60, "replay", gate="det", marker="killcam")
    frames: dict[int, dict[str, Any]] = {
        20: {"replay": [("KILL", "CAM")]},
        21: {"replay": [("KILL", "CAM")], "feed": ["Alder killed Kippers"]},
    }
    events, stats = _interpret(frames, (FEED_R, killcam))
    assert _kills(events) == [(21.0, False, True)]
    assert stats["replays"] == 1
