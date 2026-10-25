"""Golden tests for event types (scoring, tails, VLM prompt, cache keys).

They pin the behaviour before event types moved into the HUD preset data, so
the preset-driven defaults must reproduce today's values exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cutpoint.events import eventtypes, scoring, tails
from cutpoint.events import vlm_windows as vw
from cutpoint.events.common import Source
from cutpoint.events.killfeed import OcrParams
from cutpoint.events.regions import GENERIC_BANNERS, load_preset
from cutpoint.events.report import normalize


def _ocr(name: str, typ: str, text: str, t: float, **kw: Any) -> tuple[str, dict[str, Any]]:
    return name, {"kind": "ocr", "type": typ, "text": text, "t": t, "end": t + 3, **kw}


# Layer events by game shape: R6 (banners, multikill), WT (verb-based feed, log),
# no preset (audio only), as the layers emit them before events.json.
R6 = dict(
    [
        _ocr("r6_kill", "kill", "Ash > Bob", 10.0, player=True, player_kill=True, teamkill=False),
        _ocr("r6_teamkill", "kill", "Ash > Cy", 11.0, player=True, player_kill=True, teamkill=True),
        _ocr("r6_rff", "banner", "REVERSE FRIENDLY FIRE", 12.0),
        _ocr("r6_ace", "banner", "ACE", 13.0),
        _ocr("r6_match_point", "banner", "MATCH POINT", 14.0),
        _ocr("r6_double", "multikill", "DOUBLE KILL (Bob, Cy)", 15.0, player=True, kills=2),
        _ocr("r6_other_kill", "kill", "Dee > Eve", 16.0, player=False, teamkill=False),
    ]
)
WT = dict(
    [
        _ocr(
            "wt_kill",
            "kill",
            "me (T-34) > foe (Tiger)",
            20.0,
            verb="kill",
            player=True,
            player_kill=True,
            player_death=False,
            teamkill=False,
        ),
        _ocr(
            "wt_death",
            "kill",
            "foe (Tiger) > me (T-34)",
            21.0,
            verb="kill",
            player=True,
            player_kill=False,
            player_death=True,
            teamkill=False,
        ),
        _ocr("wt_log", "log", "me crashed", 22.0, verb="crash", player=True),
        _ocr("wt_banner", "banner", "TARGET DESTROYED", 23.0),
    ]
)
NONE = {
    "audio_laugh": {
        "kind": "audio",
        "t": 30.0,
        "end": 32.0,
        "label": "Laughter",
        "confidence": 0.5,
        "top_class": "Laughter",
        "peak_t": 31.0,
    },
    "audio_scream": {
        "kind": "audio",
        "t": 33.0,
        "end": 34.0,
        "label": "Scream",
        "confidence": 0.1,
        "top_class": "Scream",
        "peak_t": 33.5,
    },
    "audio_quiet": {
        "kind": "audio",
        "t": 35.0,
        "end": 36.0,
        "label": "Cheering",
        "confidence": 0.01,
        "top_class": "Cheering",
        "peak_t": 35.5,
    },
}
LAYER = {**R6, **WT, **NONE}
EVENTS = {k: normalize("a.mp4", v) for k, v in LAYER.items()}
CAND = {"start": 0.0, "end": 100.0}


def _boost(names: list[str]) -> tuple[float, dict[str, Any]]:
    return scoring.event_boost(CAND, [EVENTS[n] for n in names])


def _ev(laugh=0.0, scream=0.0, kills=0, tk=0, banners=0) -> dict[str, Any]:
    return {
        "laughter": laugh,
        "scream": scream,
        "player_kills": kills,
        "teamkills": tk,
        "banners": banners,
    }


@pytest.mark.parametrize(
    ("names", "boost", "evidence"),
    [
        # R6: 1 kill, teamkill + RFF (capped at 1 = 12), ACE + DOUBLE (2 banners = 12) -> 30 cap
        (list(R6), 30.0, _ev(kills=1, tk=2, banners=2)),
        # WT: kill and death both carry player (2 x 6), TARGET DESTROYED (6); logs do not count
        (list(WT), 18.0, _ev(kills=2, banners=1)),
        # no preset: laughter min(12, 30), scream min(6, 4)
        (list(NONE), 16.0, _ev(laugh=0.5, scream=0.1)),
        (list(LAYER), 30.0, _ev(0.5, 0.1, 3, 2, 3)),
        (["r6_kill", "r6_double"], 12.0, _ev(kills=1, banners=1)),
        (["r6_rff"], 12.0, _ev(tk=1)),
        (["r6_match_point", "r6_other_kill"], 0.0, _ev()),
        (["wt_death", "wt_log"], 6.0, _ev(kills=1)),
    ],
)
def test_event_boost(names, boost, evidence):
    assert _boost(names) == (boost, evidence)


def test_event_boost_window():
    late = {"start": 111.0, "end": 112.0, "kind": "audio", "label": "Laughter", "confidence": 1}
    assert scoring.event_boost(CAND, [late])[0] == 0.0
    assert scoring.event_boost(CAND, [{**late, "start": 109.0}])[0] == 12.0


PAYOFF = {
    "r6_kill": True,
    "r6_teamkill": True,
    "r6_rff": True,
    "r6_ace": True,
    "r6_match_point": False,
    "r6_double": True,
    "r6_other_kill": False,
    "wt_kill": True,
    "wt_death": True,
    "wt_log": True,
    "wt_banner": True,
    "audio_laugh": True,
    "audio_scream": True,
    "audio_quiet": False,
}
SEED = {
    "r6_kill": 1.2,
    "r6_teamkill": 2.0,
    "r6_rff": 2.0,
    "r6_ace": 2.0,
    "r6_match_point": None,
    "r6_double": 2.0,
    "r6_other_kill": 0.5,
    "wt_kill": 1.2,
    "wt_death": 1.2,
    "wt_log": 0.9,
    "wt_banner": 0.6,
    "audio_laugh": 1.5,
    "audio_scream": 0.5833333333333334,
    "audio_quiet": 0.32,
}


@pytest.mark.parametrize("name", list(LAYER))
def test_is_payoff(name):
    assert tails.is_payoff(EVENTS[name], tails.EventTailSettings()) is PAYOFF[name]


def test_event_tail_settings_default():
    assert tails.EventTailSettings().to_json() == {
        "enabled": True,
        "window": 10.0,
        "pad": 1.5,
        "max_ext": 14.0,
        "vlm_min": 6,
        "audio_min": 0.05,
    }


@pytest.mark.parametrize(
    ("end", "words", "events", "limit", "expected"),
    [
        # speech ends at 10: payoffs up to 20 count (kills, RFF, ACE, DOUBLE, wt_kill), cap 24
        (
            10.0,
            None,
            list(EVENTS),
            99.0,
            (
                22.5,
                "event payoff (OCR kill: me (T-34) > foe (Tiger) @20.0)",
                [
                    "OCR kill: Ash > Bob @10.0",
                    "OCR kill: Ash > Cy @11.0",
                    "OCR banner: REVERSE FRIENDLY FIRE @12.0",
                    "OCR banner: ACE @13.0",
                    "OCR multikill: DOUBLE KILL (Bob, Cy) @15.0",
                    "OCR kill: me (T-34) > foe (Tiger) @20.0",
                ],
            ),
        ),
        (  # limit caps the end
            10.0,
            None,
            list(EVENTS),
            12.0,
            (
                12.0,
                "event payoff (OCR kill: Ash > Cy @11.0)",
                ["OCR kill: Ash > Bob @10.0", "OCR kill: Ash > Cy @11.0"],
            ),
        ),
        (  # speech ends at 9: the same two kills are inside the window
            10.0,
            [{"start": 8.0, "end": 9.0}],
            ["r6_kill", "r6_teamkill"],
            99.0,
            (
                13.5,
                "event payoff (OCR kill: Ash > Cy @11.0)",
                ["OCR kill: Ash > Bob @10.0", "OCR kill: Ash > Cy @11.0"],
            ),
        ),
        (10.0, None, ["r6_match_point", "r6_other_kill"], 99.0, (10.0, None, [])),
    ],
)
def test_event_tail(end, words, events, limit, expected):
    evs = sorted((EVENTS[n] for n in events), key=lambda e: e["start"])
    assert tails.event_tail(0.0, end, words, evs, tails.EventTailSettings(), limit) == expected


def _src(tmp_path: Path) -> Source:
    return Source(tmp_path / "game.mp4", 30.0, tmp_path, [])


QWEN = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
FRAMES = (
    " recording, starting at 00:00:05.0 in game.mp4. The single audio track mixes game "
    "sound, the player's microphone and friends on voice chat. 0 frames follow, each "
    "labelled with seconds from the window start."
)
PROMPT_START = "Describe what actually happens in this window, in time order: what the player does, kills and deaths (who kills whom, with what), teamkills, round results, and how the friends react (laughing, shouting, banter, jokes). Do not list transcript lines one by one: summarise a conversation as ONE event and spend the other events on actions and reactions. Connect the talk to the action: if a conversation is followed by a kill, a death, a round win or laughter, say so explicitly ('after the argument, the player gets a kill and the team laughs'). Look at the frames for weapons, kills and deaths, and listen to the audio for laughter and tone. Use on-screen HUD text only as context; do not transcribe it.\nScoring rubric (integers 0-10):\n"
HYPE_R6 = "  hype: 0-1 walking/waiting, menus, loading, operator select, scoreboard, killcam, replay; 3-4 a single ordinary kill; 6-7 a round win, a multi-kill, a close fight; 8-10 a clutch (1 vs many), an ace, a spectacular kill.\n"
HYPE_GENERIC = "  hype: 0-1 driving/waiting, menus, hangar, loading, vehicle or respawn select, scoreboard, killcam, replay; 3-4 a single ordinary kill; 6-7 a multi-kill, a big explosion, a close call, a match win; 8-10 several kills in quick succession, a spectacular kill or a spectacular death.\n"
PROMPT_END = '  funny: 0-1 no joke; 3-4 light banter; 6-7 a clear joke or people laughing; 8-10 everyone laughing hard, an absurd mistake, a teamkill followed by laughter.\n  highlight_score: how much this window deserves a place in a highlight montage (max of the best moment\'s hype and funny, plus 1 if both).\nEvent \'t\' is seconds from the window start. Reply with JSON only: {"events": [{"t": number, "what": str, "funny": int, "hype": int}], "summary": str, "funny_score": int, "highlight_score": int, "label": "2-4 word slug"}'


@pytest.mark.parametrize(
    ("game", "intro", "hype"),
    [
        ("r6", "This is a 15-second window of a first-person Rainbow Six Siege", HYPE_R6),
        ("wt", "This is a 15-second window of a War Thunder", HYPE_GENERIC),
        ("some-game", "This is a 15-second window of a Some Game", HYPE_GENERIC),
        (None, "This is a 15-second window of a video game", HYPE_GENERIC),
    ],
)
def test_prompt(tmp_path, game, intro, hype):
    win = vw.Window("game.mp4", 5.0, 20.0, 1.0, [])
    msgs = vw.build_messages(_src(tmp_path), win, [], vw.VlmParams(), game, [], [])
    transcript = (
        "Speech transcript (automatic, may contain errors; seconds from window start):\n(none)"
    )
    assert msgs == [
        {"role": "system", "content": QWEN},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": intro + FRAMES},
                {"type": "text", "text": transcript},
                {"type": "text", "text": PROMPT_START + hype + PROMPT_END},
            ],
        },
    ]


@pytest.mark.parametrize("name", list(LAYER))
def test_seed_priority(name):
    assert vw._seed_priority(LAYER[name]) == pytest.approx(SEED[name])


def test_hints():
    evs = [{**v, "t": v["t"] - 10} for v in LAYER.values()]
    assert vw._hints(evs, 0.0, 40.0) == (
        "0.0s on-screen kill: Ash > Bob\n"
        "1.0s on-screen kill: Ash > Cy (TEAMKILL)\n"
        "2.0s on-screen banner: REVERSE FRIENDLY FIRE\n"
        "3.0s on-screen banner: ACE\n"
        "4.0s on-screen banner: MATCH POINT\n"
        "5.0s on-screen multikill: DOUBLE KILL (Bob, Cy)\n"
        "6.0s on-screen kill: Dee > Eve\n"
        "10.0s on-screen kill: me (T-34) > foe (Tiger) (the PLAYER's kill)\n"
        "11.0s on-screen kill: foe (Tiger) > me (T-34) (the PLAYER died)\n"
        "12.0s on-screen log: me crashed\n"
        "13.0s on-screen banner: TARGET DESTROYED\n"
        "31.0s audio detector: Laughter (0.50)\n"
        "33.5s audio detector: Scream (0.10)\n"
        "35.5s audio detector: Cheering (0.01)"
    )


def test_vlm_key():
    assert vw.VlmParams().key() == {
        "frames": 14,
        "h": 720,
        "imt": 640,
        "audio": True,
        "transcript": True,
        "hints": True,
        "prompt": 5,
        "model": "qwen2.5-omni-7b-q4km",
    }


R6_KEY = {
    "extra": {},
    "fps": 3.0,
    "fuzzy": 0.8,
    "game": "r6",
    "min_text_score": 0.6,
    "persist_s": 8.0,
    "regions": [
        {
            "h": 185,
            "kind": "killfeed",
            "min_change": 0.008,
            "min_fill": 0.002,
            "name": "killfeed",
            "w": 560,
            "x": 1300,
            "y": 150,
        },
        {
            "h": 110,
            "kind": "counter",
            "min_change": 0.015,
            "min_fill": 0.002,
            "name": "counter",
            "w": 400,
            "x": 760,
            "y": 200,
        },
        {
            "h": 220,
            "kind": "banner",
            "min_change": 0.03,
            "min_fill": 0.012,
            "name": "banner",
            "w": 880,
            "x": 520,
            "y": 380,
        },
        {
            "h": 80,
            "kind": "notice",
            "min_change": 0.03,
            "min_fill": 0.006,
            "name": "notice",
            "w": 720,
            "x": 600,
            "y": 740,
        },
    ],
    "tags": [],
    "v": 3,
}

WT_KEY = {
    "extra": {},
    "fps": 3.0,
    "fuzzy": 0.8,
    "game": "wt",
    "min_text_score": 0.6,
    "persist_s": 8.0,
    "preset": {
        "bigtext": True,
        "feed": {
            "colors": {
                "blue": "ally",
                "cyan": "ally",
                "green": "squad",
                "orange": "self",
                "purple": "ally",
                "red": "enemy",
                "yellow": "self",
            },
            "entry": "paren",
            "row_tol": 10,
        },
        "interp": 6,
        "phrases": [
            "YOU HAVE BEEN DESTROYED",
            "CREW KNOCKED OUT",
            "TARGET DESTROYED",
            "SHOT DOWN",
            "SET AFIRE",
            "CRITICALLY DAMAGED",
            "SEVERELY DAMAGED",
            "MISSION ACCOMPLISHED",
            "MISSION FAILED",
            "ZONE CAPTURED",
            "MULTI STRIKE",
            "TANK RESCUER",
            "WITHOUT A SCRATCH",
            "TERMINATOR",
            "AVENGER",
            "FINAL BLOW",
            "ONE SHOT ONE KILL",
            "STREAK",
            "VICTORY",
            "DEFEAT",
            "MISSION COMPLETE",
            "YOU DIED",
            "YOU WIN",
            "YOU LOSE",
            "WASTED",
            "ELIMINATED",
            "DESTROYED",
            "KILLED",
            "ACE",
            "ROUND WON",
            "ROUND LOST",
            "WON ROUND",
            "LOST ROUND",
            "LEVEL UP",
            "ACHIEVEMENT",
            "ACHIEVEMENT UNLOCKED",
            "DOUBLE KILL",
            "TRIPLE KILL",
            "MULTI KILL",
            "KILLING SPREE",
            "HEADSHOT",
            "FIRST BLOOD",
            "WINNER",
            "GAME OVER",
            "MATCH POINT",
            "OVERTIME",
            "DRAW",
            "CHAMPION",
            "RANK UP",
            "NEW RECORD",
        ],
        "preset_tags": [],
    },
    "regions": [
        {
            "det_scale": 0.6,
            "gate": "det",
            "h": 258,
            "kind": "feed",
            "min_change": 0.015,
            "min_fill": 0.002,
            "name": "killfeed",
            "w": 580,
            "x": 1340,
            "y": 430,
        },
        {
            "det_scale": 0.5,
            "every": 6,
            "gate": "det",
            "h": 270,
            "kind": "notice",
            "min_change": 0.015,
            "min_fill": 0.002,
            "name": "awards",
            "w": 1200,
            "x": 360,
            "y": 160,
        },
        {
            "det_scale": 0.5,
            "every": 6,
            "gate": "det",
            "h": 210,
            "kind": "notice",
            "min_change": 0.015,
            "min_fill": 0.002,
            "name": "messages",
            "w": 960,
            "x": 480,
            "y": 760,
        },
        {
            "det_scale": 0.35,
            "every": 6,
            "gate": "det",
            "h": 700,
            "kind": "bigtext",
            "min_change": 0.015,
            "min_fill": 0.002,
            "min_text_h": 36,
            "name": "bigtext",
            "w": 1320,
            "x": 300,
            "y": 110,
        },
    ],
    "tags": [],
    "v": 3,
}


@pytest.mark.parametrize(("game", "expected"), [("r6", R6_KEY), ("wt", WT_KEY)])
def test_ocr_key(game, expected):
    assert OcrParams().key(game, load_preset(game)) == expected


# --- preset-defined event types ------------------------------------------------

DEFUSER = {
    "name": "defuser",
    "when": [{"type": "banner", "prefix": ["DEFUSER PLANTED"]}],
    "score": 6,
    "max": 2,
    "payoff": True,
    "tail_s": 3,
    "seed": 1.5,
}


def _custom(tmp_path: Path, events: list[Any], **extra: Any) -> Path:
    """A project with a ``foo`` HUD preset; returns the project folder."""
    project = tmp_path / "proj"
    (project / "hud_presets").mkdir(parents=True, exist_ok=True)
    data = {"game": "foo", "regions": [], "banner_phrases": ["MY BANNER"], "events": events}
    (project / "hud_presets" / "foo.json").write_text(
        json.dumps({**data, **extra}), encoding="utf-8"
    )
    return project


def test_custom_event_type(tmp_path, home):
    project = _custom(tmp_path, [DEFUSER])
    preset = load_preset("foo", project)
    assert preset.phrases == ("MY BANNER", "DEFUSER PLANTED", *GENERIC_BANNERS)
    types = eventtypes.types_for(preset)
    raw = {"kind": "ocr", "type": "banner", "text": "DEFUSER PLANTED 2", "t": 12.0, "end": 15.0}
    ev = normalize("a.mp4", raw)

    boost, evidence = scoring.event_boost(CAND, [ev], types)
    assert boost == 6.0 and evidence["defuser"] == 1
    assert scoring.event_boost(CAND, [ev])[0] == 0.0  # defaults know nothing of it

    assert vw._seed_priority(raw, types) == 1.5
    assert vw._seed_priority(raw) == 0.6  # the built-in banner rule

    # payoff 12 + hold 1.0 + tail_s 3 = 16 (the default pad would give 14.5)
    (events_json := project / "events.json").write_text(
        json.dumps({"settings": {"ocr": {"game": "foo"}}, "events": [{**ev, "source": "a.mp4"}]}),
        encoding="utf-8",
    )
    loaded = tails.load_events(project)["a.mp4"]
    assert tails.is_payoff(loaded[0], tails.EventTailSettings())
    end = tails.event_tail(0.0, 10.0, None, loaded, tails.EventTailSettings(), 99.0)[0]
    assert end == 16.0
    assert events_json.exists()


def test_event_type_clamping_and_malformed(tmp_path, home):
    big = {**DEFUSER, "name": "big", "score": 60, "max": 9}
    bad = [{"name": "nowhen"}, {"name": "x", "when": [{"type": "banner"}], "score": "lots"}, 5]
    preset = load_preset("foo", _custom(tmp_path, [*bad, big, DEFUSER]))
    by_name = {t.name: t for t in eventtypes.types_for(preset)}
    assert (by_name["big"].score, by_name["big"].max) == (12.0, 3)
    assert "defuser" in by_name and "player_kills" in by_name  # the rest still loads
    assert "nowhen" not in by_name and "x" not in by_name


def test_resolve_never_raises(tmp_path, home):
    ev = {"source": str(tmp_path / "gone.mp4"), "kind": "ocr"}
    for data in (
        {},
        {"settings": None, "events": [ev]},
        {"settings": {"ocr": {"game": "nope"}}, "events": [ev]},
    ):
        got = eventtypes.resolve(tmp_path, data)
        assert all(v == eventtypes.DEFAULT_EVENT_TYPES for v in got.values())
    project = _custom(tmp_path, [DEFUSER])
    (project / "hud_presets" / "foo.json").write_text("{not json", encoding="utf-8")
    data = {"settings": {"ocr": {"game": "foo"}}, "events": [ev]}
    assert eventtypes.resolve(project, data)[ev["source"]] == eventtypes.DEFAULT_EVENT_TYPES


def test_vlm_gp_key_only_for_prompt_customisation(tmp_path, home):
    base = vw.VlmParams().key()
    plain = load_preset("foo", _custom(tmp_path, [], bigtext=False))
    assert vw.VlmParams().key(vw.game_prompt("foo", plain)) == base  # discovered-style
    for extra in ({"hype": "  hype: x\n"}, {"title": "Foo Game"}, {"view": "third-person"}):
        p = load_preset("foo", _custom(tmp_path, [], **extra))
        assert "gp" in vw.VlmParams().key(vw.game_prompt("foo", p))
    seeded = load_preset("foo", _custom(tmp_path, [DEFUSER]))
    assert "gp" in vw.VlmParams().key(vw.game_prompt("foo", seeded))
    unseeded = load_preset("foo", _custom(tmp_path, [{**DEFUSER, "seed": None}]))
    assert "gp" not in vw.VlmParams().key(vw.game_prompt("foo", unseeded))
