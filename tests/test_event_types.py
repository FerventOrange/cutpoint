"""Golden tests for event types (scoring, tails, VLM prompt, cache keys).

They pin the behaviour before event types moved into the HUD preset data, so
the preset-driven defaults reproduce the original values, except that the
player's own deaths (generic and built-in feeds alike) no longer count as player kills
and a self-kill is ``self_kills`` (half a kill), not a player kill.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from cutpoint import cache, paths
from cutpoint.events import eventtypes, regions, scoring, tails
from cutpoint.events import vlm_windows as vw
from cutpoint.events.common import Source
from cutpoint.events.killfeed import GENERIC_VERSION, FeedParser, OcrParams, derive_streaks
from cutpoint.events.regions import GENERIC_BANNERS, load_preset
from cutpoint.events.report import normalize
from hud_fixtures import boxfeed, verbfeed, write_preset


def _ocr(name: str, typ: str, text: str, t: float, **kw: Any) -> tuple[str, dict[str, Any]]:
    return name, {"kind": "ocr", "type": typ, "text": text, "t": t, "end": t + 3, **kw}


# Layer events by game shape: box feed (banners, multikill), verb feed (verb-based feed, log),
# no preset (audio only), as the layers emit them before events.json.
BOXFEED = dict(
    [
        _ocr(
            "boxfeed_kill", "kill", "Ash > Bob", 10.0, player=True, player_kill=True, teamkill=False
        ),
        _ocr(
            "boxfeed_teamkill",
            "kill",
            "Ash > Cy",
            11.0,
            player=True,
            player_kill=True,
            teamkill=True,
        ),
        _ocr("boxfeed_rff", "banner", "REVERSE FRIENDLY FIRE", 12.0),
        _ocr("boxfeed_ace", "banner", "ACE", 13.0),
        _ocr("boxfeed_match_point", "banner", "MATCH POINT", 14.0),
        _ocr("boxfeed_double", "multikill", "DOUBLE KILL (Bob, Cy)", 15.0, player=True, kills=2),
        _ocr("boxfeed_other_kill", "kill", "Dee > Eve", 16.0, player=False, teamkill=False),
        _ocr(
            "boxfeed_death",
            "kill",
            "Bob > Ash",
            17.0,
            player=True,
            player_kill=False,
            player_death=True,
            teamkill=False,
        ),
    ]
)
VERBFEED = dict(
    [
        _ocr(
            "verbfeed_kill",
            "kill",
            "me (Cube) > foe (Orb)",
            20.0,
            verb="kill",
            player=True,
            player_kill=True,
            player_death=False,
            teamkill=False,
        ),
        _ocr(
            "verbfeed_death",
            "kill",
            "foe (Orb) > me (Cube)",
            21.0,
            verb="kill",
            player=True,
            player_kill=False,
            player_death=True,
            teamkill=False,
        ),
        _ocr("verbfeed_log", "log", "me crashed", 22.0, verb="crash", player=True),
        _ocr("verbfeed_banner", "banner", "TARGET DESTROYED", 23.0),
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
LAYER = {**BOXFEED, **VERBFEED, **NONE}
EVENTS = {k: normalize("a.mp4", v) for k, v in LAYER.items()}
CAND = {"start": 0.0, "end": 100.0}


def _boost(names: list[str]) -> tuple[float, dict[str, Any]]:
    return scoring.event_boost(CAND, [EVENTS[n] for n in names])


def _ev(laugh=0.0, scream=0.0, kills=0, tk=0, banners=0, self_kills=0) -> dict[str, Any]:
    return {
        "laughter": laugh,
        "scream": scream,
        "player_kills": kills,
        "teamkills": tk,
        "banners": banners,
        "self_kills": self_kills,
    }


@pytest.mark.parametrize(
    ("names", "boost", "evidence"),
    [
        # boxfeed: 1 kill (the own death does not count), teamkill + RFF (capped at 1 = 12), ACE + DOUBLE (2 banners = 12) -> 30 cap
        (list(BOXFEED), 30.0, _ev(kills=1, tk=2, banners=2)),
        # verbfeed: only the player's kill counts (6), not the death; TARGET DESTROYED (6); logs do not count
        (list(VERBFEED), 12.0, _ev(kills=1, banners=1)),
        # no preset: laughter min(12, 30), scream min(6, 4)
        (list(NONE), 16.0, _ev(laugh=0.5, scream=0.1)),
        (list(LAYER), 30.0, _ev(0.5, 0.1, 2, 2, 3)),
        (["boxfeed_kill", "boxfeed_double"], 12.0, _ev(kills=1, banners=1)),
        (["boxfeed_rff"], 12.0, _ev(tk=1)),
        (["boxfeed_match_point", "boxfeed_other_kill"], 0.0, _ev()),
        (["boxfeed_death"], 0.0, _ev()),  # the player's own death in a built-in feed is no kill
        (["verbfeed_death", "verbfeed_log"], 0.0, _ev()),
    ],
)
def test_event_boost(names, boost, evidence):
    assert _boost(names) == (boost, evidence)


@pytest.mark.parametrize(
    ("flag", "fields", "expected"),
    [
        ("player !verb", {"player": True}, True),  # AND, negated flag false
        ("player !verb", {"player": True, "verb": "kill"}, False),  # negation fails
        ("player !verb", {"verb": "kill"}, False),  # AND, first flag missing
        ("player_kill", {"player_kill": True, "player": True}, True),  # single flag
        ("player_kill", {"player": True}, False),
        ("player_death !player_kill", {"player_death": True}, True),
        ("player_death !player_kill", {"player_death": True, "player_kill": True}, False),
        ("player_death !player_kill", {"player_kill": True}, False),
        ("", {}, True),  # empty flag matches any event of the type
    ],
)
def test_matcher_multi_flag(flag, fields, expected):
    ev = _ocr("e", "kill", "a > b", 1.0, **fields)[1]
    assert eventtypes.Matcher("kill", flag=flag).matches(ev) is expected
    # events.json keeps the layer fields under "details"
    assert eventtypes.Matcher("kill", flag=flag).matches({**ev, "details": fields}) is expected


@pytest.mark.parametrize(
    ("fields", "boost", "evidence"),
    [
        # a self-kill (player is killer and victim) is its own type: half a kill, not a kill
        ({"player_kill": True, "player_death": True}, 3.0, _ev(self_kills=1)),
        # a teamkill that kills the player is still a teamkill
        ({"player_death": True, "teamkill": True}, 12.0, _ev(tk=1)),
        # teamkills come first: a self-kill flagged as a teamkill is a teamkill
        ({"player_kill": True, "player_death": True, "teamkill": True}, 12.0, _ev(tk=1)),
    ],
)
def test_builtin_kill_flag_combinations(fields, boost, evidence):
    ev = normalize("a.mp4", _ocr("e", "kill", "zed > zed", 1.0, player=True, **fields)[1])
    assert scoring.event_boost(CAND, [ev]) == (boost, evidence)


@pytest.mark.parametrize("style", [None, "funny", "gaming"])
def test_self_kills_capped_like_kills(style):
    """3 points per self-kill in every style, at most 3 counted, like kills."""
    ev = normalize(
        "a.mp4",
        _ocr("e", "kill", "zed > zed", 1.0, player=True, player_kill=True, player_death=True)[1],
    )
    boost, evidence = scoring.event_boost(CAND, [ev] * 4, style=style)
    assert boost == 9.0 and evidence["self_kills"] == 4 and evidence["player_kills"] == 0


def test_verb_feed_self_kill_is_self_kill():
    """A generic verb feed row where the player kills themselves lands in self_kills."""
    parser = FeedParser({"kill_words": {"toppled": "kill"}}, ("BOXER",), 1.0)
    raw = parser._event("BOXER", "BOXER", "kill", None, None, 0.9, "BOXER toppled BOXER")
    assert raw["player_kill"] and raw["player_death"]
    ev = normalize("a.mp4", {"kind": "ocr", "t": 1.0, "end": 4.0, **raw})
    assert eventtypes.first_match(eventtypes.DEFAULT_EVENT_TYPES, ev).name == "self_kills"
    assert scoring.event_boost(CAND, [ev]) == (3.0, _ev(self_kills=1))


@pytest.mark.parametrize(
    ("label", "teamkills"),
    [("REVERSE FRIENDLY FIRE", 1), ("NOT REVERSE FRIENDLY FIRE", 0)],
)
def test_rff_banner_is_prefix(label, teamkills):
    ev = normalize("a.mp4", _ocr("b", "banner", label, 1.0)[1])
    assert scoring.event_boost(CAND, [ev])[1]["teamkills"] == teamkills


def test_event_boost_window():
    late = {"start": 111.0, "end": 112.0, "kind": "audio", "label": "Laughter", "confidence": 1}
    assert scoring.event_boost(CAND, [late])[0] == 0.0
    assert scoring.event_boost(CAND, [{**late, "start": 109.0}])[0] == 12.0


PAYOFF = {
    "boxfeed_kill": True,
    "boxfeed_teamkill": True,
    "boxfeed_rff": True,
    "boxfeed_ace": True,
    "boxfeed_match_point": False,
    "boxfeed_double": True,
    "boxfeed_other_kill": False,
    "boxfeed_death": True,
    "verbfeed_kill": True,
    "verbfeed_death": True,
    "verbfeed_log": True,
    "verbfeed_banner": True,
    "audio_laugh": True,
    "audio_scream": True,
    "audio_quiet": False,
}
SEED = {
    "boxfeed_kill": 1.2,
    "boxfeed_teamkill": 2.0,
    "boxfeed_rff": 2.0,
    "boxfeed_ace": 2.0,
    "boxfeed_match_point": None,
    "boxfeed_double": 2.0,
    "boxfeed_other_kill": 0.5,
    "boxfeed_death": 1.2,
    "verbfeed_kill": 1.2,
    "verbfeed_death": 1.2,
    "verbfeed_log": 0.9,
    "verbfeed_banner": 0.6,
    "audio_laugh": 1.5,
    "audio_scream": 0.5833333333333334,
    "audio_quiet": 0.32,
}


@pytest.mark.parametrize("name", list(LAYER))
def test_is_payoff(name):
    assert tails.is_payoff(EVENTS[name], tails.EventTailSettings()) is PAYOFF[name]


@pytest.mark.parametrize(
    ("end", "words", "events", "limit", "expected"),
    [
        # speech ends at 10: payoffs up to 20 count (kills, RFF, ACE, DOUBLE, boxfeed_death, verbfeed_kill), cap 24
        (
            10.0,
            None,
            list(EVENTS),
            99.0,
            (
                22.5,
                "event payoff (OCR kill: me (Cube) > foe (Orb) @20.0)",
                [
                    "OCR kill: Ash > Bob @10.0",
                    "OCR kill: Ash > Cy @11.0",
                    "OCR banner: REVERSE FRIENDLY FIRE @12.0",
                    "OCR banner: ACE @13.0",
                    "OCR multikill: DOUBLE KILL (Bob, Cy) @15.0",
                    "OCR kill: Bob > Ash @17.0",
                    "OCR kill: me (Cube) > foe (Orb) @20.0",
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
            ["boxfeed_kill", "boxfeed_teamkill"],
            99.0,
            (
                13.5,
                "event payoff (OCR kill: Ash > Cy @11.0)",
                ["OCR kill: Ash > Bob @10.0", "OCR kill: Ash > Cy @11.0"],
            ),
        ),
        (10.0, None, ["boxfeed_match_point", "boxfeed_other_kill"], 99.0, (10.0, None, [])),
    ],
)
def test_event_tail(end, words, events, limit, expected):
    evs = sorted((EVENTS[n] for n in events), key=lambda e: e["start"])
    assert tails.event_tail(0.0, end, words, evs, tails.EventTailSettings(), limit) == expected


def _src(tmp_path: Path) -> Source:
    return Source(tmp_path / "game.mp4", 30.0, tmp_path, [])


HYPE_BOXFEED = boxfeed()["hype"]
HYPE_GENERIC = "  hype: 0-1 driving/waiting, menus, hangar, loading, vehicle or respawn select, scoreboard, killcam, replay; 3-4 a single ordinary kill; 6-7 a multi-kill, a big explosion, a close call, a match win; 8-10 several kills in quick succession, a spectacular kill or a spectacular death.\n"


QWEN = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
PROMPT_PIN = (5, "fc82230bf6ad")


@pytest.mark.parametrize(
    ("game", "intro", "hype", "bare"),
    [
        ("boxfeed", "This is a 15-second window of a top-down Box Feed", HYPE_BOXFEED, False),
        ("verbfeed", "This is a 15-second window of a top-down Verb Feed", HYPE_BOXFEED, False),
        # a title but no view/hype: the intro has no view, the rubric is the generic one
        ("verbfeed", "This is a 15-second window of a Verb Feed", HYPE_GENERIC, True),
        ("some-game", "This is a 15-second window of a Some Game", HYPE_GENERIC, False),
        (None, "This is a 15-second window of a video game", HYPE_GENERIC, False),
    ],
)
def test_prompt(tmp_path, home, game, intro, hype, bare):
    verb = verbfeed()
    if bare:
        verb = {k: v for k, v in verb.items() if k not in ("view", "hype")}
    for preset in (boxfeed(), verb):
        write_preset(paths.presets_dir(), preset)
    win = vw.Window("game.mp4", 5.0, 20.0, 1.0, [])
    msgs = vw.build_messages(
        _src(tmp_path), win, [], vw.VlmParams(), vw.game_prompt(game, load_preset(game)), [], []
    )
    system, user = msgs
    assert system == {"role": "system", "content": vw.QWEN_SYSTEM}
    assert vw.QWEN_SYSTEM == QWEN
    intro_part, transcript_part, rubric_part = (c["text"] for c in user["content"])
    assert user["role"] == "user" and all(c["type"] == "text" for c in user["content"])
    # the parts that vary with the preset: the window intro and the hype rubric
    assert intro_part.startswith(intro + " recording, starting at 00:00:05.0 in game.mp4.")
    assert "0 frames follow" in intro_part
    assert transcript_part.startswith("Speech transcript") and "(none)" in transcript_part
    assert hype in rubric_part
    assert (HYPE_GENERIC in rubric_part) == (hype == HYPE_GENERIC)
    # structure: the instructions, then the rubric (hype before funny), then the JSON reply format
    order = [rubric_part.index(x) for x in ("Scoring rubric", hype, "  funny:", "highlight_score:")]
    assert order == sorted(order)
    assert rubric_part.startswith("Describe what actually happens in this window")
    assert rubric_part.endswith('"label": "2-4 word slug"}')
    reply = rubric_part[rubric_part.index("Reply with JSON only") :]
    props = vw.RESPONSE_SCHEMA["properties"]
    for name in (*props, *props["events"]["items"]["properties"]):
        assert f'"{name}"' in reply, name
    # the fixed text is cached under PROMPT_VERSION only: a change must bump it
    fixed = "\n".join(
        [
            system["content"],
            intro_part.removeprefix(intro),
            transcript_part,
            rubric_part.replace(hype, "<hype>"),
        ]
    )
    digest = hashlib.sha1(fixed.encode("utf-8")).hexdigest()[:12]
    assert (vw.PROMPT_VERSION, digest) == PROMPT_PIN, (
        "prompt text changed: bump PROMPT_VERSION and update the digest"
    )


@pytest.mark.parametrize("name", list(LAYER))
def test_seed_priority(name):
    assert vw._seed_priority(LAYER[name]) == pytest.approx(SEED[name])
    # the default event types must not change a seed (own deaths keep the built-in rule)
    got = vw._seed_priority(LAYER[name], eventtypes.DEFAULT_EVENT_TYPES)
    assert got == pytest.approx(SEED[name])


@pytest.mark.parametrize("name", ["boxfeed_death", "verbfeed_death"])
def test_default_own_death_type_sets_nothing(name):
    t = eventtypes.first_match(eventtypes.DEFAULT_EVENT_TYPES, LAYER[name])
    assert t is not None
    assert t.name == "player_deaths"
    assert (t.payoff, t.tail_s, t.seed) == (None, None, None)


def test_default_self_kill_type_keeps_payoff_and_seed():
    """Like a death, a self-kill keeps the built-in tail payoff and VLM seed."""
    raw = {"kind": "ocr", "type": "kill", "text": "zed > zed", "t": 1.0, "end": 4.0}
    raw |= {"player": True, "player_kill": True, "player_death": True}
    t = eventtypes.first_match(eventtypes.DEFAULT_EVENT_TYPES, raw)
    assert t is not None and t.name == "self_kills"
    assert (t.payoff, t.tail_s, t.seed) == (None, None, None)
    assert vw._seed_priority(raw, eventtypes.DEFAULT_EVENT_TYPES) == pytest.approx(1.2)
    assert tails.is_payoff(normalize("a.mp4", raw), tails.EventTailSettings())


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


KEY_FIELDS = {
    "v",
    "game",
    "fps",
    "fuzzy",
    "min_text_score",
    "persist_s",
    "extra",
    "tags",
    "regions",
}
REGION_FIELDS = {"name", "x", "y", "w", "h", "kind", "min_change", "min_fill"}


BUILTIN_KEY_SHA1 = "1589f1e259b5864facdd747664ef08d46a3fdfa7"
PLAIN_KEY_SHA1 = "8bd31e1301f547575c5570c1945b64ee0e50f7f2"  # the plain preset's key without ``interp`` (its own version)


def test_ocr_key(home, tmp_path, monkeypatch):
    """The OCR cache key: a plain preset adds ``preset``; a ``builtin`` one keeps the original
    key layout (so its cached results stay valid), wherever it is loaded from."""
    write_preset(paths.presets_dir(), verbfeed())
    key = OcrParams().key("verbfeed", load_preset("verbfeed"))
    assert set(key) == KEY_FIELDS | {"preset"}
    assert (key["v"], key["game"], key["fps"], key["tags"]) == (3, "verbfeed", 3.0, [])
    preset = key["preset"]
    assert set(preset) == {"phrases", "feed", "bigtext", "preset_tags", "interp"}
    assert preset["interp"] == GENERIC_VERSION
    assert preset["phrases"] == ["VERB DONE", *GENERIC_BANNERS]
    assert preset["feed"] == {"kill_words": {"toppled": "kill"}}
    assert preset["preset_tags"] == ["BOXER"]
    # the region fields are the key too: gate/every/min_text_h/det_scale only on the bigtext region
    feed, banner, bigtext = key["regions"]
    assert (feed["name"], banner["name"], bigtext["name"]) == ("feed", "banner", "bigtext")
    assert set(feed) == set(banner) == REGION_FIELDS
    assert set(bigtext) == REGION_FIELDS | {"gate", "every", "min_text_h", "det_scale"}

    write_preset(paths.presets_dir(), boxfeed())
    from_user = OcrParams().key("boxfeed", load_preset("boxfeed"))
    assert set(from_user) == KEY_FIELDS and from_user["v"] == 3
    assert from_user["regions"] == key["regions"]
    # the values count too: any change renames every cache file of the built-in (R6) presets
    assert cache.params_hash(from_user, 40) == BUILTIN_KEY_SHA1, "never change"
    no_interp = {**key, "preset": {k: v for k, v in key["preset"].items() if k != "interp"}}
    assert cache.params_hash(no_interp, 40) == PLAIN_KEY_SHA1, "plain-preset cache file names"

    bundled = tmp_path / "bundled"
    write_preset(bundled, boxfeed())
    (paths.presets_dir() / "boxfeed.json").unlink()
    monkeypatch.setattr(paths, "PRESETS_DIR", bundled)
    assert OcrParams().key("boxfeed", load_preset("boxfeed")) == from_user


# --- preset-defined event types ------------------------------------------------

PLACER = {
    "name": "placer",
    "when": [{"type": "banner", "prefix": ["BOX PLACED"]}],
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
    project = _custom(tmp_path, [PLACER])
    preset = load_preset("foo", project)
    assert preset.phrases == ("MY BANNER", *GENERIC_BANNERS, "BOX PLACED")
    types = eventtypes.types_for(preset)
    raw = {"kind": "ocr", "type": "banner", "text": "BOX PLACED 2", "t": 12.0, "end": 15.0}
    ev = normalize("a.mp4", raw)

    boost, evidence = scoring.event_boost(CAND, [ev], types)
    assert boost == 6.0 and evidence["placer"] == 1
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
    big = {**PLACER, "name": "big", "score": 60, "max": 9, "tail_s": 99, "seed": 9}
    bad = [
        {"name": "nowhen"},
        {**PLACER, "name": "laughter"},  # reserved
        {**PLACER, "name": "yes", "payoff": "yes"},  # not a bool
        {"name": "x", "when": [{"type": "banner"}], "score": "lots"},
        5,
    ]
    preset = load_preset("foo", _custom(tmp_path, [*bad, big, PLACER]))
    by_name = {t.name: t for t in eventtypes.types_for(preset)}
    assert (by_name["big"].score, by_name["big"].max) == (12.0, 3)
    assert (by_name["big"].tail_s, by_name["big"].seed) == (10.0, 3.0)
    assert "placer" in by_name and "player_kills" in by_name  # the rest still loads
    assert not {"nowhen", "x", "laughter", "yes"} & by_name.keys()


@pytest.mark.parametrize("flag", ["! verb", "!!x"])
def test_malformed_flag_type_skipped(tmp_path, home, flag):
    bad = {"name": "bangs", "when": [{"type": "kill", "flag": flag}], "score": 3}
    preset = load_preset("foo", _custom(tmp_path, [bad, PLACER]))
    by_name = {t.name: t for t in eventtypes.types_for(preset)}
    assert "bangs" not in by_name
    assert "placer" in by_name and "player_kills" in by_name  # the defaults still load


def test_resolve_never_raises(tmp_path, home):
    ev = {"source": str(tmp_path / "gone.mp4"), "kind": "ocr"}
    for data in (
        {},
        {"settings": None, "events": [ev]},
        {"settings": {"ocr": {"game": "nope"}}, "events": [ev]},
    ):
        got = eventtypes.resolve(tmp_path, data)
        assert all(v == eventtypes.DEFAULT_EVENT_TYPES for v in got.values())
    project = _custom(tmp_path, [PLACER])
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
    seeded = load_preset("foo", _custom(tmp_path, [PLACER]))
    assert "gp" not in vw.VlmParams().key(vw.game_prompt("foo", seeded))  # seeds only


def test_vlm_gp_key_against_installed_reference(tmp_path, home):
    """A project copy of an installed preset keeps the VLM key; a changed prompt adds ``gp``."""

    def key(game: str, preset) -> dict[str, Any]:
        return vw.VlmParams().key(vw.game_prompt(game, preset))

    installed = verbfeed()
    write_preset(paths.presets_dir(), installed)
    same = tmp_path / "same"
    write_preset(same / "hud_presets", installed)
    assert "gp" not in key("verbfeed", load_preset("verbfeed", same))

    changed = tmp_path / "changed"
    write_preset(changed / "hud_presets", {k: v for k, v in installed.items() if k != "hype"})
    assert "gp" in key("verbfeed", load_preset("verbfeed", changed))


def test_hype_normalised(tmp_path, home):
    def prompt(hype: str) -> vw.GamePrompt:
        return vw.game_prompt("foo", load_preset("foo", _custom(tmp_path, [], hype=hype)))

    assert prompt("hype: x") == prompt("  hype: x\n")


def test_invalid_preset_warned_once(tmp_path, home, caplog, monkeypatch):
    monkeypatch.setattr(regions, "_warned_invalid", set())
    project = _custom(tmp_path, [])
    (project / "hud_presets" / "foo.json").write_text("{not json", encoding="utf-8")
    for _ in range(2):
        load_preset("foo", project)
    assert sum("Ignoring invalid HUD preset" in r.getMessage() for r in caplog.records) == 1


def test_preset_type_replaces_builtin(tmp_path, home):
    mine = {"name": "player_kills", "when": [{"type": "kill", "flag": "player"}],
            "score": 3, "payoff": False}  # fmt: skip
    project = _custom(tmp_path, [mine])
    types = eventtypes.types_for(load_preset("foo", project))
    kill = EVENTS["boxfeed_kill"]
    boost, evidence = scoring.event_boost(CAND, [kill], types)
    assert boost == 3.0 and evidence["player_kills"] == 1
    (project / "events.json").write_text(
        json.dumps({"settings": {"ocr": {"game": "foo"}}, "events": [{**kill, "source": "a.mp4"}]}),
        encoding="utf-8",
    )
    loaded = tails.load_events(project)["a.mp4"]
    assert not tails.is_payoff(loaded[0], tails.EventTailSettings())


def test_self_kill_is_no_streak_kill():
    """Two verb-feed kills make a multikill; a self-kill between them adds nothing to it."""

    def kill(t: float, victim: str) -> dict[str, Any]:
        own = victim == "BOXER"
        return {
            "type": "kill", "t": t, "text": f"BOXER > {victim}", "victim": victim, "verb": "kill",
            "player": True, "player_kill": True, "player_death": own,
        }  # fmt: skip

    events = [kill(1.0, "zed"), kill(5.0, "BOXER")]
    derive_streaks(events)
    assert not [e for e in events if e["type"] == "multikill"]
    events.append(kill(9.0, "yan"))
    derive_streaks(events)
    (multi,) = [e for e in events if e["type"] == "multikill"]
    assert multi["text"] == "DOUBLE KILL (zed, yan)" and multi["kills"] == 2
