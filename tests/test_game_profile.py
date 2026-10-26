"""Game profile builder, OCR backend (synthetic OCR samples, invented game ``zonewar``)."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import pytest

from cutpoint.events import discover
from cutpoint.events.game_profile import (
    HINT_MIN_BURSTS,
    HINT_SHARE,
    MIN_GAP_S,
    VERSION,
    build_profile,
    match_hint,
    mine_phrases,
)
from cutpoint.events.killfeed import OcrParams, fuzzy_contains
from cutpoint.events.regions import bigtext_region, preset_from_dict
from cutpoint.events.vlm_windows import game_prompt

BANNER = {"x": 600, "y": 300, "w": 700, "h": 120}  # banner region (reference px)


def _proposal() -> dict[str, Any]:
    return {
        "game": "zonewar",
        "reference": [1920, 1080],
        "source": "discovered",
        "regions": [
            {"name": "feed", "x": 1400, "y": 40, "w": 500, "h": 160, "kind": "feed"}
            | {"gate": "det", "every": 1},
            {"name": "banner", **BANNER, "kind": "banner", "gate": "det", "every": 3},
        ],
        "banner_phrases": [],
        # an older proposal's params: menu_boxes is no longer a param and is ignored
        "discovery": {"params": {"menu_boxes": 28}},
    }


def _box(text: str, x: int = 700, y: int = 330, h: int = 50) -> dict[str, Any]:
    return {"x0": x, "y0": y, "x1": x + 400, "y1": y + h, "text": text, "score": 0.9}


def _frames(texts_by_burst: dict[int, list[str]], frames_per_burst: int = 4) -> list[dict]:
    """Burst frames; burst ``b`` shows its texts in the banner on its first frame."""
    out = []
    for b in range(10):
        for k in range(frames_per_burst):
            boxes = [_box(t) for t in texts_by_burst.get(b, [])] if k == 0 else []
            out.append(
                {
                    "id": f"rec.mp4@{b * 60 + k * 1.5:.1f}",
                    "t": b * 60 + k * 1.5,
                    "source": "rec.mp4",
                }
                | {"burst": b, "k": k, "boxes": boxes}
            )
    return out


def test_phrase_needs_two_occurrences() -> None:
    frames = _frames({0: ["ZONE SECURED"], 3: ["ZONE SECURFD"], 5: ["BRIDGE FALLEN"]})
    assert mine_phrases(_proposal(), frames) == ["ZONE SECURED"]


def test_most_frequent_spelling_wins() -> None:
    # the typo sorts first alphabetically but is seen less often
    frames = _frames({0: ["ZONE SECURED"], 3: ["ZONE SECURED"], 6: ["ZONE SECURBD"]})
    assert mine_phrases(_proposal(), frames) == ["ZONE SECURED"]


def test_phrases_capped_and_ordered() -> None:
    texts = [
        "ZONE SECURED", "BRIDGE FALLEN", "CONVOY LOST", "TOWER HELD", "GATE BREACHED",
        "PUMP ONLINE", "RADAR JAMMED", "FLAG DROPPED", "VAULT OPENED", "SIREN ACTIVE",
        "DEPOT BURNING", "LIFT ARRIVED", "SHIELD BROKEN", "CORE EXPOSED",
    ]  # fmt: skip
    frames = _frames({0: texts, 1: texts, 2: texts[:1]})
    got = mine_phrases(_proposal(), frames)
    assert len(got) == 12
    assert got[0] == texts[0]  # three bursts first


def test_rejects_gamertags_chat_static_and_generic() -> None:
    rejects = ["xX_Ash99 ELIMINATED", "[ALL] Ash: GG EZ", "ROUND WON"]
    rejects += ["XX_ASH99 DOWNED", "XANTHOS", "@Ash99 OUT", "#zonewar"]
    rejects += ["HOLD THE LINE UNTIL THE CONVOY ARRIVES", "SECTOR 5"]
    texts = {b: ["OBJECTIVE ZONE"] for b in range(10)}  # static label: every frame
    for b in (0, 4):
        texts[b] += [*rejects, "ZONE SECURED"]
    frames = _frames(texts, frames_per_burst=1)
    assert mine_phrases(_proposal(), frames) == ["ZONE SECURED"]


def test_gamertags_split_off() -> None:
    frames = _frames({0: ["N0VA HAS FLAG"], 4: ["A1B2 HAS FLAG"], 6: ["Q7R TOOK THE LEAD"]})
    assert mine_phrases(_proposal(), frames) == ["HAS FLAG"]


def test_mixed_case_kept_in_most_common_casing() -> None:
    texts = {0: ["Shield Saver"], 3: ["Shield Saver"], 6: ["Shield saver"]}
    assert mine_phrases(_proposal(), _frames(texts)) == ["Shield Saver"]


def test_inconsistent_casing_rejected() -> None:
    # the same words in a different casing in every burst: not one HUD text
    texts = {0: ["Hull Breach"], 3: ["hull BREACH"], 6: ["HULL breach"]}
    assert mine_phrases(_proposal(), _frames(texts)) == []


def test_camel_case_split_and_runtime_match() -> None:
    raw = ["blockedShieldSaver:x1", "blockedShieldSaver:x2"]
    frames = _frames({0: raw[:1], 3: raw[1:]})
    got = mine_phrases(_proposal(), frames)
    assert got == ["blocked Shield Saver"]  # parts never shown alone are not mined
    # the OCR layer matches the mined phrase in the raw, space-less text
    assert all(fuzzy_contains(t, got[0]) for t in raw)
    assert fuzzy_contains("[01:02] Kip3 blockedShieldSaver:x4", "Shield Saver")


def test_phrase_inside_log_lines() -> None:
    texts = {
        0: ["[12:03:44] Ash99 blockedShieldSaver:x1"],
        2: ["[12:09:10] Bo7 Shield Saver x2"],
        5: ["[12:15:00] Kip3 blockedShieldSaver:x1"],
    }
    # "Shield Saver" also shows alone ("x2" is a number) and in more bursts, so both are kept
    got = mine_phrases(_proposal(), _frames(texts))
    assert got == ["Shield Saver", "blocked Shield Saver"]
    assert mine_phrases(_proposal(), _frames(texts), ["Bo7"]) == got


def test_multiplier_is_a_number() -> None:
    texts = {0: ["Hull Breach x2"], 3: ["Hull Breach x3"]}
    assert mine_phrases(_proposal(), _frames(texts)) == ["Hull Breach"]


def test_name_slots_not_mined() -> None:
    rng = random.Random(7)
    names = ["Orrin", "Pell", "Quade", "Sorrel", "Tamsin", "Ulric", "Varo", "Wynn", "Yarrow"]
    names.append("Zeb")
    verbs = ["tagged", "pinned", "bumped", "tripped", "nudged"]
    frames = []
    for b in range(40):
        for k in range(4):
            # two "<name> <verb> <name>" lines in every frame
            texts = [f"{rng.choice(names)} {rng.choice(verbs)} {rng.choice(names)}" for _ in "ab"]
            if b in (5, 18, 31) and k == 0:
                texts.append("Crew Held The Gate")
            frames.append(
                {
                    "id": f"rec.mp4@{b * 60 + k * 1.5:.1f}",
                    "t": b * 60 + k * 1.5,
                    "source": "rec.mp4",
                }
                | {
                    "burst": b,
                    "k": k,
                    "boxes": [_box(t, y=310 + i * 30, h=24) for i, t in enumerate(texts)],
                }
            )
    got = mine_phrases(_proposal(), frames)
    assert "Crew Held The Gate" in got
    words = {w.upper() for w in names + verbs}
    assert not [p for p in got if set(p.upper().split()) <= words]


@pytest.mark.parametrize(
    "family",
    [
        ["AMBER SPARK", "COBALT SPARK", "FROST SPARK", "VIOLET SPARK", "GILDED SPARK"],
        ["GATE SECURED", "HILL SECURED", "DOCK SECURED", "TOWER SECURED", "BRIDGE SECURED"],
        ["ROUND GRABBED", "ROUND CEDED", "ROUND BEGUN", "ROUND HALTED"],
    ],
)
def test_banner_family_not_a_slot(family: list[str]) -> None:
    # banners sharing a word, rotated over the bursts: fixed edge words, no name slot
    texts = {b: [family[b % len(family)]] for b in range(10)}
    assert sorted(mine_phrases(_proposal(), _frames(texts))) == sorted(family)


def test_phrase_in_banner_family_kept() -> None:
    family = ["GATE SECURED", "HILL SECURED", "DOCK SECURED", "TOWER SECURED"]
    texts = {b: [family[b % 4]] for b in range(8)} | {
        8: ["ZONE SECURED NOW"],
        9: ["ZONE SECURED NOW"],
    }
    assert "ZONE SECURED NOW" in mine_phrases(_proposal(), _frames(texts))


def test_name_before_phrase_is_a_slot() -> None:
    names = ["Orrin", "Pell", "Quade", "Sorrel", "Tamsin", "Ulric"]
    texts = {b: [f"{names[b % 6]} blocked Shield Saver"] for b in range(6)}
    got = mine_phrases(_proposal(), _frames(texts))
    assert got == ["blocked Shield Saver"]


def test_only_frames_of_the_discovery_run() -> None:
    frames = _frames({0: ["ZONE SECURED"], 4: ["ZONE SECURED"]})
    proposal = _proposal()
    proposal["discovery"]["frames"] = [f["id"] for f in frames if f["burst"] < 3]
    assert mine_phrases(proposal, frames) == []  # burst 4 is from an earlier run
    assert mine_phrases(_proposal(), frames) == ["ZONE SECURED"]  # no "frames": all


def test_discovery_params_used_for_menus() -> None:
    frames = _frames({0: ["ZONE SECURED"], 4: ["ZONE SECURED"]})
    for f in frames:
        f["boxes"] += [_box(f"AMMO {i}", x=20, y=150 + i * 20, h=18) for i in range(40)]
    proposal = _proposal()
    proposal["discovery"] = {"params": {"menu_min_boxes": 10, "menu_jump": 1.0}}
    assert mine_phrases(proposal, frames) == []  # every frame is a menu with these params
    proposal["discovery"] = {"params": {}}
    # a busy HUD (40 small edge boxes per frame) is not a menu by default
    assert mine_phrases(proposal, frames) == ["ZONE SECURED"]


def test_longer_phrase_hides_part_with_same_bursts() -> None:
    texts = {0: ["Hull Breach Sealed"], 3: ["Hull Breach Sealed"], 6: ["Hull Breach Sealecl"]}
    assert mine_phrases(_proposal(), _frames(texts)) == ["Hull Breach Sealed"]


def test_player_tag_splits_log_line() -> None:
    texts = {0: ["Boxer blockedShieldSaver"], 4: ["Boxer blockedShieldSaver"]}
    assert mine_phrases(_proposal(), _frames(texts), ["Boxer"]) == ["blocked Shield Saver"]
    joined = {0: ["BoxerblockedShield"], 4: ["BoxerblockedShield"]}
    assert mine_phrases(_proposal(), _frames(joined), ["Boxer"]) == []  # "Shield" alone


def test_joined_numbers_split_off() -> None:
    # OCR joined the sector number; another burst shows the phrase inside a longer text
    texts = {0: ["HELD SECTOR5"], 3: ["HELD SECTOR7"], 6: ["WE HELD SECTOR12 AGAIN"]}
    texts |= {1: ["ALARM RAISED"], 2: ["ALARM RAISED"]}
    # the longer text counts as a third burst of "HELD SECTOR", so it ranks first
    assert mine_phrases(_proposal(), _frames(texts)) == ["HELD SECTOR", "ALARM RAISED"]


@pytest.mark.parametrize(
    "texts",
    [
        ["SPARKS LIT 2/3", "SPARKS LIT 3/3"],  # digits only
        ["3SPARKS LIT", "2SPARKS LIT"],  # digits first
        ["SPARKS LIT 2024", "SPARKS LIT 1999"],  # letter share ignores digits
    ],
)
def test_number_forms(texts: list[str]) -> None:
    assert mine_phrases(_proposal(), _frames({0: texts[:1], 5: texts[1:]})) == ["SPARKS LIT"]


def test_ordinals_dropped() -> None:
    frames = _frames({0: ["YOU PLACED 3RD"], 4: ["YOU PLACED 5TH"]})
    assert mine_phrases(_proposal(), frames) == ["YOU PLACED"]


def test_bursts_count_only_ocr_matches() -> None:
    # split, the second text contains "WAVE CLEARED"; OCR (digits kept) would not match it
    frames = _frames({0: ["WAVE CLEARED"], 3: ["WAVE 123 CLEARED BY THE SQUAD OF SIX"]})
    assert mine_phrases(_proposal(), frames) == []


def test_static_share_uses_split_texts() -> None:
    texts = {b: ["ZONE 1234 HELB"] for b in range(2, 10)}  # static label with a number
    texts |= {0: ["ZONE HELD"], 1: ["ZONE HELD"]}
    assert mine_phrases(_proposal(), _frames(texts, frames_per_burst=1)) == []


def test_player_tag_with_digits() -> None:
    frames = _frames({0: ["DOWNED ASH99"], 3: ["DOWNED ASH99"]})
    assert mine_phrases(_proposal(), frames) == ["DOWNED ASH"]
    assert mine_phrases(_proposal(), frames, ["Ash99"]) == []


def test_occurrence_sharing_only_bigrams() -> None:
    # OCR inserted and dropped letters: no common trigram, still a fuzzy occurrence
    frames = _frames({0: ["ABX CDEFZGH"], 3: ["ABCDYEFGHW"]})
    assert mine_phrases(_proposal(), frames) == ["ABX CDEFZGH"]


def test_static_label_with_ocr_variant() -> None:
    texts = {b: ["OBJECTIVE ZONE"] for b in range(10)}
    texts[9] = ["OBJECTlVE ZONE"]  # one OCR misread of the static label
    assert mine_phrases(_proposal(), _frames(texts, frames_per_burst=1)) == []


def test_generic_banner_variant() -> None:
    frames = _frames({0: ["ROUND WON"], 2: ["ROUND WON"], 4: ["ROUND VON"]})
    assert mine_phrases(_proposal(), frames) == []


def test_player_tags() -> None:
    frames = _frames({b: ["BOXER HAS THE FLAG", "ZONE SECURED"] for b in (0, 2)})
    assert mine_phrases(_proposal(), frames) == ["BOXER HAS THE FLAG", "ZONE SECURED"]
    prof = build_profile(_proposal(), frames, "zonewar", player_tags=["Boxer"])
    # the tag is split off: the rest of the banner is a candidate
    assert prof["candidate_phrases"] == ["HAS THE FLAG", "ZONE SECURED"]


def test_bigtext_region_used_with_min_height() -> None:
    proposal = _proposal()
    proposal["regions"] = proposal["regions"][:1]  # no banner: the generic big-text region
    tall = _frames({0: ["ZONE SECURED"], 4: ["ZONE SECURED"]})
    assert mine_phrases(proposal, tall) == ["ZONE SECURED"]
    for f in tall:
        for b in f["boxes"]:
            b["y1"] = b["y0"] + 20  # below the big-text minimum height
    assert mine_phrases(proposal, tall) == []


@pytest.mark.parametrize(
    ("game", "hint"),
    [
        ("zonewar", "zonewar"),
        ("zone-war-2", "zone war 2"),
        ("zwr", ""),
        ("rust", ""),
        ("new-folder", ""),
        ("my-footage", ""),
        ("zonewar-videos", ""),
        ("2024-11-14", ""),
        ("footage8", ""),
        ("clips2025", ""),
        ("Videos_2024", ""),
        ("zonewar2", "zonewar2"),
        ("zone-war-videos2", ""),
    ],
)
def test_match_hint(game: str, hint: str) -> None:
    assert match_hint(game) == hint


def test_round_trip_and_superset() -> None:
    proposal = _proposal()
    before = json.dumps(proposal, sort_keys=True)
    frames = _frames({0: ["ZONE SECURED"], 2: ["ZONE SECURED"]})
    prof = build_profile(proposal, frames, "zone-war")
    assert json.dumps(proposal, sort_keys=True) == before  # input untouched
    assert all(prof[k] == v for k, v in proposal.items() if k != "game")
    assert prof["banner_phrases"] == []  # mined phrases are inactive candidates
    assert prof["candidate_phrases"] == ["ZONE SECURED"]
    assert "banner_phrases" in prof["notes"] and "candidate_phrases" in prof["notes"]
    assert prof["source"] == "discovered"
    assert prof["profile"] == {"backend": "ocr", "version": VERSION}
    assert prof["title"] == "Zone War"
    assert prof["match"] == ["zone war"]
    preset = preset_from_dict(json.loads(json.dumps(prof)))
    assert preset.title == "Zone War"
    assert preset.phrases == preset_from_dict(proposal).phrases
    assert preset.regions == preset_from_dict(proposal).regions


def test_deterministic() -> None:
    frames = _frames({0: ["ZONE SECURED", "BRIDGE FALLEN"], 2: ["BRIDGE FALLEN", "ZONE SECURED"]})
    a = build_profile(_proposal(), frames, "zonewar")
    b = build_profile(_proposal(), list(reversed(frames)), "zonewar")
    assert json.dumps(a) == json.dumps(b)
    assert a["candidate_phrases"] == ["BRIDGE FALLEN", "ZONE SECURED"]


def test_cache_keys_unchanged_with_candidates() -> None:
    proposal = _proposal()
    prof = build_profile(proposal, _frames({0: ["ZONE SECURED"], 1: ["ZONE SECURED"]}), "zonewar")
    assert prof["candidate_phrases"] == ["ZONE SECURED"]
    plain, synth = preset_from_dict(proposal), preset_from_dict(prof)
    assert OcrParams().key("zonewar", plain) == OcrParams().key("zonewar", synth)
    assert game_prompt("zonewar", plain).key() == game_prompt("zonewar", synth).key()
    prof["banner_phrases"] = prof["candidate_phrases"]  # promoted by hand
    assert OcrParams().key("zonewar", preset_from_dict(prof)) != OcrParams().key("zonewar", plain)


def test_read_ocr_samples(tmp_path: Path) -> None:
    path = tmp_path / "ocr.jsonl"
    rec = _frames({0: ["ZONE SECURED"]})[0]
    path.write_text(json.dumps(rec) + "\nnot json\n[1]\n", encoding="utf-8")
    assert discover.read_ocr_samples(path) == [rec]
    assert discover.read_ocr_samples(tmp_path / "missing.jsonl") == []


def _burst(texts_by_k: dict[int, list[str]], region: dict | None = None) -> list[dict]:
    """Burst 20's four frames (1.5 s apart), frame ``k`` showing ``texts_by_k[k]``, after
    the 40 text-less frames of :func:`_frames`."""
    r, b = region or BANNER, 20
    return _frames({}) + [
        {"id": f"rec.mp4@{b * 60 + k * 1.5:.1f}", "t": b * 60 + k * 1.5, "source": "rec.mp4"}
        | {"burst": b, "k": k}
        | {"boxes": [_box(t, x=r["x"] + 20, y=r["y"] + 20) for t in texts_by_k.get(k, [])]}
        for k in range(4)
    ]


def test_two_frames_of_one_burst_apart_count() -> None:
    # 3 s apart in one burst: two occurrences
    assert 2 * 1.5 >= MIN_GAP_S > 1.5
    assert mine_phrases(_proposal(), _burst({0: ["Squad wiped"], 2: ["Squad wiped"]})) == [
        "Squad wiped"
    ]
    # neighbouring frames (1.5 s apart) are one occurrence
    assert mine_phrases(_proposal(), _burst({0: ["Squad wiped"], 1: ["Squad wiped"]})) == []
    # a third frame 1.5 s after the second still counts: 3 s after the first counted one
    three = _burst({0: ["Squad wiped"], 1: ["Squad wiped"], 2: ["Squad wiped"]})
    assert mine_phrases(_proposal(), three) == ["Squad wiped"]


def test_menu_frames_never_count() -> None:
    frames = _burst({0: ["Squad wiped"], 2: ["Squad wiped"]})
    assert mine_phrases(_proposal(), frames) == ["Squad wiped"]
    # frame 2 is a scoreboard: a centred table of 44 boxes
    frames[-2]["boxes"] += [
        _box(f"Nym {c * 7 + r}", x=400 + c * 300, y=250 + r * 60, h=20)
        for r in range(11)
        for c in range(4)
    ]
    assert discover.gameplay_frames(frames, discover.DiscoverParams()) == frames[:-2] + frames[-1:]
    assert mine_phrases(_proposal(), frames) == []


def test_phrase_not_counted_inside_a_word() -> None:
    # the old count slid "QORV KAL" over "HULLQORVKALBREACHED" with spaces removed
    texts = {0: ["QORV KAL"], 2: ["HULLQORVKALBREACHED"], 4: ["HULLQORVKALBREACHED"]}
    assert mine_phrases(_proposal(), _frames(texts)) == []
    # a space-less reading of a whole phrase still counts
    texts = {1: ["Hull breached"], 3: ["Hullbreached"], 6: ["Qorv k/20 Hullbreached"]}
    assert mine_phrases(_proposal(), _frames(texts)) == ["Hull breached"]


@pytest.mark.parametrize(
    ("text", "phrase", "found"),
    [
        ("HULLQORVKBREACHED", "QORVK", False),
        ("HULLQORVKBREACHED", "QORV K", False),
        ("SQUADWIPED", "QUAD WIPE", False),
        ("BLOCKEDSHIELDUPGRADE", "SHIELD UP", False),
        ("Hullbreached", "Hull breached", True),
        ("HullBreached by Nym", "Hull breached", True),
        ("Qorv k/20 Hullbreached", "Hull breached", True),
        ("Hull brea ched!", "Hull breached", True),
        ("lHull breached", "Hull breached", True),  # a stray letter OCR glued on
        ("NymSquadWiped", "Squad wiped", True),
        ("HULL BREACHED5", "HULL BREACHED", True),
    ],
)
def test_fuzzy_contains_whole_words(text: str, phrase: str, found: bool) -> None:
    assert fuzzy_contains(text, phrase) is found


def test_notice_regions_mined() -> None:
    notice = {"x": 650, "y": 700, "w": 600, "h": 80}
    proposal = _proposal()
    proposal["regions"].append({"name": "notice2", **notice, "kind": "notice", "gate": "det"})
    frames = _burst({0: ["Squad wiped"], 3: ["Squad wiped"]}, region=notice)
    for f in frames:
        for b in f["boxes"]:
            b["y1"] = b["y0"] + 31  # notice text: below the banner minimum of the OCR layer
    assert mine_phrases(proposal, frames) == ["Squad wiped"]
    proposal["regions"].pop()
    proposal["bigtext"] = False
    assert mine_phrases(proposal, frames) == []  # outside every mined region


def test_bigtext_mined_at_the_runtime_minimum_height() -> None:
    # text the OCR layer's big-text check skips is not mined there (notices are mined instead)
    proposal = _proposal()
    proposal["regions"] = proposal["regions"][:1]  # the generic big-text region only
    frames = _frames({0: ["ZONE SECURED"], 4: ["ZONE SECURED"]})
    min_h = bigtext_region().min_text_h
    for h, mined in ((min_h, True), (min_h - 5, False)):
        for f in frames:
            for b in f["boxes"]:
                b["y1"] = b["y0"] + h
        assert (mine_phrases(proposal, frames) == ["ZONE SECURED"]) is mined


def test_parts_of_long_runs_mined() -> None:
    # 7-word runs: names around a fixed phrase. Before, runs over 5 words gave nothing
    names = ["Orrin", "Pell", "Quade", "Sorrel", "Tamsin", "Ulric", "Varo", "Wynn"]
    texts = {
        b: [f"{names[b % 8]} {names[(b + 3) % 8]} hull was breached by {names[(b + 5) % 8]}"]
        for b in range(8)
    }
    got = mine_phrases(_proposal(), _frames(texts))
    assert got == ["hull was breached by"]
    words = {n.upper() for n in names}
    assert not [p for p in got if set(p.upper().split()) & words]


def test_repeated_sentence_gives_nothing() -> None:
    line = "Hold the gate until the convoy reaches the depot"
    assert mine_phrases(_proposal(), _frames({b: [line] for b in (0, 3, 6)})) == []


def test_heavy_log_cost() -> None:
    # 160 frames of 12 notice lines of 12 invented words each, most of them unique
    rng = random.Random(5)
    syl = "bdfgklmnprstvz"
    vocab = ["".join(rng.choice(syl) + rng.choice("aeiou") for _ in range(3)) for _ in range(300)]
    frames = []
    for b in range(40):
        for k in range(4):
            lines = [" ".join(rng.choice(vocab) for _ in range(12)) for _ in range(12)]
            if k == 0 and b % 9 == 0:
                lines[0] = (
                    f"{rng.choice(vocab)} {rng.choice(vocab)} Squad wiped {rng.choice(vocab)}"
                )
            frames.append(
                {
                    "id": f"rec.mp4@{b * 60 + k * 1.5:.1f}",
                    "t": b * 60 + k * 1.5,
                    "source": "rec.mp4",
                }
                | {"burst": b, "k": k}
                | {"boxes": [_box(t, y=310 + i * 8, h=40) for i, t in enumerate(lines)]}
            )
    t0 = time.perf_counter()
    got = mine_phrases(_proposal(), frames)
    assert time.perf_counter() - t0 < 10  # ~1 s; a generous bound against blow-ups
    assert got == ["Squad wiped"]


HINT = "Tap Lever Twice"  # an invented tutorial line
BANNER_TEXT = "ZONE SECURED"


def _match(hint_bursts: set[int], hint_frames: int = 4, bursts: int = 40) -> list[dict]:
    """``bursts`` bursts x 4 frames of one match: the hint line in the banner region on the
    first ``hint_frames`` frames of ``hint_bursts``; the banner on the first 2 frames of
    5 bursts spread over the match (several seconds apart from one event to the next)."""
    events = {3, 11, 19, 27, 35}
    out = []
    for b in range(bursts):
        for k in range(4):
            boxes = []
            if b in hint_bursts and k < hint_frames:
                boxes.append(_box(HINT, y=380, h=40))
            if b in events and k < 2:
                boxes.append(_box(BANNER_TEXT))
            out.append(
                {"id": f"rec.mp4@{b * 15 + k * 1.5:.1f}", "t": b * 15 + k * 1.5}
                | {"source": "rec.mp4", "burst": b, "k": k, "boxes": boxes}
            )
    return out


def test_static_hint_line_is_dropped_and_the_banner_stays() -> None:
    # the hint in 12 of 40 bursts all match long: 30% of the frames (below STATIC_SHARE)
    frames = _match(set(range(1, 40, 3)) - {37})
    share = sum(any(b["text"] == HINT for b in f["boxes"]) for f in frames) / len(frames)
    assert HINT_SHARE <= share < 0.5
    assert mine_phrases(_proposal(), frames) == [BANNER_TEXT]


def test_a_hint_seen_in_one_phase_only_is_kept() -> None:
    # 30% of the frames, all in the first third of the match: not spread over it
    frames = _match(set(range(12)))
    assert mine_phrases(_proposal(), frames) == [HINT, BANNER_TEXT]


def test_a_rarer_line_is_no_hint() -> None:
    # in 1 frame of 12 bursts spread over the match: 7.5% of the frames
    frames = _match(set(range(1, 40, 3)) - {37}, hint_frames=1)
    assert mine_phrases(_proposal(), frames) == [HINT, BANNER_TEXT]


def test_a_short_source_is_not_tested_for_hints() -> None:
    # 3 of 7 bursts spread over the source: 43% of its frames, too few bursts to tell
    frames = _match({0, 3, 6}, bursts=HINT_MIN_BURSTS - 1)
    assert HINT in mine_phrases(_proposal(), frames)
