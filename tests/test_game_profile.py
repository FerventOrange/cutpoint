"""Game profile builder, OCR backend (synthetic OCR samples, invented game ``zonewar``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cutpoint.events import discover
from cutpoint.events.game_profile import build_profile, match_hint, mine_phrases
from cutpoint.events.killfeed import OcrParams
from cutpoint.events.regions import preset_from_dict
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
                {"id": f"rec.mp4@{b * 60 + k * 1.5:.1f}", "source": "rec.mp4"}
                | {"burst": b, "k": k, "boxes": boxes}
            )
    return out


def test_phrase_needs_two_bursts() -> None:
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
    rejects = ["xX_Ash99 ELIMINATED", "[ALL] Ash: GG EZ", "Ash Was Here", "ROUND WON"]
    rejects += ["NOVA7 DOWNED", "XANTHOS", "HOLD THE LINE UNTIL THE CONVOY ARRIVES"]
    texts = {b: ["OBJECTIVE ZONE"] for b in range(10)}  # static label: every frame
    for b in (0, 4):
        texts[b] += [*rejects, "ZONE SECURED"]
    frames = _frames(texts, frames_per_burst=1)
    assert mine_phrases(_proposal(), frames) == ["ZONE SECURED"]


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
    assert prof["banner_phrases"] == ["ZONE SECURED"]


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
        ("game", ""),
        ("my-footage", ""),
        ("zonewar-videos", ""),
        ("recording", ""),
        ("2024-11-14", ""),
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
    assert all(prof[k] == v for k, v in proposal.items() if k not in ("game", "banner_phrases"))
    assert prof["source"] == "discovered"
    assert prof["profile"] == {"backend": "ocr", "version": 1}
    assert prof["title"] == "Zone War"
    assert prof["match"] == ["zone war"]
    preset = preset_from_dict(json.loads(json.dumps(prof)))
    assert preset.title == "Zone War"
    assert preset.phrases[0] == "ZONE SECURED"
    assert preset.regions == preset_from_dict(proposal).regions


def test_deterministic() -> None:
    frames = _frames({0: ["ZONE SECURED", "BRIDGE FALLEN"], 2: ["BRIDGE FALLEN", "ZONE SECURED"]})
    a = build_profile(_proposal(), frames, "zonewar")
    b = build_profile(_proposal(), list(reversed(frames)), "zonewar")
    assert json.dumps(a) == json.dumps(b)
    assert a["banner_phrases"] == ["BRIDGE FALLEN", "ZONE SECURED"]


def test_cache_keys_unchanged_without_phrases() -> None:
    proposal = _proposal()
    prof = build_profile(proposal, _frames({0: ["ZONE SECURED"]}), "zonewar")
    assert prof["banner_phrases"] == []
    plain, synth = preset_from_dict(proposal), preset_from_dict(prof)
    assert OcrParams().key("zonewar", plain) == OcrParams().key("zonewar", synth)
    assert game_prompt("zonewar", plain).key() == game_prompt("zonewar", synth).key()
    with_phrase = build_profile(
        proposal, _frames({0: ["ZONE SECURED"], 1: ["ZONE SECURED"]}), "zonewar"
    )
    assert OcrParams().key("zonewar", preset_from_dict(with_phrase)) != OcrParams().key(
        "zonewar", plain
    )


def test_read_ocr_samples(tmp_path: Path) -> None:
    path = tmp_path / "ocr.jsonl"
    rec = _frames({0: ["ZONE SECURED"]})[0]
    path.write_text(json.dumps(rec) + "\nnot json\n[1]\n", encoding="utf-8")
    assert discover.read_ocr_samples(path) == [rec]
    assert discover.read_ocr_samples(tmp_path / "missing.jsonl") == []
