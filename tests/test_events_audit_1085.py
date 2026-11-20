"""Events and HUD rows of audit #1085 that change detection or ranking: A142 (det is the only
region gate besides read), A145 (absolute feed threshold), A163 (auto window counts, CLI
ranges) and A165 (one banner list)."""

from __future__ import annotations

import argparse
import logging

import pytest

from cutpoint.events import eventtypes, killfeed, tails
from cutpoint.events import regions as rg
from cutpoint.events import vlm_windows as vw
from cutpoint.events.cli import add_arguments
from cutpoint.events.discover import MIN_REGION_SCORE, proposals_to_regions

# --------------------------------------------------------------------------- A142


def test_det_is_the_default_gate_and_the_mask_gate_is_gone() -> None:
    r = rg.Region("feed", 0, 0, 100, 100, "feed")
    assert r.gate == "det"
    assert "min_fill" not in r.key() and "min_change" not in r.key()
    assert not hasattr(killfeed, "hud_mask")


def test_an_old_mask_gate_reads_as_det_with_one_warning(caplog) -> None:
    rg._WARNED.clear()
    data = {
        "game": "oldgame",
        "regions": [
            {"name": "a", "x": 0, "y": 0, "w": 100, "h": 50, "kind": "banner", "gate": "mask"},
            {"name": "b", "x": 0, "y": 60, "w": 100, "h": 50, "kind": "feed", "gate": "mask",
             "min_fill": 0.01, "min_change": 0.1},
            {"name": "c", "x": 0, "y": 120, "w": 100, "h": 50, "kind": "score", "gate": "read"},
        ],
    }  # fmt: skip
    with caplog.at_level(logging.WARNING):
        regs = rg._regions(data, "oldgame.json")
        rg._regions(data, "oldgame.json")
    assert [r.gate for r in regs] == ["det", "det", "read"]
    assert caplog.text.count("not supported any more") == 1


# --------------------------------------------------------------------------- A145


def _proposal(name: str, kind: str, score: float, y: int) -> dict:
    return {"name": name, "kind": kind, "score": score, "x": 1400, "y": y, "w": 400, "h": 100}


def test_a_weak_second_feed_is_kept_beside_a_strong_one() -> None:
    proposals = [
        _proposal("strong", "feed", 1.0, 40),
        _proposal("weak", "feed", 0.05, 400),  # under 0.25x the best: dropped before
        _proposal("noise", "feed", MIN_REGION_SCORE / 2, 700),
    ]
    assert [r.name for r in proposals_to_regions(proposals)] == ["strong", "weak"]


# --------------------------------------------------------------------------- A163


def test_window_counts_are_auto_by_default_and_resolved_in_one_place() -> None:
    p = vw.VlmParams()
    assert p.max_windows is None and p.max_fill_windows is None
    game = vw.resolve_counts(p, False, {})
    assert (game.max_windows, game.max_fill_windows) == (60, 0)
    seeds = {"a.mp4": [{"kind": "telemetry", "type": "airtime"}] * 90}
    cam = vw.resolve_counts(p, True, seeds)
    assert (cam.max_windows, cam.max_fill_windows) == (90, 150)
    assert vw.resolve_counts(p, True, {}).max_windows == 60
    many = {"a.mp4": [{"kind": "audio", "label": "Laughter"}] * 400}
    assert vw.resolve_counts(p, True, many).max_windows == 150
    given = vw.resolve_counts(vw.VlmParams(max_windows=7, max_fill_windows=0), True, seeds)
    assert (given.max_windows, given.max_fill_windows) == (7, 0)  # 0 can be chosen now
    assert p.max_windows is None  # the params passed in stay auto


def _parse(*argv: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    add_arguments(ap)
    return ap.parse_args(["proj", *argv])


@pytest.mark.parametrize(
    "argv",
    [
        ("--max-windows", "0"),
        ("--max-windows", "-1"),
        ("--max-windows", "x"),
        ("--ocr-fps", "0"),
        ("--ocr-fps", "-2"),
        ("--ocr-fps", "nan"),
        ("--discover-bursts", "0"),
        ("--discover-bursts", "2.5"),
    ],
)
def test_bad_counts_and_rates_are_bad_args(argv) -> None:
    with pytest.raises(SystemExit) as exc:
        _parse(*argv)
    assert exc.value.code == 2


def test_good_counts_and_rates_pass() -> None:
    args = _parse("--max-windows", "5", "--ocr-fps", "1.5", "--discover-bursts", "10")
    assert (args.max_windows, args.ocr_fps, args.discover_bursts) == (5, 1.5, 10)
    assert _parse().max_windows is None


# --------------------------------------------------------------------------- A165


def test_one_banner_list_feeds_seeds_and_payoffs() -> None:
    assert set(eventtypes.PAYOFF_BANNERS) == {
        k for k, v in eventtypes.BANNER_PAYOFFS.items() if v is not None
    }
    for phrase, seed in eventtypes.BANNER_PAYOFFS.items():
        ev = {"kind": "ocr", "type": "banner", "text": phrase}
        assert vw._seed_priority(ev) == seed, phrase
        payoff = {"kind": "ocr", "type": "banner", "label": phrase}
        assert tails.is_payoff(payoff, tails.EventTailSettings()) is (seed is not None), phrase


@pytest.mark.parametrize(
    ("text", "seed"),
    [
        ("ACE", 2.0),
        ("ACE!", 2.0),  # a whole word now, not only the exact text
        ("SPACE STATION", 0.6),  # not inside another word
        ("CLUTCH 1V3", 2.0),
        ("ROUND WON - CLUTCH", 2.0),  # the highest phrase wins
        ("TEAM VICTORY", 1.3),
        ("TARGET DESTROYED", 0.6),
        ("ENEMY ELIMINATED", 0.6),  # a no-seed phrase only as the whole text
        ("ELIMINATED", None),
        ("MATCH POINT", None),
        ("CAPTURED", eventtypes.BANNER_SEED),
    ],
)
def test_banner_seed_tiers(text: str, seed: float | None) -> None:
    assert vw._seed_priority({"kind": "ocr", "type": "banner", "text": text}) == seed


def test_weak_feeds_do_not_take_the_slots_of_a_banner_and_a_notice() -> None:
    """PR #1220 F1: the best feed, then the banners, then the other feeds and notices by
    score fill the MAX_REGIONS slots."""
    proposals = [
        _proposal(f"feed{i}", "feed", 1.0 if i == 0 else 0.05, 40 + 50 * i) for i in range(6)
    ]
    proposals += [_proposal("banner", "banner", 0.03, 900), _proposal("notice", "notice", 0.3, 960)]
    names = [r.name for r in proposals_to_regions(proposals)]
    assert len(names) == 6
    assert names[:3] == ["feed0", "banner", "notice"]
    assert "banner" in names and "notice" in names
