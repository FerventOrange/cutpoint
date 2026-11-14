"""The self-checks of a finished analysis (#1073): five read-only checks that warn when a
detection silently failed. Synthetic highlights.json / events.json objects; no ranking pinned."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

import pytest

from cutpoint import cache, jsonout, rerank, selfcheck
from cutpoint.events import scoring
from cutpoint.styles import get_style
from cutpoint.ui import REFINE_WARNINGS, Job, _note_warnings
from test_event_candidates import project_of, write_events
from test_rerank import lite_layers as lite_layers
from test_rerank import make_project


# --------------------------------------------------------------------------- builders
def hl_of(cands: list[dict[str, Any]] | None = None, **top: Any) -> dict[str, Any]:
    return {
        "style": "gaming",
        "style_base": "gaming",
        "suggested_top": 0,
        "sources": [{"path": "D:/v/a.mp4"}],
        "settings": {},
        "candidates": cands or [],
        **top,
    }


def ev_of(
    stats: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
    tags: list[str] | None = ("me",),  # type: ignore[assignment]
) -> dict[str, Any]:
    ocr: dict[str, Any] = {"tags": list(tags)} if tags else {}
    return {
        "settings": {"ocr": ocr},
        "sources": ["D:/v/a.mp4"],
        "stats": {"ocr": stats if stats is not None else {}},
        "events": events or [],
    }


def stat(lines: int, hours: float, *, frames: int = 100, menu: int = 0) -> dict[str, Any]:
    return {
        "frames": frames,
        "menu_frames": menu,
        "feed_regions": {"r0": {"label": "feed", "lines": lines, "hours": hours}},
    }


def codes(found: list[dict[str, Any]]) -> list[str]:
    return [f["code"] for f in found]


def cand(i: int, label: str = "x", **signals: Any) -> dict[str, Any]:
    return {
        "id": f"h{i:03d}",
        "source": "D:/v/a.mp4",
        "start": i * 100.0,
        "end": i * 100.0 + 10,
        "label": label,
        "signals": signals,
    }


# --------------------------------------------------------------------------- feed_quiet
def test_feed_quiet_triggers_on_a_slow_feed() -> None:
    found = selfcheck.feed_quiet(hl_of(), ev_of({"a.mp4": stat(22, 1.0)}))
    assert codes(found) == ["feed_quiet"]
    assert found[0]["files"] == ["D:/v/a.mp4"]


@pytest.mark.parametrize(
    "st",
    [
        stat(300, 1.0),  # a working feed
        stat(10, 0.2),  # too little gameplay for a rate
        stat(150, 2.0, menu=50),  # 150 lines in 1 gameplay hour (menu frames excluded)
    ],
)
def test_feed_quiet_stays_quiet_below_the_threshold(st: dict[str, Any]) -> None:
    assert selfcheck.feed_quiet(hl_of(), ev_of({"a.mp4": st})) == []


def test_feed_quiet_counts_gameplay_hours_not_menu_hours() -> None:
    # 80 lines over 2 h looks quiet (40/h) but 3/4 of it is menu: 160/h of gameplay
    assert selfcheck.feed_quiet(hl_of(), ev_of({"a.mp4": stat(80, 2.0, menu=75)})) == []


@pytest.mark.parametrize("ev", [{}, {"stats": {}}, {"stats": {"ocr": {"a.mp4": {}}}}])
def test_feed_quiet_skips_a_file_without_the_keys(ev: dict[str, Any]) -> None:
    assert selfcheck.feed_quiet(hl_of(), ev) == []
    assert "feed_quiet" not in codes(selfcheck.run(hl_of(), ev))


def test_feed_checks_apply_to_gaming_only() -> None:
    hl = hl_of(style="funny", style_base="funny")
    assert selfcheck.feed_quiet(hl, ev_of({"a.mp4": stat(1, 1.0)})) == []


# --------------------------------------------------------------------------- feed_missing
def no_region() -> dict[str, Any]:
    return {"frames": 100, "menu_frames": 0, "feed_regions": {}}


def vlm_kills(n: int, source: str = "D:/v/a.mp4") -> list[dict[str, Any]]:
    return [
        {"kind": "vlm", "source": source, "description": "The player kills an enemy."}
        for _ in range(n)
    ]


def test_feed_missing_triggers_with_tags_no_region_and_kills_described() -> None:
    ev = ev_of({"a.mp4": no_region()}, vlm_kills(selfcheck.KILLS_MIN_VLM_MENTIONS))
    found = selfcheck.feed_missing(hl_of(), ev)
    assert codes(found) == ["feed_missing"] and found[0]["files"] == ["D:/v/a.mp4"]
    assert "--discover-regions" in found[0]["hint"]


def test_feed_missing_stays_quiet_in_a_game_without_kills() -> None:
    """player_tags is a global setting: a walking sim with no kills must not warn."""
    quiet = ev_of({"a.mp4": no_region()}, vlm_kills(selfcheck.KILLS_MIN_VLM_MENTIONS - 1))
    assert selfcheck.feed_missing(hl_of(), quiet) == []
    elsewhere = ev_of({"a.mp4": no_region()}, vlm_kills(30, "D:/v/b.mp4"))  # another source
    assert selfcheck.feed_missing(hl_of(), elsewhere) == []


def with_vlm(ev: dict[str, Any]) -> dict[str, Any]:
    ev["settings"]["layers"] = ["audio", "ocr", "vlm"]
    return ev


def test_feed_quiet_stays_quiet_in_a_game_without_kills() -> None:
    """With the VLM layer run and no kills described, a quiet feed is no kill feed at all."""
    stats = {"a.mp4": stat(22, 1.0)}
    quiet = with_vlm(ev_of(stats, vlm_kills(selfcheck.KILLS_MIN_VLM_MENTIONS - 1)))
    assert selfcheck.feed_quiet(hl_of(), quiet) == []
    # kills described in another source say nothing about this one: it was not described at all
    other = with_vlm(ev_of(stats, vlm_kills(30, "D:/v/b.mp4")))
    assert codes(selfcheck.feed_quiet(hl_of(), other)) == ["feed_quiet"]


def test_feed_quiet_still_fires_when_the_vlm_described_nothing() -> None:
    """The vision model was requested but unavailable: no description, no skip."""
    unavailable = with_vlm(ev_of({"a.mp4": stat(22, 1.0)}, []))
    assert codes(selfcheck.feed_quiet(hl_of(), unavailable)) == ["feed_quiet"]


def test_feed_quiet_stays_quiet_when_the_vlm_described_a_source_without_kills() -> None:
    calm = [{"kind": "vlm", "source": "D:/v/a.mp4", "description": "The player walks."}] * 5
    assert selfcheck.feed_quiet(hl_of(), ev_of({"a.mp4": stat(22, 1.0)}, calm)) == []


def test_feed_quiet_still_fires_when_kills_are_described() -> None:
    siege = with_vlm(ev_of({"a.mp4": stat(22, 1.0)}, vlm_kills(72)))
    assert codes(selfcheck.feed_quiet(hl_of(), siege)) == ["feed_quiet"]
    # no VLM layer ran: the check behaves as before
    assert codes(selfcheck.feed_quiet(hl_of(), ev_of({"a.mp4": stat(22, 1.0)}))) == ["feed_quiet"]


def test_feed_missing_hint_names_the_preset_fix() -> None:
    ev = ev_of({"a.mp4": no_region()}, vlm_kills(30))
    hint = selfcheck.feed_missing(hl_of(), ev)[0]["hint"]
    assert "hud_presets" in hint and "proposal.json" in hint


@pytest.mark.parametrize(
    "ev",
    [
        ev_of({"a.mp4": no_region()}, vlm_kills(30), tags=None),  # no player_tags
        ev_of({"a.mp4": stat(300, 1.0)}, vlm_kills(30)),  # a region exists
        ev_of({"a.mp4": {"frames": 100, "menu_frames": 100, "feed_regions": {}}}, vlm_kills(30)),
        ev_of({"a.mp4": {"frames": 100, "menu_frames": 0}}, vlm_kills(30)),  # older file
        {},
    ],
)
def test_feed_missing_stays_quiet(ev: dict[str, Any]) -> None:
    assert selfcheck.feed_missing(hl_of(), ev) == []


# --------------------------------------------------------------------------- kills_uncredited
def kill(credited: bool = False) -> dict[str, Any]:
    return {"kind": "ocr", "type": "kill", "details": {"player_kill": credited}}


def test_kills_uncredited_triggers_on_feed_kills_nobody_credited() -> None:
    found = selfcheck.kills_uncredited(hl_of(), ev_of(events=[kill() for _ in range(5)]))
    assert codes(found) == ["kills_uncredited"] and "5 kill" in found[0]["message"]


def test_kills_uncredited_triggers_on_vlm_kill_descriptions() -> None:
    vlm = [{"kind": "vlm", "description": "He killed two players"} for _ in range(10)]
    found = selfcheck.kills_uncredited(hl_of(), ev_of(events=vlm))
    assert codes(found) == ["kills_uncredited"] and "vision model" in found[0]["message"]


@pytest.mark.parametrize(
    "ev, hl",
    [
        (ev_of(events=[kill() for _ in range(4)]), hl_of()),  # below the read threshold
        (ev_of(events=[{"kind": "vlm", "description": "kill"}] * 9), hl_of()),  # below VLM
        (ev_of(events=[kill() for _ in range(5)], tags=None), hl_of()),  # no player_tags
        (ev_of(events=[kill(True)] + [kill() for _ in range(5)]), hl_of()),  # one credited
        (
            ev_of(events=[kill() for _ in range(5)]),
            hl_of([cand(1, events={"player_kills": 1})]),  # credited in a candidate
        ),
    ],
)
def test_kills_uncredited_stays_quiet(ev: dict[str, Any], hl: dict[str, Any]) -> None:
    assert selfcheck.kills_uncredited(hl, ev) == []


def test_kills_uncredited_skips_an_old_file() -> None:
    assert selfcheck.kills_uncredited(hl_of(), {}) == []


# --------------------------------------------------------------------------- signal_dominates
def kw(i: int, phrase: str = "oh no") -> dict[str, Any]:
    return cand(i, keywords=[phrase])


def ev_cand(i: int, kind: str = "banners", **extra: Any) -> dict[str, Any]:
    return cand(i, events={kind: 1}, **extra)


def test_signal_dominates_triggers_when_one_event_kind_is_behind_most_moments() -> None:
    found = selfcheck.signal_dominates(hl_of([ev_cand(i) for i in range(1, 11)]), None)
    assert codes(found) == ["signal_dominates"] and "'banners' in 10" in found[0]["message"]


def test_signal_dominates_ignores_keyword_phrases() -> None:
    """A phrase said often is what a talker says (owner decision: detections only)."""
    cands = [cand(i, keywords=["oh my god"]) for i in range(1, 11)]
    assert selfcheck.signal_dominates(hl_of(cands), None) == []


def test_signal_dominates_stays_quiet_for_a_spread_of_reasons() -> None:
    kinds = ["banners", "explosions", "announcements", "teamkills", "self_kills"]
    cands = [ev_cand(i, kinds[i % 5]) for i in range(1, 11)]
    assert selfcheck.signal_dominates(hl_of(cands), None) == []


def test_signal_dominates_needs_more_than_half() -> None:
    cands = [ev_cand(i) for i in range(1, 6)] + [ev_cand(i, "explosions") for i in range(6, 11)]
    assert selfcheck.signal_dominates(hl_of(cands), None) == []


def test_signal_dominates_floor_is_ten_moments() -> None:
    nine = [ev_cand(i) for i in range(1, 10)]
    assert selfcheck.signal_dominates(hl_of(nine), None) == []  # fewer than 10 candidates
    cands = [ev_cand(i) for i in range(1, 7)] + [cand(i, events={f"k{i}": 1}) for i in range(7, 21)]
    # suggested_top 3 is lifted to 10: 6 of the top 10 share the kind -> triggers
    assert codes(selfcheck.signal_dominates(hl_of(cands, suggested_top=3), None)) == [
        "signal_dominates"
    ]
    assert len(selfcheck.top_moments(hl_of(cands, suggested_top=3))) == 10
    assert len(selfcheck.top_moments(hl_of(cands, suggested_top=15))) == 15
    assert len(selfcheck.top_moments(hl_of(cands))) == 20  # no suggested_top: the default


def test_signal_dominates_counts_event_evidence_and_ocr_text() -> None:
    cands = [cand(i, events={"explosions": 2}) for i in range(1, 11)]
    assert "explosions" in selfcheck.signal_dominates(hl_of(cands), None)[0]["message"]
    plain = [cand(i) for i in range(1, 11)]
    ocr = [
        {"kind": "ocr", "source": "D:/v/a.mp4", "start": i * 100.0 + 1, "label": "Victory"}
        for i in range(1, 11)
    ]
    found = selfcheck.signal_dominates(hl_of(plain), ev_of(events=ocr))
    assert "'victory'" in found[0]["message"]


def test_signal_dominates_skips_candidates_without_signals() -> None:
    cands = [{"id": f"h{i}", "label": "x"} for i in range(12)]
    assert selfcheck.signal_dominates(hl_of(cands), None) == []


# --------------------------------------------------------------------------- label_dominates
def labelled(labels: list[str]) -> dict[str, Any]:
    return hl_of([{"id": f"h{i}", "label": lb} for i, lb in enumerate(labels)])


def test_label_dominates_triggers() -> None:
    found = selfcheck.label_dominates(labelled(["banners"] * 10 + [f"l{i}" for i in range(30)]))
    assert codes(found) == ["label_dominates"] and "'banners' on 10 of 40" in found[0]["message"]


def test_label_dominates_below_threshold_and_few_candidates() -> None:
    assert selfcheck.label_dominates(labelled(["banners"] * 5 + [f"l{i}" for i in range(35)])) == []
    assert (
        selfcheck.label_dominates(labelled(["banners"] * 20 + [f"l{i}" for i in range(10)])) == []
    )


@pytest.mark.parametrize(
    "label",
    ["gaming-moment", "loud-moment", "action-cam-moment", "vlm", "laughter", "cheering", "laugh"],
)
def test_label_dominates_ignores_fallback_and_reaction_labels(label: str) -> None:
    assert selfcheck.label_dominates(labelled([label] * 30 + [f"l{i}" for i in range(10)])) == []


def test_label_dominates_ignores_a_keyword_phrase_label() -> None:
    cands = [cand(i, "oh-my-god", keywords=["oh my god"]) for i in range(40)]
    assert selfcheck.label_dominates(hl_of(cands)) == []
    named = [cand(i, "oh-my-god") for i in range(40)]  # the same label with no keyword behind it
    assert codes(selfcheck.label_dominates(hl_of(named))) == ["label_dominates"]


def test_label_dominates_skips_action_cam_telemetry_labels() -> None:
    hl = hl_of([cand(i, "carve") for i in range(40)], style="action-cam", style_base="action-cam")
    assert selfcheck.label_dominates(hl) == []


def test_label_dominates_skips_candidates_without_labels() -> None:
    assert selfcheck.label_dominates(hl_of([{"id": f"h{i}"} for i in range(50)])) == []


# --------------------------------------------------------------------------- apply / run
def failing_hl() -> dict[str, Any]:
    return labelled(["banners"] * 10 + [f"l{i}" for i in range(30)])


def test_apply_records_findings_and_clears_them_when_none() -> None:
    hl = failing_hl()
    found = selfcheck.apply(hl, None)
    assert hl["settings"]["self_checks"] == found and codes(found) == ["label_dominates"]
    hl["candidates"] = [{"id": f"h{i}", "label": f"l{i}"} for i in range(20)]
    assert selfcheck.apply(hl, None) == []
    assert "self_checks" not in hl["settings"]


def test_apply_changes_no_ranking(caplog: pytest.LogCaptureFixture) -> None:
    hl = failing_hl()
    before = copy.deepcopy(hl["candidates"]), hl["suggested_top"]
    selfcheck.apply(hl, ev_of({"a.mp4": stat(1, 1.0)}))
    assert (hl["candidates"], hl["suggested_top"]) == before


def test_apply_logs_a_warning_and_one_json_warning_per_finding(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        caplog.at_level(logging.WARNING, logger="cutpoint.selfcheck"),
        jsonout.run("analyze") as res,
    ):
        selfcheck.apply(failing_hl(), None)
        selfcheck.apply(failing_hl(), None)  # events runs after analyze: it replaces the first
    ours = [w for w in res.warnings if w["code"] == "label_dominates"]
    assert len(ours) == 1 and ours[0]["message"].endswith("check what it reads in events.json")
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


def test_a_second_apply_without_findings_leaves_no_warning() -> None:
    """edit --events: the events step's clean result replaces the analysis's findings."""
    with jsonout.run("edit") as res:
        selfcheck.apply(failing_hl(), None)
        assert [w["code"] for w in res.warnings] == ["label_dominates"]
        selfcheck.apply(hl_of(), None)
    assert res.warnings == []


def test_apply_without_json_run_does_not_fail() -> None:
    assert not jsonout.active()
    assert codes(selfcheck.apply(failing_hl(), None)) == ["label_dominates"]


def test_run_never_raises_on_an_odd_file() -> None:
    odd_ev = {"stats": {"ocr": {"a.mp4": {"feed_regions": {"r": "x"}, "frames": "n"}}}}
    odd_hl = hl_of([{"label": 3, "signals": "x", "start": "nope"}] * 25)
    assert isinstance(selfcheck.run(odd_hl, odd_ev), list)
    assert selfcheck.run({}, None) == []


# --------------------------------------------------------------------------- wiring
def spy_apply(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, Any]]:
    calls: list[tuple[Any, Any]] = []
    monkeypatch.setattr(selfcheck, "apply", lambda hl, ev: calls.append((hl, ev)) or [])
    return calls


def test_rerank_ranked_runs_the_self_checks(home, tmp_path, monkeypatch) -> None:
    project = make_project(home, tmp_path)
    hl = json.loads((project / "highlights.json").read_text(encoding="utf-8"))
    calls = spy_apply(monkeypatch)
    new = rerank.ranked(project, hl, get_style("funny"))
    assert len(calls) == 1 and calls[0][0] is not hl
    assert calls[0][0]["candidates"] == new["candidates"]
    assert isinstance(calls[0][1], dict)  # the events.json object, not None


def test_rerank_ranked_check_false_runs_no_self_check(home, tmp_path, monkeypatch) -> None:
    """The UI's preview records and warns about nothing."""
    project = make_project(home, tmp_path)
    hl = json.loads((project / "highlights.json").read_text(encoding="utf-8"))
    calls = spy_apply(monkeypatch)
    new = rerank.ranked(project, hl, get_style("funny"), check=False)
    assert calls == [] and "self_checks" not in new["settings"]


def test_rerank_ranked_records_self_checks_in_the_result(home, tmp_path, monkeypatch) -> None:
    project = make_project(home, tmp_path)
    hl = json.loads((project / "highlights.json").read_text(encoding="utf-8"))
    finding = {"code": "label_dominates", "message": "m", "hint": "h", "files": []}
    monkeypatch.setattr(selfcheck, "run", lambda hl, ev: [finding])
    new = rerank.ranked(project, hl, get_style("funny"))
    assert new["settings"]["self_checks"] == [finding]


def test_the_events_command_runs_the_self_checks(tmp_path, monkeypatch) -> None:
    proj, src = project_of(tmp_path, [])
    write_events(proj, src, [41.0, 100.0])
    calls = spy_apply(monkeypatch)
    scoring.rescore_highlights(proj)
    assert len(calls) == 1 and isinstance(calls[0][1], dict)
    assert calls[0][1]["events"]


def test_the_events_command_saves_the_findings_without_touching_scores(
    tmp_path, monkeypatch
) -> None:
    proj, src = project_of(tmp_path, [])
    write_events(proj, src, [41.0, 100.0])
    scoring.rescore_highlights(proj)
    plain = cache.load_json(proj / "highlights.json")
    finding = {"code": "feed_quiet", "message": "m", "hint": "h", "files": []}
    monkeypatch.setattr(selfcheck, "run", lambda hl, ev: [finding])
    scoring.rescore_highlights(proj)
    flagged = cache.load_json(proj / "highlights.json")
    assert flagged["settings"]["self_checks"] == [finding]
    assert [(c["id"], c["score"]) for c in flagged["candidates"]] == [
        (c["id"], c["score"]) for c in plain["candidates"]
    ]


# --------------------------------------------------------------------------- ui
@pytest.mark.parametrize("code", selfcheck.CODES)
def test_every_self_check_code_is_a_refine_warning(code: str) -> None:
    assert code in REFINE_WARNINGS


def test_a_self_check_warning_reaches_the_job_log() -> None:
    job = Job("p", "analyze")
    job.result = {
        "warnings": [
            {"code": "feed_quiet", "message": "feed reads nothing. Check it", "files": ["a.mp4"]}
        ]
    }
    _note_warnings(job)
    assert any(line.endswith("feed reads nothing. Check it (a.mp4)") for line in job.logs)


def test_json_warning_is_not_noted_twice_when_the_cli_line_is_in_the_log() -> None:
    job = Job("p", "analyze")
    job.note("WARNING", "feed reads nothing. Check it")
    job.result = {
        "warnings": [{"code": "feed_quiet", "message": "feed reads nothing. Check it", "files": []}]
    }
    n = len(job.logs)
    _note_warnings(job)
    assert len(job.logs) == n
