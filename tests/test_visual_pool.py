"""Visual pool (#635): view / crew / cruise windows from cached layers, the reserved slots
of select_moments, the rescore and analyze guards."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import StubTranscriber, media_info, write
from cutpoint import analyze, cache, jsonout, scorer, ski, telemetry, visual_pool
from cutpoint.events import visual as V
from cutpoint.events import voice as VO
from cutpoint.pano import director as D
from cutpoint.pano import views
from cutpoint.planning import select_moments
from cutpoint.styles import get_style
from cutpoint.visual_pool import Part, Window

DT = 0.1
N = 3000  # 300 s


def bump(n: int, t: float, value: float, width: float = 3.0) -> np.ndarray:
    g = np.zeros(n)
    g[int((t - width / 2) / DT) : int((t + width / 2) / DT)] = value
    return g


def steady_speed(n: int, spans: list[tuple[float, float]], kmh: float = 25.0) -> np.ndarray:
    """NaN (no GPS fix) except ``kmh`` with a little wobble in ``spans``."""
    sp = np.full(n, np.nan)
    for a, b in spans:
        i, j = int(a / DT), int(b / DT)
        sp[i:j] = kmh + 0.5 * np.sin(np.arange(j - i) / 7)
    return sp


def grids(n: int = N, **over: np.ndarray) -> dict[str, np.ndarray]:
    names = ("beauty", "people", "group", "airtime", "crash", "wipeout", "carve", "stopped")
    return {k: over.get(k, np.zeros(n)) for k in names}


def part(src: str = "a.mp4", index: int = 0, offset: float = 0.0, n: int = N) -> Part:
    return Part(Path(src), index, offset, n * DT, n)


@pytest.fixture
def no_decode(monkeypatch):
    """Any ffmpeg / cv2 use fails the test."""

    def boom(*_a, **_kw):
        raise AssertionError("no decode allowed")

    class NoCv2:
        def __getattr__(self, name):
            raise AssertionError("no cv2 allowed")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setitem(sys.modules, "cv2", NoCv2())


# --------------------------------------------------------------------------- windows
def test_part_windows_finds_each_kind_outside_the_mask(no_decode):
    g = grids(
        beauty=bump(N, 40, 0.8) + bump(N, 120, 0.9),  # 120 s is on the lift
        people=bump(N, 200, 0.7, 4),
        group=bump(N, 200, 0.6, 4),
    )
    mask = np.zeros(N)
    mask[1100:1300] = 1.0
    speed = steady_speed(N, [(60, 80)])
    got = visual_pool.part_windows(part(), g, mask, speed)
    by_kind = {w.kind: w for w in got}
    assert set(by_kind) == {"view", "crew", "cruise"}
    view, crew, cruise = by_kind["view"], by_kind["crew"], by_kind["cruise"]
    assert view.start <= 40 <= view.end and 5.0 <= view.end - view.start <= 7.0
    assert [w.kind for w in got].count("view") == 1  # the lift view is masked
    assert crew.start <= 200 <= crew.end and 4.0 <= crew.end - crew.start <= 6.0
    assert cruise.start >= 60 and cruise.end <= 80 and cruise.end - cruise.start == 8.0
    assert cruise.peak > 0.9 and cruise.signals["speed_kmh"] == pytest.approx(25, abs=0.5)
    assert view.signals == {"peak": 0.8, "beauty": 0.8}


def test_weak_peaks_and_group_without_people_are_not_windows():
    g = grids(beauty=bump(N, 40, 0.45), group=bump(N, 100, 1.0, 4), people=bump(N, 100, 0.2, 4))
    assert visual_pool.part_windows(part(), g, np.zeros(N), None) == []


def test_crew_needs_a_close_person_not_distant_skiers():
    # distant skiers score people ~0.45-0.49: a high group peak alone is no crew pick
    far = grids(people=bump(N, 100, 0.49, 4), group=bump(N, 100, 1.0, 4))
    assert visual_pool.part_windows(part(), far, np.zeros(N), None) == []
    near = grids(people=bump(N, 100, visual_pool.CREW_PEOPLE_MIN, 4), group=bump(N, 100, 1.0, 4))
    got = visual_pool.part_windows(part(), near, np.zeros(N), None)
    assert [w.kind for w in got] == ["crew"] and got[0].signals["people"] == 0.6


def test_windows_are_in_source_time_for_a_recovered_part():
    p = part(offset=500.0)
    got = visual_pool.part_windows(p, grids(beauty=bump(N, 40, 0.8)), np.zeros(N), None)
    assert len(got) == 1 and 537 <= got[0].start < 540 < got[0].end <= 543
    assert 5.0 <= got[0].end - got[0].start <= 7.0


def test_a_view_in_a_source_shorter_than_five_seconds_is_the_whole_source():
    n = 40  # 4 s
    p = part(n=n)
    got = visual_pool.part_windows(p, grids(n, beauty=bump(n, 2, 0.8, 1.0)), np.zeros(n), None)
    assert [(w.kind, w.start, w.end) for w in got] == [("view", 0.0, 4.0)]


def test_a_flat_beauty_stretch_gives_a_view_of_at_most_seven_seconds():
    beauty = np.where(np.abs(np.arange(N) * DT - 40) <= 15, 0.8, 0.0)  # 30 s of scenery
    got = visual_pool.part_windows(part(), grids(beauty=beauty), np.zeros(N), None)
    assert len(got) == 1 and 5.0 <= got[0].end - got[0].start <= 7.0


def test_cruise_needs_speed_steadiness_and_eight_seconds():
    mask = np.zeros(N)
    assert all(w.kind != "cruise" for w in visual_pool.part_windows(part(), grids(), mask, None))
    short = steady_speed(N, [(60, 67)])
    assert visual_pool.part_windows(part(), grids(), mask, short) == []
    jerky = np.full(N, np.nan)
    jerky[600:900] = np.where(np.arange(300) % 40 < 20, 10.0, 40.0)
    assert visual_pool.part_windows(part(), grids(), mask, jerky) == []
    slow = steady_speed(N, [(60, 90)], kmh=5.0)  # walking
    assert visual_pool.part_windows(part(), grids(), mask, slow) == []


@pytest.mark.parametrize("grid", ["airtime", "crash", "wipeout"])
def test_cruise_vetoed_by_airtime_crash_or_wipeout(grid):
    speed = steady_speed(N, [(60, 75)])  # 15 s; a veto at 67 s leaves two 7-8 s pieces
    g = grids(**{grid: bump(N, 67.5, 1.0, 1.0)})
    assert visual_pool.part_windows(part(), g, np.zeros(N), speed) == []
    assert [w.kind for w in visual_pool.part_windows(part(), grids(), np.zeros(N), speed)] == [
        "cruise"
    ]


def test_crashes_json_vetoes_every_kind():
    g = grids(beauty=bump(N, 70, 0.8))
    speed = steady_speed(N, [(60, 75)])
    got = visual_pool.part_windows(part(), g, np.zeros(N), speed, [(66.0, 69.0)])
    assert got == []


# --------------------------------------------------------------------------- select
def win(kind: str, start: float, peak: float, src: str = "a.mp4", length: float = 4.0) -> Window:
    return Window(Path(src), 0, start, start + length, kind, peak, {"peak": peak})


def test_select_caps_per_kind_and_per_source_and_spaces_a_kind():
    ws = [win("view", 100 * i, 0.9 - 0.01 * i) for i in range(5)]
    ws += [win("crew", 50 + 100 * i, 0.8 - 0.01 * i) for i in range(5)]
    ws += [win("cruise", 70 + 100 * i, 0.7 - 0.01 * i, length=8) for i in range(5)]
    got = visual_pool.select(ws, [])
    assert len(got) == visual_pool.PER_SOURCE
    kinds = [w.kind for w in got]
    assert all(kinds.count(k) <= visual_pool.PER_KIND for k in visual_pool.KINDS)
    other = visual_pool.select([*ws, win("view", 0, 0.5, src="b.mp4")], [])
    assert len(other) == visual_pool.PER_SOURCE + 1  # the cap is per source
    close = visual_pool.select([win("view", 0, 0.9), win("view", 20, 0.8)], [])
    assert [w.start for w in close] == [0]  # within SPACING_S of a view


def test_select_deoverlaps_across_kinds_by_peak():
    got = visual_pool.select([win("crew", 10, 0.7), win("view", 12, 0.8)], [])
    assert [(w.kind, w.start) for w in got] == [("view", 12)]


def test_select_commentary_only_breaks_ties():
    quiet, talky = win("view", 10, 0.81), win("view", 200, 0.81)
    speech = {0: [(201.0, 203.0)]}
    # the same score as listed (to 0.1): the window with speech goes first
    assert visual_pool.select([quiet, talky], [], speech)[0] is talky
    assert visual_pool.select([quiet, talky], [])[0] is quiet  # no speech data: as before
    # 0.81 and 0.84 both list as 8.1 / 8.4: a different score is not a tie
    higher = win("view", 400, 0.84)
    got = visual_pool.select([quiet, talky, higher], [], speech)
    assert [w.start for w in got] == [400, 200]  # PER_KIND 2: the quiet one is out
    assert higher.peak == 0.84 and talky.peak == 0.81  # the score is never changed


def test_select_a_higher_score_without_commentary_beats_a_talky_lower_one():
    quiet, talky = win("view", 10, 0.9), win("view", 200, 0.7)
    speech = {0: [(200.0, 204.0)]}
    assert [w.start for w in visual_pool.select([talky, quiet], [], speech)] == [10, 200]
    over = visual_pool.select([win("view", 10, 0.9), win("crew", 12, 0.7)], [], {0: [(12.0, 16.0)]})
    assert [w.kind for w in over] == ["view"]  # speech over the crew shot does not lift it


def test_select_drops_windows_overlapping_a_scored_candidate():
    scored = [{"source": "a.mp4", "start": 10.0, "end": 30.0}]
    half = win("view", 28.0, 0.9)  # 2 of 4 s inside: 50% of the shorter clip
    little = win("crew", 200.0, 0.9)
    edge = Window(Path("a.mp4"), 0, 29.0, 33.0, "crew", 0.8, {})  # 1 of 4 s: kept
    other_src = win("view", 12.0, 0.9, src="b.mp4")
    got = visual_pool.select([half, little, edge, other_src], scored)
    assert half not in got and {id(w) for w in got} == {id(little), id(edge), id(other_src)}


# --------------------------------------------------------------------------- build
def _save_part(project: Path, src: Path, feats: dict[str, np.ndarray], mask=None) -> None:
    mask = np.zeros(N) if mask is None else mask
    ski.save_features(project, src, 0.0, N * DT, feats, mask, np.zeros(N), [])


def test_build_action_cam_reads_the_saved_grids(tmp_path, monkeypatch, no_decode):
    src = tmp_path / "GX01.mp4"
    _save_part(tmp_path, src, grids(beauty=bump(N, 40, 0.8), people=bump(N, 200, 0.8, 4)))
    monkeypatch.setattr(telemetry, "cached_speed", lambda _s: steady_speed(N, [(100, 120)]))
    monkeypatch.setattr(ski, "layer_features", pytest.fail)  # action-cam: saved grids only
    crashes = {"crashes": [{"source": str(src), "t0": 195.0, "t1": 197.0, "confidence": 0.2}]}
    got = visual_pool.build(tmp_path, [Part(src, 0, 0.0, N * DT, N)], get_style("action-cam"),
                            [], crashes)  # fmt: skip
    assert sorted(w.kind for w in got) == ["cruise", "view"]  # the crew is in the crash


def test_build_action_style_uses_the_visual_layer_and_still_speed(monkeypatch, no_decode):
    still = steady_speed(N, [(0, 300)], kmh=1.0)  # standing all along: the mask
    calls = []

    def layer(src, n, offset, speed):
        calls.append((src, n, offset, speed is not None))
        return {"beauty": bump(n, 40, 0.8), "people": np.zeros(n), "group": np.zeros(n)}

    monkeypatch.setattr(ski, "layer_features", layer)
    monkeypatch.setattr(telemetry, "cached_speed", lambda _s: None)
    got = visual_pool.build(Path("p"), [part()], get_style("sport"), [], None)
    assert [w.kind for w in got] == ["view"] and calls == [(Path("a.mp4"), N, 0.0, False)]
    monkeypatch.setattr(telemetry, "cached_speed", lambda _s: still)
    assert visual_pool.build(Path("p"), [part()], get_style("action"), [], None) == []


def test_action_style_nan_speed_keeps_the_real_layer_views(monkeypatch, no_decode):
    """F1: GPS gaps (NaN) and a track shorter than the analysis must not turn the real
    layer_features grids NaN (ski.pick then finds nothing)."""
    t = np.arange(0.0, N * DT, 0.5)
    beauty = np.where(np.abs(t - 40) <= 2, 0.9, 0.0)
    scores = {"beauty_zs": beauty, "group_zs": np.zeros(len(t)), "aesthetic": np.full(len(t), 6.0),
              "steady": np.ones(len(t)), "count": np.zeros(len(t)), "close": np.zeros(len(t))}  # fmt: skip
    data = V.VisualData(t=t, emb=np.zeros((len(t), 3, 1)), det=np.zeros((0, 7)),
                        box_emb=np.zeros((0, 1)), box_idx=np.zeros(0, dtype=int))  # fmt: skip
    monkeypatch.setattr(V, "load_visual", lambda _s: data)
    monkeypatch.setattr(V, "frame_scores", lambda _d: scores)
    monkeypatch.setattr(VO, "load_voice", lambda _s: None)
    monkeypatch.setattr(views, "layer_grids", lambda *_a: None)
    speed = np.full(N - 500, 5.0)  # 50 s short of the analysis: padded with NaN
    speed[300:600] = np.nan  # no fix over the view at 40 s
    monkeypatch.setattr(telemetry, "cached_speed", lambda _s: speed)
    got = visual_pool.build(Path("p"), [part()], get_style("action"), [], None)
    views_found = [w for w in got if w.kind == "view"]
    assert views_found and views_found[0].start <= 40 <= views_found[0].end
    assert all(np.isfinite(w.peak) for w in got)


def test_pool_kinds_plan_360_paths_with_the_default_director_settings():
    """A pool moment exported from a 360 project (no path planned by 360 analyze) is
    planned like any unplanned span: its kind has no director tweak."""
    assert all(D.params_for(k) == D.params_for(None) for k in visual_pool.KINDS)


def test_pool_styles():
    assert [visual_pool.is_pool_style(get_style(s)) for s in ("action-cam", "action", "sport")] == [
        True
    ] * 3
    assert not any(visual_pool.is_pool_style(get_style(s)) for s in ("gaming", "funny"))


# --------------------------------------------------------------------------- select_moments
def _hl(tmp_path: Path, n_scored: int, pool: list[dict], style: str = "action", **extra) -> Path:
    cands = [
        {"id": f"h{i:03d}", "source": "a.mp4", "start": 20.0 * i, "end": 20.0 * i + 5,
         "duration": 5.0, "score": 90.0 - i}
        for i in range(1, n_scored + 1)
    ]  # fmt: skip
    project = tmp_path / "proj"
    project.mkdir(exist_ok=True)
    data = {"style": style, "candidates": cands + pool, **extra}
    (project / "highlights.json").write_text(json.dumps(data), encoding="utf-8")
    return project


def _pool(cid: str, peak: float, start: float, duration: float = 4.0) -> dict[str, Any]:
    return {
        "id": cid, "source": "a.mp4", "start": start, "end": start + duration,
        "duration": duration, "score": round(10 * peak, 1), "signals": {"peak": peak},
        "label": "scenic view / beauty shot", "kind": "view", "pool": "visual",
    }  # fmt: skip


STRONG = [_pool("v001", 0.8, 2000), _pool("v002", 0.7, 2100), _pool("v003", 0.55, 2200)]


@pytest.mark.parametrize(("top", "slots"), [(4, 0), (5, 1), (8, 1), (16, 2), (24, 2)])
def test_reserved_slots_by_top(tmp_path, top, slots):
    project = _hl(tmp_path, 30, STRONG)
    chosen, _ = select_moments(project, top=top)
    ids = [c["id"] for c in chosen]
    assert len(chosen) == top
    assert ids[top - slots :] == ["v001", "v002"][:slots]
    assert ids[: top - slots] == [f"h{i:03d}" for i in range(1, top - slots + 1)]


def test_reserved_slot_with_suggested_top_and_max_total(tmp_path):
    project = _hl(tmp_path, 30, STRONG, suggested_top=8)
    assert [c["id"] for c in select_moments(project)[0]][-1] == "v001"
    chosen, _ = select_moments(project, max_total=60)  # 12 picks of 5 s: one slot
    assert [c["id"] for c in chosen][-1] == "v001" and len(chosen) == 12
    assert sum(c["duration"] for c in chosen) <= 60
    chosen, _ = select_moments(project, max_total=80)  # 16 picks: two
    assert [c["id"] for c in chosen][-2:] == ["v001", "v002"]


def test_reserved_slot_goes_by_the_shown_peak_so_commentary_breaks_a_tie(tmp_path):
    # analysis lists the one with speech first among equal shown scores (8.1 both)
    talky = {**_pool("v001", 0.81, 2000), "transcript": "look at that"}
    quiet = _pool("v002", 0.814, 2100)
    chosen, _ = select_moments(_hl(tmp_path, 30, [talky, quiet]), top=8)
    assert [c["id"] for c in chosen][-1] == "v001"


def test_max_total_skips_a_pool_moment_that_would_not_fit(tmp_path):
    pool = [_pool("v001", 0.9, 2000, duration=9.0), _pool("v002", 0.7, 2100)]
    chosen, _ = select_moments(_hl(tmp_path, 30, pool), max_total=60)
    assert [c["id"] for c in chosen][-1] == "v002"
    assert sum(c["duration"] for c in chosen) <= 60


def test_weak_or_overlapping_pool_falls_back_to_scored(tmp_path):
    plain = [c["id"] for c in select_moments(_hl(tmp_path, 30, []), top=8)[0]]
    weak = _hl(tmp_path, 30, [_pool("v001", 0.59, 2000)])
    assert [c["id"] for c in select_moments(weak, top=8)[0]] == plain
    over = _hl(tmp_path, 30, [_pool("v001", 0.9, 21.0), _pool("v002", 0.7, 2000)])
    assert [c["id"] for c in select_moments(over, top=8)[0]][-1] == "v002"  # v001 overlaps h001


def test_must_include_crashes_and_jumps_keep_their_slots(tmp_path):
    project = _hl(tmp_path, 30, STRONG)
    hl = json.loads((project / "highlights.json").read_text(encoding="utf-8"))
    hl["candidates"][4]["kind"] = "airtime"  # h005, the lowest of a top 5
    hl["candidates"][29] |= {"must_include": True, "crash": {"id": "c1"}}  # h030, forced
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    ids = [c["id"] for c in select_moments(project, top=5)[0]]
    assert ids == ["h001", "h002", "h003", "h005", "h030", "v001"]  # h004 gave way
    only_protected = [{**c, "must_include": True} for c in hl["candidates"][:30]]
    hl["candidates"] = only_protected + STRONG
    (project / "highlights.json").write_text(json.dumps(hl), encoding="utf-8")
    assert not any(c.get("pool") for c in select_moments(project, top=5)[0])


def test_action_crash_without_kind_keeps_its_slot(tmp_path):
    """F2: action / sport candidates carry no ``kind``; a pick over a crashes.json crash
    keeps its slot, a weak crash entry protects nothing."""
    project = _hl(tmp_path, 30, STRONG)  # action: h005 is the lowest of a top 5
    crash = {"source": "a.mp4", "t0": 101.0, "t1": 102.0, "confidence": 0.5}  # inside h005
    (project / "crashes.json").write_text(json.dumps({"crashes": [crash]}), encoding="utf-8")
    ids = [c["id"] for c in select_moments(project, top=5)[0]]
    assert ids == ["h001", "h002", "h003", "h005", "v001"]  # h004 gave way
    point = {**crash, "t1": 101.0}  # a point crash (crash._shock_spans) inside h005
    (project / "crashes.json").write_text(json.dumps({"crashes": [point]}), encoding="utf-8")
    assert [c["id"] for c in select_moments(project, top=5)[0]] == ids
    weak = {**crash, "confidence": 0.1}
    (project / "crashes.json").write_text(json.dumps({"crashes": [weak]}), encoding="utf-8")
    assert [c["id"] for c in select_moments(project, top=5)[0]][3] == "h004"
    (project / "crashes.json").write_text("[1]", encoding="utf-8")  # unusable: ignored
    assert [c["id"] for c in select_moments(project, top=5)[0]][-1] == "v001"


def test_ids_name_pool_moments(tmp_path):
    chosen, _ = select_moments(_hl(tmp_path, 3, STRONG, style="gaming"), ids=["v003", "h002"])
    assert [c["id"] for c in chosen] == ["v003", "h002"]


def test_non_pool_styles_never_pick_pool_moments(tmp_path):
    project = _hl(tmp_path, 30, STRONG, style="gaming")
    ids = [c["id"] for c in select_moments(project, top=16)[0]]
    assert ids == [f"h{i:03d}" for i in range(1, 17)]
    learned = _hl(tmp_path, 30, STRONG, style="my-ski", settings={"visual_pool": {"count": 3}})
    assert [c["id"] for c in select_moments(learned, top=8)[0]][-1] == "v001"


# --------------------------------------------------------------------------- rescore
def test_rescore_events_keeps_the_pool_last_and_unscored():
    scored = [
        {"id": "h001", "source": "a.mp4", "start": 0.0, "end": 8.0, "score": 10.0},
        {"id": "h002", "source": "a.mp4", "start": 50.0, "end": 58.0, "score": 20.0},
    ]
    pool = [{**_pool("v001", 0.9, 100), "score": 99.0}, _pool("v002", 0.5, 200)]
    hl = {"style": "action", "settings": {"target_total": 10}, "candidates": scored + pool}
    win = {"source": "a.mp4", "start": 98.0, "end": 106.0, "highlight_score": 10}
    scorer.rescore_events(hl, {"events": [], "windows": [win]}, Path("proj"))
    assert [c["id"] for c in hl["candidates"]] == ["h002", "h001", "v001", "v002"]
    assert hl["candidates"][2]["score"] == 99.0 and "score_pre_events" not in pool[0]
    assert "events" not in pool[0]["signals"]
    assert hl["suggested_top"] == 2


# --------------------------------------------------------------------------- analyze
def _stub(monkeypatch) -> None:
    def fake_probe(path):
        return media_info(
            path, 60.0, width=640, height=360, audio_streams=0,
            acodec=None, sample_rate=None, channels=None,
        )  # fmt: skip

    monkeypatch.setattr(analyze, "probe", fake_probe)
    monkeypatch.setattr(analyze, "Transcriber", StubTranscriber)


@pytest.mark.parametrize("style", ["gaming", "funny"])
def test_gaming_and_funny_never_build_a_pool(home, tmp_path, monkeypatch, style):
    write(tmp_path / "a.mp4")
    _stub(monkeypatch)

    def run(name: str) -> dict:
        proj = tmp_path / name
        analyze.analyze(tmp_path / "a.mp4", get_style(style), proj, vision=False, use_scenes=False)
        hl = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
        return {k: v for k, v in hl.items() if k not in ("created", "timings")}

    before = run("plain")
    monkeypatch.setattr(visual_pool, "build", pytest.fail)  # never called
    assert run("guarded") == before
    assert "visual_pool" not in before["settings"]
    assert not any(c.get("pool") for c in before["candidates"])


def test_action_analyze_appends_the_pool_after_the_scored(home, tmp_path, monkeypatch):
    write(tmp_path / "a.mp4")
    _stub(monkeypatch)
    seen = {}

    def fake_build(project, parts, style, scored, crashes, speech):
        seen["parts"], seen["scored"], seen["speech"] = parts, list(scored), speech
        src = parts[0].source
        return [Window(src, 0, 40.0, 44.0, "view", 0.81, {"peak": 0.81, "beauty": 0.81}),
                Window(src, 0, 10.0, 18.0, "cruise", 0.7, {"peak": 0.7})]  # fmt: skip

    monkeypatch.setattr(visual_pool, "build", fake_build)
    proj = tmp_path / "proj"
    analyze.analyze(tmp_path / "a.mp4", get_style("action"), proj, vision=False, use_scenes=False)
    hl = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
    pool = [c for c in hl["candidates"] if c.get("pool") == "visual"]
    scored = [c for c in hl["candidates"] if not c.get("pool")]
    assert hl["candidates"] == scored + pool
    assert [(c["id"], c["kind"], c["score"], c["label"]) for c in pool] == [
        ("v001", "view", 8.1, "scenic view / beauty shot"),
        ("v002", "cruise", 7.0, "casual cruising run"),
    ]
    assert pool[0]["source"] == str(tmp_path / "a.mp4") and pool[0]["duration"] == 4.0
    assert pool[0]["source_index"] == 0 and "transcript" in pool[0]
    assert all(c["id"].startswith("h") for c in scored)
    assert set(seen["speech"]) == {0}  # spoken spans by part index, for the tie-break
    assert len(seen["scored"]) == len(scored) and hl["suggested_top"] <= len(scored)
    assert hl["settings"]["visual_pool"] == {
        "count": 2, "kinds": {"view": 1, "crew": 0, "cruise": 1}
    }  # fmt: skip


@pytest.mark.parametrize(("style", "settings", "want"), [
    ("action", {"visual_pool": {"count": 2}}, 2),
    ("gaming", {}, None),
])  # fmt: skip
def test_json_counts_visual_candidates_for_pool_styles_only(tmp_path, style, settings, want):
    project = _hl(tmp_path, 3, STRONG[:2] if want else [], style=style, settings=settings)
    with jsonout.run("analyze") as res:
        analyze.report_highlights(project, project / "highlights.json")
    assert res.outputs["candidates"] == 3 + (want or 0)
    assert res.outputs.get("visual_candidates") == want
    assert ("visual_candidates" in res.outputs) == (want is not None)


def test_a_failing_pool_never_fails_the_analysis(home, tmp_path, monkeypatch, caplog):
    write(tmp_path / "a.mp4")
    _stub(monkeypatch)

    def broken(*_a):
        raise RuntimeError("no prompt cache")

    monkeypatch.setattr(visual_pool, "build", broken)
    proj = tmp_path / "proj"
    analyze.analyze(tmp_path / "a.mp4", get_style("sport"), proj, vision=False, use_scenes=False)
    hl = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
    assert not any(c.get("pool") for c in hl["candidates"])
    assert "Visual candidates skipped: no prompt cache" in caplog.text


def test_load_part_reads_this_runs_grids_while_analyze_holds_them(tmp_path):
    src = tmp_path / "GX01.mp4"
    with cache.hold_outputs(tmp_path, [ski.FEATURES_DIR]):
        _save_part(tmp_path, src, grids(beauty=bump(N, 40, 0.8)))
        assert not ski.features_path(tmp_path, src).exists()  # held back
        d = ski.load_part(tmp_path, src)
        assert d is not None and d["features"]["beauty"].max() == pytest.approx(0.8)
    assert ski.load_part(tmp_path, src) is None  # never committed
