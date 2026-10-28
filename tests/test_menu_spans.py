"""Menu time in highlight candidates: menu spans from the OCR frames, candidates dropped or
trimmed in ``analyze`` and in the ``events`` re-rank (#440)."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from conftest import write
from cutpoint import cache, sources
from cutpoint.events import pipeline, scoring
from cutpoint.events.menu_spans import MENU_DROP_SHARE, drop_menu_time, merge_menu_frames
from cutpoint.styles import get_style

SRC = "/footage/rec.mp4"


def _cand(start: float, end: float, score: float = 40.0, src: str = SRC) -> dict[str, Any]:
    return {
        "id": f"h{int(start):03d}",
        "source": src,
        "start": start,
        "end": end,
        "duration": round(end - start, 3),
        "score": score,
        "transcript": "old words",
    }


def _hl(*cands: dict[str, Any], style: str = "gaming") -> dict[str, Any]:
    return {"style": style, "settings": {}, "candidates": list(cands)}


def _ev(spans: list[list[float]], src: str = SRC, **more: Any) -> dict[str, Any]:
    stats = {Path(src).name: {"menu_frames": 1, "menu_spans": spans}}
    return {"sources": [src, *more.get("others", [])], "stats": {"ocr": stats}, "events": []}


# --------------------------------------------------------------------------- spans


def test_menu_frames_merge_into_spans() -> None:
    times = [n / 3 for n in range(60, 90)] + [n / 3 for n in range(120, 123)] + [50.0]
    assert merge_menu_frames(times, 3.0) == [[20.0, 30.0], [40.0, 41.0], [50.0, 50.33]]
    assert merge_menu_frames([], 3.0) == []


def test_a_frame_gap_splits_spans() -> None:
    # 20.0, 20.33, then 21.0: the frame at 20.67 was no menu
    assert merge_menu_frames([20.0, 20 + 1 / 3, 21.0], 3.0) == [[20.0, 20.67], [21.0, 21.33]]


# --------------------------------------------------------------------------- the rule


def test_mostly_menu_is_dropped() -> None:
    assert MENU_DROP_SHARE == 0.5
    hl = _hl(_cand(10, 20), _cand(30, 40), _cand(50, 60))
    drop_menu_time(hl, _ev([[12, 17], [30, 34.9], [55, 56]]))
    # 10-20: 5 of 10 s in the middle (half: dropped); 30-40: 4.9 s at the start (trimmed)
    assert [(c["start"], c["end"]) for c in hl["candidates"]] == [(34.9, 40), (50, 60)]


def test_start_and_end_in_menu_spans_are_trimmed() -> None:
    hl = _hl(_cand(10, 30), _cand(40, 60))
    drop_menu_time(hl, _ev([[5, 12], [27, 35], [55, 70]]))
    (a, b) = hl["candidates"]
    assert (a["start"], a["end"], a["duration"]) == (12, 27, 15)
    assert (b["start"], b["end"], b["duration"]) == (40, 55, 15)
    assert a["score"] == b["score"] == 40.0  # the analysis score stays


def test_trim_keeps_the_style_minimum_length() -> None:
    assert get_style("commentary").min_len == 20.0
    hl = _hl(_cand(10, 40), style="commentary")
    # 11 of 30 s in menu time: the start moves to 12 (28 s left), the end would leave 19 s
    drop_menu_time(hl, _ev([[5, 12], [31, 50]]))
    (c,) = hl["candidates"]
    assert (c["start"], c["end"]) == (12, 40)


def test_unknown_style_is_never_trimmed() -> None:
    hl = _hl(_cand(10, 20), style="no-such-style")
    drop_menu_time(hl, _ev([[8, 12]]))
    assert (hl["candidates"][0]["start"], hl["candidates"][0]["end"]) == (10, 20)


def test_menu_time_in_the_middle_is_kept_uncut(caplog) -> None:
    caplog.set_level(logging.INFO, logger="cutpoint.events.menu_spans")
    hl = _hl(_cand(10, 30))
    before = copy.deepcopy(hl)
    drop_menu_time(hl, _ev([[15, 20]]))
    assert hl == before
    assert [r.getMessage() for r in caplog.records] == [
        "rec.mp4: menu time in candidates: 0 dropped (0.0s), 0 trimmed (0.0s cut), "
        "1 kept with 5.0s of menu time inside"
    ]


def test_no_menu_spans_leave_the_analysis_as_it_was(caplog) -> None:
    caplog.set_level(logging.INFO, logger="cutpoint.events.menu_spans")
    hl = _hl(_cand(10, 20), _cand(30, 40, src="/footage/other.mp4"))
    before = copy.deepcopy(hl)
    for ev in (
        _ev([]),  # an OCR read without menu frames
        _ev([[100, 120]]),  # menu time outside the candidates
        {"sources": [SRC], "stats": {"ocr": {"rec.mp4": {"skipped": "no preset"}}}},
        {"sources": [SRC], "stats": {}},
        {},
    ):
        drop_menu_time(hl, ev)
        assert hl == before
    assert caplog.records == []


def test_sources_sharing_a_name_are_not_filtered() -> None:
    hl = _hl(_cand(10, 20))
    drop_menu_time(hl, _ev([[10, 20]], others=["/elsewhere/rec.mp4"]))
    assert len(hl["candidates"]) == 1


def test_log_line_counts_dropped_and_trimmed(caplog) -> None:
    caplog.set_level(logging.INFO, logger="cutpoint.events.menu_spans")
    hl = _hl(_cand(10, 20), _cand(30, 50), _cand(60, 70))
    drop_menu_time(hl, _ev([[9, 19], [28, 32]]))
    assert [r.getMessage() for r in caplog.records] == [
        "rec.mp4: menu time in candidates: 1 dropped (10.0s), 1 trimmed (2.0s cut), "
        "0 kept with 0.0s of menu time inside"
    ]


def test_a_trimmed_candidate_gets_the_transcript_of_what_is_left(tmp_path, home) -> None:
    video = write(tmp_path / "v.mp4")
    words = [
        {"word": " menu", "start": 11.0, "end": 11.5, "probability": 0.9},
        {"word": " go", "start": 14.0, "end": 14.3, "probability": 0.2},
        {"word": " now", "start": 15.0, "end": 15.3, "probability": 0.9},
    ]
    segment = {"start": 11.0, "end": 15.3, "text": "menu go now", "words": words}
    cache.save_json(cache.source_dir(video) / "transcript-small.json", [segment])
    hl = _hl(_cand(10, 30, src=str(video)))
    drop_menu_time(hl, _ev([[8, 13]], src=str(video)))
    (c,) = hl["candidates"]
    assert (c["start"], c["transcript"], c["transcript_unsure"]) == (13, "go now", [(0, 0.2)])


# --------------------------------------------------------------------------- callers


def _project(tmp_path: Path, spans: list[list[float]]) -> Path:
    project = tmp_path / "proj"
    project.mkdir()
    hl = _hl(_cand(10, 20, score=50.0), _cand(30, 50, score=40.0))
    hl["settings"]["target_total"] = 60
    cache.save_json(project / "highlights.json", hl)
    cache.save_json(project / "events.json", {**_ev(spans), "created": "t", "windows": []})
    return project


def test_the_events_rerank_drops_and_trims(tmp_path, home) -> None:
    project = _project(tmp_path, [[9, 19], [28, 32]])
    scoring.rescore_highlights(project)
    hl = cache.load_json(project / "highlights.json")
    assert [(c["id"], c["start"], c["end"]) for c in hl["candidates"]] == [("h030", 32, 50)]
    assert hl["suggested_top"] == 1
    # the default flow filters twice (analyze, then the events re-rank): a second pass is stable
    scoring.rescore_highlights(project)
    assert cache.load_json(project / "highlights.json")["candidates"] == hl["candidates"]


def test_the_events_rerank_without_menu_time_is_unchanged(tmp_path, home) -> None:
    project = _project(tmp_path, [])
    scoring.rescore_highlights(project)
    with_spans = cache.load_json(project / "highlights.json")
    other = tmp_path / "other"
    other.mkdir()
    project2 = _project(other, [])
    ev = cache.load_json(project2 / "events.json")
    del ev["stats"]  # an events.json written before menu spans
    cache.save_json(project2 / "events.json", ev)
    scoring.rescore_highlights(project2)
    assert with_spans == cache.load_json(project2 / "highlights.json")
    assert len(with_spans["candidates"]) == 2


def test_analyze_drops_candidates_in_menu_time(tmp_path, home, monkeypatch) -> None:
    from cutpoint import analyze  # noqa: PLC0415
    from cutpoint.ffmpeg_utils import MediaInfo  # noqa: PLC0415

    video = write(tmp_path / "videos" / "a.mp4")
    proj = tmp_path / "proj"

    def fake_analyze_source(src, style, tr, use_scenes, max_cand, timings, **kw):
        from fractions import Fraction  # noqa: PLC0415

        info = MediaInfo(
            src, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None
        )
        cands = [
            analyze.Candidate(src, 10.0, 20.0, 15.0, 1.0, {"loudness": 1.0}, score=50.0),
            analyze.Candidate(src, 30.0, 50.0, 35.0, 1.0, {"loudness": 1.0}, score=40.0),
        ]
        return analyze.SourceResult(src, info, 0.0, 60.0, cands, [], None)

    class StubTranscriber:
        model_name, device = "stub", "cpu"

        def __init__(self, **_):
            pass

        def close(self):
            pass

    monkeypatch.setattr(analyze, "Transcriber", StubTranscriber)
    monkeypatch.setattr(analyze, "analyze_source", fake_analyze_source)
    monkeypatch.setattr(analyze, "plan_layers", lambda *a: analyze.LayerPlan())
    monkeypatch.setattr(pipeline, "run_events", lambda *a, **k: pytest.fail("no layers"))
    ev = {**_ev([[9, 19], [28, 32]], src=str(video)), "created": "t", "windows": []}
    write(proj / "events.json", json.dumps(ev))
    analyze.analyze(video, get_style("gaming"), proj, vision=False, use_scenes=False)
    hl = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
    assert [(c["id"], c["start"], c["end"], c["duration"]) for c in hl["candidates"]] == [
        ("h001", 32, 50.0, 18.0)
    ]
    assert hl["suggested_top"] == 1


def _words(*said: tuple[str, float]) -> list[dict[str, Any]]:
    words = [{"word": f" {w}", "start": t, "end": t + 0.3, "probability": 0.9} for w, t in said]
    text = " ".join(w for w, _ in said)
    return [{"start": said[0][1], "end": said[-1][1] + 0.3, "text": text, "words": words}]


def test_a_trim_reads_the_analysis_model_transcript_once_per_source(
    tmp_path, home, monkeypatch
) -> None:
    video = write(tmp_path / "v.mp4")
    cdir = cache.source_dir(video)
    cache.save_json(cdir / "transcript-small.json", _words(("small", 14.0), ("later", 44.0)))
    cache.save_json(cdir / "transcript-large-v3.json", _words(("large", 14.0), ("later", 44.0)))
    loads = []
    real = sources.source_segments
    monkeypatch.setattr(sources, "source_segments", lambda *a: loads.append(a) or real(*a))
    hl = _hl(_cand(10, 30, src=str(video)), _cand(40, 60, src=str(video)))
    hl["settings"]["whisper_model"] = "small"
    drop_menu_time(hl, _ev([[8, 13], [38, 42]], src=str(video)))
    assert [c["transcript"] for c in hl["candidates"]] == ["small", "later"]
    assert loads == [(video, None, "small")]


@pytest.mark.parametrize(
    ("keywords", "label", "expected"),
    [
        (["insane", "wow"], "insane", (["wow"], "wow")),  # the strongest left names it
        (["insane", "bro"], "insane", (["bro"], "gaming-moment")),  # none strong left
        (["insane", "wow"], "laugh", (["wow"], "laugh")),  # not named after a keyword
    ],
)
def test_a_trim_drops_the_keywords_said_in_the_cut_off_menu(
    tmp_path, home, keywords, label, expected
) -> None:
    video = write(tmp_path / "v.mp4")
    said = [("insane", 11.0), ("wow", 15.0), ("bro", 16.0)]
    cache.save_json(cache.source_dir(video) / "transcript-small.json", _words(*said))
    c = _cand(10, 30, src=str(video))
    c["signals"], c["label"] = {"keywords": keywords, "laughter": 0.0}, label
    hl = _hl(c)
    drop_menu_time(hl, _ev([[8, 13]], src=str(video)))
    (c,) = hl["candidates"]
    assert (c["signals"]["keywords"], c["label"]) == expected
