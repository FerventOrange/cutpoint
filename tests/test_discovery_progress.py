"""The ``keyframes`` step's pct while HUD discovery and then the per-source OCR run
(fake discovery and OCR, invented game ``zonewar``)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutpoint import progress
from cutpoint.events import discover, pipeline
from cutpoint.events.killfeed import OcrParams


class _Stop(Exception):
    pass


def _lines(capsys: pytest.CaptureFixture[str]) -> list[dict]:
    return [
        json.loads(x.removeprefix("PROGRESS "))
        for x in capsys.readouterr().err.splitlines()
        if x.startswith("PROGRESS ")
    ]


@pytest.fixture
def ocr_run(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``run_events`` with the OCR layer only, two sources and fake discovery and OCR;
    it stops after the OCR stage. ``found`` is the preset ``load_preset`` returns."""
    monkeypatch.setenv(progress.ENV, "1")
    monkeypatch.setattr(progress, "MIN_INTERVAL_S", 0.0)  # every update is a line
    srcs = [
        SimpleNamespace(path=tmp_path / f"{n}.mp4", name=f"{n}.mp4", duration=600.0)
        for n in ("a", "b")
    ]
    by_path = {str(s.path): s for s in srcs}
    hl = {"sources": [{"path": str(s.path)} for s in srcs]}
    state = SimpleNamespace(found=None)
    monkeypatch.setattr(pipeline, "load_highlights", lambda project: hl)
    monkeypatch.setattr(pipeline, "preflight", lambda layers: None)
    monkeypatch.setattr(pipeline, "load_source", lambda path, dur: by_path[str(path)])
    monkeypatch.setattr(pipeline, "bad_ranges", lambda *a: {})
    monkeypatch.setattr(pipeline, "load_transcript_lines", lambda project: [])
    monkeypatch.setattr(pipeline, "detect_audio_events", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "load_preset", lambda *a: state.found)
    monkeypatch.setattr(pipeline, "OcrEngine", lambda: None)
    monkeypatch.setattr(pipeline, "detect_ocr_events", lambda *a, **k: ([], {}))

    def fake_discover(srcs, out, game, params, on_frame=None):
        for n in range(1, 9):
            on_frame(n, 8)
        return {"regions": []}

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(discover, "discover_regions", fake_discover)
    monkeypatch.setattr(pipeline, "game_prompt", stop)  # the VLM layer's first call

    def run() -> None:
        with progress.run(("keyframes",)) as steps:
            steps.start("keyframes")
            with pytest.raises(_Stop):
                pipeline.run_events(
                    tmp_path / "proj", layers=("ocr",), ocr=OcrParams(game="zonewar")
                )

    state.run = run
    return state


def test_discovery_takes_the_first_part_then_each_source(ocr_run, capsys):
    ocr_run.run()
    got = [(g["pct"], g["detail"]) for g in _lines(capsys) if g["state"] == "running"]
    pcts = [p for p, _ in got if p is not None]
    assert pcts == sorted(pcts)  # never backwards
    assert (
        int(pipeline.DISCOVERY_PCT / 8),
        "first run for zonewar: discovering HUD: 1/8 frames",
    ) in got
    assert (pipeline.DISCOVERY_PCT, "first run for zonewar: discovering HUD: 8/8 frames") in got
    assert got[-2:] == [
        (pipeline.DISCOVERY_PCT, "reading HUD text: source 1/2"),
        (75, "reading HUD text: source 2/2"),
    ]


def test_without_discovery_the_sources_take_the_whole_step(ocr_run, capsys):
    ocr_run.found = SimpleNamespace(feed={}, path=None)  # a preset matches: no discovery
    ocr_run.run()
    got = [(g["pct"], g["detail"]) for g in _lines(capsys) if g["state"] == "running"]
    assert got == [(None, None), (0, None), (50, None)]


def test_two_games_split_the_discovery_part(ocr_run, monkeypatch, capsys):
    games = {"a.mp4": "alphawar", "b.mp4": "betawar"}  # invented games
    monkeypatch.setattr(pipeline, "game_key_for_path", lambda path, *a: games[Path(path).name])
    ocr_run.run()
    got = [(g["pct"], g["detail"]) for g in _lines(capsys) if g["state"] == "running"]
    pcts = [p for p, _ in got if p is not None]
    assert pcts == sorted(pcts)  # never backwards
    second = [p for p, d in got if d and "betawar: discovering HUD:" in d]
    assert pipeline.DISCOVERY_PCT / 2 <= second[0] <= pipeline.DISCOVERY_PCT / 2 + 4
    found = [p for p, d in got if d and "discovering HUD:" in d]
    assert found[-1] == pipeline.DISCOVERY_PCT


def test_ocr_frames_report_cached_short_and_new_bursts(tmp_path, monkeypatch):
    import numpy as np  # noqa: PLC0415

    from cutpoint.events import killfeed  # noqa: PLC0415

    src = SimpleNamespace(name="rec.mp4", io_path=tmp_path / "rec.mp4")
    params = discover.DiscoverParams(ocr_workers=1)  # in-process: the fake engine applies
    bursts = [(src, 0.0), (src, 60.0), (src, 120.0)]
    cached = [f"rec.mp4@{k * params.burst_step_s:.1f}" for k in range(4)] + ["rec.mp4@60.0"]
    cache_path = tmp_path / "ocr.jsonl"
    cache_path.write_text(
        "".join(json.dumps({"id": i, "boxes": []}) + "\n" for i in cached), encoding="utf-8"
    )

    class Engine:
        def __init__(self, det_side=960):
            pass

        def __call__(self, img, min_score):
            return []

    monkeypatch.setattr(killfeed, "OcrEngine", Engine)
    monkeypatch.setattr(  # a short burst: 2 of 4 frames decode
        discover, "_burst_frames", lambda *a: [np.zeros((72, 128, 3), np.uint8)] * 2
    )
    calls: list[tuple[int, int]] = []
    frames = discover._ocr_frames(
        bursts, params, tmp_path / "frames", cache_path, lambda d, t: calls.append((d, t))
    )
    assert len(frames) == 4 + 2 + 2
    done = [d for d, _ in calls]
    assert done == sorted(done)
    assert {t for _, t in calls} == {12}
    assert calls[-1] == (12, 12)


def test_pct_rises_within_each_source(ocr_run, monkeypatch, capsys):
    def fake_ocr(*a, on_progress=None, **k):
        for share in (0.0, 0.25, 0.5, 0.5, 1.0):  # a repeat is no new line
            on_progress(share)
        return [], {}

    monkeypatch.setattr(pipeline, "detect_ocr_events", fake_ocr)
    ocr_run.run()
    got = [(g["pct"], g["detail"]) for g in _lines(capsys) if g["state"] == "running"]
    pcts = [p for p, _ in got if p is not None]
    assert pcts == sorted(pcts)  # never backwards
    first = got.index((pipeline.DISCOVERY_PCT, "reading HUD text: source 1/2"))
    # discovery took the first half: each source a quarter, in quarters of that
    assert [p for p, _ in got[first:]] == [50, 56, 62, 75, 75, 81, 87, 100]
