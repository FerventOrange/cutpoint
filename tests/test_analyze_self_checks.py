"""A plain ``cutpoint analyze`` runs the self-checks and records their findings as
``settings.self_checks`` in highlights.json: present only with findings (#1085 A587).
Whisper and the per-source analysis are faked; the checks and the saved file are real."""

from __future__ import annotations

import pytest

import cutpoint.analyze
from conftest import StubTranscriber, media_info, write
from cutpoint import cache, cli

DURATION = 2000.0


@pytest.fixture
def analyzed(home, tmp_path, monkeypatch):
    """``run(label)``: analyze one source whose 45 moments all carry ``label``; the project's
    highlights.json."""
    monkeypatch.chdir(tmp_path)
    video = write(tmp_path / "videos" / "a.mp4", "a")

    def run(label: str) -> dict:
        def analyze_source(src, style, tr, use_scenes, max_cand, timings, **kw):
            info = media_info(src, DURATION, width=640, height=360)
            cands = [
                cutpoint.analyze.Candidate(
                    src, 40.0 * i, 40.0 * i + 15.0, 40.0 * i + 7.0, 1.0, {"loudness": 1.0},
                    score=50.0 + i, label_hint=label,
                )
                for i in range(45)
            ]  # fmt: skip
            return cutpoint.analyze.SourceResult(src, info, 0.0, DURATION, cands, [], None)

        monkeypatch.setattr(cutpoint.analyze, "Transcriber", StubTranscriber)
        monkeypatch.setattr(cutpoint.analyze, "analyze_source", analyze_source)
        monkeypatch.setattr(
            cutpoint.analyze, "plan_layers", lambda *a: cutpoint.analyze.LayerPlan()
        )
        argv = ["analyze", str(video), "--style", "funny", "--no-vision", "--out", "p"]
        assert cli.main(argv) == 0
        return cache.load_json(home / "projects" / "p" / "highlights.json")

    return run


def test_a_label_on_most_moments_is_recorded_as_a_self_check(analyzed, caplog):
    hl = analyzed("boss-fight")
    assert len(hl["candidates"]) >= 40
    (finding,) = hl["settings"]["self_checks"]
    assert finding["code"] == "label_dominates" and "boss-fight" in finding["message"]
    assert "boss-fight" in caplog.text  # and warned about


def test_an_analysis_without_findings_has_no_self_checks_key(analyzed):
    hl = analyzed("")  # no hint: the fallback label is no detection
    assert hl["candidates"] and "self_checks" not in hl["settings"]
