"""Analyze: read-error recovery and profanity gating."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from conftest import write
from cutpoint import analyze, config
from cutpoint.ffmpeg_utils import MediaInfo
from cutpoint.styles import get_style


@pytest.mark.parametrize(
    ("readable_from", "expected"), [(133.0, 136.0), (None, None)], ids=["recovers", "nothing"]
)
def test_find_resume_point(monkeypatch, readable_from, expected):
    """Probes 5/10/20/40 s past the failure, then bisects back toward it."""
    monkeypatch.setattr(
        analyze, "read_window_ok", lambda src, t: readable_from is not None and t >= readable_from
    )
    got = analyze.find_resume_point(Path("a.mp4"), fail_t=100.0, duration=1000.0)
    assert got == (pytest.approx(expected) if expected else None)


def test_profanity_counts_only_next_to_another_cue():
    swear = sorted(analyze.PROFANE)[0]
    kw = [
        (10.0, 1.0, swear),  # laughter right after: kept
        (50.0, 1.0, swear),  # a strong keyword right after: kept
        (51.0, 0.9, "let's go"),
        (90.0, 1.0, swear),  # alone and quiet: dropped
    ]
    laugh = [(11.0, 1.0, "laugh")]
    kept = analyze.gate_profanity(kw, laugh, np.zeros(1000))
    assert [t for t, _, _ in kept] == [10.0, 50.0, 51.0]


def test_reanalyze_if_reachable_leaves_highlights_when_a_source_is_offline(
    tmp_path, monkeypatch, caplog
):
    here = tmp_path / "here.mp4"
    here.write_bytes(b"x")
    hl = tmp_path / "highlights.json"
    hl.write_text(
        json.dumps({"sources": [{"path": str(here)}, {"path": str(tmp_path / "gone.mp4")}]}),
        encoding="utf-8",
    )
    before = hl.read_bytes()
    monkeypatch.setattr(analyze, "reanalyze", lambda project: pytest.fail("must not reanalyze"))
    with caplog.at_level("WARNING"):
        assert analyze.reanalyze_if_reachable(tmp_path) is None
    assert "1 source(s) not reachable" in caplog.text
    assert hl.read_bytes() == before


def _remapped_project(home, tmp_path, monkeypatch, recorded):
    """Footage moved from ``old`` to ``new``; ``source_remap`` makes the copy reachable."""
    old, new = tmp_path / "old", tmp_path / "new"
    write(new / "a.mp4")
    path = str(old / "a.mp4")
    if recorded == "folder":
        write(old / "b.mp4")  # the folder input exists; only a.mp4 moved out of it
        data = {"settings": {"inputs": [str(old)]}, "sources": [{"path": path}]}
    elif recorded == "inputs":  # an older project that recorded the copy it read
        data = {"settings": {"inputs": [path]}, "sources": [{"path": str(new / "a.mp4")}]}
    else:
        data = {"sources": [{"path": path}]}
    write(home / "config.json", json.dumps({"source_remap": {str(old): str(new)}}))
    config.get_config.cache_clear()
    monkeypatch.setattr(config, "_prefix_forms", lambda p: (config.norm_path(p),))
    return write(tmp_path / "highlights.json", json.dumps(data))


@pytest.mark.parametrize("recorded", ["inputs", "sources", "folder"])
def test_reanalyze_if_reachable_leaves_highlights_when_a_path_is_only_remapped(
    home, tmp_path, monkeypatch, caplog, recorded
):
    hl = _remapped_project(home, tmp_path, monkeypatch, recorded)
    before = hl.read_bytes()
    monkeypatch.setattr(analyze, "reanalyze", lambda project: pytest.fail("must not reanalyze"))
    with caplog.at_level("WARNING"):
        assert analyze.reanalyze_if_reachable(tmp_path) is None
    assert "path(s) not found where analyze recorded them" in caplog.text
    assert hl.read_bytes() == before


def test_reanalyze_if_reachable_reanalyzes_when_every_recorded_path_exists(
    home, tmp_path, monkeypatch, caplog
):
    src = write(tmp_path / "a.mp4")
    write(
        tmp_path / "highlights.json",
        json.dumps({"settings": {"inputs": [str(src)]}, "sources": [{"path": str(src)}]}),
    )
    sentinel = tmp_path / "result.json"
    monkeypatch.setattr(analyze, "reanalyze", lambda project: sentinel)
    with caplog.at_level("WARNING"):
        assert analyze.reanalyze_if_reachable(tmp_path) is sentinel
    assert caplog.text == ""


@pytest.mark.parametrize("both", [False, True], ids=["original-only", "original-and-copy"])
def test_analyze_records_original_path_when_source_is_remapped(home, tmp_path, monkeypatch, both):
    old, new = tmp_path / "old", tmp_path / "new"
    write(old / "a.mp4")
    write(new / "a.mp4")
    write(home / "config.json", json.dumps({"source_remap": {str(old): str(new)}}))
    config.get_config.cache_clear()
    monkeypatch.setattr(config, "_prefix_forms", lambda p: (config.norm_path(p),))
    probed: list[Path] = []

    def fake_probe(path):
        probed.append(path)
        return MediaInfo(
            path, 60.0, 60.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None
        )

    class StubTranscriber:
        model_name, device = "stub", "cpu"

        def __init__(self, **_):
            pass

        def close(self):
            pass

    monkeypatch.setattr(analyze, "probe", fake_probe)
    monkeypatch.setattr(analyze, "Transcriber", StubTranscriber)
    proj = tmp_path / "proj"
    target = [old / "a.mp4", new / "a.mp4"] if both else old / "a.mp4"
    analyze.analyze(target, get_style("funny"), proj, vision=False, use_scenes=False)
    hl = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
    assert [s["path"] for s in hl["sources"]] == [str(old / "a.mp4")]
    assert (proj / "transcript.txt").read_text(encoding="utf-8").startswith(f"# {old / 'a.mp4'}")
    assert [p.resolve() for p in probed] == [(new / "a.mp4").resolve()]
