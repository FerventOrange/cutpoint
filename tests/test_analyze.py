"""Analyze: read-error recovery and profanity gating."""

from __future__ import annotations

import json
import sys
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


def _remap(home, monkeypatch, old, new, exclude=()):
    data = {"source_remap": {str(old): str(new)}, "source_remap_exclude": list(exclude)}
    write(home / "config.json", json.dumps(data))
    config.get_config.cache_clear()
    monkeypatch.setattr(config, "_prefix_forms", lambda p: (config.norm_path(p),))


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
    elif recorded == "moved-folder":  # the original folder is gone, its copy is remapped
        data = {"settings": {"inputs": [str(old)]}, "sources": [{"path": path}]}
    else:  # moved-file
        data = {"settings": {"inputs": [path]}, "sources": [{"path": path}]}
    _remap(home, monkeypatch, old, new)
    return write(tmp_path / "highlights.json", json.dumps(data))


@pytest.mark.parametrize("recorded", ["moved-file", "moved-folder"])
def test_reanalyze_if_reachable_reanalyzes_an_input_reachable_through_source_remap(
    home, tmp_path, monkeypatch, caplog, recorded
):
    _remapped_project(home, tmp_path, monkeypatch, recorded)
    sentinel = tmp_path / "result.json"
    monkeypatch.setattr(analyze, "reanalyze", lambda project: sentinel)
    with caplog.at_level("WARNING"):
        assert analyze.reanalyze_if_reachable(tmp_path) is sentinel
    assert caplog.text == ""


@pytest.mark.parametrize("recorded", ["inputs", "folder"])
def test_reanalyze_if_reachable_leaves_highlights_when_reanalysis_would_rekey_or_drop_a_source(
    home, tmp_path, monkeypatch, caplog, recorded
):
    hl = _remapped_project(home, tmp_path, monkeypatch, recorded)
    before = hl.read_bytes()
    monkeypatch.setattr(analyze, "reanalyze", lambda project: pytest.fail("must not reanalyze"))
    with caplog.at_level("WARNING"):
        assert analyze.reanalyze_if_reachable(tmp_path) is None
    assert "not found where analyze recorded them, even through source_remap" in caplog.text
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


def _stub_probe_and_transcriber(monkeypatch) -> list[Path]:
    """Stub the media probe and the transcriber; return the list of probed paths."""
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
    return probed


@pytest.mark.parametrize("both", [False, True], ids=["original-only", "original-and-copy"])
def test_analyze_records_original_path_when_source_is_remapped(home, tmp_path, monkeypatch, both):
    old, new = tmp_path / "old", tmp_path / "new"
    write(old / "a.mp4")
    write(new / "a.mp4")
    _remap(home, monkeypatch, old, new)
    probed = _stub_probe_and_transcriber(monkeypatch)
    proj = tmp_path / "proj"
    target = [old / "a.mp4", new / "a.mp4"] if both else old / "a.mp4"
    analyze.analyze(target, get_style("funny"), proj, vision=False, use_scenes=False)
    hl = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
    assert [s["path"] for s in hl["sources"]] == [str(old / "a.mp4")]
    assert (proj / "transcript.txt").read_text(encoding="utf-8").startswith(f"# {old / 'a.mp4'}")
    assert [p.resolve() for p in probed] == [(new / "a.mp4").resolve()]


def test_analyze_records_relative_input_as_absolute(home, tmp_path, monkeypatch):
    write(tmp_path / "videos" / "a.mp4")
    monkeypatch.chdir(tmp_path / "videos")
    expected = str(Path.cwd() / "a.mp4")
    _stub_probe_and_transcriber(monkeypatch)
    proj = tmp_path / "proj"
    analyze.analyze(Path("a.mp4"), get_style("funny"), proj, vision=False, use_scenes=False)
    hl = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
    assert hl["settings"]["inputs"] == [expected]


@pytest.mark.parametrize("folder", [False, True], ids=["file", "folder"])
def test_analyze_reads_a_missing_input_through_source_remap_and_records_the_original(
    home, tmp_path, monkeypatch, folder
):
    old, new = tmp_path / "old", tmp_path / "new"  # old is gone entirely
    write(new / "a.mp4")
    write(new / "skip" / "b.mp4")
    write(new / "c_unverified.mp4")  # source_remap_exclude: would be read at old, missing
    _remap(home, monkeypatch, old, new, exclude=["*unverified*"])
    probed = _stub_probe_and_transcriber(monkeypatch)
    proj = tmp_path / "proj"
    target = old if folder else old / "a.mp4"
    # The exclude is written against the original path, which only ``old`` spells.
    analyze.analyze(
        target,
        get_style("funny"),
        proj,
        vision=False,
        use_scenes=False,
        recursive=True,
        exclude=["*/old/skip/*"],
    )
    hl = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
    assert [s["path"] for s in hl["sources"]] == [str(old / "a.mp4")]
    assert hl["settings"]["inputs"] == [str(target)]
    assert [p.resolve() for p in probed] == [(new / "a.mp4").resolve()]


@pytest.mark.skipif(sys.platform != "win32", reason="mapped drives are a Windows feature")
def test_analyze_records_a_missing_mapped_drive_input_in_unc_form(home, tmp_path, monkeypatch):
    """A mapped drive is disconnected from its NAS share; the copy is read, UNC is recorded."""
    letter = next((c for c in "ZYXWVUTSRQPONMLKJIHGFED" if not Path(f"{c}:\\").exists()), None)
    if letter is None:
        pytest.skip("no free drive letter on this machine")
    drive = f"{letter}:"
    monkeypatch.setattr(config, "_drive_remote", lambda d: r"\\nas\share" if d == drive else None)
    new = tmp_path / "new"
    write(new / "a.mp4")
    write(home / "config.json", json.dumps({"source_remap": {drive + "\\Footage": str(new)}}))
    config.get_config.cache_clear()
    probed = _stub_probe_and_transcriber(monkeypatch)
    proj = tmp_path / "proj"
    analyze.analyze(
        Path(drive + "\\Footage\\a.mp4"), get_style("funny"), proj, vision=False, use_scenes=False
    )
    hl = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
    assert [s["path"] for s in hl["sources"]] == [r"\\nas\share\Footage\a.mp4"]
    assert [p.resolve() for p in probed] == [(new / "a.mp4").resolve()]
