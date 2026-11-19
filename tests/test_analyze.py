"""Analyze: read-error recovery and profanity gating."""

from __future__ import annotations

import dataclasses
import json
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import StubTranscriber, media_info, remap, set_config, write
from cutpoint import analyze, config, extras, features, reset_state
from cutpoint.ffmpeg_utils import MediaInfo
from cutpoint.styles import LAUGHTER, get_style


@pytest.mark.parametrize(
    ("readable_from", "expected"),
    [(133.0, (136.0, 133.0)), (None, None)],
    ids=["recovers", "nothing"],
)
def test_find_resume_point(monkeypatch, readable_from, expected):
    """Probes 5/10/20/40 s past the failure, then bisects back toward it; also returns where
    the unreadable stretch ends (the last failed probe's window, at most the first good one)."""
    monkeypatch.setattr(
        analyze, "read_window_ok", lambda src, t: readable_from is not None and t >= readable_from
    )
    got = analyze.find_resume_point(Path("a.mp4"), fail_t=100.0, duration=1000.0)
    assert got == (pytest.approx(expected) if expected else None)


@pytest.mark.parametrize(
    ("readable_from", "resume", "until"),
    [
        (100.5, 102.25, 101.25),  # damage under 5 s: the 5 s probe is good, bisection narrows it
        (138.0, 141.0, 138.0),  # bisected: last bad probe 135 + 3 s window, below the good 140
        (104.9, 106.0, 105.0),  # capped at the good probe: 103.75 + 3 s reaches past 105
    ],
    ids=["short", "bisected", "capped-at-good"],
)
def test_find_resume_point_says_where_the_unreadable_stretch_ends(
    monkeypatch, readable_from, resume, until
):
    monkeypatch.setattr(analyze, "read_window_ok", lambda src, t: t >= readable_from)
    got = analyze.find_resume_point(Path("a.mp4"), fail_t=100.0, duration=1000.0)
    assert got == (pytest.approx(resume), pytest.approx(until))
    assert until <= resume - analyze.RESUME_MARGIN_S  # never past the first good probe


@pytest.fixture
def recovering(home, tmp_path, monkeypatch):
    """``analyze.analyze(recover=True)`` of one source with ``analyze_source`` faked: the first
    part reports ``state.record`` as its read error, the recovered part none. ``state.seen``
    has each call's ``(offset, readable_from)``."""
    video = write(tmp_path / "videos" / "a.mp4")
    proj = tmp_path / "proj"
    state = SimpleNamespace(video=video, proj=proj, seen=[], record={}, probes=[], loud={})

    def fake_source(src, style, tr, use_scenes, max_cand, timings, offset=0.0, **kw):
        state.seen.append((offset, kw.get("readable_from")))
        info = MediaInfo(
            src, 300.0, 300.0, 640, 360, Fraction(30), "yuv420p", "h264", 0, None, None, None
        )
        rec = dict(state.record) if not offset else None
        n = int((300.0 - offset) * 10)  # with --recover the parts are picked after the loop
        x = np.arange(n) / 10
        loud = sum(
            (h * np.exp(-((x - t) ** 2)) for t, h in state.loud.get(offset, [])), np.zeros(n)
        )
        ps = analyze.PartSignals(n, {"loudness": loud}, [], [], [], [], [])
        return analyze.SourceResult(src, info, offset, 300.0 - offset, [], [], rec, ps)

    def probe_window(src, t):
        state.probes.append(t)
        return t >= 133.0

    monkeypatch.setattr(analyze, "Transcriber", StubTranscriber)
    monkeypatch.setattr(analyze, "analyze_source", fake_source)
    monkeypatch.setattr(analyze, "read_window_ok", probe_window)
    monkeypatch.setattr(analyze, "has_telemetry", lambda src, timeout=60: False)
    state.go = lambda: analyze.analyze(
        video, get_style("funny"), proj, recover=True, use_scenes=False, layers=False
    )
    return state


def _saved_record(state) -> dict:
    (path,) = state.proj.parent.rglob(analyze.READ_ERRORS_FILE)
    return json.loads(path.read_text())


def test_the_end_of_the_damage_is_saved_and_given_to_the_recovered_part(recovering):
    recovering.record = {"failure_time": 100.0}
    recovering.go()
    assert recovering.seen == [(0.0, None), (136.0, 133.0)]
    rec = _saved_record(recovering)
    assert (rec["resumed_at"], rec["unreadable_until"]) == (136.0, 133.0)


def test_a_cached_resume_point_hands_its_saved_end_of_the_damage_on(recovering):
    recovering.record = {"failure_time": 100.0, "resumed_at": 136.0, "unreadable_until": 90.0}
    recovering.go()
    assert recovering.probes == []  # nothing probed again
    assert recovering.seen == [(0.0, None), (136.0, 90.0)]
    assert _saved_record(recovering)["unreadable_until"] == 90.0


def test_an_old_record_without_the_end_of_the_damage_gives_the_part_no_lead_in(recovering):
    recovering.record = {"failure_time": 100.0, "resumed_at": 136.0}
    recovering.go()  # no crash
    assert recovering.probes == []
    assert recovering.seen == [(0.0, None), (136.0, None)]
    assert _saved_record(recovering)["unreadable_until"] is None


def test_the_recovered_parts_of_a_source_are_cut_at_one_floor(recovering):
    # #1085 A104: the part after the damage once had its own floor (a quarter of its 0.3), so
    # a 0.2 moment was kept there and dropped before the damage; now one floor holds for both
    recovering.record = {"failure_time": 100.0, "resumed_at": 136.0, "unreadable_until": 133.0}
    recovering.loud = {
        0.0: [(20.0, 1.0), (50.0, 0.5), (80.0, 0.2)],
        136.0: [(30.0, 0.3), (90.0, 0.2)],
    }
    recovering.go()
    hl = json.loads((recovering.proj / "highlights.json").read_text())
    w = get_style("funny").weights["loudness"]
    got = sorted(round(c["score"] / (100 * w), 2) for c in hl["candidates"])
    assert got == [0.3, 0.5, 1.0]


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


def test_reanalyze_refuses_a_null_sources_list(tmp_path):
    write(tmp_path / "highlights.json", json.dumps({"sources": None}))
    with pytest.raises(ValueError, match="records no analysis"):
        analyze.reanalyze(tmp_path)


def test_reanalyze_takes_a_file_without_style_and_uses_the_fallback_style(tmp_path, monkeypatch):
    """#1015: a bare {"sources": [...]} re-analyzes with the style fallback, not a ValueError."""
    src = str(tmp_path / "a.mp4")
    write(tmp_path / "highlights.json", json.dumps({"sources": [{"path": src}]}))
    seen = {}

    def fake_analyze(inputs, style, project, **kw):
        seen.update(inputs=inputs, style=style)
        return project / "highlights.json"

    monkeypatch.setattr(analyze, "analyze", fake_analyze)
    assert analyze.reanalyze(tmp_path) == tmp_path / "highlights.json"
    assert seen["inputs"] == [Path(src)]
    assert seen["style"].name == analyze._recorded_style({"sources": []}).name


def _remapped_project(home, tmp_path, monkeypatch, recorded):
    """Footage moved from ``old`` to ``new``; ``source_remap`` makes the copy reachable."""
    old, new = tmp_path / "old", tmp_path / "new"
    write(new / "a.mp4")
    path = str(old / "a.mp4")
    if recorded == "present":  # nothing moved: every recorded path exists as written
        there = str(new / "a.mp4")
        data = {"settings": {"inputs": [there]}, "sources": [{"path": there}]}
    elif recorded == "folder":
        write(old / "b.mp4")  # the folder input exists; only a.mp4 moved out of it
        data = {"settings": {"inputs": [str(old)]}, "sources": [{"path": path}]}
    elif recorded == "inputs":  # an older project that recorded the copy it read
        data = {"settings": {"inputs": [path]}, "sources": [{"path": str(new / "a.mp4")}]}
    elif recorded == "moved-folder":  # the original folder is gone, its copy is remapped
        data = {"settings": {"inputs": [str(old)]}, "sources": [{"path": path}]}
    else:  # moved-file
        data = {"settings": {"inputs": [path]}, "sources": [{"path": path}]}
    remap(home, monkeypatch, old, new)
    return write(tmp_path / "highlights.json", json.dumps(data))


@pytest.mark.parametrize("recorded", ["present", "moved-file", "moved-folder"])
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


@pytest.mark.parametrize("content", ["{}", '["style", "sources"]', '"style sources"', "5"])
def test_reanalyze_tells_a_missing_highlights_file_from_one_without_an_analysis(
    tmp_path, monkeypatch, content
):
    monkeypatch.setattr(analyze, "analyze", lambda *a, **k: pytest.fail("must not analyze"))
    with pytest.raises(FileNotFoundError, match="not found"):
        analyze.reanalyze(tmp_path)
    hl = write(tmp_path / "highlights.json", content)
    for run in (analyze.reanalyze, analyze.reanalyze_if_reachable):
        with pytest.raises(ValueError, match="records no analysis; run 'cutpoint analyze' again"):
            run(tmp_path)
    assert hl.read_text(encoding="utf-8") == content


def _stub_probe_and_transcriber(monkeypatch) -> list[Path]:
    """Stub the media probe and the transcriber; return the list of probed paths."""
    probed: list[Path] = []

    def fake_probe(path):
        probed.append(path)
        return media_info(
            path, 60.0, width=640, height=360, audio_streams=0,
            acodec=None, sample_rate=None, channels=None,
        )  # fmt: skip

    monkeypatch.setattr(analyze, "probe", fake_probe)
    monkeypatch.setattr(analyze, "Transcriber", StubTranscriber)
    return probed


@pytest.mark.parametrize("both", [False, True], ids=["original-only", "original-and-copy"])
def test_analyze_records_original_path_when_source_is_remapped(home, tmp_path, monkeypatch, both):
    old, new = tmp_path / "old", tmp_path / "new"
    write(old / "a.mp4")
    write(new / "a.mp4")
    remap(home, monkeypatch, old, new)
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


@pytest.mark.parametrize("edition", ["full", "lite"])
def test_analyze_records_the_available_features(home, tmp_path, monkeypatch, edition):
    write(tmp_path / "a.mp4")
    _stub_probe_and_transcriber(monkeypatch)
    monkeypatch.setattr(extras, "edition", lambda: edition)
    reset_state()
    proj = tmp_path / "proj"
    analyze.analyze(tmp_path / "a.mp4", get_style("funny"), proj, vision=False, use_scenes=False)
    hl = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
    expected = sorted(features.available(("events",) if edition == "full" else ()))
    assert hl["settings"]["features_available"] == expected
    assert ("vlm" in expected) is (edition == "full")


@pytest.mark.parametrize("folder", [False, True], ids=["file", "folder"])
def test_analyze_reads_a_missing_input_through_source_remap_and_records_the_original(
    home, tmp_path, monkeypatch, folder
):
    old, new = tmp_path / "old", tmp_path / "new"  # old is gone entirely
    write(new / "a.mp4")
    write(new / "skip" / "b.mp4")
    write(new / "c_unverified.mp4")  # source_remap_exclude: would be read at old, missing
    remap(home, monkeypatch, old, new, exclude=["*unverified*"])
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
    set_config(home, source_remap={drive + "\\Footage": str(new)})
    probed = _stub_probe_and_transcriber(monkeypatch)
    proj = tmp_path / "proj"
    analyze.analyze(
        Path(drive + "\\Footage\\a.mp4"), get_style("funny"), proj, vision=False, use_scenes=False
    )
    hl = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
    assert [s["path"] for s in hl["sources"]] == [r"\\nas\share\Footage\a.mp4"]
    assert [p.resolve() for p in probed] == [(new / "a.mp4").resolve()]


@pytest.mark.parametrize(
    ("top", "url", "model"), [(7, "http://ollama.example:11434", "m1"), (None, None, None)]
)
def test_analyze_stores_the_vision_options_and_reanalyze_passes_them_back(
    home, tmp_path, monkeypatch, top, url, model
):
    write(tmp_path / "a.mp4")
    _stub_probe_and_transcriber(monkeypatch)
    proj = tmp_path / "proj"
    analyze.analyze(
        tmp_path / "a.mp4",
        get_style("funny"),
        proj,
        vision=False,
        vision_top=top,
        ollama_url=url,
        ollama_model=model,
        use_scenes=False,
    )
    st = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))["settings"]
    assert (st["vision_top"], st["ollama_url"], st["ollama_model"]) == (top, url, model)

    seen: dict = {}
    monkeypatch.setattr(analyze, "analyze", lambda *a, **kw: seen.update(kw) or proj)
    analyze.reanalyze(proj)
    assert (seen["vision_top"], seen["ollama_url"], seen["ollama_model"]) == (top, url, model)


@pytest.mark.parametrize(
    "url",
    [
        "http://alice:s3cret@ollama.example:11434",
        "http://alice:s3c@ret@ollama.example:11434",
        "http://alice:s3c/ret@ollama.example:11434",
        "alice:s3cret@ollama.example",
    ],
    ids=["plain", "at-sign", "slash", "no-scheme"],
)
def test_analyze_records_the_ollama_url_without_its_login(home, tmp_path, monkeypatch, url):
    """A639: the password never reaches highlights.json; reanalyze replays the URL without it."""
    write(tmp_path / "a.mp4")
    _stub_probe_and_transcriber(monkeypatch)
    proj = tmp_path / "proj"
    analyze.analyze(
        tmp_path / "a.mp4", get_style("funny"), proj, vision=False, ollama_url=url, use_scenes=False
    )
    text = (proj / "highlights.json").read_text(encoding="utf-8")
    assert "s3c" not in text and "alice" not in text
    st = json.loads(text)["settings"]
    assert st["ollama_url"] in ("http://ollama.example:11434", "ollama.example")

    seen: dict = {}
    monkeypatch.setattr(analyze, "analyze", lambda *a, **kw: seen.update(kw) or proj)
    analyze.reanalyze(proj)
    assert seen["ollama_url"] == st["ollama_url"]


def test_reanalyze_keeps_the_analysis_time_and_records_the_rerank_time(home, tmp_path, monkeypatch):
    """A re-rank (``events``, ``crashes --reanalyze``) does not pass for the analysis' time."""
    write(tmp_path / "a.mp4")
    _stub_probe_and_transcriber(monkeypatch)
    proj = tmp_path / "proj"
    analyze.analyze(tmp_path / "a.mp4", get_style("funny"), proj, vision=False, use_scenes=False)
    path = proj / "highlights.json"
    first = json.loads(path.read_text(encoding="utf-8"))["timings"]
    assert "rerank_s" not in first
    hl = json.loads(path.read_text(encoding="utf-8"))
    hl["timings"]["total_s"] = 1234.5  # the first analysis, transcription and all
    path.write_text(json.dumps(hl), encoding="utf-8")
    for _ in range(2):  # a second re-rank keeps the first analysis' time too
        analyze.reanalyze(proj)
        timings = json.loads(path.read_text(encoding="utf-8"))["timings"]
        assert timings["total_s"] == 1234.5
        assert 0 <= timings["rerank_s"] < 1234.5


def test_reanalyze_of_an_analysis_without_a_total_passes_no_earlier_total(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    write(proj / "highlights.json", json.dumps({"style": "funny", "sources": [], "settings": {
        "inputs": [str(tmp_path / "a.mp4")]}, "timings": {}}))  # fmt: skip
    seen: dict = {}
    monkeypatch.setattr(analyze, "analyze", lambda *a, **kw: seen.update(kw) or proj)
    analyze.reanalyze(proj)
    assert seen["earlier_total_s"] is None


@pytest.mark.parametrize("pano", [False, True])
def test_reanalyze_asks_again_for_an_action_cam_style_that_fell_back(tmp_path, monkeypatch, pano):
    """#939: a project whose action-cam request ran as action re-runs with the request (the
    telemetry probe decides again); 360 projects never fall back."""
    proj = tmp_path / "proj"
    settings = {
        "inputs": [str(tmp_path / "a.mp4")],
        "must_include_crash": 0.8,
        "style_requested": "action-cam",
        "style_fallback": {"from": "action-cam", "reason": "no telemetry"},
    }
    write(
        proj / "highlights.json",
        json.dumps({"style": "action", "sources": [], "settings": settings}),
    )
    if pano:
        write(proj / "pano.json", json.dumps({"clips": []}))
    seen: list = []
    monkeypatch.setattr(analyze, "analyze", lambda *a, **kw: seen.append((a[1], kw)) or proj)
    analyze.reanalyze(proj)
    [(style, kw)] = seen
    assert (style.name, style.must_include_crash) == ("action-cam", 0.8)
    assert kw["telemetry_fallback"] is not pano


@pytest.mark.parametrize(
    ("settings", "vision"),
    [
        ({"vision_requested": True, "vision_used": False}, True),  # asked; Ollama was down
        ({"vision_requested": False, "vision_used": False}, False),
        # before vision was opt-in: an automatic run is no request, so no Ollama call now
        ({"vision_used": True}, False),
        ({}, False),
    ],
)
def test_reanalyze_uses_vision_only_when_the_earlier_run_asked_for_it(
    tmp_path, monkeypatch, settings, vision
):
    proj = tmp_path / "proj"
    write(proj / "highlights.json", json.dumps({"style": "funny", "sources": [], "settings": {
        "inputs": [str(tmp_path / "a.mp4")], **settings}}))  # fmt: skip
    seen: dict = {}
    monkeypatch.setattr(analyze, "analyze", lambda *a, **kw: seen.update(kw) or proj)
    analyze.reanalyze(proj)
    assert seen["vision"] is vision


def _words(probs, texts=None):
    texts = texts or [f"w{i}" for i in range(len(probs))]
    return [
        {"start": float(i), "end": i + 0.5, "word": t, **({} if p is None else {"probability": p})}
        for i, (t, p) in enumerate(zip(texts, probs, strict=True))
    ]


@pytest.mark.parametrize(
    ("chars", "probs", "texts", "text", "unsure"),
    [
        (
            None,
            [0.9, 0.4, 0.25],
            ["hello", "big", "world"],
            "hello big world",
            [(1, 0.4), (2, 0.25)],
        ),
        (8, [0.9, 0.3, 0.1], ["hello", "big", "world"], "hello bi", [(1, 0.3)]),  # cut mid-word
        (6, [0.2, 0.1], ["hello", "big"], "hello", [(0, 0.2)]),  # cut at a space: no empty word
        (  # spaced words split, empty ones skipped; indexes are of the snippet's words
            None,
            [0.9, 0.3, 0.2, 0.1],
            ["hello", "", "new  york", "now"],
            "hello new york now",
            [(1, 0.2), (2, 0.2), (3, 0.1)],
        ),
    ],
    ids=["by-index", "cut-mid-word", "cut-at-space", "spaced-and-empty"],
)
def test_transcript_snippet_lists_unsure_words_by_index(
    monkeypatch, chars, probs, texts, text, unsure
):
    if chars is not None:
        monkeypatch.setattr(analyze, "SNIPPET_CHARS", chars)
    got, got_unsure = analyze._transcript_snippet(_words(probs, texts))
    assert (got, got_unsure) == (text, unsure)


@pytest.mark.parametrize(
    "words",
    [_words([0.1, None]), [], _words([0.9, 0.8]), _words([analyze.UNSURE_P, 0.9])],
    ids=["missing-probability", "empty", "all-confident", "at-the-threshold"],
)
def test_transcript_snippet_without_unsure_words(words):
    assert analyze._transcript_snippet(words)[1] is None


@pytest.mark.parametrize(
    ("unsure", "present"),
    [([(0, 0.3)], True), ([], False)],
    ids=["set", "empty"],
)
def test_candidate_record_emits_transcript_unsure_only_when_set(unsure, present):
    c = analyze.Candidate(
        Path("a.mp4"), 1.0, 5.0, 3.0, 1.0, {}, transcript="hi", transcript_unsure=unsure
    )
    rec = analyze._candidate_record(c, 0, get_style("funny"))
    assert ("transcript_unsure" in rec) is present
    if present:
        assert rec["transcript_unsure"] == unsure


def test_pick_candidates_carry_word_confidence():
    style = get_style("funny")
    n = 100
    loud = np.zeros(n)
    loud[40:50] = 1.0
    words = [
        {"start": i * 0.5, "end": 0.4 + i * 0.5, "word": f"w{i}", "probability": 0.3 + i % 2}
        for i in range(n * 2)
    ]
    ps = analyze.PartSignals(n, {"loudness": loud}, [], [], words, [], [])
    res = analyze.SourceResult(Path("a.mp4"), None, 0.0, float(n), [], [], None, ps)
    cands = analyze.pick_candidates(res, style, 5, {})
    assert cands
    c = next(c for c in cands if c.transcript)
    assert c.transcript_unsure
    toks = c.transcript.split(" ")
    assert all(p == 0.3 for _, p in c.transcript_unsure)
    whole = [i for i, _ in c.transcript_unsure if i < len(toks) - 1]  # the last may be cut
    assert all(int(toks[i][1:]) % 2 == 0 for i in whole)


def _said(*pieces: str, step: float = 1.0) -> list[dict]:
    """Transcript words, one per second, as Whisper writes them (a leading space each)."""
    return [
        {"word": f" {p}", "start": i * step, "end": i * step + 0.5} for i, p in enumerate(pieces)
    ]


@pytest.mark.parametrize(
    "marker", ["[laughter]", "(laughs)", "*laughing*", "[LAUGHTER]", "(Laughing)", "[giggles]"]
)
def test_whisper_laughter_markers_are_laughter_hits(marker):
    (hit,) = analyze.laughter_hits(_said("so", marker, "anyway"))
    assert hit[0] == 1.0 and hit[1] == 1.0
    assert hit[2] == marker[1:-1].lower()


def test_a_marker_of_several_words_is_one_hit_at_its_first_word():
    (hit,) = analyze.laughter_hits(_said("ok", "[light", "laughter]", "right"))
    assert hit == (1.0, 1.0, "light laughter")


@pytest.mark.parametrize(
    "words",
    [
        ("he", "said", "laughter", "is", "funny"),
        ("stop", "laughing", "at", "me"),
        ("i", "laugh", "every", "time"),
        ("Laughter,", "he", "says."),
        ("the", "(laughing", "stock", "of", "town"),  # no closing mark
        ("what", "the", "f***", "are", "you", "laughing", "at", "s***"),  # censored profanity
        ("so", "(laughs]", "anyway"),  # mismatched delimiters
        ("so", "[laughs)", "anyway"),
    ],
)
def test_the_spoken_word_laugh_or_laughter_is_not_a_laughter_hit(words):
    assert analyze.laughter_hits(_said(*words)) == []


def test_each_laughter_word_counts_once():
    hits = analyze.laughter_hits(_said("haha", "hah", "hahaha", "hehe", "lmao", "lol"))
    assert [(t, p) for t, _, p in hits] == [
        (0.0, "haha"),
        (1.0, "hah"),
        (2.0, "haha"),
        (3.0, "hehe"),
        (4.0, "lmao"),
        (5.0, "lol"),
    ]
    assert [w for _, w, _ in hits] == [1.0, 0.8, 1.0, 0.8, 1.0, 0.8]  # the strongest match


def test_keyword_hits_count_every_match_unless_asked_for_one():
    assert len(analyze.keyword_hits(_said("haha"), LAUGHTER)) == 2
    assert len(analyze.keyword_hits(_said("haha"), LAUGHTER, once=True)) == 1


def test_laughing_interjections_still_count_beside_the_markers():
    hits = analyze.laughter_hits(_said("haha", "no", "[laughs]", "lol"))
    assert {(t, p) for t, _, p in hits} >= {(0.0, "haha"), (2.0, "laughs"), (3.0, "lol")}


def test_the_laughter_signal_comes_from_the_markers_only_for_the_word():
    n = 100
    ps = analyze.PartSignals(
        n=n,
        sig={k: np.zeros(n) for k in ("loudness", "speech", "laughter", "scene", "continuity")},
        kw_hits=[],
        laugh_hits=[],
        words=_said("laughter", "laughter", "laugh"),
        segments=[],
        cuts=[],
    )
    analyze.text_signals(ps, get_style("gaming"), "a.mp4")
    assert ps.laugh_hits == [] and not ps.sig["laughter"].any()
    ps.words = _said("so", "[laughter]")
    analyze.text_signals(ps, get_style("gaming"), "a.mp4")
    assert [p for *_, p in ps.laugh_hits] == ["laughter"] and ps.sig["laughter"].max() > 0.3


def test_analyze_records_features_by_source(home, tmp_path, monkeypatch):
    write(tmp_path / "a.mp4")
    _stub_probe_and_transcriber(monkeypatch)
    reset_state()
    proj = tmp_path / "proj"
    analyze.analyze(tmp_path / "a.mp4", get_style("funny"), proj, vision=False, use_scenes=False)
    hl = json.loads((proj / "highlights.json").read_text(encoding="utf-8"))
    st = hl["settings"]
    assert {c["source"] for c in hl["candidates"]} <= set(st["features_by_source"])
    assert all(v == st["features_available"] for v in st["features_by_source"].values())


def test_reanalyze_passes_the_recorded_context_back(home, tmp_path, monkeypatch):
    write(tmp_path / "a.mp4")
    _stub_probe_and_transcriber(monkeypatch)
    proj = tmp_path / "proj"
    seen: list[dict] = []
    real = analyze.analyze
    for words in (None, "chest mount"):
        analyze.analyze(
            tmp_path / "a.mp4", get_style("funny"), proj, vision=False, use_scenes=False,
            context=words,
        )  # fmt: skip
        monkeypatch.setattr(analyze, "analyze", lambda *a, **kw: seen.append(kw) or proj)
        analyze.reanalyze(proj)
        monkeypatch.setattr(analyze, "analyze", real)
    assert [kw["context"] for kw in seen] == [None, "chest mount"]


# --------------------------------------------------------------------------- delivery cue (#1071)
def test_a_hype_word_said_flat_keeps_a_third_of_its_weight_and_a_cued_one_all():
    quiet = np.zeros(1000)
    kw = [
        (10.0, 1.0, "oh my god"),  # laughter 1 s later: cue
        (50.0, 1.0, "oh my god"),  # alone and flat
        (90.0, 1.0, "wow"),  # a different strong keyword 2 s later: cue for both
        (92.0, 1.0, "no way"),
    ]
    got = analyze.gate_delivery(kw, [(11.0, 1.0, "laugh")], quiet)
    assert [w for _, w, _ in got] == [1.0, analyze.DELIVERY_FLAT, 1.0, 1.0]


def test_a_filler_said_twice_in_a_row_is_no_cue_for_itself():
    kw = [(10.0, 1.0, "wow"), (11.0, 1.0, "wow")]
    got = analyze.gate_delivery(kw, [], np.zeros(1000))
    assert [w for _, w, _ in got] == [analyze.DELIVERY_FLAT] * 2


def test_a_weak_keyword_is_no_cue_and_the_result_does_not_depend_on_order():
    kw = [(10.0, 1.0, "wow"), (11.0, 0.4, "yes"), (60.0, 1.0, "no way"), (61.0, 0.9, "wow")]
    quiet = np.zeros(1000)
    got = analyze.gate_delivery(kw, [], quiet)
    again = analyze.gate_delivery(kw[::-1], [], quiet)
    assert sorted(got) == sorted(again)
    assert got[0][1] == analyze.DELIVERY_FLAT  # "yes" at 0.4 is below 0.8
    assert got[2][1] == 1.0  # "wow" 0.9 beside it is strong


def test_profane_hits_are_left_to_the_profanity_gate():
    swear = sorted(analyze.PROFANE)[0]
    kw = [(10.0, 0.8, swear)]
    assert analyze.gate_delivery(kw, [], np.zeros(1000)) == kw


@pytest.mark.parametrize(
    ("at", "expected"),
    [
        ({}, False),  # silence
        ({505: 0.25}, True),  # 0.25 above a silent base
        ({505: 0.15}, False),  # a small bump
        ({515: 0.3}, True),  # up to 2 s after the word
        ({525: 0.3}, False),  # later than that belongs to something else
        ({492: 0.3}, True),  # up to 1 s before it
        ({485: 0.3}, False),
        ({505: 0.5}, True),  # as loud as a profane word needs, whatever the base
    ],
)
def test_loudness_jump_at_a_word_at_50_s(at, expected):
    x = np.zeros(1000)
    for i, v in at.items():
        x[i] = v
    assert analyze.loudness_jump(x, 50.0) is expected


def test_a_loud_stretch_before_the_word_is_the_base_not_a_jump():
    x = np.full(1000, 0.3)  # steady 0.3 over the 9 s before
    x[505] = 0.4  # only 0.1 above it
    assert not analyze.loudness_jump(x, 50.0)
    x[505] = 0.55  # absolute 0.5
    assert analyze.loudness_jump(x, 50.0)
    x = np.full(1000, 0.1)
    x[505] = 0.35  # 0.25 above the median
    assert analyze.loudness_jump(x, 50.0)


def _part(words, loudness):
    n = len(loudness)
    sig = {k: np.zeros(n) for k in ("loudness", "speech", "laughter", "scene", "continuity")}
    sig["loudness"] = loudness
    return analyze.PartSignals(n, sig, [], [], words, [], [])


def _phrase_words(*items):
    """Word dicts for ``(second, phrase)`` items (a phrase's words 0.3 s apart)."""
    out = []
    for t, phrase in items:
        for i, w in enumerate(phrase.split()):
            out.append(
                {"start": t + 0.3 * i, "end": t + 0.3 * i + 0.25, "word": w, "probability": 1.0}
            )
    return out


def _hits(style, items, n=1000, loudness=None):
    ps = _part(_phrase_words(*items), np.zeros(n) if loudness is None else loudness)
    analyze.text_signals(ps, style, "a.mp4")
    return ps.kw_hits


@pytest.mark.parametrize(
    ("style_name", "flat"), [("gaming", True), ("action", True), ("funny", False)]
)
def test_only_delivery_cue_styles_cut_a_flat_hype_word(style_name, flat):
    style = get_style(style_name)
    (hit,) = _hits(style, [(30.0, "wow")])
    assert hit[1] == pytest.approx(style.keywords["wow"] * (analyze.DELIVERY_FLAT if flat else 1))


def test_a_hype_word_said_loudly_keeps_its_weight_in_a_cue_style():
    loud = np.zeros(1000)
    loud[305] = 0.6
    (hit,) = _hits(get_style("gaming"), [(30.0, "wow")], loudness=loud)
    assert hit[1] == pytest.approx(get_style("gaming").keywords["wow"])


def test_a_repeated_phrase_is_damped_before_the_delivery_cue_judges_it():
    # "wow" (0.8) twelve times in 300 s: factor 0.5, so 0.4, below the 0.8 a cue needs. Undamped
    # it would cue the "no way" beside it; damped it does not, but "no way" cues it.
    style = get_style("gaming")
    items = [(10.0 + 20 * i, "wow") for i in range(1, 13)] + [(9.0, "no way")]
    hits = {(round(t), p): w for t, w, p in _hits(style, items, n=3000)}
    assert hits[(30, "wow")] == pytest.approx(0.8 * 0.5 * analyze.DELIVERY_FLAT)
    assert hits[(9, "no way")] == pytest.approx(style.keywords["no way"] * analyze.DELIVERY_FLAT)


def test_a_style_without_repeat_damping_keeps_the_weights():
    style = get_style("funny")
    items = [(10.0 + 20 * i, "wow") for i in range(12)]
    assert all(w == pytest.approx(0.4) for _, w, _ in _hits(style, items, n=3000))
    off = dataclasses.replace(style, repeat_damping=False)
    assert all(w == pytest.approx(0.8) for _, w, _ in _hits(off, items, n=3000))


def test_a_label_names_the_keyword_by_its_style_weight_not_its_damped_one():
    # a flat "oh my god" (1.0, cut to 0.3 with no cue) and a "wow" (0.8) on a loudness jump in
    # one candidate: the label is "oh my god"; a lone "yes" (0.4, under 0.5) names nothing
    style = get_style("gaming")
    loud = np.zeros(1000)
    loud[195:215] = 0.8  # a jump at 20 s only
    ps = _part(_phrase_words((10.0, "oh my god"), (20.0, "wow"), (50.0, "yes")), loud)
    analyze.text_signals(ps, style, "a.mp4")
    weights = {p: w for _, w, p in ps.kw_hits}
    assert weights["oh my god"] < weights["wow"]  # the score sees the damped weight
    cands = [
        analyze.Candidate(Path("a.mp4"), 5.0, 25.0, 20.0, 1.0, {}),
        analyze.Candidate(Path("a.mp4"), 45.0, 55.0, 50.0, 1.0, {}),
    ]
    both, alone = analyze.describe_candidates(cands, style, ps, 0.0)
    assert both.keywords[0] == "wow" and both.label_hint == "oh my god"
    assert alone.keywords == ["yes"] and alone.label_hint == ""


def test_reanalyze_takes_null_settings(tmp_path):
    """A100: "settings": null is an empty record, not an AttributeError."""
    here = tmp_path / "gone.mp4"
    write(
        tmp_path / "highlights.json",
        json.dumps({"sources": [{"path": str(here)}], "settings": None}),
    )
    with pytest.raises(Exception) as err:
        analyze.reanalyze(tmp_path)
    assert not isinstance(err.value, AttributeError)
    assert analyze.reanalyze_if_reachable(tmp_path) is None  # source gone: left alone


def test_reanalyze_if_reachable_tolerates_null_or_odd_sources(tmp_path, monkeypatch):
    sentinel = tmp_path / "result.json"
    monkeypatch.setattr(analyze, "reanalyze", lambda project: sentinel)
    for sources in (None, ["x", {"name": "a"}, {"path": None}]):
        (tmp_path / "highlights.json").write_text(
            json.dumps({"sources": sources}), encoding="utf-8"
        )
        assert analyze.reanalyze_if_reachable(tmp_path) is sentinel


def test_reanalyze_skips_recorded_sources_without_a_path(tmp_path, monkeypatch):
    """Non-dict entries or entries without a str path do not crash the inputs fallback."""
    seen = {}

    def fake_analyze(inputs, *a, **k):
        seen["inputs"] = inputs
        return tmp_path / "highlights.json"

    monkeypatch.setattr(analyze, "analyze", fake_analyze)
    write(
        tmp_path / "highlights.json",
        json.dumps({"sources": ["x", {"name": "n"}, {"path": 5}, {"path": "/v/a.mp4"}]}),
    )
    analyze.reanalyze(tmp_path)
    assert [str(p).replace("\\", "/") for p in seen["inputs"]] == ["/v/a.mp4"]
