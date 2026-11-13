"""Analyze: read-error recovery and profanity gating."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import StubTranscriber, media_info, remap, set_config, write
from cutpoint import analyze, config, extras, features, reset_state
from cutpoint.styles import LAUGHTER, get_style


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
