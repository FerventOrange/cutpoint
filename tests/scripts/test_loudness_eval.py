"""``scripts/loudness_eval.py``, the #512 loudness measurement harness: the metric maths, the
variant transforms (boom discount, voice tracks, reweight), the single-track no-op, the pooled
criteria and draft labels that never overwrite a label file. Synthetic signals only."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import load_script, write
from cutpoint import cache, features, scorer, transcribe
from cutpoint.analyze import DT, PartSignals, SourceResult
from cutpoint.styles import get_style

pytestmark = pytest.mark.script
le = load_script("loudness_eval")
GAMING = get_style("gaming")
N = 600  # 60 s on the analysis grid
GAME_T, REACT_T = 20.0, 45.0


def _bump(t: float, height: float, width: float = 0.5) -> np.ndarray:
    x = np.arange(N) * DT
    return height * np.exp(-0.5 * ((x - t) / width) ** 2)


def _signals(words=None, laugh=None) -> PartSignals:
    """A gaming recording: an unreacted explosion at 20 s (loudness 1), a laugh at 45 s
    (loudness 0.6 plus a transcript "haha")."""
    sig = {k: np.zeros(N) for k in ("loudness", "speech", "laughter", "scene", "continuity")}
    sig["loudness"] = _bump(GAME_T, 1.0) + _bump(REACT_T, 0.6)
    sig["laughter"] = _bump(REACT_T, 1.0)
    words = [{"word": " haha", "start": 45.0, "end": 45.4}] if words is None else words
    laugh = [(45.0, 1.0, "haha*")] if laugh is None else laugh
    segments = [
        {"start": w["start"], "end": w["end"], "text": w["word"], "words": [w]} for w in words
    ]
    return PartSignals(N, sig, [], laugh, words, segments, [])


def _result(src: Path, ps: PartSignals, streams: int = 1) -> SourceResult:
    info = SimpleNamespace(path=src, duration=N * DT, audio_streams=streams)  # the probe
    return SourceResult(src, info, 0.0, N * DT, [], [], None, ps)  # type: ignore[arg-type]


LABELS = [
    {"t": GAME_T, "kind": "game", "note": "explosion"},
    {"t": REACT_T, "kind": "reaction", "note": "laugh"},
]


def _cand(start: float, end: float, peak_t: float, score: float = 50.0, **kw) -> dict:
    return {"start": start, "end": end, "peak_t": peak_t, "score": score, **kw}


# --------------------------------------------------------------------------- parsing
def test_variants_expand_voice_and_keep_baseline_first() -> None:
    vs = le.parse_variants(["voice", "boom:0.50", "boom:0.5", "reweight:0.3", "baseline"])
    assert [v.name for v in vs] == [
        "baseline",
        "voice-mic",
        "voice-max",
        "boom:0.5",
        "reweight:0.3",
    ]
    assert vs[3].param == 0.5
    for bad in ("loud", "boom", "boom:x", "reweight:2", "boom:-1", "voice:1"):
        with pytest.raises(ValueError):
            le.parse_variants([bad])
    assert le.parse_tops(["5", "suggested", "05"]) == ["5", "suggested"]
    with pytest.raises(ValueError):
        le.parse_tops(["0"])


# --------------------------------------------------------------------------- metrics
def test_recall_counts_each_label_once_and_both_as_reaction() -> None:
    labels = [
        {"t": 10.0, "kind": "reaction"},
        {"t": 30.0, "kind": "both"},
        {"t": 50.0, "kind": "reaction"},
        {"t": 70.0, "kind": "game"},
        {"t": 90.0, "kind": "game"},
    ]
    ranked = [
        _cand(8, 14, 9),  # label 10
        _cand(9, 12, 11),  # label 10 again: still one hit
        _cand(28, 33, 30),  # both
        _cand(69, 75, 71.5),  # peak 1.5 s from the game label: FP
        _cand(85, 92, 87.0),  # contains a game label, peak 3 s away: reached, no FP
        _cand(48, 52, 50),  # outside the top 5
    ]
    m = le.top_metrics(ranked, 5, labels)
    assert (m["n"], m["reactions"], m["hit"], m["hit_ids"]) == (5, 3, 2, [0, 1])
    assert m["recall"] == round(2 / 3, 3)
    assert (m["fp"], m["fp_rate"]) == (1, 0.2)
    assert (m["games"], m["game_reached"], m["game_share"]) == (2, 2, 1.0)
    assert le.top_metrics(ranked, 6, labels)["hit"] == 3
    empty = le.top_metrics([], 5, labels)
    assert (empty["n"], empty["fp_rate"], empty["hit"]) == (0, None, 0)


def test_gained_lost_and_loud_share_against_baseline() -> None:
    labels = [{"t": 10.0, "kind": "reaction"}, {"t": 30.0, "kind": "reaction"}]
    base = le.evaluate([_cand(8, 12, 10, loud_share=0.8)], 1, labels, ["1", "suggested"])
    var = le.evaluate([_cand(28, 32, 30, loud_share=0.2)], 1, labels, ["1", "suggested"])
    le.compare(var, base)
    assert (var["at"]["1"]["gained"], var["at"]["1"]["lost"]) == (1, 1)
    assert var["loud_share"] == {"mean": 0.2, "max": 0.2}
    assert le.loud_share(0.45, 1.0, 0.45) == 1.0
    assert le.loud_share(0.45, 1.0, 0.0) == 0.0


def _clip(name: str, fp_base: int, fp_var: int, lost: int) -> dict:
    def row(fp: int, lost_: int) -> dict:
        return {
            "na": None,
            "at": {
                "suggested": {
                    "n": 4,
                    "hit": 3 - lost_,
                    "reactions": 3,
                    "fp": fp,
                    "fp_mixed": 0,
                    "games": 2,
                    "game_reached": fp,
                    "gained": 0,
                    "lost": lost_,
                }
            },
        }

    return {
        "source": name,
        "variants": {"baseline": {"pre": row(fp_base, 0)}, "boom:1": {"pre": row(fp_var, lost)}},
    }


def test_pooled_criteria() -> None:
    passing = le.pool(
        [_clip("a", 2, 1, 1), _clip("b", 2, 0, 0)], ["baseline", "boom:1"], ["suggested"]
    )
    p = passing["boom:1"]["pre"]
    assert p["at"]["suggested"]["fp"] == 1 and p["at"]["suggested"]["slots"] == 8
    c = p["criteria"]
    assert (c["fp_drop"], c["net"], c["max_1_lost_per_recording"]) == (0.75, -1, True)
    assert not c["no_net_loss"] and not c["pass"]  # one lost, none gained
    two_lost = le.pool([_clip("a", 2, 0, 2)], ["baseline", "boom:1"], ["suggested"])
    assert not two_lost["boom:1"]["pre"]["criteria"]["max_1_lost_per_recording"]
    none_lost = le.pool([_clip("a", 2, 1, 0)], ["baseline", "boom:1"], ["suggested"])
    assert none_lost["boom:1"]["pre"]["criteria"]["pass"]
    assert "criteria" not in none_lost["baseline"]["pre"]


# --------------------------------------------------------------------------- variants
def test_boom_discount_spares_speech_and_never_goes_negative() -> None:
    loud = np.array([1.0, 1.0, 1.0, 0.5])
    boom = np.array([0.5, 1.0, 1.0, 0.0])
    speech = np.array([False, False, True, False])
    assert np.allclose(le.boom_discount(loud, boom, speech, 1.0), [0.5, 0.0, 1.0, 0.5])
    assert np.allclose(le.boom_discount(loud, boom, speech, 2.0), [0.0, 0.0, 1.0, 0.5])


def test_boom_variant_protects_one_second_around_words_and_panns_voice() -> None:
    ps = _signals(words=[{"word": " go", "start": 30.0, "end": 30.2}])
    ps.sig["loudness"] = np.ones(N)
    boom = np.ones(N)
    voiced = np.zeros(N, dtype=bool)
    voiced[500] = True  # a PANNs scream at 50 s
    sig, na = le.variant_signals(le.Variant("boom:1", "boom", 1.0), ps, None, (boom, voiced))
    assert na is None
    loud = sig["loudness"]
    assert loud[int(29.1 / DT)] == 1.0 and loud[int(31.1 / DT)] == 1.0  # within 1 s of the word
    assert loud[int(28.5 / DT)] == 0.0 and loud[int(31.8 / DT)] == 0.0
    assert loud[int(49.2 / DT)] == 1.0 and loud[int(48.5 / DT)] == 0.0  # around the scream
    assert ps.sig["loudness"].min() == 1.0  # the baseline signals are untouched
    assert sig["speech"] is ps.sig["speech"]  # only loudness changes


def test_boom_speech_maps_the_pooled_grid() -> None:
    labels = ["Gunshot, gunfire", "Explosion", "Laughter"]
    probs = np.zeros((10, 3), np.float32)  # 0.32 s rows
    probs[3, 1] = 0.9  # explosion around 1.1 s
    probs[7, 2] = 0.5  # laughter around 2.4 s
    boom, voiced = le.boom_speech(probs, 0.32, labels, {"Laughter": 0.12}, 40)
    assert np.flatnonzero(boom).tolist() == [10, 11, 12] and boom.max() == pytest.approx(0.9)
    assert np.flatnonzero(voiced).tolist() == [23, 24, 25]
    empty_boom, empty_voice = le.boom_speech(np.zeros((0, 3)), 0.32, labels, {}, 5)
    assert empty_boom.shape == (5,) and not empty_voice.any()


def test_voice_picks_the_track_loud_during_words() -> None:
    talk = np.zeros(10, dtype=bool)
    talk[3:5] = True
    game = np.full(10, -20.0)
    mic = np.full(10, -60.0)
    mic[3:5] = -15.0
    assert le.pick_mic([game, mic], talk) == 1
    assert le.pick_mic([game, mic], np.zeros(10, dtype=bool)) is None
    m, g = np.array([0.2, 0.9]), np.array([1.0, 0.4])
    assert np.allclose(le.voice_signal(m, [g], "voice-mic"), [0.2, 0.9])
    assert np.allclose(le.voice_signal(m, [g], "voice-max"), [0.5, 0.9])


def test_top_metrics_counts_mixed_fps_separately() -> None:
    labels = [
        {"t": 10.0, "kind": "game"},
        {"t": 11.0, "kind": "reaction"},
        {"t": 50.0, "kind": "game"},
    ]
    ranked = [
        _cand(8, 14, 10.5),  # peak by the game label, window holds a reaction: mixed
        _cand(48, 53, 50.5),  # pure game FP
    ]
    m = le.top_metrics(ranked, 2, labels)
    assert (m["fp"], m["fp_mixed"]) == (2, 1)


def test_track_lufs_failed_chunk_leaves_no_wav_or_cache(tmp_path, monkeypatch) -> None:
    calls = []

    def decode(*_a, **_k):
        calls.append(1)
        if len(calls) == 2:
            raise le.FFmpegError("decode failed")
        return np.zeros((2, 16000), np.int16)

    monkeypatch.setattr(le, "decode_tracks", decode)
    with pytest.raises(le.FFmpegError):
        le.track_lufs(tmp_path / "a.mp4", 2, (0, 1), le.TRACK_CHUNK_S + 10, tmp_path)
    assert len(calls) == 2
    assert not list(tmp_path.glob("track*.wav"))
    assert not list(tmp_path.glob("track*-loudness.json"))


def test_distinct_tracks_falls_back_uncached_when_no_window_decodes(tmp_path, monkeypatch) -> None:
    def decode(*_a, **_k):
        raise le.FFmpegError("decode failed")

    monkeypatch.setattr(le, "decode_tracks", decode)
    assert le.distinct_tracks(tmp_path / "a.mp4", 3, 600.0, tmp_path) == (0, 1, 2)
    assert not (tmp_path / "tracks.json").exists()


def test_voice_inputs_reports_a_decode_failure_instead_of_raising(tmp_path, monkeypatch) -> None:
    src = write(tmp_path / "multi.mp4")

    def decode(*_a, **_k):
        raise le.FFmpegError("decode failed")

    monkeypatch.setattr(le, "decode_tracks", decode)
    reason = le.voice_inputs(_result(src, _signals(), streams=2), tmp_path)
    assert isinstance(reason, str)
    assert "track decode failed" in reason


def test_voice_is_a_no_op_on_a_single_distinct_track(tmp_path, monkeypatch) -> None:
    src = write(tmp_path / "solo.mp4")
    ps = _signals()
    assert le.voice_inputs(_result(src, ps, streams=1), tmp_path) == "single track"
    monkeypatch.setattr(le, "distinct_tracks", lambda *a: (0,))  # five OBS copies
    reason = le.voice_inputs(_result(src, ps, streams=5), tmp_path)
    assert reason == "single distinct track (5 tracks are copies)"
    variants = le.parse_variants(["voice"])
    out = le.run_source(
        src,
        LABELS,
        _result(src, ps),
        GAMING,
        variants,
        ["1", "suggested"],
        None,
        {"max_candidates": None},
        reason,
        None,
    )
    base = out["baseline"]["pre"]
    for name in ("voice-mic", "voice-max"):
        r = out[name]["pre"]
        assert r["na"] == reason
        assert r["top"] == base["top"] and r["at"]["1"]["hit_ids"] == base["at"]["1"]["hit_ids"]
        assert (r["at"]["1"]["gained"], r["at"]["1"]["lost"]) == (0, 0)


def test_boom_variant_removes_the_unreacted_explosion_from_the_top(tmp_path) -> None:
    src = write(tmp_path / "r6.mp4")
    ps = _signals()
    boom = _bump(GAME_T, 1.0, 1.5)
    voiced = np.zeros(N, dtype=bool)
    variants = le.parse_variants(["boom:1"])
    out = le.run_source(
        src,
        LABELS,
        _result(src, ps),
        GAMING,
        variants,
        ["1"],
        None,
        {"max_candidates": None},
        None,
        (boom, voiced),
    )
    base, var = out["baseline"]["pre"]["at"]["1"], out["boom:1"]["pre"]["at"]["1"]
    assert (base["fp"], base["hit"]) == (1, 0)  # the explosion outranks the laugh
    assert (var["fp"], var["hit"], var["gained"], var["lost"]) == (0, 1, 1, 0)
    assert out["baseline"]["pre"]["top"][0]["loud_share"] == pytest.approx(1.0, abs=0.01)


def test_reweight_lowers_loudness_only_where_events_ran() -> None:
    avail = scorer.available_for(GAMING)
    w0 = GAMING.weights["loudness"]

    def rec(peak_loud: float, peak: float, events: dict) -> dict:
        score = scorer.score(peak, GAMING, avail)
        return {
            "start": 0, "end": 10, "peak_t": 5, "peak": peak, "loud_peak": peak_loud,
            "score_pre_events": score, "score": score + events.get("boost", 0),
            "signals": {"events": events},
        }  # fmt: skip

    loud_ocr = rec(1.0, w0, {"boost": 0.0, "laughter": 0, "player_kills": 1})
    loud_vlm = rec(1.0, w0, {"boost": 0.0, "vlm": {"highlight": 0, "funny": 0}})
    plain = rec(1.0, w0, {"boost": 0.0, "laughter": 0.5, "player_kills": 0})
    assert [le.has_evidence(c) for c in (loud_ocr, loud_vlm, plain)] == [True, True, False]
    ranked, _, changed = le.reweight([loud_ocr, loud_vlm, plain], GAMING, 0.2, 90.0, avail)
    assert changed == 2
    assert ranked[0]["score"] == plain["score"]  # untouched, now first
    assert ranked[1]["score"] < plain["score"]
    weights = {**GAMING.weights, "loudness": 0.2}  # peak 0.45 - 0.25 x 1, over the new weights
    assert ranked[1]["score_pre_events"] == features.to_score(0.2, weights, avail)
    assert loud_ocr["score"] == plain["score"]  # the inputs are not modified


# --------------------------------------------------------------------------- draft labels
def test_draft_moments_find_reactions_game_peaks_and_both() -> None:
    ps = _signals()
    ps.sig["loudness"] = ps.sig["loudness"] + _bump(5.0, 0.7)
    ps.words.append({"word": " ow", "start": 5.0, "end": 5.2})  # loud, but someone talks
    drafts = le.draft_moments(ps)
    game = [d for d in drafts if d["kind"] == "game"]
    reactions = [d for d in drafts if d["kind"] == "reaction"]
    assert [round(d["t"], 1) for d in game] == [GAME_T]
    assert "no words" in game[0]["note"] and game[0]["strict"]
    assert any(abs(d["t"] - REACT_T) < 1 and "laughter" in d["note"] for d in reactions)
    assert all(abs(d["t"] - 5.0) > le.DRAFT_QUIET_S for d in game)
    boom = _bump(46.0, 0.8)
    with_boom = le.draft_moments(ps, (boom, np.zeros(N, dtype=bool)))
    assert any(d["kind"] == "both" and "boom peak" in d["note"] for d in with_boom)


def test_select_drafts_fills_each_group_with_the_strongest_weak_ones() -> None:
    def d(t, kind, strength, strict):
        return {"t": t, "kind": kind, "note": "n", "strength": strength, "strict": strict}

    found = {
        "a": [d(1, "reaction", 1.0, True), d(2, "game", 0.3, False), d(3, "game", 0.9, True)],
        "b": [d(4, "both", 0.2, False), d(5, "reaction", 0.4, False), d(6, "game", 0.35, False)],
    }
    chosen = le.select_drafts(found, minimum=2)
    assert [m["t"] for m in chosen["a"]] == [1, 3]
    assert [m["t"] for m in chosen["b"]] == [5, 6]
    assert all(m["draft"] and set(m) == {"t", "kind", "note", "draft"} for m in chosen["a"])


def test_write_drafts_never_overwrites_without_force(tmp_path) -> None:
    path = write(tmp_path / "labels" / "r6.json", '{"moments": [{"t": 1, "kind": "game"}]}')
    src = tmp_path / "r6.mp4"
    moment = [{"t": 2.0, "kind": "reaction", "note": "n", "draft": True}]
    assert not le.write_drafts(path, src, moment, force=False)
    assert json.loads(path.read_text())["moments"] == [{"t": 1, "kind": "game"}]
    assert le.write_drafts(path, src, moment, force=True)
    data = json.loads(path.read_text())
    assert data["draft"] and data["source"] == "r6.mp4" and data["moments"] == moment


@pytest.fixture
def offline(monkeypatch):
    """No Whisper, no PANNs: a stub transcriber and PANNs reported missing."""

    class Stub:
        model_name = "small"

        def __init__(self, **_k):
            pass

        def close(self):
            pass

    monkeypatch.setattr(transcribe, "Transcriber", Stub)
    monkeypatch.setattr(le, "panns_missing", lambda: "PANNs is not installed (test)")


def test_draft_mode_skips_existing_label_files_before_analysing(tmp_path, monkeypatch, offline):
    old = write(tmp_path / "labels" / "old.json", "{}")
    srcs = [write(tmp_path / "old.mp4"), write(tmp_path / "new.mp4")]
    seen = []

    def analyse(src, *_a):
        seen.append(src.name)
        return _result(src, _signals())

    monkeypatch.setattr(le, "analyse", analyse)
    assert le.main([*map(str, srcs), "--draft-labels"]) == 0
    assert seen == ["new.mp4"]
    assert old.read_text() == "{}"
    data = json.loads((tmp_path / "labels" / "new.json").read_text())
    assert data["draft"] and {m["kind"] for m in data["moments"]} == {"reaction", "game"}


def test_draft_mode_skips_a_second_source_with_the_same_stem(
    tmp_path, monkeypatch, caplog, offline
) -> None:
    srcs = [write(tmp_path / "a" / "rec.mp4"), write(tmp_path / "b" / "rec.mp4")]
    seen = []

    def analyse(src, *_a):
        seen.append(src)
        return _result(src, _signals())

    monkeypatch.setattr(le, "analyse", analyse)
    argv = [*map(str, srcs), "--draft-labels", "--labels-dir", str(tmp_path / "lab")]
    assert le.main(argv) == 0
    assert seen == [srcs[0]]
    assert json.loads((tmp_path / "lab" / "rec.json").read_text())["draft"]
    (warn,) = [r.getMessage() for r in caplog.records if "share the label file" in r.getMessage()]
    assert str(srcs[0]) in warn and str(srcs[1]) in warn


def test_eval_mode_skips_a_second_source_with_the_same_stem(
    tmp_path, monkeypatch, caplog, offline
) -> None:
    srcs = [write(tmp_path / "a" / "clip.mp4"), write(tmp_path / "b" / "clip.mp4")]
    write(tmp_path / "lab" / "clip.json", json.dumps({"source": "clip.mp4", "moments": LABELS}))
    seen = []

    def analyse(src, *_a):
        seen.append(src)
        return _result(src, _signals())

    monkeypatch.setattr(le, "analyse", analyse)
    argv = [*map(str, srcs), "--labels-dir", str(tmp_path / "lab"), "--variant", "baseline"]
    assert le.main(argv) == 0
    assert seen == [srcs[0]]
    assert any("share the label file" in r.getMessage() for r in caplog.records)


def test_draft_mode_fails_on_an_unwritable_label_folder_before_analysing(
    tmp_path, monkeypatch, caplog, offline
) -> None:
    src = write(tmp_path / "rec.mp4")

    def analyse(*_a):
        raise AssertionError("analysed before the folder check")

    monkeypatch.setattr(le, "analyse", analyse)
    monkeypatch.setattr(le, "label_dir_writable", lambda folder: False)
    assert le.main([str(src), "--draft-labels"]) == 1
    (err,) = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert str(tmp_path / "labels") in err and "--labels-dir" in err
    assert not (tmp_path / "labels").exists()


def test_label_dir_writable_probes_without_creating_the_folder(tmp_path) -> None:
    assert le.label_dir_writable(tmp_path)
    assert le.label_dir_writable(tmp_path / "new" / "labels")
    assert not (tmp_path / "new").exists()
    blocker = write(tmp_path / "file.txt")
    assert not le.label_dir_writable(blocker / "labels")


def test_label_dir_writable_is_false_when_the_probe_file_fails(tmp_path, monkeypatch) -> None:
    def refuse(*_a, **_k):
        raise PermissionError("read-only")

    monkeypatch.setattr(le.os, "open", refuse)
    assert not le.label_dir_writable(tmp_path)


def test_label_dir_writable_leaves_no_probe_file(tmp_path) -> None:
    assert le.label_dir_writable(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_label_dir_writable_warns_when_the_probe_file_stays(tmp_path, monkeypatch, caplog) -> None:
    def refuse(self, *_a, **_k):
        raise PermissionError("in use")

    monkeypatch.setattr(Path, "unlink", refuse)
    assert le.label_dir_writable(tmp_path)
    (warn,) = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert "Could not remove the probe file" in warn and str(tmp_path) in warn


def test_draft_folder_check_skips_label_files_kept_without_force(
    tmp_path, monkeypatch, offline
) -> None:
    src = write(tmp_path / "rec.mp4")
    write(tmp_path / "labels" / "rec.json", "{}")

    def analyse(*_a):
        raise AssertionError("analysed")

    monkeypatch.setattr(le, "analyse", analyse)
    monkeypatch.setattr(le, "label_dir_writable", lambda f: False)
    assert le.main([str(src), "--draft-labels"]) == 0  # nothing to write, nothing checked
    assert le.main([str(src), "--draft-labels", "--force"]) == 1


def test_same_label_file_in_another_case_is_a_clash(tmp_path, monkeypatch, caplog, offline):
    monkeypatch.setattr(le.os.path, "normcase", str.lower)  # as on Windows
    srcs = [write(tmp_path / "a" / "Rec.mp4"), write(tmp_path / "b" / "rec.mp4")]
    seen = []

    def analyse(src, *_a):
        seen.append(src)
        return _result(src, _signals())

    monkeypatch.setattr(le, "analyse", analyse)
    argv = [*map(str, srcs), "--draft-labels", "--labels-dir", str(tmp_path / "lab")]
    assert le.main(argv) == 0
    assert seen == [srcs[0]]
    assert (tmp_path / "lab" / "Rec.json").is_file()
    assert any("share the label file" in r.getMessage() for r in caplog.records)


def test_eval_main_writes_tables_and_json(tmp_path, monkeypatch, capsys, offline) -> None:
    src = write(tmp_path / "clip.mp4")
    write(tmp_path / "lab" / "clip.json", json.dumps({"source": "clip.mp4", "moments": LABELS}))
    write(tmp_path / "nolabels.mp4")
    monkeypatch.setattr(le, "analyse", lambda s, *_a: _result(s, _signals()))
    out = tmp_path / "le.json"
    argv = [str(src), str(tmp_path / "nolabels.mp4"), "--labels-dir", str(tmp_path / "lab")]
    assert le.main([*argv, "--out", str(out)]) == 0  # the default sweep
    text = capsys.readouterr().out
    assert "clip.mp4: 1 reaction, 1 game labels" in text and "Pooled" in text
    # columns keep a space between wide headers such as the suggested top's
    heads = [line.split() for line in text.splitlines() if line.startswith("variant")]
    assert len(heads) == 2  # the clip and the pooled table
    for head in heads:
        assert {"R@suggested", "FP@suggested"} <= set(head)
    assert "G@suggested" in heads[0]
    assert "boom variants skipped: PANNs is not installed" in text
    assert "reweight variants skipped" in text
    data = json.loads(out.read_text())
    assert data["variants"] == ["baseline", "voice-mic", "voice-max"]
    (clip,) = data["clips"]
    assert clip["tracks"] == "single track"
    assert clip["variants"]["voice-mic"]["pre"]["na"] == "single track"
    assert set(data["pooled"]) == {"baseline", "voice-mic", "voice-max"}
    assert data["pooled"]["voice-mic"]["pre"]["criteria"]["net"] == 0
    # the script's cache stays in its own folder
    assert (cache.cache_root() / "loudness_eval").is_dir()


def test_post_stage_rescores_with_events_and_reweight_only_touches_evidence(tmp_path) -> None:
    src = write(tmp_path / "r6.mp4")
    kill = {
        "source": str(src), "start": 44.0, "end": 44.5, "kind": "ocr", "type": "kill",
        "label": "x", "player_kill": True,
    }  # fmt: skip
    ev = {"sources": [str(src)], "events": [kill], "windows": []}
    project = tmp_path / "proj"
    write(project / "events.json", json.dumps(ev))
    found = le.events_for(src, [(project, ev)])
    assert found == (project, ev, str(src))
    assert le.events_for(tmp_path / "other.mp4", [(project, ev)]) is None
    variants = le.parse_variants(["reweight:0.2"])
    out = le.run_source(
        src, LABELS, _result(src, _signals()), GAMING, variants, ["1"], found,
        {"max_candidates": None, "language": None, "whisper_model": "small"}, None, None,
    )  # fmt: skip
    pre, post = out["baseline"]["pre"], out["baseline"]["post"]
    assert pre["at"]["1"]["fp"] == 1  # the explosion leads on the analysis score
    assert (post["at"]["1"]["fp"], post["at"]["1"]["hit"]) == (0, 1)  # the kill lifts the laugh
    rw = out["reweight:0.2"]
    assert set(rw) == {"post"} and rw["post"]["reweighted"] == 1  # the explosion has no evidence
    # the laugh's loudness share shrinks with the weight, enough to fall behind again here
    assert (rw["post"]["at"]["1"]["fp"], rw["post"]["at"]["1"]["lost"]) == (1, 1)
    assert rw["post"]["top"][0]["score"] == post["top"][1]["score"]  # the explosion, untouched
