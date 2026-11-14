"""Tests for duplicate audio tracks in clip exports (#365) and the analysis audio (#685):
detection, cache, filter graphs."""

from __future__ import annotations

import logging
import shutil
import subprocess
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from conftest import media_info, needs_ffmpeg, write
from cutpoint import analyze, audio_tracks, cache, encode, errors, ffmpeg_utils
from cutpoint.ffmpeg_utils import MediaInfo, VideoEncoder, measure_loudness
from cutpoint.styles import get_style

RNG = np.random.default_rng(365)
GAME = RNG.normal(0, 0.1, 8000)
MIC = RNG.normal(0, 0.1, 8000)
SILENT = np.zeros(8000)


def _windows(*tracks: np.ndarray, n: int = 3) -> list[np.ndarray]:
    """``n`` windows holding the same ``tracks`` (rows)."""
    return [np.stack(tracks)] * n


# -- detection --------------------------------------------------------------------------


def test_identical_tracks_become_one():
    assert audio_tracks.kept_tracks(_windows(GAME, GAME, GAME, GAME, GAME)) == (0,)


def test_a_near_copy_counts_as_a_duplicate():
    """Coding noise 40 dB down and a 0.3 dB level change (a re-encoded copy)."""
    near = GAME * 10 ** (-0.3 / 20) + RNG.normal(0, 0.001, GAME.size)
    assert audio_tracks.kept_tracks(_windows(GAME, near)) == (0,)


def test_distinct_tracks_are_all_kept():
    assert audio_tracks.kept_tracks(_windows(GAME, MIC)) == (0, 1)
    # a copy of the game track after a distinct mic: only the copy goes
    assert audio_tracks.kept_tracks(_windows(GAME, MIC, GAME)) == (0, 1)


def test_the_same_waveform_at_another_level_is_kept():
    assert audio_tracks.kept_tracks(_windows(GAME, GAME * 0.5)) == (0, 1)  # -6 dB


def test_a_silent_track_is_kept():
    assert audio_tracks.kept_tracks(_windows(GAME, SILENT)) == (0, 1)
    assert audio_tracks.same_window(SILENT, SILENT) is None


def test_one_differing_window_keeps_the_track():
    """A mic that copies the game track except where it talks is not a duplicate. Game and
    mic at the same level: only the waveform differs."""
    assert audio_tracks.same_window(GAME, MIC) is False
    same = np.stack([GAME, GAME])
    talk = np.stack([GAME, GAME + MIC])
    assert audio_tracks.kept_tracks([same, same, talk]) == (0, 1)
    # windows where both are silent are no evidence: the loud one decides
    quiet = np.stack([SILENT, SILENT])
    assert audio_tracks.kept_tracks([quiet, same, quiet]) == (0,)
    assert audio_tracks.kept_tracks([quiet, quiet]) == (0, 1)


def test_windows_spread_across_the_source_and_fit_short_ones():
    starts = audio_tracks.window_starts(600.0)
    assert starts[0] > 0 and starts[-1] + audio_tracks.WINDOW_S <= 600.0
    assert audio_tracks.window_starts(2.0) == [0.0]


def test_single_and_missing_sources_mix_every_track(monkeypatch, tmp_path):
    def fail(*_a, **_k):
        raise AssertionError("no decode expected")

    monkeypatch.setattr(audio_tracks, "decode_tracks", fail)
    assert audio_tracks.export_tracks(tmp_path / "missing.mp4", 1) == (0,)
    assert audio_tracks.export_tracks(tmp_path / "missing.mp4", 3) == (0, 1, 2)
    assert audio_tracks.export_tracks(tmp_path / "missing.mp4", 0) == ()


def test_no_decodable_window_mixes_every_track_and_is_not_cached(monkeypatch, tmp_path):
    src = tmp_path / "bad.mp4"
    src.write_bytes(b"x")

    def fail(*_a, **_k):
        raise errors.FFmpegError("read error")

    monkeypatch.setattr(audio_tracks, "decode_tracks", fail)
    assert audio_tracks.export_tracks(src, 2, 30.0) == (0, 1)
    assert not (cache.source_dir(src) / audio_tracks.TRACKS_CACHE).exists()


def _talking_mic(monkeypatch, tmp_path, *talks: tuple[float, float]) -> list[tuple]:
    """A 120 s source: track 2 = track 1 plus a mic that talks only in ``talks`` (s), all
    between the windows the source is compared on. Returns the clip-span decodes."""
    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    sr = audio_tracks.SR
    game = np.random.default_rng(1).normal(0, 0.1, 120 * sr)
    mic = np.random.default_rng(2).normal(0, 0.1, 120 * sr)
    talk = game.copy()
    for t0, t1 in talks:
        talk[int(t0 * sr) : int(t1 * sr)] += mic[int(t0 * sr) : int(t1 * sr)]
    tracks = (np.stack([game, talk]) * 32767).astype(np.int16)
    spans: list[tuple] = []

    def decode(_src, _n, *, sr, start, duration, timeout=None):
        if duration != audio_tracks.WINDOW_S:
            spans.append((start, duration, timeout))
        return tracks[:, int(start * sr) : int((start + duration) * sr)]

    monkeypatch.setattr(audio_tracks, "decode_tracks", decode)
    for t in audio_tracks.window_starts(120.0):  # the fixture must miss every window
        assert all(t1 <= t or t0 >= t + audio_tracks.WINDOW_S for t0, t1 in talks)
    assert audio_tracks.export_tracks(src, 2, 120.0) == (0,)
    return spans


def test_a_dropped_track_is_kept_for_a_clip_where_it_differs(monkeypatch, tmp_path):
    """F1: the mic talks at 41-43 s only; clips elsewhere still leave the copy out."""
    spans = _talking_mic(monkeypatch, tmp_path, (41.0, 43.0))
    src = tmp_path / "s.mp4"
    assert audio_tracks.export_tracks(src, 2, 120.0, (40.0, 5.0)) == (0, 1)
    assert audio_tracks.export_tracks(src, 2, 120.0, (50.0, 10.0)) == (0,)
    assert audio_tracks.export_tracks(src, 2, 120.0, (40.0, 5.0)) == (0, 1)  # memoised
    assert [s[:2] for s in spans] == [(40.0, 5.0), (50.0, 10.0)]


def test_a_short_callout_inside_a_long_clip_keeps_the_track(monkeypatch, tmp_path):
    """F4: 0.8 s of mic in a 50 s clip; compared as one window the copy still looked alike."""
    _talking_mic(monkeypatch, tmp_path, (35.0, 35.8))
    src = tmp_path / "s.mp4"
    assert audio_tracks.export_tracks(src, 2, 120.0, (10.0, 50.0)) == (0, 1)
    assert audio_tracks.export_tracks(src, 2, 120.0, (36.0, 10.0)) == (0,)


def test_a_clip_is_compared_to_its_end(monkeypatch, tmp_path):
    """F5: a 90 s clip with the mic at 75-85 s; only its first minute was compared."""
    spans = _talking_mic(monkeypatch, tmp_path, (75.0, 85.0))
    assert audio_tracks.export_tracks(tmp_path / "s.mp4", 2, 120.0, (5.0, 90.0)) == (0, 1)
    assert spans == [(5.0, 90.0, 90.0)]  # the whole span; the timeout grows with it


def test_the_analysis_keeps_a_track_that_differs_between_the_windows(monkeypatch, tmp_path):
    """#685 F1: the windows hear two copies, the mic talks at 75-77 s: the analysis audio
    keeps it, checked in spans across the whole source, stopping once it is back."""
    monkeypatch.setattr(audio_tracks, "ANALYSIS_SPAN_S", 50.0)
    spans = _talking_mic(monkeypatch, tmp_path, (75.0, 77.0))
    src = tmp_path / "s.mp4"
    assert audio_tracks.analysis_tracks(src, 2, 120.0) == (0, 1)
    assert [s[:2] for s in spans] == [(0.0, 50.0), (49.0, 50.0)]  # overlapping, then done
    assert audio_tracks.export_tracks(src, 2, 120.0) == (0,)  # exports check their own span
    cached = cache.load_json(cache.source_dir(src) / audio_tracks.TRACKS_CACHE)
    assert cached["tracks"] == [0] and cached["analysis"] == [0, 1]
    audio_tracks._KEPT.clear()
    spans.clear()
    assert audio_tracks.analysis_tracks(src, 2, 120.0) == (0, 1)
    assert spans == []  # cached


def test_the_analysis_leaves_out_a_copy_everywhere(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_tracks, "ANALYSIS_SPAN_S", 50.0)
    spans = _talking_mic(monkeypatch, tmp_path)
    assert audio_tracks.analysis_tracks(tmp_path / "s.mp4", 2, 120.0) == (0,)
    assert [s[:2] for s in spans] == [(0.0, 50.0), (49.0, 50.0), (98.0, 22.0)]


@pytest.mark.parametrize(
    ("length", "silent"),
    [(120.0, (49.0, 99.0)), (98.7, (98.0, 98.7))],  # the second span; the 0.7 s last span
)
def test_a_silent_span_brings_no_track_back(monkeypatch, tmp_path, length, silent):
    """#685 F4: silence on every track proves no difference; the analysis keeps the copies
    out, while an export of a silent clip mixes every track it cannot tell apart, as before."""
    monkeypatch.setattr(audio_tracks, "ANALYSIS_SPAN_S", 50.0)
    sr = audio_tracks.SR
    game = (np.random.default_rng(1).normal(0, 0.1, int(length * sr)) * 32767).astype(np.int16)
    game[int(silent[0] * sr) : int(silent[1] * sr)] = 0
    tracks = np.stack([game, game, game])
    spans: list[tuple] = []

    def decode(_src, _n, *, sr, start, duration, timeout=None):
        spans.append((start, round(duration, 3)))
        return tracks[:, int(start * sr) : int((start + duration) * sr)]

    monkeypatch.setattr(audio_tracks, "window_starts", lambda _d: [10.0, 30.0])  # sound there
    monkeypatch.setattr(audio_tracks, "decode_tracks", decode)
    src = write(tmp_path / "s.mp4")
    assert audio_tracks.export_tracks(src, 3, length) == (0,)
    spans.clear()
    assert audio_tracks.analysis_tracks(src, 3, length) == (0,)
    span = (silent[0], round(silent[1] - silent[0], 3))
    assert span in spans  # checked, silent on every track
    assert audio_tracks.export_tracks(src, 3, length, span) == (0, 1, 2)  # unchanged


def test_a_failed_span_is_not_cached(monkeypatch, tmp_path):
    """#685 F5: a span that fails to decode leaves the result for this run only; the next
    run checks the source again."""
    monkeypatch.setattr(audio_tracks, "ANALYSIS_SPAN_S", 50.0)
    spans = _talking_mic(monkeypatch, tmp_path)
    decode = audio_tracks.decode_tracks

    def flaky(src, n, *, sr, start, duration, timeout=None):
        if start == 49.0:
            raise errors.FFmpegError("timed out")
        return decode(src, n, sr=sr, start=start, duration=duration, timeout=timeout)

    monkeypatch.setattr(audio_tracks, "decode_tracks", flaky)
    src = tmp_path / "s.mp4"
    assert audio_tracks.analysis_tracks(src, 2, 120.0) == (0,)
    assert audio_tracks.analysis_tracks(src, 2, 120.0) == (0,)  # memoised for this run
    assert [s[:2] for s in spans] == [(0.0, 50.0), (98.0, 22.0)]
    assert "analysis" not in cache.load_json(cache.source_dir(src) / audio_tracks.TRACKS_CACHE)
    audio_tracks._KEPT.clear()
    monkeypatch.setattr(audio_tracks, "decode_tracks", decode)
    spans.clear()
    assert audio_tracks.analysis_tracks(src, 2, 120.0) == (0,)
    assert [s[:2] for s in spans] == [(0.0, 50.0), (49.0, 50.0), (98.0, 22.0)]  # again
    cached = cache.load_json(cache.source_dir(src) / audio_tracks.TRACKS_CACHE)
    assert cached["analysis"] == [0]


def test_the_analysis_says_when_nothing_was_compared(monkeypatch, tmp_path):
    def fail(*_a, **_k):
        raise errors.FFmpegError("read error")

    monkeypatch.setattr(audio_tracks, "decode_tracks", fail)
    src = write(tmp_path / "bad.mp4")
    assert audio_tracks.analysis_tracks(src, 2, 30.0) is None
    assert audio_tracks.analysis_tracks(tmp_path / "missing.mp4", 2, 30.0) is None
    assert audio_tracks.analysis_tracks(src, 1, 30.0) == (0,)
    assert audio_tracks.export_tracks(src, 2, 30.0) == (0, 1)  # exports mix every track


def test_chunks_overlap_and_cover_the_span():
    sr = audio_tracks.SR
    pcm = np.zeros((2, int(2.7 * sr)), dtype=np.int16)
    got = [c.shape[1] / sr for c in audio_tracks.chunks(pcm)]
    assert got == [1.0, 1.0, 1.0, 1.0, 0.7]  # starts 0, 0.5, 1, 1.5, 2; the 0.2 s tail left out
    assert [c.shape[1] for c in audio_tracks.chunks(pcm[:, : sr // 4])] == []


def test_a_failed_comparison_is_memoised(monkeypatch, tmp_path):
    """F2: the fallback to every track is not recomputed (export and peak plan agree)."""
    src = tmp_path / "s.mp4"
    src.write_bytes(b"x")
    calls: list[float] = []

    def fail(_src, _n, *, sr, start, duration, timeout=None):
        calls.append(start)
        raise errors.FFmpegError("read error")

    monkeypatch.setattr(audio_tracks, "decode_tracks", fail)
    assert audio_tracks.export_tracks(src, 2, 30.0) == (0, 1)
    n = len(calls)
    assert audio_tracks.export_tracks(src, 2, 30.0, (1.0, 2.0)) == (0, 1)
    assert len(calls) == n


# -- filter graphs ----------------------------------------------------------------------


def _audio_graph(monkeypatch, streams: int, kept: tuple[int, ...] | None = None) -> str:
    calls: list[list[str]] = []
    monkeypatch.setattr(encode, "run", lambda args, **_: calls.append(args) or "")
    if kept is not None:
        monkeypatch.setattr(encode, "export_tracks", lambda *_a: kept)
    info = media_info(Path("v.mp4"), 12.0, audio_streams=streams)
    fmt = encode.OutputFormat(64, 48, 25)
    encode._encode(VideoEncoder("cpu"), info.path, info, 1.0, 2.0, Path("o.mp4"), fmt,
                   False, None, 0.0)  # fmt: skip
    graph = calls[0][calls[0].index("-filter_complex") + 1]
    return graph.split(";", 1)[1]


def test_clip_export_audio_graph_per_track_selection(monkeypatch):
    a_fmt = encode.A_FMT
    # single track: the graph as before #365, and no track comparison
    assert _audio_graph(monkeypatch, 1) == f"[0:a:0]{a_fmt}[a]"
    # nothing to compare (no such file): every track, as before #365
    assert _audio_graph(monkeypatch, 2) == f"[0:a:0][0:a:1]amix=inputs=2:normalize=0,{a_fmt}[a]"
    # duplicates left out: the kept track alone, as a single-track source
    assert _audio_graph(monkeypatch, 5, (0,)) == f"[0:a:0]{a_fmt}[a]"
    # distinct tracks are still summed at full level
    assert _audio_graph(monkeypatch, 3, (0, 2)) == (
        f"[0:a:0][0:a:2]amix=inputs=2:normalize=0,{a_fmt}[a]"
    )


def test_clip_momentary_measures_the_tracks_the_export_mixes(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(ffmpeg_utils, "run", lambda args, **_: calls.append(args) or "")
    seen: list[tuple] = []

    def tracks(_src, _n, duration=None, span=None):
        seen.append(span)
        return (0,)

    monkeypatch.setattr(audio_tracks, "export_tracks", tracks)
    ffmpeg_utils.clip_momentary(Path("v.mp4"), 4, 1.0, 2.0)
    assert seen == [(1.0, 2.0)]  # the clip's span, as the export checks it
    assert calls[0][calls[0].index("-filter_complex") + 1] == "[0:a:0]ebur128=peak=none"


# -- real ffmpeg ------------------------------------------------------------------------


def _extract_args(monkeypatch, streams: int, kept: tuple[int, ...] | None = None) -> list[str]:
    """The ffmpeg arguments of the analysis audio extract between ``-vn`` and ``-ac``."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        ffmpeg_utils,
        "run_progress",
        lambda args, *_a, **_k: calls.append(args) or write(Path(args[-1])),
    )
    if kept is not None:
        monkeypatch.setattr(audio_tracks, "analysis_tracks", lambda *_a, **_k: kept)
    info = media_info(Path("v.mp4"), 12.0, audio_streams=streams)
    ffmpeg_utils.extract_audio(info.path, Path("cache/audio16k.wav"), info)
    args = calls[0]
    return args[args.index("-vn") + 1 : args.index("-ac")]


def test_analysis_audio_mixes_the_tracks_exports_mix(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    clock = ffmpeg_utils.AUDIO_CLOCK  # on the container's clock (#1056)
    # single track: the command line as before #685
    assert _extract_args(monkeypatch, 1) == ["-map", "0:a:0", "-af", clock]
    # every track differs: summed as before #685
    assert _extract_args(monkeypatch, 3, (0, 1, 2)) == [
        "-filter_complex", f"{ffmpeg_utils.audio_mix_filter(3)},{clock}[a]", "-map", "[a]",
    ]  # fmt: skip
    # copies left out: the kept track alone, or the kept ones summed at full level
    assert _extract_args(monkeypatch, 5, (0,)) == ["-map", "0:a:0", "-af", clock]
    assert _extract_args(monkeypatch, 3, (0, 2)) == [
        "-filter_complex",
        f"[0:a:0][0:a:2]amix=inputs=2:normalize=0:duration=longest,{clock}[a]",
        "-map", "[a]",
    ]  # fmt: skip


def test_audio_window_fallback_mixes_the_analysis_tracks(monkeypatch, tmp_path):
    """#685 F7: the cached analysis tracks when there are any, else the window's own export
    check; never a decode of the whole source."""
    seen: list[dict] = []
    monkeypatch.setattr(
        ffmpeg_utils, "decode_pcm", lambda *_a, **k: seen.append(k) or np.zeros(0, np.int16)
    )
    spans = _talking_mic(monkeypatch, tmp_path, (75.0, 77.0))
    src = tmp_path / "s.mp4"
    info = media_info(src, 120.0, audio_streams=2)
    ffmpeg_utils.read_pcm_window(src, info, 10.0, 20.0)  # no analysis yet: this window only
    ffmpeg_utils.read_pcm_window(src, info, 74.0, 78.0)
    assert [s[:2] for s in spans] == [(10.0, 10.0), (74.0, 4.0)]
    assert [k["tracks"] for k in seen] == [(0,), (0, 1)]
    assert audio_tracks.analysis_tracks(src, 2, 120.0) == (0, 1)
    audio_tracks._KEPT.clear()  # the next run: the analysis tracks come from the disk cache
    spans.clear()
    ffmpeg_utils.read_pcm_window(src, info, 10.0, 20.0)
    assert spans == [] and seen[-1]["tracks"] == (0, 1)
    assert ffmpeg_utils.decode_pcm_args("v.mp4", sr=16000, tracks=(1,))[-9:-7] == [
        "-map", "0:a:1",
    ]  # fmt: skip


def _cdir(tmp_path: Path) -> Path:
    """A cache folder with every analysis cache, audio and visual."""
    cdir = tmp_path / "cache"
    for name in (
        "audio16k.wav",
        "loudness.json",
        "transcript-small.json",
        "transcript-small-en.json",
        "events-audio-ab12.json",
        "voice-ab12.npz",
        analyze.SCENES_FILE,
        "events-ocr-ab12.json",
        "visual-ab12.npz",
        "audio-tracks.json",
    ):
        write(cdir / name)  # fmt: skip
    return cdir


AUDIO_DERIVED = {"audio16k.wav", "loudness.json", "transcript-small.json",
                 "transcript-small-en.json", "events-audio-ab12.json", "voice-ab12.npz"}  # fmt: skip


def test_a_wav_with_every_track_keeps_its_caches(tmp_path):
    cdir = _cdir(tmp_path)
    before = {f.name: f.stat().st_mtime_ns for f in cdir.iterdir()}
    analyze._audio_mix(Path("a.mp4"), cdir, (0,), 1)  # one track
    analyze._audio_mix(Path("a.mp4"), cdir, (0, 1, 2), 3)  # all distinct
    assert {f.name: f.stat().st_mtime_ns for f in cdir.iterdir()} == before


def test_a_wav_made_with_copies_is_analysed_again_once(tmp_path):
    cdir = _cdir(tmp_path)
    analyze._audio_mix(Path("a.mp4"), cdir, (0,), 5)
    names = {f.name for f in cdir.iterdir()}
    assert not names & AUDIO_DERIVED  # the hot WAV and everything made from it
    assert {
        analyze.SCENES_FILE,
        "events-ocr-ab12.json",
        "visual-ab12.npz",
        "audio-tracks.json",
    } < names
    assert cache.load_json(cdir / analyze.MIX_FILE) == {"tracks": [0]}
    for name in AUDIO_DERIVED:  # the new WAV and its caches
        write(cdir / name)
    analyze._audio_mix(Path("a.mp4"), cdir, (0,), 5)
    assert {f.name for f in cdir.iterdir()} > AUDIO_DERIVED
    # other kept tracks than the WAV was made with (a new comparison): made again
    analyze._audio_mix(Path("a.mp4"), cdir, (0, 3), 5)
    assert not {f.name for f in cdir.iterdir()} & AUDIO_DERIVED
    assert cache.load_json(cdir / analyze.MIX_FILE) == {"tracks": [0, 3]}
    for name in AUDIO_DERIVED:
        write(cdir / name)
    analyze._audio_mix(Path("a.mp4"), cdir, (0, 1, 2, 3, 4), 5)  # every track again
    assert not {f.name for f in cdir.iterdir()} & (AUDIO_DERIVED | {analyze.MIX_FILE})


def test_an_uncompared_source_keeps_its_cached_audio(monkeypatch, tmp_path):
    """#685 F2: when the tracks cannot be compared, the WAV's recorded tracks are used and
    nothing is removed (no re-extract too hot, then again on the next good run)."""
    cdir = _cdir(tmp_path)
    cache.save_json(cdir / analyze.MIX_FILE, {"tracks": [0]})
    before = {f.name: f.stat().st_mtime_ns for f in cdir.iterdir()}
    monkeypatch.setattr(analyze, "source_tracks", lambda *_a: None)
    info = media_info(Path("a.mp4"), 10.0, audio_streams=5)
    assert analyze._analysis_tracks(Path("a.mp4"), Path("a.mp4"), info, cdir) == (0,)
    assert {f.name: f.stat().st_mtime_ns for f in cdir.iterdir()} == before
    for bad in ({"tracks": [0, 7]}, {"tracks": []}, ["x"]):  # no valid record: every track
        cache.save_json(cdir / analyze.MIX_FILE, bad)
        assert analyze._analysis_tracks(Path("a.mp4"), Path("a.mp4"), info, cdir) == (
            0, 1, 2, 3, 4,
        )  # fmt: skip
    assert {f.name for f in cdir.iterdir()} > AUDIO_DERIVED


def test_a_cache_that_cannot_be_removed_is_retried(monkeypatch, tmp_path, caplog):
    """#685 F3: a WAV open in another program (Windows) stays; the record is not written, so
    the next run removes it."""
    cdir = _cdir(tmp_path)
    unlink = Path.unlink

    def locked(self, missing_ok=False):
        if self.name == "audio16k.wav":
            raise PermissionError("in use")
        unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked)
    caplog.set_level(logging.INFO, logger="cutpoint.analyze")
    analyze._audio_mix(Path("a.mp4"), cdir, (0,), 5)
    names = {f.name for f in cdir.iterdir()}
    assert names & AUDIO_DERIVED == {"audio16k.wav"}
    assert analyze.MIX_FILE not in names
    assert "could not remove audio16k.wav" in caplog.text
    assert "extracting it again" not in caplog.text  # the old WAV is kept
    caplog.clear()
    monkeypatch.setattr(Path, "unlink", unlink)
    analyze._audio_mix(Path("a.mp4"), cdir, (0,), 5)
    assert "extracting it again" in caplog.text
    assert not {f.name for f in cdir.iterdir()} & AUDIO_DERIVED
    assert cache.load_json(cdir / analyze.MIX_FILE) == {"tracks": [0]}


class _Stop(Exception):
    pass


def test_a_derived_cache_that_cannot_be_removed_keeps_the_wav(monkeypatch, tmp_path, caplog):
    """#685 F6: a voice cache open in another program stays, and so does the WAV it was made
    from (removed last), so the new WAV is never paired with a stale cache."""
    cdir = _cdir(tmp_path)
    unlink = Path.unlink

    def locked(self, missing_ok=False):
        if self.name == "voice-ab12.npz":
            raise PermissionError("in use")
        unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked)
    caplog.set_level(logging.INFO, logger="cutpoint.analyze")
    analyze._audio_mix(Path("a.mp4"), cdir, (0,), 5)
    names = {f.name for f in cdir.iterdir()}
    assert names & AUDIO_DERIVED == {"audio16k.wav", "voice-ab12.npz"}
    assert analyze.MIX_FILE not in names
    assert "could not remove voice-ab12.npz" in caplog.text
    assert "extracting it again" not in caplog.text  # the old WAV is kept
    caplog.clear()
    monkeypatch.setattr(Path, "unlink", unlink)
    analyze._audio_mix(Path("a.mp4"), cdir, (0,), 5)
    assert "extracting it again" in caplog.text
    assert not {f.name for f in cdir.iterdir()} & AUDIO_DERIVED
    assert cache.load_json(cdir / analyze.MIX_FILE) == {"tracks": [0]}


@pytest.mark.parametrize(("streams", "kept"), [(1, (0,)), (3, (0, 1, 2)), (3, (0,))])
def test_analyze_source_extracts_the_kept_tracks(tmp_path, monkeypatch, streams, kept):
    src = write(tmp_path / "a.mp4")
    info = MediaInfo(src, 100.0, 100.0, 640, 360, Fraction(30), "yuv420p", "h264", streams,
                     "aac", 16000, 1)  # fmt: skip
    cdir = tmp_path / "cache"
    for name in (
        "audio16k.wav",
        analyze.CLOCK_FILE,
        "loudness.json",
        "transcript-m.json",
    ):  # earlier
        write(cdir / name, "[[0.0], [0.0]]" if name == "loudness.json" else "[]")
    monkeypatch.setattr(analyze, "probe", lambda path: info)
    monkeypatch.setattr(analyze.cache, "source_dir", lambda s, off=0.0: cdir)
    monkeypatch.setattr(analyze, "source_tracks", lambda s, i: kept)
    extracts: list[tuple] = []
    monkeypatch.setattr(
        analyze, "extract_audio", lambda *a, **k: extracts.append(k["tracks"]) or write(a[1])
    )
    monkeypatch.setattr(analyze, "check_read", lambda *a: None)
    monkeypatch.setattr(analyze, "_loudness", lambda wav, cdir: (np.zeros(1), np.zeros(1)))

    class Tr:
        model_name = "m"
        language = None

        def transcribe(self, wav):
            raise _Stop  # transcribed again: the rest of the analysis is not needed here

    stale = len(kept) < streams
    try:
        analyze.analyze_source(src, get_style("funny"), Tr(), False, None, {})
    except _Stop:
        assert stale
    else:
        assert not stale  # the cached transcript is used
    assert extracts == ([kept] if stale else [])
    assert (cdir / analyze.MIX_FILE).exists() == stale
    assert (cdir / "audio16k.wav").exists()  # the same name either way


def _source(path: Path, *tracks: str) -> Path:
    """4 s 64x48 video with one AAC track per lavfi audio source in ``tracks``."""
    ins = ["-f", "lavfi", "-i", "testsrc=size=64x48:rate=25:duration=4"]
    maps = ["-map", "0:v"]
    for i, t in enumerate(tracks, 1):
        ins += ["-f", "lavfi", "-i", t]
        maps += ["-map", f"{i}:a"]
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-y", *ins, *maps, "-t", "4", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True,
    )  # fmt: skip
    return path


GAME_SRC = "anoisesrc=color=pink:amplitude=0.4:seed=1:duration=4:sample_rate=48000"
MIC_SRC = "anoisesrc=color=pink:amplitude=0.4:seed=2:duration=4:sample_rate=48000"


@pytest.mark.slow
@needs_ffmpeg
def test_real_tracks_are_compared_and_cached(monkeypatch, tmp_path):
    dup = _source(tmp_path / "dup.mp4", GAME_SRC, GAME_SRC, GAME_SRC)
    mixed = _source(tmp_path / "mixed.mp4", GAME_SRC, MIC_SRC, GAME_SRC)
    assert audio_tracks.export_tracks(dup, 3) == (0,)
    assert audio_tracks.export_tracks(mixed, 3) == (0, 1)
    cached = cache.load_json(cache.source_dir(dup) / audio_tracks.TRACKS_CACHE)
    assert cached == {"version": audio_tracks.TRACKS_CACHE_VERSION, "streams": 3, "tracks": [0]}

    def fail(*_a, **_k):
        raise AssertionError("cached: no decode expected")

    audio_tracks._KEPT.clear()
    monkeypatch.setattr(audio_tracks, "decode_tracks", fail)
    assert audio_tracks.export_tracks(dup, 3) == (0,)


@pytest.mark.slow
@needs_ffmpeg
def test_real_mic_between_the_windows_is_kept_for_its_clip(tmp_path):
    """F1: a 30 s source compared on 1-4, 6-9, ... s; the mic on track 2 talks at 4.2-5.8 s.

    Lossless PCM: two AAC encoders whose input differed for a while drift apart for the rest
    of the file, so with AAC the tracks would not count as copies anywhere."""
    talk = "volume=enable='between(t,4.2,5.8)':volume=1,volume=enable='not(between(t,4.2,5.8))':volume=0"
    src = tmp_path / "talk.mov"
    subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc=size=64x48:rate=25:duration=30", "-f", "lavfi", "-i",
         GAME_SRC.replace("duration=4", "duration=30"), "-f", "lavfi", "-i",
         MIC_SRC.replace("duration=4", "duration=30"), "-filter_complex",
         f"[1:a]asplit=2[g1][g2];[2:a]{talk}[m];[g2][m]amix=inputs=2:normalize=0[gm]",
         "-map", "0:v", "-map", "[g1]", "-map", "[gm]", "-t", "30", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", str(src)],
        check=True,
    )  # fmt: skip
    assert audio_tracks.export_tracks(src, 2) == (0,)
    assert audio_tracks.export_tracks(src, 2, span=(3.5, 3.0)) == (0, 1)
    assert audio_tracks.export_tracks(src, 2, span=(10.0, 3.0)) == (0,)


@pytest.mark.slow
@needs_ffmpeg
def test_duplicate_tracks_export_at_single_track_loudness(tmp_path):
    """Acceptance of #365: 3 identical tracks summed came out 9.5 dB louder, clipped."""
    single = _source(tmp_path / "single.mp4", GAME_SRC)
    dup = _source(tmp_path / "dup.mp4", GAME_SRC, GAME_SRC, GAME_SRC)
    game_mic = _source(tmp_path / "gm.mp4", GAME_SRC, MIC_SRC)
    fmt = encode.OutputFormat(64, 48, 25)
    levels = {}
    for src in (single, dup, game_mic):
        out = tmp_path / f"out-{src.name}"
        encode.export_clip(src, ffmpeg_utils.probe(src), 0.5, 3.5, out, fmt, False)
        levels[src.name] = measure_loudness(out)
    (i1, tp1), (i3, tp3) = levels["single.mp4"], levels["dup.mp4"]
    assert abs(i3 - i1) <= 1.0
    assert tp3 <= tp1 + 0.5 and tp3 < 0.0
    # game + mic: two distinct tracks of equal level, still summed (about +3 LU)
    assert levels["gm.mp4"][0] - i1 == pytest.approx(3.0, abs=1.0)


def _wav_rms_db(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
    return 10 * np.log10(np.mean((pcm / 32768.0) ** 2) + 1e-24)


@pytest.mark.slow
@needs_ffmpeg
def test_duplicate_tracks_analyse_at_single_track_level(tmp_path):
    """Acceptance of #685: 3 identical tracks summed into the analysis WAV came out 9.5 dB
    hotter; a game and a mic track are still both summed."""
    single = _source(tmp_path / "single.mp4", GAME_SRC)
    dup = _source(tmp_path / "dup.mp4", GAME_SRC, GAME_SRC, GAME_SRC)
    game_mic = _source(tmp_path / "gm.mp4", GAME_SRC, MIC_SRC)
    levels = {}
    for src in (single, dup, game_mic):
        wav = tmp_path / f"{src.stem}.wav"
        ffmpeg_utils.extract_audio(src, wav, ffmpeg_utils.probe(src))
        levels[src.stem] = _wav_rms_db(wav)
    assert levels["dup"] == pytest.approx(levels["single"], abs=0.5)
    # two distinct tracks of equal level: about +3 dB
    assert levels["gm"] - levels["single"] == pytest.approx(3.0, abs=1.0)
