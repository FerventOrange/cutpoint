"""Tests for duplicate audio tracks in clip exports (#365): detection, cache, filter graphs."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import media_info, needs_ffmpeg
from cutpoint import audio_tracks, cache, encode, ffmpeg_utils
from cutpoint.ffmpeg_utils import VideoEncoder, measure_loudness, mix_head

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


def test_a_window_with_another_waveform_at_the_same_level_keeps_the_track():
    """Game and mic at the same level: only the waveform differs (a plain bool, see
    :func:`~cutpoint.audio_tracks.same_window`)."""
    assert audio_tracks.same_window(GAME, MIC) is False
    same = np.stack([GAME, GAME])
    assert audio_tracks.kept_tracks([same, same, np.stack([GAME, MIC])]) == (0, 1)


def test_the_same_waveform_at_another_level_is_kept():
    assert audio_tracks.kept_tracks(_windows(GAME, GAME * 0.5)) == (0, 1)  # -6 dB


def test_a_silent_track_is_kept():
    assert audio_tracks.kept_tracks(_windows(GAME, SILENT)) == (0, 1)
    assert audio_tracks.same_window(SILENT, SILENT) is None


def test_one_differing_window_keeps_the_track():
    """A mic that copies the game track except where it talks is not a duplicate."""
    same = np.stack([GAME, GAME])
    talk = np.stack([GAME, GAME + MIC])
    assert audio_tracks.kept_tracks([same, same, talk]) == (0, 1)
    # windows where both are silent are no evidence: the loud one decides
    quiet = np.stack([SILENT, SILENT])
    assert audio_tracks.kept_tracks([quiet, same, quiet]) == (0,)
    assert audio_tracks.kept_tracks([quiet, quiet]) == (0, 1)


def test_windows_spread_across_the_source_and_fit_short_ones():
    starts = audio_tracks.window_starts(600.0)
    assert len(starts) == audio_tracks.WINDOWS
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
        raise ffmpeg_utils.FFmpegError("read error")

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
        raise ffmpeg_utils.FFmpegError("read error")

    monkeypatch.setattr(audio_tracks, "decode_tracks", fail)
    assert audio_tracks.export_tracks(src, 2, 30.0) == (0, 1)
    n = len(calls)
    assert audio_tracks.export_tracks(src, 2, 30.0, (1.0, 2.0)) == (0, 1)
    assert len(calls) == n


# -- filter graphs ----------------------------------------------------------------------


def test_mix_head():
    assert mix_head([0]) == "[0:a:0]"
    assert mix_head([0, 2]) == "[0:a:0][0:a:2]amix=inputs=2:normalize=0,"
    assert mix_head([0, 1], input_idx=1) == "[1:a:0][1:a:1]amix=inputs=2:normalize=0,"


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
