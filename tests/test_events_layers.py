"""Events layers (#1085 A348, A350, A711, A713-A716): the shared per-source loop, cache presence
without decompressing, the reaction group names, the transcript cues memo, whole-second VLM
windows and the cached loading-screen frames. Synthetic data only."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cutpoint import cache
from cutpoint.errors import LowMemoryError
from cutpoint.events import (
    audio_events,
    common,
    explosions,
    pipeline,
    scoring,
    tails,
    visual,
    vlm_windows,
    voice,
)


def _src(tmp_path: Path, name: str = "a.mp4", duration: float = 600.0) -> common.Source:
    cdir = tmp_path / "cache" / name
    cdir.mkdir(parents=True)
    return common.Source(tmp_path / name, duration, cdir, [])


# --------------------------------------------------------------------------- A348


def test_source_guard_lives_in_common_and_pipeline_still_names_it():
    assert pipeline.SourceGuard is common.SourceGuard


def test_for_each_source_counts_skips_and_collects(tmp_path: Path):
    a, b, c = (_src(tmp_path, n) for n in ("a.mp4", "b.mp4", "c.mp4"))

    def fn(s: common.Source) -> Any:
        if s is b:
            raise ValueError("bad")
        return None if s is c else 1

    errors: list[Exception] = []
    done = common.for_each_source(common.SourceGuard("Test"), [a, b, c], fn, errors)
    assert done == 1 and [str(e) for e in errors] == ["bad"]


def test_for_each_source_raises_when_every_source_failed(tmp_path: Path):
    a = _src(tmp_path)

    def bad(_s: common.Source) -> Any:
        raise ValueError("all")

    with pytest.raises(ValueError, match="all"):
        common.for_each_source(common.SourceGuard("Test"), [a], bad)
    # one of two counted sources failed: not the whole layer
    assert common.for_each_source(common.SourceGuard("Test"), [a], bad, total=2) == 0


def test_for_each_source_lets_low_memory_through(tmp_path: Path):
    def oom(_s: common.Source) -> Any:
        raise LowMemoryError("x", "short")

    with pytest.raises(LowMemoryError):
        common.for_each_source(common.SourceGuard("Test"), [_src(tmp_path)], oom, [])


def test_cache_paths_keep_their_names(tmp_path: Path):
    s = _src(tmp_path)
    vp, xp = voice.VoiceParams(), visual.VisualParams()
    assert voice.cache_path(s, vp) == s.cdir / f"voice-{common.params_key(vp.key())}.npz"
    assert visual.cache_path(s, xp) == s.cdir / f"visual-{common.params_key(xp.key())}.npz"
    lp = visual.LightParams()
    assert visual.light_cache_path(s, lp).name == f"visual-light-{common.params_key(lp.key())}.npz"


# --------------------------------------------------------------------------- A713


def test_npz_present_reads_only_the_zip_directory(tmp_path: Path, monkeypatch):
    path = tmp_path / "x.npz"
    assert not common.npz_present(path)
    path.write_bytes(b"not a zip")
    assert not common.npz_present(path)
    cache.save_npz(path, a=np.zeros(3), b=np.ones(2))
    monkeypatch.setattr(cache, "load_npz", lambda _p: pytest.fail("decompressed"))
    assert common.npz_present(path, ("a", "b"))
    assert not common.npz_present(path, ("a", "c"))


def test_npz_present_survives_a_damaged_zip_directory(tmp_path: Path):
    path = tmp_path / "bad.npz"
    cache.save_npz(path, a=np.zeros(3))
    raw = bytearray(path.read_bytes())
    # a central-directory entry with the UTF-8 flag set and an invalid name byte
    cd = raw.rindex(b"PK")
    raw[cd + 8] |= 0x08  # general purpose flag bit 11 (UTF-8 names)
    raw[cd + 9] |= 0x08
    raw[cd + 46] = 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(UnicodeDecodeError):
        zipfile.ZipFile(path)
    assert not common.npz_present(path, ("a",))


def test_voice_and_visual_presence_checks_load_nothing(tmp_path: Path, monkeypatch):
    s = _src(tmp_path)
    vp, xp = voice.VoiceParams(), visual.VisualParams()
    grids = {k: np.zeros(4, np.float32) for k in ("wind_db", "level_db", "vad")}
    grids |= {f"{p}_{g}": np.zeros(4, np.float32) for p in ("raw", "hp") for g in voice.GROUPS}
    cache.save_npz(voice.cache_path(s, vp), **grids)
    cache.save_npz(
        visual.cache_path(s, xp),
        t=np.zeros(1),
        emb=np.zeros((1, 3, 768), np.float16),
        det=np.zeros((0, 7)),
        box_emb=np.zeros((0, 768), np.float16),
        box_idx=np.zeros(0, np.int64),
    )
    monkeypatch.setattr(cache, "load_npz", lambda _p: pytest.fail("decompressed"))
    assert voice.run_layer([s], vp) == {"cached": 1, "computed": 0}
    assert visual.run_layer([s], xp) == {"cached": 1, "computed": 0}


def test_a_visual_cache_without_an_array_is_not_present(tmp_path: Path):
    s = _src(tmp_path)
    cache.save_npz(visual.cache_path(s, visual.VisualParams()), t=np.zeros(1))
    assert not visual._cached(s, visual.VisualParams())


# --------------------------------------------------------------------------- A711


def test_vram_monitor_polls_every_five_seconds():
    assert common.VramMonitor().interval_s == 5.0


# --------------------------------------------------------------------------- A350


def test_reaction_groups_are_named_once_with_the_same_members():
    names = {g.name for g in audio_events.DEFAULT_GROUPS}
    assert set(audio_events.REACTION_GROUPS) <= names
    assert scoring.AFTER_EVENT_SOUNDS == ("Laughter", "Scream")
    assert scoring.LAUGH_SOUNDS == ("Laughter",)
    assert scoring.SCREAM_SOUNDS == ("Scream", "Cheering")
    assert scoring.REACTION_SOUNDS == ("Laughter", "Scream", "Cheering")
    assert {"Laughter", "Scream", "Cheering"} == tails.AUDIO_PAYOFFS
    assert vlm_windows.SEED_WEIGHTS == {"Laughter": 1.0, "Scream": 0.7, "Cheering": 0.6}


# --------------------------------------------------------------------------- A716


def _seg(a: float, b: float, text: str = "words") -> dict[str, Any]:
    return {"start": a, "end": b, "text": text, "words": []}


def test_has_reaction_reads_the_transcript_cues_once(monkeypatch):
    segments = [_seg(1, 3), _seg(12, 14), _seg(30, 31, "[laughter]")]
    expect = [scoring.has_reaction(t, [], list(segments)) for t in (5.0, 10.0, 26.0, 40.0)]
    calls = []
    real = scoring._laugh_markers
    monkeypatch.setattr(scoring, "_laugh_markers", lambda s: calls.append(1) or real(s))
    assert [scoring.has_reaction(t, [], segments) for t in (5.0, 10.0, 26.0, 40.0)] == expect
    assert len(calls) == 1
    segments.append(_seg(41, 42))  # a longer list is read again
    assert scoring.has_reaction(40.0, [], segments) is True
    assert len(calls) == 2


def test_has_reaction_cues_survive_a_two_source_alternation(monkeypatch):
    one = [_seg(12, 14)]
    two = [_seg(22, 24)]
    calls = []
    real = scoring._laugh_markers
    monkeypatch.setattr(scoring, "_laugh_markers", lambda s: calls.append(1) or real(s))
    for _ in range(3):
        scoring.has_reaction(10.0, [], one)
        scoring.has_reaction(20.0, [], two)
    assert len(calls) == 2


# --------------------------------------------------------------------------- A714


def test_vlm_windows_snap_to_whole_seconds(tmp_path: Path):
    s = _src(tmp_path, duration=120.5)

    def keys(start: float, end: float) -> list[str]:
        cand = {"source": str(s.path), "start": start, "end": end, "id": "h001", "score": 50}
        got = vlm_windows.select_windows([s], [cand], {}, vlm_windows.VlmParams(), quiet=True)
        return [w.key for w in got]

    assert keys(100.02, 105.0) == keys(100.08, 105.0) == ["99.0-115.0"]
    assert keys(100.6, 112.0) == ["99.0-120.5"]  # the source's end stays


# --------------------------------------------------------------------------- A715


def test_cached_luma_reads_each_stretch_once(tmp_path: Path):
    calls: list[tuple[float, float]] = []

    def read(t0: float, t1: float) -> list[float] | None:
        calls.append((t0, t1))
        return None if t0 < 0 else [0.1 / 3, 0.5]

    luma = explosions.CachedLuma(tmp_path / "a.mp4", tmp_path, read)
    assert luma(7.0, 10.2) == [0.1 / 3, 0.5]
    assert luma(-1.0, 0.2) is None
    luma.save()
    again = explosions.CachedLuma(tmp_path / "a.mp4", tmp_path, read)
    assert again(7.0, 10.2) == [0.1 / 3, 0.5]  # exact floats from the cache
    assert again(-1.0, 0.2) is None  # an unread stretch is tried again
    assert calls == [(7.0, 10.2), (-1.0, 0.2), (-1.0, 0.2)]


def test_confirm_gives_the_same_after_black_from_the_cache(tmp_path: Path):
    def events() -> list[dict[str, Any]]:
        return [{"kind": "audio", "label": "Explosion", "t": 10.0, "start": 10.0, "end": 11.0}]

    t = np.arange(0.0, 20.0, 0.1)
    m = np.where(t < 9.6, -30.0, -10.0)
    dark = [0.5, 0.01, 0.01, 0.5]
    first, second = events(), events()
    luma = explosions.CachedLuma(tmp_path / "a.mp4", tmp_path, lambda a, b: dark)
    explosions.confirm(first, None, loudness=(t, m), cuts=[], luma=luma)
    luma.save()
    cached = explosions.CachedLuma(tmp_path / "a.mp4", tmp_path, lambda a, b: pytest.fail("read"))
    explosions.confirm(second, None, loudness=(t, m), cuts=[], luma=cached)
    assert first == second and first[0]["after_black"] is True
