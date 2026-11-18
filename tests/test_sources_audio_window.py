"""``sources.audio_window``: samples of a window from the cached analysis WAV (including the
part of a recovered source), else decoded from the source (#1085 A591). Only the ffmpeg
decode is replaced."""

from __future__ import annotations

import json
import logging
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import media_info
from cutpoint import cache, sources
from cutpoint.errors import FFmpegError

SR = sources.SR


def put_wav(cdir: Path, seconds: float) -> None:
    """An analysis WAV whose sample ``i`` has the value ``i`` (so a slice tells where it is)."""
    cdir.mkdir(parents=True, exist_ok=True)
    pcm = (np.arange(int(seconds * SR)) % 30000).astype("<i2")
    with wave.open(str(cdir / "audio16k.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


@pytest.fixture
def env(home, tmp_path, monkeypatch) -> SimpleNamespace:
    path = tmp_path / "s.mp4"
    path.write_bytes(b"x")
    calls: list[tuple] = []

    def decode(s, info, t0, t1):
        calls.append((s, t0, t1))
        return np.full(4, 16384, "<i2").tobytes()

    monkeypatch.setattr(sources, "read_pcm_window", decode)
    return SimpleNamespace(path=path, decoded=calls)


def first_index(samples: np.ndarray) -> int:
    return round(float(samples[0]) * 32768.0)


def test_a_window_inside_the_cached_wav_is_sliced_from_it_without_decoding(env):
    src, decoded = env.path, env.decoded
    put_wav(cache.source_dir(src), 4.0)
    samples, start = sources.audio_window(src, media_info(src, 4.0), 1.0, 1.5)
    assert start == 1.0 and samples.dtype == np.float32
    assert len(samples) == SR // 2
    assert first_index(samples) == SR  # sample 16000: one second in
    assert samples[1] - samples[0] == pytest.approx(1 / 32768)
    assert decoded == []


def test_a_window_before_the_start_is_clamped_to_zero(env):
    src = env.path
    put_wav(cache.source_dir(src), 2.0)
    samples, start = sources.audio_window(src, media_info(src, 2.0), -3.0, 0.5)
    assert start == 0.0 and len(samples) == SR // 2 and first_index(samples) == 0


def test_a_window_past_the_end_of_the_wav_is_decoded_from_the_source(env):
    src, decoded = env.path, env.decoded
    put_wav(cache.source_dir(src), 2.0)
    samples, start = sources.audio_window(src, media_info(src, 10.0), 1.0, 5.0)
    assert start == 1.0
    assert decoded == [(src, 1.0, 5.0)]
    assert samples.tolist() == [0.5] * 4  # the decoded PCM as float


def test_without_a_cached_wav_the_window_is_decoded_and_a_failure_gives_no_samples(
    env, monkeypatch, caplog
):
    src, decoded = env.path, env.decoded
    info = media_info(src, 10.0)
    samples, start = sources.audio_window(src, info, 2.0, 3.0)
    assert (start, len(samples)) == (2.0, 4) and decoded == [(src, 2.0, 3.0)]

    def fail(*_a):
        raise FFmpegError("Audio window 2.0-3.0s of s.mp4 failed: boom")

    monkeypatch.setattr(sources, "read_pcm_window", fail)
    with caplog.at_level(logging.WARNING):
        samples, start = sources.audio_window(src, info, 2.0, 3.0)
    assert (start, samples.dtype, len(samples)) == (2.0, np.float32, 0)
    assert "Audio window 2.0-3.0s of s.mp4 failed" in caplog.text


def test_a_window_in_a_recovered_part_reads_that_parts_wav_at_its_own_offset(env):
    src, decoded = env.path, env.decoded
    put_wav(cache.source_dir(src), 5.0)
    (cache.source_dir(src) / sources.READ_ERRORS_FILE).write_text(
        json.dumps({"resumed_at": 100.0}), encoding="utf-8"
    )
    put_wav(cache.source_dir(src, 100.0), 5.0)
    info = media_info(src, 110.0)
    samples, start = sources.audio_window(src, info, 101.0, 101.5)
    assert start == 101.0 and len(samples) == SR // 2
    assert first_index(samples) == SR  # one second into the part that starts at 100 s
    assert decoded == []
    # the gap between the parts has no cached WAV that covers it: decoded from the source
    sources.audio_window(src, info, 50.0, 51.0)
    assert decoded == [(src, 50.0, 51.0)]
