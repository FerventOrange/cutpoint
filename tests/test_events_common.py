"""``iter_audio_blocks``: the overlapping block reader behind the audio-event and voice layers."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from cutpoint.events.common import WAV_SR, AudioPart, Source, iter_audio_blocks


def write_ramp(path: Path, seconds: float) -> int:
    """A 16 kHz mono WAV whose sample ``i`` is ``i % 1000``; returns the frame count."""
    n = round(seconds * WAV_SR)
    data = (np.arange(n) % 1000).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(WAV_SR)
        w.writeframes(data.tobytes())
    return n


def source(tmp_path: Path, *parts: tuple[float, float]) -> Source:
    """A Source with one WAV part per ``(offset, seconds)``."""
    audio = []
    for k, (offset, seconds) in enumerate(parts):
        wav = tmp_path / f"p{k}.wav"
        write_ramp(wav, seconds)
        audio.append(AudioPart(offset, wav, seconds))
    return Source(tmp_path / "a.mp4", sum(s for _, s in parts), tmp_path, audio)


@pytest.mark.parametrize(
    ("seconds", "block", "overlap", "expected"),
    [
        # (start time, seconds in the block)
        (5.0, 2.0, 0.5, [(0.0, 2.0), (1.5, 2.0), (3.0, 2.0)]),  # last block ends exactly
        (4.0, 2.0, 0.5, [(0.0, 2.0), (1.5, 2.0), (3.0, 1.0)]),  # last block is partial
        (4.0, 2.0, 0.0, [(0.0, 2.0), (2.0, 2.0)]),  # no overlap: exact tiling, no empty tail
        (1.0, 2.0, 0.5, [(0.0, 1.0)]),  # shorter than one block
        (2.0, 2.0, 0.5, [(0.0, 2.0)]),  # exactly one block
        (0.0, 2.0, 0.5, []),  # empty audio
    ],
)
def test_blocks_start_every_step_and_the_last_one_is_cut_short(
    tmp_path: Path, seconds, block, overlap, expected
) -> None:
    src = source(tmp_path, (0.0, seconds))
    got = [(t, len(x) / WAV_SR) for t, x, _, _ in iter_audio_blocks(src, block, overlap)]
    assert got == pytest.approx(expected)


def test_blocks_are_float32_samples_in_minus_one_to_one_taken_from_their_offset(
    tmp_path: Path,
) -> None:
    src = source(tmp_path, (0.0, 4.0))
    blocks = list(iter_audio_blocks(src, 2.0, 0.5))
    for t, x, _, _ in blocks:
        assert x.dtype == np.float32
        first = int(t * WAV_SR)
        want = (np.arange(first, first + len(x)) % 1000) / 32768.0
        np.testing.assert_allclose(x, want)
    # consecutive blocks share the overlap: the tail of one is the head of the next
    np.testing.assert_array_equal(
        blocks[0][1][int(1.5 * WAV_SR) :], blocks[1][1][: int(0.5 * WAV_SR)]
    )


def test_each_part_is_blocked_on_its_own_and_timed_from_its_offset(tmp_path: Path) -> None:
    src = source(tmp_path, (0.0, 3.0), (10.0, 1.0))
    got = [(t, len(x) / WAV_SR) for t, x, _, _ in iter_audio_blocks(src, 2.0)]
    assert got == pytest.approx([(0.0, 2.0), (2.0, 1.0), (10.0, 1.0)])


def test_first_and_last_mark_the_edges_of_each_part(tmp_path: Path) -> None:
    """#1085 A167: a part's seam edges are no block's overlap, so callers must keep them."""
    src = source(tmp_path, (0.0, 5.0), (10.0, 3.0))
    got = [(t, first, last) for t, _, first, last in iter_audio_blocks(src, 2.0, 0.5)]
    assert got == [
        (0.0, True, False),
        (1.5, False, False),
        (3.0, False, True),
        (10.0, True, False),
        (11.5, False, True),
    ]


def test_a_source_without_audio_parts_yields_nothing(tmp_path: Path) -> None:
    assert list(iter_audio_blocks(Source(tmp_path / "a.mp4", 5.0, tmp_path, []), 2.0)) == []


@pytest.mark.parametrize(
    ("block", "overlap"),
    [
        (2.0, 2.0),
        (2.0, 3.0),
        (0.0, 0.0),
        (-1.0, 0.0),
        (2.0, -0.5),
        (1.0, 1.0 - 1e-6),
        (1e-5, 0.0),
    ],
)
def test_a_block_that_cannot_advance_is_rejected_before_decoding(
    tmp_path: Path, block: float, overlap: float
) -> None:
    src = Source(tmp_path / "a.mp4", 3.0, tmp_path, [AudioPart(0.0, tmp_path / "missing.wav", 3.0)])
    with pytest.raises(ValueError, match="overlap_s"):
        list(iter_audio_blocks(src, block, overlap))
