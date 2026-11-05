"""cutpoint.envelope: the peak level per bucket, folded from a stream of PCM."""

from __future__ import annotations

import io

import numpy as np

from cutpoint import envelope


def pcm(values: list[float]) -> bytes:
    """16-bit PCM of ``values`` (-1 to 1), each held for one bucket of ``PER_BUCKET`` samples."""
    return b"".join(
        np.full(envelope.PER_BUCKET, round(v * 32767), dtype="<i2").tobytes() for v in values
    )


def test_one_peak_per_bucket_whatever_the_read_size() -> None:
    data = pcm([0.0, 0.5, -1.0, 0.25])
    peaks = envelope.read_peaks(io.BytesIO(data))
    assert np.allclose(peaks, [0.0, 0.5, 1.0, 0.25], atol=1e-3)
    assert np.allclose(envelope.read_peaks(io.BytesIO(data + b"\x00")), peaks)  # a stray byte


def test_a_short_last_bucket_and_empty_input() -> None:
    peaks = envelope.read_peaks(io.BytesIO(pcm([0.5]) + pcm([0.8])[:20]))
    assert len(peaks) == 2
    assert np.allclose(peaks, [0.5, 0.8], atol=1e-3)
    assert len(envelope.read_peaks(io.BytesIO(b""))) == 0


def test_shrink_merges_neighbours_by_their_maximum_up_to_the_cap() -> None:
    small = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert envelope.shrink(small) == (envelope.BASE_S, [0.1, 0.2, 0.3])
    big = np.zeros(envelope.MAX_POINTS * 2 + 1, dtype=np.float32)
    big[5] = 0.7
    step, values = envelope.shrink(big)
    assert step == 0.15
    assert len(values) == envelope.MAX_POINTS * 2 // 3 + 1
    assert values[1] == 0.7 and max(values[2:]) == 0.0
