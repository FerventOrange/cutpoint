"""The visual layer's decode decision: ``keyframe_spacing`` (what ffprobe says about the
keyframes) and how ``compute_visual`` uses it (keyframe-only decode or fps=1) and its
``avoid`` ranges. ffprobe, the decoder and the models are replaced; no torch, no ffmpeg."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint.errors import FFmpegError
from cutpoint.events import visual as V
from cutpoint.events.common import Source

# --------------------------------------------------------------------------- keyframe_spacing


def probe_with(monkeypatch, data=None, error: Exception | None = None) -> list[dict]:
    """Make ``ffprobe_json`` answer ``data`` (or raise ``error``); returns the call kwargs."""
    calls: list[dict] = []

    def ffprobe_json(path, **kw):
        calls.append({"path": path, **kw})
        if error is not None:
            raise error
        return data

    monkeypatch.setattr(V, "ffprobe_json", ffprobe_json)
    return calls


def frames(*times) -> dict:
    return {"frames": [{"pts_time": str(t)} for t in times]}


@pytest.mark.parametrize(
    ("times", "expected"),
    [
        ([0.0, 2.0, 4.0, 6.0], 2.0),  # a steady GOP
        ([4.0, 0.0, 6.0, 2.0], 2.0),  # ffprobe order does not matter
        ([0.0, 1.0, 2.0, 3.0, 10.0], 1.0),  # the median ignores one long gap
        ([0.0, 0.5, 1.0], 0.5),  # the smallest usable sample: three keyframes
    ],
)
def test_the_spacing_is_the_median_gap_between_keyframes(monkeypatch, times, expected) -> None:
    probe_with(monkeypatch, frames(*times))
    assert V.keyframe_spacing(Path("a.mp4")) == pytest.approx(expected)


@pytest.mark.parametrize("data", [frames(0.0, 2.0), frames(), {}, {"frames": None}])
def test_fewer_than_three_keyframes_give_no_spacing(monkeypatch, data) -> None:
    probe_with(monkeypatch, data)
    assert V.keyframe_spacing(Path("a.mp4")) is None


def test_unreadable_entries_are_skipped_not_fatal(monkeypatch) -> None:
    bad = [{"pts_time": "N/A"}, {}, {"pts_time": None}, {"pts_time": "2"}, {"pts_time": "4"}]
    probe_with(monkeypatch, {"frames": [{"pts_time": "0"}, *bad]})
    assert V.keyframe_spacing(Path("a.mp4")) == pytest.approx(2.0)
    probe_with(monkeypatch, {"frames": [{"pts_time": "N/A"}] * 5})
    assert V.keyframe_spacing(Path("a.mp4")) is None


@pytest.mark.parametrize("error", [FFmpegError("ffprobe failed"), OSError("gone")])
def test_a_failed_probe_gives_no_spacing(monkeypatch, error) -> None:
    probe_with(monkeypatch, error=error)
    assert V.keyframe_spacing(Path("a.mp4")) is None


def test_only_the_keyframes_of_the_first_seconds_of_the_video_stream_are_probed(
    monkeypatch,
) -> None:
    calls = probe_with(monkeypatch, frames(0, 1, 2))
    V.keyframe_spacing(Path("a.mp4"))
    V.keyframe_spacing(Path("a.mp4"), probe_s=30.0)
    first, second = calls
    assert first["streams"] == "v:0" and first["entries"] == "frame=pts_time"
    assert first["options"] == ("-skip_frame", "nokey", "-read_intervals", "%+12")
    assert second["options"][-1] == "%+30"
    assert first["path"] == Path("a.mp4")


# --------------------------------------------------------------------------- compute_visual
class Models:
    """One frame in, three crops, no detections."""

    def run(self, batch):
        n = len(batch)
        return (
            np.zeros((n, 3, 768), np.float16),
            [np.zeros((0, 6))] * n,
            [np.zeros((0, 768))] * n,
        )


@pytest.fixture
def decode(tmp_path: Path, monkeypatch):
    """A source whose decoder yields frames at ``state.times`` and records how it was asked."""
    (tmp_path / "cache" / "a").mkdir(parents=True)
    src = Source(tmp_path / "a.mp4", 60.0, tmp_path / "cache" / "a", [])
    state = SimpleNamespace(times=[0.0, 1.0, 2.0], spacing=1.0, asked=[], src=src)

    def iter_frames(path, width, height, keyframes, hwaccel):
        state.asked.append((width, height, keyframes, hwaccel))
        for t in state.times:
            yield t, np.zeros((height, width, 3), np.uint8)

    monkeypatch.setattr(V, "probe", lambda p: SimpleNamespace(width=1920, height=1080))
    monkeypatch.setattr(V, "keyframe_spacing", lambda p: state.spacing)
    monkeypatch.setattr(V, "iter_frames", iter_frames)
    return state


@pytest.mark.parametrize(
    ("spacing", "keyframes", "mode"),
    [
        (1.0, True, "keyframes"),
        (1.5, True, "keyframes"),  # exactly the limit still counts as dense
        (1.6, False, "fps=1"),  # sparse keyframes would skip too much: decode every frame
        (None, False, "fps=1"),  # unknown: the safe full decode
    ],
)
def test_dense_keyframes_decode_keyframes_only_and_anything_else_decodes_at_one_fps(
    decode, spacing, keyframes, mode
) -> None:
    decode.spacing = spacing
    data = V.compute_visual(decode.src, Models(), V.VisualParams(hwaccel=False))
    assert data is not None
    assert decode.asked == [(640, 360, keyframes, False)]
    assert data.meta["mode"] == mode and data.meta["keyframe_spacing_s"] == spacing
    assert data.meta["size"] == [640, 360]


def test_the_limit_is_a_parameter(decode) -> None:
    decode.spacing = 3.0
    data = V.compute_visual(decode.src, Models(), V.VisualParams(keyframe_max_s=4.0))
    assert data.meta["mode"] == "keyframes"
    assert decode.asked[0][2:] == (True, True)  # keyframes, and the default hwaccel


def test_frames_inside_an_avoid_range_are_dropped_bounds_included(decode) -> None:
    decode.times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    data = V.compute_visual(decode.src, Models(), V.VisualParams(), avoid=[(1.0, 2.0), (4.5, 9.0)])
    assert data.t.tolist() == [0.0, 3.0, 4.0]
    assert data.emb.shape[0] == 3 and data.meta["frames"] == 3


def test_frames_less_than_half_a_second_after_the_last_kept_one_are_dropped(decode) -> None:
    decode.times = [0.0, 0.2, 0.5, 0.6, 1.4, 1.5]  # open-GOP duplicates
    data = V.compute_visual(decode.src, Models(), V.VisualParams())
    assert data.t.tolist() == pytest.approx([0.0, 0.5, 1.4])


def test_an_avoided_frame_does_not_count_as_the_last_kept_one(decode) -> None:
    decode.times = [0.0, 1.0, 1.2, 2.0]
    data = V.compute_visual(decode.src, Models(), V.VisualParams(), avoid=[(0.9, 1.1)])
    assert data.t.tolist() == pytest.approx([0.0, 1.2, 2.0])  # 1.2 is not "too close" to 1.0


def test_everything_avoided_or_nothing_decoded_gives_no_layer_and_no_cache(decode, caplog) -> None:
    params = V.VisualParams()
    assert V.compute_visual(decode.src, Models(), params, avoid=[(-1.0, 99.0)]) is None
    decode.times = []
    assert V.compute_visual(decode.src, Models(), params) is None
    assert "decoded no frames" in caplog.text
    assert not V.cache_path(decode.src, params).exists()


def test_the_result_is_cached_and_a_second_run_does_not_decode(decode) -> None:
    params = V.VisualParams()
    first = V.compute_visual(decode.src, Models(), params)
    again = V.compute_visual(decode.src, None, params)  # no models: the cache or nothing
    assert again is not None and again.t.tolist() == first.t.tolist()
    assert len(decode.asked) == 1
    assert V.compute_visual(decode.src, None, params, force=True) is None  # force skips the cache


# --------------------------------------------------------------------------- #1085 A168


def test_keyframes_spaced_out_after_the_probe_decode_again_at_one_fps(decode, caplog) -> None:
    """The probe sees 1 s keyframes, but the file spaces them 4 s apart later."""
    decode.times = [0.0, 1.0, 2.0, 3.0, 7.0, 11.0, 15.0]
    with caplog.at_level("INFO"):
        data = V.compute_visual(decode.src, Models(), V.VisualParams(hwaccel=False))
    assert [a[2] for a in decode.asked] == [True, False]  # keyframes, then fps=1
    assert data.meta["mode"] == "fps=1" and data.meta["keyframe_gap_s"] == 4.0
    assert data.meta["frames"] == len(decode.times)  # only the second pass is kept
    assert "decoding again at 1 fps" in caplog.text


def test_dense_keyframes_keep_the_one_pass(decode) -> None:
    decode.times = [0.0, 1.0, 2.0, 3.0, 4.0]
    data = V.compute_visual(decode.src, Models(), V.VisualParams())
    assert [a[2] for a in decode.asked] == [True]
    assert data.meta["mode"] == "keyframes" and data.meta["keyframe_gap_s"] == 1.0


def test_a_gap_inside_an_avoided_range_does_not_decode_again(decode) -> None:
    decode.times = [0.0, 1.0, 2.0, 9.0, 10.0]  # ffmpeg skipped a bad stretch
    data = V.compute_visual(decode.src, Models(), V.VisualParams(), avoid=[(3.0, 8.0)])
    assert [a[2] for a in decode.asked] == [True]
    assert data.meta["mode"] == "keyframes"
