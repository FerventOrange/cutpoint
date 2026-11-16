"""The light frame index of the AI Chat (#989): CLIP alone, the centre crop of a sparse set of
frames, cached apart from the full visual layer.

No CLIP, no torch, no ffmpeg: the encoder is a fake returning normalised vectors and the
decode is a stub.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cutpoint import cache, crash
from cutpoint.errors import FFmpegError, LowMemoryError
from cutpoint.events import visual as V
from cutpoint.events.common import Source

LP = V.LightParams()


def source(tmp_path: Path, duration: float = 60.0) -> Source:
    (tmp_path / "cache" / "a").mkdir(parents=True)
    return Source(tmp_path / "a.mp4", duration, tmp_path / "cache" / "a", [])


def write_npz(path: Path, t: list[float], emb_crops: int, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t=np.array(t, np.float32),
        emb=np.zeros((len(t), emb_crops, 768), np.float16),
        det=np.zeros((0, 7), np.float32),
        box_emb=np.zeros((0, 768), np.float16),
        box_idx=np.zeros(0, np.int64),
    )
    cache.save_json(path.with_suffix(".json"), meta or {})


# --------------------------------------------------------------------------- light_marks
def test_marks_are_one_per_step_from_half_a_step() -> None:
    assert V.light_marks(35.0, [], LP) == [5.0, 15.0, 25.0]


def test_candidate_frames_skip_marks_near_another() -> None:
    wide = V.LightParams(step_s=100.0)  # the timeline frame is at 50 s
    # thirds of 100-106 s fall at 101, 103, 105: 103 is within LIGHT_NEAR_S of 101
    assert V.light_marks(120.0, [(100.0, 106.0)], wide) == [50.0, 101.0, 105.0]
    assert V.LIGHT_PER_CANDIDATE == 3 and V.LIGHT_NEAR_S == 3.0


def test_a_candidate_frame_near_an_indexed_one_or_one_asked_for_before_is_left_out() -> None:
    wide = V.LightParams(step_s=100.0)
    span = [(100.0, 106.0)]
    assert V.light_marks(120.0, span, wide, have=[101.5]) == [50.0, 105.0]
    assert V.light_marks(120.0, span, wide, done=[101.0, 50.0]) == [105.0]


# --------------------------------------------------------------------------- light_todo
def test_nothing_is_todo_for_a_source_with_a_full_index(tmp_path: Path) -> None:
    src = source(tmp_path)
    assert V.light_todo(src, []) != []
    write_npz(V.cache_path(src, V.VisualParams()), [1.0], 3)
    assert V.light_todo(src, []) == []


def test_nothing_is_todo_for_an_up_to_date_light_cache_unless_forced(tmp_path: Path) -> None:
    src = source(tmp_path)
    marks = V.light_todo(src, [])
    write_npz(V.light_cache_path(src), marks, 1, {"done": marks})
    assert V.light_todo(src, []) == []
    assert V.light_todo(src, [], force=True) == marks


# --------------------------------------------------------------------------- light_minutes
def test_the_estimate_counts_clip_decode_and_the_load(monkeypatch) -> None:
    monkeypatch.setattr(V, "LIGHT_CLIP_S_PER_FRAME", 2.0)
    monkeypatch.setattr(V, "LIGHT_DECODE_S_PER_FRAME", 1.0)
    monkeypatch.setattr(V, "LIGHT_LOAD_S", 60.0)
    assert V.light_minutes(120, load=False) == 6  # 120 x 3 s
    assert V.light_minutes(120, load=True) == 7
    assert V.light_minutes(0, load=False) == 1  # at least a minute


# --------------------------------------------------------------------------- compute_light
class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, bool]] = []

    def embed(self, frames: np.ndarray, batch: int = 32, centre: bool = False) -> np.ndarray:
        self.calls.append((len(frames), batch, centre))
        out = np.zeros((len(frames), 1 if centre else 3, 768), np.float32)
        out[..., 0] = 1.0  # already normalised
        return out


@pytest.fixture
def decode(monkeypatch):
    """No ffmpeg: every mark gets a frame at its own time."""
    monkeypatch.setattr(V, "probe", lambda p: SimpleNamespace(width=1920, height=1080))
    monkeypatch.setattr(V, "keyframe_spacing", lambda p: 5.0)

    def frames(path, marks, width, height, spacing):
        for m in marks:
            yield m, m, np.zeros((height, width, 3), np.uint8)

    monkeypatch.setattr(V, "_light_frames", frames)


def test_compute_light_writes_the_light_cache_with_batches_of_eight_and_centre_crops(
    tmp_path: Path, decode
) -> None:
    src = source(tmp_path, 200.0)  # 20 timeline frames
    enc = FakeEncoder()
    data = V.compute_light(src, enc, [])
    assert data is not None and data.emb.shape == (20, 1, 768)
    assert enc.calls == [(8, 8, True), (8, 8, True), (4, 8, True)]
    assert V.light_cache_path(src).exists()
    assert not V.cache_path(src, V.VisualParams()).exists()  # the full index is not written
    assert data.meta["size"] == [640, 360]
    again = V.load_visual_light(src)
    assert again is not None and again.t.tolist() == data.t.tolist()
    # a second run finds nothing to add and embeds nothing
    V.compute_light(src, enc, [])
    assert len(enc.calls) == 3


def test_load_visual_never_returns_the_light_index(tmp_path: Path) -> None:
    src = source(tmp_path)
    write_npz(V.light_cache_path(src), [5.0], 1)
    assert V.load_visual(src) is None
    assert V.load_visual_light(src) is not None


def test_writing_the_full_index_removes_the_light_cache(tmp_path: Path, monkeypatch) -> None:
    src = source(tmp_path)
    write_npz(V.light_cache_path(src), [5.0], 1, {"done": [5.0]})
    monkeypatch.setattr(V, "probe", lambda p: SimpleNamespace(width=64, height=36))
    monkeypatch.setattr(V, "keyframe_spacing", lambda p: 1.0)
    monkeypatch.setattr(V, "iter_frames", lambda *a: iter([(1.0, np.zeros((36, 64, 3), np.uint8))]))

    class Models:  # one frame in, three crops, no detections
        def run(self, frames):
            return (
                np.zeros((len(frames), 3, 768), np.float16),
                [np.zeros((0, 6))] * len(frames),
                [np.zeros((0, 768))] * len(frames),
            )

    assert V.compute_visual(src, Models(), V.VisualParams()) is not None
    assert V.cache_path(src, V.VisualParams()).exists()
    assert not V.light_cache_path(src).exists()
    assert not V.light_cache_path(src).with_suffix(".json").exists()


# --------------------------------------------------------------------------- the encoder
def tiny_encoder() -> crash.ClipEncoder:
    """A ClipEncoder on a fake model: it embeds an image as its mean colours, spread over 768."""
    torch = pytest.importorskip("torch")

    class Model:
        def encode_image(self, x):
            return x.mean((2, 3)).repeat(1, 256) + 1.0

    enc = crash.ClipEncoder.__new__(crash.ClipEncoder)
    enc.torch, enc.device, enc.dtype, enc.model = torch, "cpu", torch.float32, Model()
    enc.mean, enc.std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    enc.on_device = True
    return enc


def test_embed_of_the_centre_crop_has_one_normalised_crop_per_frame() -> None:
    enc = tiny_encoder()
    frames = np.zeros((3, 40, 80, 3), np.uint8)  # wide: left and right crops are not the centre
    frames[:, :, :20, 0] = 255  # the left quarter is red, the centre crop (x 20-60) is black
    centre = enc.embed(frames, batch=2, centre=True)
    assert centre.shape == (3, 1, 768)
    assert np.allclose(np.linalg.norm(centre, axis=-1), 1.0, atol=1e-5)
    full = enc.embed(frames, batch=2)
    assert full.shape == (3, 3, 768)
    assert not np.allclose(full[:, 0], full[:, 1])  # the left crop saw the red
    assert np.allclose(full[:, 1], centre[:, 0], atol=1e-4)  # the centre one is the same
    assert enc.embed(frames[:0], centre=True).shape == (0, 1, 768)


# --------------------------------------------------------------------------- _light_layer
class Layer:
    """Drives ``pipeline._light_layer`` with fake sources, a fake CLIP and a fake index."""

    def __init__(self, monkeypatch) -> None:
        from cutpoint import jsonout  # noqa: PLC0415
        from cutpoint.events import pipeline  # noqa: PLC0415

        self.pipeline = pipeline
        self.sources = [
            SimpleNamespace(path=Path(f"/v/{n}.mp4"), name=f"{n}.mp4") for n in ("a", "b")
        ]
        self.made: list[SimpleNamespace] = []
        self.warned: list[tuple[str, str]] = []
        self.load_error: Exception | None = None
        self.fail: dict[str, Exception] = {}

        def make(device=None):
            if self.load_error:
                raise self.load_error
            enc = SimpleNamespace(closed=False)
            enc.close = lambda: setattr(enc, "closed", True)
            self.made.append(enc)
            return enc

        def compute(src, enc, spans, force=False):
            if src.name in self.fail:
                raise self.fail[src.name]
            return SimpleNamespace(t=np.zeros(4))

        monkeypatch.setattr(crash, "ClipEncoder", make)
        monkeypatch.setattr(V, "light_todo", lambda s, spans, force=False: [5.0, 15.0])
        monkeypatch.setattr(V, "compute_light", compute)
        monkeypatch.setattr(
            jsonout, "warn", lambda code, msg, files=None: self.warned.append((code, msg))
        )

    def run(self, encoder=None, crash_run=False):
        self.kept: list = []
        return self.pipeline._light_layer({}, self.sources, self.kept, encoder, crash_run)


@pytest.fixture
def layer(monkeypatch) -> Layer:
    return Layer(monkeypatch)


def test_the_layer_logs_the_estimate_and_counts_frames(layer, caplog) -> None:
    caplog.set_level("INFO")
    stats = layer.run()
    assert stats == {"cached": 0, "computed": 2, "frames": 8}
    assert "a light frame index for the AI Chat" in caplog.text
    assert f"(about {V.light_minutes(4, load=True)} min, 4 frames)" in caplog.text
    assert layer.made[0].closed  # nobody takes the encoder it loaded


def test_the_estimate_leaves_out_the_load_for_a_given_encoder(layer, caplog) -> None:
    caplog.set_level("INFO")
    given = SimpleNamespace(close=lambda: pytest.fail("not its encoder"))
    layer.run(encoder=given)
    assert f"(about {V.light_minutes(4, load=False)} min, 4 frames)" in caplog.text
    assert not layer.made


@pytest.mark.parametrize(("crash_run", "handed"), [(True, True), (False, False)])
def test_the_loaded_encoder_goes_to_the_crash_layer_or_is_closed(layer, crash_run, handed) -> None:
    layer.run(crash_run=crash_run)
    (enc,) = layer.made
    assert (layer.kept == [enc]) is handed and enc.closed is not handed


@pytest.mark.parametrize(
    ("where", "error"),
    [
        ("load", LowMemoryError("short of memory", "1 GB free")),
        ("index", LowMemoryError("short of memory", "1 GB free")),
        ("index", RuntimeError("CUDA out of memory")),
    ],
    ids=["load", "index-guard", "index-allocation"],
)
def test_too_little_memory_warns_closes_clip_and_does_not_raise(
    layer, caplog, where, error
) -> None:
    if where == "load":
        layer.load_error = error
    else:
        layer.fail["a.mp4"] = error
    stats = layer.run(crash_run=True)
    assert stats["low_memory"] is True
    assert "Frame index for the AI Chat skipped: not enough free memory" in caplog.text
    assert [c for c, _ in layer.warned] == ["step_failed"]
    assert layer.kept == [] and all(e.closed for e in layer.made)


def test_one_bad_source_is_skipped_and_all_failing_raises(layer, caplog) -> None:
    layer.fail["a.mp4"] = ValueError("bad file")
    stats = layer.run()
    assert stats["computed"] == 1 and "low_memory" not in stats
    assert [c for c, _ in layer.warned] == ["step_failed"]
    assert "a.mp4" in layer.warned[0][1]
    layer.fail["b.mp4"] = ValueError("also bad")
    with pytest.raises(ValueError, match="also bad"):
        layer.run()
    assert layer.made[-1].closed


# --------------------------------------------------------------------------- _light_frames
@pytest.fixture
def ffmpeg(monkeypatch) -> None:
    monkeypatch.setattr(V, "find_tool", lambda name: name)


def frame() -> np.ndarray:
    return np.zeros((4, 8, 3), np.uint8)


def test_keyframe_mode_gives_each_mark_its_nearest_keyframe(ffmpeg, monkeypatch) -> None:
    def iter_frames(*a, **k):
        yield from [(t, frame()) for t in (0.0, 4.0, 8.0, 12.0)]

    monkeypatch.setattr(V, "iter_frames", iter_frames)
    got = list(V._light_frames(Path("a.mp4"), [5.0, 15.0], 8, 4, 5.0))
    assert [(m, t) for m, t, _ in got] == [(5.0, 4.0), (15.0, 12.0)]  # 15 s is after the last


def test_seek_mode_times_the_frame_from_the_mark_and_skips_a_failed_seek(
    ffmpeg, monkeypatch
) -> None:
    def decode_raw(path, start, **kw):
        if start == 20.0:
            raise FFmpegError("no keyframe")
        yield 0.7, frame()

    monkeypatch.setattr(V, "decode_raw", decode_raw)
    got = list(V._light_frames(Path("a.mp4"), [5.0, 20.0, 30.0], 8, 4, 1.0))
    assert [(m, round(t, 1)) for m, t, _ in got] == [(5.0, 5.7), (30.0, 30.7)]


# --------------------------------------------------------------------------- compute_light
@pytest.fixture
def stub_source(monkeypatch) -> None:
    monkeypatch.setattr(V, "probe", lambda p: SimpleNamespace(width=1920, height=1080))
    monkeypatch.setattr(V, "keyframe_spacing", lambda p: 5.0)


def test_two_marks_on_one_keyframe_embed_once_and_both_are_done(
    tmp_path: Path, stub_source, monkeypatch
) -> None:
    src = source(tmp_path, 35.0)  # marks 5, 15, 25
    assert V.light_todo(src, []) == [5.0, 15.0, 25.0]
    # one keyframe at 10 s is the nearest of 5 and 15; 25 gets it too (last keyframe)
    monkeypatch.setattr(
        V,
        "_light_frames",
        lambda p, marks, w, h, s: iter([(m, 10.0, np.zeros((h, w, 3), np.uint8)) for m in marks]),
    )
    enc = FakeEncoder()
    data = V.compute_light(src, enc, [])
    assert data is not None and data.t.tolist() == [10.0]
    assert sum(n for n, _, _ in enc.calls) == 1
    assert data.meta["done"] == [5.0, 15.0, 25.0]
    assert V.light_todo(src, []) == []


def test_a_second_run_with_a_new_span_adds_frames_to_the_cache(
    tmp_path: Path, stub_source, monkeypatch
) -> None:
    src = source(tmp_path, 35.0)

    def frames(path, marks, width, height, spacing):
        for m in marks:
            yield m, m, np.zeros((height, width, 3), np.uint8)

    class Marked(FakeEncoder):
        def embed(self, frames, batch=32, centre=False):
            out = super().embed(frames, batch, centre)
            out[..., 1] = len(self.calls)  # tells which run's call made the row
            return out

    monkeypatch.setattr(V, "_light_frames", frames)
    enc = Marked()
    first = V.compute_light(src, enc, [])
    assert first is not None and first.t.tolist() == [5.0, 15.0, 25.0]
    second = V.compute_light(src, enc, [(30.0, 34.0)])
    assert second is not None
    t = second.t.tolist()
    assert t == sorted(t) and len(t) == len(second.emb) > 3
    assert {5.0, 15.0, 25.0} <= set(t)
    added = [x for x in t if x not in (5.0, 15.0, 25.0)]
    assert added and all(29.0 <= x <= 34.0 for x in added)
    marker = dict(zip(t, second.emb[:, 0, 1].tolist(), strict=True))
    assert all(marker[x] == 1 for x in (5.0, 15.0, 25.0))  # old rows stay with their times
    assert all(marker[x] == 2 for x in added)
    again = V.load_visual_light(src)
    assert again is not None and again.t.tolist() == t


# --------------------------------------------------------------------------- unreadable cache
def test_an_unreadable_light_cache_is_ignored_and_rewritten(tmp_path: Path, decode) -> None:
    src = source(tmp_path, 35.0)
    V.light_cache_path(src).write_bytes(b"not an npz at all")
    assert V.load_visual_light(src) is None
    data = V.compute_light(src, FakeEncoder(), [])
    assert data is not None
    again = V.load_visual_light(src)
    assert again is not None and again.t.tolist() == data.t.tolist() == [5.0, 15.0, 25.0]
