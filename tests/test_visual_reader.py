"""``visual.iter_frames``: a stale stderr reader must not pollute the CPU fallback pass.

Regression test for "bind loop variables in visual reader thread". No ffmpeg, no GPU:
``subprocess.Popen`` is faked and the pass order is forced with Events. The decode
loop and its reader thread live in :func:`cutpoint.ffmpeg_utils.decode_raw`, so the
fakes are installed there.
"""

from __future__ import annotations

import io
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from cutpoint import ffmpeg_utils
from cutpoint.events import visual

W, H = 2, 2
FRAME = bytes(W * H * 3)
TIMEOUT = 10.0  # only reached if the test is broken; normal runs never wait on it


def _showinfo(n: int, t: float) -> bytes:
    return f"[Parsed_showinfo_3 @ 0] n:{n} pts:{n} pts_time:{t}\n".encode()


class _Reader(threading.Thread):
    """Reader thread whose join() returns at once, like the 5 s join timing out."""

    started: ClassVar[list[_Reader]] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _Reader.started.append(self)

    def join(self, timeout=None):
        super().join(0)

    def wait_finished(self) -> None:
        super().join(TIMEOUT)
        assert not self.is_alive()


def test_stale_hw_reader_does_not_pollute_cpu_pass(monkeypatch):
    _Reader.started = []
    release_stale = threading.Event()

    def hw_stderr():
        # Outlives the hw pass: blocks until the CPU pass is reading, then writes a
        # timestamp (and, on exit, the end marker) that belongs to the hw pass.
        assert release_stale.wait(TIMEOUT)
        yield _showinfo(0, 999.0)

    def cpu_stderr():
        release_stale.set()
        _Reader.started[0].wait_finished()  # stale writes land before ours
        yield _showinfo(0, 0.5)
        yield _showinfo(1, 1.5)

    passes = [
        (io.BytesIO(b""), hw_stderr()),  # hw pass: no frames
        (io.BytesIO(FRAME * 2), cpu_stderr()),  # CPU pass: two frames
    ]
    cmds: list[list[str]] = []

    def fake_popen(cmd, **_kw):
        cmds.append(cmd)
        stdout, stderr = passes[len(cmds) - 1]
        return SimpleNamespace(
            stdout=stdout, stderr=stderr, wait=lambda: 0, poll=lambda: 0, kill=lambda: None
        )

    monkeypatch.setattr(ffmpeg_utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ffmpeg_utils, "gpu_decoder", lambda: ffmpeg_utils.HwDecoder("cuda"))
    monkeypatch.setattr(ffmpeg_utils, "find_tool", lambda name: name)
    monkeypatch.setattr(visual, "find_tool", lambda name: name)
    monkeypatch.setattr(ffmpeg_utils, "stage_inputs", lambda cmd: cmd)
    monkeypatch.setattr(
        ffmpeg_utils,
        "threading",
        SimpleNamespace(
            Thread=_Reader,
            Condition=threading.Condition,
            Event=threading.Event,
            Timer=threading.Timer,
        ),
    )

    out = list(visual.iter_frames(Path("x.mp4"), W, H, keyframes=False, hwaccel=True))

    assert len(cmds) == 2
    assert "cuda" in cmds[0] and "cuda" not in cmds[1]
    assert [t for t, _ in out] == [0.5, 1.5]
    assert all(f.shape == (H, W, 3) for _, f in out)


# ---- _aesthetic_layers: the .npz head --------------------------------------------

_DIMS = (4, 3, 2, 1)  # stands in for 768-1024-128-64-16-1


def _head(dtype=np.float32) -> dict[str, np.ndarray]:
    """A head for ``_DIMS``; layer numbers 0, 2, 10 check the numeric (not text) order."""
    rng = np.random.default_rng(1)
    out = {}
    for idx, n_in, n_out in zip((0, 2, 10), _DIMS[:-1], _DIMS[1:], strict=True):
        out[f"layers.{idx}.weight"] = rng.standard_normal((n_out, n_in)).astype(dtype)
        out[f"layers.{idx}.bias"] = rng.standard_normal(n_out).astype(dtype)
    return out


@pytest.fixture
def small_head(monkeypatch):
    """``_aesthetic_layers`` expects ``_DIMS``; its cache is cleared around the test."""
    monkeypatch.setattr(visual, "_AESTHETIC_DIMS", _DIMS)
    visual._aesthetic_layers.cache_clear()
    yield
    visual._aesthetic_layers.cache_clear()


def test_aesthetic_layers_load_npz_without_torch(tmp_path: Path, monkeypatch, small_head):
    arrays = _head()
    path = tmp_path / "head.npz"
    np.savez(path, **arrays)
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    layers = visual._aesthetic_layers(path)
    assert [w.shape for w, _ in layers] == [(3, 4), (2, 3), (1, 2)]
    for (w, b), idx in zip(layers, (0, 2, 10), strict=True):
        np.testing.assert_array_equal(w, arrays[f"layers.{idx}.weight"])
        np.testing.assert_array_equal(b, arrays[f"layers.{idx}.bias"])
        assert w.dtype == b.dtype == np.float32


def test_aesthetic_layers_missing_file_names_setup(tmp_path: Path, small_head):
    with pytest.raises(FileNotFoundError, match=r"cutpoint setup --only visual"):
        visual._aesthetic_layers(tmp_path / "head.npz")


def _object_array(path: Path) -> None:
    np.savez(path, **(_head() | {"layers.0.bias": np.array([{"x": 1}] * 3, dtype=object)}))


def _float64(path: Path) -> None:
    np.savez(path, **_head(np.float64))


def _broken_chain(path: Path) -> None:
    np.savez(path, **(_head() | {"layers.2.weight": np.zeros((2, 4), np.float32)}))


def _missing_bias(path: Path) -> None:
    np.savez(path, **{k: v for k, v in _head().items() if k != "layers.10.bias"})


def _extra_layer(path: Path) -> None:
    extra = {"layers.11.weight": np.zeros((1, 1), np.float32), "layers.11.bias": np.zeros(1)}
    np.savez(path, **(_head() | extra))


def _other_names(path: Path) -> None:
    np.savez(path, fc=np.zeros((2, 2), np.float32))


def _not_a_zip(path: Path) -> None:
    path.write_text("not a zip", encoding="utf-8")


def _bad_zip(path: Path) -> None:
    path.write_bytes(b"PK\x03\x04truncated")


def _npy(path: Path) -> None:
    with path.open("wb") as fh:
        np.save(fh, np.zeros(3, np.float32))


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (_object_array, r"not readable.*allow_pickle"),
        (_float64, r"not an aesthetic head \(expected float32 layers 4-3-2-1\)"),
        (_broken_chain, r"not an aesthetic head \(expected float32"),
        (_missing_bias, r"not an aesthetic head \(expected float32"),
        (_extra_layer, r"not an aesthetic head \(expected float32"),
        (_other_names, r"not an aesthetic head \(unexpected tensor names\)"),
        (_not_a_zip, r"not readable"),
        (_bad_zip, r"not readable"),
        (_npy, r"not readable \(not an \.npz archive\)"),
    ],
    ids=lambda v: getattr(v, "__name__", "").lstrip("_") or None,
)
def test_aesthetic_layers_rejects_a_file_that_is_not_the_head(
    tmp_path: Path, small_head, make, match
):
    path = tmp_path / "head.npz"
    make(path)
    with pytest.raises(ValueError, match=match + r".*cutpoint setup --only visual"):
        visual._aesthetic_layers(path)


# ---- pinned scores of the real aesthetic head --------------------------------

# Read at import, before conftest isolates the environment: the user's installed head
# (read only), or CUTPOINT_TEST_AESTHETIC_HEAD to check another copy of it.
_REAL_HEAD = Path(os.environ.get("CUTPOINT_TEST_AESTHETIC_HEAD") or visual.paths.aesthetic_head())


@pytest.mark.skipif(not _REAL_HEAD.is_file(), reason="real aesthetic head not installed")
def test_real_aesthetic_head_gives_pinned_scores(home: Path):
    dst = visual.paths.aesthetic_head()  # inside the isolated home
    dst.parent.mkdir(parents=True)
    dst.write_bytes(_REAL_HEAD.read_bytes())
    visual._aesthetic_layers.cache_clear()
    try:
        x = np.random.default_rng(0).standard_normal((4, 768)).astype(np.float32)
        got = visual.aesthetic(x)
    finally:
        visual._aesthetic_layers.cache_clear()
    np.testing.assert_allclose(got, [4.464581, 5.4804344, 5.008131, 5.676256], rtol=1e-6)
