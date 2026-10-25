"""``visual.iter_frames``: a stale stderr reader must not pollute the CPU fallback pass.

Regression test for "bind loop variables in visual reader thread". No ffmpeg, no GPU:
``subprocess.Popen`` is faked and the pass order is forced with Events. The decode
loop and its reader thread live in :func:`cutpoint.ffmpeg_utils.decode_raw`, so the
fakes are installed there.
"""

from __future__ import annotations

import collections
import io
import os
import pickle
import subprocess
import sys
import threading
import types
import zipfile
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


# ---- _read_state_dict / _aesthetic_layers without torch --------------------


def _torch_stub(module: str, name: str) -> type:
    return type(name, (), {"__module__": module, "__qualname__": name})


def _pickle_checkpoint(tensors: dict[str, tuple]) -> bytes:
    """A torch.save-style data.pkl: OrderedDict of _rebuild_tensor_v2 calls (protocol 2)."""

    mods = {}
    for m, names in {"torch": ["FloatStorage"], "torch._utils": ["_rebuild_tensor_v2"]}.items():
        mod = types.ModuleType(m)
        for n in names:
            setattr(mod, n, _torch_stub(m, n))
        mods[m] = mod
    saved = {m: sys.modules.get(m) for m in mods}
    sys.modules.update(mods)
    try:

        class Tensor:
            def __init__(self, key, offset, size, stride):
                self.args = (key, offset, size, stride)

            def __reduce__(self):
                key, offset, size, stride = self.args
                storage = _Storage(key, int(np.prod(size)) + offset)
                rebuild = mods["torch._utils"]._rebuild_tensor_v2
                return rebuild, (storage, offset, size, stride, False, collections.OrderedDict())

        class _Storage:
            def __init__(self, key, n):
                self.key, self.n = key, n

        class P(pickle.Pickler):
            def persistent_id(self, obj):
                if isinstance(obj, _Storage):
                    return ("storage", mods["torch"].FloatStorage, obj.key, "cpu", obj.n)
                return None

        buf = io.BytesIO()
        P(buf, protocol=2).dump(
            collections.OrderedDict((k, Tensor(*v)) for k, v in tensors.items())
        )
        return buf.getvalue()
    finally:
        for m, old in saved.items():
            if old is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = old


def _write_checkpoint(path: Path, arrays: dict[str, np.ndarray], tensors: dict[str, tuple]):

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", _pickle_checkpoint(tensors))
        zf.writestr("archive/byteorder", "little")
        for key, arr in arrays.items():
            zf.writestr(f"archive/data/{key}", arr.astype("<f4").tobytes())


def test_read_state_dict_reads_torch_zip_with_offset_and_strides(tmp_path: Path):
    plain = np.arange(6, dtype=np.float32)
    shifted = np.arange(10, dtype=np.float32)  # tensor starts at element 4
    transposed = np.arange(6, dtype=np.float32)  # a (3, 2) view with strides (1, 3)
    path = tmp_path / "m.pth"
    _write_checkpoint(
        path,
        {"0": plain, "1": shifted, "2": transposed},
        {
            "a": ("0", 0, (2, 3), (3, 1)),
            "b": ("1", 4, (6,), (1,)),
            "c": ("2", 0, (3, 2), (1, 3)),
        },
    )
    sd = visual._read_state_dict(path)
    np.testing.assert_array_equal(sd["a"], plain.reshape(2, 3))
    np.testing.assert_array_equal(sd["b"], shifted[4:])
    np.testing.assert_array_equal(sd["c"], plain.reshape(2, 3).T)


def test_read_state_dict_rejects_disallowed_global(tmp_path: Path):

    class Evil:
        def __reduce__(self):
            return os.system, ("echo pwned",)

    path = tmp_path / "evil.pth"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps(Evil(), protocol=2))
    with pytest.raises(
        ValueError, match=r"not readable without torch.*cutpoint setup --only visual"
    ):
        visual._read_state_dict(path)


def test_aesthetic_layers_load_without_torch(tmp_path: Path, monkeypatch):
    w = [np.arange(4, dtype=np.float32), np.arange(6, dtype=np.float32)]
    b = [np.ones(2, dtype=np.float32), np.ones(1, dtype=np.float32)]
    path = tmp_path / "head.pth"
    _write_checkpoint(
        path,
        {"0": w[0], "1": b[0], "2": w[1], "3": b[1]},
        {
            "layers.0.weight": ("0", 0, (2, 2), (2, 1)),
            "layers.0.bias": ("1", 0, (2,), (1,)),
            "layers.2.weight": ("2", 0, (1, 6), (6, 1)),
            "layers.2.bias": ("3", 0, (1,), (1,)),
        },
    )
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    monkeypatch.setattr(visual.paths, "aesthetic_head", lambda: path)
    visual._aesthetic_layers.cache_clear()
    try:
        layers = visual._aesthetic_layers(visual.paths.aesthetic_head())
    finally:
        visual._aesthetic_layers.cache_clear()
    assert [a.shape for a, _ in layers] == [(2, 2), (1, 6)]
    np.testing.assert_array_equal(layers[0][0], w[0].reshape(2, 2))
    np.testing.assert_array_equal(layers[1][1], b[1])


_PROBE = """
import subprocess
import sys
from pathlib import Path
from cutpoint.events import visual
try:
    visual._read_state_dict(Path(sys.argv[1]))
except ValueError as err:
    print("ValueError", err)
"""


@pytest.mark.parametrize(
    "spec",
    [
        ("0", 0, (1000, 1000), (1000, 1)),  # far larger than the 6-element storage
        ("0", 0, (3,), (-100000,)),  # negative stride
        ("0", -2, (2,), (1,)),  # negative offset
        ("0", 0, (2, 3), (1,)),  # size and stride of different lengths
    ],
    ids=["oversized", "negative-stride", "negative-offset", "rank-mismatch"],
)
def test_read_state_dict_rejects_views_outside_the_storage(tmp_path: Path, spec):
    path = tmp_path / "bad.pth"
    _write_checkpoint(path, {"0": np.arange(6, dtype=np.float32)}, {"a": spec})
    # a subprocess: an unchecked view would read foreign memory and may crash the interpreter
    run = subprocess.run(
        [sys.executable, "-c", _PROBE, str(path)], capture_output=True, text=True, timeout=60
    )
    assert run.returncode == 0, run.stderr
    assert "ValueError" in run.stdout and "not readable without torch" in run.stdout


def test_read_state_dict_rejects_non_dict_and_non_array_values(tmp_path: Path):
    plain = tmp_path / "list.pth"
    with zipfile.ZipFile(plain, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps([1, 2], protocol=2))
    with pytest.raises(
        ValueError, match=r"not readable without torch.*cutpoint setup --only visual"
    ):
        visual._read_state_dict(plain)
    values = tmp_path / "values.pth"
    with zipfile.ZipFile(values, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps({"a": 1.5}, protocol=2))
    with pytest.raises(
        ValueError, match=r"not readable without torch.*cutpoint setup --only visual"
    ):
        visual._read_state_dict(values)


def test_aesthetic_layers_rejects_a_checkpoint_that_is_not_an_aesthetic_head(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "other.pth"
    _write_checkpoint(path, {"0": np.arange(4, dtype=np.float32)}, {"fc": ("0", 0, (2, 2), (2, 1))})
    monkeypatch.setitem(sys.modules, "torch", None)
    visual._aesthetic_layers.cache_clear()
    try:
        with pytest.raises(
            ValueError, match=r"not an aesthetic head.*cutpoint setup --only visual"
        ):
            visual._aesthetic_layers(path)
    finally:
        visual._aesthetic_layers.cache_clear()
