"""One-off converter: the LAION improved-aesthetic-predictor head from the upstream pickled
``.pth`` (``torch.save`` zip) to the numpy ``.npz`` that ``cutpoint setup --only visual``
downloads. Not part of the package; it only reads the source file.

The ``.pth`` is read twice when torch is installed: with the restricted numpy unpickler below
(copied from ``cutpoint.events.visual`` before the switch to ``.npz``) and with
``torch.load(weights_only=True)``. Both must give identical float32 arrays. The ``.npz`` keeps
the state-dict keys (``layers.0.weight`` ...) and values unchanged.

Usage::

    uv run python scripts/convert_aesthetic_head.py SRC.pth DST.npz

``np.savez`` output differs byte for byte on every run (zip timestamps), so convert once and
pin the size and sha256 it prints (``AESTHETIC_SIZE``/``AESTHETIC_SHA256`` in
``cutpoint.events.setup``) for that exact file.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

_NP_STORAGE = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64}


def _read_state_dict(path: Path) -> dict[str, np.ndarray]:
    """Tensors of a ``torch.save`` zip checkpoint (a state dict of float tensors)
    as numpy arrays, without torch. Only plain tensors are unpickled.

    Raises:
        ValueError: Not such a checkpoint.
    """

    def rebuild(storage: np.ndarray, offset: int, size: tuple, stride: tuple, *_: Any) -> Any:
        ints = (int, np.integer)
        if not (
            isinstance(storage, np.ndarray)
            and isinstance(offset, ints)
            and offset >= 0
            and isinstance(size, tuple)
            and isinstance(stride, tuple)
            and len(size) == len(stride)
            and all(isinstance(v, ints) for v in size + stride)
        ):
            raise ValueError("malformed tensor")
        isz = storage.itemsize
        # np.ndarray checks the view against the buffer bounds (as_strided does not)
        return np.ndarray(
            tuple(size),
            storage.dtype,
            buffer=storage,
            offset=offset * isz,
            strides=tuple(s * isz for s in stride),
        ).copy()

    allowed = {
        ("collections", "OrderedDict"): collections.OrderedDict,
        ("torch._utils", "_rebuild_tensor_v2"): rebuild,
        ("torch._utils", "_rebuild_parameter"): lambda data, *_: data,
    }
    try:
        with zipfile.ZipFile(path) as zf:
            pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
            root = pkl[: -len("data.pkl")]
            if root + "byteorder" in zf.namelist() and zf.read(root + "byteorder") != b"little":
                raise ValueError("big-endian checkpoint")

            class _Unpickler(pickle.Unpickler):
                def find_class(self, module: str, name: str) -> Any:
                    if module == "torch" and name in _NP_STORAGE:
                        return name
                    if (module, name) in allowed:
                        return allowed[module, name]
                    raise pickle.UnpicklingError(f"{module}.{name} is not allowed")

                def persistent_load(self, pid: Any) -> Any:
                    _, kind, key, _location, _numel = pid
                    return np.frombuffer(zf.read(f"{root}data/{key}"), _NP_STORAGE[kind])

            sd = _Unpickler(io.BytesIO(zf.read(pkl))).load()
            if not isinstance(sd, dict) or not all(isinstance(v, np.ndarray) for v in sd.values()):
                raise pickle.UnpicklingError("not a state dict of tensors")
    except (
        OSError,
        KeyError,
        StopIteration,
        TypeError,
        ValueError,
        AttributeError,
        pickle.UnpicklingError,
        zipfile.BadZipFile,
    ) as err:
        raise ValueError(f"{path}: not readable without torch ({err})") from err
    return dict(sd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path, help="upstream sac+logos+ava1-l14-linearMSE.pth")
    ap.add_argument("dst", type=Path, help="output .npz (must not exist)")
    args = ap.parse_args()
    if args.dst.suffix != ".npz":
        ap.error("dst must end in .npz (np.savez would add it)")
    if args.dst.exists():
        ap.error(f"{args.dst} exists; pick a new path")

    sd = _read_state_dict(args.src)
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        print("torch not installed: read with the numpy unpickler only")
    else:
        ref = torch.load(args.src, map_location="cpu", weights_only=True)
        assert list(ref) == list(sd), (list(ref), list(sd))
        for k, v in ref.items():
            t = v.numpy()
            assert t.dtype == sd[k].dtype and t.shape == sd[k].shape, k
            assert np.array_equal(t, sd[k]), k
        print(f"torch.load and the numpy unpickler agree on {len(sd)} arrays")
    for k, v in sd.items():
        assert v.dtype == np.float32, (k, v.dtype)
        print(f"  {k:<20} {v.dtype} {v.shape}")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.dst, **sd)
    with np.load(args.dst, allow_pickle=False) as back:
        assert list(back.files) == list(sd)
        for k, v in sd.items():
            assert back[k].dtype == v.dtype and np.array_equal(back[k], v), k
    data = args.dst.read_bytes()
    print(f"wrote {args.dst}")
    print(f"size   {len(data)}")
    print(f"sha256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
