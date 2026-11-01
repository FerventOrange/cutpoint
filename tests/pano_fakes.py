"""Numpy fakes of torch, CLIP, the detector and the fisheye sampler for the 360 view-layer
tests (``views.compute_views``)."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np

from cutpoint.pano import geometry as G

N_FRAMES = 20  # 3 batches of 8, 8, 4


class _T(np.ndarray):
    """The few ``torch.Tensor`` methods ``compute_views`` uses, on numpy."""

    def to(self, *a, **k):
        return self

    def permute(self, *dims):
        return self.transpose(dims)

    def float(self):
        return self.astype(np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def view(self, *args, **kw):
        if args and all(isinstance(a, int) for a in args):
            return self.reshape(args)
        return super().view(*args, **kw)


def _interpolate(x, size, **kw):
    """Nearest resize of ``(1, 3, h, w)`` to ``size``."""
    h, w = x.shape[2:]
    yi = np.arange(size[0]) * h // size[0]
    xi = np.arange(size[1]) * w // size[1]
    return np.ascontiguousarray(x[:, :, yi][:, :, :, xi]).view(_T)


FAKE_TORCH = SimpleNamespace(
    tensor=lambda x, device=None: np.asarray(x),
    from_numpy=lambda a: a.view(_T),
    inference_mode=contextlib.nullcontext,
    cat=lambda xs, dim: np.concatenate(xs, dim).view(_T),
    isin=np.isin,
    nn=SimpleNamespace(functional=SimpleNamespace(interpolate=_interpolate)),
)


class Sampler:
    """Stand-in for ``FisheyeSampler``: each pixel is the frame's mean colour plus its ray in
    camera coordinates, so the output depends on the frame and on the view rotation."""

    def __init__(self, device, fov=G.DFISHEYE_FOV):
        pass

    def rays(self, d):
        return np.asarray(d.reshape(-1, 3), np.float32), d.shape[:-1]

    def sample(self, frames, r_vw, rays, view_rot=None):
        d, shape = rays
        full = r_vw[:, None] if view_rot is None else np.einsum("bij,kjl->bkil", r_vw, view_rot)
        cam = np.einsum("bkij,nj->bkjn", full, d)  # (B, K, 3, N)
        base = np.asarray(frames).mean((2, 3))[:, None, :, None]
        out = (base + 0.1 * cam).astype(np.float32).reshape(*full.shape[:2], 3, *shape)
        out = out.view(_T)
        return out if view_rot is not None else out[:, 0]


class Detector:
    """Per frame: the wearer (a ~90-deg box near straight ahead, off the default mount and
    off the pixel grid, so its ray is not axis-aligned), another
    person moving with the frame, and a COCO label that is not kept."""

    def __init__(self, rec):
        self.rec = rec
        self.transform = SimpleNamespace(min_size=None, max_size=None)
        self.roi_heads = SimpleNamespace(score_thresh=None)

    def __call__(self, images):
        self.rec.det_calls += 1
        outs = []
        for img in images:
            m = float(np.asarray(img).mean())
            x0 = 2.0 + (m * 997.0) % 10.0
            outs.append(
                {
                    "boxes": np.asarray(
                        [[15.3, 3.7, 24.1, 12.9], [x0, 6, x0 + 9, 15], [5, 5, 7, 7]], np.float32
                    ).view(_T),
                    "labels": np.asarray([1, 1, 3], np.int64).view(_T),
                    "scores": np.asarray([0.95, 0.4 + (m * 31.0) % 0.5, 0.9], np.float32).view(_T),
                }
            )
        return outs


class Models:
    def __init__(self, rec, **kw):
        self.rec = rec
        self.device = "cpu"
        self.det = Detector(rec)
        rec.built.append(self)

    def _embed(self, c):
        self.rec.embeds.append(len(c))
        flat = np.asarray(c, np.float32).reshape(len(c), -1)
        out = np.zeros((len(c), 768), np.float32)
        out[:, : min(768, flat.shape[1])] = flat[:, :768]
        return out.view(_T)

    def close(self):
        self.rec.closed += 1


def frames(path, width, height, keyframes, hwaccel):
    for i in range(N_FRAMES):
        frame = np.full((8, 16, 3), (i * 37) % 251, np.uint8)
        frame[:, :, 1] = (i * 11) % 7
        yield i * 1.0, frame
