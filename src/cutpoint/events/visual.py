"""Layer 1c (visual): per-second frame embeddings and person detections.

One decode pass per source feeds two GPU models:

* **CLIP ViT-L/14** (OpenAI weights via ``open_clip``): three square crops
  (left, centre, right) of every sampled frame become L2-normalised 768-d
  embeddings. They are reused by the zero-shot prompt scores (beauty shot,
  friends/group; captions of the activity profile) and the LAION "improved aesthetic
  predictor" head (trained on exactly these embeddings).
* **Faster R-CNN ResNet50-FPN v2** (torchvision, COCO weights, BSD licence):
  person boxes. Every confident person box is also
  cropped and embedded with CLIP, so zero-shot prompts can tell another
  person from the wearer's own legs, arms, gear or shadow (which a COCO
  detector happily calls "person" on a helmet camera looking down).

Frames: GoPro HEVC has a keyframe every 1.001 s, so a keyframe-only decode
(``-skip_frame nokey``, GPU decode: NVDEC + ``scale_cuda``, or VAAPI) yields about 1 fps for almost
no decode cost; reading the file is the bottleneck. Sources whose keyframes
are sparser than 1.5 s (gameplay recordings) are decoded fully through
``fps=1``. The spacing is probed on the first 12 s, so after a keyframe-only pass the gaps
between the decoded frames are checked too: a largest gap over 1.5 s (outside the
avoided ranges; a scene-adaptive encoder that spaces its keyframes out later) decodes the
source again through ``fps=1``. Frame times come from ``showinfo``.

Raw outputs are cached per source in ``cache/<source>/visual-<hash>.npz``.
Everything derived from them (:func:`frame_scores`) is cheap numpy, so the
prompts and the person rules can change without decoding again.

Without a CUDA GPU, ``analyze`` writes a **light frame index** instead (#989): CLIP only, the
centre crop of about one frame every :data:`LIGHT_STEP_S` seconds plus a few frames inside each
candidate, cached apart in ``cache/<source>/visual-light-<hash>.npz``
(:func:`compute_light`). Only the AI Chat's ``look_for`` reads it
(:func:`load_visual_light`); :func:`load_visual`, and so every ranking reader, never does. A
full index written later replaces it.
"""

from __future__ import annotations

import bisect
import hashlib
import importlib.util
import io
import logging
import re
import sys
import threading
import time
import zipfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cutpoint import cache, memguard, state
from cutpoint.errors import FFmpegError
from cutpoint.events import paths
from cutpoint.events.common import (
    Source,
    SourceGuard,
    for_each_source,
    free_torch,
    nearest_index,
    npz_cache_path,
    npz_present,
)
from cutpoint.ffmpeg_utils import decode_raw, ffprobe_json, find_tool, probe

if TYPE_CHECKING:
    from cutpoint.activity import Prompts

logger = logging.getLogger(__name__)

VISUAL_VERSION = 2
# OpenAI weights were trained with QuickGELU; plain "ViT-L-14" would load them with GELU.
CLIP_ARCH, CLIP_PRETRAINED = "ViT-L-14-quickgelu", "openai"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
COCO_PERSON = 1
DET_LABELS = (COCO_PERSON,)

# Zero-shot prompts come from the activity profile (:class:`cutpoint.activity.Prompts`:
# ``highlight`` for the beauty shot, ``group`` for friends, ``person`` for other people
# vs the wearer's own body). A frame's score is the softmax mass on the positive
# prompts among positives + negatives (CLIP logit scale 100). Functions taking
# ``prompts`` use :func:`cutpoint.activity.active_prompts` when it is None.


def _prompts(prompts: Prompts | None) -> Prompts:
    from cutpoint.activity import active_prompts  # noqa: PLC0415 - activity imports this module

    return prompts or active_prompts()


@dataclass(frozen=True)
class VisualParams:
    """Decode and model settings (all part of the cache key)."""

    width: int = 640  # decoded frame width (height keeps the aspect, even)
    keyframe_max_s: float = 1.5  # sparser keyframes: decode fully with fps=1
    batch: int = 16
    det_thresh: float = 0.3
    det_min_size: int = 360  # Faster R-CNN input scale (shorter side)
    box_min_score: float = 0.5  # person boxes this confident get a CLIP crop
    box_max: int = 6  # ... at most this many per frame (by score)
    hwaccel: bool = True

    def key(self) -> dict[str, Any]:
        """Cache-key dict (``hwaccel`` and ``batch`` do not change results)."""
        d = asdict(self)
        d.pop("hwaccel")
        d.pop("batch")
        return {
            "layer": "visual",
            "v": VISUAL_VERSION,
            "clip": [CLIP_ARCH, CLIP_PRETRAINED],
            "det": "fasterrcnn_resnet50_fpn_v2",
            **d,
        }


@dataclass
class VisualData:
    """Raw per-frame outputs for one source.

    Attributes:
        t: Frame times (s), shape ``(N,)``.
        emb: CLIP embeddings, ``(N, 3, 768)`` float16 (left, centre, right crop).
        det: Detections ``(M, 7)``: frame index, COCO label, score, x1, y1, x2, y2
            (coordinates normalised to 0-1).
        box_emb: CLIP embeddings of person-box crops ``(K, 768)`` float16.
        box_idx: Row in ``det`` of each ``box_emb`` row ``(K,)``.
        meta: Decode info (mode, elapsed seconds, frame size).
    """

    t: np.ndarray
    emb: np.ndarray
    det: np.ndarray
    box_emb: np.ndarray = field(default_factory=lambda: np.zeros((0, 768), np.float16))
    box_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_emb(self) -> np.ndarray:
        """Mean of the three crop embeddings, re-normalised, ``(N, 768)`` float32."""
        m = self.emb.astype(np.float32).mean(1)
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-6)


def cache_path(src: Source | Path, params: VisualParams) -> Path:
    """``cache/<source>/visual-<hash>.npz``."""
    return npz_cache_path(src, "visual", params.key())


def load_visual(src: Source | Path, params: VisualParams | None = None) -> VisualData | None:
    """Cached visual outputs of a source, or None if the layer has not run or the
    footage is offline (its cache key needs the file's size and mtime). Never the light
    frame index (:func:`load_visual_light`): what ranks reads this."""
    try:
        path = cache_path(src, params or VisualParams())
    except OSError as err:
        logger.warning("%s: footage not reachable (%s); no visual data for it", src, err)
        return None
    return _load_npz(path)


_VISUAL_ARRAYS = ("t", "emb", "det", "box_emb", "box_idx")  # what a visual cache must hold


def _load_npz(path: Path) -> VisualData | None:
    """The :class:`VisualData` cached in ``path`` (and its ``.json`` meta), or None when it
    is missing, damaged or lacks an array (a cache miss)."""
    z = cache.load_npz(path)
    if z is None or not set(_VISUAL_ARRAYS) <= z.keys():
        return None
    return VisualData(
        t=z["t"],
        emb=z["emb"],
        det=z["det"],
        box_emb=z["box_emb"],
        box_idx=z["box_idx"],
        meta=cache.load_json(path.with_suffix(".json")) or {},
    )


# --------------------------------------------------------------------------- decoding
KEYFRAME_PROBE_S = 12.0  # keyframe_spacing's probe of the start of a source


def keyframe_spacing(path: Path, probe_s: float = KEYFRAME_PROBE_S) -> float | None:
    """Median keyframe interval in the first ``probe_s`` seconds (None if unknown)."""
    try:
        data = ffprobe_json(
            path,
            streams="v:0",
            entries="frame=pts_time",
            options=("-skip_frame", "nokey", "-read_intervals", f"%+{probe_s:g}"),
            timeout=60,
        )
    except (OSError, FFmpegError):
        return None
    ts = []
    for frame in data.get("frames") or []:
        try:
            ts.append(float(frame["pts_time"]))
        except (KeyError, TypeError, ValueError):
            continue
    ts.sort()
    if len(ts) < 3:
        return None
    return float(np.median(np.diff(ts)))


def _keyframe_gap(decoded: list[float], avoid: list[tuple[float, float]] | None) -> float | None:
    """The largest gap between the ``decoded`` frame times of a keyframe-only pass, leaving out gaps that overlap an ``avoid`` range (bad sectors, where
    ffmpeg skips packets); None with fewer than two frames or no gap left."""
    ts = sorted(decoded)
    gaps = [b - a for a, b in pairwise(ts) if not any(lo < b and a < hi for lo, hi in avoid or [])]
    if not gaps:
        return None
    return max(gaps)


def iter_frames(path: Path, width: int, height: int, keyframes: bool, hwaccel: bool) -> Any:
    """Yield ``(t, HxWx3 uint8 frame)`` at ~1 fps from one sequential ffmpeg pass.

    Tries the GPU decoder first (NVDEC with ``vf_cuda``, VAAPI with ``vf``) and falls back
    to CPU decoding if it produced nothing. ``t`` is each frame's presentation time
    (``showinfo``).
    """
    find_tool("ffmpeg")  # a missing ffmpeg is an error, not "no frames"
    rate = "" if keyframes else "fps=1,"
    try:
        yield from decode_raw(
            path,
            size=(width, height),
            vf=f"{rate}scale={width}:{height}:flags=area,format=rgb24",
            vf_cuda=f"{rate}scale_cuda={width}:{height}:format=nv12,hwdownload,format=nv12,format=rgb24",
            keyframes_only=keyframes,
            pts=True,
            hwaccel="auto" if hwaccel else None,
        )
    except FFmpegError as err:
        logger.debug("visual decode produced no frames: %s", err)


# --------------------------------------------------------------------------- models
class WeightsMissing(FileNotFoundError):
    """Model weights ``cutpoint setup --only visual`` downloads are not on disk."""


class ClipWeightsMissing(WeightsMissing):
    """The CLIP weights ``cutpoint setup --only visual`` downloads are not on disk."""


class DetectorWeightsMissing(WeightsMissing):
    """The Faster R-CNN v2 checkpoint ``cutpoint setup --only visual`` downloads is not on
    disk."""


def square_crops(x: Any) -> list[Any]:
    """Three square crops of ``(N, C, H, W)`` frames along their long side: left, centre and
    right, or top, centre and bottom for portrait frames (views, no copies)."""
    h, w = x.shape[2], x.shape[3]
    s = min(h, w)
    if w >= h:
        return [x[:, :, :, o : o + s] for o in (0, (w - s) // 2, w - s)]
    return [x[:, :, o : o + s, :] for o in (0, (h - s) // 2, h - s)]


def load_clip(*, download: bool = False) -> Any:
    """The CLIP ViT-L/14 model (open_clip, CPU, float32) from the weights under
    :func:`paths.clip_dir`.

    Loads the checkpoint file setup wrote (``paths.clip_weights``) by path, so open_clip never
    goes through the Hugging Face Hub: no freshness check, no download. Only ``cutpoint
    setup`` passes ``download``, which loads by tag and lets the Hub fetch the weights.

    Raises:
        ClipWeightsMissing: without ``download`` when the weights are not on disk.
    """
    paths.apply_torch_home()  # also quiets the Hub's duplicate handler
    import open_clip  # noqa: PLC0415

    if download:
        model, _, _ = open_clip.create_model_and_transforms(
            CLIP_ARCH, pretrained=CLIP_PRETRAINED, cache_dir=str(paths.clip_dir())
        )
        return model
    weights = paths.clip_weights()
    if not weights.is_file():
        raise ClipWeightsMissing(
            f"CLIP weights not found in {paths.clip_dir()}; run: cutpoint setup --only visual"
        )
    # A file path is not a tag: open_clip loads it directly (the arch has QuickGELU built in).
    model, _, _ = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=str(weights))
    return model


def require_detector() -> Path:
    """The detector checkpoint at :func:`paths.detector_weights`; a cheap check (no torch
    import) that commands run before their long steps. torch and torchvision (the events
    extra) are checked first, so Cutpoint lite gets the missing extra, not the weights.

    Raises:
        ModuleNotFoundError: When torch or torchvision is not installed.
        DetectorWeightsMissing: When the checkpoint is not on disk.
    """
    for module in ("torch", "torchvision"):
        if module not in sys.modules and importlib.util.find_spec(module) is None:
            raise ModuleNotFoundError(f"No module named {module!r}", name=module)
    ckpt = paths.detector_weights()
    if not ckpt.is_file():
        raise DetectorWeightsMissing(
            f"Faster R-CNN detector weights not found in {ckpt.parent}; "
            "run: cutpoint setup --only visual"
        )
    return ckpt


def load_detector(
    det_thresh: float, min_size: int, max_size: int, *, download: bool = False
) -> Any:
    """The Faster R-CNN v2 COCO detector (torchvision, CPU) from the checkpoint at
    :func:`paths.detector_weights`.

    Builds the model without weights (``weights_backbone=None`` too, so torchvision fetches
    no ImageNet backbone) and loads the checkpoint file setup wrote, so torchvision never goes
    through ``load_state_dict_from_url``. Only ``cutpoint setup`` passes ``download``, which
    asks for ``weights=COCO_V1`` and lets torchvision fetch the file into the torch hub cache.

    Raises:
        DetectorWeightsMissing: without ``download`` when the checkpoint is not on disk.
    """
    paths.apply_torch_home()
    import torch  # noqa: PLC0415
    import torchvision  # noqa: PLC0415

    det = torchvision.models.detection
    size = {"box_score_thresh": det_thresh, "min_size": min_size, "max_size": max_size}
    if download:
        return det.fasterrcnn_resnet50_fpn_v2(
            weights=det.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, **size
        )
    ckpt = require_detector()
    # COCO_V1 has 91 categories; the same build torchvision does before loading its weights.
    model = det.fasterrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None, num_classes=91, **size
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return model


class VisualModels:
    """CLIP ViT-L/14 image encoder + Faster R-CNN v2, loaded once per run.

    ``download``: only for ``cutpoint setup``, which loads them once to fetch the weights.
    Without it both load from local files only (:class:`ClipWeightsMissing`,
    :class:`DetectorWeightsMissing` when they are not on disk).
    """

    def __init__(
        self,
        device: str | None = None,
        det_min_size: int = 360,
        det_thresh: float = 0.3,
        box_min_score: float = 0.5,
        box_max: int = 6,
        download: bool = False,
    ) -> None:
        paths.apply_torch_home()
        import open_clip  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import torchvision  # noqa: F401, PLC0415 - a missing extra fails before any load

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        if not download:  # cutpoint setup loads them once only to fetch the weights
            memguard.check("visual", gpu=self.device == "cuda")
        t0 = time.perf_counter()
        model = load_clip(download=download)
        self.clip = model.eval().to(self.device, self.dtype)
        self.tokenizer = open_clip.get_tokenizer(CLIP_ARCH)
        self.det = (
            load_detector(
                det_thresh, det_min_size, int(det_min_size * 16 / 9) + 1, download=download
            )
            .eval()
            .to(self.device)
        )
        self.box_min_score, self.box_max = box_min_score, box_max
        self.mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        logger.info(
            "Visual models (CLIP %s/%s + Faster R-CNN v2) on %s in %.1fs",
            CLIP_ARCH,
            CLIP_PRETRAINED,
            self.device,
            time.perf_counter() - t0,
        )

    def _embed(self, c: Any) -> Any:
        """CLIP image embeddings (L2-normalised) of ``(N, 3, h, w)`` 0-1 crops."""
        torch = self.torch
        c = torch.nn.functional.interpolate(
            c, size=(224, 224), mode="bicubic", antialias=True, align_corners=False
        ).clamp(0, 1)
        c = ((c - self.mean) / self.std).to(self.dtype)
        with torch.inference_mode():
            e = self.clip.encode_image(c).float()
        return e / e.norm(dim=-1, keepdim=True)

    def run(self, frames: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Frame embeddings ``(B, 3, 768)``, detections and person-crop embeddings.

        Returns:
            ``(emb, dets, box_embs)``: per frame, detections ``(K, 6)`` (label,
            score, normalised box) and ``(K, 768)`` crop embeddings (NaN rows for
            boxes that were not embedded).
        """
        torch = self.torch
        x = torch.from_numpy(frames).to(self.device).permute(0, 3, 1, 2).float() / 255
        b, _, h, w = x.shape
        c = torch.cat(square_crops(x), 0)  # (3B, 3, s, s): left block, centre block, right block
        e = self._embed(c)
        emb = e.view(3, b, -1).permute(1, 0, 2).cpu().numpy().astype(np.float16)
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    outs = self.det(list(x))
            else:
                outs = self.det(list(x))
        dets = []
        scale = torch.tensor([w, h, w, h], device=self.device, dtype=torch.float32)
        for o in outs:
            keep = torch.isin(o["labels"], torch.tensor(DET_LABELS, device=self.device))
            boxes = (o["boxes"][keep].float() / scale).cpu().numpy()
            dets.append(
                np.concatenate(
                    [
                        o["labels"][keep].float().cpu().numpy()[:, None],
                        o["scores"][keep].float().cpu().numpy()[:, None],
                        boxes,
                    ],
                    1,
                )
                if keep.any()
                else np.zeros((0, 6))
            )
        # Square crops (10% margin) around confident person boxes for CLIP.
        crops, where = [], []
        for i, d in enumerate(dets):
            # dets hold person boxes only (DET_LABELS above)
            cand = [j for j in np.argsort(-d[:, 1]) if d[j, 1] >= self.box_min_score][
                : self.box_max
            ]
            for j in cand:
                x1, y1, x2, y2 = d[j, 2:6] * [w, h, w, h]
                side = max(x2 - x1, y2 - y1) * 1.1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                a0, a1 = int(max(0, cx - side / 2)), int(min(w, cx + side / 2))
                b0, b1 = int(max(0, cy - side / 2)), int(min(h, cy + side / 2))
                if a1 - a0 < 8 or b1 - b0 < 8:
                    continue
                crops.append(
                    torch.nn.functional.interpolate(
                        x[i : i + 1, :, b0:b1, a0:a1],
                        size=(224, 224),
                        mode="bilinear",
                        antialias=True,
                        align_corners=False,
                    )
                )
                where.append((i, j))
        box_embs = [np.full((len(d), emb.shape[2]), np.nan, np.float32) for d in dets]
        if crops:
            be = self._embed(torch.cat(crops, 0)).cpu().numpy()
            for (i, j), v in zip(where, be, strict=True):
                box_embs[i][j] = v
        return emb, dets, box_embs

    def encode_text(self, prompts: tuple[str, ...]) -> np.ndarray:
        """L2-normalised text embeddings ``(P, 768)``."""
        torch = self.torch
        with torch.inference_mode():
            tok = self.tokenizer(list(prompts)).to(self.device)
            e = self.clip.encode_text(tok).float()
            e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy()

    def close(self) -> None:
        """Drop the models and release GPU memory."""
        del self.clip, self.det
        free_torch()


def _visual_pass(
    src: Source,
    models: VisualModels,
    params: VisualParams,
    size: tuple[int, int],
    keyframes: bool,
    avoid: list[tuple[float, float]] | None,
) -> tuple[_FrameBatches, list[float]]:
    """One decode of ``src`` (keyframes only or at 1 fps) through the models: the outputs of
    the frames kept (not within 0.5 s of the last kept one, not in an ``avoid`` range) and
    the time of every frame decoded."""
    acc = _FrameBatches(models)
    decoded: list[float] = []
    batch_t: list[float] = []
    batch_f: list[np.ndarray] = []

    def flush() -> None:
        acc.add(batch_t, batch_f)
        batch_t.clear()
        batch_f.clear()

    last = -1e9
    for t, frame in iter_frames(src.io_path, size[0], size[1], keyframes, params.hwaccel):
        decoded.append(t)
        if t - last < 0.5 or any(a <= t <= b for a, b in avoid or []):
            continue  # duplicate timestamps (open GOP) or an avoided range
        last = t
        batch_t.append(t)
        batch_f.append(frame)
        if len(batch_f) >= params.batch:
            flush()
    if batch_f:
        flush()
    return acc, decoded


class _FrameBatches:
    """Runs :meth:`VisualModels.run` on frame batches and assembles a :class:`VisualData`."""

    def __init__(self, models: VisualModels) -> None:
        self.models = models
        self.ts: list[float] = []
        self.embs: list[np.ndarray] = []
        self.dets: list[np.ndarray] = []
        self.bembs: list[np.ndarray] = []
        self.gpu_s = 0.0

    def add(self, batch_t: list[float], batch_f: list[np.ndarray]) -> None:
        """Run the models on one batch of frames taken at ``batch_t``."""
        g0 = time.perf_counter()
        e, d, be = self.models.run(np.stack(batch_f))
        self.gpu_s += time.perf_counter() - g0
        for j, dd in enumerate(d):
            self.dets.append(np.concatenate([np.full((len(dd), 1), len(self.ts) + j), dd], 1))
            self.bembs.append(be[j])
        self.ts.extend(batch_t)
        self.embs.append(e)

    def data(self) -> VisualData:
        """The assembled outputs (no ``meta``); needs at least one frame."""
        dets, bembs = self.dets, self.bembs
        det = np.concatenate(dets).astype(np.float32) if dets else np.zeros((0, 7), np.float32)
        be = np.concatenate(bembs) if bembs else np.zeros((0, 768), np.float32)
        has = np.flatnonzero(~np.isnan(be[:, 0])) if len(be) else np.zeros(0, np.int64)
        return VisualData(
            t=np.array(self.ts, np.float32),
            emb=np.concatenate(self.embs),
            det=det,
            box_emb=be[has].astype(np.float16),
            box_idx=has.astype(np.int64),
        )


def visual_data_from_frames(
    models: VisualModels, ts: list[float] | np.ndarray, frames: list[np.ndarray], batch: int = 16
) -> VisualData:
    """Model outputs for in-memory frames (no source cache).

    Args:
        models: Loaded models.
        ts: Frame times (s).
        frames: ``HxWx3`` uint8 RGB frames, one per time.
        batch: Frames per GPU batch.
    """
    if not len(ts):
        raise ValueError("visual_data_from_frames needs at least one frame")
    acc = _FrameBatches(models)
    for k in range(0, len(ts), batch):
        acc.add([float(t) for t in ts[k : k + batch]], list(frames[k : k + batch]))
    data = acc.data()
    data.meta = {"frames": len(ts), "gpu_s": round(acc.gpu_s, 1)}
    return data


def compute_visual(
    src: Source,
    models: VisualModels | None,
    params: VisualParams,
    force: bool = False,
    avoid: list[tuple[float, float]] | None = None,
) -> VisualData | None:
    """Run (or load) the visual layer for one source.

    Args:
        src: Source (its ``io_path`` is decoded; staged copies are used).
        models: Loaded models, or None to only read the cache.
        params: Decode / model settings.
        force: Ignore the cache.
        avoid: Time ranges whose frames are dropped (bad sectors are still read
            by the sequential pass; ffmpeg skips undecodable packets).
    """
    path = cache_path(src, params)
    if not force and (hit := load_visual(src, params)) is not None:
        return hit
    if models is None:
        return None
    t0 = time.perf_counter()
    info = probe(src.io_path)
    width = params.width
    height = round(width * info.height / max(1, info.width) / 2) * 2
    spacing = keyframe_spacing(src.io_path)
    keyframes = spacing is not None and spacing <= params.keyframe_max_s
    acc, decoded = _visual_pass(src, models, params, (width, height), keyframes, avoid)
    gap = _keyframe_gap(decoded, avoid) if keyframes else None
    if gap is not None and gap > params.keyframe_max_s:
        # the probe saw dense keyframes but the file spaces them out later: those stretches
        # would score 0 (to_grid), so decode the whole source at 1 fps instead
        logger.info(
            "%s: keyframes up to %.1fs apart after the first %.0fs: decoding again at 1 fps",
            src.name,
            gap,
            KEYFRAME_PROBE_S,
        )
        keyframes = False
        acc, _ = _visual_pass(src, models, params, (width, height), keyframes, avoid)
    ts, gpu_s = acc.ts, acc.gpu_s
    if not ts:
        logger.warning("%s: visual layer decoded no frames", src.name)
        return None
    data = acc.data()
    has = data.box_idx
    elapsed = time.perf_counter() - t0
    data.meta = {
        "source": str(src.path),
        "frames": len(ts),
        "mode": "keyframes" if keyframes else "fps=1",
        "keyframe_spacing_s": spacing,
        **({"keyframe_gap_s": round(gap, 2)} if gap is not None else {}),
        "size": [width, height],
        "elapsed_s": round(elapsed, 1),
        "gpu_s": round(gpu_s, 1),
        "person_crops": len(has),
        "params": params.key(),
    }
    cache.save_npz(
        path, t=data.t, emb=data.emb, det=data.det, box_emb=data.box_emb, box_idx=data.box_idx
    )
    cache.save_json(path.with_suffix(".json"), data.meta)
    drop_light(src)  # the full index replaces the light one for the chat
    logger.info(
        "%s: visual %d frames (%s) in %.1fs (GPU %.1fs, %.0fx realtime), %d detections",
        src.name,
        len(ts),
        data.meta["mode"],
        elapsed,
        gpu_s,
        src.duration / max(elapsed, 1e-6),
        len(data.det),
    )
    return data


# --------------------------------------------------------------------------- light index
LIGHT_VERSION = 1
LIGHT_STEP_S = 10.0  # one timeline frame every this many seconds
LIGHT_PER_CANDIDATE = 3  # frames inside each candidate ...
LIGHT_NEAR_S = 3.0  # ... unless an indexed (or planned) frame is this close
LIGHT_PASS_SPACING_S = 3.0  # keyframes this sparse: one keyframe-only pass, else a seek per frame
LIGHT_DUP_S = 0.5  # a decoded frame this close to an indexed one is a duplicate (one keyframe)
# Seconds per frame of the INFO line's estimate. CLIP ViT-L/14 centre crop, float32, on the CPU
# (#989): an 8-thread desktop with AVX2 (i7-9700K) took 0.76-0.79 s per frame in batches of 8
# (1.4-1.55 s one at a time), a 4-vCPU Linux VM without AVX2 1.73-1.84 s; 1.0 lies between.
# Decode: a keyframe seek took 0.18-0.30 s on the VM.
LIGHT_CLIP_S_PER_FRAME = 1.0
LIGHT_DECODE_S_PER_FRAME = 0.3
# Loading CLIP on the CPU: 14.9 s on the desktop, 20-29 s on the VM
LIGHT_LOAD_S = 25.0


@dataclass(frozen=True)
class LightParams:
    """Settings of the light frame index (``width`` and ``step_s`` are in its cache key; the
    candidate frames are not: a run adds the ones missing)."""

    width: int = 640  # decoded frame width (height keeps the aspect, even)
    step_s: float = LIGHT_STEP_S
    batch: int = 8  # frames per CLIP call: fastest on the CPU (#989: 1 and 16 are slower)

    def key(self) -> dict[str, Any]:
        """Cache-key dict (``batch`` does not change results)."""
        return {
            "layer": "visual_light",
            "v": LIGHT_VERSION,
            "clip": [CLIP_ARCH, CLIP_PRETRAINED],
            "crop": "centre",
            "width": self.width,
            "step_s": self.step_s,
        }


def light_cache_path(src: Source | Path, params: LightParams | None = None) -> Path:
    """``cache/<source>/visual-light-<hash>.npz``."""
    return npz_cache_path(src, "visual-light", (params or LightParams()).key())


def load_visual_light(src: Source | Path, params: LightParams | None = None) -> VisualData | None:
    """The cached light frame index of a source (``emb`` ``(N, 1, 768)``: the centre crop; no
    detections), or None. Only the AI Chat's ``look_for`` reads it."""
    try:
        path = light_cache_path(src, params)
    except OSError as err:
        logger.warning("%s: footage not reachable (%s); no frame index for it", src, err)
        return None
    return _load_npz(path)


def drop_light(src: Source | Path) -> None:
    """Remove the light frame index of ``src`` (a full one replaces it)."""
    try:
        path = light_cache_path(src)
    except OSError:
        return
    for p in (path, path.with_suffix(".json")):
        p.unlink(missing_ok=True)


def light_marks(
    duration: float,
    spans: Sequence[tuple[float, float]],
    params: LightParams | None = None,
    done: Sequence[float] = (),
    have: Sequence[float] = (),
) -> list[float]:
    """The times (s, 0.1 s steps, sorted) still to index for a source of ``duration``: one
    every ``step_s`` (from ``step_s / 2``) and :data:`LIGHT_PER_CANDIDATE` evenly inside each
    candidate span of ``spans``, a candidate time skipped when an indexed frame (``have``) or
    a time already planned is within :data:`LIGHT_NEAR_S`; times in ``done`` (asked for by an
    earlier run) are left out."""
    params = params or LightParams()
    want = [round(float(m), 1) for m in np.arange(params.step_s / 2, duration, params.step_s)]
    taken = sorted([*map(float, have), *want])
    for a, b in spans:
        a, b = max(0.0, float(a)), min(float(duration), float(b))
        for k in range(LIGHT_PER_CANDIDATE if b > a else 0):
            m = round(a + (k + 0.5) * (b - a) / LIGHT_PER_CANDIDATE, 1)
            i = bisect.bisect_left(taken, m)
            near = [taken[j] for j in (i - 1, i) if 0 <= j < len(taken)]
            if any(abs(m - x) < LIGHT_NEAR_S for x in near):
                continue
            bisect.insort(taken, m)
            want.append(m)
    asked = {round(float(x), 1) for x in done}
    return sorted({m for m in want if m not in asked})


def _light_frames(
    path: Path, marks: Sequence[float], width: int, height: int, spacing: float | None
) -> Iterator[tuple[float, float, np.ndarray]]:
    """``(mark, t, frame)`` for the sorted ``marks`` on the CPU: with keyframes at least
    :data:`LIGHT_PASS_SPACING_S` apart (or unknown), one keyframe-only pass and the keyframe
    nearest each mark (marks after the last keyframe get it); else one keyframe seek per mark,
    the first keyframe from it (a mark after the last keyframe gets none)."""
    find_tool("ffmpeg")  # a missing ffmpeg is an error, not "no frames"
    if spacing is None or spacing >= LIGHT_PASS_SPACING_S:
        i, prev = 0, None
        frames = iter_frames(path, width, height, keyframes=True, hwaccel=False)
        try:
            for t, f in frames:
                while i < len(marks) and marks[i] <= t:
                    m = marks[i]
                    yield (m, *prev) if prev is not None and m - prev[0] < t - m else (m, t, f)
                    i += 1
                if i >= len(marks):
                    break
                prev = (t, f)
        finally:
            frames.close()
        for m in marks[i:] if prev is not None else ():
            yield m, *prev
        return
    vf = f"scale={width}:{height}:flags=area,format=rgb24"
    for m in marks:
        seek = decode_raw(
            path, size=(width, height), vf=vf, start=m, keyframes_only=True, pts=True, hwaccel=None
        )
        try:
            got = next(seek, None)
        except FFmpegError as err:
            logger.debug("light index: no keyframe after %.1fs of %s: %s", m, path, err)
            got = None
        finally:
            seek.close()
        if got is not None:  # showinfo's time counts from the seek
            yield m, m + max(0.0, got[0]), got[1]


def light_todo(
    src: Source,
    spans: Sequence[tuple[float, float]],
    params: LightParams | None = None,
    force: bool = False,
) -> list[float]:
    """The times :func:`compute_light` would index for ``src`` (none when it has a full
    index: the chat reads that)."""
    if cache_path(src, VisualParams()).exists():
        return []
    old = None if force else load_visual_light(src, params)
    done = (old.meta.get("done") or []) if old is not None else []
    have = old.t.tolist() if old is not None else []
    return light_marks(src.duration, spans, params, done, have)


def light_minutes(frames: int, load: bool = True) -> int:
    """The INFO line's estimate (minutes, at least 1) of indexing ``frames`` frames, with
    loading CLIP on the CPU first when ``load``."""
    s = frames * (LIGHT_CLIP_S_PER_FRAME + LIGHT_DECODE_S_PER_FRAME) + (LIGHT_LOAD_S if load else 0)
    return max(1, round(s / 60))


def compute_light(
    src: Source,
    encoder: Any,
    spans: Sequence[tuple[float, float]],
    params: LightParams | None = None,
    force: bool = False,
) -> VisualData | None:
    """Add the missing frames of the light frame index of ``src`` (:func:`light_todo`) and
    return it (None when it has none).

    Args:
        src: Source (its ``io_path`` is decoded; staged copies are used).
        encoder: A CLIP image encoder (``cutpoint.crash.ClipEncoder``): ``embed(frames,
            batch=, centre=True)`` gives ``(N, 1, 768)``.
        spans: The source's candidate ``(start, end)`` spans.
        params: Index settings.
        force: Ignore the cached index.
    """
    params = params or LightParams()
    old = None if force else load_visual_light(src, params)
    marks = light_todo(src, spans, params, force)
    if not marks:
        return old
    t0 = time.perf_counter()
    info = probe(src.io_path)
    width = params.width
    height = round(width * info.height / max(1, info.width) / 2) * 2
    spacing = keyframe_spacing(src.io_path)
    done = [*((old.meta.get("done") or []) if old is not None else []), *marks]
    ts = sorted(old.t.tolist()) if old is not None else []
    new_t: list[float] = []
    new_e: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    clip_s = 0.0

    def flush() -> None:
        nonlocal clip_s
        c0 = time.perf_counter()
        new_e.append(
            np.asarray(encoder.embed(np.stack(batch), batch=params.batch, centre=True), np.float16)
        )
        clip_s += time.perf_counter() - c0
        batch.clear()

    for _, t, frame in _light_frames(src.io_path, marks, width, height, spacing):
        i = bisect.bisect_left(ts, t)
        if any(abs(t - ts[j]) < LIGHT_DUP_S for j in (i - 1, i) if 0 <= j < len(ts)):
            continue  # the keyframe of another mark
        ts.insert(i, t)
        new_t.append(t)
        batch.append(frame)
        if len(batch) >= params.batch:
            flush()
    if batch:
        flush()
    t_all = np.array([*(old.t.tolist() if old is not None else []), *new_t], np.float32)
    parts = [old.emb] if old is not None else []
    emb_all = np.concatenate([*parts, *new_e]).astype(np.float16) if parts or new_e else None
    if emb_all is None or not len(t_all):
        logger.warning("%s: frame index decoded no frames", src.name)
        return None
    order = np.argsort(t_all, kind="stable")
    data = VisualData(
        t=t_all[order],
        emb=emb_all[order],
        det=np.zeros((0, 7), np.float32),
        box_emb=np.zeros((0, 768), np.float16),
        box_idx=np.zeros(0, np.int64),
    )
    elapsed = time.perf_counter() - t0
    data.meta = {
        "source": str(src.path),
        "frames": len(data.t),
        "mode": "keyframe pass" if spacing is None or spacing >= LIGHT_PASS_SPACING_S else "seeks",
        "keyframe_spacing_s": spacing,
        "size": [width, height],
        "elapsed_s": round(elapsed, 1),
        "clip_s": round(clip_s, 1),
        "done": sorted({round(float(x), 1) for x in done}),
        "params": params.key(),
    }
    path = light_cache_path(src, params)
    cache.save_npz(
        path, t=data.t, emb=data.emb, det=data.det, box_emb=data.box_emb, box_idx=data.box_idx
    )
    cache.save_json(path.with_suffix(".json"), data.meta)
    logger.info(
        "%s: frame index +%d frames (%s) in %.1fs (CLIP %.1fs)",
        src.name,
        len(new_t),
        data.meta["mode"],
        elapsed,
        clip_s,
    )
    return data


# --------------------------------------------------------------------------- scores
def _prompt_file(prompts: tuple[str, ...]) -> Path:
    digest = hashlib.sha1(("|".join(prompts) + CLIP_ARCH + CLIP_PRETRAINED).encode()).hexdigest()[
        :10
    ]
    return paths.clip_dir() / f"text-{digest}.npy"


# CPU CLIP model (text use) and tokenizer by (arch, weights): kept by ``reset_state()``, dropped
# after TEXT_IDLE_S without a use (the AI Chat's visual search loads it in the ``cutpoint ui``
# server, #1085 A763; the next use loads it again, about 1-2 s).
_TEXT_MODEL: dict[tuple[str, str], tuple[Any, Any]] = state.register_container(
    __name__, "_TEXT_MODEL", {}, model=True
)
TEXT_IDLE_S = 600.0
_TEXT_LOCK = threading.Lock()  # held while the model loads, encodes or is dropped
_text_used = 0.0  # _clock() of the last use
_text_timer: Any = None  # the pending idle check (a threading.Timer), if any
_clock = time.monotonic  # tests pass a fake clock


def _start_timer(delay: float, fn: Callable[[], None]) -> Any:
    """A started daemon ``threading.Timer`` (tests replace it)."""
    timer = threading.Timer(delay, fn)
    timer.daemon = True
    timer.start()
    return timer


def _text_idle_check() -> None:
    """Drop the CPU text model once it went unused for :data:`TEXT_IDLE_S`, else check again
    when it would have. Never while it encodes (it holds ``_TEXT_LOCK``)."""
    global _text_timer
    dropped = False
    with _TEXT_LOCK:
        _text_timer = None
        if not _TEXT_MODEL:
            return
        idle = _clock() - _text_used
        if idle >= TEXT_IDLE_S:
            _TEXT_MODEL.clear()
            dropped = True
        else:
            _text_timer = _start_timer(TEXT_IDLE_S - idle, _text_idle_check)
    if dropped:
        free_torch()
        logger.debug("CLIP text model unloaded after %.0f s idle", TEXT_IDLE_S)


def _encode_text_cpu(prompts: tuple[str, ...]) -> np.ndarray:
    """The normalised text embeddings of ``prompts`` from the CPU text model, loaded when it
    is not (:func:`load_clip`); each use restarts its idle time."""
    global _text_used, _text_timer
    with _TEXT_LOCK:
        key = (CLIP_ARCH, CLIP_PRETRAINED)
        loaded = _TEXT_MODEL.get(key)
        if loaded is None:
            m = load_clip()
            import open_clip  # noqa: PLC0415

            m.visual = None  # text only: drops the image tower (most of the ~1.7 GB)
            loaded = _TEXT_MODEL[key] = (m.eval(), open_clip.get_tokenizer(CLIP_ARCH))
        model, tokenizer = loaded
        import torch  # noqa: PLC0415

        try:
            with torch.inference_mode():
                e = model.encode_text(tokenizer(list(prompts))).float()
                return (e / e.norm(dim=-1, keepdim=True)).numpy()
        finally:
            _text_used = _clock()
            if _text_timer is None:
                _text_timer = _start_timer(TEXT_IDLE_S, _text_idle_check)


def text_embeddings(
    prompts: tuple[str, ...], models: VisualModels | None = None, *, save: bool = True
) -> np.ndarray:
    """Text embeddings for ``prompts`` (cached in ``models/clip/text-<hash>.npy``; ``save``
    False neither reads nor writes that file, for free-text queries).

    On a cache miss the CLIP text tower is loaded (on the CPU unless ``models``
    is given), which takes a few seconds; from local files only
    (:func:`load_clip`, :class:`ClipWeightsMissing` without them). The CPU one is dropped
    after :data:`TEXT_IDLE_S` without a use and loaded again by the next.
    """
    path = _prompt_file(prompts)
    if save and path.exists():
        try:
            return np.load(path, allow_pickle=False)
        except cache.NPZ_ERRORS as err:
            logger.debug("%s: text embeddings unreadable (%s); recomputed", path, err)
    e = models.encode_text(prompts) if models is not None else _encode_text_cpu(prompts)
    if save:
        buf = io.BytesIO()  # a file object: numpy adds no suffix
        np.save(buf, e.astype(np.float32))
        cache.save_bytes(path, buf.getvalue(), parents=True)
    return e.astype(np.float32)


def prompt_score(
    emb: np.ndarray, pos: tuple[str, ...], neg: tuple[str, ...], models: VisualModels | None = None
) -> np.ndarray:
    """Softmax mass on ``pos`` among ``pos + neg`` (CLIP logit scale 100) per row.

    ``emb`` is ``(..., 768)``; the result has the leading shape.
    """
    txt = text_embeddings(pos + neg, models)
    logits = 100.0 * emb.astype(np.float32) @ txt.T
    logits -= logits.max(-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(-1, keepdims=True)
    return p[..., : len(pos)].sum(-1)


@state.register_cache
@lru_cache(maxsize=1)
def _aesthetic_layers(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """LAION improved-aesthetic-predictor MLP weights at ``path`` as numpy (768-1024-128-64-16-1).

    The head is a numpy ``.npz`` of float32 arrays named like the upstream state dict
    (``layers.0.weight``, ...), loaded without pickle, so both editions read it the same way
    and ``analyze`` can score a cached visual layer without torch.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run cutpoint setup --only visual")
    try:
        z = np.load(path, allow_pickle=False)
        if not isinstance(z, np.lib.npyio.NpzFile):
            raise ValueError("not an .npz archive")
        with z:
            arrays = {k: z[k] for k in z.files}
    except (OSError, ValueError, zipfile.BadZipFile) as err:
        raise ValueError(f"{path}: not readable ({err}); run cutpoint setup --only visual") from err
    if not arrays or not all(_LAYER_KEY.fullmatch(k) for k in arrays):
        raise ValueError(
            f"{path}: not an aesthetic head (unexpected tensor names); "
            "run cutpoint setup --only visual"
        )
    keys = sorted({k.rsplit(".", 1)[0] for k in arrays}, key=lambda k: int(k.split(".")[1]))
    layers = [(arrays.get(f"{k}.weight"), arrays.get(f"{k}.bias")) for k in keys]
    dims = _AESTHETIC_DIMS
    if len(layers) != len(dims) - 1 or not all(
        isinstance(w, np.ndarray)
        and isinstance(b, np.ndarray)
        and w.dtype == b.dtype == np.float32
        and w.shape == (n_out, n_in)
        and b.shape == (n_out,)
        for (w, b), n_in, n_out in zip(layers, dims, dims[1:], strict=False)
    ):
        raise ValueError(
            f"{path}: not an aesthetic head (expected float32 layers "
            f"{'-'.join(map(str, dims))}); run cutpoint setup --only visual"
        )
    return layers


_LAYER_KEY = re.compile(r"[^.]+\.\d+\.(weight|bias)")
_AESTHETIC_DIMS = (768, 1024, 128, 64, 16, 1)


def aesthetic(emb: np.ndarray) -> np.ndarray:
    """LAION aesthetic score (about 1-10; photos mostly 4-7) per embedding row.

    The head is purely linear at inference (its dropouts are inactive and it has
    no activations), applied to L2-normalised CLIP ViT-L/14 embeddings.
    """
    x = emb.astype(np.float32)
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
    for w, b in _aesthetic_layers(paths.aesthetic_head()):
        x = x @ w.T + b
    return x[..., 0]


@dataclass(frozen=True)
class PeopleRules:
    """Which person boxes count, and how close they are.

    A helmet camera looking down sees the wearer's own legs, boots, arms and
    gear, and a COCO detector calls them "person"; so does a shadow on the
    ground. Each confident box's crop is classified zero-shot (the activity
    profile's ``person`` prompts: somebody else vs the wearer's own body, gear or
    shadow) and dropped as "self" when the other-person share is
    below ``other_min``. Boxes without a crop embedding fall back to
    geometry: bottom at >= ``self_bottom`` and top at >= ``self_top``.
    """

    min_score: float = 0.6
    other_min: float = 0.5
    self_bottom: float = 0.93
    self_top: float = 0.40
    close_height: float = 0.55  # box height (share of the frame) that counts as "close"
    min_height: float = 0.06  # smaller people are in the background


def people_frames(
    data: VisualData, rules: PeopleRules | None = None, prompts: Prompts | None = None
) -> dict[str, np.ndarray]:
    """Per-frame person features (``prompts``: as :func:`self_boxes`).

    Returns:
        ``count`` (kept people), ``close`` (0-1 from the tallest kept box),
        ``area`` (largest kept box area share), ``self`` (1 when a dropped
        "self" box is visible).
    """
    r = rules or PeopleRules()
    n = len(data.t)
    out = {k: np.zeros(n, np.float32) for k in ("count", "close", "area", "self")}
    d = data.det
    if len(d) == 0:
        return out
    fi = d[:, 0].astype(int)
    lab, sc = d[:, 1].astype(int), d[:, 2]
    y1, y2 = d[:, 4], d[:, 6]
    hgt = y2 - y1
    area = (d[:, 5] - d[:, 3]) * hgt
    is_self = self_boxes(data, r, prompts)
    person = (lab == COCO_PERSON) & (sc >= r.min_score)
    keep = person & ~is_self & (hgt >= r.min_height)
    np.maximum.at(out["self"], fi[person & is_self], 1.0)
    np.add.at(out["count"], fi[keep], 1.0)
    np.maximum.at(out["close"], fi[keep], np.clip(hgt[keep] / r.close_height, 0, 1))
    np.maximum.at(out["area"], fi[keep], area[keep])
    return out


def self_boxes(
    data: VisualData, rules: PeopleRules | None = None, prompts: Prompts | None = None
) -> np.ndarray:
    """Boolean per detection row: the wearer's own body/gear, or a shadow (the
    ``person`` prompts of ``prompts``, default the active set)."""
    r = rules or PeopleRules()
    d = data.det
    out = (d[:, 6] >= r.self_bottom) & (d[:, 4] >= r.self_top)  # geometric fallback
    if len(data.box_idx):
        p = _prompts(prompts)
        other = prompt_score(data.box_emb, p.person_pos, p.person_neg)
        out[data.box_idx] = other < r.other_min
    return out


def steadiness(e: np.ndarray) -> np.ndarray:
    """Cosine similarity of consecutive embeddings ``(N, ..., D)`` -> ``(N, ...)``.

    Each frame takes the lower of its similarity to the previous and the next
    frame (1 = static view; one frame: 1).
    """
    steady = np.ones(e.shape[:-1], np.float32)
    if len(e) > 1:
        sim = (e[1:] * e[:-1]).sum(-1)
        steady[1:] = sim
        steady[0] = sim[0]
        steady = np.minimum(steady, np.r_[steady[1:], steady[-1:]])
    return steady


def frame_scores(
    data: VisualData, models: VisualModels | None = None, prompts: Prompts | None = None
) -> dict[str, np.ndarray]:
    """Per-frame scores (all 0-1 except ``aesthetic``).

    ``beauty_zs``: zero-shot beauty-shot contrast (the profile's ``highlight``
    prompts), mean over the three crops; ``group_zs``: friends/group contrast,
    max over crops; ``aesthetic``: LAION score of the centre crop; ``steady``:
    cosine similarity of consecutive centre crops (1 = static view); plus
    :func:`people_frames`. ``prompts``: default the active set.
    """
    p = _prompts(prompts)
    e = data.emb.astype(np.float32)
    centre = e[:, 1]
    steady = steadiness(centre)
    return {
        "beauty_zs": prompt_score(e, p.highlight_pos, p.highlight_neg, models).mean(1),
        "group_zs": prompt_score(e, p.group_pos, p.group_neg, models).max(1),
        "aesthetic": aesthetic(centre),
        "steady": steady,
        **people_frames(data, prompts=p),
    }


def beauty_score(sc: dict[str, np.ndarray], steady: np.ndarray | None = None) -> np.ndarray:
    """0-1 beauty-shot score per frame.

    Zero-shot contrast scaled by the aesthetic head, then by steadiness: a
    view held still (``steady`` 1) keeps its score, a shaky one keeps 40%.

    Args:
        sc: :func:`frame_scores` output.
        steady: Optional 0-1 steadiness per frame from another source
            (telemetry); default: embedding similarity of consecutive frames.
    """
    aes = np.clip((sc["aesthetic"] - 4.0) / 2.0, 0, 1)
    if steady is None:
        steady = np.clip((sc["steady"] - 0.75) / 0.2, 0, 1)
    return sc["beauty_zs"] * (0.4 + 0.6 * aes) * (0.4 + 0.6 * steady)


def friends_score(sc: dict[str, np.ndarray]) -> np.ndarray:
    """0-1 "friends close to the camera" per frame.

    The closeness of the nearest counted person (box height / ``close_height``),
    boosted a little by more people and by the zero-shot group prompt. The
    group prompt alone never exceeds 0.5, so a scene without a detected
    person cannot lift the lift/stop mask on its own.
    """
    n = np.minimum(sc["count"], 3)
    det = np.clip(sc["close"] * (0.8 + 0.1 * n), 0, 1)
    return np.maximum(det * (0.7 + 0.3 * sc["group_zs"]), 0.5 * sc["group_zs"] * (n > 0))


def to_grid(
    t: np.ndarray, values: np.ndarray, n: int, dt: float = 0.1, hold: float = 0.75
) -> np.ndarray:
    """Nearest-frame resampling of per-frame ``values`` onto an ``n``-bin grid.

    Bins farther than ``hold`` seconds from any frame get 0.
    """
    if len(t) == 0:
        return np.zeros(n)
    g = np.arange(n) * dt
    pick = nearest_index(t, g)
    out = values[pick].astype(float)
    out[np.abs(t[pick] - g) > hold] = 0.0
    return out


def run_layer(
    sources: list[Source],
    params: VisualParams,
    force: bool = False,
    bad: dict[str, list[tuple[float, float]]] | None = None,
    handoff: Callable[[VisualModels], None] | None = None,
    errors: list[Exception] | None = None,
) -> dict[str, Any]:
    """Compute the visual layer for every source that lacks it; returns stats.

    ``handoff`` is called with the loaded models before they are freed, so a
    later step can keep their CLIP model (``models.clip``) instead of loading
    it again; the detector is freed either way. A source whose layer fails is skipped
    with a WARNING and a ``step_failed`` warning; its error is appended to ``errors``
    (the caller decides), or without ``errors`` the last one is raised when every source
    failed. :class:`~cutpoint.errors.LowMemoryError` always propagates.
    """
    need = [s for s in sources if force or not _cached(s, params)]
    stats: dict[str, Any] = {"cached": len(sources) - len(need), "computed": 0}
    if not need:
        return stats
    models = VisualModels(
        det_min_size=params.det_min_size,
        det_thresh=params.det_thresh,
        box_min_score=params.box_min_score,
        box_max=params.box_max,
    )
    stats["device"] = models.device
    try:
        # Prompt embeddings are cached once so analysis never needs the GPU.
        p = _prompts(None)
        for pos, neg in (
            (p.highlight_pos, p.highlight_neg),
            (p.group_pos, p.group_neg),
            (p.person_pos, p.person_neg),
        ):
            text_embeddings(pos + neg, models)
        stats["computed"] = for_each_source(
            SourceGuard("Visual layer", log=logger),
            need,
            lambda s: compute_visual(
                s, models, params, force=True, avoid=(bad or {}).get(str(s.path))
            ),
            errors,
            total=len(sources),
        )
        if handoff is not None:
            handoff(models)
    finally:
        models.close()
    return stats


def _cached(src: Source, params: VisualParams) -> bool:
    """Whether the visual layer of ``src`` is cached: the archive holds every array
    :func:`load_visual` needs (read from its zip directory; no array is decompressed). An
    offline source has no cache key: not cached."""
    try:
        path = cache_path(src, params)
    except OSError:
        return False
    return npz_present(path, _VISUAL_ARRAYS)
