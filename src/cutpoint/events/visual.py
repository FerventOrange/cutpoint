"""Layer 1c (visual): per-second frame embeddings and person detections.

One decode pass per source feeds two GPU models:

* **CLIP ViT-L/14** (OpenAI weights via ``open_clip``): three square crops
  (left, centre, right) of every sampled frame become L2-normalised 768-d
  embeddings. They are reused by the zero-shot prompt scores (beauty shot,
  friends/group), the LAION "improved aesthetic predictor" head (trained on
  exactly these embeddings) and the personal *taste* head that
  ``cutpoint refmatch`` fits on a reference edit.
* **Faster R-CNN ResNet50-FPN v2** (torchvision, COCO weights, BSD licence):
  person, skis and snowboard boxes. Every confident person box is also
  cropped and embedded with CLIP, so zero-shot prompts can tell another
  person from the wearer's own legs, arms, skis or shadow (which a COCO
  detector happily calls "person" on a helmet camera looking down).

Frames: GoPro HEVC has a keyframe every 1.001 s, so a keyframe-only decode
(``-skip_frame nokey``, GPU decode: NVDEC + ``scale_cuda``, or VAAPI) yields about 1 fps for almost
no decode cost; reading the file is the bottleneck. Sources whose keyframes
are sparser than 1.5 s (gameplay recordings) are decoded fully through
``fps=1``. Frame times come from ``showinfo``.

Raw outputs are cached per source in ``cache/<source>/visual-<hash>.npz``.
Everything derived from them (:func:`frame_scores`) is cheap numpy, so the
prompts and the person rules can change without decoding again.
"""

from __future__ import annotations

import collections
import hashlib
import io
import logging
import pickle
import re
import time
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache, state
from cutpoint.events import paths
from cutpoint.events.common import Source, free_torch, nearest_index, params_key
from cutpoint.ffmpeg_utils import FFmpegError, decode_raw, ffprobe_json, find_tool, probe

logger = logging.getLogger(__name__)

VISUAL_VERSION = 2
# OpenAI weights were trained with QuickGELU; plain "ViT-L-14" would load them with GELU.
CLIP_ARCH, CLIP_PRETRAINED = "ViT-L-14-quickgelu", "openai"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
COCO_PERSON, COCO_SKIS, COCO_SNOWBOARD = 1, 35, 36
DET_LABELS = (COCO_PERSON, COCO_SKIS, COCO_SNOWBOARD)

# Zero-shot prompts. A frame's score is the softmax mass on the positive
# prompts among positives + negatives (CLIP logit scale 100).
BEAUTY_POS = (
    "a breathtaking mountain vista",
    "sunlit snowy mountain peaks under a blue sky",
    "snow covered trees in a winter forest",
    "a view over a valley from a mountain ridge",
    "alpenglow on snowy mountains at sunset",
    "a scenic panorama of mountains and a ski resort",
    "clouds and fog below a mountain summit",
    "a beautiful winter landscape",
)
BEAUTY_NEG = (
    "a close-up of ski boots",
    "a close-up of skis on the snow",
    "blurry snow",
    "the safety bar of a chairlift",
    "a phone screen",
    "a dark blurry image",
    "a close-up of a glove",
    "the inside of a building",
    "a flat white foggy image",
    "a close-up of a jacket",
    "a groomed ski run seen while skiing",
)
GROUP_POS = (
    "a group of friends standing together in the snow",
    "friends sitting next to me on a chairlift",
    "a selfie of friends wearing ski helmets and goggles",
    "a person in ski clothes talking to the camera",
    "people laughing together",
    "a friend skiing right in front of the camera",
)
# Person-box crops: is this somebody else, or the wearer's own body / shadow?
SELF_POS = (
    "a first-person view looking down at my own legs and ski boots",
    "my own arm and glove holding a ski pole",
    "my own skis seen from above",
    "my own snowboard and boots seen from above",
    "the shadow of a person on the snow",
    "a blurry close-up of a jacket sleeve",
)
OTHER_POS = (
    "a person wearing a ski helmet and goggles",
    "a skier",
    "a snowboarder",
    "a person sitting on a chairlift",
    "a person's face",
    "a person standing in the snow",
    "a group of people",
)
GROUP_NEG = (
    "an empty ski slope",
    "a skier far away in the distance",
    "a close-up of ski boots",
    "snowy trees with nobody around",
    "an empty chairlift",
    "a mountain landscape",
)


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
    cdir = src.cdir if isinstance(src, Source) else cache.source_dir(src)
    return cdir / f"visual-{params_key(params.key())}.npz"


def load_visual(src: Source | Path, params: VisualParams | None = None) -> VisualData | None:
    """Cached visual outputs of a source, or None if the layer has not run or the
    footage is offline (its cache key needs the file's size and mtime)."""
    try:
        path = cache_path(src, params or VisualParams())
    except OSError as err:
        logger.warning("%s: footage not reachable (%s); no visual data for it", src, err)
        return None
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        meta = cache.load_json(path.with_suffix(".json")) or {}
        return VisualData(
            t=z["t"],
            emb=z["emb"],
            det=z["det"],
            box_emb=z["box_emb"],
            box_idx=z["box_idx"],
            meta=meta,
        )


# --------------------------------------------------------------------------- decoding
def keyframe_spacing(path: Path, probe_s: float = 12.0) -> float | None:
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
            vf_cuda=f"{rate}scale_cuda={width}:{height},hwdownload,format=nv12,format=rgb24",
            keyframes_only=keyframes,
            pts=True,
            hwaccel="auto" if hwaccel else None,
        )
    except FFmpegError as err:
        logger.debug("visual decode produced no frames: %s", err)


# --------------------------------------------------------------------------- models
class VisualModels:
    """CLIP ViT-L/14 image encoder + Faster R-CNN v2, loaded once per run."""

    def __init__(
        self,
        device: str | None = None,
        det_min_size: int = 360,
        det_thresh: float = 0.3,
        box_min_score: float = 0.5,
        box_max: int = 6,
    ) -> None:
        paths.apply_torch_home()
        import open_clip  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import torchvision  # noqa: PLC0415

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        t0 = time.perf_counter()
        model, _, _ = open_clip.create_model_and_transforms(
            CLIP_ARCH, pretrained=CLIP_PRETRAINED, cache_dir=str(paths.clip_dir())
        )
        self.clip = model.eval().to(self.device, self.dtype)
        self.tokenizer = open_clip.get_tokenizer(CLIP_ARCH)
        det = torchvision.models.detection
        self.det = (
            det.fasterrcnn_resnet50_fpn_v2(
                weights=det.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
                box_score_thresh=det_thresh,
                min_size=det_min_size,
                max_size=int(det_min_size * 16 / 9) + 1,
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
        s = min(h, w)
        offs = [0, (w - s) // 2, w - s] if w >= h else [0, (h - s) // 2, h - s]
        crops = [x[:, :, :, o : o + s] if w >= h else x[:, :, o : o + s, :] for o in offs]
        c = torch.cat(crops, 0)  # (3B, 3, s, s): left block, centre block, right block
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
            cand = [
                j
                for j in np.argsort(-d[:, 1])
                if d[j, 0] == COCO_PERSON and d[j, 1] >= self.box_min_score
            ][: self.box_max]
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
    acc = _FrameBatches(models)
    batch_t: list[float] = []
    batch_f: list[np.ndarray] = []

    def flush() -> None:
        acc.add(batch_t, batch_f)
        batch_t.clear()
        batch_f.clear()

    last = -1e9
    for t, frame in iter_frames(src.io_path, width, height, keyframes, params.hwaccel):
        if t - last < 0.5 or any(a <= t <= b for a, b in avoid or []):
            continue  # duplicate timestamps (open GOP) or an avoided range
        last = t
        batch_t.append(t)
        batch_f.append(frame)
        if len(batch_f) >= params.batch:
            flush()
    if batch_f:
        flush()
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
        "size": [width, height],
        "elapsed_s": round(elapsed, 1),
        "gpu_s": round(gpu_s, 1),
        "person_crops": len(has),
        "params": params.key(),
    }
    np.savez_compressed(
        path, t=data.t, emb=data.emb, det=data.det, box_emb=data.box_emb, box_idx=data.box_idx
    )
    cache.save_json(path.with_suffix(".json"), data.meta)
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


# --------------------------------------------------------------------------- scores
def _prompt_file(prompts: tuple[str, ...]) -> Path:
    digest = hashlib.sha1(("|".join(prompts) + CLIP_ARCH + CLIP_PRETRAINED).encode()).hexdigest()[
        :10
    ]
    return paths.clip_dir() / f"text-{digest}.npy"


# CPU CLIP model (text use) and tokenizer by (arch, weights): kept by ``reset_state()``.
_TEXT_MODEL: dict[tuple[str, str], tuple[Any, Any]] = state.register_container(
    __name__, "_TEXT_MODEL", {}, model=True
)


def text_embeddings(prompts: tuple[str, ...], models: VisualModels | None = None) -> np.ndarray:
    """Text embeddings for ``prompts`` (cached in ``models/clip/text-<hash>.npy``).

    On a cache miss the CLIP text tower is loaded (on the CPU unless ``models``
    is given), which takes a few seconds once per process.
    """
    path = _prompt_file(prompts)
    if path.exists():
        return np.load(path)
    if models is not None:
        e = models.encode_text(prompts)
    else:
        key = (CLIP_ARCH, CLIP_PRETRAINED)
        loaded = _TEXT_MODEL.get(key)
        if loaded is None:
            import open_clip  # noqa: PLC0415

            m, _, _ = open_clip.create_model_and_transforms(
                CLIP_ARCH, pretrained=CLIP_PRETRAINED, cache_dir=str(paths.clip_dir())
            )
            loaded = _TEXT_MODEL[key] = (m.eval(), open_clip.get_tokenizer(CLIP_ARCH))
        model, tokenizer = loaded
        import torch  # noqa: PLC0415

        with torch.inference_mode():
            e = model.encode_text(tokenizer(list(prompts))).float()
            e = (e / e.norm(dim=-1, keepdim=True)).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, e.astype(np.float32))
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

    Without torch (no ``events`` extra) the checkpoint is read with numpy, so
    ``analyze`` can score a cached visual layer.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run cutpoint setup --only visual")
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        logger.debug("torch not installed; reading the aesthetic head with numpy")
        arrays = _read_state_dict(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        arrays = {k: v.float().numpy() for k, v in sd.items()}
    if not all(isinstance(k, str) and _LAYER_KEY.fullmatch(k) for k in arrays):
        raise ValueError(
            f"{path}: not an aesthetic head (unexpected tensor names); "
            "run cutpoint setup --only visual"
        )
    keys = sorted({k.rsplit(".", 1)[0] for k in arrays}, key=lambda k: int(k.split(".")[1]))
    return [
        (arrays[f"{k}.weight"].astype(np.float32), arrays[f"{k}.bias"].astype(np.float32))
        for k in keys
    ]


_LAYER_KEY = re.compile(r"[^.]+\.\d+\.(weight|bias)")
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
        raise ValueError(
            f"{path}: not readable without torch ({err}); run cutpoint setup --only visual"
        ) from err
    return dict(sd)


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
    skis, and a COCO detector calls them "person"; so does a shadow on the
    snow. Each confident box's crop is classified zero-shot (``OTHER_POS``
    vs ``SELF_POS``) and dropped as "self" when the other-person share is
    below ``other_min``. Boxes without a crop embedding fall back to
    geometry: bottom at >= ``self_bottom`` and top at >= ``self_top``.
    """

    min_score: float = 0.6
    other_min: float = 0.5
    self_bottom: float = 0.93
    self_top: float = 0.40
    close_height: float = 0.55  # box height (share of the frame) that counts as "close"
    min_height: float = 0.06  # smaller people are background skiers


def people_frames(data: VisualData, rules: PeopleRules | None = None) -> dict[str, np.ndarray]:
    """Per-frame person features.

    Returns:
        ``count`` (kept people), ``close`` (0-1 from the tallest kept box),
        ``area`` (largest kept box area share), ``self`` (1 when a dropped
        "self" box or skis/snowboard at the bottom edge is visible).
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
    is_self = self_boxes(data, r)
    gear = np.isin(lab, (COCO_SKIS, COCO_SNOWBOARD)) & (y2 >= r.self_bottom) & (sc >= 0.4)
    person = (lab == COCO_PERSON) & (sc >= r.min_score)
    keep = person & ~is_self & (hgt >= r.min_height)
    np.maximum.at(out["self"], fi[(person & is_self) | gear], 1.0)
    np.add.at(out["count"], fi[keep], 1.0)
    np.maximum.at(out["close"], fi[keep], np.clip(hgt[keep] / r.close_height, 0, 1))
    np.maximum.at(out["area"], fi[keep], area[keep])
    return out


def self_boxes(data: VisualData, rules: PeopleRules | None = None) -> np.ndarray:
    """Boolean per detection row: the wearer's own body/gear, or a shadow."""
    r = rules or PeopleRules()
    d = data.det
    out = (d[:, 6] >= r.self_bottom) & (d[:, 4] >= r.self_top)  # geometric fallback
    if len(data.box_idx):
        other = prompt_score(data.box_emb, OTHER_POS, SELF_POS)
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


def frame_scores(data: VisualData, models: VisualModels | None = None) -> dict[str, np.ndarray]:
    """Per-frame scores (all 0-1 except ``aesthetic``).

    ``beauty_zs``: zero-shot beauty-shot contrast, mean over the three crops;
    ``group_zs``: friends/group contrast, max over crops; ``aesthetic``: LAION
    score of the centre crop; ``steady``: cosine similarity of consecutive
    centre crops (1 = static view); plus :func:`people_frames`.
    """
    e = data.emb.astype(np.float32)
    centre = e[:, 1]
    steady = steadiness(centre)
    return {
        "beauty_zs": prompt_score(e, BEAUTY_POS, BEAUTY_NEG, models).mean(1),
        "group_zs": prompt_score(e, GROUP_POS, GROUP_NEG, models).max(1),
        "aesthetic": aesthetic(centre),
        "steady": steady,
        **people_frames(data),
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
) -> dict[str, Any]:
    """Compute the visual layer for every source that lacks it; returns stats.

    ``handoff`` is called with the loaded models before they are freed, so a
    later step can keep their CLIP model (``models.clip``) instead of loading
    it again; the detector is freed either way.
    """
    need = [s for s in sources if force or load_visual(s, params) is None]
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
        for pos, neg in ((BEAUTY_POS, BEAUTY_NEG), (GROUP_POS, GROUP_NEG), (OTHER_POS, SELF_POS)):
            text_embeddings(pos + neg, models)
        for s in need:
            if (
                compute_visual(s, models, params, force=True, avoid=(bad or {}).get(str(s.path)))
                is not None
            ):
                stats["computed"] += 1
        if handoff is not None:
            handoff(models)
    finally:
        models.close()
    return stats
