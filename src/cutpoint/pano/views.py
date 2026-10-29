"""360 proxy analysis: per-second, per-direction scores from the ``.lrv``.

One sequential keyframe decode of the 1024x512 dual-fisheye proxy (~0.8 fps:
every second keyframe, the GOP is 0.6 s) feeds the existing visual models
(:class:`cutpoint.events.visual.VisualModels`) through a GPU resampler:

* **Views for CLIP ViT-L/14**: 8 horizon-levelled perspective views (90 deg,
  yaw 0, 45, ..., 315, pitch -10) plus the downward *selfie* view at the
  wearer (``geometry.WEARER_V``, fixed in camera coordinates; the nadir for an upright
  clip), 224x224 each.
  The embeddings feed the existing zero-shot prompts (beauty, friends/group),
  the LAION aesthetic head and the crash prompts.
* **Faster R-CNN v2 on a levelled equirect** (1280x512: 1024 px for 360 deg
  plus 45 deg of wrap on each side, so people at yaw +-180 are not cut).
  Person boxes become (yaw, pitch, angular size); boxes inside the wearer cone
  (``WEARER_CONE_DEG`` around the wearer axis, in camera coordinates) are the
  wearer, the rest go through the existing own-body CLIP classifier.

Directions are in the display frame of :mod:`cutpoint.pano.geometry`
(horizon-levelled from the IMU, yaw 0 = smoothed camera forward), the same
frame the director and renderer use.

Caches per clip (keyed like every other layer, under the ``.lrv``'s cache
folder): ``pano-views-<hash>.npz/.json``. A standard visual-layer cache
(``visual-<hash>.npz``) is written too, with the left / forward / right views
as its three crops and the non-wearer people as detections, so ``cutpoint events``
skips its fisheye decode and ``cutpoint crashes`` / taste / contact sheets work
unchanged. :func:`layer_grids` gives the ski features (``beauty``, ``people``,
``group``) as the *best direction* per second, used by ``ski.layer_features``.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache, progress, state
from cutpoint.events.common import params_key
from cutpoint.pano import geometry as G

logger = logging.getLogger(__name__)

VIEWS_VERSION = 1
HORIZON_YAWS = tuple(float(y) for y in range(0, 360, 45))
SELFIE = len(HORIZON_YAWS)  # index of the selfie view
VIEW_NAMES = (*(f"yaw{int(y)}" for y in HORIZON_YAWS), "selfie")
# det columns
DET_COLS = (
    "frame",
    "label",
    "score",
    "yaw",
    "pitch",
    "width",
    "height",
    "wearer_deg",
    "vx",
    "vy",
    "vz",
)


@dataclass(frozen=True)
class ViewParams:
    """Settings of the 360 view layer (all part of the cache key)."""

    min_gap_s: float = 0.9  # keyframes closer than this are skipped (~0.8 fps)
    view_px: int = 224
    view_fov: float = 90.0
    view_pitch: float = -10.0
    selfie_fov: float = 100.0
    eq_width: int = 1024  # equirect px for 360 deg (detector input)
    eq_wrap_deg: float = 45.0
    det_thresh: float = 0.3
    box_min_score: float = 0.5
    box_max: int = 8
    batch: int = 8
    fisheye_fov: float = G.DFISHEYE_FOV

    def key(self) -> dict[str, Any]:
        """Cache-key dict (``batch`` does not change results)."""
        from cutpoint.events import visual as V  # noqa: PLC0415

        d = asdict(self)
        d.pop("batch")
        return {
            "layer": "pano-views",
            "v": VIEWS_VERSION,
            "orient": G.LEVEL_VERSION,
            "clip": [V.CLIP_ARCH, V.CLIP_PRETRAINED],
            **d,
        }


@dataclass
class PanoViews:
    """Raw per-frame outputs of one clip.

    Attributes:
        t: Frame times (s), ``(N,)``.
        emb: CLIP embeddings ``(N, 9, 768)`` float16 (8 horizon views + selfie).
        det: ``(M, 11)`` rows (see ``DET_COLS``): frame index, COCO label, score,
            box centre yaw / pitch (deg), angular width / height (deg), angle to
            the wearer axis (deg), centre ray in camera coordinates.
        box_emb: CLIP embeddings of person crops ``(K, 768)`` float16.
        box_idx: Row in ``det`` of each ``box_emb`` row.
        meta: Decode info.
        rays: Float64 centre rays of ``det`` ``(M, 3)``, the exact input of its
            ``wearer_deg``; set by :func:`compute_views`, not cached.
    """

    t: np.ndarray
    emb: np.ndarray
    det: np.ndarray
    box_emb: np.ndarray
    box_idx: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    rays: np.ndarray | None = None


def cache_path(src: Path, params: ViewParams | None = None) -> Path:
    """``cache/<lrv>/pano-views-<hash>.npz``."""
    return cache.source_dir(src) / f"pano-views-{params_key((params or ViewParams()).key())}.npz"


_STALE_AXIS: set[str] = set()  # sources whose stale-axis view layer was logged


def load_views(src: Path, params: ViewParams | None = None) -> PanoViews | None:
    """Cached view-layer outputs of a clip, or None (also when they were computed with
    another forward axis than the clip's cached orientation track has now)."""
    try:
        path = cache_path(Path(src), params)
    except OSError:
        return None
    if not path.exists():
        return None
    meta = cache.load_json(path.with_suffix(".json")) or {}
    fwd = G.cached_forward_axis(Path(src))
    if fwd is not None and meta.get("forward_axis", G.axis_name(G.FORWARD_V)) != fwd:
        if str(src) not in _STALE_AXIS:
            _STALE_AXIS.add(str(src))
            logger.info(
                "%s: view layer computed with another forward axis (upright clip); not used "
                "until `cutpoint 360 analyze` recomputes it",
                Path(src).name,
            )
        return None
    with np.load(path) as z:
        return PanoViews(
            t=z["t"],
            emb=z["emb"],
            det=z["det"],
            box_emb=z["box_emb"],
            box_idx=z["box_idx"],
            meta=meta,
        )


# --------------------------------------------------------------------------- compute
def _view_rotations(p: ViewParams) -> np.ndarray:
    return np.stack([G.view_matrix(y, p.view_pitch) for y in HORIZON_YAWS])


def _selfie_rotation(r_vw: np.ndarray, wearer_v: np.ndarray) -> np.ndarray:
    """View rotation (world <- view) looking at the wearer, horizon kept level."""
    d = r_vw.T @ wearer_v
    yaw, pitch = G.ray_yaw_pitch(d)
    return G.view_matrix(float(yaw), float(pitch))


def _wearer_deg(dv: np.ndarray, wear: np.ndarray) -> float:
    """Angle (deg) between a det centre ray and the unit wearer axis (camera coordinates)."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(dv @ wear)))))


def compute_views(
    clip: Any,
    models: Any,
    p: ViewParams,
    orient: G.Orientation,
    wearer_v: np.ndarray | None = None,
    reuse: PanoViews | None = None,
) -> PanoViews | None:
    """Run the view layer on one clip's ``.lrv`` (one sequential keyframe decode).

    Args:
        clip: :class:`cutpoint.pano.x3.X3Clip` with a proxy.
        models: Loaded :class:`cutpoint.events.visual.VisualModels`.
        p: Settings.
        orient: The clip's orientation track.
        wearer_v: Wearer axis in camera coordinates (default ``orient.wearer``).
        reuse: This clip's result for another wearer axis (same ``p``, with ``rays``):
            only what depends on the axis is recomputed (#376). The detector, the
            equirect and the person crops are skipped; their rows are kept and
            ``wearer_deg`` is recomputed from ``rays``. The 9 views are still embedded
            as one batch, as in a full run, so the selfie embeddings are byte-equal.
    """
    import torch  # noqa: PLC0415

    from cutpoint.config import remap_source  # noqa: PLC0415
    from cutpoint.events.visual import COCO_PERSON, DET_LABELS, iter_frames  # noqa: PLC0415

    src = Path(clip.lrv)
    if reuse is not None and reuse.rays is None:  # loaded from the cache: no float64 rays
        logger.debug("%s: no rays to reuse; full recompute", src.name)
        reuse = None
    dev = models.device
    sampler = G.FisheyeSampler(dev, p.fisheye_fov)
    view_rays = sampler.rays(G.flat_rays(p.view_px, p.view_px, p.view_fov, p.view_fov))
    selfie_rays = sampler.rays(G.flat_rays(p.view_px, p.view_px, p.selfie_fov, p.selfie_fov))
    eq_w = round(p.eq_width * (360 + 2 * p.eq_wrap_deg) / 360)
    eq_h = p.eq_width // 2
    # Equirect rays over 360 deg plus the wrap margin on both sides.
    lon = (np.arange(eq_w) + 0.5) / p.eq_width * 360.0 - 180.0 - p.eq_wrap_deg
    lat = ((2 * np.arange(eq_h) + 1) / eq_h - 1) * 90.0
    lo, la = np.meshgrid(np.radians(lon), np.radians(lat))
    eq_d = np.stack([np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)], -1)
    eq_rays = sampler.rays(eq_d)
    vrot = _view_rotations(p)
    det_model = models.det
    det_model.transform.min_size = (eq_h,)
    det_model.transform.max_size = eq_w + 8
    det_model.roi_heads.score_thresh = p.det_thresh

    ts: list[float] = []
    embs: list[np.ndarray] = []
    dets: list[np.ndarray] = []
    rays: list[np.ndarray] = []
    crop_rows: list[int] = []
    crop_embs: list[np.ndarray] = []
    batch_t: list[float] = []
    batch_f: list[np.ndarray] = []
    gpu_s = 0.0
    n_rows = 0
    wearer_v = orient.wearer if wearer_v is None else wearer_v
    wear = wearer_v / np.linalg.norm(wearer_v)
    person_lab = torch.tensor(DET_LABELS, device=dev)

    def flush() -> None:
        nonlocal gpu_s, n_rows
        g0 = time.perf_counter()
        b = len(batch_f)
        x = torch.from_numpy(np.stack(batch_f)).to(dev).permute(0, 3, 1, 2).float() / 255
        rv = np.stack([orient.r_vw(t) for t in batch_t])
        with torch.inference_mode():
            views = sampler.sample(x, rv, view_rays, vrot)  # (B, 8, 3, S, S)
            sel = sampler.sample(
                x, np.stack([r @ _selfie_rotation(r, wear) for r in rv]), selfie_rays
            )  # (B, 3, S, S)
            allv = torch.cat([views, sel[:, None]], 1).reshape(b * 9, 3, p.view_px, p.view_px)
            e = models._embed(allv).view(b, 9, -1)
        embs.append(e.cpu().numpy().astype(np.float16))
        if reuse is not None:  # detections and crops do not depend on the wearer axis
            ts.extend(batch_t)
            batch_t.clear()
            batch_f.clear()
            gpu_s += time.perf_counter() - g0
            return
        with torch.inference_mode():
            eq = sampler.sample(x, rv, eq_rays)  # (B, 3, H, W)
            if dev == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    outs = det_model(list(eq))
            else:
                outs = det_model(list(eq))
        crops = []
        for i, o in enumerate(outs):
            keep = torch.isin(o["labels"], person_lab)
            bx = o["boxes"][keep].float().cpu().numpy()
            lab = o["labels"][keep].cpu().numpy()
            sc = o["scores"][keep].float().cpu().numpy()
            rows: list[list[float]] = []
            n_crops = 0
            for j in np.argsort(-sc):
                x1, y1, x2, y2 = bx[j]
                cxp, cyp = (x1 + x2) / 2, (y1 + y2) / 2
                yaw = cxp / p.eq_width * 360.0 - 180.0 - p.eq_wrap_deg
                if not -180.0 <= yaw < 180.0:
                    continue  # the duplicate inside the wrap margin
                pitch = 90.0 - cyp / eq_h * 180.0
                wdeg = (x2 - x1) / p.eq_width * 360.0
                hdeg = (y2 - y1) / eq_h * 180.0
                dv = rv[i] @ G.yaw_pitch_ray(yaw, pitch)
                wa = _wearer_deg(dv, wear)
                rays.append(dv)
                rows.append(
                    [
                        len(ts) + i,
                        float(lab[j]),
                        float(sc[j]),
                        yaw,
                        pitch,
                        wdeg,
                        hdeg,
                        wa,
                        *dv.tolist(),
                    ]
                )
                if lab[j] == COCO_PERSON and sc[j] >= p.box_min_score and n_crops < p.box_max:
                    side = max(x2 - x1, y2 - y1) * 1.1
                    a0, a1 = int(max(0, cxp - side / 2)), int(min(eq_w, cxp + side / 2))
                    b0, b1 = int(max(0, cyp - side / 2)), int(min(eq_h, cyp + side / 2))
                    if a1 - a0 >= 8 and b1 - b0 >= 8:
                        crops.append(
                            torch.nn.functional.interpolate(
                                eq[i : i + 1, :, b0:b1, a0:a1].float(),
                                size=(224, 224),
                                mode="bilinear",
                                antialias=True,
                                align_corners=False,
                            )
                        )
                        crop_rows.append(n_rows + len(rows) - 1)
                        n_crops += 1
            dets.append(np.asarray(rows, np.float32).reshape(-1, len(DET_COLS)))
            n_rows += len(rows)
        if crops:
            with torch.inference_mode():
                crop_embs.append(
                    models._embed(torch.cat(crops, 0)).cpu().numpy().astype(np.float16)
                )
        ts.extend(batch_t)
        batch_t.clear()
        batch_f.clear()
        gpu_s += time.perf_counter() - g0

    t0 = time.perf_counter()
    last = -1e9
    # Decode the copy source_remap points at; caches and meta keep the original path.
    for t, frame in iter_frames(remap_source(src), 1024, 512, True, True):
        if t - last < p.min_gap_s:
            continue
        last = t
        batch_t.append(t)
        batch_f.append(frame)
        if len(batch_f) >= p.batch:
            flush()
    if batch_f:
        flush()
    if not ts:
        logger.warning("%s: 360 views decoded no frames", src.name)
        return None
    t_arr = np.asarray(ts, np.float32)
    if reuse is not None:
        if not np.array_equal(t_arr, reuse.t):
            logger.warning("%s: 360 views decoded other frames; full recompute", src.name)
            return compute_views(clip, models, p, orient, wearer_v)
        det = reuse.det.copy()
        det[:, 7] = [_wearer_deg(dv, wear) for dv in reuse.rays]
        data = replace(reuse, t=t_arr, emb=np.concatenate(embs), det=det)
        gpu_s += reuse.meta.get("gpu_s", 0.0)
        t0 -= reuse.meta.get("elapsed_s", 0.0)  # elapsed_s covers both passes
    else:
        det = np.concatenate(dets) if dets else np.zeros((0, len(DET_COLS)), np.float32)
        data = PanoViews(
            t=t_arr,
            emb=np.concatenate(embs),
            det=det,
            box_emb=(np.concatenate(crop_embs) if crop_embs else np.zeros((0, 768), np.float16)),
            box_idx=np.asarray(crop_rows, np.int64),
            rays=np.asarray(rays, np.float64).reshape(-1, 3),
        )
    elapsed = time.perf_counter() - t0
    data.meta = {
        "source": str(src),
        "clip": clip.clip_id,
        "frames": len(ts),
        "elapsed_s": round(elapsed, 1),
        "gpu_s": round(gpu_s, 1),
        "person_crops": len(data.box_idx),
        "views": list(VIEW_NAMES),
        "wearer_v": [round(float(v), 4) for v in wear],
        "params": p.key(),
    }
    if orient.forward_axis != G.axis_name(G.FORWARD_V):  # load_views checks it
        data.meta["forward_axis"] = orient.forward_axis
    return data


def save_views(src: Path, data: PanoViews, params: ViewParams) -> Path:
    """Write the pano cache and the standard visual-layer cache of one clip."""
    path = cache_path(src, params)
    np.savez_compressed(
        path, t=data.t, emb=data.emb, det=data.det, box_emb=data.box_emb, box_idx=data.box_idx
    )
    cache.save_json(path.with_suffix(".json"), data.meta)
    _SCORES.clear()  # keyed without emb or axis: drop scores of the clip's old views
    _OCCL.clear()
    write_standard_visual(src, data)
    return path


_AXES: dict[str, np.ndarray] = state.register_container(__name__, "_AXES", {})


def wearer_axis(pv: PanoViews, min_boxes: int = 15) -> np.ndarray:
    """The wearer's direction in camera coordinates for this clip.

    The camera is fixed to the wearer (helmet / backpack pole / selfie stick),
    so the wearer's own body is the most frequent large person box at a fixed
    camera direction. Median direction of confident person boxes taller than
    45 deg; the mount's default (:func:`geometry.wearer_v` of the views' forward axis)
    when there are fewer than ``min_boxes``.
    Measured: clip 030 (pole mount) -> about (0.09, 0.63, 0.69); clip 020 (4K
    mode, held in front of the face) -> a different axis, hence per clip.
    """
    key = str(pv.meta.get("source"))
    cached = _AXES.get(key)
    if cached is not None:
        return cached
    d = pv.det
    upright = pv.meta.get("forward_axis") == G.axis_name(G.LENS_V)
    axis = G.wearer_v(G.LENS_V if upright else G.FORWARD_V)
    big = (d[:, 1] == 1) & (d[:, 2] >= 0.8) & (d[:, 6] >= 45.0) if len(d) else np.zeros(0, bool)
    if big.sum() >= min_boxes:
        v = d[big, 8:11].astype(np.float64)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        m = np.median(v, 0)
        axis = m / (np.linalg.norm(m) + 1e-9)
    elif upright:  # once per source: the result is kept in _AXES
        logger.info(
            "%s: upright clip with %d large person boxes (< %d); the wearer is taken to be "
            "straight below the camera",
            Path(key).name,
            int(big.sum()),
            min_boxes,
        )
    _AXES[key] = axis
    return axis


def forward_yaw(pv: PanoViews) -> float:
    """Display yaw of the travel direction: opposite the wearer (circular median)."""
    src = pv.meta.get("source")
    o = G.load_orientation(Path(src)) if src else None
    if o is None or not len(pv.t):
        return 0.0
    wy = G.wearer_yaw(o, pv.t[:: max(1, len(pv.t) // 200)].astype(float), wearer_axis(pv))
    a = np.radians(wy + 180.0)
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))


def is_wearer(
    det: np.ndarray,
    cone_deg: float = G.WEARER_CONE_DEG,
    max_height_deg: float = 70.0,
    axis: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean per det row: the wearer (centre within the cone around ``axis``,
    default ``geometry.WEARER_V``, in camera coordinates, or a box taller than
    ``max_height_deg``).

    The height rule mislabels big close friends as the wearer; it is kept
    (default 70 deg) for the cached selection features (:func:`frame_people`,
    :func:`layer_grids`, :func:`write_standard_visual`). Director scoring and the
    geometric QA pass ``max_height_deg=inf``: the wearer is the cone only.
    """
    if not len(det):
        return np.zeros(0, bool)
    v = det[:, 8:11].astype(np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    ang = np.degrees(np.arccos(np.clip(v @ (G.WEARER_V if axis is None else axis), -1, 1)))
    return (ang <= cone_deg) | (det[:, 6] > max_height_deg)


# --------------------------------------------------------------------------- scores
@dataclass(frozen=True)
class PeopleRules360:
    """Which person detections count as *other* people, and how close they are."""

    min_score: float = 0.6
    other_min: float = 0.5  # CLIP other-vs-self share (existing own-body classifier)
    wearer_cone_deg: float = G.WEARER_CONE_DEG
    min_height_deg: float = 5.0
    close_height_deg: float = 0.55 * 90.0  # same as 0.55 of a 90-deg view
    wearer_max_height_deg: float = 70.0  # is_wearer's height rule (inf: the cone only)


def other_people(pv: PanoViews, rules: PeopleRules360 | None = None) -> np.ndarray:
    """Boolean per det row: a person who is not the wearer (cone + CLIP classifier)."""
    from cutpoint.events.visual import OTHER_POS, SELF_POS, prompt_score  # noqa: PLC0415

    r = rules or PeopleRules360()
    d = pv.det
    if not len(d):
        return np.zeros(0, bool)
    keep = (
        (d[:, 1] == 1)
        & (d[:, 2] >= r.min_score)
        & ~is_wearer(d, r.wearer_cone_deg, r.wearer_max_height_deg, axis=wearer_axis(pv))
        & (d[:, 6] >= r.min_height_deg)
    )
    if len(pv.box_idx):
        other = prompt_score(pv.box_emb, OTHER_POS, SELF_POS)
        selfish = np.zeros(len(d), bool)
        selfish[pv.box_idx] = other < r.other_min
        keep &= ~selfish
    return keep


_SCORES: dict[tuple, dict[str, np.ndarray]] = state.register_container(__name__, "_SCORES", {})
WEARER_VIEW_DEG = 60.0  # horizon views this close to the wearer's yaw see the wearer
WEARER_VIEW_DAMP = 0.25  # ... so their group / beauty prompts are scaled down


def wearer_view_mask(pv: PanoViews) -> np.ndarray | None:
    """``(N, 8)`` True where a horizon view faces the wearer (needs the cached
    orientation track), else None."""
    src = pv.meta.get("source")
    o = G.load_orientation(Path(src)) if src else None
    if o is None:
        return None
    wy = G.wearer_yaw(o, pv.t.astype(float), wearer_axis(pv))
    d = np.abs(G.wrap_deg(np.asarray(HORIZON_YAWS)[None, :] - wy[:, None]))
    return d < WEARER_VIEW_DEG


def view_scores(pv: PanoViews) -> dict[str, np.ndarray]:
    """Per-frame, per-view scores ``(N, 9)`` (cached per clip in this process).

    ``beauty`` (0-1: the existing beauty-shot score with aesthetic and
    steadiness), ``beauty_zs``, ``group_zs``, ``aesthetic``, ``steady`` and
    ``crash_zs`` (the crash prompts of ``cutpoint crashes``).
    """
    key = (pv.meta.get("source"), len(pv.t), float(pv.t[-1]) if len(pv.t) else 0.0)
    cached = _SCORES.get(key)
    if cached is not None:
        return cached
    from cutpoint.crash import CrashParams  # noqa: PLC0415
    from cutpoint.events.visual import (  # noqa: PLC0415
        BEAUTY_NEG,
        BEAUTY_POS,
        GROUP_NEG,
        GROUP_POS,
        aesthetic,
        beauty_score,
        prompt_score,
        steadiness,
    )

    e = pv.emb.astype(np.float32)
    steady = steadiness(e)
    cp = CrashParams()
    sc = {
        "beauty_zs": prompt_score(e, BEAUTY_POS, BEAUTY_NEG),
        "group_zs": prompt_score(e, GROUP_POS, GROUP_NEG),
        "aesthetic": aesthetic(e),
        "steady": steady,
        "crash_zs": prompt_score(e, cp.pos, cp.neg),
    }
    sc["beauty"] = beauty_score(sc)
    mask = wearer_view_mask(pv)
    if mask is not None:  # the wearer's helmet in a view is not "friends" / a vista
        damp = np.where(mask, WEARER_VIEW_DAMP, 1.0).astype(np.float32)
        for k in ("group_zs", "beauty"):
            sc[k][:, :SELFIE] *= damp
    _SCORES.clear()  # keep one clip
    _SCORES[key] = sc
    return sc


LIFT_POS = (
    "riding a chairlift with a view over the trees",
    "the safety bar of a chairlift",
    "people sitting on a chairlift",
    "chairlift cable and towers",
    "inside a gondola cabin",
)
LIFT_NEG = (
    "skiing down a snowy slope",
    "skiers standing on a ski run",
    "a snowy mountain landscape",
    "trees covered in snow",
    "the inside of a building",
    "a trail map sign board",
    "a ski resort information sign",
)
LIFT_MEDIAN = 5  # frames (~6 s): a lift ride is long, single-frame hits are not


def lift_scores(pv: PanoViews) -> np.ndarray:
    """Per-frame chairlift / gondola score (0-1): the best horizon view's lift
    prompts, median-filtered over :data:`LIFT_MEDIAN` frames."""
    from cutpoint.events.visual import prompt_score  # noqa: PLC0415

    z = prompt_score(pv.emb[:, :SELFIE].astype(np.float32), LIFT_POS, LIFT_NEG).max(1)
    h = LIFT_MEDIAN // 2
    if len(z) <= h:
        return z
    p = np.r_[np.full(h, z[0]), z, np.full(h, z[-1])]
    return np.median(np.lib.stride_tricks.sliding_window_view(p, LIFT_MEDIAN), 1)


def frame_people(pv: PanoViews, rules: PeopleRules360 | None = None) -> dict[str, np.ndarray]:
    """Per-frame ``count``, ``close`` (0-1, nearest other person), ``self`` (wearer seen)."""
    r = rules or PeopleRules360()
    n = len(pv.t)
    out = {k: np.zeros(n, np.float32) for k in ("count", "close", "self")}
    d = pv.det
    if not len(d):
        return out
    keep = other_people(pv, r)
    fi = d[:, 0].astype(int)
    np.add.at(out["count"], fi[keep], 1.0)
    np.maximum.at(out["close"], fi[keep], np.clip(d[keep, 6] / r.close_height_deg, 0, 1))
    np.maximum.at(
        out["self"], fi[(d[:, 1] == 1) & is_wearer(d, r.wearer_cone_deg, axis=wearer_axis(pv))], 1.0
    )
    return out


def frame_features(pv: PanoViews) -> dict[str, np.ndarray]:
    """Per-frame ski features over all directions: the best direction wins.

    ``beauty``: max over the 8 horizon views; ``group``: max over the horizon
    views (the selfie view always shows the wearer, so it is left out);
    ``people``: the existing ``friends_score`` from the nearest other person and
    the best group view; ``selfie_group``: the selfie view's group prompt.
    """
    from cutpoint.events.visual import friends_score  # noqa: PLC0415

    vs = view_scores(pv)
    pe = frame_people(pv)
    h = slice(0, SELFIE)
    group = vs["group_zs"][:, h].max(1)
    people = friends_score({"count": pe["count"], "close": pe["close"], "group_zs": group})
    return {
        "beauty": vs["beauty"][:, h].max(1),
        "group": group,
        "people": people,
        "selfie_group": vs["group_zs"][:, SELFIE],
        "count": pe["count"],
        "crash_zs": vs["crash_zs"][:, h].mean(1),
    }


def _grid_views(src: Path) -> PanoViews | None:
    """The cached view layer of a 360 source (``.lrv`` / ``.insv``) with frames, or None."""
    if not str(src).lower().endswith((".lrv", ".insv")):
        return None
    pv = load_views(Path(src))
    return pv if pv is not None and len(pv.t) else None


def layer_grids(
    src: Path, n: int, offset: float = 0.0, dt: float = 0.1
) -> dict[str, np.ndarray] | None:
    """``beauty`` / ``people`` / ``group`` on the analysis grid, or None (no 360 cache)."""
    from cutpoint.events.visual import to_grid  # noqa: PLC0415

    pv = _grid_views(src)
    if pv is None:
        return None
    ff = frame_features(pv)
    t = pv.t.astype(float) - offset
    out = {}
    for k in ("beauty", "people", "group"):
        v = ff[k]
        if len(v) >= 3:  # median of each frame and its neighbours (drops flickers)
            pad = np.r_[v[0], v, v[-1]]
            v = np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), 0)
        out[k] = to_grid(t, v, n, dt, hold=1.0)
    return out


def lift_grid(src: Path, n: int, offset: float = 0.0, dt: float = 0.1) -> np.ndarray | None:
    """:func:`lift_scores` on the analysis grid, or None (no 360 cache)."""
    from cutpoint.events.visual import to_grid  # noqa: PLC0415

    pv = _grid_views(src)
    if pv is None:
        return None
    return to_grid(pv.t.astype(float) - offset, lift_scores(pv), n, dt, hold=1.5)


FRAMING_H = (4.0, 12.0, 45.0, 60.0, 70.0)
FRAMING_U = (0.0, 1.0, 1.0, 0.0, -0.5)


def framing_utility(h_deg: np.ndarray | float) -> np.ndarray:
    """How good a person of angular height ``h_deg`` is to frame (direction scoring).

    0 up to 4 deg (a speck), rising to 1 at 12 deg, a plateau to 45 deg, 0 by
    60 deg (too close to frame well) and -0.5 above 70 deg (a face or body inches
    from the lens). Replaces the old closeness ``h / close_height_deg`` as the
    bump height, which gave a face inches away the top score.
    """
    return np.interp(np.asarray(h_deg, float), FRAMING_H, FRAMING_U)


OCCLUDE_MIN_SCORE = 0.5  # person boxes (the wearer too) this confident can occlude a bin
OCCLUDE_MIN_DEG2 = 300.0  # ... if their angular area could cover ~7 % of an 86-deg view
OCCLUDE_RAMP = 0.65  # occlude rises from this share of the QA limit to 1 at the limit
OCCL_ZS_FROM = 0.8  # the occluder prompts count above the rendered QA's occl_zs only
SELFIE_OK_POS = (
    "a selfie of a skier wearing a helmet and goggles",
    "a person's face wearing ski goggles",
    "a skier seen from above",
)
_OCCL: dict[tuple, dict[str, np.ndarray]] = state.register_container(__name__, "_OCCL", {})
_OBINS: dict[tuple, np.ndarray] = state.register_container(__name__, "_OBINS", {})


def occlusion_scores(pv: PanoViews) -> dict[str, np.ndarray]:
    """Zero-shot occluder and selfie-content scores (cached per clip in this process).

    ``occl_zs`` ``(N, 8)``: the rendered QA's occluder prompts (gloves, sleeves,
    a face inches away: :data:`cutpoint.pano.qa.OCCL_POS`) on the horizon views;
    ``selfie_ok`` ``(N,)``: :data:`SELFIE_OK_POS` against the occluders and "the
    top of a helmet" on the selfie view (a usable selfie, not a glove on the lens).
    """
    key = (pv.meta.get("source"), len(pv.t), float(pv.t[-1]) if len(pv.t) else 0.0)
    cached = _OCCL.get(key)
    if cached is not None:
        return cached
    from cutpoint.events.visual import prompt_score  # noqa: PLC0415
    from cutpoint.pano.qa import OCCL_POS, qa_negatives  # noqa: PLC0415

    e = pv.emb.astype(np.float32)
    sc = {
        "occl_zs": prompt_score(e[:, :SELFIE], OCCL_POS, qa_negatives()).astype(np.float32),
        "selfie_ok": prompt_score(
            e[:, SELFIE], SELFIE_OK_POS, (*OCCL_POS, "the top of a helmet")
        ).astype(np.float32),
    }
    _OCCL.clear()  # keep one clip
    _OCCL[key] = sc
    return sc


def occlusion_bins(
    pv: PanoViews,
    centres: np.ndarray,
    view_pitch: float = -8.0,
    view_fov: float = 86.0,
    size: tuple[int, int] = (1920, 1080),
) -> np.ndarray:
    """``(N, nb)`` 0-1: how much the biggest person box would occlude a view at each bin.

    Every confident person box (the wearer too) is projected into the view at
    the bin's yaw, ``view_pitch`` and ``view_fov`` as the geometric QA does
    (:func:`cutpoint.pano.qa_paths.project_frames`: clipped, slivers dropped).
    The score ramps to 1 at the QA's limits: ``self_area`` for the wearer's own
    body (the cone around :func:`wearer_axis`), ``close_area`` (or the full-frame
    height and width rule) for anyone else; a wearer box the QA allows
    (:func:`cutpoint.pano.qa.wearer_allowed`) scores 0. The raw equirect extent is not
    used: near the nadir it spans 100-200 deg of yaw, and seated chairlift
    friends 70 deg tall frame fine. Cached per clip and view in this process
    (every segment of a clip reuses it; one clip kept).
    """
    from cutpoint.pano import qa_paths as QP  # noqa: PLC0415
    from cutpoint.pano.qa import QAThresholds, wearer_allowed  # noqa: PLC0415

    th = QAThresholds()
    d = pv.det
    out = np.zeros((len(pv.t), len(centres)), np.float32)
    if not len(d):
        return out
    key = (
        pv.meta.get("source"),
        len(pv.t),
        len(d),
        float(np.asarray(d, np.float64).sum()),
        tuple(np.round(np.asarray(centres, float), 3)),
        view_pitch,
        view_fov,
        size,
    )
    cached = _OBINS.get(key)
    if cached is not None:
        return cached
    person, is_self = QP.person_rows(pv)
    cand = person & (d[:, 2] >= OCCLUDE_MIN_SCORE) & (d[:, 5] * d[:, 6] >= OCCLUDE_MIN_DEG2)
    for k in np.flatnonzero(cand):
        row = d[k]
        lim = th.self_area if is_self[k] else th.close_area
        dd = np.abs(G.wrap_deg(centres - row[3]))
        for b in np.flatnonzero(dd <= view_fov / 2 + row[5] / 2):
            res = G.project_box_visible(
                float(centres[b]),
                view_pitch,
                view_fov,
                size[0],
                size[1],
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
            )
            if res is None or res[1] < QP.MIN_VISIBLE:
                continue
            x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in res[0])
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            if is_self[k] and wearer_allowed((x1, y1, x2, y2), th):
                continue
            if not is_self[k] and h >= th.close_h and w >= th.close_w:
                v = 1.0
            else:
                v = float(np.clip((w * h / lim - OCCLUDE_RAMP) / (1 - OCCLUDE_RAMP), 0, 1))
            f = int(row[0])
            out[f, b] = max(out[f, b], v)
    _OBINS.clear()  # keep one clip
    _OBINS[key] = out
    return out


def direction_scores(
    pv: PanoViews,
    n_bins: int = 36,
    rules: PeopleRules360 | None = None,
    people_size: str = "framing",
    view_pitch: float = -8.0,
    view_fov: float = 86.0,
    occlude: bool = True,
) -> dict[str, np.ndarray]:
    """Per-frame scores over ``n_bins`` display-yaw bins (bin k centred on k*360/n_bins).

    ``people``: Gaussian bumps at each other person's yaw (width from the box,
    at least 10 deg), height = :func:`framing_utility` of the box height (the
    best positive bump plus the most negative one: a person inches away pulls
    the bins down); the wearer is the cone around :func:`wearer_axis` only.
    ``people_size="close"`` is the old behaviour (height = closeness, plus the
    >70 deg rule of :func:`is_wearer`; :meth:`DirectorParams.legacy`).
    ``pitch``: pitch of the strongest person per bin; ``subject`` ``(N, nb, 4)``:
    its yaw, pitch, width, height (NaN where nobody). ``group`` / ``beauty``: the
    8 view scores interpolated around the circle. ``occlude`` (0-1):
    :func:`occlusion_bins` for a view at ``view_pitch`` / ``view_fov``, max with
    the horizon views' occluder prompts above :data:`OCCL_ZS_FROM`.
    ``selfie``: the selfie view's group score; ``selfie_ok``: its usable-selfie
    score (:func:`occlusion_scores`). ``occlude=False`` leaves ``occlude`` 0
    (skips the projections when the director does not use it).
    """
    r = rules or PeopleRules360()
    legacy = people_size == "close"
    if not legacy:
        r = replace(r, wearer_max_height_deg=math.inf)
    vs = view_scores(pv)
    oc = occlusion_scores(pv)
    n = len(pv.t)
    centres = np.arange(n_bins) * 360.0 / n_bins

    def circ_interp(v8: np.ndarray) -> np.ndarray:
        xp = np.r_[np.asarray(HORIZON_YAWS), 360.0]
        return np.stack([np.interp(centres, xp, np.r_[row, row[0]]) for row in v8])

    people = np.zeros((n, n_bins), np.float32)
    worst = np.zeros((n, n_bins), np.float32)
    pitch = np.full((n, n_bins), np.nan, np.float32)
    subject = np.full((n, n_bins, 4), np.nan, np.float32)
    occl = np.zeros((n, n_bins), np.float32)
    d = pv.det
    if len(d):
        keep = other_people(pv, r)
        for row in d[keep]:
            f = int(row[0])
            if legacy:
                height = float(np.clip(row[6] / r.close_height_deg, 0, 1))
            else:
                height = float(framing_utility(row[6]))
            sig = max(10.0, float(row[5]) / 2)
            dd = G.wrap_deg(centres - row[3])
            bump = height * np.exp(-0.5 * (dd / sig) ** 2)
            if height < 0:
                worst[f] = np.minimum(worst[f], bump)
                continue
            better = bump > people[f]
            people[f] = np.where(better, bump, people[f])
            won = better & (bump > 0.05)
            pitch[f] = np.where(won, row[4], pitch[f])
            subject[f, won] = row[[3, 4, 5, 6]]
        if occlude:
            occl = occlusion_bins(pv, centres, view_pitch, view_fov)
    if n and occlude:
        zs = np.clip((oc["occl_zs"] - OCCL_ZS_FROM) / (1 - OCCL_ZS_FROM), 0, 1)
        occl = np.maximum(occl, circ_interp(zs))
    return {
        "centres": centres,
        "people": people + worst,
        "pitch": pitch,
        "subject": subject,
        "occlude": occl.astype(np.float32),
        "group": circ_interp(vs["group_zs"][:, :SELFIE]),
        "beauty": circ_interp(vs["beauty"][:, :SELFIE]),
        "selfie": vs["group_zs"][:, SELFIE].astype(np.float32),
        "selfie_ok": oc["selfie_ok"].astype(np.float32),
    }


# --------------------------------------------------------------------------- compatibility
def write_standard_visual(src: Path, pv: PanoViews) -> Path:
    """Standard ``visual-<hash>.npz`` for the ``.lrv`` built from the 360 views.

    Crops = left (yaw 270), forward (yaw 0), right (yaw 90) views. Detections =
    every non-wearer box, placed in a pseudo 90-deg view (x centred, y from the
    pitch relative to the -10 deg view pitch, sizes from the angular size), so
    the existing ``people_frames`` heights mean the same as on a GoPro frame.
    """
    from cutpoint.events import visual as V  # noqa: PLC0415

    vp = V.VisualParams()
    path = V.cache_path(src, vp)
    d = pv.det
    keep = np.flatnonzero(~is_wearer(d, axis=wearer_axis(pv))) if len(d) else np.zeros(0, int)
    std = np.zeros((len(keep), 7), np.float32)
    if len(keep):
        dk = d[keep]
        yc = 0.5 - (dk[:, 4] + 10.0) / 90.0
        std[:, 0], std[:, 1], std[:, 2] = dk[:, 0], dk[:, 1], dk[:, 2]
        std[:, 3] = 0.5 - dk[:, 5] / 180.0
        std[:, 4] = yc - dk[:, 6] / 180.0
        std[:, 5] = 0.5 + dk[:, 5] / 180.0
        std[:, 6] = yc + dk[:, 6] / 180.0
    remap = {int(old): new for new, old in enumerate(keep)}
    sel = [i for i, row in enumerate(pv.box_idx) if int(row) in remap]
    box_idx = np.asarray([remap[int(pv.box_idx[i])] for i in sel], np.int64)
    box_emb = pv.box_emb[sel] if sel else np.zeros((0, 768), np.float16)
    emb = pv.emb[:, [6, 0, 2]]
    np.savez_compressed(path, t=pv.t, emb=emb, det=std, box_emb=box_emb, box_idx=box_idx)
    cache.save_json(
        path.with_suffix(".json"),
        {
            "source": str(src),
            "frames": len(pv.t),
            "mode": "pano-views",
            "derived_from": cache_path(src).name,
            "size": [224, 224],
            "note": "360 clip: crops are the yaw 270 / 0 / 90 levelled views; detections "
            "exclude the wearer cone",
            "params": vp.key(),
        },
    )
    return path


# --------------------------------------------------------------------------- driver
def run_layer(
    clips: list[Any],
    params: ViewParams | None = None,
    force: bool = False,
    handoff: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Compute the view layer for every clip with a proxy that lacks it.

    ``handoff`` is called with the loaded models (a ``VisualModels``) before they are
    freed, so a later step can keep their CLIP model instead of loading it again (#376);
    the detector is freed either way. It is not called when every clip's views are
    cached or a clip fails.

    Returns:
        Stats: ``computed``, ``cached``, ``seconds`` and per-clip ``frames`` / ``s``.
    """
    from cutpoint.crash import CrashParams  # noqa: PLC0415
    from cutpoint.events import visual as V  # noqa: PLC0415
    from cutpoint.pano.x3 import imu_source  # noqa: PLC0415

    p = params or ViewParams()
    todo = [c for c in clips if c.lrv and (force or load_views(Path(c.lrv), p) is None)]
    stats: dict[str, Any] = {
        "cached": sum(1 for c in clips if c.lrv) - len(todo),
        "computed": 0,
        "clips": {},
    }
    if not todo:
        return stats
    t0 = time.perf_counter()
    models = V.VisualModels(
        det_min_size=512, det_thresh=p.det_thresh, box_min_score=p.box_min_score, box_max=p.box_max
    )
    try:
        cp = CrashParams()
        for pos, neg in (
            (V.BEAUTY_POS, V.BEAUTY_NEG),
            (V.GROUP_POS, V.GROUP_NEG),
            (V.OTHER_POS, V.SELF_POS),
            (cp.pos, cp.neg),
        ):
            V.text_embeddings(pos + neg, models)
        for i, c in enumerate(todo, 1):
            progress.update(progress.fraction(i - 1, len(todo)), f"clip {c.clip_id}")
            src = Path(c.lrv)
            i00 = imu_source(src)
            if i00 is None:
                logger.warning("clip %s: no _00_.insv for the IMU; skipped", c.clip_id)
                continue
            orient = G.orientation(src, i00, c.fps, max(c.frames, 1))
            c0 = time.perf_counter()
            prior = load_views(src, p)  # a forced re-run reuses the clip's wearer axis
            _AXES.pop(str(src), None)
            prior_axis = wearer_axis(prior) if prior is not None else orient.wearer
            data = compute_views(c, models, p, orient, wearer_v=prior_axis)
            if data is None:
                continue
            _AXES.pop(str(src), None)
            axis = wearer_axis(data)
            if math.degrees(math.acos(float(np.clip(axis @ prior_axis, -1, 1)))) > 25:
                logger.info(
                    "clip %s: wearer at %s in camera coordinates (not the default "
                    "mount); recomputing the selfie view",
                    c.clip_id,
                    np.round(axis, 2),
                )
                s0 = time.perf_counter()
                data = compute_views(c, models, p, orient, wearer_v=axis, reuse=data) or data
                logger.debug(
                    "clip %s: selfie view recomputed in %.1fs", c.clip_id, time.perf_counter() - s0
                )
            save_views(src, data, p)
            el = time.perf_counter() - c0
            stats["computed"] += 1
            stats["clips"][c.clip_id] = {"frames": len(data.t), "s": round(el, 1)}
            logger.info(
                "[%d/%d] clip %s: %d frames x 9 views, %d detections in %.1fs (%.0fx realtime)",
                i,
                len(todo),
                c.clip_id,
                len(data.t),
                len(data.det),
                el,
                c.duration / max(el, 1e-6),
            )
        if handoff is not None:
            handoff(models)
    finally:
        models.close()
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    return stats
