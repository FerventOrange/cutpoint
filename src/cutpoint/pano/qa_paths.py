"""Geometric direction QA from director paths alone (``cutpoint 360 qa <dir> --paths``).

No rendering and no models: the view layer's person detections (equirect
boxes in the display frame, :mod:`cutpoint.pano.views`) are projected into
the path's virtual camera (:func:`cutpoint.pano.geometry.project_box_visible`) at the
rendered QA's rate (2 fps: the path at each time, the detections of the
nearest view frame, ~0.8 fps), clipped to the frame, and go through the rendered QA's
:func:`cutpoint.pano.qa.frame_flags` and :func:`~cutpoint.pano.qa.aggregate_clip`:
same flags, score, pass rule and ``qa.json`` schema (``mode: "paths"``).

The wearer's own body is a *self* box: a detection centred inside the wearer
cone around the clip's wearer axis (:func:`views.wearer_axis`); projected into
the view it drives ``occluded`` like the rendered QA's self boxes. Unlike
:func:`views.other_people`, the >70 deg height rule and the own-body CLIP
classifier are not used: on 16 rendered clips both labelled close friends
filling the view as the wearer (``occluded`` on 42 % of the frames of a clean
clip). Projected boxes mostly outside the frame (``MIN_VISIBLE``) are dropped
and overlapping ones suppressed (``NMS_IOU``), as the detector on a rendered
frame would; a dropped box that still covers ``edge_min_area`` at the left /
right edge (a close person's head or shoulder) flags ``edge_cut``
(:func:`sliver_edge_frames`), the wearer's too unless it is allowed in frame
(:func:`qa.wearer_allowed`: small, or coming in from the bottom only). What
needs pixels is not checked:
``low_info`` and the occluder prompts (gloves, sleeves) are never flagged.
Pans and the horizon come from the path and the IMU as in the rendered QA.

The director uses :func:`path_quality` for the ``quality`` block of every path.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.ffmpeg_utils import output_dir
from cutpoint.pano import geometry as G
from cutpoint.pano import qa as Q
from cutpoint.utils import natural_key

logger = logging.getLogger(__name__)

OUT_SIZE = (1920, 1080)  # only the aspect matters
HOLD_S = 1.0  # resampled times farther than this from a view frame have no detections
MIN_VISIBLE = 0.25  # boxes mostly outside the frame (a sliver of the wearer) are not seen
# (clipped box area over its projected extent, in frame units: geometry.project_box_visible)
NMS_IOU = 0.5  # torchvision Faster R-CNN's box_nms_thresh


def person_rows(pv: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(person, is_self)`` per det row of the view layer (self: centre in the wearer cone)."""
    from cutpoint.pano.views import is_wearer, wearer_axis  # noqa: PLC0415

    d = pv.det
    if not len(d):
        return np.zeros(0, bool), np.zeros(0, bool)
    person = d[:, 1].astype(int) == 1
    wear = is_wearer(d, G.WEARER_CONE_DEG, max_height_deg=np.inf, axis=wearer_axis(pv))
    return person, person & wear


def view_frame_index(pv_t: np.ndarray, times: np.ndarray, hold: float = HOLD_S) -> np.ndarray:
    """Nearest view frame per time (-1 when farther than ``hold``)."""
    from cutpoint.events.common import nearest_index  # noqa: PLC0415

    if not len(pv_t):
        return np.full(len(times), -1)
    pick = nearest_index(pv_t, times)
    return np.where(np.abs(pv_t[pick] - times) <= hold, pick, -1)


def project_frames(
    pv: Any,
    frame_idx: np.ndarray,
    yaw: np.ndarray,
    pitch: np.ndarray,
    fov: np.ndarray,
    size: tuple[int, int] = OUT_SIZE,
    rows: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Person boxes of view frames ``frame_idx`` projected into the path's views.

    Returns:
        ``(det (M, 7), is_self (M,), slivers (K, 8))`` in :func:`qa.frame_flags`
        layout (output frame, label 1, score, x1, y1, x2, y2 clipped to 0-1); boxes
        behind the camera or outside the frame are dropped; ``slivers`` are the
        boxes mostly outside it (``MIN_VISIBLE``), left out of ``det``, plus an
        ``is_self`` column (1: the wearer).
    """
    person, is_self = rows if rows is not None else person_rows(pv)
    d = pv.det
    out: list[list[float]] = []
    sel: list[bool] = []
    slivers: list[list[float]] = []
    if len(d):
        fi = d[:, 0].astype(int)
        for i, j in enumerate(frame_idx):
            if j < 0:
                continue
            for k in np.flatnonzero(person & (fi == j)):
                res = G.project_box_visible(
                    float(yaw[i]),
                    float(pitch[i]),
                    float(fov[i]),
                    size[0],
                    size[1],
                    float(d[k, 3]),
                    float(d[k, 4]),
                    float(d[k, 5]),
                    float(d[k, 6]),
                )
                if res is None:
                    continue
                x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in res[0])
                if x2 <= x1 or y2 <= y1:
                    continue
                row = [i, 1, float(d[k, 2]), x1, y1, x2, y2]
                if res[1] < MIN_VISIBLE:
                    slivers.append([*row, float(is_self[k])])
                    continue
                out.append(row)
                sel.append(bool(is_self[k]))
    det, is_self_out = np.asarray(out, float).reshape(-1, 7), np.asarray(sel, bool)
    keep = _nms(det)
    return det[keep], is_self_out[keep], np.asarray(slivers, float).reshape(-1, 8)


def sliver_edge_frames(slivers: np.ndarray, th: Q.QAThresholds) -> set[int]:
    """Frames where a person mostly outside the view still covers ``edge_min_area`` at
    its left / right edge (the wearer too, unless :func:`qa.wearer_allowed`).

    A person inches from the lens has a huge equirect box: at the frame edge only a
    head or shoulder of it is in view, a small share of the box (dropped by
    ``MIN_VISIBLE``) but a large part of the frame, and the detector on the rendered
    frame finds it there (an edge cut). Wearer slivers count too, unlike the wearer
    boxes of :func:`qa.frame_flags`: at the frame edge the rendered QA's own-body
    classifier does not tell the wearer's helmet or arm apart and flags them (on the
    36-segment eval set, leaving them out lowered the per-clip agreement with the
    rendered edge_cut from r 0.65 to 0.55). A wearer sliver that is allowed in frame
    (``slivers`` column 7, :func:`project_frames`) does not count.
    """
    if not len(slivers):
        return set()
    x1, y1, x2, y2 = (slivers[:, c] for c in (3, 4, 5, 6))
    m = th.edge_margin
    hit = (slivers[:, 2] >= th.person_min_score) & ((x2 - x1) * (y2 - y1) >= th.edge_min_area)
    hit &= (x1 <= m) | (x2 >= 1 - m)
    hit &= ~((slivers[:, 7] > 0) & Q.wearer_allowed((x1, y1, x2, y2), th))
    return {int(i) for i in slivers[hit, 0]}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _nms(det: np.ndarray, iou: float = NMS_IOU) -> np.ndarray:
    """Rows kept by per-frame non-maximum suppression (as the detector on a rendered frame)."""
    keep = np.ones(len(det), bool)
    order = np.lexsort((-det[:, 2], det[:, 0])) if len(det) else np.zeros(0, int)
    for a_i, a in enumerate(order):
        if not keep[a]:
            continue
        for b in order[a_i + 1 :]:
            if det[b, 0] != det[a, 0]:
                break
            if keep[b] and _iou(det[a, 3:7], det[b, 3:7]) > iou:
                keep[b] = False
    return keep


def geo_frames(
    pv: Any,
    path: dict[str, Any],
    src_times: np.ndarray,
    frame_idx: np.ndarray,
    th: Q.QAThresholds,
    size: tuple[int, int] = OUT_SIZE,
    roll: np.ndarray | None = None,
    rows: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """:func:`qa.frame_flags` records for the path's view at ``src_times``."""
    from cutpoint.pano.director import evaluate  # noqa: PLC0415

    n = len(src_times)
    yaw, pitch, fov = evaluate(path, src_times)
    det, is_self, slivers = project_frames(pv, frame_idx, yaw, pitch, fov, size, rows)
    recs = Q.frame_flags(
        det,
        is_self,
        n,
        gray_std=np.ones(n),
        occl_zs=np.zeros(n),
        lowinfo_zs=np.zeros(n),
        targets=Q.targets_at(path, src_times),
        roll=roll,
        th=th,
    )
    for i in sliver_edge_frames(slivers, th):
        fl = recs[i]["flags"]
        if "edge_cut" not in fl:  # in frame_flags' order: after occluded / too_close
            fl.insert(sum(f in ("occluded", "too_close") for f in fl), "edge_cut")
    for r in recs:  # not measured without pixels
        r["occl_zs"] = r["lowinfo_zs"] = r["gray_std"] = None
    return recs


def sample_times(
    pv_t: np.ndarray, offset: float, dur: float, fps: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(source times, nearest view frame per time)`` every ``1 / fps`` over ``[offset, offset + dur)``.

    Sampling the path at the rendered QA's rate (rather than at the ~1.2 s view
    frames) keeps a single view frame from counting as a 1.2 s run.
    """
    times = offset + np.arange(0.0, dur, 1.0 / fps)
    return times, view_frame_index(pv_t.astype(float), times)


def path_quality(
    path: dict[str, Any],
    pv: Any,
    start: float,
    end: float,
    th: Q.QAThresholds | None = None,
    fps: float = 2.0,
) -> dict[str, Any]:
    """The path's ``quality`` block: geometric QA shares of ``[start, end]``."""
    th = th or Q.QAThresholds()
    size = (OUT_SIZE[1], OUT_SIZE[0]) if path.get("vertical") else OUT_SIZE
    dur = max(end - start, 0.0)
    times, idx = sample_times(pv.t, start, dur, fps)
    recs = geo_frames(pv, path, times, idx, th, size)
    for t, r in zip(times, recs, strict=True):
        r["t"] = round(float(t - start), 2)
    for r, tg in zip(recs, Q.targets_at(path, times), strict=True):
        r["target"] = tg
    pan = Q.pan_metrics(path, start, dur) if dur > 0 else None
    agg = Q.aggregate_clip(recs, fps, pan, None, th)
    s = agg["share"]
    keys = ("occluded", "too_close", "edge_cut", "too_far", "subject_present", "size_ok")
    return {
        "mode": "paths",
        "frames": len(recs),
        **{k: s[k] for k in (*keys, "safe_area")},
        "pan_p95": pan["speed_p95"] if pan else None,
        "score": agg["score"],
        "pass": agg["pass"],
        "fails": agg["fails"],
    }


# --------------------------------------------------------------------------- inputs
def find_path_inputs(qa_dir: Path) -> list[dict[str, Any]]:
    """Director paths of a clips folder (``manifest.json``), a director preview
    folder or a ``pano_paths/<name>`` folder (``*.json`` paths, video optional).

    Returns:
        :func:`qa.find_inputs`-like items plus ``duration`` (no ``video``).
    """
    manifest = cache.load_json(qa_dir / "manifest.json")
    out = []
    if manifest is not None:
        for e in manifest.get("clips", []):
            pano = e.get("pano") or {}
            path = Q.load_path(pano.get("path_file"))
            missing = bool(pano.get("path_file")) and path is None
            if path is None:
                logger.warning("%s: no director path; skipped", e.get("file"))
            out.append(
                {
                    "file": e["file"],
                    "offset": float(e["start"]),
                    "duration": float(e.get("duration") or e["end"] - e["start"]),
                    "kind": e.get("kind"),
                    "path": path,
                    "path_missing": missing,
                    "mode": pano.get("mode"),
                    "imu_levelled": Q.imu_levelled(pano),
                }
            )
        return out
    for js in sorted(qa_dir.glob("*.json"), key=natural_key):
        try:
            path = cache.read_json(js)
        except (OSError, ValueError):
            continue
        if not isinstance(path, dict) or "keys" not in path or "segment" not in path:
            continue
        seg = path["segment"]
        video = js.with_suffix(".mp4")
        out.append(
            {
                "file": video.name if video.is_file() else js.name,
                "offset": float(seg["start"]),
                "duration": float(seg["end"]) - float(seg["start"]),
                "kind": path.get("kind"),
                "path": path,
                "path_missing": False,
                "mode": "dfisheye",  # previews render the .lrv proxy
                "imu_levelled": True,
            }
        )
    return out


# --------------------------------------------------------------------------- run
def qa_path_clip(item: dict[str, Any], pv: Any, fps: float, th: Q.QAThresholds) -> dict[str, Any]:
    """Geometric QA of one clip (``qa.json`` clip record, no pixel fields)."""
    path, off, dur = item["path"], item["offset"], item["duration"]
    size = (OUT_SIZE[1], OUT_SIZE[0]) if path.get("vertical") else OUT_SIZE
    times, idx = sample_times(pv.t, off, dur, fps)
    roll = Q.horizon(item, times) if len(times) else None
    recs = geo_frames(pv, path, times, idx, th, size, roll)
    for t, tg, r in zip(times, Q.targets_at(path, times), recs, strict=True):
        r["t"] = round(float(t - off), 2)
        r["target"] = tg
    pan = Q.pan_metrics(path, off, dur)
    hor = Q.horizon_summary(Q.horizon(item, off + np.arange(0.0, dur, 1 / 30.0)))
    return Q.clip_record(item, dur, recs, fps, pan, hor, th)


def run_paths_qa(
    qa_dir: Path, fps: float = 2.0, th: Q.QAThresholds | None = None
) -> dict[str, Any]:
    """Geometric QA of every director path in ``qa_dir``; writes ``<qa_dir>_paths_qa/qa.json``."""
    from cutpoint.config import remap_source  # noqa: PLC0415
    from cutpoint.pano.views import load_views  # noqa: PLC0415

    qa_dir = Path(os.path.abspath(qa_dir))  # noqa: PTH100
    th = th or Q.QAThresholds()
    items = find_path_inputs(qa_dir)
    if not items:
        raise FileNotFoundError(f"{qa_dir}: no director paths (manifest.json or *.json paths)")
    out_dir = qa_dir.with_name(qa_dir.name + "_paths_qa")
    with output_dir(out_dir):  # a failed run leaves no empty folder behind
        t0 = time.perf_counter()
        clips, skipped = [], []
        views: dict[str, Any] = {}
        for item in items:
            path = item["path"]
            if path is None:
                skipped.append({"file": item["file"], "error": "no director path"})
                continue
            src = str(path.get("source") or "")
            if src not in views:
                views[src] = load_views(remap_source(src)) if src else None
            pv = views[src]
            if pv is None or not len(pv.t):
                logger.warning("%s: no 360 view cache for %s; skipped", item["file"], src)
                skipped.append({"file": item["file"], "error": "no 360 view cache"})
                continue
            try:
                clips.append(qa_path_clip(item, pv, fps, th))
            except (OSError, KeyError, ValueError) as err:
                logger.warning("%s: skipped (%s)", item["file"], err)
                skipped.append({"file": item["file"], "error": str(err)[:300]})
        report = {
            "version": Q.QA_VERSION,
            "source_dir": str(qa_dir),
            "mode": "paths",
            "fps": fps,
            "size": list(OUT_SIZE),
            "thresholds": asdict(th),
            "clips": clips,
            "skipped": skipped,
            "edit": Q.aggregate_edit(clips),
        }
        report["edit"]["elapsed_s"] = round(time.perf_counter() - t0, 2)
        cache.save_json(out_dir / "qa.json", report, indent=1)
    report["sheets"] = []
    report["out_dir"] = str(out_dir)
    return report
