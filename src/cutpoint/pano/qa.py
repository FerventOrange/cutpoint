"""Direction QA of rendered 360 clips (``cutpoint 360 qa``).

Ground truth is the rendered picture: every clip is decoded at ``fps`` (2) and
640x360, the visual-layer models run on those frames (3-crop CLIP, Faster
R-CNN persons, person-crop CLIP) and each frame gets flags:

``occluded``
    a confident "self" person box (the wearer's glove, arm, sleeve, own body)
    covers more than ``self_area`` of the frame, unless it only comes in from
    the bottom (:func:`wearer_allowed`), or the zero-shot occluder prompts
    (:data:`OCCL_POS`, max over the 3 crops) exceed ``occl_zs``.
``too_close``
    any confident person box (but an allowed self box) covers more than
    ``close_area`` of the frame, or is (nearly) full height and wide.
``edge_cut``
    the subject (largest kept person, if at least ``size_min_h`` tall) or any
    kept person of area >= ``edge_min_area`` touches the left / right edge, or
    the top edge while taller than ``edge_top_h`` and ending above
    ``edge_top_max_y2`` (feet cut at the bottom are fine; a box spanning the
    frame height is a tight framing, reported by ``size_ok``). A box sharing
    ``edge_self_overlap`` of the smaller one's area with a self box is the wearer
    detected twice and does not count.
``low_info``
    fog / flat snow / sky: low grey contrast plus the :data:`LOWINFO_POS`
    prompts (mean over the crops), or a very confident prompt alone.
``too_far``
    on a ``friend`` target: the subject is shorter than ``too_far_h``.
``dark``
    a covered lens or a camera being handled: mean grey below ``dark_mean`` and
    ``dark_rel`` x the clip's median, with more than ``dark_share`` of the pixels
    near black (grey < ``dark_px``); or mean grey below ``dark_abs`` (covered throughout).
``tilt``
    the rendered horizon roll (true up seen through the renderer's
    stabilised view, :func:`cutpoint.pano.render.stabilised_r_vw`) exceeds
    ``tilt_deg``.

Pan metrics come from the director path (30 fps samples of
:func:`cutpoint.pano.director.evaluate`, hard cuts excluded). Per clip:
flag shares, longest runs, flagged spans, a 0-100 score and a pass/fail by
the rule in :class:`QAThresholds`; per edit the duration-weighted summary.
When ``360 director --vl-check`` ran, the bad frames its vision model found in
the clip (the path's ``vl`` record, :func:`vl_summary`) count toward the pass
too (``max_vl_bad``); a path without that record (check skipped or the model
unavailable) is judged without it.

Inputs: a clips folder (``manifest.json`` entries with ``pano.path_file`` and
``start``) or a director preview folder (``NNN_x.mp4`` + ``NNN_x.json``; the
offset is the path's ``segment.start``). Outputs go to ``<dir>_qa/``:
``qa.json``, ``flagged_NN.jpg`` (one tile per flagged span; subject box green,
self boxes red) and ``overview_NN.jpg`` (one frame per 2 s per clip).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.utils import clip_files

logger = logging.getLogger(__name__)

QA_VERSION = 1
PROBLEM_FLAGS = ("occluded", "too_close", "edge_cut", "low_info", "too_far", "tilt", "dark")
CAPTION = {
    "occluded": "OCCL",
    "too_close": "CLOSE",
    "edge_cut": "EDGE",
    "low_info": "LOW",
    "too_far": "FAR",
    "tilt": "TILT",
    "dark": "DARK",
}

OCCL_POS = (
    "a close-up of a glove",
    "a hand covering the camera lens",
    "a blurry close-up of a jacket sleeve",
    "a ski helmet filling the frame",
    "a person's face extremely close to the camera",
    "a backpack strap close-up",
)
LOWINFO_POS = (
    "a flat white foggy image",
    "an empty white snow surface",
    "a plain sky",
    "a blurry image of snow",
    "a dark blurry image",
)


def qa_negatives() -> tuple[str, ...]:
    """Negative prompts shared by :data:`OCCL_POS` and :data:`LOWINFO_POS`."""
    from cutpoint.events.visual import BEAUTY_POS, GROUP_POS  # noqa: PLC0415

    return (
        GROUP_POS
        + BEAUTY_POS
        + ("a skier on a slope", "people standing in the snow", "a chairlift")
    )


@dataclass(frozen=True)
class QAThresholds:
    """Every QA threshold (box sizes are shares of the frame, 0-1)."""

    # per frame
    person_min_score: float = 0.6  # PeopleRules.min_score
    subject_min_h: float = 0.06  # PeopleRules.min_height
    size_min_h: float = 0.17  # 0.22 at the old 72 deg FOV floor, x tan(36) / tan(43) at 86
    size_max_h: float = 0.80
    too_far_h: float = 0.12
    close_area: float = 0.62
    close_h: float = 0.95
    close_w: float = 0.65
    edge_margin: float = 0.01
    edge_min_area: float = 0.10  # other kept people this big count for edge cuts
    edge_top_h: float = 0.30
    edge_top_max_y2: float = 0.90  # top contact of a box reaching lower is tight, not cut
    # a box sharing this much of the smaller one's area with a self box is the wearer
    # detected twice (head / goggles / gloves at the side edge), not an edge cut
    edge_self_overlap: float = 0.5
    safe_x: float = 0.05
    safe_top: float = 0.03
    self_area: float = 0.20
    # the wearer coming in from the bottom only (box reaching the bottom edge, top edge
    # this low) may be any size: wider framing shows the wearer's arms and skis there
    wearer_bottom_y2: float = 0.98
    wearer_bottom_y1: float = 0.60
    occl_zs: float = 0.8
    lowinfo_std: float = 0.07
    lowinfo_zs: float = 0.6
    lowinfo_zs_alone: float = 0.8
    # display tilt (deg): the render levels only level_strength of the camera lean, so
    # (1 - level_strength) of it shows by design. The horizon check predicts roll with the
    # render's own stabilisation model, so it only measures that intended tilt and cannot
    # catch render bugs (GitHub issue #34): an absolute cap on what is shown
    tilt_deg: float = 12.0
    # a hand over the lens: grey mean 0.04 with 86 % of the pixels below 0.1 (its edge
    # frames 0.19 / 61 %, clip median 0.44); the darkest clean frame of a ski day, on a
    # dim lift ride (median 0.29), is 0.266 / 48 %: absolute limits alone are too close
    dark_mean: float = 0.22
    dark_px: float = 0.10
    dark_share: float = 0.5
    dark_rel: float = 0.6  # ... and this far below the clip's median mean grey
    dark_abs: float = 0.08  # dark on its own (a clip covered throughout)
    # score weights
    w_occluded: float = 2.0
    w_too_close: float = 2.0
    w_edge_cut: float = 1.5
    w_low_info: float = 1.0
    w_too_far: float = 0.5
    # clip pass rule
    max_occl_close: float = 0.03
    max_occl_close_run_s: float = 1.0  # fail at a run >= this
    max_edge_cut: float = 0.08
    max_edge_run_s: float = 1.0  # fail at a run > this
    min_subject_present: float = 0.80
    max_low_info: float = 0.05
    max_pan_p95: float = 35.0
    max_pan_max: float = 55.0
    min_shot_s: float = 2.0
    max_horizon_p95: float = 12.0  # display tilt, see tilt_deg; any dark frame fails a clip too
    max_vl_bad: int = 1  # bad VL check frames (blocked / low info) a clip may have


def dark_frames(gray: np.ndarray, th: QAThresholds) -> np.ndarray:
    """Per frame of a clip's ``gray`` ``(n, h, w)`` (0-1): a covered lens / near-black
    picture, judged against the clip's own brightness."""
    g = gray.reshape(len(gray), -1)
    m = g.mean(1)
    rel = (m < th.dark_mean) & ((g < th.dark_px).mean(1) > th.dark_share)
    return (rel & (m < th.dark_rel * np.median(m))) | (m < th.dark_abs)


# --------------------------------------------------------------------------- per frame
def wearer_allowed(box: Any, th: QAThresholds) -> Any:
    """True where the wearer's in-frame box ``(x1, y1, x2, y2)`` (clipped, 0-1; arrays
    broadcast) is fine to show: at most ``self_area`` of the frame, or coming in from the
    bottom only (``y2 >= wearer_bottom_y2`` and ``y1 >= wearer_bottom_y1``).

    Shared by the director's safe-area pass, the geometric QA and :func:`frame_flags`.
    """
    x1, y1, x2, y2 = (np.asarray(v, float) for v in box)
    area = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    bottom = (y2 >= th.wearer_bottom_y2) & (y1 >= th.wearer_bottom_y1)
    return (area <= th.self_area) | bottom


def _box_rows(det: np.ndarray, i: int) -> np.ndarray:
    return det[det[:, 0].astype(int) == i] if len(det) else np.zeros((0, 7))


def _overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection of each box in ``a`` ``(N, 4)`` with each in ``b`` ``(M, 4)`` as a
    share of the smaller one's area, ``(N, M)``.

    The smaller, not ``a``'s own: the wearer's duplicate box is the larger one (a head
    and shoulders box around the self-tagged lower half, e.g. 0.21 x 0.95 around
    0.20 x 0.48), which a share of ``a`` misses. The cost: a small self box inside a
    friend's box at the edge hides that friend's edge cut too.
    """
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(hi - lo, 0.0, None).prod(2)
    aa, ab = ((x[:, 2:] - x[:, :2]).prod(1) for x in (a, b))
    return inter / np.maximum(np.minimum(aa[:, None], ab[None, :]), 1e-9)


def frame_flags(
    det: np.ndarray,
    is_self: np.ndarray,
    n: int,
    *,
    gray_std: np.ndarray,
    occl_zs: np.ndarray,
    lowinfo_zs: np.ndarray,
    targets: list[str],
    roll: np.ndarray | None,
    th: QAThresholds,
    person_label: int = 1,
    dark: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Per-frame QA records.

    Args:
        det: Detections ``(M, 7)`` (frame, label, score, x1, y1, x2, y2), normalised.
        is_self: Per detection row: the wearer's own body / gear (``self_boxes``).
        n: Number of frames.
        gray_std, occl_zs, lowinfo_zs: Per-frame grey std (0-1) and prompt scores.
        targets: Director target per frame (``friend`` / ``selfie`` / ...).
        roll: Rendered horizon roll (deg) per frame, or None.
        th: Thresholds.
        dark: Per frame: a near-black picture (:func:`dark_frames`), or None.

    Returns:
        ``{"flags", "subj", "self", "boxes", "size_ok", "safe", "occl_zs",
        "lowinfo_zs", "gray_std", "roll"}`` per frame (``subj`` = ``[x1, y1, x2, y2]``
        or None; ``boxes`` = every person detection ``[score, x1, y1, x2, y2, self]``).
    """
    out = []
    for i in range(n):
        rows = _box_rows(det, i)
        sel = is_self[det[:, 0].astype(int) == i] if len(det) else np.zeros(0, bool)
        person = (rows[:, 1].astype(int) == person_label) & (rows[:, 2] >= th.person_min_score)
        w = rows[:, 5] - rows[:, 3]
        h = rows[:, 6] - rows[:, 4]
        area = w * h
        kept = person & ~sel & (h >= th.subject_min_h)
        selfb = person & sel
        ok_self = selfb & wearer_allowed(rows[:, 3:7].T, th)  # not occluded / too close
        flags: list[str] = []
        subj = None
        if kept.any():
            k = int(np.flatnonzero(kept)[np.argmax(area[kept])])
            subj = [round(float(v), 4) for v in rows[k, 3:7]]
        # occluded: own body / gear filling the frame, or occluder prompts
        if (selfb & ~ok_self).any() or occl_zs[i] > th.occl_zs:
            flags.append("occluded")
        close = (area > th.close_area) | ((h >= th.close_h) & (w >= th.close_w))
        if (person & ~ok_self & close).any():
            flags.append("too_close")
        cand = kept & (area >= th.edge_min_area)
        if subj is not None and h[k] >= th.size_min_h:
            cand[k] = True
        if cand.any() and selfb.any():  # the wearer detected twice is not a cut
            cand &= _overlap(rows[:, 3:7], rows[selfb, 3:7]).max(1) < th.edge_self_overlap
        if cand.any():
            x1, y1, x2, y2 = (rows[cand, c] for c in (3, 4, 5, 6))
            m = th.edge_margin
            top = (y1 <= m) & (h[cand] > th.edge_top_h) & (y2 < th.edge_top_max_y2)
            hit = (x1 <= m) | (x2 >= 1 - m) | top
            if hit.any():
                flags.append("edge_cut")
        lz = float(lowinfo_zs[i])
        if (gray_std[i] < th.lowinfo_std and lz > th.lowinfo_zs) or lz > th.lowinfo_zs_alone:
            flags.append("low_info")
        size_ok = safe = None
        if subj is not None:
            sh = subj[3] - subj[1]
            size_ok = th.size_min_h <= sh <= th.size_max_h
            safe = subj[0] >= th.safe_x and subj[2] <= 1 - th.safe_x and subj[1] >= th.safe_top
        if targets[i] == "friend" and subj is not None and subj[3] - subj[1] < th.too_far_h:
            flags.append("too_far")
        r = None if roll is None else float(roll[i])
        if r is not None and abs(r) > th.tilt_deg:
            flags.append("tilt")
        if dark is not None and dark[i]:
            flags.append("dark")
        out.append(
            {
                "flags": flags,
                "subj": subj,
                "self": [[round(float(v), 4) for v in rows[j, 3:7]] for j in np.flatnonzero(selfb)],
                "boxes": [
                    [*(round(float(v), 4) for v in rows[j, 2:7]), bool(sel[j])]
                    for j in np.flatnonzero(rows[:, 1].astype(int) == person_label)
                ],
                "size_ok": size_ok,
                "safe": safe,
                "occl_zs": round(float(occl_zs[i]), 3),
                "lowinfo_zs": round(lz, 3),
                "gray_std": round(float(gray_std[i]), 4),
                "roll": None if r is None else round(r, 2),
            }
        )
    return out


# --------------------------------------------------------------------------- path metrics
def pan_metrics(
    path: dict[str, Any], start: float, dur: float, fps: float = 30.0
) -> dict[str, Any]:
    """Pan speed / acceleration / jerk / FOV rate and cuts of ``[start, start + dur]``.

    Speed is the angular rate of the view direction (yaw scaled by cos pitch,
    plus pitch) at ``fps`` samples, with differences across hard cuts dropped;
    acceleration and jerk use the yaw / pitch after a 5-sample moving average
    (key interpolation can step the velocity at keys, which makes raw second
    differences impulsive).
    """
    from cutpoint.pano.director import evaluate, shot_index  # noqa: PLC0415

    times = start + np.arange(0.0, max(dur, 1.0 / fps), 1.0 / fps)
    yaw, pitch, fov = evaluate(path, times)
    seg = shot_index(path, times)

    def rate(a: np.ndarray, b: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
        da = np.diff(a) * (1.0 if scale is None else scale[1:])
        return np.hypot(da, np.diff(b)) * fps

    same = seg[1:] == seg[:-1]
    cosp = np.cos(np.radians(pitch))
    speed = rate(yaw, pitch, cosp)[same]
    fov_rate = np.abs(np.diff(fov))[same] * fps
    acc_v, jerk_v = [], []
    for s in np.unique(seg):
        m = seg == s
        if m.sum() < 8:
            continue
        k = np.ones(5) / 5
        ys = np.convolve(np.pad(yaw[m], 2, mode="edge"), k, "valid")
        ps = np.convolve(np.pad(pitch[m], 2, mode="edge"), k, "valid")
        v = np.stack([np.diff(ys) * cosp[m][1:], np.diff(ps)], 1) * fps
        a = np.diff(v, axis=0) * fps
        acc_v.append(np.linalg.norm(a, axis=1))
        jerk_v.append(np.linalg.norm(np.diff(a, axis=0) * fps, axis=1))
    acc = np.concatenate(acc_v) if acc_v else np.zeros(0)
    jerk = np.concatenate(jerk_v) if jerk_v else np.zeros(0)
    cuts = [float(c) for c in path.get("cuts", []) if start < c < start + dur]
    bounds = [start, *cuts, start + dur]
    shots = np.diff(bounds)

    def p(x: np.ndarray, q: float) -> float:
        return round(float(np.percentile(x, q)), 1) if len(x) else 0.0

    return {
        "speed_p95": p(speed, 95),
        "speed_max": round(float(speed.max()), 1) if len(speed) else 0.0,
        "accel_p95": p(acc, 95),
        "jerk_p95": p(jerk, 95),
        "fov_rate_p95": p(fov_rate, 95),
        "cuts": len(cuts),
        "cut_times": [round(c - start, 2) for c in cuts],
        "min_shot_s": round(float(shots.min()), 2),
        "cuts_per_min": round(len(cuts) / max(dur / 60.0, 1e-6), 2),
    }


def nearest_keys(keys: list[dict[str, Any]], times: Any) -> list[dict[str, Any]]:
    """The nearest path key per (source) time (ties: the earlier key)."""
    kt = np.array([k["t"] for k in keys], float)
    idx = np.abs(kt[None, :] - np.asarray(times, float)[:, None]).argmin(1)
    return [keys[int(i)] for i in idx]


def targets_at(path: dict[str, Any] | None, times: np.ndarray) -> list[str]:
    """Director target of the nearest path key per (source) time."""
    if path is None or not path.get("keys"):
        return ["unknown"] * len(times)
    return [k["target"] for k in nearest_keys(path["keys"], times)]


def horizon_roll(orient: Any, path: dict[str, Any], times: np.ndarray) -> np.ndarray:
    """Roll (deg) of the true horizon in the rendered view at increasing source ``times``.

    The renderer's rotation is ``R = R_VW @ view`` (:func:`render.stabilised_r_vw`
    with the ``pano`` config's ``level_strength`` / ``tilt_follow_s``); the true
    up seen through it shows the tilt left in: ``u = R.T @ up`` in output
    coordinates, roll = atan2(u_x, -u_y).
    """
    from cutpoint.pano import geometry as G  # noqa: PLC0415
    from cutpoint.pano.director import evaluate  # noqa: PLC0415
    from cutpoint.pano.project import pano_config  # noqa: PLC0415
    from cutpoint.pano.render import stabilised_r_vw  # noqa: PLC0415

    cfg = pano_config()
    fps = float(orient.meta.get("fps") or 30000 / 1001)
    yaw, pitch, _ = evaluate(path, times)
    idx = np.clip(np.round(times * fps).astype(int), 0, len(orient.q) - 1)
    r_vw = stabilised_r_vw(
        orient,
        idx,
        float(cfg.get("level_strength", 0.7)),
        float(cfg.get("tilt_follow_s", 0.5)),
    )
    up = np.einsum("kji,j->ki", orient.cam_to_world(idx), [0.0, -1.0, 0.0])
    out = np.zeros(len(times))
    for i in range(len(idx)):
        r = r_vw[i] @ G.view_matrix(float(yaw[i]), float(pitch[i]))
        u = r.T @ up[i]
        out[i] = math.degrees(math.atan2(u[0], -u[1]))
    return out


# --------------------------------------------------------------------------- aggregation
def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` index ranges where ``mask`` is True."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def aggregate_clip(
    frames: list[dict[str, Any]],
    fps: float,
    pan: dict[str, Any] | None,
    horizon: dict[str, float] | None,
    th: QAThresholds,
    vl: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shares, longest runs (s), flagged spans, score and pass for one clip.

    ``frames`` need ``t``, ``target`` and the :func:`frame_flags` fields. Shares
    of the problem flags are over all frames; ``subject_present`` /
    ``size_ok`` over ``friend`` frames; ``safe_area`` over frames with a subject.
    Each frame stands for ``1 / fps`` seconds. ``vl`` is the clip's
    :func:`vl_summary` (None: no VL check, no say in the pass).
    """
    n = len(frames)
    has = {f: np.array([f in fr["flags"] for fr in frames], bool) for f in PROBLEM_FLAGS}
    has["occl_close"] = has["occluded"] | has["too_close"]
    share = {f: round(float(v.mean()), 3) if n else 0.0 for f, v in has.items()}
    runs = {
        f: round(max((b - a for a, b in runs_of(v)), default=0) / fps, 2) for f, v in has.items()
    }
    friend = [fr for fr in frames if fr["target"] == "friend"]
    subj = [fr for fr in frames if fr["subj"] is not None]
    share["subject_present"] = (
        round(sum(fr["subj"] is not None for fr in friend) / len(friend), 3) if friend else None
    )
    share["size_ok"] = (
        round(sum(bool(fr["size_ok"]) for fr in friend) / len(friend), 3) if friend else None
    )
    share["safe_area"] = (
        round(sum(bool(fr["safe"]) for fr in subj) / len(subj), 3) if subj else None
    )
    cdev = [abs((fr["subj"][0] + fr["subj"][2]) / 2 - 0.5) for fr in subj]
    centre_dev = round(float(np.percentile(cdev, 90)), 3) if cdev else None
    spans = []
    for f in PROBLEM_FLAGS:
        for a, b in runs_of(has[f]):
            spans.append(
                {"what": f, "t0": frames[a]["t"], "t1": round(frames[b - 1]["t"] + 1 / fps, 2)}
            )
    spans.sort(key=lambda s: (s["t0"], s["what"]))
    penalty = (
        th.w_occluded * share["occluded"]
        + th.w_too_close * share["too_close"]
        + th.w_edge_cut * share["edge_cut"]
        + th.w_low_info * share["low_info"]
        + th.w_too_far * share["too_far"]
    )
    score = round(float(np.clip(100 * (1 - penalty), 0, 100)), 1)
    fails = []
    count = {f: int(v.sum()) for f, v in has.items()}
    # a single flagged frame is noise, not a share (short clips: 1 frame > 3-5 %)
    if share["occl_close"] > th.max_occl_close and count["occl_close"] > 1:
        fails.append(f"occluded+too_close {share['occl_close']:.0%}")
    if runs["occl_close"] >= th.max_occl_close_run_s:
        fails.append(f"occluded/close run {runs['occl_close']:.1f}s")
    if share["edge_cut"] > th.max_edge_cut:
        fails.append(f"edge_cut {share['edge_cut']:.0%}")
    if runs["edge_cut"] > th.max_edge_run_s:
        fails.append(f"edge_cut run {runs['edge_cut']:.1f}s")
    sp = share["subject_present"]
    if sp is not None and sp < th.min_subject_present:
        fails.append(f"subject present {sp:.0%}")
    if share["low_info"] > th.max_low_info and count["low_info"] > 1:
        fails.append(f"low_info {share['low_info']:.0%}")
    if pan is not None:
        if pan["speed_p95"] > th.max_pan_p95 or pan["speed_max"] > th.max_pan_max:
            fails.append(f"pan p95 {pan['speed_p95']:.0f} max {pan['speed_max']:.0f} deg/s")
        if pan["min_shot_s"] < th.min_shot_s:
            fails.append(f"shot {pan['min_shot_s']:.1f}s")
    if count["dark"]:
        fails.append(f"dark {count['dark'] / fps:.1f}s")
    if horizon is not None and horizon["p95"] > th.max_horizon_p95:
        fails.append(f"horizon p95 {horizon['p95']:.1f} deg")
    # like the shares above, a single bad frame is noise (the check samples few frames)
    if vl is not None and vl["bad"] > th.max_vl_bad:
        fails.append(f"vl {vl['bad']}/{vl['frames']} frames bad")
    return {
        "share": share,
        "runs": runs,
        "centre_dev_p90": centre_dev,
        "score": score,
        "pass": not fails,
        "fails": fails,
        "flags": spans,
    }


def aggregate_edit(clips: list[dict[str, Any]]) -> dict[str, Any]:
    """Duration-weighted edit summary of per-clip results."""
    dur = np.array([c["duration"] for c in clips], float)
    total = float(dur.sum())
    if not clips or total <= 0:
        return {"clips": len(clips), "duration": 0.0}

    def wmean(key: str) -> float | None:
        vals = [(c["share"].get(key), d) for c, d in zip(clips, dur, strict=True)]
        vals = [(v, d) for v, d in vals if v is not None]
        if not vals:
            return None
        return round(sum(v * d for v, d in vals) / sum(d for _, d in vals), 3)

    pans = [c["pan"] for c in clips if c.get("pan")]
    hor = [c["horizon"] for c in clips if c.get("horizon")]
    vls = [c["vl"] for c in clips if c.get("vl")]
    return {
        "clips": len(clips),
        "passed": sum(bool(c["pass"]) for c in clips),
        "pass_share": round(sum(bool(c["pass"]) for c in clips) / len(clips), 3),
        "duration": round(total, 2),
        "score": round(float(np.dot([c["score"] for c in clips], dur) / total), 1),
        "share": {
            k: wmean(k)
            for k in (*PROBLEM_FLAGS, "occl_close", "subject_present", "size_ok", "safe_area")
        },
        "longest_occl_close_run_s": max(c["runs"]["occl_close"] for c in clips),
        "pan_p95_max": max((p["speed_p95"] for p in pans), default=None),
        "pan_max": max((p["speed_max"] for p in pans), default=None),
        "jerk_p95_max": max((p["jerk_p95"] for p in pans), default=None),
        "min_shot_s": min((p["min_shot_s"] for p in pans), default=None),
        "horizon_p95_max": max((h["p95"] for h in hor), default=None),
        "horizon_max": max((h["max"] for h in hor), default=None),
        "vl_checked": len(vls),
        "vl_bad": sum(v["bad"] for v in vls),
    }


# --------------------------------------------------------------------------- inputs
def load_path(ref: str | None) -> dict[str, Any] | None:
    if not ref:
        return None
    from cutpoint.config import remap_source  # noqa: PLC0415

    for p in (Path(ref), remap_source(ref)):
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def imu_levelled(pano: dict[str, Any]) -> bool:
    """True when the renderer levelled the clip with our IMU track (dual fisheye, or
    an SDK stitch whose sidecar says no FlowState), so the horizon can be checked."""
    from cutpoint.pano.project import equirect_flowstate  # noqa: PLC0415

    mode = pano.get("mode")
    if mode == "dfisheye":
        return True
    inputs = pano.get("inputs") or []
    return mode == "sdk" and bool(inputs) and equirect_flowstate(Path(inputs[0])) is False


def find_inputs(qa_dir: Path) -> list[dict[str, Any]]:
    """Rendered clips of a clips folder or director preview folder.

    Returns:
        ``{"file", "video", "offset", "kind", "path", "path_missing", "mode",
        "imu_levelled"}`` per clip (``path`` is the director path dict
        or None; ``path_missing`` when the manifest names a path file that is gone).
    """
    files = clip_files(qa_dir)
    manifest = cache.load_json(qa_dir / "manifest.json")
    out = []
    if manifest is not None:
        by_file = {e["file"]: e for e in manifest.get("clips", [])}
        for f in files:
            e = by_file.get(f.name)
            if e is None:
                continue
            pano = e.get("pano") or {}
            path = load_path(pano.get("path_file"))
            missing = bool(pano.get("path_file")) and path is None
            if missing:
                logger.warning(
                    "%s: director path %s not found; no targets, pans or horizon",
                    f.name,
                    pano["path_file"],
                )
            out.append(
                {
                    "file": f.name,
                    "video": f,
                    "offset": float(e["start"]),
                    "kind": e.get("kind"),
                    "path": path,
                    "path_missing": missing,
                    "mode": pano.get("mode"),
                    "imu_levelled": imu_levelled(pano),
                }
            )
        return out
    for f in files:
        js = f.with_suffix(".json")
        if not js.is_file():
            continue
        path = json.loads(js.read_text(encoding="utf-8"))
        out.append(
            {
                "file": f.name,
                "video": f,
                "offset": float(path["segment"]["start"]),
                "kind": path.get("kind"),
                "path": path,
                "path_missing": False,
                "mode": "dfisheye",  # director previews render the .lrv proxy
                "imu_levelled": True,
            }
        )
    return out


# --------------------------------------------------------------------------- run
def decode(video: Path, fps: float, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """``(times, frames)`` of ``video`` at ``fps``, scaled to ``size`` (RGB uint8).

    ``round=up``: the frame at each time (the default rounding shows a later one)."""
    from cutpoint.ffmpeg_utils import decode_raw  # noqa: PLC0415

    w, h = size
    ts, frames = [], []
    for t, fr in decode_raw(
        video,
        size=size,
        vf=f"fps={fps:g}:round=up,scale={w}:{h}:flags=area,format=rgb24",
        vf_cuda=f"fps={fps:g}:round=up,scale_cuda={w}:{h},hwdownload,format=nv12,format=rgb24",
        pts=True,
    ):
        ts.append(t)
        frames.append(fr)
    return np.array(ts, float), np.stack(frames) if frames else np.zeros((0, h, w, 3), np.uint8)


def horizon(item: dict[str, Any], src_times: np.ndarray) -> np.ndarray | None:
    """Per-time rendered roll, or None (not IMU-levelled, no path / cached IMU)."""
    from cutpoint.config import remap_source  # noqa: PLC0415
    from cutpoint.pano.geometry import load_orientation  # noqa: PLC0415

    path = item["path"]
    if path is None or not item.get("imu_levelled") or not path.get("source"):
        return None
    orient = load_orientation(remap_source(path["source"]))
    if orient is None:
        return None
    return horizon_roll(orient, path, src_times)


def qa_clip(
    item: dict[str, Any], models: Any, fps: float, size: tuple[int, int], th: QAThresholds
) -> dict[str, Any] | None:
    """Decode, run the models and evaluate one rendered clip."""
    from cutpoint.events import visual as V  # noqa: PLC0415

    ts, frames = decode(item["video"], fps, size)
    if not len(ts):
        logger.warning("%s: no frames decoded", item["file"])
        return None
    data = V.visual_data_from_frames(models, ts, frames)
    gray = frames.astype(np.float32).mean(3) / 255.0
    gray_std = gray.reshape(len(gray), -1).std(1)
    dark = dark_frames(gray, th)
    del frames, gray
    neg = qa_negatives()
    emb = data.emb.astype(np.float32)
    occl = V.prompt_score(emb, OCCL_POS, neg, models).max(1)
    low = V.prompt_score(emb, LOWINFO_POS, neg, models).mean(1)
    is_self = V.self_boxes(data)
    src_t = item["offset"] + ts
    roll_frames = horizon(item, src_t)
    targets = targets_at(item["path"], src_t)
    recs = frame_flags(
        data.det,
        is_self,
        len(ts),
        gray_std=gray_std,
        occl_zs=occl,
        lowinfo_zs=low,
        targets=targets,
        roll=roll_frames,
        th=th,
        dark=dark,
    )
    for t, tg, r in zip(ts, targets, recs, strict=True):
        r["t"] = round(float(t), 2)
        r["target"] = tg
    from cutpoint.ffmpeg_utils import probe  # noqa: PLC0415

    dur = float(probe(item["video"]).duration)
    pan = pan_metrics(item["path"], item["offset"], dur) if item["path"] else None
    hor = None
    if item["path"] is not None:
        hor = horizon_summary(horizon(item, item["offset"] + np.arange(0.0, dur, 1 / 30.0)))
    try:
        beauty = round(float(V.beauty_score(V.frame_scores(data, models)).mean()), 3)
    except FileNotFoundError:  # no aesthetic head installed
        beauty = None
    return clip_record(item, dur, recs, fps, pan, hor, th, beauty)


def horizon_summary(roll: np.ndarray | None) -> dict[str, float] | None:
    """``{"p95", "max"}`` of the absolute rendered roll, or None."""
    if roll is None or not len(roll):
        return None
    a = np.abs(roll)
    return {"p95": round(float(np.percentile(a, 95)), 2), "max": round(float(a.max()), 2)}


FRAME_KEYS = (
    "t",
    "target",
    "flags",
    "subj",
    "self",
    "boxes",
    "occl_zs",
    "lowinfo_zs",
    "gray_std",
    "roll",
)


def vl_summary(path: dict[str, Any] | None, offset: float, dur: float) -> dict[str, Any] | None:
    """The director's VL check of a clip's span ``[offset, offset + dur]`` (source s).

    Returns:
        ``{"model", "frames", "bad", "flags"}`` from the path's ``vl`` record, ``bad``
        and ``flags`` limited to the span (``frames`` counts the whole checked segment),
        or None when the check did not run or no frame was answered.
    """
    vl = (path or {}).get("vl")
    if not isinstance(vl, dict) or not vl.get("frames"):
        return None
    flags = [f for f in vl.get("flags", []) if offset <= float(f["t"]) <= offset + dur]
    return {"model": vl.get("model"), "frames": vl["frames"], "bad": len(flags), "flags": flags}


def clip_record(
    item: dict[str, Any],
    dur: float,
    recs: list[dict[str, Any]],
    fps: float,
    pan: dict[str, Any] | None,
    hor: dict[str, float] | None,
    th: QAThresholds,
    beauty: float | None = None,
) -> dict[str, Any]:
    """The ``qa.json`` record of one clip (rendered or geometric QA)."""
    vl = vl_summary(item.get("path"), float(item["offset"]), dur)
    agg = aggregate_clip(recs, fps, pan, hor, th, vl)
    return {
        "file": item["file"],
        "kind": item["kind"],
        "path_missing": bool(item.get("path_missing")),
        "duration": round(dur, 2),
        "share": agg["share"],
        "runs": agg["runs"],
        "pan": pan,
        "horizon": hor,
        "centre_dev_p90": agg["centre_dev_p90"],
        "beauty_mean": beauty,
        "vl": vl,
        "score": agg["score"],
        "pass": agg["pass"],
        "fails": agg["fails"],
        "flags": agg["flags"],
        "frames": [{k: r[k] for k in FRAME_KEYS} for r in recs],
    }


def _sheet_item(video: Path, stem: str, fr: dict[str, Any], what: str) -> dict[str, Any]:
    boxes = [(*b, False) for b in fr.get("self", [])]
    if fr["subj"] is not None:
        boxes.append((*fr["subj"], True))
    return {
        "path": str(video),
        "t": fr["t"],
        "caption": f"{stem[:16]} {fr['t']:.1f}s {what} {fr['target']}",
        "boxes": boxes,
    }


def write_sheets(qa_dir: Path, out_dir: Path, clips: list[dict[str, Any]]) -> list[Path]:
    """``flagged_NN.jpg`` (one tile per flagged span) and ``overview_NN.jpg``."""
    from cutpoint.events.sheets import contact_sheet  # noqa: PLC0415

    flagged, overview = [], []
    for c in clips:
        video = qa_dir / c["file"]
        stem = Path(c["file"]).stem
        by_t = {fr["t"]: fr for fr in c["frames"]}
        seen: set[float] = set()
        for s in c["flags"]:
            inside = [fr for fr in c["frames"] if s["t0"] <= fr["t"] < s["t1"]]
            fr = inside[len(inside) // 2] if inside else by_t.get(s["t0"])
            if fr is None or fr["t"] in seen:
                continue
            seen.add(fr["t"])
            what = "/".join(CAPTION[f] for f in fr["flags"])
            flagged.append(_sheet_item(video, stem, fr, what))
        step = max(1, round(2.0 / max(1e-6, _frame_dt(c["frames"]))))
        for fr in c["frames"][::step]:
            what = "/".join(CAPTION[f] for f in fr["flags"]) or "ok"
            overview.append(_sheet_item(video, stem, fr, what))
    written = []
    for name, items, cols, width, per in (
        ("flagged", flagged, 4, 480, 24),
        ("overview", overview, 8, 240, 64),
    ):
        for k in range(0, len(items), per):
            p = contact_sheet(
                items[k : k + per], out_dir / f"{name}_{k // per + 1:02d}.jpg", cols, width
            )
            if p is not None:
                written.append(p)
    return written


def _clear_sheets(out_dir: Path) -> None:
    """Delete the sheets of an earlier run (also with ``--json-only``: none go stale)."""
    for old in [*out_dir.glob("flagged_*.jpg"), *out_dir.glob("overview_*.jpg")]:
        old.unlink()


def _frame_dt(frames: list[dict[str, Any]]) -> float:
    if len(frames) < 2:
        return 0.5
    return float(np.median(np.diff([fr["t"] for fr in frames])))


def run_qa(
    qa_dir: Path,
    fps: float = 2.0,
    size: tuple[int, int] = (640, 360),
    json_only: bool = False,
    th: QAThresholds | None = None,
) -> dict[str, Any]:
    """Rendered-frame QA of every clip in ``qa_dir``; writes ``<qa_dir>_qa/``."""
    from cutpoint.events.visual import VisualModels  # noqa: PLC0415
    from cutpoint.ffmpeg_utils import FFmpegError, output_dir  # noqa: PLC0415

    # normalise "." / ".." (no name for <dir>_qa) without turning mapped drives into UNC
    qa_dir = Path(os.path.abspath(qa_dir))  # noqa: PTH100
    th = th or QAThresholds()
    items = find_inputs(qa_dir)
    if not items:
        raise FileNotFoundError(f"{qa_dir}: no rendered clips (manifest.json or NNN_x.mp4+.json)")
    out_dir = qa_dir.with_name(qa_dir.name + "_qa")
    with output_dir(out_dir):  # a failed run leaves no empty folder behind
        _clear_sheets(out_dir)
        t0 = time.perf_counter()
        models = VisualModels()
        clips, skipped = [], []
        try:
            for item in items:
                c0 = time.perf_counter()
                try:
                    res = qa_clip(item, models, fps, size, th)
                except (FFmpegError, OSError, KeyError, ValueError) as err:
                    logger.warning("%s: skipped (%s)", item["file"], err)
                    skipped.append({"file": item["file"], "error": str(err)[:300]})
                    continue
                if res is None:
                    skipped.append({"file": item["file"], "error": "no frames decoded"})
                    continue
                clips.append(res)
                logger.debug("%s: %.1fs", item["file"], time.perf_counter() - c0)
        finally:
            models.close()
        report = {
            "version": QA_VERSION,
            "source_dir": str(qa_dir),
            "mode": "rendered",
            "fps": fps,
            "size": list(size),
            "thresholds": asdict(th),
            "clips": clips,
            "skipped": skipped,
            "edit": aggregate_edit(clips),
        }
        report["edit"]["elapsed_s"] = round(time.perf_counter() - t0, 1)
        cache.save_json(out_dir / "qa.json", report, indent=1)
    report["sheets"] = [] if json_only else [str(p) for p in write_sheets(qa_dir, out_dir, clips)]
    report["out_dir"] = str(out_dir)
    return report


def log_report(report: dict[str, Any]) -> None:
    """Per-clip summary table plus the edit summary."""
    logger.info(
        "%-22s %6s %5s %4s %5s %5s %5s %5s %5s %5s %5s %5s %5s  %s",
        "clip",
        "dur",
        "score",
        "pass",
        "occl",
        "close",
        "edge",
        "low",
        "subj",
        "pan95",
        "shot",
        "hor95",
        "vl",
        "fails",
    )

    def pct(v: float | None) -> str:
        return "-" if v is None else f"{100 * v:.0f}%"

    for c in report["clips"]:
        s, pan, hor, vl = c["share"], c["pan"] or {}, c["horizon"] or {}, c.get("vl")
        logger.info(
            "%-22s %5.1fs %5.1f %4s %5s %5s %5s %5s %5s %5s %5s %5s %5s  %s",
            c["file"][:22],
            c["duration"],
            c["score"],
            "ok" if c["pass"] else "FAIL",
            pct(s["occluded"]),
            pct(s["too_close"]),
            pct(s["edge_cut"]),
            pct(s["low_info"]),
            pct(s["subject_present"]),
            f"{pan['speed_p95']:.0f}" if pan else "-",
            f"{pan['min_shot_s']:.1f}" if pan else "-",
            f"{hor['p95']:.1f}" if hor else "-",
            f"{vl['bad']}/{vl['frames']}" if vl else "-",
            "; ".join(c["fails"]),
        )
    e = report["edit"]
    sh = e.get("share", {})
    logger.info(
        "edit: %d clips, %.1fs, score %.1f, %d/%d pass; occl %s close %s edge %s low %s "
        "subject %s; longest occl/close run %.1fs; pan p95<=%s max %s; min shot %s; "
        "horizon p95<=%s (%.0fs)",
        e["clips"],
        e["duration"],
        e.get("score", 0.0),
        e.get("passed", 0),
        e["clips"],
        pct(sh.get("occluded")),
        pct(sh.get("too_close")),
        pct(sh.get("edge_cut")),
        pct(sh.get("low_info")),
        pct(sh.get("subject_present")),
        e.get("longest_occl_close_run_s", 0.0),
        e.get("pan_p95_max"),
        e.get("pan_max"),
        e.get("min_shot_s"),
        e.get("horizon_p95_max"),
        e.get("elapsed_s", 0.0),
    )
    if e.get("vl_checked"):
        logger.info(
            "vl check: %d bad frame(s) in %d checked clip(s); a clip fails above %d",
            e["vl_bad"],
            e["vl_checked"],
            report["thresholds"]["max_vl_bad"],
        )
    logger.info("wrote %s", Path(report["out_dir"]) / "qa.json")
    for p in report.get("sheets", []):
        logger.info("wrote %s", p)
