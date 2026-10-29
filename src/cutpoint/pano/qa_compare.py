"""Where the rendered and the geometric direction QA disagree on edge cuts
(``cutpoint 360 qa <dir> --compare RENDERED PATHS``).

Joins a rendered ``qa.json`` (``360 qa <dir>``) and a geometric one (``360 qa <dir>
--paths``) per clip (by file stem) and per frame time (nearest within half a frame
period), counts the frames each one flags ``edge_cut`` and attributes every frame
only the rendered QA flags to one cause, checked in this order:

``unpaired``
    the geometric QA has no frame at that time (other ``--fps``, or a shorter clip);
``wearer_overlap``
    a cut box of the rendered frame shares ``edge_self_overlap`` of the smaller
    box with a self box the geometric QA projected there, also one it dropped as
    mostly outside the view (the wearer, not tagged as self in the rendered frame);
``yaw_offset``
    the projected box matching the cut box touches the edge once shifted by the
    clip's median horizontal offset between matched rendered and projected boxes
    (``median_dx``, a share of the frame width; a constant yaw mismatch such as
    ``pano.equirect_yaw_offset``);
``stale_view``
    the geometric QA's boxes come from a view frame more than
    :data:`STALE_VIEW_S` away (or none within its hold), so they show the people
    where they were then;
``no_projected_box``
    no projected box overlaps the cut box (the proxy detector missed the person);
``other``
    a projected box is there, from a recent view frame, but the geometric QA does
    not count it as cut; ``other_detail`` says why (:data:`OTHER_DETAILS`, the
    first that applies, :func:`other_detail_of`):

    ``paths_top_edge``
        the cut box touches the top and the projected box does too, but is not
        taller than ``edge_top_h`` or reaches ``edge_top_max_y2`` (or is a large
        sliver at the top: the sliver rule counts the side edges only);
    ``paths_inside_margin``
        the projected box misses the edge margin by less than :data:`NEAR_EDGE`;
    ``position_mismatch``
        it lies farther inside, the cut box has :data:`LARGER_AREA` times its area or
        more, and it is a sliver of about the same width (within
        :data:`SLIVER_WIDTH_TOL`) whose centre is more than :data:`POSITION_DX` of the
        frame width away (cut at the bottom of the view, so smaller, but the same
        person shifted sideways);
    ``rendered_box_larger``
        it lies farther inside and the cut box has :data:`LARGER_AREA` times its
        area or more (a detection or size mismatch);
    ``paths_inside``
        it lies farther inside at a similar size (a position mismatch);
    ``paths_below_size``
        it touches the edge but is under ``subject_min_h``, or it is the paths
        subject but under ``size_min_h``, or the cut box counted as the rendered
        subject and it is under ``size_min_h``;
    ``paths_self``
        it shares ``edge_self_overlap`` with a paths self box (the wearer detected
        twice, :func:`~cutpoint.pano.qa.edge_candidates`), the cut box does not;
    ``paths_below_area``
        it is not the paths subject and under ``edge_min_area``, while the cut box
        has ``edge_min_area`` or more (counted by its area);
    ``paths_not_subject``
        the cut box counted as the rendered subject, but in the paths frame another
        box is the largest;
    ``unexplained``
        the paths rule would count it (an inconsistent pair of reports).

A cut box is a box of the rendered frame that :func:`cutpoint.pano.qa.frame_flags`
counts for its edge cut (:func:`~cutpoint.pano.qa.edge_candidates` touching an edge).
Projected boxes include those the geometric QA dropped as mostly outside the view
(``slivers`` of its frames). Each rendered-only frame records ``evidence``: the cut box
and the edge it touches, the projected box it matches and that box's distance to the
edge (:func:`explain`), and ``in_selected``: whether its source time lies in the moment
picked (the path's ``selected`` span, ``cutpoint clips``' pick before tails and padding;
else its ``segment``, as for director previews and older clip paths; None when the
reports do not say). Tails are planned and checked like the moment, so a frame outside
it still gets its cause; ``outside_selected`` counts those frames.
Writes ``edge_gap.json`` next to the rendered ``qa.json``; its edit summary also counts
the frames by the rendered clips' render ``mode`` and ``self_method`` (``by_mode``,
``by_self_method``; ``unknown`` for older reports). Per clip and for the edit,
``size_ratio`` is the median rendered over projected box width and height of the matched
non-self pairs (:func:`size_ratios`): how much larger the detector sees people than the
director's projection.
"""

from __future__ import annotations

import logging
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.pano import qa as Q

logger = logging.getLogger(__name__)

GAP_VERSION = 4  # 2: other_detail and evidence; 3: position_mismatch, by_mode, by_self_method;
# 4: size_ratio, in_selected and outside_selected instead of the outside_segment cause
GAP_FILE = "edge_gap.json"
CAUSES = (
    "unpaired",
    "wearer_overlap",
    "yaw_offset",
    "stale_view",
    "no_projected_box",
    "other",
)
OTHER_DETAILS = (
    "paths_top_edge",
    "paths_inside_margin",
    "position_mismatch",
    "rendered_box_larger",
    "paths_inside",
    "paths_below_size",
    "paths_self",
    "paths_below_area",
    "paths_not_subject",
    "unexplained",
)
COUNT_KEYS = ("frames", "rendered_edge", "paths_edge", "both", "rendered_only", "paths_only")
STALE_VIEW_S = 0.5  # view frames are ~1.2 s apart: half of that is a stale position
MATCH_IOU = 0.3  # rendered / projected box pairs for the median offset
MIN_PAIRS = 5  # fewer matched pairs: no median offset
MIN_DX = 0.01  # a smaller median offset explains nothing
NEAR_EDGE = 0.03  # a projected box this close to the cut edge just missed the margin
LARGER_AREA = 2.0  # a rendered box this many times the projected one's area: a size mismatch
SLIVER_WIDTH_TOL = 0.25  # a sliver this close in width (share of the wider) is the same person
POSITION_DX = 0.05  # ... and a centre this far off (share of the frame width) a position mismatch
BREAKDOWN_KEYS = ("frames", "rendered_edge", "paths_edge", "rendered_only")


def load_report(path: Path) -> dict[str, Any]:
    """A ``360 qa`` ``qa.json``.

    Raises:
        FileNotFoundError: If ``path`` is not a file.
        ValueError: If it is not a QA report.
    """
    if not path.is_file():
        raise FileNotFoundError(f"QA report not found: {path}")
    rep = cache.read_json(path)
    if not isinstance(rep, dict) or "clips" not in rep or "mode" not in rep:
        raise ValueError(f"{path}: not a 360 qa report (qa.json)")
    return rep


def thresholds(report: dict[str, Any]) -> Q.QAThresholds:
    """The report's thresholds (keys this version does not know are left out)."""
    known = {f.name for f in fields(Q.QAThresholds)}
    return Q.QAThresholds(
        **{k: v for k, v in (report.get("thresholds") or {}).items() if k in known}
    )


def _boxes(fr: dict[str, Any], th: Q.QAThresholds, own: bool) -> np.ndarray:
    """``(N, 4)`` confident person boxes of a frame record: self ones (``own``) or not."""
    rows = [
        b[1:5] for b in fr.get("boxes", []) if b[0] >= th.person_min_score and bool(b[5]) is own
    ]
    if own:
        rows += [b for b in fr.get("self", []) if b not in rows]
    return np.asarray(rows, float).reshape(-1, 4)


def _slivers(fr: dict[str, Any], own: bool) -> np.ndarray:
    """``(N, 4)`` boxes the geometric QA dropped as mostly outside the view (``slivers``
    of a frame record): the wearer's (``own``) or the others'."""
    rows = [b[:4] for b in fr.get("slivers", []) if bool(b[4]) is own]
    return np.asarray(rows, float).reshape(-1, 4)


def cut_boxes(fr: dict[str, Any], th: Q.QAThresholds) -> np.ndarray:
    """``(N, 4)`` boxes of a rendered frame that make it an edge cut: the candidates and
    edge rule of :func:`qa.frame_flags` (:func:`qa.edge_candidates`, :func:`qa.edge_hit`)."""
    boxes = fr.get("boxes", [])
    rows = np.asarray([[0, 1, *b[:5]] for b in boxes], float).reshape(-1, 7)
    sel = np.asarray([bool(b[5]) for b in boxes], bool)
    return rows[Q.edge_candidates(rows, sel, th) & Q.edge_hit(rows, th), 3:7]


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(hi - lo, 0.0, None).prod(2)
    aa, ab = ((x[:, 2:] - x[:, :2]).prod(1) for x in (a, b))
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def join_frames(
    rendered: list[dict[str, Any]], paths: list[dict[str, Any]], fps: float
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    """``(rendered, paths)`` frame pairs by nearest ``t`` (within half a frame period);
    frames of either side without a partner are paired with None."""
    tol = 0.5 / max(fps, 1e-6) + 1e-6
    pt = np.array([f["t"] for f in paths], float)
    used: set[int] = set()
    out: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
    for fr in rendered:
        j = int(np.abs(pt - fr["t"]).argmin()) if len(pt) else -1
        if j >= 0 and j not in used and abs(pt[j] - fr["t"]) <= tol:
            used.add(j)
            out.append((fr, paths[j]))
        else:
            out.append((fr, None))
    out += [(None, p) for j, p in enumerate(paths) if j not in used]
    return out


def median_dx(
    pairs: list[tuple[dict[str, Any] | None, dict[str, Any] | None]], th: Q.QAThresholds
) -> float | None:
    """Median horizontal offset (rendered minus projected box centre, share of the frame
    width) of non-self boxes matched by IoU >= :data:`MATCH_IOU`; None below
    :data:`MIN_PAIRS` pairs."""
    dx: list[float] = []
    for r, p in pairs:
        if r is None or p is None:
            continue
        a, b = _boxes(r, th, own=False), _boxes(p, th, own=False)
        if not len(a) or not len(b):
            continue
        iou = _iou(a, b)
        for i in range(len(a)):
            j = int(iou[i].argmax())
            if iou[i, j] >= MATCH_IOU:
                dx.append(float((a[i, 0] + a[i, 2]) / 2 - (b[j, 0] + b[j, 2]) / 2))
    return round(float(np.median(dx)), 4) if len(dx) >= MIN_PAIRS else None


def size_ratios(
    pairs: list[tuple[dict[str, Any] | None, dict[str, Any] | None]],
    th: Q.QAThresholds,
    dx: float | None,
) -> list[tuple[float, float]]:
    """``(width, height)`` ratios, rendered over projected, of the non-self boxes matched
    by IoU >= :data:`MATCH_IOU` once the projected ones are shifted by ``dx``
    (:func:`median_dx`; none when None)."""
    out: list[tuple[float, float]] = []
    shift = np.array([dx or 0.0, 0.0, dx or 0.0, 0.0])
    for r, p in pairs:
        if r is None or p is None:
            continue
        a, b = _boxes(r, th, own=False), _boxes(p, th, own=False) + shift
        if not len(a) or not len(b):
            continue
        iou = _iou(a, b)
        for i in range(len(a)):
            j = int(iou[i].argmax())
            wb, hb = b[j, 2] - b[j, 0], b[j, 3] - b[j, 1]
            if iou[i, j] >= MATCH_IOU and wb > 0 and hb > 0:
                out.append((float((a[i, 2] - a[i, 0]) / wb), float((a[i, 3] - a[i, 1]) / hb)))
    return out


def size_ratio(ratios: list[tuple[float, float]]) -> dict[str, Any]:
    """``{"width", "height", "pairs"}``: the median ratios (None below :data:`MIN_PAIRS`
    pairs) and the number of pairs."""
    if len(ratios) < MIN_PAIRS:
        return {"width": None, "height": None, "pairs": len(ratios)}
    w, h = np.median(np.asarray(ratios, float), 0)
    return {"width": round(float(w), 3), "height": round(float(h), 3), "pairs": len(ratios)}


def _area(b: Any) -> float:
    return float(max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0))


def edge_of(box: Any, th: Q.QAThresholds) -> str | None:
    """The edge a box touches by :func:`qa.edge_hit`'s rule (``left``, ``right`` or
    ``top``, first in that order), or None."""
    m = th.edge_margin
    if box[0] <= m:
        return "left"
    if box[2] >= 1 - m:
        return "right"
    if box[1] <= m and box[3] - box[1] > th.edge_top_h and box[3] < th.edge_top_max_y2:
        return "top"
    return None


def edge_gap(box: Any, edge: str | None) -> float | None:
    """Distance of ``box`` to ``edge`` (a share of the frame width, or height for the top)."""
    if edge is None:
        return None
    return round(float({"left": box[0], "right": 1 - box[2], "top": box[1]}[edge]), 4)


def same_width_offset(cut: Any, match: Any) -> bool:
    """Whether ``match`` is as wide as ``cut`` (within :data:`SLIVER_WIDTH_TOL` of the
    wider) with its centre more than :data:`POSITION_DX` of the frame width away."""
    wc, wm = float(cut[2] - cut[0]), float(match[2] - match[0])
    dx = abs(float(cut[0] + cut[2]) - float(match[0] + match[2])) / 2
    return abs(wc - wm) <= SLIVER_WIDTH_TOL * max(wc, wm, 1e-9) and dx > POSITION_DX


def other_detail_of(
    cut: Any, match: Any, sliver: bool, p: dict[str, Any], th: Q.QAThresholds
) -> str:
    """Why the paths QA did not count ``match`` (a projected box overlapping the rendered
    cut box ``cut``; ``sliver``: one it dropped as mostly outside the view) as an edge
    cut: one of :data:`OTHER_DETAILS`, the first that applies.

    Replays :func:`qa.edge_hit` and :func:`qa.edge_candidates` (boxes) or
    :func:`qa_paths.sliver_edge_frames` and :func:`qa_paths.sliver_subject_frames`
    (slivers) on the paths frame ``p``."""
    m = th.edge_margin
    edge = edge_of(cut, th)
    x1, y1, x2, y2 = (float(v) for v in match)
    h, area = y2 - y1, _area(match)
    if edge_of(match, th) is None:
        if edge == "top" and y1 <= m:  # at the top, but too short or reaching too low
            return "paths_top_edge"
        gap = edge_gap(match, edge)
        if gap is not None and gap < NEAR_EDGE:
            return "paths_inside_margin"
        if _area(cut) >= LARGER_AREA * max(area, 1e-9):
            if sliver and same_width_offset(cut, match):
                return "position_mismatch"
            return "rendered_box_larger"
        return "paths_inside"
    if h < th.subject_min_h:
        return "paths_below_size"
    if sliver and area >= th.edge_min_area and (x1 <= m or x2 >= 1 - m):
        return "unexplained"  # sliver_edge_frames counts it, a self box or not
    own = _boxes(p, th, own=True)
    if len(own) and Q._overlap(np.asarray([match], float), own).max() >= th.edge_self_overlap:
        return "paths_self"  # edge_candidates: the wearer detected twice
    kept = _boxes(p, th, own=False)
    kept = kept[(kept[:, 3] - kept[:, 1]) >= th.subject_min_h]
    kept_max = max((_area(b) for b in kept), default=0.0)
    if sliver:
        sl = _slivers(p, own=False)
        sl = sl[(sl[:, 3] - sl[:, 1]) >= th.subject_min_h]
        subj = area >= max(_area(b) for b in sl) - 1e-9 and area > kept_max
        side = x1 <= m or x2 >= 1 - m
        counted = (area >= th.edge_min_area and side) or (subj and h >= th.size_min_h)
    else:
        subj = area >= kept_max - 1e-9
        counted = area >= th.edge_min_area or (subj and h >= th.size_min_h)
    if counted:
        return "unexplained"  # the paths rule counts it: an inconsistent report
    if sliver and area >= th.edge_min_area:
        return "paths_top_edge"  # a large sliver at the top: only the side edges count
    if subj and h < th.size_min_h:  # the paths subject, too small to count
        return "paths_below_size"
    if _area(cut) >= th.edge_min_area:  # the rendered QA counted it by its area
        return "paths_below_area"
    if h < th.size_min_h:
        return "paths_below_size"
    return "paths_not_subject"


def _rounded(b: Any) -> list[float]:
    return [round(float(v), 4) for v in b]


def explain(
    r: dict[str, Any],
    p: dict[str, Any] | None,
    th: Q.QAThresholds,
    src_t: float | None,
    segment: list[float] | None,
    dx: float | None,
) -> dict[str, Any]:
    """``{"cause", "other_detail", "evidence", "in_selected"}`` of a frame only the
    rendered QA flags ``edge_cut``: the cause (one of :data:`CAUSES`), for ``other`` its
    sub-cause (one of :data:`OTHER_DETAILS`, else None), the boxes behind it and whether
    source time ``src_t`` lies in ``segment`` (the moment picked; None when either is unknown).

    ``evidence``: the rendered cut box (``rendered_box``, its ``rendered_area`` and the
    ``edge`` it touches), the projected box it matches (``paths_box``, ``paths_kind``
    ``box`` / ``sliver`` / ``wearer`` / ``wearer_sliver``, ``paths_h``, ``paths_area``)
    and that box's distance to the edge (``edge_gap``); None where there is none.

    The geometric QA's dropped boxes (``slivers``) count too: the wearer's in the wearer
    test, the others' as projected boxes to match."""
    cut = cut_boxes(r, th)
    ev: dict[str, Any] = dict.fromkeys(
        ("rendered_box", "rendered_area", "edge", "paths_box", "paths_kind"), None
    )
    ev |= dict.fromkeys(("paths_h", "paths_area", "edge_gap"), None)
    if len(cut):
        big = cut[int(np.argmax([_area(b) for b in cut]))]
        ev |= {"rendered_box": _rounded(big), "rendered_area": round(_area(big), 4)}
        ev["edge"] = edge_of(big, th)

    def done(cause: str, box: Any = None, kind: str | None = None, ci: int | None = None):
        if box is not None:
            rb = cut[ci]
            ev.update(
                rendered_box=_rounded(rb),
                rendered_area=round(_area(rb), 4),
                edge=edge_of(rb, th),
                paths_box=_rounded(box),
                paths_kind=kind,
                paths_h=round(float(box[3] - box[1]), 4),
                paths_area=round(_area(box), 4),
                edge_gap=edge_gap(box, edge_of(rb, th)),
            )
        detail = None
        if cause == "other":
            detail = other_detail_of(cut[ci], box, kind == "sliver", p, th)
        return {"cause": cause, "other_detail": detail, "evidence": ev, "in_selected": inside}

    inside = None
    if src_t is not None and segment is not None:
        inside = bool(segment[0] <= src_t <= segment[1])

    if p is None:
        return done("unpaired")
    own_b, own_s = _boxes(p, th, own=True), _slivers(p, own=True)
    oth_b, oth_s = _boxes(p, th, own=False), _slivers(p, own=False)
    match = None
    if len(cut) and len(oth_b) + len(oth_s):
        others = np.concatenate([oth_b, oth_s])
        ov = Q._overlap(cut, others)
        i, j = np.unravel_index(int(ov.argmax()), ov.shape)
        if ov[i, j] > 0:
            match = (others[j], "box" if j < len(oth_b) else "sliver", int(i))
    if len(cut) and len(own_b) + len(own_s):
        own = np.concatenate([own_b, own_s])
        ov = Q._overlap(cut, own)
        if (ov >= th.edge_self_overlap).any():
            i, j = np.unravel_index(int(ov.argmax()), ov.shape)
            kind = "wearer" if j < len(own_b) else "wearer_sliver"
            return done("wearer_overlap", own[j], kind, int(i))
    view_dt = p.get("view_dt", 0.0)  # old reports: not recorded
    stale = view_dt is None or view_dt > STALE_VIEW_S
    if match is not None:
        box = match[0]
        m = th.edge_margin
        if dx is not None and abs(dx) >= MIN_DX and (box[0] + dx <= m or box[2] + dx >= 1 - m):
            return done("yaw_offset", *match)
        return done("stale_view" if stale else "other", *match)
    return done("stale_view" if stale else "no_projected_box")


def cause_of(
    r: dict[str, Any],
    p: dict[str, Any] | None,
    th: Q.QAThresholds,
    src_t: float | None,
    segment: list[float] | None,
    dx: float | None,
) -> str:
    """The cause (one of :data:`CAUSES`) of a frame only the rendered QA flags ``edge_cut``
    (:func:`explain`)."""
    return explain(r, p, th, src_t, segment, dx)["cause"]


def compare_clip(
    rc: dict[str, Any],
    pc: dict[str, Any],
    fps: float,
    th: Q.QAThresholds,
    ratios: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Edge-cut agreement of one clip's rendered (``rc``) and geometric (``pc``) records;
    the clip's :func:`size_ratios` are added to ``ratios`` if given (the edit's median)."""
    pairs = join_frames(rc["frames"], pc["frames"], fps)
    dx = median_dx(pairs, th)
    sizes = size_ratios(pairs, th, dx)
    if ratios is not None:
        ratios += sizes
    offset = rc.get("offset", pc.get("offset"))
    # the selected moment (clips), else the planned segment (previews, older reports)
    segment = rc.get("selected") or pc.get("selected") or rc.get("segment") or pc.get("segment")
    counts = dict.fromkeys(COUNT_KEYS, 0)
    outside = 0
    causes = dict.fromkeys(CAUSES, 0)
    detail = dict.fromkeys(OTHER_DETAILS, 0)
    only: list[dict[str, Any]] = []
    for r, p in pairs:
        counts["frames"] += 1
        re_ = r is not None and "edge_cut" in r["flags"]
        pe = p is not None and "edge_cut" in p["flags"]
        counts["rendered_edge"] += re_
        counts["paths_edge"] += pe
        counts["both"] += re_ and pe
        counts["paths_only"] += pe and not re_
        if not re_ or pe:
            continue
        counts["rendered_only"] += 1
        src_t = None if offset is None else float(offset) + float(r["t"])
        got = explain(r, p, th, src_t, segment, dx)
        causes[got["cause"]] += 1
        outside += got["in_selected"] is False
        if got["other_detail"] is not None:
            detail[got["other_detail"]] += 1
        only.append({"t": r["t"], **got})
    return {
        "file": rc["file"],
        "mode": rc.get("mode"),
        "self_method": rc.get("self_method"),
        **counts,
        "outside_selected": outside,
        "median_dx": dx,
        "size_ratio": size_ratio(sizes),
        "causes": causes,
        "other_detail": detail,
        "rendered_only_frames": only,
    }


def counts_by(clips: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    """:data:`BREAKDOWN_KEYS` (plus ``clips``) of :func:`compare_clip` results summed by
    their rendered clip's ``key`` (``mode`` or ``self_method``; ``unknown`` when the
    report did not record it)."""
    out: dict[str, dict[str, int]] = {}
    for c in clips:
        g = out.setdefault(
            str(c.get(key) or "unknown"), dict.fromkeys(("clips", *BREAKDOWN_KEYS), 0)
        )
        g["clips"] += 1
        for k in BREAKDOWN_KEYS:
            g[k] += c[k]
    return dict(sorted(out.items()))


def run_compare(qa_dir: Path, a: Path, b: Path) -> dict[str, Any]:
    """Compare a rendered and a geometric ``qa.json`` (either order); writes
    :data:`GAP_FILE` next to the rendered one.

    Raises:
        FileNotFoundError: If a report is missing.
        ValueError: If the two are not one rendered and one geometric report.
    """
    from cutpoint import jsonout  # noqa: PLC0415

    reps = {}
    for f in (a, b):
        rep = load_report(f)
        reps.setdefault(rep["mode"], (f, rep))
    if set(reps) != {"rendered", "paths"}:
        raise jsonout.ArgumentError(
            f"--compare needs one rendered and one --paths qa.json; got modes "
            f"{load_report(a)['mode']!r} and {load_report(b)['mode']!r}"
        )
    (r_file, rendered), (p_file, paths) = reps["rendered"], reps["paths"]
    th = thresholds(rendered)
    fps = float(rendered.get("fps") or 2.0)
    fps_paths = float(paths.get("fps") or 2.0)
    if fps_paths != fps:
        logger.warning(
            "the reports were checked at %g and %g fps: frames pair only where their times "
            "meet (the rest are unpaired); run both at the same --fps",
            fps,
            fps_paths,
        )
    by_stem = {Path(c["file"]).stem: c for c in paths["clips"]}
    clips = []
    ratios: list[tuple[float, float]] = []
    for rc in rendered["clips"]:
        pc = by_stem.pop(Path(rc["file"]).stem, None)
        if pc is not None:
            clips.append(compare_clip(rc, pc, fps, th, ratios))
    joined = {Path(c["file"]).stem for c in clips}
    unmatched = {
        "rendered": [c["file"] for c in rendered["clips"] if Path(c["file"]).stem not in joined],
        "paths": [c["file"] for c in by_stem.values()],
    }
    edit: dict[str, Any] = {k: sum(c[k] for c in clips) for k in COUNT_KEYS}
    edit["causes"] = {k: sum(c["causes"][k] for c in clips) for k in CAUSES}
    edit["outside_selected"] = sum(c["outside_selected"] for c in clips)
    n = edit["rendered_only"]
    edit["cause_share"] = {k: round(v / n, 3) if n else 0.0 for k, v in edit["causes"].items()}
    edit["other_detail"] = {k: sum(c["other_detail"][k] for c in clips) for k in OTHER_DETAILS}
    edit["other_detail_share"] = {
        k: round(v / n, 3) if n else 0.0 for k, v in edit["other_detail"].items()
    }
    edit["size_ratio"] = size_ratio(ratios)
    edit["by_mode"] = counts_by(clips, "mode")
    edit["by_self_method"] = counts_by(clips, "self_method")
    mismatch = [
        str(rep["source_dir"])
        for rep in (rendered, paths)
        if rep.get("source_dir") and Path(rep["source_dir"]) != Path(qa_dir)
    ]
    report = {
        "version": GAP_VERSION,
        "source_dir": str(qa_dir),
        "rendered": str(r_file),
        "paths": str(p_file),
        "fps": fps,
        "fps_paths": fps_paths,
        "stale_view_s": STALE_VIEW_S,
        "clips": clips,
        "unmatched": unmatched,
        "edit": edit,
    }
    out = r_file.parent / GAP_FILE
    cache.save_json(out, report, indent=1)
    report["out_file"] = str(out)
    report["source_mismatch"] = mismatch
    report["fps_mismatch"] = fps_paths != fps
    return report


def log_compare(report: dict[str, Any]) -> None:
    """Per-clip edge-cut agreement and the cause shares."""
    logger.info(
        "%-22s %6s %6s %6s %6s %6s  %s",
        "clip",
        "frames",
        "rend",
        "paths",
        "both",
        "r-only",
        "causes",
    )
    for c in report["clips"]:
        logger.info(
            "%-22s %6d %6d %6d %6d %6d  %s",
            c["file"][:22],
            c["frames"],
            c["rendered_edge"],
            c["paths_edge"],
            c["both"],
            c["rendered_only"],
            ", ".join(f"{k} {v}" for k, v in c["causes"].items() if v) or "-",
        )
    e = report["edit"]
    logger.info(
        "edit: %d frames; edge_cut rendered %d, paths %d, both %d; rendered only %d: %s",
        e["frames"],
        e["rendered_edge"],
        e["paths_edge"],
        e["both"],
        e["rendered_only"],
        ", ".join(f"{k} {v:.0%}" for k, v in e["cause_share"].items() if v) or "-",
    )
    if e["causes"]["other"]:
        logger.info(
            "other %d: %s",
            e["causes"]["other"],
            ", ".join(f"{k} {v}" for k, v in e["other_detail"].items() if v),
        )
    for side, files in report["unmatched"].items():
        if files:
            logger.warning(
                "%d %s clip(s) without a partner: %s", len(files), side, ", ".join(files)
            )
    logger.info("wrote %s", report["out_file"])
