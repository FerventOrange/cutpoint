"""Where the rendered and the geometric direction QA disagree on edge cuts
(``cutpoint 360 qa <dir> --compare RENDERED PATHS``).

Joins a rendered ``qa.json`` (``360 qa <dir>``) and a geometric one (``360 qa <dir>
--paths``) per clip (by file stem) and per frame time (nearest within half a frame
period), counts the frames each one flags ``edge_cut`` and attributes every frame
only the rendered QA flags to one cause, checked in this order:

``unpaired``
    the geometric QA has no frame at that time (other ``--fps``, or a shorter clip);
``outside_segment``
    the frame's source time lies outside the path's planned segment (a clip's
    tails or padding, which the director's safe-area pass never fixed);
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
    not count it as cut (smaller, or inside the frame).

A cut box is a box of the rendered frame that :func:`cutpoint.pano.qa.frame_flags`
counts for its edge cut (:func:`~cutpoint.pano.qa.edge_candidates` touching an edge).
Projected boxes include those the geometric QA dropped as mostly outside the view
(``slivers`` of its frames).
Writes ``edge_gap.json`` next to the rendered ``qa.json``.
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

GAP_VERSION = 1
GAP_FILE = "edge_gap.json"
CAUSES = (
    "unpaired",
    "outside_segment",
    "wearer_overlap",
    "yaw_offset",
    "stale_view",
    "no_projected_box",
    "other",
)
COUNT_KEYS = ("frames", "rendered_edge", "paths_edge", "both", "rendered_only", "paths_only")
STALE_VIEW_S = 0.5  # view frames are ~1.2 s apart: half of that is a stale position
MATCH_IOU = 0.3  # rendered / projected box pairs for the median offset
MIN_PAIRS = 5  # fewer matched pairs: no median offset
MIN_DX = 0.01  # a smaller median offset explains nothing


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


def cause_of(
    r: dict[str, Any],
    p: dict[str, Any] | None,
    th: Q.QAThresholds,
    src_t: float | None,
    segment: list[float] | None,
    dx: float | None,
) -> str:
    """The cause (one of :data:`CAUSES`) of a frame only the rendered QA flags ``edge_cut``.

    The geometric QA's dropped boxes (``slivers``) count too: the wearer's in the wearer
    test, the others' as projected boxes to match."""
    if p is None:
        return "unpaired"
    if src_t is not None and segment is not None and not segment[0] <= src_t <= segment[1]:
        return "outside_segment"
    cut = cut_boxes(r, th)
    if len(cut):
        own = np.concatenate([_boxes(p, th, own=True), _slivers(p, own=True)])
        if len(own) and (Q._overlap(cut, own) >= th.edge_self_overlap).any():
            return "wearer_overlap"
    others = np.concatenate([_boxes(p, th, own=False), _slivers(p, own=False)])
    match = None
    if len(cut) and len(others):
        ov = Q._overlap(cut, others)
        i, j = np.unravel_index(int(ov.argmax()), ov.shape)
        if ov[i, j] > 0:
            match = others[j]
    view_dt = p.get("view_dt", 0.0)  # old reports: not recorded
    stale = view_dt is None or view_dt > STALE_VIEW_S
    if match is not None:
        m = th.edge_margin
        if dx is not None and abs(dx) >= MIN_DX and (match[0] + dx <= m or match[2] + dx >= 1 - m):
            return "yaw_offset"
        return "stale_view" if stale else "other"
    return "stale_view" if stale else "no_projected_box"


def compare_clip(
    rc: dict[str, Any], pc: dict[str, Any], fps: float, th: Q.QAThresholds
) -> dict[str, Any]:
    """Edge-cut agreement of one clip's rendered (``rc``) and geometric (``pc``) records."""
    pairs = join_frames(rc["frames"], pc["frames"], fps)
    dx = median_dx(pairs, th)
    offset = rc.get("offset", pc.get("offset"))
    segment = rc.get("segment") or pc.get("segment")
    counts = dict.fromkeys(COUNT_KEYS, 0)
    causes = dict.fromkeys(CAUSES, 0)
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
        cause = cause_of(r, p, th, src_t, segment, dx)
        causes[cause] += 1
        only.append({"t": r["t"], "cause": cause})
    return {
        "file": rc["file"],
        **counts,
        "median_dx": dx,
        "causes": causes,
        "rendered_only_frames": only,
    }


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
    for rc in rendered["clips"]:
        pc = by_stem.pop(Path(rc["file"]).stem, None)
        if pc is not None:
            clips.append(compare_clip(rc, pc, fps, th))
    joined = {Path(c["file"]).stem for c in clips}
    unmatched = {
        "rendered": [c["file"] for c in rendered["clips"] if Path(c["file"]).stem not in joined],
        "paths": [c["file"] for c in by_stem.values()],
    }
    edit: dict[str, Any] = {k: sum(c[k] for c in clips) for k in COUNT_KEYS}
    edit["causes"] = {k: sum(c["causes"][k] for c in clips) for k in CAUSES}
    n = edit["rendered_only"]
    edit["cause_share"] = {k: round(v / n, 3) if n else 0.0 for k, v in edit["causes"].items()}
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
    for side, files in report["unmatched"].items():
        if files:
            logger.warning(
                "%d %s clip(s) without a partner: %s", len(files), side, ", ".join(files)
            )
    logger.info("wrote %s", report["out_file"])
