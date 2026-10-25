"""Game-agnostic HUD region discovery for the OCR layer.

``cutpoint events --discover-regions`` (or automatically when a game has no preset)
samples short bursts of frames spread across the game's sources, runs
full-frame RapidOCR on each and finds where text *repeatedly appears and
changes*:

* Frames are taken in bursts (default 40 bursts x 4 frames, 1.5 s apart), so
  "changes over time" can be measured where it matters: within seconds, as a
  kill feed or a banner does, not across minutes.
* Menu / scoreboard frames (many text boxes, mostly static) are dropped.
* Every text box is rasterised onto a coarse grid (1920x1080 reference, 24 px
  cells). Per cell: how often text is there (occupancy) and how often its text
  differs between consecutive frames of a burst (change rate). Cells with
  frequent static text (ammo labels, minimap letters) are ignored.
* Connected groups of "dynamic" cells become regions, classified from the
  boxes inside them:
    - ``feed``: stacked lines (2+ rows per frame) near a screen edge or corner;
    - ``banner``: large text (>= 3.4% of the frame height) near the centre that
      appears briefly;
    - ``notice``: other centred text lines (objective messages, hints);
    - ``dynamic``: anything else (reported, OCR'd with the generic vocabulary).

The proposal is written as a preset JSON (see :mod:`cutpoint.events.regions`)
plus ``discovery/overlay.jpg`` (mean frame, dynamic-text heat map, proposed
rectangles) and the sampled frames as JPEGs, so the result can be checked by eye
and edited.
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint import cache
from cutpoint.events.common import Source
from cutpoint.events.regions import GENERIC_BANNERS, Region
from cutpoint.ffmpeg_utils import FFmpegError, grab_images, output_dir, probe

logger = logging.getLogger(__name__)

REF_W, REF_H = 1920, 1080
CELL = 24  # reference pixels per grid cell
GW, GH = REF_W // CELL, REF_H // CELL


@dataclass
class DiscoverParams:
    """Sampling and clustering parameters."""

    bursts: int = 40
    burst_frames: int = 4
    burst_step_s: float = 1.5
    width: int = 1280  # decode width for OCR (text >= ~12 px at 1080p stays legible)
    min_score: float = 0.5
    menu_boxes: int = 28  # frames with more boxes than this are menus/scoreboards
    static_occupancy: float = 0.45  # cells with text this often...
    static_change: float = 0.12  # ...and changing less than this are static HUD
    min_change: float = 0.2  # dynamic cell: changes in >= 20% of its occupied pairs
    min_hits: int = 3  # dynamic cell: text seen in >= this many bursts...
    min_burst_share: float = 0.08  # ...and in >= 8% of the gameplay bursts
    min_occupancy: float = 0.03  # ...and in >= 3% of the gameplay frames


# --------------------------------------------------------------------------- sampling


def _burst_frames(path: Path, t: float, n: int, step: float, width: int) -> list[np.ndarray]:
    """Decode ``n`` frames ``step`` s apart from ``t`` as BGR arrays (width ``width``)."""
    import cv2  # noqa: PLC0415

    try:
        jpegs = grab_images(
            path, t, count=n, duration=n * step, fps=f"1/{step}", width=width, quality=3
        )
    except FFmpegError:
        return []
    frames = []
    for jpg in jpegs:
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return frames


def plan_bursts(sources: list[Source], n: int) -> list[tuple[Source, float]]:
    """``n`` burst start times spread over all sources in proportion to duration."""
    total = sum(max(0.0, s.duration) for s in sources)
    if total <= 0:
        return []
    out = []
    for k in range(n):
        pos = (k + 0.5) / n * total
        for s in sources:
            if pos <= s.duration:
                lo, hi = 0.01 * s.duration, 0.99 * s.duration - 8
                out.append((s, float(np.clip(pos, lo, max(lo, hi)))))
                break
            pos -= s.duration
    return out


# --------------------------------------------------------------------------- OCR


def read_ocr_samples(path: Path) -> list[dict[str, Any]]:
    """The OCR'd burst frames cached in ``path`` (``discovery/<game>/ocr.jsonl``).

    Each record: ``{id, source, t, burst, k, frame, boxes}``; a box is
    ``{x0, y0, x1, y1, text, score}`` in reference pixels. Unreadable lines are
    skipped; a missing file gives ``[]``.
    """
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and "id" in rec:
                out.append(rec)
    return out


def boxes_in(
    frame: dict[str, Any], x: int, y: int, w: int, h: int, min_h: int = 0
) -> list[dict[str, Any]]:
    """Text boxes of an OCR'd ``frame`` whose centre lies in the rectangle
    (reference pixels) and that are at least ``min_h`` tall."""
    return [
        b
        for b in frame["boxes"]
        if x <= (b["x0"] + b["x1"]) / 2 <= x + w
        and y <= (b["y0"] + b["y1"]) / 2 <= y + h
        and b["y1"] - b["y0"] >= min_h
    ]


def _ocr_frames(
    bursts: list[tuple[Source, float]], params: DiscoverParams, frame_dir: Path, cache_path: Path
) -> list[dict[str, Any]]:
    """OCR every burst frame (cached in ``cache_path`` as JSON lines)."""
    import cv2  # noqa: PLC0415

    from cutpoint.events.killfeed import OcrEngine  # noqa: PLC0415

    done = {rec["id"]: rec for rec in read_ocr_samples(cache_path)}
    engine = None
    out: list[dict[str, Any]] = []
    fh = None
    # failed before the first frame is saved: no empty frames/ (or empty ocr.jsonl) left
    # behind; the cache is opened on the first record and closed before output_dir cleans up
    with output_dir(frame_dir), ExitStack() as stack:
        for bi, (src, t) in enumerate(bursts):
            ids = [
                f"{src.name}@{t + k * params.burst_step_s:.1f}" for k in range(params.burst_frames)
            ]
            if all(i in done for i in ids):
                out += [done[i] for i in ids]
                continue
            frames = _burst_frames(
                src.io_path, t, params.burst_frames, params.burst_step_s, params.width
            )
            if engine is None:
                engine = OcrEngine(det_side=params.width)
            for k, img in enumerate(frames):
                fid = ids[k]
                if fid in done:
                    out.append(done[fid])
                    continue
                sx = REF_W / img.shape[1]
                items = engine(img, params.min_score)
                boxes = []
                for box, text, score in items:
                    xs = [p[0] * sx for p in box]
                    ys = [p[1] * sx for p in box]
                    boxes.append(
                        {
                            "x0": round(min(xs)),
                            "y0": round(min(ys)),
                            "x1": round(max(xs)),
                            "y1": round(max(ys)),
                            "text": text,
                            "score": round(score, 3),
                        }
                    )
                fname = f"b{bi:03d}_{k}.jpg"
                cv2.imwrite(str(frame_dir / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                rec = {
                    "id": fid,
                    "source": src.name,
                    "t": round(t + k * params.burst_step_s, 2),
                    "burst": bi,
                    "k": k,
                    "frame": fname,
                    "boxes": boxes,
                }
                if fh is None:
                    fh = stack.enter_context(cache_path.open("a", encoding="utf-8"))
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                out.append(rec)
            if (bi + 1) % 10 == 0:
                logger.info("  discovery OCR: %d/%d bursts", bi + 1, len(bursts))
    return out


# --------------------------------------------------------------------------- clustering


def _norm_text(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def _cells(b: dict[str, Any]) -> tuple[slice, slice]:
    return (
        slice(max(0, b["y0"] // CELL), min(GH, b["y1"] // CELL + 1)),
        slice(max(0, b["x0"] // CELL), min(GW, b["x1"] // CELL + 1)),
    )


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (gx0, gy0, gx1, gy1) of connected True groups.

    Cells join along a text line across a one-cell gap (horizontal dilation
    only) and vertically when directly adjacent, so stacked feed rows stay one
    group but scattered world text does not chain into a screen-sized blob.
    """
    import cv2  # noqa: PLC0415

    grown = cv2.dilate(mask.astype(np.uint8), np.ones((1, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=4)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if (mask[y : y + h, x : x + w] & (labels[y : y + h, x : x + w] == i)).sum() >= 2:
            out.append((int(x), int(y), int(x + w), int(y + h)))
    return out


def _alpha_share(texts: list[str]) -> float:
    chars = "".join(texts)
    return sum(c.isalpha() for c in chars) / max(1, len(chars))


def _label_variety(texts: list[str]) -> float:
    """Distinct texts with digits removed / sightings: ~0 for readouts
    ("GEAR5", "RPM2100", "278"), high for feeds and messages (names vary)."""
    stripped = [re.sub(r"[0-9.,:%\-+]", "", t) for t in texts]
    return len({s for s in stripped if s}) / max(1, len(texts))


def _banner_proposal(frames: list[dict[str, Any]], min_h: float) -> dict[str, Any] | None:
    """One centred banner region from tall, mostly alphabetic text boxes."""
    tall = [
        (f, b)
        for f in frames
        for b in f["boxes"]
        if b["y1"] - b["y0"] >= min_h
        and 0.3 * REF_W < (b["x0"] + b["x1"]) / 2 < 0.7 * REF_W
        and 0.1 * REF_H < (b["y0"] + b["y1"]) / 2 < 0.8 * REF_H
        and _alpha_share([b["text"]]) >= 0.7
        and len(b["text"]) >= 5
    ]
    bursts = {(f["source"], f["burst"]) for f, _ in tall}
    if len(tall) < 3 or len(bursts) < 2:
        return None
    x0 = int(np.percentile([b["x0"] for _, b in tall], 10))
    x1 = int(np.percentile([b["x1"] for _, b in tall], 90))
    y0 = int(np.percentile([b["y0"] for _, b in tall], 10))
    y1 = int(np.percentile([b["y1"] for _, b in tall], 90))
    return {
        "kind": "banner",
        "x": x0,
        "y": y0,
        "w": x1 - x0,
        "h": y1 - y0,
        "score": round(len(bursts) / max(1, len(frames)) * 10, 3),
        "evidence": {
            "tall_boxes": len(tall),
            "bursts": len(bursts),
            "median_text_h": float(np.median([b["y1"] - b["y0"] for _, b in tall])),
            "samples": [b["text"] for _, b in tall[:8]],
        },
    }


def cluster_regions(
    frames: list[dict[str, Any]], params: DiscoverParams
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Propose regions from OCR'd burst frames.

    Returns:
        (proposals, maps). Each proposal: ``{name, kind, x, y, w, h, score,
        evidence}`` in reference pixels; ``maps`` holds the per-cell grids and
        ``stats``.
    """
    game_frames = [f for f in frames if len(f["boxes"]) <= params.menu_boxes]
    menu = len(frames) - len(game_frames)
    occ = np.zeros((GH, GW))
    change = np.zeros((GH, GW))
    pairs = np.zeros((GH, GW))
    burst_sets: dict[tuple[int, int], set] = {}
    by_burst: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for f in game_frames:
        bkey = (f["source"], f["burst"])
        by_burst.setdefault(bkey, []).append(f)
        seen = np.zeros((GH, GW), bool)
        for b in f["boxes"]:
            seen[_cells(b)] = True
        occ += seen
        for cell in zip(*np.nonzero(seen), strict=True):
            burst_sets.setdefault(cell, set()).add(bkey)
    bursts_hit = np.zeros((GH, GW))
    for cell, bs in burst_sets.items():
        bursts_hit[cell] = len(bs)
    for fl in by_burst.values():
        fl.sort(key=lambda f: f["k"])
        grids = []
        for f in fl:
            g: dict[tuple[int, int], str] = {}
            for b in f["boxes"]:
                ys, xs = _cells(b)
                txt = _norm_text(b["text"])
                for yy in range(ys.start, ys.stop):
                    for xx in range(xs.start, xs.stop):
                        g[(yy, xx)] = g.get((yy, xx), "") + txt
            grids.append(g)
        for g0, g1 in pairwise(grids):
            for cell in set(g0) | set(g1):
                pairs[cell] += 1
                if g0.get(cell, "") != g1.get(cell, ""):
                    change[cell] += 1
    n = max(1, len(game_frames))
    n_bursts = max(1, len(by_burst))
    occ_rate = occ / n
    chg_rate = np.divide(change, pairs, out=np.zeros_like(change), where=pairs > 0)
    static = (occ_rate >= params.static_occupancy) & (chg_rate < params.static_change)
    # HUD text recurs at the same place across many bursts; world text (signs,
    # markers, sight ranges) shows up in one or two.
    recurring = bursts_hit >= max(params.min_hits, params.min_burst_share * n_bursts)
    dynamic = (
        recurring & (occ_rate >= params.min_occupancy) & (chg_rate >= params.min_change) & ~static
    )
    comps = _components(dynamic)

    proposals: list[dict[str, Any]] = []
    for gx0, gy0, gx1, gy1 in comps:
        x0, y0, x1, y1 = gx0 * CELL, gy0 * CELL, gx1 * CELL, gy1 * CELL
        if (x1 - x0) * (y1 - y0) > 0.25 * REF_W * REF_H:
            continue  # scattered text, not a HUD element
        inside = []
        for f in game_frames:
            bs = [
                b
                for b in f["boxes"]
                if x0 - CELL <= (b["x0"] + b["x1"]) / 2 <= x1 + CELL
                and y0 - CELL <= (b["y0"] + b["y1"]) / 2 <= y1 + CELL
            ]
            if bs:
                inside.append((f, bs))
        if not inside:
            continue
        texts = [b["text"] for _, bs in inside for b in bs]
        heights = [b["y1"] - b["y0"] for _, bs in inside for b in bs]
        rows = [len({round((b["y0"] + b["y1"]) / 2 / 14) for b in bs}) for _, bs in inside]
        cx = (x0 + x1) / 2 / REF_W
        med_h = float(np.median(heights))
        mean_rows = float(np.mean(rows))
        appear = len(inside) / n
        variety = _label_variety(texts)
        alpha = _alpha_share(texts)
        side = min(cx, 1 - cx) < 0.3  # feeds hug the left/right edge
        if variety < 0.15 or alpha < 0.45:
            kind = "readout"  # gauges, compass, ammo: labels fixed, numbers change
        elif mean_rows >= 1.3 and side and med_h < 0.034 * REF_H:
            kind = "feed"
        elif med_h >= 0.034 * REF_H and 0.25 < cx < 0.75:
            kind = "banner"
        elif 0.3 < cx < 0.7:
            kind = "notice"
        else:
            kind = "dynamic"
        score = float(
            chg_rate[gy0:gy1, gx0:gx1].mean()
            * appear
            * variety
            * len(set(map(_norm_text, texts))) ** 0.5
        )
        proposals.append(
            {
                "kind": kind,
                "x": x0,
                "y": y0,
                "w": x1 - x0,
                "h": y1 - y0,
                "score": round(score, 3),
                "evidence": {
                    "frames_with_text": len(inside),
                    "appear_rate": round(appear, 3),
                    "bursts": int(bursts_hit[gy0:gy1, gx0:gx1].max()),
                    "median_text_h": round(med_h, 1),
                    "mean_rows": round(mean_rows, 2),
                    "label_variety": round(variety, 3),
                    "alpha_share": round(alpha, 3),
                    "change_rate": round(float(chg_rate[gy0:gy1, gx0:gx1].mean()), 3),
                    "samples": texts[:8],
                },
            }
        )
    if not any(p["kind"] == "banner" for p in proposals):
        ban = _banner_proposal(game_frames, 0.034 * REF_H)
        if ban:
            proposals.append(ban)
    order = {"feed": 0, "banner": 1, "notice": 2, "dynamic": 3, "readout": 4}
    proposals.sort(key=lambda p: (order.get(p["kind"], 9), -p["score"]))
    counts: dict[str, int] = {}
    for p in proposals:
        counts[p["kind"]] = counts.get(p["kind"], 0) + 1
        p["name"] = p["kind"] if counts[p["kind"]] == 1 else f"{p['kind']}{counts[p['kind']]}"
    stats = {
        "frames": len(frames),
        "menu_frames": menu,
        "gameplay_frames": len(game_frames),
        "bursts": n_bursts,
        "dynamic_cells": int(dynamic.sum()),
        "static_cells": int(static.sum()),
        "components": len(comps),
    }
    return proposals, {
        "stats": stats,
        "occ": occ_rate,
        "chg": chg_rate * recurring,
        "dynamic": dynamic,
        "static": static,
    }


# --------------------------------------------------------------------------- output


def proposals_to_regions(
    proposals: list[dict[str, Any]], max_regions: int = 6, pad: int = 12
) -> list[Region]:
    """Turn the best proposals into :class:`Region` objects (padded, clipped).

    ``feed`` regions keep the det gate at full rate; banners and notices are
    sampled at a third of the OCR rate. ``dynamic`` proposals are left out
    unless nothing else was found.
    """
    best = max((p["score"] for p in proposals if p["kind"] == "feed"), default=0.0)
    keep = [
        p
        for p in proposals
        if (p["kind"] == "feed" and p["score"] >= 0.25 * best)
        or (p["kind"] in ("banner", "notice") and p["score"] >= 0.02)
    ]
    keep = keep or [p for p in proposals if p["kind"] != "readout"]
    out = []
    for p in keep[:max_regions]:
        x, y = max(0, p["x"] - pad), max(0, p["y"] - pad)
        w = min(REF_W - x, p["w"] + 2 * pad)
        h = min(REF_H - y, p["h"] + 2 * pad)
        kind = {"feed": "feed", "banner": "banner", "notice": "notice"}.get(p["kind"], "notice")
        out.append(
            Region(p["name"], x, y, w, h, kind, gate="det", every=1 if kind == "feed" else 3)
        )
    return out


def write_overlay(
    frames: list[dict[str, Any]],
    frame_dir: Path,
    maps: dict[str, Any],
    proposals: list[dict[str, Any]],
    path: Path,
) -> None:
    """Mean gameplay frame + dynamic-text heat map + proposed rectangles."""
    import cv2  # noqa: PLC0415

    acc, n = None, 0
    for f in frames[:: max(1, len(frames) // 40)]:
        img = cv2.imread(str(frame_dir / f["frame"]))
        if img is None:
            continue
        img = cv2.resize(img, (REF_W, REF_H)).astype(np.float32)
        acc = img if acc is None else acc + img
        n += 1
    base = (
        (acc / max(1, n) * 0.6).astype(np.uint8)
        if acc is not None
        else np.zeros((REF_H, REF_W, 3), np.uint8)
    )
    heat = (np.clip(maps["chg"] * (maps["occ"] > 0), 0, 1) * 255).astype(np.uint8)
    heat = cv2.resize(heat, (REF_W, REF_H), interpolation=cv2.INTER_NEAREST)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    mask = heat > 20
    base[mask] = (0.5 * base[mask] + 0.5 * colored[mask]).astype(np.uint8)
    static = cv2.resize(
        maps["static"].astype(np.uint8), (REF_W, REF_H), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    base[static] = (0.5 * base[static] + np.array([0, 0, 0]) + np.array([60, 60, 60])).astype(
        np.uint8
    )
    colors = {
        "feed": (0, 255, 0),
        "banner": (0, 200, 255),
        "notice": (255, 200, 0),
        "dynamic": (200, 200, 200),
    }
    for p in proposals:
        c = colors.get(p["kind"], (255, 255, 255))
        cv2.rectangle(base, (p["x"], p["y"]), (p["x"] + p["w"], p["y"] + p["h"]), c, 3)
        cv2.putText(
            base,
            f"{p['name']} {p['score']:.2f}",
            (p["x"] + 4, max(20, p["y"] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            c,
            2,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), base, [cv2.IMWRITE_JPEG_QUALITY, 85])


def discover_regions(
    sources: list[Source], out_dir: Path, game: str, params: DiscoverParams | None = None
) -> dict[str, Any]:
    """Run discovery on ``sources`` and write the proposal into ``out_dir``.

    Args:
        sources: Sources of one game (decoded at the sampled times only).
        out_dir: Folder for ``frames/``, ``ocr.jsonl``, ``overlay.jpg`` and
            ``proposal.json`` (usually ``<project>/discovery/<game>``).
        game: Preset key the proposal is for.
        params: Sampling / clustering parameters.

    Returns:
        The proposal: a preset dict (``game``, ``regions``, ``banner_phrases``,
        ``discovery`` stats and per-region evidence).
    """
    params = params or DiscoverParams()
    t0 = time.perf_counter()
    bursts = plan_bursts(sources, params.bursts)
    logger.info(
        "Region discovery for %s: %d bursts x %d frames over %d source(s)",
        game,
        len(bursts),
        params.burst_frames,
        len(sources),
    )
    frames = _ocr_frames(bursts, params, out_dir / "frames", out_dir / "ocr.jsonl")
    proposals, maps = cluster_regions(frames, params)
    regions = proposals_to_regions(proposals)
    info = probe(sources[0].io_path) if sources else None
    write_overlay(
        [f for f in frames if len(f["boxes"]) <= params.menu_boxes],
        out_dir / "frames",
        maps,
        proposals,
        out_dir / "overlay.jpg",
    )
    preset = {
        "game": game,
        "reference": [REF_W, REF_H],
        "source": "discovered",
        "regions": [asdict(r) for r in regions],
        "banner_phrases": [],
        "discovery": {
            **maps["stats"],
            "params": asdict(params),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "source_resolution": [info.width, info.height] if info else None,
            "proposals": proposals,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.save_json(out_dir / "proposal.json", preset)
    logger.info(
        "Discovery: %d gameplay / %d menu frames, %d proposals (%s) in %.0fs; overlay %s",
        maps["stats"]["gameplay_frames"],
        maps["stats"]["menu_frames"],
        len(proposals),
        ", ".join(f"{p['name']}@{p['x']},{p['y']} {p['w']}x{p['h']}" for p in proposals[:6]),
        time.perf_counter() - t0,
        out_dir / "overlay.jpg",
    )
    return preset


__all__ = [
    "GENERIC_BANNERS",
    "DiscoverParams",
    "boxes_in",
    "cluster_regions",
    "discover_regions",
    "read_ocr_samples",
]
