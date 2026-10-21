"""Contact sheets for checking the visual layer by eye (``cutpoint events --sheets``).

Writes JPEG grids to ``<project>/visual_check/``:

* ``beauty_top.jpg``: highest beauty-shot scores (one per 20 s of a source)
* ``friends_top.jpg``: highest friends scores, with person boxes (green =
  counted, red = dropped as the wearer's own body / skis)
* ``lift_people.jpg``: lift and stopped footage with detections, to check
  that the wearer's boots and skis do not count as friends
* ``beauty_bottom.jpg``: lowest beauty scores (boots, lift bar, phone, dark)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from cutpoint.ffmpeg_utils import FFmpegError, grab_images

logger = logging.getLogger(__name__)


def grab(path: Path, t: float, width: int = 480) -> np.ndarray | None:
    """One BGR frame at ``t`` (fast seek), scaled to ``width``."""
    import cv2  # noqa: PLC0415

    try:  # seek rounded to 0.01 s, as the sheets have always been cut
        png = grab_images(path, round(max(0.0, t), 2), width=width, fmt="png")[0]
    except FFmpegError:
        return None
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def contact_sheet(
    items: list[dict[str, Any]], out: Path, cols: int = 4, width: int = 480
) -> Path | None:
    """Tile frames with captions and optional boxes into one JPEG.

    Args:
        items: ``{"path", "t", "caption", "boxes": [(x1, y1, x2, y2, kept), ...]}``
            with box coordinates normalised to 0-1.
        out: Output JPEG.
        cols: Tiles per row.
        width: Tile width (px).
    """
    import cv2  # noqa: PLC0415

    tiles = []
    for it in items:
        img = grab(Path(it["path"]), it["t"], width)
        if img is None:
            continue
        h, w = img.shape[:2]
        for x1, y1, x2, y2, kept in it.get("boxes", []):
            cv2.rectangle(
                img,
                (int(x1 * w), int(y1 * h)),
                (int(x2 * w), int(y2 * h)),
                (0, 220, 0) if kept else (0, 0, 255),
                2,
            )
        cv2.rectangle(img, (0, h - 22), (w, h), (0, 0, 0), -1)
        cv2.putText(
            img,
            it["caption"][:70],
            (4, h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(img)
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    blank = np.zeros_like(tiles[0])
    while len(tiles) % cols:
        tiles.append(blank)
    grid = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def _spread_top(
    cands: list[tuple[float, str, float, Any]], n: int, gap_s: float = 20.0, reverse: bool = True
) -> list[tuple[float, str, float, Any]]:
    """Best ``n`` candidates ``(score, source, t, extra)``, at least ``gap_s`` apart per source."""
    out: list[tuple[float, str, float, Any]] = []
    for c in sorted(cands, key=lambda c: c[0], reverse=reverse):
        if all(c[1] != o[1] or abs(c[2] - o[2]) >= gap_s for o in out):
            out.append(c)
        if len(out) >= n:
            break
    return out


def write_visual_sheets(project: Path, sources: list[Any], n: int = 24) -> list[Path]:
    """Write the contact sheets for a project's sources (visual layer must have run)."""
    from cutpoint.events import visual as V  # noqa: PLC0415
    from cutpoint.telemetry import extract  # noqa: PLC0415

    rules = V.PeopleRules()
    beauty, friends, lift = [], [], []
    boxes_of: dict[tuple[str, int], list[tuple[float, ...]]] = {}
    for s in sources:
        d = V.load_visual(s)
        if d is None:
            continue
        sc = V.frame_scores(d)
        fr = V.friends_score(sc)
        tel = extract(s.path, s.duration)
        masked = np.zeros(len(d.t), bool)
        if tel is not None:
            g = tel.grid
            idx = np.clip((d.t / 0.1).astype(int), 0, tel.n - 1)
            masked = (g["lift_mask"][idx] > 0) | (g["still_mask"][idx] > 0)
        is_self = V.self_boxes(d, rules)
        for row, me in zip(d.det, is_self, strict=True):
            fi, lab, score, x1, y1, x2, y2 = row
            if int(lab) != V.COCO_PERSON or score < rules.min_score:
                continue
            kept = not me and y2 - y1 >= rules.min_height
            boxes_of.setdefault((str(s.path), int(fi)), []).append((x1, y1, x2, y2, kept))
        for i, t in enumerate(d.t):
            extra = (i, sc["beauty_zs"][i], sc["aesthetic"][i], fr[i], int(sc["count"][i]))
            beauty.append((float(V.beauty_score(sc)[i]), str(s.path), float(t), extra))
            friends.append((float(fr[i]), str(s.path), float(t), extra))
            if masked[i] and boxes_of.get((str(s.path), i)):
                lift.append(
                    (float(np.random.default_rng(i).random()), str(s.path), float(t), extra)
                )
    out_dir = project / "visual_check"
    written = []

    def items(sel: list[tuple[float, str, float, Any]], what: str) -> list[dict[str, Any]]:
        res = []
        for score, path, t, (i, bz, ae, f, cnt) in sel:
            res.append(
                {
                    "path": _io(path),
                    "t": t,
                    "caption": f"{Path(path).stem} {t:.0f}s {what} {score:.2f} "
                    f"zs {bz:.2f} aes {ae:.1f} fr {f:.2f} n{cnt}",
                    "boxes": boxes_of.get((path, i), []),
                }
            )
        return res

    for name, sel, what in (
        ("beauty_top", _spread_top(beauty, n), "beauty"),
        ("beauty_bottom", _spread_top(beauty, n // 2, reverse=False), "beauty"),
        ("friends_top", _spread_top(friends, n), "friends"),
        ("lift_people", _spread_top(lift, n), "lift"),
    ):
        p = contact_sheet(items(sel, what), out_dir / f"{name}.jpg")
        if p:
            written.append(p)
            logger.info("Wrote %s", p)
    return written


def _io(path: str) -> str:
    from cutpoint.config import remap_source  # noqa: PLC0415

    return str(remap_source(path))
